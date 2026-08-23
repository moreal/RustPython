#!/usr/bin/env python3
"""PR performance regression gate for RustPython.

Measures retired instruction counts (callgrind "Ir") for a fixed set of
Python workloads executed by a RustPython binary, and compares two such
measurements (base vs head) with a relative threshold.

Why instruction counts instead of wall-clock time?

* GitHub Actions runners are shared, throttled machines; wall-clock numbers
  routinely swing by 10-40% between runs, which forces either huge thresholds
  (misses real regressions) or a flaky gate (noise blocks unrelated PRs).
* Instruction counts from callgrind are deterministic for a deterministic
  program: repeated runs of the same binary+workload differ by well under
  0.1% (see benches/perf_ci/README.md for measured numbers), and are immune
  to CPU contention, frequency scaling, and co-tenant noise.
* Determinism also means results are comparable when base and head are
  measured in parallel processes, which keeps the job inside its time budget.

Instruction count is a proxy for time (it cannot see cache/branch effects),
but for an interpreter hot-loop it tracks real cost closely and, above all,
it makes the gate reproducible: a red gate is always caused by the diff.

Workloads fall into three tiers, because they measure three different things
and a single measurement recipe cannot serve all of them:

* "boot" -- interpreter startup and the import machinery. The whole process
  *is* the subject, so the raw count is reported. An in-process harness
  structurally cannot see these.
* "micro" -- one interpreter axis in a tight loop. Here the process is not
  the subject: booting the VM costs ~135M instructions, which is 6-24% of a
  micro workload's raw count, so a regression in the loop body shows up
  diluted and by a different factor per workload. Each micro workload is
  therefore measured twice, at N and at N/10 iterations, and the reported
  value is the difference: every fixed cost (boot, imports, setup) cancels
  exactly, so a 10% regression in the body reads as 10% everywhere and one
  threshold means the same thing for every workload. Measured intercepts land
  within 1-3% of the standalone `-c pass` floor across workloads whose
  per-iteration cost spans a factor of 24, so the linearity this relies on
  holds. A zero-iteration baseline would be cheaper but is not safe: some
  workloads degenerate at N=0 (an empty list has no lst[0]).
* "macro" -- whole programs, both numeric kernels (nbody, spectral_norm,
  scimark) and application-style benchmarks (richards, deltablue, raytrace).
  End-to-end cost is the question being asked, boot is a smaller share
  (4.5-24%), and not all of them expose an iteration knob, so the raw count
  is reported.

The interpreter is run with PYTHONHASHSEED=0 so that str/bytes hashing, and
therefore dict layout, is identical across runs. Each measurement run first
purges the bytecode cache and then repopulates it with the binary it is about
to measure, so no run is made cheaper by bytecode another run compiled and
both binaries are measured in the same steady state (see
purge_bytecode_cache and warm_bytecode_cache).

Usage:
  scripts/perf_ci.py list
  scripts/perf_ci.py measure --binary target/release/rustpython -o head.json
  scripts/perf_ci.py compare base.json head.json --threshold 0.02
"""

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERF_DIR = os.path.join("benches", "perf_ci")
MICRO = os.path.join(PERF_DIR, "micro")
PYPERF = os.path.join(PERF_DIR, "pyperformance")
# Benchmarks that predate this gate and are also driven by benches/execution.rs.
BENCHMARKS = os.path.join("benches", "benchmarks")

# How much smaller the second measurement point of a micro workload is. The
# baseline run costs roughly (fixed + body/10), so this buys exact fixed-cost
# cancellation for about 15% extra time on the micro tier.
MICRO_BASELINE_DIVISOR = 10

# Workload table. Each entry carries its tier (see the module docstring), the
# argv passed after the binary, optional extra environment, and -- for micro
# workloads -- the iteration count fed to the script through PERF_CI_N.
#
# Macro and boot workloads are sized so that a single run takes roughly
# 50-500 ms natively, a few seconds under callgrind's ~50x slowdown.
WORKLOADS = {
    # -- boot: the process itself is the subject ---------------------------
    "startup": {"tier": "boot", "argv": ["-c", "pass"]},
    "import_stdlib": {
        "tier": "boot",
        "argv": [os.path.join(MICRO, "import_stdlib.py")],
    },
    # -- micro: one interpreter axis each, fixed cost cancelled ------------
    "int_arith": {
        "tier": "micro",
        "argv": [os.path.join(MICRO, "int_arith.py")],
        "iterations": 200000,
    },
    "float_arith": {
        "tier": "micro",
        "argv": [os.path.join(MICRO, "float_arith.py")],
        "iterations": 200000,
    },
    "call_function": {
        "tier": "micro",
        "argv": [os.path.join(MICRO, "call_function.py")],
        "iterations": 100000,
    },
    "method_call": {
        "tier": "micro",
        "argv": [os.path.join(MICRO, "method_call.py")],
        "iterations": 100000,
    },
    "attr_load": {
        "tier": "micro",
        "argv": [os.path.join(MICRO, "attr_load.py")],
        "iterations": 100000,
    },
    "dict_ops": {
        "tier": "micro",
        "argv": [os.path.join(MICRO, "dict_ops.py")],
        "iterations": 50000,
    },
    "list_ops": {
        "tier": "micro",
        "argv": [os.path.join(MICRO, "list_ops.py")],
        "iterations": 50000,
    },
    "str_ops": {
        "tier": "micro",
        "argv": [os.path.join(MICRO, "str_ops.py")],
        "iterations": 20000,
    },
    "class_create": {
        "tier": "micro",
        "argv": [os.path.join(MICRO, "class_create.py")],
        "iterations": 2000,
    },
    "exceptions": {
        "tier": "micro",
        "argv": [os.path.join(MICRO, "exceptions.py")],
        "iterations": 50000,
    },
    # -- macro: whole programs, already in the repo ------------------------
    "nbody": {"tier": "macro", "argv": [os.path.join(BENCHMARKS, "nbody.py")]},
    "fannkuch": {
        "tier": "macro",
        "argv": [os.path.join(BENCHMARKS, "fannkuch.py")],
        "env": {"PERF_CI_FANNKUCH_ARG": "8"},
    },
    "mandelbrot": {
        "tier": "macro",
        "argv": [os.path.join(BENCHMARKS, "mandelbrot.py")],
    },
    "json_loads": {
        "tier": "macro",
        "argv": [os.path.join(BENCHMARKS, "json_loads.py")],
    },
    # -- macro: vendored from pyperformance (see that directory's README) --
    # Sizes are shrunk from upstream defaults via CLI args or env so each run
    # stays affordable under callgrind.
    "chaos": {
        "tier": "macro",
        "argv": [
            os.path.join(PYPERF, "bm_chaos.py"),
            "--iterations",
            "500",
            "--width",
            "128",
            "--height",
            "128",
        ],
    },
    "raytrace": {
        "tier": "macro",
        "argv": [
            os.path.join(PYPERF, "bm_raytrace.py"),
            "--width",
            "24",
            "--height",
            "24",
        ],
    },
    "deltablue": {
        "tier": "macro",
        "argv": [os.path.join(PYPERF, "bm_deltablue.py")],
        "env": {"PERF_CI_DELTABLUE_N": "30"},
    },
    "float": {
        "tier": "macro",
        "argv": [os.path.join(PYPERF, "bm_float.py")],
        "env": {"PERF_CI_FLOAT_POINTS": "20000"},
    },
    "nqueens": {
        "tier": "macro",
        "argv": [os.path.join(PYPERF, "bm_nqueens.py")],
        "env": {"PERF_CI_NQUEENS_COUNT": "7"},
    },
    "spectral_norm": {
        "tier": "macro",
        "argv": [os.path.join(PYPERF, "bm_spectral_norm.py")],
        "env": {"PERF_CI_SPECTRAL_NORM_N": "60"},
    },
    "richards": {"tier": "macro", "argv": [os.path.join(PYPERF, "bm_richards.py")]},
    "scimark_sor": {
        "tier": "macro",
        "argv": [os.path.join(PYPERF, "bm_scimark.py"), "sor"],
        "env": {"PERF_CI_SCIMARK_SOR_N": "40"},
    },
    "scimark_monte_carlo": {
        "tier": "macro",
        "argv": [os.path.join(PYPERF, "bm_scimark.py"), "monte_carlo"],
        "env": {"PERF_CI_SCIMARK_MONTE_CARLO_N": "20000"},
    },
}

TIER_ORDER = ("boot", "micro", "macro")

DEFAULT_THRESHOLD = 0.02


def workload_env(spec, iterations=None):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env.setdefault("RUSTPYTHONPATH", os.path.join(REPO_ROOT, "Lib"))
    env.update(spec.get("env", {}))
    if iterations is not None:
        env["PERF_CI_N"] = str(iterations)
    return env


def run_callgrind(binary, spec, out_file, iterations=None):
    """Run one workload under callgrind and return its instruction count."""
    cmd = [
        "valgrind",
        "--tool=callgrind",
        "--callgrind-out-file=%s" % out_file,
        "--quiet",
        binary,
        # Never write .pyc files. The cache is populated up front by
        # warm_bytecode_cache; keeping the measured runs read-only means a
        # workload can neither pay to compile bytecode for a later one nor
        # race another measurement writing the same file.
        "-B",
    ] + spec["argv"]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=workload_env(spec, iterations),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "workload failed (exit %d): %s\n%s\n%s"
            % (proc.returncode, " ".join(cmd), proc.stdout[-2000:], proc.stderr[-2000:])
        )
    ir = parse_ir(out_file)
    os.unlink(out_file)
    return ir


def measure_one(binary, name, out_dir):
    spec = WORKLOADS[name]
    out_file = os.path.join(out_dir, "callgrind.%s.out" % name)
    start = time.monotonic()
    if spec["tier"] == "micro":
        # Two points, so every fixed cost cancels in the difference.
        hi_n = spec["iterations"]
        lo_n = hi_n // MICRO_BASELINE_DIVISOR
        hi = run_callgrind(binary, spec, out_file, hi_n)
        lo = run_callgrind(binary, spec, out_file, lo_n)
        record = {
            "ir": hi - lo,
            "tier": "micro",
            "ir_at_n": hi,
            "ir_at_baseline": lo,
            "iterations": hi_n,
            "baseline_iterations": lo_n,
        }
    else:
        record = {"ir": run_callgrind(binary, spec, out_file), "tier": spec["tier"]}
    return name, record, time.monotonic() - start


def parse_ir(out_file):
    events = None
    with open(out_file) as f:
        for line in f:
            if line.startswith("events:"):
                events = line.split()[1:]
            elif line.startswith("summary:"):
                values = [int(v) for v in line.split()[1:]]
                if events is None or "Ir" not in events:
                    raise RuntimeError("no Ir event in %s" % out_file)
                return values[events.index("Ir")]
    raise RuntimeError("no summary line in %s" % out_file)


def purge_bytecode_cache():
    """Delete every __pycache__ directory the workloads could import from.

    RustPython caches compiled bytecode next to the source it imports, and
    compiling a stdlib module costs far more than loading it from that cache
    (measured: ~3.8x total instructions for import_stdlib, ~2x for chaos).
    A cache left behind by an earlier run therefore makes a measurement depend
    on what ran before it. Measuring base and then head in one checkout let
    head reuse the bytecode base had just compiled, so head came out up to 46%
    "faster" with an identical interpreter -- a bias towards whichever binary
    is measured second, which is the direction that hides a regression.
    """
    removed = 0
    roots = (
        os.path.join(REPO_ROOT, "Lib"),
        os.path.join(REPO_ROOT, PERF_DIR),
        os.path.join(REPO_ROOT, BENCHMARKS),
    )
    for root in roots:
        for dirpath, dirnames, _ in os.walk(root):
            if "__pycache__" in dirnames:
                dirnames.remove("__pycache__")
                shutil.rmtree(os.path.join(dirpath, "__pycache__"), ignore_errors=True)
                removed += 1
    return removed


def warm_bytecode_cache(binary, names):
    """Run each workload once, untimed, to populate the cache for `binary`.

    Purging alone would leave every measurement paying bytecode compilation
    for the stdlib modules its workload imports. That cost is unrelated to the
    interpreter hot paths this gate exists to protect: it inflates the job and,
    worse, dilutes sensitivity, because a regression in the measured code is
    divided by a much larger total (compiling `os` and `argparse` alone adds
    ~60% to fannkuch). Warming with the very binary about to be measured keeps
    the comparison symmetric -- each binary compiles its own bytecode -- while
    the measured runs observe steady-state execution.

    Micro workloads are warmed at their baseline iteration count: the point is
    to compile the imports, not to do the work twice.

    Run sequentially: concurrent writers of the same .pyc would race.
    """
    for name in names:
        spec = WORKLOADS[name]
        iterations = None
        if spec["tier"] == "micro":
            iterations = spec["iterations"] // MICRO_BASELINE_DIVISOR
        proc = subprocess.run(
            [binary] + spec["argv"],
            cwd=REPO_ROOT,
            env=workload_env(spec, iterations),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "warm-up of workload %r failed (exit %d):\n%s"
                % (name, proc.returncode, proc.stderr[-2000:])
            )


def tier_sort_key(name):
    return (TIER_ORDER.index(WORKLOADS[name]["tier"]), name)


def cmd_measure(args):
    binary = os.path.abspath(args.binary)
    names = args.bench or sorted(WORKLOADS)
    unknown = set(names) - set(WORKLOADS)
    if unknown:
        sys.exit("unknown workloads: %s" % ", ".join(sorted(unknown)))
    jobs = args.jobs or max(1, (os.cpu_count() or 2) - 1)
    removed = purge_bytecode_cache()
    print("purged %d stale __pycache__ directories" % removed, flush=True)
    warm_start = time.monotonic()
    warm_bytecode_cache(binary, names)
    print(
        "warmed bytecode cache with the measured binary in %.0fs"
        % (time.monotonic() - warm_start),
        flush=True,
    )
    results = {}
    wall = time.monotonic()
    with tempfile.TemporaryDirectory() as out_dir:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [
                pool.submit(measure_one, binary, name, out_dir) for name in names
            ]
            for fut in concurrent.futures.as_completed(futures):
                name, record, elapsed = fut.result()
                results[name] = record
                note = ""
                if record["tier"] == "micro":
                    note = " = %s - %s" % (
                        format(record["ir_at_n"], ","),
                        format(record["ir_at_baseline"], ","),
                    )
                print(
                    "%-22s %-6s %14s Ir%s  (%.1fs under callgrind)"
                    % (name, record["tier"], format(record["ir"], ","), note, elapsed),
                    flush=True,
                )
    wall = time.monotonic() - wall
    payload = {
        "binary": binary,
        "unit": "instructions (callgrind Ir); micro tier is fixed-cost corrected",
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print("measured %d workloads in %.0fs -> %s" % (len(results), wall, args.output))


def read_measurements(path):
    with open(path) as f:
        results = json.load(f)["results"]
    # Measurements written before the tiering change stored a bare integer.
    return {
        name: value if isinstance(value, dict) else {"ir": value, "tier": "?"}
        for name, value in results.items()
    }


def cmd_compare(args):
    base = read_measurements(args.base)
    head = read_measurements(args.head)

    common = set(base) & set(head)
    if not common:
        sys.exit("no common workloads between %s and %s" % (args.base, args.head))
    common = sorted(
        common, key=lambda n: tier_sort_key(n) if n in WORKLOADS else (9, n)
    )
    only_base = sorted(set(base) - set(head))
    only_head = sorted(set(head) - set(base))

    rows = []
    regressions = []
    for name in common:
        b, h = base[name]["ir"], head[name]["ir"]
        delta = (h - b) / b
        status = "ok"
        if delta > args.threshold:
            status = "REGRESSION"
            regressions.append((name, delta))
        elif delta < -args.threshold:
            status = "improved"
        tier = head[name].get("tier") or base[name].get("tier") or "?"
        rows.append((name, tier, b, h, delta, status))

    geomean = (
        math.exp(
            sum(math.log(head[n]["ir"] / base[n]["ir"]) for n in common) / len(common)
        )
        - 1.0
    )

    lines = []
    lines.append(
        "| workload | tier | base instructions | head instructions | delta | status |"
    )
    lines.append("|---|---|---:|---:|---:|---|")
    for name, tier, b, h, delta, status in rows:
        mark = {"REGRESSION": "❌", "improved": "✅", "ok": ""}[status]
        lines.append(
            "| %s | %s | %s | %s | %+.2f%% | %s %s |"
            % (name, tier, format(b, ","), format(h, ","), delta * 100, mark, status)
        )
    lines.append("")
    lines.append(
        "Geometric mean delta: **%+.2f%%** (threshold per workload: +%.1f%%)"
        % (geomean * 100, args.threshold * 100)
    )
    lines.append("")
    lines.append(
        "`micro` rows are corrected for fixed cost: each is measured at N and "
        "N/%d iterations and the difference is reported, so interpreter boot "
        "and imports cancel out. `boot` and `macro` rows are whole-process "
        "counts." % MICRO_BASELINE_DIVISOR
    )
    for name in only_base:
        lines.append("- `%s` only present in base measurement" % name)
    for name in only_head:
        lines.append("- `%s` only present in head measurement" % name)
    report = "\n".join(lines)

    print(report)
    summary_path = args.summary or os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("## Performance gate (callgrind instruction counts)\n\n")
            f.write(report)
            f.write("\n")

    if regressions:
        print()
        print(
            "FAIL: %d workload(s) regressed more than %.1f%%:"
            % (len(regressions), args.threshold * 100)
        )
        for name, delta in regressions:
            print("  %-22s %+.2f%%" % (name, delta * 100))
        sys.exit(1)
    print()
    print("PASS: no workload regressed more than %.1f%%" % (args.threshold * 100))


def cmd_list(_args):
    for name in sorted(WORKLOADS, key=tier_sort_key):
        spec = WORKLOADS[name]
        extra = " ".join("%s=%s" % kv for kv in sorted(spec.get("env", {}).items()))
        if spec["tier"] == "micro":
            extra = (
                "PERF_CI_N=%d (baseline %d) "
                % (
                    spec["iterations"],
                    spec["iterations"] // MICRO_BASELINE_DIVISOR,
                )
            ) + extra
        print("%-22s %-6s %s %s" % (name, spec["tier"], " ".join(spec["argv"]), extra))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("measure", help="measure Ir for each workload")
    p.add_argument("--binary", required=True, help="rustpython binary to measure")
    p.add_argument("-o", "--output", required=True, help="output JSON path")
    p.add_argument(
        "--bench",
        action="append",
        help="measure only this workload (repeatable; default: all)",
    )
    p.add_argument(
        "--jobs",
        type=int,
        help="parallel callgrind processes (default: cpu_count - 1); "
        "instruction counts are unaffected by concurrency",
    )
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser("compare", help="compare two measurement files")
    p.add_argument("base", help="JSON produced by `measure` for the base commit")
    p.add_argument("head", help="JSON produced by `measure` for the head commit")
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="max allowed relative Ir increase per workload (default: %(default)s)",
    )
    p.add_argument(
        "--summary",
        help="markdown summary output path (default: $GITHUB_STEP_SUMMARY)",
    )
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("list", help="list workloads")
    p.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
