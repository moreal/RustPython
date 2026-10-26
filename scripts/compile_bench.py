#!/usr/bin/env python3
"""Compare compile-only latency for CPython and two RustPython builds.

Run the controller under CPython; workers use each target interpreter.
No third-party packages are required. See compile_bench.md for reproduction.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SYNTHETIC_SIZES = {
    "assigns": (1000, 2000, 4000, 8000),
    "unique_defs": (1000, 2000, 4000, 8000),
    "same_name_defs": (500, 1000, 2000, 4000),
    "same_line_lambdas": (500, 1000, 2000, 4000),
    "nested_globals": (500, 1000, 2000, 4000),
    "frozenset_membership": (400, 800, 1600, 3200),
    "folded_lists": (1000, 2000, 4000, 8000),
    "nested_lambdas": (10, 20, 40, 80, 120),
    "compacted_nested_lambdas": (10, 20, 40, 80, 120),
    "skipped_nested_lambdas": (10, 20, 40, 80, 120),
}
LABELS = ("cpython", "base", "head")


def synthetic_source(name, size):
    if size <= 0:
        raise ValueError("size must be positive")
    if name == "assigns":
        return "\n".join(f"x{i} = {i}" for i in range(size))
    if name == "unique_defs":
        return "\n".join(f"def f{i}():\n    return {i}" for i in range(size))
    if name == "same_name_defs":
        return "\n".join(f"def f():\n    return {i}" for i in range(size))
    if name == "same_line_lambdas":
        return "values = (" + ",".join(f"lambda: {i}" for i in range(size)) + ",)"
    if name == "nested_globals":
        return "\n".join(
            f"x{i} = {i}\ndef f{i}():\n    global x{i}\n    return x{i}"
            for i in range(size)
        )
    if name == "frozenset_membership":
        return "x in {" + ",".join(str(i) for i in range(size)) + "}"
    if name == "folded_lists":
        return "\n".join(f"x{i} = [{i}, {i + 1}, {i + 2}]" for i in range(size))
    if name == "nested_lambdas":
        return "x=" + "lambda:" * size + "0"
    if name == "compacted_nested_lambdas":
        return "x=" + "lambda: 1 if False else " * size + "0"
    if name == "skipped_nested_lambdas":
        return (
            "x=False\n(x): ("
            + "lambda:" * size
            + "0)\ndef following():\n    return 42\n"
        )
    raise ValueError(f"unknown workload: {name}")


def make_cases(root, workloads, sizes=None):
    cases = []
    for name in workloads:
        if name.startswith("stdlib:"):
            module = name.removeprefix("stdlib:")
            if not module.isidentifier():
                raise ValueError(f"invalid stdlib module: {module}")
            entries = [
                (None, (root / "Lib" / f"{module}.py").read_text(encoding="utf-8"))
            ]
        else:
            if name not in SYNTHETIC_SIZES:
                raise ValueError(f"unknown workload: {name}")
            entries = [
                (size, synthetic_source(name, size))
                for size in (sizes or SYNTHETIC_SIZES[name])
            ]
        for size, source in entries:
            case_id = name if size is None else f"{name}/{size}"
            encoded = source.encode("utf-8")
            cases.append(
                {
                    "id": case_id,
                    "workload": name,
                    "size": size,
                    "source": source,
                    "filename": f"<compile-bench:{case_id}>",
                    "source_bytes": len(encoded),
                    "source_sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
    return cases


def measure_source(source, filename, samples, warmups):
    timings = []
    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for index in range(warmups + samples):
            start = time.perf_counter_ns()
            code = compile(
                source, filename, "exec", flags=0, dont_inherit=True, optimize=0
            )
            elapsed = time.perf_counter_ns() - start
            # Destruction of the returned object is outside the measured interval.
            del code
            if index >= warmups:
                timings.append(elapsed)
    finally:
        if was_enabled:
            gc.enable()
    return timings


def worker_main():
    request = json.load(sys.stdin)
    results = []
    for case in request["cases"]:
        try:
            values = measure_source(
                case["source"], case["filename"], request["samples"], request["warmups"]
            )
            results.append({"id": case["id"], "status": "ok", "samples_ns": values})
        except Exception as error:
            results.append(
                {
                    "id": case["id"],
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    print(
        json.dumps(
            {
                "runtime": {
                    "implementation": sys.implementation.name,
                    "version": sys.version,
                    "executable": sys.executable,
                },
                "results": results,
            }
        )
    )


def summarize_case(case, samples):
    timings = {}
    for label in LABELS:
        values = samples[label]
        timings[label] = (
            {
                "samples_ns": values,
                "median_ns": statistics.median(values),
                "min_ns": min(values),
                "stdev_ns": statistics.stdev(values) if len(values) > 1 else 0,
            }
            if values
            else None
        )

    def ratio(numerator, denominator):
        if timings[numerator] and timings[denominator]:
            return timings[numerator]["median_ns"] / timings[denominator]["median_ns"]
        return None

    return {
        **{key: value for key, value in case.items() if key != "source"},
        "timings": timings,
        "base_over_head": ratio("base", "head"),
        "head_over_cpython": ratio("head", "cpython"),
    }


def render_report(data):
    out = [
        "# Compile-only benchmark",
        "",
        f"Measured at {data['measured_at']}; {data['host']}.",
        "",
        "Only `compile(source, filename, 'exec', flags=0, dont_inherit=True, optimize=0)` is timed. "
        "Source generation, decoding, startup, imports, GC collection, code disposal and code execution are excluded. "
        "Automatic cyclic GC is disabled during each case. Times are median ± sample standard deviation, in milliseconds. "
        "Single-machine measurements are not statistical proof of a regression.",
        "",
        "| Input | Bytes | CPython ms | base ms | head ms | base/head speedup | head/CPython |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def fmt_ratio(value):
        return f"{value:.2f}x" if value is not None else "-"

    for row in data["rows"]:
        cells = []
        for label in LABELS:
            timing = row["timings"][label]
            cells.append(
                f"{timing['median_ns'] / 1e6:.3f} ± {timing['stdev_ns'] / 1e6:.3f}"
                if timing
                else "ERROR"
            )
        out.append(
            f"| {row['id']} | {row['source_bytes']} | "
            + " | ".join(cells)
            + f" | {fmt_ratio(row['base_over_head'])} | {fmt_ratio(row['head_over_cpython'])} |"
        )
    out.extend(
        [
            "",
            "## Growth",
            "",
            "Last two sizes per synthetic workload: elapsed-time ratio and exponent "
            "`log(time ratio) / log(size ratio)` (approximately 1 = linear, 2 = quadratic). "
            "This is a local scaling observation, not an asymptotic guarantee.",
            "",
            "| Workload | Sizes | CPython growth (exponent) | base growth (exponent) | head growth (exponent) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for name in SYNTHETIC_SIZES:
        rows = sorted(
            (row for row in data["rows"] if row["workload"] == name),
            key=lambda row: row["size"],
        )
        if len(rows) < 2:
            continue
        before, after = rows[-2:]
        cells = []
        for label in LABELS:
            a, b = before["timings"][label], after["timings"][label]
            if a and b:
                growth = b["median_ns"] / a["median_ns"]
                exponent = math.log(growth) / math.log(after["size"] / before["size"])
                cells.append(f"{growth:.2f}x ({exponent:.2f})")
            else:
                cells.append("ERROR")
        out.append(
            f"| {name} | {before['size']} → {after['size']} | "
            + " | ".join(cells)
            + " |"
        )
    out.extend(
        [
            "",
            "## Reproduction",
            "",
            "```sh",
            "cd " + shlex.quote(data["cwd"]),
            data["command"],
            "```",
            "",
            f"Round orders: `{data['round_orders']}`. Raw samples, executable hashes, revisions and source "
            "hashes are in `results.json`; the exact input text is in `sources.json`.",
            "",
        ]
    )
    for label in LABELS:
        info = data["interpreters"][label]
        out.append(
            f"- {label}: `{info['path']}`, ref `{info['ref']}`, SHA-256 `{info['sha256']}`; "
            f"{info.get('runtime', {}).get('version', 'unavailable').replace(chr(10), ' ')}"
        )
    if data["errors"]:
        out.extend(["", "## Errors", ""] + [f"- {error}" for error in data["errors"]])
    return "\n".join(out) + "\n"


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for label in LABELS:
        parser.add_argument(f"--{label}", type=Path, required=True)
    parser.add_argument("--base-ref", default="unspecified")
    parser.add_argument("--head-ref", default="unspecified")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent.parent
    )
    parser.add_argument(
        "--workloads",
        default=",".join(
            [
                *SYNTHETIC_SIZES,
                "stdlib:functools",
                "stdlib:dataclasses",
                "stdlib:typing",
            ]
        ),
    )
    parser.add_argument("--sizes", help="override synthetic sizes, comma separated")
    parser.add_argument("--rounds", type=positive, default=3)
    parser.add_argument(
        "--samples", type=positive, default=3, help="samples per case per round"
    )
    parser.add_argument(
        "--warmups", type=int, default=1, help="discarded compiles per case per round"
    )
    parser.add_argument(
        "--timeout",
        type=positive,
        default=600,
        help="seconds per interpreter per round",
    )
    parser.add_argument("--output", type=Path, default=Path("target/compile-bench"))
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("--warmups must be nonnegative")
    if args.sizes:
        try:
            args.sizes = sorted({positive(value) for value in args.sizes.split(",")})
        except (ValueError, argparse.ArgumentTypeError) as error:
            parser.error(str(error))
    return args


def main():
    if sys.argv[1:] == ["--worker"]:
        worker_main()
        return 0
    args = parse_args()
    cases = make_cases(args.root.resolve(), args.workloads.split(","), args.sizes)
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("duplicate workloads")
    script = str(Path(__file__).resolve())
    data = {
        "schema_version": 1,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(),
        "cwd": str(Path.cwd()),
        "command": shlex.join([sys.executable, script, *sys.argv[1:]]),
        "settings": {
            "rounds": args.rounds,
            "samples_per_round": args.samples,
            "warmups": args.warmups,
        },
        "cases": [
            {key: value for key, value in case.items() if key != "source"}
            for case in cases
        ],
        "interpreters": {},
        "round_orders": [],
        "errors": [],
        "runs": [],
    }
    samples = {case["id"]: {label: [] for label in LABELS} for case in cases}
    failed = set()
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONHOME")
    }
    env.update(PYTHONHASHSEED="0", RUSTPYTHONPATH=str(args.root.resolve() / "Lib"))
    for label in LABELS:
        path = getattr(args, label).resolve(strict=True)
        data["interpreters"][label] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "ref": getattr(args, f"{label}_ref", "CPython reference"),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sources.json").write_text(json.dumps(cases, indent=2) + "\n")
    for round_index in range(args.rounds):
        offset = round_index % len(LABELS)
        order = list(LABELS[offset:] + LABELS[:offset])
        data["round_orders"].append(order)
        ordered_cases = cases if round_index % 2 == 0 else list(reversed(cases))
        request = json.dumps(
            {"cases": ordered_cases, "samples": args.samples, "warmups": args.warmups}
        )
        for label in order:
            print(
                f"Round {round_index + 1}/{args.rounds}: {label} ({len(cases)} inputs)",
                file=sys.stderr,
                flush=True,
            )
            try:
                process = subprocess.run(
                    [data["interpreters"][label]["path"], script, "--worker"],
                    input=request,
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=args.timeout,
                    env=env,
                    cwd=args.root,
                )
                run = json.loads(process.stdout)
                if [row["id"] for row in run["results"]] != [
                    case["id"] for case in ordered_cases
                ]:
                    raise ValueError(
                        "worker returned an incomplete or reordered case catalog"
                    )
                data["interpreters"][label]["runtime"] = run["runtime"]
                data["runs"].append({"round": round_index + 1, "label": label, **run})
                for row in run["results"]:
                    if row["status"] == "ok" and len(row["samples_ns"]) == args.samples:
                        samples[row["id"]][label].extend(row["samples_ns"])
                    else:
                        failed.add((row["id"], label))
                        data["errors"].append(
                            f"{label}, round {round_index + 1}, {row['id']}: {row.get('error', 'incomplete samples')}"
                        )
            except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
                detail = str(getattr(error, "stderr", None) or error)
                data["errors"].append(f"{label}, round {round_index + 1}: {detail}")
                failed.update((case["id"], label) for case in cases)
    for case_id, label in failed:
        samples[case_id][label] = []
    data["rows"] = [summarize_case(case, samples[case["id"]]) for case in cases]
    (args.output / "results.json").write_text(json.dumps(data, indent=2) + "\n")
    (args.output / "report.md").write_text(render_report(data))
    print(args.output.resolve() / "report.md")
    return 1 if data["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
