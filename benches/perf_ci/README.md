# PR performance regression gate workloads

These workloads back the `Performance gate` workflow
(`.github/workflows/perf-ci.yaml`), which blocks pull requests that regress
interpreter performance. They are driven by `scripts/perf_ci.py`, which runs
each file under `valgrind --tool=callgrind` and compares retired instruction
counts (Ir) between the PR head and its merge base.

## Why instruction counts

Wall-clock timings on shared CI runners are noisy (10-40% run-to-run swings),
which makes any wall-clock threshold either too loose or flaky. Instruction
counts of a deterministic program are reproducible to well under 0.1%
(`PYTHONHASHSEED=0` is pinned so dict/str hashing is deterministic), are
immune to CPU contention and frequency scaling, and remain comparable when
measurements run in parallel. Ir cannot observe cache or branch-predictor
effects, so the gate can miss (rare) regressions that change memory locality
without changing instruction count — the scheduled criterion benchmarks
(`cron-ci.yaml`) still track wall-clock trends over time for that.

Role split:

* `cron-ci.yaml` `benchmark` job — scheduled criterion wall-clock runs,
  published to the website; long-term trend data, never blocks PRs.
* This gate — per-PR, deterministic, blocks regressions.

Two runs of the gate share a checkout, so bytecode caching has to be handled
explicitly: RustPython writes `.pyc` files next to the sources it imports, and
loading a cached module is far cheaper than compiling it (measured: ~3.8x
fewer instructions for `import_stdlib`, ~2x for `chaos`). Whichever binary ran
second would otherwise inherit the first one's compiled bytecode and look up
to 46% faster with no interpreter change at all. `scripts/perf_ci.py`
therefore purges the cache at the start of every measurement run and warms it
with the binary it is about to measure, then measures with `-B` so the timed
runs cannot mutate it. Both binaries are thus compared in the same steady
state, and the numbers do not depend on measurement order.

## The three tiers

Workloads measure three different things, and one measurement recipe cannot
serve all of them. `scripts/perf_ci.py list` shows each workload's tier.

**`boot`** — `startup` (`-c pass`) and `import_stdlib`. The whole process is
the subject, so the raw instruction count is reported. An in-process harness
such as the criterion benches structurally cannot measure these.

**`micro`** — one interpreter axis in a tight loop (`micro/`). Here the process
is *not* the subject: booting the VM costs ~135M instructions, which is 6-24%
of a micro workload's raw count, so a regression in the loop body would show up
diluted, and by a different factor for each workload — a 10% regression reads
as anywhere from 4.6% to 9.4%, which means one threshold would silently mean
something different everywhere. Each micro workload is therefore measured
twice, at N and at N/10 iterations, and the reported value is the difference.
Every fixed cost — boot, imports, setup — cancels exactly, so 10% reads as 10%
on every workload. This costs about 15% extra time on the micro tier.

The linearity that relies on was checked: fitting the two points and
extrapolating back to zero iterations lands within 1-3% of the standalone
`-c pass` floor across workloads whose per-iteration cost spans a factor of 24.
A zero-iteration baseline would be cheaper still, but is not safe — some
workloads degenerate at N=0 (an empty list has no `lst[0]`).

**`macro`** — whole programs: numeric kernels (nbody, spectral_norm, scimark,
fannkuch, mandelbrot, float, nqueens) and application-style benchmarks
(richards, deltablue, raytrace, chaos, json_loads). End-to-end cost is the
question being asked, boot is a smaller share (4.5-24%), and not all of them
expose an iteration knob, so the raw count is reported. These catch regressions
that only appear when features interact — an optimization can win on a micro
axis and lose in realistic code.

## Layout

* `micro/` — microbenchmarks written for this gate, one axis per file. Each
  reads its iteration count from `PERF_CI_N` so the two-point measurement can
  drive it.
* `pyperformance/` — benchmark kernels vendored from
  [pyperformance](https://github.com/python/pyperformance) 1.14.0 (MIT
  license, see `pyperformance/COPYING`). Modifications are limited to
  `# RUSTPYTHON perf_ci` blocks that let the gate shrink workload sizes via
  environment variables (upstream defaults are kept).
  `pyperformance/pyperf.py` is a minimal local stand-in for
  the real pyperf harness so the kernels run unmodified; it is our code, not
  vendored.
* `../benchmarks/` — the benchmarks that predate this gate. They are standalone
  scripts, so the gate drives them directly rather than vendoring copies:
  `nbody`, `fannkuch`, `mandelbrot`, `json_loads`. `benches/execution.rs` still
  runs the same files under criterion.

Note that `../microbenchmarks/` is *not* used here. Those files are fragments
split on `# ---` with an externally injected `ITERATIONS`, not standalone
scripts, so they only run under the in-process criterion harness. They also
lack the axes this gate most needs — no file there touches attribute access —
and eight of the twenty-one are disabled in `microbenchmarks.rs` for memory
blowups.

The full pyperformance harness is not used because it hard-requires building
`psutil` (a CPython C extension) into a venv managed by the measured
interpreter, which RustPython cannot currently do. Wall-clock statistics from
pyperf would also not fit the deterministic instruction-count approach.

## Running locally

Drive the harness with CPython 3.14 (the version CONTRIBUTING.md requires and
the one CI uses); the script itself only needs the stdlib, plus `valgrind` on
the system. All workloads also run unmodified under CPython, which is handy
for sanity-checking a workload change.

```shell
cargo build --release
python3 scripts/perf_ci.py measure --binary target/release/rustpython -o head.json
git stash        # or check out the base commit and rebuild
python3 scripts/perf_ci.py measure --binary target/release/rustpython -o base.json
python3 scripts/perf_ci.py compare base.json head.json
```

Useful flags: `--bench <name>` (repeatable) to measure one workload while
iterating, `--jobs N` for parallelism, `--threshold 0.05` to loosen the gate.

Workload sizes are tuned so one full measurement of one binary takes a few
minutes (callgrind slows execution ~50x). When adding one, make it
deterministic — fixed seeds, no wall-clock dependence, no filesystem or network
I/O in the hot path — then add an entry to `WORKLOADS` in
`scripts/perf_ci.py`. A `micro` entry must read its loop count from
`PERF_CI_N` and stay well-defined at N/10; a `macro` or `boot` entry should run
in 50-500 ms natively.
