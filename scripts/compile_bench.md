# Compile-only comparison

`compile_bench.py` compares CPython, the PR base, and the PR head on the same
machine. Unlike execution benchmarks, it measures only the public
`compile(source, filename, "exec", flags=0, dont_inherit=True, optimize=0)` call:
parsing, symbol analysis, code generation, and conversion to a runtime code
object. It does not execute the result or isolate individual compiler phases.

## Build and run

Use clean checkouts at the intended base and head commits. Build both with
the same Rust toolchain, release profile, features, and environment. For PR 37:

```sh
# Run in each checkout; use distinct target directories.
cargo +1.99.0 build --release --locked --bin rustpython

# Run this controller under CPython, not under a target being measured.
python3.15 scripts/compile_bench.py \
    --cpython /absolute/path/to/python3.15 \
    --base /absolute/path/to/base/target/release/rustpython \
    --head /absolute/path/to/head/target/release/rustpython \
    --base-ref d1c3cb745af41fef0b1d273c24b4b7658daa3fa5 \
    --head-ref 3d0bf54c22becfe73f38086e68ec0a95038aa273 \
    --output target/compile-bench/pr37
```

Executable paths must be actual binaries, not shell aliases or version-manager
shims. Revision labels are supplied by the caller; executable hashes and target
version strings are recorded independently. CPython build settings are not
identical to RustPython's; compare the two RustPython builds to evaluate the PR.
RustPython's incremental build can retain an older Git revision in `sys.version`
because the VM build script does not watch Git HEAD. Verify checkout revisions
and cleanliness directly; the version string alone does not establish provenance.
Keep the machine idle during measurement, and do not run builds concurrently.

## Workloads and measurement

The default catalog contains 46 inputs: growing assignments, uniquely named
functions, repeatedly redefined functions, same-line lambdas, nested explicit
globals, constant-set membership, folded list literals, deeply nested lambdas, nested lambdas with
dead-branch constants that require compaction, entirely skipped nested lambdas
followed by a live function, and the exact text of
`Lib/functools.py`, `Lib/dataclasses.py`, and `Lib/typing.py` from `--root`.
Every interpreter receives the same source text, filename, and compiler flags.

Sources are generated or read once in the controller. Each worker decodes the
catalog and imports its dependencies before timing. Cyclic GC collection runs
before each case and automatic cyclic GC is disabled during it; the prior GC
state is restored afterward. Each sample times a single compile, retaining its
result until after the clock stops. Process startup, warmups, reading, source
generation, explicit GC collection, result destruction, and code execution are
not part of the measured interval.

By default there are three independent worker rounds, one warmup and three
samples per input per round (nine retained samples). Interpreter order rotates
CPython/base/head → base/head/CPython → head/CPython/base. Input order reverses
in the middle round. The controller fixes `PYTHONHASHSEED=0`, clears inherited
`PYTHONPATH`/`PYTHONHOME`, and sets `RUSTPYTHONPATH` to `--root/Lib`.

```sh
# A quick smoke run, not a meaningful performance comparison.
python3.15 scripts/compile_bench.py \
    --cpython /absolute/path/to/python3.15 \
    --base /absolute/path/to/base/rustpython \
    --head /absolute/path/to/head/rustpython \
    --workloads assigns,unique_defs --sizes 10,20 \
    --rounds 1 --samples 1 --warmups 0 --output target/compile-bench/smoke

# Contract tests, including actual worker/controller subprocesses.
cargo run --release -- scripts/test_compile_bench.py
```

## Results and limitations

The output directory contains:

- `report.md`: median ± sample standard deviation in milliseconds, base/head
  speedup, head/CPython ratio, and last-two-size growth ratios/exponents.
- `results.json`: all samples in nanoseconds, per-round results, errors,
  executable hashes/versions, caller-supplied revisions, source hashes, settings,
  machine information, working directory, and the original command.
- `sources.json`: exact input text, allowing future comparisons against the
  same fixtures even when the standard library changes.

A failed case is shown as `ERROR`, has no speedup ratio, and makes the controller
exit unsuccessfully. A timeout or crashed worker marks every input in that
worker's round as failed. Successful samples from other rounds are retained in
the raw per-round records, but do not silently turn that failure into a valid
aggregate. `--timeout` is a per-worker, per-round limit, not a per-case limit.

These are elapsed-time observations, not significance tests. A small change or
one noisy workload needs a repeated controlled experiment before calling it a
regression. The reported exponent is a local observation over the two largest
sizes, not proof of asymptotic complexity. Real modules have no synthetic-size
growth estimate. Existing output files in the selected output directory are
replaced; choose a new directory to preserve an earlier run.

## PR 37 measurement (2026-10-26)

On macOS 27.0.1 arm64, CPython 3.15.1 was compared with clean base
`d1c3cb745af41fef0b1d273c24b4b7658daa3fa5` and head
`3d0bf54c22becfe73f38086e68ec0a95038aa273`. Both RustPython binaries were built
with Rust 1.99.0 and the same `--release --locked --bin rustpython` command and
default features, before timing. The Rust source trees were unchanged during
measurement; the new benchmark scripts were uncommitted in the head worktree.

All 32 inputs compiled successfully under all three interpreters, with nine
samples per input/target. Representative medians, in milliseconds:

| Input | CPython | base | head | base/head speedup |
| --- | ---: | ---: | ---: | ---: |
| Assignments, 8,000 | 15.640 | 122.912 | 9.732 | 12.63x |
| Unique functions, 8,000 | 52.676 | 1474.817 | 62.958 | 23.43x |
| Same-name functions, 4,000 | 23.615 | 215.839 | 31.264 | 6.90x |
| Same-line lambdas, 4,000 | 16.240 | 113.275 | 25.805 | 4.39x |
| Nested globals, 4,000 | 562.853 | 591.078 | 41.936 | 14.09x |
| Constant-set membership, 3,200 | 2.594 | 46.550 | 2.515 | 18.51x |

For the last size doubling, head time grew by 1.99–2.14x in these six
workloads, versus base's 3.45–4.55x. Deeply nested lambdas still showed
approximately quadratic growth in both RustPython builds.

The initially slower cases were repeated separately using nine rotated rounds,
five samples and two warmups per case/round (45 samples per target). All eight
inputs in that confirmation run compiled successfully. The slowdown did not
disappear:

| Input | CPython ms | base ms | head ms | Head time change |
| --- | ---: | ---: | ---: | ---: |
| Nested lambdas, depth 120 | 0.443 | 10.597 | 11.507 | +8.6% |
| `functools.py` | 4.209 | 5.170 | 5.970 | +15.5% |
| `dataclasses.py` | 4.865 | 6.274 | 7.322 | +16.7% |
| `typing.py` | 11.726 | 19.219 | 19.028 | -1.0% |

Thus the original head removes the measured wide-input quadratic behavior, but is not
a uniform compile-time improvement. The repeated module/nesting slowdowns are
follow-up investigation targets; these observations do not identify a compiler
phase or prove the cause. The tiny `typing.py` difference should not be treated
as an improvement.

Raw artifacts for this local run are under `target/compile-bench/pr37/` and
`target/compile-bench/pr37-confirmation/` (ignored by Git). The binary SHA-256s
recorded in both runs are:

- CPython: `cd353074fb2780e9ed5bdaf2ed6aeab5bd9fc0c883041ea2228da4b653d97903`
- base: `548afe03bf05879c4473e977fb978c87172f6cd6bdde906e8bcaa4f67d1c5622`
- head: `f0d9ca6bbbe06f4e9cc53bee0d2a07b79d65d6d7de9fa6178db67aabc88c10d0`

## Measured follow-up improvements (2026-10-26)

The slow cases above were investigated rather than accepted as a trade-off.
Sampling identified byte-at-a-time digest hashing on real modules and repeated
runtime code hashing on deep nesting. Four changes were tested independently,
in order, keeping the previous binary as the next experiment's control:

| Change | Focused input | Before ms | After ms |
| --- | --- | ---: | ---: |
| Replace the custom FNV accumulator with the existing RapidHasher dependency | `functools.py` | 5.569 | 4.829 |
| Reuse dictionary `get_or_insert` to hash a missing constant only once | Nested lambdas, depth 120 | 9.968 | 6.837 |
| Fold code attribute hashes without temporary Python tuples/integers | Nested lambdas, depth 120 | 6.845 | 4.504 |
| Move retained constants during compaction instead of cloning their descendants | Dead-branch nested lambdas, depth 120 | 6.136 | 4.987 |

Each row is a separate controlled run, with nine rotated worker rounds, two
warmups and five retained samples per case/round. Do not add the absolute time
differences across runs. The compaction change left ordinary nested lambdas
effectively unchanged (4.494 vs 4.484 ms), consistent with the proposed cause.
Small module changes from the last three individual steps are not separately
claimed as improvements. Reports and raw samples are in
`target/compile-bench/experiment-{rapid-hash,single-lookup,code-hash,move-constants}/`.

These changes introduce no new cache or public API. Digest collisions still
require equality checks. Code hashing preserves the previous attribute hash
values, callback order, and exceptions, including custom constant hashes on
every call; it does not memoize observable Python hashes. Regression tests cover
single hashing on cache misses/hits, hash exceptions, tuple subclass overrides,
custom callback ordering, and retained code/descendant allocation identity.

A final three-way run compiled all 37 inputs successfully, with nine samples
per input/target. Representative medians from
`target/compile-bench/pr37-improved/`:

| Input | CPython ms | PR base ms | Improved head ms | Base/head speedup |
| --- | ---: | ---: | ---: | ---: |
| Assignments, 8,000 | 15.285 | 123.481 | 9.427 | 13.10x |
| Unique functions, 8,000 | 50.819 | 1449.611 | 50.047 | 28.96x |
| Same-name functions, 4,000 | 22.900 | 213.873 | 23.139 | 9.24x |
| Same-line lambdas, 4,000 | 16.125 | 111.268 | 18.038 | 6.17x |
| Nested globals, 4,000 | 570.008 | 583.471 | 33.486 | 17.42x |
| Constant-set membership, 3,200 | 2.744 | 46.074 | 2.391 | 19.27x |

The previously regressing inputs were confirmed in two additional, separate
45-sample comparisons. Within each comparison the targets rotated as above:

| Input | Original PR head ms | Improved ms | Reduction | PR base ms (separate run) | Improved ms (separate run) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nested lambdas, depth 120 | 10.596 | 4.497 | 57.6% | 9.893 | 4.434 |
| Dead-branch nested lambdas, depth 120 | 12.484 | 4.910 | 60.7% | 11.734 | 4.891 |
| `functools.py` | 5.481 | 4.476 | 18.3% | 4.789 | 4.459 |
| `dataclasses.py` | 6.794 | 5.553 | 18.3% | 5.937 | 5.536 |
| `typing.py` | 17.707 | 14.370 | 18.8% | 17.703 | 14.390 |

The original module regressions were no longer observed against the PR base.
These remain local elapsed-time observations, not cross-platform guarantees or
formal significance tests. For `typing.py`, the improved samples in the
original-head comparison had a standard deviation of 1.346 ms, while the PR
base samples in the separate base comparison had a standard deviation of
1.436 ms. Inspect raw rounds rather than relying on one aggregate.

The improved binary SHA-256 is
`56d53005476a9cc9ea3df17dc1a6cb30a8a80f424a50572501ddd4b3e7bb5dbb`;
the reference binary hashes are unchanged. The improved source is the original
head `3d0bf54c22becfe73f38086e68ec0a95038aa273` plus the four changes described
above. It was uncommitted when measured, and all builds finished before timing.
Confirmation artifacts are in `target/compile-bench/pr37-followup-confirmation/`
(original head as control) and `target/compile-bench/pr37-improved-confirmation/`
(PR base as control). Each report includes its exact reproduction command.

### Scope-consumption follow-up and five-change verification run

A fifth, allocation-focused cleanup separates choosing/advancing a symbol
table from cloning it for activation. Consuming a skipped scope no longer
clones and immediately discards its descendants. The regression test failed
before the change because an activation stack with capacity 1 grew to 4; after
the change, it stays at 1, including ordinary and hidden annotation blocks.
Actual scope activation still creates an independent copy, preserving replay.

No elapsed-time improvement is claimed for this step. An initial unused-boolean
input still compiled nested scopes in no-emission mode and showed no benefit;
it did not isolate the intended operation. A second, non-simple annotation
input also showed no measurable difference (depth 120: 0.063 vs 0.064 ms).
Both negative results are retained in `target/compile-bench/experiment-skip-scopes/`
and `target/compile-bench/experiment-skipped-annotations/`. The default skipped
scope input now uses the non-simple annotation, not the initial boolean input.

The five-change binary was compared again with CPython and the PR base
on all 42 inputs then present, with nine samples per input/target and no failures.
The report is `target/compile-bench/pr37-final-v2/report.md`; its binary SHA-256
is `05951c2452cea87a491d0ea6614298ee65af4e4b5d1fec2f9656b55a8d27ab74`.
Representative final medians:

| Input | CPython ms | PR base ms | Final head ms | Base/head speedup |
| --- | ---: | ---: | ---: | ---: |
| Unique functions, 8,000 | 51.726 | 1423.324 | 49.259 | 28.89x |
| Nested lambdas, depth 120 | 0.423 | 9.840 | 4.539 | 2.17x |
| `functools.py` | 3.924 | 4.787 | 4.553 | 1.05x |
| `dataclasses.py` | 4.554 | 5.868 | 5.604 | 1.05x |
| `typing.py` | 10.890 | 17.595 | 14.528 | 1.21x |

At that point deep nesting still grew approximately quadratically: ordinary
lambdas grew 2.27x from depth 80 to 120. Sampling identified constant-operand
cloning, recursive runtime constant hashing, and symbol-table subtree cloning.

### Borrowed operands and compiler-local hash policy

Three further changes were investigated independently, using the previous
binary as each experiment's control:

1. Return borrowed pool entries from `get_const_value`, owning only synthetic
   small constants and materializing a copy when a fold actually constructs a
   new constant. Read-only inspection no longer clones nested code objects.
2. Use fallible amortized growth instead of reserving exactly one slot before
   each new constant. The allocation regression test observed 1,024 capacity
   changes for 1,024 insertions before the fix and requires fewer than 32 after
   it; allocation errors still propagate before insertion.
3. Give the private compiler constant dictionary a lookup-only hash policy.
   Compilation creates and canonicalizes child constants before their parent,
   including each code object's `co_consts` tuple. Equal compiler code therefore
   shares that tuple's identity. Hashing its identity avoids descending through
   the same child code chain again at every ancestor. Stored keys and equality
   remain unchanged; public code hashing and comparison do not use this policy.

The first and third changes produced these isolated 45-sample results. Each
comparison used nine rotated rounds, two warmups and five retained samples;
absolute differences from separate runs must not be added together.

| Change | Input | Before ms | After ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| Borrow constant operands | Nested lambdas, depth 120 | 4.517 | 1.949 | 2.32x |
| Borrow constant operands | Dead-branch nested lambdas, depth 120 | 5.032 | 2.334 | 2.16x |
| Compiler-local hash | Nested lambdas, depth 120 | 1.909 | 1.078 | 1.77x |
| Compiler-local hash | Dead-branch nested lambdas, depth 120 | 2.289 | 1.328 | 1.72x |

Borrowing reduced the three measured module medians by approximately 3% in its
isolated run. The subsequent compiler-local hash change left them effectively
unchanged (for example, `dataclasses.py`: 5.390 vs 5.385 ms). No speedup is claimed
for amortized reservation: folded-list compilation at size 8,000 changed from
24.819 ± 0.624 to 24.407 ± 0.573 ms, while some other medians rose slightly.
The small differences overlap sample variation; the reason to keep that
one-line change is the directly tested allocation-growth bound.

These experiment records are in
`target/compile-bench/experiment-borrow-operands/`,
`target/compile-bench/experiment-amortized-constants/`,
`target/compile-bench/experiment-compiler-key-all/`, and
`target/compile-bench/experiment-compiler-key-confirmation/`.
The last comparison's control binary SHA-256 is
`0e21cacb9cd7e829c7574e8b4ece03c8639496381e3635d0add58339cc740be5`;
its improved binary SHA-256 is
`25033d7557ae487714bcd334dd1644edf8142acac1a45cfa94532b13be8fdac2`.
They represent seven and eight local follow-ups respectively, not new committed
revisions. Source and executable provenance is recorded per experiment.

The compiler-hash regression test bounds real code-name hash callbacks while
constructing a 12-level chain: the old path invoked them 210 times; the new
path must use at most 36. Sharing tests retain distinct code constants, boolean
versus integer, signed float and complex zeros, and fresh NaNs, including Code
inside Tuple/Frozenset/Slice containers. The private policy relies on bottom-up
canonicalization and must not be reused for arbitrary Python-created code.

A proposed public code-equality shortcut for identical `co_consts` tuples was
rejected. Reconstructing frozenset constant keys can hash nested code and raise
a user exception even when both sides share that tuple. A targeted probe against
both CPython and RustPython confirmed the observable exception; a characterization
test preserves it. Public comparison is unchanged.

Depth 80 to 120 still grows faster than linearly after these changes (ordinary
lambdas: 1.89x, local exponent 1.56 in the compiler-hash confirmation). Scope
activation still clones symbol-table descendants to preserve replay. These
measurements do not claim all compiler bottlenecks or all quadratic behavior
have been eliminated.

### Removing duplicate tuple-key hashing

The compiler-hash wide-input confirmation found small costs worth investigating:
at size 8,000 assignments changed from 9.257 to 9.404 ms and folded lists from
25.030 to 25.404 ms. These are small, noisy differences, not proof of a broad
regression. The record is
`target/compile-bench/experiment-compiler-key-wide-confirmation/`.

Inspection found redundant work in a tuple constant key: it contains both the
element keys and the original tuple, so hashing the entire pair walks both.
The private compiler lookup now hashes only the element keys. Equal complete
keys necessarily have equal element keys; additional collisions still use the
unchanged complete-key equality. The original tuple remains stored for equality
and decoding. This shortcut is limited to the two-element keys generated by
`constant_key` for exact tuples; scalar and opaque container lookup is unchanged.

A regression test using a real cached Code inside a tuple observed two code-name
hash callbacks before the change and requires exactly one afterward. All ten
constant-cache tests passed after the change. Independent correctness reviews
found no missing premise in the current private call path.

The isolated comparison compiled all 46 inputs successfully with 45 samples per
input/target (`target/compile-bench/experiment-tuple-key-all/`). Representative
medians before/after were 48.524/47.560 ms for 8,000 unique functions,
25.275/25.016 ms for 8,000 folded lists, and 9.385/9.401 ms for 8,000 assignments.
The small timing differences are not claimed as robust speedups. In particular,
the improved `dataclasses.py` samples had a 5.276 ms standard deviation with
several slow samples; the raw samples are retained rather than discarded.
Removing directly observed duplicate work is the reason to retain this small change.

### Nine-change final comparison

The final binary includes all nine local follow-ups above, still uncommitted at
measurement time. Its SHA-256 is
`1e1c81d4e676efda90b36e8b2c586c59be3ac2042f9a32ef08203a28fe2e2c6d`;
the PR base and CPython binary hashes are unchanged. All 46 inputs succeeded
under all three interpreters in
`target/compile-bench/pr37-final-round2-nine/` (nine samples per input/target).

| Input | CPython ms | PR base ms | Final head ms | Base/head speedup |
| --- | ---: | ---: | ---: | ---: |
| Unique functions, 8,000 | 51.416 | 1418.517 | 46.844 | 30.28x |
| Folded lists, 8,000 | 40.228 | 763.121 | 25.122 | 30.38x |
| Nested lambdas, depth 120 | 0.430 | 9.959 | 1.103 | 9.03x |
| `functools.py` | 3.973 | 4.856 | 4.246 | 1.14x |
| `dataclasses.py` | 4.603 | 5.990 | 5.381 | 1.11x |
| `typing.py` | 11.126 | 17.874 | 13.948 | 1.28x |

A separate 45-sample comparison directly tests the combined effect of the four
latest changes against the five-change snapshot; it does not add differences
from separate experiments. The report is
`target/compile-bench/pr37-round2-nine-confirmation/`:

| Input | Five-change control ms | Nine-change head ms | Speedup |
| --- | ---: | ---: | ---: |
| Nested lambdas, depth 120 | 4.438 | 1.074 | 4.13x |
| Dead-branch nested lambdas, depth 120 | 5.019 | 1.341 | 3.74x |
| `functools.py` | 4.455 | 4.236 | 1.05x |
| `dataclasses.py` | 5.605 | 5.345 | 1.05x |
| `typing.py` | 14.366 | 13.708 | 1.05x |

The final head still takes about 2.5 times CPython's time on deep lambdas, and
the depth-80-to-120 growth remains superlinear. A separate six-second sampling
profile (not a timed benchmark) now has SymbolTable/Vec<SymbolTable> cloning as
its two most frequent leaf frames, ahead of memory movement and SymbolTable
destruction. The raw profile is
`target/compile-bench/profile-nine-nested.sample.txt`. A private active-scope
frame borrowing immutable descendants is a follow-up design candidate, not an
implemented change; finally replay, annotation cursor state and inlined
comprehension symbol overrides must be preserved.

Final local validation of the nine-change source passed: 1,462 Rust tests across
41 suites, 128 C-API tests (four ignored), and `test_code`, `test_compile`,
`test_symtable`, `test_scope`, and `test_marshal` (reported run=381, skipped=85).
The benchmark's 14 contract tests passed under both RustPython and CPython.
Workspace and separate C-API Clippy checks passed with warnings denied, as did
formatting and the applicable repository hooks. Logs are in
`target/compile-bench/verification-nine-*.log`. These are local results; the
follow-ups had not been committed or validated by remote CI at measurement time.

### Borrowed active symbol frames

The tenth local follow-up implements the scope-frame design above. The analyzed
`SymbolTable` tree stays immutable; a private `SymbolTableFrame` borrows its
metadata, symbols and children. Each activation owns only child references,
cursors and annotation-frame progress. Symbols use copy-on-write only at the
two existing comprehension override sites. Finally replay gets fresh activation
state, while taking/restoring an annotation frame retains its progress. Public
symbol-table ownership, compile APIs and symbol analysis are unchanged; no new
cache, dependency, arena, reference counting or unsafe code is introduced.

Two regression tests first failed on metadata/descendant pointer identity and
unchanged symbol pointer identity respectively. They now pass without copying
the underlying allocations. Additional tests cover fresh replay state, restored
annotation cursors, legacy comprehension child insertion, nested closures in
finally replay, annotation scopes and enclosing-symbol restoration. The three
runtime fixtures also passed against the unchanged nine-change binary and
CPython, including both future-annotation forms where exercised.

The isolated comparison uses the nine-change snapshot as control, not the PR
base. All 46 inputs succeeded with 45 samples per input/target in
`target/compile-bench/experiment-symbol-frames-all/`. A separate deep-input run
also succeeded with 45 samples per input/target in
`target/compile-bench/experiment-symbol-frames-deep/`:

| Nested lambda depth | CPython ms | Nine-change control ms | Borrowed-frame ms | Control/head speedup |
| --- | ---: | ---: | ---: | ---: |
| 120 | 0.449 | 1.258 | 0.593 | 2.12x |
| 240 | 0.989 | 5.885 | 1.468 | 4.01x |
| 400 | 2.027 | 17.798 | 2.867 | 6.21x |
| 800 | 5.457 | 73.395 | 7.671 | 9.57x |

Doubling depth from 400 to 800 grows elapsed time by 4.12x in the control,
2.68x in the new build, and 2.69x in CPython. This removes the observed
scope-activation subtree-copy bottleneck, not every superlinear compiler cost.
The new build is still about 1.41 times CPython's time at depth 800. All three
interpreters also compiled the boundary probes at depths 120/160/240/400/800
(`target/compile-bench/scope-frame-depth-boundary.log`).

The first all-input comparison had an approximately 2.5% higher assignment
median at size 8,000. Rather than treating this as either proven regression or
noise, a separate 45-sample wide-input confirmation was run, without concurrent
builds/tests (`target/compile-bench/experiment-symbol-frames-wide-confirmation/`):

| Input | Nine-change control ms | Borrowed-frame ms |
| --- | ---: | ---: |
| Assignments, 8,000 | 9.927 ± 0.356 | 10.017 ± 0.435 |
| Folded lists, 8,000 | 26.148 ± 0.637 | 26.251 ± 0.456 |
| `functools.py` | 4.923 ± 0.285 | 4.787 ± 0.278 |
| `dataclasses.py` | 6.117 ± 0.241 | 6.026 ± 0.300 |
| `typing.py` | 15.662 ± 0.493 | 15.478 ± 0.501 |

These small differences overlap sample variation and are not robust speedup or
regression claims. Absolute times differ from earlier experiments even for
CPython; only within-run comparisons are used to attribute this change.

A separate six-second profile of repeated depth-120 compilation contains no
`SymbolTable` clone frames. Its most frequent leaf frames are now instruction
sequence iteration (270 samples), memory movement (254) and allocator work,
rather than subtree cloning. The raw profile is
`target/compile-bench/profile-symbol-frames-nested.sample.txt` (3,323 main-thread
samples). This is a bottleneck-location observation, not a timed benchmark or a
claim that every allocation is unnecessary.

The combined PR-base/head/CPython comparison succeeded on all 46 inputs with
nine samples per input/target in
`target/compile-bench/pr37-final-symbol-frames/`:

| Input | CPython ms | PR base ms | Ten-change head ms | Base/head speedup |
| --- | ---: | ---: | ---: | ---: |
| Unique functions, 8,000 | 60.374 | 1625.407 | 52.502 | 30.96x |
| Folded lists, 8,000 | 47.246 | 790.673 | 27.793 | 28.45x |
| Nested lambdas, depth 120 | 0.478 | 10.904 | 0.567 | 19.22x |
| `functools.py` | 4.701 | 5.532 | 5.041 | 1.10x |
| `dataclasses.py` | 5.090 | 7.081 | 6.040 | 1.17x |
| `typing.py` | 12.512 | 20.151 | 15.699 | 1.28x |

The new binary SHA-256 is
`14a114077bf5e526a6da4ff67afd06e20a5527e61fbe9a8d397753e16436799e`;
the nine-change control, PR base and CPython hashes are recorded above and in
each experiment's `results.json`. These ten follow-ups were uncommitted when
measured and had no remote CI result for that source state.

Final local validation of the ten-change source passed: 1,470 Rust tests across
41 suites (20 ignored), 128 C-API tests (four ignored), and all nine Python
modules (`test_code`, `test_compile`, `test_symtable`, `test_scope`,
`test_marshal`, `test_listcomps`, `test_setcomps`, `test_dictcomps`,
`test_genexps`; reported run=460, skipped=85). The benchmark's 14 contract tests
passed under both RustPython and CPython. Workspace and separate C-API Clippy
checks passed with warnings denied; formatting and applicable repository hooks
passed. The hook runner also warned about an old unhealthy Python 3.14.0 cache
versus installed 3.14.7; the applicable hooks themselves all passed. Logs are in
`target/compile-bench/verification-symbol-frames-*.log`. Independent code and
measurement reviews found no actionable issues; these local results do not
replace remote CI for the measured source.

### Separating linear instructions from graph metadata

Two further local changes reduce instruction storage without changing public
APIs or bytecode semantics. Raw location snapshots used only by Python
`InstructionSequence.addop` are stored in a lazy per-sequence prefix instead of
every native instruction slot. Native appends do not read or grow that prefix.
Insertion moves entries with `copy_within` and preserves the old slot metadata,
including the existing snapshot-duplication behavior.

A private `LinearInstructionInfo` then separates linear instruction data from
the public CFG `InstructionInfo`: block targets and graph-handler metadata are
normalized away before compaction and reconstructed as absent when rebuilding
the CFG. The distinct assembler exception-handler, target and offset fields
remain intact. Raw source positions, semantic positions and line overrides
are retained; shared location/sizing helpers avoid duplicate implementations.
No new dependency, unsafe code, growth policy or public type change is needed.

On this arm64 build, `InstructionInfo` remains 80 bytes. Instruction sequence
entries shrink from 120 to 96 bytes, a 20% reduction per backing-array slot.
The sequence header grows from 112 to 136 bytes for the optional prefix, so the
20% figure is not a claim about total compiler memory. Existing tests for the
100/200-slot growth policy and insertion metadata are unchanged. Seven new
tests cover storage bounds, mixed native/public signed coordinates, insertion,
growth, independent clones/nesting, label conversion, native suffixes without
snapshot growth, and semantic-location/line-override round trips. Resource
tests were observed failing before the relevant changes and passing afterward;
behavioral characterization fixtures also passed against the prior design.

The intermediate candidates are retained rather than reported as successes:

- A 104-byte entry with snapshots extended on every native append had no robust
  general CPU gain (`experiment-instruction-layout-all/` and `-deep/`).
- Keeping snapshots as a prefix removed those hot-path accesses, but CPU changes
  remained small (`experiment-instruction-prefix-confirmation/`,
  `experiment-instruction-final-all/`, `-deep/`, `-wide-repeat/`). Its five-process
  depth-800 peak-RSS median was 58,785,792 bytes versus 58,523,648 for control.
  Smaller requested slots alone did not reduce whole-process RSS.
- Separating CFG-only metadata produced the final 96-byte entry. The isolated
  prefix-to-linear comparison used 13 inputs with 45 samples each and succeeded
  (`experiment-linear-payload-isolated/`); small timing differences are not
  claimed as robust speedups.

All experiment directories above are under `target/compile-bench/` and retain
raw samples, errors, exact inputs, source hashes, binary hashes and commands.
The original two intermediate binary hashes are
`1e4c53ac337655e78b1c57a4ab7dab6217783b4cd8114ef31327f16eacd46919`
and `7e987b62ae8aff6b0081f4328f5b6fc73c1fe799538c111405f4be32d2d37044`.
The final binary hash is
`78d238b987428d3f1231f5f8b1c13ab84847006d2939b0d3470e829acdc2ab82`;
the ten-change control is `14a114077bf5e526a6da4ff67afd06e20a5527e61fbe9a8d397753e16436799e`.

The final control/head/CPython comparison succeeded on all 46 inputs with 45
samples each (`experiment-linear-payload-all/`). A separate deep-input run also
succeeded with 45 samples per input/target (`experiment-linear-payload-deep/`):

| Input | CPython ms | Ten-change control ms | Linear-payload head ms |
| --- | ---: | ---: | ---: |
| Unique functions, 8,000 | 53.388 | 47.366 | 47.054 |
| Folded lists, 8,000 | 41.798 | 25.394 | 25.218 |
| `functools.py` | 3.975 | 4.285 | 4.306 |
| `dataclasses.py` | 4.607 | 5.495 | 5.506 |
| `typing.py` | 11.249 | 14.350 | 14.267 |
| Nested lambdas, depth 800 (separate deep run) | 4.749 | 6.505 | 6.597 |

The robust result is reduced storage and measured memory, not a broad compile
latency speedup. Small positive and negative timing differences remain; do not
claim that the CPython CPU gap or every memory-movement bottleneck is resolved.

A separate wide-input repeat retained 75 samples per input/target
(`experiment-linear-payload-wide-repeat/`), also without errors:

| Input | Ten-change control ms | Linear-payload head ms |
| --- | ---: | ---: |
| Assignments, 8,000 | 9.352 ± 0.248 | 9.385 ± 0.327 |
| Unique functions, 8,000 | 46.566 ± 1.464 | 46.007 ± 1.236 |
| Folded lists, 8,000 | 24.492 ± 0.386 | 24.529 ± 0.712 |
| `functools.py` | 4.267 ± 0.248 | 4.272 ± 0.186 |
| `dataclasses.py` | 5.410 ± 0.435 | 5.437 ± 0.415 |
| `typing.py` | 13.780 ± 0.455 | 13.893 ± 0.437 |

These differences overlap sample variation; neither a general speedup nor an
absence-of-regressions guarantee follows from them.

Whole-process memory was measured separately from timing with five fresh
processes per interpreter. Each disables cyclic GC, compiles the same depth-800
source once with the benchmark flags, and retains its result. `/usr/bin/time -l`
records maximum RSS and peak memory footprint; these include interpreter startup
and all compilation phases, not just instruction arrays. Median values:

| Metric (MiB) | CPython | Ten-change control | Linear-payload head | Control/head reduction |
| --- | ---: | ---: | ---: | ---: |
| Maximum resident set | 27.61 | 55.83 | 52.30 | 6.3% |
| Peak memory footprint | 20.09 | 40.44 | 36.73 | 9.2% |

The exact control/head medians are 58,540,032/54,837,248 RSS bytes and
42,402,392/38,519,384 footprint bytes. All five samples remain in
`target/compile-bench/linear-memory-{control,final,cpython}-{1..5}.log`.
This single-machine, single-workload result does not promise the same reduction
for every program, and it is not a compiler-phase allocation measurement.

Separate six-second depth-120 sampling profiles are
`profile-linear-control.sample.txt` and `profile-linear-final.sample.txt` under
the same directory. Main-thread sample counts are 3,568 and 3,518. The leading
leaves remain instruction-slot allocation/initialization (334 versus 279 samples)
and memory movement (343 versus 273). Their observed shares decrease, but these
profiles cover the entire repeated-compile loop, including result destruction;
they are not timed-interval CPU measurements or proof of a latency speedup.

Direct Python InstructionSequence growth, signed-coordinate and nesting probes
passed under the final head, control and CPython
(`linear-payload-public-probes.log`). Independent reviews found no semantic
defect. One lint blocker, a now-unused private sizing wrapper, was reproduced
with Clippy and removed; the common helper and its original comment remain.

Final local validation passed with Rust 1.99.0: the workspace test command
excluding wasm, venvlauncher and C-API reported 1,477 passed, zero failed and
20 ignored across 41 suites. Separate C-API tests reported 128 passed and four
ignored. Both workspace and separate C-API Clippy checks passed with
`-D warnings`. The release interpreter passed all nine selected Python modules
(`test_code`, `test_compile`, `test_symtable`, `test_scope`, `test_marshal`,
`test_listcomps`, `test_setcomps`, `test_dictcomps`, `test_genexps`): 460 tests
reported run, 85 skipped. Benchmark contract tests passed all 14 cases on both
RustPython and CPython; formatting, diff whitespace checks and applicable hooks
also passed. The hook runner warned about an unrelated unhealthy cached Python
3.14.0 environment versus installed 3.14.7; every applicable hook listed in its
output passed.

The first C-API build stopped with `No space left on device` before tests ran.
Only this worktree's regenerable debug incremental cache was removed; source,
release binaries and experiment data were retained. C-API and subsequent checks
were then rerun successfully. Both failed and successful logs remain under
`target/compile-bench/verification-instruction-final-*`. The final release hash
still matches the measured binary above. These changes were local and
uncommitted when measured; the older PR CI results do not validate that source
state.
