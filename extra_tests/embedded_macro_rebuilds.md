# Embedded Python macro rebuild regression

Run from a RustPython checkout with the project's Rust toolchain and Cargo dependencies available:

```sh
python3 extra_tests/embedded_macro_rebuilds.py \
  --repo . --output /tmp/rustpython-embedded-rebuilds
```

The output directory must not exist. It retains the generated standalone consumer, Cargo verbose logs, dep-info, runtime marker output, and a JSON report. The driver uses the existing `CARGO_TARGET_DIR` when set; otherwise it creates a target directory inside the output directory. It never cleans a Cargo cache. Dependencies must already be cached because all Cargo calls use `--offline`.

This Python script is host-side build orchestration. The independent Rust consumer invokes the real `py_compile!` and `py_freeze!` macros, decodes their frozen bytecode at runtime, and prints the embedded string constants. It has no build script, avoiding the invalidation behavior of `rustpython-pylib`'s build script.

The initial build is followed by a no-op build, then one Python-only edit at a time. Checks cover both file macros, recursive directory input, nested invoking Rust source paths with spaces, absolute input paths, inline `source=` controls, symlink targets when supported, and previously skipped syntax-error input. Hashes and modification times verify that Rust sources, Cargo.toml, and Cargo.lock remain unchanged between builds. Static frozen libraries also exercise const-compatible expansion.

Known directory-file removal is checked separately. Adding a completely new file is reported as a boundary: per-file dependency markers cannot discover new directory entries until some known input or Rust source triggers another build. The subsequent known-file removal must trigger recompilation and discover the new file. This regression does not require or introduce a directory-watching redesign.

The default run requires fixed behavior. To confirm the original stale-build bug on an unpatched checkout, pass `--expect-stale` with a different output directory. Symlinks are skipped only if the host does not permit their creation; the report records this explicitly.

Additional guards cover `macro_rules!` inputs whose literal originates at the invocation or definition, and macros inside included Rust files. These ensure dependency markers follow the same source file as Python compilation. Files skipped because their contents are invalid UTF-8 must become visible after a Python-only edit makes them valid.

Two further boundaries are explicit:

- Rustc records symlink targets in dep-info. Editing the target contents is tracked, but retargeting the symlink alone can leave Cargo `Fresh` with old bytecode. The driver records this, then verifies another known-input rebuild discovers the new target.
- Non-UTF-8 operating-system paths cannot be represented by the string literal required by `include_bytes!`. Compilation retains its previous behavior, but these paths have no emitted dependency marker. On Unix, the driver checks compilation directly with rustc, confirms the input is absent from dep-info, and, when `make` is available, checks that a dep-info-driven build retains stale bytecode after a Python-only edit. Cargo itself cannot handle these non-UTF-8 source paths on the tested toolchain, so this compatibility check is separate from the Cargo tests.
