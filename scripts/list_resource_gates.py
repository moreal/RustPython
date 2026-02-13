#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STRICT_GATES = {"requires", "requires_resource"}
CONDITIONAL_GATES = {"is_resource_enabled"}
GATE_NAMES = STRICT_GATES | CONDITIONAL_GATES
MODULE_SETUP_NAMES = {"setUpModule"}
CLASS_SETUP_NAMES = {"setUpClass", "setUp", "asyncSetUp"}


@dataclass
class GateCall:
    file: str
    line: int
    col: int
    scope: str
    scope_kind: str
    gate: str
    resource: str | None
    strict: bool


@dataclass
class Scope:
    scope_id: int
    kind: str
    file: str
    qualname: str
    name: str
    lineno: int
    parent_id: int | None
    strict_resources: set[str] = field(default_factory=set)
    conditional_resources: set[str] = field(default_factory=set)


@dataclass
class FileAnalysis:
    path: str
    calls: list[GateCall]
    scopes: dict[int, Scope]


class Analyzer(ast.NodeVisitor):
    def __init__(self, relpath: str, source: str, tree: ast.AST) -> None:
        self.relpath = relpath
        self.source = source
        self.tree = tree

        self.support_aliases: set[str] = set()
        self.test_aliases: set[str] = set()
        self.requires_names: set[str] = set()
        self.requires_resource_names: set[str] = set()
        self.is_resource_enabled_names: set[str] = set()
        self.has_support_star_import = False

        self.calls: list[GateCall] = []
        self.scopes: dict[int, Scope] = {}
        self.scope_stack: list[Scope] = []
        self.next_scope_id = 0

        self._collect_import_aliases()

    def analyze(self) -> FileAnalysis:
        module_scope = self._new_scope(
            kind="module",
            name="<module>",
            qualname="<module>",
            lineno=1,
            parent_id=None,
        )
        self.scope_stack.append(module_scope)
        self.visit(self.tree)
        self.scope_stack.pop()

        return FileAnalysis(path=self.relpath, calls=self.calls, scopes=self.scopes)

    def _new_scope(
        self,
        *,
        kind: str,
        name: str,
        qualname: str,
        lineno: int,
        parent_id: int | None,
    ) -> Scope:
        scope = Scope(
            scope_id=self.next_scope_id,
            kind=kind,
            file=self.relpath,
            qualname=qualname,
            name=name,
            lineno=lineno,
            parent_id=parent_id,
        )
        self.scopes[scope.scope_id] = scope
        self.next_scope_id += 1
        return scope

    def _current_scope(self) -> Scope:
        return self.scope_stack[-1]

    def _collect_import_aliases(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "test":
                        self.test_aliases.add(alias.asname or "test")
                    elif alias.name == "test.support":
                        if alias.asname:
                            self.support_aliases.add(alias.asname)
                        else:
                            self.test_aliases.add("test")
            elif isinstance(node, ast.ImportFrom):
                if node.module == "test":
                    for alias in node.names:
                        if alias.name == "support":
                            self.support_aliases.add(alias.asname or "support")
                elif node.module == "test.support":
                    for alias in node.names:
                        if alias.name == "*":
                            self.has_support_star_import = True
                            continue
                        if alias.name == "requires":
                            self.requires_names.add(alias.asname or "requires")
                        elif alias.name == "requires_resource":
                            self.requires_resource_names.add(
                                alias.asname or "requires_resource"
                            )
                        elif alias.name == "is_resource_enabled":
                            self.is_resource_enabled_names.add(
                                alias.asname or "is_resource_enabled"
                            )

    def _is_support_object(self, expr: ast.expr) -> bool:
        if isinstance(expr, ast.Name):
            return expr.id in self.support_aliases

        if isinstance(expr, ast.Attribute) and expr.attr == "support":
            return isinstance(expr.value, ast.Name) and expr.value.id in self.test_aliases

        return False

    def _resolve_gate(self, func: ast.expr) -> tuple[str, bool] | None:
        if isinstance(func, ast.Name):
            if (
                func.id in self.requires_names
                or self.has_support_star_import
                and func.id == "requires"
            ):
                return ("requires", True)
            if (
                func.id in self.requires_resource_names
                or self.has_support_star_import
                and func.id == "requires_resource"
            ):
                return ("requires_resource", True)
            if (
                func.id in self.is_resource_enabled_names
                or self.has_support_star_import
                and func.id == "is_resource_enabled"
            ):
                return ("is_resource_enabled", False)
            return None

        if isinstance(func, ast.Attribute):
            if func.attr in GATE_NAMES and self._is_support_object(func.value):
                return (func.attr, func.attr in STRICT_GATES)

        return None

    @staticmethod
    def _extract_resource(call: ast.Call) -> str | None:
        if not call.args:
            return None
        arg = call.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        return None

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        parent = self._current_scope()
        qualname = node.name if parent.kind == "module" else f"{parent.qualname}.{node.name}"

        scope = self._new_scope(
            kind="class",
            name=node.name,
            qualname=qualname,
            lineno=node.lineno,
            parent_id=parent.scope_id,
        )

        self.scope_stack.append(scope)
        for deco in node.decorator_list:
            self.visit(deco)
        for stmt in node.body:
            self.visit(stmt)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._visit_function_like(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._visit_function_like(node)

    def _visit_function_like(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent = self._current_scope()
        qualname = node.name if parent.kind == "module" else f"{parent.qualname}.{node.name}"

        scope = self._new_scope(
            kind="function",
            name=node.name,
            qualname=qualname,
            lineno=node.lineno,
            parent_id=parent.scope_id,
        )

        self.scope_stack.append(scope)
        for deco in node.decorator_list:
            self.visit(deco)
        for stmt in node.body:
            self.visit(stmt)
        self.scope_stack.pop()

    def visit_Call(self, node: ast.Call) -> Any:
        gate = self._resolve_gate(node.func)
        if gate is not None:
            gate_name, is_strict = gate
            resource = self._extract_resource(node)
            current = self._current_scope()

            if resource is not None:
                if is_strict:
                    current.strict_resources.add(resource)
                else:
                    current.conditional_resources.add(resource)

            self.calls.append(
                GateCall(
                    file=self.relpath,
                    line=node.lineno,
                    col=node.col_offset + 1,
                    scope=current.qualname,
                    scope_kind=current.kind,
                    gate=gate_name,
                    resource=resource,
                    strict=is_strict,
                )
            )

        self.generic_visit(node)


def gather_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def is_subpath(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def normalize_path_for_report(path: Path, report_root: Path) -> str:
    if is_subpath(path, report_root):
        return path.relative_to(report_root).as_posix()
    return path.as_posix()


def load_and_analyze(path: Path, report_root: Path) -> tuple[FileAnalysis | None, str | None]:
    relpath = normalize_path_for_report(path, report_root)
    try:
        with tokenize.open(path) as f:
            source = f.read()
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        return None, f"{relpath}: read error: {exc}"

    try:
        tree = ast.parse(source, filename=relpath)
    except SyntaxError as exc:
        return None, f"{relpath}:{exc.lineno}: {exc.msg}"

    analyzer = Analyzer(relpath=relpath, source=source, tree=tree)
    return analyzer.analyze(), None


def effective_strict_resources(scopes: dict[int, Scope], scope_id: int) -> set[str]:
    def child_setup_resources(parent_id: int, names: set[str]) -> set[str]:
        resources: set[str] = set()
        for scope in scopes.values():
            if scope.parent_id != parent_id:
                continue
            if scope.kind != "function":
                continue
            if scope.name not in names:
                continue
            resources |= scope.strict_resources
        return resources

    resources: set[str] = set()
    ancestors: list[Scope] = []
    cursor: Scope | None = scopes.get(scope_id)
    while cursor is not None:
        ancestors.append(cursor)
        resources |= cursor.strict_resources
        cursor = scopes.get(cursor.parent_id) if cursor.parent_id is not None else None

    for ancestor in ancestors:
        if ancestor.kind == "module":
            resources |= child_setup_resources(ancestor.scope_id, MODULE_SETUP_NAMES)
        elif ancestor.kind == "class":
            resources |= child_setup_resources(ancestor.scope_id, CLASS_SETUP_NAMES)
    return resources


def is_test_scope(scope: Scope) -> bool:
    return scope.kind == "function" and scope.name.startswith("test")


def summarize(analyses: list[FileAnalysis]) -> dict[str, Any]:
    gate_calls: list[GateCall] = []
    files_with_strict: dict[str, set[str]] = {}
    files_with_conditional: dict[str, set[str]] = {}

    tests_with_multi_resources: list[dict[str, Any]] = []

    for analysis in analyses:
        gate_calls.extend(analysis.calls)

        strict_in_file: set[str] = set()
        conditional_in_file: set[str] = set()
        for scope in analysis.scopes.values():
            strict_in_file |= scope.strict_resources
            conditional_in_file |= scope.conditional_resources

        if strict_in_file:
            files_with_strict[analysis.path] = strict_in_file
        if conditional_in_file:
            files_with_conditional[analysis.path] = conditional_in_file

        for scope in analysis.scopes.values():
            if not is_test_scope(scope):
                continue
            effective = effective_strict_resources(analysis.scopes, scope.scope_id)
            if len(effective) < 2:
                continue
            tests_with_multi_resources.append(
                {
                    "file": analysis.path,
                    "test": f"{analysis.path}::{scope.qualname}",
                    "resources": sorted(effective),
                    "line": scope.lineno,
                }
            )

    multi_resource_files = [
        {"file": file, "resources": sorted(resources)}
        for file, resources in sorted(files_with_strict.items())
        if len(resources) >= 2
    ]

    tests_with_multi_resources.sort(key=lambda item: (item["file"], item["line"], item["test"]))

    gate_calls_sorted = sorted(
        gate_calls,
        key=lambda c: (c.file, c.line, c.col, c.gate),
    )

    return {
        "gate_call_count": len(gate_calls_sorted),
        "gate_calls": [
            {
                "file": c.file,
                "line": c.line,
                "col": c.col,
                "scope": c.scope,
                "scope_kind": c.scope_kind,
                "gate": c.gate,
                "resource": c.resource,
                "strict": c.strict,
            }
            for c in gate_calls_sorted
        ],
        "files_with_strict_resources": [
            {"file": file, "resources": sorted(resources)}
            for file, resources in sorted(files_with_strict.items())
        ],
        "files_with_conditional_resources": [
            {"file": file, "resources": sorted(resources)}
            for file, resources in sorted(files_with_conditional.items())
        ],
        "files_with_multiple_strict_resources": multi_resource_files,
        "tests_with_multiple_effective_strict_resources": tests_with_multi_resources,
    }


def print_text_report(report: dict[str, Any], scanned_files: int, skipped: list[str]) -> None:
    print(f"Scanned {scanned_files} files")
    print(f"Gate calls found: {report['gate_call_count']}")
    print(
        "Files with multiple strict resources: "
        f"{len(report['files_with_multiple_strict_resources'])}"
    )
    print(
        "Tests with multiple effective strict resources: "
        f"{len(report['tests_with_multiple_effective_strict_resources'])}"
    )

    if skipped:
        print("\nSkipped files (parse errors):")
        for item in skipped:
            print(f"- {item}")

    print("\nGate usage sites:")
    for call in report["gate_calls"]:
        resource = call["resource"] if call["resource"] is not None else "<dynamic>"
        strict = "strict" if call["strict"] else "conditional"
        print(
            f"- {call['file']}:{call['line']}:{call['col']} "
            f"{call['scope']} {call['gate']}({resource!r}) [{strict}]"
        )

    print("\nFiles with multiple strict resources:")
    for item in report["files_with_multiple_strict_resources"]:
        print(f"- {item['file']}: {', '.join(item['resources'])}")

    print("\nTests with multiple effective strict resources:")
    for item in report["tests_with_multiple_effective_strict_resources"]:
        print(f"- {item['test']} (line {item['line']}): {', '.join(item['resources'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect test.support resource-gate usage and list tests that "
            "require multiple resources simultaneously."
        )
    )
    parser.add_argument(
        "--root",
        default="Lib/test",
        help="Root directory to scan (default: Lib/test)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output instead of text",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    scan_root = (repo_root / args.root).resolve()
    if not scan_root.exists():
        raise SystemExit(f"scan root does not exist: {scan_root}")
    if not scan_root.is_dir():
        raise SystemExit(f"scan root is not a directory: {scan_root}")

    root_outside_repo = not is_subpath(scan_root, repo_root)
    report_root = scan_root if root_outside_repo else repo_root

    files = gather_python_files(scan_root)
    analyses: list[FileAnalysis] = []
    skipped: list[str] = []

    for path in files:
        analysis, error = load_and_analyze(path, report_root)
        if error is not None:
            skipped.append(error)
            continue
        assert analysis is not None
        analyses.append(analysis)

    report = summarize(analyses)
    output = {
        "root": args.root,
        "root_outside_repo": root_outside_repo,
        "report_root": report_root.as_posix(),
        "scanned_files": len(files),
        "parsed_files": len(analyses),
        "skipped_files": skipped,
        **report,
    }

    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print_text_report(report, scanned_files=len(files), skipped=skipped)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
