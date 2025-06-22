#!/usr/bin/env python3
"""
RustPython Test Decorator Tool

This tool automatically adds @expectedFailure decorators and TODO comments
to failing test methods in CPython test suites being ported to RustPython.
It handles inheritance properly by placing decorators at the appropriate level.
"""

import libcst as cst
from typing import Dict, Set, List, Tuple, Optional
import subprocess
import re
from pathlib import Path
import argparse


class TestFailureAnalyzer:
    """Analyzes test failures and determines where to place decorators"""

    def __init__(self):
        self.test_results: Dict[str, Dict[str, bool]] = {}  # class.method -> pass/fail
        self.class_hierarchy: Dict[str, Set[str]] = {}  # parent -> children
        self.method_definitions: Dict[str, str] = {}  # method -> defining class

    def run_tests(self, test_file: str) -> Dict[str, bool]:
        """Run tests and collect failure information"""
        try:
            # Run pytest with verbose output to get individual test results
            result = subprocess.run(
                ["cargo", "run", "--", test_file, "-v", "-b"],
                capture_output=True,
                text=True
            )

            # Parse test results
            test_results = {}
            for line in result.stderr.split('\n'):
                # Match test result lines (e.g., "ERROR: test_unicode (__main__.ArrayReconstructorTest.test_unicode)")
                match = re.match(r'(PASSED|FAIL|ERROR):\s+(\w+)\s+\(__main__\.(\w+)\.(\w+)\)', line)
                if match:
                    status, method_name, class_name, _ = match.groups()
                    test_key = f"{class_name}.{method_name}"
                    test_results[test_key] = status == "PASSED"

            return test_results

        except Exception as e:
            print(f"Error running tests: {e}")
            return {}


class ClassHierarchyVisitor(cst.CSTVisitor):
    """Visitor to build class hierarchy and method definitions"""

    def __init__(self):
        self.current_class: Optional[str] = None
        self.class_hierarchy: Dict[str, Set[str]] = {}
        self.method_definitions: Dict[str, str] = {}
        self.class_bases: Dict[str, List[str]] = {}

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        class_name = node.name.value
        self.current_class = class_name

        # Extract base classes
        bases = []
        for arg in node.bases:
            if isinstance(arg.value, cst.Name):
                bases.append(arg.value.value)
            elif isinstance(arg.value, cst.Attribute):
                # Handle cases like unittest.TestCase
                bases.append(self._get_full_name(arg.value))

        self.class_bases[class_name] = bases

        # Build hierarchy (reverse mapping)
        for base in bases:
            if base not in self.class_hierarchy:
                self.class_hierarchy[base] = set()
            self.class_hierarchy[base].add(class_name)

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        self.current_class = None

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        if self.current_class and node.name.value.startswith('test_'):
            method_key = f"{self.current_class}.{node.name.value}"
            self.method_definitions[method_key] = self.current_class

    def _get_full_name(self, node: cst.Attribute) -> str:
        """Get full name from attribute node"""
        parts = []
        current: cst.BaseExpression = node
        while isinstance(current, cst.Attribute):
            parts.append(current.attr.value)
            current = current.value
        if isinstance(current, cst.Name):
            parts.append(current.value)
        return '.'.join(reversed(parts))


class TestDecoratorTransformer(cst.CSTTransformer):
    """Transformer to add decorators and comments to failing tests"""

    def __init__(self, methods_to_decorate: Dict[str, Set[str]]):
        self.methods_to_decorate = methods_to_decorate
        self.current_class: Optional[str] = None
        self.added_decorators: Set[str] = set()

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.current_class = node.name.value

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        self.current_class = None
        return updated_node

    def leave_FunctionDef(self, original_node: cst.FunctionDef, node: cst.FunctionDef) -> cst.FunctionDef:
        if not self.current_class:
            return node

        method_name = node.name.value
        if not method_name.startswith('test_'):
            return node

        # Check if this method needs decoration
        if method_name in self.methods_to_decorate.get(self.current_class, set()):
            # Check if already has expectedFailure decorator
            has_decorator = any(
                isinstance(d.decorator, cst.Name) and d.decorator.value == "expectedFailure"
                or isinstance(d.decorator, cst.Attribute) and d.decorator.attr.value == "expectedFailure"
                for d in node.decorators
            )

            if not has_decorator:
                # Add TODO comment
                todo_comment = cst.EmptyLine(
                    comment=cst.Comment("# TODO: RUSTPYTHON")
                )

                # Add expectedFailure decorator
                decorator = cst.Decorator(
                    decorator=cst.Attribute(
                        value=cst.Name("unittest"),
                        attr=cst.Name("expectedFailure")
                    )
                )

                # Create new decorators list
                new_decorators = list(node.decorators) + [decorator]

                # Add comment before the method
                new_leading_lines = [cst.EmptyLine(), todo_comment] + list(line for line in node.leading_lines if not isinstance(line, cst.EmptyLine))

                return node.with_changes(
                    decorators=new_decorators,
                    leading_lines=new_leading_lines
                )

        return node


class OverrideMethodTransformer(cst.CSTTransformer):
    """Transformer to add override methods in child classes"""

    def __init__(self, overrides_to_add: Dict[str, Set[str]]):
        self.overrides_to_add = overrides_to_add
        self.current_class: Optional[str] = None

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self.current_class = node.name.value

    def leave_ClassDef(self, original_node: cst.ClassDef, node: cst.ClassDef) -> cst.ClassDef:
        if self.current_class in self.overrides_to_add:
            methods_to_override = self.overrides_to_add[self.current_class]

            # Create override methods
            new_methods = []
            for method_name in methods_to_override:
                # Create a simple override method that calls super()
                override_method = cst.FunctionDef(
                    name=cst.Name(method_name),
                    params=cst.Parameters([
                        cst.Param(name=cst.Name("self"))
                    ]),
                    body=cst.IndentedBlock([
                        cst.SimpleStatementLine([
                            cst.Expr(
                                cst.Call(
                                    func=cst.Attribute(
                                        value=cst.Call(
                                            func=cst.Name("super")
                                        ),
                                        attr=cst.Name(method_name)
                                    )
                                )
                            )
                        ])
                    ]),
                    decorators=[
                        cst.Decorator(
                            decorator=cst.Attribute(
                                value=cst.Name("unittest"),
                                attr=cst.Name("expectedFailure")
                            )
                        )
                    ],
                    leading_lines=[
                        cst.EmptyLine(
                            whitespace=cst.SimpleWhitespace("    "),
                            comment=cst.Comment("# TODO RUSTPYTHON: Fix this test")
                        )
                    ]
                )
                new_methods.append(override_method)

            # Add new methods to the class body
            if new_methods:
                new_body = list(node.body.body) + new_methods
                return node.with_changes(
                    body=node.body.with_changes(body=new_body)
                )

        self.current_class = None
        return node


def analyze_test_failures(test_file: str, test_results: Dict[str, bool],
                         hierarchy_visitor: ClassHierarchyVisitor) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Analyze test failures and determine where to place decorators.
    Returns: (methods_to_decorate, overrides_to_add)
    """
    methods_to_decorate: Dict[str, Set[str]] = {}
    overrides_to_add: Dict[str, Set[str]] = {}

    # Group failures by method name
    method_failures: Dict[str, Set[str]] = {}  # method -> set of failing classes
    for test_key, passed in test_results.items():
        if not passed:
            class_name, method_name = test_key.split('.')
            if method_name not in method_failures:
                method_failures[method_name] = set()
            method_failures[method_name].add(class_name)

    # For each failing method, determine where to place the decorator
    for method_name, failing_classes in method_failures.items():
        # Find all classes that have this method
        all_classes_with_method = set()
        for test_key in hierarchy_visitor.method_definitions:
            if test_key.endswith(f".{method_name}"):
                class_name = test_key.split('.')[0]
                all_classes_with_method.add(class_name)

        # Check if all child classes of a parent fail
        for parent_class in hierarchy_visitor.class_hierarchy:
            children = hierarchy_visitor.class_hierarchy[parent_class]
            children_with_method = children & all_classes_with_method

            if children_with_method and children_with_method.issubset(failing_classes):
                # All children with this method fail - decorate parent
                if parent_class not in methods_to_decorate:
                    methods_to_decorate[parent_class] = set()
                methods_to_decorate[parent_class].add(method_name)

                # Remove from failing_classes as we've handled them
                failing_classes -= children_with_method

        # Handle remaining failures - need overrides in specific child classes
        for class_name in failing_classes:
            # Check if method is defined in this class or inherited
            test_key = f"{class_name}.{method_name}"
            if test_key in hierarchy_visitor.method_definitions:
                defining_class = hierarchy_visitor.method_definitions[test_key]
                if defining_class == class_name:
                    # Method defined in this class - decorate it
                    if class_name not in methods_to_decorate:
                        methods_to_decorate[class_name] = set()
                    methods_to_decorate[class_name].add(method_name)
                else:
                    # Method inherited - need override
                    if class_name not in overrides_to_add:
                        overrides_to_add[class_name] = set()
                    overrides_to_add[class_name].add(method_name)

    return methods_to_decorate, overrides_to_add


def process_test_file(file_path: str, dry_run: bool = False) -> None:
    """Process a single test file"""
    print(f"Processing {file_path}...")

    # Read the file
    with open(file_path, 'r') as f:
        source_code = f.read()

    # Parse with libcst
    module = cst.parse_module(source_code)

    # Build class hierarchy
    hierarchy_visitor = ClassHierarchyVisitor()
    module.visit(hierarchy_visitor)

    # Run tests and collect failures
    analyzer = TestFailureAnalyzer()
    test_results = analyzer.run_tests(file_path)

    if not test_results:
        print("No test results found. Make sure the file contains valid unittest tests.")
        return

    # Analyze failures and determine decorator placement
    methods_to_decorate, overrides_to_add = analyze_test_failures(
        file_path, test_results, hierarchy_visitor
    )

    # Apply transformations
    if methods_to_decorate:
        module = module.visit(TestDecoratorTransformer(methods_to_decorate))

    if overrides_to_add:
        module = module.visit(OverrideMethodTransformer(overrides_to_add))

    # Write back the modified code
    if dry_run:
        print("Dry run - would make the following changes:")
        print(f"Methods to decorate: {methods_to_decorate}")
        print(f"Override methods to add: {overrides_to_add}")
    else:
        with open(file_path, 'w') as f:
            f.write(module.code)
        print(f"Updated {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Add @expectedFailure decorators to failing RustPython tests"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Test files to process"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files"
    )

    args = parser.parse_args()

    for file_path in args.files:
        if Path(file_path).exists():
            process_test_file(file_path, args.dry_run)
        else:
            print(f"File not found: {file_path}")


if __name__ == "__main__":
    main()
