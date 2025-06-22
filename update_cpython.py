#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "libcst",
# ]
# ///
# -*- coding: utf-8 -*-

"""
Script to update RustPython files using CPython source code.

Usage:
    uv run update_cpython.py [version] [path1] [path2] ... [--no-mark-failures]

    version: CPython version (e.g., 3.13.5)
    path: Path to update (e.g., Lib/os.py or Lib/test/test_os.py)
    --no-mark-failures: Do not add @unittest.expectedFailure decorator to failing tests

Environment Variables:
    LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               Default: INFO

Examples:
    # Basic usage with default logging
    uv run update_cpython.py 3.13.0 Lib/os.py

    # Enable debug logging for performance analysis
    LOG_LEVEL=DEBUG uv run update_cpython.py 3.13.0 Lib/os.py

    # Quiet mode (only errors)
    LOG_LEVEL=ERROR uv run update_cpython.py 3.13.0 Lib/os.py
"""

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import NewType
from pathlib import Path

import libcst as cst


def setup_logger():
    """Setup logger with environment variable control."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Convert string level to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    level = level_map.get(log_level, logging.INFO)

    # Configure logging
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    return logging.getLogger(__name__)


logger = setup_logger()


@contextmanager
def time_logger(operation_name: str, log_level: int = logging.DEBUG):
    """Context manager for timing and logging operations."""
    start_time = time.time()
    logger.log(log_level, f"Starting {operation_name}")

    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        logger.log(
            log_level, f"Completed {operation_name} in {elapsed_time:.2f} seconds"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Update RustPython files using CPython source code."
    )
    parser.add_argument("version", help="CPython version (e.g., 3.13.0)")
    parser.add_argument(
        "paths", nargs="+", help="Paths to update (e.g., Lib/os.py Lib/test/test_os.py)"
    )
    parser.add_argument(
        "--no-mark-failures",
        action="store_true",
        help="Do not add @unittest.expectedFailure to failing tests",
    )

    args = parser.parse_args()

    # Validate version format
    if not re.match(r"^\d+\.\d+\.\d+$", args.version):
        parser.error(
            "Version must be in the format 'number.number.number' (e.g., 3.13.0)"
        )

    return args


def clone_cpython(version: str) -> str:
    """Clone CPython repository and checkout a specific version."""
    with time_logger(f"clone CPython {version}"):
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"cpython_{version}_")
        logger.debug(f"Created temporary directory: {temp_dir}")

        try:
            # Clone repository
            tag = f"v{version}"
            logger.debug(f"Cloning repository with tag: {tag}")

            with time_logger("git clone", logging.DEBUG):
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "--branch",
                        tag,
                        "https://github.com/python/cpython.git",
                        temp_dir,
                    ],
                    check=True,
                    capture_output=True,
                )

            return temp_dir
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone CPython repository: {e}")
            logger.debug(f"Git clone stderr: {e.stderr.decode()}")
            shutil.rmtree(temp_dir)
            sys.exit(1)


def get_file_list(directory: str, rel_path: str) -> set[str]:
    """Get a list of files from the directory."""
    logger.debug(f"Getting file list from {directory}/{rel_path}")

    base_path = Path(directory) / rel_path

    if not base_path.exists():
        logger.debug(f"Path does not exist: {base_path}")
        return set()

    if base_path.is_file():
        logger.debug(f"Single file found: {base_path}")
        return {str(base_path.relative_to(directory))}

    result = set()
    with time_logger(f"scan files in {rel_path}", logging.DEBUG):
        for path in base_path.glob("**/*"):
            if path.is_file():
                result.add(str(path.relative_to(directory)))

    logger.debug(f"Found {len(result)} files")
    return result


def run_test_and_get_failures(test_file: str) -> list[str]:
    """Run tests and return names of failing test cases."""
    with time_logger(f"run test: {test_file}"):
        try:
            logger.debug(f"Executing: cargo run -- {test_file}")

            with time_logger("test execution", logging.DEBUG):
                result = subprocess.run(
                    ["cargo", "run", "--", test_file], capture_output=True, text=True
                )

            # Extract failing test case names
            failures = []
            for line in result.stderr.splitlines():
                # Check all lines starting with FAIL or ERROR
                match = re.search(r"(?:FAIL|ERROR): ([A-Za-z0-9_\.]+) \((.+)\)", line)
                if match:
                    # Extract test name and class info
                    test_name = match.group(1)  # Simple method name
                    full_class_name = match.group(2)  # module.class.method format

                    # Remove __main__. prefix if it exists
                    if full_class_name.startswith("__main__."):
                        full_class_name = full_class_name[9:]

                    # Use class.method format if class exists, otherwise just method
                    if "." in full_class_name:
                        failures.append(full_class_name)
                    else:
                        failures.append(test_name)

            logger.info(f"Found {len(failures)} failing tests")
            logger.debug(f"Failing tests: {failures}")

            return failures
        except subprocess.CalledProcessError as e:
            logger.error(f"Test execution failed: {e}")
            return []


class TestModifier(cst.CSTTransformer):
    """LibCST transformer to add decorators to failing test methods"""

    def __init__(self, failures: list[str]):
        super().__init__()
        self.failures = failures

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        class_name = original_node.name.value

        # Check function definitions in the class and add decorators to failing methods
        new_body = []
        for item in updated_node.body.body:
            if isinstance(item, cst.FunctionDef):
                method_name = item.name.value
                class_method = f"{class_name}.{method_name}"

                # Check if in failure list
                if class_method in self.failures or method_name in self.failures:
                    # Add unittest.expectedFailure decorator
                    expected_failure = cst.Decorator(
                        decorator=cst.Attribute(
                            value=cst.Name("unittest"), attr=cst.Name("expectedFailure")
                        )
                    )

                    # Add RUSTPYTHON marker
                    todo_rustpython_comment = cst.EmptyLine(
                        comment=cst.Comment("# TODO: RUSTPYTHON")
                    )

                    # Add new decorators to existing decorators
                    new_decorators = list(
                        (
                            todo_rustpython_comment,
                            expected_failure,
                        )
                    ) + list(item.decorators)

                    # Create modified function definition
                    item = item.with_changes(decorators=new_decorators)

            new_body.append(item)

        # Update ClassDef with modified body
        return updated_node.with_changes(
            body=updated_node.body.with_changes(body=new_body)
        )


def mark_failing_tests(test_file: str, failing_tests: list[str]) -> bool:
    """Add decorators to failing test cases."""
    if not failing_tests:
        logger.debug(f"No failing tests to mark in {test_file}")
        return False

    with time_logger(f"mark {len(failing_tests)} failing tests in {test_file}"):
        logger.debug(f"Tests to mark: {failing_tests}")

        with open(test_file, "r", encoding="utf-8") as f:
            file_content = f.read()

        logger.debug(f"Read file content ({len(file_content)} characters)")

        # Parse file using LibCST
        with time_logger("parse file", logging.DEBUG):
            tree = cst.parse_module(file_content)

        # Apply transformation
        with time_logger("apply transformations", logging.DEBUG):
            modifier = TestModifier(failing_tests)
            modified_tree = tree.visit(modifier)

        with time_logger("write modified file", logging.DEBUG):
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(modified_tree.code)

    return False


ProcessResult = NewType("ProcessResult", list[str])


def process_path(
    update_path: str, cpython_dir: str, mark_failures: bool
) -> ProcessResult:
    """Process a single path and return list of updated test files."""
    with time_logger(f"process path '{update_path}'"):
        test_files = []

        rel_path = update_path
        if rel_path.startswith("Lib/"):
            rel_path = rel_path[4:]  # Remove "Lib/"

        logger.debug(f"Getting file lists for {update_path}")
        cpython_files = get_file_list(cpython_dir, update_path)
        rustpython_files = get_file_list(".", update_path)

        to_add = cpython_files - rustpython_files
        to_remove = rustpython_files - cpython_files
        to_update = cpython_files.intersection(rustpython_files)

        logger.info(
            f"Files to add: {len(to_add)}, remove: {len(to_remove)}, update: {len(to_update)}"
        )

        logger.info(f"Summary of changes for path '{update_path}':")
        logger.info(f"Files to add: {len(to_add)}")
        for file in sorted(to_add):
            logger.debug(f"  + {file}")

        logger.info(f"Files to remove: {len(to_remove)}")
        for file in sorted(to_remove):
            logger.debug(f"  - {file}")

        logger.info(f"Files to update: {len(to_update)}")
        for file in sorted(to_update):
            logger.debug(f"  ~ {file}")

        # Remove files
        if to_remove:
            with time_logger(f"remove {len(to_remove)} files", logging.DEBUG):
                for file in to_remove:
                    file_path = Path(file)
                    logger.debug(f"Deleting file: {file_path}")
                    file_path.unlink()

        # Add and update files
        if to_add or to_update:
            with time_logger(
                f"copy {len(to_add) + len(to_update)} files", logging.DEBUG
            ):
                for file in to_add.union(to_update):
                    cpython_file = Path(cpython_dir) / file
                    rustpython_file = Path(update_path)

                    if update_path.startswith("Lib/") and file.startswith("Lib/"):
                        rustpython_file = Path(file)
                    elif update_path.startswith("Lib/"):
                        rustpython_file = Path("Lib") / file

                    rustpython_file.parent.mkdir(parents=True, exist_ok=True)

                    logger.debug(f"Copying: {cpython_file} -> {rustpython_file}")
                    shutil.copy2(cpython_file, rustpython_file)

                    if re.match(r".*test_.*\.py$", str(rustpython_file)):
                        test_files.append(str(rustpython_file))

        logger.info(f"Found {len(test_files)} test files")
        return ProcessResult(test_files)


def check_project_root():
    if not Path("Cargo.toml").exists() or not Path("Lib").is_dir():
        logger.error("Not in RustPython project root directory")
        sys.exit(1)


def check_paths_exist(update_paths: list[str]):
    for update_path in update_paths:
        rustpython_path = Path(update_path)
        if not rustpython_path.exists() and not rustpython_path.parent.exists():
            logger.error(f"Path does not exist: {update_path}")
            sys.exit(1)


def main():
    args = parse_args()
    cpython_version = args.version
    update_paths = args.paths
    mark_failures = not args.no_mark_failures

    logger.info(f"Starting update process for CPython {cpython_version}")
    logger.info(f"Paths to update: {update_paths}")
    logger.debug(f"Mark failures: {mark_failures}")

    with time_logger(f"entire update process for CPython {cpython_version}"):
        # Check that current working directory is the RustPython project root
        check_project_root()

        # Check that the paths to update exist
        check_paths_exist(update_paths)

        # Clone CPython source code - only clone once
        cpython_dir = clone_cpython(cpython_version)

        try:
            all_test_files = []

            # Process each path
            with time_logger("process all paths"):
                for update_path in update_paths:
                    logger.info(f"===== Processing path '{update_path}' =====")
                    test_files = process_path(update_path, cpython_dir, mark_failures)
                    all_test_files.extend(test_files)

            # Run tests and mark failures - run once after processing all paths
            if mark_failures and all_test_files:
                logger.info("===== Running tests and marking failing test cases =====")
                logger.info(f"Running tests for {len(all_test_files)} test files")

                with time_logger("test processing"):
                    total_failures = 0
                    for test_file in all_test_files:
                        failing_tests = run_test_and_get_failures(test_file)
                        total_failures += len(failing_tests)
                        if failing_tests:
                            mark_failing_tests(test_file, failing_tests)

                    logger.debug(f"Total failures: {total_failures}")

        finally:
            # Clean up temporary directory
            with time_logger("cleanup", logging.DEBUG):
                logger.debug(f"Cleaning up temporary directory: {cpython_dir}")
                shutil.rmtree(cpython_dir)


if __name__ == "__main__":
    main()
