#!/usr/bin/env python3
"""Fail when an example project resolves a crate to two incompatible versions
that the workspace resolves to one.

Usage: check_example_lockfiles.py WORKSPACE_LOCK EXAMPLE_LOCK...
"""

import re
import sys

import tomllib


def compatible_versions(lockfile: str) -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    with open(lockfile, "rb") as f:
        packages = tomllib.load(f)["package"]
    for package in packages:
        if not package.get("source", "").startswith("registry+"):
            continue
        major, minor, patch = re.match(
            r"(\d+)\.(\d+)\.(\d+)", package["version"]
        ).groups()
        if major != "0":
            series = major
        elif minor != "0":
            series = f"0.{minor}"
        else:
            series = f"0.0.{patch}"
        found.setdefault(package["name"], {})[series] = package["version"]
    return found


def main() -> int:
    workspace_lock, *example_locks = sys.argv[1:]
    workspace = compatible_versions(workspace_lock)
    failed = False
    for example_lock in example_locks:
        for name, versions in compatible_versions(example_lock).items():
            if len(versions) > 1 and len(workspace.get(name, {})) <= 1:
                found = ", ".join(versions.values())
                expected = ", ".join(workspace.get(name, {}).values()) or "none"
                print(
                    f"{example_lock}: {name} resolves to {found}; "
                    f"{workspace_lock} has {expected}"
                )
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
