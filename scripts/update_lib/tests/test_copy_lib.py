"""Tests for copy_lib.py - library copying with dependencies."""

import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import ANY, patch


class TestCopySingle(unittest.TestCase):
    """Tests for _copy_single helper function."""

    def test_copies_file(self):
        """Test copying a single file."""
        from update_lib.cmd_copy_lib import _copy_single

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            src = tmpdir / "source.py"
            src.write_text("content")
            dst = tmpdir / "dest.py"

            _copy_single(src, dst, verbose=False)

            self.assertTrue(dst.exists())
            self.assertEqual(dst.read_text(), "content")

    def test_copies_directory(self):
        """Test copying a directory."""
        from update_lib.cmd_copy_lib import _copy_single

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            src = tmpdir / "source_dir"
            src.mkdir()
            (src / "file.py").write_text("content")
            dst = tmpdir / "dest_dir"

            _copy_single(src, dst, verbose=False)

            self.assertTrue(dst.exists())
            self.assertTrue((dst / "file.py").exists())

    def test_removes_existing_before_copy(self):
        """Test that existing destination is removed before copy."""
        from update_lib.cmd_copy_lib import _copy_single

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            src = tmpdir / "source.py"
            src.write_text("new content")
            dst = tmpdir / "dest.py"
            dst.write_text("old content")

            _copy_single(src, dst, verbose=False)

            self.assertEqual(dst.read_text(), "new content")


class TestCopyLib(unittest.TestCase):
    """Tests for copy_lib function."""

    def test_raises_on_path_without_lib(self):
        """Test that copy_lib raises ValueError when path doesn't contain /Lib/."""
        from update_lib.cmd_copy_lib import copy_lib

        with self.assertRaises(ValueError) as ctx:
            copy_lib(pathlib.Path("some/path/without/lib.py"))

        self.assertIn("/Lib/", str(ctx.exception))


class TestGenerateFiles(unittest.TestCase):
    """Tests for repository-owned generated module dependencies."""

    @patch("update_lib.cmd_copy_lib.subprocess.run")
    def test_opcode_generates_python_and_rust_metadata(self, mock_run):
        from update_lib.cmd_copy_lib import _generate_files

        repo_root = pathlib.Path(__file__).parents[3].resolve()
        cpython_root = pathlib.Path("cpython").resolve()
        outputs = _generate_files("opcode", "cpython", verbose=False)

        self.assertEqual(
            outputs,
            [
                repo_root / "Lib/_opcode_metadata.py",
                repo_root / "crates/compiler-core/src/bytecode/opcode_metadata.rs",
            ],
        )
        self.assertEqual(mock_run.call_count, 2)
        mock_run.assert_any_call(
            [
                sys.executable,
                "tools/opcode_metadata/generate_py_opcode_metadata.py",
            ],
            cwd=repo_root,
            env=ANY,
            check=True,
        )
        mock_run.assert_any_call(
            [
                sys.executable,
                "tools/opcode_metadata/generate_rs_opcode_metadata.py",
            ],
            cwd=repo_root,
            env=ANY,
            check=True,
        )
        for call in mock_run.call_args_list:
            self.assertEqual(call.kwargs["env"]["CPYTHON_ROOT"], str(cpython_root))

    @patch("update_lib.cmd_copy_lib.subprocess.run")
    def test_generator_failure_is_reported(self, mock_run):
        from update_lib.cmd_copy_lib import _generate_files

        mock_run.side_effect = subprocess.CalledProcessError(1, "generator")

        with self.assertRaises(subprocess.CalledProcessError):
            _generate_files("opcode", "cpython", verbose=False)


if __name__ == "__main__":
    unittest.main()
