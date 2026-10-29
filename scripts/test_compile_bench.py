"""Contract tests for the compile-only benchmark, without timing thresholds."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import compile_bench as bench


class CompileBenchmarkTests(unittest.TestCase):
    def test_generated_workloads_compile_at_small_and_large_sizes(self):
        for name in bench.SYNTHETIC_SIZES:
            for size in (1, 8):
                with self.subTest(name=name, size=size):
                    code = compile(bench.synthetic_source(name, size), "<test>", "exec")
                    self.assertIsNotNone(code)

    def test_measurement_compiles_but_does_not_execute_source(self):
        samples = bench.measure_source(
            "raise RuntimeError('must not execute')", "<test>", 3, 1
        )
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(value > 0 for value in samples))

    def test_summary_keeps_raw_samples_and_uses_median_for_ratios(self):
        row = bench.summarize_case(
            {"id": "assigns/2", "size": 2, "workload": "assigns", "source_bytes": 13},
            {"cpython": [10, 20, 30], "base": [100, 200, 300], "head": [40, 50, 90]},
        )
        self.assertEqual(row["timings"]["head"]["median_ns"], 50)
        self.assertEqual(row["timings"]["head"]["samples_ns"], [40, 50, 90])
        self.assertEqual(row["base_over_head"], 4.0)
        self.assertEqual(row["head_over_cpython"], 2.5)

    def test_failed_measurement_cannot_appear_as_a_speedup(self):
        row = bench.summarize_case(
            {"id": "assigns/1", "size": 1, "workload": "assigns", "source_bytes": 6},
            {"cpython": [10], "base": [], "head": [5]},
        )
        self.assertIsNone(row["base_over_head"])
        self.assertIsNone(row["timings"]["base"])

    def test_worker_reports_syntax_error_without_dropping_case(self):
        payload = {
            "cases": [{"id": "broken", "source": "def :", "filename": "<broken>"}],
            "samples": 1,
            "warmups": 0,
        }
        result = subprocess.run(
            [sys.executable, str(Path(bench.__file__).resolve()), "--worker"],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=True,
        )
        rows = json.loads(result.stdout)["results"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "broken")
        self.assertEqual(rows[0]["status"], "error")
        self.assertIn("SyntaxError", rows[0]["error"])

    def test_cli_runs_real_interpreters_and_writes_reproducible_results(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(bench.__file__).resolve()),
                    "--cpython",
                    sys.executable,
                    "--base",
                    sys.executable,
                    "--head",
                    sys.executable,
                    "--workloads",
                    "assigns",
                    "--sizes",
                    "2,4",
                    "--rounds",
                    "3",
                    "--samples",
                    "1",
                    "--warmups",
                    "0",
                    "--output",
                    directory,
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads((Path(directory) / "results.json").read_text())
            report = (Path(directory) / "report.md").read_text()
        self.assertEqual(len(data["cases"]), 2)
        self.assertEqual(
            data["round_orders"],
            [
                ["cpython", "base", "head"],
                ["base", "head", "cpython"],
                ["head", "cpython", "base"],
            ],
        )
        self.assertEqual(len(data["rows"][0]["timings"]["base"]["samples_ns"]), 3)
        self.assertIn("Growth", report)
        self.assertIn("assigns/4", report)
        self.assertIn("sha256", data["interpreters"]["base"])
        self.assertEqual(data["cwd"], str(Path.cwd()))

    def test_unexecutable_target_is_reported_without_losing_other_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = root / "not-an-executable"
            invalid.write_text("not a Python interpreter\n")
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(bench.__file__).resolve()),
                    "--cpython",
                    sys.executable,
                    "--base",
                    str(invalid),
                    "--head",
                    sys.executable,
                    "--workloads",
                    "assigns",
                    "--sizes",
                    "1",
                    "--rounds",
                    "1",
                    "--samples",
                    "1",
                    "--warmups",
                    "0",
                    "--output",
                    str(root / "results"),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 1)
            data = json.loads((root / "results" / "results.json").read_text())
        self.assertTrue(data["errors"])
        self.assertIsNone(data["rows"][0]["timings"]["base"])
        self.assertIsNotNone(data["rows"][0]["timings"]["head"])
        self.assertIsNone(data["rows"][0]["base_over_head"])


if __name__ == "__main__":
    unittest.main()
