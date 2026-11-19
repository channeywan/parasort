#!/usr/bin/env python3
"""Exercise the real STREAM tool, its provenance, and rejection of invalid runs."""
import csv
import io
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StreamBandwidthTests(unittest.TestCase):
    def run_tool(self, arguments, bind="close", **environment):
        env = dict(os.environ, OMP_PROC_BIND=bind, OMP_PLACES="cores")
        env.update(environment)
        return subprocess.run(
            [str(ROOT / "bin/stream_bw"), *arguments],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )

    def test_verified_rows_and_binding_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "bandwidth.csv"
            for bind in ("close", "spread"):
                with self.subTest(bind=bind):
                    result = self.run_tool(
                        ["--n", "65537", "--threads", "2", "--repeat", "5", "--csv", str(destination)],
                        bind,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    rows = list(csv.DictReader(io.StringIO(result.stdout)))
                    self.assertEqual({row["kernel"] for row in rows}, {"copy", "triad"})
                    self.assertEqual(len(rows), 2)
                    for row in rows:
                        self.assertEqual(row["threads"], "2")
                        self.assertEqual(row["n"], "65537")
                        self.assertEqual(row["repeats"], "5")
                        self.assertEqual(row["warmups"], "1")
                        self.assertEqual(row["verified"], "1")
                        self.assertEqual(row["bind"], bind)
                        self.assertEqual(row["places"], "cores")
                        self.assertEqual(row["bind_requested"], bind)
                        self.assertTrue(row["host"])
                        self.assertTrue(row["timestamp"].endswith("Z"))
                        best, median = float(row["best_GBps"]), float(row["median_GBps"])
                        self.assertTrue(math.isfinite(best) and math.isfinite(median))
                        self.assertGreater(median, 0)
                        self.assertGreaterEqual(best, median)
            with destination.open() as source:
                stored = list(csv.DictReader(source))
            self.assertEqual(len(stored), 4)
            self.assertEqual({row["bind"] for row in stored}, {"close", "spread"})

    def test_size_notation_and_one_element_boundaries(self):
        for size in ("1", "2^10"):
            result = self.run_tool(["--n", size, "--threads", "1", "--repeat", "2", "--csv", ""])
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = list(csv.DictReader(io.StringIO(result.stdout)))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["verified"] == "1" for row in rows))
            self.assertEqual(rows[0]["n"], "1" if size == "1" else "1024")

    def test_invalid_parameters_never_emit_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "must_not_exist.csv"
            for arguments in (["--n", "0"], ["--threads", "0"], ["--repeat", "0"], ["--n", "-1"], ["--n", "2^64"], ["--unknown"]):
                with self.subTest(arguments=arguments):
                    result = self.run_tool([*arguments, "--csv", str(destination)])
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(result.stdout)
                    self.assertFalse(destination.exists())

    def test_runtime_team_limit_is_not_reported_as_requested_threads(self):
        result = self.run_tool(
            ["--n", "1024", "--threads", "2", "--repeat", "1", "--csv", ""],
            OMP_THREAD_LIMIT="1",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("OpenMP created 1 threads; requested 2", result.stderr)
        self.assertFalse(result.stdout)

    def test_incompatible_existing_csv_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "existing.csv"
            destination.write_text("unrelated,data\n1,2\n")
            result = self.run_tool(["--n", "1024", "--threads", "1", "--repeat", "1", "--csv", str(destination)])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("different header", result.stderr)
            self.assertEqual(destination.read_text(), "unrelated,data\n1,2\n")
            self.assertFalse(result.stdout)


if __name__ == "__main__":
    unittest.main()
