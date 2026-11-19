#!/usr/bin/env python3
"""数据可信性回归：没有绑定元数据的 MPI 测量不可参与性能统计。"""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import pandas as pd

SPEC = importlib.util.spec_from_file_location("parasort_summary", Path(__file__).resolve().parents[1] / "analysis/summarize.py")
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)

class BindingDataTests(unittest.TestCase):
    def test_pre_fix_rows_are_excluded_and_binding_modes_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = SUMMARY.ROOT
            SUMMARY.ROOT = Path(directory)
            (SUMMARY.ROOT / "results").mkdir()
            records = []
            for impl, binding, verified in [("mpi", "", 1), ("hybrid", "", 1), ("pipeline", "", 1), ("mpi", "core", 1), ("mpi", "hwthread", 1), ("mpi", "core", 0), ("omp", "", 1)]:
                records.append({"impl": impl, "verified": verified, "extra": "seed=42" + (";mpi_bind=" + binding if binding else "")})
            pd.DataFrame(records).to_csv(SUMMARY.ROOT / "results/E2.csv", index=False)
            try:
                actual = SUMMARY.read_results()
                self.assertEqual(len(actual), 3)
                self.assertEqual(set(actual[actual.impl == "mpi"].mpi_bind), {"core", "hwthread"})
                self.assertIn("mpi_bind", SUMMARY.TUNING)
            finally:
                SUMMARY.ROOT = previous

if __name__ == "__main__":
    unittest.main()
