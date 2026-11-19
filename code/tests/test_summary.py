#!/usr/bin/env python3
# test_summary.py：验证算法输出和实验方法的边界条件。
# 对应 DESIGN.md 第 9 节。
# 配置、数据来源及缺失状态由统一框架记录。
"""数据可信性回归：没有绑定元数据的 MPI 测量不可参与性能统计。"""

import importlib.util
import tempfile
import unittest
from pathlib import Path
import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    "parasort_summary", Path(__file__).resolve().parents[1] / "analysis/summarize.py"
)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class BindingDataTests(unittest.TestCase):
    def test_failed_gpu_binding_is_not_merged_with_successful_request(self):
        records = []
        for actual, elapsed in (("numa", 10), ("none(setaffinity_22)", 100)):
            records.append(
                dict(
                    experiment="E11",
                    quick=False,
                    host="test-host",
                    impl="gpu_multi",
                    variant="sample",
                    dist="uniform",
                    n=1 << 30,
                    procs=1,
                    threads=2,
                    gpus=2,
                    total_ms=elapsed,
                    verified=1,
                    extra="gpu_affinity=numa;gpu_affinity_actual=" + actual,
                )
            )
        data = SUMMARY.summarize(pd.DataFrame(records))
        self.assertEqual(len(data), 2)
        self.assertEqual(
            dict(zip(data.gpu_affinity_actual, data.median_ms)),
            {"numa": 10, "none(setaffinity_22)": 100},
        )
        self.assertTrue((data.repeats == 1).all())

    def test_pre_fix_rows_are_excluded_and_binding_modes_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = SUMMARY.ROOT
            SUMMARY.ROOT = Path(directory)
            (SUMMARY.ROOT / "results").mkdir()
            records = []
            for impl, binding, verified in [
                ("mpi", "", 1),
                ("hybrid", "", 1),
                ("pipeline", "", 1),
                ("mpi", "core", 1),
                ("mpi", "hwthread", 1),
                ("mpi", "core", 0),
                ("omp", "", 1),
            ]:
                records.append(
                    {
                        "impl": impl,
                        "verified": verified,
                        "extra": "seed=42"
                        + (";mpi_bind=" + binding if binding else ""),
                    }
                )
            pd.DataFrame(records).to_csv(SUMMARY.ROOT / "results/E2.csv", index=False)
            try:
                actual = SUMMARY.read_results()
                self.assertEqual(len(actual), 3)
                self.assertEqual(
                    set(actual[actual.impl == "mpi"].mpi_bind), {"core", "hwthread"}
                )
                self.assertIn("mpi_bind", SUMMARY.TUNING)
            finally:
                SUMMARY.ROOT = previous


if __name__ == "__main__":
    unittest.main()
