#!/usr/bin/env python3
# test_bandwidth_analysis.py：验证算法输出和实验方法的边界条件。
# 对应 DESIGN.md 第 9 节。
# 配置、数据来源及缺失状态由统一框架记录。
"""Bandwidth provenance and merge traffic must stay separate from sorting work."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "bandwidth_summary", ROOT / "analysis/summarize.py"
)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
sys.path.insert(0, str(ROOT / "scripts"))
import bench


class BandwidthTests(unittest.TestCase):
    def test_merge_traffic_is_only_defined_for_psrs_merge(self):
        data = pd.DataFrame(
            [
                dict(impl=impl, variant=variant, n=1_000_000, merge_ms=ms)
                for impl, variant, ms in [
                    ("omp", "psrs", 2),
                    ("mpi", "allgather", 4),
                    ("omp", "local", 2),
                    ("pipeline", "block", 2),
                    ("gpu", "sample", 2),
                    ("hybrid", "psrs", 0),
                ]
            ]
        )
        actual = summary.merge_bandwidth(data)
        self.assertEqual(actual.iloc[:2].tolist(), [4, 2])
        self.assertTrue(actual.iloc[2:].isna().all())

    def test_stream_reader_rejects_unverified_and_nonfinite_without_mode_mixing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            common = dict(
                threads=128, n=1 << 28, bind="close", kernel="triad", median_GBps=100
            )
            for filename, rows in [
                (
                    "E14.csv",
                    [
                        dict(common, best_GBps=120, verified=1),
                        dict(common, best_GBps=999, verified=0),
                        dict(common, best_GBps=float("inf"), verified=1),
                    ],
                ),
                ("E14_quick.csv", [dict(common, best_GBps=300, verified=1)]),
            ]:
                pd.DataFrame(rows).to_csv(root / "results" / filename, index=False)
            with patch.object(summary, "ROOT", root):
                data = summary.read_bandwidth()
        self.assertEqual(len(data), 2)
        self.assertEqual(data.loc[~data.quick, "best_GBps"].tolist(), [120])
        self.assertEqual(data.loc[data.quick, "best_GBps"].tolist(), [300])

    def test_full_and_quick_stream_matrix_both_cover_all_threads_and_bindings(self):
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        for quick in (False, True):
            with contextlib.redirect_stdout(io.StringIO()):
                bench.main(["E14", "--dry-run"] + (["--quick"] if quick else []))
            configs = bench.plan["configurations"]
            self.assertEqual(len(configs), 16)
            self.assertEqual({c["n"] for c in configs}, {1 << (20 if quick else 28)})
            self.assertEqual(
                {(c["threads"], c["bind"]) for c in configs},
                {
                    (p, b)
                    for p in (1, 2, 4, 8, 16, 32, 64, 128)
                    for b in ("close", "spread")
                },
            )
            self.assertTrue(all(c["repeats"] == 5 for c in configs))
            self.assertTrue(
                all(
                    c["environment"]["OMP_PLACES"] == "threads"
                    for c in configs
                    if c["threads"] == 128
                )
            )


if __name__ == "__main__":
    unittest.main()
