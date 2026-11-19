#!/usr/bin/env python3
# test_coverage_matrix.py：验证算法输出和实验方法的边界条件。
# 对应 DESIGN.md 第 9 节。
# 配置、数据来源及缺失状态由统一框架记录。
"""The preview matrix must cover the prescribed SMT, NUMA and Scatter pairs."""

import contextlib
import io
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bench


class CoverageMatrixTests(unittest.TestCase):
    def preview(self, experiment, quick=False):
        previous = Path.cwd()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                bench.main([experiment, "--dry-run"] + (["--quick"] if quick else []))
            return bench.plan["configurations"]
        finally:
            os.chdir(previous)

    def test_e2_smt_places_and_two_hybrid_topologies(self):
        configs = self.preview("E2")
        omp = [
            c for c in configs if c["executable"] == "psrs_omp" and c["threads"] == 128
        ]
        self.assertEqual(len(omp), 1)
        self.assertEqual(omp[0]["environment"]["OMP_PLACES"], "threads")
        hybrids = [c for c in configs if c["executable"] == "psrs_hybrid"]
        for p in (16, 32, 64, 128):
            self.assertEqual(
                {c["procs"] for c in hybrids if c["procs"] * c["threads"] == p}, {2, 8}
            )

    def test_e3_weak_scaling_includes_bucket_and_eight_rank_hybrid(self):
        configs = self.preview("E3")
        for p in (1, 2, 4, 8, 16, 32, 64, 128):
            self.assertTrue(
                any(
                    c["executable"] == "bucket_omp"
                    and c["threads"] == p
                    and c["n"] == (1 << 24) * p
                    for c in configs
                )
            )
        hybrids = [c for c in configs if c["executable"] == "psrs_hybrid"]
        self.assertEqual(
            {(c["procs"], c["threads"], c["n"]) for c in hybrids},
            {(8, p // 8, (1 << 24) * p) for p in (8, 16, 32, 64, 128)},
        )

    def test_e9_both_modes_keep_scatter_local_pairs_and_all_hybrid_ranks(self):
        for quick in (False, True):
            configs = self.preview("E9", quick)
            self.assertEqual(
                {c["procs"] for c in configs if c["executable"] == "psrs_hybrid"},
                {2, 4, 8},
            )
            for p in (8, 64):
                pair = [
                    c
                    for c in configs
                    if c["executable"] == "psrs_mpi"
                    and c["procs"] == p
                    and c["variant"] in (None, "psrs")
                ]
                self.assertEqual(
                    {("--scatter" in c["options"]) for c in pair}, {False, True}
                )
                self.assertEqual(len(pair), 2)
                self.assertEqual(
                    len({(c["n"], c["dist"], c["threads"]) for c in pair}), 1
                )


if __name__ == "__main__":
    unittest.main()
