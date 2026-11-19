#!/usr/bin/env python3
# test_e4b_analysis.py：验证算法输出和实验方法的边界条件。
# 对应 DESIGN.md 第 9 节。
# 配置、数据来源及缺失状态由统一框架记录。
"""E4b analysis keeps distinct measured contexts and handles CSV tuning types."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd


def load_analysis(name):
    path = Path(__file__).resolve().parents[1] / "analysis" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"e4b_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


with tempfile.TemporaryDirectory() as cache:
    with patch.dict(os.environ, {"MPLCONFIGDIR": cache}):
        PLOT = load_analysis("plot")
SUMMARY = load_analysis("summarize")


def record(dist="hotkeys", p=64, k=1, tie=1, host="host-a", quick=False, **extra):
    offset = 10 if host == "host-b" else 0
    row = {
        "experiment": "E4b",
        "quick": quick,
        "host": host,
        "impl": "omp",
        "variant": "psrs",
        "dist": dist,
        "n": 1 << (18 if quick else 28),
        "procs": 1,
        "threads": p,
        "gpus": 0,
        "units": p,
        "seed": "42",
        "oversample": str(float(k)),
        "tiebreak": str(float(tie)),
        "median_ms": 100 / (k + 1) + offset,
        "splitters_ms": k / 10 + offset,
        "imbalance": 1 + 1 / k + offset,
    }
    row.update(extra)
    return row


def matrix(quick=False):
    rows = {}
    for host in ("host-a", "host-b"):
        for dist in ("uniform", "normal", "hotkeys"):
            for k in (1, 2, 4, 8, 16, 32):
                rows[host, dist, 64, k, 1] = record(dist, 64, k, 1, host, quick)
        for dist in ("hotkeys", "few_unique"):
            for p in (8, 16, 32, 64, 128):
                for tie in (0, 1):
                    # The common hotkeys/p=64/k=1/tie-on measurement is reused.
                    rows[host, dist, p, 1, tie] = record(dist, p, 1, tie, host, quick)
    return list(rows.values())


class E4bAnalysisTests(unittest.TestCase):
    def test_scans_select_exact_scale_mode_and_numeric_tuning_values(self):
        rows = matrix()
        rows += [
            record(n=1 << 27),
            record(impl="seq"),
            record(variant="serial_init"),
            record(procs=2),
            record(gpus=1),
            record(units=8),
            record(k=3),
            record(dist="power"),
            record(quick=True, n=1 << 28),
        ]
        over, tie = PLOT.e4b_subsets(pd.DataFrame(rows), False)
        self.assertEqual(len(over), 36)
        self.assertEqual(len(tie), 40)
        self.assertEqual(set(over.oversample), {1, 2, 4, 8, 16, 32})
        self.assertEqual(set(tie.threads), {8, 16, 32, 64, 128})
        self.assertEqual(set(tie.tiebreak), {0, 1})
        for quick in (False, True):
            data = pd.DataFrame(matrix(False) + matrix(True))
            over, tie = PLOT.e4b_subsets(data, quick)
            self.assertEqual(set(over.n), {1 << (18 if quick else 28)})
            self.assertEqual(set(tie.quick), {quick})

    def test_plots_keep_hosts_separate_and_write_both_formats(self):
        rows = matrix()
        rows += [record(k=2, n=1 << 27, median_ms=9999, splitters_ms=9999)]
        snapshots = {}
        real_save = PLOT.save

        def save(fig, name):
            snapshots[name] = [
                [
                    (line.get_label(), list(line.get_xdata()), list(line.get_ydata()))
                    for line in ax.lines
                ]
                for ax in fig.axes
            ]
            real_save(fig, name)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(PLOT, "DEST", Path(directory)), patch.object(
                PLOT, "save", side_effect=save
            ):
                PLOT.plot_e4b(pd.DataFrame(rows), False)
            expected = {
                "E4b_oversample.png",
                "E4b_oversample.pdf",
                "E4b_tiebreak.png",
                "E4b_tiebreak.pdf",
            }
            self.assertEqual(
                {path.name for path in Path(directory).iterdir()}, expected
            )
            self.assertTrue(
                all(path.stat().st_size > 1000 for path in Path(directory).iterdir())
            )
        axes = snapshots["E4b_oversample"]
        self.assertEqual(len(axes), 7)  # Four panels and three independent time axes.
        first_panel = axes[0]
        self.assertEqual(len(first_panel), 2)
        self.assertTrue(any("host-a" in name for name, _, _ in first_panel))
        self.assertTrue(any("host-b" in name for name, _, _ in first_panel))
        self.assertEqual(
            {tuple(values) for _, _, values in first_panel},
            {
                tuple(1 + 1 / k for k in (1, 2, 4, 8, 16, 32)),
                tuple(11 + 1 / k for k in (1, 2, 4, 8, 16, 32)),
            },
        )
        self.assertEqual(len(axes[3]), 6)  # Three distributions on each host.
        self.assertFalse(any(9999 in values for ax in axes for _, _, values in ax))
        for ax in snapshots["E4b_tiebreak"]:
            self.assertEqual(len(ax), 4)  # On/off for each host, without pooling.
            self.assertTrue(all(points == [8, 16, 32, 64, 128] for _, points, _ in ax))

    def test_e4b_table_has_eight_chinese_columns_and_explicit_tuning_keys(self):
        table = SUMMARY.e4b_table(
            pd.DataFrame(
                [
                    record(p=8, tie=0),
                    record(p=64, k=32, tie=1),
                ]
            )
        )
        self.assertEqual(len(table.columns), 8)
        self.assertEqual(set(table["过采样 k"]), {1, 32})
        self.assertEqual(set(table["处理单元 p"]), {8, 64})
        self.assertEqual(set(table["复合键"]), {"开启", "关闭"})
        self.assertIn("分割点时间 (ms)", table.columns)


if __name__ == "__main__":
    unittest.main()
