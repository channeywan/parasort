#!/usr/bin/env python3
# test_plot.py：验证算法输出和实验方法的边界条件。
# 对应 DESIGN.md 第 9 节。
# 配置、数据来源及缺失状态由统一框架记录。
"""Plot regressions cover configuration identity, data modes and output provenance."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

CODE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "tested_plot", CODE / "analysis" / "plot.py"
)
plot = importlib.util.module_from_spec(spec)
with tempfile.TemporaryDirectory() as cache, patch.dict(
    os.environ, {"MPLCONFIGDIR": cache}
):
    spec.loader.exec_module(plot)
from summarize import DEFAULTS


def row(impl="omp", p=32, ranks=1, quick=False, experiment="E2", host="a", **overrides):
    procs = p if impl == "mpi" else ranks if impl == "hybrid" else 1
    threads = 1 if impl == "mpi" else p // procs
    values = dict(
        DEFAULTS,
        experiment=experiment,
        quick=quick,
        host=host,
        impl=impl,
        variant="psrs",
        dist="uniform",
        n=1 << 30,
        procs=procs,
        threads=threads,
        gpus=0,
        units=p,
        median_ms=120 / p,
        min_ms=100 / p,
        max_ms=140 / p,
        baseline_ms=120,
        speedup=p,
        efficiency=1,
        karp_flatt=0.01,
        bind="false" if impl == "mpi" else "close",
        places="threads" if impl != "mpi" and threads > 64 // procs else "cores",
        launcher="mpich" if impl in ("mpi", "hybrid") else "none",
        mpi_bind=(
            ("core" if p <= 64 else "hwthread")
            if impl == "mpi"
            else (
                {2: "socket", 8: "numa"}.get(procs, "none")
                if impl == "hybrid"
                else "none"
            )
        ),
        scatter_ms=0,
        compute_ms=3,
        merge_GBps=10,
    )
    values.update({field: 1.0 for field in plot.PHASES})
    values.update(overrides)
    return values


class PlotTests(unittest.TestCase):
    def tearDown(self):
        plot.plt.close("all")

    def test_scaling_curves_keep_rank_families_and_custom_contexts(self):
        rows = [row("mpi", p) for p in (32, 64, 128)]
        rows += [
            row("hybrid", p, ranks=ranks) for ranks in (2, 8) for p in (32, 64, 128)
        ]
        rows += [row("mpi", p, host="b") for p in (32, 64, 128)]
        fig, ax = plot.plt.subplots()
        plot.curves(pd.DataFrame(rows), "speedup", ax, "加速比")
        groups = [
            container for container in ax.containers if hasattr(container, "lines")
        ]
        self.assertEqual(len(groups), 4)
        self.assertTrue(
            all(
                list(container.lines[0].get_xdata()) == [32, 64, 128]
                for container in groups
            )
        )
        names = [container.get_label() for container in groups]
        self.assertTrue(any("2 个 MPI 进程" in name for name in names))
        self.assertTrue(any("8 个 MPI 进程" in name for name in names))
        self.assertTrue(any("主机=b" in name for name in names))
        self.assertEqual(ax.get_xscale(), "log")
        ideal = next(line for line in ax.lines if line.get_label() == "理想加速比 y=p")
        self.assertEqual(list(ideal.get_xdata()), list(ideal.get_ydata()))
        self.assertEqual(ideal.get_xdata()[-1], 128)

    def test_e14_quick_never_uses_full_stream_or_sort_overlay(self):
        stream = pd.DataFrame(
            [
                dict(
                    threads=8,
                    host="a",
                    n=1 << 20,
                    bind="close",
                    bind_requested="close",
                    kernel="copy",
                    places="cores",
                    best_GBps=value,
                    quick=quick,
                )
                for quick, value in ((True, 100), (False, 999))
            ]
        )
        data = pd.DataFrame(
            [
                row("omp", 8, quick=False, merge_GBps=777),
                row("omp", 8, quick=True, merge_GBps=888),
            ]
        )
        snapshots = {}

        def capture(fig, name):
            snapshots[name] = [
                (line.get_label(), list(line.get_ydata())) for line in fig.axes[0].lines
            ]

        with patch.object(plot, "save", side_effect=capture):
            plot.plot_e14(data, stream, True)
            plot.plot_e14(data, stream, False)
        self.assertEqual([values for _, values in snapshots["E14_quick"]], [[100]])
        full = [value for _, values in snapshots["E14"] for value in values]
        self.assertIn(999, full)
        self.assertIn(777, full)
        self.assertNotIn(888, full)

    def test_scatter_component_and_gpu_phase_breakdown_remain_visible(self):
        data = pd.DataFrame(
            [
                row(
                    "mpi",
                    p,
                    experiment="E9",
                    scatter=str(scatter),
                    scatter_ms=25 if scatter else 0,
                )
                for p in (8, 64)
                for scatter in (0, 1)
            ]
        )
        captured = {}
        with patch.object(
            plot, "save", side_effect=lambda fig, name: captured.setdefault(name, fig)
        ):
            plot.plot_e9(data, False)
        self.assertIn(25, [bar.get_height() for bar in captured["E9"].axes[3].patches])
        self.assertIn("未计入", captured["E9"]._parasort["caption"])
        gpu = pd.DataFrame(
            [
                row(
                    "gpu_multi",
                    2,
                    experiment="E11",
                    gpus=2,
                    variant="sample",
                    gpu_affinity=affinity,
                    exchange_path="pinned_host",
                )
                for affinity in ("numa", "none")
            ]
        )
        with patch.object(
            plot, "save", side_effect=lambda fig, name: captured.setdefault(name, fig)
        ):
            plot.plot_e11(gpu, False)
        self.assertIn("E11_phases", captured)
        self.assertGreaterEqual(
            len(captured["E11_phases"].axes[0].patches), len(plot.PHASES) * 2
        )
        names = [line.get_label() for line in captured["E11"].axes[0].lines]
        self.assertTrue(any("GPU 本地 NUMA" in name for name in names))
        self.assertTrue(any("不绑定" in name for name in names))
        self.assertTrue(all("锁页主机内存中转" in name for name in names))

    def test_save_routes_quick_images_and_registers_actual_files(self):
        with tempfile.TemporaryDirectory() as directory:
            figures = Path(directory) / "figures"
            registry = plot.FigureRegistry(figures)
            with patch.object(plot, "DEST", figures), patch.object(
                plot, "REGISTRY", registry
            ):
                fig, ax = plot.plt.subplots()
                ax.plot([1, 2], [1, 2], label="实测加速比")
                plot.describe(fig, "E2", True)
                plot.save(fig, "E2_quick")
                registry.write()
            self.assertFalse((figures / "E2_quick.png").exists())
            self.assertTrue((figures / "quick" / "E2_quick.png").is_file())
            records = json.loads((figures / "manifest.json").read_text())
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["quick"], True)
            self.assertEqual(
                set(records[0]),
                {"paths", "title", "experiment", "quick", "source_table", "series"},
            )
            self.assertTrue(
                all((Path(directory) / path).is_file() for path in records[0]["paths"])
            )


if __name__ == "__main__":
    unittest.main()
