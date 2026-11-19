#!/usr/bin/env python3
# test_export_for_report.py：验证算法输出和实验方法的边界条件。
# 对应 DESIGN.md 第 9 节。
# 配置、数据来源及缺失状态由统一框架记录。
"""Synthetic-data tests for the report exporter's provenance and selection rules."""

import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "analysis/export_for_report.py"
if not MODULE_PATH.exists():
    MODULE_PATH = Path(__file__).with_name("export_for_report.py")
SPEC = importlib.util.spec_from_file_location("export_for_report", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Exporter = MODULE.Exporter
PHASES = MODULE.PHASES


def row(
    impl="omp",
    variant="psrs",
    n=1 << 30,
    procs=1,
    threads=8,
    total=10,
    verified=1,
    **extra,
):
    settings = dict(
        seed="42",
        oversample="1",
        buckets_per_thread="1",
        schedule="static",
        tiebreak="1",
        bind="close",
        batch="0",
        naive_atomics="0",
        scatter="0",
        mempolicy="default",
        mpi_bind="none",
        launcher="none",
        places="cores",
        gpu_affinity="legacy",
        exchange_path="none",
    )
    if impl in {"mpi", "hybrid", "pipeline"}:
        settings.update(
            mpi_bind=(
                "core" if impl != "hybrid" else ("socket" if procs == 2 else "numa")
            ),
            launcher="mpich",
            bind="false" if impl != "hybrid" else "close",
        )
    host = extra.pop("host", "host-a")
    dist = extra.pop("dist", "uniform")
    gpus = extra.pop("gpus", 0)
    settings.update({key: str(value) for key, value in extra.items()})
    result = dict(
        timestamp="2026-11-19T00:00:00Z",
        host=host,
        impl=impl,
        variant=variant,
        dist=dist,
        n=n,
        procs=procs,
        threads=threads,
        gpus=gpus,
        repeat_id=0,
        total_ms=total,
        imbalance=1.1,
        verified=verified,
        extra=";".join(f"{key}={value}" for key, value in settings.items()),
    )
    result.update({field: 1.0 for field in PHASES})
    return result


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="parasort-export-test-")
        self.root = Path(self.temp.name)
        self.results = self.root / "results"
        self.tables = self.root / "report/tables"
        self.output = self.root / "report/data_bundle"
        self.analysis = self.root / "analysis"
        for path in (self.results, self.tables, self.analysis):
            path.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, rows):
        path = self.results / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def export(self, quick=False):
        exporter = Exporter(
            self.results, self.tables, self.output, quick, analysis=self.analysis
        )
        numbers = exporter.run()
        self.assertEqual(
            numbers, json.loads((self.output / "key_numbers.json").read_text())
        )
        return numbers

    def test_missing_full_never_fills_from_quick(self):
        self.write(
            "E2_quick.csv",
            [
                row(impl="seq", variant="std", n=262144, threads=1, total=100),
                row(n=262144, total=10),
            ],
        )
        numbers = self.export()
        for key in MODULE.REQUIRED:
            if key != "affinity_fix":
                self.assertIsNone(numbers[key]["value"], key)
        self.assertEqual(numbers["sources"], [])
        self.assertEqual(numbers["mode"], "full")
        self.assertTrue(numbers["missing"])
        quick = self.export(quick=True)
        self.assertEqual(quick["mode"], "quick")
        self.assertEqual(quick["best_cpu"]["value"]["total_ms"]["value"], 10)
        self.assertTrue(quick["best_cpu"]["value"]["total_ms"]["quick"])
        self.assertIsNone(quick["seq_std_2p30_ms"]["value"])

    def test_gpu_binding_failure_is_a_separate_measurement_not_numa_success(self):
        self.write(
            "E11.csv",
            [
                row(
                    impl="gpu_multi",
                    variant="sample",
                    threads=2,
                    gpus=2,
                    total=elapsed,
                    gpu_affinity="numa",
                    gpu_affinity_actual=actual,
                    exchange_path="pinned_host",
                )
                for actual, elapsed in (("numa", 10), ("none(setaffinity_22)", 100))
            ],
        )
        numbers = self.export()
        gpu = numbers["e11_gpu"]["value"]
        measured = gpu["measurements"]
        self.assertEqual(len(measured), 2)
        self.assertEqual(
            {
                item["config"]["gpu_affinity_actual"]: item["total_ms"]["value"]
                for item in measured
            },
            {"numa": 10, "none(setaffinity_22)": 100},
        )
        comparison = gpu["affinity"][0]
        self.assertEqual(comparison["numa"]["total_ms"]["value"], 10)
        self.assertEqual(comparison["numa"]["config"]["gpu_affinity_actual"], "numa")
        self.assertIsNone(comparison["none"]["value"])
        self.assertNotIn("none_over_numa", comparison)

    def test_failed_gpu_binding_alone_cannot_supply_the_numa_ablation(self):
        self.write(
            "E11.csv",
            [
                row(
                    impl="gpu_multi",
                    variant="sample",
                    threads=2,
                    gpus=2,
                    total=100,
                    gpu_affinity="numa",
                    gpu_affinity_actual="none(numa_node_unknown)",
                )
            ],
        )
        gpu = self.export()["e11_gpu"]["value"]
        self.assertEqual(len(gpu["measurements"]), 1)
        self.assertIsNone(gpu["affinity"][0]["numa"]["value"])
        self.assertEqual(gpu["dual_vs_single_thrust"], [])

    def test_verified_filter_and_mode_separation(self):
        raw = [
            row(impl="seq", variant="std", threads=1, total=100),
            row(total=10),
            row(total=12),
            row(total=0.0001, verified=0),
        ]
        self.write("E2.csv", raw)
        self.write("E2_quick.csv", [row(total=0.000001)])
        numbers = self.export()
        metric = numbers["best_cpu"]["value"]["total_ms"]
        self.assertEqual(metric["value"], 11)
        self.assertEqual(metric["raw_sources"][0]["rows"], [3, 4])
        self.assertFalse(metric["quick"])
        self.assertAlmostEqual(
            numbers["best_cpu"]["value"]["speedup"]["value"], 100 / 11
        )
        self.assertEqual(numbers["sources"][0]["rows_verified"], 3)
        self.assertEqual(numbers["sources"][0]["rows_excluded"], 1)

    def test_hybrid_topologies_stay_distinct_for_64_128(self):
        self.write(
            "E2.csv",
            [
                row(impl="seq", variant="std", threads=1, total=1000),
                row(impl="hybrid", procs=2, threads=32, total=20),
                row(impl="hybrid", procs=2, threads=64, total=10, places="threads"),
                row(impl="hybrid", procs=8, threads=8, total=25),
                row(impl="hybrid", procs=8, threads=16, total=5, places="threads"),
                row(impl="mpi", procs=64, threads=1, total=30),
                row(impl="mpi", procs=128, threads=1, total=20, mpi_bind="hwthread"),
            ],
        )
        numbers = self.export()
        comparisons = numbers["speedup_64_vs_128"]["value"]
        self.assertEqual(len(comparisons), 3)
        hybrid = {
            entry["family"]["procs"]: entry
            for entry in comparisons
            if entry["family"]["impl"] == "hybrid"
        }
        self.assertEqual(hybrid[2]["speedup_128_over_64"]["value"], 2)
        self.assertEqual(hybrid[8]["speedup_128_over_64"]["value"], 5)
        mpi = next(entry for entry in comparisons if entry["family"]["impl"] == "mpi")
        self.assertEqual(mpi["speedup_128_over_64"]["value"], 1.5)
        self.assertEqual(mpi["p128"]["config"]["mpi_bind"], "hwthread")

    def test_unmatched_baseline_does_not_cross_host_or_seed(self):
        self.write(
            "E2.csv",
            [
                row(impl="seq", variant="std", threads=1, host="host-b", total=100),
                row(impl="seq", variant="std", threads=1, total=100, seed="43"),
                row(total=10),
            ],
        )
        numbers = self.export()
        record = numbers["best_cpu"]["value"]
        self.assertIsNone(record["speedup"]["value"])
        self.assertTrue(
            any(item["path"] == "best_cpu.speedup" for item in numbers["missing"])
        )

    def test_legacy_mpi_excluded(self):
        self.write(
            "E2.csv",
            [
                row(impl="mpi", procs=8, threads=1, total=0.001, mpi_bind="none"),
                row(total=10),
            ],
        )
        numbers = self.export()
        self.assertEqual(numbers["best_cpu"]["value"]["config"]["impl"], "omp")
        self.assertTrue(any("mpi_bind" in issue for issue in numbers["load_issues"]))

    def test_bandwidth_verified_positive_rows_only(self):
        base = dict(
            threads=128,
            bind="close",
            kernel="Triad",
            best_GBps=150.0,
            median_GBps=140.0,
            n=1 << 28,
            repeats=5,
            warmups=1,
            host="host-a",
            places="threads",
            timestamp="2026-11-19",
            verified=1,
        )
        self.write(
            "E14.csv",
            [base, dict(base, best_GBps=9999, verified=0), dict(base, best_GBps=-1)],
        )
        merge = row(n=1000000, total=10)
        merge["merge_ms"] = 2
        self.write("E5.csv", [merge])
        numbers = self.export()
        bandwidth = numbers["e14_bandwidth"]["value"]
        self.assertEqual(len(bandwidth["stream"]), 1)
        self.assertEqual(bandwidth["stream"][0]["best_GBps"]["value"], 150)
        self.assertEqual(bandwidth["psrs_merge"][0]["value"], 4)
        self.assertEqual(bandwidth["psrs_merge"][0]["raw_sources"][0]["rows"], [2])

    def test_scatter_time_separate_from_sort(self):
        self.write(
            "E9.csv",
            [
                row(
                    impl="mpi",
                    procs=8,
                    threads=1,
                    total=10,
                    scatter="0",
                    scatter_ms="0",
                ),
                row(
                    impl="mpi", procs=8, threads=1, total=9, scatter="1", scatter_ms="4"
                ),
            ],
        )
        numbers = self.export()
        records = numbers["e9_comm"]["value"]["measurements"]
        scatter = next(entry for entry in records if entry["config"]["scatter"] == "1")
        self.assertEqual(scatter["total_ms"]["value"], 9)
        self.assertEqual(scatter["scatter_ms"]["value"], 4)
        self.assertEqual(scatter["sort_plus_scatter_ms"]["value"], 13)
        self.assertIn("excludes generation", scatter["sort_plus_scatter_ms"]["formula"])

    def test_five_numbers_reproduce_from_raw_and_table_sources(self):
        self.write(
            "E2.csv",
            [
                row(impl="seq", variant="std", threads=1, total=100),
                row(total=10),
                row(total=12),
            ],
        )
        (self.tables / "E2.md").write_text(
            "| median_ms | speedup | efficiency | imbalance |\n|---|---|---|---|\n|100|1|1|1.1|\n|11|9.0909|1.13636|1.1|\n"
        )
        numbers = self.export()
        best = numbers["best_cpu"]["value"]
        leaves = [
            numbers["seq_std_2p30_ms"],
            best["total_ms"],
            best["speedup"],
            best["efficiency"],
            best["imbalance"],
        ]
        expected = [100, 11, 100 / 11, 100 / 11 / 8, 1.1]
        with (self.results / "E2.csv").open() as source:
            raw = list(csv.DictReader(source))
        for leaf, value in zip(leaves, expected):
            self.assertAlmostEqual(leaf["value"], value)
            self.assertEqual(leaf["source"], "tables/E2.md")
            for source in leaf["raw_sources"]:
                for line in source["rows"]:
                    self.assertEqual(raw[line - 2]["verified"], "1")

    def test_registry_compact_tables_and_report_preserved(self):
        self.write("E2.csv", [row(total=10)])
        report = self.tables.parent / "report.md"
        report.write_text("Untouched report body.\n")
        figures = self.tables.parent / "figures"
        figures.mkdir()
        (figures / "manifest.json").write_text(
            json.dumps(
                [
                    dict(
                        paths=["figures/E2.png", "figures/E2.pdf"],
                        title="强扩展",
                        experiment="E2",
                        quick=False,
                        source_table="tables/E2.md",
                        series=["MPI PSRS", "OpenMP PSRS"],
                    ),
                    dict(
                        paths=["figures/quick/E2_quick.png"],
                        title="快速",
                        experiment="E2",
                        quick=True,
                        source_table="tables/E2_quick.md",
                        series=["快速曲线"],
                    ),
                ]
            )
        )
        self.export()
        self.assertEqual(report.read_text(), "Untouched report body.\n")
        manifest = (self.output / "figures_manifest.md").read_text()
        self.assertIn("tables/E2.md", manifest)
        self.assertIn("MPI PSRS", manifest)
        self.assertNotIn("E2_quick", manifest)
        for path in (self.output / "tables").glob("*.md"):
            for line in path.read_text().splitlines():
                if line.startswith("|"):
                    self.assertLessEqual(len(line.split("|")) - 2, 8, path)

    def test_latest_fit_point_adapter_ignores_stale_fit_json(self):
        self.write(
            "E2.csv",
            [
                row(impl="seq", variant="std", threads=1, total=100),
                row(threads=1, total=100),
                row(threads=2, total=60),
                row(threads=4, total=40),
            ],
        )
        (self.tables / "E13_fits.json").write_text('[{"f":0.999,"sse":9999}]')
        (self.analysis / "summarize.py").write_text("""
def fit_models(summary):
    data=summary[(summary.impl=='omp')&(summary.variant=='psrs')].sort_values('units')
    keys=['experiment','quick','host','impl','variant','dist','n','procs','threads','gpus','seed','oversample','buckets_per_thread','schedule','tiebreak','bind','batch','naive_atomics','scatter','mempolicy','mpi_bind','launcher','places','gpu_affinity','gpu_affinity_requested','exchange_path','sweep']
    points=[{'p':int(row.units),'time_ms':float(row.median_ms),'baseline_ms':float(row.baseline_ms),'config':{key:row[key] for key in keys}} for _,row in data.iterrows()]
    return [{'quick':False,'host':'host-a','impl':'omp','variant':'psrs','dist':'uniform','n':1073741824,'f':0.2,'alpha_ms':0.0,'sse':0.0,'points':len(points),'hybrid_ranks':0,'bind_policy':'close','places_policy':'automatic','data_points':points}]
""")
        numbers = self.export()
        fit = numbers["e13_amdahl"]["value"][0]
        self.assertEqual(fit["f"]["value"], 0.2)
        self.assertEqual(fit["rmse_ms"]["value"], 0)
        self.assertEqual(len(fit["points"]), 3)
        self.assertEqual(fit["f"]["raw_sources"][0]["rows"], [2, 3, 4, 5])
        self.assertIsNone(fit["points"][0]["karp_flatt"]["value"])

    def test_all_comparison_sections_populate_from_matched_fixtures(self):
        self.write(
            "E4.csv",
            [
                row(variant="local", dist="normal", total=20),
                row(dist="normal", total=10),
                row(variant="local", dist="power", total=30),
                row(dist="power", total=10),
                row(dist="hotkeys", total=20, tiebreak="0"),
                row(dist="hotkeys", total=10),
                row(dist="few_unique", total=20, tiebreak="0"),
                row(dist="few_unique", total=10),
            ],
        )
        self.write(
            "E4b.csv",
            [
                row(dist=dist, threads=64, total=10, oversample=k)
                for dist in ("uniform", "normal", "hotkeys")
                for k in (1, 2, 4, 8, 16, 32)
            ]
            + [
                row(dist=dist, threads=64, total=20, tiebreak="0")
                for dist in ("hotkeys", "few_unique")
            ]
            + [row(dist="few_unique", threads=64, total=10)],
        )
        self.write(
            "E6.csv",
            [
                row(dist="normal", total=20),
                row(
                    dist="normal", total=10, buckets_per_thread="4", schedule="dynamic"
                ),
            ],
        )
        self.write(
            "E7.csv",
            [
                row(total=10),
                row(variant="serial_init", total=20),
                row(total=12, bind="spread"),
            ],
        )
        self.write(
            "E8.csv",
            [
                row(impl="pthreads", variant="locked", threads=4, total=30),
                row(impl="pthreads", variant="local", threads=4, total=10),
                row(impl="pthreads", variant="locked", threads=8, total=50),
            ],
        )
        self.write(
            "E10.csv",
            [
                row(impl="pipeline", variant="elem", n=4, procs=4, threads=1, total=1),
                row(
                    impl="pipeline",
                    variant="block",
                    n=1000,
                    procs=4,
                    threads=1,
                    total=20,
                    batch="64",
                ),
                row(impl="mpi", n=1000, procs=4, threads=1, total=10),
            ],
        )
        self.write(
            "E11.csv",
            [
                row(
                    impl="gpu",
                    variant="thrust",
                    threads=1,
                    gpus=1,
                    total=20,
                    compute_ms="15",
                    gpu_affinity="numa",
                    gpu_affinity_actual="numa",
                ),
                row(
                    impl="gpu_multi",
                    variant="sample",
                    threads=2,
                    gpus=2,
                    total=10,
                    compute_ms="5",
                    gpu_affinity="numa",
                    gpu_affinity_actual="numa",
                    exchange_path="pinned_host",
                ),
                row(
                    impl="gpu_multi",
                    variant="sample",
                    threads=2,
                    gpus=2,
                    total=15,
                    compute_ms="8",
                    gpu_affinity="none",
                    gpu_affinity_actual="none(disabled)",
                    exchange_path="pinned_host",
                ),
            ],
        )
        self.write(
            "E12.csv",
            [row(impl="seq", variant="std", threads=1, total=100), row(total=10)],
        )
        self.write(
            "E1.csv",
            [
                row(impl="mpi", procs=2, threads=1, total=1, mpi_bind="none"),
                row(verified=0),
            ],
        )
        numbers = self.export()
        self.assertEqual(len(numbers["e4_skew"]["value"]), 2)
        self.assertEqual(
            numbers["e4_skew"]["value"][0]["total_ms_left_over_right"]["value"], 2
        )
        self.assertEqual(len(numbers["e4b_oversample_best"]["value"]), 3)
        self.assertEqual(
            numbers["e6_best_schedule"]["value"][0]["relative_to_static_k1"]["value"], 2
        )
        self.assertEqual(
            numbers["e10_pipeline"]["value"]["best_block_by_n_p"][0][
                "block_over_psrs_time"
            ]["value"],
            2,
        )
        self.assertEqual(
            numbers["e11_gpu"]["value"]["dual_vs_single_thrust"][0]["value"], 2
        )
        self.assertEqual(
            numbers["e11_gpu"]["value"]["affinity"][0]["none_over_numa"]["value"], 1.5
        )
        self.assertEqual(
            next(
                record
                for record in numbers["e11_gpu"]["value"]["measurements"]
                if record["config"]["gpus"] == 2
            )["exchange_paths"]["value"],
            ["pinned_host"],
        )
        self.assertEqual(numbers["correctness"]["value"]["passed"]["value"], 1)
        for key in (
            "e4_skew",
            "e4_hotkeys_tiebreak",
            "e4b_oversample_best",
            "e4b_tiebreak_vs_p",
            "e6_best_schedule",
            "e7_numa",
            "e8_lock",
            "e10_pipeline",
            "e11_gpu",
            "e12_overview",
            "correctness",
        ):
            self.assertIsNotNone(numbers[key]["value"], key)

    def test_live_summarizer_api_preserves_transitions_and_hybrid_topology(self):
        analysis = Path(
            os.environ.get("PARASORT_LIVE_ANALYSIS", str(MODULE_PATH.parent))
        )
        if not (analysis / "summarize.py").exists():
            self.skipTest("live summarizer not installed beside staged exporter")
        rows = [row(impl="seq", variant="std", threads=1, total=1000)]
        rows += [
            row(
                impl="mpi",
                procs=p,
                threads=1,
                total=1000 / p + 20,
                mpi_bind="core" if p <= 64 else "hwthread",
            )
            for p in (8, 16, 32, 64, 128)
        ]
        rows += [
            row(
                impl="hybrid",
                procs=2,
                threads=p // 2,
                total=1000 / p + 10,
                places="threads" if p > 64 else "cores",
            )
            for p in (16, 64, 128)
        ]
        rows += [
            row(
                impl="hybrid",
                procs=8,
                threads=p // 8,
                total=1000 / p + 5,
                places="threads" if p > 64 else "cores",
            )
            for p in (16, 64, 128)
        ]
        self.write("E2.csv", rows)
        exporter = Exporter(self.results, self.tables, self.output, analysis=analysis)
        numbers = exporter.run()
        fits = numbers["e13_amdahl"]["value"]
        self.assertTrue(fits)
        mpi = next(fit for fit in fits if fit["config"]["impl"] == "mpi")
        self.assertEqual(
            {point["p"]["value"] for point in mpi["points"]}, {8, 16, 32, 64, 128}
        )
        self.assertEqual(mpi["config"]["bind_policy"], "mpi_auto_core_to_hwthread")
        self.assertEqual(mpi["points"][-1]["config"]["mpi_bind"], "hwthread")
        hybrids = {
            fit["config"]["hybrid_ranks"]: fit
            for fit in fits
            if fit["config"]["impl"] == "hybrid"
        }
        self.assertEqual(set(hybrids), {2, 8})
        for ranks, fit in hybrids.items():
            self.assertTrue(
                all(point["config"]["procs"] == ranks for point in fit["points"])
            )
        ratios = numbers["speedup_64_vs_128"]["value"]
        self.assertEqual(len(ratios), 3)
        self.assertTrue(
            all(entry["speedup_128_over_64"]["value"] is not None for entry in ratios)
        )

    def test_explicit_and_user_bindings_are_not_paired_across_64_128(self):
        analysis = Path(
            os.environ.get("PARASORT_LIVE_ANALYSIS", str(MODULE_PATH.parent))
        )
        for live in (False, True):
            if live and not (analysis / "summarize.py").exists():
                continue
            self.write(
                "E2.csv",
                [
                    row(impl="mpi", procs=64, threads=1, total=30),
                    row(impl="mpi", procs=128, threads=1, total=20, mpi_bind="user"),
                    row(threads=64, total=30),
                    row(threads=128, total=20, places="{0}"),
                ],
            )
            exporter = Exporter(
                self.results,
                self.tables,
                self.output,
                analysis=analysis if live else self.analysis,
            )
            numbers = exporter.run()
            comparisons = numbers["speedup_64_vs_128"]["value"]
            self.assertEqual(len(comparisons), 4)
            self.assertTrue(
                all(
                    entry["speedup_128_over_64"]["value"] is None
                    for entry in comparisons
                )
            )
            self.write(
                "E2.csv",
                [
                    row(impl="mpi", procs=64, threads=1, total=30, mpi_bind="user"),
                    row(impl="mpi", procs=128, threads=1, total=20, mpi_bind="user"),
                ],
            )
            exporter = Exporter(
                self.results,
                self.tables,
                self.output,
                analysis=analysis if live else self.analysis,
            )
            comparisons = exporter.run()["speedup_64_vs_128"]["value"]
            self.assertEqual(len(comparisons), 2)
            self.assertTrue(
                all(
                    entry["speedup_128_over_64"]["value"] is None
                    for entry in comparisons
                )
            )

    def test_structured_environment_gpu_os_numa_and_link(self):
        snapshot = """PRETTY_NAME="Ubuntu 22.04.5 LTS"
Linux host-a 5.15.0 x86_64 GNU/Linux
Model name: AMD EPYC 7542
CPU(s): 128
Core(s) per socket: 32
Socket(s): 2
Thread(s) per core: 2
NUMA node(s): 8
NUMA node2 CPU(s): 16-23,80-87
Mem: 251Gi 10Gi 241Gi
HYDRA build details:
    Version: 5.0.2
|   0  NVIDIA GeForce RTX 4090 D      Off |   00000000:21:00.0 Off |                  Off |
| 30%   34C    P8             21W /  425W |       2MiB /  24564MiB |      0%      Default |
|   1  NVIDIA GeForce RTX 4090 D      Off |   00000000:81:00.0 Off |                  Off |
| 30%   31C    P8             19W /  425W |       2MiB /  24564MiB |      0%      Default |
        GPU0 GPU1 CPU Affinity NUMA Affinity GPU NUMA ID
GPU0 X SYS 16-23,80-87 2 N/A
GPU1 SYS X 56-63,120-127 7 N/A
0000:21:00.0 numa_node=2
0000:81:00.0 numa_node=7
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
"""
        (self.results / "env.txt").write_text(snapshot)
        self.export()
        structured = (
            (self.output / "environment.md").read_text().split("原始环境快照")[0]
        )
        for fact in (
            "| OS 发行版 | Ubuntu 22.04.5 LTS |",
            "| GPU 0 型号 | NVIDIA GeForce RTX 4090 D |",
            "| GPU 1 显存 | 24564 MiB |",
            "| GPU 0 NUMA 节点 | 2 |",
            "| GPU 1 NUMA 节点 | 7 |",
            "| GPU0 ↔ GPU1 连接类型 | SYS |",
            "64（32 核/路 × 2 路）",
            "| MPI 实现与版本 | 5.0.2 |",
        ):
            self.assertIn(fact, structured)
        self.assertIn("部分 GPU 快照的采集状态", structured)
        self.assertIn("T5_residency.log", (self.output / "methods.md").read_text())

    def test_relative_leaf_sources_and_three_significant_digit_tables(self):
        self.write(
            "E2.csv",
            [row(impl="seq", variant="std", threads=1, total=100), row(total=10)],
        )
        self.write(
            "E14.csv",
            [
                dict(
                    threads=128,
                    bind="close",
                    kernel="triad",
                    best_GBps=150.1234,
                    median_GBps=140.1234,
                    n=1 << 28,
                    repeats=5,
                    warmups=1,
                    host="host-a",
                    places="threads",
                    verified=1,
                )
            ],
        )
        self.write(
            "archive_pre_bindfix/E2_quick.csv",
            [row(impl="mpi", procs=2, threads=1, n=262144, total=40, mpi_bind="none")],
        )
        self.write(
            "E2_quick.csv", [row(impl="mpi", procs=2, threads=1, n=262144, total=10)]
        )
        numbers = self.export()
        direct = numbers["best_cpu"]["value"]["total_ms"]
        self.assertEqual(direct["source"], "../code/results/E2.csv")
        self.assertEqual(direct["raw_sources"][0]["file"], "code/results/E2.csv")
        diagnostic = numbers["affinity_fix"]["value"]["timings"][0]
        self.assertTrue(
            diagnostic["before_ms"]["source"].startswith(
                "../code/results/archive_pre_bindfix/"
            )
        )
        self.assertTrue(diagnostic["after_ms"]["source"].startswith("../code/results/"))
        self.assertEqual(len(numbers["diagnostic_sources"]), 2)
        e2 = (self.output / "tables/E2.md").read_text()
        self.assertIn("1.07e+09", e2)
        self.assertIn("主机：host-a；种子：42", e2)
        e14 = (self.output / "tables/E14.md").read_text()
        self.assertIn("| 主机 | 规模 n |", e14)
        self.assertIn("host-a | 2.68e+08", e14)
        self.assertNotIn("150.1234", e14)


if __name__ == "__main__":
    unittest.main()
