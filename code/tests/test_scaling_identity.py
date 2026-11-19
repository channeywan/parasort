#!/usr/bin/env python3
# test_scaling_identity.py：验证算法输出和实验方法的边界条件。
# 对应 DESIGN.md 第 9 节。
# 配置、数据来源及缺失状态由统一框架记录。
"""Scaling curves normalize expected placement transitions, preserving provenance."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    "scaling_summary", Path(__file__).resolve().parents[1] / "analysis/summarize.py"
)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def sample(
    impl="omp",
    p=64,
    ranks=2,
    n=1 << 30,
    total=None,
    experiment="E2",
    quick=False,
    host="host-a",
    seed="42",
    dist="uniform",
    tuning=None,
    verified=1,
):
    if impl == "hybrid":
        procs, threads = ranks, p // ranks
        mpi_bind = {2: "socket", 4: "numa:2", 8: "numa"}[ranks]
    elif impl == "mpi":
        procs, threads = p, 1
        mpi_bind = "core" if p <= 64 else "hwthread"
    else:
        procs, threads, mpi_bind = 1, p, "none"
    extra = {
        "seed": seed,
        "oversample": "1",
        "buckets_per_thread": "1",
        "schedule": "static",
        "tiebreak": "1",
        "bind": "false" if impl == "mpi" else "close",
        "places": "threads" if p > 64 and impl != "mpi" else "cores",
        "mpi_bind": mpi_bind,
        "launcher": "mpich" if impl in ("mpi", "hybrid") else "none",
        "scatter_ms": "2.5",
        "compute_ms": "8",
    }
    extra.update(tuning or {})
    # None represents a genuinely missing legacy CSV field, not the string None.
    extra = {key: value for key, value in extra.items() if value is not None}
    row = {
        "experiment": experiment,
        "quick": quick,
        "host": host,
        "impl": impl,
        "variant": "std" if impl == "seq" else "psrs",
        "dist": dist,
        "n": n,
        "procs": procs,
        "threads": threads,
        "gpus": 0,
        "repeat_id": 0,
        "total_ms": 10 + 120 / p if total is None else total,
        "imbalance": 1.2,
        "verified": verified,
        "extra": ";".join(f"{key}={value}" for key, value in extra.items()),
    }
    for phase in (
        "local_sort",
        "sampling",
        "splitters",
        "partition",
        "exchange",
        "merge",
        "h2d",
        "d2h",
    ):
        row[phase + "_ms"] = 1.0
    return row


def scaling_summary(rows):
    return SUMMARY.summarize(pd.DataFrame(rows))


class ScalingIdentityTests(unittest.TestCase):
    def test_fresh_summary_extracts_verified_scatter_and_actual_places(self):
        rows = [
            sample(total=10, tuning={"scatter_ms": "2"}),
            sample(total=30, tuning={"scatter_ms": "4"}),
            sample(total=999, tuning={"scatter_ms": "999"}, verified=0),
            sample(total=50, tuning={"places": None, "scatter_ms": None}),
            sample("seq", p=1, total=100),
        ]
        raw = pd.DataFrame(rows)
        before = raw.copy(deep=True)
        summary = SUMMARY.summarize(raw)
        pd.testing.assert_frame_equal(raw, before)
        omp = summary[summary.impl == "omp"]
        self.assertEqual(len(omp), 2)
        normal = omp[omp.places == "cores"].iloc[0]
        self.assertEqual(
            (normal.median_ms, normal.min_ms, normal.max_ms, normal.repeats),
            (20, 10, 30, 2),
        )
        self.assertEqual(normal.scatter_ms, 3)
        self.assertEqual(normal.baseline_ms, 100)
        self.assertEqual(normal.speedup, 5)
        legacy = omp[omp.places == "unset"].iloc[0]
        self.assertTrue(pd.isna(legacy.scatter_ms))
        self.assertIn("places", SUMMARY.TUNING)
        self.assertTrue(set(SUMMARY.GROUP_KEYS).issubset(summary.columns))

    def test_read_results_keeps_places_and_numeric_scatter_only_for_verified_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results").mkdir()
            pd.DataFrame(
                [
                    sample("mpi", p=32, tuning={"scatter_ms": "1.25"}),
                    sample("mpi", p=128, tuning={"scatter_ms": "3.75"}),
                    sample("mpi", p=64, tuning={"scatter_ms": "999"}, verified=0),
                    sample("omp", tuning={"places": None, "scatter_ms": None}),
                ]
            ).drop(columns=["experiment", "quick"]).to_csv(
                root / "results/E2.csv", index=False
            )
            with patch.object(SUMMARY, "ROOT", root):
                actual = SUMMARY.read_results()
        self.assertEqual(len(actual), 3)
        mpi = actual[actual.impl == "mpi"]
        self.assertEqual(set(mpi.scatter_ms), {1.25, 3.75})
        self.assertEqual(set(mpi.mpi_bind), {"core", "hwthread"})
        self.assertEqual(set(actual[actual.impl == "omp"].places), {"unset"})

    def test_expected_smt_transitions_form_complete_fits_with_actual_configs(self):
        rows = [sample("seq", p=1, total=100)]
        for p in (32, 64, 128):
            rows.extend(
                [
                    sample("mpi", p),
                    sample("omp", p),
                    sample("hybrid", p, ranks=2),
                    sample("hybrid", p, ranks=8),
                ]
            )
        summary = scaling_summary(rows)
        measured = summary[summary.impl != "seq"]
        before = measured.copy(deep=True)
        identities = SUMMARY.scaling_identity(measured)
        pd.testing.assert_frame_equal(measured, before)
        keys = SUMMARY.scaling_keys(measured)
        self.assertTrue(
            {
                "host",
                "dist",
                "quick",
                "hybrid_ranks",
                "bind_policy",
                "places_policy",
                "scale_n",
            }.issubset(keys)
        )
        self.assertTrue(any(key in keys for key in ("seed", "_identity_seed")))
        groups = list(SUMMARY.scaling_groups(measured))
        self.assertEqual(sorted(len(group) for _, group in groups), [3, 3, 3, 3])
        self.assertEqual(
            set(identities[identities.impl == "hybrid"].hybrid_ranks), {2, 8}
        )
        self.assertEqual(set(identities[identities.impl == "mpi"].hybrid_ranks), {0})
        fits = SUMMARY.fit_models(summary)
        self.assertEqual(len(fits), 4)
        self.assertEqual(
            {int(fit["hybrid_ranks"]) for fit in fits if fit["impl"] == "hybrid"},
            {2, 8},
        )
        for fit in fits:
            self.assertEqual(fit["points"], 3)
            self.assertEqual(
                [point["p"] for point in fit["data_points"]], [32, 64, 128]
            )
            self.assertIn("bind_policy", fit)
            self.assertIn("places_policy", fit)
            for point in fit["data_points"]:
                self.assertTrue(set(SUMMARY.GROUP_KEYS).issubset(point["config"]))
                self.assertEqual(point["baseline_ms"], 100)
                self.assertEqual(point["time_ms"], 10 + 120 / point["p"])
            actual = [point["config"] for point in fit["data_points"]]
            if fit["impl"] == "mpi":
                self.assertEqual(
                    [config["mpi_bind"] for config in actual],
                    ["core", "core", "hwthread"],
                )
            else:
                self.assertEqual(
                    [config["places"] for config in actual],
                    ["cores", "cores", "threads"],
                )

    def test_user_wrong_and_legacy_placements_stay_separate(self):
        rows = []
        for p in (32, 64, 128):
            rows.extend(
                [
                    sample("mpi", p),
                    sample("omp", p),
                    sample("hybrid", p, ranks=2),
                    sample("mpi", p, tuning={"mpi_bind": "user"}),
                    sample("omp", p, tuning={"places": None}),
                ]
            )
        rows.extend(
            [
                sample("mpi", 128, tuning={"mpi_bind": "core"}),
                sample("omp", 128, tuning={"places": "cores"}),
                sample("hybrid", 128, ranks=2, tuning={"places": "cores"}),
            ]
        )
        summary = scaling_summary(rows)
        groups = list(SUMMARY.scaling_groups(summary))
        self.assertEqual(
            sorted(len(group) for _, group in groups), [1, 1, 1, 3, 3, 3, 3, 3]
        )
        for _, group in groups:
            if "user" in set(group.mpi_bind):
                self.assertEqual(set(group.mpi_bind), {"user"})
            if "unset" in set(group.places):
                self.assertEqual(set(group.places), {"unset"})

    def test_weak_scaling_uses_keys_per_unit_and_preserves_context(self):
        unit = 1 << 20
        rows = [sample("omp", p, n=unit * p, experiment="E3") for p in (1, 2, 4, 8)]
        rows += [
            sample("omp", 2, n=unit * 2, experiment="E3", host="host-b"),
            sample("omp", 4, n=unit * 4, experiment="E3", seed="43"),
            sample("omp", 8, n=unit * 8, experiment="E3", dist="normal"),
            sample("omp", 8, n=unit * 16, experiment="E3"),
        ]
        summary = scaling_summary(rows)
        normal = summary[
            (summary.host == "host-a")
            & (summary.seed == "42")
            & (summary.dist == "uniform")
            & (summary.n / summary.units == unit)
        ]
        self.assertEqual(len(list(SUMMARY.scaling_groups(normal))), 4)
        self.assertEqual(len(list(SUMMARY.scaling_groups(normal, weak=True))), 1)
        weak = SUMMARY.scaling_identity(summary, weak=True)
        self.assertEqual(
            sorted(
                len(group) for _, group in SUMMARY.scaling_groups(summary, weak=True)
            ),
            [1, 1, 1, 1, 4],
        )
        self.assertEqual(set(weak.scale_n), {unit, 2 * unit})
        self.assertEqual(set(weak.n), {row["n"] for row in rows})


if __name__ == "__main__":
    unittest.main()
