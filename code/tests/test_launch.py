#!/usr/bin/env python3
# test_launch.py：验证算法输出和实验方法的边界条件。
# 对应 DESIGN.md 第 9 节。
# 配置、数据来源及缺失状态由统一框架记录。
"""Policy regression tests without requiring MPI, GPUs, or a specific host."""

import contextlib
import importlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import launch


class LauncherPolicyTests(unittest.TestCase):
    def setUp(self):
        self.topo = launch.Topology(64, 128, (32, 32), (8,) * 8)
        self.topology = patch.object(launch, "topology", return_value=self.topo)
        self.implementation = patch.object(launch, "detect_mpi", return_value="mpich")
        self.topology_query = self.topology.start()
        self.detect = self.implementation.start()
        self.addCleanup(self.topology.stop)
        self.addCleanup(self.implementation.stop)

    def flags(
        self, exe="psrs_mpi", procs=2, threads=1, implementation="mpich", **extra
    ):
        self.detect.return_value = implementation
        env = {"OMP_PROC_BIND": "close", "OMP_PLACES": "cores", **extra}
        return launch.launch_flags(exe, procs, threads, env), env

    def test_pure_mpi_overrides_global_openmp_binding(self):
        for exe in ("psrs_mpi", "pipeline_sort", "affinity_probe"):
            with self.subTest(exe=exe):
                flags, env = self.flags(exe, 8)
                self.assertEqual(flags, ["-bind-to", "core", "-map-by", "core"])
                self.assertEqual(env["OMP_NUM_THREADS"], "1")
                self.assertEqual(env["OMP_PROC_BIND"], "false")
                self.assertEqual(env["PARASORT_LAUNCHER"], "mpich")
                self.assertEqual(env["PARASORT_MPI_BIND"], "core")
        _, env = self.flags("psrs_mpi", 8, 64)
        self.assertEqual(env["OMP_NUM_THREADS"], "1")

    def test_pure_smt_starts_above_physical_core_capacity(self):
        for procs, expected in ((64, "core"), (65, "hwthread"), (128, "hwthread")):
            flags, env = self.flags(procs=procs)
            self.assertEqual(flags, ["-bind-to", expected, "-map-by", expected])
            self.assertEqual(env["PARASORT_MPI_BIND"], expected)
        self.topology_query.return_value = launch.Topology(4, 8, (4,), (4,))
        self.assertEqual(self.flags(procs=5)[1]["PARASORT_MPI_BIND"], "hwthread")

    def test_hybrid_uses_disjoint_hardware_domains(self):
        for procs, threads, expected, places in (
            (2, 8, "socket", "cores"),
            (4, 4, "numa:2", "cores"),
            (8, 8, "numa", "cores"),
            (2, 64, "socket", "threads"),
            (4, 32, "numa:2", "threads"),
            (8, 16, "numa", "threads"),
        ):
            with self.subTest(procs=procs, threads=threads):
                flags, env = self.flags("psrs_hybrid", procs, threads)
                self.assertEqual(flags, ["-bind-to", expected, "-map-by", expected])
                self.assertEqual(env["OMP_NUM_THREADS"], str(threads))
                self.assertEqual(env["OMP_PROC_BIND"], "close")
                self.assertEqual(env["OMP_PLACES"], places)
        self.assertEqual(
            self.flags("affinity_probe", 2, 8)[1]["PARASORT_MPI_BIND"], "socket"
        )

    def test_other_rank_counts_use_nonoverlapping_groups(self):
        for procs, expected in (
            (1, "core:64"),
            (3, "core:21"),
            (7, "core:9"),
            (16, "core:4"),
        ):
            self.assertEqual(
                self.flags("psrs_hybrid", procs, 2)[0],
                ["-bind-to", expected, "-map-by", expected],
            )
        self.topology_query.return_value = launch.Topology(4, 8, (4,), (4,))
        flags, env = self.flags("psrs_hybrid", 8, 2)
        self.assertEqual(flags, ["-bind-to", "hwthread:1", "-map-by", "hwthread:1"])
        self.assertEqual(env["OMP_PLACES"], "threads")
        # An 8-core one-socket host cannot use eight NUMA-domain mappings.
        self.assertEqual(self.flags("psrs_hybrid", 2, 2)[0][1], "core:2")

    def test_openmpi_maps_with_matching_processing_element_units(self):
        self.assertEqual(
            self.flags(procs=8, implementation="openmpi")[0],
            ["--bind-to", "core", "--map-by", "core"],
        )
        self.assertEqual(
            self.flags(procs=128, implementation="openmpi")[0],
            ["--use-hwthread-cpus", "--bind-to", "hwthread", "--map-by", "hwthread"],
        )
        for procs, threads, expected, binding in (
            (2, 8, "ppr:1:package:PE=8", "core"),
            (4, 16, "ppr:2:package:PE=16", "core"),
            (8, 16, "ppr:1:numa:PE=16", "hwthread"),
            (3, 2, "slot:PE=2", "core"),
        ):
            flags, env = self.flags("psrs_hybrid", procs, threads, "openmpi")
            self.assertEqual(flags[-4:], ["--bind-to", binding, "--map-by", expected])
            self.assertEqual("--use-hwthread-cpus" in flags, binding == "hwthread")
            self.assertEqual(env["PARASORT_LAUNCHER"], "openmpi")

    def test_user_flags_preserve_quoting_and_warn(self):
        warning = io.StringIO()
        with contextlib.redirect_stderr(warning):
            flags, env = self.flags(
                MPIEXEC_FLAGS='-hosts "node 1,node 2" -bind-to none'
            )
        self.assertEqual(flags, ["-hosts", "node 1,node 2", "-bind-to", "none"])
        self.assertEqual(env["PARASORT_MPI_BIND"], "user")
        self.assertEqual(env["OMP_PROC_BIND"], "false")
        self.assertIn("WARNING: MPIEXEC_FLAGS", warning.getvalue())
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.flags(MPIEXEC_FLAGS="")[0], [])
            # Explicit user oversubscription still overrides automatic policy.
            self.assertEqual(
                self.flags(procs=256, MPIEXEC_FLAGS="--oversubscribe")[0],
                ["--oversubscribe"],
            )

    def test_invalid_counts_fail_before_launch(self):
        for procs, threads in ((0, 1), (1, 0), (-1, 1), (129, 1)):
            with self.subTest(procs=procs, threads=threads), self.assertRaises(
                ValueError
            ):
                self.flags(procs=procs, threads=threads)

    def test_bench_supplies_binding_environment_for_each_child(self):
        bench = importlib.import_module("bench")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            (root / "results").mkdir()
            (root / "bin" / "psrs_mpi").touch()
            with patch.multiple(
                bench,
                ROOT=root,
                MPI=True,
                GPU=0,
                create=True,
                args=SimpleNamespace(quick=True, experiment="E2"),
                R=2,
                result=root / "results" / "E2_quick.csv",
                plan={"runs": 0, "skipped": []},
            ):
                with patch.dict(os.environ, {"OMP_PROC_BIND": "close"}, clear=True):
                    with patch.object(
                        bench, "run_command"
                    ) as run, contextlib.redirect_stdout(io.StringIO()):
                        bench.run("psrs_mpi", 1024, procs=8)
                    command = run.call_args.args[0]
                    self.assertEqual(
                        command[1:7], ["-bind-to", "core", "-map-by", "core", "-n", "8"]
                    )
                    child_env = run.call_args.kwargs["env"]
                    self.assertEqual(child_env["OMP_PROC_BIND"], "false")
                    self.assertEqual(child_env["OMP_NUM_THREADS"], "1")
                    self.assertEqual(child_env["PARASORT_MPI_BIND"], "core")
                    self.assertEqual(os.environ["OMP_PROC_BIND"], "close")
                    self.assertEqual(bench.plan["runs"], 1)

    def test_importing_bench_does_not_run_experiments(self):
        sys.modules.pop("bench", None)
        with patch("subprocess.run") as run, patch(
            "subprocess.check_output"
        ) as check, patch("os.chdir") as cd:
            with patch.object(sys, "argv", ["unrelated", "--invalid-option"]):
                importlib.import_module("bench")
        run.assert_not_called()
        check.assert_not_called()
        cd.assert_not_called()


class MpiDetectionTests(unittest.TestCase):
    def tearDown(self):
        launch.detect_mpi.cache_clear()
        launch.topology.cache_clear()

    def test_detects_hydra_and_openmpi(self):
        for version, implementation in (
            ("HYDRA build details: Version: 5.0.2", "mpich"),
            ("mpiexec (Open MPI) 5.0.8", "openmpi"),
        ):
            launch.detect_mpi.cache_clear()
            with patch("subprocess.check_output", return_value=version) as query:
                self.assertEqual(
                    launch.detect_mpi("/custom/mpi", "/custom/bin", "/custom/lib"),
                    implementation,
                )
                launch.detect_mpi("/custom/mpi", "/custom/bin", "/custom/lib")
                self.assertEqual(query.call_count, 1)
                self.assertEqual(query.call_args.args[0], ["/custom/mpi", "--version"])
                self.assertEqual(
                    query.call_args.kwargs["env"]["LD_LIBRARY_PATH"], "/custom/lib"
                )

    def test_unknown_or_missing_launcher_is_actionable(self):
        with patch(
            "subprocess.check_output", return_value="unrecognized MPI"
        ), self.assertRaisesRegex(RuntimeError, "unsupported MPI launcher"):
            launch.detect_mpi("other")
        with patch(
            "subprocess.check_output", side_effect=FileNotFoundError
        ), self.assertRaisesRegex(RuntimeError, "cannot identify MPI launcher"):
            launch.detect_mpi("missing")

    def test_topology_counts_smt_and_respects_allowed_cpus(self):
        # Socket-local core IDs can repeat on another socket; SMT siblings do not
        # create new physical cores. The final row is outside the allowed set.
        rows = "# CPU,Core,Socket,Node\n0,0,0,0\n1,0,0,0\n2,0,1,1\n3,0,1,1\n4,1,1,1\n"
        with patch("os.sched_getaffinity", return_value={0, 1, 2, 3}), patch(
            "subprocess.check_output", return_value=rows
        ):
            self.assertEqual(launch.topology(), launch.Topology(2, 4, (1, 1), (1, 1)))


if __name__ == "__main__":
    unittest.main()
