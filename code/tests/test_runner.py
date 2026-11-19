#!/usr/bin/env python3
# test_runner.py：验证算法输出和实验方法的边界条件。
# 对应 DESIGN.md 第 9 节。
# 配置、数据来源及缺失状态由统一框架记录。
"""Runner regressions using temporary results, fake experiments and one real child.

These tests do not require sorting binaries, MPI, GPUs or an idle benchmark host.
"""

import argparse
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_experiments as runner


class TemporaryResults(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = io.StringIO()
        self.stdout = contextlib.redirect_stdout(self.output)
        self.stderr = contextlib.redirect_stderr(self.output)
        self.stdout.__enter__()
        self.stderr.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.addCleanup(self.stderr.__exit__, None, None, None)

    def state(self, experiment, quick, success=True, csv=True):
        results = self.root / "results"
        results.mkdir(exist_ok=True)
        stem = experiment + ("_quick" if quick else "")
        runner.atomic_json(
            results / f"{stem}_status.json", {"success": success, "quick": quick}
        )
        if csv:
            (results / f"{stem}.csv").write_text("verified,total_ms\n1,1\n")


class RunnerPolicyTests(TemporaryResults):
    def test_atomic_status_preserves_previous_document_on_replace_failure(self):
        path = self.root / "nested" / "status.json"
        runner.atomic_json(path, {"success": False})
        with patch.object(runner.os, "replace", side_effect=OSError("disk error")):
            with self.assertRaisesRegex(OSError, "disk error"):
                runner.atomic_json(path, {"success": True})
        self.assertEqual(json.loads(path.read_text()), {"success": False})
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_resume_requires_success_matching_mode_and_canonical_csv(self):
        results = self.root / "results"
        self.assertFalse(runner.completed("E2", True, results))
        self.state("E2", True)
        self.assertTrue(runner.completed("E2", True, results))
        self.assertFalse(runner.completed("E2", False, results))
        self.state("E2", False, success=False)
        self.assertFalse(runner.completed("E2", False, results))
        self.state("E2", False)
        self.assertTrue(runner.completed("E2", False, results))
        (results / "E2.csv").unlink()
        self.assertFalse(runner.completed("E2", False, results))
        # A damaged or mismatched marker must not suppress another attempt.
        (results / "E2_quick_status.json").write_text("{")
        self.assertFalse(runner.completed("E2", True, results))
        runner.atomic_json(
            results / "E2_quick_status.json", {"success": True, "quick": False}
        )
        self.assertFalse(runner.completed("E2", True, results))
        self.state("E13", False, csv=False)
        self.assertTrue(runner.completed("E13", False, results))

    def test_selection_is_ordered_deduplicated_and_rejects_unknown_experiments(self):
        parser = argparse.ArgumentParser()
        self.assertEqual(runner.selection("E4,E2,E4", parser), ["E2", "E4"])
        self.assertEqual(runner.selection(None, parser), list(runner.EXPERIMENTS))
        with self.assertRaises(SystemExit) as error:
            runner.selection("E2,E99", parser)
        self.assertEqual(error.exception.code, 2)

    def test_only_resume_mode_separation_and_force(self):
        attempts = []

        def execute(command, **kwargs):
            experiment = next(arg for arg in command if arg in runner.EXPERIMENTS)
            quick = "--quick" in command
            attempts.append((experiment, quick))
            self.assertEqual(kwargs["cwd"], self.root)
            self.assertIsNotNone(kwargs["stdout"])
            self.state(experiment, quick)

        with patch.object(runner, "ROOT", self.root), patch.object(
            runner, "install_termination_handler"
        ), patch.object(runner, "preflight") as preflight, patch.object(
            runner, "run_command", side_effect=execute
        ), patch.object(
            runner.subprocess, "run"
        ):
            runner.main(["--quick", "--only", "E4,E2"])
            self.assertEqual(attempts, [("E2", True), ("E4", True)])
            runner.main(["--quick", "--only", "E2,E4"])
            self.assertEqual(len(attempts), 2)
            # A full run is independent of a successful quick run.
            runner.main(["--only", "E2"])
            runner.main(["--quick", "--only", "E2", "--force"])
            self.assertEqual(attempts[-2:], [("E2", False), ("E2", True)])
            self.assertEqual(preflight.call_count, 3)
        for experiment in ("E2", "E4"):
            log = (self.root / "results" / "logs" / f"{experiment}.log").read_text()
            self.assertIn("mode=quick", log)
            self.assertIn("bench.py", log)

    def test_failed_experiment_is_retried_after_successful_predecessor(self):
        attempts = []
        fail_once = True

        def execute(command, **kwargs):
            nonlocal fail_once
            experiment = next(arg for arg in command if arg in runner.EXPERIMENTS)
            attempts.append(experiment)
            if experiment == "E4" and fail_once:
                fail_once = False
                self.state(experiment, True, success=False, csv=False)
                raise subprocess.CalledProcessError(1, command)
            self.state(experiment, True)

        with patch.object(runner, "ROOT", self.root), patch.object(
            runner, "install_termination_handler"
        ), patch.object(runner, "preflight"), patch.object(
            runner, "run_command", side_effect=execute
        ), patch.object(
            runner.subprocess, "run"
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                runner.main(["--quick", "--only", "E2,E4,E5"])
            self.assertEqual(attempts, ["E2", "E4"])
            runner.main(["--quick", "--only", "E2,E4,E5"])
        self.assertEqual(attempts, ["E2", "E4", "E4", "E5"])

    def test_runner_dry_run_bypasses_preflight_resume_and_all_writes(self):
        with patch.object(runner, "ROOT", self.root), patch.object(
            runner, "preflight"
        ) as preflight, patch.object(runner, "completed") as completed, patch.object(
            runner, "run_command"
        ) as execute, patch.object(
            runner.subprocess, "run"
        ) as preview:
            runner.main(["--quick", "--only", "E2,E11", "--dry-run"])
        preflight.assert_not_called()
        completed.assert_not_called()
        execute.assert_not_called()
        self.assertEqual(preview.call_count, 2)
        for call in preview.call_args_list:
            self.assertIn("--dry-run", call.args[0])
            self.assertIn("--quick", call.args[0])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_load_gate_requires_override_or_interactive_confirmation(self):
        with patch.object(
            runner.Path, "read_text", return_value="3.25 1.0 1.0 1/2 3"
        ), patch.object(runner.sys.stdin, "isatty", return_value=False), patch.object(
            runner.subprocess, "check_output", return_value=""
        ) as query:
            with self.assertRaisesRegex(RuntimeError, "--ignore-load"):
                runner.preflight()
            query.assert_not_called()
            runner.preflight(ignore_load=True)
            query.assert_called_once()
        self.assertIn("WARNING", self.output.getvalue())
        for answer, accepted in (("no", False), ("yes", True)):
            with self.subTest(answer=answer), patch.object(
                runner.Path, "read_text", return_value="3.25 1.0 1.0"
            ), patch.object(runner.sys.stdin, "isatty", return_value=True), patch(
                "builtins.input", return_value=answer
            ), patch.object(
                runner.subprocess, "check_output", return_value=""
            ):
                if accepted:
                    runner.preflight()
                else:
                    with self.assertRaisesRegex(RuntimeError, "cancelled"):
                        runner.preflight()

    def test_missing_gpu_query_does_not_block_cpu_experiments(self):
        with patch.object(
            runner.Path, "read_text", return_value="0.25 1.0 1.0"
        ), patch.object(
            runner.subprocess, "check_output", side_effect=FileNotFoundError
        ):
            runner.preflight()
        self.assertIn("GPU process query unavailable", self.output.getvalue())


class CommandLifecycleTests(TemporaryResults):
    def nested_interrupt(self, signum):
        """Exercise runner -> bench -> correctness -> stubborn sorter sessions."""
        ready = self.root / "stubborn.pid"
        child = (
            "import os,signal,time; from pathlib import Path; "
            "signal.signal(signal.SIGINT,signal.SIG_IGN); "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(30)"
        )
        command = [sys.executable, "-B", "-c", child]
        traces = []
        pidfiles = [ready]
        # Increasing grace at each enclosing Python driver allows the inner
        # driver to kill and reap a sorter that ignores both graceful signals.
        for level, grace in enumerate((0.2, 2, 3)):
            pidfile = self.root / f"driver_{level}.pid"
            trace = self.root / f"driver_{level}.jsonl"
            pidfiles.append(pidfile)
            traces.append(trace)
            driver = (
                "import os,sys; from pathlib import Path; "
                f"sys.path.insert(0,{str(SCRIPTS)!r}); "
                "from run_experiments import run_command,install_termination_handler\n"
                "install_termination_handler()\n"
                f"Path({str(pidfile)!r}).write_text(str(os.getpid()))\n"
                "try:\n"
                f" run_command({command!r},trace_path={str(trace)!r},"
                f"cleanup_timeout={grace!r})\n"
                "except KeyboardInterrupt:\n"
                " sys.exit(130)\n"
                "except InterruptedError:\n"
                " sys.exit(1)\n"
            )
            command = [sys.executable, "-B", "-c", driver]
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        pids = []
        try:
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                self.assertIsNone(
                    process.poll(), "driver exited before sorter readiness"
                )
                if ready.exists() and ready.read_text():
                    break
                time.sleep(0.02)
            self.assertTrue(ready.exists(), "stubborn sorter did not become ready")
            pids = [int(path.read_text()) for path in pidfiles]
            process.send_signal(signum)
            self.assertEqual(
                process.wait(timeout=6), 130 if signum == signal.SIGINT else 1
            )
            # Checking every session catches the former nested-timeout race:
            # the outer driver exited while an unresponsive sorter survived.
            for pid in pids:
                with self.subTest(pid=pid), self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
            for level, trace in enumerate(traces):
                records = [json.loads(line) for line in trace.read_text().splitlines()]
                self.assertEqual(
                    [record["event"] for record in records], ["started", "finished"]
                )
                self.assertIn(
                    (
                        "KeyboardInterrupt"
                        if signum == signal.SIGINT
                        else "InterruptedError"
                    ),
                    records[-1]["error"],
                )
                if level == 0:
                    self.assertEqual(records[-1]["returncode"], -signal.SIGKILL)
        finally:
            # Failures in this test must not themselves leave driver sessions.
            pids.extend(
                int(path.read_text())
                for path in pidfiles
                if path.exists() and path.read_text()
            )
            for pid in set(pids + [process.pid]):
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.poll() is None:
                process.wait(timeout=5)

    @unittest.skipUnless(hasattr(os, "killpg"), "requires POSIX process groups")
    def test_nested_sigint_reaps_stubborn_sorter_and_all_drivers(self):
        self.nested_interrupt(signal.SIGINT)

    @unittest.skipUnless(hasattr(os, "killpg"), "requires POSIX process groups")
    def test_nested_sigterm_reaps_stubborn_sorter_and_all_drivers(self):
        self.nested_interrupt(signal.SIGTERM)

    def test_trace_records_failed_command_and_binding_environment(self):
        trace = self.root / "commands.jsonl"
        environment = dict(
            os.environ,
            OMP_NUM_THREADS="8",
            OMP_PROC_BIND="close",
            PARASORT_MPI_BIND="socket",
            UNRELATED_SECRET="do not record",
        )
        command = [sys.executable, "-c", "raise SystemExit(7)"]
        with self.assertRaises(subprocess.CalledProcessError):
            runner.run_command(
                command,
                env=environment,
                trace_path=trace,
                metadata={"experiment": "E2", "quick": True},
            )
        records = [json.loads(line) for line in trace.read_text().splitlines()]
        self.assertEqual(
            [record["event"] for record in records], ["started", "finished"]
        )
        self.assertEqual(records[0]["command_id"], records[1]["command_id"])
        self.assertEqual(records[1]["returncode"], 7)
        self.assertIn("CalledProcessError", records[1]["error"])
        self.assertEqual(records[0]["environment"]["PARASORT_MPI_BIND"], "socket")
        self.assertNotIn("UNRELATED_SECRET", records[0]["environment"])

    @unittest.skipUnless(hasattr(os, "killpg"), "requires POSIX process groups")
    def test_sigint_is_forwarded_and_child_is_reaped(self):
        ready = self.root / "child.pid"
        trace = self.root / "interrupted.jsonl"
        child_code = (
            "import os,time; from pathlib import Path; "
            f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)"
        )
        driver = (
            "import sys; "
            f"sys.path.insert(0, {str(SCRIPTS)!r}); "
            "from run_experiments import run_command\n"
            "try:\n"
            f' run_command({[sys.executable, "-c", child_code]!r}, '
            f"trace_path={str(trace)!r})\n"
            "except KeyboardInterrupt:\n"
            " sys.exit(130)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-B", "-c", driver],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        child_pid = None
        try:
            deadline = time.monotonic() + 8
            while not ready.exists() and time.monotonic() < deadline:
                self.assertIsNone(
                    process.poll(), "driver exited before launching its child"
                )
                time.sleep(0.02)
            self.assertTrue(ready.exists(), "child did not become ready")
            child_pid = int(ready.read_text())
            process.send_signal(signal.SIGINT)
            self.assertEqual(process.wait(timeout=12), 130)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual(
                [record["event"] for record in records], ["started", "finished"]
            )
            self.assertIn("KeyboardInterrupt", records[-1]["error"])
            self.assertNotEqual(records[-1]["returncode"], 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            if child_pid is not None:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


class BenchTransactionTests(TemporaryResults):
    def setUp(self):
        super().setUp()
        self.bench = importlib.import_module("bench")
        self.previous_cwd = Path.cwd()
        self.addCleanup(os.chdir, self.previous_cwd)
        self.addCleanup(signal.signal, signal.SIGTERM, signal.getsignal(signal.SIGTERM))
        environment = patch.dict(os.environ, {})
        environment.start()
        self.addCleanup(environment.stop)

    def test_failure_or_interruption_keeps_canonical_csv_then_retry_replaces_it(self):
        for failure in (RuntimeError("fake sorter failed"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                results = self.root / "results"
                results.mkdir(exist_ok=True)
                canonical = results / "E2_quick.csv"
                canonical.write_text("verified,total_ms\n1,99\n")

                def fail():
                    self.bench.result.write_text("verified,total_ms\n1,42\n")
                    raise failure

                def succeed():
                    self.bench.result.write_text("verified,total_ms\n1,7\n")

                with patch.object(self.bench, "ROOT", self.root), patch.object(
                    self.bench, "available_gpu", return_value=0
                ), patch.object(
                    self.bench.shutil, "which", return_value=None
                ), patch.object(
                    self.bench.subprocess, "run"
                ), patch.object(
                    self.bench, "execute_experiment", side_effect=fail
                ):
                    with self.assertRaises(type(failure)):
                        self.bench.main(["E2", "--quick"])
                self.assertEqual(canonical.read_text(), "verified,total_ms\n1,99\n")
                state = json.loads((results / "E2_quick_status.json").read_text())
                self.assertFalse(state["success"])
                self.assertTrue(state.get("error"))
                self.assertTrue(state.get("partial_csv"))
                partial = Path(state["partial_csv"])
                if not partial.is_absolute():
                    partial = self.root / partial
                self.assertEqual(partial.read_text(), "verified,total_ms\n1,42\n")
                self.assertFalse(runner.completed("E2", True, results))
                with patch.object(self.bench, "ROOT", self.root), patch.object(
                    self.bench, "available_gpu", return_value=0
                ), patch.object(
                    self.bench.shutil, "which", return_value=None
                ), patch.object(
                    self.bench.subprocess, "run"
                ), patch.object(
                    self.bench, "execute_experiment", side_effect=succeed
                ):
                    self.bench.main(["E2", "--quick"])
                self.assertEqual(canonical.read_text(), "verified,total_ms\n1,7\n")
                self.assertTrue(runner.completed("E2", True, results))

    def test_real_failed_sorter_is_traced_and_retry_does_not_duplicate_rows(self):
        binary = self.root / "bin" / "seq_sort"
        binary.parent.mkdir()
        fail = self.root / "fail"
        fail.touch()
        binary.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            "from pathlib import Path\n"
            "csv = Path(sys.argv[sys.argv.index('--csv') + 1])\n"
            "with csv.open('a') as output:\n"
            " if csv.stat().st_size == 0: output.write('verified,total_ms\\n')\n"
            " output.write('1,7\\n')\n"
            f"if Path({str(fail)!r}).exists(): sys.exit(9)\n"
        )
        binary.chmod(0o755)
        results = self.root / "results"
        results.mkdir()
        canonical = results / "E2_quick.csv"
        canonical.write_text("verified,total_ms\n1,99\n")
        with patch.object(self.bench, "ROOT", self.root), patch.object(
            self.bench, "available_gpu", return_value=0
        ), patch.object(self.bench.shutil, "which", return_value=None), patch.object(
            self.bench,
            "execute_experiment",
            side_effect=lambda: self.bench.run("seq_sort", 32),
        ):
            with self.assertRaises(subprocess.CalledProcessError) as error:
                self.bench.main(["E2", "--quick"])
            self.assertEqual(error.exception.returncode, 9)
            self.assertEqual(canonical.read_text(), "verified,total_ms\n1,99\n")
            fail.unlink()
            self.bench.main(["E2", "--quick"])
        self.assertEqual(canonical.read_text(), "verified,total_ms\n1,7\n")
        records = [
            json.loads(line)
            for line in (results / "commands.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            [record["event"] for record in records],
            ["started", "finished", "started", "finished"],
        )
        self.assertEqual([records[index]["returncode"] for index in (1, 3)], [9, 0])
        self.assertNotEqual(records[0]["run_id"], records[2]["run_id"])

    def test_bench_dry_run_lists_missing_cpu_mpi_and_gpu_without_artifacts(self):
        with patch.object(self.bench, "ROOT", self.root), patch.object(
            self.bench, "available_gpu", side_effect=AssertionError("GPU query")
        ), patch.object(
            self.bench.subprocess, "run", side_effect=AssertionError("executed")
        ), patch.object(
            self.bench, "launch_flags", return_value=["-bind-to", "core"]
        ):
            self.bench.main(["E12", "--quick", "--dry-run"])
        output = self.output.getvalue()
        for binary in (
            "seq_sort",
            "psrs_omp",
            "psrs_mpi",
            "psrs_hybrid",
            "gpu_sort",
            "gpu_multi",
        ):
            self.assertIn(binary, output)
        self.assertIn("ESTIMATE", output.upper())
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
