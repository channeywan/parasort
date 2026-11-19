#!/usr/bin/env python3
# test_portable_package.py：验证追溯路径清理与提交包的目录边界。
# 对应 DESIGN.md 第 9 节的可复现构建与提交要求。
# 使用临时项目和真实 ZIP 操作，不需要排序负载或外部服务。
"""Portable path, logging and package regressions with temporary fixtures."""

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

CODE = Path(__file__).resolve().parents[1]
SCRIPTS = CODE / "scripts"
sys.path.insert(0, str(SCRIPTS))
from portable_paths import portable_paths
import run_experiments as runner
from sanitize_paths import sanitize_tree


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="parasort-portable-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "project"
        self.code = self.root / "code"
        self.code.mkdir(parents=True)

    def test_recursive_copy_prefix_boundaries_and_conda_aliases(self):
        legacy = self.base / "legacy"
        prefix = legacy / ".conda"
        (prefix / "conda-meta").mkdir(parents=True)
        (self.root / ".conda").symlink_to(prefix, target_is_directory=True)
        event = {
            "command": [str(self.code / "bin/seq_sort"), str(prefix / "bin/python")],
            "nested": {
                "csv": self.code / "results/E2.csv",
                "report": str(self.root / "report/report.md"),
                "message": f"failed: '{legacy}/code/bin/program':12",
                "compiler": f"-I{prefix}/include -L{prefix}/lib",
            },
            "neighbor": str(self.base / "project-copy/code/not-ours"),
            "system": "/usr/bin/g++",
            "value": [1, 2.5, True, None],
        }
        before = copy.deepcopy(event)
        result = portable_paths(event, self.code, conda_prefix=prefix)
        self.assertEqual(event, before)
        self.assertEqual(result["command"], ["./bin/seq_sort", "../.conda/bin/python"])
        self.assertEqual(result["nested"]["csv"], "./results/E2.csv")
        self.assertEqual(result["nested"]["report"], "../report/report.md")
        self.assertEqual(result["nested"]["message"], "failed: './bin/program':12")
        self.assertEqual(
            result["nested"]["compiler"], "-I../.conda/include -L../.conda/lib"
        )
        self.assertEqual(result["neighbor"], event["neighbor"])
        self.assertEqual(result["system"], event["system"])
        self.assertEqual(result["value"], event["value"])

    def test_runner_serializes_portable_copies_and_executes_original_argv(self):
        script = self.code / "real child.py"
        script.write_text(
            "from pathlib import Path\n"
            "Path(__file__).with_suffix('.done').write_text('executed')\n"
        )
        command = [sys.executable, str(script)]
        original = list(command)
        trace = self.code / "commands.jsonl"
        status = self.code / "status.json"
        event = {"command": command, "csv": str(self.code / "results/E2.csv")}
        with patch.object(runner, "ROOT", self.code):
            runner.atomic_json(status, event)
            runner.run_command(command, cwd=self.code, trace_path=trace)
        self.assertEqual(command, original)
        self.assertEqual(script.with_suffix(".done").read_text(), "executed")
        serialized = json.loads(status.read_text())
        self.assertEqual(serialized["csv"], "./results/E2.csv")
        events = [json.loads(line) for line in trace.read_text().splitlines()]
        self.assertEqual([event["event"] for event in events], ["started", "finished"])
        self.assertTrue(
            all(event["command"][1] == "./real child.py" for event in events)
        )
        self.assertEqual(events[-1]["returncode"], 0)

    def test_sanitize_dry_run_idempotence_and_raw_csv_preservation(self):
        results = self.code / "results"
        (results / "validation").mkdir(parents=True)
        legacy = self.base / "old 项目"
        status = results / "E2_status.json"
        status.write_text(json.dumps({"csv": str(legacy / "code/results/E2.csv")}))
        trace = results / "commands.jsonl"
        trace.write_text(
            json.dumps({"command": [str(legacy / "code/bin/seq_sort")]})
            + "\n"
            + f"old malformed record: {legacy}/code/bin/seq_sort\n"
        )
        log = results / "validation/build.log"
        log.write_text(
            f"RUN {self.code}/bin/program --csv {legacy}/code/results/E2.csv\n"
        )
        (results / "env.txt").write_text(f"environment={legacy}/.conda\n")
        csv = results / "E2.csv"
        csv.write_text("verified,total_ms,extra\n1,12.3,seed=42\n")
        raw = csv.read_bytes()
        originals = {path: path.read_bytes() for path in (status, trace, log)}
        changed = sanitize_tree(results, self.code, [legacy], dry_run=True)
        self.assertEqual(len(changed), 4)
        self.assertTrue(
            all(path.read_bytes() == value for path, value in originals.items())
        )
        sanitize_tree(results, self.code, [legacy])
        self.assertEqual(json.loads(status.read_text())["csv"], "./results/E2.csv")
        self.assertNotIn(str(legacy), trace.read_text())
        self.assertIn("./bin/program", log.read_text())
        self.assertEqual(csv.read_bytes(), raw)
        self.assertEqual(sanitize_tree(results, self.code, [legacy]), [])

    def package_fixture(self):
        scripts = self.code / "scripts"
        scripts.mkdir()
        shutil.copyfile(SCRIPTS / "package.sh", scripts / "package.sh")
        (self.root / "report/figures/screenshots").mkdir(parents=True)
        (self.root / "report/tables").mkdir()
        (self.root / "report/data_bundle").mkdir()
        for relative, contents in {
            "code/Makefile": "all:\n\t@echo fixture build\n",
            "code/main.cpp": "int main() { return 0; }\n",
            "environment.yml": "name: fixture\nchannels: [conda-forge]\n",
            "DESIGN.md": "excluded design",
            "IMPLEMENTATION.md": "excluded implementation",
            "report/report.md": "report body preserved",
            "report/report.pdf": "pdf fixture",
            "report/report.docx": "docx fixture",
            "report/tables/E2.md": "measured table",
            "report/figures/E2.png": "figure fixture",
            "report/figures/screenshots/.gitkeep": "",
            "report/data_bundle/key_numbers.json": "excluded bundle",
            "code/results/E2.csv": "verified,total_ms\n1,12.3\n",
            "code/results/archive_pre_bindfix/E2_quick.csv": "excluded archive",
            "code/results/.mpl-cache/cache": "excluded cache",
            "code/results/.pending/partial.csv": "excluded intermediate",
            "code/bin/program": "excluded binary",
            "code/build/object.o": "excluded build",
            "code/loose.o": "excluded object",
            "code/__pycache__/module.pyc": "excluded bytecode",
            "code/.conda/private": "excluded environment",
        }.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
        return scripts / "package.sh"

    @unittest.skipUnless(shutil.which("zipinfo"), "zipinfo is required by package.sh")
    def test_package_only_code_report_excludes_runtime_and_runs_from_any_cwd(self):
        package = self.package_fixture()
        completed = subprocess.run(
            ["bash", str(package), "000", "测试"],
            cwd=self.base,
            env=dict(os.environ, PYTHON=sys.executable),
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        archive = self.base / "000-测试.zip"
        self.assertTrue(archive.is_file())
        with zipfile.ZipFile(archive) as source:
            names = source.namelist()
            relative = {name.removeprefix("000-测试/") for name in names}
            self.assertTrue(
                all(
                    not name or name.split("/")[0] in {"code", "report"}
                    for name in relative
                )
            )
            self.assertIn("code/environment.yml", relative)
            self.assertIn("report/figures/screenshots/.gitkeep", relative)
            self.assertIn("report/report.pdf", relative)
            self.assertIn("report/report.docx", relative)
            self.assertIn("code/results/E2.csv", relative)
            self.assertFalse(
                any(
                    "data_bundle" in name or "archive_pre_bindfix" in name
                    for name in names
                )
            )
            self.assertFalse(
                any(
                    ".conda" in name
                    or "__pycache__" in name
                    or "/bin/" in name
                    or "/build/" in name
                    for name in names
                )
            )
            self.assertEqual(
                source.read("000-测试/report/report.md"), b"report body preserved"
            )
            self.assertEqual(
                source.read("000-测试/code/environment.yml"),
                (self.root / "environment.yml").read_bytes(),
            )
            self.assertIsNone(source.testzip())

    @unittest.skipUnless(shutil.which("zipinfo"), "zipinfo is required by package.sh")
    def test_package_rejects_external_symlink_without_publishing_content(self):
        package = self.package_fixture()
        outside = self.base / "outside.txt"
        outside.write_text("external content must not be packaged")
        (self.code / "outside.txt").symlink_to(outside)
        completed = subprocess.run(
            ["bash", str(package), "000", "测试"],
            cwd=self.base,
            env=dict(os.environ, PYTHON=sys.executable),
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("符号链接", completed.stderr)
        self.assertFalse((self.base / "000-测试.zip").exists())


if __name__ == "__main__":
    unittest.main()
