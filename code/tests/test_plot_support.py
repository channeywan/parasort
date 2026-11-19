#!/usr/bin/env python3
# test_plot_support.py：验证算法输出和实验方法的边界条件。
# 对应 DESIGN.md 第 9 节。
# 配置、数据来源及缺失状态由统一框架记录。
"""验证中文字体、图例身份与快速图片清单，不依赖实验测量。"""

import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

with tempfile.TemporaryDirectory() as cache:
    with patch.dict(os.environ, {"MPLCONFIGDIR": cache}):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ft2font import FT2Font
        from matplotlib.text import Text

        SPEC = importlib.util.spec_from_file_location(
            "tested_plot_support",
            Path(__file__).resolve().parents[1] / "analysis/plot_support.py",
        )
        SUPPORT = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(SUPPORT)


class ChineseFontTests(unittest.TestCase):
    def test_regular_file_draws_chinese_without_glyph_or_lookup_warnings(self):
        log = io.StringIO()
        handler = logging.StreamHandler(log)
        logger = logging.getLogger("matplotlib.font_manager")
        logger.addHandler(handler)
        self.addCleanup(logger.removeHandler, handler)
        with matplotlib.rc_context({"font.family": ["SimHei"], "font.weight": "bold"}):
            font = SUPPORT.configure_chinese_font()
            path = Path(font.get_file())
            self.assertTrue(path.is_file())
            face = FT2Font(str(path))
            self.assertTrue(
                all(ord(char) in face.get_charmap() for char in SUPPORT.REQUIRED_GLYPHS)
            )
            self.assertNotIn("bold", face.style_name.lower())
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fig, ax = plt.subplots()
                ax.plot([1, 2], [2, 3], label="OpenMP 等宽桶")
                ax.set(
                    title="强扩展与归并带宽", xlabel="处理单元数 p", ylabel="时间 (ms)"
                )
                ax.legend()
                note = ax.text(1, 2.5, "各阶段取进程最大值", fontsize=13, weight="bold")
                SUPPORT.apply_font(fig)
                self.assertEqual(note.get_fontsize(), 13)
                for text in fig.findobj(match=Text):
                    self.assertEqual(Path(text.get_fontproperties().get_file()), path)
                fig.savefig(io.BytesIO(), format="png")
                fig.savefig(io.BytesIO(), format="pdf")
                plt.close(fig)
            messages = [str(item.message) for item in caught]
            self.assertFalse(
                any("glyph" in message.lower() for message in messages), messages
            )
            self.assertNotIn("findfont", log.getvalue().lower())
            self.assertNotIn("fontweight", log.getvalue().lower())

    def test_invalid_or_latin_only_fonts_require_conda_installation(self):
        latin = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "SimHei.ttf"
            invalid.write_text("这不是字体文件", encoding="utf-8")
            with patch.object(
                SUPPORT, "_font_candidates", return_value=[invalid, latin]
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "conda install.*font-ttf-noto-cjk"
                ):
                    SUPPORT.configure_chinese_font()


class ChineseLabelTests(unittest.TestCase):
    def test_failed_gpu_affinity_is_explicit_without_changing_success_labels(self):
        gpu = {"impl": "gpu_multi", "variant": "sample", "gpu_affinity": "numa"}
        normal = SUPPORT.label(gpu)
        self.assertEqual(SUPPORT.label(dict(gpu, gpu_affinity_actual="numa")), normal)
        failed = SUPPORT.label(dict(gpu, gpu_affinity_actual="none(setaffinity_22)"))
        self.assertIn("请求未生效", failed)
        self.assertIn("实际未绑定", failed)
        self.assertNotIn("GPU 本地 NUMA", failed)
        self.assertIn("部分生效", SUPPORT.label(dict(gpu, gpu_affinity_actual="mixed")))

    def test_labels_keep_relevant_variants_and_fixed_hybrid_rank_count(self):
        self.assertIn(
            "关闭复合键",
            SUPPORT.label({"impl": "omp", "variant": "psrs", "tiebreak": "0.0"}),
        )
        hybrid = SUPPORT.label({"impl": "hybrid", "variant": "psrs", "procs": 8.0})
        self.assertIn("8 个 MPI 进程", hybrid)
        self.assertNotIn("8.0", hybrid)
        gpu = {
            "impl": "gpu",
            "variant": "sample",
            "naive_atomics": "0.0",
            "gpu_affinity": "numa",
        }
        self.assertIn("层次化原子", SUPPORT.label(gpu))
        self.assertIn("GPU 本地 NUMA", SUPPORT.label(gpu))
        gpu.update(naive_atomics="1.0", gpu_affinity="none")
        self.assertIn("全局原子", SUPPORT.label(gpu))
        self.assertIn("不绑定", SUPPORT.label(gpu))
        self.assertEqual(SUPPORT.tuning_label("quick", True), "快速验证")
        self.assertEqual(SUPPORT.DISTRIBUTION_LABELS["power"], "幂律分布")
        self.assertEqual(SUPPORT.PHASE_LABELS["scatter_ms"], "初始数据分发")


class FigureRegistryTests(unittest.TestCase):
    def test_migration_keeps_full_figures_and_manifest_paths_are_report_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report"
            figures = report / "figures"
            figures.mkdir(parents=True)
            (figures / "E2.png").write_bytes("完整图片".encode("utf-8"))
            (figures / "E2_quick.png").write_bytes("快速图片".encode("utf-8"))
            (figures / "E2_quick.pdf").write_bytes("快速文档".encode("utf-8"))
            moved = SUPPORT.migrate_quick(figures)
            self.assertEqual(
                {path.name for path in moved}, {"E2_quick.png", "E2_quick.pdf"}
            )
            self.assertTrue(all(path.parent == figures / "quick" for path in moved))
            self.assertEqual(
                (figures / "E2.png").read_bytes(), "完整图片".encode("utf-8")
            )
            self.assertFalse(list(figures.glob("*_quick.*")))
            self.assertEqual(SUPPORT.migrate_quick(figures), [])
            registry = SUPPORT.FigureRegistry(figures)
            registry.add(
                "E2_quick",
                "强扩展快速验证",
                "E2",
                True,
                "tables/E2_quick.md",
                ["OpenMP PSRS 的实测加速比"],
                moved,
            )
            registry.add(
                "E2",
                "强扩展",
                "E2",
                False,
                "tables/E2.md",
                ["MPI PSRS 的实测加速比"],
                [figures / "E2.png"],
            )
            # 再次保存同一张图时更新记录，不重复登记。
            registry.add(
                "E2",
                "强扩展与效率",
                "E2",
                False,
                "tables/E2.md",
                ["MPI PSRS 的实测加速比"],
                ["figures/E2.png"],
            )
            manifest = registry.write()
            records = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(len(records), 2)
            self.assertEqual(records[-1]["title"], "强扩展与效率")
            fields = {"paths", "title", "experiment", "quick", "source_table", "series"}
            for record in records:
                self.assertEqual(set(record), fields)
                self.assertIsInstance(record["quick"], bool)
                self.assertIsInstance(record["series"], list)
                for path in record["paths"]:
                    self.assertFalse(Path(path).is_absolute())
                    self.assertTrue((report / path).is_file())
            self.assertIn("强扩展", manifest.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
