#!/usr/bin/env python3
# plot_support.py：从真实测量生成统计与报告材料。
# 对应 DESIGN.md 第 7 节。
# 配置、数据来源及缺失状态由统一框架记录。
"""报告图表共用的中文标签、字体验证和图片清单。

字体必须指向能够绘制中文的真实常规字体文件；缺失时给出 conda 安装命令。
清单只登记调用者实际生成的图片，避免把旧图误当作本轮测量结果。
"""

import json
import math
import os
from pathlib import Path
import sys

import matplotlib
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text

IMPLEMENTATION_LABELS = {
    ("seq", "std"): "串行 std::sort",
    ("seq", "bucket"): "串行桶排序",
    ("pthreads", "locked"): "Pthreads 互斥锁桶排序",
    ("pthreads", "local"): "Pthreads 无锁桶排序",
    ("omp", "local"): "OpenMP 等宽桶",
    ("omp", "psrs"): "OpenMP PSRS",
    ("omp", "serial_init"): "OpenMP PSRS（串行初始化）",
    ("omp", "gnu_parallel"): "GNU parallel sort（参考）",
    ("omp", "omp_task_qsort"): "OpenMP 任务快速排序（参考）",
    ("ref", "gnu_parallel"): "GNU parallel sort（参考）",
    ("ref", "omp_task_qsort"): "OpenMP 任务快速排序（参考）",
    ("mpi", "psrs"): "MPI PSRS",
    ("mpi", "allgather"): "MPI PSRS（Allgather 对照）",
    ("hybrid", "psrs"): "MPI+OpenMP 混合",
    ("pipeline", "elem"): "MPI 逐元素流水线",
    ("pipeline", "block"): "MPI 分块流水线",
    ("gpu", "thrust"): "单卡 Thrust 排序",
    ("gpu", "sample"): "单卡 GPU 采样排序",
    ("gpu_multi", "sample"): "双卡 GPU 采样排序",
}
DISTRIBUTION_LABELS = {
    "uniform": "均匀分布",
    "normal": "正态分布",
    "power": "幂律分布",
    "hotkeys": "热点键",
    "sorted": "已排序",
    "reverse": "逆序",
    "nearly_sorted": "近乎有序",
    "few_unique": "少量不同键",
}
PHASE_LABELS = {
    "local_sort_ms": "本地排序",
    "sampling_ms": "采样",
    "splitters_ms": "选择分割点",
    "partition_ms": "划分",
    "scatter_ms": "初始数据分发",
    "exchange_ms": "数据交换",
    "merge_ms": "多路归并",
    "h2d_ms": "主机传至显卡",
    "d2h_ms": "显卡传回主机",
}
FIELD_LABELS = {
    "experiment": "实验",
    "quick": "运行模式",
    "host": "主机",
    "impl": "实现",
    "variant": "实现变体",
    "dist": "输入分布",
    "n": "数据规模 n",
    "procs": "MPI 进程数",
    "threads": "线程数",
    "gpus": "GPU 数量",
    "units": "处理单元数 p",
    "hybrid_ranks": "混合模式 MPI 进程数",
    "scale_n": "规模基准",
    "seed": "随机种子",
    "repeats": "测量次数",
    "median_ms": "总时间 (ms)",
    "total_ms": "总时间 (ms)",
    "min_ms": "最短时间 (ms)",
    "max_ms": "最长时间 (ms)",
    "baseline_ms": "串行基线时间 (ms)",
    "compute_ms": "计算时间 (ms)",
    "speedup": "加速比",
    "efficiency": "并行效率",
    "imbalance": "失衡度",
    "karp_flatt": "Karp–Flatt 实验串行比例",
    "merge_GBps": "归并有效带宽 (GB/s)",
    "best_GBps": "最佳带宽 (GB/s)",
    "median_GBps": "带宽中位数 (GB/s)",
    "kernel": "带宽测试内核",
    "oversample": "过采样系数 k",
    "buckets_per_thread": "每线程桶数",
    "schedule": "调度方式",
    "tiebreak": "复合键",
    "bind": "OpenMP 绑定方式",
    "places": "OpenMP 放置单位",
    "batch": "批大小",
    "naive_atomics": "原子操作方式",
    "scatter": "初始数据分发方式",
    "mempolicy": "内存策略",
    "mpi_bind": "MPI 绑定对象",
    "launcher": "MPI 实现",
    "gpu_affinity": "GPU 驱动线程绑定",
    "exchange_path": "双卡交换路径",
    "bind_policy": "绑定策略",
    "places_policy": "放置策略",
}
FIELD_LABELS.update(
    {field: title + "时间 (ms)" for field, title in PHASE_LABELS.items()}
)
TUNING_LABELS = {
    "schedule": {"static": "静态", "dynamic": "动态", "guided": "引导"},
    "bind": {"false": "不绑定", "close": "紧邻", "spread": "分散", "unset": "未记录"},
    "places": {"cores": "物理核", "threads": "硬件线程", "unset": "未记录"},
    "mempolicy": {"default": "系统默认", "interleave": "交错内存"},
    "mpi_bind": {
        "none": "无 MPI 绑定",
        "core": "物理核",
        "hwthread": "硬件线程",
        "socket": "CPU 插槽",
        "numa": "NUMA 节点",
        "numa:2": "两个 NUMA 节点",
        "user": "用户自定",
    },
    "launcher": {"mpich": "MPICH", "openmpi": "Open MPI", "none": "不适用"},
    "tiebreak": {"0": "关闭", "1": "开启"},
    "naive_atomics": {"0": "层次化原子", "1": "全局原子"},
    "scatter": {"0": "各进程本地生成", "1": "根进程生成后分发"},
    "gpu_affinity": {"numa": "GPU 本地 NUMA", "none": "不绑定", "legacy": "绑定未记录"},
    "exchange_path": {
        "peer": "显卡直接互访",
        "pinned_host": "锁页主机内存中转",
        "none": "无跨卡交换",
    },
    "kernel": {
        "Copy": "复制",
        "Triad": "三向运算",
        "copy": "复制",
        "triad": "三向运算",
    },
    "quick": {"true": "快速验证", "false": "完整实验"},
}

PREFERRED_FONT = Path(
    "/usr/share/fonts/opentype/OTF/SimplifiedChinese/NotoSerifCJKsc-Regular.otf"
)
FONT_INSTALL_COMMAND = "conda install -c conda-forge font-ttf-noto-cjk"
REQUIRED_GLYPHS = (
    "时间加速比效率处理单元数线程进程数据规模采样选择分割点归并交换带宽"
    "实验测量暂无完整快速验证开关闭合键不绑定紧邻分散物理核硬件主机显卡"
    "均匀正态幂律热点已排序逆序近乎有序少量不同任务互斥锁流水线"
    "−×α0123456789()/%"
)
REQUIRED_GLYPHS += "".join(
    str(value)
    for mapping in (
        IMPLEMENTATION_LABELS,
        DISTRIBUTION_LABELS,
        PHASE_LABELS,
        FIELD_LABELS,
    )
    for value in mapping.values()
)
REQUIRED_GLYPHS += "".join(
    value for mapping in TUNING_LABELS.values() for value in mapping.values()
)
_CHINESE_FONT = None


def _font_candidates():
    """Search conda and system font directories, preferring the known regular face."""
    roots = [Path("/usr/share/fonts"), Path("/usr/local/share/fonts")]
    prefixes = [Path(sys.prefix), Path(__file__).resolve().parents[2] / ".conda"]
    if os.environ.get("CONDA_PREFIX"):
        prefixes.insert(0, Path(os.environ["CONDA_PREFIX"]))
    roots.extend(prefix / "share/fonts" for prefix in prefixes)
    paths = {PREFERRED_FONT}
    for root in roots:
        if root.is_dir():
            paths.update(
                path
                for path in root.rglob("*")
                if path.suffix.lower() in (".ttf", ".otf", ".ttc")
            )

    def priority(path):
        name = path.name.lower()
        return (
            path != PREFERRED_FONT,
            "regular" not in name,
            "cjk" not in name,
            "noto" not in name,
            str(path),
        )

    return sorted(paths, key=priority)


def _valid_regular_font(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        face = FT2Font(str(path))
        style = (face.style_name or "").lower()
        if any(
            word in style
            for word in ("bold", "black", "light", "medium", "italic", "oblique")
        ):
            return None
        coverage = face.get_charmap()
        if any(ord(char) not in coverage for char in REQUIRED_GLYPHS):
            return None
        return face.family_name
    except (OSError, RuntimeError, ValueError):
        return None


def configure_chinese_font():
    """Return a validated file-based Chinese font; never fall back to Latin glyphs."""
    global _CHINESE_FONT
    for candidate in _font_candidates():
        family = _valid_regular_font(candidate)
        if family is None:
            continue
        path = str(Path(candidate).resolve())
        font_manager.fontManager.addfont(path)
        matplotlib.rcParams.update(
            {
                "font.family": [family],
                "font.weight": "normal",
                "axes.titleweight": "normal",
                "axes.labelweight": "normal",
                "axes.unicode_minus": True,
            }
        )
        _CHINESE_FONT = FontProperties(fname=path, style="normal", weight="normal")
        return _CHINESE_FONT.copy()
    raise RuntimeError(
        "未找到能够完整绘制简体中文的常规字体文件。请在项目 conda 环境中执行："
        + FONT_INSTALL_COMMAND
    )


def apply_font(fig):
    """Apply the selected physical font file to every text artist, preserving size."""
    selected = _CHINESE_FONT or configure_chinese_font()
    for text in fig.findobj(match=Text):
        properties = selected.copy()
        properties.set_size(text.get_fontsize())
        text.set_fontproperties(properties)


def _value(row, key, default=""):
    value = row.get(key, default) if hasattr(row, "get") else getattr(row, key, default)
    return default if value is None or str(value) in ("nan", "<NA>", "None") else value


def _token(value):
    text = str(value).strip()
    try:
        number = float(text)
        if math.isfinite(number) and number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def tuning_label(field, value):
    """Translate metadata values while retaining numbers and unknown provenance."""
    token = _token(value)
    if token.lower() in ("true", "false"):
        token = token.lower()
        if field in ("tiebreak", "naive_atomics", "scatter"):
            token = "1" if token == "true" else "0"
    mapping = TUNING_LABELS.get(field, {})
    if token in mapping:
        return mapping[token]
    if field == "gpu_affinity" and token.startswith("none("):
        return "不绑定（拓扑信息不可用）"
    if field == "mpi_bind" and ":" in token:
        object_name, count = token.split(":", 1)
        if object_name in mapping:
            return count + " 个" + mapping[object_name]
    return token


def label(row):
    """Readable implementation identity, including algorithm and placement controls."""
    impl, variant = str(_value(row, "impl")), str(_value(row, "variant"))
    value = IMPLEMENTATION_LABELS.get(
        (impl, variant), "未知实现（" + impl + "/" + variant + "）"
    )
    if impl == "hybrid":
        value += "（" + _token(_value(row, "procs", 1)) + " 个 MPI 进程）"
    if (
        variant in ("psrs", "allgather")
        and tuning_label("tiebreak", _value(row, "tiebreak", 1)) == "关闭"
    ):
        value += "（关闭复合键）"
    if impl == "gpu" and variant == "sample":
        value += (
            "（" + tuning_label("naive_atomics", _value(row, "naive_atomics", 0)) + "）"
        )
    if impl in ("gpu", "gpu_multi"):
        value += (
            "（"
            + tuning_label("gpu_affinity", _value(row, "gpu_affinity", "legacy"))
            + "）"
        )
    return value


def migrate_quick(figures_dir):
    """Move old top-level quick PNG/PDF artifacts into their dedicated directory."""
    figures_dir = Path(figures_dir)
    moved = []
    for suffix in ("png", "pdf"):
        for source in sorted(figures_dir.glob(f"*_quick.{suffix}")):
            if not source.is_file():
                continue
            target = figures_dir / "quick" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            moved.append(target)
    return moved


class FigureRegistry:
    """Manifest of generated figures, with paths relative to ``report/``."""

    def __init__(self, figures_dir=None, *, report_dir=None):
        if figures_dir is None:
            if report_dir is None:
                raise TypeError("必须指定 figures_dir 或 report_dir")
            figures_dir = Path(report_dir) / "figures"
        self.figures_dir = Path(figures_dir)
        self.report_dir = (
            Path(report_dir) if report_dir is not None else self.figures_dir.parent
        )
        self.records = []
        self._indices = {}

    def _relative_path(self, value):
        path = Path(value)
        if path.is_absolute():
            path = path.resolve().relative_to(self.report_dir.resolve())
        if ".." in path.parts:
            raise ValueError("清单路径必须位于 report 目录内")
        return path.as_posix()

    def add(self, name, title, experiment, quick, source_table, series, paths):
        record = {
            "paths": [self._relative_path(path) for path in paths],
            "title": str(title),
            "experiment": str(experiment),
            "quick": bool(quick),
            "source_table": self._relative_path(source_table),
            "series": [series] if isinstance(series, str) else list(series),
        }
        identity = (str(name), bool(quick))
        if identity in self._indices:
            self.records[self._indices[identity]] = record
        else:
            self._indices[identity] = len(self.records)
            self.records.append(record)
        return record

    def write(self):
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        path = self.figures_dir / "manifest.json"
        path.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path
