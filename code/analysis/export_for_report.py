#!/usr/bin/env python3
"""Export verified measurements and factual methods for the report author.

This implements the data interface in IMPROVEMENTS.md T7. It never edits report.md,
never combines full and quick measurements, and uses only pandas beyond stdlib.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = [f"E{i}" for i in range(1, 15)] + ["E4b"]
REQUIRED = {
    "seq_std_2p30_ms": "E2", "best_cpu": "E2", "best_by_impl": "E2",
    "speedup_64_vs_128": "E2", "e4_skew": "E4", "e4_hotkeys_tiebreak": "E4",
    "e4b_oversample_best": "E4b", "e4b_tiebreak_vs_p": "E4b", "e5_phase_share": "E5",
    "e6_best_schedule": "E6", "e7_numa": "E7", "e8_lock": "E8", "e9_comm": "E9",
    "e10_pipeline": "E10", "e11_gpu": "E11", "e12_overview": "E12",
    "e13_amdahl": "E2", "e14_bandwidth": "E14", "correctness": "E1",
    "affinity_fix": "E2",
}
TUNING = [
    "seed", "oversample", "buckets_per_thread", "schedule", "tiebreak", "bind",
    "batch", "naive_atomics", "scatter", "mempolicy", "mpi_bind", "launcher",
    "places", "gpu_affinity", "gpu_affinity_requested", "exchange_path", "sweep",
]
DEFAULTS = {"seed": "42", "oversample": "1", "buckets_per_thread": "1", "schedule": "static",
            "tiebreak": "1", "bind": "unset", "batch": "0", "naive_atomics": "0", "scatter": "0",
            "mempolicy": "default", "mpi_bind": "none", "launcher": "none", "places": "unset",
            "gpu_affinity": "legacy", "gpu_affinity_requested": "", "exchange_path": "none", "sweep": ""}
NUMERIC_TUNING = {"seed", "oversample", "buckets_per_thread", "tiebreak", "batch", "naive_atomics", "scatter"}
IDENTITY = ["experiment", "quick", "host", "impl", "variant", "dist", "n", "procs", "threads", "gpus"] + TUNING
NUMERIC_CONFIG = {"n", "procs", "threads", "gpus"}
PHASES = ["local_sort_ms", "sampling_ms", "splitters_ms", "partition_ms", "exchange_ms", "merge_ms", "h2d_ms", "d2h_ms"]
METRICS = ["total_ms", "imbalance", "compute_ms", "scatter_ms"] + PHASES
IMPL_LABELS = {
    ("seq", "std"): "串行 std::sort", ("omp", "psrs"): "OpenMP PSRS",
    ("omp", "local"): "OpenMP 等宽桶", ("pthreads", "local"): "Pthreads 无锁桶",
    ("pthreads", "locked"): "Pthreads 共享锁桶", ("mpi", "psrs"): "MPI PSRS",
    ("mpi", "allgather"): "MPI PSRS Allgather", ("hybrid", "psrs"): "MPI+OpenMP 混合",
    ("omp", "gnu_parallel"): "GNU parallel sort（参考）", ("omp", "omp_task_qsort"): "OpenMP 任务快排（参考）",
    ("ref", "gnu_parallel"): "GNU parallel sort（参考）", ("ref", "omp_task_qsort"): "OpenMP 任务快排（参考）",
    ("gpu", "thrust"): "单卡 Thrust", ("gpu", "sample"): "单卡样本排序",
    ("gpu", "dual"): "双卡排序", ("gpu_multi", "sample"): "双卡样本排序", ("pipeline", "elem"): "逐元素流水线", ("pipeline", "block"): "块流水线",
}
DISTS = {"uniform": "均匀", "normal": "正态", "power": "幂律", "hotkeys": "热点键", "few_unique": "少量不同键", "sorted": "已排序", "reverse": "逆序", "nearly_sorted": "近乎有序"}


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def extras(value: Any) -> dict[str, str]:
    return dict(part.split("=", 1) for part in str(value).split(";") if "=" in part)


def numeric(value: Any) -> float:
    return float(value) if finite(value) else float("nan")


def signature(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=scalar)


def md_table(columns: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        value = scalar(value)
        if value is None:
            return "缺失"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"{value:.3g}"
        return str(value).replace("|", "/").replace("\n", " ")
    return "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |\n" + "".join("| " + " | ".join(cell(value) for value in row) + " |\n" for row in rows)


class Exporter:
    def __init__(self, results: Path, tables: Path, output: Path, quick: bool = False,
                 figures: Path | None = None, analysis: Path | None = None):
        self.results, self.tables, self.output = results, tables, output
        self.quick = quick
        self.figures = figures or tables.parent / "figures"
        self.analysis = analysis or ROOT / "analysis"
        self.missing: list[dict[str, Any]] = []
        self.inventory: list[dict[str, Any]] = []
        self.diagnostic_inventory: list[dict[str, Any]] = []
        self.load_issues: list[str] = []
        self.raw: list[dict[str, Any]] = []
        self.groups: list[dict[str, Any]] = []
        self.bandwidth: list[dict[str, Any]] = []
        self.statuses: list[dict[str, Any]] = []
        self._missing_seen: set[str] = set()
        self._summary_module = None
        self._summary_loaded = False

    def summarizer(self):
        if self._summary_loaded:
            return self._summary_module
        self._summary_loaded = True
        path = self.analysis / "summarize.py"
        if not path.exists():
            return None
        spec = importlib.util.spec_from_file_location("parasort_report_summarize", path)
        module = importlib.util.module_from_spec(spec)
        # Exporting a bundle need not create bytecode beside the source script.
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous
        self._summary_module = module
        return module

    def tuning_value(self, key: str, value: Any) -> str:
        module = self.summarizer()
        defaults = dict(DEFAULTS, **getattr(module, "DEFAULTS", {}))
        if value is None or (isinstance(value, float) and not finite(value)):
            return defaults.get(key, "")
        text = str(value).strip()
        if key in NUMERIC_TUNING and finite(text) and float(text).is_integer():
            return str(int(float(text)))
        return text

    def path_name(self, path: Path) -> str:
        if path.parent == self.results:
            return "code/results/" + path.name
        if path.parent == self.results / "archive_pre_bindfix":
            return "code/results/archive_pre_bindfix/" + path.name
        return path.name

    def missing_value(self, path: str, reason: str, config: dict | None = None,
                      unit: str = "", quick: bool | None = None) -> dict:
        item = {"path": path, "reason": reason, "expected_config": config or {}}
        key = signature(item)
        if key not in self._missing_seen:
            self._missing_seen.add(key)
            self.missing.append(item)
        return {"value": None, "unit": unit, "source": None, "config": config or {},
                "quick": self.quick if quick is None else quick}

    def source(self, experiment: str, quick: bool | None = None) -> str:
        quick = self.quick if quick is None else quick
        name = f"{experiment}{'_quick' if quick else ''}.md"
        if (self.tables / name).exists():
            return "tables/" + name
        return f"../code/results/{experiment}{'_quick' if quick else ''}.csv"

    def leaf(self, value: Any, unit: str, group: dict | None = None, *, path: str = "",
             rows: list[dict] | None = None, config: dict | None = None,
             source: str | None = None, statistic: str = "median", formula: str | None = None,
             inputs: list[dict] | None = None, quick: bool | None = None) -> dict:
        config = config if config is not None else (group["config"] if group else {})
        rows = rows if rows is not None else (group["rows"] if group else [])
        quick = bool(group["config"].get("quick", self.quick)) if quick is None and group else (self.quick if quick is None else quick)
        if value is None or (isinstance(value, (int, float)) and not finite(value)):
            return self.missing_value(path, "没有可验证的数值或匹配的比较配置", config, unit, quick)
        references: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            references[row["_file"]].append(int(row["_line"]))
        result = {
            "value": scalar(value), "unit": unit,
            "source": source or self.source(config.get("experiment", "E2"), quick),
            "config": config, "quick": quick, "statistic": statistic,
            "raw_sources": [{"file": file, "rows": sorted(set(lines)), "verified": 1}
                            for file, lines in sorted(references.items())],
        }
        if formula:
            result["formula"] = formula
        if inputs:
            result["inputs"] = inputs
        return result

    def read_csv(self, path: Path, experiment: str, quick: bool, *, inventory: bool = True) -> list[dict]:
        try:
            frame = pd.read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            self.load_issues.append(f"{self.path_name(path)}: {error}")
            return []
        if "verified" not in frame:
            self.load_issues.append(f"{self.path_name(path)} 缺少 verified 列，未导出")
            return []
        verified = pd.to_numeric(frame["verified"], errors="coerce").eq(1)
        target_inventory = self.inventory if inventory else self.diagnostic_inventory
        if not any(item["file"] == self.path_name(path) for item in target_inventory):
            target_inventory.append({"file": self.path_name(path), "rows_total": len(frame),
                                   "rows_verified": int(verified.sum()),
                                   "rows_excluded": int((~verified).sum()),
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "quick": quick})
        records = []
        for index, series in frame[verified].iterrows():
            row = {key: scalar(value) for key, value in series.items()}
            row.update(extras(row.get("extra", "")))
            row.update(experiment=experiment, quick=quick, _file=self.path_name(path), _line=int(index) + 2)
            if experiment != "E1" and row.get("impl") in {"mpi", "hybrid", "pipeline"} and row.get("mpi_bind", "none") in {"", "none", "unset"} and path.parent.name != "archive_pre_bindfix":
                self.load_issues.append(f"{self.path_name(path)}:{index + 2} 缺少修复后的 mpi_bind，未导出")
                continue
            records.append(row)
        return records

    def load(self) -> None:
        suffix = "_quick" if self.quick else ""
        for experiment in EXPERIMENTS:
            path = self.results / f"{experiment}{suffix}.csv"
            if not path.exists():
                continue
            rows = self.read_csv(path, experiment, self.quick)
            if experiment == "E14":
                for row in rows:
                    if all(key in row for key in ("threads", "bind", "kernel", "best_GBps", "median_GBps")) and all(finite(row.get(key)) and float(row[key]) > 0 for key in ("threads", "best_GBps", "median_GBps")):
                        self.bandwidth.append(row)
                    else:
                        self.load_issues.append(f"{row['_file']}:{row['_line']} 缺少 STREAM 字段")
            else:
                self.raw.extend(rows)
        buckets: dict[str, list[dict]] = defaultdict(list)
        configs: dict[str, dict] = {}
        for row in self.raw:
            if not all(key in row for key in ("host", "impl", "variant", "dist", "n", "procs", "threads", "gpus", "total_ms")) or not finite(row["total_ms"]):
                self.load_issues.append(f"{row['_file']}:{row['_line']} 缺少排序字段或有效 total_ms")
                continue
            config = {key: self.tuning_value(key, row.get(key)) if key in TUNING else row.get(key, "") for key in IDENTITY}
            for key in NUMERIC_CONFIG:
                config[key] = int(config[key])
            config["quick"] = bool(config["quick"])
            key = signature(config)
            buckets[key].append(row)
            configs[key] = config
        for key in sorted(buckets):
            rows = buckets[key]
            metrics = {}
            for field in METRICS:
                values = pd.Series([numeric(row.get(field)) for row in rows], dtype=float).dropna()
                metrics[field] = float(values.median()) if not values.empty else None
            total = [float(row["total_ms"]) for row in rows]
            metrics.update(min_ms=min(total), max_ms=max(total), repeats=len(total))
            self.groups.append({"config": configs[key], "metrics": metrics, "rows": rows})
        for path in sorted(self.results.glob(f"E*{suffix}_status.json")):
            # The suffix is exact: default full mode must not ingest *_quick_status.json.
            if not self.quick and path.name.endswith("_quick_status.json"):
                continue
            try:
                status = json.loads(path.read_text())
                if bool(status.get("quick", "_quick" in path.name)) != self.quick:
                    continue
                status["_file"] = self.path_name(path)
                self.statuses.append(status)
            except (OSError, ValueError) as error:
                self.load_issues.append(f"{path.name}: {error}")

    def select(self, experiment: str | None = None, **filters) -> list[dict]:
        def matches(group: dict) -> bool:
            config = group["config"]
            return (experiment is None or config["experiment"] == experiment) and all(str(config.get(key, "")) == str(value) for key, value in filters.items())
        return [group for group in self.groups if matches(group)]

    def units(self, group: dict) -> int:
        config = group["config"]
        return config["gpus"] or config["procs"] * config["threads"]

    def family(self, group: dict, remove: set[str] | None = None) -> str:
        config = dict(group["config"])
        module = self.summarizer()
        if config["experiment"] == "E2" and not remove and hasattr(module, "scaling_identity"):
            frame = pd.DataFrame([dict(config, units=self.units(group))])
            prepared = module.scaling_identity(frame)
            keys = module.scaling_keys(prepared)
            identity = {key.removeprefix("_identity_"): scalar(prepared.iloc[0][key]) for key in keys}
            identity["experiment"] = config["experiment"]
            if config["mpi_bind"] == "user":
                # The CSV's opaque "user" tag does not prove that two distinct
                # MPIEXEC_FLAGS allocations use a matched mapping across p.
                identity["user_parallelism"] = [config["procs"], config["threads"]]
            if config["impl"] == "hybrid":
                identity["procs"] = config["procs"]
            return signature(identity)
        ignored = {"procs", "threads", "places", "mpi_bind"} | (remove or set())
        if config["impl"] == "hybrid":
            ignored.discard("procs")
        expected = "core" if config["procs"] <= 64 else "hwthread"
        mpi_auto = config["impl"] in {"mpi", "pipeline"} and config["threads"] == 1 and config["bind"] == "false" and config["launcher"] in {"mpich", "openmpi"} and config["mpi_bind"] == expected
        hybrid_auto = config["impl"] == "hybrid" and config["bind"] == "close" and config["launcher"] in {"mpich", "openmpi"} and config["mpi_bind"] == {2: "socket", 4: "numa:2", 8: "numa"}.get(config["procs"])
        if not (mpi_auto or hybrid_auto):
            ignored.discard("mpi_bind")
        capacity = 64 // config["procs"] if hybrid_auto else 64
        automatic_places = (config["impl"] in {"omp", "ref"} and config["procs"] == 1) or hybrid_auto
        if not automatic_places or config["places"] != ("cores" if config["threads"] <= capacity else "threads"):
            ignored.discard("places")
        if config["mpi_bind"] == "user":
            ignored -= {"procs", "threads", "mpi_bind", "places"}
        return signature({key: value for key, value in config.items() if key not in ignored})

    def baseline(self, group: dict) -> dict | None:
        config = group["config"]
        candidates = [item for item in self.select(impl="seq", variant="std") if all(item["config"][key] == config[key] for key in ("host", "quick", "n", "dist", "seed"))]
        same = [item for item in candidates if item["config"]["experiment"] == config["experiment"]]
        candidates = same or candidates
        if len(candidates) > 1:
            exact = [item for item in candidates if item["config"]["bind"] == config["bind"]]
            candidates = exact if len(exact) == 1 else candidates
        return candidates[0] if len(candidates) == 1 else None

    def metric(self, group: dict, field: str, path: str, unit: str = "ms") -> dict:
        return self.leaf(group["metrics"].get(field), unit, group, path=path)

    def ratio(self, numerator: dict, denominator: dict, path: str, *, field: str = "total_ms", scale: float = 1, unit: str = "x", formula: str | None = None) -> dict:
        top, bottom = numerator["metrics"].get(field), denominator["metrics"].get(field)
        value = top / bottom * scale if finite(top) and finite(bottom) and bottom > 0 else None
        config = {"numerator": numerator["config"], "denominator": denominator["config"]}
        return self.leaf(value, unit, rows=numerator["rows"] + denominator["rows"], config=config,
                         source=self.source(denominator["config"]["experiment"]), path=path,
                         formula=formula or f"median({field}, numerator) / median({field}, denominator) * {scale}",
                         inputs=[self.metric(numerator, field, path + ".numerator"), self.metric(denominator, field, path + ".denominator")])

    def record(self, group: dict, path: str, fields: list[str] | None = None, speedup: bool = True) -> dict:
        record = {"config": group["config"], "p": self.leaf(self.units(group), "units", group, statistic="procs*threads or gpus")}
        for field in fields or ["total_ms", "imbalance"]:
            record[field] = self.metric(group, field, path + "." + field, "ratio" if field == "imbalance" else "ms")
        for field, unit, statistic in (("min_ms", "ms", "min"), ("max_ms", "ms", "max"), ("repeats", "runs", "count verified rows")):
            record[field] = self.leaf(group["metrics"][field], unit, group, statistic=statistic)
        if speedup and group["config"]["impl"] != "seq":
            baseline = self.baseline(group)
            if baseline:
                record["speedup"] = self.ratio(baseline, group, path + ".speedup")
                record["efficiency"] = self.ratio(baseline, group, path + ".efficiency", scale=1 / self.units(group), unit="ratio")
            else:
                for name, unit in (("speedup", "x"), ("efficiency", "ratio")):
                    record[name] = self.missing_value(path + "." + name, "缺少同主机、种子、规模、分布的唯一串行基线", group["config"], unit)
        return record

    def section(self, key: str, value: Any, experiment: str | None = None, *, quick: bool | None = None, note: str | None = None) -> dict:
        if value is None or value == [] or value == {}:
            return self.missing_value(key, "缺少所选模式下 verified=1 的完整匹配配置", {"experiment": experiment or REQUIRED[key]})
        result = {"value": value, "source": self.source(experiment or REQUIRED[key], quick), "quick": self.quick if quick is None else quick}
        if note:
            result["note"] = note
        return result

    def best_cpu(self) -> dict:
        rows = [group for group in self.select("E2", dist="uniform") if group["config"]["gpus"] == 0 and group["config"]["impl"] != "seq"]
        if not rows:
            return {}
        target = max(group["config"]["n"] for group in rows) if self.quick else 1 << 30
        rows = [group for group in rows if group["config"]["n"] == target]
        return self.record(min(rows, key=lambda item: item["metrics"]["total_ms"]), "best_cpu") if rows else {}

    def best_by_impl(self) -> list[dict]:
        grouped = defaultdict(list)
        for group in self.select("E2", dist="uniform"):
            if group["config"]["gpus"] == 0 and group["config"]["impl"] != "seq" and (self.quick or group["config"]["n"] == 1 << 30):
                grouped[self.family(group)].append(group)
        return [self.record(min(values, key=lambda item: item["metrics"]["total_ms"]), "best_by_impl") for _, values in sorted(grouped.items())]

    def scaling_64_128(self) -> list[dict]:
        grouped = defaultdict(dict)
        for group in self.select("E2", dist="uniform"):
            if group["config"]["gpus"] == 0 and group["config"]["impl"] != "seq" and self.units(group) in {64, 128}:
                grouped[self.family(group)][self.units(group)] = group
        result = []
        for family, points in sorted(grouped.items()):
            record = {"family": json.loads(family)}
            for p in (64, 128):
                record[f"p{p}"] = self.record(points[p], f"speedup_64_vs_128.p{p}") if p in points else self.missing_value(f"speedup_64_vs_128.p{p}", "缺少同一实现和拓扑的处理单元配置", {**json.loads(family), "p": p})
            record["speedup_128_over_64"] = self.ratio(points[64], points[128], "speedup_64_vs_128.ratio") if len(points) == 2 else self.missing_value("speedup_64_vs_128.ratio", "缺少 p=64 或 p=128 的匹配配置", json.loads(family), "x")
            result.append(record)
        return result

    def paired(self, experiment: str, distributions: list[str], left_filter: dict, right_filter: dict, path: str, *, remove: set[str], allowed_procs: set[int] | None = None) -> list[dict]:
        pairs = []
        for dist in distributions:
            rows = [row for row in self.select(experiment, dist=dist) if allowed_procs is None or row["config"]["procs"] in allowed_procs]
            if not rows:
                self.missing_value(f"{path}.{dist}", "缺少所需分布", {"experiment": experiment, "dist": dist})
                continue
            def identity(group):
                return signature({key: value for key, value in group["config"].items() if key not in remove})
            left = {identity(group): group for group in rows if all(str(group["config"].get(key)) == str(value) for key, value in left_filter.items())}
            right = {identity(group): group for group in rows if all(str(group["config"].get(key)) == str(value) for key, value in right_filter.items())}
            for key in sorted(left.keys() | right.keys()):
                record = {"config": json.loads(key)}
                for name, values in (("left", left), ("right", right)):
                    record[name] = self.record(values[key], f"{path}.{dist}.{name}", speedup=False) if key in values else self.missing_value(f"{path}.{dist}.{name}", "缺少比较配置", {**json.loads(key), **(left_filter if name == "left" else right_filter)})
                if key in left and key in right:
                    for field in ("total_ms", "imbalance"):
                        record[field + "_left_over_right"] = self.ratio(left[key], right[key], f"{path}.{dist}.{field}", field=field)
                pairs.append(record)
        return pairs

    def oversample(self) -> list[dict]:
        grouped = defaultdict(list)
        for group in self.select("E4b", impl="omp", variant="psrs", tiebreak="1"):
            if group["config"]["dist"] in {"uniform", "normal", "hotkeys"} and self.units(group) == 64 and int(group["config"].get("oversample") or 0) in {1, 2, 4, 8, 16, 32}:
                config = {key: value for key, value in group["config"].items() if key not in {"oversample", "sweep"}}
                grouped[signature(config)].append(group)
        result = []
        for _, rows in sorted(grouped.items()):
            best = min(rows, key=lambda item: item["metrics"]["total_ms"])
            result.append({"best": self.record(best, "e4b_oversample_best", ["total_ms", "imbalance", "splitters_ms"], False), "observed_k": self.leaf(sorted({int(row["config"]["oversample"]) for row in rows}), "oversample", rows=[raw for row in rows for raw in row["rows"]], config={key: value for key, value in best["config"].items() if key not in {"oversample", "sweep"}}, statistic="observed parameter values")})
        return result

    def phase_share(self) -> list[dict]:
        result = []
        for group in self.select("E5"):
            if group["config"]["impl"] == "seq":
                continue
            record = self.record(group, "e5_phase_share", ["total_ms"], False)
            total = group["metrics"]["total_ms"]
            record["phases"] = {field: self.leaf((value / total * 100) if finite(value) and total > 0 else None, "%", group, path=f"e5_phase_share.{field}", source="tables/summary.csv", formula=f"median({field}) / median(total_ms) * 100") for field in PHASES for value in [group["metrics"].get(field)]}
            result.append(record)
        return result

    def scheduling(self) -> list[dict]:
        grouped = defaultdict(list)
        for group in self.select("E6", impl="omp", variant="psrs"):
            key = signature({key: value for key, value in group["config"].items() if key not in {"buckets_per_thread", "schedule"}})
            grouped[key].append(group)
        result = []
        for key, rows in sorted(grouped.items()):
            best = min(rows, key=lambda item: item["metrics"]["total_ms"])
            baseline = [row for row in rows if row["config"]["schedule"] == "static" and str(row["config"]["buckets_per_thread"]) == "1"]
            record = {"config": json.loads(key), "best": self.record(best, "e6_best_schedule.best", speedup=False)}
            record["relative_to_static_k1"] = self.ratio(baseline[0], best, "e6_best_schedule.relative_to_static_k1") if len(baseline) == 1 else self.missing_value("e6_best_schedule.relative_to_static_k1", "缺少唯一 static、k=1 基线", json.loads(key), "x")
            result.append(record)
        return result

    def lock_contention(self) -> dict:
        rows = self.select("E8", impl="pthreads")
        locked = [row for row in rows if row["config"]["variant"] == "locked"]
        return {"measurements": [self.record(row, "e8_lock", speedup=False) for row in rows], "fastest_locked": self.record(min(locked, key=lambda item: item["metrics"]["total_ms"]), "e8_lock.fastest_locked", speedup=False) if locked else self.missing_value("e8_lock.fastest_locked", "缺少 locked 配置")} if rows else {}

    def communication(self) -> dict:
        rows = [row for row in self.select("E9") if row["config"]["impl"] in {"mpi", "hybrid"}]
        if not rows:
            return {}
        pairs = self.paired("E9", ["uniform"], {"impl": "mpi", "variant": "psrs", "scatter": "1"}, {"impl": "mpi", "variant": "psrs", "scatter": "0"}, "e9_comm.scatter", remove={"scatter"}, allowed_procs={8, 64})
        measures = []
        for row in rows:
            record = self.record(row, "e9_comm", ["total_ms", "exchange_ms", "scatter_ms"], False)
            total, scatter = row["metrics"]["total_ms"], row["metrics"].get("scatter_ms")
            record["sort_plus_scatter_ms"] = self.leaf(total + scatter if finite(scatter) else None, "ms", row, path="e9_comm.sort_plus_scatter_ms", formula="median(total_ms) + median(scatter_ms); excludes generation")
            measures.append(record)
        return {"measurements": measures, "scatter_vs_local": pairs, "note": "total_ms 是排序时间，不包括生成和 Scatter；scatter_ms 单列记录，sort_plus_scatter_ms 仍不包括 root 生成。"}

    def pipeline(self) -> dict:
        elem, block = [], defaultdict(list)
        for row in self.select("E10", impl="pipeline"):
            if row["config"]["variant"] == "elem":
                elem.append(self.record(row, "e10_pipeline.elem", speedup=False))
            else:
                block[self.family(row, {"batch"}) + str(self.units(row))].append(row)
        bests = []
        for rows in block.values():
            best = min(rows, key=lambda item: item["metrics"]["total_ms"])
            c = best["config"]
            matches = [row for row in self.select("E10", impl="mpi", variant="psrs") if all(row["config"][key] == c[key] for key in ("host", "n", "dist", "seed", "procs", "threads", "quick"))]
            record = {"best_block": self.record(best, "e10_pipeline.best_block", speedup=False)}
            record["block_over_psrs_time"] = self.ratio(best, matches[0], "e10_pipeline.block_over_psrs_time") if len(matches) == 1 else self.missing_value("e10_pipeline.block_over_psrs_time", "缺少同规模、同 rank 数的唯一 PSRS-MPI 对照", c, "x")
            bests.append(record)
        return {"elem": elem, "best_block_by_n_p": bests} if elem or bests else {}

    def gpu(self) -> dict:
        rows = [row for row in self.select("E11") if row["config"]["gpus"] > 0]
        if not rows:
            return {}
        result = []
        for row in rows:
            record = self.record(row, "e11_gpu", ["total_ms", "compute_ms", "h2d_ms", "d2h_ms"], False)
            metrics = row["metrics"]
            copies = metrics.get("h2d_ms"), metrics.get("d2h_ms")
            value = sum(copies) / metrics["total_ms"] * 100 if all(finite(part) for part in copies) and metrics["total_ms"] > 0 else None
            record["copy_share"] = self.leaf(value, "%", row, path="e11_gpu.copy_share", formula="(median(h2d_ms) + median(d2h_ms)) / median(total_ms) * 100")
            paths = sorted({str(value) for raw in row["rows"] for key, value in raw.items() if key in {"exchange_path", "transfer", "xfer", "peer", "p2p", "exchange"} and value not in {None, ""}})
            record["exchange_paths"] = {"value": paths or None, "source": self.source("E11"), "config": row["config"], "quick": self.quick, "raw_sources": self.leaf(0, "", row)["raw_sources"]}
            result.append(record)
        comparisons = []
        for dual in rows:
            if dual["config"]["gpus"] != 2 or str(dual["config"].get("gpu_affinity", "")).startswith("none"):
                continue
            c = dual["config"]
            singles = [row for row in rows if row["config"]["gpus"] == 1 and row["config"]["variant"] == "thrust" and all(row["config"][key] == c[key] for key in ("host", "n", "dist", "seed", "quick", "gpu_affinity"))]
            if len(singles) == 1:
                comparisons.append(self.ratio(singles[0], dual, "e11_gpu.dual_speedup"))
            else:
                comparisons.append(self.missing_value("e11_gpu.dual_speedup", "缺少同规模、分布、种子的唯一单卡 Thrust 对照", c, "x"))
        affinity = []
        by_affinity = defaultdict(dict)
        for row in rows:
            if row["config"]["gpus"] != 2:
                continue
            request = row["config"].get("gpu_affinity_requested") or row["config"].get("gpu_affinity", "")
            mode = "none" if str(request).startswith("none") else "numa" if str(request).startswith("numa") else "unknown"
            key = signature({key: value for key, value in row["config"].items() if key not in {"gpu_affinity", "gpu_affinity_requested"}})
            by_affinity[key][mode] = row
        for key, values in by_affinity.items():
            # The requested affinity ablation is at n=2^30. Quick mode records
            # its actual scaled n; do not demand none at every ordinary GPU size.
            if json.loads(key)["n"] != 1 << 30 and "none" not in values:
                continue
            record = {"config": json.loads(key)}
            for mode in ("numa", "none"):
                record[mode] = self.record(values[mode], "e11_gpu.affinity." + mode, speedup=False) if mode in values else self.missing_value("e11_gpu.affinity." + mode, "缺少双卡 NUMA/none 对照", {**json.loads(key), "gpu_affinity_requested": mode})
            if "numa" in values and "none" in values:
                record["none_over_numa"] = self.ratio(values["none"], values["numa"], "e11_gpu.affinity.none_over_numa")
            affinity.append(record)
        return {"measurements": result, "dual_vs_single_thrust": comparisons, "affinity": affinity}

    def bandwidth_numbers(self) -> dict:
        streams = []
        for row in self.bandwidth:
            config = {key: row.get(key) for key in ("experiment", "quick", "host", "n", "threads", "bind", "bind_requested", "places", "kernel", "repeats", "warmups")}
            streams.append({"config": config, **{field: self.leaf(numeric(row.get(field)) if finite(row.get(field)) else None, "GB/s", rows=[row], config=config, source=self.source("E14"), path="e14_bandwidth." + field, statistic="best of measured repeats" if field == "best_GBps" else "median") for field in ("best_GBps", "median_GBps")}})
        effective = []
        for row in self.groups:
            c, merge = row["config"], row["metrics"].get("merge_ms")
            if c["experiment"] in {"E2", "E5"} and c["variant"] == "psrs" and c["impl"] in {"omp", "mpi", "hybrid"} and finite(merge) and merge > 0:
                effective.append(self.leaf(8 * c["n"] / (merge * 1e6), "GB/s", row, source="tables/summary.csv", path="e14_bandwidth.psrs_merge", formula="8*n / (median(merge_ms) * 10^6)"))
        if not streams and effective:
            self.missing_value("e14_bandwidth.stream", "缺少所选模式下 verified=1 的 STREAM CSV", {"experiment": "E14"})
        return {"stream": streams, "psrs_merge": effective} if streams or effective else {}

    def correctness(self) -> dict:
        rows = [row for row in self.raw if row["experiment"] == "E1"]
        if not rows:
            return {}
        # Counts describe the exported verified dataset; rejected rows remain visible
        # in the source inventory and load issues, never disguised as successful runs.
        config = {"experiment": "E1", "quick": self.quick, "scope": "exported verified rows"}
        result = {"total": self.leaf(len(rows), "cases", rows=rows, config=config, source=self.source("E1"), statistic="count verified rows"), "passed": self.leaf(len(rows), "cases", rows=rows, config=config, source=self.source("E1"), statistic="count verified rows"), "scope": "仅统计导出集合中 verified=1 的 CSV 行；排除行数见 sources，不代表失败测试已通过。"}
        for key in ("n", "dist", "impl", "variant"):
            values = sorted({row.get(key) for row in rows if row.get(key) is not None}, key=lambda value: (type(value).__name__, value))
            result[key] = self.leaf(values, "observed values", rows=rows, config=config, source=self.source("E1"), statistic="unique observed values")
        result["p"] = self.leaf(sorted({int(row.get("procs", 1)) * int(row.get("threads", 1)) for row in rows}), "units", rows=rows, config=config, source=self.source("E1"), statistic="unique procs*threads")
        return result

    def summary_frame(self) -> pd.DataFrame:
        records = []
        for group in self.groups:
            row = dict(group["config"], **group["metrics"])
            row["median_ms"] = row["total_ms"]
            row["units"] = self.units(group)
            baseline = self.baseline(group)
            row["baseline_ms"] = baseline["metrics"]["total_ms"] if baseline else float("nan")
            row["speedup"] = row["baseline_ms"] / row["median_ms"] if row["median_ms"] > 0 else float("nan")
            row["efficiency"] = row["speedup"] / row["units"]
            row["karp_flatt"] = (1 / row["speedup"] - 1 / row["units"]) / (1 - 1 / row["units"]) if row["units"] > 1 and finite(row["speedup"]) and row["speedup"] > 0 else float("nan")
            records.append(row)
        return pd.DataFrame(records)

    def amdahl(self) -> list[dict]:
        """Use the live summarizer API, not a possibly stale E13_fits.json file."""
        summary = self.summary_frame()
        module = self.summarizer()
        if summary.empty or module is None:
            return []
        try:
            if hasattr(module, "summarize"):
                raw = pd.DataFrame([dict(row, **group["config"]) for group in self.groups if group["config"]["experiment"] != "E1" for row in group["rows"]])
                summary = module.summarize(raw)
            fits = module.fit_models(summary)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            self.load_issues.append("Amdahl 汇总接口不可用: " + str(error))
            return []
        result = []
        for fit in fits:
            if bool(fit.get("quick", self.quick)) != self.quick:
                continue
            ignored = {"quick", "points", "sse", "f", "alpha_ms", "rmse", "point_configs", "data_points", "curve_id", "family", "hybrid_ranks", "bind_policy", "places_policy"}
            if fit.get("data_points"):
                points = []
                valid_fit = True
                for expected in fit["data_points"]:
                    target = expected.get("config", {})
                    matches = [group for group in self.select("E2") if all(key not in group["config"] or str(group["config"][key]) == str(value) for key, value in target.items())]
                    if len(matches) != 1 or not math.isclose(matches[0]["metrics"]["total_ms"], float(expected["time_ms"]), rel_tol=1e-10, abs_tol=1e-10):
                        valid_fit = False
                        break
                    matched_baseline = self.baseline(matches[0])
                    if not matched_baseline or not math.isclose(matched_baseline["metrics"]["total_ms"], float(expected["baseline_ms"]), rel_tol=1e-10, abs_tol=1e-10):
                        valid_fit = False
                        break
                    points.append(matches[0])
                if not valid_fit:
                    self.missing_value("e13_amdahl.points", "拟合点与当前 verified 原始分组不一致", {key: value for key, value in fit.items() if key not in ignored})
                    continue
            else:
                # Compatibility with the first-stage summarizer. The T6 API carries
                # exact point configurations, which are required for transitioned
                # bindings and distinct hybrid rank topologies.
                points = [group for group in self.select("E2") if all(key not in group["config"] or str(group["config"][key]) == str(value) for key, value in fit.items() if key not in ignored)]
            if len({self.units(point) for point in points}) < 3:
                self.missing_value("e13_amdahl.points", "拟合缺少至少三个可验证处理单元数", {key: value for key, value in fit.items() if key not in ignored})
                continue
            raw = [row for point in points for row in point["rows"]]
            raw += [row for point in points for base in [self.baseline(point)] if base for row in base["rows"]]
            config = {key: scalar(value) for key, value in fit.items() if key not in ignored or key in {"hybrid_ranks", "bind_policy", "places_policy"}}
            record = {"config": config, "f": self.leaf(fit.get("f"), "ratio", rows=raw, config=config, source="tables/E13.md", path="e13_amdahl.f", statistic="constrained model fit"), "alpha_ms": self.leaf(fit.get("alpha_ms"), "ms", rows=raw, config=config, source="tables/E13.md", path="e13_amdahl.alpha", statistic="constrained model fit"), "sse": self.leaf(fit.get("sse"), "ms^2", rows=raw, config=config, source="tables/E13.md", path="e13_amdahl.sse", statistic="sum squared fit residuals"), "rmse_ms": self.leaf(math.sqrt(float(fit["sse"]) / len(points)) if finite(fit.get("sse")) else None, "ms", rows=raw, config=config, source="tables/E13.md", path="e13_amdahl.rmse_ms", statistic="root mean squared fit residual", formula="sqrt(sse / number of fitted points)"), "points": []}
            for point in sorted(points, key=self.units):
                entry = self.record(point, "e13_amdahl.points", ["total_ms"], True)
                baseline = self.baseline(point)
                p = self.units(point)
                speedup = baseline["metrics"]["total_ms"] / point["metrics"]["total_ms"] if baseline and point["metrics"]["total_ms"] > 0 else None
                value = (1 / speedup - 1 / p) / (1 - 1 / p) if finite(speedup) and speedup > 0 and p > 1 else None
                entry["karp_flatt"] = self.leaf(value, "ratio", point, rows=point["rows"] + (baseline["rows"] if baseline else []), path="e13_amdahl.karp_flatt", source="tables/summary.csv", formula="(1/speedup - 1/p) / (1 - 1/p)") if p > 1 else {"value": None, "unit": "ratio", "source": "tables/summary.csv", "config": point["config"], "quick": self.quick, "note": "p=1 时未定义"}
                record["points"].append(entry)
            result.append(record)
        return result

    def affinity_fix(self) -> dict:
        before_path = self.results / "archive_pre_bindfix/E2_quick.csv"
        after_path = self.results / "E2_quick.csv"
        before = self.read_csv(before_path, "E2", True, inventory=False) if before_path.exists() else []
        after = self.read_csv(after_path, "E2", True, inventory=False) if after_path.exists() else []
        records = []
        for p in (2, 4, 8):
            left = [row for row in before if row.get("impl") == "mpi" and row.get("variant") == "psrs" and int(row.get("procs", 0)) == p and int(row.get("threads", 0)) == 1]
            right = [row for row in after if row.get("impl") == "mpi" and row.get("variant") == "psrs" and int(row.get("procs", 0)) == p and int(row.get("threads", 0)) == 1]
            by_key = defaultdict(lambda: {"before": [], "after": []})
            for side, rows in (("before", left), ("after", right)):
                for row in rows:
                    key = signature({key: row.get(key) for key in ("host", "n", "dist", "seed", "procs", "threads", "impl", "variant")})
                    by_key[key][side].append(row)
            for key, pair in sorted(by_key.items()):
                config = json.loads(key)
                record = {"config": config}
                for side, rows in pair.items():
                    value = float(pd.Series([float(row["total_ms"]) for row in rows]).median()) if rows else None
                    record[side + "_ms"] = self.leaf(value, "ms", rows=rows, config=config, quick=True, path=f"affinity_fix.p{p}.{side}", source="../" + rows[0]["_file"] if rows else None)
                if pair["before"] and pair["after"]:
                    ratio = record["before_ms"]["value"] / record["after_ms"]["value"] if record["after_ms"]["value"] > 0 else None
                    record["before_over_after"] = self.leaf(ratio, "x", rows=pair["before"] + pair["after"], config=config, quick=True, source="../code/results/E2_quick.csv", path="affinity_fix.ratio", formula="median(before total_ms) / median(after total_ms)")
                records.append(record)
        log_path = self.results / "validation/affinity_before.log"
        masks = []
        if log_path.exists():
            for line in log_path.read_text(errors="replace").splitlines():
                if re.search(r"rank=\d+.*thread=\d+.*cpuset=", line):
                    masks.append(line)
        if not records and not masks:
            return {}
        return {"timings": records, "before_log": {"source": "../code/results/validation/affinity_before.log", "thread_masks": masks}, "note": "独立的快速诊断，quick=true；每条 config 中 n 为实际规模，不作为完整实验结果。"}

    def key_numbers(self) -> dict:
        result: dict[str, Any] = {"schema_version": "1", "mode": "quick" if self.quick else "full", "generated_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(), "sources": self.inventory, "load_issues": self.load_issues}
        baselines = self.select("E2", impl="seq", variant="std", n=1 << 30, dist="uniform")
        result["seq_std_2p30_ms"] = self.metric(baselines[0], "total_ms", "seq_std_2p30_ms") if len(baselines) == 1 else self.missing_value("seq_std_2p30_ms", "缺少 n=2^30、uniform 的唯一串行 std::sort 基线", {"experiment": "E2", "impl": "seq", "variant": "std", "n": 1 << 30, "dist": "uniform"}, "ms")
        values = {
            "best_cpu": self.best_cpu(), "best_by_impl": self.best_by_impl(), "speedup_64_vs_128": self.scaling_64_128(),
            "e4_skew": self.paired("E4", ["normal", "power"], {"impl": "omp", "variant": "local"}, {"impl": "omp", "variant": "psrs", "tiebreak": "1"}, "e4_skew", remove={"variant"}) if self.select("E4") else [],
            "e4_hotkeys_tiebreak": self.paired("E4", ["hotkeys", "few_unique"], {"impl": "omp", "variant": "psrs", "tiebreak": "0"}, {"impl": "omp", "variant": "psrs", "tiebreak": "1"}, "e4_hotkeys_tiebreak", remove={"tiebreak"}) if self.select("E4") else [],
            "e4b_oversample_best": self.oversample(),
            "e4b_tiebreak_vs_p": self.paired("E4b", ["hotkeys", "few_unique"], {"impl": "omp", "variant": "psrs", "tiebreak": "0", "oversample": "1"}, {"impl": "omp", "variant": "psrs", "tiebreak": "1", "oversample": "1"}, "e4b_tiebreak_vs_p", remove={"tiebreak"}) if self.select("E4b") else [],
            "e5_phase_share": self.phase_share(), "e6_best_schedule": self.scheduling(),
            "e7_numa": [self.record(row, "e7_numa", ["total_ms"], False) for row in self.select("E7") if row["config"]["impl"] != "seq"],
            "e8_lock": self.lock_contention(), "e9_comm": self.communication(), "e10_pipeline": self.pipeline(),
            "e11_gpu": self.gpu(), "e12_overview": [self.record(row, "e12_overview", ["total_ms"]) for row in self.select("E12")],
            "e13_amdahl": self.amdahl(), "e14_bandwidth": self.bandwidth_numbers(), "correctness": self.correctness(), "affinity_fix": self.affinity_fix(),
        }
        for key, value in values.items():
            result[key] = self.section(key, value, quick=True if key == "affinity_fix" else None)
        result["diagnostic_sources"] = self.diagnostic_inventory
        result["missing"] = self.missing
        return result

    def compact_tables(self, key_numbers: dict) -> None:
        destination = self.output / "tables"
        destination.mkdir(exist_ok=True)
        for experiment in [f"E{i}" for i in range(2, 15)] + ["E4b"]:
            title = f"# {experiment} {'快速验证数据' if self.quick else '完整实验数据'}\n\n"
            if experiment == "E14":
                rows = [[row.get(key) for key in ("host", "n", "threads", "bind", "places", "kernel", "best_GBps", "median_GBps")] for row in self.bandwidth]
                body = md_table(["主机", "规模 n", "线程数", "绑定", "位置", "内核", "最佳带宽 (GB/s)", "中位带宽 (GB/s)"], rows) if rows else "缺少 verified=1 的 STREAM 数据。\n"
            elif experiment == "E13":
                families = defaultdict(list)
                for fit in key_numbers["e13_amdahl"]["value"] or []:
                    config = fit["config"]
                    families[(config.get("host", "未知"), config.get("seed", "未知"))].append(fit)
                sections = []
                for (host, seed), fits in sorted(families.items()):
                    rows = []
                    for fit in fits:
                        config = fit["config"]
                        label = IMPL_LABELS.get((config.get("impl"), config.get("variant")), f"{config.get('impl', '')}/{config.get('variant', '')}")
                        rows.append([label, config.get("n"), config.get("hybrid_ranks", 0) or "不适用", fit["f"]["value"], fit["alpha_ms"]["value"], fit["sse"]["value"], len(fit["points"])])
                    sections.append(f"## 主机：{host}；种子：{seed}\n\n" + md_table(["实现", "规模 n", "混合 rank 数", "串行比例 f", "通信系数 α (ms)", "平方误差 (ms²)", "数据点数"], rows))
                body = "\n".join(sections) if sections else "缺少可验证的拟合数据。\n"
            else:
                contexts = defaultdict(list)
                for group in self.select(experiment):
                    c, metrics = group["config"], group["metrics"]
                    name = IMPL_LABELS.get((c["impl"], c["variant"]), f"{c['impl']}/{c['variant']}")
                    shown = {
                        "E2": ("mpi_bind", "places"), "E3": ("mpi_bind", "places"),
                        "E4": ("tiebreak",), "E4b": ("oversample", "tiebreak"),
                        "E5": ("mpi_bind", "places"), "E6": ("buckets_per_thread", "schedule"),
                        "E7": ("bind", "mempolicy"), "E9": ("mpi_bind", "scatter"),
                        "E10": ("batch", "mpi_bind"), "E11": ("gpu_affinity", "naive_atomics", "exchange_path"),
                        "E12": ("mpi_bind", "places", "gpu_affinity"),
                    }.get(experiment, ())
                    tuning = ",".join(f"{key}={c[key]}" for key in shown if c.get(key) not in {"", None})
                    baseline = self.baseline(group)
                    speedup = baseline["metrics"]["total_ms"] / metrics["total_ms"] if baseline and metrics["total_ms"] > 0 else None
                    contexts[(c["host"], c["seed"])].append([name, c["n"], f"{c['procs']}×{c['threads']}" if not c["gpus"] else f"{c['gpus']} 卡", DISTS.get(c["dist"], c["dist"]), tuning, metrics["total_ms"], speedup, metrics.get("imbalance")])
                sections = [f"## 主机：{host}；种子：{seed}\n\n" + md_table(["实现", "元素数 n", "进程×线程", "分布", "调优参数", "中位时间 (ms)", "加速比", "失衡度"], rows) for (host, seed), rows in sorted(contexts.items())]
                body = "\n".join(sections) if sections else "缺少所选模式下 verified=1 的数据。\n"
            (destination / f"{experiment}.md").write_text(title + body + "\n仅含可验证实测；显示值保留三位有效数字。完整配置及原始 CSV 行见 key_numbers.json。\n")

    def manifest(self) -> None:
        registry = None
        for name in ("manifest.json", "figures_manifest.json", "registry.json"):
            path = self.figures / name
            if path.exists():
                try:
                    registry = json.loads(path.read_text())
                    break
                except (OSError, ValueError):
                    pass
        if isinstance(registry, dict):
            registry = registry.get("figures", registry.get("items", []))
        rows = []
        if isinstance(registry, list):
            for item in registry:
                if bool(item.get("quick", False)) != self.quick:
                    continue
                paths = item.get("paths", [item.get("path", "")])
                if isinstance(paths, dict):
                    paths = list(paths.values())
                elif isinstance(paths, str):
                    paths = [paths]
                for path in paths:
                    rows.append([path, item.get("title", "未登记图题"), item.get("experiment", ""), item.get("source_table", item.get("source", item.get("sources", ""))), item.get("meaning", item.get("series", item.get("description", "")))])
        elif self.figures.exists():
            base = self.figures / "quick" if self.quick else self.figures
            for path in sorted(base.glob("E*.png")):
                if not self.quick and "_quick" in path.stem:
                    continue
                experiment = path.stem.split("_")[0]
                rows.append([str(path.relative_to(self.figures.parent)), "图题尚未登记", experiment, self.source(experiment), "请重新运行 T8 绘图脚本生成曲线含义注册表"])
        body = md_table(["图文件（相对 report）", "图题", "实验", "数据来源", "曲线或柱组含义"], rows) if rows else "当前模式尚无图表注册记录。\n"
        (self.output / "figures_manifest.md").write_text("# 图表清单\n\n" + body)

    def environment(self) -> None:
        path = self.results / "env.txt"
        text = path.read_text(errors="replace") if path.exists() else ""
        clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)

        def recorded(pattern: str) -> list[str]:
            return list(dict.fromkeys(value.strip().strip('"') for value in re.findall(pattern, clean, re.MULTILINE | re.IGNORECASE) if value.strip()))

        def display(values: list[str]) -> str:
            return "；".join(values) if values else "env.txt 未记录"

        patterns = {
            "CPU 型号": r"^Model name:\s*(.+)$", "逻辑 CPU 数": r"^CPU\(s\):\s*(.+)$",
            "每核线程数": r"^Thread\(s\) per core:\s*(.+)$", "每路物理核数": r"^Core\(s\) per socket:\s*(.+)$",
            "CPU 路数": r"^Socket\(s\):\s*(.+)$", "NUMA 节点数": r"^NUMA node\(s\):\s*(.+)$",
            "内存总量": r"^Mem:\s*(\S+)", "OS 发行版": r"^PRETTY_NAME=(.+)$", "内核": r"^(Linux .+)$",
            "gcc": r"^(g\+\+ \(.+)$", "nvcc": r"^(Cuda compilation tools,.+)$",
            "Python": r"^((?:Python )?3\.\d+\.\d+.*)$", "Git commit": r"^(?:git commit(?: id)?|git_commit|commit)\s*[:=]\s*(.+)$",
            "负载": r"^(?:loadavg|load average)\s*[:=]\s*(.+)$",
        }
        rows = [[title, display(recorded(pattern))] for title, pattern in patterns.items()]
        sockets = recorded(patterns["CPU 路数"])
        cores_per_socket = recorded(patterns["每路物理核数"])
        if len(sockets) == len(cores_per_socket) == 1 and sockets[0].isdigit() and cores_per_socket[0].isdigit():
            count = int(sockets[0]) * int(cores_per_socket[0])
            rows.append(["物理核总数", f"{count}（{cores_per_socket[0]} 核/路 × {sockets[0]} 路）"])
        else:
            rows.append(["物理核总数", "记录存在缺失或多个拓扑；见每路核数与 CPU 路数"])
        mpi = recorded(r"HYDRA build details:\s*\n\s*Version:\s*([^\n]+)")
        mpi += recorded(r"^MPICH Version:\s*(.+)$")
        mpi += recorded(r"^(mpiexec \(OpenRTE\).+)$")
        rows.append(["MPI 实现与版本", display(list(dict.fromkeys(mpi)))])
        numa = defaultdict(list)
        for node, cpus in re.findall(r"^NUMA node(\d+) CPU\(s\):\s*(.+)$", clean, re.MULTILINE):
            if cpus.strip() not in numa[node]:
                numa[node].append(cpus.strip())
        for node, cpus in sorted(numa.items(), key=lambda item: int(item[0])):
            rows.append([f"NUMA 节点 {node} CPU 列表", display(cpus)])

        # nvidia-smi's model/PCI and memory values occupy adjacent table lines.
        # Keep recorded values from the snapshots, rather than querying hardware
        # on the computer that exports the report bundle.
        gpus = defaultdict(lambda: defaultdict(list))
        current_gpu = None
        topology_headers = []
        links = defaultdict(list)
        pci_nodes = defaultdict(list)
        for line in clean.splitlines():
            model = re.match(r"^\|\s*(\d+)\s+(NVIDIA.*?)\s{2,}(?:On|Off|N/A)\s*\|\s*([0-9a-fA-F:.]+)", line)
            if model:
                current_gpu = model.group(1)
                gpus[current_gpu]["model"].append(model.group(2).strip())
                parts = model.group(3).lower().split(":")
                pci = f"{int(parts[0], 16):04x}:{parts[1]}:{parts[2]}" if len(parts) == 3 else model.group(3).lower()
                gpus[current_gpu]["pci"].append(pci)
            memory = re.search(r"\b\d+\s*MiB\s*/\s*(\d+)\s*MiB\b", line)
            if memory and current_gpu is not None:
                gpus[current_gpu]["vram"].append(memory.group(1) + " MiB")
                current_gpu = None
            if "CPU Affinity" in line and "GPU" in line:
                topology_headers = re.findall(r"GPU\d+", line)
            tokens = line.split()
            if topology_headers and "CPU Affinity" not in line and tokens and re.fullmatch(r"GPU\d+", tokens[0]) and len(tokens) >= len(topology_headers) + 3:
                identifier = tokens[0][3:]
                gpus[identifier]["cpus"].append(tokens[len(topology_headers) + 1])
                gpus[identifier]["numa"].append(tokens[len(topology_headers) + 2])
                for index, other in enumerate(topology_headers):
                    if tokens[0] != other:
                        pair = tuple(sorted((tokens[0], other)))
                        links[pair].append(tokens[index + 1])
            pci_node = re.match(r"^([0-9a-fA-F:.]+)\s+numa_node=(-?\d+)\s*$", line)
            if pci_node:
                pci_nodes[pci_node.group(1).lower()].append(pci_node.group(2))
        for gpu, facts in sorted(gpus.items(), key=lambda item: int(item[0])):
            for pci in facts["pci"]:
                facts["numa"].extend(pci_nodes.get(pci, []))
            for field, label in (("model", "型号"), ("vram", "显存"), ("pci", "PCI 地址"), ("numa", "NUMA 节点"), ("cpus", "CPU Affinity")):
                rows.append([f"GPU {gpu} {label}", display(list(dict.fromkeys(facts[field])))])
        if not gpus:
            for label in ("GPU 型号", "GPU 显存", "GPU NUMA 节点"):
                rows.append([label, "env.txt 未记录"])
        if links:
            rows.extend([f"{pair[0]} ↔ {pair[1]} 连接类型", display(list(dict.fromkeys(values)))] for pair, values in sorted(links.items()))
        else:
            rows.append(["GPU 间连接类型", "env.txt 未记录"])
        failures = list(dict.fromkeys(line.strip() for line in clean.splitlines() if "NVIDIA-SMI has failed" in line))
        if failures:
            rows.append(["部分 GPU 快照的采集状态", "；".join(failures)])
        starts = [status["started"] for status in self.statuses if finite(status.get("started"))]
        ends = [status["finished"] for status in self.statuses if finite(status.get("finished"))]
        for title, values, operation in (("实验开始时间", starts, min), ("实验结束时间", ends, max)):
            rows.append([title, datetime.fromtimestamp(operation(values), ZoneInfo("Asia/Tokyo")).isoformat() if values else "状态文件未记录"])
        (self.output / "environment.md").write_text("# 实验环境\n\n来源：../code/results/env.txt；时间范围取所选模式的状态文件。表格合并已保存快照中的可读事实，不同值并列保留；未记录的信息不从导出机器当前环境补写。\n\n" + md_table(["项目", "值"], rows) + "\n原始环境快照：\n\n```text\n" + clean + "\n```\n")

    def methods(self) -> None:
        lines = ["# 实验方法事实\n", "- 仅采用所选模式 CSV 中 verified=1 的行；快速模式和完整模式独立导出。", "- 排序总时间和各阶段分别取同一完整配置的中位数，并保留最小值、最大值及重复数；阶段中位数之和不替代总时间。", "- STREAM 使用 CSV 记录的最佳值和中位数；不由排序时间推算 STREAM 带宽。", "- 归并有效带宽为 8n 字节 / merge_ms；本地排序和划分不是单遍读写，不估算有效带宽。", "- CPU 总时间以稳态墙钟覆盖排序调用，不包括输入生成及验证；每组预热不写入 CSV。", "- MPI 排序 total_ms 在输入生成/可选 Scatter 之后开始，scatter_ms 单独记录；阶段为各 rank 最大值，阶段比例可合计超过 100%。", "- 流水线总时间包含流水线及结果汇集；GPU 总时间采用主机墙钟，设备阶段采用 CUDA 事件，双卡控制阶段包含主机时钟。", "- 加速比只用同主机、seed、n、分布、模式的唯一串行 std::sort 基线；缺少匹配时留空。", "- E9 的 sort_plus_scatter_ms 是两个实测中位数之和，仍不包括 root 输入生成。", "- T5 运行环境的 NUMA 页放置回退已在 ../code/results/validation/T5_residency.log 中诊断：默认 first-touch 请求可能受运行环境限制而回退。该验证日志是方法约束证据，不作为完整 E14 带宽数据来源；正式数值仅来自所选模式的 E14 CSV。\n"]
        for experiment in EXPERIMENTS:
            groups = self.select(experiment)
            lines.append(f"## {experiment}\n")
            if not groups and not (experiment == "E14" and self.bandwidth):
                lines.append("- 没有所选模式的 verified=1 数据。\n")
                continue
            if experiment == "E14":
                lines.append("- STREAM 配置：" + signature([{key: row.get(key) for key in ("threads", "bind", "kernel")} for row in self.bandwidth]) + "\n")
                continue
            rows = [[IMPL_LABELS.get((group["config"]["impl"], group["config"]["variant"]), group["config"]["impl"] + "/" + group["config"]["variant"]), group["config"]["n"], group["config"]["dist"], f"{group['config']['procs']}×{group['config']['threads']}", group["metrics"]["repeats"], ";".join(f"{key}={group['config'][key]}" for key in TUNING if group["config"].get(key) not in {"", None})] for group in groups]
            lines.append(md_table(["实现", "n", "分布", "进程×线程", "CSV 正式重复数", "实际 extra 参数"], rows))
        lines.append("## 跳过配置与读取问题\n")
        skipped = [{"status": status["_file"], "skipped": status.get("skipped", [])} for status in self.statuses if status.get("skipped")]
        lines.append("```json\n" + json.dumps(skipped, ensure_ascii=False, indent=2) + "\n```\n")
        lines.extend("- " + issue for issue in self.load_issues)
        (self.output / "methods.md").write_text("\n".join(lines) + "\n")

    def implementation_notes(self) -> None:
        text = """# 实现事实（供算法章节引用）

- 空块和短块只贡献有效样本，MPI 使用 Gatherv/Allgatherv；未用最大键填充伪样本。
- PSRS 默认 oversample=1；平衡界依赖等长块、样本和取整条件，参数扫描记录实际 oversample。
- GPU 使用每块 1024 槽共享哈希表聚合 512 个输入，再更新全局直方图；散射按活跃桶批量预留位置。
- bitonic 桶排序容量为 8192 个 uint32 元素（32 KiB），更大桶回退 Thrust。
- MPI 交换计数后分配接收缓冲并 first-touch，接收分配包含在 exchange 时间，extra 记录 receive_allocation_timed=1。
- quick 缩小规模并减少重复；任何快速时间均不推算十亿元素的完整时间。
- 双卡 GPU 的设备阶段使用 CUDA 事件，跨卡控制使用主机墙钟；总时间为主机墙钟。
- 主机、种子、模式、规模、分布、调优参数分别统计；无匹配基线时加速比为空。
- 纯 MPI/流水线 rank 按 core 放置，超过物理核数时按 hwthread；OMP_NUM_THREADS=1、OMP_PROC_BIND=false。
- 混合 2 rank 使用 socket；8 rank 使用 NUMA；4 rank 每 rank 使用两个 NUMA 节点。OpenMP close，超出 rank 物理核容量时使用 threads places。
- GPU 默认请求 NUMA 绑定：读取 CUDA PCI 地址对应的 sysfs NUMA/CPU 列表；绑定、pinned 内存分配与 first-touch 在驱动该 GPU 的同一线程上完成，失败状态写入 extra。
- 新增 E4b 的 oversample 扫描及 tie-break/p 扫描；E14 测 Copy/Triad；E2/E3/E9 补混合 NUMA 拓扑和 Scatter 对照。
- T5 的 NUMA 页放置诊断记录在 ../code/results/validation/T5_residency.log；默认 first-touch 可能受运行环境限制而回退，完整数据包不从此验证日志提取带宽数值。
- 公共默认值：seed=42，threads=1，repeat=5，oversample=1，buckets_per_thread=1，schedule=static，tiebreak=true，gpu_affinity=numa。
- GPU 桶数为不小于 ceil(n/4096) 的最小二次幂；GPU block 大小与核函数具体实现一致，单卡路径为 256 线程/块。
- 流水线 block 默认批大小为 max(1, 最小阶段容量/8)，elem 版批大小为 1；显式 --batch 记录在 CSV。
- MPI 计数接口使用 int，超出可表示范围会报错；双卡无可用 peer 访问时经 pinned host 中转，实际路径由 extra 记录。

来源：IMPLEMENTATION.md「对设计细节的必要调整」、公共 Config、GPU 与 MPI 源码、T1–T6 实现；本文件记录实现，不提供性能判断。
"""
        (self.output / "implementation_notes.md").write_text(text)

    def readme(self, key_numbers: dict) -> None:
        rows = [[source["file"], source["rows_total"], source["rows_verified"], source["rows_excluded"], source["sha256"]] for source in self.inventory]
        contents = """# 报告数据包

- `key_numbers.json`：关键数字、原始 CSV 行号、完整配置、统计方法、推导公式和缺失原因；`source` 相对 report，`raw_sources.file` 相对仓库根目录。
- `tables/`：每实验一张精简表，最多八列，三位有效数字；JSON 保留计算精度。
- `figures_manifest.md`：图题、实验、来源及曲线/柱组说明。
- `environment.md`：保存的环境快照及实验时间范围；不采集导出机器的当前配置。
- `methods.md`：实际参数矩阵、重复数、计时口径、绑定和跳过配置。
- `implementation_notes.md`：算法与参数的实现事实。

本包不会改写 report/report.md。缺失数值为 null；无测量时不外推、不补零。默认完整模式；--include-quick 只选择快速模式，二者不混合。affinity_fix 是单独标记 quick=true 的修复诊断，不构成完整性能结论。
"""
        contents += f"\n生成时间：{key_numbers['generated_at']}\n\n所选模式：{key_numbers['mode']}\n\n读取来源：\n\n"
        contents += md_table(["CSV 文件", "总行数", "verified=1 行数", "排除行数", "SHA256"], rows) if rows else "所选模式暂无 CSV。\n"
        if self.diagnostic_inventory:
            contents += "\n独立的修复快速诊断来源（不混入完整性能统计）：\n\n" + md_table(["CSV 文件", "总行数", "verified=1 行数", "排除行数", "SHA256"], [[source["file"], source["rows_total"], source["rows_verified"], source["rows_excluded"], source["sha256"]] for source in self.diagnostic_inventory])
        contents += "\n原始行号从 2 开始，包含 CSV 表头；只对该快照哈希有效。raw_sources 引用的行均为 verified=1。排序配置缺少修复后的 MPI 绑定标记时会排除，并列入读取问题。\n"
        (self.output / "README.md").write_text(contents)

    def run(self) -> dict:
        self.output.mkdir(parents=True, exist_ok=True)
        self.load()
        numbers = self.key_numbers()
        (self.output / "key_numbers.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        self.compact_tables(numbers)
        self.manifest()
        self.environment()
        self.methods()
        self.implementation_notes()
        self.readme(numbers)
        return numbers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-quick", action="store_true", help="只导出快速 CSV；不混入完整 CSV")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--tables-dir", type=Path, default=ROOT.parent / "report/tables")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "report/data_bundle")
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "analysis")
    args = parser.parse_args()
    exporter = Exporter(args.results_dir, args.tables_dir, args.output_dir, args.include_quick, args.figures_dir, args.analysis_dir)
    result = exporter.run()
    print(f"Exported {result['mode']} data to {args.output_dir}; {len(result['sources'])} CSV files, {len(result['missing'])} missing entries")


if __name__ == "__main__":
    main()
