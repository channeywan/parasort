#!/usr/bin/env python3
# summarize.py：从真实测量生成统计与报告材料。
# 对应 DESIGN.md 第 7 节。
# 配置、数据来源及缺失状态由统一框架记录。
"""Only verified raw CSV measurements are summarized; quick/full runs never mix."""

import json, math
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT.parent / "report/tables"
TUNING = [
    "seed",
    "oversample",
    "buckets_per_thread",
    "schedule",
    "tiebreak",
    "bind",
    "places",
    "batch",
    "naive_atomics",
    "scatter",
    "mempolicy",
    "mpi_bind",
    "launcher",
    "gpu_affinity",
    "exchange_path",
]
DEFAULTS = {
    "seed": "42",
    "oversample": "1",
    "buckets_per_thread": "1",
    "schedule": "static",
    "tiebreak": "1",
    "batch": "0",
    "naive_atomics": "0",
    "scatter": "0",
    "bind": "unset",
    "places": "unset",
    "mempolicy": "default",
    "mpi_bind": "none",
    "launcher": "none",
    "gpu_affinity": "legacy",
    "exchange_path": "none",
}
GROUP_KEYS = [
    "experiment",
    "quick",
    "host",
    "impl",
    "variant",
    "dist",
    "n",
    "procs",
    "threads",
    "gpus",
] + TUNING
PHASE_FIELDS = [
    "imbalance",
    "compute_ms",
    "scatter_ms",
    "local_sort_ms",
    "sampling_ms",
    "splitters_ms",
    "partition_ms",
    "exchange_ms",
    "merge_ms",
    "h2d_ms",
    "d2h_ms",
]
SCALING_KEYS = [
    "quick",
    "host",
    "impl",
    "variant",
    "dist",
    "scale_n",
    "hybrid_ranks",
    "bind_policy",
    "places_policy",
] + [key for key in TUNING if key not in ("bind", "places", "mpi_bind")]
NUMERIC_TUNING = {
    "seed",
    "oversample",
    "buckets_per_thread",
    "tiebreak",
    "batch",
    "naive_atomics",
    "scatter",
}


def extras(text):
    return dict(part.split("=", 1) for part in str(text).split(";") if "=" in part)


def read_results():
    frames = []
    for path in sorted((ROOT / "results").glob("*.csv")):
        if not path.name.startswith("E") or path.stem.split("_")[0] in (
            "E1",
            "E13",
            "E14",
        ):
            continue
        data = pd.read_csv(path)
        if data.empty:
            continue
        data = data[pd.to_numeric(data.verified, errors="coerce") == 1].copy()
        data["experiment"] = path.stem.split("_")[0]
        data["quick"] = "_quick" in path.stem
        extra = data.extra.map(extras)
        valid = ~data.impl.isin(["mpi", "hybrid", "pipeline"]) | extra.map(
            lambda d: d.get("mpi_bind", "none") != "none"
        )
        data = data[valid].copy()
        extra = extra[valid]
        for key in TUNING:
            data[key] = extra.map(lambda d: d.get(key, DEFAULTS[key]))
        for field in ("compute_ms", "scatter_ms"):
            data[field] = pd.to_numeric(
                extra.map(lambda d: d.get(field, float("nan"))), errors="coerce"
            )
        frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_bandwidth():
    frames = []
    for path in sorted((ROOT / "results").glob("E14*.csv")):
        if path.stem not in ("E14", "E14_quick"):
            continue
        data = pd.read_csv(path)
        if data.empty:
            continue
        data = data[pd.to_numeric(data.verified, errors="coerce") == 1].copy()
        for field in ("threads", "n", "best_GBps", "median_GBps"):
            data[field] = pd.to_numeric(data[field], errors="coerce")
            data = data[data[field].map(math.isfinite)]
        data = data[
            (data.threads > 0)
            & (data.n > 0)
            & (data.best_GBps > 0)
            & (data.median_GBps > 0)
        ]
        data["quick"] = path.stem.endswith("_quick")
        data["source"] = "results/" + path.name
        frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def merge_bandwidth(data):
    """A read and a write of each uint32 key; only PSRS merge is a single pass."""
    values = pd.Series(float("nan"), index=data.index)
    valid = (
        data.impl.isin(("omp", "mpi", "hybrid"))
        & data.variant.isin(("psrs", "allgather", "serial_init"))
        & (data.merge_ms > 0)
        & (data.n > 0)
    )
    values.loc[valid] = 8 * data.loc[valid, "n"] / (data.loc[valid, "merge_ms"] * 1e6)
    return values


def markdown(data):
    def cell(x):
        if pd.isna(x):
            return "—"
        if isinstance(x, float):
            return f"{x:.5g}"
        return str(x).replace("|", "/")

    return (
        "| "
        + " | ".join(data.columns)
        + " |\n| "
        + " | ".join(["---"] * len(data.columns))
        + " |\n"
        + "".join(
            "| " + " | ".join(cell(x) for x in row) + " |\n"
            for row in data.itertuples(index=False, name=None)
        )
    )


def e4b_table(data):
    """Keep the two E4b scans readable without discarding their tuning keys."""
    table = data[
        [
            "dist",
            "n",
            "units",
            "oversample",
            "tiebreak",
            "median_ms",
            "splitters_ms",
            "imbalance",
        ]
    ].copy()
    for column in ("n", "units", "oversample", "tiebreak"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.sort_values(["dist", "n", "units", "oversample", "tiebreak"])
    table["dist"] = table.dist.replace(
        {
            "uniform": "均匀分布",
            "normal": "正态分布",
            "hotkeys": "热点键",
            "few_unique": "少量不同键",
        }
    )
    table["tiebreak"] = table.tiebreak.map({0: "关闭", 1: "开启"}).fillna("未知")
    table.columns = [
        "分布",
        "规模 n",
        "处理单元 p",
        "过采样 k",
        "复合键",
        "总时间 (ms)",
        "分割点时间 (ms)",
        "失衡度",
    ]
    return table


def _tuning_value(value, key):
    """Unify CSV numeric spelling, without changing a configuration's meaning."""
    if pd.isna(value):
        return DEFAULTS.get(key, "unset")
    text = str(value).strip()
    if key in NUMERIC_TUNING:
        try:
            number = float(text)
            if math.isfinite(number) and number.is_integer():
                return str(int(number))
        except ValueError:
            pass
    return text


def scaling_identity(data, weak=False):
    """Attach scaling policies while keeping every actual binding column intact.

    The experiment host has 64 physical cores. Only the expected automatic SMT
    transition is normalized; an explicit user mapping or an unexpected binding
    remains its own context. Hybrid rank counts are always fixed per curve.
    """
    result = data.copy()
    for key in TUNING:
        if key not in result:
            result[key] = DEFAULTS[key]
        # Identity columns are separate from actual metadata, so a caller can still
        # trace each fitted point to exactly what was emitted in the source CSV.
        result["_identity_" + key] = result[key].map(
            lambda value: _tuning_value(value, key)
        )
    if "units" not in result:
        result["units"] = pd.to_numeric(result.procs) * pd.to_numeric(result.threads)
    units = pd.to_numeric(result.units, errors="coerce")
    result["scale_n"] = (
        pd.to_numeric(result.n, errors="coerce") / units if weak else result.n
    )
    result["hybrid_ranks"] = (
        pd.to_numeric(result.procs, errors="coerce")
        .where(result.impl == "hybrid", 0)
        .astype(int)
    )
    bindings = []
    places = []
    for _, row in result.iterrows():
        impl = row.impl
        procs = int(row.procs)
        threads = int(row.threads)
        bind = str(row["bind"]).strip().lower()
        mapping = str(row.mpi_bind).strip().lower()
        place = str(row.places).strip().lower()
        launcher = str(row.launcher).strip().lower()
        expected = "core" if procs <= 64 else "hwthread"
        mpi_auto = (
            impl in ("mpi", "pipeline")
            and threads == 1
            and bind == "false"
            and launcher in ("mpich", "openmpi")
            and mapping == expected
        )
        hybrid_object = {2: "socket", 4: "numa:2", 8: "numa"}.get(procs)
        hybrid_auto = (
            impl == "hybrid"
            and bind == "close"
            and launcher in ("mpich", "openmpi")
            and hybrid_object is not None
            and mapping == hybrid_object
        )
        if mpi_auto:
            binding = "mpi_auto_core_to_hwthread"
        elif hybrid_auto:
            binding = "hybrid_auto_" + hybrid_object
        elif impl in ("mpi", "pipeline", "hybrid"):
            binding = "explicit:mpi_bind=" + mapping + ";omp_bind=" + bind
        else:
            binding = "omp_bind=" + bind
        capacity = 64 // procs if hybrid_auto else 64
        expected_place = "cores" if threads <= capacity else "threads"
        can_transition = (impl in ("omp", "ref") and procs == 1) or hybrid_auto
        if can_transition and place == expected_place:
            place_policy = "auto_cores_to_threads"
        else:
            place_policy = "explicit:" + place
        bindings.append(binding)
        places.append(place_policy)
    result["bind_policy"] = bindings
    result["places_policy"] = places
    return result


def scaling_keys(data=None, weak=False):
    """Return stable columns for grouping a prepared scaling frame.

    ``experiment`` is intentionally left to the caller's selection, so the helper
    can also compare E2/E5 without treating unlike host/seed/tuning as one curve.
    """
    return [("_identity_" + key if key in TUNING else key) for key in SCALING_KEYS]


def scaling_groups(data, weak=False):
    prepared = scaling_identity(data, weak)
    return prepared.groupby(scaling_keys(prepared, weak), dropna=False, sort=True)


def _json_value(value):
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def fit_models(summary):
    if summary.empty:
        return []
    fits = []
    data = summary[summary.experiment == "E2"]
    for _, group in scaling_groups(data):
        if group.iloc[0].impl == "seq":
            continue
        valid = pd.to_numeric(group.baseline_ms, errors="coerce").map(
            lambda value: math.isfinite(value) and value > 0
        )
        valid &= pd.to_numeric(group.median_ms, errors="coerce").map(
            lambda value: math.isfinite(value) and value >= 0
        )
        group = group[valid].sort_values("units")
        points = list(
            group[["units", "median_ms", "baseline_ms"]].itertuples(
                index=False, name=None
            )
        )
        if len(set(p for p, _, _ in points)) < 3:
            continue
        best = None
        # Fit T(p)=Tseq*(f+(1-f)/p)+alpha*(p-1), f in [0,1], alpha>=0.
        for i in range(1001):
            f = i / 1000
            den = sum((p - 1) ** 2 for p, _, _ in points)
            alpha = (
                max(
                    0,
                    sum((p - 1) * (t - b * (f + (1 - f) / p)) for p, t, b in points)
                    / den,
                )
                if den
                else 0
            )
            error = sum(
                (t - (b * (f + (1 - f) / p) + alpha * (p - 1))) ** 2
                for p, t, b in points
            )
            if best is None or error < best[0]:
                best = (error, f, alpha)
        row = group.iloc[0]
        record = {
            key: _json_value(row[key])
            for key in ["quick", "host", "impl", "variant", "dist", "n"]
        }
        for key in TUNING:
            values = group[key].drop_duplicates()
            record[key] = _json_value(values.iloc[0]) if len(values) == 1 else "varies"
        record.update(
            sse=best[0],
            f=best[1],
            alpha_ms=best[2],
            points=len(points),
            hybrid_ranks=int(row.hybrid_ranks),
            bind_policy=row.bind_policy,
            places_policy=row.places_policy,
            data_points=[
                {
                    "p": int(point.units),
                    "time_ms": float(point.median_ms),
                    "baseline_ms": float(point.baseline_ms),
                    "config": {
                        key: _json_value(point[key])
                        for key in GROUP_KEYS
                        if key in point
                    },
                }
                for _, point in group.iterrows()
            ],
        )
        fits.append(record)
    return fits


def summarize(raw):
    """Compute fresh statistics from verified raw rows, without writing artifacts."""
    if raw.empty:
        return pd.DataFrame()
    raw = raw.copy()
    if "verified" in raw:
        raw = raw[pd.to_numeric(raw.verified, errors="coerce") == 1].copy()
    if raw.empty:
        return pd.DataFrame()
    extra = (
        raw.extra.map(extras)
        if "extra" in raw
        else pd.Series([{} for _ in range(len(raw))], index=raw.index)
    )
    for key in TUNING:
        if key not in raw:
            raw[key] = extra.map(lambda value: value.get(key, DEFAULTS[key]))
        raw[key] = raw[key].map(lambda value: _tuning_value(value, key))
    for field in PHASE_FIELDS:
        if field not in raw:
            raw[field] = extra.map(lambda value: value.get(field, float("nan")))
        raw[field] = pd.to_numeric(raw[field], errors="coerce")
    # An MPI row without a binding record predates T1 and cannot enter statistics.
    valid = ~raw.impl.isin(("mpi", "hybrid", "pipeline")) | (raw.mpi_bind != "none")
    raw = raw[valid].copy()
    if raw.empty:
        return pd.DataFrame()
    grouped = raw.groupby(GROUP_KEYS, dropna=False)
    summary = grouped.total_ms.agg(
        median_ms="median", min_ms="min", max_ms="max", repeats="size"
    ).reset_index()
    for field in PHASE_FIELDS:
        summary = summary.merge(
            grouped[field].median().rename(field).reset_index(), on=GROUP_KEYS
        )
    summary["units"] = summary.apply(
        lambda x: int(x.gpus) if x.gpus else int(x.procs * x.threads), axis=1
    )
    summary["merge_GBps"] = merge_bandwidth(summary)
    baseline = summary[(summary.impl == "seq") & (summary.variant == "std")]

    def lookup(row):
        subset = baseline[
            (baseline.host == row.host)
            & (baseline.quick == row.quick)
            & (baseline.seed == row.seed)
            & (baseline.n == row.n)
            & (baseline.dist == row.dist)
        ]
        same = subset[subset.experiment == row.experiment]
        if not same.empty:
            subset = same
        return float(subset.median_ms.median()) if not subset.empty else float("nan")

    summary["baseline_ms"] = summary.apply(lookup, axis=1)
    summary["speedup"] = summary.baseline_ms / summary.median_ms
    summary["efficiency"] = summary.speedup / summary.units
    summary["karp_flatt"] = summary.apply(
        lambda r: (
            (1 / r.speedup - 1 / r.units) / (1 - 1 / r.units)
            if r.units > 1 and pd.notna(r.speedup) and r.speedup > 0
            else float("nan")
        ),
        axis=1,
    )
    return summary


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    raw = read_results()
    bandwidth = read_bandwidth()
    if not bandwidth.empty:
        bandwidth.to_csv(DEST / "bandwidth.csv", index=False)
        for quick, data in bandwidth.groupby("quick"):
            table = data[
                ["host", "n", "threads", "bind", "kernel", "best_GBps", "median_GBps"]
            ]
            (DEST / f'E14{"_quick" if quick else ""}.md').write_text(
                "# E14 "
                + ("快速验证" if quick else "完整实验")
                + "\n\n"
                + markdown(table)
                + "\n\nGB/s 采用十进制字节。Copy=16n 字节，Triad=24n 字节；每配置热身一次、测量五次，逐次验证输出。\n"
            )
    else:
        (DEST / "bandwidth.csv").unlink(missing_ok=True)
    if raw.empty:
        (DEST / "README.md").write_text(
            "TODO(待服务器实测)：尚无 verified=1 的实验 CSV。\n"
        )
        print("No verified experiment data")
        return
    summary = summarize(raw)
    if summary.empty:
        print("No verified bound experiment data")
        return
    summary.to_csv(DEST / "summary.csv", index=False)
    cols = [
        "impl",
        "variant",
        "dist",
        "n",
        "procs",
        "threads",
        "gpus",
        "tiebreak",
        "buckets_per_thread",
        "schedule",
        "median_ms",
        "min_ms",
        "max_ms",
        "speedup",
        "efficiency",
        "imbalance",
    ]
    for (e, quick), data in summary.groupby(["experiment", "quick"]):
        body = (
            "\n".join(
                f"## 主机：{host}\n\n" + markdown(e4b_table(group))
                for host, group in data.groupby("host", dropna=False)
            )
            if e == "E4b"
            else markdown(data[cols])
        )
        (DEST / f'{e}{"_quick" if quick else ""}.md').write_text(
            f'# {e} {"快速验证" if quick else "完整实验"}\n\n'
            + body
            + "\n加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。\n"
        )
    fits = fit_models(summary)
    (DEST / "E13_fits.json").write_text(json.dumps(fits, indent=2, ensure_ascii=False))
    (DEST / "E13.md").write_text(
        "# Amdahl 模型拟合\n\nT(p) = Tseq·[f+(1−f)/p] + α·(p−1)，约束 f∈[0,1]、α≥0。\n\n"
        + (
            markdown(
                pd.DataFrame(fits)[
                    ["quick", "impl", "variant", "n", "f", "alpha_ms", "sse", "points"]
                ]
            )
            if fits
            else "TODO(待服务器实测)：E2 至少需要三个不同处理单元数。"
        )
        + "\n\nKarp–Flatt 指标见 summary.csv；模型参数仅为经验拟合，不能据此断言内在串行比例。\n"
    )
    print(
        f"Wrote {len(summary)} measured configurations and {len(fits)} fits to {DEST}"
    )


if __name__ == "__main__":
    main()
