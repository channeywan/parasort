#!/usr/bin/env python3
# bench.py：组织实验参数、运行配置和追溯记录。
# 对应 DESIGN.md 第 6 节。
# 配置、数据来源及缺失状态由统一框架记录。
"""E1–E14 and E4b matrices, with transactional per-experiment CSV output."""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import uuid

# --dry-run also remains free of bytecode artifacts when called directly.
if "--dry-run" in sys.argv:
    sys.dont_write_bytecode = True
from launch import launch_flags
from run_experiments import (
    EXPERIMENTS,
    atomic_json,
    install_termination_handler,
    run_command,
)

ROOT = Path(__file__).resolve().parents[1]
DISTS = (
    "uniform",
    "normal",
    "power",
    "hotkeys",
    "sorted",
    "reverse",
    "nearly_sorted",
    "few_unique",
)


def size(exponent):
    return min((1 << exponent) // 16, 1 << 18) if args.quick else 1 << exponent


def available_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return len(out.strip().splitlines())
    except (OSError, subprocess.SubprocessError):
        return 0


def estimate_seconds(n, parallelism, repeats):
    # Console planning hint only; never a measured value or report input.
    return (
        90.0
        * (n / (1 << 30))
        * (math.log2(max(2, n)) / 30)
        / max(1, parallelism)
        * (repeats + 1)
    )


def run(
    exe,
    n,
    dist="uniform",
    threads=1,
    procs=1,
    variant=None,
    extra=(),
    env=None,
    prefix=(),
):
    dry_run = getattr(args, "dry_run", False)
    binary = ROOT / "bin" / exe
    if not dry_run and (
        not binary.exists()
        or (
            (exe.startswith("psrs_m") or exe in ("psrs_hybrid", "pipeline_sort"))
            and not MPI
        )
        or (exe.startswith("gpu") and GPU < (2 if exe == "gpu_multi" else 1))
    ):
        reason = f"{exe}: missing binary, MPI runtime or usable GPU"
        if reason not in plan["skipped"]:
            plan["skipped"].append(reason)
            print("SKIP", reason, flush=True)
        return
    repeats = (2 if args.quick else 3) if exe == "seq_sort" else R
    command = [
        str(binary),
        "--n",
        str(n),
        "--dist",
        dist,
        "--threads",
        str(threads),
        "--repeat",
        str(repeats),
        "--csv",
        str(result),
    ]
    if variant:
        command += ["--variant", variant]
    command += list(extra)
    child_env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    child_env.update(env or {})
    if threads == 128 and "OMP_PLACES" not in (env or {}):
        child_env["OMP_PLACES"] = "threads"
    for key in ("PARASORT_LAUNCHER", "PARASORT_MPI_BIND"):
        child_env.pop(key, None)
    if exe in ("psrs_mpi", "psrs_hybrid", "pipeline_sort"):
        try:
            flags = launch_flags(exe, procs, threads, child_env)
        except RuntimeError:
            if not dry_run:
                raise
            # A plan can be inspected before MPI is installed; real execution
            # still requires identification and verified automatic bindings.
            flags = ["<automatic-bind-after-MPI-detection>"]
        command = [
            child_env.get("MPIEXEC", "mpiexec"),
            *flags,
            "-n",
            str(procs),
        ] + command
    command = list(prefix) + command
    omp = {
        key: child_env.get(key, "")
        for key in (
            "OMP_NUM_THREADS",
            "OMP_PROC_BIND",
            "OMP_PLACES",
            "PARASORT_LAUNCHER",
            "PARASORT_MPI_BIND",
            "PARASORT_MEMPOLICY",
        )
    }
    print(
        "DRY RUN" if dry_run else "RUN",
        shlex.join(command),
        shlex.join([key + "=" + value for key, value in omp.items() if value]),
        flush=True,
    )
    configuration = {
        "executable": exe,
        "n": n,
        "dist": dist,
        "threads": threads,
        "procs": procs,
        "variant": variant,
        "options": list(extra),
        "repeats": repeats,
        "warmups": 1,
        "environment": omp,
        "command": command,
    }
    plan.setdefault("configurations", []).append(configuration)
    if dry_run:
        estimate = estimate_seconds(n, threads * procs, repeats)
        plan["estimated_seconds"] = plan.get("estimated_seconds", 0) + estimate
        print(f"ESTIMATE: {estimate:.3f} s (planning heuristic only)", flush=True)
        return
    run_command(
        command,
        cwd=ROOT,
        env=child_env,
        timeout=240 if args.quick else None,
        trace_path=ROOT / "results" / "commands.jsonl",
        metadata={
            "experiment": args.experiment,
            "quick": args.quick,
            "run_id": plan.get("run_id"),
            "configuration": configuration,
        },
    )
    plan["runs"] += 1
    if "status" in globals():
        atomic_json(status, plan)


def baseline(n, dist="uniform"):
    if (n, dist) not in base_done:
        run("seq_sort", n, dist, variant="std")
        base_done.add((n, dist))


def stream(n, threads, bind):
    binary = ROOT / "bin" / "stream_bw"
    if not args.dry_run and not binary.exists():
        plan["skipped"].append("stream_bw: binary not built")
        return
    child_env = dict(
        os.environ,
        OMP_NUM_THREADS=str(threads),
        OMP_PROC_BIND=bind,
        OMP_PLACES="threads" if threads == 128 else "cores",
    )
    command = [
        str(binary),
        "--n",
        str(n),
        "--threads",
        str(threads),
        "--repeat",
        "5",
        "--csv",
        str(result),
    ]
    configuration = {
        "executable": "stream_bw",
        "n": n,
        "threads": threads,
        "bind": bind,
        "kernels": ["copy", "triad"],
        "repeats": 5,
        "warmups": 1,
        "environment": {
            key: child_env[key]
            for key in ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES")
        },
        "command": command,
    }
    plan["configurations"].append(configuration)
    print(
        "DRY RUN" if args.dry_run else "RUN",
        shlex.join(command),
        f'OMP_PROC_BIND={bind} OMP_PLACES={child_env["OMP_PLACES"]}',
        flush=True,
    )
    if args.dry_run:
        estimate = estimate_seconds(n, threads, 5)
        plan["estimated_seconds"] = plan.get("estimated_seconds", 0) + estimate
        print(f"ESTIMATE: {estimate:.3f} s (planning heuristic only)", flush=True)
        return
    run_command(
        command,
        cwd=ROOT,
        env=child_env,
        timeout=240 if args.quick else None,
        trace_path=ROOT / "results" / "commands.jsonl",
        metadata={
            "experiment": "E14",
            "quick": args.quick,
            "run_id": plan["run_id"],
            "configuration": configuration,
        },
    )
    plan["runs"] += 1
    atomic_json(status, plan)


def cpu(n, p, dist="uniform"):
    run("bucket_omp", n, dist, p)
    run("psrs_omp", n, dist, p)
    for variant in ("gnu_parallel", "omp_task_qsort"):
        run("ref_sorts", n, dist, p, variant=variant)
    run("psrs_mpi", n, dist, procs=p)
    run("psrs_hybrid", n, dist, threads=max(1, p // 2), procs=min(p, 2))
    if p >= 16:
        run("psrs_hybrid", n, dist, threads=p // 8, procs=8)


def execute_experiment():
    e = args.experiment
    if e == "E1":
        command = [
            os.environ.get("PYTHON", sys.executable),
            "-B",
            "tests/correctness.py",
            "--csv",
            str(result),
            "--validation-json",
            str(validation_result),
            "--run-id",
            plan["run_id"],
        ]
        if args.quick:
            command += ["--quick"]
        if args.dry_run:
            command += ["--dry-run"]
        run_command(
            command,
            cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            trace_path=None if args.dry_run else ROOT / "results" / "commands.jsonl",
            metadata={
                "experiment": "E1",
                "quick": args.quick,
                "run_id": plan["run_id"],
                "role": "correctness_driver",
            },
            cleanup_timeout=20,
        )
    elif e == "E2":
        n = size(30)
        baseline(n)
        for p in PS:
            cpu(n, p)
    elif e == "E3":
        unit = (1 << 14) if args.quick else (1 << 24)
        for p in PS:
            n = unit * p
            baseline(n)
            run("psrs_omp", n, threads=p)
            run("bucket_omp", n, threads=p)
            run("psrs_mpi", n, procs=p)
            if p >= 8:
                run("psrs_hybrid", n, threads=p // 8, procs=8)
            else:
                plan["skipped"].append(
                    f"psrs_hybrid E3: fixed 8 ranks requires p>=8; p={p}"
                )
    elif e == "E4":
        n = size(28)
        p = 8 if args.quick else 64
        for dist in DISTS:
            baseline(n, dist)
            run("bucket_omp", n, dist, p)
            run("psrs_omp", n, dist, p, extra=["--no-tiebreak"])
            run("psrs_omp", n, dist, p)
    elif e == "E4b":
        n = (1 << 18) if args.quick else (1 << 28)
        seen = set()
        plan["matrix_notes"] = [
            "hotkeys、p=64、k=1、开启 tie-break 的相同配置由两组扫描共用，每轮只执行一次。"
        ]

        def measure(dist, p, k, tiebreak):
            identity = (dist, p, k, tiebreak)
            if identity in seen:
                return
            seen.add(identity)
            options = ["--oversample", str(k)]
            if not tiebreak:
                options += ["--no-tiebreak"]
            run(
                "psrs_omp",
                n,
                dist,
                threads=p,
                variant="psrs",
                extra=options,
                env={"OMP_PLACES": "threads" if p == 128 else "cores"},
            )

        for dist in ("uniform", "normal", "hotkeys"):
            for k in (1, 2, 4, 8, 16, 32):
                measure(dist, 64, k, True)
        for dist in ("hotkeys", "few_unique"):
            for p in (8, 16, 32, 64, 128):
                for tiebreak in (True, False):
                    measure(dist, p, 1, tiebreak)
    elif e == "E5":
        n = size(30)
        baseline(n)
        for p in ([2, 4, 8] if args.quick else [8, 32, 128]):
            run("psrs_omp", n, threads=p)
            run("psrs_mpi", n, procs=p)
    elif e == "E6":
        n = size(28)
        p = 8 if args.quick else 64
        for dist in ("normal", "power"):
            baseline(n, dist)
            for b in (1, 4, 16):
                for schedule in ("static", "dynamic", "guided"):
                    run(
                        "psrs_omp",
                        n,
                        dist,
                        p,
                        extra=["--buckets-per-thread", str(b), "--schedule", schedule],
                    )
    elif e == "E7":
        n = size(30)
        p = 8 if args.quick else 64
        baseline(n)
        for bind in ("false", "close", "spread"):
            for variant in ("psrs", "serial_init"):
                run(
                    "psrs_omp",
                    n,
                    threads=p,
                    variant=variant,
                    env={"OMP_PROC_BIND": bind},
                )
        if args.dry_run or shutil.which("numactl"):
            probe = (
                None
                if args.dry_run
                else subprocess.run(
                    ["numactl", "--interleave=all", "true"], stderr=subprocess.DEVNULL
                )
            )
            if args.dry_run or probe.returncode == 0:
                run(
                    "psrs_omp",
                    n,
                    threads=p,
                    prefix=["numactl", "--interleave=all"],
                    env={"PARASORT_MEMPOLICY": "interleave"},
                )
            else:
                plan["skipped"].append("NUMA interleave: set_mempolicy denied")
    elif e == "E8":
        n = size(24)
        baseline(n)
        for p in (PS if args.quick else PS[:-1]):
            for variant in ("locked", "local"):
                run("bucket_pthreads", n, threads=p, variant=variant)
    elif e == "E9":
        n = size(30)
        baseline(n)
        for p in PS[1:]:
            for variant in ("psrs", "allgather"):
                run("psrs_mpi", n, procs=p, variant=variant)
        total = 8 if args.quick else 128
        for ranks in (2, 4, 8):
            run("psrs_hybrid", n, threads=total // ranks, procs=ranks)
        for p in (8, 64):
            if p not in PS[1:]:
                run("psrs_mpi", n, procs=p)
            run("psrs_mpi", n, procs=p, extra=["--scatter"])
    elif e == "E10":
        for p in ([2, 4, 8] if args.quick else [4, 8, 16, 32, 64]):
            run("pipeline_sort", p, procs=p, variant="elem")
        for exponent in ([20] if args.quick else [20, 22, 24]):
            n = size(exponent)
            baseline(n)
            for p in ([2, 4] if args.quick else [2, 4, 8, 16, 32]):
                for divisor in (32, 8, 2):
                    run(
                        "pipeline_sort",
                        n,
                        procs=p,
                        variant="block",
                        extra=["--batch", str(max(1, n // p // divisor))],
                    )
                run("psrs_mpi", n, procs=p)
    elif e == "E11":
        for exponent in ([24, 26] if args.quick else [24, 26, 28, 30]):
            n = (1 << (exponent - 10)) if args.quick else size(exponent)
            baseline(n)
            run("gpu_sort", n, variant="thrust")
            run("gpu_sort", n, variant="sample")
            run("gpu_sort", n, variant="sample", extra=["--naive-atomics"])
            run("gpu_multi", n)
        n = (1 << 18) if args.quick else (1 << 30)
        baseline(n)
        for affinity in ("numa", "none"):
            run("gpu_multi", n, extra=["--gpu-affinity", affinity])
    elif e == "E12":
        n = size(30)
        baseline(n)
        cpu(n, PS[-1])
        run("gpu_sort", n, variant="thrust")
        run("gpu_sort", n, variant="sample")
        run("gpu_multi", n)
    elif e == "E14":
        n = (1 << 20) if args.quick else (1 << 28)
        for bind in ("close", "spread"):
            for p in (1, 2, 4, 8, 16, 32, 64, 128):
                stream(n, p, bind)
    elif e == "E13":
        python = os.environ.get("PYTHON", sys.executable)
        for script in (
            "analysis/summarize.py",
            "analysis/plot.py",
            "analysis/export_for_report.py",
        ):
            command = [python, "-B", script]
            print("DRY RUN" if args.dry_run else "RUN", shlex.join(command), flush=True)
            if not args.dry_run:
                run_command(
                    command,
                    cwd=ROOT,
                    trace_path=ROOT / "results" / "commands.jsonl",
                    metadata={
                        "experiment": "E13",
                        "quick": args.quick,
                        "run_id": plan["run_id"],
                        "role": "analysis",
                    },
                )


def main(argv=None):
    global args, R, PS, result, status, plan, base_done, GPU, MPI, validation_result
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")
    R = 2 if args.quick else 5
    PS = [1, 2, 4, 8] if args.quick else [1, 2, 4, 8, 16, 32, 64, 128]
    stem = args.experiment + ("_quick" if args.quick else "")
    canonical = ROOT / "results" / f"{stem}.csv"
    status = ROOT / "results" / f"{stem}_status.json"
    run_id = uuid.uuid4().hex
    plan = {
        "success": False,
        "experiment": args.experiment,
        "quick": args.quick,
        "run_id": run_id,
        "started": time.time(),
        "runs": 0,
        "skipped": [],
        "configurations": [],
        "csv": str(canonical),
    }
    base_done = set()
    if args.dry_run:
        result = canonical
        validation_result = ROOT / "results" / f"{stem}_validation.json"
        GPU = 2
        MPI = True
        execute_experiment()
        print(
            f"ESTIMATE {args.experiment}: {plan.get('estimated_seconds',0):.3f} s; excludes build/analysis/communication overhead.",
            flush=True,
        )
        return 0
    install_termination_handler()
    pending = ROOT / "results" / ".pending"
    pending.mkdir(parents=True, exist_ok=True)
    result = pending / f"{stem}.{run_id}.csv"
    validation_result = pending / f"{stem}.{run_id}_validation.json"
    plan["partial_csv"] = str(result)
    atomic_json(status, plan)
    try:
        GPU = available_gpu()
        MPI = bool(shutil.which(os.environ.get("MPIEXEC", "mpiexec")))
        execute_experiment()
        if args.experiment == "E1":
            validation = json.loads(validation_result.read_text())
            if not validation.get("success"):
                raise RuntimeError("correctness validation did not succeed")
            plan["runs"] = validation.get("passed", 0)
            plan["skipped"] = validation.get("skipped", [])
        if args.experiment != "E13":
            if not result.is_file():
                raise RuntimeError(
                    "experiment produced no measurement CSV; see skipped configurations"
                )
            os.replace(result, canonical)
        if args.experiment == "E1":
            os.replace(validation_result, ROOT / "results" / f"{stem}_validation.json")
        plan["success"] = True
        plan["partial_csv"] = None
    except BaseException as exc:
        plan["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        plan["finished"] = time.time()
        atomic_json(status, plan)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(
            "Experiment interrupted; partial data retained and success=false. Rerun to restart this experiment.",
            file=sys.stderr,
        )
        sys.exit(130)
    except (InterruptedError, RuntimeError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
