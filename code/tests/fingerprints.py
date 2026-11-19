#!/usr/bin/env python3
"""Capture actual sorted-output fingerprints and compare formatting revisions.

DESIGN.md §4 verification is applied after each implementation's timing interval.
The element pipeline retains its defined n=p exception; other cases use 2^20.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from launch import launch_flags


def specifications():
    cases = []

    def add(name, exe, variant, threads=8, ranks=1, options=(), n=1 << 20):
        cases.append(
            dict(
                name=name,
                executable=exe,
                variant=variant,
                threads=threads,
                ranks=ranks,
                options=list(options),
                n=n,
            )
        )

    add("seq_std", "seq_sort", "std", 1)
    add("seq_bucket", "seq_sort", "bucket", 1)
    add("pthreads_locked", "bucket_pthreads", "locked")
    add("pthreads_local", "bucket_pthreads", "local")
    add("omp_bucket", "bucket_omp", "local")
    add("omp_psrs", "psrs_omp", "psrs")
    add("omp_psrs_no_tie", "psrs_omp", "psrs", options=["--no-tiebreak"])
    add("gnu_parallel", "ref_sorts", "gnu_parallel")
    add("omp_task_qsort", "ref_sorts", "omp_task_qsort")
    add("mpi_psrs", "psrs_mpi", "psrs", 1, 8)
    add("mpi_allgather", "psrs_mpi", "allgather", 1, 8)
    add("hybrid_2x4", "psrs_hybrid", "psrs", 4, 2)
    add("hybrid_8x1", "psrs_hybrid", "psrs", 1, 8)
    add("pipeline_block", "pipeline_sort", "block", 1, 4, options=["--batch", "32768"])
    add("pipeline_elem", "pipeline_sort", "elem", 1, 8, n=8)
    add("gpu_thrust", "gpu_sort", "thrust", 1)
    add("gpu_sample_hierarchical", "gpu_sort", "sample", 1)
    add("gpu_sample_naive", "gpu_sort", "sample", 1, options=["--naive-atomics"])
    add("gpu_dual", "gpu_multi", "sample", 2)
    return cases


def capture(destination):
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    measurements = destination.with_suffix(".csv")
    if measurements.exists():
        raise RuntimeError(
            "Snapshot CSV already exists; choose a new snapshot path: "
            + str(measurements)
        )
    try:
        gpu = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            .strip()
            .splitlines()
        )
    except (OSError, subprocess.SubprocessError):
        gpu = []
    snapshot = {
        "seed": 42,
        "distribution": "uniform",
        "default_n": 1 << 20,
        "cases": {},
        "skipped": [],
        "note": "Fingerprints read actual sorted output after timing. pipeline_elem uses n=p by definition.",
    }
    for case in specifications():
        exe = case["executable"]
        if not (ROOT / "bin" / exe).is_file():
            snapshot["skipped"].append(
                {"case": case["name"], "reason": "binary unavailable"}
            )
            continue
        mpi = exe in ("psrs_mpi", "psrs_hybrid", "pipeline_sort")
        if mpi and not shutil.which(os.environ.get("MPIEXEC", "mpiexec")):
            snapshot["skipped"].append(
                {"case": case["name"], "reason": "MPI launcher unavailable"}
            )
            continue
        if exe.startswith("gpu") and len(gpu) < (2 if exe == "gpu_multi" else 1):
            snapshot["skipped"].append(
                {"case": case["name"], "reason": "required GPU count unavailable"}
            )
            continue
        env = dict(
            os.environ,
            PARASORT_FINGERPRINT="1",
            OMP_NUM_THREADS=str(case["threads"]),
            OMP_PROC_BIND="close",
            OMP_PLACES="cores",
        )
        for key in ("PARASORT_MPI_BIND", "PARASORT_LAUNCHER"):
            env.pop(key, None)
        command = [
            str(ROOT / "bin" / exe),
            "--variant",
            case["variant"],
            "--n",
            str(case["n"]),
            "--seed",
            "42",
            "--dist",
            "uniform",
            "--threads",
            str(case["threads"]),
            "--repeat",
            "1",
            "--csv",
            str(measurements),
            *case["options"],
        ]
        if mpi:
            flags = launch_flags(exe, case["ranks"], case["threads"], env)
            command = [
                env.get("MPIEXEC", "mpiexec"),
                *flags,
                "-n",
                str(case["ranks"]),
                *command,
            ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
        if result.returncode:
            raise RuntimeError(case["name"] + " failed: " + result.stderr)
        with measurements.open() as stream:
            row = list(csv.DictReader(stream))[-1]
        assert row["verified"] == "1", case["name"]
        extra = dict(
            part.split("=", 1) for part in row["extra"].split(";") if "=" in part
        )
        fingerprint = {
            key: int(extra[key]) for key in ("output_count", "output_sum", "output_xor")
        }
        assert fingerprint["output_count"] == case["n"], case["name"]
        snapshot["cases"][case["name"]] = {"config": case, "fingerprint": fingerprint}
        print("PASS fingerprint", case["name"], fingerprint, flush=True)
    comparable = [
        case["fingerprint"]
        for case in snapshot["cases"].values()
        if case["config"]["n"] == 1 << 20
    ]
    assert comparable and all(
        value == comparable[0] for value in comparable
    ), "implementations disagree on output multiset"
    destination.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    print(
        "Captured",
        len(snapshot["cases"]),
        "implementations; skipped",
        snapshot["skipped"],
    )


def compare(before, after):
    left = json.loads(before.read_text())
    right = json.loads(after.read_text())
    assert (
        left == right
    ), "Output fingerprints or tested configurations changed across formatting revisions"
    print(
        "PASS: identical actual-output fingerprints for",
        len(left["cases"]),
        "implementations/variants.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    snapshot = commands.add_parser("capture")
    snapshot.add_argument("destination", type=Path)
    comparison = commands.add_parser("compare")
    comparison.add_argument("before", type=Path)
    comparison.add_argument("after", type=Path)
    args = parser.parse_args()
    if args.operation == "capture":
        capture(args.destination)
    else:
        compare(args.before, args.after)


if __name__ == "__main__":
    main()
