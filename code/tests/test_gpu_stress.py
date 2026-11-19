#!/usr/bin/env python3
"""Run the 64 GPU regression cases into a selected validation CSV.

DESIGN.md sections 4.4 and 5.7–5.8 define output verification and GPU variants.
Importing this module does not launch processes or access a CUDA device.
"""

import argparse
import csv
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DISTRIBUTIONS = (
    "uniform",
    "normal",
    "power",
    "hotkeys",
    "sorted",
    "reverse",
    "nearly_sorted",
    "few_unique",
)


def cases():
    for n in (1000003, 16777216):
        for distribution in DISTRIBUTIONS:
            for executable, variant, extra in (
                ("gpu_sort", "thrust", ()),
                ("gpu_sort", "sample", ()),
                ("gpu_sort", "sample", ("--naive-atomics",)),
                ("gpu_multi", "sample", ()),
            ):
                yield n, distribution, executable, variant, extra


def csv_rows(path):
    if not path.exists():
        return []
    with path.open() as source:
        return list(csv.DictReader(source))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "results/GPU_validation.csv")
    args = parser.parse_args(argv)
    destination = args.csv.resolve()
    initial_count = len(csv_rows(destination))
    environment = dict(os.environ, OMP_NUM_THREADS="2", OMP_PROC_BIND="close")
    completed = 0
    previous_n = None
    for n, distribution, executable, variant, extra in cases():
        if previous_n is not None and previous_n != n:
            print("PASS GPU n=", previous_n, flush=True)
        command = [
            str(ROOT / "bin" / executable),
            "--n",
            str(n),
            "--dist",
            distribution,
            "--variant",
            variant,
            "--repeat",
            "1",
            "--csv",
            str(destination),
            *extra,
        ]
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            env=environment,
            timeout=120,
        )
        completed += 1
        previous_n = n
    if previous_n is not None:
        print("PASS GPU n=", previous_n, flush=True)
    recorded = csv_rows(destination)[initial_count:]
    if len(recorded) != completed or any(
        row.get("verified") != "1" for row in recorded
    ):
        raise RuntimeError(
            "GPU regression CSV is incomplete or has unverified measurements"
        )
    if completed != 64:
        raise RuntimeError("GPU regression must cover exactly 64 cases")
    print("TOTAL GPU cases", completed, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
