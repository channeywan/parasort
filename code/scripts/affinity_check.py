#!/usr/bin/env python3
"""Reproduce pre-fix rank overlap, then verify the centralized launch policy."""
import argparse
import csv
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def topology():
    """Count physical cores in the calling process's actual allowed CPU set."""
    allowed = os.sched_getaffinity(0)
    output = subprocess.check_output(
        ["lscpu", "-p=CPU,CORE,SOCKET,NODE"], text=True)
    cores, sockets, nodes = set(), {}, {}
    for line in output.splitlines():
        if line.startswith("#"):
            continue
        cpu, core, socket, node = (int(value) for value in line.split(","))
        if cpu not in allowed:
            continue
        key = (socket, core)
        cores.add(key)
        sockets.setdefault(socket, set()).add(key)
        if node >= 0:
            nodes.setdefault(node, set()).add(key)
    return cores, sockets, nodes


def command_log(command, log, env, expected, timeout=120):
    keys = ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "PARASORT_LAUNCHER", "PARASORT_MPI_BIND")
    settings = " ".join(f"{key}={env[key]}" for key in keys if key in env)
    text = "ENV " + settings + "\n$ " + shlex.join(command) + "\n"
    print(text, end="", flush=True)
    log.write(text)
    log.flush()
    result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=timeout)
    print(result.stdout, end="", flush=True)
    log.write(result.stdout)
    log.flush()
    marker = "AFFINITY " + expected
    if marker not in result.stdout or ((result.returncode == 0) != (expected == "OK")):
        raise RuntimeError(f"expected {marker}, received exit code {result.returncode}")


def sanity(log):
    """An independent workload guard: rank count must improve real MPI sorting."""
    result_file = ROOT / "results/validation/mpi_sanity.csv"
    # This validation artifact contains only the current, identically sized run.
    result_file.unlink(missing_ok=True)
    medians = {}
    for procs in (2, 8):
        command = [sys.executable, "scripts/launch.py", "--exe", "psrs_mpi",
                   "--procs", str(procs), "--threads", "1", "--",
                   "--n", str(1 << 22), "--repeat", "5", "--csv", str(result_file)]
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=ROOT, check=True, stdout=log,
                       stderr=subprocess.STDOUT, timeout=180)
    with result_file.open() as source:
        rows = list(csv.DictReader(source))
    for procs in (2, 8):
        selected = [row for row in rows if int(row["procs"]) == procs]
        if not selected or any(row["verified"] != "1" for row in selected):
            raise RuntimeError(f"MPI sanity p={procs} has missing/unverified rows")
        medians[procs] = statistics.median(float(row["total_ms"]) for row in selected)
    message = (f"MPI SANITY n={1 << 22}: p=2 {medians[2]:.6g} ms; "
               f"p=8 {medians[8]:.6g} ms")
    print(message, flush=True)
    log.write(message + "\n")
    if medians[8] >= medians[2]:
        raise RuntimeError("MPI SANITY BAD: p=8 must be faster than p=2; investigate before full runs")
    print("MPI SANITY OK", flush=True)
    log.write("MPI SANITY OK\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanity", action="store_true",
                        help="also require MPI n=2^22 p=8 to outperform p=2 (five repeats)")
    parser.add_argument("--no-build", action="store_true", help="use already-built binaries")
    args = parser.parse_args()
    os.chdir(ROOT)
    if not args.no_build:
        targets = ["bin/affinity_probe"] + (["bin/psrs_mpi"] if args.sanity else [])
        subprocess.run(["make", *targets], check=True)
    cores, sockets, nodes = topology()
    if len(cores) < 16 or len(sockets) != 2 or min(map(len, sockets.values())) < 8:
        raise RuntimeError(
            "unsupported validation topology: required p=2/4/8/16 and hybrid 2x8 "
            "need at least 16 allowed physical cores across exactly 2 sockets; "
            f"found {len(cores)} cores, socket core counts "
            f"{[len(value) for value in sockets.values()]}. "
            "Run the probe manually with configurations supported by this allocation.")
    validation = ROOT / "results/validation"
    validation.mkdir(parents=True, exist_ok=True)
    launcher = os.environ.get("MPIEXEC", "mpiexec")
    version = subprocess.check_output([launcher, "--version"], text=True,
                                      stderr=subprocess.STDOUT)
    before_env = dict(os.environ, OMP_NUM_THREADS="1", OMP_PROC_BIND="close", OMP_PLACES="cores")
    # Hydra's original default has no binding flags. Open MPI may bind by default,
    # so explicitly unbind there to reproduce the same original failure mode.
    unbound = [] if "HYDRA" in version.upper() else ["--bind-to", "none", "--map-by", "slot"]
    with (validation / "affinity_before.log").open("w") as log:
        log.write("Pre-fix reproduction: unbound MPI, OMP_PROC_BIND=close, OMP_PLACES=cores, OMP_NUM_THREADS=1\n")
        log.write("MPI implementation: " + version.splitlines()[0] + "\n")
        for procs in (2, 4, 8):
            command_log([launcher, *unbound, "-n", str(procs), "bin/affinity_probe"],
                        log, before_env, "BAD")
    tests = [(procs, 1) for procs in (2, 4, 8, 16)] + [(2, 8)]
    # These extra configurations check the T1 2-NUMA/rank and 1-NUMA/rank rules.
    if len(nodes) == 8 and min(map(len, nodes.values())) >= 2:
        tests += [(4, 4), (8, 2)]
    else:
        print(f"SKIP optional hybrid 4/8-rank checks: found {len(nodes)} allowed NUMA nodes; need 8", flush=True)
    with (validation / "affinity_after.log").open("w") as log:
        from launch import launch_flags
        log.write(f"Allowed physical cores: {len(cores)}; sockets: {len(sockets)}; NUMA nodes: {len(nodes)}\n")
        log.write("MPI implementation: " + version.splitlines()[0] + "\n")
        for procs, threads in tests:
            effective_env = dict(os.environ)
            launch_flags("affinity_probe", procs, threads, effective_env)
            command_log([sys.executable, "scripts/launch.py", "--exe", "affinity_probe",
                         "--procs", str(procs), "--threads", str(threads), "--"],
                        log, effective_env, "OK")
        if args.sanity:
            sanity(log)
    print("All requested affinity checks passed. Logs: results/validation/affinity_{before,after}.log", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"AFFINITY CHECK FAILED: {error}", file=sys.stderr)
        sys.exit(1)
