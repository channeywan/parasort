#!/usr/bin/env python3
"""MPI binding policy shared by experiments, regression tests and affinity probes.

``launch_flags`` returns launcher flags and updates the supplied child environment.
Importing this module never starts a process; MPI and topology queries happen lazily.
"""
import argparse
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import shlex
import subprocess
import sys


@dataclass(frozen=True)
class Topology:
    physical_cores: int
    logical_cpus: int
    socket_cores: tuple
    numa_cores: tuple


@lru_cache(maxsize=None)
def detect_mpi(launcher, path=None, library_path=None):
    """Identify the selected executable, never a different mpiexec on PATH."""
    env = dict(os.environ)
    if path is not None:
        env['PATH'] = path
    if library_path is not None:
        env['LD_LIBRARY_PATH'] = library_path
    try:
        version = subprocess.check_output(
            [launcher, '--version'], text=True, stderr=subprocess.STDOUT,
            timeout=10, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f'cannot identify MPI launcher {launcher!r}: {exc}') from exc
    if 'HYDRA' in version.upper():
        return 'mpich'
    if 'OPEN MPI' in version.upper() or 'OPENRTE' in version.upper():
        return 'openmpi'
    raise RuntimeError(f'unsupported MPI launcher {launcher!r}; expected MPICH/Hydra or Open MPI')


@lru_cache(maxsize=1)
def topology():
    """Count physical cores in the caller's allowed CPU set, including SMT siblings."""
    allowed = os.sched_getaffinity(0)
    sockets, numas, all_cores = defaultdict(set), defaultdict(set), set()
    rows = subprocess.check_output(
        ['lscpu', '-p=CPU,CORE,SOCKET,NODE'], text=True, timeout=10)
    logical = 0
    for row in rows.splitlines():
        if not row or row.startswith('#'):
            continue
        cpu, core, socket, node = row.split(',')
        if int(cpu) not in allowed:
            continue
        key = (int(socket), int(core))
        all_cores.add(key)
        sockets[int(socket)].add(key)
        if node and node != '-':
            numas[int(node)].add(key)
        logical += 1
    if not all_cores:
        raise RuntimeError('lscpu did not report any CPU in the allowed cpuset')
    return Topology(len(all_cores), logical,
                    tuple(len(sockets[k]) for k in sorted(sockets)),
                    tuple(len(numas[k]) for k in sorted(numas)))


def _placement(procs, topo):
    """Return Hydra object, physical capacity and logical processing-element mode."""
    if procs == 2 and len(topo.socket_cores) == 2:
        return 'socket', min(topo.socket_cores), False
    if procs == 4 and len(topo.numa_cores) == 8:
        return 'numa:2', min(sum(topo.numa_cores[i:i + 2]) for i in range(0, 8, 2)), False
    if procs == 8 and len(topo.numa_cores) == 8:
        return 'numa', min(topo.numa_cores), False
    # Nonstandard rank counts (including correctness p=3/7) must not wrap
    # socket/NUMA objects onto another rank. Disjoint contiguous core groups
    # use the available cores evenly and may leave a small remainder unused.
    if procs <= topo.physical_cores:
        capacity = topo.physical_cores // procs
        return f'core:{capacity}', capacity, False
    capacity = topo.logical_cpus // procs
    return f'hwthread:{capacity}', capacity, True


def launch_flags(exe, procs, threads, env=None):
    """Generate MPI flags and set explicit OpenMP/CSV metadata in ``env``.

    ``MPIEXEC_FLAGS`` is parsed with shell quoting and, when present (even if
    empty), completely replaces automatic mapping. A warning remains visible
    in the experiment log so such measurements cannot be mistaken for defaults.
    """
    if procs < 1 or threads < 1:
        raise ValueError('MPI ranks and OpenMP threads must be positive')
    env = os.environ if env is None else env
    launcher = env.get('MPIEXEC', 'mpiexec')
    implementation = detect_mpi(launcher, env.get('PATH'), env.get('LD_LIBRARY_PATH'))
    topo = topology()
    hybrid = Path(exe).name == 'psrs_hybrid' or (Path(exe).name == 'affinity_probe' and threads > 1)
    env['OMP_NUM_THREADS'] = str(threads if hybrid else 1)
    env['OMP_PROC_BIND'] = 'close' if hybrid else 'false'
    env['OMP_PLACES'] = 'cores'
    env['PARASORT_LAUNCHER'] = implementation

    if 'MPIEXEC_FLAGS' in env:
        env['PARASORT_MPI_BIND'] = 'user'
        # A user mapping can provide less than the whole-node share. Prefer
        # explicit hardware-thread places when even that share needs SMT.
        if hybrid and threads > max(1, topo.physical_cores // procs):
            env['OMP_PLACES'] = 'threads'
        print('WARNING: MPIEXEC_FLAGS is explicitly set; using user flags instead '
              'of automatic MPI binding. Verify with check_affinity.sh.', file=sys.stderr, flush=True)
        return shlex.split(env['MPIEXEC_FLAGS'])
    if procs > topo.logical_cpus:
        raise ValueError(f'{procs} MPI ranks exceed the {topo.logical_cpus} allowed logical CPUs; '
                         'distinct automatic binding is impossible')

    if not hybrid:
        binding = 'core' if procs <= topo.physical_cores else 'hwthread'
        env['PARASORT_MPI_BIND'] = binding
        if implementation == 'mpich':
            return ['-bind-to', binding, '-map-by', binding]
        return (['--use-hwthread-cpus'] if binding == 'hwthread' else []) + [
            '--bind-to', binding, '--map-by', binding]

    binding, capacity, logical_mode = _placement(procs, topo)
    use_smt = logical_mode or threads > capacity
    if use_smt:
        env['OMP_PLACES'] = 'threads'
    env['PARASORT_MPI_BIND'] = binding
    if implementation == 'mpich':
        return ['-bind-to', binding, '-map-by', binding]

    # Open MPI counts PE in cores by default and in hardware threads with
    # --use-hwthread-cpus. Binding and mapping therefore use the same unit.
    pe = threads
    if binding == 'socket':
        mapping = f'ppr:1:package:PE={pe}'
    elif binding == 'numa':
        mapping = f'ppr:1:numa:PE={pe}'
    elif binding == 'numa:2':
        mapping = f'ppr:2:package:PE={pe}'
    else:
        mapping = f'slot:PE={pe}'
    return (['--use-hwthread-cpus'] if use_smt else []) + [
        '--bind-to', 'hwthread' if use_smt else 'core', '--map-by', mapping]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', required=True, help='binary name or path')
    parser.add_argument('--procs', required=True, type=int)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    env = dict(os.environ)
    flags = launch_flags(args.exe, args.procs, args.threads, env)
    binary = Path(args.exe)
    if not binary.is_file():
        binary = Path(__file__).resolve().parents[1] / 'bin' / args.exe
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    command = [env.get('MPIEXEC', 'mpiexec'), *flags, '-n', str(args.procs), str(binary), *arguments]
    print('LAUNCH', shlex.join(command), 'OMP_NUM_THREADS=' + env['OMP_NUM_THREADS'],
          'OMP_PROC_BIND=' + env['OMP_PROC_BIND'], 'OMP_PLACES=' + env['OMP_PLACES'],
          'launcher=' + env['PARASORT_LAUNCHER'], 'mpi_bind=' + env['PARASORT_MPI_BIND'], flush=True)
    return subprocess.call(command, env=env)


if __name__ == '__main__':
    sys.exit(main())
