#!/usr/bin/env python3
"""Run experiments with mode-specific resume, logs and preflight checks."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = tuple(f'E{i}' for i in range(1, 5)) + ('E4b',) + tuple(f'E{i}' for i in range(5, 13)) + ('E14', 'E13')
ENV_FIELDS = ('OMP_NUM_THREADS', 'OMP_PROC_BIND', 'OMP_PLACES',
              'PARASORT_LAUNCHER', 'PARASORT_MPI_BIND', 'PARASORT_MEMPOLICY')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def trace(path, event):
    if path is None:
        return
    with Path(path).open('a') as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + '\n')
        stream.flush()


def stop_process(process, signum, grace=8):
    """Forward interruption to the command's process group and reap it."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_command(command, *, env=None, cwd=None, stdout=None, timeout=None,
                trace_path=None, metadata=None, cleanup_timeout=8):
    """Trace attempts before launch, including failures and interrupted children.

    Python drivers receive longer cleanup timeouts than their nested commands:
    runner -> bench (32 s), bench -> correctness (20 s), backend (8 s).
    This gives every driver time to kill and reap a stubborn backend before an
    outer driver forcibly ends its process group.
    """
    environment = dict(os.environ) if env is None else env
    record = dict(metadata or {}, command=list(command), command_id=uuid.uuid4().hex,
                  environment={key: environment[key] for key in ENV_FIELDS if key in environment})
    trace(trace_path, dict(record, event='started', time=time.time()))
    process = None
    code = None
    error = None
    try:
        process = subprocess.Popen(command, env=environment, cwd=cwd, stdout=stdout,
                                   stderr=subprocess.STDOUT if stdout is not None else None,
                                   start_new_session=True)
        code = process.wait(timeout=timeout)
        if code:
            raise subprocess.CalledProcessError(code, command)
        return code
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        if process is not None:
            stop_process(process, signal.SIGINT if isinstance(exc, KeyboardInterrupt) else signal.SIGTERM, cleanup_timeout)
            code = process.returncode
        raise
    finally:
        trace(trace_path, dict(record, event='finished', time=time.time(), returncode=code, error=error))


def install_termination_handler():
    def terminate(signum, frame):
        raise InterruptedError(f'terminated by signal {signum}')
    signal.signal(signal.SIGTERM, terminate)


def completed(experiment, quick, results=None):
    results = ROOT / 'results' if results is None else Path(results)
    stem = experiment + ('_quick' if quick else '')
    try:
        state = json.loads((results / f'{stem}_status.json').read_text())
    except (OSError, ValueError):
        return False
    return (state.get('success') is True and state.get('quick') is quick and
            (experiment == 'E13' or (results / f'{stem}.csv').is_file()))


def preflight(ignore_load=False):
    load = float(Path('/proc/loadavg').read_text().split()[0])
    print(f'PRECHECK: one-minute load={load:.2f}', flush=True)
    if load > 2:
        print('WARNING: one-minute load exceeds 2; performance may be affected.', file=sys.stderr, flush=True)
        if ignore_load:
            print('PRECHECK: load warning explicitly overridden by --ignore-load.', flush=True)
        elif not sys.stdin.isatty():
            raise RuntimeError('load is above 2; rerun interactively to confirm or pass --ignore-load')
        elif input('Continue despite current load? [y/N] ').strip().lower() not in ('y', 'yes'):
            raise RuntimeError('experiment run cancelled because of current load')
    try:
        processes = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'],
            text=True, stderr=subprocess.STDOUT, timeout=10).strip()
        if processes:
            print('WARNING: existing GPU compute processes:\n' + processes, file=sys.stderr, flush=True)
        else:
            print('PRECHECK: no existing GPU compute processes.', flush=True)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f'PRECHECK: GPU process query unavailable ({exc}); GPU backends will check availability.', flush=True)


def selection(value, parser):
    if value is None:
        return list(EXPERIMENTS)
    requested = value.split(',')
    unknown = sorted(set(requested) - set(EXPERIMENTS))
    if unknown:
        parser.error('unknown experiment(s): ' + ', '.join(unknown))
    return [name for name in EXPERIMENTS if name in requested]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--only', metavar='E2,E4,...')
    parser.add_argument('--force', action='store_true', help='rerun completed experiments')
    parser.add_argument('--dry-run', action='store_true', help='preview every selected configuration without writes')
    parser.add_argument('--ignore-load', action='store_true', help='explicitly accept a one-minute load above 2')
    args = parser.parse_args(argv)
    chosen = selection(args.only, parser)
    python = os.environ.get('PYTHON', sys.executable)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    build = ['make', '-j' + os.environ.get('BUILD_JOBS', '8'), 'all']
    if args.dry_run:
        print('DRY RUN: preview all selected configurations (including completed/unavailable backends); no files are written.', flush=True)
        print('BUILD (not executed):', shlex.join(build), flush=True)
        for experiment in chosen:
            command = [python, '-B', str(ROOT / 'scripts' / 'bench.py'), experiment, '--dry-run']
            if args.quick:
                command.append('--quick')
            subprocess.run(command, check=True, cwd=ROOT, env=env)
        return 0
    install_termination_handler()
    pending = []
    for experiment in chosen:
        if not args.force and completed(experiment, args.quick):
            print(f'RESUME: skip successful {experiment} ({"quick" if args.quick else "full"}).', flush=True)
        else:
            pending.append(experiment)
    if not pending:
        return 0
    preflight(args.ignore_load)
    logs = ROOT / 'results' / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    env['PARASORT_IGNORE_LOAD'] = str(args.ignore_load).lower()
    subprocess.run(['bash', str(ROOT / 'scripts' / 'env_info.sh'), 'start'], cwd=ROOT, env=env, check=True)
    try:
        subprocess.run(build, cwd=ROOT, env=env, check=True)
        for experiment in pending:
            command = [python, '-B', str(ROOT / 'scripts' / 'bench.py'), experiment]
            if args.quick:
                command.append('--quick')
            log = logs / f'{experiment}.log'
            print(f'RUN {experiment} ({"quick" if args.quick else "full"}) -> {log}', flush=True)
            with log.open('a') as output:
                output.write('\n=== ' + datetime.now(timezone.utc).isoformat() +
                             f' mode={"quick" if args.quick else "full"} command={shlex.join(command)} ===\n')
                output.flush()
                run_command(command, cwd=ROOT, env=env, stdout=output, cleanup_timeout=32)
            print(f'DONE {experiment}', flush=True)
    finally:
        subprocess.run(['bash', str(ROOT / 'scripts' / 'env_info.sh'), 'finish'], cwd=ROOT, env=env, check=False)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted; completed experiments are preserved. Rerun the same command to resume.', file=sys.stderr)
        sys.exit(130)
    except (RuntimeError, InterruptedError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
