#!/usr/bin/env python3
"""Boundary and adversarial correctness matrix; supports artifact-free planning."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

if '--dry-run' in sys.argv:
    sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from launch import launch_flags
from run_experiments import atomic_json,install_termination_handler,run_command
from bench import estimate_seconds
DISTS=('uniform','normal','power','hotkeys','sorted','reverse','nearly_sorted','few_unique')

def check(exe,variant,n,dist,t=1,ranks=0,options=()):
    command=[str(ROOT/'bin'/exe),'--variant',variant,'--n',str(n),'--dist',dist,'--threads',str(t),'--repeat','1','--csv',str(result),*options]
    child_env=dict(os.environ,OMP_NUM_THREADS=str(t))
    for key in ('PARASORT_LAUNCHER','PARASORT_MPI_BIND'):child_env.pop(key,None)
    if ranks:
        try:flags=launch_flags(exe,ranks,t,child_env)
        except RuntimeError:
            if not args.dry_run:raise
            flags=['<automatic-bind-after-MPI-detection>']
        command=[child_env.get('MPIEXEC','mpiexec'),*flags,'-n',str(ranks),*command]
    configuration={'executable':exe,'variant':variant,'n':n,'dist':dist,'threads':t,'procs':ranks or 1,'repeats':1,'warmups':1,'options':list(options)}
    if args.dry_run:
        print('DRY RUN',shlex.join(command),flush=True)
        print(f'ESTIMATE: {estimate_seconds(n,t*(ranks or 1),1):.3f} s (planning heuristic only)',flush=True)
    else:
        run_command(command,cwd=ROOT,env=child_env,timeout=90,trace_path=ROOT/'results'/'commands.jsonl',metadata={'experiment':'E1','quick':args.quick,'run_id':args.run_id,'configuration':configuration})
    stats['passed']+=1

spec=[('seq_sort','std'),('seq_sort','bucket'),('bucket_pthreads','locked'),('bucket_pthreads','local'),('bucket_omp','local'),('psrs_omp','psrs'),('ref_sorts','gnu_parallel'),('ref_sorts','omp_task_qsort')]

def execute_tests():
    for command in (['make','bin/test_core'],['bin/test_core']):
     if args.dry_run:print('DRY RUN',shlex.join(command),flush=True)
     else:run_command(command,cwd=ROOT,trace_path=ROOT/'results'/'commands.jsonl',metadata={'experiment':'E1','quick':args.quick,'run_id':args.run_id,'role':'core_test'})
    for exe,variant in spec:
     for n in NS:
      for dist in DISTS:
       for t in ([1] if exe=='seq_sort' else PS):check(exe,variant,n,dist,t)
     print('PLAN' if args.dry_run else 'PASS',exe,variant,flush=True)
    # Adversarial duplicate-heavy data exercises all three schedules, oversampling and overdecomposition.
    for schedule in ('static','dynamic','guided'):
     for tie in ([],['--no-tiebreak']):check('psrs_omp','psrs',10003,'few_unique',3,options=['--oversample','4','--buckets-per-thread','4','--schedule',schedule,*tie])
    if not args.cpu_only and (args.dry_run or shutil.which(os.environ.get('MPIEXEC','mpiexec'))):
     for exe in ('psrs_mpi','psrs_hybrid','pipeline_sort'):
      if not args.dry_run and not (ROOT/'bin'/exe).exists():stats['skipped'].append(exe);continue
      for n in ([0,7,1003] if args.quick else NS):
       for dist in (['uniform','few_unique'] if args.quick else DISTS):
        for ranks in ([1,3] if args.quick else PS):check(exe,'block' if exe=='pipeline_sort' else 'psrs',n,dist,2 if exe=='psrs_hybrid' else 1,ranks)
      print('PLAN' if args.dry_run else 'PASS',exe,flush=True)
     if args.dry_run or (ROOT/'bin'/'psrs_mpi').exists():
      for variant,extra in [('allgather',[]),('psrs',['--scatter']),('psrs',['--no-tiebreak'])]:check('psrs_mpi',variant,1003,'hotkeys',1,3,extra)
     if args.dry_run or (ROOT/'bin'/'pipeline_sort').exists():
      for ranks in PS:check('pipeline_sort','elem',ranks,'uniform',1,ranks)
    else:stats['skipped'].append('MPI: disabled or missing runtime')
    try:
     gpu=['planned GPU 0','planned GPU 1'] if args.dry_run else subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True,stderr=subprocess.DEVNULL,timeout=10).strip().splitlines()
    except (OSError,subprocess.SubprocessError):gpu=[]
    if gpu and not args.cpu_only:
     for exe,variants in [('gpu_sort',['thrust','sample']),('gpu_multi',['sample'])]:
      if exe=='gpu_multi' and len(gpu)<2:stats['skipped'].append('gpu_multi: fewer than 2 GPUs');continue
      if not args.dry_run and not (ROOT/'bin'/exe).exists():stats['skipped'].append(exe+': not built');continue
      for variant in variants:
       for n in NS:
        for dist in DISTS:check(exe,variant,n,dist)
      if exe=='gpu_sort':check(exe,'sample',1000003,'few_unique',options=['--naive-atomics'])
      print('PLAN' if args.dry_run else 'PASS',exe,flush=True)
    else:stats['skipped'].append('CUDA runtime: no usable NVIDIA driver/GPU')

def main(argv=None):
    global args,NS,PS,result,stats
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--cpu-only',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--csv',type=Path)
    parser.add_argument('--validation-json',type=Path)
    parser.add_argument('--run-id',default=None)
    args=parser.parse_args(argv)
    os.chdir(ROOT)
    os.environ.setdefault('OMP_PROC_BIND','close');os.environ.setdefault('OMP_PLACES','cores')
    NS=[0,1,2,7,1000] if args.quick else [0,1,2,7,1000,1000003]
    PS=[1,3] if args.quick else [1,2,3,7,8,16]
    result=args.csv or ROOT/'results'/('E1_quick.csv' if args.quick else 'E1.csv')
    validation=args.validation_json or ROOT/'results'/('E1_quick_validation.json' if args.quick else 'E1_validation.json')
    stats={'success':False,'passed':0,'skipped':[],'started':time.time(),'quick':args.quick,'run_id':args.run_id}
    if not args.dry_run:install_termination_handler()
    try:
        execute_tests()
        stats['success']=True
        print('PLAN cases:' if args.dry_run else 'PASS cases:',stats['passed'],'SKIP:',stats['skipped'],flush=True)
    except BaseException as exc:
        stats['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        stats['finished']=time.time()
        if not args.dry_run:atomic_json(validation,stats)
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:sys.exit(130)
    except (InterruptedError,RuntimeError,subprocess.SubprocessError) as exc:
        print(str(exc),file=sys.stderr)
        sys.exit(1)
