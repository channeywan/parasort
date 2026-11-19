#!/usr/bin/env python3
import argparse,csv,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
sys.path.insert(0,str(ROOT/'scripts'))
from launch import launch_flags
p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');p.add_argument('--cpu-only',action='store_true');args=p.parse_args()
DISTS=('uniform','normal','power','hotkeys','sorted','reverse','nearly_sorted','few_unique')
NS=[0,1,2,7,1000] if args.quick else [0,1,2,7,1000,1000003]
PS=[1,3] if args.quick else [1,2,3,7,8,16]
result=ROOT/'results'/('E1_quick.csv' if args.quick else 'E1.csv');stats={'success':False,'passed':0,'skipped':[],'started':time.time()}
os.environ.setdefault('OMP_PROC_BIND','close');os.environ.setdefault('OMP_PLACES','cores')
def check(exe,variant,n,dist,t=1,ranks=0,options=()):
 cmd=[str(ROOT/'bin'/exe),'--variant',variant,'--n',str(n),'--dist',dist,'--threads',str(t),'--repeat','1','--csv',str(result),*options]
 child_env=dict(os.environ)
 for key in ('PARASORT_LAUNCHER','PARASORT_MPI_BIND'):child_env.pop(key,None)
 if ranks:cmd=[child_env.get('MPIEXEC','mpiexec'),*launch_flags(exe,ranks,t,child_env),'-n',str(ranks),*cmd]
 subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,timeout=90,env=child_env);stats['passed']+=1
spec=[('seq_sort','std'),('seq_sort','bucket'),('bucket_pthreads','locked'),('bucket_pthreads','local'),('bucket_omp','local'),('psrs_omp','psrs'),('ref_sorts','gnu_parallel'),('ref_sorts','omp_task_qsort')]
try:
 subprocess.run(['make','bin/test_core'],check=True,stdout=subprocess.DEVNULL);subprocess.run(['bin/test_core'],check=True)
 for exe,variant in spec:
  for n in NS:
   for dist in DISTS:
    for t in ([1] if exe=='seq_sort' else PS):check(exe,variant,n,dist,t)
  print('PASS',exe,variant,flush=True)
 # Adversarial duplicate-heavy data exercises all three schedules, oversampling and overdecomposition.
 for schedule in ('static','dynamic','guided'):
  for tie in ([],['--no-tiebreak']):check('psrs_omp','psrs',10003,'few_unique',3,options=['--oversample','4','--buckets-per-thread','4','--schedule',schedule,*tie])
 if not args.cpu_only and shutil.which(os.environ.get('MPIEXEC','mpiexec')):
  for exe in ('psrs_mpi','psrs_hybrid','pipeline_sort'):
   if not (ROOT/'bin'/exe).exists():stats['skipped'].append(exe);continue
   for n in ([0,7,1003] if args.quick else NS):
    for dist in (['uniform','few_unique'] if args.quick else DISTS):
     for ranks in ([1,3] if args.quick else PS):check(exe,'block' if exe=='pipeline_sort' else 'psrs',n,dist,2 if exe=='psrs_hybrid' else 1,ranks)
   print('PASS',exe,flush=True)
  for variant,extra in [('allgather',[]),('psrs',['--scatter']),('psrs',['--no-tiebreak'])]:check('psrs_mpi',variant,1003,'hotkeys',1,3,extra)
  for ranks in PS:check('pipeline_sort','elem',ranks,'uniform',1,ranks)
 else:stats['skipped'].append('MPI: disabled or missing runtime')
 try:
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True,stderr=subprocess.DEVNULL,timeout=10).strip().splitlines()
 except (OSError,subprocess.SubprocessError):gpu=[]
 if gpu and not args.cpu_only:
  for exe,variants in [('gpu_sort',['thrust','sample']),('gpu_multi',['sample'])]:
   if exe=='gpu_multi' and len(gpu)<2:stats['skipped'].append('gpu_multi: fewer than 2 GPUs');continue
   if not (ROOT/'bin'/exe).exists():stats['skipped'].append(exe+': not built');continue
   for variant in variants:
    for n in NS:
     for dist in DISTS:check(exe,variant,n,dist)
   if exe=='gpu_sort':check(exe,'sample',1000003,'few_unique',options=['--naive-atomics'])
   print('PASS',exe,flush=True)
 else:stats['skipped'].append('CUDA runtime: no usable NVIDIA driver/GPU')
 stats['success']=True
 print('PASS cases:',stats['passed'],'SKIP:',stats['skipped'],flush=True)
finally:
 stats['finished']=time.time();(ROOT/'results'/('E1_quick_validation.json' if args.quick else 'E1_validation.json')).write_text(json.dumps(stats,indent=2))
