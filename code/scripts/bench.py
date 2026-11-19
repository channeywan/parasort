#!/usr/bin/env python3
"""E1–E13 experiment plans. Full runs use the design scales; quick runs cap n at 2^18."""
import argparse,csv,json,os,shlex,shutil,subprocess,time
from launch import launch_flags
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DISTS=('uniform','normal','power','hotkeys','sorted','reverse','nearly_sorted','few_unique')
def size(exponent):return min((1<<exponent)//16,1<<18) if args.quick else 1<<exponent
def available_gpu():
 try:
  out=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True,stderr=subprocess.DEVNULL,timeout=10)
  return len(out.strip().splitlines())
 except (OSError,subprocess.SubprocessError):return 0
def run(exe,n,dist='uniform',threads=1,procs=1,variant=None,extra=(),env=None,prefix=()):
 binary=ROOT/'bin'/exe
 if not binary.exists() or ((exe.startswith('psrs_m') or exe in ('psrs_hybrid','pipeline_sort')) and not MPI) or (exe.startswith('gpu') and GPU<(2 if exe=='gpu_multi' else 1)):
  reason=f'{exe}: missing binary, MPI runtime or usable GPU';
  if reason not in plan['skipped']:plan['skipped'].append(reason);print('SKIP',reason,flush=True)
  return
 cmd=[str(binary),'--n',str(n),'--dist',dist,'--threads',str(threads),'--repeat',str((2 if args.quick else 3) if exe=='seq_sort' else R),'--csv',str(result)]
 if variant:cmd+=['--variant',variant]
 cmd+=list(extra)
 child_env=dict(os.environ,**(env or {}))
 for key in ('PARASORT_LAUNCHER','PARASORT_MPI_BIND'):child_env.pop(key,None)
 if exe in ('psrs_mpi','psrs_hybrid','pipeline_sort'):
  flags=launch_flags(exe,procs,threads,child_env)
  cmd=[child_env.get('MPIEXEC','mpiexec'),*flags,'-n',str(procs)]+cmd
 cmd=list(prefix)+cmd
 print('RUN',shlex.join(cmd),'OMP_NUM_THREADS='+child_env.get('OMP_NUM_THREADS',str(threads)),'OMP_PROC_BIND='+child_env.get('OMP_PROC_BIND',''),'OMP_PLACES='+child_env.get('OMP_PLACES',''),flush=True)
 subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,env=child_env,timeout=240 if args.quick else None)
 plan['runs']+=1
 # Preserve the concrete executable, which the fixed CSV schema has no field for.
 with (ROOT/'results'/'commands.jsonl').open('a') as f:f.write(json.dumps({'experiment':args.experiment,'quick':args.quick,'command':cmd,'time':time.time()})+'\n')
def baseline(n,dist='uniform'):
 if (n,dist) not in base_done:run('seq_sort',n,dist,variant='std');base_done.add((n,dist))
def cpu(n,p,dist='uniform'):
 run('bucket_omp',n,dist,p);run('psrs_omp',n,dist,p)
 for variant in ('gnu_parallel','omp_task_qsort'):run('ref_sorts',n,dist,p,variant=variant)
 run('psrs_mpi',n,dist,procs=p);run('psrs_hybrid',n,dist,threads=max(1,p//2),procs=min(p,2))
def main():
 global args,R,PS,result,status,plan,base_done,GPU,MPI
 p=argparse.ArgumentParser();p.add_argument('experiment',choices=[f'E{i}' for i in range(1,14)]);p.add_argument('--quick',action='store_true');args=p.parse_args()
 os.chdir(ROOT);os.environ.setdefault('OMP_PROC_BIND','close');os.environ.setdefault('OMP_PLACES','cores')
 R=2 if args.quick else 5;PS=[1,2,4,8] if args.quick else [1,2,4,8,16,32,64,128]
 subprocess.run(['bash','scripts/env_info.sh'],check=True)
 result=ROOT/'results'/f'{args.experiment}{"_quick" if args.quick else ""}.csv'
 status=ROOT/'results'/f'{args.experiment}{"_quick" if args.quick else ""}_status.json'
 plan={'success':False,'experiment':args.experiment,'quick':args.quick,'started':time.time(),'runs':0,'skipped':[]}
 base_done=set()
 GPU=available_gpu();MPI=bool(shutil.which(os.environ.get('MPIEXEC','mpiexec')))
 try:
  e=args.experiment
  if e=='E1':subprocess.run(['bash','tests/test_correctness.sh']+(['--quick'] if args.quick else []),check=True)
  elif e=='E2':
   n=size(30);baseline(n)
   for p in PS:cpu(n,p)
  elif e=='E3':
   unit=(1<<14) if args.quick else (1<<24)
   for p in PS:
    n=unit*p;baseline(n);run('psrs_omp',n,threads=p);run('psrs_mpi',n,procs=p)
  elif e=='E4':
   n=size(28);p=8 if args.quick else 64
   for dist in DISTS:
    baseline(n,dist);run('bucket_omp',n,dist,p);run('psrs_omp',n,dist,p,extra=['--no-tiebreak']);run('psrs_omp',n,dist,p)
  elif e=='E5':
   n=size(30);baseline(n)
   for p in ([2,4,8] if args.quick else [8,32,128]):run('psrs_omp',n,threads=p);run('psrs_mpi',n,procs=p)
  elif e=='E6':
   n=size(28);p=8 if args.quick else 64
   for dist in ('normal','power'):
    baseline(n,dist)
    for b in (1,4,16):
     for schedule in ('static','dynamic','guided'):run('psrs_omp',n,dist,p,extra=['--buckets-per-thread',str(b),'--schedule',schedule])
  elif e=='E7':
   n=size(30);p=8 if args.quick else 64;baseline(n)
   for bind in ('false','close','spread'):
    for variant in ('psrs','serial_init'):run('psrs_omp',n,threads=p,variant=variant,env={'OMP_PROC_BIND':bind})
   if shutil.which('numactl'):
    probe=subprocess.run(['numactl','--interleave=all','true'],stderr=subprocess.DEVNULL)
    if probe.returncode==0:run('psrs_omp',n,threads=p,prefix=['numactl','--interleave=all'],env={'PARASORT_MEMPOLICY':'interleave'})
    else:plan['skipped'].append('NUMA interleave: set_mempolicy denied')
  elif e=='E8':
   n=size(24);baseline(n)
   for p in (PS if args.quick else PS[:-1]):
    for variant in ('locked','local'):run('bucket_pthreads',n,threads=p,variant=variant)
  elif e=='E9':
   n=size(30);baseline(n)
   for p in PS[1:]:
    for variant in ('psrs','allgather'):run('psrs_mpi',n,procs=p,variant=variant)
   total=8 if args.quick else 128
   for ranks in ([2,4] if args.quick else [2,4,8]):run('psrs_hybrid',n,threads=total//ranks,procs=ranks)
   run('psrs_mpi',n,procs=2,extra=['--scatter'])
  elif e=='E10':
   for p in ([2,4,8] if args.quick else [4,8,16,32,64]):run('pipeline_sort',p,procs=p,variant='elem')
   for exponent in ([20] if args.quick else [20,22,24]):
    n=size(exponent);baseline(n)
    for p in ([2,4] if args.quick else [2,4,8,16,32]):
     for divisor in (32,8,2):run('pipeline_sort',n,procs=p,variant='block',extra=['--batch',str(max(1,n//p//divisor))])
     run('psrs_mpi',n,procs=p)
  elif e=='E11':
   for exponent in ([24,26] if args.quick else [24,26,28,30]):
    n=(1<<(exponent-10)) if args.quick else size(exponent);baseline(n)
    run('gpu_sort',n,variant='thrust');run('gpu_sort',n,variant='sample');run('gpu_sort',n,variant='sample',extra=['--naive-atomics']);run('gpu_multi',n)
   n=(1<<18) if args.quick else (1<<30)
   baseline(n)
   for affinity in ('numa','none'):run('gpu_multi',n,extra=['--gpu-affinity',affinity])
  elif e=='E12':
   n=size(30);baseline(n);cpu(n,PS[-1]);run('gpu_sort',n,variant='thrust');run('gpu_sort',n,variant='sample');run('gpu_multi',n)
  elif e=='E13':
   python=os.environ.get('PYTHON','python3');subprocess.run([python,'analysis/summarize.py'],check=True);subprocess.run([python,'analysis/plot.py'],check=True)
  plan['success']=True
 finally:
  plan['finished']=time.time();status.write_text(json.dumps(plan,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
