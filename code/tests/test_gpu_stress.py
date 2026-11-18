import subprocess,os
from pathlib import Path
root=Path(__file__).resolve().parents[1];env=dict(os.environ,OMP_NUM_THREADS='2',OMP_PROC_BIND='close')
count=0
for n in (1000003,16777216):
 for dist in ('uniform','normal','power','hotkeys','sorted','reverse','nearly_sorted','few_unique'):
  for exe,variant in [('gpu_sort','thrust'),('gpu_sort','sample'),('gpu_multi','sample')]:
   subprocess.run([str(root/'bin'/exe),'--n',str(n),'--dist',dist,'--variant',variant,'--repeat','1','--csv',str(root/'results/GPU_validation.csv')],check=True,stdout=subprocess.DEVNULL,env=env,timeout=120);count+=1
   if exe=='gpu_sort' and variant=='sample':
    subprocess.run([str(root/'bin'/exe),'--n',str(n),'--dist',dist,'--variant',variant,'--naive-atomics','--repeat','1','--csv',str(root/'results/GPU_validation.csv')],check=True,stdout=subprocess.DEVNULL,env=env,timeout=120);count+=1
 print('PASS GPU n=',n,flush=True)
print('TOTAL GPU cases',count,flush=True)
