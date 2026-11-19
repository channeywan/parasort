#!/usr/bin/env python3
"""Generate E2–E13 PNG/PDF figures without fabricating missing backend measurements."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT.parent/'report/figures';TABLES=ROOT.parent/'report/tables'
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2})
fonts={f.name for f in font_manager.fontManager.ttflist}
for font in ['Noto Sans CJK SC','Noto Serif CJK SC','SimHei']:
 if font in fonts:plt.rcParams['font.family']=font;break
PHASES=['local_sort_ms','sampling_ms','splitters_ms','partition_ms','exchange_ms','merge_ms','h2d_ms','d2h_ms']
def label(row):
 value=f'{row.impl}/{row.variant}'
 if row.impl=='omp' and row.variant=='psrs' and str(row.tiebreak)=='0':value+=' no-tie'
 if row.impl=='gpu' and row.variant=='sample':value+=' naive' if str(row.naive_atomics)=='1' else ' hierarchical'
 if row.impl=='gpu_multi':value+=' '+str(row.get('gpu_affinity','legacy'))
 return value
def save(fig,name):
 fig.tight_layout();fig.savefig(DEST/f'{name}.png',dpi=200);fig.savefig(DEST/f'{name}.pdf');plt.close(fig)
def placeholder(e,quick,text):
 fig,ax=plt.subplots(figsize=(7,3));ax.axis('off');ax.text(.5,.55,f'{e}: measurements unavailable',ha='center',fontsize=14);ax.text(.5,.35,text,ha='center',wrap=True);save(fig,f'{e}{"_quick" if quick else ""}')
def curves(data,field,ax,title):
 for name,g in data.groupby('label'):
  g=g.sort_values('units')
  error=None
  if field=='median_ms':error=[(g.median_ms-g.min_ms).clip(lower=0),(g.max_ms-g.median_ms).clip(lower=0)]
  elif field in ('speedup','efficiency'):
   scale=g.units if field=='efficiency' else 1
   error=[(g[field]-g.baseline_ms/g.max_ms/scale).clip(lower=0),(g.baseline_ms/g.min_ms/scale-g[field]).clip(lower=0)]
  ax.errorbar(g.units,g[field],yerr=error,marker='o',capsize=2,label=name)
 ax.set(xlabel='Processing units',ylabel=field,title=title);ax.legend(fontsize=6)
def pivot_bars(data,index,columns,metric,ax):
 values=data.pivot_table(index=index,columns=columns,values=metric,aggfunc='median')
 error=None
 if metric=='median_ms':
  low=data.pivot_table(index=index,columns=columns,values='min_ms',aggfunc='min').reindex_like(values)
  high=data.pivot_table(index=index,columns=columns,values='max_ms',aggfunc='max').reindex_like(values)
  error={col:[(values[col]-low[col]).clip(lower=0).tolist(),(high[col]-values[col]).clip(lower=0).tolist()] for col in values}
 values.plot.bar(ax=ax,rot=45,yerr=error,capsize=2)
E4B_DISTRIBUTIONS={'uniform':'均匀分布','normal':'正态分布','hotkeys':'热点键','few_unique':'少量不同键'}
E4B_CONTEXT=('host','n','seed','impl','variant','procs','gpus','buckets_per_thread',
             'schedule','bind','batch','naive_atomics','scatter','mempolicy',
             'mpi_bind','launcher','gpu_affinity','exchange_path')

def e4b_subsets(data,quick):
 """Select the specified scans, accepting tuning strings such as ``1.0``.

 The p=64, k=1, hotkeys/tie-on observation belongs to both scans and is reused.
 Other scales and implementations cannot silently enter an E4b curve.
 """
 data=data[data.experiment=='E4b'].copy()
 if 'quick' in data:
  mode=data.quick.map(lambda value:str(value).lower() in ('true','1','1.0'))
  data=data[mode==bool(quick)].copy()
 for column in ('n','units','threads','procs','gpus','oversample','tiebreak'):
  data[column]=pd.to_numeric(data[column],errors='coerce')
 data=data[(data.n==(1<<18 if quick else 1<<28))&(data.impl=='omp')&
           (data.variant=='psrs')&(data.procs==1)&(data.gpus==0)&
           (data.units==data.threads)]
 oversample=data[(data.threads==64)&(data.tiebreak==1)&
                 data.oversample.isin((1,2,4,8,16,32))&
                 data.dist.isin(('uniform','normal','hotkeys'))].copy()
 tiebreak=data[(data.oversample==1)&data.threads.isin((8,16,32,64,128))&
               data.tiebreak.isin((0,1))&data.dist.isin(('hotkeys','few_unique'))].copy()
 return oversample,tiebreak

def e4b_context_groups(data):
 """Keep hosts, seeds and fixed tuning settings on separate measured curves."""
 if data.empty:return []
 keys=[key for key in E4B_CONTEXT if key in data.columns]
 varying=[key for key in keys if data[key].nunique(dropna=False)>1]
 groups=[]
 for _,group in data.groupby(keys,dropna=False,sort=True):
  row=group.iloc[0]
  suffix='，'+', '.join(f'{key}={row[key]}' for key in varying) if varying else ''
  groups.append((suffix,group))
 return groups

def e4b_xaxis(ax,values,label):
 ax.set_xscale('log',base=2)
 ax.set_xticks(values,[str(value) for value in values])
 ax.set_xlabel(label)

def plot_e4b(data,quick):
 oversample,tiebreak=e4b_subsets(data,quick)
 suffix='_quick' if quick else ''
 fig,axes=plt.subplots(2,2,figsize=(13,8))
 for ax,dist in zip(axes.flat,('uniform','normal','hotkeys')):
  right=ax.twinx()
  for index,(context,group) in enumerate(e4b_context_groups(oversample[oversample.dist==dist])):
   group=group.sort_values('oversample')
   color=f'C{index%10}'
   ax.plot(group.oversample,group.imbalance,'o-',color=color,label='失衡度'+context)
   right.plot(group.oversample,group.median_ms,'s--',color=color,label='总时间'+context)
  ax.set(title=E4B_DISTRIBUTIONS[dist],ylabel='失衡度')
  right.set_ylabel('总时间 (ms)')
  right.grid(False)
  e4b_xaxis(ax,[1,2,4,8,16,32],'过采样系数 k')
  handles,labels=ax.get_legend_handles_labels()
  extra_handles,extra_labels=right.get_legend_handles_labels()
  if handles:ax.legend(handles+extra_handles,labels+extra_labels,fontsize=7)
  else:ax.text(.5,.5,'暂无符合参数的实测数据',ha='center',transform=ax.transAxes)
 ax=axes[1,1]
 for index,dist in enumerate(('uniform','normal','hotkeys')):
  for context,group in e4b_context_groups(oversample[oversample.dist==dist]):
   group=group.sort_values('oversample')
   ax.plot(group.oversample,group.splitters_ms,'o-',color=f'C{index}',
           label=E4B_DISTRIBUTIONS[dist]+context)
 ax.set(title='选择分割点的开销',ylabel='分割点时间 (ms)')
 e4b_xaxis(ax,[1,2,4,8,16,32],'过采样系数 k')
 if ax.lines:ax.legend(fontsize=7)
 else:ax.text(.5,.5,'暂无符合参数的实测数据',ha='center',transform=ax.transAxes)
 save(fig,'E4b_oversample'+suffix)

 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for ax,dist in zip(axes,('hotkeys','few_unique')):
  for context,group in e4b_context_groups(tiebreak[tiebreak.dist==dist]):
   for tie,values in group.groupby('tiebreak'):
    values=values.sort_values('threads')
    ax.plot(values.threads,values.imbalance,'o-' if tie else 's--',
            color='C0' if tie else 'C1',
            label=('开启复合键' if tie else '关闭复合键')+context)
  ax.set(title=E4B_DISTRIBUTIONS[dist],ylabel='失衡度')
  e4b_xaxis(ax,[8,16,32,64,128],'处理单元数 p')
  if ax.lines:ax.legend(fontsize=7)
  else:ax.text(.5,.5,'暂无符合参数的实测数据',ha='center',transform=ax.transAxes)
 save(fig,'E4b_tiebreak'+suffix)

def main():
 DEST.mkdir(parents=True,exist_ok=True);path=TABLES/'summary.csv'
 if not path.exists():print('Run summarize.py after collecting CSV results');return
 data=pd.read_csv(path,dtype={'tiebreak':str,'naive_atomics':str,'buckets_per_thread':str,'batch':str,'bind':str});data['label']=data.apply(label,axis=1)
 for quick in sorted(data.quick.unique()):
  all_data=data[data.quick==quick];suffix='_quick' if quick else ''
  plot_e4b(all_data,quick)
  for number in range(2,14):
   e=f'E{number}';d=all_data[all_data.experiment==e].copy()
   if e=='E13':
    d=all_data[(all_data.experiment=='E2')&(all_data.impl!='seq')].copy()
   if d.empty:placeholder(e,quick,'TODO: run this experiment on a supported platform.');continue
   d['tick']=d.label+' p='+d.units.astype(str)
   if e=='E2':
    d=d[d.impl!='seq'];fig,axes=plt.subplots(1,2,figsize=(11,4));curves(d,'speedup',axes[0],'Strong scaling');curves(d,'efficiency',axes[1],'Efficiency');maximum=d.units.max();axes[0].plot([1,maximum],[1,maximum],'k--',label='ideal');axes[0].legend(fontsize=6);save(fig,e+suffix)
   elif e=='E3':
    fig,axes=plt.subplots(1,2,figsize=(11,4));d=d[d.impl!='seq'];curves(d,'median_ms',axes[0],'Weak scaling time (constant keys/unit)');curves(d,'speedup',axes[1],'Measured scaled speedup vs serial at same n');axes[1].plot([1,d.units.max()],[1,d.units.max()],'k--',label='Gustafson f=0');axes[1].legend(fontsize=6);save(fig,e+suffix)
   elif e=='E4':
    fig,axes=plt.subplots(1,2,figsize=(12,4));d=d[d.impl!='seq'];
    for ax,metric,title in zip(axes,['median_ms','imbalance'],['Distribution vs time (ms)','Distribution vs imbalance']):pivot_bars(d,'dist','label',metric,ax);ax.set_title(title);ax.set_ylabel(metric)
    save(fig,e+suffix)
   elif e in ('E5','E9','E11'):
    if e=='E11' and not (d.gpus>0).any():placeholder(e,quick,'CUDA runtime unavailable. CPU baselines exist; GPU values were not invented.');continue
    d=d[d.impl!='seq'];fig,axes=plt.subplots(1,2,figsize=(12,4));d=d.sort_values(['label','n','units']);x=range(len(d));axes[0].bar(x,d.median_ms,yerr=[d.median_ms-d.min_ms,d.max_ms-d.median_ms],capsize=2,label='including all overhead');axes[0].set_xticks(list(x),d.tick+' n='+d.n.astype(str),rotation=80,fontsize=6);axes[0].set_ylabel('Time (ms)');axes[0].set_title(e+' total time')
    if e=='E11':axes[0].scatter(list(x),d.compute_ms,color='red',s=12,label='compute (no H2D/D2H)');axes[0].legend(fontsize=6)
    d.set_index(d.tick+' n='+d.n.astype(str))[PHASES].plot.bar(stacked=True,ax=axes[1],rot=80);axes[1].set_title('Phase maxima (MPI maxima are not additive)');axes[1].set_ylabel('Time (ms)');axes[1].legend(fontsize=5);save(fig,e+suffix)
   elif e=='E6':
    d['tuning']=d.buckets_per_thread.astype(str)+'/'+d.schedule;fig,axes=plt.subplots(1,2,figsize=(11,4));
    for ax,(dist,g) in zip(axes,d[d.impl!='seq'].groupby('dist')):pivot_bars(g,'buckets_per_thread','schedule','median_ms',ax);ax.set_title(dist+' scheduling');ax.set_ylabel('Time (ms)')
    save(fig,e+suffix)
   elif e=='E7':
    d=d[d.impl!='seq'];d['tuning']=d.variant+'/'+d.bind+'/'+d.mempolicy;fig,ax=plt.subplots(figsize=(8,4));ax.bar(d.tuning,d.median_ms,yerr=[d.median_ms-d.min_ms,d.max_ms-d.median_ms],capsize=2);ax.tick_params(axis='x',rotation=40);ax.set(title='NUMA initialization and binding',ylabel='Time (ms)');save(fig,e+suffix)
   elif e=='E8':
    fig,ax=plt.subplots(figsize=(7,4));curves(d[d.impl!='seq'],'median_ms',ax,'Pthreads lock contention');ax.set_yscale('log');save(fig,e+suffix)
   elif e=='E10':
    fig,ax=plt.subplots(figsize=(9,4));d=d[d.impl!='seq'];d['config']=d.label+' n='+d.n.astype(str)+' b='+d.batch.astype(str)
    for name,g in d.groupby('config'):ax.errorbar(g.units,g.median_ms,yerr=[g.median_ms-g.min_ms,g.max_ms-g.median_ms],marker='o',capsize=2,label=name)
    ax.set(xlabel='MPI ranks',ylabel='Time (ms)',title='Pipeline batch size and PSRS');ax.legend(fontsize=5);save(fig,e+suffix)
   elif e=='E12':
    fig,ax=plt.subplots(figsize=(9,4));ax.bar(d.tick,d.median_ms,yerr=[d.median_ms-d.min_ms,d.max_ms-d.median_ms],capsize=2);ax.tick_params(axis='x',rotation=50);ax.set(yscale='log',ylabel='Time (ms)',title='Global comparison (available implementations)');save(fig,e+suffix)
   elif e=='E13':
    fig,axes=plt.subplots(1,2,figsize=(11,4));curves(d,'median_ms',axes[0],'Amdahl + communication overhead fit');curves(d[d.units>1],'karp_flatt',axes[1],'Karp–Flatt experimental serial fraction')
    fits=json.loads((TABLES/'E13_fits.json').read_text()) if (TABLES/'E13_fits.json').exists() else []
    for record in fits:
     if record['quick']!=bool(quick):continue
     g=d[(d.impl==record['impl'])&(d.variant==record['variant'])&(d.n==record['n'])&(d.host==record['host'])].sort_values('units')
     if g.empty:continue
     prediction=g.baseline_ms*(record['f']+(1-record['f'])/g.units)+record['alpha_ms']*(g.units-1)
     axes[0].plot(g.units,prediction,'--',label=f"fit {record['impl']}/{record['variant']} f={record['f']:.3f}")
    axes[0].legend(fontsize=5);save(fig,e+suffix)
 print(f'Figures saved to {DEST}')
if __name__=='__main__':main()
