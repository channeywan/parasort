#!/usr/bin/env python3
"""Only verified raw CSV measurements are summarized; quick/full runs never mix."""
import json,math
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT.parent/'report/tables'
TUNING=['seed','oversample','buckets_per_thread','schedule','tiebreak','bind','batch','naive_atomics','scatter','mempolicy']
def extras(text):
 return dict(part.split('=',1) for part in str(text).split(';') if '=' in part)
def read_results():
 frames=[]
 for path in sorted((ROOT/'results').glob('*.csv')):
  if not path.name.startswith('E') or path.stem.split('_')[0] in ('E1','E13'):continue
  data=pd.read_csv(path)
  if data.empty:continue
  data=data[data.verified==1].copy();data['experiment']=path.stem.split('_')[0];data['quick']='_quick' in path.stem
  extra=data.extra.map(extras)
  for key in TUNING:data[key]=extra.map(lambda d:d.get(key,{'seed':'42','oversample':'1','buckets_per_thread':'1','schedule':'static','tiebreak':'1','batch':'0','naive_atomics':'0','scatter':'0','bind':'unset','mempolicy':'default'}[key]))
  data['compute_ms']=extra.map(lambda d:float(d.get('compute_ms','nan')))
  frames.append(data)
 return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
def markdown(data):
 def cell(x):
  if pd.isna(x):return '—'
  if isinstance(x,float):return f'{x:.5g}'
  return str(x).replace('|','/')
 return '| '+' | '.join(data.columns)+' |\n| '+' | '.join(['---']*len(data.columns))+' |\n'+''.join('| '+' | '.join(cell(x) for x in row)+' |\n' for row in data.itertuples(index=False,name=None))
def fit_models(summary):
 fits=[];data=summary[summary.experiment=='E2']
 identity=['quick','host','impl','variant','dist','n']+TUNING
 for name,group in data.groupby(identity,dropna=False):
  if group.iloc[0].impl=='seq':continue
  group=group.dropna(subset=['baseline_ms']).sort_values('units');points=list(group[['units','median_ms','baseline_ms']].itertuples(index=False,name=None))
  if len(set(p for p,_,_ in points))<3:continue
  best=None
  # Fit T(p)=Tseq*(f+(1-f)/p)+alpha*(p-1), f in [0,1], alpha>=0.
  for i in range(1001):
   f=i/1000;den=sum((p-1)**2 for p,_,_ in points)
   alpha=max(0,sum((p-1)*(t-b*(f+(1-f)/p)) for p,t,b in points)/den) if den else 0
   error=sum((t-(b*(f+(1-f)/p)+alpha*(p-1)))**2 for p,t,b in points)
   if best is None or error<best[0]:best=(error,f,alpha)
  record=dict(zip(identity,name));record.update(sse=best[0],f=best[1],alpha_ms=best[2],points=len(points));fits.append(record)
 return fits
def main():
 DEST.mkdir(parents=True,exist_ok=True);raw=read_results()
 if raw.empty:
  (DEST/'README.md').write_text('TODO(待服务器实测)：尚无 verified=1 的实验 CSV。\n');print('No verified experiment data');return
 keys=['experiment','quick','host','impl','variant','dist','n','procs','threads','gpus']+TUNING
 grouped=raw.groupby(keys,dropna=False)
 summary=grouped.total_ms.agg(median_ms='median',min_ms='min',max_ms='max',repeats='size').reset_index()
 for field in ['imbalance','compute_ms','local_sort_ms','sampling_ms','splitters_ms','partition_ms','exchange_ms','merge_ms','h2d_ms','d2h_ms']:
  summary=summary.merge(grouped[field].median().rename(field).reset_index(),on=keys)
 summary['units']=summary.apply(lambda x:int(x.gpus) if x.gpus else int(x.procs*x.threads),axis=1)
 baseline=summary[(summary.impl=='seq')&(summary.variant=='std')]
 def lookup(row):
  subset=baseline[(baseline.host==row.host)&(baseline.quick==row.quick)&(baseline.seed==row.seed)&(baseline.n==row.n)&(baseline.dist==row.dist)]
  same=subset[subset.experiment==row.experiment]
  if not same.empty:subset=same
  return float(subset.median_ms.median()) if not subset.empty else float('nan')
 summary['baseline_ms']=summary.apply(lookup,axis=1);summary['speedup']=summary.baseline_ms/summary.median_ms;summary['efficiency']=summary.speedup/summary.units
 summary['karp_flatt']=summary.apply(lambda r:(1/r.speedup-1/r.units)/(1-1/r.units) if r.units>1 and pd.notna(r.speedup) and r.speedup>0 else float('nan'),axis=1)
 summary.to_csv(DEST/'summary.csv',index=False)
 cols=['impl','variant','dist','n','procs','threads','gpus','tiebreak','buckets_per_thread','schedule','median_ms','min_ms','max_ms','speedup','efficiency','imbalance']
 for (e,quick),data in summary.groupby(['experiment','quick']):
  (DEST/f'{e}{"_quick" if quick else ""}.md').write_text(f'# {e} {"快速验证" if quick else "完整实验"}\n\n'+markdown(data[cols])+'\n加速比仅使用同一主机、seed、规模、分布的 seq/std 实测中位数。空值表示缺少基线。\n')
 fits=fit_models(summary);(DEST/'E13_fits.json').write_text(json.dumps(fits,indent=2,ensure_ascii=False))
 (DEST/'E13.md').write_text('# Amdahl 模型拟合\n\nT(p) = Tseq·[f+(1−f)/p] + α·(p−1)，约束 f∈[0,1]、α≥0。\n\n'+(markdown(pd.DataFrame(fits)[['quick','impl','variant','n','f','alpha_ms','sse','points']]) if fits else 'TODO(待服务器实测)：E2 至少需要三个不同处理单元数。')+'\n\nKarp–Flatt 指标见 summary.csv；模型参数仅为经验拟合，不能据此断言内在串行比例。\n')
 print(f'Wrote {len(summary)} measured configurations and {len(fits)} fits to {DEST}')
if __name__=='__main__':main()
