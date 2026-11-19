#!/usr/bin/env python3
"""Render measured experiments with Chinese labels and traceable figure metadata."""
import json
import math
from pathlib import Path
import sys
import textwrap

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import TUNING, scaling_groups, scaling_identity, scaling_keys
from plot_support import (DISTRIBUTION_LABELS, FIELD_LABELS, PHASE_LABELS,
                          TUNING_LABELS, FigureRegistry, apply_font,
                          configure_chinese_font, label, migrate_quick)

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT.parent / 'report' / 'figures'
TABLES = ROOT.parent / 'report' / 'tables'
REGISTRY = None
P_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
PHASES = ['local_sort_ms', 'sampling_ms', 'splitters_ms', 'partition_ms',
          'exchange_ms', 'merge_ms', 'h2d_ms', 'd2h_ms']
TITLES = {
    'E2': '强扩展：加速比与效率', 'E3': '弱扩展：总时间与加速比',
    'E4': '输入分布与负载均衡', 'E4b': '过采样与重复键对照',
    'E5': '排序阶段时间', 'E6': '桶数与任务调度',
    'E7': 'NUMA 初始化与线程绑定', 'E8': 'Pthreads 锁竞争',
    'E9': 'MPI 通信与混合配置', 'E10': '流水线批大小与 PSRS 对照',
    'E11': 'GPU 排序与数据传输', 'E12': '各实现总时间对照',
    'E13': 'Amdahl 拟合与 Karp–Flatt 指标',
    'E14': 'STREAM 带宽与 PSRS 归并有效带宽',
}
CONTEXT_LABELS = {
    'host': '主机', 'seed': '种子', 'dist': '分布', 'scale_n': '规模 n',
    'oversample': '过采样 k', 'buckets_per_thread': '每线程桶数',
    'schedule': '调度', 'batch': '批大小', 'scatter': '输入生成',
    'mempolicy': '内存策略', 'launcher': 'MPI 启动器',
    'bind_policy': '绑定策略', 'places_policy': '线程位置',
}
MPI_CAPTION = '各阶段取各 rank 最大值，堆叠高度可能大于总时间。'
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .2,
                     'axes.titlepad': 11})
configure_chinese_font()


def number(value):
    try:
        value = float(value)
        return str(int(value)) if value.is_integer() else f'{value:.3g}'
    except (TypeError, ValueError):
        return str(value)


def translated(key, value):
    value = str(value)
    if key == 'dist':
        return DISTRIBUTION_LABELS.get(value, value)
    if key == 'scatter':
        return '根进程生成后分发' if value in ('1', '1.0') else '各进程本地生成'
    if key == 'bind_policy':
        if value.startswith('explicit:mpi_bind='):
            mapping,bind=value.removeprefix('explicit:mpi_bind=').split(';omp_bind=',1)
            return 'MPI '+TUNING_LABELS['mpi_bind'].get(mapping,mapping)+' / OpenMP '+TUNING_LABELS['bind'].get(bind,bind)
        return {'mpi_auto_core_to_hwthread': '自动物理核/硬件线程',
                'hybrid_auto_socket': '每插槽一进程',
                'hybrid_auto_numa': '每 NUMA 一进程',
                'hybrid_auto_numa:2': '每两 NUMA 一进程',
                'omp_bind=close': '近邻绑定', 'omp_bind=spread': '分散绑定',
                'omp_bind=false': '不绑定'}.get(value, value.replace('explicit:', '显式：'))
    if key == 'places_policy':
        return '自动物理核/硬件线程' if value == 'auto_cores_to_threads' else '显式：'+TUNING_LABELS.get('places',{}).get(value.removeprefix('explicit:'),value.removeprefix('explicit:'))
    return TUNING_LABELS.get(key, {}).get(value, value)


def wrap(value, width=24):
    return '\n'.join(textwrap.wrap(str(value), width=width, break_long_words=False))


def source_table(experiment, quick):
    candidate = TABLES / f'{experiment}{"_quick" if quick else ""}.md'
    return 'tables/' + candidate.name if candidate.is_file() else 'tables/summary.csv'


def describe(fig, experiment, quick, title=None, series=None, source=None, caption=None):
    title = title or TITLES[experiment]
    fig._parasort = dict(title=title, experiment=experiment, quick=bool(quick),
                        source_table=source or source_table(experiment, quick),
                        series=list(series or []), caption=caption)
    mode = '快速验证' if quick else '完整实验'
    fig.suptitle(f'{experiment}　{title}（{mode}）', fontsize=12)
    return fig


def save(fig, name):
    """One output path policy for every chart, including legacy public callers."""
    quick = name.endswith('_quick')
    metadata = getattr(fig, '_parasort', None)
    if metadata is None:
        experiment = name.split('_')[0]
        describe(fig, experiment, quick)
        metadata = fig._parasort
    target = DEST / 'quick' if quick else DEST
    target.mkdir(parents=True, exist_ok=True)
    handles, labels = [], []
    for ax in fig.axes:
        for handle, title in zip(*ax.get_legend_handles_labels()):
            if title and not title.startswith('_') and title not in labels:
                handles.append(handle)
                labels.append(title)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    columns = 2 if any(len(title) > 30 for title in labels) else 3
    rows = math.ceil(len(labels) / columns) if labels else 0
    height = fig.get_size_inches()[1]
    legend_inches = rows * .34 + (.3 if metadata.get('caption') else .08)
    if legend_inches > height * .32:
        fig.set_size_inches(fig.get_size_inches()[0], height + legend_inches * .65)
        height = fig.get_size_inches()[1]
    bottom = min(.48, (legend_inches + .18) / height) if rows or metadata.get('caption') else .05
    if handles:
        fig.legend(handles, [wrap(title, 35) for title in labels], loc='lower center',
                   bbox_to_anchor=(.5, .015), ncol=columns, fontsize=7,
                   frameon=False, columnspacing=1.8)
    if metadata.get('caption'):
        fig.text(.5, bottom - .025, metadata['caption'], ha='center', va='top', fontsize=8)
    apply_font(fig)
    fig.tight_layout(rect=(0, bottom, 1, .94))
    paths = []
    for extension in ('png', 'pdf'):
        path = target / f'{name}.{extension}'
        fig.savefig(path, dpi=200, bbox_inches='tight')
        paths.append(path)
    if REGISTRY is not None:
        meanings=list(metadata['series'] or labels)
        for ax in fig.axes:
            if not ax.patches:continue
            for tick in list(ax.get_xticklabels())+list(ax.get_yticklabels()):
                text=tick.get_text().replace('\n',' ')
                if any('\u4e00'<=char<='\u9fff' for char in text):
                    meaning='柱配置：'+text
                    if meaning not in meanings:meanings.append(meaning)
        REGISTRY.add(name=name, title=metadata['title'], experiment=metadata['experiment'],
                     quick=metadata['quick'], source_table=metadata['source_table'],
                     series=meanings or ['暂无本模式的已验证实测曲线。'], paths=paths)
    plt.close(fig)


def placeholder(experiment, quick, text='暂无本模式的已验证实测数据。'):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    ax.text(.5, .5, text, ha='center', va='center', transform=ax.transAxes)
    describe(fig, experiment, quick, series=['暂无实测曲线；不根据快速数据推算完整结果。'])
    save(fig, experiment + ('_quick' if quick else ''))


def p_axis(ax, maximum=128):
    ax.set_xscale('log', base=2)
    ax.set_xticks(P_VALUES, [str(value) for value in P_VALUES])
    ax.set_xlabel('处理单元数 p')
    ax.set_xlim(.85,max(1.2,float(maximum)*1.1))


def context_name(row, prepared, weak=False, exclude=()):
    if 'scale_n' not in row or 'bind_policy' not in row:
        row=scaling_identity(pd.DataFrame([row]),weak=weak).iloc[0]
    base = label(row)
    details = []
    family=prepared[(prepared.impl==row.impl)&(prepared.variant==row.variant)] if not prepared.empty else prepared
    if row.impl=='hybrid' and 'hybrid_ranks' in family:
        family=family[family.hybrid_ranks==int(row.procs)]
    fields = ['host', 'seed', 'dist', 'scale_n', 'oversample', 'buckets_per_thread',
              'schedule', 'batch', 'scatter', 'mempolicy', 'launcher',
              'bind_policy', 'places_policy']
    for field in fields:
        column = '_identity_' + field if '_identity_' + field in prepared else field
        if field in exclude or column not in family or family[column].nunique(dropna=False) <= 1:
            continue
        value = row.get(column, row.get(field))
        key = '每单元规模' if weak and field == 'scale_n' else CONTEXT_LABELS.get(field, field)
        details.append(key + '=' + translated(field, value))
    return base + ('，' + '，'.join(details) if details else '')


def curves(data, field, ax, title, weak=False):
    """Connect only matching hosts, seeds, fixed tuning and hybrid topologies."""
    if data.empty:
        ax.text(.5, .5, '暂无符合参数的实测数据', ha='center', transform=ax.transAxes)
    else:
        prepared = scaling_identity(data, weak=weak)
        for _, group in scaling_groups(data, weak=weak):
            group = group.sort_values('units')
            group = group[pd.to_numeric(group[field], errors='coerce').notna()]
            if group.empty:
                continue
            error = None
            if field == 'median_ms':
                error = [(group.median_ms-group.min_ms).clip(lower=0),
                         (group.max_ms-group.median_ms).clip(lower=0)]
            elif field in ('speedup', 'efficiency'):
                scale = group.units if field == 'efficiency' else 1
                error = [(group[field]-group.baseline_ms/group.max_ms/scale).clip(lower=0),
                         (group.baseline_ms/group.min_ms/scale-group[field]).clip(lower=0)]
            ax.errorbar(group.units, group[field], yerr=error, marker='o', capsize=2,
                        label=context_name(group.iloc[0], prepared, weak))
    maximum=float(data.units.max()) if not data.empty else 128
    ideal=[value for value in P_VALUES if value<=maximum]
    if field == 'speedup':
        ax.plot(ideal, ideal, 'k--', linewidth=1, label='理想加速比 y=p')
    elif field == 'efficiency':
        ax.axhline(1, color='black', linestyle='--', linewidth=1, label='理想效率 1')
    p_axis(ax,maximum)
    ax.set(title=title, ylabel=FIELD_LABELS.get(field, field))


def measured_bars(data, ax, title, field='median_ms', horizontal=False):
    """Each bar is one summary row; never take a median across hidden contexts."""
    data = data.sort_values(['impl', 'variant', 'n', 'procs', 'threads'])
    names = [wrap(label(row) + f'，p={number(row.units)}，n={number(row.n)}', 28)
             for _, row in data.iterrows()]
    error = None
    if field == 'median_ms':
        error = [(data.median_ms-data.min_ms).clip(lower=0),
                 (data.max_ms-data.median_ms).clip(lower=0)]
    prepared=scaling_identity(data)
    names=[wrap(context_name(row,prepared,exclude=('scale_n',))+f'，p={number(row.units)}，n={number(row.n)}',28) for _,row in data.iterrows()]
    positions = list(range(len(data)))
    if horizontal:
        ax.barh(positions, data[field], xerr=error, capsize=2)
        ax.set_yticks(positions, names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(FIELD_LABELS.get(field, field))
    else:
        ax.bar(positions, data[field], yerr=error, capsize=2)
        ax.set_xticks(positions, names, rotation=65, ha='right', fontsize=7)
        ax.set_ylabel(FIELD_LABELS.get(field, field))
    ax.set_title(title)
    return data


def phase_bars(data, ax, horizontal=False):
    prepared=scaling_identity(data)
    names = [context_name(row,prepared,exclude=('scale_n',) if horizontal else ())+f'，p={number(row.units)}'+(f'，n={number(row.n)}' if horizontal else '') for _, row in data.iterrows()]
    positions = list(range(len(data)))
    bottom = pd.Series(0., index=data.index)
    for field in PHASES:
        values = data[field].fillna(0)
        if not (values > 0).any():
            continue
        if horizontal:ax.barh(positions,values,left=bottom,label=PHASE_LABELS[field])
        else:ax.bar(positions, values, bottom=bottom, label=PHASE_LABELS[field])
        bottom += values
    if horizontal:
        ax.set_yticks(positions,[wrap(name,32) for name in names],fontsize=8)
        ax.invert_yaxis()
        ax.set(title='阶段时间（独立统计）',xlabel='时间 (ms)')
    else:
        ax.set_xticks(positions, [wrap(name, 18) for name in names], rotation=65,ha='right',fontsize=7)
        ax.set(title='阶段时间（独立统计）', ylabel='时间 (ms)')


E4B_DISTRIBUTIONS = DISTRIBUTION_LABELS
E4B_CONTEXT = ('host', 'n', 'seed', 'impl', 'variant', 'procs', 'gpus',
               'buckets_per_thread', 'schedule', 'bind_policy', 'places_policy',
               'batch', 'naive_atomics', 'scatter', 'mempolicy', 'launcher',
               'gpu_affinity', 'exchange_path')


def e4b_subsets(data, quick):
    data = data[data.experiment == 'E4b'].copy()
    if 'quick' in data:
        mode = data.quick.map(lambda value: str(value).lower() in ('true', '1', '1.0'))
        data = data[mode == bool(quick)].copy()
    for column in ('n', 'units', 'threads', 'procs', 'gpus', 'oversample', 'tiebreak'):
        data[column] = pd.to_numeric(data[column], errors='coerce')
    data = data[(data.n == (1 << 18 if quick else 1 << 28)) & (data.impl == 'omp') &
                (data.variant == 'psrs') & (data.procs == 1) & (data.gpus == 0) &
                (data.units == data.threads)]
    oversample = data[(data.threads == 64) & (data.tiebreak == 1) &
                      data.oversample.isin((1, 2, 4, 8, 16, 32)) &
                      data.dist.isin(('uniform', 'normal', 'hotkeys'))].copy()
    tiebreak = data[(data.oversample == 1) & data.threads.isin((8, 16, 32, 64, 128)) &
                    data.tiebreak.isin((0, 1)) & data.dist.isin(('hotkeys', 'few_unique'))].copy()
    return oversample, tiebreak


def e4b_context_groups(data):
    if data.empty:
        return []
    data = scaling_identity(data)
    keys = [key for key in E4B_CONTEXT if key in data]
    varying = [key for key in keys if data[key].nunique(dropna=False) > 1]
    groups = []
    for _, group in data.groupby(keys, dropna=False, sort=True):
        row = group.iloc[0]
        suffix = '，' + '，'.join(CONTEXT_LABELS.get(key, key)+'='+translated(key, row[key])
                                for key in varying) if varying else ''
        groups.append((suffix, group))
    return groups


def e4b_xaxis(ax, values, axis_label):
    ax.set_xscale('log', base=2)
    ax.set_xticks(values, [str(value) for value in values])
    ax.set_xlabel(axis_label)


def plot_e4b(data, quick):
    oversample, tiebreak = e4b_subsets(data, quick)
    suffix = '_quick' if quick else ''
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, dist in zip(axes.flat, ('uniform', 'normal', 'hotkeys')):
        right = ax.twinx()
        for index, (context, group) in enumerate(e4b_context_groups(oversample[oversample.dist == dist])):
            group = group.sort_values('oversample')
            color = f'C{index % 10}'
            ax.plot(group.oversample, group.imbalance, 'o-', color=color, label='失衡度'+context)
            right.plot(group.oversample, group.median_ms, 's--', color=color, label='总时间'+context)
        ax.set(title=DISTRIBUTION_LABELS[dist], ylabel='失衡度')
        right.set_ylabel('总时间 (ms)')
        right.grid(False)
        e4b_xaxis(ax, [1, 2, 4, 8, 16, 32], '过采样系数 k')
        if not ax.lines:
            ax.text(.5, .5, '暂无符合参数的实测数据', ha='center', transform=ax.transAxes)
    ax = axes[1, 1]
    for index, dist in enumerate(('uniform', 'normal', 'hotkeys')):
        for context, group in e4b_context_groups(oversample[oversample.dist == dist]):
            group = group.sort_values('oversample')
            ax.plot(group.oversample, group.splitters_ms, 'o-', color=f'C{index}',
                    label=DISTRIBUTION_LABELS[dist]+context)
    ax.set(title='选择分割点的开销', ylabel='分割点时间 (ms)')
    e4b_xaxis(ax, [1, 2, 4, 8, 16, 32], '过采样系数 k')
    describe(fig, 'E4b', quick, title='过采样：失衡度、总时间与分割点开销')
    save(fig, 'E4b_oversample'+suffix)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, dist in zip(axes, ('hotkeys', 'few_unique')):
        for context, group in e4b_context_groups(tiebreak[tiebreak.dist == dist]):
            for tie, values in group.groupby('tiebreak'):
                values = values.sort_values('threads')
                ax.plot(values.threads, values.imbalance, 'o-' if tie else 's--',
                        color='C0' if tie else 'C1',
                        label=('开启复合键' if tie else '关闭复合键')+context)
        ax.set(title=DISTRIBUTION_LABELS[dist], ylabel='失衡度')
        e4b_xaxis(ax, [8, 16, 32, 64, 128], '处理单元数 p')
    describe(fig, 'E4b', quick, title='重复键：复合键开关与处理单元数')
    save(fig, 'E4b_tiebreak'+suffix)


def plot_e14(data, bandwidth, quick):
    fig, ax = plt.subplots(figsize=(12, 6))
    stream = bandwidth[bandwidth.quick == quick].copy() if not bandwidth.empty else bandwidth
    if not stream.empty:
        stream['places_policy'] = stream.apply(
            lambda row: '自动物理核/硬件线程' if str(row.places) == ('cores' if row.threads <= 64 else 'threads') else '显式：'+str(row.places), axis=1)
        keys = [key for key in ('host', 'n', 'bind', 'bind_requested', 'kernel', 'places_policy') if key in stream]
        for config, group in stream.groupby(keys, dropna=False):
            group = group.sort_values('threads')
            row = group.iloc[0]
            kernel = '复制' if row.kernel == 'copy' else '三项运算'
            name = f'STREAM {kernel}，{translated("bind", row.bind)}，n={number(row.n)}，主机={row.host}'
            ax.plot(group.threads, group.best_GBps, 'o-', label=name)
    if not quick and not data.empty and 'merge_GBps' in data:
        merge = data[(~data.quick) & data.experiment.isin(('E2', 'E5')) & data.merge_GBps.notna()]
        for experiment, selected in merge.groupby('experiment'):
            prepared = scaling_identity(selected)
            for _, group in scaling_groups(selected):
                group = group.sort_values('units')
                ax.plot(group.units, group.merge_GBps, 's--',
                        label=experiment+' 归并，'+context_name(group.iloc[0], prepared))
    if not ax.lines:
        ax.text(.5, .5, '暂无本模式的已验证实测数据', ha='center', transform=ax.transAxes)
    p_axis(ax)
    ax.set(ylabel='带宽 (GB/s)', title='最佳实测带宽与归并有效带宽')
    describe(fig, 'E14', quick, source='tables/bandwidth.csv',
             series=[line.get_label() for line in ax.lines] +
                    ([] if quick else ['PSRS 归并曲线仅取完整 E2/E5，来源 tables/summary.csv。']))
    save(fig, 'E14'+('_quick' if quick else ''))


def distribution_bars(data, metric, ax):
    positions = {dist: index for index, dist in enumerate(DISTRIBUTION_LABELS)}
    labels = list(positions)
    contexts = []
    # Each exact configuration owns its curve; the distribution is the x variable.
    prepared = scaling_identity(data)
    keys = [key for key in scaling_keys(prepared) if key != 'dist']
    for _, group in prepared.groupby(keys, dropna=False):
        contexts.append(group)
    width = .78/max(1, len(contexts))
    for index, group in enumerate(contexts):
        x = [positions[dist]+(index-(len(contexts)-1)/2)*width for dist in group.dist]
        ax.bar(x, group[metric], width=width, label=context_name(group.iloc[0], prepared,exclude=('dist',)))
    ax.set_xticks(list(positions.values()), [DISTRIBUTION_LABELS[dist] for dist in labels], rotation=35, ha='right')
    ax.set_ylabel(FIELD_LABELS.get(metric, metric))


def plot_e6(data, quick):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dist in zip(axes, ('normal', 'power')):
        chosen = data[(data.dist == dist) & (data.impl != 'seq')]
        prepared = scaling_identity(chosen)
        keys = [key for key in scaling_keys(prepared) if key not in ('_identity_buckets_per_thread', '_identity_schedule')]
        contexts = list(prepared.groupby(keys, dropna=False))
        schedules = ('static', 'dynamic', 'guided')
        width = .75/max(1, len(contexts)*len(schedules))
        count = 0
        for _, group in contexts:
            for schedule in schedules:
                values = group[group.schedule == schedule].sort_values('buckets_per_thread', key=lambda column: pd.to_numeric(column))
                x = [list((1, 4, 16)).index(int(float(value))) + (count-(len(contexts)*3-1)/2)*width for value in values.buckets_per_thread]
                suffix=context_name(group.iloc[0],prepared,exclude=('buckets_per_thread','schedule')).removeprefix(label(group.iloc[0]))
                ax.bar(x, values.median_ms, width=width, label=translated('schedule', schedule)+suffix)
                count += 1
        ax.set_xticks([0, 1, 2], ['1', '4', '16'])
        ax.set(title=DISTRIBUTION_LABELS[dist], xlabel='每线程桶数', ylabel='时间 (ms)')
    describe(fig, 'E6', quick)
    save(fig, 'E6'+('_quick' if quick else ''))


def plot_e9(data, quick):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    pure = data[(data.impl == 'mpi') & (data.scatter.astype(str).isin(('0', '0.0')))]
    curves(pure, 'median_ms', axes[0, 0], '纯 MPI 总时间')
    curves(pure, 'exchange_ms', axes[0, 1], '纯 MPI 交换时间')
    hybrid = data[data.impl == 'hybrid'].sort_values('procs')
    positions = list(range(len(hybrid)))
    axes[1, 0].bar([value-.18 for value in positions], hybrid.median_ms, width=.36, label='总时间')
    axes[1, 0].bar([value+.18 for value in positions], hybrid.exchange_ms, width=.36, label='交换时间')
    hybrid_context=scaling_identity(hybrid)
    axes[1, 0].set_xticks(positions, [f'{int(row.procs)} 进程 × {int(row.threads)} 线程'+context_name(row,hybrid_context).removeprefix(label(row)) for _, row in hybrid.iterrows()], rotation=25)
    axes[1, 0].set(title='固定处理单元总数的混合配置', ylabel='时间 (ms)')
    contrast = data[(data.impl == 'mpi') & (data.variant == 'psrs') & data.units.isin((8,64))].sort_values(['units','scatter'])
    positions=list(range(len(contrast)))
    axes[1,1].bar(positions,contrast.median_ms,label='排序总时间')
    if 'scatter_ms' in contrast:
        axes[1,1].bar(positions,contrast.scatter_ms,bottom=contrast.median_ms,label='初始 Scatter 分发')
    prepared=scaling_identity(contrast)
    names=[f'p={int(row.units)}，'+translated('scatter',row.scatter)+context_name(row,prepared,exclude=('scatter',)).removeprefix(label(row)) for _,row in contrast.iterrows()]
    axes[1,1].set_xticks(positions,[wrap(name,15) for name in names],rotation=25,ha='right',fontsize=8)
    axes[1,1].set(title='已记录的排序与初始分发时间',ylabel='时间 (ms)')
    describe(fig, 'E9', quick, caption='Scatter 单独计时；此处显示排序与分发，未计入两种方式的数据生成时间。')
    save(fig, 'E9'+('_quick' if quick else ''))


def pipeline_family(row):
    if row.impl != 'pipeline' or row.variant != 'block':
        return label(row)
    batch = float(row.batch)
    ratio = float(row.n)/(float(row.procs)*batch) if batch else 0
    if ratio in (2, 8, 32):
        return '分块流水线，批大小=n/('+number(ratio)+'p)'
    return '分块流水线，批大小='+number(batch)


def plot_e10(data, quick):
    chosen = data[data.impl != 'seq']
    scales = sorted(chosen.loc[chosen.variant != 'elem', 'n'].unique())
    panels = len(scales)+1
    fig, axes = plt.subplots(math.ceil(panels/2), 2, figsize=(13, 4*math.ceil(panels/2)), squeeze=False)
    elem = chosen[chosen.variant == 'elem']
    curves(elem, 'median_ms', axes.flat[0], '逐元素流水线（n=p）', weak=True)
    for ax, n in zip(list(axes.flat)[1:], scales):
        selected = chosen[(chosen.n == n) & (chosen.variant != 'elem')].copy()
        selected['pipeline_family'] = selected.apply(pipeline_family, axis=1)
        prepared = scaling_identity(selected)
        keys = [key for key in scaling_keys(prepared) if key != '_identity_batch']+['pipeline_family']
        for _, group in prepared.groupby(keys, dropna=False):
            group = group.sort_values('units')
            row=group.iloc[0]
            name=row.pipeline_family+context_name(row,prepared,exclude=('batch','scale_n')).removeprefix(label(row))
            ax.plot(group.units, group.median_ms, 'o-', label=name)
        p_axis(ax)
        ax.set(title='规模 n='+number(n), ylabel='时间 (ms)')
    for ax in list(axes.flat)[panels:]:
        ax.axis('off')
    describe(fig, 'E10', quick)
    save(fig, 'E10'+('_quick' if quick else ''))


def plot_e11(data, quick):
    chosen = data[data.gpus > 0].copy()
    if chosen.empty:
        return placeholder('E11', quick, '暂无已验证的 GPU 实测数据。')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    prepared = scaling_identity(chosen)
    keys = [key for key in scaling_keys(prepared) if key != 'scale_n']
    for _, group in prepared.groupby(keys, dropna=False):
        group = group.sort_values('n')
        name = context_name(group.iloc[0],prepared,exclude=('scale_n',))
        path=str(group.iloc[0].get('exchange_path','none'))
        if path!='none':name+='，'+TUNING_LABELS.get('exchange_path',{}).get(path,path)
        axes[0].plot(group.n, group.median_ms, 'o-', label=name)
        if group.compute_ms.notna().any():
            axes[1].plot(group.n, group.compute_ms, 'o-', label=name+'：计算')
        transfer = group.h2d_ms+group.d2h_ms
        axes[1].plot(group.n, transfer, 's--', label=name+'：主机/设备复制')
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set(xlabel='键数量 n', ylabel='时间 (ms)')
    axes[0].set_title('包含全部开销的总时间')
    axes[1].set_title('计算与主机/设备复制时间')
    describe(fig, 'E11', quick, caption='计算时间不含 H2D/D2H；双卡交换路径取自原始记录。')
    save(fig, 'E11'+('_quick' if quick else ''))
    ordered=chosen.sort_values(['n','impl','variant','gpu_affinity','naive_atomics'])
    fig,ax=plt.subplots(figsize=(13,max(6,len(ordered)*.44)))
    phase_bars(ordered,ax,horizontal=True)
    describe(fig,'E11',quick,title='GPU 阶段时间分解',source='tables/summary.csv',caption='各柱取对应配置的实测阶段中位数；NUMA 绑定及跨卡路径保留在配置标签和原始表中。')
    save(fig,'E11_phases'+('_quick' if quick else ''))


def plot_e13(data, quick, fits):
    selected = data[(data.experiment == 'E2') & (data.impl != 'seq')]
    if selected.empty:
        return placeholder('E13', quick)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    curves(selected, 'median_ms', axes[0], '实测总时间与 Amdahl 拟合')
    curves(selected[selected.units > 1], 'karp_flatt', axes[1], 'Karp–Flatt 实验串行比例')
    for record in fits:
        if bool(record['quick']) != bool(quick):
            continue
        points = record.get('data_points', [])
        if not points:
            continue
        p = [point['p'] for point in points]
        predicted = [point['baseline_ms']*(record['f']+(1-record['f'])/point['p'])+record['alpha_ms']*(point['p']-1) for point in points]
        row = scaling_identity(pd.DataFrame([points[0]['config']])).iloc[0]
        name = context_name(row,scaling_identity(selected))+f'：拟合 f={record["f"]:.3g}'
        axes[0].plot(p, predicted, '--', label=name)
    describe(fig, 'E13', quick, source='tables/E13_fits.json')
    save(fig, 'E13'+('_quick' if quick else ''))


def plot_experiment(experiment, data, quick, fits):
    suffix = '_quick' if quick else ''
    if experiment == 'E13':
        return plot_e13(data, quick, fits)
    selected = data[data.experiment == experiment].copy()
    if selected.empty:
        return placeholder(experiment, quick)
    if experiment == 'E2':
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        selected = selected[selected.impl != 'seq']
        curves(selected, 'speedup', axes[0], '强扩展加速比')
        curves(selected, 'efficiency', axes[1], '并行效率')
    elif experiment == 'E3':
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        selected = selected[selected.impl != 'seq']
        curves(selected, 'median_ms', axes[0], '每单元规模固定时的总时间', weak=True)
        curves(selected, 'speedup', axes[1], '相对同规模串行基线的加速比', weak=True)
    elif experiment == 'E4':
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        selected = selected[selected.impl != 'seq']
        distribution_bars(selected, 'median_ms', axes[0])
        distribution_bars(selected, 'imbalance', axes[1])
        axes[0].set_title('不同分布的总时间')
        axes[1].set_title('不同分布的负载失衡度')
    elif experiment == 'E5':
        selected = selected[selected.impl != 'seq']
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ordered = measured_bars(selected, axes[0], '总时间（中位数与最小/最大范围）')
        phase_bars(ordered, axes[1])
        describe(fig, experiment, quick, caption=MPI_CAPTION,source='tables/summary.csv')
        return save(fig, experiment+suffix)
    elif experiment == 'E6':
        return plot_e6(selected, quick)
    elif experiment == 'E7':
        selected = selected[selected.impl != 'seq']
        fig, ax = plt.subplots(figsize=(12, max(5, len(selected)*.48)))
        selected = selected.sort_values(['variant', 'bind', 'mempolicy'])
        names = [label(row)+'，'+translated('bind', row.bind)+'，'+translated('mempolicy', row.mempolicy) for _, row in selected.iterrows()]
        ax.barh(list(range(len(selected))), selected.median_ms,
                xerr=[selected.median_ms-selected.min_ms, selected.max_ms-selected.median_ms], capsize=2)
        ax.set_yticks(list(range(len(selected))), [wrap(name, 28) for name in names])
        ax.invert_yaxis()
        ax.set(xlabel='时间 (ms)', title='初始化、绑定与内存策略的独立配置')
    elif experiment == 'E8':
        fig, ax = plt.subplots(figsize=(11, 5))
        curves(selected[selected.impl != 'seq'], 'median_ms', ax, '锁竞争与线程本地桶')
        ax.set_yscale('log')
    elif experiment == 'E9':
        return plot_e9(selected, quick)
    elif experiment == 'E10':
        return plot_e10(selected, quick)
    elif experiment == 'E11':
        return plot_e11(selected, quick)
    elif experiment == 'E12':
        fig, ax = plt.subplots(figsize=(13, max(5, len(selected)*.48)))
        measured_bars(selected, ax, '完整排序总时间', horizontal=True)
        ax.set_xscale('log')
    else:
        return placeholder(experiment, quick)
    describe(fig, experiment, quick)
    save(fig, experiment+suffix)


def main():
    global REGISTRY
    DEST.mkdir(parents=True, exist_ok=True)
    migrate_quick(DEST)
    REGISTRY = FigureRegistry(DEST)
    path = TABLES/'summary.csv'
    if not path.is_file():
        raise RuntimeError('缺少统计表，请先运行 analysis/summarize.py。')
    data = pd.read_csv(path, dtype={key: str for key in TUNING})
    data['quick'] = data.quick.map(lambda value: str(value).lower() in ('true', '1', '1.0'))
    bandwidth = pd.read_csv(TABLES/'bandwidth.csv') if (TABLES/'bandwidth.csv').is_file() else pd.DataFrame()
    if not bandwidth.empty:
        bandwidth['quick'] = bandwidth.quick.map(lambda value: str(value).lower() in ('true', '1', '1.0'))
    fits = json.loads((TABLES/'E13_fits.json').read_text()) if (TABLES/'E13_fits.json').is_file() else []
    for quick in (False, True):
        selected = data[data.quick == quick]
        plot_e4b(selected, quick)
        plot_e14(data, bandwidth, quick)
        for number in range(2, 14):
            plot_experiment(f'E{number}', selected, quick, fits)
    # Historical generated images remain discoverable after quick-file migration.
    known = {path for record in REGISTRY.records for path in record['paths']}
    groups = {}
    for path in sorted(DEST.rglob('*')):
        if path.suffix.lower() in ('.png', '.pdf'):
            relative = str(path.relative_to(DEST.parent))
            if relative not in known:
                groups.setdefault((path.parent, path.stem), []).append(path)
    for (_, name), paths in groups.items():
        experiment = name.split('_')[0]
        quick = name.endswith('_quick') or paths[0].parent.name == 'quick'
        REGISTRY.add(name=name, title=TITLES.get(experiment, '历史生成图：'+name),
                     experiment=experiment, quick=quick,
                     source_table=source_table(experiment, quick),
                     series=['历史生成图；以对应实测表为准。'], paths=paths)
    REGISTRY.write()
    print(f'图表与图片清单已保存至 {DEST}')


if __name__ == '__main__':
    main()
