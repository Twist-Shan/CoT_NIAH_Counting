"""Describe every frozen Top-1..8 ablation and plot direct Aurora results."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
METRICS=['ar_accuracy','ar_answered','next_marker_correct','next_marker_identifiable',
         'immediate_marker_correct','immediate_marker_valid','joint_marker_and_count_failure',
         'eos_reached','generation_capped','answer_and_eos_correct','normalized_count_shift',
         'absolute_count_shift','new_tokens','trace_exact']
COLORS={'nonthinking':'#B52F6B','thinking':'#007EAB'}
GRAY,INK='#8190A5','#161923'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(frame):
    lines=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join('—' if pd.isna(v) else f'{v:.2f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def load_results(out,protocol):
    rows,by_count=[],[]
    observed=0
    for mode in ['thinking','nonthinking']:
        paths=[out/mode/'clean.csv']+sorted((out/mode/'arms').glob('*.csv'))
        expected=1+sum(a['mode']==mode for a in protocol['arms'])*len(protocol['scopes'][mode])
        assert len(paths)==expected,(mode,len(paths),expected)
        for path in paths:
            header=pd.read_csv(path,nrows=0).columns
            wanted=['key','mode','count','primary_eligible','boundary_count1','scope','arm','top_k','repeat','heads','overlap']+METRICS
            f=pd.read_csv(path,usecols=[c for c in wanted if c in header]).fillna({'heads':''})
            assert not f.key.duplicated().any()
            assert len(f)==(98 if mode=='thinking' else 100)
            observed+=len(f)
            for support,g in [('primary',f.loc[f.primary_eligible]),('all_available',f),('count1_boundary',f.loc[f.boundary_count1])]:
                if g.empty:continue
                row={'mode':mode,'scope':f.scope.iloc[0],'arm':f.arm.iloc[0],
                     'top_k':int(f.top_k.iloc[0]),'repeat':int(f['repeat'].iloc[0]),
                     'heads':f.heads.iloc[0],'overlap':int(f.overlap.iloc[0]),'support':support,'inputs':len(g)}
                for metric in METRICS:
                    row[metric]=g[metric].mean() if metric in g else np.nan
                    row[metric+'_n']=int(g[metric].notna().sum()) if metric in g else 0
                rows.append(row)
            for count,g in f.groupby('count'):
                row={'mode':mode,'scope':f.scope.iloc[0],'arm':f.arm.iloc[0],'top_k':int(f.top_k.iloc[0]),
                     'repeat':int(f['repeat'].iloc[0]),'count':int(count),'inputs':len(g)}
                row.update({c:g[c].mean() for c in METRICS if c in g})
                by_count.append(row)
    expected=sum((1+sum(a['mode']==m for a in protocol['arms'])*len(protocol['scopes'][m]))*(98 if m=='thinking' else 100) for m in ['thinking','nonthinking'])
    assert observed==expected
    result=pd.DataFrame(rows)
    result.to_csv(out/'all_condition_summary.csv',index=False)
    pd.DataFrame(by_count).to_csv(out/'per_count_summary.csv',index=False)
    return result,observed


def contrasts(summary):
    rows=[]
    for (mode,scope,support,k),g in summary.loc[summary.arm.ne('clean')].groupby(['mode','scope','support','top_k']):
        selected=g.loc[g.arm.eq('selected')].iloc[0]
        controls=g.loc[g.arm.eq('control')]
        assert len(controls)>0
        for metric in METRICS:
            if pd.isna(selected[metric]):continue
            rows.append({'mode':mode,'scope':scope,'support':support,'top_k':k,'metric':metric,
                         'inputs':int(selected.inputs),'controls':len(controls),
                         'overlap':int(controls.overlap.iloc[0]),
                         'selected':selected[metric],'control_mean':controls[metric].mean(),
                         'control_min':controls[metric].min(),'control_max':controls[metric].max(),
                         'control_minus_selected':controls[metric].mean()-selected[metric],
                         'selected_valid_n':int(selected[metric+'_n']),
                         'control_valid_n_min':int(controls[metric+'_n'].min()),
                         'control_valid_n_max':int(controls[metric+'_n'].max())})
    return pd.DataFrame(rows)


def panel(ax,summary,comparison,mode,metric,title,scope='sustained'):
    f=comparison.loc[comparison['mode'].eq(mode)&comparison.scope.eq(scope)&comparison.support.eq('primary')&comparison.metric.eq(metric)].sort_values('top_k')
    x=f.top_k.to_numpy()
    ax.fill_between(x,100*f.control_min.to_numpy(),100*f.control_max.to_numpy(),color=GRAY,alpha=.17,linewidth=0)
    ax.plot(x,100*f.control_mean,'o--',color=GRAY,label='Control mean',markersize=4.8)
    ax.plot(x,100*f.selected,'o-',color=COLORS[mode],label='Selected',markersize=5.6)
    clean=summary.loc[summary['mode'].eq(mode)&summary.arm.eq('clean')&summary.support.eq('primary'),metric].iloc[0]
    ax.axhline(100*clean,color=INK,ls=':',lw=1.4,label='Clean')
    ax.set(title=title,xlim=(.75,8.25),ylim=(-3,103),xlabel='Number of ablated heads',ylabel='Accuracy (%)')
    ax.set_xticks(range(1,9));ax.set_yticks([0,25,50,75,100])
    if mode=='nonthinking':
        upper=max(30,int(np.ceil(max(100*clean,100*f.control_max.max(),100*f.selected.max())/10))*10)
        ax.set_ylim(-1,upper+1);ax.set_yticks(range(0,upper+1,10))
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#E8E8ED',linewidth=.8)
    ax.set_axisbelow(True)


def plot(out,summary,comparison):
    plt.rcParams.update({'font.family':'Times New Roman','font.size':12,'axes.titlesize':14,
        'axes.labelsize':12.5,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':11.5,
        'legend.frameon':False,'text.color':INK,'axes.labelcolor':INK,'axes.titlecolor':INK,
        'lines.linewidth':1.8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(1,3,figsize=(11.4,3.85))
    specs=[('nonthinking','ar_accuracy','A. Broad: final count'),
           ('thinking','next_marker_correct','B. Targeted: next marker'),
           ('thinking','ar_accuracy','C. Targeted: final count')]
    for ax,spec in zip(axs,specs):panel(ax,summary,comparison,*spec)
    fig.subplots_adjust(left=.065,right=.987,bottom=.27,top=.86,wspace=.36)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles=[Line2D([],[],color=COLORS['nonthinking'],marker='o'),
             Line2D([],[],color=COLORS['thinking'],marker='o'),
             Line2D([],[],color=GRAY,marker='o',ls='--'),
             Line2D([],[],color=INK,ls=':'),Patch(facecolor=GRAY,alpha=.17)]
    labels=['Broad selected','Targeted selected','Control mean','Clean','Control range']
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.005),ncol=5)
    for ext in ['pdf','png','svg']:
        fig.savefig(out/f'top1to8_sustained_ablation.{ext}',dpi=240)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(8.1,3.85))
    for ax,scope,title in zip(axs,['answer_query_only','sustained'],['A. Initial answer query only','B. Sustained from answer query']):
        panel(ax,summary,comparison,'nonthinking','ar_accuracy',title,scope)
    fig.subplots_adjust(left=.09,right=.98,bottom=.27,top=.86,wspace=.35)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.015),ncol=3)
    for ext in ['pdf','png']:
        fig.savefig(out/f'broad_timing_reference.{ext}',dpi=240)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=ROOT/'work/v58_top1to8_final_query_20260908')
    args=p.parse_args();out=args.input
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='complete'
    for name,digest in manifest['files'].items():assert sha(out/name)==digest,name
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    summary,observed=load_results(out,protocol)
    comparison=contrasts(summary);comparison.to_csv(out/'selected_vs_all_controls.csv',index=False)
    plot(out,summary,comparison)
    primary=comparison.loc[comparison.support.eq('primary')&comparison.scope.eq('sustained')]
    def point(mode,k,metric):
        return primary.loc[primary['mode'].eq(mode)&primary.top_k.eq(k)&primary.metric.eq(metric)].iloc[0]
    t4,t8=point('thinking',4,'next_marker_correct'),point('thinking',8,'next_marker_correct')
    nt4,nt8=point('nonthinking',4,'ar_accuracy'),point('nonthinking',8,'ar_accuracy')
    rows=[]
    for k in range(1,9):
        row={'K':k}
        for mode,metric,name in [('nonthinking','ar_accuracy','NT count'),('thinking','next_marker_correct','T marker'),('thinking','ar_accuracy','T count')]:
            q=primary.loc[primary['mode'].eq(mode)&primary.top_k.eq(k)&primary.metric.eq(metric)].iloc[0]
            row[name+' selected (%)']=100*q.selected
            row[name+' control (%)']=100*q.control_mean
        rows.append(row)
    main_table=pd.DataFrame(rows)
    main_table.to_csv(out/'compact_accuracy_table.csv',index=False)
    control_rows=[]
    for k in range(1,9):
        row={'K':k}
        for mode,tag in [('nonthinking','NT'),('thinking','T')]:
            plan=protocol['plans'][mode][str(k)]
            row[tag+' controls']=len(plan['controls']);row[tag+' overlap']=plan['minimum_overlap']
        control_rows.append(row)
    validrows=[]
    for mode in ['nonthinking','thinking']:
        for k in range(1,9):
            q=primary.loc[primary['mode'].eq(mode)&primary.top_k.eq(k)&primary.metric.eq('ar_answered')].iloc[0]
            validrows.append({'Mode':mode,'K':k,'Selected count parsed (%)':100*q.selected,'Control count parsed (%)':100*q.control_mean})
    eq=json.loads((out/'cached_equivalence.json').read_text())
    clean=summary.loc[summary.arm.eq('clean')&summary.support.eq('primary'),['mode','inputs','ar_accuracy','next_marker_correct','ar_answered']]
    report=f'''# Top-1 到 Top-8 消融：固定最后检索 query（2026-09-08）

状态：已完成。v58 单训练种子，step 10000；全部 {observed:,} 条输出均保留。没有依据干预结果筛选头、control 或样本，没有新增显著性检验。

## 设置与评分

Selected 取 200 个 discovery 输入上冻结的全局 ranking 前 K 个 heads，K=1..8。Control 匹配每层头数，优先排除 selected，结构上不足时采用最少必要重叠；穷举全部可行组合。共 602 组 controls（NT 488、T 114），每种模式各 8 个 selected 条件。不同 K 的 control pool 不同，K>=5 开始存在必要重叠。

NT 在原始 Ans query 开始干预，100 个输入（每个 count 十个）。T 在 clean 自然轨迹的第 N 个 Sep，即最后一个 marker 的 query 开始干预；干预后不提供 gold token。100 个原输入中两条 clean 轨迹未到达该 query，事先固定为 anchor 不可用。98 个可用输入中，count>=2 的 88 个构成主结果，count=1 的十个单独保存，所有条件共用同一集合。先前 marker 内容是否正确不作为筛选条件。

两种模式主要结果均保持干预到 EOS 或 64-token 上限。完整前缀计算始终保留早先位置的干预；缓存实现保留干预后的实际 KV states。额外运行 NT 初始 answer-query-only 版本作为正文 broad 实验时程参考。主实验 targeted 是最后一次检索起点；此前 synthetic 从第一次 Sep 开始的结果继续保留，两者不可混称。

全词表 greedy，不强制合法 marker 或数字。T 的 next-marker 分数使用 query 后、trace close/Ans/EOS 前的第一个 character token；允许先出现非 character 的格式 token，但绝不跳过错误 character 去寻找后续正确答案。立即下一 token 的合法率与正确率另外保存。目标 marker 正确之后，其他输出错误不扣 marker 分数。Count 仍用原严格解析器：第一个 Ans 后紧邻的 atomic number，缺失或无法解析计 count 错误；EOS 不影响已解析 count 正确率。

Broad 的 |ablated count - clean count|/N 仅在两者可解析时有定义，缺失保持 NaN，另记录有效样本数。它不能代替全样本 count 准确率或把无效输出当作零变化。

## Clean 基线

{table(clean)}

## 全部 K 的主要准确率

表内 selected 与 control 使用相同输入；control 为所有可行组合的等权平均。

{table(main_table)}

Targeted 的 next-marker 指标显示，Top-4 selected 准确率为 {100*t4.selected:.2f}%，control 为 {100*t4.control_mean:.2f}%；Top-8 分别为 {100*t8.selected:.2f}% 和 {100*t8.control_mean:.2f}%。在这些输入和冻结控制集合下，selected 对 next-marker 选择造成更大损伤。最终 count 上，control 损伤仍更大，不能将 next-marker 的作用解释为整个计数过程的统一重要性排序。

Broad 的 count 指标没有支持 selected 更具因果针对性：Top-4 selected 为 {100*nt4.selected:.2f}%，control 为 {100*nt4.control_mean:.2f}%；Top-8 分别为 {100*nt8.selected:.2f}% 和 {100*nt8.control_mean:.2f}%。Selected 相对 clean 的下降说明这些 heads 参与当前任务，但 control 损伤更大，不能据此声称 broad-score 选头优于同层 control。解析失败的影响需结合下表判断。

![Sustained ablation](top1to8_sustained_ablation.png)

图中 A 为 NT count 准确率（100 输入），B/C 分别为 T next-marker/count 准确率（同一 88 输入）；彩色为 selected，灰色线为 control 均值，浅灰区间为 control 最小至最大值，虚线为 clean。A 与 B/C 的纵轴范围不同。区间表示 head-set 差异，不是置信区间。

## Control 数与必要重叠

{table(pd.DataFrame(control_rows))}

## 答案可解析率

{table(pd.DataFrame(validrows))}

## Broad 干预时程参考

![Broad timing](broad_timing_reference.png)

两图均为 100 个输入上的 count 准确率。初始 query-only 对应正文原 broad 时程；sustained 对应用户要求的延长干预。

## 验证

11 项控制组合、query 位置、评分与真实切片 CPU 检查通过；8 项缓存 RoPE、单层／跨层干预、不同长度前缀与完整前缀轨迹对照检查通过。

GPU 上 clean 前缀续写与原始 clean 轨迹逐 token 一致；batch=1 与不同长度 batch 的轨迹一致。Top-4 selected 和 Top-8 control 的真实 pre-O slices 检查通过，其他切片不变且无 hook 泄漏。

缓存与完整前缀等价核查：{json.dumps(eq,ensure_ascii=False)}。

## 可复现文件

- `protocol.json`：运行前冻结的 ranking、602 组 controls、输入、checkpoint 与代码哈希。
- `anchor_registry.csv`、`prefixes.jsonl`：所有原始输入的 query 可用性和固定 clean 前缀。
- 两模式 `arms/*.csv`：每个条件完整输出；没有覆盖旧实验。
- `all_condition_summary.csv`：所有条件的全部指标，含 primary、all-available、count1。
- `selected_vs_all_controls.csv`：每个 K 的均值、范围、差值和数值位移有效样本数。
- `per_count_summary.csv`：逐 count 结果。
- `validation.json`、`cached_equivalence.json`、`cache_execution.json`、`manifest.json`：实现检查与来源校验。

所有图为新增结果图，使用正文 Aurora 模式颜色和 A. B. C. 标注。正文、appendix 和既有图未改动。本轮只有一个训练种子；不能将遍历 control 组合当作独立训练重复。
'''
    (out/'RESULTS.md').write_text(report,encoding='utf-8')
    print(main_table.to_string(index=False))
    print('Rows:',observed,'Report:',out/'RESULTS.md')


if __name__=='__main__':main()
