"""Build the uniform-budget body-mechanism report; never mix historical panels."""
from __future__ import annotations
import argparse
import base64
import hashlib
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_v58_synthetic_report import geometry_projection_widget
from v58_alignment_core import paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'work/v58_final'
ALIGN = DATA/'analysis/v58_alignment_supplement_20260905'
LEGACY = DATA/'analysis/v58_unified_legacy_20260905'
EXTRA = DATA/'analysis/v58_unified_additional_20260905'
ASSETS = ROOT/'reports/assets/v58_unified_20260905'
MODES = ['nonthinking', 'thinking']
COLORS = {'nonthinking': '#ce7241', 'thinking': '#267cb0'}


def read(path):
    return pd.read_csv(path)


def pct(x):
    return f'{100*x:.1f}%'


def table(frame, columns=None, digits=3):
    if columns is not None:
        frame = frame[columns]
    return '<div class="table-scroll">'+frame.to_html(index=False, border=0, na_rep='—', float_format=lambda x: f'{x:.{digits}f}')+'</div>'


def fig(fig, filename, caption):
    path = ASSETS/filename
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return '<figure><img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"><figcaption>'+caption+'</figcaption></figure>'


def conclusion(text):
    return f'<p class="conclusion"><b>当前结论。</b>{text}</p>'


def contrast(left, right, metric, keys=('key',), scale=1):
    a = left.groupby(list(keys)).agg(value=(metric, 'mean'), block=('block', 'first')).reset_index()
    b = right.groupby(list(keys))[metric].mean().reset_index(name='control')
    merged = a.merge(b, on=list(keys), validate='one_to_one')
    assert len(merged) == len(a) == len(b)
    result = paired_bootstrap((merged.value-merged.control).to_numpy()*scale, merged.block.to_numpy())
    return f"{result['effect']:.2f} [{result['ci_low']:.2f}, {result['ci_high']:.2f}]"


def behavior_and_ablation(ablation):
    clean = ablation.loc[ablation.arm.eq('clean')]
    bycount = clean.groupby(['mode', 'count']).ar_accuracy.agg(['mean', 'size']).reset_index()
    fig1, axes = plt.subplots(1, 2, figsize=(11.5, 4), layout='constrained')
    for mode in MODES:
        frame = bycount.loc[bycount['mode'].eq(mode)]
        axes[0].plot(frame['count'], frame['mean']*100, 'o-', color=COLORS[mode], label=mode)
        sub = ablation.loc[ablation['mode'].eq(mode)]
        for arm, style in [('selected', '-'), ('random', '--')]:
            d = sub.loc[sub.arm.eq(arm)].groupby('top_k').ar_accuracy.mean()
            axes[1].plot(d.index, d.values*100, 'o'+style, color=COLORS[mode], label=mode+' '+('selected' if arm=='selected' else 'matched controls'))
    axes[0].set(xlabel='True count', ylabel='Free-running answer accuracy (%)', xticks=range(1, 11), ylim=(-3, 103))
    axes[1].set(xlabel='Ablated heads K', ylabel='Free-running answer accuracy (%)', xticks=[1, 2, 4], ylim=(-3, 103))
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    image = fig(fig1, 'behavior_ablation.png', '图1｜左：每个数字10个相同confirmation输入的答案准确率；横轴为真实count，纵轴为自由生成正确率。右：同一100个输入的K剂量曲线，实线为按discovery选择的检索头，虚线为同层、等头数、不重叠对照的均值。K=1/2各3个不同对照，K=4仅1个可行补集；虚线不代表随机分布。无误差条，配对区间见正文。')
    summary = clean.groupby('mode')[['ar_accuracy', 'trace_exact', 'trace_marker_count_accuracy']].mean().reset_index()
    effects = []
    for mode in MODES:
        f = ablation.loc[ablation['mode'].eq(mode)]
        for k in [1, 2, 4]:
            selected = f.loc[f.arm.eq('selected') & f.top_k.eq(k)]
            controls = f.loc[f.arm.eq('random') & f.top_k.eq(k)]
            effects.append({'mode': mode, 'K': k, 'selected_answer': selected.ar_accuracy.mean(),
                'control_answer': controls.ar_accuracy.mean(),
                'selected_answered': selected.ar_answered.mean(), 'control_answered': controls.ar_answered.mean(),
                'selected_minus_control_pp_95CI': contrast(selected, controls, 'ar_accuracy', scale=100),
                'selected_trace_exact': selected.trace_exact.mean(), 'control_trace_exact': controls.trace_exact.mean()})
    return image, summary, pd.DataFrame(effects), bycount


def plot_dynamics():
    data = pd.concat([read(LEGACY/m/'dynamics_attention_summary.csv') for m in MODES])
    behavior = pd.concat([read(LEGACY/m/'dynamics_behavior_trials.csv') for m in MODES])
    image = '<p><b>原始指标主图。</b>固定物理head、固定100题、固定k-to-k配对；不按checkpoint重新选头。Targeted score为每题先平均所有k的 A(q_k, needle_k)，再对100题等权平均。q_k为预测第k个marker的前一位置。数值是占全部可见key的attention mass，不除以所有heads总分，也不除以全部needles的mass。示例：对正确needle分配0.4，其他keys合计0.6，则score为0.4。所有前向使用gold trace。</p>'
    order = [f'L{l}H{h}' for l in range(1,5) for h in range(8)]
    for role, modes, filename, vmax in [
        ('broad', MODES, 'broad_absolute_heatmaps.png', float(data.broad.max())),
        ('targeted', ['thinking'], 'targeted_absolute_heatmaps.png', 1.0),
    ]:
        raw_fig, raw_axes = plt.subplots(len(modes), 2, figsize=(13, 5*len(modes)), squeeze=False, layout='constrained')
        for row, mode in enumerate(modes):
            f = data.loc[data['mode'].eq(mode)].copy()
            f['physical_head'] = ['L%dH%d'%(l,h) for l,h in zip(f.layer,f['head'])]
            matrix = f.pivot(index='physical_head', columns='step', values=role).reindex(order)
            for col in range(2):
                mask = matrix.columns.to_numpy()>0 if col else np.ones(len(matrix.columns), bool)
                ax = raw_axes[row,col]
                im = ax.pcolormesh(matrix.columns.to_numpy()[mask], np.arange(32), matrix.iloc[:,mask].to_numpy(), shading='nearest', cmap='magma', vmin=0, vmax=vmax)
                ax.set(yticks=np.arange(32), yticklabels=order, title=f'{mode}: '+('absolute k-to-k attention mass' if role=='targeted' else 'answer-query broad score'), xlabel='Optimizer step (log scale)' if col else 'Optimizer step')
                if col: ax.set_xscale('log')
                ax.invert_yaxis(); ax.tick_params(axis='y', labelsize=6)
                ax.axvline(1500,color='#67bdce',ls='--',lw=1)
                raw_fig.colorbar(im,ax=ax,label='Correct-needle attention mass (0–1)' if role=='targeted' else 'Needle mass × effective coverage')
        caption = ('图4a｜Broad原始分数：上Non-thinking，下Thinking。最终answer query指向正文needles的总mass M，乘以coverage exp(H(p))/N，其中p为needles内部归一化attention，H为自然对数熵。两mode、全部steps共享固定色标，色标上限为两mode全时段最大值；该指标衡量answer-side正文覆盖，不测trace-query broad，也不证明最终聚合。' if role=='broad' else '图4b｜Thinking原始k-to-k attention mass，全部steps固定0–1色标。深层增强和浅层减弱可以直接跨checkpoint比较，无跨head归一化。')
        image += fig(raw_fig,filename,caption+' 横轴左为steps、右为log steps；纵轴为固定32个物理heads（层1–4，头0–7）。相同100题、count 1–10各10题，每题内k等权后题目等权；每100步评估，log省略step0，虚线为1500步loss-scope切换。无误差条，单训练seed。')
    th = data.loc[data['mode'].eq('thinking')].copy()
    th['role_share'] = th.targeted / th.groupby('step').targeted.transform('sum')
    th.to_csv(ASSETS/'thinking_head_dynamics.csv',index=False)
    layer = th.groupby(['step','layer'])[['targeted','correct_occurrence_top1']].mean().reset_index()
    layer.to_csv(ASSETS/'thinking_layer_dynamics.csv',index=False)
    diag, dax = plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for l,g in layer.groupby('layer'):
        dax[0].plot(g.step,g.targeted,label=f'L{l}')
        dax[1].plot(g.step,g.correct_occurrence_top1,label=f'L{l}')
    for l,h in [(1,6),(4,5)]:
        g=th.loc[th.layer.eq(l)&th['head'].eq(h)]
        dax[2].plot(g.step,g.targeted,label=f'L{l}H{h} mass')
        dax[2].plot(g.step,g.role_share,ls='--',label=f'L{l}H{h} share')
    dax[1].axhline(sum(1/k for k in range(1,11))/10,color='gray',ls=':',label='Uniform-needle reference')
    for ax,title in zip(dax,['Layer mean k-to-k mass','Layer mean correct-needle Top-1','Absolute mass vs head-role share']):
        ax.set(xlabel='Optimizer step',ylabel=title,ylim=(0,1)); ax.legend(fontsize=7); ax.grid(alpha=.2); ax.axvline(1500,color='gray',ls='--',lw=1)
    image+=fig(diag,'targeted_layer_diagnostics.png','图4c｜左：每层8个heads的平均原始k-to-k mass。中：正确needle在全部needles中attention排名第一的比例，再对题目及heads等权平均；灰线为均匀选择needle参考值29.29%，包含count=1，非统计显著性阈值。右：L1H6与L4H5原始mass（实线）及跨head份额（虚线），两头因本次问题事后选取，仅作描述。横轴steps，全部使用相同100题，无误差条、单训练seed。')
    fig1, axes = plt.subplots(4, 2, figsize=(13, 17), layout='constrained')
    for row, (mode, role) in enumerate([('nonthinking', 'broad'), ('thinking', 'broad'), ('thinking', 'targeted'), ('thinking', 'successor')]):
        f = data.loc[data['mode'].eq(mode)].copy()
        f['physical_head'] = ['L%dH%d'%(l,h) for l,h in zip(f.layer,f['head'])]
        matrix = f.pivot(index='physical_head', columns='step', values=role)
        order = [f'L{l}H{h}' for l in range(1,5) for h in range(8)]
        matrix = matrix.reindex(order)
        matrix = matrix.div(matrix.sum(axis=0), axis=1).fillna(0)
        for col in range(2):
            steps = matrix.columns.to_numpy(); mask = steps>0 if col else np.ones(len(steps), bool)
            x = steps[mask].astype(float); z = matrix.iloc[:, mask].to_numpy()
            im = axes[row,col].pcolormesh(x, np.arange(32), z, shading='nearest', cmap='magma', vmin=0, vmax=max(.16, matrix.to_numpy().max()))
            if col:
                axes[row,col].set_xscale('log')
            axes[row,col].set(xlabel='Optimizer step (log scale)' if col else 'Optimizer step',
                yticks=np.arange(32), yticklabels=order, title=f'{mode}: {role} role share')
            axes[row,col].tick_params(axis='y', labelsize=6)
            axes[row,col].axvline(1500, color='#67bdce', lw=1, ls='--')
            axes[row,col].invert_yaxis()
            fig1.colorbar(im, ax=axes[row,col], label='Head score / sum over 32 heads')
    image += '<details><summary>辅助图：跨head归一化份额（不用于比较原始强度）</summary>'+fig(fig1, 'role_heatmaps.png', '图4d｜四行依次为Non-thinking broad、Thinking broad、Thinking targeted、Thinking successor。固定100题与32个物理heads；颜色为head score除以同一checkpoint的32头score总和，每行全时段固定色标。横轴分别为steps/log steps，虚线为1500步；log省略step0。份额下降可来自其他heads增强，不能单独证明该head原始能力下降。')+'</details>'
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for mode in MODES:
        b = behavior.loc[behavior['mode'].eq(mode)].groupby('step').ar_accuracy.mean()
        axes2[0].plot(b.index, 100*b, 'o-', label=mode, color=COLORS[mode])
        f = data.loc[data['mode'].eq(mode)]
        sites = json.loads((ALIGN/mode/'frozen_sites.json').read_text())
        role = 'broad' if mode=='nonthinking' else 'targeted'
        heads = [tuple(x[:2]) for x in sites['ranking'][role][:4]]
        selected = f.loc[[tuple(x) in heads for x in f[['layer','head']].to_numpy()]]
        axes2[1].plot(selected.groupby('step')[role].mean(), label=mode+' fixed Top-4', color=COLORS[mode])
        share = selected.groupby('step')[role].sum()/f.groupby('step')[role].sum()
        axes2[2].plot(share, label=mode, color=COLORS[mode])
    axes2[0].set(xlabel='Optimizer step', ylabel='Free-running accuracy (%)', ylim=(0,103))
    axes2[1].set(xlabel='Optimizer step', ylabel='Mean fixed-bank role score')
    axes2[2].set(xlabel='Optimizer step', ylabel='Fixed Top-4 role share', ylim=(0,1)); axes2[2].axhline(4/32,ls=':',color='gray')
    for ax in axes2:
        ax.legend(fontsize=8); ax.grid(alpha=.2); ax.axvline(1500, ls='--', color='gray', lw=.8)
    image += fig(fig2, 'dynamics_behavior_bank.png', '图5｜横轴均为steps。左：预先固定11个checkpoint、每mode每点相同100个输入的自由生成准确率；中：最终200个discovery输入冻结的Top-4 bank的平均原始role score；右：该bank占32个heads总分的比例，点线为4/32参考份额。Broad score与targeted mass定义不同，不比较两者绝对数值高低；不在每个checkpoint重新选头。')
    fig3, axes3 = plt.subplots(1, 2, figsize=(12,4), layout='constrained')
    for mode in MODES:
        f=read(LEGACY/mode/'dynamics_causal_trials.csv')
        for metric,style in [('ar_accuracy','-'),('trace_exact',':')]:
            if mode=='nonthinking' and metric=='trace_exact':
                continue
            s=f.loc[f.condition.eq('selected')].groupby('step')[metric].mean()
            c=f.loc[f.condition.eq('control')].groupby('step')[metric].mean()
            axes3[0].plot(s.index,100*(c-s),'o'+style,label=mode+' '+metric,color=COLORS[mode])
    transport=[]
    for folder in sorted((LEGACY/'thinking').glob('transport_step_*')):
        frame=read(folder/'transport_trials.csv'); frame['step']=int(folder.name.rsplit('_',1)[1]); transport.append(frame)
    transport=pd.concat(transport)
    for condition,style in [('value_selected','-'),('value_control','--')]:
        f=transport.loc[transport.condition.eq(condition)&transport.top_k.eq(2)].groupby('step').restoration.mean()
        axes3[1].plot(f.index,f,'o'+style,label=condition)
    axes3[0].set(xlabel='Optimizer step',ylabel='Control accuracy minus selected accuracy (pp)')
    axes3[1].set(xlabel='Optimizer step',ylabel='Patched minus damaged marker logit margin')
    for ax in axes3:
        ax.axhline(0,color='gray',lw=.8);ax.grid(alpha=.2);ax.legend(fontsize=8)
    image+=fig(fig3,'dynamics_causal_transport.png','图5b｜同一11个checkpoint、每条件相同100输入。左：Top-2选中头相对三个等层对照的额外损伤，纵轴为control正确率减selected正确率（百分点）；实线为最终答案，点线为Thinking trace exact，正值表示选中头损伤更大。右：Thinking对应source的V恢复后相对损坏baseline的marker logit margin增益，实线选中Top-2，虚线同层对照均值。没有归一化小分母，也没有在训练中重新选头。')
    return image


def continuation_section(registry):
    root = LEGACY/'continuation'
    plan = json.loads((root/'plan.json').read_text())
    selected = json.loads((root/'selected_layers.json').read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    assert plan['discovery_prompts']==20 and plan['confirmation_prompts']==10
    pairs = read(root/'frozen_pairs.csv')
    assert set(pairs.loc[pairs.split.eq('confirmation'), 'prompt_sha256']) == set(registry.loc[registry.split.eq('confirmation') & registry['count'].eq(10),'key'])
    rows, selection_rows = [], []
    for scope in ['item_end_w1', 'item_span_w2']:
        info = selected['scopes'][scope]
        for row in info['layer_summaries']:
            selection_rows.append(dict(scope=scope, selected_layer=info['selected_layer'], **row))
        if info['selected_layer'] is None:
            continue
        trials = read(root/scope/'rollout_trials.csv')
        for metric in ['donor_marker_adoption','donor_continuation_adoption','donor_prefix_h2','donor_prefix_h3','donor_prefix_h4']:
            f = trials.loc[trials[metric].notna()].copy()
            if metric == 'donor_marker_adoption':
                f = f.loc[f.successor_identity_distinct]
            f['key'] = f.pair_id
            f['block'] = f.prompt_sha256.map(registry.set_index('key').block)
            d = f.loc[f.condition.eq('full_donor_patch')]
            c = f.loc[f.condition.eq('self_patch')]
            o = f.loc[f.condition.str.startswith('full_norm_orthogonal')]
            def promptmean(x):
                return x.groupby('prompt_sha256')[metric].mean().mean()
            rows.append({'scope': scope, 'layer': info['selected_layer'], 'metric': metric,
                'eligible_pairs': len(d), 'eligible_prompts': d.prompt_sha256.nunique(),
                'self': promptmean(c), 'donor': promptmean(d), 'orthogonal': promptmean(o),
                'donor_minus_orth_pp_95CI': contrast(d,o,metric,scale=100) if len(d) else 'not identifiable'})
    return pd.DataFrame(rows), pd.DataFrame(selection_rows), manifest


def continuation_plot(frame, selection):
    figure, axes = plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    for scope,f in selection.groupby('scope'):
        axes[0].plot(f.layer,f.median_prompt_mean,'o-',label=scope)
        axes[0].axvline(f.selected_layer.iloc[0],color='#888',ls=':',lw=.8)
    axes[0].set(xlabel='Patched post-block layer',ylabel='Discovery donor/receiver log-odds shift',xticks=[1,2,3,4])
    for scope,style in [('item_span_w2','-'),('item_end_w1',':')]:
        f=frame.loc[frame.scope.eq(scope)]
        x=np.arange(len(f))
        axes[1].plot(x,100*f.donor,'o'+style,label=scope+' donor')
        axes[1].plot(x,100*f.orthogonal,'s'+style,label=scope+' orth.')
        values=np.array([[float(t) for t in re.findall(r'-?\d+(?:\.\d+)?',s)] for s in f.donor_minus_orth_pp_95CI])
        offset=.05 if scope=='item_span_w2' else -.05
        axes[2].errorbar(x+offset,values[:,0],yerr=[np.maximum(0,values[:,0]-values[:,1]),np.maximum(0,values[:,2]-values[:,0])],fmt='o',capsize=3,label=scope)
    for ax in axes[1:]:
        ax.set(xticks=np.arange(5),xticklabels=['Next','q<=4','h=2','h=3','h=4'],xlabel='Eligible continuation endpoint')
    axes[1].set_ylabel('Donor-prefix adoption (%)');axes[2].set_ylabel('Donor minus norm control (pp)')
    axes[2].axhline(0,color='gray',lw=.8)
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    return fig(figure,'progress_continuation.png','图3｜左：20个discovery prompts的可识别pair选择层，横轴post-block层，纵轴donor/receiver logodds变化的prompt平均后中位数；竖虚线为冻结选层L1。中：统一10-prompt confirmation子集中的donor与等范数对照前缀adoption；右：两者配对百分点差及prompt-clustered 95%区间。横轴五个指标依次含26、37、30、26、20个可识别pairs，均来自8个prompts；各指标分母不同，连线仅辅助阅读。')


def build(output):
    output = output.resolve()
    ASSETS.mkdir(parents=True, exist_ok=True)
    for root in [ALIGN, LEGACY, EXTRA]:
        assert json.loads((root/'manifest.json').read_text())['status']=='complete', root
    assert json.loads((LEGACY/'unified_sample_audit.json').read_text())['status']=='passed'
    registry = read(ALIGN/'input_registry.csv')
    assert len(registry)==300 and registry.key.nunique()==300
    expected = set(registry.loc[registry.split.eq('confirmation'),'key'])
    trials = pd.concat([read(ALIGN/m/'trials.csv') for m in MODES])
    ablation = pd.concat([read(ALIGN/m/'ablation.csv') for m in MODES])
    for (_,_,_,_), f in ablation.groupby(['mode','arm','top_k','repeat']):
        assert set(f.key)==expected and len(f)==100
    for mode in MODES:
        for family in ['source','answer_source']:
            for _,f in trials.loc[trials['mode'].eq(mode)&trials.family.eq(family)].groupby(['arm','layer'],dropna=False):
                assert set(f.key)==expected and len(f)==100
    image1, behavior, abl_effects, bycount = behavior_and_ablation(ablation)
    clean_geometry = pd.concat([read(LEGACY/m/'geometry/clean_layer_metrics.csv') for m in MODES])
    selections = pd.concat([read(LEGACY/m/'geometry/clean_selections.csv') for m in MODES])
    selected_geometry = selections.loc[selections.selector.eq('ncc_balanced_accuracy')]
    cloud = pd.concat([read(LEGACY/m/'geometry/projection_cloud.csv') for m in MODES])
    cloud['sample'] = cloud.prompt_sha256.map({k:i for i,k in enumerate(registry.key)})
    widget = geometry_projection_widget(cloud).replace('<option value="0">L0 · embedding output</option>','')
    probe = pd.concat([read(LEGACY/m/'geometry/frozen_probe_trials.csv') for m in MODES])
    # Balanced accuracy weights the ten labels equally, not the more numerous low-k occurrences.
    probe_summary = probe.groupby(['mode','endpoint','layer','condition','top_k','repeat','occurrence']).ncc_correct.mean().groupby(['mode','endpoint','layer','condition','top_k','repeat']).mean().reset_index()
    probe_selected=[]
    for m in MODES:
        depths=json.loads((LEGACY/m/'geometry/frozen_depths.json').read_text())
        for ep,l in depths.items():
            f=probe_summary.loc[probe_summary.endpoint.eq(ep)&probe_summary.layer.eq(l)]
            probe_selected.append(f)
    factorial=read(LEGACY/'thinking/factorial_trials.csv')
    transport=read(LEGACY/'thinking/transport_trials.csv')
    transport_summary=transport.groupby(['condition','top_k','repeat']).agg(prompts=('key','nunique'),clean_margin=('clean_margin','mean'),damaged_margin=('corrupt_margin','mean'),patched_margin=('margin','mean'),restoration=('restoration','mean')).reset_index()
    source=trials.loc[trials.family.eq('source')].groupby(['mode','arm','layer'],dropna=False)[['accuracy','margin','expected_abs_error','running_centroid_distance_l1','running_centroid_distance_l2']].mean().reset_index()
    answer_source=trials.loc[trials.family.eq('answer_source')].groupby(['mode','arm'])[['accuracy','margin','expected_abs_error','count_probability_mass']].mean().reset_index()
    answer=trials.loc[trials.family.eq('answer')].copy()
    answer['donor_adoption']=np.where(answer.offset.notna(), (answer.predicted_count==answer['count']+answer.offset).astype(float), np.nan)
    answer_summary=answer.groupby(['mode','arm','layer'])[['accuracy','donor_adoption','margin']].mean().reset_index()
    native_edges = np.array([(int(n),int(n+o)) in {(1,2),(2,1),(5,6),(6,5)} if pd.notna(o) else False for n,o in zip(answer['count'],answer.offset)])
    donor_native = answer.loc[answer.arm.eq('adjacent_donor') & native_edges]
    native_summary=donor_native.groupby(['mode','layer']).agg(pairs=('key','size'),donor_adoption=('donor_adoption','mean'),receiver_accuracy=('accuracy','mean')).reset_index()
    bridge=trials.loc[trials.family.eq('terminal_bridge')].groupby(['arm','scope'],dropna=False)[['accuracy','margin']].mean().reset_index()
    relay=trials.loc[trials.family.eq('terminal_relay')].copy()
    relay_summary=relay.groupby(['arm','reset'])[['accuracy','margin']].mean().reset_index()
    serial=trials.loc[trials.family.eq('serial')].groupby(['source_layer','source_restored','retrieval','late'])[['accuracy','margin','retrieval_centroid_distance_l1','answer_centroid_distance_l4']].mean().reset_index()
    source_next=read(EXTRA/'thinking/source_next_trials.csv')
    source_next_summary=source_next.groupby(['arm','ordinary_control']).agg(prompts=('key','nunique'),nonempty=('blank_tokens',lambda x:int((x>0).sum())),marker_accuracy=('next_marker_correct','mean'),margin=('marker_margin','mean')).reset_index()
    vector=pd.concat([read(EXTRA/m/'count_vector_trials.csv') for m in MODES])
    vector_summary=vector.groupby(['mode','arm','layer'])[['donor_count_adoption','directed_expected_shift']].mean().reset_index()
    progress, progress_selection, progress_manifest=continuation_section(registry)
    progress_image=continuation_plot(progress,progress_selection)
    images_dynamics=plot_dynamics()
    gd=pd.concat([read(EXTRA/m/'geometry_dynamics_trials.csv') for m in MODES])
    gds=gd.groupby(['endpoint','step','occurrence']).ncc_correct.mean().groupby(['endpoint','step']).mean().reset_index()
    figg, axg=plt.subplots(1,1,figsize=(9,4),layout='constrained')
    for ep,f in gds.groupby('endpoint'):
        axg.plot(f.step,100*f.ncc_correct,'o-',label=ep)
    axg.set(xlabel='Optimizer step',ylabel='Balanced NCC accuracy (%)',ylim=(0,103));axg.legend(fontsize=8);axg.grid(alpha=.2)
    geometry_dynamic_image=fig(figg,'geometry_dynamics.png','图6｜横轴为预先固定的11个训练checkpoint；纵轴为十个标签等权的confirmation NCC准确率。每个endpoint的物理层由最终discovery CV确定后固定，每个checkpoint单独用相同200个discovery输入拟合标准化、PCA16与centroids，再评相同100个confirmation输入。Running状态包含每条输入的全部occurrences。该图包含重新拟合的probe，不代表固定decoder跨checkpoint的迁移。')
    nt=behavior.loc[behavior['mode'].eq('nonthinking'),'ar_accuracy'].iloc[0]; th=behavior.loc[behavior['mode'].eq('thinking'),'ar_accuracy'].iloc[0]
    coverage=pd.DataFrame([
        ['Behavior / Top-K','200 / 100','10/count; both modes','K=1,2: 3 controls; K=4: 1','完成'],
        ['Four-endpoint geometry / post-Top-K NCC','200 / 100','1100 / 550 running states; 200 / 100 answer states','same input + occurrence keys','完成'],
        ['Source formation / restoration','200 / 100','100 per mode per condition','embedding + L1–L4, ordinary control','完成'],
        ['Answer state / count directions','200 / 100','180 directed pairs/depth; Native slice 40','self + context / 3 norm controls','完成'],
        ['NT retrieval write / serial factorial','200 / 100','100 per combination','2 source depths × 2 × 3 × 3','完成；positive mediation待检验'],
        ['Thinking source-next / answer source','200 / 100','100 per condition','empty-history support reported separately','完成'],
        ['Thinking terminal bridge / relay','200 / 100','100 / 180 pairs per condition','same damaged baseline / downstream reset','完成；bridge范围受限'],
        ['Read → separate carrier → commit','200 / 100','90 inputs with N≥2','two-token item lacks separate post-marker commit','架构不适用；不能计为复现'],
        ['Progress free continuation','20 / 10 (count-10 subset)','40 discovery / 60 confirmation pairs/scope','8 conditions; 2 scopes','完成'],
        ['Training dynamics','fixed 200 / 100','same inputs at every checkpoint','100-step attention; 11 behavior milestones','完成；single training seed'],
    ],columns=['Experiment','Discovery / confirmation prompts','Realized unit','Control / restriction','Status'])
    cfg=json.loads((DATA/'config.json').read_text())
    train=read(DATA/'tables/train_metrics.csv')
    sampling=read(DATA/'tables/training_sampling_distribution.csv')
    sampling=sampling.loc[sampling.dimension.eq('accepted_counts')].copy()
    sampling['fraction']=sampling.examples/sampling.total_training_examples
    setup=pd.DataFrame([
        ['Data','Shakespeare chars；target set=3 chars；100 sets；query first；256-char context；count1–10'],
        ['Sampler','max-entropy set×count；训练窗口重新随机排列字符；集合顺序打乱；corpus split 80/10/10'],
        ['Models','两份独立初始化、独立优化的12,658,176参数模型；4层×8heads；d512；MLP2048；RoPE base10000；384 positions'],
        ['Training','每mode 10000 steps×batch128=1.28M examples；AdamW lr3e-4，warmup500，wd.01，clip1，BF16；single seed1234'],
        ['Trace','<Think> (<Sep> marker)^N </Think> <Ans> <N> <EOS>；无显式running index；NT只有<Ans> <N> <EOS>'],
        ['Loss steps1–1500','所有非padding位置的teacher-forced next-token cross-entropy'],
        ['Loss steps1501–10000','task-output；分区归一化 count/marker/structure系数8/8/16；T权重份额25/25/50%，NT33.3/66.7%'],
        ['Readout','count1–10为atomic单token且独立输出行；其余词表输入/输出embedding tied；两mode相同；无TTT或联合mode训练'],
        ['Checkpoint','每100steps保存fp16科学snapshot；每500steps保存optimizer恢复状态；本轮不重训、不改trace'],
    ],columns=['Setting','Value'])
    # Training loss has a changing objective; never concatenate as one homogeneous risk.
    trainfig, tax=plt.subplots(1,1,figsize=(9,3.6),layout='constrained')
    losscol=next((c for c in ['train_total_loss','loss','train_loss'] if c in train),None)
    if losscol:
        for mode,f in train.groupby('mode'):
            tax.plot(f.step,f[losscol],label=mode)
        tax.axvline(1500,color='gray',ls='--'); tax.set(xlabel='Optimizer step',ylabel='Logged training loss');tax.legend();tax.grid(alpha=.2)
    loss_image=fig(trainfig,'training_loss.png','图0｜原始训练日志。横轴为optimizer step，纵轴为当时训练目标下记录的loss。1500步后loss作用域与分区归一化发生变化，因此切换前后绝对loss不应视作同一定义的连续风险；该训练日志不属于confirmation实验样本量。')
    from v58_report_narrative import render_report
    context = dict(locals())
    context.update(table_fn=table, fig_fn=fig, conclusion_fn=conclusion, ALIGN=ALIGN)
    body = render_report(context)
    css='''body{margin:0;background:#f2f4f5;color:#23303c;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1240px;margin:auto;padding:40px;background:white}h1{font-size:32px}h2{margin-top:55px;border-top:1px solid #ccd6dc;padding-top:24px}h3{margin-top:30px}.summary,.conclusion{padding:15px 20px;background:#edf5f8;border-left:4px solid #357ea2}.subtitle,.caption,figcaption{color:#596671}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px 11px;text-align:left;border-bottom:1px solid #dde3e7;vertical-align:top}th{background:#f1f5f7}tr:nth-child(even){background:#fafbfc}.table-scroll{overflow:auto;margin:16px 0;max-height:650px}figure{margin:25px 0}figure img{width:100%;height:auto}figcaption{font-size:14px;padding:8px}details{background:#f7f9fa;padding:12px;margin:16px 0}code{overflow-wrap:anywhere;font-size:13px}.geometry-panel{height:760px;width:100%}.geometry-view-title{display:flex;gap:20px;align-items:center;flex-wrap:wrap}select{padding:6px}a{color:#21749d}@media(max-width:700px){main{padding:18px}.geometry-panel{height:650px}}'''
    css += '''html{scroll-behavior:smooth}h2[id]{scroll-margin-top:20px}header{padding:14px 0 26px}.eyebrow{font-size:12px;letter-spacing:.16em;color:#357ea2}nav{background:#f5f8fa;border:1px solid #dce5ec;padding:20px 26px;border-radius:8px}nav h2{margin:0;padding:0;border:0;font-size:20px}nav ol{columns:2;column-gap:36px;list-style:none;padding:0}nav li{break-inside:avoid;margin:8px 0}nav a{text-decoration:none}.protocol{border-left:3px solid #92aabe;padding:1px 18px;background:#fafbfc;margin:18px 0}.example,.equation{background:#f3f7fa;border:1px solid #dce5ed;border-radius:6px;padding:16px 20px;margin:18px 0}.equation{font-family:Georgia,"Microsoft YaHei",serif;line-height:2;overflow-x:auto}.ledger summary{cursor:pointer;font-weight:600}pre{overflow-x:auto;background:#f4f6f8;padding:16px;border-radius:6px}.conclusion{margin:20px 0}figure img{display:block}p{overflow-wrap:break-word}@media(max-width:800px){nav ol{columns:1}h1{font-size:27px}.table-scroll{max-height:500px}}@media print{nav{display:none}main{max-width:none;padding:0}body{background:white}.table-scroll{max-height:none}h2{break-after:avoid}figure{break-inside:avoid}}'''
    output.write_text('<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NiaH Synthetic — Unified Mechanism Audit</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>',encoding='utf-8')
    for name,frame in [('coverage',coverage),('ablation_effects',abl_effects),('continuation',progress),('geometry_selected',selected_geometry),('behavior',behavior),('answer_native_slice',native_summary)]:
        frame.to_csv(ASSETS/(name+'.csv'),index=False)
    manifest={'schema_version':'v58_uniform_panel_report_v1','report':str(output),'report_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'inputs_sha256':hashlib.sha256((ALIGN/'input_registry.csv').read_bytes()).hexdigest(),'confirmation_prompts':100,
        'count10_confirmation_prompts':10,'historical_results_excluded':True,'progress_validation':progress_manifest.get('validation')}
    manifest['experiment_manifest_sha256']={str(root.relative_to(DATA)) : hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest() for root in [ALIGN,LEGACY,EXTRA]}
    manifest['ui_validation']='static syntax/data coverage and PNG visual checks passed; browser file navigation blocked; interactions not exercised'
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    if output.name=='NiaH_Synthetic_report.html':
        output.with_name('NiaH_Synthetic_report_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(manifest,indent=2,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'reports/NiaH_Synthetic_report_unified.html')
    build(p.parse_args().output)
