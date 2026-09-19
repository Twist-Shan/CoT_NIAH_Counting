"""One figure per model: shared linear/log length fits across all N."""
from pathlib import Path
import json
import time
import hashlib
import base64
import sys
from itertools import product
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_realistic_niah_v3_2_empirical_laws import (
    clip_probability, condition_fold, HEADLINE_ACCURACY,
)
ANALYSIS=ROOT/'outputs/anvil_realistic_niah_v3_3_long_context_20260906_holdout/analysis'
ASSETS=ROOT/'reports/assets/niah_empirical_all_n_length'
FOLDERS={'L_k':ANALYSIS/'v3_3_all_n_linear_length_20260909','logL':ANALYSIS/'v3_3_all_n_log_length_20260909'}
MODELS=('Gemma4-31B','Qwen3-32B')
MODES=('direct','native_thinking')
MODE_RULE={'direct':'L_k','native_thinking':'logL'}
HOLDOUT=ANALYSIS/'v3_3_regression_scan/tables/n_fixed_shared_length_metrics.csv'
SHORT_RANGE=ROOT/'outputs/anvil_realistic_niah_v3_1_20260819_formal/analysis/v3_2_n_fixed_shared_length_20260909/tables/metrics.csv'


def image_uri(path):
    return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')


def load_results():
    manifests={term:json.loads((p/'manifest.json').read_text(encoding='utf-8')) for term,p in FOLDERS.items()}
    assert manifests['L_k']['input_sha256']==manifests['logL']['input_sha256']
    assert all(m['fits']==4 and m['archived_cv_reproduced'] for m in manifests.values())
    data={name:pd.concat([pd.read_csv(folder/f'{name}.csv').assign(length_term=term) for term,folder in FOLDERS.items()],ignore_index=True) for name in ('metrics','coefficients','cell_predictions','per_N_metrics')}
    cell=data['cell_predictions'];keys=['model','mode','N','L']
    old=cell.loc[cell.length_term.eq('L_k')].sort_values(keys)
    new=cell.loc[cell.length_term.eq('logL')].sort_values(keys)
    assert old[keys].reset_index(drop=True).equals(new[keys].reset_index(drop=True))
    assert np.array_equal(old.observed.to_numpy(),new.observed.to_numpy())
    return data,manifests


def comparison_rows(metrics):
    rows=[]
    for (model,mode),group in metrics.groupby(['model','mode'],sort=False):
        g=group.set_index('length_term');lin=g.loc['L_k'];log=g.loc['logL']
        rows.append(dict(model=model,mode=mode,linear_cell_R2=lin.cell_R2,log_cell_R2=log.cell_R2,
            linear_CV_R2=lin.cell_CV_R2,log_CV_R2=log.cell_CV_R2,
            linear_CV_log_loss=lin.cv_log_loss,log_CV_log_loss=log.cv_log_loss,
            relative_log_loss_reduction=(lin.cv_log_loss-log.cv_log_loss)/lin.cv_log_loss,
            preferred='logL' if log.cv_log_loss<lin.cv_log_loss else 'L_k'))
    return pd.DataFrame(rows)


def evaluate_mode_rule(data):
    """Compare a common form per mode without changing fits, cells or folds."""
    metrics=data['metrics'].copy()
    cells=data['cell_predictions'].copy()
    assert cells.n_requests.eq(30).all()
    assert metrics.n_rows.eq(7140).all()
    assert not cells.duplicated(['model','mode','N','L','length_term']).any()
    p=clip_probability(cells.oof.to_numpy())
    cells['cv_log_loss']=-cells.observed*np.log(p)-(1-cells.observed)*np.log1p(-p)
    for key,group in cells.groupby(['model','mode','length_term']):
        saved=metrics.set_index(['model','mode','length_term']).loc[key,'cv_log_loss']
        assert np.isclose(np.average(group.cv_log_loss,weights=group.n_requests),saved,atol=1e-10,rtol=0)
    levels=tuple(sorted(cells.N.unique()));lengths=tuple(sorted(cells.L.unique()))
    cells['fold']=condition_fold(cells,levels,lengths)
    paired=cells.pivot(index=['model','mode','N','L','fold'],columns='length_term',values='cv_log_loss').reset_index()
    paired['linear_minus_log']=paired.L_k-paired.logL
    indexed=metrics.set_index(['model','mode','length_term'])
    groups=[]
    for model,mode in product(MODELS,MODES):
        lin=indexed.loc[(model,mode,'L_k')];log=indexed.loc[(model,mode,'logL')]
        selected=indexed.loc[(model,mode,MODE_RULE[mode])]
        best=min(lin.cv_log_loss,log.cv_log_loss)
        groups.append(dict(model=model,mode=mode,rule_form=MODE_RULE[mode],
            linear_CV_log_loss=lin.cv_log_loss,log_CV_log_loss=log.cv_log_loss,
            rule_CV_log_loss=selected.cv_log_loss,best_CV_log_loss=best,
            rule_cell_R2=selected.cell_R2,rule_cell_CV_R2=selected.cell_CV_R2,
            absolute_cost=selected.cv_log_loss-best,relative_cost=selected.cv_log_loss/best-1,
            n_requests=int(selected.n_rows)))
    groups=pd.DataFrame(groups)
    mode_rows=[]
    for mode in MODES:
        block=metrics.loc[metrics['mode'].eq(mode)]
        values={term:np.average(g.cv_log_loss,weights=g.n_rows) for term,g in block.groupby('length_term')}
        mode_rows.append(dict(mode=mode,linear_CV_log_loss=values['L_k'],log_CV_log_loss=values['logL'],
            preferred=min(values,key=values.get),rule_form=MODE_RULE[mode]))
    schemes=[]
    for direct,native in product(('L_k','logL'),repeat=2):
        selected=metrics.loc[(metrics['mode'].eq('direct')&metrics.length_term.eq(direct))|
            (metrics['mode'].eq('native_thinking')&metrics.length_term.eq(native))]
        schemes.append(dict(non_thinking_form=direct,native_thinking_form=native,
            mean_CV_log_loss=np.average(selected.cv_log_loss,weights=selected.n_rows),
            is_requested=direct=='L_k' and native=='logL',numeric_parameters=int(selected.n_parameters.sum())))
    schemes=pd.DataFrame(schemes)
    oracle=np.average(groups.best_CV_log_loss,weights=groups.n_requests)
    schemes['absolute_cost_vs_group_best']=schemes.mean_CV_log_loss-oracle
    schemes['relative_cost_vs_group_best']=schemes.mean_CV_log_loss/oracle-1
    schemes['group_best_CV_log_loss']=oracle
    holdout=pd.read_csv(HOLDOUT)
    holdout=holdout.loc[holdout.outcome_family.eq(HEADLINE_ACCURACY)&
        holdout.evaluation_scheme.eq('fit_v3_1_score_v3_3_holdout')&
        holdout.length_term.isin(['L_k','logL'])].copy()
    assert len(holdout)==8 and holdout.n_rows.eq(3780).all()
    holdout=holdout.pivot(index=['model_label','prompt_mode'],columns='length_term',values='holdout_log_loss').reset_index()
    holdout.columns.name=None
    holdout=holdout.rename(columns={'model_label':'model','prompt_mode':'mode','L_k':'linear_holdout_log_loss','logL':'log_holdout_log_loss'})
    short=pd.read_csv(SHORT_RANGE)
    short=short.loc[short.outcome_family.eq(HEADLINE_ACCURACY)&short.prompt_mode.isin(MODES)&short.length_term.isin(['L_k','logL'])]
    assert len(short)==48
    short=short.pivot(index=['comparison_slot','prompt_mode'],columns='length_term',values='primary_loss').reset_index()
    short['log_preferred']=short.logL<short.L_k
    return dict(groups=groups,mode_summary=pd.DataFrame(mode_rows),schemes=schemes,
        paired_cells=paired,folds=paired.groupby(['model','mode','fold'],as_index=False)[['L_k','logL','linear_minus_log']].mean(),
        per_N=paired.groupby(['model','mode','N'],as_index=False)[['L_k','logL','linear_minus_log']].mean(),
        per_L=paired.groupby(['model','mode','L'],as_index=False)[['L_k','logL','linear_minus_log']].mean(),
        frozen_holdout=holdout,short_range=short)


def draw_mode_rule(model,data):
    """A requested-form display; model-specific winners remain in the comparison."""
    levels=sorted(data['cell_predictions'].N.unique())
    colors=plt.colormaps['plasma_r'](np.linspace(.12,.95,len(levels)))
    fig,axes=plt.subplots(1,2,figsize=(13.7,4.8),sharex=True,sharey=True)
    plotted=[]
    for ax,mode in zip(axes,MODES):
        term=MODE_RULE[mode]
        coefs=data['coefficients'];metrics=data['metrics'];cells=data['cell_predictions']
        c=coefs.loc[coefs.model.eq(model)&coefs['mode'].eq(mode)&coefs.length_term.eq(term)].set_index('term').estimate
        metric=metrics.loc[metrics.model.eq(model)&metrics['mode'].eq(mode)&metrics.length_term.eq(term)].iloc[0]
        for n,color in zip(levels,colors):
            points=cells.loc[cells.model.eq(model)&cells['mode'].eq(mode)&cells.length_term.eq(term)&cells.N.eq(n)].sort_values('L')
            grid=np.geomspace(1000,100000,250)
            feature=grid/1000 if term=='L_k' else np.log(grid/1000)
            prediction=expit(c[f'alpha_N={n}']+c[f'beta_{term}']*feature)
            ax.plot(grid/1000,prediction,color=color,lw=1.9,ls='-' if term=='L_k' else '--')
            ax.scatter(points.L/1000,points.observed,s=15,marker='o' if mode=='direct' else '^',facecolor='none',edgecolor=color,lw=.7,alpha=.52)
            plotted.extend(dict(model=model,mode=mode,length_term=term,N=int(n),L=float(l),prediction=float(p)) for l,p in zip(grid,prediction))
        label='Non-thinking | Linear L' if mode=='direct' else 'Native-thinking | Log length ln L'
        ax.set_title(f'{label}\nCell R² = {metric.cell_R2:.3f}  ·  CV log loss = {metric.cv_log_loss:.3f}',loc='left',fontsize=11.5,pad=10)
        ax.set_xscale('log');ax.set_xlim(.94,106);ax.set_ylim(-.025,1.025)
        ax.set_xticks([1,2,5,10,20,50,100]);ax.set_xticklabels(['1','2','5','10','20','50','100'])
        ax.set_yticks([0,.25,.5,.75,1]);ax.grid(alpha=.15)
        ax.axvline(20,color='#8E99AA',ls=':',lw=.8);ax.spines[['top','right']].set_visible(False)
        ax.set_xlabel('Passage length (thousand tokens; log scale)')
    axes[0].set_ylabel('Probability of a correct count')
    cax=fig.add_axes([.932,.23,.015,.50]);cmap=ListedColormap(colors)
    norm=BoundaryNorm(np.arange(len(levels)+1)-.5,len(levels))
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=np.arange(len(levels)))
    bar.ax.set_yticklabels([str(n) for n in levels]);bar.set_label('Target count N',labelpad=8)
    fig.suptitle(f'{model}: a common length form for each mode',x=.065,ha='left',fontsize=18,fontweight='bold')
    fig.text(.065,.90,'logit p = alpha_N + beta f(L) | all 14 N retained | a separate slope for each model and mode',fontsize=10,color='#526173')
    fig.text(.065,.055,'Open markers: observed (30 requests/cell) · curves: full-data fits · dotted vertical: 20k',fontsize=9,color='#526173')
    note='Qwen Non-thinking: this display uses linear L; log L has 2.15% lower CV log loss.' if model=='Qwen3-32B' else 'For Gemma, the displayed forms also minimize CV log loss within each mode.'
    fig.text(.065,.014,note,fontsize=9.5,color='#7B4B19' if model=='Qwen3-32B' else '#526173')
    fig.subplots_adjust(left=.065,right=.90,top=.75,bottom=.19,wspace=.17)
    for ext in ('png','pdf'):fig.savefig(ASSETS/f'{model}_mode_length_rule.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return plotted


def mode_rule_section(data,audit):
    schemes=audit['schemes'];chosen=schemes.loc[schemes.is_requested].iloc[0]
    rows=[]
    for _,r in audit['groups'].iterrows():
        rows.append({'模型':r.model,'模式':'Non-thinking' if r['mode']=='direct' else 'Native-thinking',
            '统一形式':'线性 L' if r.rule_form=='L_k' else '对数 ln L','R²':f'{r.rule_cell_R2:.3f}',
            '预测误差':f'{r.rule_CV_log_loss:.4f}','比本组最优形式多出的误差':f'{100*r.relative_cost:.2f}%'})
    def table(rows):
        return pd.DataFrame(rows).to_html(index=False,classes='data-table',border=0)
    figures=''
    for number,model in enumerate(MODELS,3):
        figures+=f'<figure><img src="{image_uri(ASSETS/f"{model}_mode_length_rule.png")}" alt="{model} 采用 Non-thinking 线性和 Native-thinking 对数的统一形式"><figcaption><strong>图 P{number}</strong><span>{model}。左图指定 Non-thinking 使用线性 L，右图指定 Native-thinking 使用对数 ln L；每个 N 各有一个截距，每种模式共用一个长度系数。横轴是原文长度，采用对数刻度；纵轴是正确率；颜色表示 N，空心点为观测正确率，实线/虚线为两种模式的完整数据拟合。包含全部 14 个 N、1k–100k 和每个条件 30 次请求，未绘制误差条。竖直点线表示 20k。R² 越大表示观测点整体贴合越好，CV log loss 越小表示留出预测误差越低。Qwen 左图采用便于统一描述的线性形式，其最优形式仍是对数。</span></figcaption></figure>'
    scheme_rows=[]
    for _,r in schemes.iterrows():
        scheme_rows.append({'Non-thinking':'线性 L' if r.non_thinking_form=='L_k' else '对数 ln L',
            'Native-thinking':'线性 L' if r.native_thinking_form=='L_k' else '对数 ln L','平均预测误差':f'{r.mean_CV_log_loss:.4f}'})
    holdout=[]
    for _,r in audit['frozen_holdout'].iterrows():
        holdout.append({'模型':r.model,'模式':'Non-thinking' if r['mode']=='direct' else 'Native-thinking',
            '线性：长程预测误差':f'{r.linear_holdout_log_loss:.4f}','对数：长程预测误差':f'{r.log_holdout_log_loss:.4f}'})
    q=audit['groups'].loc[audit['groups'].model.eq('Qwen3-32B')&audit['groups']['mode'].eq('direct')].iloc[0]
    return f'''<div id="mode-length-rule"><h4>两种模式能否各用一种统一形式？</h4>
<p><strong>在这两个模型、1k–100k 的整体拟合中，可以将“Non-thinking 用线性、Native-thinking 用对数”作为统一近似。</strong>两模型等权平均后，这一组合在四种统一方案中预测误差最低。平均误差为 {chosen.mean_CV_log_loss:.4f}；每组分别选最优形式时为 {chosen.group_best_CV_log_loss:.4f}，统一后增加 {100*chosen.relative_cost_vs_group_best:.2f}%。</p>
<p>这里的预测误差是交叉验证 log loss，越小越好。例如从 0.30 降到 0.27 表示该误差降低 10%，不表示准确率提高 10 个百分点（这是说明性示例）。</p>
<div class="table-wrap">{table(rows)}</div>
<p><strong>Qwen Non-thinking 仍是例外。</strong>它在线性形式下的误差是 {q.linear_CV_log_loss:.4f}，对数形式是 {q.log_CV_log_loss:.4f}。选择线性会比本组最优形式多出 {100*q.relative_cost:.2f}% 的误差；对数在五个验证折中都更好。因此，目前可以保留一个简洁的统一描述，但“Non-thinking 每个模型都更适合线性”还没有得到支持。</p>
<p>下面每个模型一张图，按统一方案展示。原来的两种形式比较和逐个 N 的详细图保留在后面的折叠区。</p>
{figures}
<p><strong>适用范围：</strong>上述结论来自两模型在整个 1k–100k 范围内重新拟合。用 1k–20k 拟合后直接预测 25k–100k 时，Qwen Non-thinking 的线性误差为 0.9930，对数为 0.5641，差距明显增大。原 V3.2 的 12 个比较槽中，Non-thinking 有 8 个更适合对数。因此，统一近似的适用范围需要保留，尚不能作为所有模型或长程外推的共同规律。</p>
<details><summary>统一方案比较、短程拟合后的长程预测与验证细节</summary>
<p>所有方案复用同一批数据、同一组 N 截距、同一五折条件切分和已经保存的八组拟合。两个模型与两种模式仍分别估计系数，每组 15 个系数，总计 60 个；统一的是长度形式。四组请求数相同，合并请求误差等于四组等权平均。这里是看到候选结果后的探索性比较，尚未进行独立数据确认。</p>
<div class="table-wrap">{table(scheme_rows)}</div>
<p>以下为原有冻结短程系数的外推结果，仅用于说明适用范围；每组在 25k–100k 评价 3,780 次请求。两种形式使用相同数据，误差越小越好。</p>
<div class="table-wrap">{table(holdout)}</div>
<p>逐个 N、逐个 L 和五个验证折的配对误差保存在 mode_rule_per_N.csv、mode_rule_per_L.csv 和 mode_rule_folds.csv。五折训练集相互重叠，这里的方向一致性不作五次独立实验或显著性检验解释。统一主图来自完整数据拟合，不能替代外推验证。</p></details></div>'''


def draw_model(model,data):
    cells=data['cell_predictions'];coefs=data['coefficients'];metrics=data['metrics']
    levels=sorted(cells.N.unique());colors=plt.colormaps['plasma_r'](np.linspace(.12,.95,len(levels)))
    fig,axes=plt.subplots(2,2,figsize=(13.7,8.6),sharex=True,sharey=True)
    plotted=[]
    for row,term in enumerate(('L_k','logL')):
        for col,mode in enumerate(MODES):
            ax=axes[row,col]
            c=coefs.loc[coefs.model.eq(model)&coefs['mode'].eq(mode)&coefs.length_term.eq(term)].set_index('term').estimate
            metric=metrics.loc[metrics.model.eq(model)&metrics['mode'].eq(mode)&metrics.length_term.eq(term)].iloc[0]
            for n,color in zip(levels,colors):
                points=cells.loc[cells.model.eq(model)&cells['mode'].eq(mode)&cells.length_term.eq(term)&cells.N.eq(n)].sort_values('L')
                grid=np.geomspace(1000,100000,250);feature=grid/1000 if term=='L_k' else np.log(grid/1000)
                p=expit(c[f'alpha_N={n}']+c[f'beta_{term}']*feature)
                ax.plot(grid/1000,p,color=color,linewidth=1.7,linestyle='-' if term=='L_k' else '--',alpha=.94)
                ax.scatter(points.L/1000,points.observed,s=14,marker='o' if mode=='direct' else '^',facecolor='none',edgecolor=color,linewidth=.7,alpha=.5,zorder=3)
                plotted.extend(dict(model=model,mode=mode,length_term=term,N=int(n),L=float(l),prediction=float(v)) for l,v in zip(grid,p))
            label='Non-thinking' if mode=='direct' else 'Native-thinking'
            form='Linear L' if term=='L_k' else 'Log length ln L'
            ax.set_title(f'{label} | {form}\nCell R² = {metric.cell_R2:.3f}  ·  CV D² = {metric.cv_d2_vs_global_intercept:.3f}',loc='left',fontsize=11.2,pad=9)
            ax.set_xscale('log');ax.set_xlim(.94,106);ax.set_ylim(-.025,1.025)
            ax.set_xticks([1,2,5,10,20,50,100]);ax.set_xticklabels(['1','2','5','10','20','50','100'])
            ax.set_yticks([0,.25,.5,.75,1]);ax.grid(alpha=.15)
            ax.axvline(20,color='#8E99AA',linestyle=':',linewidth=.8)
            ax.spines[['top','right']].set_visible(False)
            if col==0:ax.set_ylabel('Probability of a correct count')
            if row==1:ax.set_xlabel('Passage length (thousand tokens; log scale)')
    cax=fig.add_axes([.93,.17,.016,.62]);cmap=ListedColormap(colors);norm=BoundaryNorm(np.arange(len(levels)+1)-.5,len(levels))
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=np.arange(len(levels)))
    bar.ax.set_yticklabels([str(n) for n in levels]);bar.set_label('Target count N',labelpad=8)
    fig.suptitle(f'{model}: linear vs log length across all N',x=.065,ha='left',fontsize=18,fontweight='bold')
    fig.text(.065,.925,'logit p = alpha_N + beta f(L) | one shared slope per mode and form | 1k–100k',fontsize=10.5,color='#526173')
    fig.text(.065,.014,'Open markers: observed (30 requests/cell) · curves: full-data fits · dotted vertical: 20k · same axes in all panels',fontsize=9,color='#526173')
    fig.subplots_adjust(left=.065,right=.905,top=.84,bottom=.09,wspace=.15,hspace=.34)
    for ext in ('png','pdf'):fig.savefig(ASSETS/f'{model}_length_comparison.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    return plotted


def build_section():
    data,_=load_results();compare=comparison_rows(data['metrics'])
    audit=evaluate_mode_rule(data)
    rows=[]
    for _,r in compare.iterrows():
        rows.append({'模型':r.model,'模式':'Non-thinking' if r['mode']=='direct' else 'Native-thinking',
            '线性 L：R²':f'{r.linear_cell_R2:.3f}','对数 ln L：R²':f'{r.log_cell_R2:.3f}',
            '留出预测更好的形式':'对数 ln L' if r.preferred=='logL' else '线性 L'})
    table=pd.DataFrame(rows).to_html(index=False,classes='data-table',border=0)
    figures=''
    details=''
    for number,model in enumerate(MODELS,3):
        figures+=f'<figure><img src="{image_uri(ASSETS/f"{model}_length_comparison.png")}" alt="{model} 的线性与对数长度回归，包含全部 N 和两种模式"><figcaption><strong>图 P{number} 补充比较</strong><span>{model}，每个模型一张图。左列为 Non-thinking，右列为 Native-thinking；上行为线性 L，下行为对数 ln L。颜色表示 N，空心点是观测正确率，线是完整数据拟合。四个面板都用相同的对数横轴，范围为 1k–100k。竖直点线标出 20k。R² 衡量图中观测点的整体贴合，CV D² 衡量留出条件上的预测。</span></figcaption></figure>'
        for term,suffix in [('L_k',''),('logL','_logL')]:
            label='线性 L' if term=='L_k' else '对数 ln L'
            details+=f'<figure><img src="{image_uri(ASSETS/f"{model}_all_N{suffix}.png")}" alt="{model} {label} 各个 N 的详细曲线"><figcaption>{model} · {label}。每个小图固定 N，橙色实线/圆点为 Non-thinking，紫色虚线/三角为 Native-thinking；原文长度使用线性刻度，纵轴是正确率，点为观测、线为拟合。</figcaption></figure>'
    numeric=data['per_N_metrics'].to_html(index=False,classes='data-table',border=0,float_format=lambda x:f'{x:.4f}',na_rep='未定义')
    coeff=data['coefficients'][['model','mode','length_term','term','estimate','ci95_low','ci95_high']].to_html(index=False,classes='data-table',border=0,float_format=lambda x:f'{x:.5f}',na_rep='不可用')
    return r'''<div id="all-n-linear-length"><h3>全部 N：比较线性 L 和对数 ln L</h3>
<p>保留全部 14 个 N，每个 N 有自己的起点。我们分别尝试一个共同的线性长度系数，和一个共同的对数长度系数。两个模型、两种模式分别拟合，没有交叉项。每组、每种形式均为 14 个截距加 1 个长度系数。</p>
<p>数据范围是 1k–100k，共 28,560 次请求。这里使用全部 N；前面的 N=10 图是单独检验，结果需要分别理解。</p>
<div class="table-wrap">@@TABLE@@</div>
<p><strong>Native-thinking：</strong>两个模型都更适合对数长度；Gemma 的 R² 从 0.907 提高到 0.949，Qwen 从 0.646 提高到 0.794。留出条件上的预测也支持这一方向。<strong>Non-thinking：</strong>Gemma 更适合线性长度，Qwen 略偏向对数长度，仍没有统一形式。</p>
@@MODE_RULE@@
<details><summary>每个模型的线性与对数对比图</summary>@@FIGURES@@</details>
<p>整体 R² 同时包含 N 和 L 的解释能力。即使整体分数较高，也可能有个别 N 拟合较差。例如 Qwen Non-thinking 的 N=10 在长端回升，两种共同负斜率都无法完整解释。</p>
<details><summary>逐个 N 的详细图</summary>@@DETAILS@@</details>
<details><summary>系数、各 N 指标与计算方式</summary>
<div class="math-block">\[\operatorname{logit}p(N,L)=\alpha_N+\beta f(L),\qquad f(L)\in\{L/1000,\ln(L/1000)\}.\]</div>
<p>共享的是 logit 概率上的长度系数，概率图中的曲线不必平行。两种形式采用同一 Bernoulli GLM、同一五折留条件规则、同一批样本；全部 N 保留。用较低的 CV log loss 比较两种形式，不进行新的多候选筛选。曲线使用全数据拟合；当前比较不代表 100k 以外的外推。</p>
<p>整体 R² 使用每组全部 238 个条件；各 N 的 R² 使用该 N 的 17 个长度。负值表示不如该组观测均值；零方差时标为未定义。系数区间使用 HC3 95% 区间，未校正多重比较。</p>
<div class="table-wrap">@@NUMERIC@@</div><div class="table-wrap">@@COEFF@@</div>
<p>复现：python -s scripts/analyze_niah_all_n_linear_length.py --length-term logL；python -s scripts/build_niah_all_n_length_comparison.py；python -s scripts/build_niah_empirical_n_fixed_addendum.py。线性版本不带 --length-term 参数。比较数据、图片与清单在 reports/assets/niah_empirical_all_n_length。</p></details></div>'''.replace('@@TABLE@@',table).replace('@@MODE_RULE@@',mode_rule_section(data,audit)).replace('@@FIGURES@@',figures).replace('@@DETAILS@@',details).replace('@@NUMERIC@@',numeric).replace('@@COEFF@@',coeff)


def main():
    start=time.perf_counter();ASSETS.mkdir(parents=True,exist_ok=True)
    data,manifests=load_results();compare=comparison_rows(data['metrics'])
    audit=evaluate_mode_rule(data)
    compare.to_csv(ASSETS/'comparison_summary.csv',index=False)
    data['metrics'].to_csv(ASSETS/'comparison_metrics.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    plotted=[]
    for model in MODELS:plotted.extend(draw_model(model,data))
    pd.DataFrame(plotted).to_csv(ASSETS/'comparison_plotted_predictions.csv.gz',index=False,compression='gzip')
    rule_plotted=[]
    for model in MODELS:rule_plotted.extend(draw_mode_rule(model,data))
    pd.DataFrame(rule_plotted).to_csv(ASSETS/'mode_rule_plotted_predictions.csv.gz',index=False,compression='gzip')
    for name,frame in audit.items():frame.to_csv(ASSETS/f'mode_rule_{name}.csv',index=False)
    inputs=[p/name for p in FOLDERS.values() for name in ('manifest.json','metrics.csv','coefficients.csv','cell_predictions.csv','per_N_metrics.csv')]
    inputs.extend([HOLDOUT,SHORT_RANGE])
    selected=audit['schemes'].loc[audit['schemes'].is_requested].iloc[0]
    summary=dict(common_mode_form=MODE_RULE,mean_CV_log_loss=float(selected.mean_CV_log_loss),
        group_best_CV_log_loss=float(selected.group_best_CV_log_loss),
        relative_cost_vs_group_best=float(selected.relative_cost_vs_group_best),
        is_best_common_scheme=bool(selected.mean_CV_log_loss==audit['schemes'].mean_CV_log_loss.min()),
        inference='exploratory comparison of saved out-of-fold predictions; no independent confirmation',
        refits=0,bootstrap=0)
    (ASSETS/'comparison_manifest.json').write_text(json.dumps(dict(fits=8,requests=28560,unique_conditions=952,
        figures=4,comparison_figures=2,mode_rule_figures=2,mode_rule=summary,
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        elapsed_seconds=time.perf_counter()-start),indent=2),encoding='utf-8')
    print(compare.to_string(index=False))
    print(audit['schemes'].to_string(index=False))


if __name__=='__main__':main()
