"""Build and insert the reproducible piecewise / additive-law report extension.

python -s scripts/build_niah_empirical_piecewise_addendum.py
Reads completed analysis tables; never modifies frozen data or prior sections.
"""
from __future__ import annotations
import base64
import html
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import analyze_realistic_niah_v3_2_piecewise_n3 as analysis

ROOT=Path(__file__).resolve().parents[1]
OUT=analysis.ANALYSIS/'v3_2_piecewise_n3_20260909'
START='<!-- PIECEWISE_N3_ADDENDUM_START -->'
END='<!-- PIECEWISE_N3_ADDENDUM_END -->'
MODE={'direct':'Non-thinking','native_thinking':'Native-thinking','enumeration_index':'Index enumeration','enumeration_bullet':'Bullet enumeration'}
TARGET={analysis.ext.MAE_FAMILY:'Trimmed MAE',analysis.core.BIAS_FAMILY:'Trimmed bias'}
VARIANT={'full_selected':'原版 · 全候选','full_additive':'原版 · 无交叉项选式','tail_selected':'分段 · 全候选','tail_additive':'分段 · 无交叉项选式','full_N_L':'原版 · 固定 N+Lk','tail_N_L':'分段 · 固定 N+Lk'}


def formula(cid):
    if cid=='intercept': return 'a'
    terms=cid.split('__')
    names={'N':'N','L_k':'Lk','logN':'ln N','logL':'ln Lk','invN':'1/N','N_x_L_k':'N·Lk','N_x_logL':'N·ln Lk','logN_x_L_k':'ln N·Lk','logN_x_logL':'ln N·ln Lk','invN_x_L_k':'Lk/N','invN_x_logL':'ln Lk/N'}
    return 'a + '+' + '.join(f'{chr(98+i)}·({names[t]})' for i,t in enumerate(terms))


def prepare():
    start=time.perf_counter()
    core,ext=analysis.core,analysis.ext
    cfg=ext.load_parent_frozen_config(core.DEFAULT_CONFIG,core.DEFAULT_FREEZE)
    levels=(tuple(cfg['immutable_input']['N_levels']),tuple(cfg['immutable_input']['L_levels']))
    candidates=core.load_candidates(cfg)+ext.load_inverse_candidates(ext.DEFAULT_EXTENSION_CONFIG)
    registry={c.id:c for c in candidates}
    additive=tuple(c for c in candidates if c.interaction is None)
    old_summary=pd.concat([pd.read_csv(analysis.ANALYSIS/'v3_2_trimmed_count_error_extension/tables/mae_mode_candidate_summary.csv'),pd.read_csv(analysis.ANALYSIS/'v3_2_inverse_n_candidate_extension/tables/mode_candidate_summary.csv')])
    old_summary=old_summary[old_summary.outcome_family.isin(analysis.FAMILIES)]
    tail_summary=pd.read_csv(OUT/'tables/tail_candidate_summary.csv')
    selections={}
    for prefix,summary in [('full',old_summary),('tail',tail_summary)]:
        selections[prefix+'_selected']=core.select_all(summary,candidates)
        selections[prefix+'_additive']=core.select_all(summary[summary.candidate.isin([c.id for c in additive])],additive)
    cells=pd.read_csv(OUT/'tables/source_cells.csv.gz')
    cells=cells[cells.bias_law_eligible].copy()
    rows,coefficients,predictions=[],[],[]
    for (slot,mode),block in cells.groupby(['comparison_slot','prompt_mode']):
        for family in analysis.FAMILIES:
            for variant in VARIANT:
                tail=variant.startswith('tail')
                cid='N__L_k' if variant.endswith('N_L') else selections[variant].loc[lambda x:x.outcome_family.eq(family)&x.prompt_mode.eq(mode),'selected_candidate'].item()
                candidate=registry[cid]
                oof,fitted=analysis.predict(block,family,candidate,levels,tail_only=tail,plateau='constant' if tail else None)
                for domain,mask in [('all',np.ones(len(block),bool)),('N_ge4',block.N.gt(3).to_numpy())]:
                    rows.append(dict(comparison_slot=slot,prompt_mode=mode,outcome_family=family,variant=variant,candidate=cid,domain=domain,**core.continuous_metrics(block.loc[mask,family],oof[mask])))
                train=block[block.N.gt(3)] if tail else block
                fit=core.fit_ols(train[family].to_numpy(),core.design_matrix(train,candidate),robust=True)
                for i,term in enumerate(('intercept',*candidate.terms)):
                    coefficients.append(dict(comparison_slot=slot,prompt_mode=mode,outcome_family=family,variant=variant,candidate=cid,term=term,estimate=fit.params[i],ci95_low=fit.conf_int()[i,0],ci95_high=fit.conf_int()[i,1]))
                if tail:
                    coefficients.append(dict(comparison_slot=slot,prompt_mode=mode,outcome_family=family,variant=variant,candidate=cid,term='low_N_constant',estimate=block.loc[block.N.le(3),family].mean(),ci95_low=np.nan,ci95_high=np.nan))
                p=block[['comparison_slot','prompt_mode','N','L']].copy()
                p['outcome_family'],p['variant'],p['fitted'],p['observed'],p['oof']=family,variant,fitted,block[family],oof
                predictions.append(p)
    detail=pd.DataFrame(rows)
    summary=detail.groupby(['outcome_family','prompt_mode','variant','candidate','domain']).agg(median_r2=('r2','median'),q25_r2=('r2',lambda s:s.quantile(.25)),valid_r2=('r2','count'),median_prediction_mae=('mae','median')).reset_index()
    for name,table in [('report_version_model_comparison',detail),('report_version_summary',summary),('report_version_coefficients',pd.DataFrame(coefficients)),('report_version_predictions',pd.concat(predictions))]:
        table.to_csv(OUT/'tables'/f'{name}.csv',index=False)
    selection=pd.concat([s.assign(version=k) for k,s in selections.items()])
    selection.to_csv(OUT/'tables/report_version_selections.csv',index=False)
    # Verify full-domain original scores against the existing published tables.
    oldmetrics=pd.concat([pd.read_csv(analysis.ANALYSIS/'v3_2_trimmed_count_error_extension/tables/mae_selected_model_fit_metrics.csv'),pd.read_csv(analysis.ANALYSIS/'v3_2_inverse_n_candidate_extension/tables/selected_model_fit_metrics.csv')])
    checked=detail[detail.variant.eq('full_selected')&detail.domain.eq('all')].merge(oldmetrics,on=['comparison_slot','prompt_mode','outcome_family','candidate'],validate='one_to_one')
    assert len(checked)==96
    np.testing.assert_allclose(checked.r2,checked.cv_r2,atol=1e-9,equal_nan=True)
    np.testing.assert_allclose(checked.mae,checked.cv_mae,atol=1e-9)
    core.write_json(OUT/'report_extension_analysis_manifest.json',dict(elapsed_seconds=time.perf_counter()-start,baseline_checks=96,source_cells_sha256=core.file_sha256(OUT/'tables/source_cells.csv.gz'),candidates=len(candidates),additive_candidates=len(additive),condition_folds='Unchanged original grid indices',estimands='Unchanged 10% symmetric trimming',exploratory=True))
    plot(summary)


def plot(summary):
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    variants=['full_selected','full_additive','tail_selected','tail_additive']
    labels=['Original\nall candidates','Original\nadditive','Piecewise\nall candidates','Piecewise\nadditive']
    for row,family in enumerate(analysis.FAMILIES):
        for col,mode in enumerate(['direct','native_thinking']):
            ax=axes[row,col]
            for domain,offset,color,label in [('all',-.19,'#34699A','Full grid CV'),('N_ge4',.19,'#DB9251','N >= 4 CV')]:
                b=summary[summary.outcome_family.eq(family)&summary.prompt_mode.eq(mode)&summary.domain.eq(domain)].set_index('variant')
                vals=b.loc[variants,'median_r2'].to_numpy()
                ax.bar(np.arange(4)+offset,vals,width=.36,label=label,color=color)
                for x,y in zip(np.arange(4)+offset,vals): ax.text(x,y+.014,f'{y:.3f}',ha='center',fontsize=8)
            ax.set_xticks(np.arange(4),labels,fontsize=9)
            ax.set_ylim(0,1.05); ax.set_ylabel('Median held-condition R²')
            ax.set_title(f'{MODE[mode]} / {TARGET[family]}');ax.grid(axis='y',alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=2,fontsize=9,frameon=False)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/'figures/additive_version_comparison.png',dpi=170);plt.close(fig)


def table(frame):
    return '<div class="table-wrap">'+frame.to_html(index=False,escape=True,float_format=lambda v:f'{v:.4f}',border=0,na_rep='—')+'</div>'


def build_piecewise_main_figures():
    """Reuse the original main-figure renderer and evaluate all 12 slot laws."""
    import build_niah_empirical_law_v3_2_report as main
    main.set_plot_style()
    cells=pd.read_csv(OUT/'tables/source_cells.csv.gz')
    coefficients=pd.read_csv(OUT/'tables/report_version_coefficients.csv')
    scores=pd.read_csv(OUT/'tables/report_version_summary.csv')
    saved=pd.read_csv(OUT/'tables/report_version_predictions.csv')
    ns=sorted(cells.N.unique());ls=sorted(cells.L.unique())
    modes=('direct','native_thinking')
    audit=[]
    for variant in ['tail_additive','tail_selected']:
        for family in analysis.FAMILIES:
            coef=coefficients[coefficients.variant.eq(variant)&coefficients.outcome_family.eq(family)]
            laws=scores[scores.variant.eq(variant)&scores.outcome_family.eq(family)&scores.domain.eq('N_ge4')].rename(columns={'candidate':'selected_candidate','median_r2':'median_primary_score'})
            fitted={}
            for mode in modes:
                f=main.predictions_by_slot(coef[coef.term.ne('low_N_constant')],mode,family,ns,ls)
                assert f.shape==(12,len(ns),len(ls))
                low=coef[coef.prompt_mode.eq(mode)&coef.term.eq('low_N_constant')].set_index('comparison_slot').reindex(main.SLOT_ORDER).estimate.to_numpy()
                assert np.isfinite(low).all()
                f[:,np.asarray(ns)<=3,:]=low[:,None,None]
                fitted[mode]=f
                # Predictions on every eligible cell must reproduce saved fits.
                b=saved[saved.variant.eq(variant)&saved.outcome_family.eq(family)&saved.prompt_mode.eq(mode)]
                ni={n:i for i,n in enumerate(ns)};li={l:i for i,l in enumerate(ls)};si={s:i for i,s in enumerate(main.SLOT_ORDER)}
                values=np.asarray([f[si[r.comparison_slot],ni[r.N],li[r.L]] for r in b.itertuples()])
                np.testing.assert_allclose(values,b.fitted,atol=1e-9,rtol=1e-9)
                audit.append(dict(variant=variant,family=family,mode=mode,checked_cells=len(b)))
            short='mae' if family==analysis.ext.MAE_FAMILY else 'bias'
            for axis in ['N','L']:
                name=f'piecewise_main_{variant}_{short}_by_{axis}.png'
                main.plot_aggregate_law_curves(cells,coef,laws,family=family,column=family,modes=modes,x_axis=axis,
                    path=OUT/'figures'/name,title=f'Piecewise {TARGET[family]} | '+('additive laws' if variant=='tail_additive' else 'all-candidate selected laws'),
                    ylabel='10% trimmed conditional MAE (count units; symlog)' if short=='mae' else '10% trimmed signed bias (count units)',
                    fitted_override=fitted,piecewise_breakpoint=3)
    analysis.core.write_json(OUT/'piecewise_main_figure_validation.json',dict(passed=True,prediction_checks=audit,aggregation='Q25/Q50/Q75 across 12 slot-specific laws; raw cell observations match original main figure',boundary='No connecting segment from N=3 to N=4',figures=8))


def main_figures_html():
    groups=[]
    for variant in ['tail_additive','tail_selected']:
        content=[]
        for metric in ['mae','bias']:
            for axis in ['N','L']:
                path=OUT/'figures'/f'piecewise_main_{variant}_{metric}_by_{axis}.png'
                data=base64.b64encode(path.read_bytes()).decode()
                label=f'{variant} {metric} by {axis}'
                content.append(f'<figure><img src="data:image/png;base64,{data}" alt="Piecewise main figure: {label}"><figcaption><strong>FIGURE E · {html.escape(metric.upper())} / {axis}</strong><span>分段回归主图：'+('无交叉项' if variant=='tail_additive' else '全候选选式')+f'。横轴 {axis}；颜色'+('固定全部 8 个 L。' if axis=='N' else '固定全部 14 个 N。')+'左右面板为 Non-thinking 与 Native-thinking。空心点和误差条为原始 cell 指标跨 12 个模型的 Q50 与 [Q25,Q75]；线和色带为 12 套分段方程预测的 Q50 与 [Q25,Q75]。IQR 表示模型异质性，未作为置信区间。N≤3 为模型特异常数，N≥4 为回归后段；N=3 与 N=4 之间不连接。标题 R² 来自 N≥4 条件 CV，曲线来自全数据拟合。MAE 负预测保留。</span></figcaption></figure>')
        group=''.join(content)
        if variant=='tail_selected': group='<details><summary>同样式的含交叉项分段主图（MAE / bias；N / L 两轴）</summary>'+group+'</details>'
        groups.append(group)
    return '<h3>E.3a 分段回归主图：与正文相同的汇总方式</h3><p>优先展示无交叉项版本，含交叉项版本放在下方展开栏中。沿用正文黄→紫配色、完整 N/L 网格和双模式面板。对 12 套模型方程先计算预测，再汇总分位数；未先平均系数，也未将不同 L 混合成一条曲线。低 N 常数不依赖 L，因此 L 横轴上的 N≤3 拟合线会重合。</p>'+''.join(groups)


def build_section():
    summary=pd.read_csv(OUT/'tables/report_version_summary.csv')
    coefficients=pd.read_csv(OUT/'tables/report_version_coefficients.csv')
    def readable(frame):
        f=frame.copy();f['candidate']=f.candidate.map(formula)
        f['prompt_mode']=f.prompt_mode.map(MODE);f['outcome_family']=f.outcome_family.map(TARGET);f['variant']=f.variant.map(VARIANT)
        return f.rename(columns={'outcome_family':'目标','prompt_mode':'模式','variant':'版本','candidate':'公式（含截距）','median_r2':'CV R² 中位数','q25_r2':'CV R² Q25','valid_r2':'R² 有效模型数','median_prediction_mae':'预测 MAE 中位数'})
    headline=summary[summary.prompt_mode.isin(['direct','native_thinking'])&~summary.variant.str.endswith('N_L')]
    high=readable(headline[headline.domain.eq('N_ge4')].drop(columns='domain'))
    full=readable(headline[headline.domain.eq('all')].drop(columns='domain'))
    additive=readable(summary[summary.variant.isin(['full_additive','tail_additive'])].copy())
    additive['domain']=additive.domain.map({'all':'全部 N','N_ge4':'仅 N≥4'})
    coeff=coefficients[coefficients.variant.isin(['full_additive','tail_additive'])].copy()
    coeff['prompt_mode']=coeff.prompt_mode.map(MODE);coeff['outcome_family']=coeff.outcome_family.map(TARGET);coeff['variant']=coeff.variant.map(VARIANT)
    coeff=coeff.rename(columns={'comparison_slot':'模型','prompt_mode':'模式','outcome_family':'目标','variant':'版本','term':'项','estimate':'系数','ci95_low':'HC3 95% 下限','ci95_high':'HC3 95% 上限'}).drop(columns='candidate')
    image=base64.b64encode((OUT/'figures/additive_version_comparison.png').read_bytes()).decode()
    piece=base64.b64encode((OUT/'figures/piecewise_comparison.png').read_bytes()).decode()
    main_figures=main_figures_html() if (OUT/'figures/piecewise_main_tail_additive_mae_by_N.png').exists() else ''
    return START+rf'''
<section id="piecewise-n3" class="appendix"><div class="section-head"><div class="section-no">Appendix E · 2026-09-09</div><div><h2>分段回归与无交叉项版本</h2><p class="lede">在同一批数据上比较原版全 N 回归与 N≤3 常数段 / N≥4 后段回归，并分别限制为无交叉项公式。</p></div></div>
<div class="conclusion"><strong>当前结论：</strong>前段常数设定对原生思考的 MAE 与 bias 拟合有小幅改善。为便于解释，无交叉项版本保留为明确的简化描述；在直接作答和原生思考上，它的交叉验证性能低于含交叉项版本。以下同时呈现性能代价，不将简化公式表述为更优拟合。</div>
<h3>E.1 定义、评估范围与前段设定</h3>
<p>沿用原报告的 10% 双侧截尾 conditional MAE 与 signed bias（预测数量 − N），可解析数至少 20 的 cell 等权。原版使用全部 N；新版仅用 N≥4 拟合后段，每个模型、模式分别估计系数。Lk=L/1000，所有 log 均为自然对数。</p>
<div class="math-block">\[\widehat y(N,L)=\begin{{cases}}c_0,&amp;N\le3,\\a+b\,u(N)+c\,v(L_k),&amp;N\ge4.\end{{cases}}\]</div>
<p>上式是无交叉项分段版；全候选版本允许额外项 d·u(N)·v(Lk)。c₀ 是该模型、模式低 N cell 的均值，交叉验证时只用训练折估计。未强制断点连续，也未裁剪 MAE 负预测。原生思考低 N trimmed bias 接近零，但直接作答低 N 准确率约 88.6%，因此统一设为零并非所有模式均合适。零常数敏感性结果单独保留在分析目录。</p>
<p>保持原始五折条件划分 (index(N)+index(L)) mod 5，删除低 N 时不重编号。每张表中所有版本使用完全相同的评估 cell；“全部 N”与“仅 N≥4”的 R² 分母不同，不能跨这两个评估域直接比较。R² 和预测 MAE 先按模型计算，再取模型间中位数；恒定响应的 R² 保留为缺失值。预测 MAE 衡量回归预测误差，与回归目标 trimmed MAE 是不同量。</p>
<h3>E.2 原版与分段版：相同 N≥4 评估域</h3>{table(high)}
<h3>E.3 完整分段函数：相同全域评估</h3>{table(full)}
<figure><img src="data:image/png;base64,{image}" alt="Cross-validation comparison of original and piecewise additive laws"><figcaption><strong>FIGURE E1</strong><span>横轴为原版/分段版与全候选/无交叉项选式；纵轴为 12 个模型的 CV R² 中位数。蓝色评估全部 N，橙色只评估 N≥4。同色柱之间可直接比较；选式仍依据各自训练范围的原规则。本图不显示置信区间。</span></figcaption></figure>
{main_figures}
<h3>E.4 无交叉项公式的解释与全部模式结果</h3>
<p>无交叉项版本从原有 18 个候选中筛出 12 个加性候选，沿用效应阈值、R² 容忍区间与简单性优先规则重新选式。该规则不等同于单纯最小化预测 MAE。表中的公式分别在全部 N 或 N≥4 上选择；模型间共享函数形式，系数各自估计。</p>
<p>对 a+bN+cLk，b 表示固定长度时 N 增加 1 的预测变化，c 表示固定 N 时长度增加 1000 tokens 的预测变化。对 a+bN+c ln Lk，长度倍增的预测变化为 c ln 2。加性模型假设这两种效应互不依赖，便于分别报告 N 与 L 的影响；这种解释是回归模型的数学性质，尚不能确认为因果机制。</p>
{table(additive)}
<details><summary>固定 N+Lk 的原版 / 分段版对照（不重新选择自变量变换）</summary>{table(readable(summary[summary.variant.str.endswith('N_L')]))}</details>
<details><summary>原版与分段版无交叉项：各模型系数及 HC3 95% 区间</summary><p>low_N_constant 是分段前段的描述性常数，未计算区间；后段 HC3 区间为给定所选公式的条件区间，未包含选式不确定性。</p>{table(coeff)}</details>
<details><summary>原版与分段版全候选拟合曲线</summary><figure><img src="data:image/png;base64,{piece}" alt="Piecewise fitted curves by mode"><figcaption>点和线分别为观测值、全数据拟合预测，按模型与长度等权平均；灰色背景为 N≤3。曲线用于描述，表格使用留出条件预测。由于各 N 的可用模型可能不同，聚合线可能产生折点。</figcaption></figure></details>
<h3>E.5 结论范围与复现</h3>
<p>用于简洁的经验描述时，可同时报告加性公式与性能损失。若以现有网格上的预测拟合为主要标准，直接作答与原生思考仍支持保留交叉项。枚举 bias 的共享结构较弱且不稳定，不宜推广为统一规律。</p>
<p>断点 3 由观察数据后的分析请求指定，本节属于探索性敏感性分析；条件 CV 没有在外层嵌套公式选择，尚无新独立测试集验证。全候选后段的 LOMO 结果已保存；无交叉项限制下未额外执行 LOMO。此前正文和 Appendix A–D 的数字仍对应原版全候选拟合。</p>
<details><summary>本节数据、脚本与验证</summary><p>分析目录：outputs/anvil_realistic_niah_v3_1_20260819_formal/analysis/v3_2_piecewise_n3_20260909。核心对照表：report_version_summary.csv；逐模型指标：report_version_model_comparison.csv；系数：report_version_coefficients.csv。原版 96 组模型指标已与原报告核对一致。</p><pre>python -s scripts/analyze_realistic_niah_v3_2_piecewise_n3.py
python -s scripts/build_niah_empirical_piecewise_addendum.py</pre></details>
</section>
'''+END


def inject(report):
    report=re.sub(re.escape(START)+r'.*?'+re.escape(END)+r'\n?','',report,flags=re.S)
    section=build_section()
    anchor='<section id="repro"'
    if anchor not in report: raise ValueError('Missing reproducibility section anchor')
    report=report.replace(anchor,section+'\n'+anchor,1)
    nav='<a href="#piecewise-n3">Appendix E · 分段 / 加性</a>'
    if nav not in report: report=report.replace('<a href="#repro">',nav+'<a href="#repro">',1)
    assert report.count('id="piecewise-n3"')==1
    return report


if __name__=='__main__':
    prepare()
    build_piecewise_main_figures()
    target=ROOT/'reports/NiaH_Empirical-law_report.html'
    prior=target.read_text(encoding='utf-8')
    backup=OUT/'report_before_addendum.html'
    if not backup.exists(): backup.write_text(prior,encoding='utf-8')
    updated=inject(prior)
    target.write_text(updated,encoding='utf-8')
    analysis.core.write_json(OUT/'report_extension_build_manifest.json',dict(report=str(target),sha256=analysis.core.file_sha256(target),original_backup_sha256=analysis.core.file_sha256(backup)))
    parent_manifest=ROOT/'reports/niah_empirical_law_v3_2_assets/report_build_manifest.json'
    if parent_manifest.exists():
        payload=json.loads(parent_manifest.read_text(encoding='utf-8'))
        payload['output_sha256']=analysis.core.file_sha256(target)
        payload['piecewise_additive_extension']={'date':'2026-09-09','section':'piecewise-n3','manifest':str(OUT/'report_extension_analysis_manifest.json'),'original_report_sha256':analysis.core.file_sha256(backup)}
        analysis.core.write_json(parent_manifest,payload)
    print(target)
