"""Exploratory N<=3 plateau / N>=4 regression sensitivity, without new inference.

Run from repository root: python -s scripts/analyze_realistic_niah_v3_2_piecewise_n3.py
Uses unchanged cached cell estimands, 18 candidates, original condition folds,
HC3 and selection gates. Baseline and refits are evaluated on identical cells.
The breakpoint is user-specified after observing data; this is not confirmatory.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels

import analyze_realistic_niah_v3_2_empirical_laws as core
import analyze_realistic_niah_v3_2_count_error_extension as ext

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'outputs/anvil_realistic_niah_v3_1_20260819_formal/analysis'
FAMILIES = (ext.MAE_FAMILY, core.BIAS_FAMILY)


def predict(block, family, candidate, levels, *, tail_only=False, plateau=None):
    """Full-grid OOF and descriptive full-fit predictions; no clipping."""
    x = core.design_matrix(block, candidate)
    y = block[family].to_numpy(float)
    tail = block.N.to_numpy() > 3
    folds = core.condition_fold(block, *levels)
    oof = np.full(len(block), np.nan)
    for fold in range(5):
        test = folds == fold
        train = folds != fold
        fitmask = train & tail if tail_only else train
        assert not np.any(fitmask & test)
        assert np.linalg.matrix_rank(x[fitmask]) == x.shape[1]
        beta = np.linalg.lstsq(x[fitmask], y[fitmask], rcond=None)[0]
        oof[test] = x[test] @ beta
        if plateau is not None:
            lowtrain = train & ~tail
            assert lowtrain.any()
            oof[test & ~tail] = 0.0 if plateau == 'zero' else y[lowtrain].mean()
    fitmask = tail if tail_only else np.ones(len(y), bool)
    beta = np.linalg.lstsq(x[fitmask], y[fitmask], rcond=None)[0]
    fitted = x @ beta
    if plateau is not None:
        fitted[~tail] = 0.0 if plateau == 'zero' else y[~tail].mean()
    assert np.isfinite(oof).all() and np.isfinite(fitted).all()
    return oof, fitted


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ANALYSIS / 'v3_2_piecewise_n3_20260909')
    args = parser.parse_args()
    out = args.output
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    (out / 'figures').mkdir(exist_ok=True)
    source = ANALYSIS / 'v3_2_trimmed_count_error_extension/tables/count_error_cells.csv.gz'
    config = ext.load_parent_frozen_config(core.DEFAULT_CONFIG, core.DEFAULT_FREEZE)
    levels = (tuple(config['immutable_input']['N_levels']), tuple(config['immutable_input']['L_levels']))
    candidates = core.load_candidates(config) + ext.load_inverse_candidates(ext.DEFAULT_EXTENSION_CONFIG)
    registry = {c.id: c for c in candidates}
    cells = pd.read_csv(source)
    assert len(cells) == 5376 and not cells.duplicated(['comparison_slot','prompt_mode','N','L']).any()
    assert int(cells.n_total.sum()) == 161280
    old = pd.concat([
        pd.read_csv(ANALYSIS / 'v3_2_trimmed_count_error_extension/tables/mae_selected_mode_laws.csv'),
        pd.read_csv(ANALYSIS / 'v3_2_inverse_n_candidate_extension/tables/selected_mode_laws.csv'),
    ])
    old = old.loc[old.outcome_family.isin(FAMILIES)]
    eligible = cells.loc[cells.bias_law_eligible.astype(bool)].copy()
    tail = eligible.loc[eligible.N.gt(3)]
    metrics, coefficients = [], []
    for (slot, mode), block in tail.groupby(['comparison_slot', 'prompt_mode']):
        for family in FAMILIES:
            for candidate in candidates:
                if family == core.BIAS_FAMILY:
                    m, c = core.fit_bias_candidate(block, candidate, *levels)
                else:
                    m, c = ext.fit_continuous_candidate(block, candidate, outcome_family=family,
                        outcome_column=family, n_levels=levels[0], l_levels=levels[1])
                metrics.append(dict(m, comparison_slot=slot, prompt_mode=mode))
                coefficients.extend(dict(v, comparison_slot=slot, prompt_mode=mode) for v in c)
    metrics = pd.DataFrame(metrics)
    coefficients = core.apply_coefficient_bh(pd.DataFrame(coefficients))
    summary = core.summarize_candidates(metrics, coefficients, candidates)
    selected = core.select_all(summary, candidates)
    print('Tail candidate fitting and selection complete', flush=True)
    # Reuse family-specific LOMO implementation (structure selection across slots).
    lomos = []
    for family in FAMILIES:
        lomos.append(ext.lomo_for_family(metrics.loc[metrics.outcome_family.eq(family)],
            coefficients.loc[coefficients.outcome_family.eq(family)], candidates, selected, family))
    lomo = pd.concat(lomos, ignore_index=True)
    selected = core.add_lomo_summary(selected, lomo)
    selected['evidence_reading'] = selected.apply(core.evidence_reading, axis=1)
    print('LOMO complete', flush=True)
    rows, predictions, plateau_rows = [], [], []
    for (slot, mode), block in eligible.groupby(['comparison_slot', 'prompt_mode']):
        block = block.reset_index(drop=True)
        low = block.N.le(3).to_numpy()
        for family in FAMILIES:
            old_id = old.loc[old.outcome_family.eq(family) & old.prompt_mode.eq(mode), 'selected_candidate'].item()
            new_id = selected.loc[selected.outcome_family.eq(family) & selected.prompt_mode.eq(mode), 'selected_candidate'].item()
            plateau_rows.append(dict(comparison_slot=slot, prompt_mode=mode, outcome_family=family,
                constant=block.loc[low, family].mean(), max_abs_cell=block.loc[low, family].abs().max()))
            variants = [('original',old_id,False,None),
                ('refit_fixed_zero',old_id,True,'zero'), ('refit_fixed_constant',old_id,True,'constant'),
                ('refit_selected_zero',new_id,True,'zero'), ('refit_selected_constant',new_id,True,'constant')]
            for variant, cid, tail_only, plateau in variants:
                oof, fitted = predict(block, family, registry[cid], levels, tail_only=tail_only, plateau=plateau)
                for domain, mask in [('all',np.ones(len(block),bool)),('N_ge4',~low),('N_le3',low)]:
                    m = core.continuous_metrics(block.loc[mask,family].to_numpy(), oof[mask])
                    rows.append(dict(comparison_slot=slot,prompt_mode=mode,outcome_family=family,
                        variant=variant,candidate=cid,domain=domain,n_cells=int(mask.sum()),**m))
                p = block[['comparison_slot','prompt_mode','N','L']].copy()
                p['outcome_family'],p['variant'],p['observed'],p['oof'],p['fitted'] = family,variant,block[family],oof,fitted
                predictions.append(p)
    comparisons = pd.DataFrame(rows)
    predictions = pd.concat(predictions, ignore_index=True)
    aggregate = comparisons.groupby(['outcome_family','prompt_mode','domain','variant']).agg(
        median_r2=('r2','median'),median_mae=('mae','median'),median_rmse=('rmse','median')).reset_index()
    paired = comparisons.merge(comparisons.loc[comparisons.variant.eq('original'),
        ['comparison_slot','prompt_mode','outcome_family','domain','mae','r2']],
        on=['comparison_slot','prompt_mode','outcome_family','domain'],suffixes=('','_original'),validate='many_to_one')
    paired['delta_mae'] = paired.mae - paired.mae_original
    paired['delta_r2'] = paired.r2 - paired.r2_original
    gain = paired.groupby(['outcome_family','prompt_mode','domain','variant']).agg(
        median_paired_delta_mae=('delta_mae','median'),median_paired_delta_r2=('delta_r2','median'),
        models_improved_mae=('delta_mae',lambda s:int((s < -1e-12).sum()))).reset_index()
    aggregate = aggregate.merge(gain,validate='one_to_one')
    low_summary = cells.loc[cells.N.le(3)].groupby('prompt_mode').agg(
        cells=('N','size'),requests=('n_total','sum'),correct=('n_correct','sum'),
        trimmed_mae=(ext.MAE_FAMILY,'mean'),trimmed_bias=(core.BIAS_FAMILY,'mean'),raw_mae=('conditional_mae','mean'))
    low_summary['accuracy'] = low_summary.correct / low_summary.requests
    for name, table in [('tail_candidate_metrics',metrics),('tail_candidate_coefficients',coefficients),
        ('tail_candidate_summary',summary),('tail_selected_laws',selected),('tail_lomo',lomo),
        ('model_comparison',comparisons),('comparison_summary',aggregate),('paired_comparison',paired),
        ('plateau_constants',pd.DataFrame(plateau_rows)),('low_N_diagnostics',low_summary.reset_index())]:
        table.to_csv(out / 'tables' / f'{name}.csv',index=False)
    predictions.to_csv(out / 'tables/predictions.csv.gz',index=False)
    cells.to_csv(out / 'tables/source_cells.csv.gz',index=False)
    manifest = dict(status='complete',exploratory=True,breakpoint=3,source=str(source.resolve()),
        source_sha256=core.file_sha256(source),input_request_sha256=config['immutable_input']['request_level_sha256'],
        command='python -s scripts/analyze_realistic_niah_v3_2_piecewise_n3.py',
        source_cells=len(cells),eligible_cells=len(eligible),tail_cells=len(tail),candidates=len(candidates),
        condition_folds='Original (index(N)+index(L)) mod 5; original N indices retained',
        plateau='zero or equal-cell constant per model and mode; OOF constant fitted on training low-N cells only',
        continuity_constraint=False,mae_clipping=False,selection='Original gates; HC3/BH; 18 candidates; LOMO structure selection',
        caveat='Condition CV is not nested over formula selection; no fresh independent test set or breakpoint search.',
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
        statsmodels=statsmodels.__version__,elapsed_seconds=time.perf_counter()-start)
    core.write_json(out/'analysis_manifest.json',manifest)
    compare_no_interaction(out, eligible, candidates, levels, summary)
    plot_results(predictions, out)
    write_report(out, aggregate, selected, low_summary, manifest)
    print(aggregate.loc[aggregate.domain.eq('N_ge4') & aggregate.variant.isin(['original','refit_fixed_zero','refit_selected_zero'])].to_string(index=False))
    print(f'Output: {out.resolve()}',flush=True)


def compare_no_interaction(out, cells, candidates, levels, summary):
    """Compare matched additive and interaction tails on identical OOF cells."""
    start = time.perf_counter()
    additive = tuple(c for c in candidates if c.interaction is None)
    allowed = {c.id for c in additive}
    selected = core.select_all(summary.loc[summary.candidate.isin(allowed)], additive)
    selected.to_csv(out/'tables/no_interaction_selected_laws.csv',index=False)
    registry = {c.id:c for c in candidates}
    rows, coeffs = [], []
    for (slot,mode), block in cells.groupby(['comparison_slot','prompt_mode']):
        for family in FAMILIES:
            cid = selected.loc[selected.outcome_family.eq(family)&selected.prompt_mode.eq(mode),'selected_candidate'].item()
            variants = [('N_L', 'N__L_k'), ('N_L_interaction','N__L_k__N_x_L_k'),
                ('selected_no_interaction',cid)]
            for label,candidate_id in variants:
                candidate = registry[candidate_id]
                oof, fitted = predict(block,family,candidate,levels,tail_only=True,plateau='constant')
                mask=block.N.gt(3).to_numpy()
                fit=core.fit_ols(block.loc[mask,family].to_numpy(),core.design_matrix(block.loc[mask],candidate),robust=True)
                for j,term in enumerate(('intercept',*candidate.terms)):
                    coeffs.append(dict(comparison_slot=slot,prompt_mode=mode,outcome_family=family,
                        variant=label,candidate=candidate_id,term=term,estimate=fit.params[j],
                        ci95_low=fit.conf_int()[j,0],ci95_high=fit.conf_int()[j,1]))
                for domain,evalmask in [('N_ge4',mask),('all',np.ones(len(block),bool))]:
                    rows.append(dict(comparison_slot=slot,prompt_mode=mode,outcome_family=family,
                        variant=label,candidate=candidate_id,domain=domain,
                        **core.continuous_metrics(block.loc[evalmask,family],oof[evalmask])))
    detail=pd.DataFrame(rows)
    reference=detail.loc[detail.variant.eq('N_L_interaction'),['comparison_slot','prompt_mode','outcome_family','domain','mae','r2']]
    detail=detail.merge(reference,on=['comparison_slot','prompt_mode','outcome_family','domain'],suffixes=('','_interaction'),validate='many_to_one')
    detail['delta_mae_vs_interaction']=detail.mae-detail.mae_interaction
    detail['delta_r2_vs_interaction']=detail.r2-detail.r2_interaction
    table=detail.groupby(['outcome_family','prompt_mode','domain','variant','candidate']).agg(
        median_r2=('r2','median'),valid_r2_models=('r2','count'),median_mae=('mae','median'),
        median_delta_mae_vs_interaction=('delta_mae_vs_interaction','median'),
        models_lower_mae=('delta_mae_vs_interaction',lambda s:int((s < -1e-12).sum()))).reset_index()
    detail.to_csv(out/'tables/no_interaction_model_comparison.csv',index=False)
    table.to_csv(out/'tables/no_interaction_summary.csv',index=False)
    pd.DataFrame(coeffs).to_csv(out/'tables/no_interaction_coefficients.csv',index=False)
    lines=['# 排除 N≤3 后，去掉交叉项是否改善回归？','',
        '结论：对直接作答与原生思考，去掉交叉项后，MAE 和 bias 的交叉验证拟合均变差。前段设为常数后，高 N 区间仍需要检验 N 与 L 的交互。',
        '索引与项目符号枚举的 MAE 也更支持含交叉项的候选；枚举 bias 的共享规律仍弱，不能据此推广所有模式都需要交叉项。','',
        '后段统一使用 N≥4；前段使用每个模型、模式的训练集低 N 常数。N_L=a+bN+c(L/1000)，N_L_interaction 再加 dN(L/1000)。',
        'selected_no_interaction 沿用原来的 R² 容忍区间、效应阈值及简单性优先规则，在 12 个无交叉项候选中选式；不等同于单纯使预测 MAE 最小的候选。','',
        '## N≥4 的相同留出条件','',table.loc[table.domain.eq('N_ge4')].to_markdown(index=False,floatfmt='.4f'),'',
        'R² 和预测 MAE 是模型间中位数。models_lower_mae 表示该形式相对 N_L_interaction 的预测 MAE 更低的模型数；这是配对比较，不能用两组中位数之差替代。','',
        '## 完整分段函数的全域评估','',table.loc[table.domain.eq('all')].to_markdown(index=False,floatfmt='.4f'),'',
        '## 解释与限制','',
        '分段常数解决低 N 的水平段；交叉项允许高 N 区间中 N 的斜率随 L 改变。两个设定施加不同约束，因此加入前段常数并不必然使后段交叉项冗余。',
        '无交叉项可以作为简化描述，但现有比较不支持它在直接作答与原生思考上具有更好的预测拟合。原因的机制解释尚未验证。',
        '这是固定断点的事后敏感性分析；仍使用原有非嵌套条件交叉验证，尚无新测试集验证。无交叉项限制下未额外运行 LOMO，主分析的全候选 LOMO 结果单独保存。','',
        '详细系数与 HC3 区间：tables/no_interaction_coefficients.csv；逐模型结果：tables/no_interaction_model_comparison.csv。',
        f'本比较耗时 {time.perf_counter()-start:.2f} 秒；由同一脚本自动生成。']
    (out/'no_interaction_report.md').write_text('\n'.join(lines),encoding='utf-8')
    core.write_json(out/'no_interaction_timing.json',dict(elapsed_seconds=time.perf_counter()-start))
    print(table.loc[table.domain.eq('N_ge4')].to_string(index=False),flush=True)


def plot_results(predictions, out):
    modes = ['direct','native_thinking','enumeration_index','enumeration_bullet']
    fig, axes = plt.subplots(2,4,figsize=(17,8),sharex=True)
    for row,family in enumerate(FAMILIES):
        for col,mode in enumerate(modes):
            ax=axes[row,col]
            b=predictions.loc[predictions.outcome_family.eq(family)&predictions.prompt_mode.eq(mode)]
            obs=b.loc[b.variant.eq('original')].groupby('N')['observed'].mean()
            ax.plot(obs.index,obs.values,'o',color='black',ms=4,label='Observed')
            for variant,label,color in [('original','Original','#777777'),('refit_fixed_constant','Tail, old formula','#0072B2'),('refit_selected_constant','Tail, reselected','#D55E00')]:
                s=b.loc[b.variant.eq(variant)].groupby('N').fitted.mean()
                ax.plot(s.index[s.index<=3],s[s.index<=3],color=color,lw=2)
                ax.plot(s.index[s.index>=4],s[s.index>=4],color=color,lw=2,label=label)
            ax.axvspan(1,3,color='gray',alpha=.1)
            ax.axhline(0,color='gray',lw=.5)
            ax.set_title(mode.replace('_',' '))
            ax.set_xlabel('N')
            ax.set_ylabel('Trimmed MAE' if row==0 else 'Trimmed bias')
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('N <= 3 plateau; N >= 4 refit (equal means across model slots and lengths)')
    fig.tight_layout()
    fig.savefig(out/'figures/piecewise_comparison.png',dpi=170)
    fig.savefig(out/'figures/piecewise_comparison.pdf')
    plt.close(fig)


def write_report(out, aggregate, selected, low_summary, manifest):
    lines=['# N≤3 常数段与 N≥4 回归：探索性敏感性分析','',
        '结论：排除 N≤3 后，直接作答和原生思考仍选择 N+L_k+N×L_k，原生思考的 MAE 与 bias 后段拟合小幅改善。',
        '索引枚举 MAE 在重选交互项后改善较明显；保持原式时未出现同等改善。枚举 bias 仍缺乏稳定的共享结构。',
        '前段设零并非所有模式均适用，应同时参照拟合常数版本及低 N 原始准确率。','',
        '沿用报告中的 10% 双侧截尾 conditional MAE 和 signed bias；bias = 预测数量 − N。',
        '对每个模型、模式分别拟合：g(N,L)=c（N≤3），g(N,L)=a+Σ b_j x_j(N,L)（N≥4）。',
        'c=0 和低 N 样本拟合常数均评估；未施加断点连续性约束。MAE 负预测不裁剪。',
        '仅高 N 数据参与后段回归；保持原始五折条件划分。新旧方法都在相同评估域、相同 cell 上比较。',
        '原式重拟合用于隔离排除低 N 的影响；另按原规则从 18 个候选中重选后段公式。','',
        '## 低 N 前提核验','',low_summary.to_markdown(floatfmt='.5f'),'',
        '## 同一 N≥4 评估域','',
        'R² 和回归预测 MAE 均先按模型计算，再取模型间中位数（最多 12 个）；常数响应的 R² 无定义并保留为 NaN，中位数排除 NaN。回归预测 MAE 与被解释的 trimmed MAE 是不同量。',
        aggregate.loc[aggregate.domain.eq('N_ge4') & aggregate.variant.isin(['original','refit_fixed_zero','refit_selected_zero'])].to_markdown(index=False,floatfmt='.4f'),'',
        '## 全域评估（包含前段）','',aggregate.loc[aggregate.domain.eq('all')].to_markdown(index=False,floatfmt='.4f'),'',
        '## 后段公式与留一模型结构稳定性','',
        selected[['outcome_family','prompt_mode','selected_candidate','median_primary_score','lomo_formula_stability']].to_markdown(index=False,floatfmt='.4f'),'',
        'N、L_k=L/1000、logN=ln(N)、logL=ln(L/1000)、invN=1/N；每个公式包含截距。系数和 HC3 区间见 tables/tail_candidate_coefficients.csv；低 N 常数见 tables/plateau_constants.csv。','',
        '![分段曲线](figures/piecewise_comparison.png)','',
        '图中点为观测 cell 指标，线为全样本拟合预测，均对模型和长度等权平均；灰色背景为 N≤3。图用于描述，评估指标来自 OOF 预测。','',
        '## 限制与复现','',
        '断点 3 是观察数据后指定的探索性设定。公式选择未嵌套在条件交叉验证外层，因此重选结果不等同于独立测试集性能。LOMO 衡量结构对模型集合的敏感性；不表示系数可跨模型直接迁移。',
        '截尾指标接近零不能推出原始请求完全无误；低 N 总体均值也不能证明各个模型和长度均为平段。',
        '全零响应导致 OLS 的 AIC/BIC 为负无穷，R² 无定义；这些情况未作为拟合失败忽略，选式仍使用原有 CV 和效应阈值。',
        '原始分析与原始请求未修改。没有运行新推理；结论仅适用于现有模型与 N、L 网格。','',
        '```powershell','python -s scripts/analyze_realistic_niah_v3_2_piecewise_n3.py','```','',
        f'运行时长：{manifest["elapsed_seconds"]:.1f} 秒。输入哈希、版本和过滤规则见 analysis_manifest.json。']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__ == '__main__':
    main()
