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




if __name__ == '__main__':
    main()
