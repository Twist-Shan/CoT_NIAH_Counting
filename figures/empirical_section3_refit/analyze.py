"""Re-evaluate the two Section 3 hypotheses on the displayed data.

Run: python -s figures/empirical_section3_refit/analyze.py
No inference, smoothing, empirical-law selection, or manuscript writes occur here.
See ANALYSIS_DESIGN.md for the exploratory analysis plan and interpretation.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = ROOT / 'figures/empirical_law_1x4'
SHORT = ROOT / ('realistic/outputs/'
    'realistic_niah_v3_1_20260819_formal/analysis/'
    'v3_2_inverse_n_candidate_extension/tables/cell_outcomes.csv.gz')
RERUN = ROOT / ('realistic/outputs/'
    'qwen3_32b_yarn_off_20260913/final_results/cell_summary.csv')
NT_FORMS = ('gaussian_sqrt', 'gaussian_linear', 'gaussian_power')
T_FORMS = ('interaction_linear', 'interaction_log', 'additive_linear',
           'additive_log', 'interaction_free_length')
INPUTS = (SHORT, RERUN, FIG/'long_context_observations.csv',
          FIG/'observed_count_panel_quantiles.csv', FIG/'build_manifest.json')


def scores(y, p, *, request_mean=True):
    y, p = np.asarray(y, float), np.asarray(p, float)
    residual = p-y
    sst = np.sum((y-y.mean())**2)
    result = dict(n_cells=len(y), r2=float(1-np.sum(residual**2)/sst) if sst else None,
                  rmse_pp=float(100*np.sqrt(np.mean(residual**2))),
                  mae_pp=float(100*np.mean(np.abs(residual))),
                  bias_pp=float(100*residual.mean()),
                  outside_probability_range=int(np.sum((p < -1e-10) | (p > 1+1e-10))))
    if request_mean:
        # Clipping is for the secondary numerical score only, never prediction.
        q = np.clip(p, 1e-9, 1-1e-9)
        result['log_loss'] = float(np.mean(-y*np.log(q)-(1-y)*np.log1p(-q)))
        result['request_brier'] = float(np.mean((p-y)**2 + y*(1-y)))
    return result


def gaussian_values(frame, pars, lengths, gamma):
    ix = pd.Index(lengths).get_indexer(frame.L)
    assert (ix >= 0).all()
    exponent = pars[-1] if gamma is None else gamma
    logn = np.log(frame.N.to_numpy(float))
    z = np.exp(pars[ix]-exponent*logn)
    p = erf(z/np.sqrt(2))
    derivative = np.sqrt(2/np.pi)*z*np.exp(-z*z/2)
    jac = np.zeros((len(frame), len(pars)))
    jac[np.arange(len(frame)), ix] = derivative
    if gamma is None:
        jac[:, -1] = -derivative*logn
    return p, jac


def fit_gaussian(frame, form):
    lengths = sorted(frame.L.unique().tolist())
    gamma = {'gaussian_sqrt': .5, 'gaussian_linear': 1., 'gaussian_power': None}[form]
    starts = [gamma] if gamma is not None else [.5, 1., 2., 3.]
    fits = []
    y = frame.observed.to_numpy(float)
    for start in starts:
        logk = []
        for length in lengths:
            rows = frame.loc[frame.L.eq(length)]
            mid = rows.iloc[np.argmin(np.abs(rows.observed.to_numpy()-.5))].N
            logk.append(np.log(.6744897501960817)+start*np.log(mid))
        initial = np.r_[logk, start] if gamma is None else np.asarray(logk)
        lower = np.r_[np.full(len(lengths), -8.), .05] if gamma is None else np.full(len(lengths), -8.)
        upper = np.r_[np.full(len(lengths), 40.), 12.] if gamma is None else np.full(len(lengths), 20.)
        fit = least_squares(lambda t: gaussian_values(frame, t, lengths, gamma)[0]-y,
            initial, jac=lambda t: gaussian_values(frame, t, lengths, gamma)[1],
            bounds=(lower, upper), max_nfev=1500, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        fits.append(fit)
    best = min(fits, key=lambda f: f.cost)
    assert best.success, (form, best.message)
    return dict(form=form, parameters=best.x.tolist(), lengths=lengths,
                gamma=float(best.x[-1] if gamma is None else gamma),
                gamma_free=gamma is None, n_parameters=len(best.x),
                optimizer_success=bool(best.success), cost=float(best.cost),
                optimality=float(best.optimality), nfev=int(best.nfev),
                boundary_count=int(np.count_nonzero(best.active_mask)))


def hazard_design(frame, form, lengths=None):
    n = frame.N.to_numpy(float)/20
    if form == 'interaction_free_length':
        assert lengths is not None
        ix = pd.Index(lengths).get_indexer(frame.L)
        assert (ix >= 0).all(), 'Flexible per-length model cannot predict an unseen length'
        x = np.zeros((len(frame), 1+len(lengths)))
        x[:, 0] = 1
        x[np.arange(len(frame)), 1+ix] = n
        return x
    length_k = frame.L.to_numpy(float)/1000
    u = (length_k-1)/19 if form.endswith('_linear') else np.log(length_k)/np.log(20)
    assert (u >= -1e-12).all()
    return np.column_stack([np.ones(len(frame)), n,
                             n*u if form.startswith('interaction_') else u])


def fit_hazard(frame, form):
    lengths = sorted(frame.L.unique().tolist())
    x = hazard_design(frame, form, lengths)
    y = frame.observed.to_numpy(float)
    fits = []
    for start in (.05, .5, 2.):
        initial = np.full(x.shape[1], start)
        initial[0] = .001
        fit = least_squares(lambda t: np.exp(-x@t)-y, initial,
            jac=lambda t: -np.exp(-x@t)[:, None]*x,
            bounds=(np.zeros(x.shape[1]), np.full(x.shape[1], 500.)),
            max_nfev=1000, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        fits.append(fit)
    best = min(fits, key=lambda f: f.cost)
    assert best.success, (form, best.message)
    return dict(form=form, parameters=best.x.tolist(), lengths=lengths,
                n_parameters=len(best.x), optimizer_success=bool(best.success),
                cost=float(best.cost), optimality=float(best.optimality),
                nfev=int(best.nfev), boundary_count=int(np.count_nonzero(best.active_mask)))


def fit_model(frame, form):
    return fit_gaussian(frame, form) if form.startswith('gaussian_') else fit_hazard(frame, form)


def predict(frame, fit):
    if fit['form'].startswith('gaussian_'):
        p = gaussian_values(frame, np.asarray(fit['parameters']), fit['lengths'],
                            None if fit['gamma_free'] else fit['gamma'])[0]
    else:
        p = np.exp(-hazard_design(frame, fit['form'], fit['lengths'])@np.asarray(fit['parameters']))
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return p


def check_jacobians():
    # Numerical differentiation checks the gradients used by the optimizer.
    frame = pd.DataFrame({'N': [1, 5, 20, 3], 'L': [1000, 1000, 20000, 20000]})
    for gamma, pars in [(1., np.array([1., 2.])), (None, np.array([1., 2., 1.3]))]:
        _, jac = gaussian_values(frame, pars, [1000, 20000], gamma)
        numeric = np.column_stack([
            (gaussian_values(frame, pars+np.eye(len(pars))[j]*1e-5, [1000, 20000], gamma)[0]
             - gaussian_values(frame, pars-np.eye(len(pars))[j]*1e-5, [1000, 20000], gamma)[0])/2e-5
            for j in range(len(pars))])
        assert np.allclose(jac, numeric, atol=1e-8, rtol=1e-5)
    for form in T_FORMS:
        x = hazard_design(frame, form, [1000, 20000])
        pars = np.full(x.shape[1], .3)
        jac = -np.exp(-x@pars)[:, None]*x
        numeric = np.column_stack([(np.exp(-x@(pars+np.eye(len(pars))[j]*1e-5))
            -np.exp(-x@(pars-np.eye(len(pars))[j]*1e-5)))/2e-5 for j in range(len(pars))])
        assert np.allclose(jac, numeric, atol=1e-8, rtol=1e-5)


def load_data():
    original = pd.read_csv(SHORT)
    original = original.loc[original.prompt_mode.isin(['direct', 'native_thinking'])].copy()
    assert len(original) == 2688 and original.n_total.eq(30).all()
    assert np.allclose(original.parsed_exact_accuracy, original.n_correct/30)
    original = original.rename(columns={'comparison_slot': 'model', 'prompt_mode': 'mode',
        'parsed_exact_accuracy': 'observed', 'n_total': 'n_requests'})
    original['source'] = 'original_12group'
    current = pd.read_csv(FIG/'long_context_observations.csv')
    assert len(current) == 952 and current.n_requests.eq(30).all()
    assert current.n_seeds.eq(30).all() and not current.duplicated(['model', 'mode', 'N', 'L']).any()
    official = pd.read_csv(RERUN).rename(columns={'mode': 'mode'})
    qwen = current.loc[current.model.eq('Qwen3-32B')].merge(official, on=['mode', 'N', 'L'], validate='1:1')
    assert len(qwen) == 476 and np.allclose(qwen.observed, qwen.exact_correct/30)
    assert qwen.batch.eq('qwen_yarn_off').all()
    current['source'] = current.batch
    medians = original.groupby(['mode', 'N', 'L'])['observed'].median().reset_index()
    old = pd.read_csv(FIG/'observed_count_panel_quantiles.csv')
    check = medians.merge(old, on=['mode', 'N', 'L'], validate='1:1')
    assert len(check) == 224 and np.allclose(check.observed, check.observed_median, atol=1e-12)
    medians['model'] = 'Median across 12 groups'
    medians['source'] = 'original_12group_median'
    # No n_requests is invented for the cross-model median.
    return original, current, medians


def main():
    start = time.perf_counter()
    check_jacobians()
    original, current, medians = load_data()
    datasets = [('median_short', 'Median across 12 groups', medians)]
    datasets += [('original_short', model, frame) for model, frame in original.groupby('model')]
    for model, frame in current.groupby('model'):
        datasets.append(('current_full', model, frame))
        datasets.append(('current_long', model, frame.loc[frame.L.ge(25000)]))
    all_scores, predictions, parameters, fold_fits, linear_diagnostics = [], [], [], [], []
    for scope, model, whole in datasets:
        print(json.dumps({'scope': scope, 'model': model, 'elapsed': round(time.perf_counter()-start, 1)}), flush=True)
        for mode, frame in whole.groupby('mode'):
            frame = frame.sort_values(['L', 'N']).reset_index(drop=True)
            frame = frame[['N', 'L', 'observed', 'source']].copy()
            forms = NT_FORMS if mode == 'direct' else T_FORMS
            request_mean = scope != 'median_short'
            for form in forms:
                identity = dict(scope=scope, model=model, mode=mode, form=form)
                fit = fit_model(frame, form)
                parameters.append(dict(**identity, **{k:v for k,v in fit.items() if k != 'form'}))
                p = predict(frame, fit)
                all_scores.append(dict(**identity, validation='in_sample', **scores(frame.observed, p, request_mean=request_mean)))
                for validation, held_column in [('leave_count_out', 'N'), ('leave_length_out', 'L')]:
                    if held_column == 'L' and (mode == 'direct' or form == 'interaction_free_length'):
                        continue
                    held_predictions = np.full(len(frame), np.nan)
                    baseline_predictions = np.full(len(frame), np.nan)
                    for value in sorted(frame[held_column].unique()):
                        held = frame[held_column].eq(value).to_numpy()
                        train = frame.loc[~held]
                        fitted = fit_model(train, form)
                        held_predictions[held] = predict(frame.loc[held], fitted)
                        baseline_predictions[held] = train.observed.mean()
                        fold_fits.append(dict(**identity, validation=validation, held_value=int(value),
                            **{k:v for k,v in fitted.items() if k != 'form'}))
                    assert np.isfinite(held_predictions).all()
                    metric = scores(frame.observed, held_predictions, request_mean=request_mean)
                    metric['mse_reduction_vs_training_mean'] = float(1-np.mean((frame.observed-held_predictions)**2)/np.mean((frame.observed-baseline_predictions)**2))
                    all_scores.append(dict(**identity, validation=validation, **metric))
                    for row, value in zip(frame.itertuples(), held_predictions):
                        predictions.append(dict(**identity, validation=validation, N=int(row.N), L=int(row.L),
                                                observed=float(row.observed), prediction=float(value), source=row.source))
                for row, value in zip(frame.itertuples(), p):
                    predictions.append(dict(**identity, validation='in_sample', N=int(row.N), L=int(row.L),
                                            observed=float(row.observed), prediction=float(value), source=row.source))
            if mode == 'native_thinking':
                # Unconstrained linear error is a separate diagnostic, not a
                # probability model whose outputs may be silently clipped.
                for form in ('interaction_linear', 'interaction_log'):
                    x = hazard_design(frame, form)
                    target = 1-frame.observed.to_numpy()
                    coef = np.linalg.lstsq(x, target, rcond=None)[0]
                    raw = 1-x@coef
                    linear_diagnostics.append(dict(scope=scope, model=model, form=form,
                        coefficients=coef.tolist(), **scores(frame.observed, raw, request_mean=False)))
        # Checkpoint results make progress reviewable during a long local run.
        pd.DataFrame(all_scores).to_csv(OUT/'fit_scores.csv', index=False)
        (OUT/'full_fit_parameters.json').write_text(json.dumps(parameters, indent=2), encoding='utf-8')

    extrapolation = []
    for model, whole in current.loc[current['mode'].eq('native_thinking')].groupby('model'):
        train = whole.loc[whole.L.le(20000)].copy()
        test = whole.loc[whole.L.ge(25000)].copy()
        assert len(train) == 112 and len(test) == 126
        for form in T_FORMS[:4]:
            fit = fit_model(train, form)
            p = predict(test, fit)
            extrapolation.append(dict(model=model, form=form, **scores(test.observed, p),
                                       training_parameters=fit['parameters']))
            for row, value in zip(test.itertuples(), p):
                predictions.append(dict(scope='short_to_long', model=model, mode='native_thinking',
                    form=form, validation='frozen_extrapolation', N=int(row.N), L=int(row.L),
                    observed=float(row.observed), prediction=float(value), source=row.source))

    original.groupby(['model', 'mode'])['observed'].mean().unstack().to_csv(OUT/'original_group_accuracy.csv')
    pd.DataFrame(predictions).to_csv(OUT/'predictions.csv', index=False)
    pd.DataFrame(extrapolation).to_csv(OUT/'extrapolation_scores.csv', index=False)
    pd.DataFrame(linear_diagnostics).to_csv(OUT/'linear_error_diagnostics.csv', index=False)
    (OUT/'fold_parameters.json').write_text(json.dumps(fold_fits, indent=2), encoding='utf-8')
    manifest = dict(objective='Nonlinear least squares on unsmoothed cell means, or separately on model medians',
        validation=['leave_count_out', 'leave_length_out where length parameters transfer', 'short_to_long for current Thinking data'],
        count_grid=sorted(original.N.unique().tolist()), candidates={'Non-thinking': NT_FORMS, 'Thinking': T_FORMS},
        retained_N1=True, altered_main_figure=False, iid_confidence_intervals_used=False,
        gamma_bounds=[.05, 12], log_kappa_bounds=[-8,40], hazard_bounds=[0,500],
        linear_length_basis='(L/1000-1)/19', log_length_basis='ln(L/1000)/ln(20)', count_scale='N/20',
        original_short_cells=len(original), current_full_cells=len(current),
        qwen_current_cells_verified_against_official=476,
        cross_model_medians_verified=224, analytic_jacobians_numerically_checked=True,
        full_fits=len(parameters), fold_fits=len(fold_fits),
        all_optimizers_succeeded=all(p['optimizer_success'] for p in parameters+fold_fits),
        versions={'numpy':np.__version__, 'pandas':pd.__version__, 'scipy':scipy.__version__},
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        elapsed_seconds=time.perf_counter()-start)
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'complete':True,'full_fits':len(parameters),'fold_fits':len(fold_fits),
                      'elapsed_seconds':manifest['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
