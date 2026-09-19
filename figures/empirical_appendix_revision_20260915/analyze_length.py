"""Fit five explicit, bounded length models without rerunning model inference."""
from pathlib import Path
import hashlib
import json
import runpy
import time

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BASE = ROOT / 'figures/empirical_section3_refit'
API = runpy.run_path(str(BASE / 'analyze.py'))
FORMS = ('phi_linear', 'phi_log', 'hazard_linear', 'hazard_log', 'reciprocal')


def values(frame, pars, form):
    n = frame.N.to_numpy(float)
    x = (frame.L.to_numpy(float)/1000-1)/99
    z = np.log(frame.L.to_numpy(float)/1000)/np.log(100)
    assert np.all((x >= 0) & (x <= 1)), 'Predeclared domain is 1k--100k'
    u = z if form.endswith('_log') else x
    h0, eta, b = pars
    baseline = np.exp(-h0-eta*n/20)
    if form.startswith('phi_'):
        t = 1-b*u
        p = baseline*t**n
        db = -baseline*n*u*t**(n-1)
    elif form.startswith('hazard_'):
        p = baseline*np.exp(-b*u*n/20)
        db = -p*u*n/20
    else:
        t = 1+b*x/20
        p = baseline*t**(-n)
        db = -p*n*x/(20*t)
    jac = np.column_stack([-p, -n*p/20, db])
    assert np.isfinite(p).all() and np.isfinite(jac).all()
    return p, jac


def fit(frame, form):
    limit = 1. if form.startswith('phi_') else 500.
    fits = []
    for b in ([.01, .15, .7] if limit == 1 else [.1, 1., 8.]):
        result = least_squares(lambda t: values(frame, t, form)[0]-frame.observed.to_numpy(),
            [.001, .1, b], jac=lambda t: values(frame, t, form)[1],
            bounds=([0., 0., 0.], [500., 500., limit]),
            ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=1500)
        fits.append(result)
    result = min(fits, key=lambda r: r.cost)
    assert result.success, (form, result.message)
    return dict(form=form, parameters=result.x.tolist(), n_parameters=3,
                optimizer_success=bool(result.success), cost=float(result.cost),
                optimality=float(result.optimality), nfev=int(result.nfev),
                boundary_mask=result.active_mask.tolist())


def predict(frame, fitted):
    p = values(frame, fitted['parameters'], fitted['form'])[0]
    assert ((0 <= p) & (p <= 1)).all()
    return p


def effective_phi(lengths, fitted):
    # Remove the model-wide baseline before extracting the per-step probability.
    frame = pd.DataFrame({'N': np.ones(len(lengths)), 'L': lengths})
    h0 = fitted['parameters'][0]
    adjusted = dict(fitted, parameters=[0., *fitted['parameters'][1:]])
    return 1-predict(frame, adjusted)


def main():
    began = time.perf_counter()
    test = pd.DataFrame({'N':[1, 3, 10, 20], 'L':[1000, 20000, 70000, 100000]})
    pars = np.array([.03, .2, .25])
    for form in FORMS:
        jac = values(test, pars, form)[1]
        numerical = np.column_stack([(values(test, pars+1e-6*np.eye(3)[i], form)[0]
            - values(test, pars-1e-6*np.eye(3)[i], form)[0])/2e-6 for i in range(3)])
        assert np.allclose(jac, numerical, atol=1e-8, rtol=1e-5)
    # Endpoint behavior, including N=1 at phi=1, must remain finite and correct.
    endpoint = values(test, [0., 0., 1.], 'phi_linear')
    assert endpoint[0][0] == 1 and endpoint[0][-1] == 0
    original, current, medians = API['load_data']()
    datasets = [('median_short', 'Median across 12 groups', medians)]
    datasets += [('original_short', model, f) for model, f in original.groupby('model')]
    for model, frame in current.groupby('model'):
        datasets += [('current_full', model, frame),
                     ('current_long', model, frame.loc[frame.L.ge(25000)])]
    fits, folds, metrics, predictions = [], [], [], []

    def record(identity, validation, frame, p):
        metrics.append(dict(**identity, validation=validation,
            **API['scores'](frame.observed, p, request_mean=identity['scope']!='median_short')))
        for row, prediction in zip(frame.itertuples(), p):
            predictions.append(dict(**identity, validation=validation, N=int(row.N),
                L=int(row.L), observed=float(row.observed), prediction=float(prediction)))

    for scope, model, whole in datasets:
        frame = whole.loc[whole['mode'].eq('native_thinking')].sort_values(['L','N']).reset_index(drop=True)
        assert frame.N.nunique() == 14
        for form in FORMS:
            identity = dict(scope=scope, model=model, mode='native_thinking', form=form)
            fitted = fit(frame, form)
            fits.append(dict(**identity, **{k:v for k,v in fitted.items() if k!='form'}))
            record(identity, 'in_sample', frame, predict(frame, fitted))
            held_predictions = np.full(len(frame), np.nan)
            for length in sorted(frame.L.unique()):
                held = frame.L.eq(length)
                fold = fit(frame.loc[~held], form)
                held_predictions[held] = predict(frame.loc[held], fold)
                folds.append(dict(**identity, held_length=int(length),
                    **{k:v for k,v in fold.items() if k!='form'}))
            assert np.isfinite(held_predictions).all()
            record(identity, 'leave_length_out', frame, held_predictions)
        print(json.dumps({'scope':scope,'model':model,'seconds':round(time.perf_counter()-began,1)}),flush=True)

    for model, whole in current.loc[current['mode'].eq('native_thinking')].groupby('model'):
        train, test = whole.loc[whole.L.le(20000)], whole.loc[whole.L.ge(25000)]
        assert len(train)==112 and len(test)==126
        for form in FORMS:
            fitted = fit(train, form)
            identity = dict(scope='short_to_long',model=model,mode='native_thinking',form=form)
            fits.append(dict(**identity, **{k:v for k,v in fitted.items() if k!='form'}))
            record(identity,'frozen_extrapolation',test,predict(test,fitted))

    pd.DataFrame(metrics).to_csv(OUT/'length_scores.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'length_predictions.csv',index=False)
    (OUT/'length_parameters.json').write_text(json.dumps(fits,indent=2),encoding='utf-8')
    (OUT/'length_fold_parameters.json').write_text(json.dumps(folds,indent=2),encoding='utf-8')
    manifest=dict(
        forms=FORMS,parameters_per_model=3,domain_tokens=[1000,100000],
        linear_basis='(L/1000-1)/99',log_basis='ln(L/1000)/ln(100)',
        parameter_scaling='eta=20*(-ln(q0)); hazard and reciprocal coefficients also scaled by 20',
        direct_phi_shape_bound=[0,1],all_14_counts_retained=True,
        raw_condition_means=True,median_treated_as_binomial=False,
        validation=['leave_length_out','frozen_short_to_long'],
        all_optimizers_succeeded=all(f['optimizer_success'] for f in fits+folds),
        jacobians_checked=True,endpoint_behavior_checked=True,full_fits=len(fits),fold_fits=len(folds),
        versions={'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        primary_analysis_manifest_sha256=hashlib.sha256((BASE/'analysis_manifest.json').read_bytes()).hexdigest(),
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in API['INPUTS']},
        elapsed_seconds=time.perf_counter()-began)
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'complete':True,'seconds':manifest['elapsed_seconds'],'full_fits':len(fits),'folds':len(folds)}),flush=True)


if __name__=='__main__':
    main()
