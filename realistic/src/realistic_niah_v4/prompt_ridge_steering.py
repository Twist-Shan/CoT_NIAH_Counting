"""Discovery-only ridge directions and audited one-shot prompt interventions."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .modeling import _tensor_from_output, _replace_output_tensor


def fit_direction(endpoints, counts, span_states, *, random_seed=20260910):
    x = np.asarray(endpoints, dtype=np.float32)
    y = np.asarray(counts, dtype=np.float32)
    spans = np.asarray(span_states, dtype=np.float32)
    if x.ndim != 2 or y.shape != (len(x),) or spans.ndim != 2 or spans.shape[1] != x.shape[1]:
        raise ValueError('Invalid training shapes')
    if len(x) < 33 or not all(np.isfinite(a).all() for a in (x,y,spans)):
        raise ValueError('Insufficient or nonfinite training data')
    pca = PCA(n_components=min(32,x.shape[1]), svd_solver='randomized', random_state=20260806)
    z = pca.fit_transform(x)
    scaler = StandardScaler().fit(z)
    ridge = Ridge(alpha=1.0).fit(scaler.transform(z), y)
    scaled = ridge.coef_.astype(np.float64) / scaler.scale_
    w = pca.components_.astype(np.float64).T @ scaled
    intercept = float(ridge.intercept_ - scaler.mean_ @ scaled - pca.mean_.astype(np.float64) @ w)
    norm = float(np.linalg.norm(w))
    if norm <= 1e-12:
        raise ValueError('Degenerate ridge direction')
    v = w / norm
    sigma = float(np.std(spans.astype(np.float64) @ v))
    if not np.isfinite(sigma) or sigma <= 1e-10:
        raise ValueError('Degenerate discovery projection variance')
    rng = np.random.default_rng(random_seed)
    r = rng.normal(size=len(v))
    r -= v * (r @ v)
    r /= np.linalg.norm(r)
    direct = x.astype(np.float64) @ w + intercept
    predicted = ridge.predict(scaler.transform(pca.transform(x)))
    projection_error = float(np.max(np.abs(direct-predicted)))
    if projection_error > 1e-3:
        raise RuntimeError(f'Incorrect ridge back-projection: {projection_error}')
    audit = dict(training_rows=len(x), training_span_tokens=len(spans), hidden_size=x.shape[1],
        pca_components=pca.n_components_, ridge_alpha=1.0, sigma=sigma, weight_norm=norm,
        random_cosine=float(v@r), projection_max_error=projection_error,
        training_r2=float(1-np.square(direct-y).sum()/np.square(y-y.mean()).sum()),
        training_mae=float(np.abs(direct-y).mean()), random_seed=random_seed,
        intended_endpoint_count_shift_per_beta=float(sigma*norm))
    arrays = dict(weight=w.astype(np.float32), unit=v.astype(np.float32),
        random_unit=r.astype(np.float32), intercept=np.array(intercept), sigma=np.array(sigma))
    return arrays, audit


def realized_replacement(selected, direction, magnitude):
    return (selected.float() + direction.to(device=selected.device, dtype=torch.float32)*float(magnitude)).to(selected.dtype)


def matched_random_replacement(selected, direction, target_norm, magnitude):
    """Deterministic magnitude search; never search for a favorable random direction."""
    if target_norm <= 0 or magnitude == 0:
        return selected.clone(), 0.0
    def proposal(scale):
        value = realized_replacement(selected, direction, magnitude*scale)
        norm = float(torch.linalg.vector_norm(value.float()-selected.float()))
        return value, norm
    lower, upper = 0., 2.
    value, actual = proposal(upper)
    for _ in range(12):
        if actual >= target_norm: break
        upper *= 2
        value, actual = proposal(upper)
    else: raise RuntimeError('Cannot bracket matched random norm')
    best, best_norm, best_scale = value, actual, upper
    for _ in range(24):
        scale = (lower+upper)/2
        value, actual = proposal(scale)
        if abs(actual-target_norm) < abs(best_norm-target_norm):
            best, best_norm, best_scale = value, actual, scale
        if abs(actual/target_norm-1) <= .005: break
        if actual < target_norm: lower=scale
        else: upper=scale
    if abs(best_norm/target_norm-1) > .05:
        raise RuntimeError('Realized random perturbation norm differs by >5%')
    return best, float(best_scale)


@contextmanager
def additive_span_hook(adapter, encoding, *, layer, positions, probe, beta, condition):
    positions = tuple(int(p) for p in positions)
    if not positions or len(set(positions)) != len(positions) or min(positions)<0 or max(positions)>=encoding.query_position:
        raise ValueError('Positions must be distinct prompt positions before the answer query')
    if not 0 <= layer < adapter.num_layers or condition not in ('ridge','random','noop'):
        raise ValueError('Invalid layer or condition')
    audit = {'applications':0}
    def hook(_module, _args, output):
        hidden = _tensor_from_output(output)
        if hidden.shape[1] != encoding.sequence_length:
            return output
        if hidden.shape[0] != 1: raise RuntimeError('Prefill batch must be one')
        selected = hidden[:,list(positions),:]
        v = torch.as_tensor(probe['unit'],device=hidden.device)
        w = torch.as_tensor(probe['weight'],device=hidden.device)
        magnitude = float(beta)*float(probe['sigma']) if condition!='noop' else 0.
        ridge_value = realized_replacement(selected,v,magnitude)
        target_norm = float(torch.linalg.vector_norm(ridge_value.float()-selected.float()))
        scale=1.
        if condition=='random':
            r = torch.as_tensor(probe['random_unit'],device=hidden.device)
            value,scale=matched_random_replacement(selected,r,target_norm,magnitude)
        else: value=ridge_value
        delta=value.float()-selected.float()
        actual=float(torch.linalg.vector_norm(delta))
        if not torch.isfinite(value).all(): raise RuntimeError('Nonfinite intervention')
        if condition!='noop' and beta!=0 and actual<=0:
            raise RuntimeError('Nonzero strength was rounded to a no-op')
        audit.update(applications=audit['applications']+1, target_norm=target_norm,
            realized_norm=actual, realized_norm_ratio=actual/target_norm if target_norm else 1.,
            random_magnitude_scale=scale, changed_fraction=float((delta!=0).float().mean()),
            endpoint_probe_shift=float(delta[0,-1]@w),
            mean_probe_shift=float((delta@w).mean()),
            endpoint_probe_before=float(selected[0,-1].float()@w+float(probe['intercept'])),
            relative_state_norm=actual/max(float(torch.linalg.vector_norm(selected.float())),1e-12),
            intended_probe_shift=float(magnitude*torch.linalg.vector_norm(w)))
        updated=hidden.clone()
        updated[:,list(positions),:]=value
        return _replace_output_tensor(output,updated)
    handle=adapter.layers[layer].register_forward_hook(hook)
    try: yield audit
    finally: handle.remove()
    if audit['applications']!=1:
        raise RuntimeError(f'Expected one prompt intervention: {audit}')
