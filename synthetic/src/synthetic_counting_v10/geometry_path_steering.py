from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .core import Vocab, balanced_examples, count_bin, count_prediction
from .report_followups import load_run
from .state_causal import (
    SITE_NONTHINKING_FINAL,
    SITE_THINKING_FINAL,
    SITE_THINKING_FIXED,
    _capture_residuals,
    render_site,
)
from .training import load_final_model


FINAL_SITES = (
    SITE_NONTHINKING_FINAL,
    SITE_THINKING_FINAL,
    SITE_THINKING_FIXED,
)


def centroid_chord_point(
    centroids: Mapping[int, np.ndarray], start: int, end: int, alpha: float
) -> tuple[np.ndarray, float]:
    """Interpolate directly between two count centroids."""
    progress = float(np.clip(alpha, 0.0, 1.0))
    start_value = np.asarray(centroids[int(start)], dtype=np.float32)
    end_value = np.asarray(centroids[int(end)], dtype=np.float32)
    point = start_value + progress * (end_value - start_value)
    count_coordinate = float(start) + progress * (float(end) - float(start))
    return point.astype(np.float32), count_coordinate


def centroid_polyline_point(
    centroids: Mapping[int, np.ndarray], start: int, end: int, alpha: float
) -> tuple[np.ndarray, float]:
    """Move by normalized arc length along consecutive count centroids.

    At alpha=1 this returns the same endpoint as ``centroid_chord_point``.
    Intermediate points differ when the count-centroid trajectory is curved.
    """
    progress = float(np.clip(alpha, 0.0, 1.0))
    if int(start) == int(end):
        return np.asarray(centroids[int(start)], dtype=np.float32).copy(), float(start)
    step = 1 if int(end) > int(start) else -1
    labels = list(range(int(start), int(end) + step, step))
    points = [np.asarray(centroids[label], dtype=np.float32) for label in labels]
    lengths = np.asarray(
        [float(np.linalg.norm(right - left)) for left, right in zip(points[:-1], points[1:])],
        dtype=float,
    )
    total = float(lengths.sum())
    if total <= 1e-12:
        return centroid_chord_point(centroids, start, end, progress)
    target_distance = progress * total
    traversed = 0.0
    for index, length in enumerate(lengths):
        if target_distance <= traversed + length or index == len(lengths) - 1:
            local = (target_distance - traversed) / max(float(length), 1e-12)
            local = float(np.clip(local, 0.0, 1.0))
            point = points[index] + local * (points[index + 1] - points[index])
            count_coordinate = float(labels[index]) + local * float(step)
            return point.astype(np.float32), count_coordinate
        traversed += float(length)
    return points[-1].copy(), float(end)


def _centroids_for_site_layer(
    arrays: Mapping[str, np.ndarray], site: str, layer: int, count_min: int, count_max: int
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for count in range(int(count_min), int(count_max) + 1):
        key = f"{site}__L{int(layer) + 1}__C{count}"
        if key not in arrays:
            raise KeyError(f"Missing count centroid: {key}")
        result[count] = np.asarray(arrays[key], dtype=np.float32)
    return result


def _patched_residual_batch(
    model,
    input_ids: torch.Tensor,
    layer: int,
    position: int,
    replacement_states: torch.Tensor,
) -> torch.Tensor:
    batch_size = int(replacement_states.shape[0])
    repeated = input_ids.repeat(batch_size, 1)

    def hook(_module, _args, output):
        is_tuple = isinstance(output, tuple)
        hidden = (output[0] if is_tuple else output).clone()
        hidden[:, int(position)] = replacement_states.to(hidden.device, dtype=hidden.dtype)
        return (hidden, *output[1:]) if is_tuple else hidden

    handle = model.transformer.h[int(layer)].register_forward_hook(hook)
    try:
        return model(input_ids=repeated).logits[:, int(position)].detach()
    finally:
        handle.remove()


def _regression_slope(frame: pd.DataFrame) -> tuple[float, float, float]:
    clean = frame[["intended_count_shift", "causal_expected_shift"]].dropna()
    if len(clean) < 2 or clean.intended_count_shift.nunique() < 2:
        return math.nan, math.nan, math.nan
    x = clean.intended_count_shift.to_numpy(dtype=float)
    y = clean.causal_expected_shift.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ beta
    denominator = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - prediction) ** 2).sum()) / denominator if denominator > 1e-12 else math.nan
    return float(beta[1]), float(beta[0]), r2


def summarize_geometry_path_steering(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame()
    group_columns = [
        "site",
        "mode",
        "count_bin",
        "layer",
        "method",
        "abs_offset",
        "alpha",
    ]
    summary = (
        detail.groupby(group_columns, as_index=False)
        .agg(
            n=("receiver_count", "size"),
            baseline_accuracy=("baseline_correct", "mean"),
            donor_hit=("follows_donor", "mean"),
            intended_hit=("follows_intended_integer", "mean"),
            expected_shift=("causal_expected_shift", "mean"),
            intended_shift=("intended_count_shift", "mean"),
            mean_transport_fraction=("pair_transport_fraction", "mean"),
            path_tracking_mae=("path_tracking_error", "mean"),
            donor_probability=("donor_probability", "mean"),
            intended_probability=("intended_probability", "mean"),
            mean_state_update_norm=("state_update_norm", "mean"),
            mean_curve_chord_distance=("curve_chord_state_distance", "mean"),
        )
        .sort_values(group_columns)
    )
    regression_rows: list[dict[str, Any]] = []
    regression_groups = ["site", "mode", "count_bin", "layer", "method", "abs_offset", "alpha"]
    for keys, frame in detail.groupby(regression_groups, dropna=False):
        slope, intercept, r2 = _regression_slope(frame)
        regression_rows.append(
            {
                **dict(zip(regression_groups, keys)),
                "n": len(frame),
                "transport_slope": slope,
                "transport_intercept": intercept,
                "transport_r2": r2,
                "path_tracking_mae": float(frame.path_tracking_error.mean()),
                "donor_hit": float(frame.follows_donor.mean()),
                "intended_hit": float(frame.follows_intended_integer.mean()),
            }
        )
    return summary, pd.DataFrame(regression_rows)


@torch.no_grad()
def evaluate_geometry_path_steering(
    models: Mapping[str, Any],
    cfg,
    vocab: Vocab,
    centroid_arrays: Mapping[str, np.ndarray],
    *,
    examples_per_count: int,
    alphas: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    nonadjacent_offsets: Sequence[int] = (-10, -5, -3, -2, 2, 3, 5, 10),
    sites: Sequence[str] = FINAL_SITES,
    patch_batch_size: int = 24,
) -> pd.DataFrame:
    examples = balanced_examples(
        cfg,
        vocab,
        int(examples_per_count),
        int(cfg.seed) + 1_070_000,
    )
    rows: list[dict[str, Any]] = []
    total = len(sites) * len(examples) * int(cfg.n_layer)
    progress = tqdm(total=total, desc="v10 centroid-path steering")
    for site in sites:
        mode = "nonthinking" if site == SITE_NONTHINKING_FINAL else "thinking"
        model = models[mode]
        for example_idx, example in enumerate(examples):
            item = render_site(example, vocab, site, cfg)
            ids = torch.tensor([item.input_ids], dtype=torch.long, device=cfg.device)
            position = int(item.spans.ans_pos)
            baseline_logits, receiver_states = _capture_residuals(model, ids, position)
            baseline_pred, baseline_expected, _ = count_prediction(baseline_logits, vocab)
            receiver_count = int(example.count)
            for layer, receiver_state_tensor in enumerate(receiver_states):
                centroids = _centroids_for_site_layer(
                    centroid_arrays, site, layer, cfg.count_min, cfg.count_max
                )
                receiver_centroid = centroids[receiver_count]
                receiver_state = receiver_state_tensor[0].detach().float().cpu().numpy()
                specifications: list[dict[str, Any]] = []

                for offset in (-1, 1):
                    donor_count = receiver_count + offset
                    if not cfg.count_min <= donor_count <= cfg.count_max:
                        continue
                    donor_centroid = centroids[donor_count]
                    specifications.extend(
                        [
                            {
                                "method": "adjacent_centroid_transplant",
                                "donor_count": donor_count,
                                "alpha": 1.0,
                                "path_count_coordinate": float(donor_count),
                                "state": donor_centroid,
                                "curve_chord_state_distance": 0.0,
                            },
                            {
                                "method": "adjacent_delta_transport",
                                "donor_count": donor_count,
                                "alpha": 1.0,
                                "path_count_coordinate": float(donor_count),
                                "state": receiver_state + donor_centroid - receiver_centroid,
                                "curve_chord_state_distance": 0.0,
                            },
                        ]
                    )

                for offset in nonadjacent_offsets:
                    donor_count = receiver_count + int(offset)
                    if not cfg.count_min <= donor_count <= cfg.count_max:
                        continue
                    for alpha in alphas:
                        chord_point, chord_coordinate = centroid_chord_point(
                            centroids, receiver_count, donor_count, float(alpha)
                        )
                        curve_point, curve_coordinate = centroid_polyline_point(
                            centroids, receiver_count, donor_count, float(alpha)
                        )
                        state_distance = float(np.linalg.norm(curve_point - chord_point))
                        specifications.extend(
                            [
                                {
                                    "method": "nonadjacent_chord_transport",
                                    "donor_count": donor_count,
                                    "alpha": float(alpha),
                                    "path_count_coordinate": chord_coordinate,
                                    "state": receiver_state + chord_point - receiver_centroid,
                                    "curve_chord_state_distance": state_distance,
                                },
                                {
                                    "method": "nonadjacent_curve_transport",
                                    "donor_count": donor_count,
                                    "alpha": float(alpha),
                                    "path_count_coordinate": curve_coordinate,
                                    "state": receiver_state + curve_point - receiver_centroid,
                                    "curve_chord_state_distance": state_distance,
                                },
                            ]
                        )

                if not specifications:
                    progress.update(1)
                    continue
                patch_batch_size = max(1, int(patch_batch_size))
                patched_rows: list[tuple[torch.Tensor, dict[str, Any]]] = []
                for start in range(0, len(specifications), patch_batch_size):
                    batch_specs = specifications[start : start + patch_batch_size]
                    replacements = torch.tensor(
                        np.stack([spec["state"] for spec in batch_specs]),
                        dtype=torch.float32,
                        device=cfg.device,
                    )
                    patched_logits = _patched_residual_batch(
                        model, ids, layer, position, replacements
                    )
                    patched_rows.extend(zip(patched_logits, batch_specs))
                for row_logits, spec in patched_rows:
                    patched_pred, patched_expected, probabilities = count_prediction(
                        row_logits, vocab
                    )
                    donor_count = int(spec["donor_count"])
                    path_count_coordinate = float(spec["path_count_coordinate"])
                    intended_integer = int(
                        np.clip(round(path_count_coordinate), cfg.count_min, cfg.count_max)
                    )
                    donor_offset = donor_count - receiver_count
                    intended_shift = path_count_coordinate - receiver_count
                    update_norm = float(
                        np.linalg.norm(np.asarray(spec["state"]) - receiver_state)
                    )
                    rows.append(
                        {
                            "site": site,
                            "mode": mode,
                            "example_idx": example_idx,
                            "count_bin": count_bin(receiver_count),
                            "receiver_count": receiver_count,
                            "donor_count": donor_count,
                            "donor_offset": donor_offset,
                            "abs_offset": abs(donor_offset),
                            "layer": layer,
                            "token_position": position,
                            "method": spec["method"],
                            "alpha": float(spec["alpha"]),
                            "path_count_coordinate": path_count_coordinate,
                            "intended_integer_count": intended_integer,
                            "intended_count_shift": intended_shift,
                            "baseline_pred": baseline_pred,
                            "baseline_expected": baseline_expected,
                            "baseline_correct": float(baseline_pred == receiver_count),
                            "patched_pred": patched_pred,
                            "patched_expected": patched_expected,
                            "causal_expected_shift": patched_expected - baseline_expected,
                            "pair_transport_fraction": (
                                (patched_expected - baseline_expected) / intended_shift
                                if abs(intended_shift) > 1e-12
                                else math.nan
                            ),
                            "path_tracking_error": abs(patched_expected - path_count_coordinate),
                            "follows_donor": float(patched_pred == donor_count),
                            "follows_receiver": float(patched_pred == receiver_count),
                            "follows_intended_integer": float(patched_pred == intended_integer),
                            "donor_probability": float(probabilities[donor_count - 1]),
                            "intended_probability": float(probabilities[intended_integer - 1]),
                            "state_update_norm": update_norm,
                            "curve_chord_state_distance": float(
                                spec["curve_chord_state_distance"]
                            ),
                        }
                    )
                progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def _curve_chord_comparison(detail: pd.DataFrame) -> pd.DataFrame:
    frame = detail[detail.method.str.startswith("nonadjacent_")].copy()
    if frame.empty:
        return pd.DataFrame()
    keys = [
        "site",
        "mode",
        "example_idx",
        "count_bin",
        "receiver_count",
        "donor_count",
        "donor_offset",
        "abs_offset",
        "layer",
        "alpha",
    ]
    wide = frame.pivot_table(
        index=keys,
        columns="method",
        values=["patched_expected", "path_tracking_error", "patched_pred"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}__{method}" for metric, method in wide.columns]
    wide = wide.reset_index()
    curve = "nonadjacent_curve_transport"
    chord = "nonadjacent_chord_transport"
    wide["curve_minus_chord_expected"] = (
        wide[f"patched_expected__{curve}"] - wide[f"patched_expected__{chord}"]
    )
    wide["curve_minus_chord_tracking_error"] = (
        wide[f"path_tracking_error__{curve}"] - wide[f"path_tracking_error__{chord}"]
    )
    wide["same_prediction"] = (
        wide[f"patched_pred__{curve}"] == wide[f"patched_pred__{chord}"]
    ).astype(float)
    return wide


def run_geometry_path_steering(
    run_dir: str | Path,
    *,
    examples_per_count: int = 4,
    alphas: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    nonadjacent_offsets: Sequence[int] = (-10, -5, -3, -2, 2, 3, 5, 10),
    patch_batch_size: int = 24,
    device: str | None = None,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    out_dir = run_dir / "analysis" / "geometry_path_steering"
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    paths = {
        "detail": tables / "geometry_path_steering.csv",
        "summary": tables / "geometry_path_steering_summary.csv",
        "regression": tables / "geometry_path_steering_regression.csv",
        "curve_chord": tables / "curve_chord_pair_comparison.csv",
    }
    if not overwrite and all(path.exists() for path in paths.values()):
        return {name: pd.read_csv(path) for name, path in paths.items()}

    cfg, vocab = load_run(run_dir, device=device)
    centroid_path = run_dir / "analysis" / "state_causal" / "centroids.npz"
    if not centroid_path.exists():
        raise FileNotFoundError(
            f"Missing independent count centroids: {centroid_path}. Run the v10 state_causal stage first."
        )
    centroid_arrays = np.load(centroid_path)
    models = {
        mode: load_final_model(cfg, vocab, run_dir, mode)
        for mode in cfg.modes
    }
    detail = evaluate_geometry_path_steering(
        models,
        cfg,
        vocab,
        centroid_arrays,
        examples_per_count=int(examples_per_count),
        alphas=tuple(float(value) for value in alphas),
        nonadjacent_offsets=tuple(int(value) for value in nonadjacent_offsets),
        patch_batch_size=int(patch_batch_size),
    )
    summary, regression = summarize_geometry_path_steering(detail)
    curve_chord = _curve_chord_comparison(detail)
    outputs = {
        "detail": detail,
        "summary": summary,
        "regression": regression,
        "curve_chord": curve_chord,
    }
    for name, frame in outputs.items():
        frame.to_csv(paths[name], index=False)
    endpoint = curve_chord[curve_chord.alpha == 1.0]
    manifest = {
        "run_dir": str(run_dir.resolve()),
        "examples_per_count": int(examples_per_count),
        "alphas": [float(value) for value in alphas],
        "nonadjacent_offsets": [int(value) for value in nonadjacent_offsets],
        "patch_batch_size": int(patch_batch_size),
        "sites": list(FINAL_SITES),
        "definitions": {
            "adjacent_centroid_transplant": "replace h_r by the independent centroid mu_t for t=r+-1",
            "adjacent_delta_transport": "preserve receiver residual epsilon and set h'=h_r+(mu_t-mu_r)",
            "nonadjacent_chord_transport": "move the receiver residual along the straight endpoint chord",
            "nonadjacent_curve_transport": "move by normalized arc length along the piecewise-linear chain of adjacent centroids",
        },
        "endpoint_sanity": {
            "rows": int(len(endpoint)),
            "max_abs_expected_difference": (
                float(endpoint.curve_minus_chord_expected.abs().max())
                if not endpoint.empty
                else math.nan
            ),
            "prediction_agreement": (
                float(endpoint.same_prediction.mean()) if not endpoint.empty else math.nan
            ),
            "note": "At alpha=1 the curve and chord offsets are mathematically identical.",
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    for model in models.values():
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs
