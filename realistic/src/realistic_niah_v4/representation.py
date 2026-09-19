from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .modeling import DecoderAdapter, capture_post_block_states, capture_span_states
from .prompts import PromptEncoding
from .spec import CAPTURE_SCHEMA_VERSION, V4Config


def _stable_seed(label: str, seed: int) -> int:
    digest = hashlib.sha256(f"{label}:{int(seed)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_647


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _numpy_dtype(name: str) -> np.dtype[Any]:
    dtype = np.dtype(str(name))
    if dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("V4 hidden_save_dtype must be float16 or float32")
    return dtype


@torch.inference_mode()
def capture_representation_shards(
    model: Any,
    adapter: DecoderAdapter,
    encodings: Iterable[PromptEncoding],
    *,
    output_dir: str | Path,
    save_dtype: str = "float16",
    overwrite: bool = False,
) -> Path:
    """Capture post-block hidden states at every active needle span.

    One shard per prompt makes the long GPU job restartable. The forward hooks
    retain only 2 * layers * N * hidden values, never all 10k token states.
    """

    output = Path(output_dir)
    dtype = _numpy_dtype(save_dtype)
    index_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for example_index, encoding in enumerate(encodings):
        if encoding.count < 1 or len(encoding.needle_spans) != encoding.count:
            raise ValueError(
                "Prompt-reading representation capture requires exactly one "
                "active needle span per registered count"
            )
        if encoding.stimulus_id in seen:
            raise ValueError(f"Duplicate capture stimulus: {encoding.stimulus_id}")
        seen.add(encoding.stimulus_id)
        relative = (
            Path("shards") / encoding.design_variant / f"{encoding.stimulus_id}.npz"
        )
        shard = output / relative
        if shard.exists() and not overwrite:
            with np.load(shard, allow_pickle=False) as saved:
                if set(saved.files) != {
                    "layer_indices",
                    "span_end",
                    "span_mean",
                }:
                    raise RuntimeError(f"Incomplete V4 capture shard: {shard}")
                shape = tuple(int(value) for value in saved["span_end"].shape)
                if saved["span_mean"].shape != shape:
                    raise RuntimeError(f"Pooling shape mismatch in {shard}")
        else:
            captured = capture_span_states(
                model,
                adapter,
                encoding,
                encoding.needle_spans,
            )
            span_end = captured["span_end"].numpy().astype(dtype, copy=False)
            span_mean = captured["span_mean"].numpy().astype(dtype, copy=False)
            layer_indices = captured["layer_indices"].numpy()
            if span_end.shape != span_mean.shape:
                raise RuntimeError("V4 pooling capture shapes disagree")
            shard.parent.mkdir(parents=True, exist_ok=True)
            temporary = shard.with_name(shard.name + ".tmp")
            with temporary.open("wb") as handle:
                np.savez(
                    handle,
                    layer_indices=layer_indices,
                    span_end=span_end,
                    span_mean=span_mean,
                )
            temporary.replace(shard)
            shape = tuple(int(value) for value in span_end.shape)
        index_rows.append(
            {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "example_index": int(example_index),
                "stimulus_id": encoding.stimulus_id,
                "design_variant": encoding.design_variant,
                "model_label": encoding.model_label,
                "answer_format": encoding.answer_format,
                "seed": int(encoding.seed),
                "split": encoding.split,
                "count": int(encoding.count),
                "sequence_length": int(encoding.sequence_length),
                "query_position": int(encoding.query_position),
                "poolings": ["span_end", "span_mean"],
                "array_shape": list(shape),
                "save_dtype": str(dtype),
                "shard_path": relative.as_posix(),
            }
        )
        print(
            "[v4 representation] "
            f"{example_index + 1} variant={encoding.design_variant} "
            f"seed={encoding.seed} shard={shard}",
            flush=True,
        )
    if not index_rows:
        raise ValueError("No V4 representation encodings were supplied")
    index_path = output / "capture_index.jsonl"
    _atomic_jsonl(index_path, index_rows)
    _atomic_json(
        output / "capture_manifest.json",
        {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "rows": len(index_rows),
            "model_labels": sorted({str(row["model_label"]) for row in index_rows}),
            "answer_formats": sorted(
                {str(row["answer_format"]) for row in index_rows}
            ),
            "design_variants": sorted(
                {str(row["design_variant"]) for row in index_rows}
            ),
            "splits": sorted({str(row["split"]) for row in index_rows}),
            "counts": sorted({int(row["count"]) for row in index_rows}),
            "poolings": ["span_end", "span_mean"],
            "restartable_shards": True,
            "full_sequence_hidden_states_materialized": False,
        },
    )
    return index_path


@torch.inference_mode()
def capture_answer_query_representation_shards(
    model: Any,
    adapter: DecoderAdapter,
    encodings: Iterable[PromptEncoding],
    *,
    output_dir: str | Path,
    save_dtype: str = "float16",
    layers: Iterable[int] | None = None,
    overwrite: bool = False,
) -> Path:
    """Capture the prompt-final ``Total:`` query residual at every layer.

    This capture is deliberately independent of geometric steering.  Steering
    may use a sparse intervention layer grid, whereas a representation sweep
    needs every post-block decoder state.  Each restartable shard stores only
    ``[layers, hidden]`` for one prompt; no sequence-wide hidden states are
    materialized or serialized.
    """

    output = Path(output_dir)
    dtype = _numpy_dtype(save_dtype)
    if layers is None:
        selected_layers = tuple(range(int(adapter.num_layers)))
    else:
        selected_layers = tuple(sorted({int(layer) for layer in layers}))
    if not selected_layers:
        raise ValueError("At least one answer-query layer is required")
    invalid = [
        layer
        for layer in selected_layers
        if not 0 <= int(layer) < int(adapter.num_layers)
    ]
    if invalid:
        raise ValueError(f"Invalid answer-query decoder layers: {invalid}")

    index_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for example_index, encoding in enumerate(encodings):
        if encoding.stimulus_id in seen:
            raise ValueError(
                f"Duplicate answer-query capture stimulus: {encoding.stimulus_id}"
            )
        seen.add(encoding.stimulus_id)
        relative = (
            Path("shards") / encoding.design_variant / f"{encoding.stimulus_id}.npz"
        )
        shard = output / relative
        if shard.exists() and not overwrite:
            with np.load(shard, allow_pickle=False) as saved:
                if set(saved.files) != {
                    "layer_indices",
                    "query_states",
                    "query_position",
                    "sequence_length",
                }:
                    raise RuntimeError(
                        f"Incomplete answer-query representation shard: {shard}"
                    )
                observed_layers = tuple(
                    int(value) for value in saved["layer_indices"]
                )
                query_states = np.asarray(saved["query_states"])
                query_position = int(saved["query_position"][0])
                sequence_length = int(saved["sequence_length"][0])
            if observed_layers != selected_layers:
                raise RuntimeError(
                    f"Answer-query layer grid mismatch in {shard}: "
                    f"{observed_layers} != {selected_layers}"
                )
            if query_states.ndim != 2 or query_states.shape[0] != len(
                selected_layers
            ):
                raise RuntimeError(
                    f"Invalid answer-query state shape in {shard}: "
                    f"{query_states.shape}"
                )
            if query_position != int(encoding.query_position):
                raise RuntimeError(f"Answer-query position mismatch in {shard}")
            if sequence_length != int(encoding.sequence_length):
                raise RuntimeError(f"Sequence-length mismatch in {shard}")
        else:
            _logits, captured = capture_post_block_states(
                model,
                adapter,
                encoding,
                [int(encoding.query_position)],
                layers=selected_layers,
            )
            query_states = np.stack(
                [captured[layer][0].numpy() for layer in selected_layers],
                axis=0,
            ).astype(dtype, copy=False)
            if query_states.ndim != 2 or not np.isfinite(query_states).all():
                raise RuntimeError("Invalid answer-query residual capture")
            shard.parent.mkdir(parents=True, exist_ok=True)
            temporary = shard.with_name(shard.name + ".tmp")
            with temporary.open("wb") as handle:
                np.savez(
                    handle,
                    layer_indices=np.asarray(selected_layers, dtype=np.int64),
                    query_states=query_states,
                    query_position=np.asarray(
                        [encoding.query_position], dtype=np.int64
                    ),
                    sequence_length=np.asarray(
                        [encoding.sequence_length], dtype=np.int64
                    ),
                )
            temporary.replace(shard)

        index_rows.append(
            {
                "schema_version": "realistic_niah_v4_answer_query_capture_v1",
                "example_index": int(example_index),
                "stimulus_id": encoding.stimulus_id,
                "design_variant": encoding.design_variant,
                "model_label": encoding.model_label,
                "answer_format": encoding.answer_format,
                "seed": int(encoding.seed),
                "split": encoding.split,
                "count": int(encoding.count),
                "sequence_length": int(encoding.sequence_length),
                "query_position": int(encoding.query_position),
                "position": "prompt_final_total_query",
                "layer_indices": list(selected_layers),
                "array_shape": [int(value) for value in query_states.shape],
                "save_dtype": str(dtype),
                "shard_path": relative.as_posix(),
            }
        )
        print(
            "[v4 answer-query representation] "
            f"{example_index + 1} variant={encoding.design_variant} "
            f"seed={encoding.seed} count={encoding.count} shard={shard}",
            flush=True,
        )
    if not index_rows:
        raise ValueError("No V4 answer-query encodings were supplied")
    index_path = output / "capture_index.jsonl"
    _atomic_jsonl(index_path, index_rows)
    _atomic_json(
        output / "capture_manifest.json",
        {
            "schema_version": "realistic_niah_v4_answer_query_capture_v1",
            "rows": len(index_rows),
            "model_labels": sorted(
                {str(row["model_label"]) for row in index_rows}
            ),
            "answer_formats": sorted(
                {str(row["answer_format"]) for row in index_rows}
            ),
            "design_variants": sorted(
                {str(row["design_variant"]) for row in index_rows}
            ),
            "splits": sorted({str(row["split"]) for row in index_rows}),
            "counts": sorted({int(row["count"]) for row in index_rows}),
            "layer_indices": list(selected_layers),
            "position": "prompt_final_total_query",
            "restartable_shards": True,
            "full_sequence_hidden_states_materialized": False,
        },
    )
    return index_path


def load_capture_index(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_variant_pooling(
    *,
    index_path: Path,
    records: Sequence[dict[str, Any]],
    variant: str,
    pooling: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted(
        [row for row in records if row["design_variant"] == variant],
        key=lambda row: int(row["seed"]),
    )
    if not selected:
        raise ValueError(f"No representation shards for {variant}")
    arrays: list[np.ndarray] = []
    layers: np.ndarray | None = None
    for row in selected:
        shard = index_path.parent / str(row["shard_path"])
        with np.load(shard, allow_pickle=False) as saved:
            current_layers = np.asarray(saved["layer_indices"], dtype=int)
            if layers is None:
                layers = current_layers
            elif not np.array_equal(layers, current_layers):
                raise RuntimeError("V4 capture shards use different layers")
            arrays.append(np.asarray(saved[pooling]))
    tensor = np.stack(arrays, axis=0)
    seeds = np.asarray([int(row["seed"]) for row in selected], dtype=int)
    return tensor, seeds, np.asarray(layers, dtype=int)


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or float(np.var(y_true)) <= 0:
        return math.nan
    return float(r2_score(y_true, y_pred))


def _safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return math.nan
    if float(np.std(y_true)) <= 0 or float(np.std(y_pred)) <= 0:
        return math.nan
    return float(pearsonr(y_true, y_pred).statistic)


def _ridge_probe(
    x: np.ndarray,
    y: np.ndarray,
    seeds: np.ndarray,
    discovery_mask: np.ndarray,
    confirmation_mask: np.ndarray,
    *,
    alphas: Sequence[float],
) -> tuple[dict[str, float], np.ndarray]:
    x_train = np.asarray(x[discovery_mask], dtype=np.float32)
    y_train = np.asarray(y[discovery_mask], dtype=np.float32)
    groups = np.asarray(seeds[discovery_mask], dtype=int)
    unique_groups = np.unique(groups)
    folds = min(5, len(unique_groups))
    if folds < 2:
        raise ValueError("At least two discovery seeds are required")
    splitter = GroupKFold(n_splits=folds)
    alpha_scores: list[tuple[float, float]] = []
    for alpha in alphas:
        fold_scores = []
        for train_indices, validation_indices in splitter.split(
            x_train, y_train, groups
        ):
            estimator = make_pipeline(
                StandardScaler(),
                Ridge(alpha=float(alpha), solver="lsqr"),
            )
            estimator.fit(x_train[train_indices], y_train[train_indices])
            prediction = estimator.predict(x_train[validation_indices])
            fold_scores.append(_safe_r2(y_train[validation_indices], prediction))
        alpha_scores.append((float(alpha), float(np.nanmean(fold_scores))))
    best_alpha, best_cv_r2 = max(alpha_scores, key=lambda item: item[1])
    estimator = make_pipeline(
        StandardScaler(),
        Ridge(alpha=best_alpha, solver="lsqr"),
    )
    estimator.fit(x_train, y_train)
    prediction = np.full(len(y), np.nan, dtype=float)
    prediction[discovery_mask] = estimator.predict(x_train)
    prediction[confirmation_mask] = estimator.predict(
        np.asarray(x[confirmation_mask], dtype=np.float32)
    )
    test_true = y[confirmation_mask]
    test_prediction = prediction[confirmation_mask]
    return (
        {
            "ridge_alpha": best_alpha,
            "discovery_group_cv_r2": best_cv_r2,
            "discovery_fit_r2": _safe_r2(y[discovery_mask], prediction[discovery_mask]),
            "confirmation_r2": _safe_r2(test_true, test_prediction),
            "confirmation_mae": float(mean_absolute_error(test_true, test_prediction)),
            "confirmation_pearson": _safe_pearson(test_true, test_prediction),
        },
        prediction,
    )


def _linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    left_gram = left @ left.T
    right_gram = right @ right.T
    numerator = float(np.sum(left_gram * right_gram))
    denominator = math.sqrt(
        float(np.sum(left_gram * left_gram)) * float(np.sum(right_gram * right_gram))
    )
    return numerator / denominator if denominator > 0 else math.nan


def _distance_geometry_correlation(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    def pairwise(values: np.ndarray) -> np.ndarray:
        difference = values[:, None, :] - values[None, :, :]
        distances = np.sqrt(np.sum(difference * difference, axis=-1))
        indices = np.triu_indices(len(values), k=1)
        return distances[indices]

    left_distances = pairwise(left)
    right_distances = pairwise(right)
    return _safe_pearson(left_distances, right_distances)


def _curve_metrics(
    x: np.ndarray,
    index: np.ndarray,
    discovery_mask: np.ndarray,
    confirmation_mask: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    labels = np.arange(1, 11, dtype=int)
    discovery_centroids = np.stack(
        [x[discovery_mask & (index == label)].mean(axis=0) for label in labels]
    )
    confirmation_centroids = np.stack(
        [x[confirmation_mask & (index == label)].mean(axis=0) for label in labels]
    )
    grand = discovery_centroids.mean(axis=0)
    signal_variance = float(np.mean(np.sum((discovery_centroids - grand) ** 2, axis=1)))
    residual = x[confirmation_mask] - discovery_centroids[index[confirmation_mask] - 1]
    noise_variance = float(np.mean(np.sum(residual * residual, axis=1)))
    deltas = np.diff(discovery_centroids, axis=0)
    lengths = np.sqrt(np.sum(deltas * deltas, axis=1))
    adjacent_cosines: list[float] = []
    for left, right in zip(deltas, deltas[1:]):
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        adjacent_cosines.append(
            float(np.dot(left, right) / denominator) if denominator > 0 else math.nan
        )
    chord = float(np.linalg.norm(discovery_centroids[-1] - discovery_centroids[0]))
    path = float(lengths.sum())
    metrics = {
        "signal_rms": math.sqrt(max(0.0, signal_variance)),
        "confirmation_noise_rms": math.sqrt(max(0.0, noise_variance)),
        "noise_to_signal_ratio": (
            math.sqrt(noise_variance / signal_variance)
            if signal_variance > 0
            else math.nan
        ),
        "centroid_adjacent_step_mean": float(lengths.mean()),
        "centroid_adjacent_step_cv": (
            float(lengths.std(ddof=0) / lengths.mean())
            if float(lengths.mean()) > 0
            else math.nan
        ),
        "centroid_adjacent_direction_cosine": float(np.nanmean(adjacent_cosines)),
        "centroid_path_to_chord_ratio": (path / chord if chord > 0 else math.nan),
        "discovery_confirmation_linear_cka": _linear_cka(
            discovery_centroids, confirmation_centroids
        ),
        "discovery_confirmation_distance_correlation": (
            _distance_geometry_correlation(discovery_centroids, confirmation_centroids)
        ),
    }
    return metrics, discovery_centroids


def _per_seed_rows(
    *,
    x: np.ndarray,
    index: np.ndarray,
    seed_labels: np.ndarray,
    confirmation_mask: np.ndarray,
    centroids: np.ndarray,
    prediction: np.ndarray,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grand = centroids.mean(axis=0)
    signal_rms = float(np.sqrt(np.mean(np.sum((centroids - grand) ** 2, axis=1))))
    for seed in sorted(np.unique(seed_labels[confirmation_mask])):
        mask = confirmation_mask & (seed_labels == int(seed))
        residual = x[mask] - centroids[index[mask] - 1]
        residual_rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        rows.append(
            {
                **metadata,
                "seed": int(seed),
                "probe_mae": float(np.mean(np.abs(prediction[mask] - index[mask]))),
                "curve_residual_rms": residual_rms,
                "curve_residual_to_signal": (
                    residual_rms / signal_rms if signal_rms > 0 else math.nan
                ),
            }
        )
    return rows


def _bootstrap_paired_delta(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    repetitions: int = 10_000,
) -> dict[str, float | bool]:
    if left.shape != right.shape or left.ndim != 1 or len(left) == 0:
        raise ValueError("Paired bootstrap inputs must be nonempty 1D peers")
    delta = np.asarray(right - left, dtype=float)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(delta), size=(int(repetitions), len(delta)))
    means = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "delta_mean": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "increase_ci_excludes_zero": bool(low > 0),
        "decrease_ci_excludes_zero": bool(high < 0),
        "bootstrap_repetitions": int(repetitions),
    }


def _plot_shared_pca(
    *,
    index_path: Path,
    records: Sequence[dict[str, Any]],
    config: V4Config,
    pooling: str,
    layer: int,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variant_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for variant in config.design_variants:
        tensor, seeds, layers = _load_variant_pooling(
            index_path=index_path,
            records=records,
            variant=variant,
            pooling=pooling,
        )
        layer_matches = np.flatnonzero(layers == int(layer))
        if len(layer_matches) != 1:
            raise RuntimeError(f"Layer {layer} is absent or duplicated")
        variant_data[variant] = (
            np.asarray(tensor[:, layer_matches[0]], dtype=np.float32),
            seeds,
        )
    reference, reference_seeds = variant_data["v4.1"]
    discovery = np.isin(reference_seeds, config.discovery_seeds)
    reference_flat = reference[discovery].reshape(-1, reference.shape[-1])
    pca = PCA(
        n_components=int(config.pca_components),
        svd_solver="randomized",
        random_state=0,
    )
    pca.fit(reference_flat)

    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(config.needle_counts)))
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)
    point_rows: list[dict[str, Any]] = []
    for axis, variant in zip(axes, config.design_variants):
        tensor, seeds = variant_data[variant]
        flat = tensor.reshape(-1, tensor.shape[-1])
        projected = pca.transform(flat).reshape(
            len(seeds), 10, int(config.pca_components)
        )
        discovery_mask = np.isin(seeds, config.discovery_seeds)
        confirmation_mask = np.isin(seeds, config.confirmation_seeds)
        discovery_centroid = projected[discovery_mask].mean(axis=0)
        confirmation_centroid = projected[confirmation_mask].mean(axis=0)
        for seed_index, seed in enumerate(seeds):
            split = (
                "discovery" if seed in set(config.discovery_seeds) else "confirmation"
            )
            for count_index, count in enumerate(config.needle_counts):
                x_value, y_value = projected[seed_index, count_index, :2]
                point_rows.append(
                    {
                        "design_variant": variant,
                        "pooling": pooling,
                        "layer": int(layer),
                        "seed": int(seed),
                        "split": split,
                        "count_index": int(count),
                        "pc1": float(x_value),
                        "pc2": float(y_value),
                    }
                )
                axis.scatter(
                    x_value,
                    y_value,
                    s=12 if split == "discovery" else 22,
                    alpha=0.16 if split == "discovery" else 0.48,
                    marker="o" if split == "discovery" else "x",
                    color=colors[count_index],
                    linewidths=0.7,
                )
        axis.plot(
            discovery_centroid[:, 0],
            discovery_centroid[:, 1],
            color="#555555",
            linestyle="--",
            linewidth=1.6,
            label="discovery mean",
        )
        axis.plot(
            confirmation_centroid[:, 0],
            confirmation_centroid[:, 1],
            color="#111111",
            linewidth=2.0,
            label="confirmation mean",
        )
        for count_index, count in enumerate(config.needle_counts):
            axis.scatter(
                confirmation_centroid[count_index, 0],
                confirmation_centroid[count_index, 1],
                s=42,
                color=colors[count_index],
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
            axis.annotate(
                str(count),
                confirmation_centroid[count_index, :2],
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        axis.set_title(variant)
        axis.grid(alpha=0.15)
        axis.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[-1].legend(loc="best", fontsize=8)
    figure.suptitle(f"{pooling}, layer {layer}: PCA fit on v4.1 discovery seeds")
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"shared_pca_{pooling}_layer_{layer}.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    points_path = output_dir / f"shared_pca_{pooling}_layer_{layer}.csv"
    pd.DataFrame(point_rows).to_csv(points_path, index=False)
    pca_metadata = {
        "pooling": pooling,
        "layer": int(layer),
        "fit_variant": "v4.1",
        "fit_split": "discovery",
        "explained_variance_ratio": [
            float(value) for value in pca.explained_variance_ratio_
        ],
    }
    return figure_path, points_path, pca_metadata


def analyze_representation_captures(
    *,
    capture_index_path: str | Path,
    output_dir: str | Path,
    config: V4Config,
) -> dict[str, Path]:
    """Analyze index-aligned trajectories and locate seed sensitivity."""

    config.validate()
    index_path = Path(capture_index_path)
    records = load_capture_index(index_path)
    output = Path(output_dir)
    if not records:
        raise ValueError("The V4 representation capture index is empty")
    observed_variants = {str(row["design_variant"]) for row in records}
    if observed_variants != set(config.design_variants):
        raise ValueError("Representation captures do not cover all registered variants")
    model_labels = {str(row["model_label"]) for row in records}
    if len(model_labels) != 1:
        raise ValueError("Analyze one V4 model per capture index")

    metric_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    layer_indices_reference: np.ndarray | None = None
    for pooling in config.hidden_state_poolings:
        for variant in config.design_variants:
            tensor, seeds, layer_indices = _load_variant_pooling(
                index_path=index_path,
                records=records,
                variant=variant,
                pooling=pooling,
            )
            if layer_indices_reference is None:
                layer_indices_reference = layer_indices
            elif not np.array_equal(layer_indices_reference, layer_indices):
                raise RuntimeError("Variants have inconsistent captured layers")
            if tensor.shape[0] != len(config.seeds) or tensor.shape[2] != 10:
                raise RuntimeError(
                    f"Unexpected V4 capture tensor shape: {tensor.shape}"
                )
            seed_labels = np.repeat(seeds, 10)
            index_labels = np.tile(np.arange(1, 11, dtype=int), len(seeds))
            discovery_mask = np.isin(seed_labels, np.asarray(config.discovery_seeds))
            confirmation_mask = np.isin(
                seed_labels, np.asarray(config.confirmation_seeds)
            )
            for layer_axis, layer in enumerate(layer_indices):
                x = np.asarray(
                    tensor[:, layer_axis].reshape(-1, tensor.shape[-1]),
                    dtype=np.float32,
                )
                probe, prediction = _ridge_probe(
                    x,
                    index_labels,
                    seed_labels,
                    discovery_mask,
                    confirmation_mask,
                    alphas=config.ridge_alphas,
                )
                curve, centroids = _curve_metrics(
                    x,
                    index_labels,
                    discovery_mask,
                    confirmation_mask,
                )
                metadata = {
                    "model_label": next(iter(model_labels)),
                    "design_variant": variant,
                    "pooling": pooling,
                    "layer": int(layer),
                }
                metric_rows.append({**metadata, **probe, **curve})
                seed_rows.extend(
                    _per_seed_rows(
                        x=x,
                        index=index_labels,
                        seed_labels=seed_labels,
                        confirmation_mask=confirmation_mask,
                        centroids=centroids,
                        prediction=prediction,
                        metadata=metadata,
                    )
                )
            del tensor

    metrics = pd.DataFrame(metric_rows)
    per_seed = pd.DataFrame(seed_rows)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "representation_layer_metrics.csv"
    seed_path = output / "representation_confirmation_by_seed.csv"
    metrics.to_csv(metrics_path, index=False)
    per_seed.to_csv(seed_path, index=False)

    primary_layers: dict[str, int] = {}
    sensitivity_rows: list[dict[str, Any]] = []
    sensitivity_calls: list[dict[str, Any]] = []
    pca_metadata: list[dict[str, Any]] = []
    figures: list[str] = []
    for pooling in config.hidden_state_poolings:
        selection = metrics[
            (metrics["design_variant"] == "v4.1") & (metrics["pooling"] == pooling)
        ].sort_values(
            ["discovery_group_cv_r2", "layer"],
            ascending=[False, True],
        )
        if selection.empty:
            raise RuntimeError(f"No v4.1 layer metrics for {pooling}")
        primary_layer = int(selection.iloc[0]["layer"])
        primary_layers[pooling] = primary_layer
        figure, points, pca_info = _plot_shared_pca(
            index_path=index_path,
            records=records,
            config=config,
            pooling=pooling,
            layer=primary_layer,
            output_dir=output / "figures",
        )
        figures.extend([str(figure), str(points)])
        pca_metadata.append(pca_info)
        for metric_name in ("probe_mae", "curve_residual_to_signal"):
            earliest: str | None = None
            for left_variant, right_variant in zip(
                config.design_variants, config.design_variants[1:]
            ):
                subset = per_seed[
                    (per_seed["pooling"] == pooling)
                    & (per_seed["layer"] == primary_layer)
                    & (per_seed["design_variant"].isin([left_variant, right_variant]))
                ][["design_variant", "seed", metric_name]]
                pivot = subset.pivot(
                    index="seed",
                    columns="design_variant",
                    values=metric_name,
                ).dropna()
                comparison = _bootstrap_paired_delta(
                    pivot[left_variant].to_numpy(dtype=float),
                    pivot[right_variant].to_numpy(dtype=float),
                    seed=_stable_seed(
                        f"{pooling}:{metric_name}:{left_variant}:{right_variant}",
                        primary_layer,
                    ),
                )
                row = {
                    "pooling": pooling,
                    "primary_layer": primary_layer,
                    "metric": metric_name,
                    "left_variant": left_variant,
                    "right_variant": right_variant,
                    "paired_confirmation_seeds": len(pivot),
                    **comparison,
                }
                sensitivity_rows.append(row)
                if earliest is None and bool(comparison["increase_ci_excludes_zero"]):
                    earliest = right_variant
            sensitivity_calls.append(
                {
                    "pooling": pooling,
                    "metric": metric_name,
                    "primary_layer": primary_layer,
                    "earliest_step_with_positive_paired_delta_ci": earliest,
                    "decision_rule": (
                        "earliest adjacent relaxation whose 95% paired "
                        "seed-bootstrap CI for the increase excludes zero"
                    ),
                }
            )

    sensitivity_path = output / "seed_sensitivity_paired_bootstrap.csv"
    pd.DataFrame(sensitivity_rows).to_csv(sensitivity_path, index=False)
    summary_path = output / "representation_summary.json"
    _atomic_json(
        summary_path,
        {
            "schema_version": "realistic_niah_v4_representation_analysis_v1",
            "model_label": next(iter(model_labels)),
            "primary_layer_selection": {
                "source_variant": "v4.1",
                "source_split": "discovery",
                "criterion": "maximum grouped-seed ridge CV R2",
                "layers": primary_layers,
            },
            "seed_sensitivity_calls": sensitivity_calls,
            "pca": pca_metadata,
            "figures_and_points": figures,
            "interpretation_guardrail": (
                "v4.1 is identity/order/position aligned. Evidence for an "
                "abstract counter requires persistence through v4.3 and v4.4."
            ),
        },
    )
    return {
        "layer_metrics": metrics_path,
        "confirmation_by_seed": seed_path,
        "seed_sensitivity": sensitivity_path,
        "summary": summary_path,
    }


def label_representation_analysis_by_generation(
    *,
    analysis_dir: str | Path,
    generation_labels_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Attach actual N=10 greedy outcomes to saved representation analyses.

    No hidden states are recomputed. Every occurrence point in an N=10 prompt
    inherits that prompt's strict final-output label, allowing span-end and
    span-mean trajectory quality to be compared descriptively by correctness.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    analysis = Path(analysis_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(generation_labels_path)
    required = {
        "stimulus_id",
        "design_variant",
        "model_label",
        "seed",
        "split",
        "gold_count",
        "outcome_group",
        "is_correct",
        "format_valid",
        "parsed_count",
        "count_error",
    }
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Generation labels are missing columns: {missing}")
    labels = labels[labels["gold_count"].astype(int) == 10].copy()
    if labels.empty or labels.duplicated(["design_variant", "seed"]).any():
        raise ValueError("N=10 generation labels are empty or duplicated")
    label_columns = [
        "stimulus_id",
        "design_variant",
        "model_label",
        "seed",
        "split",
        "outcome_group",
        "is_correct",
        "format_valid",
        "parsed_count",
        "count_error",
    ]

    per_seed_path = analysis / "representation_confirmation_by_seed.csv"
    per_seed = pd.read_csv(per_seed_path)
    labeled_seed = per_seed.merge(
        labels[label_columns],
        on=["model_label", "design_variant", "seed"],
        how="left",
        validate="many_to_one",
    )
    if labeled_seed["outcome_group"].isna().any():
        raise RuntimeError("Some confirmation representation rows lack behavior labels")
    labeled_seed_path = output / "representation_confirmation_by_seed_labeled.csv"
    labeled_seed.to_csv(labeled_seed_path, index=False)
    metric_columns = [
        "probe_mae",
        "curve_residual_rms",
        "curve_residual_to_signal",
    ]
    grouped = labeled_seed.groupby(
        ["model_label", "design_variant", "pooling", "layer", "outcome_group"],
        as_index=False,
        dropna=False,
    )
    outcome_summary = grouped.agg(
        seeds=("seed", "nunique"),
        **{metric: (metric, "mean") for metric in metric_columns},
    )
    standard = grouped[metric_columns].std(ddof=1).rename(
        columns={metric: f"{metric}_sd" for metric in metric_columns}
    )
    outcome_summary = outcome_summary.merge(
        standard,
        on=["model_label", "design_variant", "pooling", "layer", "outcome_group"],
        how="left",
    )
    outcome_summary_path = output / "representation_outcome_summary.csv"
    outcome_summary.to_csv(outcome_summary_path, index=False)

    figure_output = output / "figures"
    figure_output.mkdir(parents=True, exist_ok=True)
    pca_outputs: list[str] = []
    colors = {"correct": "tab:blue", "wrong": "tab:red", "invalid": "0.5"}
    for pca_path in sorted((analysis / "figures").glob("shared_pca_*.csv")):
        points = pd.read_csv(pca_path)
        labeled_points = points.merge(
            labels[label_columns],
            on=["design_variant", "seed", "split"],
            how="left",
            validate="many_to_one",
        )
        if labeled_points["outcome_group"].isna().any():
            raise RuntimeError(f"PCA points lack final-output labels: {pca_path}")
        labeled_points_path = output / f"{pca_path.stem}_labeled.csv"
        labeled_points.to_csv(labeled_points_path, index=False)
        confirmation = labeled_points[labeled_points["split"] == "confirmation"]
        variants = sorted(confirmation["design_variant"].unique())
        figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
        for axis, variant in zip(axes.flat, variants):
            frame = confirmation[confirmation["design_variant"] == variant]
            for (_seed, outcome), trajectory in frame.groupby(
                ["seed", "outcome_group"], sort=True
            ):
                trajectory = trajectory.sort_values("count_index")
                axis.plot(
                    trajectory["pc1"],
                    trajectory["pc2"],
                    color=colors[str(outcome)],
                    alpha=0.35,
                    linewidth=0.8,
                )
                axis.scatter(
                    trajectory["pc1"],
                    trajectory["pc2"],
                    color=colors[str(outcome)],
                    alpha=0.5,
                    s=11,
                )
            axis.set_title(str(variant))
            axis.set_xlabel("shared PCA component 1")
            axis.set_ylabel("shared PCA component 2")
            axis.grid(alpha=0.15)
        present = [
            outcome
            for outcome in ("correct", "wrong", "invalid")
            if outcome in set(confirmation["outcome_group"])
        ]
        figure.legend(
            handles=[
                Line2D([0], [0], color=colors[outcome], label=outcome)
                for outcome in present
            ],
            loc="upper center",
            ncol=max(1, len(present)),
        )
        pooling = str(points["pooling"].iloc[0])
        layer = int(points["layer"].iloc[0])
        figure.suptitle(
            f"N=10 confirmation trajectories by actual output: "
            f"{pooling}, layer {layer}"
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        figure_path = figure_output / f"{pca_path.stem}_by_outcome.png"
        figure.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        pca_outputs.extend([str(labeled_points_path), str(figure_path)])

    manifest_path = output / "representation_outcome_manifest.json"
    _atomic_json(
        manifest_path,
        {
            "schema_version": "realistic_niah_v4_representation_outcomes_v1",
            "label_source": str(Path(generation_labels_path)),
            "label_rule": "strict actual greedy-generated numeric answer",
            "represented_prompt_count": len(labels),
            "outcome_counts": {
                str(key): int(value)
                for key, value in labels["outcome_group"].value_counts().items()
            },
            "scope": (
                "N=10 representation prompts only; each occurrence trajectory "
                "inherits its prompt-level final-output label"
            ),
            "interpretation": (
                "descriptive association between trajectory geometry and final "
                "accuracy; not a causal classification result"
            ),
            "files": {
                "confirmation_by_seed_labeled": str(labeled_seed_path),
                "outcome_summary": str(outcome_summary_path),
                "pca_outputs": pca_outputs,
            },
        },
    )
    return {
        "manifest": manifest_path,
        "confirmation_by_seed_labeled": labeled_seed_path,
        "outcome_summary": outcome_summary_path,
    }
