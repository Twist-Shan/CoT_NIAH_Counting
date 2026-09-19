from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from .behavior import count_logit_metrics
from .modeling import DecoderAdapter, query_attention_outputs
from .prompts import PromptEncoding, TokenSpan


Head = tuple[int, int]
BEHAVIOR_COLUMNS = (
    "gold_count",
    "predicted_count_among_candidates",
    "correct_count_logit",
    "correct_count_margin",
    "correct_count_probability",
    "expected_count",
    "candidate_counts",
    "candidate_logits",
    "candidate_probabilities",
)


def _span_mass(
    row: np.ndarray,
    span: TokenSpan,
    *,
    key_start: int,
) -> float:
    local_start = max(0, int(span.start) - int(key_start))
    local_end = min(len(row), int(span.end) - int(key_start))
    if local_end <= local_start:
        return 0.0
    return float(np.asarray(row[local_start:local_end], dtype=float).sum())


def _coverage(values: np.ndarray, epsilon: float) -> float:
    total = float(values.sum())
    if len(values) == 0 or total <= epsilon:
        return 0.0
    probabilities = values / (total + epsilon)
    entropy = float(-np.sum(probabilities * np.log(probabilities + epsilon)))
    return float(math.exp(entropy) / len(values))


def broad_attention_metrics(
    row: np.ndarray | torch.Tensor,
    needle_spans: Sequence[TokenSpan],
    hard_negative_spans: Sequence[TokenSpan],
    *,
    key_start: int = 0,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    values = (
        row.detach().float().cpu().numpy()
        if isinstance(row, torch.Tensor)
        else np.asarray(row, dtype=float)
    )
    if values.ndim != 1:
        raise ValueError("broad_attention_metrics expects one attention row")
    if not needle_spans:
        raise ValueError("At least one active needle span is required")
    negative_by_slot = {int(span.slot_index): span for span in hard_negative_spans}
    negatives = [
        negative_by_slot[int(span.slot_index)]
        for span in needle_spans
        if int(span.slot_index) in negative_by_slot
    ]
    if len(negatives) != len(needle_spans):
        raise ValueError("Every active needle needs a matched hard negative")

    needle_mass = np.asarray(
        [_span_mass(values, span, key_start=key_start) for span in needle_spans],
        dtype=float,
    )
    negative_mass = np.asarray(
        [_span_mass(values, span, key_start=key_start) for span in negatives],
        dtype=float,
    )
    needle_per_token = np.asarray(
        [
            mass / max(1, int(span.model_token_length))
            for mass, span in zip(needle_mass, needle_spans)
        ]
    )
    negative_per_token = np.asarray(
        [
            mass / max(1, int(span.model_token_length))
            for mass, span in zip(negative_mass, negatives)
        ]
    )
    total_mass = float(needle_mass.sum())
    coverage = _coverage(needle_mass, epsilon)
    length_normalized_coverage = _coverage(needle_per_token, epsilon)
    mean_mass = float(needle_mass.mean())
    mass_cv = (
        float(needle_mass.std(ddof=0) / mean_mass) if mean_mass > epsilon else math.nan
    )
    return {
        "needle_count": len(needle_spans),
        "broad_mass": total_mass,
        "broad_coverage": coverage,
        "broad_primary": total_mass * coverage,
        "broad_length_normalized_coverage": length_normalized_coverage,
        "broad_length_normalized_primary": (total_mass * length_normalized_coverage),
        "broad_contrast": float(needle_per_token.mean() - negative_per_token.mean()),
        "mean_needle_mass": mean_mass,
        "mean_needle_mass_per_token": float(needle_per_token.mean()),
        "mean_hard_negative_mass": float(negative_mass.mean()),
        "mean_hard_negative_mass_per_token": float(negative_per_token.mean()),
        "needle_mass_cv": mass_cv,
        "needle_effective_number": coverage * len(needle_spans),
        "needle_span_masses": json.dumps(needle_mass.tolist()),
        "needle_span_mass_per_token": json.dumps(needle_per_token.tolist()),
        "hard_negative_span_masses": json.dumps(negative_mass.tolist()),
        "attention_row_sum": float(values.sum()),
        "attention_key_start": int(key_start),
        "attention_key_length": len(values),
    }


def _attention_numpy_dtype(name: str) -> np.dtype[Any]:
    dtype = np.dtype(str(name))
    if dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("V4 attention_save_dtype must be float16 or float32")
    return dtype


@torch.inference_mode()
def _collect_attention_encoding(
    model: Any,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    example_index: int,
) -> tuple[pd.DataFrame, list[torch.Tensor], list[int]]:
    attention_rows, key_starts, query_logits = query_attention_outputs(
        model,
        adapter,
        encoding,
    )
    first_tokens = [
        int(token_ids[0])
        for _, token_ids in encoding.count_candidate_answer_token_ids
        if token_ids
    ]
    single_token_distinct = (
        len(first_tokens) == len(encoding.count_candidate_answer_token_ids)
        and all(
            len(token_ids) == 1
            for _, token_ids in encoding.count_candidate_answer_token_ids
        )
        and len(set(first_tokens)) == len(first_tokens)
    )
    behavior = (
        count_logit_metrics(query_logits, encoding) if single_token_distinct else {}
    )
    behavior_status = (
        "single_token_candidate_softmax"
        if single_token_distinct
        else "deferred_joint_sequence_probability"
    )
    rows: list[dict[str, Any]] = []
    for layer, (layer_rows, key_start) in enumerate(zip(attention_rows, key_starts)):
        for head in range(layer_rows.shape[0]):
            metrics = broad_attention_metrics(
                layer_rows[head],
                encoding.needle_spans,
                encoding.hard_negative_spans,
                key_start=key_start,
            )
            rows.append(
                {
                    "example_index": int(example_index),
                    "stimulus_id": encoding.stimulus_id,
                    "design_variant": encoding.design_variant,
                    "model_label": encoding.model_label,
                    "seed": encoding.seed,
                    "split": encoding.split,
                    "count": encoding.count,
                    "sequence_length": encoding.sequence_length,
                    "query_position": encoding.query_position,
                    "layer": layer,
                    "head": head,
                    "layer_type": adapter.layer_types[layer],
                    "answer_behavior_status": behavior_status,
                    **behavior,
                    **metrics,
                }
            )
    if not rows:
        raise ValueError("No V4 attention rows were collected")
    return pd.DataFrame(rows), attention_rows, key_starts


@torch.inference_mode()
def collect_attention_rows(
    model: Any,
    adapter: DecoderAdapter,
    encodings: Iterable[PromptEncoding],
) -> pd.DataFrame:
    frames = [
        _collect_attention_encoding(
            model,
            adapter,
            encoding,
            example_index=example_index,
        )[0]
        for example_index, encoding in enumerate(encodings)
    ]
    if not frames:
        raise ValueError("No V4 attention rows were collected")
    return pd.concat(frames, ignore_index=True)


def _write_raw_attention_shard(
    path: Path,
    *,
    attention_rows: Sequence[torch.Tensor],
    key_starts: Sequence[int],
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    save_dtype: str,
) -> None:
    dtype = _attention_numpy_dtype(save_dtype)
    if len(attention_rows) != adapter.num_layers:
        raise RuntimeError("Raw attention capture has the wrong layer count")
    arrays: dict[str, np.ndarray] = {
        "key_starts": np.asarray(key_starts, dtype=np.int64),
        "query_position": np.asarray([encoding.query_position], dtype=np.int64),
        "sequence_length": np.asarray([encoding.sequence_length], dtype=np.int64),
        "layer_types": np.asarray(adapter.layer_types),
    }
    for layer, row in enumerate(attention_rows):
        values = row.detach().cpu().numpy().astype(dtype, copy=False)
        if values.ndim != 2 or values.shape[0] != adapter.num_heads[layer]:
            raise RuntimeError(
                f"Invalid raw attention shape at layer {layer}: {values.shape}"
            )
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite raw attention at layer {layer}")
        arrays[f"layer_{layer:03d}"] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        # Deliberately uncompressed: float16 query rows are already compact,
        # and compression would make the 1,200-prompt capture CPU-bound.
        np.savez(handle, **arrays)
    temporary.replace(path)


def _validate_raw_attention_shard(
    path: Path,
    *,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
) -> None:
    expected = {
        "key_starts",
        "query_position",
        "sequence_length",
        "layer_types",
        *(f"layer_{layer:03d}" for layer in range(adapter.num_layers)),
    }
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != expected:
            raise RuntimeError(f"Incomplete V4 raw-attention shard: {path}")
        if int(saved["query_position"][0]) != int(encoding.query_position):
            raise RuntimeError(f"Raw-attention query mismatch: {path}")
        if int(saved["sequence_length"][0]) != int(encoding.sequence_length):
            raise RuntimeError(f"Raw-attention length mismatch: {path}")
        key_starts = np.asarray(saved["key_starts"], dtype=int)
        if key_starts.shape != (adapter.num_layers,):
            raise RuntimeError(f"Raw-attention key-start mismatch: {path}")
        for layer in range(adapter.num_layers):
            values = np.asarray(saved[f"layer_{layer:03d}"])
            if (
                values.ndim != 2
                or values.shape[0] != adapter.num_heads[layer]
                or values.shape[1]
                != int(encoding.query_position) + 1 - int(key_starts[layer])
                or not np.isfinite(values).all()
            ):
                raise RuntimeError(f"Invalid raw-attention layer {layer} in {path}")


@torch.inference_mode()
def capture_attention_shards(
    model: Any,
    adapter: DecoderAdapter,
    encodings: Iterable[PromptEncoding],
    *,
    output_dir: str | Path,
    save_raw_rows: bool = True,
    save_dtype: str = "float16",
    overwrite: bool = False,
) -> Path:
    """Capture restartable per-prompt metrics and raw answer-query rows."""

    output = Path(output_dir)
    _attention_numpy_dtype(save_dtype)
    index_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for example_index, encoding in enumerate(encodings):
        if encoding.stimulus_id in seen:
            raise ValueError(f"Duplicate attention stimulus: {encoding.stimulus_id}")
        seen.add(encoding.stimulus_id)
        relative = (
            Path("shards") / encoding.design_variant / f"{encoding.stimulus_id}.csv.gz"
        )
        shard = output / relative
        raw_relative = (
            Path("raw_shards") / encoding.design_variant / f"{encoding.stimulus_id}.npz"
        )
        raw_shard = output / raw_relative
        complete = shard.exists() and (not save_raw_rows or raw_shard.exists())
        if complete and not overwrite:
            frame = pd.read_csv(shard, compression="gzip")
            if frame.empty or set(frame["stimulus_id"]) != {encoding.stimulus_id}:
                raise RuntimeError(f"Invalid V4 attention shard: {shard}")
            if save_raw_rows:
                _validate_raw_attention_shard(
                    raw_shard,
                    adapter=adapter,
                    encoding=encoding,
                )
        else:
            frame, raw_rows, key_starts = _collect_attention_encoding(
                model,
                adapter,
                encoding,
                example_index=example_index,
            )
            frame["example_index"] = int(example_index)
            shard.parent.mkdir(parents=True, exist_ok=True)
            temporary = shard.with_name(shard.name + ".tmp")
            frame.to_csv(
                temporary,
                index=False,
                compression="gzip",
            )
            temporary.replace(shard)
            if save_raw_rows:
                _write_raw_attention_shard(
                    raw_shard,
                    attention_rows=raw_rows,
                    key_starts=key_starts,
                    adapter=adapter,
                    encoding=encoding,
                    save_dtype=save_dtype,
                )
        first = frame.iloc[0]
        index_rows.append(
            {
                "stimulus_id": encoding.stimulus_id,
                "design_variant": encoding.design_variant,
                "model_label": encoding.model_label,
                "seed": int(encoding.seed),
                "split": encoding.split,
                "count": int(encoding.count),
                "rows": len(frame),
                "shard_path": relative.as_posix(),
                "raw_attention_saved": bool(save_raw_rows),
                "raw_attention_dtype": str(save_dtype) if save_raw_rows else None,
                "raw_attention_shard_path": (
                    raw_relative.as_posix() if save_raw_rows else None
                ),
                "raw_attention_bytes": (
                    int(raw_shard.stat().st_size) if save_raw_rows else 0
                ),
                "answer_behavior_status": str(first["answer_behavior_status"]),
                **{
                    column: (
                        int(first[column])
                        if column
                        in {
                            "gold_count",
                            "predicted_count_among_candidates",
                        }
                        else float(first[column])
                        if column
                        in {
                            "correct_count_logit",
                            "correct_count_margin",
                            "correct_count_probability",
                            "expected_count",
                        }
                        else str(first[column])
                    )
                    for column in BEHAVIOR_COLUMNS
                    if column in frame.columns
                },
            }
        )
        print(
            "[v4 attention] "
            f"{example_index + 1} {encoding.design_variant} "
            f"seed={encoding.seed} N={encoding.count}",
            flush=True,
        )
    if not index_rows:
        raise ValueError("No V4 attention encodings were supplied")
    index_path = output / "attention_capture_index.jsonl"
    temporary = index_path.with_name(index_path.name + ".tmp")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(index_path)
    return index_path


def load_attention_shards(index_path: str | Path) -> pd.DataFrame:
    path = Path(index_path)
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("The V4 attention capture index is empty")
    frames = [
        pd.read_csv(
            path.parent / str(row["shard_path"]),
            compression="gzip",
        )
        for row in records
    ]
    return pd.concat(frames, ignore_index=True)


def summarize_attention_rows(
    detail: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "design_variant",
        "split",
        "count",
        "layer",
        "head",
        "layer_type",
        "broad_primary",
        "broad_mass",
        "broad_coverage",
        "broad_contrast",
    }
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"Attention table is missing columns: {missing}")
    metrics = [
        "broad_primary",
        "broad_mass",
        "broad_coverage",
        "broad_length_normalized_primary",
        "broad_length_normalized_coverage",
        "broad_contrast",
        "mean_needle_mass_per_token",
        "mean_hard_negative_mass_per_token",
        "needle_mass_cv",
        "needle_effective_number",
    ]

    def aggregate(groups: list[str]) -> pd.DataFrame:
        grouped = detail.groupby(groups, as_index=False, dropna=False)
        mean = grouped[metrics].mean()
        std = (
            grouped[metrics]
            .std(ddof=1)
            .rename(columns={metric: f"{metric}_sd" for metric in metrics})
        )
        count = grouped.size().rename(columns={"size": "examples"})
        merged = mean.merge(std, on=groups, how="left").merge(
            count, on=groups, how="left"
        )
        for metric in metrics:
            merged[f"{metric}_sem"] = merged[f"{metric}_sd"] / np.sqrt(
                merged["examples"].clip(lower=1)
            )
        return merged

    summary = aggregate(
        [
            "model_label",
            "design_variant",
            "split",
            "layer",
            "head",
            "layer_type",
        ]
    )
    by_count = aggregate(
        [
            "model_label",
            "design_variant",
            "split",
            "count",
            "layer",
            "head",
            "layer_type",
        ]
    )
    return summary, by_count


def rank_broad_heads(
    summary: pd.DataFrame,
    *,
    design_variant: str,
    metric: str = "broad_primary",
    split: str = "discovery",
    require_positive_contrast: bool = True,
) -> list[Head]:
    frame = summary[
        (summary["split"] == split) & (summary["design_variant"] == str(design_variant))
    ].copy()
    if frame.empty:
        raise ValueError(
            f"No attention summary rows for variant={design_variant!r}, split={split!r}"
        )
    if metric not in frame:
        raise ValueError(f"Unknown attention ranking metric: {metric}")
    if require_positive_contrast:
        eligible = frame[frame["broad_contrast"] > 0]
        if not eligible.empty:
            frame = eligible
    frame = frame.sort_values(
        [metric, "broad_contrast", "layer", "head"],
        ascending=[False, False, True, True],
    )
    return [(int(row.layer), int(row.head)) for row in frame.itertuples(index=False)]


def matched_random_heads(
    selected: Sequence[Head],
    adapter: DecoderAdapter,
    *,
    seed: int,
    excluded: Sequence[Head] = (),
    exclude_selected: bool = True,
) -> list[Head]:
    """Sample an exact per-layer matched head bank.

    ``excluded`` removes explicitly forbidden heads.  By default selected
    heads are also removed, preserving the legacy ablation control.  A
    phenotype-vs-random top-k study can set ``exclude_selected=False`` to draw
    an unbiased layer-stratified subset from the complete layer population;
    overlap with the ranked set is then measured and retained in the output.
    Sampling never uses replacement.
    """

    rng = random.Random(int(seed))
    selected_set = {(int(layer), int(head)) for layer, head in selected}
    excluded_set = {(int(layer), int(head)) for layer, head in excluded}
    result: list[Head] = []
    counts_by_layer: dict[int, int] = {}
    for layer, _head in selected:
        counts_by_layer[int(layer)] = counts_by_layer.get(int(layer), 0) + 1
    for layer, requested in counts_by_layer.items():
        candidates = [
            (layer, head)
            for head in range(adapter.num_heads[layer])
            if (not exclude_selected or (layer, head) not in selected_set)
            and (layer, head) not in excluded_set
        ]
        if len(candidates) < requested:
            raise RuntimeError(
                "Cannot construct a unique, non-selected layer-matched "
                f"control for layer {layer}: need {requested}, "
                f"available {len(candidates)}"
            )
        result.extend(rng.sample(candidates, k=requested))
    return result


def save_head_ranking(ranking: Sequence[Head], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "ranking": [
                    {"layer": int(layer), "head": int(head)} for layer, head in ranking
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def load_head_ranking(path: str | Path) -> list[Head]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(int(item["layer"]), int(item["head"])) for item in payload["ranking"]]


def analyze_attention_table(
    detail: pd.DataFrame,
    *,
    output_dir: str | Path,
    metric: str = "broad_primary",
    top_k_stability: int = 8,
) -> dict[str, Any]:
    """Write summaries, discovery-only rankings, and broadness plots."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "attention_head_detail.csv"
    summary_path = output / "attention_head_summary.csv"
    by_count_path = output / "attention_head_by_count.csv"
    detail.to_csv(detail_path, index=False)
    summary, by_count = summarize_attention_rows(detail)
    summary.to_csv(summary_path, index=False)
    by_count.to_csv(by_count_path, index=False)

    missing_behavior = sorted(set(BEHAVIOR_COLUMNS) - set(detail.columns))
    if missing_behavior:
        raise ValueError(
            f"Attention table is missing answer-query outcomes: {missing_behavior}"
        )
    behavior = (
        detail.sort_values(["design_variant", "seed", "count", "layer", "head"])
        .drop_duplicates("stimulus_id")[
            [
                "stimulus_id",
                "design_variant",
                "model_label",
                "seed",
                "split",
                "count",
                "sequence_length",
                "query_position",
                *BEHAVIOR_COLUMNS,
            ]
        ]
        .copy()
    )
    behavior["candidate_correct"] = (
        behavior["predicted_count_among_candidates"].astype(int)
        == behavior["gold_count"].astype(int)
    ).astype(float)
    behavior["expected_count_absolute_error"] = np.abs(
        behavior["expected_count"].astype(float) - behavior["gold_count"].astype(float)
    )
    behavior_path = output / "answer_query_behavior.csv"
    behavior.to_csv(behavior_path, index=False)
    behavior_groups = ["model_label", "design_variant", "split", "count"]
    behavior_summary = behavior.groupby(
        behavior_groups,
        as_index=False,
    ).agg(
        examples=("stimulus_id", "count"),
        seeds=("seed", "nunique"),
        candidate_accuracy=("candidate_correct", "mean"),
        mean_correct_count_probability=("correct_count_probability", "mean"),
        mean_correct_count_margin=("correct_count_margin", "mean"),
        mean_expected_count=("expected_count", "mean"),
        mean_expected_count_absolute_error=(
            "expected_count_absolute_error",
            "mean",
        ),
    )
    behavior_summary_path = output / "answer_query_behavior_by_count.csv"
    behavior_summary.to_csv(behavior_summary_path, index=False)

    # N=1 has coverage identically equal to one and cannot identify broad
    # aggregation. Discovery rankings therefore average N=2..10 only.
    ranking_detail = detail[
        (detail["split"] == "discovery") & (detail["count"] >= 2)
    ].copy()
    ranking_summary, _ = summarize_attention_rows(ranking_detail)
    variants = sorted(str(value) for value in detail["design_variant"].unique())
    rankings: dict[str, list[Head]] = {}
    ranking_paths: dict[str, str] = {}
    figure_paths: list[str] = []
    for variant in variants:
        ranking = rank_broad_heads(
            ranking_summary,
            design_variant=variant,
            metric=metric,
            split="discovery",
        )
        rankings[variant] = ranking
        ranking_path = output / f"head_ranking_{variant.replace('.', '_')}.json"
        save_head_ranking(ranking, ranking_path)
        ranking_paths[variant] = str(ranking_path)

        frame = ranking_summary[
            (ranking_summary["design_variant"] == variant)
            & (ranking_summary["split"] == "discovery")
        ]
        heatmap = frame.pivot(index="layer", columns="head", values=metric)
        figure, axis = plt.subplots(figsize=(max(7.0, 0.3 * heatmap.shape[1]), 7.0))
        image = axis.imshow(
            heatmap.to_numpy(dtype=float),
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
        )
        axis.set_title(f"{variant}: discovery {metric} (N=2..10)")
        axis.set_xlabel("head")
        axis.set_ylabel("layer")
        axis.set_xticks(np.arange(heatmap.shape[1]))
        axis.set_xticklabels(heatmap.columns.astype(int), fontsize=7)
        layer_tick_stride = max(1, heatmap.shape[0] // 16)
        tick_indices = np.arange(0, heatmap.shape[0], layer_tick_stride)
        axis.set_yticks(tick_indices)
        axis.set_yticklabels(
            heatmap.index.to_numpy()[tick_indices].astype(int), fontsize=7
        )
        figure.colorbar(image, ax=axis, label=metric)
        figure.tight_layout()
        figure_path = output / (f"broad_head_heatmap_{variant.replace('.', '_')}.png")
        figure.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        figure_paths.append(str(figure_path))

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, variant in zip(axes.flat, variants):
        frame = behavior_summary[behavior_summary["design_variant"] == variant]
        for split, color in (("discovery", "#4C78A8"), ("confirmation", "#E45756")):
            selected = frame[frame["split"] == split].sort_values("count")
            if not selected.empty:
                axis.plot(
                    selected["count"],
                    selected["mean_expected_count"],
                    marker="o",
                    color=color,
                    label=split,
                )
        counts = sorted(int(value) for value in frame["count"].unique())
        if counts:
            axis.plot(counts, counts, color="black", linestyle="--", alpha=0.45)
        axis.set_title(variant)
        axis.set_xlabel("gold count")
        axis.set_ylabel("candidate-softmax expected count")
        axis.grid(alpha=0.15)
        axis.legend()
    figure.suptitle("Non-thinking answer-query behavior")
    figure.tight_layout()
    behavior_figure_path = output / "answer_query_expected_count.png"
    figure.savefig(behavior_figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    figure_paths.append(str(behavior_figure_path))

    # Show whether each top head is broad over all ten spans on held-out seeds.
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, variant in zip(axes.flat, variants):
        top_layer, top_head = rankings[variant][0]
        selected = detail[
            (detail["design_variant"] == variant)
            & (detail["split"] == "confirmation")
            & (detail["count"] == 10)
            & (detail["layer"] == top_layer)
            & (detail["head"] == top_head)
        ]
        masses = [
            np.asarray(json.loads(value), dtype=float)
            for value in selected["needle_span_masses"]
        ]
        if masses:
            matrix = np.stack(masses)
            mean = matrix.mean(axis=0)
            sem = matrix.std(axis=0, ddof=1) / math.sqrt(len(matrix))
            axis.bar(
                np.arange(1, 11),
                mean,
                yerr=sem,
                color="#4C78A8",
                alpha=0.85,
                capsize=2,
            )
        axis.set_title(f"{variant}: L{top_layer}H{top_head}")
        axis.set_xlabel("needle occurrence index")
        axis.set_ylabel("answer-query attention mass")
        axis.grid(axis="y", alpha=0.15)
    figure.suptitle("Held-out N=10 span masses for the top discovery broad head")
    figure.tight_layout()
    mass_figure_path = output / "top_broad_head_span_masses.png"
    figure.savefig(mass_figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    figure_paths.append(str(mass_figure_path))

    stability_rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(variants):
        for right in variants[left_index + 1 :]:
            left_top = set(rankings[left][: int(top_k_stability)])
            right_top = set(rankings[right][: int(top_k_stability)])
            union = left_top | right_top
            stability_rows.append(
                {
                    "left_variant": left,
                    "right_variant": right,
                    "top_k": int(top_k_stability),
                    "intersection": len(left_top & right_top),
                    "jaccard": (
                        len(left_top & right_top) / len(union) if union else math.nan
                    ),
                }
            )
    stability_path = output / "head_ranking_stability.csv"
    pd.DataFrame(stability_rows).to_csv(stability_path, index=False)
    manifest_path = output / "attention_analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "realistic_niah_v4_attention_analysis_v1",
                "ranking_split": "discovery",
                "ranking_counts": list(range(2, 11)),
                "ranking_metric": metric,
                "require_positive_contrast": True,
                "rankings": ranking_paths,
                "figures": figure_paths,
                "answer_query_behavior": str(behavior_path),
                "answer_query_behavior_by_count": str(behavior_summary_path),
                "confirmation_use": (
                    "visualization and causal validation only; never head selection"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "detail": detail_path,
        "summary": summary_path,
        "by_count": by_count_path,
        "behavior": behavior_path,
        "behavior_by_count": behavior_summary_path,
        "rankings": rankings,
        "ranking_paths": ranking_paths,
        "stability": stability_path,
        "manifest": manifest_path,
    }
