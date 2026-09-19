from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .attention import Head
from .prompts import PromptEncoding, TokenSpan


POOLINGS = ("span_end", "span_mean", "span_sum")
POOL_METRICS = (
    "pool_sum",
    "pool_mean",
    "pool_coverage",
    "pool_primary",
    "pool_contrast",
    "pool_enrichment",
    "pool_cv",
    "pool_effective_number",
    "pool_min",
    "pool_max",
    "pool_min_to_mean",
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv_gzip_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Empty JSONL input: {path}")
    return rows


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    raise ValueError(f"Cannot interpret value as a strict boolean: {value!r}")


def _span_arrays(
    rows: np.ndarray,
    spans: Sequence[TokenSpan],
    *,
    key_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return end weights, span means, span sums, and visibility by span.

    The three reductions answer different questions and must not be conflated:

    - ``span_end`` is the attention weight on the final token only;
    - ``span_mean`` is attention density per model token in the complete span;
    - ``span_sum`` is the literal fraction of the query row assigned anywhere
      inside the complete span.
    """

    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Expected [heads, keys] attention rows")
    ends: list[np.ndarray] = []
    means: list[np.ndarray] = []
    sums: list[np.ndarray] = []
    visible: list[bool] = []
    key_end = int(key_start) + values.shape[1]
    for span in spans:
        full = int(span.start) >= int(key_start) and int(span.end) <= key_end
        visible.append(full)
        if full:
            local_start = int(span.start) - int(key_start)
            local_end = int(span.end) - int(key_start)
            segment = values[:, local_start:local_end]
            ends.append(values[:, local_end - 1])
            means.append(segment.mean(axis=1, dtype=np.float32))
            sums.append(segment.sum(axis=1, dtype=np.float32))
        else:
            ends.append(np.zeros(values.shape[0], dtype=np.float32))
            means.append(np.zeros(values.shape[0], dtype=np.float32))
            sums.append(np.zeros(values.shape[0], dtype=np.float32))
    if not spans:
        return (
            np.zeros((values.shape[0], 0), dtype=np.float32),
            np.zeros((values.shape[0], 0), dtype=np.float32),
            np.zeros((values.shape[0], 0), dtype=np.float32),
            np.zeros(0, dtype=bool),
        )
    return (
        np.stack(ends, axis=1),
        np.stack(means, axis=1),
        np.stack(sums, axis=1),
        np.asarray(visible, dtype=bool),
    )


def _pool_scalar_arrays(
    evidence: np.ndarray,
    negatives: np.ndarray,
    *,
    row_sums: np.ndarray,
    key_length: int,
    density_evidence: np.ndarray | None = None,
    density_negatives: np.ndarray | None = None,
    epsilon: float = 1e-12,
) -> dict[str, np.ndarray]:
    values = np.asarray(evidence, dtype=np.float64)
    controls = np.asarray(negatives, dtype=np.float64)
    if values.ndim != 2 or controls.shape != values.shape or values.shape[1] == 0:
        raise ValueError("Needle and hard-negative evidence must be [heads, N]")
    density_values = (
        values
        if density_evidence is None
        else np.asarray(density_evidence, dtype=np.float64)
    )
    density_controls = (
        controls
        if density_negatives is None
        else np.asarray(density_negatives, dtype=np.float64)
    )
    if density_values.shape != values.shape or density_controls.shape != values.shape:
        raise ValueError("Density evidence must align with occurrence evidence")
    total = values.sum(axis=1)
    mean = values.mean(axis=1)
    probabilities = values / np.maximum(total[:, None], epsilon)
    entropy = -np.sum(
        np.where(
            probabilities > 0,
            probabilities * np.log(np.maximum(probabilities, epsilon)),
            0.0,
        ),
        axis=1,
    )
    coverage = np.where(
        total > epsilon,
        np.exp(entropy) / values.shape[1],
        0.0,
    )
    std = values.std(axis=1, ddof=0)
    global_mean = np.asarray(row_sums, dtype=np.float64) / max(1, int(key_length))
    minimum = values.min(axis=1)
    maximum = values.max(axis=1)
    return {
        "pool_sum": total,
        "pool_mean": mean,
        "pool_coverage": coverage,
        "pool_primary": total * coverage,
        # Contrast and enrichment are density comparisons.  In particular,
        # span-sum keeps literal full-span mass for total/coverage/primary but
        # uses span-mean density here so a longer record is not automatically
        # declared more needle-selective than its matched negative.
        "pool_contrast": density_values.mean(axis=1)
        - density_controls.mean(axis=1),
        "pool_enrichment": density_values.mean(axis=1)
        / np.maximum(global_mean, epsilon),
        "pool_cv": np.where(mean > epsilon, std / mean, np.nan),
        "pool_effective_number": coverage * values.shape[1],
        "pool_min": minimum,
        "pool_max": maximum,
        "pool_min_to_mean": minimum / np.maximum(mean, epsilon),
    }


def layer_pooling_metrics(
    attention_rows: np.ndarray,
    needle_spans: Sequence[TokenSpan],
    hard_negative_spans: Sequence[TokenSpan],
    *,
    key_start: int,
) -> dict[str, dict[str, Any]]:
    """Compute end, mean-density, and literal full-span broadness metrics.

    Span-end uses one attention weight per occurrence. Span-mean first averages
    attention over every token in an occurrence, preventing longer realistic
    records from receiving an automatic advantage. Span-sum instead preserves
    the literal attention mass assigned to all tokens in each record. All
    three then apply the V10 mass/coverage/effective-number logic across
    occurrences; span-sum eligibility controls remain length-normalized.
    """

    rows = np.asarray(attention_rows, dtype=np.float32)
    negative_by_slot = {int(span.slot_index): span for span in hard_negative_spans}
    negatives = [negative_by_slot[int(span.slot_index)] for span in needle_spans]
    if len(negatives) != len(needle_spans):
        raise ValueError("Every active needle requires a matched hard negative")
    needle_end, needle_mean, needle_sum, needle_visible = _span_arrays(
        rows, needle_spans, key_start=key_start
    )
    negative_end, negative_mean, negative_sum, negative_visible = _span_arrays(
        rows, negatives, key_start=key_start
    )
    row_sums = rows.sum(axis=1, dtype=np.float64)
    base = {
        "attention_row_sum": row_sums,
        "attention_key_start": int(key_start),
        "attention_key_length": int(rows.shape[1]),
        "visible_needle_count": int(needle_visible.sum()),
        "all_needles_visible": bool(needle_visible.all()),
        "all_hard_negatives_visible": bool(negative_visible.all()),
    }
    result: dict[str, dict[str, Any]] = {}
    for pooling, evidence, controls, density_evidence, density_controls in (
        ("span_end", needle_end, negative_end, needle_end, negative_end),
        ("span_mean", needle_mean, negative_mean, needle_mean, negative_mean),
        ("span_sum", needle_sum, negative_sum, needle_mean, negative_mean),
    ):
        result[pooling] = {
            **base,
            **_pool_scalar_arrays(
                evidence,
                controls,
                row_sums=row_sums,
                key_length=rows.shape[1],
                density_evidence=density_evidence,
                density_negatives=density_controls,
            ),
            "needle_occurrence_values": evidence,
            "hard_negative_occurrence_values": controls,
        }
    return result


def _label_table(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
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
        "omission_count",
    }
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Generation label table is missing columns: {missing}")
    if labels["stimulus_id"].duplicated().any():
        raise ValueError("Generation label table contains duplicate stimulus IDs")
    if not set(labels["outcome_group"]).issubset({"correct", "wrong", "invalid"}):
        raise ValueError("Generation label table contains an unknown outcome")
    return labels


def _validate_capture_grid(
    records: Sequence[dict[str, Any]],
    labels: pd.DataFrame,
    encodings: Mapping[str, PromptEncoding],
) -> None:
    capture_ids = [str(row["stimulus_id"]) for row in records]
    if len(capture_ids) != len(set(capture_ids)):
        raise ValueError("Attention capture index contains duplicate stimulus IDs")
    capture_set = set(capture_ids)
    label_set = set(labels["stimulus_id"].astype(str))
    encoding_set = set(encodings)
    if capture_set != label_set or capture_set != encoding_set:
        raise ValueError(
            "Attention, generation-label, and rendered-prompt grids disagree: "
            f"attention={len(capture_set)} labels={len(label_set)} "
            f"encodings={len(encoding_set)}"
        )
    cells = {
        (
            str(row["design_variant"]),
            int(row["seed"]),
            int(row["count"]),
        )
        for row in records
    }
    if len(cells) != len(records):
        raise ValueError("Attention capture grid is not unique by variant/seed/count")
    labels_by_id = labels.set_index("stimulus_id", drop=False)
    for record in records:
        stimulus_id = str(record["stimulus_id"])
        encoding = encodings[stimulus_id]
        label = labels_by_id.loc[stimulus_id]
        expected = {
            "design_variant": encoding.design_variant,
            "model_label": encoding.model_label,
            "seed": int(encoding.seed),
            "count": int(encoding.count),
        }
        observed = {
            "design_variant": str(record["design_variant"]),
            "model_label": str(record["model_label"]),
            "seed": int(record["seed"]),
            "count": int(record["count"]),
        }
        if observed != expected:
            raise ValueError(
                f"Attention metadata mismatch for {stimulus_id}: "
                f"observed={observed} expected={expected}"
            )
        if (
            str(label["design_variant"]) != encoding.design_variant
            or str(label["model_label"]) != encoding.model_label
            or int(label["seed"]) != int(encoding.seed)
            or int(label["gold_count"]) != int(encoding.count)
        ):
            raise ValueError(f"Behavior metadata mismatch for {stimulus_id}")


def capture_pooling_metric_shards(
    *,
    attention_index_path: str | Path,
    generation_labels_path: str | Path,
    encodings: Iterable[PromptEncoding],
    output_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """Stream raw answer-query rows into restartable pooled metric shards."""

    attention_index = Path(attention_index_path)
    capture_root = attention_index.parent
    records = _read_jsonl(attention_index)
    labels = _label_table(Path(generation_labels_path))
    labels_by_id = labels.set_index("stimulus_id", drop=False)
    encoding_by_id = {item.stimulus_id: item for item in encodings}
    _validate_capture_grid(records, labels, encoding_by_id)
    output = Path(output_dir)
    index_rows: list[dict[str, Any]] = []
    for example_index, record in enumerate(records):
        stimulus_id = str(record["stimulus_id"])
        encoding = encoding_by_id[stimulus_id]
        label = labels_by_id.loc[stimulus_id]
        relative = (
            Path("shards") / encoding.design_variant / f"{encoding.stimulus_id}.csv.gz"
        )
        shard = output / relative
        raw_value = record.get("raw_attention_shard_path")
        if not raw_value:
            raise RuntimeError(f"Raw attention row is unavailable for {stimulus_id}")
        raw_path = capture_root / str(raw_value)
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        if shard.exists() and not overwrite:
            frame = pd.read_csv(shard, compression="gzip")
            if (
                frame.empty
                or set(frame["stimulus_id"]) != {stimulus_id}
                or set(frame["pooling"]) != set(POOLINGS)
            ):
                raise RuntimeError(f"Invalid pooling-metric shard: {shard}")
        else:
            frames: list[pd.DataFrame] = []
            with np.load(raw_path, allow_pickle=False) as raw:
                if int(raw["query_position"][0]) != int(encoding.query_position):
                    raise RuntimeError(f"Raw query-position mismatch: {raw_path}")
                if int(raw["sequence_length"][0]) != int(
                    encoding.sequence_length
                ):
                    raise RuntimeError(f"Raw sequence-length mismatch: {raw_path}")
                key_starts = np.asarray(raw["key_starts"], dtype=int)
                layer_types = np.asarray(raw["layer_types"]).astype(str)
                for layer in range(len(key_starts)):
                    rows = np.asarray(raw[f"layer_{layer:03d}"], dtype=np.float32)
                    metrics = layer_pooling_metrics(
                        rows,
                        encoding.needle_spans,
                        encoding.hard_negative_spans,
                        key_start=int(key_starts[layer]),
                    )
                    for pooling in POOLINGS:
                        values = metrics[pooling]
                        head_count = rows.shape[0]
                        data: dict[str, Any] = {
                            "stimulus_id": np.repeat(stimulus_id, head_count),
                            "design_variant": np.repeat(
                                encoding.design_variant, head_count
                            ),
                            "model_label": np.repeat(encoding.model_label, head_count),
                            "seed": np.repeat(int(encoding.seed), head_count),
                            "split": np.repeat(encoding.split, head_count),
                            "count": np.repeat(int(encoding.count), head_count),
                            "sequence_length": np.repeat(
                                int(encoding.sequence_length), head_count
                            ),
                            "query_position": np.repeat(
                                int(encoding.query_position), head_count
                            ),
                            "layer": np.repeat(layer, head_count),
                            "head": np.arange(head_count, dtype=int),
                            "layer_type": np.repeat(layer_types[layer], head_count),
                            "pooling": np.repeat(pooling, head_count),
                            "outcome_group": np.repeat(
                                str(label["outcome_group"]), head_count
                            ),
                            "is_correct": np.repeat(
                                _as_bool(label["is_correct"]), head_count
                            ),
                            "format_valid": np.repeat(
                                _as_bool(label["format_valid"]), head_count
                            ),
                            "predicted_count": np.repeat(
                                label["parsed_count"], head_count
                            ),
                            "count_error": np.repeat(label["count_error"], head_count),
                            "omission_count": np.repeat(
                                label["omission_count"], head_count
                            ),
                            "visible_needle_count": np.repeat(
                                values["visible_needle_count"], head_count
                            ),
                            "all_needles_visible": np.repeat(
                                values["all_needles_visible"], head_count
                            ),
                            "all_hard_negatives_visible": np.repeat(
                                values["all_hard_negatives_visible"], head_count
                            ),
                            "attention_key_start": np.repeat(
                                values["attention_key_start"], head_count
                            ),
                            "attention_key_length": np.repeat(
                                values["attention_key_length"], head_count
                            ),
                            "attention_row_sum": values["attention_row_sum"],
                        }
                        for metric in POOL_METRICS:
                            data[metric] = values[metric]
                        frames.append(pd.DataFrame(data))
            frame = pd.concat(frames, ignore_index=True)
            _write_csv_gzip_atomic(frame, shard)
        index_rows.append(
            {
                "stimulus_id": stimulus_id,
                "design_variant": encoding.design_variant,
                "seed": int(encoding.seed),
                "count": int(encoding.count),
                "rows": len(frame),
                "shard_path": relative.as_posix(),
            }
        )
        if (example_index + 1) % 10 == 0 or example_index == 0:
            print(
                f"[v4 attention analyze] pooling metrics "
                f"{example_index + 1}/{len(records)} {stimulus_id}",
                flush=True,
            )
    index_path = output / "pooling_metric_index.jsonl"
    temporary = index_path.with_name(index_path.name + ".tmp")
    output.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(index_path)
    return index_path


def load_pooling_metric_shards(index_path: str | Path) -> pd.DataFrame:
    path = Path(index_path)
    records = _read_jsonl(path)
    return pd.concat(
        [
            pd.read_csv(path.parent / str(row["shard_path"]), compression="gzip")
            for row in records
        ],
        ignore_index=True,
    )


def rank_broad_candidates(
    detail: pd.DataFrame,
    *,
    top_k: int = 8,
) -> tuple[pd.DataFrame, dict[tuple[str, str], list[Head]]]:
    """Rank discovery heads without silently accepting negative controls."""

    discovery = detail[
        (detail["split"] == "discovery")
        & (detail["count"] >= 2)
    ].copy()
    groups = [
        "model_label",
        "design_variant",
        "pooling",
        "layer",
        "head",
        "layer_type",
    ]
    discovery["full_visibility"] = (
        discovery["all_needles_visible"].astype(bool)
        & discovery["all_hard_negatives_visible"].astype(bool)
    )
    visibility = discovery.groupby(groups, as_index=False).agg(
        ranking_grid_examples=("stimulus_id", "nunique"),
        full_visibility_rate=("full_visibility", "mean"),
    )
    eligible = discovery[discovery["full_visibility"]].copy()
    summary = eligible.groupby(groups, as_index=False).agg(
        examples=("stimulus_id", "nunique"),
        seeds=("seed", "nunique"),
        **{metric: (metric, "mean") for metric in POOL_METRICS},
    )
    summary = summary.merge(visibility, on=groups, how="left", validate="one_to_one")
    summary["positive_needle_control_contrast"] = summary["pool_contrast"] > 0
    summary["needle_density_enrichment_gt_one"] = summary["pool_enrichment"] > 1
    summary["is_broad_candidate"] = (
        np.isclose(summary["full_visibility_rate"], 1.0)
        & summary["positive_needle_control_contrast"]
        & summary["needle_density_enrichment_gt_one"]
    )
    summary["candidate_rank"] = np.nan
    rankings: dict[tuple[str, str], list[Head]] = {}
    for (variant, pooling), frame in summary.groupby(
        ["design_variant", "pooling"], sort=True
    ):
        candidates = frame[frame["is_broad_candidate"]].sort_values(
            ["pool_primary", "pool_coverage", "pool_contrast", "layer", "head"],
            ascending=[False, False, False, True, True],
        )
        indices = candidates.index.to_numpy()
        summary.loc[indices, "candidate_rank"] = np.arange(1, len(indices) + 1)
        rankings[(str(variant), str(pooling))] = [
            (int(row.layer), int(row.head))
            for row in candidates.head(int(top_k)).itertuples(index=False)
        ]
    return summary, rankings


def summarize_outcomes(detail: pd.DataFrame) -> pd.DataFrame:
    groups = [
        "model_label",
        "design_variant",
        "pooling",
        "split",
        "count",
        "outcome_group",
        "layer",
        "head",
        "layer_type",
    ]
    grouped = detail.groupby(groups, as_index=False, dropna=False)
    summary = grouped.agg(
        examples=("stimulus_id", "nunique"),
        seeds=("seed", "nunique"),
        **{metric: (metric, "mean") for metric in POOL_METRICS},
    )
    standard = grouped[list(POOL_METRICS)].std(ddof=1).rename(
        columns={metric: f"{metric}_sd" for metric in POOL_METRICS}
    )
    summary = summary.merge(standard, on=groups, how="left")
    for metric in POOL_METRICS:
        summary[f"{metric}_sem"] = summary[f"{metric}_sd"] / np.sqrt(
            summary["examples"].clip(lower=1)
        )
    return summary


def _head_occurrence_values(
    raw: Mapping[str, np.ndarray],
    encoding: PromptEncoding,
    head: Head,
    pooling: str,
) -> np.ndarray:
    layer, head_index = (int(head[0]), int(head[1]))
    rows = np.asarray(raw[f"layer_{layer:03d}"], dtype=np.float32)
    key_start = int(np.asarray(raw["key_starts"], dtype=int)[layer])
    metrics = layer_pooling_metrics(
        rows[head_index : head_index + 1],
        encoding.needle_spans,
        encoding.hard_negative_spans,
        key_start=key_start,
    )[pooling]
    if not metrics["all_needles_visible"]:
        raise RuntimeError(
            f"Selected broad head L{layer}H{head_index} cannot see every needle "
            f"in {encoding.stimulus_id}"
        )
    return np.asarray(metrics["needle_occurrence_values"][0], dtype=float)


def extract_ranked_occurrence_diagnostics(
    *,
    attention_index_path: str | Path,
    generation_labels_path: str | Path,
    encodings: Mapping[str, PromptEncoding],
    rankings: Mapping[tuple[str, str], Sequence[Head]],
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract occurrence profiles for discovery-selected candidate heads."""

    attention_index = Path(attention_index_path)
    capture_root = attention_index.parent
    records = _read_jsonl(attention_index)
    labels = _label_table(Path(generation_labels_path)).set_index(
        "stimulus_id", drop=False
    )
    occurrence_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    for example_index, record in enumerate(records):
        stimulus_id = str(record["stimulus_id"])
        encoding = encodings[stimulus_id]
        label = labels.loc[stimulus_id]
        raw_path = capture_root / str(record["raw_attention_shard_path"])
        with np.load(raw_path, allow_pickle=False) as raw:
            selected_layers = sorted(
                {
                    int(layer)
                    for pooling in POOLINGS
                    for layer, _head in rankings.get(
                        (encoding.design_variant, pooling), ()
                    )
                }
            )
            cached_raw: dict[str, np.ndarray] = {
                "key_starts": np.asarray(raw["key_starts"], dtype=int),
                **{
                    f"layer_{layer:03d}": np.asarray(
                        raw[f"layer_{layer:03d}"], dtype=np.float32
                    )
                    for layer in selected_layers
                },
            }
            for pooling in POOLINGS:
                heads = list(rankings.get((encoding.design_variant, pooling), ()))
                if not heads:
                    continue
                per_head = np.stack(
                    [
                        _head_occurrence_values(
                            cached_raw, encoding, head, pooling
                        )
                        for head in heads
                    ]
                )
                totals = per_head.sum(axis=1, keepdims=True)
                normalized = np.divide(
                    len(encoding.needle_spans) * per_head,
                    totals,
                    out=np.zeros_like(per_head),
                    where=totals > 1e-12,
                )
                ensemble_raw = per_head.mean(axis=0)
                ensemble_share = normalized.mean(axis=0)
                order = np.argsort(ensemble_share, kind="stable")
                ranks = np.empty(len(order), dtype=int)
                ranks[order] = np.arange(1, len(order) + 1)
                total = float(ensemble_raw.sum())
                if total > 1e-12:
                    probabilities = ensemble_raw / total
                    entropy = float(
                        -np.sum(
                            probabilities
                            * np.log(np.maximum(probabilities, 1e-12))
                        )
                    )
                    coverage = math.exp(entropy) / len(ensemble_raw)
                else:
                    coverage = 0.0
                omission_value = label["omission_count"]
                omission = (
                    int(omission_value) if pd.notna(omission_value) else None
                )
                predicted_value = label["parsed_count"]
                predicted = (
                    int(predicted_value) if pd.notna(predicted_value) else None
                )
                undercount = (
                    predicted is not None and predicted < len(encoding.needle_spans)
                )
                tail_indices = (
                    set(range(predicted, len(encoding.needle_spans)))
                    if undercount
                    else set()
                )
                bottom_indices = (
                    set(int(index) for index in order[:omission])
                    if omission is not None and omission > 0
                    else set()
                )
                candidate_slots = (
                    [
                        int(encoding.needle_spans[index].slot_index)
                        for index in order[:omission]
                    ]
                    if omission is not None and omission > 0
                    else []
                )
                prefix_share = (
                    float(ensemble_share[:predicted].mean())
                    if undercount and predicted > 0
                    else math.nan
                )
                tail_share = (
                    float(ensemble_share[predicted:].mean())
                    if undercount
                    else math.nan
                )
                prompt_rows.append(
                    {
                        "stimulus_id": stimulus_id,
                        "design_variant": encoding.design_variant,
                        "model_label": encoding.model_label,
                        "seed": int(encoding.seed),
                        "split": encoding.split,
                        "count": int(encoding.count),
                        "pooling": pooling,
                        "outcome_group": str(label["outcome_group"]),
                        "is_correct": _as_bool(label["is_correct"]),
                        "predicted_count": predicted_value,
                        "count_error": label["count_error"],
                        "omission_count": omission_value,
                        "selected_head_count": len(heads),
                        "selected_heads": json.dumps(
                            [
                                {"layer": int(layer), "head": int(head)}
                                for layer, head in heads
                            ]
                        ),
                        "ensemble_coverage": coverage,
                        "ensemble_effective_number": coverage
                        * len(ensemble_raw),
                        "minimum_normalized_share": float(ensemble_share.min()),
                        "maximum_normalized_share": float(ensemble_share.max()),
                        "undercount_tail_prefix_mean_share": prefix_share,
                        "undercount_tail_mean_share": tail_share,
                        "undercount_tail_to_prefix_ratio": (
                            tail_share / prefix_share
                            if np.isfinite(tail_share)
                            and np.isfinite(prefix_share)
                            and prefix_share > 1e-12
                            else math.nan
                        ),
                        "bottom_k_tail_overlap": (
                            len(bottom_indices & tail_indices)
                            if undercount
                            else None
                        ),
                        "bottom_k_tail_overlap_fraction": (
                            len(bottom_indices & tail_indices) / len(tail_indices)
                            if tail_indices
                            else math.nan
                        ),
                        "attention_implied_missed_slot_candidates": json.dumps(
                            candidate_slots
                        ),
                    }
                )
                for occurrence_index, (span, raw_value, share, rank) in enumerate(
                    zip(
                        encoding.needle_spans,
                        ensemble_raw,
                        ensemble_share,
                        ranks,
                    ),
                    start=1,
                ):
                    occurrence_rows.append(
                        {
                            "stimulus_id": stimulus_id,
                            "design_variant": encoding.design_variant,
                            "model_label": encoding.model_label,
                            "seed": int(encoding.seed),
                            "split": encoding.split,
                            "count": int(encoding.count),
                            "pooling": pooling,
                            "outcome_group": str(label["outcome_group"]),
                            "is_correct": _as_bool(label["is_correct"]),
                            "predicted_count": label["parsed_count"],
                            "count_error": label["count_error"],
                            "omission_count": omission_value,
                            "occurrence_index": int(occurrence_index),
                            "slot_index": int(span.slot_index),
                            "span_start": int(span.start),
                            "span_end": int(span.end),
                            "normalized_depth": float(
                                (span.end - 1) / max(1, encoding.query_position)
                            ),
                            "ensemble_raw_attention": float(raw_value),
                            "ensemble_normalized_share": float(share),
                            "low_attention_rank": int(rank),
                            "attention_implied_missed_candidate": bool(
                                omission is not None
                                and omission > 0
                                and rank <= omission
                            ),
                            "beyond_predicted_count_in_undercount": bool(
                                undercount and occurrence_index > int(predicted)
                            ),
                        }
                    )
        if (example_index + 1) % 25 == 0 or example_index == 0:
            print(
                f"[v4 attention analyze] occurrence profiles "
                f"{example_index + 1}/{len(records)} {stimulus_id}",
                flush=True,
            )
    occurrences = pd.DataFrame(occurrence_rows)
    prompts = pd.DataFrame(prompt_rows)
    if occurrences.empty or prompts.empty:
        raise RuntimeError("No ranked-head occurrence diagnostics were extracted")

    threshold_source = occurrences[
        (occurrences["split"] == "discovery")
        & (occurrences["outcome_group"] == "correct")
        & (occurrences["count"] >= 2)
    ]
    thresholds = (
        threshold_source.groupby(
            ["design_variant", "pooling", "count"], as_index=False
        )["ensemble_normalized_share"]
        .quantile(0.10)
        .rename(
            columns={
                "ensemble_normalized_share": "correct_discovery_q10_share"
            }
        )
    )
    occurrences = occurrences.merge(
        thresholds,
        on=["design_variant", "pooling", "count"],
        how="left",
        validate="many_to_one",
    )
    occurrences["low_attention_threshold_available"] = occurrences[
        "correct_discovery_q10_share"
    ].notna()
    occurrences["below_correct_discovery_q10"] = pd.Series(
        pd.NA, index=occurrences.index, dtype="boolean"
    )
    available = occurrences["low_attention_threshold_available"]
    occurrences.loc[available, "below_correct_discovery_q10"] = (
        occurrences.loc[available, "ensemble_normalized_share"]
        < occurrences.loc[available, "correct_discovery_q10_share"]
    )
    low_counts = (
        occurrences.groupby(["stimulus_id", "pooling"], as_index=False)
        .agg(
            low_attention_threshold_available=(
                "low_attention_threshold_available",
                "all",
            ),
            low_attention_occurrence_count=(
                "below_correct_discovery_q10",
                lambda values: values.astype("Int64").sum(min_count=1),
            ),
            bottom_occurrence_share=("ensemble_normalized_share", "min"),
        )
    )
    prompts = prompts.merge(
        low_counts,
        on=["stimulus_id", "pooling"],
        how="left",
        validate="one_to_one",
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv_gzip_atomic(occurrences, output / "occurrence_attention.csv.gz")
    prompts.to_csv(output / "omission_diagnostics.csv", index=False)
    thresholds.to_csv(output / "low_attention_thresholds.csv", index=False)
    return occurrences, prompts


def _count_adjusted_delta(frame: pd.DataFrame, metric: str) -> tuple[float, int]:
    deltas: list[float] = []
    weights: list[float] = []
    for _count, cell in frame.groupby("count"):
        correct = cell[cell["outcome_group"] == "correct"][metric].dropna()
        wrong = cell[cell["outcome_group"] == "wrong"][metric].dropna()
        if correct.empty or wrong.empty:
            continue
        deltas.append(float(wrong.mean() - correct.mean()))
        weights.append(float(2 * len(correct) * len(wrong) / (len(correct) + len(wrong))))
    if not deltas:
        return math.nan, 0
    return float(np.average(deltas, weights=weights)), len(deltas)


def confirmation_outcome_effects(
    detail: pd.DataFrame,
    rankings: Mapping[tuple[str, str], Sequence[Head]],
    *,
    bootstrap_replicates: int = 1000,
    seed: int = 90421,
) -> pd.DataFrame:
    """Estimate count-adjusted wrong-minus-correct effects on held-out seeds."""

    def vectorized_effects(
        selected: pd.DataFrame,
        metric: str,
        seeds: np.ndarray,
        *,
        replicates: int,
        rng: np.random.Generator,
    ) -> tuple[float, int, np.ndarray]:
        """Match seed-cluster resampling without repeated DataFrame concat."""

        if not len(seeds):
            return math.nan, 0, np.asarray([], dtype=float)
        counts = np.asarray(sorted(selected["count"].unique()), dtype=int)
        seed_index = {int(value): index for index, value in enumerate(seeds)}
        count_index = {int(value): index for index, value in enumerate(counts)}
        outcome_index = {"correct": 0, "wrong": 1}
        sums = np.zeros((len(seeds), len(counts), 2), dtype=float)
        sample_sizes = np.zeros_like(sums)
        finite = selected[np.isfinite(pd.to_numeric(selected[metric], errors="coerce"))]
        grouped = finite.groupby(["seed", "count", "outcome_group"])[metric].agg(
            ["sum", "count"]
        )
        for (seed_value, count_value, outcome), row in grouped.iterrows():
            seed_axis = seed_index[int(seed_value)]
            count_axis = count_index[int(count_value)]
            outcome_axis = outcome_index[str(outcome)]
            sums[seed_axis, count_axis, outcome_axis] = float(row["sum"])
            sample_sizes[seed_axis, count_axis, outcome_axis] = float(row["count"])

        def effects_from_multiplicity(multiplicity: np.ndarray) -> np.ndarray:
            total_sums = np.einsum("rs,sco->rco", multiplicity, sums)
            total_sizes = np.einsum("rs,sco->rco", multiplicity, sample_sizes)
            means = np.divide(
                total_sums,
                total_sizes,
                out=np.zeros_like(total_sums),
                where=total_sizes > 0,
            )
            valid = (total_sizes[:, :, 0] > 0) & (total_sizes[:, :, 1] > 0)
            weights = np.divide(
                2.0 * total_sizes[:, :, 0] * total_sizes[:, :, 1],
                total_sizes[:, :, 0] + total_sizes[:, :, 1],
                out=np.zeros_like(total_sizes[:, :, 0]),
                where=valid,
            )
            deltas = means[:, :, 1] - means[:, :, 0]
            denominator = weights.sum(axis=1)
            return np.divide(
                (deltas * weights).sum(axis=1),
                denominator,
                out=np.full(multiplicity.shape[0], np.nan, dtype=float),
                where=denominator > 0,
            )

        observed_multiplicity = np.ones((1, len(seeds)), dtype=int)
        observed = effects_from_multiplicity(observed_multiplicity)[0]
        observed_sizes = sample_sizes.sum(axis=0)
        cells = int(
            np.sum((observed_sizes[:, 0] > 0) & (observed_sizes[:, 1] > 0))
        )
        if len(seeds) < 2 or not np.isfinite(observed):
            return float(observed), cells, np.asarray([], dtype=float)
        sampled_indices = rng.integers(
            0, len(seeds), size=(int(replicates), len(seeds))
        )
        multiplicity = np.stack(
            [
                np.bincount(sample, minlength=len(seeds))
                for sample in sampled_indices
            ],
            axis=0,
        )
        bootstrap = effects_from_multiplicity(multiplicity)
        return float(observed), cells, bootstrap[np.isfinite(bootstrap)]

    rows: list[dict[str, Any]] = []
    metrics = (
        "pool_primary",
        "pool_coverage",
        "pool_enrichment",
        "pool_min_to_mean",
    )
    rng = np.random.default_rng(int(seed))
    for (variant, pooling), heads in sorted(rankings.items()):
        for head_rank, (layer, head) in enumerate(heads, start=1):
            selected = detail[
                (detail["split"] == "confirmation")
                & (detail["design_variant"] == variant)
                & (detail["pooling"] == pooling)
                & (detail["layer"] == int(layer))
                & (detail["head"] == int(head))
                & detail["outcome_group"].isin(["correct", "wrong"])
            ].copy()
            seeds = np.asarray(sorted(selected["seed"].unique()), dtype=int)
            for metric in metrics:
                observed, cells, bootstrap_values = vectorized_effects(
                    selected,
                    metric,
                    seeds,
                    replicates=int(bootstrap_replicates),
                    rng=rng,
                )
                bootstrap = bootstrap_values.tolist()
                rows.append(
                    {
                        "design_variant": variant,
                        "pooling": pooling,
                        "head_rank": int(head_rank),
                        "layer": int(layer),
                        "head": int(head),
                        "metric": metric,
                        "confirmation_examples": selected["stimulus_id"].nunique(),
                        "correct_examples": int(
                            (selected["outcome_group"] == "correct").sum()
                        ),
                        "wrong_examples": int(
                            (selected["outcome_group"] == "wrong").sum()
                        ),
                        "counts_with_both_outcomes": int(cells),
                        "wrong_minus_correct_count_adjusted": observed,
                        "bootstrap_valid_replicates": len(bootstrap),
                        "bootstrap_ci_low": (
                            float(np.quantile(bootstrap, 0.025))
                            if bootstrap
                            else math.nan
                        ),
                        "bootstrap_ci_high": (
                            float(np.quantile(bootstrap, 0.975))
                            if bootstrap
                            else math.nan
                        ),
                        "descriptive_only_small_wrong_group": bool(
                            (selected["outcome_group"] == "wrong").sum() < 5
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _plot_behavior_accuracy(labels: pd.DataFrame, output: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = (
        labels.groupby(["design_variant", "split", "gold_count"], as_index=False)
        .agg(examples=("stimulus_id", "count"), accuracy=("is_correct", "mean"))
    )
    variants = sorted(summary["design_variant"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    for axis, variant in zip(axes.flat, variants):
        frame = summary[summary["design_variant"] == variant]
        for split, marker in (("discovery", "o"), ("confirmation", "s")):
            selected = frame[frame["split"] == split].sort_values("gold_count")
            axis.plot(
                selected["gold_count"],
                selected["accuracy"],
                marker=marker,
                label=split,
            )
        axis.set_title(str(variant))
        axis.set_xlabel("gold count")
        axis.set_ylabel("strict greedy accuracy")
        axis.set_ylim(-0.03, 1.03)
        axis.set_xticks(range(1, 11))
        axis.grid(alpha=0.2)
        axis.legend()
    figure.suptitle("Actual non-thinking greedy outputs")
    figure.tight_layout()
    path = output / "behavior_accuracy_by_count.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_shared_heatmaps(summary: pd.DataFrame, output: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = sorted(summary["design_variant"].unique())
    paths: list[Path] = []
    for pooling in POOLINGS:
        pool_frame = summary[
            (summary["pooling"] == pooling)
            & np.isclose(summary["full_visibility_rate"], 1.0)
        ]
        for metric in ("pool_primary", "pool_coverage", "pool_enrichment"):
            finite = pool_frame[metric].to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            if not len(finite):
                continue
            vmin = 0.0
            vmax = 1.0 if metric == "pool_coverage" else float(
                np.quantile(finite, 0.995)
            )
            vmax = max(vmax, np.finfo(float).eps)
            figure, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
            image = None
            for axis, variant in zip(axes.flat, variants):
                frame = pool_frame[pool_frame["design_variant"] == variant]
                heatmap = frame.pivot(index="layer", columns="head", values=metric)
                if heatmap.empty:
                    axis.text(0.5, 0.5, "no fully visible layer", ha="center")
                    axis.set_axis_off()
                    continue
                heatmap = heatmap.reindex(
                    index=range(
                        int(pool_frame["layer"].min()),
                        int(pool_frame["layer"].max()) + 1,
                    ),
                    columns=range(
                        int(pool_frame["head"].min()),
                        int(pool_frame["head"].max()) + 1,
                    ),
                )
                image = axis.imshow(
                    heatmap.to_numpy(dtype=float),
                    aspect="auto",
                    interpolation="nearest",
                    cmap="magma",
                    vmin=vmin,
                    vmax=vmax,
                )
                axis.set_title(str(variant))
                axis.set_xlabel("head")
                axis.set_ylabel("layer (blank = incomplete visibility)")
                if heatmap.shape[1] <= 32:
                    axis.set_xticks(np.arange(heatmap.shape[1]))
                    axis.set_xticklabels(heatmap.columns.astype(int), fontsize=6)
                stride = max(1, heatmap.shape[0] // 12)
                positions = np.arange(0, heatmap.shape[0], stride)
                axis.set_yticks(positions)
                axis.set_yticklabels(
                    heatmap.index.to_numpy()[positions].astype(int), fontsize=7
                )
            if image is not None:
                figure.colorbar(
                    image,
                    ax=axes.ravel().tolist(),
                    label=f"{metric} (shared scale; clipped at 99.5th percentile)",
                    shrink=0.82,
                )
            figure.suptitle(
                f"Discovery broad-head map, {pooling}, N=2..10"
            )
            figure.subplots_adjust(top=0.91, right=0.9, hspace=0.28, wspace=0.22)
            path = output / f"broad_heatmaps_{pooling}_{metric}.png"
            figure.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(figure)
            paths.append(path)
    return paths


def _pooling_comparison(
    summary: pd.DataFrame,
    rankings: Mapping[tuple[str, str], Sequence[Head]],
    output: Path,
    *,
    top_k: int,
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fully_visible = summary[np.isclose(summary["full_visibility_rate"], 1.0)]
    end = fully_visible[fully_visible["pooling"] == "span_end"]
    keys = ["model_label", "design_variant", "layer", "head", "layer_type"]
    comparisons = ("span_mean", "span_sum")
    merged_by_pooling: dict[str, pd.DataFrame] = {}
    for comparison in comparisons:
        other = fully_visible[fully_visible["pooling"] == comparison]
        merged_by_pooling[comparison] = end[keys + ["pool_primary"]].merge(
            other[keys + ["pool_primary"]],
            on=keys,
            suffixes=("_span_end", f"_{comparison}"),
            how="inner",
            validate="one_to_one",
        )
    variants = sorted(fully_visible["design_variant"].unique())
    rows: list[dict[str, Any]] = []
    figure, axes = plt.subplots(
        len(comparisons), len(variants), figsize=(14, 7.2), squeeze=False
    )
    for row_index, comparison in enumerate(comparisons):
        merged = merged_by_pooling[comparison]
        for column_index, variant in enumerate(variants):
            axis = axes[row_index, column_index]
            frame = merged[merged["design_variant"] == variant]
            x = frame["pool_primary_span_end"].to_numpy(dtype=float)
            y = frame[f"pool_primary_{comparison}"].to_numpy(dtype=float)
            axis.scatter(x, y, s=10, alpha=0.35)
            axis.set_xscale("symlog", linthresh=1e-8)
            axis.set_yscale("symlog", linthresh=1e-8)
            axis.set_title(str(variant))
            axis.set_xlabel("span-end primary")
            axis.set_ylabel(f"{comparison.replace('_', '-')} primary")
            axis.grid(alpha=0.15)
            end_top = set(
                rankings.get((str(variant), "span_end"), ())[:top_k]
            )
            other_top = set(
                rankings.get((str(variant), comparison), ())[:top_k]
            )
            rank_x = pd.Series(x).rank().to_numpy(dtype=float)
            rank_y = pd.Series(y).rank().to_numpy(dtype=float)
            correlation = (
                float(np.corrcoef(rank_x, rank_y)[0, 1])
                if len(frame) >= 2
                else math.nan
            )
            union = end_top | other_top
            rows.append(
                {
                    "design_variant": variant,
                    "left_pooling": "span_end",
                    "right_pooling": comparison,
                    "heads_compared": len(frame),
                    "spearman_primary": correlation,
                    "top_k": int(top_k),
                    "top_k_intersection": len(end_top & other_top),
                    "top_k_jaccard": (
                        len(end_top & other_top) / len(union)
                        if union
                        else math.nan
                    ),
                }
            )
    figure.suptitle("Does endpoint retrieval align with complete-span retrieval?")
    figure.tight_layout()
    figure_path = output / "span_end_alignment_heads.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    table_path = output / "span_end_alignment_heads.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    return figure_path, table_path


def _ranking_stability(
    rankings: Mapping[tuple[str, str], Sequence[Head]],
    *,
    top_k: int,
) -> pd.DataFrame:
    keys = sorted(rankings)
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1 :]:
            if left[0] != right[0] and left[1] != right[1]:
                continue
            left_set = set(rankings[left][: int(top_k)])
            right_set = set(rankings[right][: int(top_k)])
            union = left_set | right_set
            rows.append(
                {
                    "left_variant": left[0],
                    "left_pooling": left[1],
                    "right_variant": right[0],
                    "right_pooling": right[1],
                    "top_k": int(top_k),
                    "intersection": len(left_set & right_set),
                    "jaccard": len(left_set & right_set) / len(union)
                    if union
                    else math.nan,
                }
            )
    return pd.DataFrame(rows)


def discovery_seed_bootstrap_stability(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    top_k: int,
    replicates: int = 500,
    seed: int = 77231,
) -> pd.DataFrame:
    """Measure how often discovery-seed bootstraps recover each top head."""

    rows: list[dict[str, Any]] = []
    base_rng = np.random.default_rng(int(seed))
    for (variant, pooling), candidates in summary[
        summary["is_broad_candidate"]
    ].groupby(["design_variant", "pooling"], sort=True):
        candidate_keys = [
            (int(row.layer), int(row.head))
            for row in candidates.sort_values("candidate_rank").itertuples(index=False)
        ]
        if not candidate_keys:
            continue
        frame = detail[
            (detail["split"] == "discovery")
            & (detail["count"] >= 2)
            & (detail["design_variant"] == variant)
            & (detail["pooling"] == pooling)
        ]
        per_seed = frame.groupby(["seed", "layer", "head"], as_index=False)[
            "pool_primary"
        ].mean()
        pivot = per_seed.pivot(
            index="seed", columns=["layer", "head"], values="pool_primary"
        )
        pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(candidate_keys))
        if pivot.isna().any().any():
            raise RuntimeError(
                f"Incomplete discovery seed/head matrix for {variant} {pooling}"
            )
        values = pivot.to_numpy(dtype=float)
        selections = np.zeros(values.shape[1], dtype=int)
        rank_sum = np.zeros(values.shape[1], dtype=float)
        rank_square_sum = np.zeros(values.shape[1], dtype=float)
        local_seed = int(base_rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(local_seed)
        for _ in range(int(replicates)):
            sampled = rng.integers(0, values.shape[0], size=values.shape[0])
            scores = values[sampled].mean(axis=0)
            order = np.argsort(-scores, kind="stable")
            ranks = np.empty(len(order), dtype=float)
            ranks[order] = np.arange(1, len(order) + 1, dtype=float)
            selected = order[: min(int(top_k), len(order))]
            selections[selected] += 1
            rank_sum += ranks
            rank_square_sum += ranks**2
        for index, (layer, head) in enumerate(candidate_keys):
            mean_rank = rank_sum[index] / int(replicates)
            variance = max(
                0.0,
                rank_square_sum[index] / int(replicates) - mean_rank**2,
            )
            global_row = candidates[
                (candidates["layer"] == int(layer))
                & (candidates["head"] == int(head))
            ].iloc[0]
            rows.append(
                {
                    "design_variant": str(variant),
                    "pooling": str(pooling),
                    "layer": int(layer),
                    "head": int(head),
                    "global_candidate_rank": int(global_row["candidate_rank"]),
                    "discovery_seeds": values.shape[0],
                    "bootstrap_replicates": int(replicates),
                    "top_k": int(top_k),
                    "top_k_selection_frequency": float(
                        selections[index] / int(replicates)
                    ),
                    "bootstrap_mean_rank": float(mean_rank),
                    "bootstrap_rank_sd": float(math.sqrt(variance)),
                }
            )
    return pd.DataFrame(rows)


def _plot_outcome_curves(
    detail: pd.DataFrame,
    rankings: Mapping[tuple[str, str], Sequence[Head]],
    output: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    variants = sorted(detail["design_variant"].unique())
    for pooling in POOLINGS:
        for metric in ("pool_coverage", "pool_enrichment"):
            figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
            for axis, variant in zip(axes.flat, variants):
                heads = rankings.get((str(variant), pooling), ())
                if not heads:
                    axis.text(0.5, 0.5, "no eligible candidate head", ha="center")
                    continue
                layer, head = heads[0]
                frame = detail[
                    (detail["split"] == "confirmation")
                    & (detail["design_variant"] == variant)
                    & (detail["pooling"] == pooling)
                    & (detail["layer"] == int(layer))
                    & (detail["head"] == int(head))
                ]
                for outcome, marker in (("correct", "o"), ("wrong", "x")):
                    selected = frame[frame["outcome_group"] == outcome]
                    grouped = selected.groupby("count")[metric].agg(
                        ["mean", "sem", "count"]
                    )
                    if grouped.empty:
                        continue
                    axis.errorbar(
                        grouped.index,
                        grouped["mean"],
                        yerr=grouped["sem"].fillna(0),
                        marker=marker,
                        capsize=2,
                        label=f"{outcome} (n={int(grouped['count'].sum())})",
                    )
                axis.set_title(f"{variant}: L{layer}H{head}")
                axis.set_xlabel("gold count")
                axis.set_ylabel(metric)
                axis.set_xticks(range(1, 11))
                axis.grid(alpha=0.2)
                axis.legend(fontsize=8)
            figure.suptitle(
                f"Held-out correct versus wrong: {pooling} top discovery head"
            )
            figure.tight_layout()
            path = output / f"outcome_curves_{pooling}_{metric}.png"
            figure.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(figure)
            paths.append(path)
    return paths


def _plot_occurrence_profiles(occurrences: pd.DataFrame, output: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    variants = sorted(occurrences["design_variant"].unique())
    for pooling in POOLINGS:
        figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
        for axis, variant in zip(axes.flat, variants):
            frame = occurrences[
                (occurrences["split"] == "confirmation")
                & (occurrences["design_variant"] == variant)
                & (occurrences["pooling"] == pooling)
                & (occurrences["count"] == 10)
            ]
            for outcome, marker in (("correct", "o"), ("wrong", "x")):
                selected = frame[frame["outcome_group"] == outcome]
                grouped = selected.groupby("occurrence_index")[
                    "ensemble_normalized_share"
                ].agg(["mean", "sem", "count"])
                if grouped.empty:
                    continue
                axis.errorbar(
                    grouped.index,
                    grouped["mean"],
                    yerr=grouped["sem"].fillna(0),
                    marker=marker,
                    capsize=2,
                    label=f"{outcome} ({selected['stimulus_id'].nunique()} prompts)",
                )
            axis.axhline(1.0, color="black", linestyle="--", alpha=0.4)
            axis.set_title(str(variant))
            axis.set_xlabel("needle occurrence index")
            axis.set_ylabel("normalized attention share (uniform = 1)")
            axis.set_xticks(range(1, 11))
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        figure.suptitle(
            f"Held-out N=10 occurrence profile, top-8 ensemble, {pooling}"
        )
        figure.tight_layout()
        path = output / f"occurrence_profiles_n10_{pooling}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def _plot_omission_diagnostics(prompts: pd.DataFrame, output: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    under = prompts[
        (prompts["split"] == "confirmation")
        & (prompts["outcome_group"] == "wrong")
        & (prompts["omission_count"].fillna(0) > 0)
    ].copy()
    if under.empty:
        return []
    variants = sorted(under["design_variant"].unique())
    paths: list[Path] = []

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
    for axis, variant in zip(axes.flat, variants):
        frame = under[under["design_variant"] == variant]
        for pooling, marker in (
            ("span_end", "o"),
            ("span_mean", "x"),
            ("span_sum", "s"),
        ):
            selected = frame[
                (frame["pooling"] == pooling)
                & frame["low_attention_threshold_available"].astype(bool)
            ]
            axis.scatter(
                selected["omission_count"],
                selected["low_attention_occurrence_count"],
                marker=marker,
                alpha=0.65,
                label=f"{pooling} (n={len(selected)})",
            )
        low_values = frame["low_attention_occurrence_count"].dropna()
        low_max = float(low_values.max()) if not low_values.empty else 0.0
        limit = int(max(float(frame["omission_count"].max()), low_max))
        axis.plot([0, limit], [0, limit], color="black", linestyle="--", alpha=0.4)
        axis.set_title(str(variant))
        axis.set_xlabel("actual undercount magnitude")
        axis.set_ylabel("occurrences below correct-discovery q10")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Do low-attention occurrences track omitted counts?")
    figure.tight_layout()
    path = output / "omission_count_vs_low_attention_count.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharey=True)
    for axis, variant in zip(axes.flat, variants):
        frame = under[under["design_variant"] == variant]
        values = [
            frame[frame["pooling"] == pooling][
                "undercount_tail_to_prefix_ratio"
            ].dropna()
            for pooling in POOLINGS
        ]
        if any(len(value) for value in values):
            axis.boxplot(values, labels=list(POOLINGS), showfliers=True)
        axis.axhline(1.0, color="black", linestyle="--", alpha=0.4)
        axis.set_title(str(variant))
        axis.set_xlabel("attention pooling")
        axis.set_ylabel("mean share after predicted boundary / before boundary")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Undercount stopped-early diagnostic (ratio < 1 means weaker tail attention)"
    )
    figure.tight_layout()
    path = output / "undercount_tail_to_prefix_attention.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    paths.append(path)
    return paths


def nested_increment_diagnostics(
    occurrences: pd.DataFrame,
    prompts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test whether a newly activated needle receives attention when count advances.

    V4 families are nested: moving N-1 to N toggles exactly the Nth slot from a
    matched control to a needle. A failed +1 prediction transition therefore
    supplies a more targeted missed-evidence diagnostic than a scalar
    undercount alone.
    """

    groups = ["model_label", "design_variant", "seed", "pooling"]
    ordered = prompts.sort_values([*groups, "count"]).copy()
    ordered["previous_predicted_count"] = ordered.groupby(groups)[
        "predicted_count"
    ].shift(1)
    valid_pair = (
        ordered["count"].astype(int) >= 2
    ) & ordered["predicted_count"].notna() & ordered[
        "previous_predicted_count"
    ].notna()
    ordered["prediction_increment"] = np.where(
        valid_pair,
        ordered["predicted_count"] - ordered["previous_predicted_count"],
        np.nan,
    )
    ordered["increment_status"] = "invalid_pair"
    ordered.loc[
        valid_pair & (ordered["prediction_increment"] == 1), "increment_status"
    ] = "registered_plus_one"
    ordered.loc[
        valid_pair & (ordered["prediction_increment"] < 1), "increment_status"
    ] = "failed_to_increment"
    ordered.loc[
        valid_pair & (ordered["prediction_increment"] > 1), "increment_status"
    ] = "overshot_increment"

    newly_active = occurrences[
        occurrences["occurrence_index"].astype(int)
        == occurrences["count"].astype(int)
    ][
        [
            "stimulus_id",
            "pooling",
            "occurrence_index",
            "slot_index",
            "normalized_depth",
            "ensemble_raw_attention",
            "ensemble_normalized_share",
            "low_attention_rank",
            "correct_discovery_q10_share",
            "below_correct_discovery_q10",
        ]
    ].rename(
        columns={
            "ensemble_raw_attention": "new_needle_raw_attention",
            "ensemble_normalized_share": "new_needle_normalized_share",
            "low_attention_rank": "new_needle_low_attention_rank",
            "correct_discovery_q10_share": "new_needle_correct_discovery_q10_share",
            "below_correct_discovery_q10": "new_needle_below_correct_discovery_q10",
        }
    )
    diagnostics = ordered.merge(
        newly_active,
        on=["stimulus_id", "pooling"],
        how="left",
        validate="one_to_one",
    )
    if diagnostics["new_needle_normalized_share"].isna().any():
        raise RuntimeError("A nested V4 prompt lacks its newly activated occurrence")
    summary = (
        diagnostics[diagnostics["count"].astype(int) >= 2]
        .groupby(
            [
                "model_label",
                "design_variant",
                "pooling",
                "split",
                "increment_status",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            transitions=("stimulus_id", "count"),
            seeds=("seed", "nunique"),
            mean_new_needle_normalized_share=(
                "new_needle_normalized_share",
                "mean",
            ),
            mean_new_needle_raw_attention=("new_needle_raw_attention", "mean"),
            low_attention_rate=(
                "new_needle_below_correct_discovery_q10",
                "mean",
            ),
        )
    )
    return diagnostics, summary


def _plot_nested_increment_diagnostics(
    diagnostics: pd.DataFrame,
    output: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = diagnostics[
        (diagnostics["split"] == "confirmation")
        & diagnostics["increment_status"].isin(
            ["registered_plus_one", "failed_to_increment"]
        )
    ]
    if frame.empty:
        return []
    variants = sorted(frame["design_variant"].unique())
    paths: list[Path] = []
    for pooling in POOLINGS:
        figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharey=True)
        for axis, variant in zip(axes.flat, variants):
            selected = frame[
                (frame["design_variant"] == variant)
                & (frame["pooling"] == pooling)
            ]
            statuses = ["registered_plus_one", "failed_to_increment"]
            values = [
                selected[selected["increment_status"] == status][
                    "new_needle_normalized_share"
                ].dropna()
                for status in statuses
            ]
            if any(len(value) for value in values):
                axis.boxplot(
                    values,
                    labels=["+1 registered", "failed +1"],
                    showfliers=True,
                )
            axis.axhline(1.0, color="black", linestyle="--", alpha=0.4)
            axis.set_title(str(variant))
            axis.set_ylabel("newly activated needle share (uniform = 1)")
            axis.grid(axis="y", alpha=0.2)
        figure.suptitle(
            f"Nested N-1→N transition: newly activated needle attention, {pooling}"
        )
        figure.tight_layout()
        path = output / f"nested_increment_new_needle_{pooling}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def _binned_attention(
    row: np.ndarray,
    *,
    key_start: int,
    bin_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(row, dtype=float)
    starts = np.arange(0, len(values), int(bin_size), dtype=int)
    masses = np.add.reduceat(values, starts)
    centers = int(key_start) + starts + np.minimum(
        int(bin_size) / 2,
        (len(values) - starts) / 2,
    )
    return centers, masses


def _plot_representative_maps(
    *,
    attention_index_path: Path,
    encodings: Mapping[str, PromptEncoding],
    prompts: pd.DataFrame,
    rankings: Mapping[tuple[str, str], Sequence[Head]],
    output: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = {
        str(row["stimulus_id"]): row for row in _read_jsonl(attention_index_path)
    }
    capture_root = attention_index_path.parent
    paths: list[Path] = []
    base = prompts[prompts["pooling"] == "span_end"].copy()
    for variant in sorted(base["design_variant"].unique()):
        confirmation = base[
            (base["design_variant"] == variant)
            & (base["split"] == "confirmation")
        ]
        wrong = confirmation[confirmation["outcome_group"] == "wrong"].copy()
        if wrong.empty:
            continue
        wrong["absolute_error"] = wrong["count_error"].abs()
        wrong = wrong.sort_values(
            ["omission_count", "absolute_error", "seed"],
            ascending=[False, False, True],
        )
        wrong_row = wrong.iloc[0]
        matched = confirmation[
            (confirmation["outcome_group"] == "correct")
            & (confirmation["count"] == int(wrong_row["count"]))
        ]
        if matched.empty:
            matched = confirmation[confirmation["outcome_group"] == "correct"]
        if matched.empty:
            continue
        correct_row = matched.sort_values("seed").iloc[0]
        examples = [("correct", correct_row), ("wrong", wrong_row)]
        figure, axes = plt.subplots(
            2, len(POOLINGS), figsize=(17, 7.5), sharex=True, squeeze=False
        )
        for row_axis, (outcome, metadata) in enumerate(examples):
            stimulus_id = str(metadata["stimulus_id"])
            encoding = encodings[stimulus_id]
            record = records[stimulus_id]
            raw_path = capture_root / str(record["raw_attention_shard_path"])
            with np.load(raw_path, allow_pickle=False) as raw:
                key_starts = np.asarray(raw["key_starts"], dtype=int)
                for column_axis, pooling in enumerate(POOLINGS):
                    axis = axes[row_axis, column_axis]
                    heads = rankings.get((str(variant), pooling), ())
                    if not heads:
                        axis.text(0.5, 0.5, "no candidate head", ha="center")
                        continue
                    layer, head = heads[0]
                    values = np.asarray(
                        raw[f"layer_{int(layer):03d}"][int(head)], dtype=float
                    )
                    key_start = int(key_starts[int(layer)])
                    x, mass = _binned_attention(values, key_start=key_start)
                    axis.plot(x, np.maximum(mass, 1e-10), linewidth=0.8)
                    axis.set_yscale("log")
                    for occurrence_index, span in enumerate(
                        encoding.needle_spans, start=1
                    ):
                        axis.axvspan(
                            span.start,
                            span.end,
                            alpha=0.18,
                            color="tab:orange",
                        )
                        axis.axvline(
                            span.end - 1,
                            alpha=0.35,
                            linewidth=0.5,
                            color="tab:red",
                        )
                        if occurrence_index in {1, len(encoding.needle_spans)}:
                            axis.text(
                                span.end - 1,
                                axis.get_ylim()[1],
                                str(occurrence_index),
                                fontsize=7,
                                va="top",
                                ha="center",
                            )
                    predicted = metadata["predicted_count"]
                    predicted_text = (
                        "invalid" if pd.isna(predicted) else str(int(predicted))
                    )
                    axis.set_title(
                        f"{outcome}, {pooling}, L{layer}H{head}, "
                        f"gold={encoding.count}, pred={predicted_text}"
                    )
                    axis.set_xlabel("absolute key-token position (64-token bins)")
                    axis.set_ylabel("answer-query attention mass")
                    axis.grid(alpha=0.15)
        figure.suptitle(
            f"{variant}: matched representative answer-query maps; "
            "orange=needle span, red=span end"
        )
        figure.tight_layout()
        path = output / f"representative_map_{str(variant).replace('.', '_')}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def analyze_labeled_attention(
    *,
    attention_index_path: str | Path,
    generation_labels_path: str | Path,
    encodings: Iterable[PromptEncoding],
    output_dir: str | Path,
    top_k: int = 8,
    overwrite_pooling_metrics: bool = False,
) -> dict[str, Path]:
    """Run behavior-stratified V10-style analysis on saved answer-query rows."""

    output = Path(output_dir)
    tables = output / "tables"
    figures = output / "figures"
    rankings_dir = output / "rankings"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    rankings_dir.mkdir(parents=True, exist_ok=True)
    encoding_by_id = {item.stimulus_id: item for item in encodings}
    if len(encoding_by_id) == 0:
        raise ValueError("No rendered encodings were supplied for attention analysis")

    pooling_index = capture_pooling_metric_shards(
        attention_index_path=attention_index_path,
        generation_labels_path=generation_labels_path,
        encodings=encoding_by_id.values(),
        output_dir=output / "pooling_metric_capture",
        overwrite=overwrite_pooling_metrics,
    )
    detail = load_pooling_metric_shards(pooling_index)
    detail_path = tables / "pooling_head_detail.csv.gz"
    _write_csv_gzip_atomic(detail, detail_path)
    summary, rankings = rank_broad_candidates(detail, top_k=int(top_k))
    summary_path = tables / "discovery_head_summary.csv"
    summary.to_csv(summary_path, index=False)

    ranking_paths: list[Path] = []
    for (variant, pooling), top_heads in sorted(rankings.items()):
        full = summary[
            (summary["design_variant"] == variant)
            & (summary["pooling"] == pooling)
            & summary["is_broad_candidate"]
        ].sort_values("candidate_rank")
        payload = {
            "schema_version": "realistic_niah_v4_broad_head_ranking_v2",
            "design_variant": variant,
            "pooling": pooling,
            "selection_split": "discovery",
            "selection_counts": list(range(2, 11)),
            "eligibility": {
                "all_needles_visible": True,
                "all_hard_negatives_visible": True,
                "mean_needle_minus_hard_negative_density_positive": True,
                "mean_needle_density_enrichment_gt_one": True,
            },
            "ranking_metric": "pool_primary",
            "top_k_for_diagnostics": int(top_k),
            "top_heads": [
                {"layer": int(layer), "head": int(head)}
                for layer, head in top_heads
            ],
            "full_candidate_ranking": [
                {
                    "rank": int(row.candidate_rank),
                    "layer": int(row.layer),
                    "head": int(row.head),
                    "layer_type": str(row.layer_type),
                    "pool_primary": float(row.pool_primary),
                    "pool_coverage": float(row.pool_coverage),
                    "pool_contrast": float(row.pool_contrast),
                    "pool_enrichment": float(row.pool_enrichment),
                }
                for row in full.itertuples(index=False)
            ],
        }
        path = rankings_dir / (
            f"{str(variant).replace('.', '_')}_{pooling}.json"
        )
        _atomic_json(path, payload)
        ranking_paths.append(path)

    outcome_summary = summarize_outcomes(detail)
    outcome_summary_path = tables / "head_outcomes_by_count.csv.gz"
    _write_csv_gzip_atomic(outcome_summary, outcome_summary_path)
    effects = confirmation_outcome_effects(detail, rankings)
    effects_path = tables / "confirmation_wrong_minus_correct_effects.csv"
    effects.to_csv(effects_path, index=False)
    stability = _ranking_stability(rankings, top_k=int(top_k))
    stability_path = tables / "head_ranking_stability.csv"
    stability.to_csv(stability_path, index=False)
    seed_stability = discovery_seed_bootstrap_stability(
        detail,
        summary,
        top_k=int(top_k),
    )
    seed_stability_path = tables / "discovery_seed_bootstrap_head_stability.csv"
    seed_stability.to_csv(seed_stability_path, index=False)

    occurrences, prompts = extract_ranked_occurrence_diagnostics(
        attention_index_path=attention_index_path,
        generation_labels_path=generation_labels_path,
        encodings=encoding_by_id,
        rankings=rankings,
        output_dir=tables,
    )
    nested, nested_summary = nested_increment_diagnostics(occurrences, prompts)
    nested_path = tables / "nested_increment_diagnostics.csv"
    nested_summary_path = tables / "nested_increment_summary.csv"
    nested.to_csv(nested_path, index=False)
    nested_summary.to_csv(nested_summary_path, index=False)
    labels = _label_table(Path(generation_labels_path))
    behavior_summary = (
        labels.groupby(
            ["model_label", "design_variant", "split", "gold_count"],
            as_index=False,
        )
        .agg(
            examples=("stimulus_id", "count"),
            correct=("is_correct", "sum"),
            accuracy=("is_correct", "mean"),
            format_valid_rate=("format_valid", "mean"),
        )
    )
    behavior_summary_path = tables / "behavior_accuracy_by_count.csv"
    behavior_summary.to_csv(behavior_summary_path, index=False)

    figure_paths: list[Path] = [_plot_behavior_accuracy(labels, figures)]
    figure_paths.extend(_plot_shared_heatmaps(summary, figures))
    comparison_figure, comparison_table = _pooling_comparison(
        summary, rankings, figures, top_k=int(top_k)
    )
    figure_paths.append(comparison_figure)
    figure_paths.extend(_plot_outcome_curves(detail, rankings, figures))
    figure_paths.extend(_plot_occurrence_profiles(occurrences, figures))
    figure_paths.extend(_plot_omission_diagnostics(prompts, figures))
    figure_paths.extend(_plot_nested_increment_diagnostics(nested, figures))
    figure_paths.extend(
        _plot_representative_maps(
            attention_index_path=Path(attention_index_path),
            encodings=encoding_by_id,
            prompts=prompts,
            rankings=rankings,
            output=figures,
        )
    )

    manifest_path = output / "attention_analysis_manifest.json"
    _atomic_json(
        manifest_path,
        {
            "schema_version": "realistic_niah_v4_labeled_attention_analysis_v3",
            "behavior_label_source": str(Path(generation_labels_path)),
            "behavior_label_rule": (
                "strict correctness of the actual greedy-generated continuation; "
                "no candidate probabilities are used"
            ),
            "answer_query_position": (
                "last prompt token in the already-rendered Total: prefix, before "
                "the generated numeric continuation"
            ),
            "poolings": {
                "span_end": "one attention weight at each needle span's final token",
                "span_mean": (
                    "mean attention per token within each complete needle span"
                ),
                "span_sum": (
                    "literal sum of attention over every token in each complete "
                    "needle span; contrast and enrichment use per-token density "
                    "to control record length"
                ),
            },
            "v10_metrics": [
                "needle evidence sum",
                "entropy-normalized coverage",
                "effective occurrence number",
                "evidence-times-coverage primary score",
            ],
            "ranking": {
                "split": "discovery",
                "counts": list(range(2, 11)),
                "top_k": int(top_k),
                "requires_full_span_visibility": True,
                "requires_positive_matched_hard_negative_contrast": True,
                "requires_needle_density_enrichment_gt_one": True,
                "no_negative_contrast_fallback": True,
            },
            "outcome_comparison": {
                "split": "confirmation",
                "head_selection_reused_from_discovery": True,
                "effect": "count-adjusted wrong minus correct",
                "uncertainty": "seed-cluster bootstrap 95% interval",
                "invalid_generations_kept_separate": True,
            },
            "omission_guardrail": (
                "A scalar undercount does not reveal which occurrence the model "
                "missed. Bottom-k attention occurrences are therefore labeled "
                "attention-implied missed candidates, not ground-truth omissions."
            ),
            "nested_increment_diagnostic": (
                "Within each nested panel/seed family, N-1 to N activates exactly "
                "one new needle. The analysis compares attention to that newly "
                "activated occurrence when the generated prediction does versus "
                "does not increment by one; association is not causal proof."
            ),
            "files": {
                "pooling_metric_index": str(pooling_index),
                "pooling_detail": str(detail_path),
                "discovery_head_summary": str(summary_path),
                "head_outcomes_by_count": str(outcome_summary_path),
                "confirmation_effects": str(effects_path),
                "ranking_stability": str(stability_path),
                "discovery_seed_bootstrap_stability": str(seed_stability_path),
                "pooling_comparison": str(comparison_table),
                "behavior_summary": str(behavior_summary_path),
                "occurrence_attention": str(tables / "occurrence_attention.csv.gz"),
                "omission_diagnostics": str(tables / "omission_diagnostics.csv"),
                "nested_increment_diagnostics": str(nested_path),
                "nested_increment_summary": str(nested_summary_path),
                "rankings": [str(path) for path in ranking_paths],
                "figures": [str(path) for path in figure_paths],
            },
        },
    )
    return {
        "manifest": manifest_path,
        "pooling_metric_index": pooling_index,
        "detail": detail_path,
        "head_summary": summary_path,
        "outcome_summary": outcome_summary_path,
        "outcome_effects": effects_path,
        "seed_stability": seed_stability_path,
        "omission_diagnostics": tables / "omission_diagnostics.csv",
        "nested_increment_diagnostics": nested_path,
        "occurrence_attention": tables / "occurrence_attention.csv.gz",
        "behavior_summary": behavior_summary_path,
    }
