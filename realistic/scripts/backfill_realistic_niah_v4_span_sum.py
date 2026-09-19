from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from realistic_niah_v4.attention_outcomes import (  # noqa: E402
    POOLINGS,
    POOL_METRICS,
    _atomic_json,
    _pool_scalar_arrays,
    _ranking_stability,
    _write_csv_gzip_atomic,
    confirmation_outcome_effects,
    discovery_seed_bootstrap_stability,
    extract_ranked_occurrence_diagnostics,
    load_pooling_metric_shards,
    nested_increment_diagnostics,
    rank_broad_candidates,
    summarize_outcomes,
)
from realistic_niah_v4.prompts import PromptEncoding, TokenSpan  # noqa: E402


MODELS = ("Qwen3-8B", "Gemma4-E4B")


def _native_path(path: Path) -> Path:
    """Allow NumPy to open long Windows paths without changing file identity."""

    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _span_lookup(base_analysis: Path) -> dict[str, tuple[TokenSpan, ...]]:
    path = base_analysis / "tables" / "occurrence_attention.csv.gz"
    frame = pd.read_csv(
        path,
        compression="gzip",
        usecols=[
            "stimulus_id",
            "pooling",
            "occurrence_index",
            "slot_index",
            "span_start",
            "span_end",
        ],
    )
    frame = frame[frame["pooling"] == "span_end"].drop_duplicates(
        ["stimulus_id", "occurrence_index"]
    )
    result: dict[str, tuple[TokenSpan, ...]] = {}
    for stimulus_id, selected in frame.groupby("stimulus_id", sort=False):
        selected = selected.sort_values("occurrence_index")
        spans = tuple(
            TokenSpan(
                slot_index=int(row.slot_index),
                start=int(row.span_start),
                end=int(row.span_end),
                active=True,
                kind="needle",
                canonical_length=int(row.span_end) - int(row.span_start),
                model_token_length=int(row.span_end) - int(row.span_start),
            )
            for row in selected.itertuples(index=False)
        )
        result[str(stimulus_id)] = spans
    return result


def _span_sum_metrics(
    rows: np.ndarray,
    spans: Sequence[TokenSpan],
    *,
    key_start: int,
) -> dict[str, np.ndarray]:
    values = np.asarray(rows, dtype=np.float32)
    evidence = np.zeros((values.shape[0], len(spans)), dtype=np.float32)
    key_end = int(key_start) + values.shape[1]
    for occurrence_index, span in enumerate(spans):
        if int(span.start) < int(key_start) or int(span.end) > key_end:
            continue
        local_start = int(span.start) - int(key_start)
        local_end = int(span.end) - int(key_start)
        evidence[:, occurrence_index] = values[:, local_start:local_end].sum(
            axis=1, dtype=np.float32
        )
    metrics = _pool_scalar_arrays(
        evidence,
        np.zeros_like(evidence),
        row_sums=values.sum(axis=1, dtype=np.float64),
        key_length=values.shape[1],
    )
    return metrics


def _write_rankings(
    summary: pd.DataFrame,
    rankings: Mapping[tuple[str, str], Sequence[tuple[int, int]]],
    output: Path,
    *,
    top_k: int,
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for (variant, pooling), top_heads in sorted(rankings.items()):
        full = summary[
            (summary["design_variant"] == variant)
            & (summary["pooling"] == pooling)
            & summary["is_broad_candidate"]
        ].sort_values("candidate_rank")
        payload = {
            "schema_version": "realistic_niah_v4_broad_head_ranking_v3",
            "design_variant": variant,
            "pooling": pooling,
            "selection_split": "discovery",
            "selection_counts": list(range(2, 11)),
            "eligibility": {
                "all_needles_visible": True,
                "all_hard_negatives_visible": True,
                "mean_needle_minus_hard_negative_density_positive": True,
                "mean_needle_density_enrichment_gt_one": True,
                "span_sum_length_control": (
                    "span_mean density is used for contrast and enrichment"
                ),
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
                    "pool_sum": float(row.pool_sum),
                    "pool_coverage": float(row.pool_coverage),
                    "pool_contrast": float(row.pool_contrast),
                    "pool_enrichment": float(row.pool_enrichment),
                }
                for row in full.itertuples(index=False)
            ],
        }
        path = output / f"{str(variant).replace('.', '_')}_{pooling}.json"
        _atomic_json(path, payload)
        paths.append(path)
    return paths


def _end_alignment_table(
    summary: pd.DataFrame,
    rankings: Mapping[tuple[str, str], Sequence[tuple[int, int]]],
    *,
    top_k: int,
) -> pd.DataFrame:
    visible = summary[np.isclose(summary["full_visibility_rate"], 1.0)]
    keys = ["model_label", "design_variant", "layer", "head", "layer_type"]
    endpoint = visible[visible["pooling"] == "span_end"]
    rows: list[dict[str, Any]] = []
    for other_pooling in ("span_mean", "span_sum"):
        other = visible[visible["pooling"] == other_pooling]
        merged = endpoint[keys + ["pool_primary"]].merge(
            other[keys + ["pool_primary"]],
            on=keys,
            how="inner",
            suffixes=("_span_end", f"_{other_pooling}"),
            validate="one_to_one",
        )
        for variant, frame in merged.groupby("design_variant", sort=True):
            end_top = set(rankings[(str(variant), "span_end")][: int(top_k)])
            other_top = set(
                rankings[(str(variant), other_pooling)][: int(top_k)]
            )
            union = end_top | other_top
            rows.append(
                {
                    "design_variant": str(variant),
                    "left_pooling": "span_end",
                    "right_pooling": other_pooling,
                    "heads_compared": int(len(frame)),
                    "spearman_primary": float(
                        frame["pool_primary_span_end"].corr(
                            frame[f"pool_primary_{other_pooling}"],
                            method="spearman",
                        )
                    ),
                    "top_k": int(top_k),
                    "top_k_intersection": int(len(end_top & other_top)),
                    "top_k_jaccard": (
                        float(len(end_top & other_top) / len(union))
                        if union
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _reconstruct_encodings(
    records: Sequence[Mapping[str, Any]],
    spans: Mapping[str, tuple[TokenSpan, ...]],
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, PromptEncoding]:
    result: dict[str, PromptEncoding] = {}
    for record in records:
        stimulus_id = str(record["stimulus_id"])
        needle_spans = spans[stimulus_id]
        hard_negatives = tuple(
            TokenSpan(
                slot_index=int(span.slot_index),
                start=int(span.start),
                end=int(span.end),
                active=False,
                kind="hard_negative",
                canonical_length=int(span.canonical_length),
                model_token_length=int(span.model_token_length),
            )
            for span in needle_spans
        )
        item = metadata[stimulus_id]
        result[stimulus_id] = PromptEncoding(
            stimulus_id=stimulus_id,
            design_variant=str(record["design_variant"]),
            seed=int(record["seed"]),
            split=str(record["split"]),
            count=int(record["count"]),
            model_label=str(record["model_label"]),
            answer_format="numeric",
            text="",
            generation_prompt="",
            input_ids=(0,),
            attention_mask=(1,),
            query_position=int(item["query_position"]),
            slot_spans=needle_spans,
            needle_spans=needle_spans,
            hard_negative_spans=hard_negatives,
            count_candidate_texts=(),
            count_candidate_answer_token_ids=(),
            count_candidate_token_ids=(),
        )
    return result


def backfill_model(
    run_root: Path,
    model: str,
    *,
    output_name: str,
    top_k: int,
    overwrite: bool,
) -> Path:
    model_root = run_root / model / "numeric"
    base_analysis = model_root / "attention" / "analysis"
    output = model_root / "attention" / output_name
    tables = output / "tables"
    rankings_dir = output / "rankings"
    pooling_output = output / "pooling_metric_capture"
    tables.mkdir(parents=True, exist_ok=True)
    rankings_dir.mkdir(parents=True, exist_ok=True)
    pooling_output.mkdir(parents=True, exist_ok=True)

    span_by_id = _span_lookup(base_analysis)
    attention_index_path = (
        model_root / "attention" / "capture" / "attention_capture_index.jsonl"
    )
    attention_records = _read_jsonl(attention_index_path)
    attention_by_id = {
        str(record["stimulus_id"]): record for record in attention_records
    }
    if set(attention_by_id) != set(span_by_id):
        raise RuntimeError(
            f"{model}: span coverage {len(span_by_id)} != raw rows "
            f"{len(attention_by_id)}"
        )

    base_pooling_index = (
        base_analysis / "pooling_metric_capture" / "pooling_metric_index.jsonl"
    )
    base_pooling_records = _read_jsonl(base_pooling_index)
    base_pooling_by_id = {
        str(record["stimulus_id"]): record for record in base_pooling_records
    }
    capture_root = attention_index_path.parent
    metadata: dict[str, dict[str, Any]] = {}
    output_index: list[dict[str, Any]] = []
    for record_index, record in enumerate(attention_records):
        stimulus_id = str(record["stimulus_id"])
        relative = (
            Path("shards")
            / str(record["design_variant"])
            / f"{stimulus_id}.csv.gz"
        )
        destination = pooling_output / relative
        if destination.exists() and not overwrite:
            combined = pd.read_csv(destination, compression="gzip")
            if set(combined["pooling"]) != set(POOLINGS):
                raise RuntimeError(f"Incomplete cached span-sum shard: {destination}")
        else:
            base_record = base_pooling_by_id[stimulus_id]
            base_shard = (
                base_pooling_index.parent / str(base_record["shard_path"])
            )
            base = pd.read_csv(base_shard, compression="gzip")
            if set(base["pooling"]) != {"span_end", "span_mean"}:
                raise RuntimeError(f"Unexpected base poolings in {base_shard}")
            mean = base[base["pooling"] == "span_mean"].sort_values(
                ["layer", "head"]
            )
            raw_path = capture_root / str(record["raw_attention_shard_path"])
            sum_frames: list[pd.DataFrame] = []
            with np.load(_native_path(raw_path), allow_pickle=False) as raw:
                key_starts = np.asarray(raw["key_starts"], dtype=int)
                for layer in sorted(mean["layer"].unique()):
                    selected = mean[mean["layer"] == int(layer)].sort_values("head")
                    rows = np.asarray(raw[f"layer_{int(layer):03d}"], dtype=np.float32)
                    expected_heads = np.arange(rows.shape[0], dtype=int)
                    if not np.array_equal(
                        selected["head"].to_numpy(dtype=int), expected_heads
                    ):
                        raise RuntimeError(
                            f"{model}/{stimulus_id}/L{layer}: head grid mismatch"
                        )
                    metrics = _span_sum_metrics(
                        rows,
                        span_by_id[stimulus_id],
                        key_start=int(key_starts[int(layer)]),
                    )
                    summed = selected.copy()
                    summed["pooling"] = "span_sum"
                    for metric_name in POOL_METRICS:
                        if metric_name in {"pool_contrast", "pool_enrichment"}:
                            # These two are intentionally copied from span-mean:
                            # they test per-token density against matched controls.
                            continue
                        summed[metric_name] = metrics[metric_name]
                    sum_frames.append(summed)
            combined = pd.concat([base, *sum_frames], ignore_index=True)
            _write_csv_gzip_atomic(combined, destination)
        first = combined.iloc[0]
        metadata[stimulus_id] = {
            "query_position": int(first["query_position"]),
            "sequence_length": int(first["sequence_length"]),
        }
        output_index.append(
            {
                "stimulus_id": stimulus_id,
                "design_variant": str(record["design_variant"]),
                "seed": int(record["seed"]),
                "count": int(record["count"]),
                "rows": int(len(combined)),
                "shard_path": relative.as_posix(),
            }
        )
        if (record_index + 1) % 10 == 0 or record_index == 0:
            print(
                f"[span-sum] {model} pooled shards "
                f"{record_index + 1}/{len(attention_records)}",
                flush=True,
            )

    output_index_path = pooling_output / "pooling_metric_index.jsonl"
    temporary = output_index_path.with_name(output_index_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output_index:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(output_index_path)

    detail = load_pooling_metric_shards(output_index_path)
    detail_path = tables / "pooling_head_detail.csv.gz"
    _write_csv_gzip_atomic(detail, detail_path)
    summary, rankings = rank_broad_candidates(detail, top_k=int(top_k))
    summary_path = tables / "discovery_head_summary.csv"
    summary.to_csv(summary_path, index=False)
    ranking_paths = _write_rankings(
        summary, rankings, rankings_dir, top_k=int(top_k)
    )

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
        detail, summary, top_k=int(top_k)
    )
    seed_stability_path = tables / "discovery_seed_bootstrap_head_stability.csv"
    seed_stability.to_csv(seed_stability_path, index=False)
    alignment = _end_alignment_table(summary, rankings, top_k=int(top_k))
    alignment_path = tables / "span_end_alignment_heads.csv"
    alignment.to_csv(alignment_path, index=False)

    encodings = _reconstruct_encodings(attention_records, span_by_id, metadata)
    labels_path = model_root / "behavior" / "capture" / "generation_labels.csv"
    occurrences, prompts = extract_ranked_occurrence_diagnostics(
        attention_index_path=attention_index_path,
        generation_labels_path=labels_path,
        encodings=encodings,
        rankings=rankings,
        output_dir=tables,
    )
    nested, nested_summary = nested_increment_diagnostics(occurrences, prompts)
    nested_path = tables / "nested_increment_diagnostics.csv"
    nested_summary_path = tables / "nested_increment_summary.csv"
    nested.to_csv(nested_path, index=False)
    nested_summary.to_csv(nested_summary_path, index=False)

    manifest_path = output / "attention_analysis_manifest.json"
    _atomic_json(
        manifest_path,
        {
            "schema_version": "realistic_niah_v4_labeled_attention_analysis_v3",
            "source_analysis": str(base_analysis),
            "source_raw_attention_index": str(attention_index_path),
            "behavior_label_source": str(labels_path),
            "poolings": {
                "span_end": "one final-token attention weight per occurrence",
                "span_mean": "complete-span attention divided by model-token length",
                "span_sum": "literal attention sum over every token in the span",
            },
            "span_sum_length_control": (
                "pool_contrast and pool_enrichment reuse span_mean per-token "
                "density; pool_sum, coverage, primary, CV, extrema, occurrence "
                "profiles, and omission diagnostics use literal span sums"
            ),
            "top_k": int(top_k),
            "files": {
                "pooling_metric_index": str(output_index_path),
                "pooling_detail": str(detail_path),
                "discovery_head_summary": str(summary_path),
                "head_outcomes_by_count": str(outcome_summary_path),
                "confirmation_effects": str(effects_path),
                "ranking_stability": str(stability_path),
                "discovery_seed_bootstrap_stability": str(seed_stability_path),
                "span_end_alignment": str(alignment_path),
                "occurrence_attention": str(
                    tables / "occurrence_attention.csv.gz"
                ),
                "omission_diagnostics": str(tables / "omission_diagnostics.csv"),
                "nested_increment_diagnostics": str(nested_path),
                "nested_increment_summary": str(nested_summary_path),
                "rankings": [str(path) for path in ranking_paths],
            },
        },
    )
    print(f"[span-sum] {model} complete: {manifest_path}", flush=True)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill literal complete-needle-span attention mass from saved "
            "answer-query rows without rerunning either model."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--output-name", default="analysis_span_sum_v3")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        parser.error(f"unknown models: {unknown}")
    for model in models:
        backfill_model(
            args.run_root,
            model,
            output_name=args.output_name,
            top_k=int(args.top_k),
            overwrite=bool(args.overwrite),
        )


if __name__ == "__main__":
    main()
