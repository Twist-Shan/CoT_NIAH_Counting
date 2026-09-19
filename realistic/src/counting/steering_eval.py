from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from dataset_generation.dynamic_niah_v2 import (
    DynamicNiahV2Config,
    generate_dynamic_niah_dataset_v2,
    write_dynamic_niah_v2,
)

from .steering import (
    SteeringExample,
    run_counting_steering_sweep,
    run_needle_span_counting_steering_sweep,
    write_csv,
)

TUPLE_MEANING = (
    "no_steering_acc, steering_acc, "
    "no_steering_avg_counts, steering_avg_counts"
)


def steering_eval_insertion_positions(
    *,
    num_needles: int,
    base_positions: Sequence[int | None],
    target_haystack_tokens: int,
) -> tuple[int, ...]:
    """Return a non-null insertion-position pattern of length ``num_needles``.

    The steering-eval datasets always use randomized insertion. These positions
    therefore act as the non-null mask required by the generator; the actual
    per-example positions are sampled later.
    """

    k = int(num_needles)
    if k <= 0:
        raise ValueError(f"num_needles must be positive, got {num_needles}")
    non_null_base = [int(pos) for pos in base_positions if pos is not None]
    if len(non_null_base) >= k:
        return tuple(non_null_base[:k])

    margin = min(50, max(0, int(target_haystack_tokens) // 10))
    low = margin
    high = max(low + k, int(target_haystack_tokens) - margin)
    if k == 1:
        return (max(0, int(target_haystack_tokens) // 2),)
    step = max(1, (high - low) // (k - 1))
    return tuple(int(low + i * step) for i in range(k))


def build_steering_eval_dataset_config(
    base_config: DynamicNiahV2Config,
    *,
    gold_count: int,
    num_examples: int,
    randomize_needle_min_separation: int | None = None,
    output_dir: str | Path,
) -> DynamicNiahV2Config:
    """Build a randomized held-out counting dataset config for one gold count."""

    k = int(gold_count)
    out = Path(output_dir)
    positions = steering_eval_insertion_positions(
        num_needles=k,
        base_positions=base_config.insertion_positions,
        target_haystack_tokens=base_config.target_haystack_tokens,
    )
    return replace(
        base_config,
        num_examples=int(num_examples),
        num_needles=k,
        insertion_positions=positions,
        randomize_needle_insertion=True,
        randomize_needle_min_separation=(
            base_config.randomize_needle_min_separation
            if randomize_needle_min_separation is None
            else int(randomize_needle_min_separation)
        ),
        output_dir=str(out),
        data_save_path=str(out / "dynamic_niah_v2.jsonl"),
        output_pred_jsonl=None,
        output_metrics_json=None,
        run_dir=None,
        run_name=None,
        save_data=True,
    )


def generate_steering_eval_datasets(
    *,
    base_config: DynamicNiahV2Config,
    output_root: str | Path,
    max_needles: int,
    num_examples: int,
    randomize_needle_min_separation: int | None = None,
) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Generate and save one randomized eval dataset for each count 1..K."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    datasets: dict[int, list[dict[str, Any]]] = {}
    metadata: list[dict[str, Any]] = []
    for k in range(1, int(max_needles) + 1):
        dataset_dir = root / f"dataset_{k}"
        cfg = build_steering_eval_dataset_config(
            base_config,
            gold_count=k,
            num_examples=int(num_examples),
            randomize_needle_min_separation=randomize_needle_min_separation,
            output_dir=dataset_dir,
        )
        rows = generate_dynamic_niah_dataset_v2(cfg)
        written = write_dynamic_niah_v2(rows, cfg)
        datasets[k] = rows
        metadata.append(
            {
                "dataset_k": k,
                "num_examples": len(rows),
                "dataset_dir": str(dataset_dir),
                "dataset_path": written["jsonl"],
                "config_path": written["config"],
                "config": asdict(cfg),
            }
        )
    return datasets, metadata


def steering_eval_examples(
    datasets: dict[int, Sequence[dict[str, Any]]],
) -> tuple[list[SteeringExample], dict[int, dict[str, Any]], dict[str, Any]]:
    """Convert eval datasets into globally indexed steering examples."""

    examples: list[SteeringExample] = []
    index_lookup: dict[int, dict[str, Any]] = {}
    for dataset_k in sorted(datasets):
        for local_idx, row in enumerate(datasets[dataset_k]):
            global_idx = len(examples)
            row_id = str(row.get("id", local_idx))
            example = SteeringExample(
                dataset_index=global_idx,
                row_id=f"dataset_{dataset_k}:{row_id}",
                group=f"dataset_{dataset_k}",
                row=dict(row),
                scored_row=None,
            )
            examples.append(example)
            index_lookup[global_idx] = {
                "dataset_k": int(dataset_k),
                "local_index": int(local_idx),
                "source_row_id": row_id,
            }
    summary = {
        "num_datasets": len(datasets),
        "num_examples": len(examples),
        "examples_by_dataset": {
            str(k): len(rows) for k, rows in sorted(datasets.items())
        },
    }
    return examples, index_lookup, summary


def _prediction_count(row: dict[str, Any], key: str) -> int | float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def _mean_numeric(values: Sequence[int | float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _metric_tuple(rows: Sequence[dict[str, Any]]) -> tuple[float, float, float | None, float | None]:
    n = len(rows)
    if n == 0:
        return (0.0, 0.0, None, None)
    no_steering_acc = sum(bool(row.get("baseline_exact_match")) for row in rows) / n
    steering_acc = sum(bool(row.get("steered_exact_match")) for row in rows) / n
    no_steering_avg = _mean_numeric(
        [_prediction_count(row, "baseline_prediction_count") for row in rows]
    )
    steering_avg = _mean_numeric(
        [_prediction_count(row, "steered_prediction_count") for row in rows]
    )
    return (
        no_steering_acc,
        steering_acc,
        no_steering_avg,
        steering_avg,
    )


def _format_metric_tuple(values: tuple[float, float, float | None, float | None]) -> str:
    formatted: list[str] = []
    for value in values:
        if value is None:
            formatted.append("null")
        else:
            formatted.append(f"{float(value):.6g}")
    return "(" + ", ".join(formatted) + ")"


def _intervention_target(row: dict[str, Any], steering_position_mode: str) -> str:
    if steering_position_mode == "last_token":
        return "last_token"
    ordinal = row.get("needle_ordinal")
    if ordinal is None:
        return "needle_span"
    return f"needle_{int(ordinal) + 1}"


def annotate_steering_eval_details(
    detail_rows: Sequence[dict[str, Any]],
    *,
    index_lookup: dict[int, dict[str, Any]],
    steering_position_mode: str,
) -> list[dict[str, Any]]:
    """Add eval-dataset identifiers and intervention labels to detail rows."""

    annotated: list[dict[str, Any]] = []
    for row in detail_rows:
        out = dict(row)
        dataset_index = int(out["dataset_index"])
        if dataset_index not in index_lookup:
            raise KeyError(f"Missing steering-eval index metadata for {dataset_index}")
        meta = index_lookup[dataset_index]
        out["eval_dataset_k"] = int(meta["dataset_k"])
        out["eval_example_index"] = int(meta["local_index"])
        out["source_row_id"] = meta["source_row_id"]
        out["intervention_target"] = _intervention_target(
            out, steering_position_mode
        )
        annotated.append(out)
    return annotated


def summarize_steering_test_eval_details(
    detail_rows: Sequence[dict[str, Any]],
    *,
    dataset_ks: Sequence[int],
) -> list[dict[str, Any]]:
    """Build compact per-setting summary rows with one tuple per eval dataset."""

    grouped: dict[tuple[int, str, float], list[dict[str, Any]]] = {}
    for row in detail_rows:
        key = (
            int(row["layer"]),
            str(row["intervention_target"]),
            float(row["beta"]),
        )
        grouped.setdefault(key, []).append(dict(row))

    summaries: list[dict[str, Any]] = []
    for (layer, target, beta), rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "layer": layer,
            "intervention_target": target,
            "steering_coeff": beta,
        }
        for dataset_k in dataset_ks:
            dataset_rows = [
                row for row in rows if int(row["eval_dataset_k"]) == int(dataset_k)
            ]
            column = f"Dataset {int(dataset_k)} Before/After steering"
            summary[column] = (
                ""
                if not dataset_rows
                else _format_metric_tuple(_metric_tuple(dataset_rows))
            )
        summaries.append(summary)
    return summaries


def write_steering_test_eval_summary_csv(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
) -> Path:
    """Write summary CSV with a leading row explaining the dataset tuples."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["layer", "intervention_target", "steering_coeff"]
    dataset_fields = sorted(
        {key for row in rows for key in row if key.startswith("Dataset ")},
        key=lambda item: int(item.split()[1]),
    )
    fieldnames.extend(dataset_fields)
    explanation = {
        "layer": "# tuple_meaning",
        "intervention_target": f"({TUPLE_MEANING})",
        "steering_coeff": "",
    }
    for field in dataset_fields:
        explanation[field] = f"({TUPLE_MEANING})"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(explanation)
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return output_path


def run_steering_test_eval(
    *,
    model: Any,
    tokenizer: Any,
    datasets: dict[int, Sequence[dict[str, Any]]],
    layers: Sequence[int],
    probe_dir: str | Path,
    betas: Sequence[float],
    max_new_tokens: int,
    thinking_mode: bool,
    vector_source: str,
    steering_position_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the configured steering pipeline on fresh count-1..K datasets."""

    if steering_position_mode not in {"last_token", "needle_span"}:
        raise ValueError(
            "steering_position_mode must be 'last_token' or 'needle_span', "
            f"got {steering_position_mode!r}"
        )
    examples, index_lookup, example_summary = steering_eval_examples(datasets)
    if not examples:
        raise ValueError("No steering-test examples were generated")

    if steering_position_mode == "last_token":
        detail_rows, _summary_rows, metadata = run_counting_steering_sweep(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            layers=layers,
            probe_dir=probe_dir,
            betas=betas,
            max_new_tokens=max_new_tokens,
            thinking_mode=thinking_mode,
            vector_source=vector_source,
        )
    else:
        detail_rows, _summary_rows, metadata = run_needle_span_counting_steering_sweep(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            layers=layers,
            probe_dir=probe_dir,
            betas=betas,
            max_new_tokens=max_new_tokens,
            thinking_mode=thinking_mode,
            vector_source=vector_source,
        )

    annotated = annotate_steering_eval_details(
        detail_rows,
        index_lookup=index_lookup,
        steering_position_mode=steering_position_mode,
    )
    summary = summarize_steering_test_eval_details(
        annotated, dataset_ks=sorted(datasets)
    )
    metadata = dict(metadata)
    metadata.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "eval_type": "steering_test_eval",
            "steering_position_mode": steering_position_mode,
            "steering_vector_source": str(vector_source),
            "probe_dir": str(probe_dir),
            "num_eval_datasets": len(datasets),
            "dataset_ks": [int(k) for k in sorted(datasets)],
            "examples": example_summary,
            "summary_tuple_meaning": TUPLE_MEANING,
        }
    )
    return annotated, summary, metadata


def write_json(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path


def write_steering_test_eval_outputs(
    *,
    detail_rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    tables_dir: str | Path,
    tensors_dir: str | Path,
) -> dict[str, str]:
    """Write detailed predictions, compact summary, metadata, and vector refs."""

    table_root = Path(tables_dir)
    tensor_root = Path(tensors_dir)
    table_root.mkdir(parents=True, exist_ok=True)
    tensor_root.mkdir(parents=True, exist_ok=True)
    predictions_path = table_root / "steering_test_eval_predictions.csv"
    summary_path = table_root / "steering_test_eval_summary.csv"
    metadata_path = table_root / "steering_test_eval_metadata.json"
    vector_refs_path = tensor_root / "steering_vector_references.json"

    write_csv(detail_rows, predictions_path)
    write_steering_test_eval_summary_csv(summary_rows, summary_path)
    write_json(metadata_path, metadata)
    write_json(
        vector_refs_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "steering_vector_source": metadata.get("steering_vector_source"),
            "probe_dir": metadata.get("probe_dir"),
            "layers": metadata.get("layers"),
            "sigma_by_layer": metadata.get("sigma_by_layer"),
        },
    )
    return {
        "predictions_csv": str(predictions_path),
        "summary_csv": str(summary_path),
        "metadata_json": str(metadata_path),
        "vector_references_json": str(vector_refs_path),
    }
