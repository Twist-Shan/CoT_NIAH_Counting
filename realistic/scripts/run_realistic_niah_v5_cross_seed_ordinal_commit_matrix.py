#!/usr/bin/env python3
"""Cross-seed causal ordinal-equivariance matrix for unnumbered commits.

Natural post-block states h_1,...,h_9 from a discovery donor trace are each
patched into one fixed natural boundary of a different confirmation trace.
The 9 x 9 matrix scores receiver-native successor bullets 2,...,10.  Row and
column centering removes state-wide and bullet-lexical baselines.  A positive
matching diagonal means donor h_k preferentially selects receiver item k+1
despite unrelated donor/receiver item content.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v4.modeling import capture_post_block_states  # noqa: E402
from realistic_niah_v5.count_stream import (  # noqa: E402
    _prefill_with_state_replacements,
    build_answer_source_registry,
)
from realistic_niah_v5.unnumbered_counter_restore import (  # noqa: E402
    audit_no_count_enumeration_trace,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
)
from scripts.run_realistic_niah_v5_cross_seed_counter_recurrence import (  # noqa: E402
    prefix_through_boundary,
    score_native_item_candidates,
)
from scripts.run_realistic_niah_v5_unindexed_full_commit_successor import (  # noqa: E402
    _transition_candidates,
)


def _pairs(values: Sequence[int]) -> tuple[tuple[int, int], ...]:
    flat = tuple(int(value) for value in values)
    if not flat or len(flat) % 2:
        raise ValueError("Seed pairs must be supplied as donor receiver groups")
    result = tuple(zip(flat[::2], flat[1::2]))
    if any(donor == receiver for donor, receiver in result):
        raise ValueError("Cross-seed donor and receiver must differ")
    donors = [donor for donor, _receiver in result]
    receivers = [receiver for _donor, receiver in result]
    if len(set(donors)) != len(donors) or len(set(receivers)) != len(receivers):
        raise ValueError("Donor and receiver seeds must each be unique")
    if set(donors) & set(receivers):
        raise ValueError("Discovery donors and confirmation receivers must be disjoint")
    return result


def _read_rows(path: Path, seeds: Sequence[int]) -> dict[int, dict[str, Any]]:
    wanted = set(int(value) for value in seeds)
    selected: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        seed = int(row.get("seed", -1))
        if seed not in wanted:
            continue
        if seed in selected:
            raise ValueError(f"Duplicate selected seed {seed}")
        audit = audit_no_count_enumeration_trace(row)
        if not bool(audit["eligible"]):
            raise ValueError(f"Seed {seed} fails unindexed gate: {audit['reasons']}")
        if int(row.get("gold_count", -1)) != 10:
            raise ValueError(f"Seed {seed} is not N=10")
        selected[seed] = row
    if set(selected) != wanted:
        raise ValueError(f"Missing selected seeds {sorted(wanted-set(selected))}")
    return selected


def _record_strings(row: Mapping[str, Any]) -> tuple[str, ...]:
    records = row.get("gold_records")
    if not isinstance(records, list) or len(records) != 10:
        raise ValueError("Expected ten gold records")
    return tuple(
        json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for record in records
    )


def _matrix_metrics(matrix: np.ndarray) -> dict[str, Any]:
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("Ordinal score matrix must be square and nontrivial")
    centered = (
        matrix
        - matrix.mean(axis=1, keepdims=True)
        - matrix.mean(axis=0, keepdims=True)
        + matrix.mean()
    )
    diagonal = np.diag(centered)
    predictions = np.argmax(centered, axis=1)
    expected = np.arange(matrix.shape[0])
    pairwise: list[float] = []
    for left in range(matrix.shape[0]):
        for right in range(left + 1, matrix.shape[0]):
            pairwise.append(
                float(
                    (matrix[left, left] - matrix[left, right])
                    - (matrix[right, left] - matrix[right, right])
                )
            )
    shifts = {
        str(shift): float(
            np.mean([centered[row, (row + shift) % len(centered)] for row in expected])
        )
        for shift in range(1, len(centered))
    }
    return {
        "doubly_centered_matrix": centered.tolist(),
        "matching_diagonal_mean": float(np.mean(diagonal)),
        "matching_diagonal_min": float(np.min(diagonal)),
        "assignment_exact_accuracy": float(np.mean(predictions == expected)),
        "predicted_successor_occurrences": [int(value) + 2 for value in predictions],
        "mean_pairwise_matching_double_difference": float(np.mean(pairwise)),
        "positive_pairwise_matching_fraction": float(np.mean(np.asarray(pairwise) > 0)),
        "cyclic_shift_diagonal_means": shifts,
        "matching_minus_best_cyclic_shift": float(
            np.mean(diagonal) - max(shifts.values())
        ),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--seed-pairs", type=int, nargs="+", required=True)
    parser.add_argument("--donor-occurrences", type=int, nargs="+", default=tuple(range(1, 10)))
    parser.add_argument("--receiver-occurrence", type=int, default=5)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs = _pairs(args.seed_pairs)
    occurrences = tuple(int(value) for value in args.donor_occurrences)
    if not occurrences or len(set(occurrences)) != len(occurrences):
        raise ValueError("Donor occurrences must be unique and nonempty")
    if min(occurrences) < 1 or max(occurrences) > 9:
        raise ValueError("Every donor occurrence must own a successor")
    receiver_occurrence = int(args.receiver_occurrence)
    if not 1 <= receiver_occurrence < 10:
        raise ValueError("Receiver occurrence must own a successor")
    seeds = tuple(sorted({seed for pair in pairs for seed in pair}))
    rows = _read_rows(args.generations, seeds)
    model, tokenizer, adapter = _model(args)
    layers = tuple(int(value) for value in args.layers)
    if len(set(layers)) != len(layers):
        raise ValueError("Patch layers must be unique")
    if not layers or min(layers) < 0 or max(layers) >= int(adapter.num_layers) - 1:
        raise ValueError("Every patch layer must leave a downstream block")

    results: list[dict[str, Any]] = []
    for pair_index, (donor_seed, receiver_seed) in enumerate(pairs, start=1):
        donor_row = rows[donor_seed]
        receiver_row = rows[receiver_seed]
        donor_encoding, donor_registry = build_answer_source_registry(donor_row, tokenizer)
        receiver_encoding, receiver_registry = build_answer_source_registry(
            receiver_row, tokenizer
        )
        donor_endpoints = tuple(int(end) - 1 for _start, end in donor_registry.trace_items)
        receiver_endpoints = tuple(
            int(end) - 1 for _start, end in receiver_registry.trace_items
        )
        _logits, captured = capture_post_block_states(
            model,
            adapter,
            donor_encoding,
            tuple(donor_endpoints[k - 1] for k in occurrences),
            layers=layers,
        )
        prefix = prefix_through_boundary(
            receiver_encoding, receiver_endpoints[receiver_occurrence - 1]
        )
        candidates = _transition_candidates(
            receiver_encoding,
            receiver_registry,
            receiver_occurrence=receiver_occurrence,
        )
        donor_records = set(_record_strings(donor_row))
        receiver_records = set(_record_strings(receiver_row))
        for layer in layers:
            sum_rows: list[list[float]] = []
            mean_rows: list[list[float]] = []
            patch_norms: list[float] = []
            for state in captured[layer]:
                prefill, applications, norm = _prefill_with_state_replacements(
                    model,
                    adapter,
                    prefix,
                    layer=layer,
                    positions=(int(prefix.sequence_length) - 1,),
                    states=torch.as_tensor(state).reshape(1, -1),
                )
                if int(applications) != 1:
                    raise RuntimeError("Cross-seed commit patch did not apply once")
                scored = score_native_item_candidates(
                    model,
                    prefix,
                    prefill,
                    candidates,
                    target=receiver_occurrence + 1,
                )
                sum_rows.append(
                    [float(value) for value in scored["sum_logprob_scores"]]
                )
                mean_rows.append(
                    [float(value) for value in scored["mean_logprob_scores"]]
                )
                patch_norms.append(float(norm))
            candidate_indices = [occurrence for occurrence in occurrences]
            sum_matrix = np.asarray(sum_rows, dtype=float)[:, candidate_indices]
            mean_matrix = np.asarray(mean_rows, dtype=float)[:, candidate_indices]
            result = {
                "schema_version": "cross_seed_ordinal_commit_matrix_v1",
                "model_label": str(args.model),
                "donor_seed": donor_seed,
                "receiver_seed": receiver_seed,
                "pair_index": pair_index,
                "layer": layer,
                "receiver_occurrence": receiver_occurrence,
                "donor_occurrences": list(occurrences),
                "target_successor_occurrences": [value + 1 for value in occurrences],
                "sum_logprob_matrix": sum_matrix.tolist(),
                "mean_logprob_matrix": mean_matrix.tolist(),
                "sum_logprob_metrics": _matrix_metrics(sum_matrix),
                "mean_logprob_metrics": _matrix_metrics(mean_matrix),
                "patch_realized_l2_norms": patch_norms,
                "exact_record_overlap_count": len(donor_records & receiver_records),
                "donor_receiver_same_seed": False,
                "diagnostic_total_suffix_used": False,
                "visible_item_indices_used": False,
                "count_subspace_used": False,
                "selection_uses_outcomes": False,
                "format_conditioned_auxiliary": True,
            }
            results.append(result)
            print(
                f"[cross-seed-ordinal] {pair_index}/{len(pairs)} "
                f"donor={donor_seed} receiver={receiver_seed} L{layer} "
                f"sum_acc={result['sum_logprob_metrics']['assignment_exact_accuracy']:.3f} "
                f"sum_diag={result['sum_logprob_metrics']['matching_diagonal_mean']:+.3f}",
                flush=True,
            )

    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(args.output / "pair_matrices.jsonl", results)
    summary = [
        {
            "layer": layer,
            **{
                f"{metric}.{key}": float(
                    np.mean(
                        [row[metric][key] for row in results if row["layer"] == layer]
                    )
                )
                for metric in ("sum_logprob_metrics", "mean_logprob_metrics")
                for key in (
                    "matching_diagonal_mean",
                    "matching_diagonal_min",
                    "assignment_exact_accuracy",
                    "mean_pairwise_matching_double_difference",
                    "positive_pairwise_matching_fraction",
                    "matching_minus_best_cyclic_shift",
                )
            },
        }
        for layer in layers
    ]
    manifest = {
        "schema_version": "cross_seed_ordinal_commit_matrix_v1",
        "status": "PASS",
        "model_label": str(args.model),
        "layers": list(layers),
        "seed_pairs": [list(value) for value in pairs],
        "pair_count": len(pairs),
        "donor_occurrences": list(occurrences),
        "receiver_occurrence": receiver_occurrence,
        "summary": summary,
        "exact_record_overlap_counts": [
            int(value["exact_record_overlap_count"]) for value in results
        ],
        "diagnostic_total_suffix_used": False,
        "visible_item_indices_used": False,
        "count_subspace_used": False,
        "selection_uses_outcomes": False,
        "formal_frozen_prompt_claim_allowed": False,
        "claim_scope": "format-conditioned unnumbered reasoning auxiliary",
    }
    _atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
