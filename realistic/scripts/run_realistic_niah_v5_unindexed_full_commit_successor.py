#!/usr/bin/env python3
"""Test whether an unnumbered item-boundary commit controls its successor.

For each frozen same-trace receiver/donor pair, this runner replaces the
receiver's post-block boundary vector by a complete natural donor vector and
scores the ten native bullet strings as possible continuations.  The primary
effects are paired differences, so fixed lexical preferences among the ten
items cancel within a pair.  No numeric Total suffix or count subspace is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
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
from realistic_niah_v5.native_loop import (  # noqa: E402
    full_commit_specificity_condition_states,
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


CONDITIONS = (
    "self_patch",
    "full_donor_patch",
    "full_delta_norm_matched_orthogonal_r0",
    "full_delta_norm_matched_orthogonal_r1",
    "full_delta_norm_matched_orthogonal_r2",
    "opposite_full_delta_patch",
    "shuffled_natural_donor_patch",
)


def _read_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        request_id = str(row["request_id"])
        if request_id in rows:
            raise ValueError(f"Duplicate request_id {request_id}")
        audit = audit_no_count_enumeration_trace(row)
        if not bool(audit["eligible"]):
            raise ValueError(
                f"Seed {row.get('seed')} fails unindexed gate: {audit['reasons']}"
            )
        if int(row.get("gold_count", -1)) != 10:
            raise ValueError(f"Seed {row.get('seed')} is not N=10")
        rows[request_id] = row
    return rows


def _validated_plan(path: Path, rows: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    plan = pd.read_csv(path)
    required = {
        "request_id",
        "pair_sha256",
        "seed",
        "receiver_occurrence",
        "donor_occurrence",
        "donor_offset",
        "shuffled_donor_occurrence",
        "selection_rank_used",
        "specificity_outcome_blind",
    }
    missing = sorted(required - set(plan.columns))
    if missing:
        raise ValueError(f"Frozen plan lacks columns {missing}")
    if plan["pair_sha256"].astype(str).duplicated().any():
        raise ValueError("Frozen plan contains duplicate pair hashes")
    if plan["selection_rank_used"].map(str).str.lower().isin({"true", "1"}).any():
        raise ValueError("Frozen plan used a selection rank")
    if not plan["specificity_outcome_blind"].map(str).str.lower().isin(
        {"true", "1"}
    ).all():
        raise ValueError("Frozen plan is not outcome blind")
    missing_rows = sorted(set(plan["request_id"].astype(str)) - set(rows))
    if missing_rows:
        raise ValueError(f"Frozen plan references missing rows: {missing_rows}")
    for value in plan.itertuples(index=False):
        receiver = int(value.receiver_occurrence)
        donor = int(value.donor_occurrence)
        shuffled = int(value.shuffled_donor_occurrence)
        if donor - receiver != int(value.donor_offset):
            raise ValueError("Frozen donor offset is inconsistent")
        if len({receiver, donor, shuffled}) != 3:
            raise ValueError("Receiver, donor, and shuffled donor must be distinct")
        if not all(1 <= item < 10 for item in (receiver, donor, shuffled)):
            raise ValueError("Every commit must own a successor")
    return plan.sort_values(["seed", "donor_offset"], kind="stable").reset_index(
        drop=True
    )


def _transition_candidates(
    encoding: Any,
    registry: Any,
    *,
    receiver_occurrence: int,
) -> dict[int, tuple[int, ...]]:
    items = tuple((int(start), int(end)) for start, end in registry.trace_items)
    receiver = int(receiver_occurrence)
    if len(items) != 10 or not 1 <= receiver < 10:
        raise ValueError("Expected ten items and a nonterminal receiver")
    receiver_end = items[receiver - 1][1]
    natural_next_start = items[receiver][0]
    separator = tuple(
        int(value) for value in encoding.input_ids[receiver_end:natural_next_start]
    )
    candidates = {
        occurrence: separator
        + tuple(int(value) for value in encoding.input_ids[start:end])
        for occurrence, (start, end) in enumerate(items, start=1)
    }
    if any(not tokens for tokens in candidates.values()):
        raise ValueError("A successor candidate is empty")
    if len(set(candidates.values())) != 10:
        raise ValueError("Successor candidates are not unique")
    return candidates


@torch.inference_mode()
def _condition_scores(
    model: Any,
    adapter: Any,
    prefix: Any,
    *,
    layer: int,
    position: int,
    states: Mapping[str, torch.Tensor],
    candidates: Mapping[int, Sequence[int]],
    receiver_target: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        state = torch.as_tensor(states[condition]).reshape(1, -1)
        prefill, applications, norm = _prefill_with_state_replacements(
            model,
            adapter,
            prefix,
            layer=int(layer),
            positions=(int(position),),
            states=state,
        )
        scored = score_native_item_candidates(
            model,
            prefix,
            prefill,
            candidates,
            target=int(receiver_target),
        )
        results[condition] = {
            **scored,
            "patch_applications": int(applications),
            "patch_realized_l2_norm": float(norm),
        }
    return results


def _contrast_rows(trials: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_hash, group in trials.groupby("pair_sha256", sort=False):
        by_condition = {
            str(row.condition): row for row in group.itertuples(index=False)
        }
        full = by_condition["full_donor_patch"]
        self_row = by_condition["self_patch"]
        shuffled = by_condition["shuffled_natural_donor_patch"]
        random_values = [
            by_condition[f"full_delta_norm_matched_orthogonal_r{index}"]
            for index in range(3)
        ]
        common = {
            "pair_sha256": str(pair_hash),
            "seed": int(full.seed),
            "receiver_occurrence": int(full.receiver_occurrence),
            "donor_occurrence": int(full.donor_occurrence),
            "donor_offset": int(full.donor_offset),
            "shuffled_donor_occurrence": int(full.shuffled_donor_occurrence),
        }
        rows.extend(
            [
                {
                    **common,
                    "estimand": "full_vs_self_intended_successor",
                    "effect": float(
                        full.donor_vs_receiver_sum_logodds
                        - self_row.donor_vs_receiver_sum_logodds
                    ),
                },
                {
                    **common,
                    "estimand": "full_vs_mean_norm_controls_intended_successor",
                    "effect": float(
                        full.donor_vs_receiver_sum_logodds
                        - np.mean(
                            [value.donor_vs_receiver_sum_logodds for value in random_values]
                        )
                    ),
                },
                {
                    **common,
                    "estimand": "ordinal_donor_identity_double_difference",
                    "effect": float(
                        full.donor_vs_shuffled_sum_logodds
                        - shuffled.donor_vs_shuffled_sum_logodds
                    ),
                },
            ]
        )
    return rows


def _seed_summary(contrasts: pd.DataFrame) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for estimand, group in contrasts.groupby("estimand", sort=True):
        seed_values = group.groupby("seed", sort=True)["effect"].mean()
        values = seed_values.to_numpy(float)
        result.append(
            {
                "estimand": str(estimand),
                "seed_count": int(len(values)),
                "pair_count": int(len(group)),
                "mean_seed_effect": float(np.mean(values)),
                "median_seed_effect": float(np.median(values)),
                "positive_seed_fraction": float(np.mean(values > 0)),
                "seed_effects": [float(value) for value in values],
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--random-seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _read_rows(args.generations)
    plan = _validated_plan(args.plan, rows)
    model, tokenizer, adapter = _model(args)
    if not 0 <= int(args.layer) < int(adapter.num_layers) - 1:
        raise ValueError("Patch layer must leave a downstream decoder block")

    trials: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(plan.itertuples(index=False), start=1):
        row = rows[str(pair.request_id)]
        encoding, registry = build_answer_source_registry(row, tokenizer)
        endpoints = tuple(int(end) - 1 for _start, end in registry.trace_items)
        receiver = int(pair.receiver_occurrence)
        donor = int(pair.donor_occurrence)
        shuffled = int(pair.shuffled_donor_occurrence)
        positions = (endpoints[receiver - 1], endpoints[donor - 1], endpoints[shuffled - 1])
        _logits, captured = capture_post_block_states(
            model,
            adapter,
            encoding,
            positions,
            layers=[int(args.layer)],
        )
        receiver_state, donor_state, shuffled_state = captured[int(args.layer)]
        controls, control_audit = full_commit_specificity_condition_states(
            receiver_state,
            donor_state,
            shuffled_donor_state=shuffled_state,
            random_seed=int(args.random_seed) + int(pair.seed) * 1009,
            random_replicates=3,
        )
        states = {
            "self_patch": receiver_state,
            "full_donor_patch": donor_state,
            **controls,
        }
        prefix = prefix_through_boundary(encoding, endpoints[receiver - 1])
        candidates = _transition_candidates(
            encoding,
            registry,
            receiver_occurrence=receiver,
        )
        scored = _condition_scores(
            model,
            adapter,
            prefix,
            layer=int(args.layer),
            position=int(prefix.sequence_length) - 1,
            states=states,
            candidates=candidates,
            receiver_target=receiver + 1,
        )
        for condition, value in scored.items():
            sums = value["sum_logprob_scores"]
            means = value["mean_logprob_scores"]
            common = {
                "schema_version": "unindexed_full_commit_successor_v1",
                "model_label": str(args.model),
                "request_id": str(pair.request_id),
                "pair_sha256": str(pair.pair_sha256),
                "seed": int(pair.seed),
                "condition": condition,
                "layer": int(args.layer),
                "receiver_occurrence": receiver,
                "donor_occurrence": donor,
                "donor_offset": int(pair.donor_offset),
                "shuffled_donor_occurrence": shuffled,
                "receiver_successor_occurrence": receiver + 1,
                "donor_successor_occurrence": donor + 1,
                "shuffled_donor_successor_occurrence": shuffled + 1,
                "diagnostic_total_suffix_used": False,
                "visible_item_indices_used": False,
                "count_subspace_used": False,
                "full_commit_vector_used": True,
                "selection_rank_used": False,
                "format_conditioned_auxiliary": True,
                "donor_vs_receiver_sum_logodds": float(
                    sums[donor] - sums[receiver]
                ),
                "donor_vs_shuffled_sum_logodds": float(
                    sums[donor] - sums[shuffled]
                ),
                "donor_vs_receiver_mean_logodds": float(
                    means[donor] - means[receiver]
                ),
                "donor_vs_shuffled_mean_logodds": float(
                    means[donor] - means[shuffled]
                ),
                **value,
            }
            if condition in control_audit["condition_audit"]:
                common.update(control_audit["condition_audit"][condition])
            trials.append(common)
        print(
            f"[unindexed-full-commit] {pair_index}/{len(plan)} "
            f"seed={pair.seed} r={receiver} d={donor} wrong={shuffled}",
            flush=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(args.output / "trials.jsonl", trials)
    trial_frame = pd.DataFrame(trials)
    contrast_rows = _contrast_rows(trial_frame)
    _atomic_jsonl(args.output / "pair_contrasts.jsonl", contrast_rows)
    summary = _seed_summary(pd.DataFrame(contrast_rows))
    _atomic_json(
        args.output / "manifest.json",
        {
            "schema_version": "unindexed_full_commit_successor_v1",
            "status": "PASS",
            "model_label": str(args.model),
            "layer": int(args.layer),
            "pair_count": int(len(plan)),
            "seed_count": int(plan["seed"].nunique()),
            "conditions": list(CONDITIONS),
            "summary": summary,
            "diagnostic_total_suffix_used": False,
            "visible_item_indices_used": False,
            "count_subspace_used": False,
            "selection_uses_outcomes": False,
            "formal_frozen_prompt_claim_allowed": False,
            "claim_scope": "format-conditioned unnumbered reasoning auxiliary",
        },
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
