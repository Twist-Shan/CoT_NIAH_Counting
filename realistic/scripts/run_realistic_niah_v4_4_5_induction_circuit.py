from __future__ import annotations

"""Run follow-up 22: synthetic relation assay plus canonical edge removal."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from realistic_niah_v4.behavior import parse_numeric_completion
from realistic_niah_v4.prompts import PromptEncoding, render_v4_prompt
from realistic_niah_v4.spec import V4Config, resolve_model_spec
from realistic_niah_v4.stimuli import load_stimuli
from realistic_niah_v4_4_3.interventions import (
    candidate_sequence_metrics,
    capture_query_bundle,
)
from realistic_niah_v4_4_5.followup_edges import (
    freeze_anchor_token_from_encodings,
    natural_edge_delta,
    registered_forbidden_positions,
    repeated_anchor_candidates,
    select_attention_mass_control,
)
from realistic_niah_v4_4_5.followup_runtime import (
    capture_endpoint_states,
    position_pre_o_deltas,
    selective_position_attention_outputs,
)
from realistic_niah_v4_4_5.restoration import (
    active_broad_metrics,
    generate_answer_completion_from_prefill,
)
from realistic_niah_v4.modeling import load_registered_model


SCHEMA = "realistic_niah_v4_4_5_induction_circuit_detail_v1"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != dict(value):
        raise RuntimeError(f"Existing frozen artifact differs: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def strict_fields(result: Mapping[str, Any], gold: int) -> dict[str, Any]:
    parsed = parse_numeric_completion(str(result.get("completion_text", "")))
    prediction = parsed.get("parsed_count")
    prediction = None if prediction is None else int(prediction)
    return {
        "strict_prediction": prediction,
        "strict_correct": bool(prediction == int(gold)),
        "strict_absolute_error": 10 if prediction is None else abs(prediction - int(gold)),
        "strict_completion": str(result.get("completion_text", "")),
        "strict_format_valid": bool(parsed.get("format_valid", False)),
    }


def completed_keys(path: Path) -> set[tuple[int, int, str]]:
    result: set[tuple[int, int, str]] = set()
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (int(row["seed"]), int(row["gold_count"]), str(row["arm"]))
        if key in result:
            raise RuntimeError(f"Duplicate existing row {key}")
        result.add(key)
    return result


def single_token_pool(tokenizer: Any, *, minimum: int = 256) -> tuple[int, ...]:
    special = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    candidates = []
    for token_id in sorted(set(int(value) for value in tokenizer.get_vocab().values())):
        if token_id in special:
            continue
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if not text.strip() or any(ord(char) < 32 for char in text):
            continue
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded == [token_id]:
            candidates.append(token_id)
        if len(candidates) >= int(minimum):
            break
    if len(candidates) < int(minimum):
        raise RuntimeError(f"Only {len(candidates)} stable single-token items were found")
    return tuple(candidates)


def synthetic_encoding(
    *,
    model_label: str,
    base_ids: Sequence[int],
    query_position: int,
) -> PromptEncoding:
    ids = tuple(int(value) for value in base_ids)
    if len(ids) != int(query_position) + 1:
        raise ValueError("Synthetic query must be the final token")
    return PromptEncoding(
        stimulus_id="synthetic_induction",
        design_variant="synthetic_induction",
        seed=0,
        split="synthetic",
        count=0,
        model_label=model_label,
        answer_format="numeric",
        text="token_id_synthetic_assay",
        generation_prompt="token_id_synthetic_assay",
        input_ids=ids,
        attention_mask=tuple([1] * len(ids)),
        query_position=int(query_position),
        slot_spans=(),
        needle_spans=(),
        hard_negative_spans=(),
        count_candidate_texts=(),
        count_candidate_answer_token_ids=(),
        count_candidate_token_ids=(),
    )


@torch.inference_mode()
def run_synthetic_assay(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    *,
    model_label: str,
    candidate_heads: Sequence[Sequence[int]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool = single_token_pool(tokenizer)
    rng = random.Random(int(config["random_seed"]))
    layers = tuple(sorted({int(layer) for layer, _head in candidate_heads}))
    length = int(config["sequence_length"])
    a = int(config["anchor_a_position"])
    x = int(config["successor_a_position"])
    b = int(config["anchor_b_position"])
    y = int(config["successor_b_position"])
    q = int(config["query_position"])
    if length != q + 1 or not (a + 1 == x < b and b + 1 == y < q):
        raise ValueError("Synthetic position registry is inconsistent")
    rows: list[dict[str, Any]] = []
    for sequence_index in range(int(config["sequences"])):
        chosen = rng.sample(pool, 8)
        anchor_a, anchor_b, unique_anchor, successor_a, successor_b, ordinary, fill1, fill2 = chosen
        # Keep all unregistered positions free of the assay tokens. Otherwise
        # a random filler collision can create an unintended previous match.
        filler_pool = [token for token in pool if token not in set(chosen)]
        if len(filler_pool) < length:
            raise RuntimeError("Synthetic token pool is too small for collision-free filler")
        filler = rng.sample(filler_pool, length)
        if getattr(tokenizer, "bos_token_id", None) is not None:
            filler[0] = int(tokenizer.bos_token_id)
        conditions: dict[str, tuple[list[int], int, int]] = {}
        repeated = list(filler)
        repeated[a], repeated[x], repeated[b], repeated[y], repeated[q] = (
            anchor_a,
            successor_a,
            anchor_b,
            successor_b,
            anchor_a,
        )
        conditions["repeated_consistent"] = (repeated, x, y)
        unique = list(repeated)
        unique[q] = unique_anchor
        conditions["unique_anchor"] = (unique, x, y)
        reassigned = list(repeated)
        reassigned[a], reassigned[b] = anchor_b, anchor_a
        conditions["successor_reassignment"] = (reassigned, y, x)
        ordinary_repeat = list(repeated)
        # Preserve a repeated token at the two registered predecessor sites,
        # but make the final query a different, previously unseen token. This
        # controls repeated-token/position statistics without supplying a
        # query-matched previous-occurrence-to-successor relation.
        ordinary_repeat[a], ordinary_repeat[b], ordinary_repeat[q] = (
            ordinary,
            ordinary,
            unique_anchor,
        )
        conditions["same_position_ordinary_repeat"] = (ordinary_repeat, x, y)
        for condition in config["conditions"]:
            ids, correct, control = conditions[str(condition)]
            encoding = synthetic_encoding(
                model_label=model_label, base_ids=ids, query_position=q
            )
            attention, starts = selective_position_attention_outputs(
                model, adapter, encoding, query_position=q, layers=layers
            )
            for rank, (layer, head) in enumerate(candidate_heads, start=1):
                layer, head = int(layer), int(head)
                start = int(starts[layer])
                correct_relative, control_relative = correct - start, control - start
                if min(correct_relative, control_relative) < 0 or max(
                    correct_relative, control_relative
                ) >= attention[layer].shape[-1]:
                    raise RuntimeError("Synthetic successor lies outside attention row")
                correct_mass = float(attention[layer][head, correct_relative])
                control_mass = float(attention[layer][head, control_relative])
                rows.append(
                    {
                        "schema_version": "realistic_niah_v4_4_5_induction_synthetic_v1",
                        "model_label": model_label,
                        "sequence_index": sequence_index,
                        "condition": str(condition),
                        "canonical_rank": rank,
                        "layer": layer,
                        "head": head,
                        "correct_successor_position": int(correct),
                        "matched_non_successor_position": int(control),
                        "correct_successor_mass": correct_mass,
                        "matched_non_successor_mass": control_mass,
                        "relation_advantage": correct_mass - control_mass,
                    }
                )
    summaries: list[dict[str, Any]] = []
    threshold = float(config["minimum_relation_advantage"])
    collapse_ratio = float(config["maximum_collapse_ratio"])
    for rank, (layer, head) in enumerate(candidate_heads, start=1):
        selected = [
            row
            for row in rows
            if int(row["layer"]) == int(layer) and int(row["head"]) == int(head)
        ]
        by_condition = {
            condition: np.asarray(
                [
                    float(row["relation_advantage"])
                    for row in selected
                    if row["condition"] == condition
                ],
                dtype=float,
            )
            for condition in config["conditions"]
        }
        repeated_mean = float(by_condition["repeated_consistent"].mean())
        reassigned_mean = float(by_condition["successor_reassignment"].mean())
        unique_abs = float(np.abs(by_condition["unique_anchor"]).mean())
        ordinary_abs = float(
            np.abs(by_condition["same_position_ordinary_repeat"]).mean()
        )
        signal = max(1e-12, 0.5 * (abs(repeated_mean) + abs(reassigned_mean)))
        passed = bool(
            repeated_mean >= threshold
            and reassigned_mean >= threshold
            and unique_abs <= collapse_ratio * signal
            and ordinary_abs <= collapse_ratio * signal
        )
        summaries.append(
            {
                "canonical_rank": rank,
                "layer": int(layer),
                "head": int(head),
                "repeated_relation_mean": repeated_mean,
                "reassignment_follow_mean": reassigned_mean,
                "unique_anchor_abs_mean": unique_abs,
                "ordinary_repeat_abs_mean": ordinary_abs,
                "collapse_reference": signal,
                "passed": passed,
            }
        )
    retained = [row for row in summaries if row["passed"]][
        : int(config["max_retained_heads"])
    ]
    audit = {
        "schema_version": "realistic_niah_v4_4_5_induction_synthetic_audit_v1",
        "status": "PASS",
        "sequences": int(config["sequences"]),
        "conditions": list(config["conditions"]),
        "candidate_heads": len(candidate_heads),
        "rows": len(rows),
        "head_summaries": summaries,
        "retained_heads": retained,
        "decision_rule": (
            "canonical-rank-first among heads passing repeated and reassignment "
            "advantages plus unique/ordinary collapse; maximum one primary head"
        ),
    }
    return rows, audit


def identity_repeated_pairs(encoding: PromptEncoding, anchor_token: int) -> tuple[tuple[int, int], ...]:
    # N=1 is a registered structural negative control: there is no previous
    # occurrence, hence no previous-match -> successor edge to remove.  This
    # is different from losing the frozen anchor on an N>=2 prompt, which
    # remains a hard registration failure.
    if len(encoding.needle_spans) < 2:
        return ()
    candidates = repeated_anchor_candidates(encoding)
    if int(anchor_token) not in candidates:
        raise RuntimeError(
            "Frozen anchor is absent from a relation-present confirmation prompt"
        )
    occurrences = tuple(candidates[int(anchor_token)])
    return tuple(
        (int(occurrences[index][0]), int(occurrences[index - 1][1]))
        for index in range(1, len(occurrences))
    )


def broad_summary(bundle: Any, encoding: PromptEncoding, *, layer: int, heads: Sequence[int]) -> dict[str, float]:
    rows = active_broad_metrics(
        bundle.alpha_by_layer[int(layer)],
        key_start=bundle.alpha_key_start_by_layer[int(layer)],
        spans=encoding.needle_spans,
    )
    selected = [row for row in rows if int(row["head"]) in set(int(v) for v in heads)]
    if len(selected) != len(set(int(value) for value in heads)):
        raise RuntimeError("Frozen retrieval head registry was not fully captured")
    return {
        "retrieval_bank_needle_mass_mean": float(np.mean([row["needle_mass"] for row in selected])),
        "retrieval_bank_coverage_mean": float(np.mean([row["coverage"] for row in selected])),
        "retrieval_bank_broad_score_mean": float(np.mean([row["broad_score"] for row in selected])),
    }


@torch.inference_mode()
def evaluate_arm(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    encoding: PromptEncoding,
    *,
    endpoint_layer: int,
    retrieval_layer: int,
    retrieval_heads: Sequence[int],
    deltas: Mapping[tuple[int, int, int], torch.Tensor],
    max_new_tokens: int,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, Any]]:
    with position_pre_o_deltas(adapter, deltas=deltas) as intervention_audit:
        with capture_endpoint_states(
            adapter, encoding, layer=int(endpoint_layer)
        ) as endpoint_audit:
            bundle = capture_query_bundle(
                model,
                adapter,
                encoding,
                layers=(int(retrieval_layer),),
                capture_attention=True,
                capture_values=False,
                audit_cache_equivalence=False,
                retain_prefill_output=True,
            )
        if bundle.reusable_prefill_output is None:
            raise RuntimeError("Induction arm retained no causal prefill")
        strict = generate_answer_completion_from_prefill(
            model,
            tokenizer,
            encoding,
            bundle.reusable_prefill_output,
            max_new_tokens=int(max_new_tokens),
        )
    states = endpoint_audit.get("states")
    if not isinstance(states, torch.Tensor) or states.shape[0] != len(
        encoding.needle_spans
    ):
        raise RuntimeError("Endpoint-state capture failed")
    metrics = {
        **candidate_sequence_metrics(bundle.candidate_log_scores, encoding),
        **strict_fields(strict, int(encoding.count)),
        **broad_summary(
            bundle, encoding, layer=int(retrieval_layer), heads=retrieval_heads
        ),
        "attention_source": "cache_reconstruction_only_no_equivalence_comparison",
        "strict_generation_reused_causal_prefill": True,
    }
    return metrics, states, dict(intervention_audit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("Qwen3-8B", "Gemma4-E4B"))
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--v4-config", default="configs/realistic_niah_v4_4_5_stimuli.json")
    parser.add_argument("--experiment-config", default="configs/realistic_niah_v4_4_5_induction_circuit.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    experiment_path = Path(args.experiment_config).resolve()
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if experiment.get("schema_version") != "realistic_niah_v4_4_5_induction_circuit_v1":
        raise ValueError("Unexpected induction-circuit schema")
    stimuli_path = Path(args.stimuli).resolve()
    if sha256(stimuli_path) != str(experiment["stimulus_sha256"]):
        raise RuntimeError("Frozen stimulus hash mismatch")
    model_label = str(args.model)
    model_config = experiment["models"][model_label]
    config = V4Config.from_json(args.v4_config)
    spec = resolve_model_spec(model_label)
    root = Path(args.output_dir).resolve() / model_label
    root.mkdir(parents=True, exist_ok=True)
    rows = {
        (int(row["seed"]), int(row["gold_count"])): row
        for row in load_stimuli(stimuli_path)
        if str(row.get("design_variant")) == "v4.4"
    }
    required = {
        (seed, count)
        for seed in tuple(experiment["discovery_seeds"]) + tuple(experiment["confirmation_seeds"])
        for count in experiment["counts"]
    }
    if not required.issubset(rows):
        raise RuntimeError(f"Missing canonical stimuli: {sorted(required - set(rows))[:5]}")
    provenance = {
        "schema_version": "realistic_niah_v4_4_5_induction_run_v1",
        "model": model_label,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
        "experiment_config": str(experiment_path),
        "experiment_config_sha256": sha256(experiment_path),
        "stimuli": str(stimuli_path),
        "stimuli_sha256": sha256(stimuli_path),
        "edge_intervention_boundary": experiment["canonical"]["intervention"],
        "scientific_boundary": "natural alpha-times-V edge contribution removal; not recomputed counterfactual QK routing and not a unique-channel claim",
    }
    write_json(root / "run_provenance.json", provenance)

    model, tokenizer, adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=experiment["model_torch_dtype"],
        attention_backend=experiment["attention_prefix_backend"],
    )
    synthetic_rows, synthetic_audit = run_synthetic_assay(
        model,
        tokenizer,
        adapter,
        model_label=model_label,
        candidate_heads=model_config["candidate_heads"],
        config=experiment["synthetic"],
    )
    synthetic_path = root / "synthetic_rows.jsonl"
    if not synthetic_path.exists():
        for row in synthetic_rows:
            append_jsonl(synthetic_path, row)
    elif sum(1 for line in synthetic_path.read_text().splitlines() if line.strip()) != len(synthetic_rows):
        raise RuntimeError("Existing synthetic rows have the wrong size")
    write_json(root / "synthetic_audit.json", synthetic_audit)
    retained = synthetic_audit["retained_heads"]
    if not retained:
        completion = {
            "schema_version": "realistic_niah_v4_4_5_induction_complete_v1",
            "status": "complete_no_retained_head",
            "model": model_label,
            "synthetic_rows": len(synthetic_rows),
            "canonical_rows": 0,
            "decision": "classical induction criteria failed before canonical causal confirmation",
        }
        write_json(root / "complete.json", completion)
        print(json.dumps(completion, indent=2))
        return
    source_layer, source_head = int(retained[0]["layer"]), int(retained[0]["head"])

    discovery_encodings = [
        render_v4_prompt(
            rows[(int(seed), 10)], tokenizer=tokenizer, model_spec=spec, config=config, answer_format="numeric"
        )
        for seed in experiment["discovery_seeds"]
    ]
    anchor_token, anchor_audit = freeze_anchor_token_from_encodings(discovery_encodings)
    registration = {
        "schema_version": "realistic_niah_v4_4_5_induction_registration_v1",
        "source_layer": source_layer,
        "source_head": source_head,
        "anchor_token_id": int(anchor_token),
        "anchor_token_text": tokenizer.decode([int(anchor_token)]),
        **anchor_audit,
    }
    write_json(root / "canonical_registration.json", registration)

    detail_path = root / "detail.jsonl"
    state_dir = root / "endpoint_states"
    completed = completed_keys(detail_path)
    arms = tuple(str(value) for value in experiment["canonical"]["arms"])
    structural_counts = {
        int(value)
        for value in experiment["canonical"]["structural_no_previous_match_counts"]
    }
    primary_relation_counts = {
        int(value) for value in experiment["canonical"]["primary_relation_counts"]
    }
    if structural_counts | primary_relation_counts != {
        int(value) for value in experiment["counts"]
    } or structural_counts & primary_relation_counts:
        raise RuntimeError("Canonical structural/primary count registry is not a partition")
    confirmation_encodings: dict[tuple[int, int], PromptEncoding] = {}
    registration_coverage: list[dict[str, Any]] = []
    for seed in experiment["confirmation_seeds"]:
        for count in experiment["counts"]:
            encoding = render_v4_prompt(
                rows[(int(seed), int(count))],
                tokenizer=tokenizer,
                model_spec=spec,
                config=config,
                answer_format="numeric",
            )
            pairs = identity_repeated_pairs(encoding, anchor_token)
            expected_edges = 0 if int(count) in structural_counts else int(count) - 1
            if len(pairs) != expected_edges:
                raise RuntimeError(
                    "Frozen canonical anchor has incomplete confirmation coverage: "
                    f"seed={seed}, count={count}, observed={len(pairs)}, "
                    f"expected={expected_edges}"
                )
            confirmation_encodings[(int(seed), int(count))] = encoding
            registration_coverage.append(
                {
                    "seed": int(seed),
                    "gold_count": int(count),
                    "structural_no_previous_match": bool(
                        int(count) in structural_counts
                    ),
                    "registered_edges": len(pairs),
                }
            )
    write_json(
        root / "canonical_registration_coverage.json",
        {
            "schema_version": (
                "realistic_niah_v4_4_5_induction_registration_coverage_v1"
            ),
            "status": "PASS",
            "units": len(registration_coverage),
            "structural_no_previous_match_counts": sorted(structural_counts),
            "primary_relation_counts": sorted(primary_relation_counts),
            "rows": registration_coverage,
        },
    )
    expected = {
        (int(seed), int(count), arm)
        for seed in experiment["confirmation_seeds"]
        for count in experiment["counts"]
        for arm in arms
    }
    if not completed.issubset(expected):
        raise RuntimeError("Existing canonical rows are outside the frozen registry")

    # The first Gemma attention/capture forward can exercise lazy backend
    # initialization.  It must never become a recorded arm: in V2 that made
    # the very first structural natural row differ from the two later empty-
    # delta rows even though all three had zero hook sites.  Exercise the exact
    # source-capture + causal-prefill path once, discard every readout/state,
    # and retain the strict 1e-9 equality audit for the subsequently recorded
    # N=1 arms.  This warm-up is repeated after every process restart.
    warmup_config = experiment["canonical"].get("discarded_noop_warmup", {})
    if not bool(warmup_config.get("enabled")) or bool(
        warmup_config.get("recorded", True)
    ):
        raise RuntimeError("Canonical discarded no-op warm-up is not enabled")
    warmup_seed = int(warmup_config["seed"])
    warmup_count = int(warmup_config["count"])
    if warmup_seed not in {int(value) for value in experiment["confirmation_seeds"]}:
        raise RuntimeError("Canonical warm-up seed is outside confirmation")
    if warmup_count not in structural_counts:
        raise RuntimeError("Canonical warm-up count is not a structural no-match count")
    warmup_encoding = confirmation_encodings[(warmup_seed, warmup_count)]
    warmup_source_bundle = capture_query_bundle(
        model,
        adapter,
        warmup_encoding,
        layers=(source_layer,),
        capture_attention=False,
        capture_values=True,
        audit_cache_equivalence=False,
    )
    if source_layer not in warmup_source_bundle.value_by_layer:
        raise RuntimeError("Canonical no-op warm-up failed to capture source values")
    warmup_metrics, warmup_states, warmup_intervention = evaluate_arm(
        model,
        tokenizer,
        adapter,
        warmup_encoding,
        endpoint_layer=int(model_config["endpoint_geometry_layer"]),
        retrieval_layer=int(model_config["retrieval_layer"]),
        retrieval_heads=model_config["retrieval_heads"],
        deltas={},
        max_new_tokens=int(
            experiment["canonical"]["strict_generation_max_new_tokens"]
        ),
    )
    if int(warmup_intervention.get("sites", -1)) != 0:
        raise RuntimeError("Discarded canonical no-op warm-up applied an intervention")
    if warmup_states.shape[0] != len(warmup_encoding.needle_spans):
        raise RuntimeError("Discarded canonical no-op warm-up state capture is incomplete")
    if any(
        not np.isfinite(float(warmup_metrics[name]))
        for name in (
            "expected_count",
            "retrieval_bank_broad_score_mean",
            "correct_count_margin",
        )
    ):
        raise RuntimeError("Discarded canonical no-op warm-up produced nonfinite readouts")
    write_json(
        root / "canonical_noop_warmup.json",
        {
            "schema_version": "realistic_niah_v4_4_5_induction_noop_warmup_v1",
            "status": "PASS",
            "seed": warmup_seed,
            "gold_count": warmup_count,
            "recorded": False,
            "intervention_sites": 0,
            "source_capture_present": True,
            "endpoint_states": int(warmup_states.shape[0]),
            "reason": "discard_lazy_attention_capture_initialization_before_recorded_arms",
        },
    )
    del warmup_source_bundle, warmup_metrics, warmup_states, warmup_intervention
    print(
        f"[induction] {model_label} discarded canonical no-op warm-up "
        f"seed={warmup_seed} N={warmup_count}",
        flush=True,
    )

    for seed in experiment["confirmation_seeds"]:
        for count in experiment["counts"]:
            if all((int(seed), int(count), arm) in completed for arm in arms):
                continue
            encoding = confirmation_encodings[(int(seed), int(count))]
            source_bundle = capture_query_bundle(
                model,
                adapter,
                encoding,
                layers=(source_layer,),
                capture_attention=False,
                capture_values=True,
                audit_cache_equivalence=False,
            )
            value_states = source_bundle.value_by_layer[source_layer]
            candidate_deltas: dict[tuple[int, int, int], torch.Tensor] = {}
            control_deltas: dict[tuple[int, int, int], torch.Tensor] = {}
            edge_audits: list[dict[str, Any]] = []
            forbidden = registered_forbidden_positions(encoding)
            ordinary = set(range(int(encoding.query_position))) - forbidden
            for query, candidate_key in identity_repeated_pairs(encoding, anchor_token):
                attention, starts = selective_position_attention_outputs(
                    model,
                    adapter,
                    encoding,
                    query_position=query,
                    layers=(source_layer,),
                )
                row = attention[source_layer][source_head]
                key_start = int(starts[source_layer])
                if not key_start <= int(candidate_key) < key_start + len(row):
                    edge_audits.append(
                        {
                            "query_position": int(query),
                            "candidate_key": int(candidate_key),
                            "reachable": False,
                            "attention_key_start": key_start,
                        }
                    )
                    continue
                full_row = torch.zeros(int(query) + 1, dtype=torch.float32)
                full_row[key_start : key_start + len(row)] = row
                control_key, match = select_attention_mass_control(
                    full_row,
                    query=int(query),
                    target_key=int(candidate_key),
                    allowed=(key for key in ordinary if key >= key_start),
                    excluded=(),
                    bin_width=int(experiment["canonical"]["distance_bin_width"]),
                )
                if not bool(match["exact_distance_bin"]):
                    raise RuntimeError(
                        "No exact distance-bin control exists for a registered induction edge"
                    )
                candidate_delta, candidate_audit = natural_edge_delta(
                    row,
                    value_states,
                    keys=(int(candidate_key),),
                    key_start=key_start,
                    query_head=source_head,
                    query_heads=int(adapter.num_heads[source_layer]),
                    head_dim=int(adapter.head_dims[source_layer]),
                )
                control_delta, control_audit = natural_edge_delta(
                    row,
                    value_states,
                    keys=(int(control_key),),
                    key_start=key_start,
                    query_head=source_head,
                    query_heads=int(adapter.num_heads[source_layer]),
                    head_dim=int(adapter.head_dims[source_layer]),
                )
                candidate_deltas[(source_layer, source_head, int(query))] = candidate_delta
                control_deltas[(source_layer, source_head, int(query))] = control_delta
                edge_audits.append(
                    {
                        "query_position": int(query),
                        "candidate_key": int(candidate_key),
                        "control_key": int(control_key),
                        "reachable": True,
                        "attention_key_start": key_start,
                        **{f"match_{key}": value for key, value in match.items()},
                        **{f"candidate_{key}": value for key, value in candidate_audit.items()},
                        **{f"control_{key}": value for key, value in control_audit.items()},
                    }
                )
            deltas_by_arm = {
                "natural": {},
                "candidate_edge_block": candidate_deltas,
                "mass_distance_control": control_deltas,
            }
            for arm in arms:
                key = (int(seed), int(count), arm)
                if key in completed:
                    continue
                metrics, states, intervention_audit = evaluate_arm(
                    model,
                    tokenizer,
                    adapter,
                    encoding,
                    endpoint_layer=int(model_config["endpoint_geometry_layer"]),
                    retrieval_layer=int(model_config["retrieval_layer"]),
                    retrieval_heads=model_config["retrieval_heads"],
                    deltas=deltas_by_arm[arm],
                    max_new_tokens=int(experiment["canonical"]["strict_generation_max_new_tokens"]),
                )
                state_path = state_dir / f"seed{int(seed)}_count{int(count)}_{arm}.npz"
                state_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    state_path,
                    states=states.numpy().astype(np.float16),
                    running_index=np.arange(1, len(encoding.needle_spans) + 1),
                )
                reachable = [row for row in edge_audits if row.get("reachable")]
                append_jsonl(
                    detail_path,
                    {
                        "schema_version": SCHEMA,
                        "model_label": model_label,
                        "seed": int(seed),
                        "gold_count": int(count),
                        "arm": arm,
                        "source_layer": source_layer,
                        "source_head": source_head,
                        "anchor_token_id": int(anchor_token),
                        "structural_no_previous_match": bool(
                            int(count) in structural_counts
                        ),
                        "registered_edges": len(edge_audits),
                        "reachable_edges": len(reachable),
                        "blocked_edge_attention_mass_sum": float(
                            sum(
                                float(row.get("candidate_edge_attention_mass", 0.0))
                                for row in reachable
                            )
                            if arm == "candidate_edge_block"
                            else sum(
                                float(row.get("control_edge_attention_mass", 0.0))
                                for row in reachable
                            )
                            if arm == "mass_distance_control"
                            else 0.0
                        ),
                        "endpoint_state_path": str(state_path.relative_to(root).as_posix()),
                        "intervention_sites": int(intervention_audit["sites"]),
                        **metrics,
                    },
                )
                completed.add(key)
                print(
                    f"[induction] {model_label} seed={seed} N={count} arm={arm} "
                    f"edges={len(reachable)} E={metrics['expected_count']:.3f}",
                    flush=True,
                )
            audit_path = root / "edge_audits" / f"seed{int(seed)}_count{int(count)}.json"
            write_json(audit_path, {"seed": int(seed), "gold_count": int(count), "edges": edge_audits})
    missing = sorted(expected - completed)
    if missing:
        raise RuntimeError(f"Induction run ended with missing rows: {missing[:5]}")
    completion = {
        "schema_version": "realistic_niah_v4_4_5_induction_complete_v1",
        "status": "complete",
        "model": model_label,
        "synthetic_rows": len(synthetic_rows),
        "canonical_rows": len(expected),
        "unique_canonical_keys": len(completed),
        "retained_head": [source_layer, source_head],
        "anchor_token_id": int(anchor_token),
        "structural_no_previous_match_counts": sorted(structural_counts),
        "primary_relation_counts": sorted(primary_relation_counts),
    }
    write_json(root / "complete.json", completion)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
