from __future__ import annotations

"""Run follow-up 23: nuisance factorial and targeted outside-context edges."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from realistic_niah_v4.behavior import parse_numeric_completion
from realistic_niah_v4.modeling import capture_post_block_states, load_registered_model
from realistic_niah_v4.prompts import PromptEncoding, render_v4_prompt
from realistic_niah_v4.spec import V4Config, resolve_model_spec
from realistic_niah_v4.stimuli import load_stimuli
from realistic_niah_v4_4_3.interventions import candidate_sequence_metrics, capture_query_bundle
from realistic_niah_v4_4_5.followup_edges import (
    context_halo_positions,
    derive_factorial_encoding,
    natural_edge_delta,
    registered_forbidden_positions,
    select_attention_mass_control,
    select_control_feasible_candidates,
    select_deterministic_random_control,
)
from realistic_niah_v4_4_5.followup_runtime import (
    run_answer_query_edge_arm,
    selective_position_attention_outputs,
)
from realistic_niah_v4_4_5.restoration import (
    active_broad_metrics,
    generate_answer_completion_from_prefill,
)


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
        raise RuntimeError(f"Existing artifact differs: {path}")
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


def identity_replacements(
    base: PromptEncoding,
    pool: Sequence[PromptEncoding],
) -> tuple[tuple[int, ...], ...]:
    result = []
    for span in base.needle_spans:
        source = tuple(base.input_ids[int(span.start) : int(span.end)])
        options: list[tuple[int, int, tuple[int, ...]]] = []
        for donor in pool:
            if int(donor.seed) == int(base.seed):
                continue
            for donor_span in donor.needle_spans:
                tokens = tuple(donor.input_ids[int(donor_span.start) : int(donor_span.end)])
                if len(tokens) == len(source) and tokens != source:
                    options.append((int(donor.seed), int(donor_span.slot_index), tokens))
        if not options:
            raise RuntimeError(
                f"No exact-length identity donor for seed={base.seed} slot={span.slot_index} len={len(source)}"
            )
        result.append(min(options, key=lambda value: (value[0], value[1]))[2])
    return tuple(result)


def completed_factorial(path: Path) -> set[tuple[int, str]]:
    if not path.exists():
        return set()
    values = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = (int(row["seed"]), str(row["cell"]))
            if key in values:
                raise RuntimeError(f"Duplicate factorial row {key}")
            values.add(key)
    return values


def completed_outside(path: Path) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    values = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = (int(row["seed"]), int(row["gold_count"]), str(row["arm"]))
            if key in values:
                raise RuntimeError(f"Duplicate outside-context row {key}")
            values.add(key)
    return values


def visible_positions(positions: Sequence[int], *, key_start: int, key_length: int) -> list[int]:
    return [
        int(value)
        for value in positions
        if int(key_start) <= int(value) < int(key_start) + int(key_length)
    ]


@torch.inference_mode()
def freeze_source_head(
    model: Any,
    adapter: Any,
    encodings: Sequence[PromptEncoding],
    *,
    candidates: Sequence[Sequence[int]],
    halo_width: int,
) -> tuple[tuple[int, int], list[dict[str, Any]]]:
    layers = tuple(sorted({int(layer) for layer, _head in candidates}))
    accum: dict[tuple[int, int], list[float]] = {
        (int(layer), int(head)): [] for layer, head in candidates
    }
    rows: list[dict[str, Any]] = []
    for encoding in encodings:
        attention, starts = selective_position_attention_outputs(
            model,
            adapter,
            encoding,
            query_position=int(encoding.query_position),
            layers=layers,
        )
        halo = set(context_halo_positions(encoding, width=int(halo_width)))
        forbidden = registered_forbidden_positions(encoding)
        ordinary = set(range(int(encoding.query_position))) - forbidden
        for rank, (layer, head) in enumerate(candidates, start=1):
            layer, head = int(layer), int(head)
            row = attention[layer][head]
            start = int(starts[layer])
            visible_halo = visible_positions(halo, key_start=start, key_length=len(row))
            visible_other = visible_positions(
                ordinary - halo, key_start=start, key_length=len(row)
            )
            if not visible_halo or not visible_other:
                score = float("-inf")
                halo_density = other_density = 0.0
            else:
                halo_density = float(
                    row[torch.as_tensor([key - start for key in visible_halo])].mean()
                )
                other_density = float(
                    row[torch.as_tensor([key - start for key in visible_other])].mean()
                )
                score = halo_density - other_density
            if np.isfinite(score):
                accum[(layer, head)].append(score)
            rows.append(
                {
                    "seed": int(encoding.seed),
                    "canonical_rank": rank,
                    "layer": layer,
                    "head": head,
                    "halo_density": halo_density,
                    "nonhalo_ordinary_density": other_density,
                    "halo_specificity": score,
                }
            )
    summary = []
    for rank, (layer, head) in enumerate(candidates, start=1):
        values = accum[(int(layer), int(head))]
        summary.append(
            {
                "canonical_rank": rank,
                "layer": int(layer),
                "head": int(head),
                "mean_halo_specificity": float(np.mean(values)) if values else float("-inf"),
                "discovery_seed_count": len(values),
            }
        )
    valid = [row for row in summary if np.isfinite(row["mean_halo_specificity"])]
    if not valid:
        raise RuntimeError("No candidate source head had visible halo and ordinary keys")
    winner = max(valid, key=lambda row: (row["mean_halo_specificity"], -row["canonical_rank"]))
    return (int(winner["layer"]), int(winner["head"])), summary


def broad_metrics(
    rows: torch.Tensor,
    *,
    key_start: int,
    encoding: PromptEncoding,
    heads: Sequence[int],
) -> dict[str, float]:
    metrics = active_broad_metrics(rows, key_start=int(key_start), spans=encoding.needle_spans)
    selected = [row for row in metrics if int(row["head"]) in set(int(v) for v in heads)]
    if len(selected) != len(set(int(value) for value in heads)):
        raise RuntimeError("Frozen retrieval head registry was not captured")
    return {
        "retrieval_bank_needle_mass_mean": float(np.mean([row["needle_mass"] for row in selected])),
        "retrieval_bank_coverage_mean": float(np.mean([row["coverage"] for row in selected])),
        "retrieval_bank_broad_score_mean": float(np.mean([row["broad_score"] for row in selected])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("Qwen3-8B", "Gemma4-E4B"))
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--v4-config", default="configs/realistic_niah_v4_4_5_stimuli.json")
    parser.add_argument("--experiment-config", default="configs/realistic_niah_v4_4_5_noise_factorial.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    experiment_path = Path(args.experiment_config).resolve()
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if experiment.get("schema_version") != "realistic_niah_v4_4_5_noise_factorial_v2":
        raise ValueError("Unexpected noise-factorial schema")
    stimuli_path = Path(args.stimuli).resolve()
    if sha256(stimuli_path) != str(experiment["stimulus_sha256"]):
        raise RuntimeError("Frozen stimulus hash mismatch")
    model_label = str(args.model)
    model_config = experiment["models"][model_label]
    v4_config = V4Config.from_json(args.v4_config)
    spec = resolve_model_spec(model_label)
    root = Path(args.output_dir).resolve() / model_label
    root.mkdir(parents=True, exist_ok=True)
    stimuli = {
        (int(row["seed"]), int(row["gold_count"])): row
        for row in load_stimuli(stimuli_path)
        if str(row.get("design_variant")) == "v4.4"
    }
    all_seeds = tuple(experiment["discovery_seeds"]) + tuple(experiment["confirmation_seeds"])
    required = {
        (int(seed), int(count))
        for seed in all_seeds
        for count in experiment["outside_context"]["counts"]
    }
    if not required.issubset(stimuli):
        raise RuntimeError(f"Missing canonical stimuli: {sorted(required - set(stimuli))[:5]}")
    write_json(
        root / "run_provenance.json",
        {
            "schema_version": "realistic_niah_v4_4_5_noise_factorial_run_v1",
            "model": model_label,
            "model_id": spec.model_id,
            "model_revision": spec.revision,
            "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
            "experiment_config": str(experiment_path),
            "experiment_config_sha256": sha256(experiment_path),
            "stimuli": str(stimuli_path),
            "stimuli_sha256": sha256(stimuli_path),
            "scientific_boundary": "controlled token-level nuisance deformations and natural answer-query edge contributions; not a unique decomposition of observational prompt noise",
        },
    )
    model, tokenizer, adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=experiment["model_torch_dtype"],
        attention_backend=experiment["attention_prefix_backend"],
    )
    base_encodings = {
        int(seed): render_v4_prompt(
            stimuli[(int(seed), int(experiment["factorial"]["gold_count"]))],
            tokenizer=tokenizer,
            model_spec=spec,
            config=v4_config,
            answer_format="numeric",
        )
        for seed in all_seeds
    }

    factorial_path = root / "factorial_rows.jsonl"
    factorial_done = completed_factorial(factorial_path)
    factorial_expected = {
        (int(seed), str(cell)) for seed in all_seeds for cell in experiment["factorial"]["cells"]
    }
    factorial_state_dir = root / "factorial_states"
    for seed in all_seeds:
        base = base_encodings[int(seed)]
        replacements = identity_replacements(base, tuple(base_encodings.values()))
        for cell in experiment["factorial"]["cells"]:
            key = (int(seed), str(cell))
            if key in factorial_done:
                continue
            identity, context, position = (bool(int(value)) for value in str(cell))
            derived, manipulation_audit = derive_factorial_encoding(
                base,
                identity_replacements=replacements,
                identity=identity,
                context=context,
                position=position,
                context_width=int(experiment["factorial"]["context_width"]),
            )
            endpoints = tuple(int(span.end) - 1 for span in derived.needle_spans)
            _logits, captured = capture_post_block_states(
                model,
                adapter,
                derived,
                endpoints,
                layers=(int(model_config["geometry_layer"]),),
            )
            states = captured[int(model_config["geometry_layer"])]
            if states.shape[0] != 10 or not torch.isfinite(states).all():
                raise RuntimeError("Factorial endpoint capture failed")
            state_path = factorial_state_dir / f"seed{int(seed)}_cell{cell}.npz"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                state_path,
                states=states.numpy().astype(np.float16),
                running_index=np.arange(1, 11),
            )
            append_jsonl(
                factorial_path,
                {
                    "schema_version": "realistic_niah_v4_4_5_noise_factorial_row_v1",
                    "model_label": model_label,
                    "seed": int(seed),
                    "split": "discovery" if int(seed) in set(experiment["discovery_seeds"]) else "confirmation",
                    "cell": str(cell),
                    "identity": identity,
                    "context": context,
                    "position": position,
                    "geometry_layer": int(model_config["geometry_layer"]),
                    "state_path": str(state_path.relative_to(root).as_posix()),
                    "manipulation_audit": manipulation_audit,
                },
            )
            factorial_done.add(key)
            print(f"[noise-factorial] {model_label} seed={seed} cell={cell}", flush=True)
    if factorial_done != factorial_expected:
        raise RuntimeError("Factorial stage ended with missing or extra keys")

    discovery_encodings = [base_encodings[int(seed)] for seed in experiment["discovery_seeds"]]
    source, source_summary = freeze_source_head(
        model,
        adapter,
        discovery_encodings,
        candidates=model_config["source_candidate_heads"],
        halo_width=int(experiment["outside_context"]["halo_width"]),
    )
    source_layer, source_head = source
    registration = {
        "schema_version": "realistic_niah_v4_4_5_outside_context_registration_v2",
        "source_layer": source_layer,
        "source_head": source_head,
        "source_head_summary": source_summary,
        "candidate_key_rule": "natural-attention ranked ordinary halo keys, capped per exact distance bin by unique nonhalo-control capacity, at most top_keys_per_head",
    }
    write_json(root / "outside_context_registration.json", registration)

    outside_path = root / "outside_context_rows.jsonl"
    outside_done = completed_outside(outside_path)
    outside_arms = tuple(str(value) for value in experiment["outside_context"]["arms"])
    outside_expected = {
        (int(seed), int(count), arm)
        for seed in experiment["confirmation_seeds"]
        for count in experiment["outside_context"]["counts"]
        for arm in outside_arms
    }
    outside_state_dir = root / "outside_context_states"
    for seed in experiment["confirmation_seeds"]:
        for count in experiment["outside_context"]["counts"]:
            if all((int(seed), int(count), arm) in outside_done for arm in outside_arms):
                continue
            encoding = render_v4_prompt(
                stimuli[(int(seed), int(count))],
                tokenizer=tokenizer,
                model_spec=spec,
                config=v4_config,
                answer_format="numeric",
            )
            bundle = capture_query_bundle(
                model,
                adapter,
                encoding,
                layers=(source_layer,),
                capture_attention=True,
                capture_values=True,
                audit_cache_equivalence=False,
            )
            row = bundle.alpha_by_layer[source_layer][source_head]
            key_start = int(bundle.alpha_key_start_by_layer[source_layer])
            halo = visible_positions(
                context_halo_positions(
                    encoding, width=int(experiment["outside_context"]["halo_width"])
                ),
                key_start=key_start,
                key_length=len(row),
            )
            forbidden = registered_forbidden_positions(encoding)
            ordinary = set(range(int(encoding.query_position))) - forbidden
            nonhalo = ordinary - set(halo)
            ranked_halo_keys = tuple(
                sorted(
                    halo,
                    key=lambda key: (-float(row[int(key) - key_start]), int(key)),
                )
            )
            eligible_nonhalo = tuple(
                sorted(key for key in nonhalo if int(key) >= int(key_start))
            )
            candidate_keys, feasibility_audit = select_control_feasible_candidates(
                query=int(encoding.query_position),
                ranked_candidates=ranked_halo_keys,
                allowed_controls=eligible_nonhalo,
                bin_width=int(experiment["outside_context"]["distance_bin_width"]),
                max_candidates=int(experiment["outside_context"]["top_keys_per_head"]),
            )
            full_row = torch.zeros(int(encoding.query_position) + 1, dtype=torch.float32)
            full_row[key_start : key_start + len(row)] = row
            random_keys: list[int] = []
            mass_keys: list[int] = []
            random_audits, mass_audits = [], []
            for index, candidate_key in enumerate(candidate_keys):
                random_key, random_audit = select_deterministic_random_control(
                    query=int(encoding.query_position),
                    target_key=int(candidate_key),
                    allowed=eligible_nonhalo,
                    excluded=set(candidate_keys) | set(random_keys),
                    bin_width=int(experiment["outside_context"]["distance_bin_width"]),
                    label=f"{model_label}:{seed}:{count}:{index}",
                )
                mass_key, mass_audit = select_attention_mass_control(
                    full_row,
                    query=int(encoding.query_position),
                    target_key=int(candidate_key),
                    allowed=eligible_nonhalo,
                    excluded=set(candidate_keys) | set(mass_keys),
                    bin_width=int(experiment["outside_context"]["distance_bin_width"]),
                )
                if not bool(random_audit["exact_distance_bin"]):
                    raise RuntimeError(
                        "No exact distance-bin random control exists for a halo edge"
                    )
                if not bool(mass_audit["exact_distance_bin"]):
                    raise RuntimeError(
                        "No exact distance-bin attention-mass control exists for a halo edge"
                    )
                random_keys.append(random_key)
                mass_keys.append(mass_key)
                random_audits.append(random_audit)
                mass_audits.append(mass_audit)
            value_states = bundle.value_by_layer[source_layer]
            candidate_delta, candidate_audit = natural_edge_delta(
                row,
                value_states,
                keys=candidate_keys,
                key_start=key_start,
                query_head=source_head,
                query_heads=int(adapter.num_heads[source_layer]),
                head_dim=int(adapter.head_dims[source_layer]),
            )
            random_delta, random_delta_audit = natural_edge_delta(
                row,
                value_states,
                keys=random_keys,
                key_start=key_start,
                query_head=source_head,
                query_heads=int(adapter.num_heads[source_layer]),
                head_dim=int(adapter.head_dims[source_layer]),
            )
            mass_delta, mass_delta_audit = natural_edge_delta(
                row,
                value_states,
                keys=mass_keys,
                key_start=key_start,
                query_head=source_head,
                query_heads=int(adapter.num_heads[source_layer]),
                head_dim=int(adapter.head_dims[source_layer]),
            )
            arm_deltas = {
                "natural": torch.zeros_like(candidate_delta),
                "candidate_halo_edge_block": candidate_delta,
                "distance_random_control": random_delta,
                "attention_mass_control": mass_delta,
            }
            edge_audit_path = root / "outside_context_edge_audits" / f"seed{int(seed)}_count{int(count)}.json"
            write_json(
                edge_audit_path,
                {
                    "schema_version": "realistic_niah_v4_4_5_outside_context_edge_audit_v2",
                    "seed": int(seed),
                    "gold_count": int(count),
                    "query_position": int(encoding.query_position),
                    "distance_bin_width": int(experiment["outside_context"]["distance_bin_width"]),
                    "candidate_keys": list(candidate_keys),
                    "ranked_halo_keys": list(ranked_halo_keys),
                    "candidate_feasibility": feasibility_audit,
                    "random_keys": random_keys,
                    "mass_keys": mass_keys,
                    "candidate": candidate_audit,
                    "random": random_delta_audit,
                    "mass": mass_delta_audit,
                    "random_match_audits": random_audits,
                    "mass_match_audits": mass_audits,
                },
            )
            for arm in outside_arms:
                key = (int(seed), int(count), arm)
                if key in outside_done:
                    continue
                scored, prefill, readout_rows, readout_starts, states, hook_audit = run_answer_query_edge_arm(
                    model,
                    adapter,
                    encoding,
                    source_layer=source_layer,
                    source_deltas={source_head: arm_deltas[arm]},
                    readout_layers=(int(model_config["retrieval_layer"]),),
                    state_layers=(int(model_config["late_state_layer"]),),
                )
                strict = generate_answer_completion_from_prefill(
                    model,
                    tokenizer,
                    encoding,
                    prefill,
                    max_new_tokens=int(experiment["outside_context"]["strict_generation_max_new_tokens"]),
                )
                retrieval_layer = int(model_config["retrieval_layer"])
                metrics = {
                    **candidate_sequence_metrics(scored.candidate_log_scores, encoding),
                    **strict_fields(strict, int(count)),
                    **broad_metrics(
                        readout_rows[retrieval_layer],
                        key_start=readout_starts[retrieval_layer],
                        encoding=encoding,
                        heads=model_config["retrieval_heads"],
                    ),
                }
                state_path = outside_state_dir / f"seed{int(seed)}_count{int(count)}_{arm}.npz"
                state_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    state_path,
                    answer_state=states[int(model_config["late_state_layer"])].numpy().astype(np.float16),
                )
                arm_audit = (
                    {"edge_count": 0.0, "edge_attention_mass": 0.0, "edge_contribution_norm": 0.0}
                    if arm == "natural"
                    else candidate_audit
                    if arm == "candidate_halo_edge_block"
                    else random_delta_audit
                    if arm == "distance_random_control"
                    else mass_delta_audit
                )
                append_jsonl(
                    outside_path,
                    {
                        "schema_version": "realistic_niah_v4_4_5_outside_context_row_v1",
                        "model_label": model_label,
                        "seed": int(seed),
                        "gold_count": int(count),
                        "arm": arm,
                        "source_layer": source_layer,
                        "source_head": source_head,
                        "retrieval_layer": retrieval_layer,
                        "late_state_layer": int(model_config["late_state_layer"]),
                        "edge_count": int(arm_audit["edge_count"]),
                        "edge_attention_mass": float(arm_audit["edge_attention_mass"]),
                        "edge_contribution_norm": float(arm_audit["edge_contribution_norm"]),
                        "answer_state_path": str(state_path.relative_to(root).as_posix()),
                        **hook_audit,
                        **metrics,
                    },
                )
                outside_done.add(key)
                print(
                    f"[outside-context] {model_label} seed={seed} N={count} arm={arm} E={metrics['expected_count']:.3f}",
                    flush=True,
                )
    if outside_done != outside_expected:
        raise RuntimeError("Outside-context stage ended with missing or extra keys")
    completion = {
        "schema_version": "realistic_niah_v4_4_5_noise_factorial_complete_v1",
        "status": "complete",
        "model": model_label,
        "factorial_rows": len(factorial_expected),
        "outside_context_rows": len(outside_expected),
        "source_head": [source_layer, source_head],
    }
    write_json(root / "complete.json", completion)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
