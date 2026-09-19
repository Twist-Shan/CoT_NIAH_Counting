#!/usr/bin/env python3
"""Compare residual and distributed count carriers under one native transition.

This is a Qwen discovery/confirmation runner.  Every condition uses the same
marker-scrubbed native trace, receiver boundary, frozen L24 count decoder, and
native next-item candidate bank.  The intervention carriers are:

* a complete natural donor residual clamp;
* the donor delta projected into an OOF count subspace;
* the projected residual clamp plus an OOF all-history K/V count field; and
* matched residual-only and residual+K/V orthogonal controls.

The runner records the realized current-count first stage, retention after the
unchanged next native item, and whether next-item candidate scores remain tied
to the receiver rather than switching to the donor's successor.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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

from realistic_niah_v5.boundary_attention_bottleneck import (  # noqa: E402
    select_post_item_boundary_position,
)
from realistic_niah_v5.boundary_counter_probe import (  # noqa: E402
    count_probe_subspace,
    fit_dual_ridge_count_probe,
    norm_matched_orthogonal_replacement,
)
from realistic_niah_v5.bullet_counter_site_diagnostics import (  # noqa: E402
    build_diagnostic_bases,
)
from realistic_niah_v5.indexed_counter_patch import (  # noqa: E402
    capture_decoder_block_input_states,
)
from realistic_niah_v5.kv_counter_transition import (  # noqa: E402
    item_bin_positions,
)
from realistic_niah_v5.unified_carrier_transition import (  # noqa: E402
    carrier_capture_positions,
    carrier_no_cache_tail_logits,
    interpolated_boundary_targets,
    no_cache_tail_logits,
    summarize_carrier_trials,
)
from scripts.run_realistic_niah_v5_boundary_equivariance import (  # noqa: E402
    decode_count_probe,
)
from scripts.run_realistic_niah_v5_count_stream import (  # noqa: E402
    _atomic_json,
    _atomic_jsonl,
    _model,
)
from scripts.run_realistic_niah_v5_cross_seed_counter_recurrence import (  # noqa: E402
    prefix_through_boundary,
)
from scripts.run_realistic_niah_v5_kv_counter_transition import (  # noqa: E402
    BankSpec,
    build_kv_directions,
    build_or_load_raw_kv_panel,
    fit_kv_field,
    orthogonal_kv_tangents,
)


CARRIERS = (
    "whole_state",
    "residual_count_subspace",
    "residual_count_subspace_orthogonal",
    "residual_count_plus_kv",
    "residual_count_plus_kv_orthogonal",
)


def _read_rows(path: Path, seeds: Sequence[int]) -> dict[int, dict[str, Any]]:
    wanted = {int(value) for value in seeds}
    selected: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        seed = int(row["seed"])
        if seed in wanted:
            if seed in selected:
                raise ValueError(f"Duplicate requested seed {seed}")
            selected[seed] = row
    if set(selected) != wanted:
        raise ValueError(f"Missing requested seeds: {sorted(wanted-set(selected))}")
    return selected


def _transition_candidates(
    encoding: Any,
    registry: Any,
    *,
    receiver_occurrence: int,
) -> dict[int, tuple[int, ...]]:
    items = tuple((int(start), int(end)) for start, end in registry.trace_items)
    receiver = int(receiver_occurrence)
    if len(items) != 10 or not 1 <= receiver < 10:
        raise ValueError("Expected ten native items and a nonterminal receiver")
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
        raise ValueError("A native successor candidate is empty")
    if len(set(candidates.values())) != 10:
        raise ValueError("Native successor candidates are not unique")
    return candidates


def build_or_load_boundary_panel(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    *,
    path: Path,
    rows: Mapping[int, Mapping[str, Any]],
    seeds: Sequence[int],
    layers: Sequence[int],
) -> np.ndarray:
    """Cache discovery residual states as [seed,layer,count,width]."""

    active_seeds = tuple(int(value) for value in seeds)
    active_layers = tuple(int(value) for value in layers)
    if path.exists():
        payload = np.load(path)
        if np.asarray(payload["seeds"]).tolist() != list(active_seeds):
            raise ValueError("Frozen boundary-panel seeds changed")
        if np.asarray(payload["layers"]).tolist() != list(active_layers):
            raise ValueError("Frozen boundary-panel layers changed")
        values = np.asarray(payload["states"], dtype=np.float32)
        if values.shape[:3] != (len(active_seeds), len(active_layers), 10):
            raise ValueError("Frozen boundary-panel shape changed")
        return values

    values: np.ndarray | None = None
    for seed_index, seed in enumerate(active_seeds):
        source, _blank, registry, _audit = build_diagnostic_bases(
            rows[seed],
            tokenizer,
            random_seed=20260830 + seed,
            construction="targeted_explicit_count_scrub",
        )
        positions = tuple(
            select_post_item_boundary_position(
                source, registry, tokenizer, occurrence=occurrence
            )[0]
            for occurrence in range(1, 11)
        )
        captured = capture_decoder_block_input_states(
            model, adapter, source, positions, layers=active_layers
        )
        if values is None:
            width = int(captured[active_layers[0]].shape[-1])
            values = np.empty(
                (len(active_seeds), len(active_layers), 10, width),
                dtype=np.float32,
            )
        for layer_index, layer in enumerate(active_layers):
            panel = captured[layer].numpy().astype(np.float32)
            if panel.shape != values[seed_index, layer_index].shape:
                raise ValueError("Boundary residual width changed across layers")
            values[seed_index, layer_index] = panel
        print(f"[unified-carrier] captured boundary panel seed={seed}", flush=True)
    if values is None:
        raise RuntimeError("No discovery boundary states were captured")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        states=values.astype(np.float16),
        seeds=np.asarray(active_seeds, dtype=np.int64),
        layers=np.asarray(active_layers, dtype=np.int64),
    )
    temporary.replace(path)
    return values


def fit_residual_bases(
    panel: np.ndarray,
    *,
    layers: Sequence[int],
    train_indices: Sequence[int],
    alpha: float,
) -> dict[int, np.ndarray]:
    values = np.asarray(panel, dtype=np.float32)
    active_layers = tuple(int(value) for value in layers)
    indices = np.asarray(tuple(int(value) for value in train_indices), dtype=np.int64)
    if values.ndim != 4 or indices.size < 2:
        raise ValueError("Residual basis fitting needs a four-dimensional panel")
    if int(values.shape[1]) != len(active_layers) or int(values.shape[2]) != 10:
        raise ValueError("Residual boundary panel geometry changed")
    labels = np.tile(np.arange(1, 11, dtype=np.int64), indices.size)
    result: dict[int, np.ndarray] = {}
    for layer_index, layer in enumerate(active_layers):
        states = values[indices, layer_index]
        probe = fit_dual_ridge_count_probe(
            states.reshape(-1, states.shape[-1]), labels, alpha=float(alpha)
        )
        result[layer] = count_probe_subspace(probe)
    return result


def _fit_fold_geometry(
    *,
    fit_mode: str,
    discovery_seeds: Sequence[int],
    evaluation_seeds: Sequence[int],
    folds: int,
    boundary_panel: np.ndarray,
    kv_panel: np.ndarray,
    layers: Sequence[int],
    alpha: float,
) -> tuple[
    dict[int, int],
    dict[int, dict[int, np.ndarray]],
    dict[int, tuple[dict[Any, np.ndarray], dict[Any, np.ndarray]]],
]:
    active_discovery = tuple(int(value) for value in discovery_seeds)
    seed_to_index = {seed: index for index, seed in enumerate(active_discovery)}
    if fit_mode == "oof":
        fold_by_seed = {
            seed: index % int(folds)
            for index, seed in enumerate(sorted(active_discovery))
        }
        if any(seed not in fold_by_seed for seed in evaluation_seeds):
            raise ValueError("OOF evaluation seeds must belong to discovery")
        active_folds = tuple(range(int(folds)))
    elif fit_mode == "full_discovery":
        if set(evaluation_seeds) & set(active_discovery):
            raise ValueError("Full-discovery evaluation seeds must be held out")
        fold_by_seed = {int(seed): 0 for seed in evaluation_seeds}
        active_folds = (0,)
    else:
        raise ValueError("fit_mode must be oof or full_discovery")

    residual_by_fold: dict[int, dict[int, np.ndarray]] = {}
    kv_by_fold: dict[
        int, tuple[dict[Any, np.ndarray], dict[Any, np.ndarray]]
    ] = {}
    for fold in active_folds:
        if fit_mode == "oof":
            train_indices = [
                seed_to_index[seed]
                for seed in active_discovery
                if fold_by_seed[seed] != fold
            ]
        else:
            train_indices = list(range(len(active_discovery)))
        residual_by_fold[fold] = fit_residual_bases(
            boundary_panel,
            layers=layers,
            train_indices=train_indices,
            alpha=float(alpha),
        )
        kv_by_fold[fold] = fit_kv_field(
            kv_panel,
            layers=layers,
            train_indices=train_indices,
            alpha=float(alpha),
        )
        print(
            f"[unified-carrier] fit fold={fold} train_n={len(train_indices)}",
            flush=True,
        )
    return fold_by_seed, residual_by_fold, kv_by_fold


def _candidate_score_payload(
    candidates: Mapping[int, Sequence[int]],
    *,
    target: int,
    sum_scores: Sequence[float],
) -> dict[str, Any]:
    """Match the registered native-candidate score summary from raw sums."""

    ordered = [(key, tuple(int(value) for value in candidates[key])) for key in range(1, 11)]
    values = np.asarray(tuple(float(value) for value in sum_scores), dtype=np.float64)
    if values.shape != (10,) or any(not tokens for _key, tokens in ordered):
        raise ValueError("Native candidate scoring requires ten nonempty score rows")
    means = np.asarray(
        [values[index] / len(tokens) for index, (_key, tokens) in enumerate(ordered)],
        dtype=np.float64,
    )
    target_index = int(target) - 1
    if not 0 <= target_index < 10:
        raise ValueError("Native candidate target must be in 1..10")
    other_mean = np.delete(means, target_index)
    other_sum = np.delete(values, target_index)
    shifted = means - float(np.max(means))
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return {
        "target_occurrence": int(target),
        "predicted_occurrence_mean_logprob": int(np.argmax(means)) + 1,
        "predicted_occurrence_sum_logprob": int(np.argmax(values)) + 1,
        "target_exact_mean_logprob": bool(int(np.argmax(means)) == target_index),
        "target_exact_sum_logprob": bool(int(np.argmax(values)) == target_index),
        "target_mean_logprob_margin": float(means[target_index] - np.max(other_mean)),
        "target_sum_logprob_margin": float(values[target_index] - np.max(other_sum)),
        "target_probability_mean_logprob": float(probabilities[target_index]),
        "mean_logprob_scores": means.tolist(),
        "sum_logprob_scores": values.tolist(),
        "candidate_token_counts": [len(tokens) for _key, tokens in ordered],
        "baseline_corrected": False,
    }


def _score_native_item_candidates_no_cache(
    model: Any,
    adapter: Any,
    prefix: Any,
    candidates: Mapping[int, Sequence[int]],
    *,
    receiver_successor: int,
    boundary_targets: Mapping[int, np.ndarray | torch.Tensor] | None = None,
    kv_directions: Mapping[
        tuple[int, str], Mapping[int, np.ndarray | torch.Tensor]
    ] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score native successors without retaining a long-prefix K/V cache."""

    if (boundary_targets is None) != (kv_directions is None):
        raise ValueError("Residual and K/V intervention arguments must be paired")
    scores: list[float] = []
    forward_audits: list[dict[str, Any]] = []
    sequence_lengths: list[int] = []
    for occurrence in range(1, 11):
        tokens = tuple(int(value) for value in candidates[occurrence])
        if not tokens:
            raise ValueError("A native successor candidate is empty")
        # For L target tokens, prefix + candidate[:-1] produces exactly the L
        # causal logit rows that predict candidate[0], ..., candidate[L-1].
        extended = replace(
            prefix,
            input_ids=tuple(prefix.input_ids) + tokens[:-1],
            attention_mask=tuple(prefix.attention_mask) + (1,) * (len(tokens) - 1),
        )
        if boundary_targets is None:
            logits = no_cache_tail_logits(
                model,
                extended,
                logits_to_keep=len(tokens),
            )
            audit: dict[str, Any] = {}
        else:
            logits, audit = carrier_no_cache_tail_logits(
                model,
                adapter,
                extended,
                boundary_position=int(prefix.sequence_length) - 1,
                boundary_targets=boundary_targets,
                kv_directions=kv_directions or {},
                logits_to_keep=len(tokens),
            )
        if tuple(logits.shape[:1]) != (len(tokens),):
            raise RuntimeError("Candidate logit window and token count disagree")
        targets = torch.tensor(tokens, dtype=torch.long).unsqueeze(1)
        selected = torch.log_softmax(logits, dim=-1).gather(1, targets)
        scores.append(float(selected.sum().item()))
        forward_audits.append({"occurrence": occurrence, **audit})
        sequence_lengths.append(int(extended.sequence_length))
    return (
        _candidate_score_payload(
            candidates,
            target=int(receiver_successor),
            sum_scores=scores,
        ),
        {
            "mode": "sequential_full_forward_no_cache",
            "candidate_forward_count": 10,
            "sequence_lengths": sequence_lengths,
            "forwards": forward_audits,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B",), required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--discovery-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--fit-mode", choices=("oof", "full_discovery"), default="oof")
    parser.add_argument("--raw-kv-panel", type=Path, required=True)
    parser.add_argument("--boundary-panel", type=Path, required=True)
    parser.add_argument("--frozen-probes", type=Path, required=True)
    parser.add_argument("--receivers", type=int, nargs="+", default=(5,))
    parser.add_argument("--doses", type=int, nargs="+", default=(-1, 1))
    parser.add_argument("--clamp-start-layer", type=int, default=14)
    parser.add_argument("--read-layer", type=int, default=24)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--oof-folds", type=int, default=4)
    parser.add_argument("--whole-scale", type=float, default=1.0)
    parser.add_argument("--subspace-scale", type=float, default=1.0)
    parser.add_argument("--kv-scale", type=float, default=1.0)
    parser.add_argument("--carriers", nargs="+", choices=CARRIERS, default=CARRIERS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.command = "unified-carrier-transition"

    discovery_seeds = tuple(int(value) for value in args.discovery_seeds)
    evaluation_seeds = tuple(int(value) for value in args.evaluation_seeds)
    receivers = tuple(int(value) for value in args.receivers)
    doses = tuple(sorted({int(value) for value in args.doses}))
    carriers = tuple(dict.fromkeys(str(value) for value in args.carriers))
    read_layer = int(args.read_layer)
    layers = tuple(range(int(args.clamp_start_layer), read_layer))
    folds = int(args.oof_folds)
    if len(discovery_seeds) < folds or folds < 2:
        raise ValueError("Discovery fitting requires at least two folds")
    if not evaluation_seeds or not receivers or not doses:
        raise ValueError("Evaluation seeds, receivers, and doses must be nonempty")
    if any(receiver < 2 or receiver > 8 for receiver in receivers):
        raise ValueError("Receivers must permit adjacent donor successors")
    if any(dose not in {-1, 1} for dose in doses):
        raise ValueError("The first unified comparison freezes adjacent donors only")
    if any(not np.isfinite(float(value)) or float(value) <= 0 for value in (
        args.whole_scale,
        args.subspace_scale,
        args.kv_scale,
    )):
        raise ValueError("Every carrier scale must be finite and positive")

    all_seeds = tuple(dict.fromkeys(discovery_seeds + evaluation_seeds))
    rows = _read_rows(args.generations, all_seeds)
    discovery_rows = {seed: rows[seed] for seed in discovery_seeds}
    evaluation_rows = {seed: rows[seed] for seed in evaluation_seeds}
    probe_npz = np.load(args.frozen_probes)
    alpha = float(np.asarray(probe_npz["alpha"])[0])
    frozen_layers = set(int(value) for value in np.asarray(probe_npz["frozen_layers"]))
    if read_layer not in frozen_layers:
        raise ValueError("Read layer has no frozen confirmation probe")
    read_probe = {
        "mean": np.asarray(probe_npz[f"layer_{read_layer}_mean"], dtype=np.float32),
        "weights": np.asarray(
            probe_npz[f"layer_{read_layer}_weights"], dtype=np.float32
        ),
        "alpha": alpha,
    }

    model, tokenizer, adapter = _model(args)
    boundary_panel = build_or_load_boundary_panel(
        model,
        tokenizer,
        adapter,
        path=args.boundary_panel,
        rows=discovery_rows,
        seeds=discovery_seeds,
        layers=layers,
    )
    kv_panel = build_or_load_raw_kv_panel(
        model,
        tokenizer,
        adapter,
        path=args.raw_kv_panel,
        rows=discovery_rows,
        seeds=discovery_seeds,
        layers=layers,
        bins=int(args.bins),
    )
    fold_by_seed, residual_by_fold, kv_by_fold = _fit_fold_geometry(
        fit_mode=str(args.fit_mode),
        discovery_seeds=discovery_seeds,
        evaluation_seeds=evaluation_seeds,
        folds=folds,
        boundary_panel=boundary_panel,
        kv_panel=kv_panel,
        layers=layers,
        alpha=alpha,
    )

    results: list[dict[str, Any]] = []
    kv_spec = BankSpec("all_history_kv", "all_history", "kv", layers)
    for seed in evaluation_seeds:
        row = evaluation_rows[seed]
        fold = fold_by_seed[seed]
        residual_bases = residual_by_fold[fold]
        kv_bases, kv_tangents = kv_by_fold[fold]
        orthogonal_kv = orthogonal_kv_tangents(
            kv_bases,
            kv_tangents,
            seed=20261225 + seed * 100,
        )
        source, _blank, registry, scrub_audit = build_diagnostic_bases(
            row,
            tokenizer,
            random_seed=20260830 + seed,
            construction="targeted_explicit_count_scrub",
        )
        boundary_positions = {
            occurrence: select_post_item_boundary_position(
                source, registry, tokenizer, occurrence=occurrence
            )[0]
            for occurrence in range(1, 11)
        }
        captured = capture_decoder_block_input_states(
            model,
            adapter,
            source,
            tuple(boundary_positions.values()),
            layers=layers + (read_layer,),
        )
        bins_by_occurrence = item_bin_positions(
            registry.trace_items, bins=int(args.bins)
        )

        for receiver in receivers:
            receiver_position = boundary_positions[receiver]
            next_position = boundary_positions[receiver + 1]
            read_indices = (receiver - 1, receiver)
            clean_read_states = captured[read_layer][list(read_indices)]
            clean_current = decode_count_probe(
                read_probe, clean_read_states[0].numpy()
            )
            clean_next = decode_count_probe(read_probe, clean_read_states[1].numpy())
            prefix = prefix_through_boundary(source, receiver_position)
            candidates = _transition_candidates(
                source, registry, receiver_occurrence=receiver
            )
            clean_scores, clean_candidate_audit = _score_native_item_candidates_no_cache(
                model,
                adapter,
                prefix,
                candidates,
                receiver_successor=receiver + 1,
            )

            for dose in doses:
                donor = receiver + dose
                receiver_states = {
                    layer: captured[layer][receiver - 1].numpy()
                    for layer in layers
                }
                donor_states = {
                    layer: captured[layer][donor - 1].numpy()
                    for layer in layers
                }
                whole_targets, whole_deltas = interpolated_boundary_targets(
                    receiver_states,
                    donor_states,
                    scale=float(args.whole_scale),
                )
                subspace_targets, subspace_deltas = interpolated_boundary_targets(
                    receiver_states,
                    donor_states,
                    scale=float(args.subspace_scale),
                    bases=residual_bases,
                )
                orthogonal_targets: dict[int, np.ndarray] = {}
                orthogonal_deltas: dict[int, np.ndarray] = {}
                for layer in layers:
                    target, delta = norm_matched_orthogonal_replacement(
                        receiver_states[layer],
                        subspace_deltas[layer],
                        residual_bases[layer],
                        seed=20261301 + seed * 10000 + (dose + 2) * 100 + layer,
                    )
                    orthogonal_targets[layer] = target
                    orthogonal_deltas[layer] = delta
                aligned_kv = build_kv_directions(
                    kv_spec,
                    receiver=receiver,
                    dose=dose,
                    scale=float(args.kv_scale),
                    bins_by_occurrence=bins_by_occurrence,
                    tangents=kv_tangents,
                )
                control_kv = build_kv_directions(
                    kv_spec,
                    receiver=receiver,
                    dose=dose,
                    scale=float(args.kv_scale),
                    bins_by_occurrence=bins_by_occurrence,
                    tangents=orthogonal_kv,
                )
                conditions = {
                    "whole_state": (whole_targets, {}, whole_deltas),
                    "residual_count_subspace": (
                        subspace_targets,
                        {},
                        subspace_deltas,
                    ),
                    "residual_count_subspace_orthogonal": (
                        orthogonal_targets,
                        {},
                        orthogonal_deltas,
                    ),
                    "residual_count_plus_kv": (
                        subspace_targets,
                        aligned_kv,
                        subspace_deltas,
                    ),
                    "residual_count_plus_kv_orthogonal": (
                        orthogonal_targets,
                        control_kv,
                        orthogonal_deltas,
                    ),
                }

                for carrier in carriers:
                    boundary_targets, kv_directions, planned_deltas = conditions[carrier]
                    states, full_audit = carrier_capture_positions(
                        model,
                        adapter,
                        source,
                        boundary_position=receiver_position,
                        boundary_targets=boundary_targets,
                        kv_directions=kv_directions,
                        read_positions=(receiver_position, next_position),
                        read_layer=read_layer,
                    )
                    current = decode_count_probe(read_probe, states[0].numpy())
                    later = decode_count_probe(read_probe, states[1].numpy())
                    scored, candidate_audit = _score_native_item_candidates_no_cache(
                        model,
                        adapter,
                        prefix,
                        candidates,
                        receiver_successor=receiver + 1,
                        boundary_targets=boundary_targets,
                        kv_directions=kv_directions,
                    )
                    current_shift = float(
                        current["probe_softmax_expected_count"]
                        - clean_current["probe_softmax_expected_count"]
                    )
                    next_shift = float(
                        later["probe_softmax_expected_count"]
                        - clean_next["probe_softmax_expected_count"]
                    )
                    sum_scores = np.asarray(scored["sum_logprob_scores"], dtype=float)
                    clean_sum = np.asarray(
                        clean_scores["sum_logprob_scores"], dtype=float
                    )
                    mean_scores = np.asarray(
                        scored["mean_logprob_scores"], dtype=float
                    )
                    clean_mean = np.asarray(
                        clean_scores["mean_logprob_scores"], dtype=float
                    )
                    receiver_index = receiver
                    donor_index = donor
                    results.append(
                        {
                            "schema_version": "unified_carrier_transition_v2",
                            "model_label": str(args.model),
                            "seed": seed,
                            "request_id": str(row["request_id"]),
                            "oof_fold": fold,
                            "fit_mode": str(args.fit_mode),
                            "receiver": receiver,
                            "donor": donor,
                            "dose": dose,
                            "carrier": carrier,
                            "whole_scale": float(args.whole_scale),
                            "subspace_scale": float(args.subspace_scale),
                            "kv_scale": float(args.kv_scale),
                            "read_layer": read_layer,
                            "clamp_layers": list(layers),
                            "scrub_construction": scrub_audit["construction"],
                            "expected_current": donor,
                            "expected_next": donor + 1,
                            "clean_current_prediction": int(
                                clean_current["probe_prediction"]
                            ),
                            "clean_next_prediction": int(clean_next["probe_prediction"]),
                            "current_prediction": int(current["probe_prediction"]),
                            "next_prediction": int(later["probe_prediction"]),
                            "current_soft": float(
                                current["probe_softmax_expected_count"]
                            ),
                            "next_soft": float(later["probe_softmax_expected_count"]),
                            "clean_current_soft": float(
                                clean_current["probe_softmax_expected_count"]
                            ),
                            "clean_next_soft": float(
                                clean_next["probe_softmax_expected_count"]
                            ),
                            "current_shift": current_shift,
                            "next_shift": next_shift,
                            "current_exact": bool(
                                int(current["probe_prediction"]) == donor
                            ),
                            "next_exact": bool(
                                int(later["probe_prediction"]) == donor + 1
                            ),
                            "probe_scores_current": current["probe_scores"],
                            "probe_scores_next": later["probe_scores"],
                            "receiver_successor_occurrence": receiver + 1,
                            "donor_successor_occurrence": donor + 1,
                            "predicted_successor_sum_logprob": int(
                                scored["predicted_occurrence_sum_logprob"]
                            ),
                            "clean_predicted_successor_sum_logprob": int(
                                clean_scores["predicted_occurrence_sum_logprob"]
                            ),
                            "predicted_successor_mean_logprob": int(
                                scored["predicted_occurrence_mean_logprob"]
                            ),
                            "clean_predicted_successor_mean_logprob": int(
                                clean_scores["predicted_occurrence_mean_logprob"]
                            ),
                            "receiver_successor_argmax_mean_logprob": bool(
                                int(scored["predicted_occurrence_mean_logprob"])
                                == receiver + 1
                            ),
                            "donor_successor_argmax_mean_logprob": bool(
                                int(scored["predicted_occurrence_mean_logprob"])
                                == donor + 1
                            ),
                            "receiver_successor_argmax_sum_logprob": bool(
                                int(scored["predicted_occurrence_sum_logprob"])
                                == receiver + 1
                            ),
                            "donor_successor_argmax_sum_logprob": bool(
                                int(scored["predicted_occurrence_sum_logprob"])
                                == donor + 1
                            ),
                            "receiver_successor_mean_logprob_change": float(
                                mean_scores[receiver_index]
                                - clean_mean[receiver_index]
                            ),
                            "donor_vs_receiver_mean_logodds_change": float(
                                (
                                    mean_scores[donor_index]
                                    - mean_scores[receiver_index]
                                )
                                - (
                                    clean_mean[donor_index]
                                    - clean_mean[receiver_index]
                                )
                            ),
                            "receiver_successor_sum_logprob_change": float(
                                sum_scores[receiver_index] - clean_sum[receiver_index]
                            ),
                            "donor_vs_receiver_sum_logodds_change": float(
                                (sum_scores[donor_index] - sum_scores[receiver_index])
                                - (clean_sum[donor_index] - clean_sum[receiver_index])
                            ),
                            "mean_logprob_scores": mean_scores.tolist(),
                            "clean_mean_logprob_scores": clean_mean.tolist(),
                            "sum_logprob_scores": sum_scores.tolist(),
                            "clean_sum_logprob_scores": clean_sum.tolist(),
                            "candidate_token_counts": scored[
                                "candidate_token_counts"
                            ],
                            "planned_boundary_l2_norms": {
                                str(layer): float(np.linalg.norm(planned_deltas[layer]))
                                for layer in layers
                            },
                            "full_trace_audit": full_audit,
                            "candidate_scoring_audit": candidate_audit,
                            "clean_candidate_scoring_audit": clean_candidate_audit,
                            "candidate_scoring_mode": "sequential_full_forward_no_cache",
                            "tokens_changed_by_intervention": False,
                            "diagnostic_suffix_used": False,
                        }
                    )
                print(
                    f"[unified-carrier] seed={seed} receiver={receiver} "
                    f"donor={donor} complete",
                    flush=True,
                )

    clean_units: dict[tuple[int, int], tuple[int, int, int]] = {}
    for row in results:
        unit = (int(row["seed"]), int(row["receiver"]))
        clean_readout = (
            int(row["receiver_successor_occurrence"]),
            int(row["clean_predicted_successor_mean_logprob"]),
            int(row["clean_predicted_successor_sum_logprob"]),
        )
        if unit in clean_units and clean_units[unit] != clean_readout:
            raise RuntimeError("Clean candidate readout changed within one trial unit")
        clean_units[unit] = clean_readout
    summary = {
        "schema_version": "unified_carrier_transition_v2",
        "model_label": str(args.model),
        "fit_mode": str(args.fit_mode),
        "discovery_seeds": list(discovery_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "receivers": list(receivers),
        "doses": list(doses),
        "carriers": list(carriers),
        "whole_scale": float(args.whole_scale),
        "subspace_scale": float(args.subspace_scale),
        "kv_scale": float(args.kv_scale),
        "read_layer": read_layer,
        "clamp_layers": list(layers),
        "first_stage_selection_only_allowed_for_scale_calibration": True,
        "next_state_and_candidate_outcomes_blinded_during_scale_calibration": True,
        "outcomes": summarize_carrier_trials(results),
        "clean_candidate_baseline": {
            "unit_count": len(clean_units),
            "natural_successor_mean_logprob_argmax": int(
                sum(
                    natural == predicted_mean
                    for natural, predicted_mean, _ in clean_units.values()
                )
            ),
            "natural_successor_sum_logprob_argmax": int(
                sum(
                    natural == predicted_sum
                    for natural, _, predicted_sum in clean_units.values()
                )
            ),
        },
        "input_tokens_changed_by_intervention": False,
        "diagnostic_suffix_used": False,
        "native_successor_identity_audited": True,
        "primary_successor_identity_readout": (
            "paired_intervention_minus_clean_donor_vs_receiver_mean_logodds"
        ),
        "global_successor_argmax_diagnostic_only": True,
        "sum_logprob_identity_retained_for_audit_only": True,
        "candidate_scoring_mode": "sequential_full_forward_no_cache",
    }
    _atomic_jsonl(args.output, results)
    _atomic_json(args.summary, summary)
    print(json.dumps(summary["outcomes"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
