from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .attention_causal import _capture_cproj_inputs, _patched_head_forward
from .config import V10Config
from .core import (
    IGNORE_INDEX,
    Example,
    Rendered,
    Spans,
    Vocab,
    count_bin,
    count_prediction,
    margin,
    render,
)
from .report_followups import load_run, nested_example_pair
from .successor_conversion import (
    _replace_tensor_output,
    _tensor_from_output,
    capture_sublayer_states,
)
from .successor_mlp_features import (
    _feature_margin_coefficients,
    capture_mlp_intermediates,
    patched_mlp_intermediate_forward,
)
from .training import load_final_model


CONFLICT_PROMPT = "prompt_count_only"
CONFLICT_TRACE = "trace_count_and_length"
CONFLICT_FINAL_INDEX = "final_trace_index_only"
CONFLICT_MARKERS = "trace_marker_identity_control"
CONFLICTS = (CONFLICT_PROMPT, CONFLICT_TRACE, CONFLICT_FINAL_INDEX, CONFLICT_MARKERS)

COUNT_BINS = ("1-10", "11-20", "21-30")
HEAD_SUPPORTS = (1, 4, 8)
FEATURE_SUPPORTS = (8, 32, 128, 512, 1024)


@dataclass(frozen=True)
class ConflictCase:
    conflict_type: str
    receiver_count: int
    donor_count: int
    clean: Rendered
    conflict: Rendered
    target_count: int
    alternative_count: int


def _thinking_render(
    prompt: Example,
    vocab: Vocab,
    *,
    trace_count: int,
    trace_markers: list[str],
    trace_indices: list[int] | None = None,
    target_count: int | None = None,
) -> Rendered:
    """Render a controlled CoT input while keeping prompt and trace independently editable."""
    trace_count = int(trace_count)
    if len(trace_markers) != trace_count:
        raise ValueError("trace_markers must contain exactly trace_count entries")
    indices = list(range(1, trace_count + 1)) if trace_indices is None else list(trace_indices)
    if len(indices) != trace_count:
        raise ValueError("trace_indices must contain exactly trace_count entries")
    if any(not 1 <= int(value) <= len(vocab.numbers) for value in indices):
        raise ValueError("all controlled trace indices must be valid shared numeric tokens")

    trace: list[str] = []
    for value, marker in zip(indices, trace_markers):
        trace.extend([vocab.number_token(int(value)), marker])
    prompt_start = 1
    prompt_end = prompt_start + len(prompt.seq_tokens)
    think_pos = prompt_end
    trace_start = think_pos + 1
    trace_positions = list(range(trace_start, trace_start + len(trace)))
    index_positions = trace_positions[0::2]
    marker_positions = trace_positions[1::2]
    close_pos = trace_start + len(trace)
    ans_pos = close_pos + 1
    count_pos = ans_pos + 1
    eos_pos = count_pos + 1
    output_count = int(prompt.count if target_count is None else target_count)
    tokens = [
        "<BOS>",
        *prompt.seq_tokens,
        "<Think>",
        *trace,
        "</Think>",
        "<Ans>",
        vocab.number_token(output_count),
        "<EOS>",
    ]
    labels = [IGNORE_INDEX] * len(tokens)
    for position in range(trace_start, len(tokens)):
        labels[position] = vocab.token_to_id[tokens[position]]
    spans = Spans(
        bos_pos=0,
        prompt_start=prompt_start,
        prompt_end_exclusive=prompt_end,
        think_pos=think_pos,
        trace_index_positions=index_positions,
        trace_marker_positions=marker_positions,
        think_close_pos=close_pos,
        ans_pos=ans_pos,
        count_pos=count_pos,
        eos_pos=eos_pos,
    )
    return Rendered(
        mode="thinking",
        tokens=tokens,
        input_ids=vocab.encode(tokens),
        labels=labels,
        spans=spans,
        prompt_needle_positions=[prompt_start + value for value in prompt.needle_positions],
        count=output_count,
    )


def build_conflict_case(
    cfg: V10Config,
    vocab: Vocab,
    receiver_count: int,
    donor_count: int,
    seed: int,
    conflict_type: str,
) -> ConflictCase:
    """Construct natural donor and one controlled prompt/trace conflict."""
    receiver, donor = nested_example_pair(cfg, vocab, receiver_count, donor_count, seed)
    clean = render(donor, vocab, "thinking")

    if conflict_type == CONFLICT_PROMPT:
        # Donor prompt count m, receiver trace count n. Trace syntax and Ans position
        # are those of n, so only prompt evidence favors m.
        conflict = _thinking_render(
            donor,
            vocab,
            trace_count=receiver_count,
            trace_markers=receiver.needle_markers,
            target_count=donor_count,
        )
        target, alternative = donor_count, receiver_count
    elif conflict_type == CONFLICT_TRACE:
        # Receiver prompt count n, donor trace count m. Both trace content and trace
        # length/Ans absolute position now favor m.
        conflict = _thinking_render(
            receiver,
            vocab,
            trace_count=donor_count,
            trace_markers=donor.needle_markers,
            target_count=donor_count,
        )
        target, alternative = donor_count, receiver_count
    elif conflict_type == CONFLICT_FINAL_INDEX:
        # Prompt, marker trace, trace length, and Ans position remain n. Only the final
        # shared numeric index token is changed from <n> to <m>.
        indices = list(range(1, receiver_count + 1))
        indices[-1] = donor_count
        conflict = _thinking_render(
            receiver,
            vocab,
            trace_count=receiver_count,
            trace_markers=receiver.needle_markers,
            trace_indices=indices,
            target_count=donor_count,
        )
        target, alternative = donor_count, receiver_count
    elif conflict_type == CONFLICT_MARKERS:
        # A fixed-length negative control: rotate every trace marker identity while
        # prompt count, numeric trace, length, and Ans position remain n.
        rotated = [
            vocab.markers[(vocab.markers.index(marker) + 1) % len(vocab.markers)]
            for marker in receiver.needle_markers
        ]
        conflict = _thinking_render(
            receiver,
            vocab,
            trace_count=receiver_count,
            trace_markers=rotated,
            target_count=receiver_count,
        )
        clean = render(receiver, vocab, "thinking")
        target, alternative = receiver_count, donor_count
    else:
        raise ValueError(f"Unknown conflict type: {conflict_type}")

    return ConflictCase(
        conflict_type=conflict_type,
        receiver_count=int(receiver_count),
        donor_count=int(donor_count),
        clean=clean,
        conflict=conflict,
        target_count=int(target),
        alternative_count=int(alternative),
    )


def _pair_specs() -> list[tuple[int, int]]:
    return [(5, 8), (8, 5), (15, 18), (18, 15), (25, 28), (28, 25)]


def _count_pair_margin(logits: torch.Tensor, vocab: Vocab, target: int, alternative: int) -> float:
    values = logits.detach().float()
    return float(values[vocab.number_id(target)] - values[vocab.number_id(alternative)])


def _normalized_recovery(clean: float, conflict: float, patched: float) -> float:
    denominator = float(clean) - float(conflict)
    if not np.isfinite(denominator) or abs(denominator) < 1e-8:
        return math.nan
    return (float(patched) - float(conflict)) / denominator


def _behavior_row(
    logits: torch.Tensor,
    vocab: Vocab,
    case: ConflictCase,
    *,
    example_index: int,
) -> dict[str, Any]:
    pred, expected, _ = count_prediction(logits, vocab)
    return {
        "conflict_type": case.conflict_type,
        "example_index": int(example_index),
        "receiver_count": case.receiver_count,
        "donor_count": case.donor_count,
        "count_bin": count_bin(case.receiver_count),
        "target_count": case.target_count,
        "alternative_count": case.alternative_count,
        "prediction": pred,
        "expected_count": expected,
        "target_margin": _count_pair_margin(
            logits, vocab, case.target_count, case.alternative_count
        ),
        "follows_receiver": float(pred == case.receiver_count),
        "follows_donor": float(pred == case.donor_count),
        "follows_target": float(pred == case.target_count),
    }


def _final_query_rankings(run_dir: Path) -> dict[str, list[tuple[int, int]]]:
    frame = pd.read_csv(
        run_dir / "analysis" / "attention_causal" / "tables" / "attention_head_summary.csv"
    )
    final = frame[(frame["mode"] == "thinking") & (frame["query_kind"] == "final_count_query")]
    if final.empty:
        raise ValueError("thinking final_count_query attention summary is empty")

    def ranking(column: str) -> list[tuple[int, int]]:
        return [
            (int(row.layer), int(row.head))
            for row in final.sort_values(column, ascending=False).itertuples(index=False)
        ]

    return {
        "prompt_broad": ranking("broad_attention_score"),
        "trace_indices": ranking("trace_indices_mass"),
        "trace_markers": ranking("trace_markers_mass"),
    }


@torch.no_grad()
def _patched_component_forward(
    model,
    receiver_ids: torch.Tensor,
    receiver_position: int,
    donor_states: dict[tuple[int, str], torch.Tensor],
    *,
    layer: int,
    component: str,
) -> torch.Tensor:
    if component == "attn_out":
        module = model.transformer.h[int(layer)].attn
    elif component == "mlp_out":
        module = model.transformer.h[int(layer)].mlp
    elif component == "post_mlp":
        module = model.transformer.h[int(layer)]
    else:
        raise ValueError(f"Unknown component: {component}")

    def hook(_module, _args, output):
        hidden = _tensor_from_output(output)
        patched = hidden.clone()
        patched[:, int(receiver_position), :] = donor_states[(int(layer), component)].to(
            device=hidden.device, dtype=hidden.dtype
        )
        return _replace_tensor_output(output, patched)

    handle = module.register_forward_hook(hook)
    try:
        return model(input_ids=receiver_ids).logits[0, int(receiver_position)].detach()
    finally:
        handle.remove()


@torch.no_grad()
def run_cot_source_and_bridge(
    model,
    cfg: V10Config,
    vocab: Vocab,
    run_dir: Path,
    *,
    behavior_examples_per_pair: int = 4,
    patch_examples_per_pair: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = next(model.parameters()).device
    behavior_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    rankings = _final_query_rankings(run_dir)
    layers = set(range(int(cfg.n_layer)))
    total = len(_pair_specs()) * len(CONFLICTS) * int(behavior_examples_per_pair)
    progress = tqdm(total=total, desc="v10 CoT final-source conflicts and bridge patching")

    for receiver_count, donor_count in _pair_specs():
        for conflict_type in CONFLICTS:
            for example_index in range(int(behavior_examples_per_pair)):
                seed = cfg.seed + 3_710_000 + receiver_count * 10_000 + donor_count * 100 + example_index
                case = build_conflict_case(
                    cfg, vocab, receiver_count, donor_count, seed, conflict_type
                )
                clean_ids = torch.tensor([case.clean.input_ids], dtype=torch.long, device=device)
                conflict_ids = torch.tensor([case.conflict.input_ids], dtype=torch.long, device=device)
                clean_pos = int(case.clean.spans.ans_pos)
                conflict_pos = int(case.conflict.spans.ans_pos)
                clean_logits = model(input_ids=clean_ids).logits[0, clean_pos].detach()
                conflict_logits = model(input_ids=conflict_ids).logits[0, conflict_pos].detach()
                behavior_rows.append(
                    {
                        **_behavior_row(
                            conflict_logits, vocab, case, example_index=example_index
                        ),
                        "clean_prediction": count_prediction(clean_logits, vocab)[0],
                        "clean_expected_count": count_prediction(clean_logits, vocab)[1],
                        "clean_target_margin": _count_pair_margin(
                            clean_logits, vocab, case.target_count, case.alternative_count
                        ),
                        "ans_position_clean": clean_pos,
                        "ans_position_conflict": conflict_pos,
                        "ans_position_shift": conflict_pos - clean_pos,
                        "ans_position_abs_shift": abs(conflict_pos - clean_pos),
                    }
                )

                if example_index < int(patch_examples_per_pair):
                    clean_margin = _count_pair_margin(
                        clean_logits, vocab, case.target_count, case.alternative_count
                    )
                    conflict_margin = _count_pair_margin(
                        conflict_logits, vocab, case.target_count, case.alternative_count
                    )
                    donor_cproj = _capture_cproj_inputs(model, clean_ids, layers)
                    _, donor_states = capture_sublayer_states(model, clean_ids, clean_pos)

                    for family, ranking in rankings.items():
                        for support in HEAD_SUPPORTS:
                            heads = ranking[: min(int(support), len(ranking))]
                            logits = _patched_head_forward(
                                model,
                                conflict_ids,
                                donor_cproj,
                                heads,
                                donor_pos=clean_pos,
                                receiver_pos=conflict_pos,
                            )[conflict_pos]
                            pred, expected, _ = count_prediction(logits, vocab)
                            patched_margin = _count_pair_margin(
                                logits, vocab, case.target_count, case.alternative_count
                            )
                            patch_rows.append(
                                {
                                    "conflict_type": conflict_type,
                                    "example_index": example_index,
                                    "receiver_count": receiver_count,
                                    "donor_count": donor_count,
                                    "count_bin": count_bin(receiver_count),
                                    "intervention_family": "head_slices",
                                    "intervention": family,
                                    "layer": -1,
                                    "support_size": len(heads),
                                    "patched_heads": " ".join(
                                        f"L{layer + 1}H{head}" for layer, head in heads
                                    ),
                                    "clean_margin": clean_margin,
                                    "conflict_margin": conflict_margin,
                                    "patched_margin": patched_margin,
                                    "normalized_recovery": _normalized_recovery(
                                        clean_margin, conflict_margin, patched_margin
                                    ),
                                    "prediction": pred,
                                    "expected_count": expected,
                                    "follows_target": float(pred == case.target_count),
                                }
                            )

                    for layer in range(int(cfg.n_layer)):
                        for component in ("attn_out", "mlp_out", "post_mlp"):
                            logits = _patched_component_forward(
                                model,
                                conflict_ids,
                                conflict_pos,
                                donor_states,
                                layer=layer,
                                component=component,
                            )
                            pred, expected, _ = count_prediction(logits, vocab)
                            patched_margin = _count_pair_margin(
                                logits, vocab, case.target_count, case.alternative_count
                            )
                            patch_rows.append(
                                {
                                    "conflict_type": conflict_type,
                                    "example_index": example_index,
                                    "receiver_count": receiver_count,
                                    "donor_count": donor_count,
                                    "count_bin": count_bin(receiver_count),
                                    "intervention_family": "sublayer_or_residual",
                                    "intervention": component,
                                    "layer": layer,
                                    "support_size": -1,
                                    "patched_heads": "",
                                    "clean_margin": clean_margin,
                                    "conflict_margin": conflict_margin,
                                    "patched_margin": patched_margin,
                                    "normalized_recovery": _normalized_recovery(
                                        clean_margin, conflict_margin, patched_margin
                                    ),
                                    "prediction": pred,
                                    "expected_count": expected,
                                    "follows_target": float(pred == case.target_count),
                                }
                            )
                progress.update(1)
    progress.close()
    return pd.DataFrame(behavior_rows), pd.DataFrame(patch_rows)


def summarize_cot_behavior(detail: pd.DataFrame) -> pd.DataFrame:
    detail = detail.copy()
    if "ans_position_shift" not in detail.columns:
        detail["ans_position_shift"] = detail["ans_position_conflict"] - detail["ans_position_clean"]
    if "ans_position_abs_shift" not in detail.columns:
        detail["ans_position_abs_shift"] = detail["ans_position_shift"].abs()
    return (
        detail.groupby(["conflict_type", "count_bin"], as_index=False)
        .agg(
            n=("prediction", "size"),
            follows_receiver=("follows_receiver", "mean"),
            follows_donor=("follows_donor", "mean"),
            follows_target=("follows_target", "mean"),
            mean_prediction=("prediction", "mean"),
            mean_expected_count=("expected_count", "mean"),
            mean_target_margin=("target_margin", "mean"),
            mean_ans_position_shift=("ans_position_shift", "mean"),
            mean_abs_ans_position_shift=("ans_position_abs_shift", "mean"),
        )
        .sort_values(["conflict_type", "count_bin"])
        .reset_index(drop=True)
    )


def summarize_cot_bridge(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby(
            [
                "conflict_type",
                "count_bin",
                "intervention_family",
                "intervention",
                "layer",
                "support_size",
            ],
            as_index=False,
        )
        .agg(
            n=("prediction", "size"),
            mean_normalized_recovery=("normalized_recovery", "mean"),
            median_normalized_recovery=("normalized_recovery", "median"),
            follows_target=("follows_target", "mean"),
            mean_patched_margin=("patched_margin", "mean"),
        )
        .sort_values(
            ["conflict_type", "count_bin", "intervention_family", "intervention", "layer"]
        )
        .reset_index(drop=True)
    )


def _nonthinking_pair_specs() -> list[tuple[int, int]]:
    specs: list[tuple[int, int]] = []
    for lo, hi in ((1, 10), (11, 20), (21, 30)):
        midpoint = (lo + hi) / 2
        for receiver in range(lo, hi + 1):
            donor = min(hi, receiver + 3) if receiver <= midpoint else max(lo, receiver - 3)
            if donor != receiver:
                specs.append((receiver, donor))
    return specs


def _feature_replacements(
    donor: torch.Tensor,
    receiver: torch.Tensor,
    ranking: torch.Tensor,
    *,
    support_sizes: Iterable[int],
    random_replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    rng = np.random.default_rng(int(seed))
    n_inner = int(receiver.numel())
    conditions: list[dict[str, Any]] = []
    replacements: list[torch.Tensor] = []
    for support in support_sizes:
        size = min(int(support), n_inner)
        top = ranking[:size].long()
        replacement = receiver.clone()
        replacement[top] = donor[top]
        conditions.append({"family": "ranked", "support_size": size, "replicate": 0})
        replacements.append(replacement)
        for replicate in range(int(random_replicates)):
            indices = torch.tensor(rng.choice(n_inner, size=size, replace=False), dtype=torch.long)
            random_replacement = receiver.clone()
            random_replacement[indices] = donor[indices]
            conditions.append(
                {"family": "random", "support_size": size, "replicate": replicate}
            )
            replacements.append(random_replacement)
    return conditions, replacements


@torch.no_grad()
def run_nonthinking_mlp_transport(
    model,
    cfg: V10Config,
    vocab: Vocab,
    *,
    fit_examples_per_pair: int = 1,
    eval_examples_per_pair: int = 2,
    random_replicates: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit disjoint MLP feature rankings and causally transplant n→m evidence."""
    device = next(model.parameters()).device
    layers = tuple(range(int(cfg.n_layer)))
    accumulators: dict[tuple[str, int], dict[str, Any]] = {}
    feature_rows: list[dict[str, Any]] = []
    pair_specs = _nonthinking_pair_specs()

    for receiver_count, donor_count in tqdm(pair_specs, desc="fit non-thinking MLP features"):
        for example_index in range(int(fit_examples_per_pair)):
            seed = cfg.seed + 4_210_000 + receiver_count * 10_000 + donor_count * 100 + example_index
            receiver, donor = nested_example_pair(
                cfg, vocab, receiver_count, donor_count, seed
            )
            receiver_item = render(receiver, vocab, "nonthinking")
            donor_item = render(donor, vocab, "nonthinking")
            receiver_ids = torch.tensor([receiver_item.input_ids], dtype=torch.long, device=device)
            donor_ids = torch.tensor([donor_item.input_ids], dtype=torch.long, device=device)
            _, receiver_features = capture_mlp_intermediates(
                model, receiver_ids, receiver_item.spans.ans_pos, layers
            )
            _, donor_features = capture_mlp_intermediates(
                model, donor_ids, donor_item.spans.ans_pos, layers
            )
            for layer in layers:
                delta = donor_features[layer].float() - receiver_features[layer].float()
                coeff = _feature_margin_coefficients(
                    model, layer, vocab.number_id(donor_count), vocab.number_id(receiver_count)
                ).detach().float()
                evidence = delta * coeff
                key = (count_bin(receiver_count), layer)
                if key not in accumulators:
                    accumulators[key] = {
                        "n": 0,
                        "delta": torch.zeros_like(delta, device="cpu"),
                        "evidence": torch.zeros_like(evidence, device="cpu"),
                        "abs_evidence": torch.zeros_like(evidence, device="cpu"),
                        "positive": torch.zeros_like(evidence, device="cpu"),
                    }
                acc = accumulators[key]
                acc["n"] += 1
                acc["delta"] += delta.cpu()
                acc["evidence"] += evidence.cpu()
                acc["abs_evidence"] += evidence.abs().cpu()
                acc["positive"] += evidence.gt(0).float().cpu()

    fitted: dict[tuple[str, int], torch.Tensor] = {}
    for (bin_name, layer), acc in accumulators.items():
        n = max(1, int(acc["n"]))
        mean_evidence = acc["evidence"] / n
        ranking = torch.argsort(mean_evidence, descending=True)
        fitted[(bin_name, layer)] = ranking
        rank_lookup = torch.empty_like(ranking)
        rank_lookup[ranking] = torch.arange(len(ranking))
        for feature in range(int(mean_evidence.numel())):
            feature_rows.append(
                {
                    "count_bin": bin_name,
                    "layer": layer,
                    "feature": feature,
                    "n_fit_pairs": n,
                    "mean_activation_delta": float(acc["delta"][feature] / n),
                    "mean_projected_evidence": float(mean_evidence[feature]),
                    "mean_abs_projected_evidence": float(acc["abs_evidence"][feature] / n),
                    "positive_evidence_rate": float(acc["positive"][feature] / n),
                    "signed_rank": int(rank_lookup[feature]) + 1,
                }
            )

    patch_rows: list[dict[str, Any]] = []
    for receiver_count, donor_count in tqdm(pair_specs, desc="eval non-thinking MLP transport"):
        for example_index in range(int(eval_examples_per_pair)):
            seed = (
                cfg.seed
                + 4_810_000
                + receiver_count * 10_000
                + donor_count * 100
                + example_index
            )
            receiver, donor = nested_example_pair(
                cfg, vocab, receiver_count, donor_count, seed
            )
            receiver_item = render(receiver, vocab, "nonthinking")
            donor_item = render(donor, vocab, "nonthinking")
            receiver_ids = torch.tensor([receiver_item.input_ids], dtype=torch.long, device=device)
            donor_ids = torch.tensor([donor_item.input_ids], dtype=torch.long, device=device)
            receiver_logits, receiver_features = capture_mlp_intermediates(
                model, receiver_ids, receiver_item.spans.ans_pos, layers
            )
            donor_logits, donor_features = capture_mlp_intermediates(
                model, donor_ids, donor_item.spans.ans_pos, layers
            )
            receiver_pred, receiver_expected, _ = count_prediction(receiver_logits, vocab)
            donor_pred, donor_expected, _ = count_prediction(donor_logits, vocab)
            receiver_margin = _count_pair_margin(
                receiver_logits, vocab, donor_count, receiver_count
            )
            donor_margin = _count_pair_margin(donor_logits, vocab, donor_count, receiver_count)
            for layer in layers:
                conditions, replacements = _feature_replacements(
                    donor_features[layer],
                    receiver_features[layer],
                    fitted[(count_bin(receiver_count), layer)],
                    support_sizes=FEATURE_SUPPORTS,
                    random_replicates=random_replicates,
                    seed=seed + layer * 10_000,
                )
                logits_batch = patched_mlp_intermediate_forward(
                    model,
                    receiver_ids,
                    receiver_item.spans.ans_pos,
                    layer,
                    replacements,
                    batch_size=max(1, cfg.analysis_batch_size),
                )
                for condition, logits in zip(conditions, logits_batch):
                    pred, expected, _ = count_prediction(logits, vocab)
                    patched_margin = _count_pair_margin(
                        logits, vocab, donor_count, receiver_count
                    )
                    patch_rows.append(
                        {
                            "receiver_count": receiver_count,
                            "donor_count": donor_count,
                            "donor_offset": donor_count - receiver_count,
                            "count_bin": count_bin(receiver_count),
                            "example_index": example_index,
                            "layer": layer,
                            **condition,
                            "receiver_prediction": receiver_pred,
                            "donor_prediction": donor_pred,
                            "patched_prediction": pred,
                            "receiver_expected_count": receiver_expected,
                            "donor_expected_count": donor_expected,
                            "patched_expected_count": expected,
                            "causal_expected_shift": expected - receiver_expected,
                            "receiver_margin": receiver_margin,
                            "donor_margin": donor_margin,
                            "patched_margin": patched_margin,
                            "normalized_margin_recovery": _normalized_recovery(
                                donor_margin, receiver_margin, patched_margin
                            ),
                            "follows_donor": float(pred == donor_count),
                            "follows_receiver": float(pred == receiver_count),
                        }
                    )
    return pd.DataFrame(feature_rows), pd.DataFrame(patch_rows)


def summarize_nonthinking_transport(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, frame in detail.groupby(
        ["count_bin", "layer", "family", "support_size"], sort=False
    ):
        x = frame["donor_offset"].to_numpy(dtype=float)
        y = frame["causal_expected_shift"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        prediction = design @ beta
        denominator = float(((y - y.mean()) ** 2).sum())
        r2 = (
            1.0 - float(((y - prediction) ** 2).sum()) / denominator
            if denominator > 1e-12
            else math.nan
        )
        rows.append(
            {
                "count_bin": keys[0],
                "layer": int(keys[1]),
                "family": keys[2],
                "support_size": int(keys[3]),
                "n_pairs": len(frame),
                "transport_slope": float(beta[1]),
                "transport_intercept": float(beta[0]),
                "transport_r2": r2,
                "mean_normalized_margin_recovery": float(
                    frame["normalized_margin_recovery"].mean()
                ),
                "follows_donor": float(frame["follows_donor"].mean()),
                "follows_receiver": float(frame["follows_receiver"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["count_bin", "layer", "family", "support_size"]
    )


def _plot_source_attribution(summary: pd.DataFrame, path: Path) -> None:
    labels = {
        CONFLICT_PROMPT: "prompt=m, trace=n",
        CONFLICT_TRACE: "prompt=n, trace=m",
        CONFLICT_FINAL_INDEX: "only final trace index=m",
        CONFLICT_MARKERS: "trace markers rotated",
    }
    grouped = (
        summary.groupby("conflict_type", as_index=False)[["follows_receiver", "follows_donor"]]
        .mean()
        .set_index("conflict_type")
        .reindex(CONFLICTS)
    )
    x = np.arange(len(grouped))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.bar(x - width / 2, grouped["follows_receiver"], width, label="predict receiver n")
    ax.bar(x + width / 2, grouped["follows_donor"], width, label="predict donor m")
    ax.set_xticks(x, [labels[value] for value in grouped.index], rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of conflict examples")
    ax.set_title("CoT final answer under independently controlled prompt/trace evidence")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_bridge(summary: pd.DataFrame, path: Path) -> None:
    # A normalized recovery is meaningful only when corruption opens a clean-to-
    # conflict margin gap. The other controls already predict the clean target.
    component = summary[
        (summary["intervention_family"] == "sublayer_or_residual")
        & summary["conflict_type"].isin((CONFLICT_PROMPT, CONFLICT_FINAL_INDEX))
    ].copy()
    component["label"] = component.apply(
        lambda row: f"L{int(row.layer) + 1} {row.intervention}", axis=1
    )
    pivot = component.pivot_table(
        index="label",
        columns="conflict_type",
        values="mean_normalized_recovery",
        aggfunc="mean",
    ).reindex(columns=[CONFLICT_PROMPT, CONFLICT_FINAL_INDEX])
    order: list[str] = []
    n_layers = int(component["layer"].max()) + 1 if not component.empty else 0
    for layer in range(n_layers):
        order.extend([f"L{layer + 1} attn_out", f"L{layer + 1} mlp_out", f"L{layer + 1} post_mlp"])
    pivot = pivot.reindex([value for value in order if value in pivot.index])
    fig, ax = plt.subplots(figsize=(7.8, 7.1))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)), [value.replace("_", " ") for value in pivot.columns], rotation=18, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_xlabel("controlled conflict")
    ax.set_ylabel("clean component patched at <Ans>")
    ax.set_title("CoT final scalar-count bridge at <Ans>")
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            value = pivot.iloc[row, column]
            if np.isfinite(value):
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="normalized clean-margin recovery")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_mlp_transport(summary: pd.DataFrame, path: Path) -> None:
    ranked = summary[summary["family"] == "ranked"]
    random_rows = summary[summary["family"] == "random"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), sharey=True)
    for ax, bin_name in zip(axes, COUNT_BINS):
        for layer in sorted(ranked.layer.unique()):
            frame = ranked[(ranked.count_bin == bin_name) & (ranked.layer == layer)].sort_values("support_size")
            ax.plot(frame.support_size, frame.transport_slope, marker="o", label=f"Layer {int(layer) + 1}")
        control = random_rows[random_rows.count_bin == bin_name].groupby("support_size", as_index=False).transport_slope.mean()
        ax.plot(control.support_size, control.transport_slope, color="#6b7280", linestyle="--", marker=".", label="random mean")
        ax.axhline(0, color="#9ca3af", linewidth=1)
        ax.axhline(1, color="#111827", linewidth=1, linestyle=":")
        ax.set_xscale("log", base=2)
        ax.set_title(f"receiver count {bin_name}")
        ax.set_xlabel("number of patched post-GELU features")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("expected-count transport slope")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        "Non-thinking MLP feature transplant: sparse receiver-to-donor count transport",
        y=0.98,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=5,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.81))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_final_bridge_causal(
    run_dir: str | Path,
    *,
    device: str | None = None,
    behavior_examples_per_pair: int = 4,
    patch_examples_per_pair: int = 2,
    feature_fit_examples_per_pair: int = 1,
    feature_eval_examples_per_pair: int = 2,
    random_replicates: int = 3,
    skip_completed: bool = True,
) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    out_dir = run_dir / "analysis" / "final_bridge_causal"
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    paths = {
        "cot_conflict_behavior": tables / "cot_conflict_behavior.csv",
        "cot_conflict_summary": tables / "cot_conflict_summary.csv",
        "cot_bridge_patching": tables / "cot_final_bridge_patching.csv",
        "cot_bridge_summary": tables / "cot_final_bridge_patching_summary.csv",
        "nonthinking_feature_stats": tables / "nonthinking_mlp_feature_stats.csv",
        "nonthinking_feature_patching": tables / "nonthinking_mlp_feature_patching.csv",
        "nonthinking_transport_summary": tables / "nonthinking_mlp_transport_summary.csv",
    }
    if skip_completed and all(path.exists() and path.stat().st_size > 0 for path in paths.values()):
        outputs = {name: pd.read_csv(path) for name, path in paths.items()}
        _plot_source_attribution(outputs["cot_conflict_summary"], figures / "cot_conflict_source_attribution.png")
        _plot_bridge(outputs["cot_bridge_summary"], figures / "cot_final_bridge_component_recovery.png")
        _plot_mlp_transport(outputs["nonthinking_transport_summary"], figures / "nonthinking_mlp_feature_transport.png")
        return outputs

    cfg, vocab = load_run(run_dir, device=device)
    thinking_model = load_final_model(cfg, vocab, run_dir, "thinking")
    cot_behavior, cot_patch = run_cot_source_and_bridge(
        thinking_model,
        cfg,
        vocab,
        run_dir,
        behavior_examples_per_pair=behavior_examples_per_pair,
        patch_examples_per_pair=patch_examples_per_pair,
    )
    del thinking_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    nonthinking_model = load_final_model(cfg, vocab, run_dir, "nonthinking")
    feature_stats, feature_patch = run_nonthinking_mlp_transport(
        nonthinking_model,
        cfg,
        vocab,
        fit_examples_per_pair=feature_fit_examples_per_pair,
        eval_examples_per_pair=feature_eval_examples_per_pair,
        random_replicates=random_replicates,
    )
    del nonthinking_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    outputs = {
        "cot_conflict_behavior": cot_behavior,
        "cot_conflict_summary": summarize_cot_behavior(cot_behavior),
        "cot_bridge_patching": cot_patch,
        "cot_bridge_summary": summarize_cot_bridge(cot_patch),
        "nonthinking_feature_stats": feature_stats,
        "nonthinking_feature_patching": feature_patch,
        "nonthinking_transport_summary": summarize_nonthinking_transport(feature_patch),
    }
    for name, frame in outputs.items():
        frame.to_csv(paths[name], index=False)

    _plot_source_attribution(outputs["cot_conflict_summary"], figures / "cot_conflict_source_attribution.png")
    _plot_bridge(outputs["cot_bridge_summary"], figures / "cot_final_bridge_component_recovery.png")
    _plot_mlp_transport(outputs["nonthinking_transport_summary"], figures / "nonthinking_mlp_feature_transport.png")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "analysis": "v10 final scalar bridge and non-thinking MLP feature causality",
                "behavior_examples_per_pair": behavior_examples_per_pair,
                "patch_examples_per_pair": patch_examples_per_pair,
                "feature_fit_examples_per_pair": feature_fit_examples_per_pair,
                "feature_eval_examples_per_pair": feature_eval_examples_per_pair,
                "random_replicates": random_replicates,
                "pair_specs": _pair_specs(),
                "feature_pair_specs": _nonthinking_pair_specs(),
                "conflicts": list(CONFLICTS),
                "head_supports": list(HEAD_SUPPORTS),
                "feature_supports": list(FEATURE_SUPPORTS),
                "fit_eval_disjoint": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return outputs
