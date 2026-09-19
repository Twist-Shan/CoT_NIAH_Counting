from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import V10Config
from .core import (
    Example,
    Vocab,
    count_bin,
    count_prediction,
    make_example,
    margin,
    render,
)
from .report_followups import load_run, nested_example_pair
from .training import load_final_model


def _post_block_states(
    model,
    input_ids: torch.Tensor,
    position: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Capture the actual post-block residual used by a block-output patch.

    Hugging Face's final ``hidden_states`` entry is after the final layer norm,
    whereas a GPT-2 block forward hook sees the pre-final-LN residual. Capturing
    with the same hook location used for patching keeps all layers in a
    representation-matched space.
    """
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer] = hidden[:, int(position), :].detach().clone()

        return hook

    for layer, block in enumerate(model.transformer.h):
        handles.append(block.register_forward_hook(make_hook(layer)))
    try:
        logits = model(input_ids=input_ids).logits[0].detach()
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(range(len(model.transformer.h))) - set(captured))
    if missing:
        raise RuntimeError(f"Failed to capture post-block residuals for layers {missing}")
    return logits, [captured[layer] for layer in range(len(model.transformer.h))]


def _patched_forward(
    model,
    receiver_ids: torch.Tensor,
    layer: int,
    receiver_position: int,
    donor_state: torch.Tensor,
) -> torch.Tensor:
    """Replace one token's post-block residual and return all-position logits."""

    def hook(_module, _args, output):
        is_tuple = isinstance(output, tuple)
        hidden = (output[0] if is_tuple else output).clone()
        hidden[:, int(receiver_position), :] = donor_state.to(
            device=hidden.device, dtype=hidden.dtype
        )
        return (hidden, *output[1:]) if is_tuple else hidden

    handle = model.transformer.h[int(layer)].register_forward_hook(hook)
    try:
        return model(input_ids=receiver_ids).logits[0].detach()
    finally:
        handle.remove()


def _pair_margin(logits: torch.Tensor, vocab: Vocab, donor: int, receiver: int) -> float:
    if int(donor) == int(receiver):
        return math.nan
    return float(logits[vocab.number_id(donor)] - logits[vocab.number_id(receiver)])


def _normalized_recovery(
    patched: float,
    receiver: float,
    donor: float,
) -> float:
    denominator = float(donor) - float(receiver)
    if not np.isfinite(denominator) or abs(denominator) < 1e-8:
        return math.nan
    return (float(patched) - float(receiver)) / denominator


def _prefix_nested_pair(
    cfg: V10Config,
    vocab: Vocab,
    short_count: int,
    long_count: int,
    seed: int,
) -> tuple[Example, Example]:
    """Create count-m/count-n prompts whose ordered trace is prefix-aligned."""
    short_count, long_count = int(short_count), int(long_count)
    if not cfg.count_min <= short_count < long_count <= cfg.count_max:
        raise ValueError("prefix pair requires count_min <= short_count < long_count <= count_max")
    rng = random.Random(int(seed))
    positions = sorted(rng.sample(range(cfg.seq_len), long_count))
    markers = [rng.choice(vocab.markers) for _ in positions]
    base_noise = [rng.choice(vocab.noise) for _ in range(cfg.seq_len)]

    def build(count: int) -> Example:
        selected_positions = positions[:count]
        selected_markers = markers[:count]
        tokens = list(base_noise)
        for position, marker in zip(selected_positions, selected_markers):
            tokens[position] = marker
        return Example(tokens, count, list(selected_positions), list(selected_markers), int(seed))

    return build(short_count), build(long_count)


def _truncated_thinking_ids(
    example: Example,
    vocab: Vocab,
    trace_count: int,
) -> tuple[list[int], int, int]:
    """Render prompt + first m trace pairs + forced close + Ans, without gold tail."""
    trace: list[str] = []
    for k in range(1, int(trace_count) + 1):
        trace.extend([vocab.number_token(k), example.needle_markers[k - 1]])
    tokens = [
        "<BOS>",
        *example.seq_tokens,
        "<Think>",
        *trace,
        "</Think>",
        "<Ans>",
    ]
    marker_position = 1 + len(example.seq_tokens) + 2 * int(trace_count)
    ans_position = len(tokens) - 1
    return vocab.encode(tokens), marker_position, ans_position


def _thinking_prefix_ids(
    example: Example,
    vocab: Vocab,
    trace_progress: int,
) -> tuple[list[int], int]:
    """Render only prompt + a trace prefix ending at ``M_progress``.

    No ``</Think>``, ``<Ans>``, final count, or later gold trace token is
    included. This is the input used by the free-rollout transplant test.
    """
    trace_progress = int(trace_progress)
    if not 1 <= trace_progress <= int(example.count):
        raise ValueError(
            f"trace_progress must be in 1..example.count, got {trace_progress} "
            f"for count={example.count}"
        )
    trace: list[str] = []
    for k in range(1, trace_progress + 1):
        trace.extend([vocab.number_token(k), example.needle_markers[k - 1]])
    tokens = ["<BOS>", *example.seq_tokens, "<Think>", *trace]
    return vocab.encode(tokens), len(tokens) - 1


def _decode_free_rollout(
    suffix_ids: list[int],
    vocab: Vocab,
    *,
    receiver_progress: int,
) -> dict[str, Any]:
    """Parse a freely generated suffix without assuming a well-formed trace."""
    tokens = vocab.decode(suffix_ids)
    close_index = next(
        (index for index, token in enumerate(tokens) if token == "</Think>"), None
    )
    trace_tokens = tokens if close_index is None else tokens[:close_index]
    generated_indices = [
        value
        for token_id in suffix_ids[: len(trace_tokens)]
        if (value := vocab.number_from_id(token_id)) is not None
    ]
    generated_markers = [token for token in trace_tokens if token in vocab.markers]
    ans_index = next(
        (
            index
            for index, token in enumerate(tokens)
            if token == "<Ans>" and (close_index is None or index > close_index)
        ),
        None,
    )
    final_count = None
    if ans_index is not None:
        for token_id in suffix_ids[ans_index + 1 :]:
            value = vocab.number_from_id(token_id)
            if value is not None:
                final_count = int(value)
                break
    stop_index = (
        int(generated_indices[-1]) if generated_indices else int(receiver_progress)
    )
    return {
        "generated_suffix": " ".join(tokens),
        "closed_trace": float(close_index is not None),
        "emitted_ans": float(ans_index is not None),
        "emitted_eos": float("<EOS>" in tokens),
        "tokens_until_close": (
            float(close_index + 1) if close_index is not None else math.nan
        ),
        "trace_markers_after_patch": int(len(generated_markers)),
        "first_generated_index": (
            float(generated_indices[0]) if generated_indices else math.nan
        ),
        "last_generated_index": (
            float(generated_indices[-1]) if generated_indices else math.nan
        ),
        "inferred_stop_index": stop_index,
        "generated_index_sequence": ",".join(str(value) for value in generated_indices),
        "final_count": float(final_count) if final_count is not None else math.nan,
    }


@torch.no_grad()
def _free_rollout_with_residual_patch(
    model,
    prefix_ids: list[int],
    vocab: Vocab,
    *,
    layer: int | None = None,
    patch_position: int | None = None,
    replacement_state: torch.Tensor | None = None,
    patch_policy: str = "none",
    max_new_tokens: int = 32,
) -> list[int]:
    """Greedy rollout after a one-shot or persistent post-Layer patch.

    ``one_shot`` changes only the forward pass that predicts the first token
    after the truncated trace. ``persistent`` reapplies the same replacement
    at the original receiver anchor on every full-prefix forward pass.
    """
    if patch_policy not in {"none", "one_shot", "persistent"}:
        raise ValueError(f"Unknown patch_policy: {patch_policy}")
    if patch_policy != "none" and (
        layer is None or patch_position is None or replacement_state is None
    ):
        raise ValueError("A residual patch requires layer, position, and replacement_state")
    device = next(model.parameters()).device
    generated: list[int] = []
    for step in range(int(max_new_tokens)):
        current = torch.tensor([prefix_ids + generated], dtype=torch.long, device=device)
        should_patch = patch_policy == "persistent" or (
            patch_policy == "one_shot" and step == 0
        )
        if should_patch:
            logits = _patched_forward(
                model,
                current,
                int(layer),
                int(patch_position),
                replacement_state,
            )
        else:
            logits = model(input_ids=current).logits[0].detach()
        next_id = int(logits[-1].argmax().item())
        generated.append(next_id)
        if next_id == vocab.eos_id:
            break
    return generated


def _fit_trace_progress_centroids(
    model,
    cfg: V10Config,
    vocab: Vocab,
    sites: Iterable[tuple[int, int]],
    *,
    examples_per_site: int,
    seed: int,
) -> dict[tuple[int, int], list[torch.Tensor]]:
    """Fit independent post-Layer centroids for (prompt total, trace progress)."""
    centroids: dict[tuple[int, int], list[torch.Tensor]] = {}
    device = next(model.parameters()).device
    for total_count, trace_progress in sorted(set(sites)):
        states_by_layer: list[list[torch.Tensor]] = [
            [] for _ in range(int(cfg.n_layer))
        ]
        for example_index in range(int(examples_per_site)):
            rng = random.Random(
                int(seed)
                + int(total_count) * 100_000
                + int(trace_progress) * 1_000
                + example_index
            )
            example = make_example(
                cfg,
                vocab,
                rng,
                count=int(total_count),
                seed=int(seed) + example_index,
            )
            ids, position = _thinking_prefix_ids(example, vocab, int(trace_progress))
            tensor = torch.tensor([ids], dtype=torch.long, device=device)
            _, states = _post_block_states(model, tensor, position)
            for layer, state in enumerate(states):
                states_by_layer[layer].append(state.detach().float().cpu())
        centroids[(int(total_count), int(trace_progress))] = [
            torch.stack(layer_states).mean(dim=0)
            for layer_states in states_by_layer
        ]
    return centroids


def _regression_summary(
    frame: pd.DataFrame,
    groups: list[str],
    *,
    x_column: str = "donor_offset",
    y_column: str = "causal_expected_shift",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, part in frame.groupby(groups, sort=False, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        finite = part[np.isfinite(part[x_column]) & np.isfinite(part[y_column])]
        if len(finite) < 2 or finite[x_column].nunique() < 2:
            slope = intercept = r2 = math.nan
        else:
            x = finite[x_column].to_numpy(dtype=float)
            y = finite[y_column].to_numpy(dtype=float)
            design = np.column_stack([np.ones(len(x)), x])
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            prediction = design @ beta
            denominator = float(((y - y.mean()) ** 2).sum())
            slope, intercept = float(beta[1]), float(beta[0])
            r2 = (
                1.0 - float(((y - prediction) ** 2).sum()) / denominator
                if denominator > 1e-12
                else math.nan
            )
        row = {
            **dict(zip(groups, keys)),
            "n_rows": int(len(part)),
            "transport_slope": slope,
            "transport_intercept": intercept,
            "transport_r2": r2,
        }
        for column in (
            "follows_donor",
            "follows_receiver",
            "normalized_recovery",
            "close_margin_shift",
            "close_predicted",
        ):
            if column in part:
                row[f"mean_{column}"] = float(part[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _with_patch_direction(frame: pd.DataFrame) -> pd.DataFrame:
    """Label donor/receiver count ordering without discarding either direction."""
    required = {"donor_count", "receiver_count"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Cannot label patch direction; missing columns: {sorted(missing)}")
    result = frame.copy()
    donor = result["donor_count"].to_numpy(dtype=float)
    receiver = result["receiver_count"].to_numpy(dtype=float)
    result["patch_direction"] = np.select(
        [donor > receiver, donor < receiver],
        ["donor_gt_receiver", "donor_lt_receiver"],
        default="same_count",
    )
    return result


def _directional_regression_summary(
    frame: pd.DataFrame,
    groups: list[str],
) -> pd.DataFrame:
    """Fit transport separately for upward and downward count transplants."""
    directed = _with_patch_direction(frame)
    directed = directed[directed.patch_direction != "same_count"].copy()
    return _regression_summary(
        directed,
        [*groups, "patch_direction"],
    )


@torch.no_grad()
def run_final_answer_patching(
    models: dict[str, Any],
    cfg: V10Config,
    vocab: Vocab,
    *,
    examples_per_pair: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    pair_specs = [
        (receiver, int(offset))
        for receiver in range(cfg.count_min, cfg.count_max + 1)
        for offset in (0, *cfg.patch_offsets)
        if cfg.count_min <= receiver + int(offset) <= cfg.count_max
    ]
    total = len(cfg.modes) * len(pair_specs) * int(examples_per_pair) * cfg.n_layer
    progress = tqdm(total=total, desc="v10 hidden patch: final Ans")
    for mode in cfg.modes:
        model = models[mode]
        device = next(model.parameters()).device
        for receiver_count, offset in pair_specs:
            donor_count = receiver_count + offset
            for example_index in range(int(examples_per_pair)):
                seed = cfg.seed + 1_210_000 + receiver_count * 10_000 + (offset + 30) * 100 + example_index
                receiver, donor = nested_example_pair(
                    cfg, vocab, receiver_count, donor_count, seed
                )
                receiver_item = render(receiver, vocab, mode)
                donor_item = render(donor, vocab, mode)
                receiver_ids = torch.tensor([receiver_item.input_ids], dtype=torch.long, device=device)
                donor_ids = torch.tensor([donor_item.input_ids], dtype=torch.long, device=device)
                receiver_logits_all, receiver_states = _post_block_states(
                    model, receiver_ids, receiver_item.spans.ans_pos
                )
                donor_logits_all, donor_states = _post_block_states(
                    model, donor_ids, donor_item.spans.ans_pos
                )
                receiver_logits = receiver_logits_all[receiver_item.spans.ans_pos]
                donor_logits = donor_logits_all[donor_item.spans.ans_pos]
                receiver_pred, receiver_expected, _ = count_prediction(receiver_logits, vocab)
                donor_pred, donor_expected, _ = count_prediction(donor_logits, vocab)
                receiver_pair_margin = _pair_margin(
                    receiver_logits, vocab, donor_count, receiver_count
                )
                donor_pair_margin = _pair_margin(
                    donor_logits, vocab, donor_count, receiver_count
                )
                for layer, donor_state in enumerate(donor_states):
                    patched_all = _patched_forward(
                        model,
                        receiver_ids,
                        layer,
                        receiver_item.spans.ans_pos,
                        donor_state,
                    )
                    patched_logits = patched_all[receiver_item.spans.ans_pos]
                    patched_pred, patched_expected, _ = count_prediction(patched_logits, vocab)
                    patched_pair_margin = _pair_margin(
                        patched_logits, vocab, donor_count, receiver_count
                    )
                    rows.append(
                        {
                            "mode": mode,
                            "site": "final_answer_query",
                            "receiver_count": receiver_count,
                            "donor_count": donor_count,
                            "count_bin": count_bin(receiver_count),
                            "donor_offset": offset,
                            "example_index": example_index,
                            "layer": layer + 1,
                            "receiver_position": receiver_item.spans.ans_pos,
                            "donor_position": donor_item.spans.ans_pos,
                            "position_delta": donor_item.spans.ans_pos - receiver_item.spans.ans_pos,
                            "receiver_pred": receiver_pred,
                            "receiver_expected": receiver_expected,
                            "donor_pred": donor_pred,
                            "donor_expected": donor_expected,
                            "patched_pred": patched_pred,
                            "patched_expected": patched_expected,
                            "causal_expected_shift": patched_expected - receiver_expected,
                            "follows_donor": float(patched_pred == donor_count),
                            "follows_receiver": float(patched_pred == receiver_count),
                            "receiver_pair_margin": receiver_pair_margin,
                            "donor_pair_margin": donor_pair_margin,
                            "patched_pair_margin": patched_pair_margin,
                            "normalized_recovery": _normalized_recovery(
                                patched_pair_margin, receiver_pair_margin, donor_pair_margin
                            ),
                            "donor_receiver_state_l2": float(
                                torch.linalg.vector_norm(
                                    donor_state - receiver_states[layer]
                                ).item()
                            ),
                        }
                    )
                    progress.update(1)
    progress.close()
    detail = _with_patch_direction(pd.DataFrame(rows))
    summary = _regression_summary(
        detail,
        ["mode", "site", "count_bin", "layer"],
    )
    return detail, summary


@torch.no_grad()
def run_trace_final_patching(
    model,
    cfg: V10Config,
    vocab: Vocab,
    *,
    examples_per_pair: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    pair_specs = [
        (receiver, int(offset))
        for receiver in range(cfg.count_min, cfg.count_max + 1)
        for offset in cfg.patch_offsets
        if cfg.count_min <= receiver + int(offset) <= cfg.count_max
    ]
    progress = tqdm(
        total=len(pair_specs) * int(examples_per_pair) * cfg.n_layer,
        desc="v10 hidden patch: trace final-to-final",
    )
    device = next(model.parameters()).device
    for receiver_count, offset in pair_specs:
        donor_count = receiver_count + offset
        for example_index in range(int(examples_per_pair)):
            seed = cfg.seed + 1_220_000 + receiver_count * 10_000 + (offset + 30) * 100 + example_index
            receiver, donor = _prefix_nested_pair(
                cfg,
                vocab,
                min(receiver_count, donor_count),
                max(receiver_count, donor_count),
                seed,
            )
            if receiver.count != receiver_count:
                receiver, donor = donor, receiver
            receiver_item = render(receiver, vocab, "thinking")
            donor_item = render(donor, vocab, "thinking")
            receiver_pos = receiver_item.spans.trace_marker_positions[-1]
            donor_pos = donor_item.spans.trace_marker_positions[-1]
            receiver_ids = torch.tensor([receiver_item.input_ids], dtype=torch.long, device=device)
            donor_ids = torch.tensor([donor_item.input_ids], dtype=torch.long, device=device)
            receiver_all, _ = _post_block_states(model, receiver_ids, receiver_pos)
            donor_all, donor_states = _post_block_states(model, donor_ids, donor_pos)
            receiver_ans = receiver_all[receiver_item.spans.ans_pos]
            donor_ans = donor_all[donor_item.spans.ans_pos]
            receiver_pred, receiver_expected, _ = count_prediction(receiver_ans, vocab)
            donor_pred, donor_expected, _ = count_prediction(donor_ans, vocab)
            receiver_pair_margin = _pair_margin(receiver_ans, vocab, donor_count, receiver_count)
            donor_pair_margin = _pair_margin(donor_ans, vocab, donor_count, receiver_count)
            baseline_close_margin = margin(
                receiver_all[receiver_pos], vocab.think_close_id, vocab.number_ids
            )
            for layer, donor_state in enumerate(donor_states):
                patched_all = _patched_forward(
                    model, receiver_ids, layer, receiver_pos, donor_state
                )
                patched_ans = patched_all[receiver_item.spans.ans_pos]
                patched_pred, patched_expected, _ = count_prediction(patched_ans, vocab)
                patched_pair_margin = _pair_margin(
                    patched_ans, vocab, donor_count, receiver_count
                )
                patched_close_margin = margin(
                    patched_all[receiver_pos], vocab.think_close_id, vocab.number_ids
                )
                rows.append(
                    {
                        "mode": "thinking",
                        "site": "trace_final_marker_to_final_marker",
                        "receiver_count": receiver_count,
                        "donor_count": donor_count,
                        "count_bin": count_bin(receiver_count),
                        "donor_offset": offset,
                        "example_index": example_index,
                        "layer": layer + 1,
                        "receiver_position": receiver_pos,
                        "donor_position": donor_pos,
                        "position_delta": donor_pos - receiver_pos,
                        "receiver_pred": receiver_pred,
                        "receiver_expected": receiver_expected,
                        "donor_pred": donor_pred,
                        "donor_expected": donor_expected,
                        "patched_pred": patched_pred,
                        "patched_expected": patched_expected,
                        "causal_expected_shift": patched_expected - receiver_expected,
                        "follows_donor": float(patched_pred == donor_count),
                        "follows_receiver": float(patched_pred == receiver_count),
                        "receiver_pair_margin": receiver_pair_margin,
                        "donor_pair_margin": donor_pair_margin,
                        "patched_pair_margin": patched_pair_margin,
                        "normalized_recovery": _normalized_recovery(
                            patched_pair_margin, receiver_pair_margin, donor_pair_margin
                        ),
                        "baseline_close_margin": baseline_close_margin,
                        "patched_close_margin": patched_close_margin,
                        "close_margin_shift": patched_close_margin - baseline_close_margin,
                    }
                )
                progress.update(1)
    progress.close()
    detail = _with_patch_direction(pd.DataFrame(rows))
    summary = _regression_summary(
        detail,
        ["mode", "site", "count_bin", "layer"],
    )
    return detail, summary


@torch.no_grad()
def run_trace_early_stop_patching(
    model,
    cfg: V10Config,
    vocab: Vocab,
    *,
    examples_per_pair: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    positive_offsets = sorted({int(value) for value in cfg.patch_offsets if int(value) > 0})
    pairs = [
        (short, short + offset)
        for short in range(cfg.count_min, cfg.count_max)
        for offset in positive_offsets
        if short + offset <= cfg.count_max
    ]
    progress = tqdm(
        total=len(pairs) * int(examples_per_pair) * cfg.n_layer,
        desc="v10 hidden patch: trace early stop",
    )
    device = next(model.parameters()).device
    for donor_count, receiver_count in pairs:
        for example_index in range(int(examples_per_pair)):
            seed = cfg.seed + 1_230_000 + receiver_count * 10_000 + donor_count * 100 + example_index
            donor, receiver = _prefix_nested_pair(
                cfg, vocab, donor_count, receiver_count, seed
            )
            donor_item = render(donor, vocab, "thinking")
            receiver_item = render(receiver, vocab, "thinking")
            donor_pos = donor_item.spans.trace_marker_positions[-1]
            receiver_pos = receiver_item.spans.trace_marker_positions[donor_count - 1]
            if donor_pos != receiver_pos:
                raise AssertionError("prefix-aligned traces must share the M_m token position")
            donor_ids = torch.tensor([donor_item.input_ids], dtype=torch.long, device=device)
            receiver_ids = torch.tensor([receiver_item.input_ids], dtype=torch.long, device=device)
            truncated_tokens, truncated_pos, truncated_ans_pos = _truncated_thinking_ids(
                receiver, vocab, donor_count
            )
            if truncated_pos != receiver_pos:
                raise AssertionError("truncated receiver must preserve the M_m position")
            truncated_ids = torch.tensor([truncated_tokens], dtype=torch.long, device=device)

            donor_all, donor_states = _post_block_states(model, donor_ids, donor_pos)
            receiver_all, _ = _post_block_states(model, receiver_ids, receiver_pos)
            truncated_all, _ = _post_block_states(model, truncated_ids, truncated_pos)
            next_id = vocab.number_id(donor_count + 1)
            baseline_close_margin = float(
                receiver_all[receiver_pos, vocab.think_close_id]
                - receiver_all[receiver_pos, next_id]
            )
            donor_close_margin = float(
                donor_all[donor_pos, vocab.think_close_id]
                - donor_all[donor_pos, next_id]
            )
            truncated_ans = truncated_all[truncated_ans_pos]
            truncated_pred, truncated_expected, _ = count_prediction(truncated_ans, vocab)
            donor_ans = donor_all[donor_item.spans.ans_pos]
            donor_pred, donor_expected, _ = count_prediction(donor_ans, vocab)
            baseline_pair_margin = _pair_margin(
                truncated_ans, vocab, donor_count, receiver_count
            )
            donor_pair_margin = _pair_margin(donor_ans, vocab, donor_count, receiver_count)
            for layer, donor_state in enumerate(donor_states):
                patched_full = _patched_forward(
                    model, receiver_ids, layer, receiver_pos, donor_state
                )
                patched_close_margin = float(
                    patched_full[receiver_pos, vocab.think_close_id]
                    - patched_full[receiver_pos, next_id]
                )
                patched_local_id = int(patched_full[receiver_pos].argmax().item())

                patched_truncated = _patched_forward(
                    model, truncated_ids, layer, truncated_pos, donor_state
                )
                patched_ans = patched_truncated[truncated_ans_pos]
                patched_pred, patched_expected, _ = count_prediction(patched_ans, vocab)
                patched_pair_margin = _pair_margin(
                    patched_ans, vocab, donor_count, receiver_count
                )
                rows.append(
                    {
                        "mode": "thinking",
                        "site": "trace_prefix_early_stop",
                        "receiver_count": receiver_count,
                        "donor_count": donor_count,
                        "count_bin": count_bin(receiver_count),
                        "donor_offset": donor_count - receiver_count,
                        "trace_gap": receiver_count - donor_count,
                        "example_index": example_index,
                        "layer": layer + 1,
                        "receiver_position": receiver_pos,
                        "donor_position": donor_pos,
                        "position_delta": donor_pos - receiver_pos,
                        "baseline_close_margin": baseline_close_margin,
                        "donor_close_margin": donor_close_margin,
                        "patched_close_margin": patched_close_margin,
                        "close_margin_shift": patched_close_margin - baseline_close_margin,
                        "close_recovery": _normalized_recovery(
                            patched_close_margin, baseline_close_margin, donor_close_margin
                        ),
                        "close_predicted": float(patched_local_id == vocab.think_close_id),
                        "baseline_truncated_pred": truncated_pred,
                        "baseline_truncated_expected": truncated_expected,
                        "donor_pred": donor_pred,
                        "donor_expected": donor_expected,
                        "patched_pred": patched_pred,
                        "patched_expected": patched_expected,
                        "causal_expected_shift": patched_expected - truncated_expected,
                        "follows_donor": float(patched_pred == donor_count),
                        "follows_receiver": float(patched_pred == receiver_count),
                        "receiver_pair_margin": baseline_pair_margin,
                        "donor_pair_margin": donor_pair_margin,
                        "patched_pair_margin": patched_pair_margin,
                        "normalized_recovery": _normalized_recovery(
                            patched_pair_margin, baseline_pair_margin, donor_pair_margin
                        ),
                        "uses_gold_trace_tail": False,
                        "forced_close_and_ans_tokens": True,
                    }
                )
                progress.update(1)
    progress.close()
    detail = pd.DataFrame(rows)
    summary = _regression_summary(
        detail,
        ["mode", "site", "count_bin", "layer"],
    )
    return detail, summary


def _rollout_summary(detail: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "n": ("example_index", "size"),
        "closed_rate": ("closed_trace", "mean"),
        "ans_rate": ("emitted_ans", "mean"),
        "eos_rate": ("emitted_eos", "mean"),
        "mean_tokens_until_close": ("tokens_until_close", "mean"),
        "mean_trace_markers_after_patch": ("trace_markers_after_patch", "mean"),
        "mean_first_generated_index": ("first_generated_index", "mean"),
        "mean_stop_index": ("inferred_stop_index", "mean"),
        "mean_final_count": ("final_count", "mean"),
        "receiver_total_hit": ("final_matches_receiver_total", "mean"),
        "donor_total_hit": ("final_matches_donor_total", "mean"),
        "receiver_progress_hit": ("final_matches_receiver_progress", "mean"),
        "donor_progress_hit": ("final_matches_donor_progress", "mean"),
        "receiver_successor_rate": ("first_index_follows_receiver_progress", "mean"),
        "donor_successor_rate": ("first_index_follows_donor_progress", "mean"),
    }
    summary = (
        detail.groupby(
            [
                "scenario_id",
                "scenario_family",
                "intervention",
                "patch_policy",
                "layer",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(**metrics)
        .sort_values(
            ["scenario_family", "scenario_id", "patch_policy", "intervention", "layer"]
        )
    )
    for column in (
        "receiver_count",
        "receiver_progress",
        "donor_count",
        "donor_progress",
        "progress_offset",
        "total_count_offset",
        "uses_gold_trace_tail",
        "forced_close_or_ans",
    ):
        metadata = (
            detail[["scenario_id", column]]
            .drop_duplicates()
            .set_index("scenario_id")[column]
        )
        summary[column] = summary["scenario_id"].map(metadata)
    return summary


def default_rollout_scenarios(count_max: int = 30) -> list[dict[str, Any]]:
    """Balanced total/progress mismatches for the free-rollout diagnostic.

    A scenario is written as ``receiver(total, progress) <- donor(total, progress)``.
    The suite separates scalar total from local trace progress, including cases
    where they move together, independently, or in opposite directions.
    """
    scenarios = [
        ("both_up_small", "both_up", 5, 4, 10, 7),
        ("both_up_mid", "both_up", 10, 7, 20, 14),
        ("both_up_high", "both_up", 20, 14, 30, 24),
        # Keep receiver progress below donor total so the matched-progress
        # donor control exists for every factorial scenario.
        ("both_down_small", "both_down", 10, 4, 5, 2),
        ("both_down_mid", "both_down", 20, 9, 10, 5),
        ("both_down_high", "both_down", 30, 18, 20, 10),
        ("progress_up_only", "progress_only", 10, 4, 10, 7),
        ("progress_down_only", "progress_only", 20, 14, 20, 7),
        ("total_up_only", "total_only", 10, 7, 20, 7),
        ("total_down_only", "total_only", 20, 7, 10, 7),
        ("total_up_progress_down", "opposed", 10, 7, 20, 4),
        ("total_down_progress_up", "opposed", 20, 7, 10, 9),
    ]
    output = []
    for scenario_id, family, rc, rp, dc, dp in scenarios:
        if max(rc, dc) > int(count_max):
            continue
        output.append(
            {
                "scenario_id": scenario_id,
                "scenario_family": family,
                "receiver_count": rc,
                "receiver_progress": rp,
                "donor_count": dc,
                "donor_progress": dp,
            }
        )
    if not output and int(count_max) >= 5:
        output.append(
            {
                "scenario_id": "debug_both_up",
                "scenario_family": "both_up",
                "receiver_count": 3,
                "receiver_progress": 2,
                "donor_count": 5,
                "donor_progress": 4,
            }
        )
    return output


def _rollout_factor_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Separate donor-total and donor-progress effects across mismatch cases."""
    detail = detail.copy()
    if "total_count_offset" not in detail.columns:
        detail["total_count_offset"] = (
            detail["donor_count"] - detail["receiver_count"]
        )
    if "progress_offset" not in detail.columns:
        detail["progress_offset"] = (
            detail["donor_progress"] - detail["receiver_progress"]
        )
    if "closed_trace" not in detail.columns:
        detail["closed_trace"] = math.nan
    scenario_means = (
        detail.groupby(
            [
                "scenario_id",
                "scenario_family",
                "intervention",
                "patch_policy",
                "layer",
                "receiver_count",
                "receiver_progress",
                "donor_count",
                "donor_progress",
                "total_count_offset",
                "progress_offset",
            ],
            as_index=False,
            dropna=False,
        )[
            [
                "first_generated_index",
                "inferred_stop_index",
                "final_count",
                "closed_trace",
            ]
        ]
        .mean()
    )
    scenario_means["first_index_shift"] = (
        scenario_means["first_generated_index"]
        - scenario_means["receiver_progress"]
        - 1
    )
    scenario_means["stop_index_shift"] = (
        scenario_means["inferred_stop_index"] - scenario_means["receiver_count"]
    )
    scenario_means["final_count_shift"] = (
        scenario_means["final_count"] - scenario_means["receiver_count"]
    )
    rows: list[dict[str, Any]] = []
    for keys, frame in scenario_means.groupby(
        ["intervention", "patch_policy", "layer"], dropna=False
    ):
        intervention, policy, layer = keys
        for outcome in (
            "first_index_shift",
            "stop_index_shift",
            "final_count_shift",
        ):
            valid = frame.dropna(subset=[outcome, "total_count_offset", "progress_offset"])
            if len(valid) < 4:
                continue
            design = np.column_stack(
                [
                    np.ones(len(valid)),
                    valid["total_count_offset"].to_numpy(dtype=float),
                    valid["progress_offset"].to_numpy(dtype=float),
                ]
            )
            target = valid[outcome].to_numpy(dtype=float)
            coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
            predicted = design @ coefficients
            residual = float(np.sum((target - predicted) ** 2))
            total = float(np.sum((target - target.mean()) ** 2))
            rows.append(
                {
                    "intervention": intervention,
                    "patch_policy": policy,
                    "layer": int(layer),
                    "outcome": outcome,
                    "n_scenarios": int(len(valid)),
                    "intercept": float(coefficients[0]),
                    "donor_total_coefficient": float(coefficients[1]),
                    "donor_progress_coefficient": float(coefficients[2]),
                    "r2": float(1.0 - residual / total) if total > 1e-12 else math.nan,
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def run_misaligned_trace_rollout_patching(
    model,
    cfg: V10Config,
    vocab: Vocab,
    *,
    receiver_count: int = 5,
    receiver_progress: int = 4,
    donor_count: int = 10,
    donor_progress: int = 7,
    examples: int = 8,
    centroid_examples: int = 12,
    max_new_tokens: int = 32,
    scenarios: Iterable[dict[str, Any]] | None = None,
    scenario_id: str = "single_mismatch",
    scenario_family: str = "single",
    layers: Iterable[int] | None = None,
    patch_policies: Iterable[str] = ("one_shot", "persistent"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transport donor total/progress into a receiver prefix and free-roll out.

    The default target is the concrete counterfactual requested for v10:
    ``donor(total=10, progress=7) -> receiver(total=5, progress=4)``.
    Both prefixes end at a trace marker. No later trace token, close token,
    answer prefix, or final count is supplied to the receiver. Passing
    ``scenarios`` runs a balanced mismatch suite and concatenates the rows.
    """
    if scenarios is not None:
        details: list[pd.DataFrame] = []
        for scenario in scenarios:
            frame, _ = run_misaligned_trace_rollout_patching(
                model,
                cfg,
                vocab,
                receiver_count=int(scenario["receiver_count"]),
                receiver_progress=int(scenario["receiver_progress"]),
                donor_count=int(scenario["donor_count"]),
                donor_progress=int(scenario["donor_progress"]),
                examples=examples,
                centroid_examples=centroid_examples,
                max_new_tokens=max_new_tokens,
                scenario_id=str(scenario["scenario_id"]),
                scenario_family=str(scenario["scenario_family"]),
                layers=layers,
                patch_policies=patch_policies,
            )
            details.append(frame)
        if not details:
            raise ValueError("The rollout scenario suite is empty")
        detail = pd.concat(details, ignore_index=True)
        return detail, _rollout_summary(detail)

    receiver_count = int(receiver_count)
    receiver_progress = int(receiver_progress)
    donor_count = int(donor_count)
    donor_progress = int(donor_progress)
    if not 1 <= receiver_progress < receiver_count <= cfg.count_max:
        raise ValueError("receiver must satisfy 1 <= progress < total <= count_max")
    if not 1 <= donor_progress < donor_count <= cfg.count_max:
        raise ValueError("donor must satisfy 1 <= progress < total <= count_max")
    if receiver_progress >= donor_count:
        raise ValueError(
            "receiver progress must be below donor total so the same-progress donor control exists"
        )

    selected_layers = tuple(
        int(layer) for layer in (layers if layers is not None else range(1, int(cfg.n_layer) + 1))
    )
    if not selected_layers or any(
        layer < 1 or layer > int(cfg.n_layer) for layer in selected_layers
    ):
        raise ValueError(
            f"layers must be non-empty 1-based indices within 1..{int(cfg.n_layer)}"
        )
    selected_policies = tuple(str(policy) for policy in patch_policies)
    if not selected_policies or any(
        policy not in {"one_shot", "persistent"} for policy in selected_policies
    ):
        raise ValueError("patch_policies must contain one_shot and/or persistent")

    centroid_sites = (
        (receiver_count, receiver_progress),
        (donor_count, receiver_progress),
        (donor_count, donor_progress),
    )
    centroids = _fit_trace_progress_centroids(
        model,
        cfg,
        vocab,
        centroid_sites,
        examples_per_site=int(centroid_examples),
        seed=int(cfg.seed) + 1_240_000,
    )
    device = next(model.parameters()).device
    rows: list[dict[str, Any]] = []
    total_rollouts = int(examples) * (
        1 + len(selected_layers) * 7 * len(selected_policies)
    )
    progress = tqdm(total=total_rollouts, desc="v10 hidden patch: later progress free rollout")

    def append_row(
        *,
        example_index: int,
        intervention: str,
        policy: str,
        layer: int,
        suffix: list[int],
    ) -> None:
        parsed = _decode_free_rollout(
            suffix,
            vocab,
            receiver_progress=receiver_progress,
        )
        first_index = parsed["first_generated_index"]
        final_count = parsed["final_count"]
        rows.append(
            {
                "mode": "thinking",
                "site": "misaligned_trace_progress_free_rollout",
                "scenario_id": str(scenario_id),
                "scenario_family": str(scenario_family),
                "example_index": int(example_index),
                "receiver_count": receiver_count,
                "receiver_progress": receiver_progress,
                "donor_count": donor_count,
                "donor_progress": donor_progress,
                "progress_offset": donor_progress - receiver_progress,
                "total_count_offset": donor_count - receiver_count,
                "intervention": intervention,
                "patch_policy": policy,
                "layer": int(layer),
                "uses_gold_trace_tail": False,
                "forced_close_or_ans": False,
                **parsed,
                "first_index_follows_receiver_progress": float(
                    np.isfinite(first_index)
                    and int(first_index) == receiver_progress + 1
                ),
                "first_index_follows_donor_progress": float(
                    np.isfinite(first_index)
                    and int(first_index) == donor_progress + 1
                ),
                "final_matches_receiver_total": float(
                    np.isfinite(final_count) and int(final_count) == receiver_count
                ),
                "final_matches_donor_total": float(
                    np.isfinite(final_count) and int(final_count) == donor_count
                ),
                "final_matches_receiver_progress": float(
                    np.isfinite(final_count) and int(final_count) == receiver_progress
                ),
                "final_matches_donor_progress": float(
                    np.isfinite(final_count) and int(final_count) == donor_progress
                ),
            }
        )
        progress.update(1)

    for example_index in range(int(examples)):
        receiver_rng = random.Random(int(cfg.seed) + 1_250_000 + example_index)
        donor_rng = random.Random(int(cfg.seed) + 1_260_000 + example_index)
        semantic_control_rng = random.Random(int(cfg.seed) + 1_270_000 + example_index)
        receiver = make_example(
            cfg, vocab, receiver_rng, count=receiver_count, seed=example_index
        )
        donor = make_example(
            cfg, vocab, donor_rng, count=donor_count, seed=example_index
        )
        semantic_control = make_example(
            cfg,
            vocab,
            semantic_control_rng,
            count=receiver_count,
            seed=example_index,
        )
        receiver_prefix, receiver_position = _thinking_prefix_ids(
            receiver, vocab, receiver_progress
        )
        donor_early_prefix, donor_early_position = _thinking_prefix_ids(
            donor, vocab, receiver_progress
        )
        donor_late_prefix, donor_late_position = _thinking_prefix_ids(
            donor, vocab, donor_progress
        )
        control_prefix, control_position = _thinking_prefix_ids(
            semantic_control, vocab, receiver_progress
        )
        receiver_ids = torch.tensor([receiver_prefix], dtype=torch.long, device=device)
        donor_early_ids = torch.tensor(
            [donor_early_prefix], dtype=torch.long, device=device
        )
        donor_late_ids = torch.tensor(
            [donor_late_prefix], dtype=torch.long, device=device
        )
        control_ids = torch.tensor([control_prefix], dtype=torch.long, device=device)
        _, receiver_states = _post_block_states(
            model, receiver_ids, receiver_position
        )
        _, donor_early_states = _post_block_states(
            model, donor_early_ids, donor_early_position
        )
        _, donor_late_states = _post_block_states(
            model, donor_late_ids, donor_late_position
        )
        _, semantic_control_states = _post_block_states(
            model, control_ids, control_position
        )

        baseline_suffix = _free_rollout_with_residual_patch(
            model,
            receiver_prefix,
            vocab,
            max_new_tokens=max_new_tokens,
        )
        append_row(
            example_index=example_index,
            intervention="baseline_no_patch",
            policy="none",
            layer=0,
            suffix=baseline_suffix,
        )

        for layer_number in selected_layers:
            layer_index = layer_number - 1
            receiver_state = receiver_states[layer_index]
            centroid_device = {
                "device": receiver_state.device,
                "dtype": receiver_state.dtype,
            }
            mu_receiver_early = centroids[(receiver_count, receiver_progress)][
                layer_index
            ].to(**centroid_device)
            mu_donor_early = centroids[(donor_count, receiver_progress)][layer_index].to(
                **centroid_device
            )
            mu_donor_late = centroids[(donor_count, donor_progress)][layer_index].to(
                **centroid_device
            )
            replacements = {
                "self_state_control": receiver_state,
                "same_semantics_cross_prompt_control": semantic_control_states[layer_index],
                "same_progress_donor_total_full": donor_early_states[layer_index],
                "later_progress_donor_full": donor_late_states[layer_index],
                "total_only_centroid_delta": (
                    receiver_state + mu_donor_early - mu_receiver_early
                ),
                "progress_only_centroid_delta": (
                    receiver_state + mu_donor_late - mu_donor_early
                ),
                "combined_centroid_delta": (
                    receiver_state + mu_donor_late - mu_receiver_early
                ),
            }
            for intervention, replacement in replacements.items():
                for policy in selected_policies:
                    suffix = _free_rollout_with_residual_patch(
                        model,
                        receiver_prefix,
                        vocab,
                        layer=layer_index,
                        patch_position=receiver_position,
                        replacement_state=replacement,
                        patch_policy=policy,
                        max_new_tokens=max_new_tokens,
                    )
                    append_row(
                        example_index=example_index,
                        intervention=intervention,
                        policy=policy,
                        layer=layer_number,
                        suffix=suffix,
                    )
    progress.close()
    detail = pd.DataFrame(rows)
    return detail, _rollout_summary(detail)


def _plot_layer_lines(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    y_column: str,
    ylabel: str,
    hue: str | None = None,
) -> None:
    bins = [value for value in ("1-10", "11-20", "21-30") if value in set(summary.count_bin)]
    fig, axes = plt.subplots(1, len(bins), figsize=(5.0 * len(bins), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, bin_name in zip(axes, bins):
        part = summary[summary.count_bin == bin_name]
        if hue and hue in part:
            for label, group in part.groupby(hue, sort=False):
                ax.plot(group.layer, group[y_column], marker="o", linewidth=2, label=str(label))
        else:
            ax.plot(part.layer, part[y_column], marker="o", linewidth=2, color="#2563eb")
        ax.axhline(0.0, color="#64748b", linestyle="--", linewidth=1)
        ax.set_title(f"receiver count {bin_name}")
        ax.set_xlabel("patched post-Layer residual")
        ax.set_xticks(sorted(part.layer.unique()))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    if hue:
        axes[-1].legend(title=hue, frameon=False)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_rollout_heatmaps(
    summary: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    labels = {
        "self_state_control": "self control",
        "same_semantics_cross_prompt_control": "same semantics control",
        "same_progress_donor_total_full": "raw donor: total only",
        "later_progress_donor_full": "raw donor: total + progress",
        "total_only_centroid_delta": "centroid delta: total only",
        "progress_only_centroid_delta": "centroid delta: progress only",
        "combined_centroid_delta": "centroid delta: combined",
    }
    order = list(labels)
    frame = summary[
        (summary.layer == 4)
        & (summary.patch_policy == "one_shot")
        & summary.intervention.isin(order)
    ].copy()
    scenario_order = list(dict.fromkeys(summary["scenario_id"].astype(str)))
    scenario_meta = (
        summary.drop_duplicates("scenario_id")
        .set_index("scenario_id")
        [["receiver_count", "receiver_progress", "donor_count", "donor_progress"]]
    )
    scenario_labels = {
        scenario: (
            f"R{int(row.receiver_count)}@I{int(row.receiver_progress)} <- "
            f"D{int(row.donor_count)}@I{int(row.donor_progress)}"
        )
        for scenario, row in scenario_meta.iterrows()
    }
    paths: list[Path] = []
    for metric, title, name in (
        (
            "mean_first_generated_index",
            "First freely generated trace index across total/progress mismatches",
            "misaligned_rollout_first_index.png",
        ),
        (
            "mean_stop_index",
            "Where does the freely generated trace stop?",
            "misaligned_rollout_stop_index.png",
        ),
        (
            "mean_final_count",
            "Which scalar count is emitted after each mismatch patch?",
            "misaligned_rollout_final_count.png",
        ),
    ):
        pivot = frame.pivot(index="scenario_id", columns="intervention", values=metric)
        pivot = pivot.reindex(index=scenario_order, columns=order).rename(columns=labels)
        values = pivot.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        vmin = float(np.min(finite)) if len(finite) else 0.0
        vmax = float(np.max(finite)) if len(finite) else 1.0
        fig, ax = plt.subplots(figsize=(12.6, 8.2))
        image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Layer-4 one-shot intervention")
        ax.set_ylabel("receiver prefix <- donor state")
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=28, ha="right")
        ax.set_yticks(
            range(len(pivot.index)),
            [scenario_labels.get(str(value), str(value)) for value in pivot.index],
        )
        middle = (vmin + vmax) / 2
        for row in range(len(pivot.index)):
            for column in range(len(pivot.columns)):
                value = values[row, column]
                if np.isfinite(value):
                    color = "white" if value < middle else "black"
                    ax.text(column, row, f"{value:.1f}", ha="center", va="center", color=color)
        fig.subplots_adjust(left=0.24, right=0.91, bottom=0.25, top=0.88)
        colorbar_axis = fig.add_axes([0.93, 0.24, 0.018, 0.58])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(metric.replace("mean_", "mean ").replace("_", " "))
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.text(
            0.5,
            0.025,
            (
                "R = receiver prompt total/progress; D = donor prompt total/progress. "
                "No gold trace tail or forced close/answer is supplied."
            ),
            ha="center",
            fontsize=9,
        )
        path = output_dir / name
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_rollout_factor_coefficients(
    factor_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    labels = {
        "later_progress_donor_full": "raw donor: total + progress",
        "total_only_centroid_delta": "centroid delta: total only",
        "progress_only_centroid_delta": "centroid delta: progress only",
        "combined_centroid_delta": "centroid delta: combined",
    }
    frame = factor_summary[
        (factor_summary.patch_policy == "one_shot")
        & (factor_summary.layer == 4)
        & factor_summary.intervention.isin(labels)
    ].copy()
    outcomes = [
        ("first_index_shift", "first generated index shift"),
        ("stop_index_shift", "trace stop-index shift"),
        ("final_count_shift", "final scalar-count shift"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.6), sharey=True)
    for ax, (outcome, title) in zip(axes, outcomes):
        part = frame[frame.outcome == outcome].set_index("intervention").reindex(labels)
        y = np.arange(len(part))
        ax.barh(y - 0.18, part.donor_total_coefficient, height=0.34, label="donor total offset")
        ax.barh(y + 0.18, part.donor_progress_coefficient, height=0.34, label="donor progress offset")
        ax.axvline(0.0, color="#111827", linewidth=1)
        ax.axvline(1.0, color="#16a34a", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("partial OLS coefficient")
        ax.set_yticks(y, [labels.get(str(value), str(value)) for value in part.index])
        ax.grid(axis="x", alpha=0.2)
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "What does a trace-marker residual transport: total count or local progress?",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_donor_gt_receiver_transport(
    final_directional: pd.DataFrame,
    trace_directional: pd.DataFrame,
    output_path: Path,
) -> None:
    """Compare upward scalar transport at final-answer and final-marker sites."""
    bins = ("1-10", "11-20", "21-30")
    rows = (
        ("nonthinking", "final <Ans> residual", final_directional),
        ("thinking", "final <Ans> residual", final_directional),
        ("thinking", "final trace-marker residual", trace_directional),
    )
    fig, axes = plt.subplots(
        len(rows), len(bins), figsize=(14.5, 10.0), sharex=True, sharey=True
    )
    for row_index, (mode, site_label, frame) in enumerate(rows):
        selected = frame[frame.patch_direction == "donor_gt_receiver"].copy()
        if "mode" in selected.columns:
            selected = selected[selected["mode"] == mode]
        for column_index, bin_name in enumerate(bins):
            ax = axes[row_index, column_index]
            part = selected[selected.count_bin == bin_name].sort_values("layer")
            if not part.empty:
                ax.plot(
                    part.layer,
                    part.transport_slope,
                    marker="o",
                    linewidth=2.2,
                    color="#2563eb" if "<Ans>" in site_label else "#f97316",
                )
            ax.axhline(1.0, color="#16a34a", linestyle="--", linewidth=1.2)
            ax.axhline(0.0, color="#64748b", linestyle=":", linewidth=1.0)
            ax.set_title(f"receiver count {bin_name}")
            ax.set_xticks([1, 2, 3, 4])
            ax.grid(alpha=0.22)
            if row_index == len(rows) - 1:
                ax.set_xlabel("patched post-Layer residual")
            if column_index == 0:
                ax.set_ylabel(
                    f"{mode}: {site_label}\nupward transport slope"
                )
    fig.suptitle(
        "Donor count > receiver count: does the patched state transport the larger scalar?",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        (
            "Slope is fit only on donor_count > receiver_count pairs. "
            "1 = one-for-one transport of the larger donor count; 0 = no scalar transport."
        ),
        ha="center",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_hidden_state_patching_plots(
    output_dir: Path,
    final_summary: pd.DataFrame,
    trace_summary: pd.DataFrame,
    early_summary: pd.DataFrame,
    rollout_summary: pd.DataFrame | None = None,
    rollout_factor_summary: pd.DataFrame | None = None,
    final_directional: pd.DataFrame | None = None,
    trace_directional: pd.DataFrame | None = None,
) -> list[Path]:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    specs = [
        (
            final_summary,
            "final_answer_transport_by_layer.png",
            "Cross-count residual transplant at the final <Ans> query",
            "transport_slope",
            "expected-count transport slope",
            "mode",
        ),
        (
            trace_summary,
            "trace_final_transport_by_layer.png",
            "CoT final-marker m-to-n residual transplant",
            "transport_slope",
            "final-answer expected-count transport slope",
            None,
        ),
        (
            early_summary,
            "trace_early_stop_close_margin.png",
            "Early-stop test: donor final-marker state at receiver interior marker",
            "mean_close_margin_shift",
            "change in logit(</Think>) - logit(<m+1>)",
            None,
        ),
        (
            early_summary,
            "trace_early_stop_count_transport.png",
            "Forced-close readout without a gold trace tail",
            "transport_slope",
            "expected-count transport slope",
            None,
        ),
    ]
    for frame, name, title, column, ylabel, hue in specs:
        path = figures / name
        _plot_layer_lines(
            frame,
            path,
            title=title,
            y_column=column,
            ylabel=ylabel,
            hue=hue,
        )
        paths.append(path)
    if rollout_summary is not None and not rollout_summary.empty:
        paths.extend(
            _plot_rollout_heatmaps(
                rollout_summary,
                figures,
            )
        )
    if rollout_factor_summary is not None and not rollout_factor_summary.empty:
        path = figures / "misaligned_rollout_factor_coefficients.png"
        _plot_rollout_factor_coefficients(rollout_factor_summary, path)
        paths.append(path)
    if (
        final_directional is not None
        and not final_directional.empty
        and trace_directional is not None
        and not trace_directional.empty
    ):
        path = figures / "donor_gt_receiver_scalar_transport.png"
        _plot_donor_gt_receiver_transport(
            final_directional,
            trace_directional,
            path,
        )
        paths.append(path)
    return paths


def run_hidden_state_patching(
    run_dir: str | Path,
    *,
    device: str | None = None,
    examples_per_pair: int | None = None,
    rollout_examples: int | None = None,
    rollout_centroid_examples: int | None = None,
    rollout_max_new_tokens: int = 32,
    rollout_receiver_count: int = 5,
    rollout_receiver_progress: int = 4,
    rollout_donor_count: int = 10,
    rollout_donor_progress: int = 7,
    rollout_scenarios: Iterable[dict[str, Any]] | None = None,
    rollout_layers: Iterable[int] | None = None,
    rollout_patch_policies: Iterable[str] = ("one_shot",),
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    cfg, vocab = load_run(run_dir, device=device)
    examples_per_pair = int(
        examples_per_pair
        if examples_per_pair is not None
        else (1 if cfg.preset == "debug" else 2)
    )
    rollout_examples = int(
        rollout_examples
        if rollout_examples is not None
        else (1 if cfg.preset == "debug" else 8)
    )
    rollout_centroid_examples = int(
        rollout_centroid_examples
        if rollout_centroid_examples is not None
        else (2 if cfg.preset == "debug" else 12)
    )
    rollout_scenarios = list(
        rollout_scenarios
        if rollout_scenarios is not None
        else default_rollout_scenarios(cfg.count_max)
    )
    if not rollout_scenarios:
        rollout_scenarios = [
            {
                "scenario_id": "legacy_single_case",
                "scenario_family": "legacy_single_case",
                "receiver_count": int(rollout_receiver_count),
                "receiver_progress": int(rollout_receiver_progress),
                "donor_count": int(rollout_donor_count),
                "donor_progress": int(rollout_donor_progress),
            }
        ]
    rollout_layers = tuple(
        int(layer)
        for layer in (
            rollout_layers if rollout_layers is not None else (int(cfg.n_layer),)
        )
    )
    rollout_patch_policies = tuple(str(value) for value in rollout_patch_policies)
    output_dir = run_dir / "analysis" / "hidden_state_patching"
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    legacy_expected = {
        "final_detail": tables / "final_answer_patching.csv",
        "final_summary": tables / "final_answer_patching_summary.csv",
        "trace_detail": tables / "trace_final_patching.csv",
        "trace_summary": tables / "trace_final_patching_summary.csv",
        "early_detail": tables / "trace_early_stop_patching.csv",
        "early_summary": tables / "trace_early_stop_patching_summary.csv",
    }
    rollout_expected = {
        "rollout_detail": tables / "misaligned_trace_rollout.csv",
        "rollout_summary": tables / "misaligned_trace_rollout_summary.csv",
        "rollout_factor_summary": (
            tables / "misaligned_trace_rollout_factor_summary.csv"
        ),
    }
    directional_expected = {
        "final_directional_summary": (
            tables / "final_answer_patching_directional_summary.csv"
        ),
        "trace_directional_summary": (
            tables / "trace_final_patching_directional_summary.csv"
        ),
    }
    legacy_ready = not overwrite and all(
        path.exists() and path.stat().st_size > 0
        for path in legacy_expected.values()
    )
    rollout_ready = not overwrite and all(
        path.exists() and path.stat().st_size > 0
        for path in rollout_expected.values()
    )
    expected_scenario_ids = {
        str(scenario["scenario_id"]) for scenario in rollout_scenarios
    }
    if rollout_ready:
        try:
            cached_rollout = pd.read_csv(rollout_expected["rollout_detail"])
            cached_ids = set(cached_rollout["scenario_id"].astype(str))
            cached_layers = set(
                cached_rollout.loc[cached_rollout["layer"] > 0, "layer"].astype(int)
            )
            cached_policies = set(cached_rollout["patch_policy"].astype(str)) - {
                "none"
            }
            rollout_ready = (
                cached_ids == expected_scenario_ids
                and cached_layers == set(rollout_layers)
                and cached_policies == set(rollout_patch_policies)
            )
        except (KeyError, pd.errors.EmptyDataError):
            rollout_ready = False
    outputs: dict[str, pd.DataFrame] = {}
    models: dict[str, Any] = {}
    if legacy_ready:
        outputs.update(
            {name: pd.read_csv(path) for name, path in legacy_expected.items()}
        )
    else:
        models = {
            mode: load_final_model(cfg, vocab, run_dir, mode)
            for mode in cfg.modes
        }
        final_detail, final_summary = run_final_answer_patching(
            models,
            cfg,
            vocab,
            examples_per_pair=examples_per_pair,
        )
        trace_detail, trace_summary = run_trace_final_patching(
            models["thinking"],
            cfg,
            vocab,
            examples_per_pair=examples_per_pair,
        )
        early_detail, early_summary = run_trace_early_stop_patching(
            models["thinking"],
            cfg,
            vocab,
            examples_per_pair=examples_per_pair,
        )
        outputs.update(
            {
                "final_detail": final_detail,
                "final_summary": final_summary,
                "trace_detail": trace_detail,
                "trace_summary": trace_summary,
                "early_detail": early_detail,
                "early_summary": early_summary,
            }
        )
        for name, path in legacy_expected.items():
            outputs[name].to_csv(path, index=False)

    # Directional summaries are cheap derived artifacts. Rebuild them from the
    # cached detail tables so an existing expensive run never has to retrain or
    # rerun forward passes merely to separate upward from downward transport.
    outputs["final_detail"] = _with_patch_direction(outputs["final_detail"])
    outputs["trace_detail"] = _with_patch_direction(outputs["trace_detail"])
    outputs["final_directional_summary"] = _directional_regression_summary(
        outputs["final_detail"],
        ["mode", "site", "count_bin", "layer"],
    )
    outputs["trace_directional_summary"] = _directional_regression_summary(
        outputs["trace_detail"],
        ["site", "count_bin", "layer"],
    )
    outputs["final_detail"].to_csv(legacy_expected["final_detail"], index=False)
    outputs["trace_detail"].to_csv(legacy_expected["trace_detail"], index=False)
    for name, path in directional_expected.items():
        outputs[name].to_csv(path, index=False)

    if rollout_ready:
        outputs.update(
            {name: pd.read_csv(path) for name, path in rollout_expected.items()}
        )
    else:
        if "thinking" not in models:
            models["thinking"] = load_final_model(
                cfg, vocab, run_dir, "thinking"
            )
        rollout_detail, rollout_summary = run_misaligned_trace_rollout_patching(
            models["thinking"],
            cfg,
            vocab,
            receiver_count=rollout_receiver_count,
            receiver_progress=rollout_receiver_progress,
            donor_count=rollout_donor_count,
            donor_progress=rollout_donor_progress,
            examples=rollout_examples,
            centroid_examples=rollout_centroid_examples,
            max_new_tokens=rollout_max_new_tokens,
            scenarios=rollout_scenarios,
            layers=rollout_layers,
            patch_policies=rollout_patch_policies,
        )
        outputs["rollout_detail"] = rollout_detail
        outputs["rollout_summary"] = rollout_summary
        outputs["rollout_factor_summary"] = _rollout_factor_summary(
            rollout_detail
        )
        for name, path in rollout_expected.items():
            outputs[name].to_csv(path, index=False)
    manifest = {
        "experiment": "v10_hidden_state_patching",
        "representation_space": "post-transformer-layer residual captured and patched at the same block-output hook",
        "examples_per_pair": examples_per_pair,
        "free_rollout_protocol": {
            "scenario_count": len(rollout_scenarios),
            "scenarios": rollout_scenarios,
            "examples": int(rollout_examples),
            "centroid_examples_per_site": int(rollout_centroid_examples),
            "max_new_tokens": int(rollout_max_new_tokens),
            "layers": list(rollout_layers),
            "patch_policies": list(rollout_patch_policies),
            "uses_gold_trace_tail": False,
            "forced_close_or_ans": False,
        },
        "final_answer_modes": list(cfg.modes),
        "patch_direction_definition": {
            "donor_gt_receiver": "donor_count > receiver_count",
            "donor_lt_receiver": "donor_count < receiver_count",
            "same_count": "donor_count == receiver_count",
            "directional_regression_note": (
                "Upward and downward transport slopes are fit separately; "
                "the donor_gt_receiver result is not averaged with reverse patches."
            ),
        },
        "trace_tests": [
            "final_marker_m_to_final_marker_n",
            "prefix_M_m_early_close_teacher_forced",
            "prefix_M_m_forced_close_readout_without_gold_trace_tail",
            "systematic_total_progress_mismatch_free_rollout_suite",
        ],
        "teacher_forcing_note": (
            "The early-close local metric uses the full receiver sequence only to score the token "
            "immediately after M_m. Its downstream readout uses a separate truncated input that "
            "contains forced </Think><Ans> but no later gold trace or answer token."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_hidden_state_patching_plots(
        output_dir,
        outputs["final_summary"],
        outputs["trace_summary"],
        outputs["early_summary"],
        outputs["rollout_summary"],
        outputs["rollout_factor_summary"],
        outputs["final_directional_summary"],
        outputs["trace_directional_summary"],
    )
    return outputs
