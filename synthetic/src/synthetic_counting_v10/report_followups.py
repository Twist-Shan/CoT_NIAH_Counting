from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .attention_causal import (
    Head,
    _capture_cproj_inputs,
    _head_label,
    _head_mask,
    _patched_head_forward,
    marker_identity_corruption,
    normalized_recovery,
)
from .config import V10Config, config_from_dict
from .core import (
    Example,
    Vocab,
    balanced_examples,
    collate,
    count_bin,
    count_prediction,
    margin,
    render,
)
from .state_causal import (
    SITE_NONTHINKING_FINAL,
    SITE_THINKING_FINAL,
    SITE_THINKING_FIXED,
    _forward_residual_patch,
    render_site,
)
from .training import load_final_model


def load_run(run_dir: str | Path, device: str | None = None) -> tuple[V10Config, Vocab]:
    run_dir = Path(run_dir)
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    chosen = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = replace(cfg, device=chosen)
    return cfg, Vocab.load(run_dir / "vocab.json")


def load_rankings(run_dir: str | Path) -> dict[str, list[Head]]:
    obj = json.loads(
        (Path(run_dir) / "analysis" / "attention_causal" / "head_rankings.json").read_text(
            encoding="utf-8"
        )
    )
    return {name: [(int(layer), int(head)) for layer, head in heads] for name, heads in obj.items()}


def _all_heads(cfg: V10Config) -> list[Head]:
    return [(layer, head) for layer in range(cfg.n_layer) for head in range(cfg.n_head)]


def _count_margin(logits: torch.Tensor, vocab: Vocab, count: int) -> float:
    return margin(logits, vocab.number_id(count), vocab.number_ids)


def _local_query_positions(item, site: str) -> list[int]:
    if site == "ans_token":
        return [int(item.spans.ans_pos)]
    if site == "trace_index_tokens":
        return [int(position) for position in item.spans.trace_index_positions]
    raise ValueError(f"Unknown position-local ablation site: {site}")


def _forward_with_position_local_head_ablation(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    heads: Iterable[Head],
    positions_by_row: list[list[int]],
):
    """Zero selected pre-c_proj head slices only at the requested query rows."""
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(int(layer), []).append(int(head))
    head_dim = int(model.config.n_embd) // int(model.config.n_head)
    handles = []
    for layer, layer_heads in by_layer.items():
        def hook(_module, args, layer_heads=tuple(layer_heads)):
            value = args[0].clone()
            for row_idx, positions in enumerate(positions_by_row):
                valid = [int(position) for position in positions if 0 <= int(position) < value.shape[1]]
                if not valid:
                    continue
                for head in layer_heads:
                    start = int(head) * head_dim
                    value[row_idx, valid, start : start + head_dim] = 0.0
            return (value, *args[1:])

        handles.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(hook))
    try:
        return model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def batched_teacher_forced_metrics(
    model,
    vocab: Vocab,
    examples: list[Example],
    mode: str,
    device: str | torch.device,
    *,
    head_mask: torch.Tensor | None = None,
    local_ablation_heads: Iterable[Head] | None = None,
    local_ablation_site: str | None = None,
    batch_size: int = 24,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(examples), int(batch_size)):
        batch = examples[start : start + int(batch_size)]
        rendered = [render(example, vocab, mode) for example in batch]
        ids, _, attention_mask = collate(rendered, vocab, device)
        local_heads = list(local_ablation_heads or [])
        if local_heads:
            if head_mask is not None:
                raise ValueError("Global head_mask and position-local ablation cannot be combined.")
            if local_ablation_site is None:
                raise ValueError("local_ablation_site is required when local_ablation_heads is set.")
            positions_by_row = [
                _local_query_positions(item, local_ablation_site) for item in rendered
            ]
            logits = _forward_with_position_local_head_ablation(
                model,
                ids,
                attention_mask,
                local_heads,
                positions_by_row,
            ).logits
        else:
            logits = model(input_ids=ids, attention_mask=attention_mask, head_mask=head_mask).logits
        for row_idx, (example, item) in enumerate(zip(batch, rendered)):
            count_logits = logits[row_idx, item.spans.ans_pos]
            pred, expected, _ = count_prediction(count_logits, vocab)
            marker_correct = 0.0
            marker_margin_sum = 0.0
            marker_n = 0
            index_correct = 0.0
            index_n = 0
            if mode == "thinking":
                for k, (index_pos, marker_pos) in enumerate(
                    zip(item.spans.trace_index_positions, item.spans.trace_marker_positions), start=1
                ):
                    marker_logits = logits[row_idx, index_pos]
                    target_marker = item.input_ids[marker_pos]
                    marker_correct += float(int(marker_logits.argmax().item()) == target_marker)
                    marker_margin_sum += margin(marker_logits, target_marker, vocab.marker_ids)
                    marker_n += 1
                    previous_pos = item.spans.think_pos if k == 1 else item.spans.trace_marker_positions[k - 2]
                    target_index = vocab.number_id(k)
                    index_correct += float(int(logits[row_idx, previous_pos].argmax().item()) == target_index)
                    index_n += 1
            rows.append(
                {
                    "example_idx": start + row_idx,
                    "mode": mode,
                    "count": example.count,
                    "count_bin": count_bin(example.count),
                    "final_count_correct": float(pred == example.count),
                    "final_count_margin": _count_margin(count_logits, vocab, example.count),
                    "final_count_expected": expected,
                    "trace_marker_correct_sum": marker_correct,
                    "trace_marker_margin_sum": marker_margin_sum,
                    "trace_marker_n": marker_n,
                    "trace_index_correct_sum": index_correct,
                    "trace_index_n": index_n,
                }
            )
    return pd.DataFrame(rows)


def aggregate_metrics(detail: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = [("all", detail)] + list(detail.groupby("count_bin", sort=False))
    for label, frame in groups:
        marker_n = float(frame["trace_marker_n"].sum())
        index_n = float(frame["trace_index_n"].sum())
        rows.append(
            {
                "count_bin": label,
                "n_examples": len(frame),
                "final_count_accuracy": float(frame["final_count_correct"].mean()),
                "final_count_margin": float(frame["final_count_margin"].mean()),
                "final_count_expected_mae": float(
                    np.mean(np.abs(frame["final_count_expected"] - frame["count"]))
                ),
                "trace_marker_accuracy": (
                    float(frame["trace_marker_correct_sum"].sum() / marker_n) if marker_n else math.nan
                ),
                "trace_marker_margin": (
                    float(frame["trace_marker_margin_sum"].sum() / marker_n) if marker_n else math.nan
                ),
                "trace_index_accuracy": (
                    float(frame["trace_index_correct_sum"].sum() / index_n) if index_n else math.nan
                ),
            }
        )
    return rows


def _append_drop_columns(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame[frame["family"] == "baseline"].set_index(["mode", "count_bin"])
    for metric in (
        "final_count_accuracy",
        "final_count_margin",
        "trace_marker_accuracy",
        "trace_marker_margin",
        "trace_index_accuracy",
    ):
        frame[f"drop_{metric}"] = frame.apply(
            lambda row: (
                float(baseline.loc[(row["mode"], row["count_bin"]), metric] - row[metric])
                if np.isfinite(row[metric])
                and np.isfinite(baseline.loc[(row["mode"], row["count_bin"]), metric])
                else math.nan
            ),
            axis=1,
        )
    return frame


@torch.no_grad()
def run_strict_ablation_suite(
    models: dict[str, Any],
    cfg: V10Config,
    vocab: Vocab,
    rankings: dict[str, list[Head]],
    *,
    examples_per_count: int = 3,
    random_replicates: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    examples = balanced_examples(cfg, vocab, examples_per_count, cfg.seed + 910_000)
    all_heads = _all_heads(cfg)
    rows: list[dict[str, Any]] = []
    single_rows: list[dict[str, Any]] = []
    baseline_by_mode: dict[str, pd.DataFrame] = {}

    for mode in cfg.modes:
        model = models[mode]
        baseline_detail = batched_teacher_forced_metrics(
            model, vocab, examples, mode, cfg.device, batch_size=cfg.analysis_batch_size
        )
        baseline_by_mode[mode] = baseline_detail
        for metric_row in aggregate_metrics(baseline_detail):
            rows.append(
                {
                    "mode": mode,
                    "family": "baseline",
                    "replicate": 0,
                    "top_n": 0,
                    "masked_heads": "",
                    **metric_row,
                }
            )

        for layer, head in tqdm(all_heads, desc=f"v10 single-head ablation: {mode}"):
            masked = batched_teacher_forced_metrics(
                model,
                vocab,
                examples,
                mode,
                cfg.device,
                head_mask=_head_mask(model, [(layer, head)], cfg.device),
                batch_size=cfg.analysis_batch_size,
            )
            for metric_row in aggregate_metrics(masked):
                single_rows.append(
                    {
                        "mode": mode,
                        "layer": layer,
                        "head": head,
                        "masked_head": _head_label([(layer, head)]),
                        **metric_row,
                    }
                )

        if mode == "nonthinking":
            named = {"direct_broad_top": rankings["direct_broad"]}
            primary = rankings["direct_broad"]
        else:
            named = {
                "targeted_retrieval_top": rankings["targeted_retrieval"],
                "trace_readout_top": rankings["trace_readout"],
                "successor_top": rankings["successor"],
            }
            primary = rankings["targeted_retrieval"]
        named["primary_bottom"] = list(reversed(primary))
        for replicate in range(int(random_replicates)):
            shuffled = list(all_heads)
            random.Random(cfg.seed + 911_000 + replicate).shuffle(shuffled)
            named[f"random_{replicate}"] = shuffled

        total = len(named) * len(all_heads)
        progress = tqdm(total=total, desc=f"v10 cumulative ablation: {mode}")
        for family, ranking in named.items():
            replicate = int(family.split("_")[-1]) if family.startswith("random_") else 0
            family_label = "random" if family.startswith("random_") else family
            for top_n in range(1, len(ranking) + 1):
                heads = ranking[:top_n]
                masked = batched_teacher_forced_metrics(
                    model,
                    vocab,
                    examples,
                    mode,
                    cfg.device,
                    head_mask=_head_mask(model, heads, cfg.device),
                    batch_size=cfg.analysis_batch_size,
                )
                for metric_row in aggregate_metrics(masked):
                    rows.append(
                        {
                            "mode": mode,
                            "family": family_label,
                            "replicate": replicate,
                            "top_n": top_n,
                            "masked_heads": _head_label(heads),
                            **metric_row,
                        }
                    )
                progress.update(1)
        progress.close()

    cumulative = _append_drop_columns(pd.DataFrame(rows))
    single = pd.DataFrame(single_rows)
    baseline = cumulative[cumulative["family"] == "baseline"].set_index(["mode", "count_bin"])
    for metric in (
        "final_count_accuracy",
        "final_count_margin",
        "trace_marker_accuracy",
        "trace_marker_margin",
        "trace_index_accuracy",
    ):
        single[f"drop_{metric}"] = single.apply(
            lambda row: (
                float(baseline.loc[(row["mode"], row["count_bin"]), metric] - row[metric])
                if np.isfinite(row[metric])
                and np.isfinite(baseline.loc[(row["mode"], row["count_bin"]), metric])
                else math.nan
            ),
            axis=1,
        )
    return single, cumulative


def _layer_matched_order(
    primary: list[Head],
    *,
    seed: int | None = None,
    reverse_within_layer: bool = False,
) -> list[Head]:
    """Match the primary ranking's layer sequence while changing heads within each layer."""
    pools: dict[int, list[Head]] = {}
    for layer, _head in primary:
        pools.setdefault(int(layer), [])
    for layer in pools:
        layer_heads = [head for head in primary if int(head[0]) == layer]
        if reverse_within_layer:
            layer_heads = list(reversed(layer_heads))
        elif seed is not None:
            random.Random(int(seed) + layer).shuffle(layer_heads)
        pools[layer] = layer_heads
    offsets = {layer: 0 for layer in pools}
    result: list[Head] = []
    for layer, _head in primary:
        index = offsets[int(layer)]
        result.append(pools[int(layer)][index])
        offsets[int(layer)] = index + 1
    return result


@torch.no_grad()
def run_position_local_ablation_suite(
    models: dict[str, Any],
    cfg: V10Config,
    vocab: Vocab,
    rankings: dict[str, list[Head]],
    *,
    examples_per_count: int = 8,
    random_replicates: int = 8,
) -> pd.DataFrame:
    """Ablate head outputs only at the semantic query used by each mechanism."""
    examples = balanced_examples(cfg, vocab, examples_per_count, cfg.seed + 930_000)
    specifications = (
        ("nonthinking", "direct_broad", "ans_token"),
        ("thinking", "targeted_retrieval", "trace_index_tokens"),
        ("thinking", "trace_readout", "ans_token"),
    )
    rows: list[dict[str, Any]] = []
    for mode, mechanism, site in specifications:
        model = models[mode]
        primary = list(rankings[mechanism])
        families: dict[str, tuple[int, list[Head]]] = {
            "ranked_top": (0, primary),
            "global_reverse": (0, list(reversed(primary))),
            "layer_matched_low": (
                0,
                _layer_matched_order(primary, reverse_within_layer=True),
            ),
        }
        for replicate in range(int(random_replicates)):
            families[f"layer_matched_random_{replicate}"] = (
                replicate,
                _layer_matched_order(
                    primary,
                    seed=cfg.seed + 931_000 + 101 * replicate,
                ),
            )

        baseline = batched_teacher_forced_metrics(
            model,
            vocab,
            examples,
            mode,
            cfg.device,
            batch_size=cfg.analysis_batch_size,
        )
        for metric_row in aggregate_metrics(baseline):
            rows.append(
                {
                    "mode": mode,
                    "mechanism": mechanism,
                    "site": site,
                    "family": "baseline",
                    "replicate": 0,
                    "top_n": 0,
                    "masked_heads": "",
                    **metric_row,
                }
            )

        total = len(families) * len(primary)
        progress = tqdm(total=total, desc=f"v10 position-local ablation: {mechanism}")
        for family, (replicate, ranking) in families.items():
            family_label = "layer_matched_random" if family.startswith("layer_matched_random_") else family
            for top_n in range(1, len(ranking) + 1):
                heads = ranking[:top_n]
                detail = batched_teacher_forced_metrics(
                    model,
                    vocab,
                    examples,
                    mode,
                    cfg.device,
                    local_ablation_heads=heads,
                    local_ablation_site=site,
                    batch_size=cfg.analysis_batch_size,
                )
                for metric_row in aggregate_metrics(detail):
                    rows.append(
                        {
                            "mode": mode,
                            "mechanism": mechanism,
                            "site": site,
                            "family": family_label,
                            "replicate": replicate,
                            "top_n": top_n,
                            "masked_heads": _head_label(heads),
                            **metric_row,
                        }
                    )
                progress.update(1)
        progress.close()

    frame = pd.DataFrame(rows)
    baseline = frame[frame["family"] == "baseline"].set_index(
        ["mode", "mechanism", "site", "count_bin"]
    )
    for metric in (
        "final_count_accuracy",
        "final_count_margin",
        "trace_marker_accuracy",
        "trace_marker_margin",
        "trace_index_accuracy",
    ):
        frame[f"drop_{metric}"] = frame.apply(
            lambda row: (
                float(
                    baseline.loc[
                        (row["mode"], row["mechanism"], row["site"], row["count_bin"]),
                        metric,
                    ]
                    - row[metric]
                )
                if np.isfinite(row[metric])
                and np.isfinite(
                    baseline.loc[
                        (row["mode"], row["mechanism"], row["site"], row["count_bin"]),
                        metric,
                    ]
                )
                else math.nan
            ),
            axis=1,
        )
    return frame


def nested_example_pair(
    cfg: V10Config,
    vocab: Vocab,
    receiver_count: int,
    donor_count: int,
    seed: int,
) -> tuple[Example, Example]:
    rng = random.Random(int(seed))
    maximum = max(int(receiver_count), int(donor_count))
    candidate_positions = rng.sample(range(cfg.seq_len), maximum)
    priority_positions = list(candidate_positions)
    marker_by_position = {position: rng.choice(vocab.markers) for position in candidate_positions}
    base_noise = [rng.choice(vocab.noise) for _ in range(cfg.seq_len)]

    def build(count: int) -> Example:
        positions = sorted(priority_positions[: int(count)])
        sequence = list(base_noise)
        markers = [marker_by_position[position] for position in positions]
        for position, marker in zip(positions, markers):
            sequence[position] = marker
        return Example(sequence, int(count), positions, markers, seed)

    return build(receiver_count), build(donor_count)


def _regression_summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, part in frame.groupby(groups, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        x = part["donor_offset"].to_numpy(dtype=float)
        y = part["causal_expected_shift"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        prediction = design @ beta
        denominator = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - float(((y - prediction) ** 2).sum()) / denominator if denominator > 1e-12 else math.nan
        rows.append(
            {
                **dict(zip(groups, keys)),
                "n_pairs": len(part),
                "shift_slope": float(beta[1]),
                "shift_intercept": float(beta[0]),
                "shift_r2": r2,
                "shift_mae_to_requested_offset": float(np.mean(np.abs(y - x))),
                "follows_donor": float(part["follows_donor"].mean()),
                "follows_receiver": float(part["follows_receiver"].mean()),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def run_nested_count_head_patching(
    models: dict[str, Any],
    cfg: V10Config,
    vocab: Vocab,
    rankings: dict[str, list[Head]],
    *,
    random_replicates: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    top_ns = [1, 2, 4, 8, 16]
    all_heads = _all_heads(cfg)
    for mode in cfg.modes:
        model = models[mode]
        primary_name = "direct_broad" if mode == "nonthinking" else "trace_readout"
        primary = rankings[primary_name]
        named: dict[str, tuple[str, int, list[Head]]] = {
            "primary_top": ("primary_top", 0, primary),
            "primary_bottom": ("primary_bottom", 0, list(reversed(primary))),
        }
        for replicate in range(int(random_replicates)):
            shuffled = list(all_heads)
            random.Random(cfg.seed + 920_000 + replicate).shuffle(shuffled)
            named[f"random_{replicate}"] = ("random", replicate, shuffled)

        pair_specs = [
            (receiver_count, int(offset))
            for receiver_count in range(cfg.count_min, cfg.count_max + 1)
            for offset in cfg.patch_offsets
            if cfg.count_min <= receiver_count + int(offset) <= cfg.count_max
        ]
        progress = tqdm(total=len(pair_specs) * len(named) * len(top_ns), desc=f"v10 nested head patch: {mode}")
        for receiver_count, offset in pair_specs:
            donor_count = receiver_count + offset
            receiver, donor = nested_example_pair(
                cfg,
                vocab,
                receiver_count,
                donor_count,
                cfg.seed + 921_000 + receiver_count * 101 + donor_count,
            )
            receiver_item = render(receiver, vocab, mode)
            donor_item = render(donor, vocab, mode)
            receiver_ids = torch.tensor([receiver_item.input_ids], dtype=torch.long, device=cfg.device)
            donor_ids = torch.tensor([donor_item.input_ids], dtype=torch.long, device=cfg.device)
            receiver_pos = receiver_item.spans.ans_pos
            donor_pos = donor_item.spans.ans_pos
            baseline_logits = model(input_ids=receiver_ids).logits[0, receiver_pos]
            donor_logits = model(input_ids=donor_ids).logits[0, donor_pos]
            baseline_pred, baseline_expected, _ = count_prediction(baseline_logits, vocab)
            donor_pred, donor_expected, _ = count_prediction(donor_logits, vocab)
            donor_inputs = _capture_cproj_inputs(model, donor_ids, set(range(cfg.n_layer)))
            for _, (family, replicate, ranking) in named.items():
                for top_n in top_ns:
                    heads = ranking[:top_n]
                    patched_logits = _patched_head_forward(
                        model, receiver_ids, donor_inputs, heads, donor_pos, receiver_pos
                    )[receiver_pos]
                    patched_pred, patched_expected, _ = count_prediction(patched_logits, vocab)
                    rows.append(
                        {
                            "mode": mode,
                            "ranking_source": primary_name,
                            "family": family,
                            "replicate": replicate,
                            "receiver_count": receiver_count,
                            "donor_count": donor_count,
                            "donor_offset": offset,
                            "count_bin": count_bin(receiver_count),
                            "top_n": top_n,
                            "patched_heads": _head_label(heads),
                            "receiver_position": receiver_pos,
                            "donor_position": donor_pos,
                            "position_delta": donor_pos - receiver_pos,
                            "baseline_pred": baseline_pred,
                            "baseline_expected": baseline_expected,
                            "donor_pred": donor_pred,
                            "donor_expected": donor_expected,
                            "patched_pred": patched_pred,
                            "patched_expected": patched_expected,
                            "causal_expected_shift": patched_expected - baseline_expected,
                            "follows_donor": float(patched_pred == donor_count),
                            "follows_receiver": float(patched_pred == receiver_count),
                        }
                    )
                    progress.update(1)
        progress.close()
    detail = pd.DataFrame(rows)
    summary = detail.groupby(
        ["mode", "family", "replicate", "count_bin", "donor_offset", "top_n"], as_index=False
    ).mean(numeric_only=True)
    regression = _regression_summary(
        detail, ["mode", "family", "replicate", "top_n"]
    )
    return detail, summary, regression


@torch.no_grad()
def run_retrieval_control_patching(
    model,
    cfg: V10Config,
    vocab: Vocab,
    rankings: dict[str, list[Head]],
    *,
    examples_per_count: int = 2,
    random_replicates: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    examples = balanced_examples(cfg, vocab, examples_per_count, cfg.seed + 930_000)
    targeted = rankings["targeted_retrieval"]
    all_heads = _all_heads(cfg)
    named: dict[str, tuple[str, int, list[Head]]] = {
        "targeted_top": ("targeted_top", 0, targeted),
        "targeted_bottom": ("targeted_bottom", 0, list(reversed(targeted))),
    }
    for replicate in range(int(random_replicates)):
        shuffled = list(all_heads)
        random.Random(cfg.seed + 931_000 + replicate).shuffle(shuffled)
        named[f"random_{replicate}"] = ("random", replicate, shuffled)
    top_ns = [1, 2, 4, 8, 16]
    rows: list[dict[str, Any]] = []
    progress = tqdm(total=len(examples) * 2 * len(named) * len(top_ns), desc="v10 retrieval controls")
    for example_idx, example in enumerate(examples):
        k_values = sorted(set([max(1, (example.count + 1) // 2), example.count]))
        for k in k_values:
            corrupt = marker_identity_corruption(example, k, vocab)
            clean_item = render(example, vocab, "thinking")
            corrupt_item = render(corrupt, vocab, "thinking")
            clean_ids = torch.tensor([clean_item.input_ids], dtype=torch.long, device=cfg.device)
            corrupt_ids = torch.tensor([corrupt_item.input_ids], dtype=torch.long, device=cfg.device)
            query_pos = clean_item.spans.trace_index_positions[k - 1]
            clean_target = vocab.token_to_id[example.needle_markers[k - 1]]
            corrupt_target = vocab.token_to_id[corrupt.needle_markers[k - 1]]
            candidates = [clean_target, corrupt_target]
            clean_logits = model(input_ids=clean_ids).logits[0, query_pos]
            corrupt_logits = model(input_ids=corrupt_ids).logits[0, query_pos]
            clean_margin = margin(clean_logits, clean_target, candidates)
            corrupt_margin = margin(corrupt_logits, clean_target, candidates)
            donor = _capture_cproj_inputs(model, clean_ids, set(range(cfg.n_layer)))
            for _, (family, replicate, ranking) in named.items():
                for top_n in top_ns:
                    heads = ranking[:top_n]
                    patched_logits = _patched_head_forward(
                        model, corrupt_ids, donor, heads, query_pos, query_pos
                    )[query_pos]
                    patched_margin = margin(patched_logits, clean_target, candidates)
                    rows.append(
                        {
                            "example_idx": example_idx,
                            "count": example.count,
                            "count_bin": count_bin(example.count),
                            "query_k": k,
                            "query_role": "final" if k == example.count else "interior",
                            "family": family,
                            "replicate": replicate,
                            "top_n": top_n,
                            "patched_heads": _head_label(heads),
                            "clean_margin": clean_margin,
                            "corrupt_margin": corrupt_margin,
                            "patched_margin": patched_margin,
                            "normalized_recovery": normalized_recovery(
                                clean_margin, corrupt_margin, patched_margin
                            ),
                        }
                    )
                    progress.update(1)
    progress.close()
    detail = pd.DataFrame(rows)
    summary = detail.groupby(
        ["family", "replicate", "query_role", "count_bin", "top_n"], as_index=False
    ).mean(numeric_only=True)
    return detail, summary


def centroid_mean_pca(run_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = Path(run_dir)
    arrays = np.load(run_dir / "analysis" / "state_causal" / "centroids.npz")
    sites = sorted({key.split("__L", 1)[0] for key in arrays.files})
    coordinate_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for site in sites:
        for layer in range(1, 5):
            labels = sorted(
                int(key.rsplit("__C", 1)[1])
                for key in arrays.files
                if key.startswith(f"{site}__L{layer}__C")
            )
            if not labels:
                continue
            values = np.stack([arrays[f"{site}__L{layer}__C{label}"] for label in labels])
            centered = values - values.mean(axis=0, keepdims=True)
            _, singular, vh = np.linalg.svd(centered, full_matrices=False)
            coordinates = centered @ vh[:6].T
            eigenvalues = singular**2
            ratios = eigenvalues / max(float(eigenvalues.sum()), 1e-12)
            deltas = np.diff(values, axis=0)
            delta_norms = np.linalg.norm(deltas, axis=1)
            adjacent_cosines = []
            turning_angles = []
            for left, right in zip(deltas[:-1], deltas[1:]):
                cosine = float(left @ right / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12))
                adjacent_cosines.append(cosine)
                turning_angles.append(math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0)))))
            path_length = float(delta_norms.sum())
            chord = float(np.linalg.norm(values[-1] - values[0]))
            effective = float(eigenvalues.sum() ** 2 / max(float((eigenvalues**2).sum()), 1e-12))
            geometry_rows.append(
                {
                    "site": site,
                    "layer": layer - 1,
                    **{
                        f"pc{index + 1}_variance": float(ratios[index]) if index < len(ratios) else 0.0
                        for index in range(6)
                    },
                    "pc2_cumulative": float(ratios[:2].sum()),
                    "pc3_cumulative": float(ratios[:3].sum()),
                    "pc6_cumulative": float(ratios[:6].sum()),
                    "effective_dimension": effective,
                    "adjacent_delta_cosine_mean": float(np.mean(adjacent_cosines)),
                    "mean_turning_angle_degrees": float(np.mean(turning_angles)),
                    "path_chord_ratio": path_length / max(chord, 1e-12),
                }
            )
            for label, coordinate in zip(labels, coordinates):
                coordinate_rows.append(
                    {
                        "site": site,
                        "layer": layer - 1,
                        "state_label": label,
                        **{
                            f"pc{index + 1}": float(coordinate[index]) if index < coordinate.shape[0] else 0.0
                            for index in range(6)
                        },
                    }
                )
    return pd.DataFrame(coordinate_rows), pd.DataFrame(geometry_rows)


@torch.no_grad()
def run_centroid_transplants(
    models: dict[str, Any],
    cfg: V10Config,
    vocab: Vocab,
    run_dir: str | Path,
    *,
    examples_per_count: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = Path(run_dir)
    centroids = np.load(run_dir / "analysis" / "state_causal" / "centroids.npz")
    examples = balanced_examples(cfg, vocab, examples_per_count, cfg.seed + 940_000)
    sites = (SITE_NONTHINKING_FINAL, SITE_THINKING_FINAL, SITE_THINKING_FIXED)
    rows: list[dict[str, Any]] = []
    total = sum(
        1
        for example in examples
        for offset in cfg.patch_offsets
        if cfg.count_min <= example.count + int(offset) <= cfg.count_max
    ) * len(sites) * cfg.n_layer
    progress = tqdm(total=total, desc="v10 centroid residual transplant")
    for site in sites:
        mode = "nonthinking" if site == SITE_NONTHINKING_FINAL else "thinking"
        model = models[mode]
        for example_idx, example in enumerate(examples):
            item = render_site(example, vocab, site, cfg)
            ids = torch.tensor([item.input_ids], dtype=torch.long, device=cfg.device)
            position = item.spans.ans_pos
            baseline_logits = model(input_ids=ids).logits[0, position]
            baseline_pred, baseline_expected, _ = count_prediction(baseline_logits, vocab)
            for offset in cfg.patch_offsets:
                donor_count = example.count + int(offset)
                if not cfg.count_min <= donor_count <= cfg.count_max:
                    continue
                for layer in range(cfg.n_layer):
                    key = f"{site}__L{layer + 1}__C{donor_count}"
                    donor_state = torch.tensor(
                        centroids[key][None, :], dtype=torch.float32, device=cfg.device
                    )
                    patched_logits = _forward_residual_patch(
                        model, ids, layer, position, donor_state
                    )
                    patched_pred, patched_expected, _ = count_prediction(patched_logits, vocab)
                    rows.append(
                        {
                            "site": site,
                            "mode": mode,
                            "example_idx": example_idx,
                            "receiver_count": example.count,
                            "donor_count": donor_count,
                            "donor_offset": int(offset),
                            "count_bin": count_bin(example.count),
                            "layer": layer,
                            "token_position": position,
                            "baseline_pred": baseline_pred,
                            "baseline_expected": baseline_expected,
                            "patched_pred": patched_pred,
                            "patched_expected": patched_expected,
                            "causal_expected_shift": patched_expected - baseline_expected,
                            "follows_donor": float(patched_pred == donor_count),
                            "follows_receiver": float(patched_pred == example.count),
                        }
                    )
                    progress.update(1)
    progress.close()
    detail = pd.DataFrame(rows)
    summary = detail.groupby(
        ["site", "mode", "count_bin", "donor_offset", "layer"], as_index=False
    ).mean(numeric_only=True)
    return detail, summary


def run_report_followups(
    run_dir: str | Path,
    *,
    device: str | None = None,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    out_dir = run_dir / "analysis" / "report_followups"
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    expected = [
        "single_head_ablation.csv",
        "strict_cumulative_ablation.csv",
        "nested_head_patching.csv",
        "nested_head_patching_summary.csv",
        "nested_head_patching_regression.csv",
        "retrieval_control_patching.csv",
        "retrieval_control_patching_summary.csv",
        "centroid_mean_pca_coordinates.csv",
        "centroid_mean_geometry.csv",
        "centroid_transplant.csv",
        "centroid_transplant_summary.csv",
    ]
    if not overwrite and all((tables / name).exists() for name in expected):
        return {Path(name).stem: pd.read_csv(tables / name) for name in expected}

    cfg, vocab = load_run(run_dir, device=device)
    rankings = load_rankings(run_dir)
    models = {mode: load_final_model(cfg, vocab, run_dir, mode) for mode in cfg.modes}
    outputs: dict[str, pd.DataFrame] = {}

    single, cumulative = run_strict_ablation_suite(models, cfg, vocab, rankings)
    outputs["single_head_ablation"] = single
    outputs["strict_cumulative_ablation"] = cumulative

    nested, nested_summary, nested_regression = run_nested_count_head_patching(
        models, cfg, vocab, rankings
    )
    outputs["nested_head_patching"] = nested
    outputs["nested_head_patching_summary"] = nested_summary
    outputs["nested_head_patching_regression"] = nested_regression

    retrieval, retrieval_summary = run_retrieval_control_patching(
        models["thinking"], cfg, vocab, rankings
    )
    outputs["retrieval_control_patching"] = retrieval
    outputs["retrieval_control_patching_summary"] = retrieval_summary

    coordinates, geometry = centroid_mean_pca(run_dir)
    outputs["centroid_mean_pca_coordinates"] = coordinates
    outputs["centroid_mean_geometry"] = geometry

    centroid_rows, centroid_summary = run_centroid_transplants(
        models, cfg, vocab, run_dir
    )
    outputs["centroid_transplant"] = centroid_rows
    outputs["centroid_transplant_summary"] = centroid_summary

    for name, frame in outputs.items():
        frame.to_csv(tables / f"{name}.csv", index=False)
    manifest = {
        "run_dir": str(run_dir.resolve()),
        "device": cfg.device,
        "definitions": {
            "single_head_ablation": "globally zero one head and measure metric drop against the unmodified model",
            "strict_cumulative_ablation": "mask top-1 through top-16 ranked heads with bottom and six random-order controls",
            "nested_head_patching": "donor and receiver share base noise and a nested needle set; patch c_proj pre-projection head slices at the final query",
            "centroid_mean_pca": "average hidden states by exact count first, then fit PCA to the 30 class means separately at every site and layer",
            "centroid_transplant": "replace the receiver residual at one layer/query with the independent training centroid for donor count m",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for model in models.values():
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs
