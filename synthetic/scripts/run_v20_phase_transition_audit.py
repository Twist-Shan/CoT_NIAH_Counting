#!/usr/bin/env python3
"""Run a focused, resumable phase-transition audit for the v20 checkpoint series.

The audit keeps raw tensors in memory only long enough to reduce them to compact
statistics.  It measures per-k retrieval routing and QK discrimination at every
100-step snapshot, causal/patching evidence at a denser milestone grid, and
high-power autoregressive accuracy around the candidate transition window.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares

from synthetic_counting_v20.config import V20Config, config_from_dict
from synthetic_counting_v20.data import (
    V20Example,
    V20Rendered,
    V20Vocab,
    collate_v20,
    load_corpus_split,
    load_corpus_text,
    load_suite_manifests,
    render_v20,
)
from synthetic_counting_v20.model import _apply_rope, build_model
from synthetic_counting_v20.needle_pool import load_needle_pool
from synthetic_counting_v20.phase_transition import _causal_rows
from synthetic_counting_v20.training import (
    _parse_generation,
    autoregressive_task_evaluation,
    checkpoint_steps,
)
from synthetic_counting_v20.v10_port_analysis import (
    _attention_pattern_value_patch,
    _capture_attention_internals,
    _marker_margin,
    _normalized_recovery,
    _residual_patch,
    _retrieval_corruption,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "colab_results" / "v20_main_RoPE_count1-30_seed1234"
AUDIT_DIR = Path("analysis") / "phase_transition_audit"
ROUTING_STEPS = tuple(range(0, 10_001, 100))
INTERVENTION_STEPS = (
    0,
    1_000,
    1_500,
    2_000,
    2_500,
    3_000,
    3_500,
    4_000,
    4_500,
    5_000,
    5_500,
    6_000,
    6_500,
    7_000,
    8_000,
    9_000,
    10_000,
)
HIGH_POWER_AR_STEPS = tuple(range(3_000, 7_001, 500))


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def batches(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def balanced(
    examples: Sequence[V20Example],
    count_max: int,
    per_count: int,
    *,
    offset: int = 0,
) -> list[V20Example]:
    selected: list[V20Example] = []
    for count in range(1, count_max + 1):
        bucket = [item for item in examples if int(item.count or 0) == count]
        values = bucket[offset : offset + per_count]
        if len(values) != per_count:
            raise ValueError(
                f"count={count}: need {offset + per_count} examples, found {len(bucket)}"
            )
        selected.extend(values)
    return selected


def parse_steps(values: Sequence[str], default: Sequence[int]) -> tuple[int, ...]:
    if not values:
        return tuple(int(value) for value in default)
    result: set[int] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                pieces = [int(item) for item in part.split(":")]
                if len(pieces) not in (2, 3):
                    raise ValueError(f"invalid step range: {part}")
                start, stop = pieces[:2]
                stride = pieces[2] if len(pieces) == 3 else 100
                result.update(range(start, stop + 1, stride))
            else:
                result.add(int(part))
    return tuple(sorted(result))


def load_inputs(
    run_dir: Path, device: str
) -> tuple[
    V20Config,
    V20Vocab,
    list[V20Example],
    list[V20Example],
    list[V20Example],
]:
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    cfg = replace(cfg, device=device)
    vocab = V20Vocab.load(run_dir / "vocab.json")
    corpus = load_corpus_text()
    split = load_corpus_split(run_dir / "data/corpus_split.json", cfg, corpus)
    pool = load_needle_pool(
        run_dir / "data/needle_pool.json",
        cfg,
        split_fingerprint=split.split_fingerprint,
        vocab_fingerprint=vocab.fingerprint,
    )
    curves, test_suites = load_suite_manifests(
        run_dir / "data/loss_suite_manifests.json",
        split_fingerprint=split.split_fingerprint,
        pool_fingerprint=pool.pool_fingerprint,
    )
    heldout = list(curves["heldout"]["task"])
    reporting = balanced(
        heldout,
        cfg.count_max_threshold,
        cfg.phase_examples_per_count,
        offset=cfg.phase_head_selection_examples_per_count,
    )
    port_reporting: list[V20Example] = []
    split_at = cfg.phase_head_selection_examples_per_count
    for count in range(1, cfg.count_max_threshold + 1):
        bucket = [item for item in heldout if int(item.count or 0) == count]
        port_reporting.extend(bucket[split_at:])
    final_test = balanced(
        list(test_suites["task"]),
        cfg.count_max_threshold,
        cfg.final_examples_per_count,
    )
    return cfg, vocab, reporting, port_reporting, final_test


def checkpoint_map(run_dir: Path, mode: str) -> dict[int, Path]:
    return {
        int(step): Path(shard)
        for step, shard in checkpoint_steps(run_dir, "rope", mode)
    }


def iter_checkpoint_models(
    cfg: V20Config,
    vocab: V20Vocab,
    run_dir: Path,
    mode: str,
    steps: Iterable[int],
) -> Iterator[tuple[int, torch.nn.Module]]:
    available = checkpoint_map(run_dir, mode)
    selected = sorted(set(int(step) for step in steps))
    missing = [step for step in selected if step not in available]
    if missing:
        raise FileNotFoundError(f"missing {mode} checkpoints: {missing}")
    grouped: dict[Path, list[int]] = {}
    for step in selected:
        grouped.setdefault(available[step], []).append(step)
    model = build_model(cfg, vocab, "rope", cfg.device).eval()
    for shard, shard_steps in grouped.items():
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        for step in sorted(shard_steps):
            model.load_state_dict(payload["model_state_dicts"][str(step)])
            yield step, model
        del payload
    del model


def exposure_lookup(run_dir: Path) -> dict[tuple[int, int], float]:
    frame = pd.read_csv(run_dir / "tables/training_token_exposure_by_k.csv")
    frame = frame[frame["mode"] == "thinking"]
    result = {
        (int(row.step), int(row.k)): float(row.trace_index_token_exposure)
        for row in frame.itertuples(index=False)
    }
    for k in range(1, 31):
        result[(0, k)] = 0.0
    return result


@torch.inference_mode()
def routing_batch_rows(
    model: torch.nn.Module,
    cfg: V20Config,
    vocab: V20Vocab,
    items: Sequence[V20Rendered],
    *,
    step: int,
    layer: int,
    head: int,
) -> list[dict[str, Any]]:
    captured: dict[str, torch.Tensor] = {}
    attention = model.layers[layer - 1].attention
    previous = attention.intervention

    def capture(query, key, value, weights):
        captured["query"] = query.detach()
        captured["key"] = key.detach()
        captured["weights"] = weights.detach()
        return value, weights

    attention.intervention = capture
    try:
        ids, _, mask = collate_v20(list(items), vocab, cfg.device)
        model(input_ids=ids, attention_mask=mask, output_attentions=False)
    finally:
        attention.intervention = previous
    query = captured["query"][:, head]
    key = captured["key"][:, head]
    weights = captured["weights"][:, head]
    scale = math.sqrt(float(attention.head_dim))
    rows: list[dict[str, Any]] = []
    for row, item in enumerate(items):
        assert item.spans is not None and item.count is not None
        candidates = list(item.prompt_needle_positions)
        for k, query_position in enumerate(item.spans.trace_index_positions, start=1):
            source_position = candidates[k - 1]
            scores = (
                query[row, int(query_position)]
                @ key[row, candidates].transpose(0, 1)
            ).float() / scale
            correct_score = scores[k - 1]
            wrong_scores = torch.cat((scores[: k - 1], scores[k:]))
            best_wrong = wrong_scores.max() if len(wrong_scores) else correct_score
            rows.append(
                {
                    "step": int(step),
                    "count": int(item.count),
                    "k": int(k),
                    "targeted_mass": float(
                        weights[row, int(query_position), int(source_position)].float().cpu()
                    ),
                    "qk_correct_score": float(correct_score.cpu()),
                    "qk_best_wrong_score": float(best_wrong.cpu()),
                    "qk_margin": float((correct_score - best_wrong).cpu()),
                    "correct_occurrence_top1": float(int(scores.argmax()) == k - 1),
                }
            )
    return rows


def run_routing(
    run_dir: Path,
    cfg: V20Config,
    vocab: V20Vocab,
    examples: Sequence[V20Example],
    steps: Sequence[int],
) -> Path:
    output = run_dir / AUDIT_DIR / "tables/routing_qk_by_k.csv"
    roles = json.loads(
        (run_dir / "analysis/phase_transition/fixed_head_roles.json").read_text(
            encoding="utf-8"
        )
    )
    layer = int(roles["targeted_retrieval"]["layer"])
    head = int(roles["targeted_retrieval"]["head"])
    items = [render_v20(example, vocab, "thinking") for example in examples]
    exposure = exposure_lookup(run_dir)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (step, model) in enumerate(
        iter_checkpoint_models(cfg, vocab, run_dir, "thinking", steps), start=1
    ):
        sample_rows: list[dict[str, Any]] = []
        for batch in batches(items, min(8, cfg.analysis_batch_size)):
            sample_rows.extend(
                routing_batch_rows(
                    model,
                    cfg,
                    vocab,
                    batch,
                    step=step,
                    layer=layer,
                    head=head,
                )
            )
        sample = pd.DataFrame(sample_rows)
        summary = sample.groupby(["step", "k"], as_index=False).agg(
            targeted_mass=("targeted_mass", "mean"),
            qk_correct_score=("qk_correct_score", "mean"),
            qk_best_wrong_score=("qk_best_wrong_score", "mean"),
            qk_margin=("qk_margin", "mean"),
            correct_occurrence_top1=("correct_occurrence_top1", "mean"),
            observations=("targeted_mass", "size"),
        )
        summary["semantic_token_exposure"] = [
            exposure.get((int(row.step), int(row.k)), math.nan)
            for row in summary.itertuples(index=False)
        ]
        summary["layer"] = layer
        summary["head"] = head
        rows.extend(summary.to_dict("records"))
        if index % 10 == 0 or index == len(steps):
            print(
                f"[routing] {index}/{len(steps)} step={step} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    frame = pd.DataFrame(rows).sort_values(["step", "k"])
    atomic_csv(frame, output)
    return output


def retrieval_ranking(run_dir: Path) -> list[tuple[int, int]]:
    candidates = (
        (
            run_dir / "analysis/v10_port/tables/head_rankings.csv",
            ("thinking_targeted", "targeted_retrieval"),
        ),
        (
            run_dir / "analysis/phase_transition/tables/fixed_head_rankings.csv",
            ("targeted_retrieval", "thinking_targeted"),
        ),
    )
    attempted: list[str] = []
    for path, role_names in candidates:
        attempted.append(str(path))
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if not {"role", "rank", "layer", "head"}.issubset(frame.columns):
            continue
        selected = frame[frame["role"].isin(role_names)].sort_values("rank")
        if not selected.empty:
            return [
                (int(row.layer), int(row.head))
                for row in selected.itertuples(index=False)
            ]
    raise FileNotFoundError(
        "No usable targeted-retrieval head ranking was found; checked: "
        + ", ".join(attempted)
    )


def prepare_retrieval_pairs(
    examples: Sequence[V20Example], vocab: V20Vocab, amount: int = 24
) -> tuple[
    list[V20Example],
    list[V20Rendered],
    list[V20Rendered],
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
]:
    eligible = [item for item in examples if int(item.count or 0) >= 3][:amount]
    if len(eligible) != amount:
        raise ValueError(f"retrieval patching needs {amount} examples")
    ks = [max(2, int(item.count or 0) // 2) for item in eligible]
    pairs = [
        _retrieval_corruption(example, vocab, k)
        for example, k in zip(eligible, ks, strict=True)
    ]
    clean_items = [pair[0] for pair in pairs]
    corrupt_items = [pair[1] for pair in pairs]
    target_ids = [pair[2] for pair in pairs]
    alternative_ids = [pair[3] for pair in pairs]
    query_positions = [
        item.spans.trace_index_positions[k - 1]
        for item, k in zip(clean_items, ks, strict=True)
    ]
    source_positions = [
        item.prompt_needle_positions[k - 1]
        for item, k in zip(clean_items, ks, strict=True)
    ]
    return (
        eligible,
        clean_items,
        corrupt_items,
        ks,
        target_ids,
        alternative_ids,
        query_positions,
        source_positions,
    )


def recovery_summary(
    *,
    step: int,
    intervention: str,
    top_n: int,
    residual_layer: int | None,
    clean_margin: np.ndarray,
    corrupt_margin: np.ndarray,
    patched_margin: np.ndarray,
) -> dict[str, Any]:
    recovery = _normalized_recovery(clean_margin, corrupt_margin, patched_margin)
    valid = np.isfinite(recovery)
    values = recovery[valid]
    return {
        "step": int(step),
        "intervention": intervention,
        "top_n": int(top_n),
        "residual_layer": residual_layer,
        "observations": int(valid.sum()),
        "normalized_recovery_mean": float(np.mean(values)) if len(values) else math.nan,
        "normalized_recovery_median": float(np.median(values)) if len(values) else math.nan,
        "normalized_recovery_sem": (
            float(np.std(values, ddof=1) / math.sqrt(len(values)))
            if len(values) > 1
            else math.nan
        ),
        "clean_margin": float(np.mean(clean_margin)),
        "corrupt_margin": float(np.mean(corrupt_margin)),
        "patched_margin": float(np.mean(patched_margin)),
        "identification_gap": float(np.mean(clean_margin - corrupt_margin)),
        "margin_restoration": float(np.mean(patched_margin - corrupt_margin)),
        "recovery_reliable": float(abs(float(np.mean(clean_margin - corrupt_margin))) >= 1.0),
        "clean_correct": float(np.mean(clean_margin > 0)),
        "corrupt_correct": float(np.mean(corrupt_margin > 0)),
        "patched_correct": float(np.mean(patched_margin > 0)),
    }


def run_patching(
    run_dir: Path,
    cfg: V20Config,
    vocab: V20Vocab,
    examples: Sequence[V20Example],
    steps: Sequence[int],
) -> Path:
    output = run_dir / AUDIT_DIR / "tables/retrieval_transport_recovery.csv"
    ranking = retrieval_ranking(run_dir)
    (
        _,
        clean_items,
        corrupt_items,
        ks,
        target_ids,
        alternative_ids,
        query_positions,
        source_positions,
    ) = prepare_retrieval_pairs(examples, vocab)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (step, model) in enumerate(
        iter_checkpoint_models(cfg, vocab, run_dir, "thinking", steps), start=1
    ):
        clean_output, clean_values = _capture_attention_internals(
            model, clean_items, vocab, cfg.device
        )
        corrupt_output, _ = _capture_attention_internals(
            model, corrupt_items, vocab, cfg.device
        )
        assert clean_output.hidden_states is not None
        clean_margin = _marker_margin(
            clean_output.logits, clean_items, ks, target_ids, alternative_ids
        )
        corrupt_margin = _marker_margin(
            corrupt_output.logits, corrupt_items, ks, target_ids, alternative_ids
        )
        rows.append(
            recovery_summary(
                step=step,
                intervention="corrupt_baseline",
                top_n=0,
                residual_layer=None,
                clean_margin=clean_margin,
                corrupt_margin=corrupt_margin,
                patched_margin=corrupt_margin,
            )
        )
        for top_n in (1, 2):
            with _attention_pattern_value_patch(
                model,
                ranking[:top_n],
                query_positions,
                source_positions,
                donor_values=clean_values,
            ):
                ids, _, mask = collate_v20(corrupt_items, vocab, cfg.device)
                with torch.inference_mode():
                    changed = model(input_ids=ids, attention_mask=mask)
            patched_margin = _marker_margin(
                changed.logits, corrupt_items, ks, target_ids, alternative_ids
            )
            rows.append(
                recovery_summary(
                    step=step,
                    intervention="value_only_at_target_source",
                    top_n=top_n,
                    residual_layer=None,
                    clean_margin=clean_margin,
                    corrupt_margin=corrupt_margin,
                    patched_margin=patched_margin,
                )
            )
        for residual_layer in range(1, cfg.n_layer + 1):
            clean_residual = torch.stack(
                [
                    clean_output.hidden_states[residual_layer][row, position]
                    for row, position in enumerate(query_positions)
                ]
            )
            with _residual_patch(model, residual_layer, query_positions, clean_residual):
                ids, _, mask = collate_v20(corrupt_items, vocab, cfg.device)
                with torch.inference_mode():
                    changed = model(input_ids=ids, attention_mask=mask)
            patched_margin = _marker_margin(
                changed.logits, corrupt_items, ks, target_ids, alternative_ids
            )
            rows.append(
                recovery_summary(
                    step=step,
                    intervention="residual_stream",
                    top_n=0,
                    residual_layer=residual_layer,
                    clean_margin=clean_margin,
                    corrupt_margin=corrupt_margin,
                    patched_margin=patched_margin,
                )
            )
        print(
            f"[patching] {index}/{len(steps)} step={step} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    frame = pd.DataFrame(rows).sort_values(["step", "intervention", "top_n"])
    atomic_csv(frame, output)
    return output


def run_causality(
    run_dir: Path,
    cfg: V20Config,
    vocab: V20Vocab,
    examples: Sequence[V20Example],
    steps: Sequence[int],
) -> Path:
    output = run_dir / AUDIT_DIR / "tables/local_head_causal_damage.csv"
    manifest = json.loads(
        (run_dir / "analysis/phase_transition/manifest.json").read_text(encoding="utf-8")
    )
    fixed = {
        role: (int(value[0]), int(value[1]))
        for role, value in manifest["fixed_heads"].items()
    }
    controls = {
        role: (int(value[0]), int(value[1]))
        for role, value in manifest["same_layer_score_matched_controls"].items()
    }
    items = [render_v20(example, vocab, "thinking") for example in examples]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, (step, model) in enumerate(
        iter_checkpoint_models(cfg, vocab, run_dir, "thinking", steps), start=1
    ):
        rows.extend(
            _causal_rows(cfg, vocab, model, items, fixed, controls, step=step)
        )
        print(
            f"[causal] {index}/{len(steps)} step={step} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame["causal_damage"] = -frame["margin_change_from_baseline"]
    atomic_csv(frame, output)
    return output


def completed_ar_group(
    frame: pd.DataFrame,
    *,
    step: int,
    mode: str,
    examples_per_count: int,
    count_max: int,
) -> bool:
    if frame.empty:
        return False
    selected = frame[(frame["step"] == step) & (frame["mode"] == mode)]
    counts = selected.groupby("count").size()
    return len(counts) == count_max and bool(counts.eq(examples_per_count).all())


def apply_rope_at_position(
    tensor: torch.Tensor, *, position: int, base: float
) -> torch.Tensor:
    width = tensor.shape[-1]
    inverse = 1.0 / (
        float(base)
        ** (
            torch.arange(0, width, 2, device=tensor.device, dtype=torch.float32)
            / width
        )
    )
    angles = (float(position) * inverse).to(tensor.dtype)
    cosine, sine = angles.cos()[None, None, None], angles.sin()[None, None, None]
    even, odd = tensor[..., 0::2], tensor[..., 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    ).flatten(-2)


@torch.inference_mode()
def cached_prefix_forward(
    model: torch.nn.Module, input_ids: torch.Tensor
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    hidden = model.token_embedding(input_ids)
    caches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer in model.layers:
        normalized = layer.ln_attention(hidden)
        batch, length, _ = normalized.shape
        qkv = layer.attention.qkv(normalized).view(
            batch,
            length,
            3,
            layer.attention.n_head,
            layer.attention.head_dim,
        )
        query, key, value = (
            part.transpose(1, 2) for part in qkv.unbind(dim=2)
        )
        query = _apply_rope(query, base=layer.attention.rope_base)
        key = _apply_rope(key, base=layer.attention.rope_base)
        context = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=True
        )
        projected = layer.attention.output(
            context.transpose(1, 2).contiguous().view(batch, length, -1)
        )
        hidden = hidden + projected
        hidden = hidden + layer.mlp(layer.ln_mlp(hidden))
        caches.append((key, value))
    logits = F.linear(model.final_norm(hidden), model.unembedding_weight)
    return logits, caches


@torch.inference_mode()
def cached_incremental_forward(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    caches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    position: int,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    hidden = model.token_embedding(token_ids[:, None])
    updated: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer, (past_key, past_value) in zip(model.layers, caches, strict=True):
        normalized = layer.ln_attention(hidden)
        batch = normalized.shape[0]
        qkv = layer.attention.qkv(normalized).view(
            batch,
            1,
            3,
            layer.attention.n_head,
            layer.attention.head_dim,
        )
        query, key, value = (
            part.transpose(1, 2) for part in qkv.unbind(dim=2)
        )
        query = apply_rope_at_position(
            query, position=position, base=layer.attention.rope_base
        )
        key = apply_rope_at_position(
            key, position=position, base=layer.attention.rope_base
        )
        all_key = torch.cat((past_key, key), dim=2)
        all_value = torch.cat((past_value, value), dim=2)
        context = F.scaled_dot_product_attention(
            query, all_key, all_value, dropout_p=0.0, is_causal=False
        )
        projected = layer.attention.output(
            context.transpose(1, 2).contiguous().view(batch, 1, -1)
        )
        hidden = hidden + projected
        hidden = hidden + layer.mlp(layer.ln_mlp(hidden))
        updated.append((all_key, all_value))
    logits = F.linear(model.final_norm(hidden), model.unembedding_weight)
    return logits, updated


@torch.inference_mode()
def cached_autoregressive_task_evaluation(
    model: torch.nn.Module,
    cfg: V20Config,
    vocab: V20Vocab,
    examples: list[V20Example],
    *,
    mode: str,
    step: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not examples:
        return pd.DataFrame(rows)
    model.eval()
    batch_size = min(cfg.analysis_batch_size, len(examples))
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        prefixes: list[list[str]] = []
        for example in chunk:
            item = render_v20(example, vocab, mode)
            assert item.spans is not None
            stop = (
                item.spans.ans_pos + 1
                if mode == "nonthinking"
                else item.spans.think_pos + 1
            )
            prefixes.append(item.tokens[:stop])
        lengths = {len(tokens) for tokens in prefixes}
        if len(lengths) != 1:
            raise ValueError(f"cached AR needs equal prefix lengths, found {lengths}")
        prefix_length = lengths.pop()
        generated = torch.tensor(
            [vocab.encode(tokens) for tokens in prefixes], device=cfg.device
        )
        logits, caches = cached_prefix_forward(model, generated)
        done = torch.zeros(len(chunk), dtype=torch.bool, device=cfg.device)
        max_new_tokens = (
            4 if mode == "nonthinking" else cfg.max_render_len - prefix_length + 2
        )
        for generation_index in range(max_new_tokens):
            next_ids = logits[:, -1].argmax(dim=-1)
            next_ids = torch.where(
                done, torch.full_like(next_ids, vocab.eos_id), next_ids
            )
            generated = torch.cat((generated, next_ids[:, None]), dim=1)
            done |= next_ids.eq(vocab.eos_id)
            if bool(done.all()) or generation_index + 1 == max_new_tokens:
                break
            logits, caches = cached_incremental_forward(
                model,
                next_ids,
                caches,
                position=prefix_length + generation_index,
            )
        for index, example in enumerate(chunk):
            rows.append(
                {
                    "step": step,
                    "position_encoding": "rope",
                    "mode": mode,
                    "row_id": start + index,
                    "set_id": example.set_id,
                    "count": example.count,
                    "count_bin": cfg.count_bin(int(example.count)),
                    "corpus_region": example.corpus_region,
                    "corpus_start": example.corpus_start,
                    "prompt_sha256": example.prompt_sha256,
                    **_parse_generation(
                        vocab.decode(generated[index]), vocab, example, mode
                    ),
                }
            )
    return pd.DataFrame(rows)


def validate_cached_ar(
    run_dir: Path,
    cfg: V20Config,
    vocab: V20Vocab,
    examples: Sequence[V20Example],
    *,
    step: int = 5_000,
) -> None:
    selected = balanced(examples, cfg.count_max_threshold, 1)
    for mode in ("nonthinking", "thinking"):
        _, model = next(
            iter_checkpoint_models(cfg, vocab, run_dir, mode, (step,))
        )
        reference = autoregressive_task_evaluation(
            model,
            cfg,
            vocab,
            list(selected),
            position_encoding="rope",
            mode=mode,
            step=step,
        )
        cached = cached_autoregressive_task_evaluation(
            model, cfg, vocab, list(selected), mode=mode, step=step
        )
        if reference["generated_tokens"].tolist() != cached["generated_tokens"].tolist():
            mismatches = np.flatnonzero(
                reference["generated_tokens"].to_numpy()
                != cached["generated_tokens"].to_numpy()
            )
            raise AssertionError(
                f"cached AR differs for {mode} at rows {mismatches[:10].tolist()}"
            )
        print(
            f"[cache-check] {mode} step={step}: "
            f"{len(cached)} generations exactly match",
            flush=True,
        )


def run_high_power_ar(
    run_dir: Path,
    cfg: V20Config,
    vocab: V20Vocab,
    examples: Sequence[V20Example],
    steps: Sequence[int],
    *,
    examples_per_count: int,
) -> Path:
    table_dir = run_dir / AUDIT_DIR / "tables"
    detail_path = table_dir / "high_power_ar_detail.csv"
    existing = pd.read_csv(detail_path) if detail_path.exists() else pd.DataFrame()
    selected = balanced(examples, cfg.count_max_threshold, examples_per_count)
    columns = [
        "step",
        "position_encoding",
        "mode",
        "row_id",
        "set_id",
        "count",
        "count_bin",
        "corpus_region",
        "corpus_start",
        "prompt_sha256",
        "ar_pred_count",
        "ar_answered",
        "ar_accuracy",
        "ar_abs_error",
        "ar_abs_error_with_missing_penalty",
        "trace_exact",
        "trace_ordered_marker_accuracy",
        "trace_marker_recall",
    ]
    started = time.perf_counter()
    total = len(steps) * 2
    finished = 0
    for mode in ("nonthinking", "thinking"):
        needed = [
            step
            for step in steps
            if not completed_ar_group(
                existing,
                step=step,
                mode=mode,
                examples_per_count=examples_per_count,
                count_max=cfg.count_max_threshold,
            )
        ]
        finished += len(steps) - len(needed)
        for step, model in iter_checkpoint_models(
            cfg, vocab, run_dir, mode, needed
        ):
            frame = cached_autoregressive_task_evaluation(
                model,
                cfg,
                vocab,
                list(selected),
                mode=mode,
                step=step,
            )[columns]
            if existing.empty:
                existing = frame
            else:
                keep = ~(
                    (existing["step"] == step) & (existing["mode"] == mode)
                )
                existing = pd.concat((existing[keep], frame), ignore_index=True)
            existing = existing.sort_values(["step", "mode", "count", "row_id"])
            atomic_csv(existing, detail_path)
            finished += 1
            print(
                f"[high-power-ar] {finished}/{total} {mode} step={step} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    summary = existing.groupby(["step", "mode", "count"], as_index=False).agg(
        examples=("ar_accuracy", "size"),
        ar_accuracy=("ar_accuracy", "mean"),
        ar_answer_rate=("ar_answered", "mean"),
        mean_abs_error=("ar_abs_error_with_missing_penalty", "mean"),
        trace_exact=("trace_exact", "mean"),
        trace_ordered_marker_accuracy=("trace_ordered_marker_accuracy", "mean"),
    )
    atomic_csv(summary, table_dir / "high_power_ar_by_count.csv")
    overall = existing.groupby(["step", "mode"], as_index=False).agg(
        examples=("ar_accuracy", "size"),
        successes=("ar_accuracy", "sum"),
        ar_accuracy=("ar_accuracy", "mean"),
        ar_answer_rate=("ar_answered", "mean"),
        mean_abs_error=("ar_abs_error_with_missing_penalty", "mean"),
        trace_exact=("trace_exact", "mean"),
        trace_ordered_marker_accuracy=("trace_ordered_marker_accuracy", "mean"),
    )
    atomic_csv(overall, table_dir / "high_power_ar_summary.csv")
    return detail_path


def weighted_step_summary(frame: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for step, group in frame.groupby("step"):
        weights = group["observations"].to_numpy(dtype=float)
        row: dict[str, Any] = {"step": int(step), "observations": int(weights.sum())}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            row[metric] = (
                float(np.average(values[valid], weights=weights[valid]))
                if bool(valid.any())
                else math.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("step")


def model_bic(residual: np.ndarray, parameters: int) -> float:
    residual = np.asarray(residual, dtype=float)
    n = len(residual)
    rss = max(float(np.sum(residual**2)), 1e-15)
    return n * math.log(rss / max(n, 1)) + parameters * math.log(max(n, 2))


def fit_transition_series(
    x: Sequence[float],
    y: Sequence[float],
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    weight_array = (
        np.ones_like(y_array)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    valid = np.isfinite(x_array) & np.isfinite(y_array) & np.isfinite(weight_array)
    x_array, y_array, weight_array = x_array[valid], y_array[valid], weight_array[valid]
    if len(x_array) < 6 or float(np.ptp(x_array)) <= 0:
        return {
            "observations": len(x_array),
            "smooth_model": "insufficient",
            "smooth_bic": math.nan,
            "changepoint_model": "insufficient",
            "changepoint_bic": math.nan,
            "delta_bic_smooth_minus_changepoint": math.nan,
            "candidate_x": math.nan,
            "classification": "insufficient",
        }
    order = np.argsort(x_array)
    x_array, y_array, weight_array = (
        x_array[order],
        y_array[order],
        weight_array[order],
    )
    weight_array = weight_array / max(float(np.mean(weight_array)), 1e-12)
    root_weight = np.sqrt(weight_array)
    x_min, x_max = float(x_array.min()), float(x_array.max())
    x_scaled = (x_array - x_min) / (x_max - x_min)

    linear_design = np.column_stack((np.ones(len(x_scaled)), x_scaled))
    linear_coef = np.linalg.lstsq(
        linear_design * root_weight[:, None], y_array * root_weight, rcond=None
    )[0]
    linear_pred = linear_design @ linear_coef
    smooth_candidates: list[tuple[str, float, list[float]]] = [
        (
            "linear_continuous",
            model_bic(root_weight * (linear_pred - y_array), 2),
            linear_coef.tolist(),
        )
    ]

    span = max(float(np.ptp(y_array)), float(np.std(y_array)), 1e-3)

    def sigmoid_prediction(parameters: np.ndarray) -> np.ndarray:
        baseline, amplitude, center, log_scale = parameters
        scale = math.exp(float(log_scale))
        argument = np.clip((x_scaled - center) / scale, -60, 60)
        return baseline + amplitude / (1.0 + np.exp(-argument))

    for center in (0.25, 0.5, 0.75):
        for sign in (1.0, -1.0):
            initial = np.asarray(
                [float(np.median(y_array[:2])), sign * span, center, math.log(0.12)]
            )
            result = least_squares(
                lambda parameters: root_weight
                * (sigmoid_prediction(parameters) - y_array),
                initial,
                bounds=(
                    np.asarray([-np.inf, -np.inf, 0.0, math.log(0.005)]),
                    np.asarray([np.inf, np.inf, 1.0, math.log(5.0)]),
                ),
                max_nfev=5_000,
            )
            prediction = sigmoid_prediction(result.x)
            smooth_candidates.append(
                (
                    "sigmoid",
                    model_bic(root_weight * (prediction - y_array), 4),
                    result.x.tolist(),
                )
            )
    smooth_model, smooth_bic, smooth_parameters = min(
        smooth_candidates, key=lambda item: item[1]
    )
    smooth_center_x = math.nan
    smooth_width_10_90 = math.nan
    smooth_width_fraction = math.nan
    if smooth_model == "sigmoid":
        smooth_center_x = x_min + float(smooth_parameters[2]) * (x_max - x_min)
        smooth_width_10_90 = (
            2.0
            * math.log(9.0)
            * math.exp(float(smooth_parameters[3]))
            * (x_max - x_min)
        )
        smooth_width_fraction = smooth_width_10_90 / (x_max - x_min)

    minimum_side = max(2, int(math.ceil(0.15 * len(x_scaled))))
    cp_candidates: list[tuple[str, float, float, list[float]]] = []
    for cp_index in range(minimum_side, len(x_scaled) - minimum_side):
        cp = float((x_scaled[cp_index - 1] + x_scaled[cp_index]) / 2)
        hinge = np.maximum(0.0, x_scaled - cp)
        continuous = np.column_stack((np.ones(len(x_scaled)), x_scaled, hinge))
        coefficient = np.linalg.lstsq(
            continuous * root_weight[:, None], y_array * root_weight, rcond=None
        )[0]
        prediction = continuous @ coefficient
        cp_candidates.append(
            (
                "continuous_slope_changepoint",
                model_bic(root_weight * (prediction - y_array), 4),
                cp,
                coefficient.tolist(),
            )
        )
        jump = (x_scaled >= cp).astype(float)
        discontinuous = np.column_stack(
            (np.ones(len(x_scaled)), x_scaled, jump, hinge)
        )
        coefficient = np.linalg.lstsq(
            discontinuous * root_weight[:, None], y_array * root_weight, rcond=None
        )[0]
        prediction = discontinuous @ coefficient
        cp_candidates.append(
            (
                "level_and_slope_changepoint",
                model_bic(root_weight * (prediction - y_array), 5),
                cp,
                coefficient.tolist(),
            )
        )
    cp_model, cp_bic, cp_scaled, cp_parameters = min(
        cp_candidates, key=lambda item: item[1]
    )
    delta = float(smooth_bic - cp_bic)
    if delta >= 10:
        classification = "strong_changepoint_preference"
    elif delta >= 6:
        classification = "moderate_changepoint_preference"
    elif delta <= -6:
        classification = "smooth_preference"
    else:
        classification = "inconclusive"
    return {
        "observations": len(x_array),
        "x_min": x_min,
        "x_max": x_max,
        "smooth_model": smooth_model,
        "smooth_bic": float(smooth_bic),
        "smooth_parameters_json": json.dumps(smooth_parameters),
        "smooth_center_x": smooth_center_x,
        "smooth_width_10_90": smooth_width_10_90,
        "smooth_width_fraction": smooth_width_fraction,
        "changepoint_model": cp_model,
        "changepoint_bic": float(cp_bic),
        "changepoint_parameters_json": json.dumps(cp_parameters),
        "delta_bic_smooth_minus_changepoint": delta,
        "candidate_x": x_min + cp_scaled * (x_max - x_min),
        "classification": classification,
    }


def combined_ar_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / AUDIT_DIR / "tables/high_power_ar_summary.csv"
    audit = pd.read_csv(path) if path.exists() else pd.DataFrame()
    final_path = run_dir / "tables/final_autoregressive_summary.csv"
    if final_path.exists():
        final = pd.read_csv(final_path).copy()
        final = final.rename(
            columns={
                "ar_final_accuracy": "ar_accuracy",
                "ar_answer_rate": "ar_answer_rate",
            }
        )
        final["successes"] = final["ar_accuracy"] * final["examples"]
        keep = [
            "step",
            "mode",
            "examples",
            "successes",
            "ar_accuracy",
            "ar_answer_rate",
            "trace_exact",
            "trace_ordered_marker_accuracy",
        ]
        final = final[[column for column in keep if column in final.columns]]
        if audit.empty:
            audit = final
        else:
            audit = pd.concat(
                (audit[~audit["step"].isin(final["step"])], final),
                ignore_index=True,
                sort=False,
            )
    return audit.sort_values(["step", "mode"])


def run_model_comparison(run_dir: Path) -> tuple[Path, Path]:
    table_dir = run_dir / AUDIT_DIR / "tables"
    routing = pd.read_csv(table_dir / "routing_qk_by_k.csv")
    per_k_rows: list[dict[str, Any]] = []
    for metric in ("targeted_mass", "qk_margin", "correct_occurrence_top1"):
        for k, line in routing.groupby("k"):
            for axis, column in (
                ("training_step", "step"),
                ("semantic_token_exposure", "semantic_token_exposure"),
            ):
                result = fit_transition_series(
                    line[column], line[metric], line["observations"]
                )
                per_k_rows.append(
                    {"evidence_family": "routing", "metric": metric, "k": int(k), "axis": axis, **result}
                )
    per_k = pd.DataFrame(per_k_rows)
    per_k_path = table_dir / "per_k_transition_model_comparison.csv"
    atomic_csv(per_k, per_k_path)

    series_rows: list[dict[str, Any]] = []
    routing_summary = weighted_step_summary(
        routing,
        ("targeted_mass", "qk_margin", "correct_occurrence_top1"),
    )
    for metric in ("targeted_mass", "qk_margin", "correct_occurrence_top1"):
        result = fit_transition_series(
            routing_summary["step"],
            routing_summary[metric],
            routing_summary["observations"],
        )
        series_rows.append(
            {"evidence_family": "routing", "metric": metric, "group": "all_k", "axis": "training_step", **result}
        )

    dense_roles_path = (
        run_dir / "analysis/phase_transition/tables/dense_fixed_head_dynamics.csv"
    )
    if dense_roles_path.exists():
        dense_roles = pd.read_csv(dense_roles_path)
        dense_roles = dense_roles[dense_roles["is_fixed_role_head"] == 1]
        for role, line in dense_roles.groupby("role"):
            result = fit_transition_series(line["step"], line["score"], line["observations"])
            series_rows.append(
                {
                    "evidence_family": "attention_role",
                    "metric": "fixed_head_role_score",
                    "group": role,
                    "axis": "training_step",
                    **result,
                }
            )

    patching = pd.read_csv(table_dir / "retrieval_transport_recovery.csv")
    patching = patching[
        (patching["intervention"] == "value_only_at_target_source")
        | ((patching["intervention"] == "residual_stream") & (patching["residual_layer"].isin([2, 3])))
    ].copy()
    patching["group"] = [
        (
            f"value_top{int(row.top_n)}"
            if row.intervention == "value_only_at_target_source"
            else f"residual_L{int(row.residual_layer)}"
        )
        for row in patching.itertuples(index=False)
    ]
    for group, line in patching.groupby("group"):
        result = fit_transition_series(
            line["step"], line["margin_restoration"], line["observations"]
        )
        series_rows.append(
            {"evidence_family": "transport", "metric": "patched_minus_corrupt_margin", "group": group, "axis": "training_step", **result}
        )

    causal = pd.read_csv(table_dir / "local_head_causal_damage.csv")
    causal = causal[causal["intervention"] == "fixed_head_zero"]
    for role, line in causal.groupby("role"):
        result = fit_transition_series(line["step"], line["causal_damage"])
        series_rows.append(
            {"evidence_family": "causality", "metric": "correct_token_margin_damage", "group": role, "axis": "training_step", **result}
        )

    ar = combined_ar_summary(run_dir)
    for mode, line in ar.groupby("mode"):
        result = fit_transition_series(line["step"], line["ar_accuracy"], line["examples"])
        series_rows.append(
            {"evidence_family": "behavior", "metric": "high_power_ar_accuracy", "group": mode, "axis": "training_step", **result}
        )
    series = pd.DataFrame(series_rows)
    series_path = table_dir / "aggregate_transition_model_comparison.csv"
    atomic_csv(series, series_path)
    return per_k_path, series_path


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 145,
            "savefig.dpi": 175,
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.constrained_layout.use": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_routing_heatmaps(run_dir: Path) -> Path:
    frame = pd.read_csv(run_dir / AUDIT_DIR / "tables/routing_qk_by_k.csv")
    specifications = (
        ("targeted_mass", "Targeted attention mass", "viridis", 0.0, 1.0),
        ("qk_margin", "Correct − best-wrong QK margin", "coolwarm", None, None),
        ("correct_occurrence_top1", "Correct occurrence top-1 accuracy", "viridis", 0.0, 1.0),
    )
    fig, axes = plt.subplots(3, 1, figsize=(14.2, 10.6), sharex=True)
    steps = sorted(int(value) for value in frame["step"].unique())
    ks = sorted(int(value) for value in frame["k"].unique())
    for ax, (metric, title, cmap, vmin, vmax) in zip(axes, specifications, strict=True):
        matrix = (
            frame.pivot(index="k", columns="step", values=metric)
            .reindex(index=ks, columns=steps)
            .to_numpy(dtype=float)
        )
        if metric == "qk_margin":
            limit = float(np.nanquantile(np.abs(matrix), 0.98))
            vmin, vmax = -limit, limit
        image = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            extent=(steps[0] - 50, steps[-1] + 50, ks[0] - 0.5, ks[-1] + 0.5),
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.axvline(1500, color="white", linestyle=":", linewidth=1.4)
        ax.set(title=title, ylabel="semantic k", yticks=[1, 5, 10, 15, 20, 25, 30])
        colorbar = fig.colorbar(image, ax=ax, fraction=0.022, pad=0.018)
        colorbar.set_label(metric.replace("_", " "))
    axes[-1].set_xlabel("optimizer training step")
    axes[-1].set_xticks(np.arange(0, 10_001, 1_000))
    fig.suptitle("Per-k retrieval routing and discrimination across training", fontsize=15)
    output = run_dir / AUDIT_DIR / "figures/per_k_routing_qk_dynamics.png"
    savefig(fig, output)
    return output


def wilson(successes: float, observations: int) -> tuple[float, float]:
    if observations <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    estimate = float(successes) / observations
    denominator = 1 + z * z / observations
    center = (estimate + z * z / (2 * observations)) / denominator
    radius = z * math.sqrt(
        estimate * (1 - estimate) / observations
        + z * z / (4 * observations * observations)
    ) / denominator
    return center - radius, center + radius


def plot_high_power_ar_by_count(run_dir: Path) -> Path:
    frame = pd.read_csv(
        run_dir / AUDIT_DIR / "tables/high_power_ar_by_count.csv"
    )
    specifications = (
        ("nonthinking", "ar_accuracy", "Nonthinking final-count exact"),
        ("thinking", "ar_accuracy", "Thinking final-count exact"),
        ("thinking", "trace_exact", "Thinking whole-trace exact"),
        (
            "thinking",
            "trace_ordered_marker_accuracy",
            "Thinking ordered-marker accuracy",
        ),
    )
    steps = sorted(int(value) for value in frame["step"].unique())
    counts = list(range(1, 31))
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 8.5), sharex=True, sharey=True)
    image = None
    for ax, (mode, metric, title) in zip(axes.flat, specifications, strict=True):
        selected = frame[frame["mode"] == mode]
        matrix = (
            selected.pivot(index="count", columns="step", values=metric)
            .reindex(index=counts, columns=steps)
            .to_numpy(dtype=float)
        )
        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(steps[0] - 250, steps[-1] + 250, 0.5, 30.5),
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        ax.set(title=title, ylabel="true count n", yticks=[1, 5, 10, 15, 20, 25, 30])
    for ax in axes[-1]:
        ax.set_xlabel("optimizer training step")
        ax.set_xticks(steps)
        ax.set_xticklabels([f"{step/1000:g}k" for step in steps])
    if image is None:
        raise ValueError("high-power AR heatmap could not be created")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label("exact accuracy / ordered-marker accuracy")
    fig.suptitle("High-power autoregressive dynamics · 50 held-out prompts per count", fontsize=15)
    output = run_dir / AUDIT_DIR / "figures/high_power_ar_by_count_dynamics.png"
    savefig(fig, output)
    return output


def plot_functionwise_evidence(run_dir: Path) -> Path:
    table_dir = run_dir / AUDIT_DIR / "tables"
    routing = pd.read_csv(table_dir / "routing_qk_by_k.csv")
    routing_summary = weighted_step_summary(
        routing, ("targeted_mass", "qk_margin", "correct_occurrence_top1")
    )
    patching = pd.read_csv(table_dir / "retrieval_transport_recovery.csv")
    causal = pd.read_csv(table_dir / "local_head_causal_damage.csv")
    ar = combined_ar_summary(run_dir)
    fits = pd.read_csv(table_dir / "aggregate_transition_model_comparison.csv")
    colors = {"thinking": "#d97745", "nonthinking": "#315f9f"}
    fig, axes = plt.subplots(2, 3, figsize=(16.4, 9.1))

    ax = axes[0, 0]
    for mode, line in ar.groupby("mode"):
        line = line.sort_values("step")
        low, high = zip(
            *(wilson(row.successes, int(row.examples)) for row in line.itertuples(index=False))
        )
        y = line["ar_accuracy"].to_numpy(dtype=float)
        ax.errorbar(
            line["step"],
            y,
            yerr=np.asarray([y - np.asarray(low), np.asarray(high) - y]),
            marker="o",
            capsize=3,
            linewidth=2,
            color=colors[mode],
            label=mode,
        )
    ax.set(title="High-power free-generation behavior", ylabel="AR exact accuracy", ylim=(-0.03, 1.04))
    ax.legend(loc="lower right")

    ax = axes[0, 1]
    ax.plot(routing_summary["step"], routing_summary["targeted_mass"], color="#6f4aa8", label="targeted mass")
    ax.plot(routing_summary["step"], routing_summary["correct_occurrence_top1"], color="#16877d", label="correct top-1")
    ax.set(title="Retrieval routing", ylabel="mass / accuracy", ylim=(-0.03, 1.04))
    ax.legend(loc="upper left")

    ax = axes[0, 2]
    ax.plot(routing_summary["step"], routing_summary["qk_margin"], color="#b54a62")
    ax.axhline(0, color="#222", linewidth=1)
    ax.set(title="Retrieval discrimination", ylabel="correct − best-wrong QK score")

    ax = axes[1, 0]
    selected = patching[
        (patching["intervention"] == "value_only_at_target_source")
        | (
            (patching["intervention"] == "residual_stream")
            & (patching["residual_layer"].isin([3, 4]))
        )
    ].copy()
    selected["group"] = [
        (
            f"value_top{int(row.top_n)}"
            if row.intervention == "value_only_at_target_source"
            else f"residual_L{int(row.residual_layer)}"
        )
        for row in selected.itertuples(index=False)
    ]
    labels = {
        "value_top1": "value-only top-1 head",
        "value_top2": "value-only top-2 heads",
        "residual_L3": "post-L3 residual",
        "residual_L4": "post-L4 residual upper bound",
    }
    for group, line in selected.groupby("group"):
        line = line.sort_values("step")
        ax.errorbar(
            line["step"],
            line["margin_restoration"],
            marker="o",
            capsize=2,
            label=labels[group],
        )
    ax.axhline(0, color="#222", linewidth=1)
    ax.set(title="Identity transport patch effect", ylabel="patched − corrupt marker margin")
    ax.legend(loc="best")

    ax = axes[1, 1]
    fixed = causal[causal["intervention"] == "fixed_head_zero"]
    for role, line in fixed.groupby("role"):
        ax.plot(line["step"], line["causal_damage"], marker="o", label=role)
    ax.axhline(0, color="#222", linewidth=1)
    ax.set(title="Role-head causal necessity", ylabel="correct-token margin damage")
    ax.legend(loc="upper left")

    ax = axes[1, 2]
    labels = []
    values = []
    colors_fit = []
    family_abbreviation = {
        "routing": "routing",
        "attention_role": "role",
        "transport": "transport",
        "causality": "causal",
        "behavior": "behavior",
    }
    group_abbreviation = {
        "all_k": "all-k",
        "marker_successor": "successor",
        "targeted_retrieval": "targeted",
        "residual_L2": "residual L2",
        "residual_L3": "residual L3",
        "value_top1": "value top-1",
        "value_top2": "value top-2",
    }
    metric_abbreviation = {
        "targeted_mass": "targeted mass",
        "qk_margin": "QK margin",
        "correct_occurrence_top1": "occurrence top-1",
    }
    for row in fits.itertuples(index=False):
        group_label = (
            metric_abbreviation.get(row.metric, row.metric)
            if row.evidence_family == "routing"
            else group_abbreviation.get(row.group, row.group)
        )
        label = (
            f"{family_abbreviation.get(row.evidence_family, row.evidence_family)} · "
            f"{group_label}"
        )
        labels.append(label)
        values.append(float(row.delta_bic_smooth_minus_changepoint))
        colors_fit.append("#b54a62" if values[-1] >= 10 else "#60829c")
    positions = np.arange(len(values))
    ax.barh(positions, values, color=colors_fit)
    ax.axvline(0, color="#222", linewidth=1)
    ax.axvline(10, color="#b54a62", linestyle="--", linewidth=1)
    ax.set_yticks(positions, labels)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.invert_yaxis()
    ax.set(title="Model comparison", xlabel="ΔBIC = smooth − changepoint")

    for ax in axes.flat[:5]:
        ax.axvline(1500, color="#222", linestyle=":", linewidth=1.1)
        ax.set_xlabel("optimizer training step")
    fig.suptitle("Function-wise transition profiles across training", fontsize=15)
    output = run_dir / AUDIT_DIR / "figures/synchronized_phase_evidence.png"
    savefig(fig, output)
    return output


def plot_exposure_alignment(run_dir: Path) -> Path:
    frame = pd.read_csv(run_dir / AUDIT_DIR / "tables/routing_qk_by_k.csv")
    selected_ks = (1, 5, 10, 15, 20, 25, 30)
    specifications = (
        ("targeted_mass", "Targeted attention mass", (0.0, 1.0)),
        ("qk_margin", "Correct − best-wrong QK margin", None),
        ("correct_occurrence_top1", "Correct occurrence top-1 accuracy", (0.0, 1.0)),
    )
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(selected_ks)))
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
    for ax, (metric, title, ylim) in zip(axes, specifications, strict=True):
        for k, color in zip(selected_ks, colors, strict=True):
            line = frame[frame["k"] == k].sort_values("semantic_token_exposure")
            ax.plot(
                line["semantic_token_exposure"] / 1_000_000,
                line[metric],
                color=color,
                label=f"k={k}",
            )
        ax.set(title=title, xlabel="semantic index-token exposure (millions)", ylabel=metric.replace("_", " "))
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[-1].legend(title="semantic k", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Retrieval dynamics re-parameterized by each k's own token exposure", fontsize=15)
    output = run_dir / AUDIT_DIR / "figures/routing_by_semantic_exposure.png"
    savefig(fig, output)
    return output


def plot_per_k_model_evidence(run_dir: Path) -> Path:
    frame = pd.read_csv(
        run_dir / AUDIT_DIR / "tables/per_k_transition_model_comparison.csv"
    )
    metrics = ("targeted_mass", "qk_margin", "correct_occurrence_top1")
    axes_names = ("training_step", "semantic_token_exposure")
    fig, axes = plt.subplots(2, 3, figsize=(14.7, 7.6), sharex=True)
    for row_index, axis_name in enumerate(axes_names):
        for col_index, metric in enumerate(metrics):
            ax = axes[row_index, col_index]
            line = frame[(frame["axis"] == axis_name) & (frame["metric"] == metric)].sort_values("k")
            values = line["delta_bic_smooth_minus_changepoint"].to_numpy(dtype=float)
            ax.plot(line["k"], values, marker="o", markersize=3.5, color="#315f9f")
            ax.fill_between(line["k"], 0, values, where=values >= 10, color="#b54a62", alpha=0.22)
            ax.axhline(0, color="#222", linewidth=1)
            ax.axhline(10, color="#b54a62", linestyle="--", linewidth=1)
            ax.set_yscale("symlog", linthresh=5)
            ax.set_title(metric.replace("_", " "))
            if col_index == 0:
                ax.set_ylabel(f"{axis_name}\nΔBIC")
            if row_index == 1:
                ax.set_xlabel("semantic k")
    fig.suptitle("Per-k evidence for a changepoint over a smooth trajectory", fontsize=15)
    output = run_dir / AUDIT_DIR / "figures/per_k_changepoint_model_evidence.png"
    savefig(fig, output)
    return output


def run_plots(run_dir: Path) -> list[Path]:
    setup_style()
    return [
        plot_routing_heatmaps(run_dir),
        plot_high_power_ar_by_count(run_dir),
        plot_functionwise_evidence(run_dir),
        plot_exposure_alignment(run_dir),
        plot_per_k_model_evidence(run_dir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--part",
        action="append",
        choices=("routing", "patching", "causal", "cache-check", "ar", "fit", "plots"),
        default=[],
    )
    parser.add_argument("--routing-steps", action="append", default=[])
    parser.add_argument("--intervention-steps", action="append", default=[])
    parser.add_argument("--ar-steps", action="append", default=[])
    parser.add_argument("--ar-examples-per-count", type=int, default=50)
    parser.add_argument("--ar-batch-size", type=int, default=128)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    parts = args.part or ["routing", "patching", "causal", "ar", "fit", "plots"]
    routing_steps = parse_steps(args.routing_steps, ROUTING_STEPS)
    intervention_steps = parse_steps(args.intervention_steps, INTERVENTION_STEPS)
    ar_steps = parse_steps(args.ar_steps, HIGH_POWER_AR_STEPS)
    cfg, vocab, reporting, port_reporting, final_test = load_inputs(run_dir, args.device)
    started = time.perf_counter()
    if "cache-check" in parts:
        validate_cached_ar(run_dir, cfg, vocab, final_test)
    if "routing" in parts:
        run_routing(run_dir, cfg, vocab, reporting, routing_steps)
    if "patching" in parts:
        run_patching(run_dir, cfg, vocab, port_reporting, intervention_steps)
    if "causal" in parts:
        run_causality(run_dir, cfg, vocab, reporting, intervention_steps)
    if "ar" in parts:
        ar_cfg = replace(cfg, analysis_batch_size=int(args.ar_batch_size))
        run_high_power_ar(
            run_dir,
            ar_cfg,
            vocab,
            final_test,
            ar_steps,
            examples_per_count=args.ar_examples_per_count,
        )
    if "fit" in parts:
        run_model_comparison(run_dir)
    figures: list[Path] = []
    if "plots" in parts:
        figures = run_plots(run_dir)
    manifest_path = run_dir / AUDIT_DIR / "manifest.json"
    existing_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    figure_records = (
        [str(path.relative_to(run_dir)) for path in figures]
        if figures
        else list(existing_manifest.get("figures", []))
    )
    artifact_parts = {
        "routing": run_dir / AUDIT_DIR / "tables/routing_qk_by_k.csv",
        "patching": run_dir / AUDIT_DIR / "tables/retrieval_transport_recovery.csv",
        "causal": run_dir / AUDIT_DIR / "tables/local_head_causal_damage.csv",
        "ar": run_dir / AUDIT_DIR / "tables/high_power_ar_summary.csv",
        "fit": run_dir
        / AUDIT_DIR
        / "tables/aggregate_transition_model_comparison.csv",
        "plots": run_dir / AUDIT_DIR / "figures/synchronized_phase_evidence.png",
    }
    completed_parts = sorted(
        set(existing_manifest.get("completed_parts", []))
        | set(parts)
        | {part for part, path in artifact_parts.items() if path.exists()}
    )
    manifest = {
        "run": run_dir.name,
        "device": args.device,
        "routing_steps": list(routing_steps),
        "intervention_steps": list(intervention_steps),
        "high_power_ar_steps": list(ar_steps),
        "high_power_ar_examples_per_count": args.ar_examples_per_count,
        "fixed_targeted_head": json.loads(
            (run_dir / "analysis/phase_transition/fixed_head_roles.json").read_text(
                encoding="utf-8"
            )
        )["targeted_retrieval"],
        "definitions": {
            "targeted_mass": "attention mass from trace index k to the matching prompt occurrence k",
            "qk_margin": "scaled pre-softmax QK score at the correct occurrence minus the largest score among all other prompt occurrences",
            "correct_occurrence_top1": "fraction for which the correct prompt occurrence has the largest scaled QK score among prompt occurrences",
            "normalized_recovery": "(patched marker margin - corrupt marker margin) / (clean marker margin - corrupt marker margin)",
            "margin_restoration": "patched correct-marker logit margin minus corrupt correct-marker logit margin; used for dynamics because normalized recovery is unstable when the clean-corrupt gap is near zero",
            "causal_damage": "baseline correct-token margin minus the margin after position-local fixed-head zeroing",
            "semantic_token_exposure": "cumulative number of training trace-index tokens for the corresponding semantic k",
            "delta_bic": "BIC(smooth best of linear/sigmoid) minus BIC(best slope/level changepoint); positive favors changepoint",
        },
        "limitations": [
            "one training seed only",
            "fixed roles selected at the final checkpoint on a disjoint selection split",
            "routing uses one reporting example per true count and therefore fewer observations at high k",
            "changepoint BIC is descriptive because neighboring checkpoints are autocorrelated",
            "post-L4 residual patching copies the final query state and is therefore an upper-bound intervention rather than a localized mechanism claim",
        ],
        "figures": figure_records,
        "completed_parts": completed_parts,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(manifest, manifest_path)
    print(f"PHASE_AUDIT_DIR={run_dir / AUDIT_DIR}", flush=True)


if __name__ == "__main__":
    main()
