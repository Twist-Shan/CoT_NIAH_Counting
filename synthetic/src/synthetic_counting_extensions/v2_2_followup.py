from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    from synthetic_niah_v5.model import GPT2LMHeadModel
except ModuleNotFoundError:
    from transformers import GPT2LMHeadModel


SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<Ans>", "<Think/>", "</Think>"]
NOISE_TOKENS = [f"<N{i}>" for i in range(64)]
MARKER_TOKENS = [f"<{chr(ord('A') + i)}>" for i in range(10)]
NUMBER_TOKENS = [f"<{i}>" for i in range(1, 11)]


@dataclass
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @classmethod
    def build(cls) -> "Vocab":
        tokens = SPECIAL_TOKENS + NOISE_TOKENS + MARKER_TOKENS + NUMBER_TOKENS
        return cls({tok: i for i, tok in enumerate(tokens)}, tokens)

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return cls(dict(obj["token_to_id"]), list(obj["id_to_token"]))

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id[t] for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[int(i)] for i in ids]

    def count_to_token(self, count: int) -> str:
        return f"<{int(count)}>"

    @property
    def numeric_ids(self) -> list[int]:
        return [self.token_to_id[t] for t in NUMBER_TOKENS]


@dataclass
class BaseExample:
    seq_tokens: list[str]
    count: int
    needle_positions: list[int]
    needle_markers: list[str]
    seed: int | None = None


def sample_base_example(seq_len: int, rng: random.Random, *, count: int, seed: int | None = None) -> BaseExample:
    positions = sorted(rng.sample(range(seq_len), int(count)))
    markers = [rng.choice(MARKER_TOKENS) for _ in positions]
    seq = [rng.choice(NOISE_TOKENS) for _ in range(seq_len)]
    for pos, marker in zip(positions, markers):
        seq[pos] = marker
    return BaseExample(seq, int(count), positions, markers, seed)


def balanced_examples(seq_len: int, examples_per_count: int, seed: int) -> list[BaseExample]:
    rng = random.Random(seed)
    examples: list[BaseExample] = []
    for count in range(1, 11):
        for i in range(examples_per_count):
            examples.append(sample_base_example(seq_len, rng, count=count, seed=seed + 1000 * count + i))
    rng.shuffle(examples)
    return examples


def render_thinking(ex: BaseExample, vocab: Vocab, *, trace_count_override: int | None = None) -> dict[str, Any]:
    trace: list[str] = []
    index_positions: list[int] = []
    marker_positions: list[int] = []
    pre_index_positions: list[int] = []
    think_start_pos = 1 + len(ex.seq_tokens)
    pos = think_start_pos + 1
    prev_marker_pos = think_start_pos
    trace_count = ex.count if trace_count_override is None else int(trace_count_override)
    for k in range(1, trace_count + 1):
        marker = ex.needle_markers[(k - 1) % len(ex.needle_markers)]
        pre_index_positions.append(prev_marker_pos)
        trace.extend([vocab.count_to_token(k), marker])
        index_positions.append(pos)
        marker_positions.append(pos + 1)
        prev_marker_pos = pos + 1
        pos += 2
    tokens = ["<BOS>"] + ex.seq_tokens + ["<Think/>"] + trace + ["</Think>", "<Ans>", vocab.count_to_token(trace_count), "<EOS>"]
    think_end_pos = think_start_pos + 1 + len(trace)
    ans_pos = think_end_pos + 1
    return {
        "tokens": tokens,
        "input_ids": vocab.encode(tokens),
        "count": ex.count,
        "trace_count": trace_count,
        "needle_markers": ex.needle_markers,
        "prompt_needle_positions": [1 + p for p in ex.needle_positions],
        "anchors": {
            "prompt_start": 1,
            "prompt_end_exclusive": 1 + len(ex.seq_tokens),
            "think_start": think_start_pos,
            "think_end": think_end_pos,
            "ans_token": ans_pos,
            "index_positions": index_positions,
            "marker_positions": marker_positions,
            "pre_index_positions": pre_index_positions,
        },
    }


def load_v2_thinking_model(v2_run_dir: str | Path, device: str):
    v2_run_dir = Path(v2_run_dir)
    ckpt = v2_run_dir / "checkpoints" / "final" / "thinking"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing v2 thinking checkpoint: {ckpt}")
    try:
        model = GPT2LMHeadModel.from_pretrained(ckpt, attn_implementation="eager")
    except TypeError:
        model = GPT2LMHeadModel.from_pretrained(ckpt)
        model.config._attn_implementation = "eager"
    model.to(device)
    model.eval()
    vocab_path = v2_run_dir / "checkpoints" / "final" / "vocab.json"
    vocab = Vocab.load(vocab_path) if vocab_path.exists() else Vocab.build()
    cfg = {"seq_len": 256, "seed": 1234}
    cfg_path = v2_run_dir / "config.json"
    if cfg_path.exists():
        cfg.update(json.loads(cfg_path.read_text(encoding="utf-8")))
    return model, vocab, cfg


def _attention_entropy(weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0 or len(weights) <= 1:
        return math.nan
    p = weights / total
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum() / math.log(len(weights)))


def _margin(logits: torch.Tensor, target_id: int, competitor_ids: list[int]) -> float:
    values = logits.detach().float().cpu()
    competitors = [idx for idx in competitor_ids if int(idx) != int(target_id)]
    if not competitors:
        return math.nan
    return float(values[target_id] - values[competitors].max())


def _num_layers_heads(model: torch.nn.Module) -> tuple[int, int]:
    cfg = model.config
    n_layer = int(getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", 0)))
    n_head = int(getattr(cfg, "n_head", getattr(cfg, "num_attention_heads", 0)))
    if n_layer <= 0 or n_head <= 0:
        raise ValueError("Could not infer n_layer/n_head from model.config")
    return n_layer, n_head


def _head_mask(
    model: torch.nn.Module,
    device: str | torch.device,
    masked_heads: list[tuple[int, int]],
) -> torch.Tensor:
    n_layer, n_head = _num_layers_heads(model)
    mask = torch.ones((n_layer, n_head), dtype=torch.float32, device=device)
    for layer, head in masked_heads:
        layer_idx = int(layer) - 1
        head_idx = int(head)
        if 0 <= layer_idx < n_layer and 0 <= head_idx < n_head:
            mask[layer_idx, head_idx] = 0.0
    return mask


def _count_margin(logits: torch.Tensor, vocab: Vocab, count: int) -> float:
    return _margin(logits, vocab.token_to_id[vocab.count_to_token(int(count))], vocab.numeric_ids)


def _pred_count(logits: torch.Tensor, vocab: Vocab) -> int:
    return int(logits.detach().float().cpu()[vocab.numeric_ids].argmax().item()) + 1


def _top_heads(df: pd.DataFrame, metric: str, n: int) -> list[tuple[int, int]]:
    if df.empty or metric not in df.columns:
        return []
    rows = df.sort_values(metric, ascending=False).head(int(n))
    return [(int(row.layer), int(row.head)) for row in rows.itertuples(index=False)]


def _dedupe_heads(heads: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for head in heads:
        if head not in seen:
            seen.add(head)
            out.append(head)
    return out


def _head_group_specs(
    answer_head: pd.DataFrame,
    *,
    n_layer: int,
    n_head: int,
    seed: int,
) -> dict[str, list[tuple[int, int]]]:
    all_heads = [(layer, head) for layer in range(1, n_layer + 1) for head in range(n_head)]
    rng = random.Random(seed)
    specs: dict[str, list[tuple[int, int]]] = {}
    for n in [1, 2, 4, 8]:
        specs[f"top_all_trace_marker_{n}"] = _top_heads(answer_head, "all_trace_marker_mass", n)
        specs[f"top_last_trace_marker_{n}"] = _top_heads(answer_head, "last_trace_marker_mass", n)
        if n <= len(all_heads):
            specs[f"random_{n}"] = sorted(rng.sample(all_heads, n))
    if n_layer >= 4:
        specs["layer4_all"] = [(4, head) for head in range(n_head)]
    if n_layer >= 2:
        specs["layers2_to_4_all"] = [
            (layer, head)
            for layer in range(2, min(4, n_layer) + 1)
            for head in range(n_head)
        ]
    specs["all_heads"] = all_heads
    return {name: _dedupe_heads(heads) for name, heads in specs.items() if heads}


@torch.no_grad()
def run_v2_2_followup(
    v2_run_dir: str | Path,
    *,
    examples_per_count: int = 50,
    causal_examples_per_count: int | None = None,
    device: str | None = None,
    seed_offset: int = 88_000,
) -> dict[str, pd.DataFrame]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, cfg = load_v2_thinking_model(v2_run_dir, device)
    examples = balanced_examples(int(cfg.get("seq_len", 256)), examples_per_count, int(cfg.get("seed", 1234)) + seed_offset)
    if causal_examples_per_count is None:
        causal_examples_per_count = min(10, examples_per_count)
    causal_examples = balanced_examples(
        int(cfg.get("seq_len", 256)),
        int(causal_examples_per_count),
        int(cfg.get("seed", 1234)) + seed_offset + 17,
    )
    out_dir = Path(v2_run_dir) / "v2_2_followup_mechanism"
    tables = out_dir / "tables"
    figs = out_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    transition_rows: list[dict[str, Any]] = []
    next_index_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    override_rows: list[dict[str, Any]] = []
    successor_causal_rows: list[dict[str, Any]] = []
    answer_mask_rows: list[dict[str, Any]] = []
    close_id = vocab.token_to_id["</Think>"]
    number_ids = vocab.numeric_ids

    for ex_idx, ex in enumerate(examples):
        rendered = render_thinking(ex, vocab)
        ids = torch.tensor([rendered["input_ids"]], dtype=torch.long, device=device)
        out = model(input_ids=ids, output_attentions=True)
        logits = out.logits[0]
        attns = [a[0].detach().cpu().numpy() for a in out.attentions or []]
        a = rendered["anchors"]
        prompt_needles = rendered["prompt_needle_positions"]
        prompt_set = set(prompt_needles)
        prompt_noise = [p for p in range(a["prompt_start"], a["prompt_end_exclusive"]) if p not in prompt_set]
        index_positions = a["index_positions"]
        marker_positions = a["marker_positions"]
        ans_pos = a["ans_token"]
        for k, marker_pos in enumerate(marker_positions, start=1):
            target_token = vocab.count_to_token(k + 1) if k < len(marker_positions) else "</Think>"
            target_id = vocab.token_to_id[target_token]
            competitors = number_ids + [close_id]
            for layer, probs in enumerate(attns, start=1):
                for head in range(probs.shape[0]):
                    row = probs[head, marker_pos]
                    next_prompt = prompt_needles[k] if k < len(prompt_needles) else None
                    transition_rows.append(
                        {
                            "example_idx": ex_idx,
                            "count": ex.count,
                            "k": k,
                            "is_final_marker": float(k == ex.count),
                            "layer": layer,
                            "head": head,
                            "target_next_token": target_token,
                            "next_token_margin": _margin(logits[marker_pos], target_id, competitors),
                            "current_index_mass": float(row[index_positions[k - 1]]),
                            "current_marker_self_mass": float(row[marker_pos]),
                            "previous_marker_mass": float(row[marker_positions[k - 2]]) if k > 1 else 0.0,
                            "next_prompt_needle_mass": float(row[next_prompt]) if next_prompt is not None else 0.0,
                            "all_prompt_needles_mass": float(row[prompt_needles].sum()) if prompt_needles else 0.0,
                            "prompt_noise_mass": float(row[prompt_noise].sum()) if prompt_noise else 0.0,
                        }
                    )
        for next_k in range(2, len(index_positions) + 1):
            index_pos = index_positions[next_k - 1]
            correct_prompt_pos = prompt_needles[next_k - 1]
            previous_marker_pos = marker_positions[next_k - 2]
            previous_index_pos = index_positions[next_k - 2]
            for layer, probs in enumerate(attns, start=1):
                for head in range(probs.shape[0]):
                    row = probs[head, index_pos]
                    prompt_weights = row[prompt_needles]
                    if len(prompt_weights):
                        ordered = np.argsort(-prompt_weights)
                        top_prompt_pos = prompt_needles[int(ordered[0])]
                        rank_matches = np.where(np.array(prompt_needles)[ordered] == correct_prompt_pos)[0]
                        correct_rank = int(rank_matches[0]) + 1 if len(rank_matches) else math.nan
                    else:
                        top_prompt_pos = None
                        correct_rank = math.nan
                    next_index_rows.append(
                        {
                            "example_idx": ex_idx,
                            "count": ex.count,
                            "k": next_k,
                            "layer": layer,
                            "head": head,
                            "query_position": "index_token_k",
                            "correct_prompt_needle_mass": float(row[correct_prompt_pos]),
                            "correct_top1": float(top_prompt_pos == correct_prompt_pos),
                            "correct_rank": correct_rank,
                            "all_prompt_needles_mass": float(row[prompt_needles].sum()) if prompt_needles else 0.0,
                            "prompt_noise_mass": float(row[prompt_noise].sum()) if prompt_noise else 0.0,
                            "previous_trace_marker_mass": float(row[previous_marker_pos]),
                            "previous_trace_index_mass": float(row[previous_index_pos]),
                            "self_mass": float(row[index_pos]),
                        }
                    )
        for layer, probs in enumerate(attns, start=1):
            for head in range(probs.shape[0]):
                row = probs[head, ans_pos]
                marker_weights = row[marker_positions]
                index_weights = row[index_positions]
                answer_rows.append(
                    {
                        "example_idx": ex_idx,
                        "count": ex.count,
                        "layer": layer,
                        "head": head,
                        "all_trace_marker_mass": float(marker_weights.sum()),
                        "all_trace_index_mass": float(index_weights.sum()),
                        "last_trace_marker_mass": float(marker_weights[-1]) if len(marker_weights) else 0.0,
                        "last_trace_index_mass": float(index_weights[-1]) if len(index_weights) else 0.0,
                        "trace_marker_entropy": _attention_entropy(marker_weights),
                        "trace_index_entropy": _attention_entropy(index_weights),
                        "prompt_needles_mass": float(row[prompt_needles].sum()) if prompt_needles else 0.0,
                        "prompt_noise_mass": float(row[prompt_noise].sum()) if prompt_noise else 0.0,
                    }
                )
        if ex_idx < max(20, examples_per_count):
            for trace_count in range(1, 11):
                r = render_thinking(ex, vocab, trace_count_override=trace_count)
                ids2 = torch.tensor([r["input_ids"]], dtype=torch.long, device=device)
                logits2 = model(input_ids=ids2).logits[0, r["anchors"]["ans_token"]]
                pred_offset = int(logits2[vocab.numeric_ids].argmax().item())
                pred_count = pred_offset + 1
                override_rows.append(
                    {
                        "example_idx": ex_idx,
                        "prompt_count": ex.count,
                        "trace_count": trace_count,
                        "pred_count": pred_count,
                        "follows_trace": float(pred_count == trace_count),
                        "follows_prompt": float(pred_count == ex.count),
                    }
                )

    n_layer, n_head = _num_layers_heads(model)
    for ex_idx, ex in enumerate(causal_examples):
        rendered = render_thinking(ex, vocab)
        ids = torch.tensor([rendered["input_ids"]], dtype=torch.long, device=device)
        clean_logits = model(input_ids=ids).logits[0]
        a = rendered["anchors"]
        marker_positions = a["marker_positions"]
        for layer in range(1, n_layer + 1):
            for head in range(n_head):
                masked_logits = model(input_ids=ids, head_mask=_head_mask(model, device, [(layer, head)])).logits[0]
                for k, marker_pos in enumerate(marker_positions, start=1):
                    target_token = vocab.count_to_token(k + 1) if k < len(marker_positions) else "</Think>"
                    target_id = vocab.token_to_id[target_token]
                    competitors = number_ids + [close_id]
                    clean_margin = _margin(clean_logits[marker_pos], target_id, competitors)
                    masked_margin = _margin(masked_logits[marker_pos], target_id, competitors)
                    successor_causal_rows.append(
                        {
                            "example_idx": ex_idx,
                            "count": ex.count,
                            "k": k,
                            "is_final_marker": float(k == ex.count),
                            "layer": layer,
                            "head": head,
                            "target_next_token": target_token,
                            "clean_margin": clean_margin,
                            "masked_margin": masked_margin,
                            "margin_drop": clean_margin - masked_margin,
                        }
                    )

    transition = pd.DataFrame(transition_rows)
    next_index = pd.DataFrame(next_index_rows)
    answer = pd.DataFrame(answer_rows)
    override = pd.DataFrame(override_rows)
    transition_head = transition.groupby(["layer", "head"], as_index=False).mean(numeric_only=True)
    next_index_head = next_index.groupby(["layer", "head"], as_index=False).mean(numeric_only=True)
    answer_head = answer.groupby(["layer", "head"], as_index=False).mean(numeric_only=True)
    override_summary = override.groupby(["prompt_count", "trace_count"], as_index=False).mean(numeric_only=True)
    successor_causal = pd.DataFrame(successor_causal_rows)
    successor_causal_head = successor_causal.groupby(["layer", "head"], as_index=False).mean(numeric_only=True)

    group_specs = _head_group_specs(
        answer_head,
        n_layer=n_layer,
        n_head=n_head,
        seed=int(cfg.get("seed", 1234)) + seed_offset + 311,
    )
    for ex_idx, ex in enumerate(causal_examples):
        rendered = render_thinking(ex, vocab)
        ids = torch.tensor([rendered["input_ids"]], dtype=torch.long, device=device)
        ans_pos = rendered["anchors"]["ans_token"]
        clean_logits = model(input_ids=ids).logits[0, ans_pos]
        clean_margin = _count_margin(clean_logits, vocab, ex.count)
        clean_pred = _pred_count(clean_logits, vocab)
        for group_name, heads in group_specs.items():
            masked_logits = model(input_ids=ids, head_mask=_head_mask(model, device, heads)).logits[0, ans_pos]
            masked_margin = _count_margin(masked_logits, vocab, ex.count)
            masked_pred = _pred_count(masked_logits, vocab)
            answer_mask_rows.append(
                {
                    "example_idx": ex_idx,
                    "count": ex.count,
                    "group_name": group_name,
                    "n_masked_heads": len(heads),
                    "masked_heads": " ".join(f"L{layer}H{head}" for layer, head in heads),
                    "clean_margin": clean_margin,
                    "masked_margin": masked_margin,
                    "margin_drop": clean_margin - masked_margin,
                    "clean_pred": clean_pred,
                    "masked_pred": masked_pred,
                    "clean_accuracy": float(clean_pred == ex.count),
                    "masked_accuracy": float(masked_pred == ex.count),
                    "accuracy_drop": float(clean_pred == ex.count) - float(masked_pred == ex.count),
                    "pred_shift": float(masked_pred - clean_pred),
                }
            )
    answer_mask = pd.DataFrame(answer_mask_rows)
    answer_mask_summary = answer_mask.groupby(["group_name", "n_masked_heads"], as_index=False).mean(numeric_only=True)
    outputs = {
        "successor_transition_rows": transition,
        "successor_transition_head_summary": transition_head,
        "next_index_retrieval_rows": next_index,
        "next_index_retrieval_head_summary": next_index_head,
        "successor_head_ablation_rows": successor_causal,
        "successor_head_ablation_head_summary": successor_causal_head,
        "answer_trace_attention_rows": answer,
        "answer_trace_attention_head_summary": answer_head,
        "answer_multihead_mask_rows": answer_mask,
        "answer_multihead_mask_summary": answer_mask_summary,
        "trace_length_override": override,
        "trace_length_override_summary": override_summary,
    }
    for name, df in outputs.items():
        df.to_csv(tables / f"{name}.csv", index=False)
    make_plots(outputs, out_dir)
    make_report(outputs, out_dir)
    return outputs


def _heatmap(df: pd.DataFrame, metric: str, title: str, path: Path, vmin: float | None = None, vmax: float | None = None) -> None:
    import matplotlib.pyplot as plt

    if df.empty or metric not in df.columns:
        return
    mat = df.pivot(index="layer", columns="head", values=metric)
    values = mat.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(mat.columns)), labels=[str(c) for c in mat.columns])
    ax.set_yticks(range(len(mat.index)), labels=[str(i) for i in mat.index])
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    finite = values[np.isfinite(values)]
    midpoint = float(np.nanmean(finite)) if finite.size else 0.5
    for y, row in enumerate(values):
        for x, val in enumerate(row):
            if np.isfinite(val):
                ax.text(x, y, f"{val:.2f}", ha="center", va="center", color="white" if val < midpoint else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _barh(df: pd.DataFrame, metric: str, title: str, path: Path, *, top_n: int | None = None) -> None:
    import matplotlib.pyplot as plt

    if df.empty or metric not in df.columns:
        return
    data = df.sort_values(metric, ascending=True)
    if top_n is not None and len(data) > top_n:
        data = data.tail(top_n)
    labels = data["group_name"].astype(str).tolist()
    values = data[metric].astype(float).tolist()
    fig_h = max(4.0, 0.38 * len(data) + 1.2)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.barh(labels, values, color="#2f6fed")
    ax.axvline(0, color="#222", linewidth=1)
    ax.set_xlabel(metric)
    ax.set_title(title)
    for y, value in enumerate(values):
        ax.text(value, y, f" {value:.3f}", va="center", ha="left" if value >= 0 else "right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_plots(outputs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figs = out_dir / "figures"
    transition = outputs["successor_transition_head_summary"]
    next_index = outputs["next_index_retrieval_head_summary"]
    successor_causal = outputs["successor_head_ablation_head_summary"]
    answer = outputs["answer_trace_attention_head_summary"]
    answer_mask = outputs["answer_multihead_mask_summary"]
    _heatmap(transition, "next_token_margin", "Successor transition: next-index/close logit margin", figs / "successor_next_token_margin.png")
    _heatmap(transition, "current_marker_self_mass", "Successor transition: current marker self mass", figs / "successor_current_marker_self_mass.png", vmin=0)
    _heatmap(transition, "next_prompt_needle_mass", "Successor transition: next prompt needle mass", figs / "successor_next_prompt_needle_mass.png", vmin=0)
    _heatmap(next_index, "correct_prompt_needle_mass", "Next index token: correct prompt needle mass", figs / "next_index_correct_prompt_needle_mass.png", vmin=0)
    _heatmap(next_index, "correct_top1", "Next index token: correct prompt needle is top-1", figs / "next_index_correct_top1.png", vmin=0, vmax=1)
    _heatmap(successor_causal, "margin_drop", "Single-head ablation: successor margin drop", figs / "successor_margin_drop_by_head.png")
    _heatmap(answer, "all_trace_marker_mass", "Final answer: all trace marker mass", figs / "answer_all_trace_marker_mass.png", vmin=0)
    _heatmap(answer, "last_trace_marker_mass", "Final answer: last trace marker mass", figs / "answer_last_trace_marker_mass.png", vmin=0)
    _barh(answer_mask, "margin_drop", "Final answer multi-head mask: count-margin drop", figs / "answer_multihead_mask_margin_drop.png", top_n=16)
    _barh(answer_mask, "accuracy_drop", "Final answer multi-head mask: accuracy drop", figs / "answer_multihead_mask_accuracy_drop.png", top_n=16)
    override = outputs["trace_length_override_summary"]
    if not override.empty:
        mat = override.pivot(index="prompt_count", columns="trace_count", values="follows_trace")
        values = mat.to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(mat.columns)), labels=[str(c) for c in mat.columns])
        ax.set_yticks(range(len(mat.index)), labels=[str(i) for i in mat.index])
        ax.set_xlabel("forced trace count")
        ax.set_ylabel("prompt count")
        for y, row in enumerate(values):
            for x, val in enumerate(row):
                if np.isfinite(val):
                    ax.text(x, y, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Final answer follows overridden trace length")
        fig.tight_layout()
        fig.savefig(figs / "trace_length_override_follows_trace.png", dpi=180)
        plt.close(fig)


def _img(name: str) -> str:
    return f"<img src='../figures/{name}' style='max-width:100%;border:1px solid #ddd;border-radius:8px'>"


def make_report(outputs: dict[str, pd.DataFrame], out_dir: Path) -> None:
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    trans = outputs["successor_transition_head_summary"].sort_values("next_token_margin", ascending=False)
    next_idx = outputs["next_index_retrieval_head_summary"].sort_values("correct_top1", ascending=False)
    succ_causal = outputs["successor_head_ablation_head_summary"].sort_values("margin_drop", ascending=False)
    ans = outputs["answer_trace_attention_head_summary"].sort_values("all_trace_marker_mass", ascending=False)
    ans_mask = outputs["answer_multihead_mask_summary"].sort_values("margin_drop", ascending=False)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>v2.2 follow-up mechanism</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;line-height:1.55;max-width:1150px;margin:32px auto;padding:0 20px;color:#172033}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ddd;padding:8px}}th{{background:#f4f7fb}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}code{{background:#eef2f7;padding:2px 5px;border-radius:5px}}</style>
</head><body>
<h1>v2.2 follow-up: successor localization and aggregation</h1>
<p>This report asks what happens after L3 targeted retrieval. It separates three claims: (1) descriptive attention at <code>marker_k</code>; (2) descriptive retrieval at <code>index_token_{{k+1}}</code>; (3) causal evidence from single-head and multi-head masks.</p>
<h2>Descriptive successor transition</h2>
<p>The raw next-token logit margin is a model-level quantity and is copied across heads; it is retained as a sanity check, not as head-specific evidence.</p>
<div class="grid"><div>{_img('successor_next_token_margin.png')}</div><div>{_img('successor_current_marker_self_mass.png')}</div><div>{_img('successor_next_prompt_needle_mass.png')}</div><div>{_img('next_index_correct_prompt_needle_mass.png')}</div><div>{_img('next_index_correct_top1.png')}</div><div>{_img('successor_margin_drop_by_head.png')}</div></div>
<h2>Final answer aggregation</h2>
<div class="grid"><div>{_img('answer_all_trace_marker_mass.png')}</div><div>{_img('answer_last_trace_marker_mass.png')}</div><div>{_img('answer_multihead_mask_margin_drop.png')}</div><div>{_img('answer_multihead_mask_accuracy_drop.png')}</div><div>{_img('trace_length_override_follows_trace.png')}</div></div>
<h2>Top successor-transition descriptive heads</h2>{trans.head(12).to_html(index=False)}
<h2>Top next-index retrieval heads</h2>{next_idx.head(12).to_html(index=False)}
<h2>Top successor causal single-head ablations</h2>{succ_causal.head(12).to_html(index=False)}
<h2>Top final-answer trace-attention heads</h2>{ans.head(12).to_html(index=False)}
<h2>Final-answer multi-head masks</h2>{ans_mask.head(20).to_html(index=False)}
</body></html>"""
    (report_dir / "report.html").write_text(html, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v2.2 follow-up mechanism diagnostics")
    p.add_argument("--v2-run-dir", required=True)
    p.add_argument("--examples-per-count", type=int, default=50)
    p.add_argument("--causal-examples-per-count", type=int, default=None)
    p.add_argument("--device", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_v2_2_followup(
        args.v2_run_dir,
        examples_per_count=args.examples_per_count,
        causal_examples_per_count=args.causal_examples_per_count,
        device=args.device,
    )


if __name__ == "__main__":
    main()
