from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data import BaseExample, count_bin
from .render import RenderedExample, render_for_model
from .vocab import Vocab


@dataclass(frozen=True)
class Anchor:
    anchor_name: str
    position: int
    anchor_k: int | None = None
    prefix_count: int | None = None
    leakage_prone: bool = False


@dataclass
class HiddenCache:
    metadata: pd.DataFrame
    hidden: np.ndarray

    @property
    def d_model(self) -> int:
        return int(self.hidden.shape[1]) if self.hidden.ndim == 2 else 0


def hook_layer(hook_name: str, n_layers: int) -> int:
    if hook_name == "embed":
        return -1
    if hook_name == "final_norm":
        return int(n_layers)
    return int(hook_name.rsplit("_", 1)[1])


def extract_anchors(rendered: RenderedExample, example: BaseExample, model_type: str) -> list[Anchor]:
    spans = rendered.spans
    anchors: list[Anchor] = []
    for k, pos in enumerate(rendered.prompt_needle_token_positions, start=1):
        anchors.append(Anchor("prompt_marker_k", pos, anchor_k=k, prefix_count=k))
    if model_type == "non_thinking":
        anchors.extend(
            [
                Anchor("last_prompt_token", spans.ans_pos - 1),
                Anchor("pre_ans_pos", spans.ans_pos - 1),
                Anchor("ans_token", spans.ans_pos),
            ]
        )
        return anchors

    if spans.think_open_pos is None or spans.think_close_pos is None:
        return anchors
    anchors.extend(
        [
            Anchor("think_start", spans.think_open_pos),
            Anchor("think_end", spans.think_close_pos),
            Anchor("pre_ans_pos", spans.ans_pos - 1),
            Anchor("ans_token", spans.ans_pos),
        ]
    )
    for k, marker_pos in enumerate(spans.trace_marker_positions, start=1):
        index_pos = spans.trace_index_positions[k - 1]
        pre_index_pos = spans.think_open_pos if k == 1 else spans.trace_marker_positions[k - 2]
        anchors.append(Anchor("pre_index_k", pre_index_pos, anchor_k=k, prefix_count=k))
        anchors.append(Anchor("index_k_pos", index_pos, anchor_k=k, prefix_count=k, leakage_prone=True))
        anchors.append(Anchor("marker_k_pos", marker_pos, anchor_k=k, prefix_count=k))
        anchors.append(Anchor("post_marker_k", marker_pos, anchor_k=k, prefix_count=k))
    return anchors


def _hook_names(n_layers: int) -> list[str]:
    names = ["embed"]
    for layer_idx in range(n_layers):
        names.append(f"resid_pre_layer_{layer_idx}")
        names.append(f"resid_post_layer_{layer_idx}")
    names.append("final_norm")
    return names


@torch.no_grad()
def collect_hidden_cache(
    models: dict[str, Any],
    examples: list[BaseExample],
    vocab: Vocab,
    cfg: Any,
) -> HiddenCache:
    rows: list[dict[str, Any]] = []
    hidden_rows: list[np.ndarray] = []
    n_layers = int(cfg.model.n_layer)
    expected_hooks = set(_hook_names(n_layers))
    for model_type, model in models.items():
        model.eval()
        eval_mode = "direct" if model_type == "non_thinking" else "oracle_trace"
        for example in examples:
            rendered = render_for_model(example, vocab, model_type)
            input_ids = torch.tensor([rendered.input_ids], dtype=torch.long, device=cfg.device)
            attention_mask = (input_ids != vocab.pad_id).long()
            out = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            if out.hidden_states is None:
                continue
            anchors = extract_anchors(rendered, example, model_type)
            for hook_name, states in out.hidden_states.items():
                if hook_name not in expected_hooks:
                    continue
                layer = hook_layer(hook_name, n_layers)
                for anchor in anchors:
                    if not (0 <= anchor.position < states.size(1)):
                        continue
                    token_id = int(input_ids[0, anchor.position].detach().cpu())
                    rows.append(
                        {
                            "row_id": len(rows),
                            "example_id": example.example_id,
                            "model_type": model_type,
                            "eval_mode": eval_mode,
                            "count": int(example.count),
                            "final_count": int(example.count),
                            "count_bin": count_bin(example.count),
                            "seq_len": int(example.seq_len),
                            "anchor_name": anchor.anchor_name,
                            "anchor_k": anchor.anchor_k if anchor.anchor_k is not None else "",
                            "layer": int(layer),
                            "hook_name": hook_name,
                            "position": int(anchor.position),
                            "absolute_token_id": token_id,
                            "token_string": vocab.id_to_token[token_id],
                            "prefix_count": anchor.prefix_count if anchor.prefix_count is not None else "",
                            "trace_length_tokens": int(2 * example.count) if model_type == "thinking" else "",
                            "leakage_prone": bool(anchor.leakage_prone),
                        }
                    )
                    hidden_rows.append(states[0, anchor.position].detach().cpu().numpy().astype(np.float32))
    hidden = np.stack(hidden_rows, axis=0) if hidden_rows else np.zeros((0, int(cfg.model.n_embd)), dtype=np.float32)
    return HiddenCache(pd.DataFrame(rows), hidden)


def save_hidden_cache(cache: HiddenCache, run_dir: Path) -> None:
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.metadata.to_csv(cache_dir / "hidden_cache_metadata.csv", index=False)
    np.save(cache_dir / "hidden_cache.npy", cache.hidden)


def load_hidden_cache(run_dir: Path) -> HiddenCache:
    cache_dir = run_dir / "cache"
    metadata = pd.read_csv(cache_dir / "hidden_cache_metadata.csv")
    hidden = np.load(cache_dir / "hidden_cache.npy")
    return HiddenCache(metadata, hidden)
