from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .cache import extract_anchors
from .data import BaseExample
from .generation import predict_count_from_prefixes
from .hooks import ReplacementPatchSpec, make_replacement_hook
from .render import non_thinking_eval_prefix, render_for_model, thinking_oracle_trace_prefix
from .vocab import Vocab


PATCH_COLUMNS = [
    "model_type",
    "eval_mode",
    "anchor_name",
    "anchor_k",
    "hook_name",
    "layer",
    "donor_count",
    "receiver_count",
    "base_pred",
    "donor_pred",
    "patched_pred",
    "patched_moves_toward_donor",
    "logit_recovery_toward_donor_count",
    "causal_effect_size",
    "n_examples",
]

FINAL_PATCH_ANCHORS = {"ans_token", "pre_ans_pos", "last_prompt_token", "think_end", "think_start"}
TRACE_PATCH_ANCHORS = {"pre_index_k", "index_k_pos", "marker_k_pos", "post_marker_k", "prompt_marker_k"}


def _prefix(example: BaseExample, model_type: str, vocab: Vocab) -> list[int]:
    if model_type == "non_thinking":
        return non_thinking_eval_prefix(example, vocab)
    return thinking_oracle_trace_prefix(example, vocab)


def _anchor_position(example: BaseExample, model_type: str, anchor_name: str, anchor_k: Any, vocab: Vocab) -> int | None:
    rendered = render_for_model(example, vocab, model_type)
    want_k = None
    if str(anchor_k).strip() not in {"", "nan", "None"}:
        try:
            want_k = int(anchor_k)
        except ValueError:
            want_k = None
    for anchor in extract_anchors(rendered, example, model_type):
        if anchor.anchor_name != anchor_name:
            continue
        if want_k is not None and anchor.anchor_k != want_k:
            continue
        return anchor.position
    return None


@torch.no_grad()
def _hidden_at(model, prefix: list[int], hook_name: str, position: int, vocab: Vocab, device: str):
    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    out = model(input_ids, attention_mask=(input_ids != vocab.pad_id).long(), output_hidden_states=True)
    if out.hidden_states is None or hook_name not in out.hidden_states:
        return None
    return out.hidden_states[hook_name][0, position].detach().cpu()


def _select_patch_configs(run_dir: Path, limit: int = 6) -> pd.DataFrame:
    artifacts = run_dir / "artifacts"
    tables = run_dir / "tables"
    direction_path = artifacts / "directions.csv"
    metrics_path = tables / "direction_metrics.csv"
    if not direction_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(direction_path)
    if metrics_path.exists() and metrics_path.stat().st_size > 0:
        metrics = pd.read_csv(metrics_path)
        if "projection_r2" in metrics.columns:
            key_cols = ["model_type", "eval_mode", "anchor_name", "anchor_k", "hook_name", "layer", "direction_type", "target"]
            left = df.copy()
            right = metrics.copy()
            for col in key_cols:
                if col not in left.columns:
                    left[col] = ""
                if col not in right.columns:
                    right[col] = ""
                left[f"__key_{col}"] = left[col].fillna("").astype(str)
                right[f"__key_{col}"] = right[col].fillna("").astype(str)
            merge_cols = [f"__key_{col}" for col in key_cols]
            keep = merge_cols + ["projection_r2"]
            df = left.merge(right[keep].drop_duplicates(merge_cols), on=merge_cols, how="left").drop(columns=merge_cols)
    if "projection_r2" not in df.columns:
        df["projection_r2"] = 0.0
    final_sub = df[
        df["target"].eq("final_count")
        & df["anchor_name"].isin(FINAL_PATCH_ANCHORS)
    ]
    trace_sub = df[
        df["target"].eq("prefix_count")
        & df["anchor_name"].isin(TRACE_PATCH_ANCHORS)
    ]
    sub = pd.concat([final_sub, trace_sub], ignore_index=True)
    resid = sub[sub["hook_name"].astype(str).str.startswith("resid_post_layer_")]
    if not resid.empty:
        sub = resid
    if sub.empty:
        sub = df.copy()
    return (
        sub.sort_values("projection_r2", ascending=False)
        .drop_duplicates(["model_type", "eval_mode", "anchor_name", "anchor_k", "hook_name", "layer"])
        .head(limit)
    )


def run_patching(models: dict[str, Any], examples: list[BaseExample], vocab: Vocab, cfg: Any, run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    configs = _select_patch_configs(run_dir, limit=max(1, int(cfg.steering.max_direction_configs)))
    if configs.empty:
        out = pd.DataFrame(columns=PATCH_COLUMNS)
        out.to_csv(run_dir / "tables" / "interchange_patching_results.csv", index=False)
        return out
    by_count: dict[int, list[BaseExample]] = {}
    for ex in examples:
        by_count.setdefault(ex.count, []).append(ex)
    pairs: list[tuple[BaseExample, BaseExample]] = []
    counts = sorted(by_count)
    for receiver_count in counts:
        for donor_count in counts:
            if receiver_count == donor_count:
                continue
            pairs.append((by_count[receiver_count][0], by_count[donor_count][0]))
    if not pairs and len(examples) >= 2:
        pairs = [(examples[0], examples[-1])]

    for _, cfg_row in configs.iterrows():
        model_type = str(cfg_row["model_type"])
        model = models[model_type]
        hook_name = str(cfg_row["hook_name"])
        anchor_name = str(cfg_row["anchor_name"])
        anchor_k = cfg_row.get("anchor_k", "")
        for receiver, donor in pairs[:90]:
            receiver_pos = _anchor_position(receiver, model_type, anchor_name, anchor_k, vocab)
            donor_pos = _anchor_position(donor, model_type, anchor_name, anchor_k, vocab)
            if receiver_pos is None or donor_pos is None:
                continue
            donor_hidden = _hidden_at(model, _prefix(donor, model_type, vocab), hook_name, donor_pos, vocab, cfg.device)
            if donor_hidden is None:
                continue
            receiver_prefix = _prefix(receiver, model_type, vocab)
            donor_prefix = _prefix(donor, model_type, vocab)
            base = predict_count_from_prefixes(model, [receiver_prefix], [receiver.count], vocab, cfg.device, batch_size=1)[0]
            donor_pred = predict_count_from_prefixes(model, [donor_prefix], [donor.count], vocab, cfg.device, batch_size=1)[0]
            hook = make_replacement_hook(ReplacementPatchSpec(hook_name=hook_name, position=receiver_pos, donor_hidden=donor_hidden))
            patched = predict_count_from_prefixes(model, [receiver_prefix], [receiver.count], vocab, cfg.device, batch_size=1, hook_fn=hook)[0]
            base_pred = int(base["pred_count"])
            donor_count = int(donor.count)
            patched_pred = int(patched["pred_count"])
            base_dist = abs(base_pred - donor_count)
            patched_dist = abs(patched_pred - donor_count)
            rows.append(
                {
                    "model_type": model_type,
                    "eval_mode": str(cfg_row["eval_mode"]),
                    "anchor_name": anchor_name,
                    "anchor_k": anchor_k,
                    "hook_name": hook_name,
                    "layer": int(cfg_row["layer"]),
                    "donor_count": donor.count,
                    "receiver_count": receiver.count,
                    "base_pred": base_pred,
                    "donor_pred": int(donor_pred["pred_count"]),
                    "patched_pred": patched_pred,
                    "patched_moves_toward_donor": bool(patched_dist < base_dist),
                    "logit_recovery_toward_donor_count": float(patched["gold_logit"] - base["gold_logit"]),
                    "causal_effect_size": float(patched_pred - base_pred),
                    "n_examples": 1,
                }
            )
    out = pd.DataFrame(rows, columns=PATCH_COLUMNS)
    out.to_csv(run_dir / "tables" / "interchange_patching_results.csv", index=False)
    return out
