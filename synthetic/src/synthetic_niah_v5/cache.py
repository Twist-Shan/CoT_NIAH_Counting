from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import balanced_examples, render_nonthinking, render_thinking
from .model import make_model
from .train import load_checkpoint
from .vocab import Vocab


def final_checkpoint_path(run_dir: Path) -> Path:
    final_path = run_dir / "checkpoints" / "final.pt"
    if final_path.exists():
        return final_path
    step_paths = sorted((run_dir / "checkpoints").glob("step_*.pt"))
    if not step_paths:
        raise FileNotFoundError("No checkpoints available for cache/probe/attention.")
    return step_paths[-1]


def _records_for_rendered(rendered, hidden_states, example_id: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    anchors: list[tuple[str, int, str, int, bool]] = [
        ("mode_pos", rendered.spans.mode_pos, "final_count", 0, False),
        ("think_open_pos", rendered.spans.think_open_pos, "final_count", 0, False),
        ("think_close_pos", rendered.spans.think_close_pos, "final_count", 0, False),
        ("pre_count_pos", rendered.spans.pre_count_pos, "final_count", 0, False),
        ("count_pos", rendered.spans.count_pos, "final_count", 0, True),
    ]
    for idx, pos in enumerate(rendered.prompt_needle_token_positions, start=1):
        anchors.append((f"prompt_marker_{idx}", pos, "final_count", 0, False))
    if rendered.variant == "thinking":
        for idx, pos in enumerate(rendered.spans.trace_marker_positions, start=1):
            anchors.append((f"trace_marker_{idx}", pos, "prefix_count", idx, False))
            if pos + 1 < len(rendered.input_ids):
                anchors.append((f"post_trace_marker_{idx}", pos + 1, "prefix_count", idx, False))
    for layer, hidden in enumerate(hidden_states):
        h = hidden[0].detach().cpu().numpy()
        for anchor_name, pos, target, prefix_value, leakage in anchors:
            if 0 <= pos < h.shape[0]:
                records.append(
                    {
                        "example_id": example_id,
                        "mode": rendered.variant,
                        "anchor_name": anchor_name,
                        "target": target,
                        "target_value": prefix_value if target == "prefix_count" else len(rendered.gold_trace_markers),
                        "layer": layer,
                        "hook_name": "hidden_state",
                        "position": int(pos),
                        "trace_len": len(rendered.gold_trace_markers),
                        "leakage_prone": bool(leakage),
                    }
                )
    return records


@torch.no_grad()
def _collect_one(model, rendered, device: str | torch.device) -> tuple[list[dict[str, Any]], np.ndarray]:
    input_ids = torch.tensor([rendered.input_ids], dtype=torch.long, device=device)
    out = model(input_ids=input_ids, output_hidden_states=True)
    hidden_states = list(out.hidden_states or [])
    vectors: list[np.ndarray] = []
    rows = _records_for_rendered(rendered, hidden_states, example_id=0)
    for row in rows:
        vectors.append(hidden_states[int(row["layer"])][0, int(row["position"])].detach().cpu().numpy())
    return rows, np.stack(vectors) if vectors else np.empty((0, int(model.config.n_embd)))


@torch.no_grad()
def run_cache(cfg: dict[str, Any], vocab: Vocab, run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = run_dir / "cache"
    tables = run_dir / "tables"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    examples = balanced_examples(
        int(cfg["train"]["seq_len"]),
        int(cfg["train"]["probe_examples_per_count"]),
        int(cfg["train"]["seed"]) + 7000,
        int(cfg["train"]["count_min"]),
        int(cfg["train"]["count_max"]),
    )
    model = make_model(cfg["model"], cfg["device"])
    load_checkpoint(model, final_checkpoint_path(run_dir), cfg["device"])
    model.eval()
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for example_id, ex in enumerate(examples):
        for rendered in [
            render_thinking(ex, vocab, trace_indices=bool(cfg["trace_indices"])),
            render_nonthinking(ex, vocab),
        ]:
            local_rows, local_vectors = _collect_one(model, rendered, cfg["device"])
            start_idx = len(vectors)
            for offset, row in enumerate(local_rows):
                row["example_id"] = example_id
                row["hidden_index"] = start_idx + offset
                rows.append(row)
            vectors.extend(list(local_vectors))
    index = pd.DataFrame(rows)
    hidden = np.stack(vectors) if vectors else np.empty((0, int(cfg["model"]["n_embd"])))
    np.savez_compressed(cache_dir / "hidden_cache.npz", hidden=hidden)
    index.to_csv(tables / "hidden_cache_index.csv", index=False)

    sim_rows: list[dict[str, Any]] = []
    for example_id, ex in enumerate(examples):
        think = render_thinking(ex, vocab, trace_indices=bool(cfg["trace_indices"]))
        non = render_nonthinking(ex, vocab)
        t_ids = torch.tensor([think.input_ids[: think.spans.think_close_pos + 1]], dtype=torch.long, device=cfg["device"])
        n_ids = torch.tensor([non.input_ids[: non.spans.think_close_pos + 1]], dtype=torch.long, device=cfg["device"])
        t_out = model(input_ids=t_ids, output_hidden_states=True)
        n_out = model(input_ids=n_ids, output_hidden_states=True)
        for layer, (t_h, n_h) in enumerate(zip(t_out.hidden_states or [], n_out.hidden_states or [])):
            t_vec = t_h[0, -1]
            n_vec = n_h[0, -1]
            sim_rows.append(
                {
                    "example_id": example_id,
                    "anchor_name": "think_close_pos",
                    "layer": layer,
                    "cosine_similarity": float(F.cosine_similarity(t_vec, n_vec, dim=0).detach().cpu()),
                    "count": ex.count,
                }
            )
    sim = pd.DataFrame(sim_rows)
    sim.to_csv(tables / "mode_hidden_similarity.csv", index=False)
    return index, sim
