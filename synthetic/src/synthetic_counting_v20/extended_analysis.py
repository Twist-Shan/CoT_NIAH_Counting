"""Supplemental high-power behavior and interactive attention diagnostics for v20."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .attention_metrics import broad_profile_metrics as _broad_profile_metrics
from .config import V20Config, config_from_dict
from .data import V20Example, V20Rendered, V20Vocab, collate_v20, load_corpus_split, load_corpus_text, load_suite_manifests, render_v20
from .model import build_model
from .needle_pool import load_needle_pool
from .training import atomic_csv, checkpoint_steps


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _balanced_split(
    examples: Sequence[V20Example], count_max: int, per_count: int, *, offset: int = 0
) -> list[V20Example]:
    result: list[V20Example] = []
    for count in range(1, count_max + 1):
        bucket = [item for item in examples if int(item.count or 0) == count]
        selected = bucket[offset : offset + per_count]
        if len(selected) != per_count:
            raise ValueError(f"count={count}: need {offset + per_count} held-out examples")
        result.extend(selected)
    return result


@torch.inference_mode()
def _broad_metric_matrices(
    model,
    cfg: V20Config,
    vocab: V20Vocab,
    items: Sequence[V20Rendered],
) -> tuple[dict[str, np.ndarray], int]:
    totals = {
        name: np.zeros((cfg.n_layer, cfg.n_head), dtype=np.float64)
        for name in (
            "total_target_mass",
            "effective_number",
            "effective_coverage",
            "normalized_entropy",
            "broad_score",
            "entropy_broad_score",
            "legacy_entropy_broad_score",
        )
    }
    observations = 0
    batch_size = min(8, cfg.analysis_batch_size)
    for start in range(0, len(items), batch_size):
        batch = list(items[start : start + batch_size])
        ids, _, mask = collate_v20(batch, vocab, cfg.device)
        output = model(input_ids=ids, attention_mask=mask, output_attentions=True)
        assert output.attentions is not None
        for row, item in enumerate(batch):
            assert item.spans is not None
            needles = list(item.prompt_needle_positions)
            for layer, weights in enumerate(output.attentions):
                values = (
                    weights[row, :, item.spans.ans_pos, needles]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                metrics = _broad_profile_metrics(values)
                for name, metric_values in metrics.items():
                    totals[name][layer] += metric_values
            observations += 1
    denominator = max(observations, 1)
    return {name: values / denominator for name, values in totals.items()}, observations


@torch.inference_mode()
def _broad_score_matrix(
    model,
    cfg: V20Config,
    vocab: V20Vocab,
    items: Sequence[V20Rendered],
) -> tuple[np.ndarray, int]:
    """Backward-compatible wrapper returning the effective-coverage score."""

    metrics, observations = _broad_metric_matrices(model, cfg, vocab, items)
    return metrics["broad_score"], observations


def _best_head(matrix: np.ndarray) -> tuple[int, int]:
    layer, head = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
    return int(layer + 1), int(head)


def collect_dense_attention_roles(run_dir: str | Path, *, device: str | None = None) -> Path:
    """Aggregate four head-role maps over all saved scientific checkpoints."""

    run_dir = Path(run_dir).resolve()
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    from dataclasses import replace

    cfg = replace(cfg, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = V20Vocab.load(run_dir / "vocab.json")
    corpus = load_corpus_text()
    split = load_corpus_split(run_dir / "data/corpus_split.json", cfg, corpus)
    pool = load_needle_pool(
        run_dir / "data/needle_pool.json",
        cfg,
        split_fingerprint=split.split_fingerprint,
        vocab_fingerprint=vocab.fingerprint,
    )
    curves, _ = load_suite_manifests(
        run_dir / "data/loss_suite_manifests.json",
        split_fingerprint=split.split_fingerprint,
        pool_fingerprint=pool.pool_fingerprint,
    )
    heldout = list(curves["heldout"]["task"])
    selection = _balanced_split(
        heldout, cfg.count_max_threshold, cfg.phase_head_selection_examples_per_count
    )
    reporting = _balanced_split(
        heldout,
        cfg.count_max_threshold,
        cfg.phase_examples_per_count,
        offset=cfg.phase_head_selection_examples_per_count,
    )

    fixed_roles = json.loads(
        (run_dir / "analysis/phase_transition/fixed_head_roles.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    broad_fixed: dict[str, dict[str, int]] = {}
    for mode in ("nonthinking", "thinking"):
        entries = checkpoint_steps(run_dir, "rope", mode)
        if not entries:
            raise FileNotFoundError(f"no dense snapshots for rope/{mode}")
        by_shard: dict[Path, list[int]] = {}
        for step, shard in entries:
            by_shard.setdefault(shard, []).append(step)
        model = build_model(cfg, vocab, "rope", cfg.device).eval()
        final_shard = entries[-1][1]
        payload = torch.load(final_shard, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dicts"][str(cfg.train_steps)])
        selection_matrix, _ = _broad_score_matrix(
            model,
            cfg,
            vocab,
            [render_v20(example, vocab, mode) for example in selection],
        )
        fixed_layer, fixed_head = _best_head(selection_matrix)
        role = f"{mode}_broad"
        broad_fixed[role] = {"layer": fixed_layer, "head": fixed_head}
        del payload

        reporting_items = [render_v20(example, vocab, mode) for example in reporting]
        for shard, steps in by_shard.items():
            payload = torch.load(shard, map_location="cpu", weights_only=False)
            for step in sorted(steps):
                model.load_state_dict(payload["model_state_dicts"][str(step)])
                metric_matrices, observations = _broad_metric_matrices(
                    model, cfg, vocab, reporting_items
                )
                matrix = metric_matrices["broad_score"]
                for layer in range(cfg.n_layer):
                    for head in range(cfg.n_head):
                        rows.append(
                            {
                                "step": int(step),
                                "role": role,
                                "mode": mode,
                                "layer": layer + 1,
                                "head": head,
                                "score": float(matrix[layer, head]),
                                "broad_score": float(matrix[layer, head]),
                                "total_target_mass": float(
                                    metric_matrices["total_target_mass"][layer, head]
                                ),
                                "effective_number": float(
                                    metric_matrices["effective_number"][layer, head]
                                ),
                                "effective_coverage": float(
                                    metric_matrices["effective_coverage"][layer, head]
                                ),
                                "normalized_entropy": float(
                                    metric_matrices["normalized_entropy"][layer, head]
                                ),
                                "entropy_broad_score": float(
                                    metric_matrices["entropy_broad_score"][layer, head]
                                ),
                                "legacy_entropy_broad_score": float(
                                    metric_matrices["legacy_entropy_broad_score"][
                                        layer, head
                                    ]
                                ),
                                "is_fixed_role_head": float(
                                    (layer + 1, head) == (fixed_layer, fixed_head)
                                ),
                                "observations": observations,
                                "selection_split": "disjoint_final_checkpoint",
                                "reporting_split": "heldout_reporting",
                                "score_definition": (
                                    "mean(total_target_mass*exp(shannon_entropy)"
                                    "/num_target_occurrences)"
                                ),
                            }
                        )
            del payload
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del model

    existing = pd.read_csv(
        run_dir / "analysis/phase_transition/tables/dense_fixed_head_dynamics.csv"
    ).copy()
    existing["mode"] = "thinking"
    existing["selection_split"] = "disjoint_final_checkpoint"
    existing["reporting_split"] = "heldout_reporting"
    combined = pd.concat((pd.DataFrame(rows), existing), ignore_index=True, sort=False)
    role_order = {
        "nonthinking_broad": 0,
        "thinking_broad": 1,
        "targeted_retrieval": 2,
        "marker_successor": 3,
    }
    combined["_order"] = combined["role"].map(role_order)
    combined = combined.sort_values(["step", "_order", "layer", "head"]).drop(columns="_order")
    output_dir = run_dir / "analysis/extended"
    table_path = output_dir / "tables/attention_role_dynamics.csv"
    atomic_csv(combined, table_path)
    roles = {
        **broad_fixed,
        "targeted_retrieval": fixed_roles["targeted_retrieval"],
        "marker_successor": fixed_roles["marker_successor"],
    }
    _atomic_text(output_dir / "fixed_attention_roles.json", json.dumps(roles, indent=2))
    return output_dir




__all__ = [
    "_broad_profile_metrics",
    "collect_dense_attention_roles",
]
