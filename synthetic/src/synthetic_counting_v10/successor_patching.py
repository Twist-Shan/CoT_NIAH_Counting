from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm

from .attention_causal import (
    Head,
    _capture_cproj_inputs,
    _head_label,
    _patched_head_forward,
    normalized_recovery,
)
from .config import V10Config
from .core import Example, Vocab, count_bin, render
from .report_followups import load_rankings, load_run
from .training import load_final_model


TOP_NS = (1, 2, 4, 8, 12, 16)
DIRECTION_CONTINUE = "continue_into_close"
DIRECTION_CLOSE = "close_into_continue"


def successor_nested_example_pair(
    cfg: V10Config,
    vocab: Vocab,
    k: int,
    seed: int,
) -> tuple[Example, Example]:
    """Return count-k and count-(k+1) prompts sharing the first k needles.

    The extra needle is after every shared needle. Consequently, the two
    teacher-forced traces are token-identical through marker M_k, while the
    correct token after M_k differs: the short prompt closes and the long
    prompt continues with numeric token <k+1>.
    """
    k = int(k)
    if not cfg.count_min <= k < cfg.count_max:
        raise ValueError(f"successor k must be in {cfg.count_min}..{cfg.count_max - 1}")
    rng = random.Random(int(seed))
    positions = sorted(rng.sample(range(cfg.seq_len), k + 1))
    shared_positions = positions[:k]
    extra_position = positions[k]
    markers = [rng.choice(vocab.markers) for _ in range(k + 1)]
    base_noise = [rng.choice(vocab.noise) for _ in range(cfg.seq_len)]

    def build(count: int, selected_positions: list[int], selected_markers: list[str]) -> Example:
        sequence = list(base_noise)
        for position, marker in zip(selected_positions, selected_markers):
            sequence[position] = marker
        return Example(
            sequence,
            int(count),
            list(selected_positions),
            list(selected_markers),
            int(seed),
        )

    short = build(k, shared_positions, markers[:k])
    long = build(k + 1, [*shared_positions, extra_position], markers)
    return short, long


def _all_heads(cfg: V10Config) -> list[Head]:
    return [(layer, head) for layer in range(cfg.n_layer) for head in range(cfg.n_head)]


def _two_token_margin(logits: torch.Tensor, target_id: int, alternative_id: int) -> float:
    values = logits.detach().float().cpu()
    return float(values[int(target_id)] - values[int(alternative_id)])


def _ranking_families(
    cfg: V10Config,
    rankings: dict[str, list[Head]],
    random_replicates: int,
) -> list[tuple[str, int, list[Head], str]]:
    families: list[tuple[str, int, list[Head], str]] = [
        ("successor_top", 0, rankings["successor"], "marker_query"),
        ("targeted_top", 0, rankings["targeted_retrieval"], "marker_query"),
        # Same selected slices, but copied from the preceding <k> row. This is
        # a row-specificity control rather than a different head ranking.
        ("successor_wrong_row", 0, rankings["successor"], "index_query"),
    ]
    all_heads = _all_heads(cfg)
    for replicate in range(int(random_replicates)):
        shuffled = list(all_heads)
        random.Random(cfg.seed + 970_000 + replicate).shuffle(shuffled)
        families.append(("random", replicate, shuffled, "marker_query"))
    return families


@torch.no_grad()
def run_successor_patching_rows(
    model,
    cfg: V10Config,
    vocab: Vocab,
    rankings: dict[str, list[Head]],
    *,
    examples_per_k: int = 2,
    random_replicates: int = 4,
    top_ns: tuple[int, ...] = TOP_NS,
) -> pd.DataFrame:
    """Patch local head outputs at M_k and test continue-versus-close decisions."""
    families = _ranking_families(cfg, rankings, random_replicates)
    rows: list[dict[str, Any]] = []
    k_values = range(cfg.count_min, cfg.count_max)
    total = len(k_values) * int(examples_per_k) * 2 * len(families) * len(top_ns)
    progress = tqdm(total=total, desc="v10 successor patching")
    layers = set(range(cfg.n_layer))
    model_device = next(model.parameters()).device

    for k in k_values:
        for example_index in range(int(examples_per_k)):
            seed = cfg.seed + 971_000 + k * 1_000 + example_index
            short_example, long_example = successor_nested_example_pair(cfg, vocab, k, seed)
            short_item = render(short_example, vocab, "thinking")
            long_item = render(long_example, vocab, "thinking")
            short_query = int(short_item.spans.trace_marker_positions[k - 1])
            long_query = int(long_item.spans.trace_marker_positions[k - 1])
            short_index_query = int(short_item.spans.trace_index_positions[k - 1])
            long_index_query = int(long_item.spans.trace_index_positions[k - 1])
            if short_query != long_query or short_index_query != long_index_query:
                raise AssertionError("nested traces must align through M_k")
            short_trace_prefix = short_item.tokens[short_item.spans.think_pos : short_query + 1]
            long_trace_prefix = long_item.tokens[long_item.spans.think_pos : long_query + 1]
            if short_trace_prefix != long_trace_prefix:
                raise AssertionError("nested teacher-forced traces must be identical through M_k")

            short_ids = torch.tensor([short_item.input_ids], dtype=torch.long, device=model_device)
            long_ids = torch.tensor([long_item.input_ids], dtype=torch.long, device=model_device)
            short_logits = model(input_ids=short_ids).logits[0, short_query]
            long_logits = model(input_ids=long_ids).logits[0, long_query]
            short_donor = _capture_cproj_inputs(model, short_ids, layers)
            long_donor = _capture_cproj_inputs(model, long_ids, layers)
            continue_id = vocab.number_id(k + 1)
            close_id = vocab.think_close_id

            directions = (
                {
                    "direction": DIRECTION_CONTINUE,
                    "clean_ids": long_ids,
                    "receiver_ids": short_ids,
                    "clean_logits": long_logits,
                    "corrupt_logits": short_logits,
                    "donor_inputs": long_donor,
                    "marker_donor_pos": long_query,
                    "index_donor_pos": long_index_query,
                    "receiver_pos": short_query,
                    "target_id": continue_id,
                    "alternative_id": close_id,
                },
                {
                    "direction": DIRECTION_CLOSE,
                    "clean_ids": short_ids,
                    "receiver_ids": long_ids,
                    "clean_logits": short_logits,
                    "corrupt_logits": long_logits,
                    "donor_inputs": short_donor,
                    "marker_donor_pos": short_query,
                    "index_donor_pos": short_index_query,
                    "receiver_pos": long_query,
                    "target_id": close_id,
                    "alternative_id": continue_id,
                },
            )

            for spec in directions:
                clean_margin = _two_token_margin(
                    spec["clean_logits"], spec["target_id"], spec["alternative_id"]
                )
                corrupt_margin = _two_token_margin(
                    spec["corrupt_logits"], spec["target_id"], spec["alternative_id"]
                )
                for family, replicate, ranking, donor_row in families:
                    donor_pos = (
                        spec["marker_donor_pos"]
                        if donor_row == "marker_query"
                        else spec["index_donor_pos"]
                    )
                    for top_n in top_ns:
                        heads = ranking[: int(top_n)]
                        patched_logits = _patched_head_forward(
                            model,
                            spec["receiver_ids"],
                            spec["donor_inputs"],
                            heads,
                            int(donor_pos),
                            int(spec["receiver_pos"]),
                        )[int(spec["receiver_pos"])]
                        patched_margin = _two_token_margin(
                            patched_logits, spec["target_id"], spec["alternative_id"]
                        )
                        rows.append(
                            {
                                "decision_k": k,
                                "count_bin": count_bin(k),
                                "example_index": example_index,
                                "direction": spec["direction"],
                                "family": family,
                                "replicate": replicate,
                                "top_n": int(top_n),
                                "patched_heads": _head_label(heads),
                                "donor_row": donor_row,
                                "donor_position": int(donor_pos),
                                "receiver_position": int(spec["receiver_pos"]),
                                "clean_margin": clean_margin,
                                "corrupt_margin": corrupt_margin,
                                "patched_margin": patched_margin,
                                "normalized_recovery": normalized_recovery(
                                    clean_margin, corrupt_margin, patched_margin
                                ),
                                "clean_target_correct": float(clean_margin > 0),
                                "corrupt_target_correct": float(corrupt_margin > 0),
                                "patched_target_correct": float(patched_margin > 0),
                            }
                        )
                        progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def summarize_successor_patching(detail: pd.DataFrame) -> pd.DataFrame:
    groups = ["direction", "family", "replicate", "count_bin", "top_n", "donor_row"]
    return (
        detail.groupby(groups, as_index=False)
        .agg(
            n_pairs=("decision_k", "size"),
            clean_margin=("clean_margin", "mean"),
            corrupt_margin=("corrupt_margin", "mean"),
            patched_margin=("patched_margin", "mean"),
            normalized_recovery=("normalized_recovery", "mean"),
            clean_target_correct=("clean_target_correct", "mean"),
            corrupt_target_correct=("corrupt_target_correct", "mean"),
            patched_target_correct=("patched_target_correct", "mean"),
        )
        .sort_values(["direction", "family", "replicate", "count_bin", "top_n"])
        .reset_index(drop=True)
    )


def run_successor_patching(
    run_dir: str | Path,
    *,
    examples_per_k: int = 2,
    random_replicates: int = 4,
    device: str | None = None,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    out_dir = run_dir / "analysis" / "successor_patching"
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    detail_path = tables / "successor_head_patching.csv"
    summary_path = tables / "successor_head_patching_summary.csv"
    if not overwrite and detail_path.exists() and summary_path.exists():
        return {
            "successor_head_patching": pd.read_csv(detail_path),
            "successor_head_patching_summary": pd.read_csv(summary_path),
        }

    cfg, vocab = load_run(run_dir, device=device)
    rankings = load_rankings(run_dir)
    model = load_final_model(cfg, vocab, run_dir, "thinking")
    detail = run_successor_patching_rows(
        model,
        cfg,
        vocab,
        rankings,
        examples_per_k=examples_per_k,
        random_replicates=random_replicates,
    )
    summary = summarize_successor_patching(detail)
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "run_dir": str(run_dir.resolve()),
        "device": cfg.device,
        "examples_per_k": int(examples_per_k),
        "decision_k_min": int(cfg.count_min),
        "decision_k_max": int(cfg.count_max - 1),
        "random_replicates": int(random_replicates),
        "top_ns": list(TOP_NS),
        "pair_definition": (
            "count-k and count-(k+1) prompts share noise, the first k needles, and the full "
            "teacher-forced trace through M_k; the longer prompt adds one later needle"
        ),
        "continue_margin": "logit(<k+1>) - logit(</Think>)",
        "close_margin": "logit(</Think>) - logit(<k+1>)",
        "patch_site": "pre-c_proj per-head output at the receiver M_k query",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "successor_head_patching": detail,
        "successor_head_patching_summary": summary,
    }
