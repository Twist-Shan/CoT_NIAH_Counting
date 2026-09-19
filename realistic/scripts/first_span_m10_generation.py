from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from realistic_niah_v4.causal_generation import (
    _base_metadata,
    _baseline_metadata,
    _validate_baseline_label,
    generate_with_head_ablation,
    intervention_outcome,
)


Head = tuple[int, int]


def _heads(rows: Sequence[Mapping[str, Any]]) -> tuple[Head, ...]:
    return tuple((int(row["layer"]), int(row["head"])) for row in rows)


def run_generation_head_ablation_v2_m10(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    encodings: Sequence[Any],
    *,
    baseline_labels: Mapping[str, Mapping[str, Any]],
    rankings: Mapping[str, Sequence[Head]],
    top_ns: Sequence[int],
    random_replicates: int,
    require_full_sweep: bool,
    valid_counts: Sequence[int],
    max_new_tokens: int,
    control_json: str | Path,
    require_correct_baseline: bool = False,
) -> pd.DataFrame:
    """Run ranked sets against frozen layer+M10-matched control sets."""

    del require_full_sweep
    plan = json.loads(Path(control_json).read_text(encoding="utf-8"))
    if int(random_replicates) != 3:
        raise ValueError("M10 campaign expects exactly three frozen controls")
    rows: list[dict[str, Any]] = []
    for example_index, encoding in enumerate(encodings):
        label = baseline_labels.get(encoding.stimulus_id)
        if label is None:
            raise KeyError(f"Missing baseline generation label: {encoding.stimulus_id}")
        _validate_baseline_label(encoding, label)
        if require_correct_baseline and str(label.get("outcome_group")) != "correct":
            raise ValueError(f"Ineligible clean baseline: {encoding.stimulus_id}")
        cache: dict[tuple[Head, ...], dict[str, Any]] = {}
        for bank, ranking in sorted(rankings.items()):
            normalized = tuple((int(layer), int(head)) for layer, head in ranking)
            for top_n in sorted({int(value) for value in top_ns}):
                selected = normalized[:top_n]
                frozen = sorted(plan["sets"][str(top_n)], key=lambda row: int(row["replicate"]))
                conditions = [("ranked", -1, selected, len(selected), "discovery_ranked_prefix")]
                for row in frozen:
                    heads = _heads(row["heads"])
                    conditions.append(
                        (
                            "layer_matched_random",
                            int(row["replicate"]),
                            heads,
                            int(row["ranked_control_overlap"]),
                            "frozen_layer_and_discovery_M10_matched",
                        )
                    )
                for condition, replicate, heads, overlap, population in conditions:
                    if heads not in cache:
                        cache[heads] = generate_with_head_ablation(
                            model,
                            tokenizer,
                            adapter,
                            encoding,
                            heads,
                            scope="answer_query",
                            max_new_tokens=max_new_tokens,
                        )
                    rows.append(
                        {
                            **_base_metadata(encoding),
                            **_baseline_metadata(label),
                            "example_index": int(example_index),
                            "head_bank": str(bank),
                            "scope": "answer_query",
                            "condition": condition,
                            "top_n": int(top_n),
                            "random_replicate": int(replicate),
                            "heads": ",".join(f"L{layer}H{head}" for layer, head in heads),
                            "ranked_random_head_overlap": int(overlap),
                            "random_sampling_population": population,
                            "analysis_population": "all_examples",
                            **intervention_outcome(
                                cache[heads],
                                encoding,
                                label,
                                valid_counts=valid_counts,
                            ),
                        }
                    )
        print(
            "[v4 M10-matched ablation] "
            f"{example_index + 1}/{len(encodings)} seed={encoding.seed} N={encoding.count}",
            flush=True,
        )
    if not rows:
        raise ValueError("No M10-matched head-ablation rows were produced")
    return pd.DataFrame(rows)
