from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v20.data import (
    V20Vocab,
    balanced_v20_examples,
    character_token,
    load_corpus_text,
)
from synthetic_counting_v20.pipeline import load_prepared_v20_data
from synthetic_counting_v44.preflight import run_preflight as _base_preflight


def run_preflight(run_dir: str | Path) -> dict[str, Any]:
    """Audit v47 full support, permutation invariants, and fixed horizon."""

    root = Path(run_dir)
    result = _base_preflight(root, expected_version="v47")
    cfg = config_from_dict(json.loads((root / "config.json").read_text(encoding="utf-8")))
    if not cfg.permute_task_context_tokens:
        raise ValueError("v47 requires permuted counting-task contexts")
    if cfg.train_steps != 10_000:
        raise ValueError("v47 requires the fixed 10,000-step endpoint")

    text = load_corpus_text()
    vocab = V20Vocab.build(cfg, text)
    split, pool, _, _ = load_prepared_v20_data(cfg, vocab, text, root)
    examples = balanced_v20_examples(
        cfg,
        vocab,
        text,
        split,
        pool,
        1,
        cfg.seed + 147_000,
        region_name="validation",
    )
    if sorted(int(example.count or 0) for example in examples) != list(range(1, 11)):
        raise ValueError("v47 permutation audit is not balanced over counts 1..10")

    changed_order = 0
    for example in examples:
        source = text[example.corpus_start : example.corpus_end]
        source_tokens = [character_token(character) for character in source]
        if Counter(source_tokens) != Counter(example.seq_tokens):
            raise ValueError("task-context permutation changed the source-window multiset")
        if source_tokens != example.seq_tokens:
            changed_order += 1
        if len(example.needle_positions) != int(example.count or 0):
            raise ValueError("permuted needle positions disagree with the target count")
        observed_markers = tuple(
            example.seq_tokens[position] for position in example.needle_positions
        )
        if observed_markers != example.needle_markers:
            raise ValueError("permuted needle positions and trace markers disagree")
    if changed_order != len(examples):
        raise ValueError("at least one v47 audit window was not actually permuted")

    result.update(
        {
            "training_endpoint": 10_000,
            "task_context_order": (
                "fresh_random_permutation_of_each_selected_source_window"
            ),
            "permutation_audit_examples": len(examples),
            "permutation_multiset_invariant": True,
            "permutation_target_count_invariant": True,
            "permutation_trace_marker_order_matches_context": True,
            "permutation_changed_order_fraction": 1.0,
        }
    )
    output = root / "analysis" / "preflight_v47.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit v47 before optimization")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(run_preflight(args.run_dir), indent=2), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["run_preflight"]
