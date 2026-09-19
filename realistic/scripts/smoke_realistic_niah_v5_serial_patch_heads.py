#!/usr/bin/env python3
"""One-row GPU smoke for the joint terminal-patch/head-ablation forward path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import load_registered_model  # noqa: E402
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.count_stream import (  # noqa: E402
    run_full_state_patch_head_readout_factorial_trials,
)
from realistic_niah_v5.pipeline import read_jsonl, registered_records  # noqa: E402
from realistic_niah_v5.spec import V5Config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["Qwen3-8B", "Gemma4-E4B"], required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    args = parser.parse_args()

    config = V5Config.load(ROOT / "configs" / "realistic_niah_v5.json")
    candidates = [
        row
        for row in registered_records(
            read_jsonl(args.generations), config, model_label=args.model
        )
        if int(row["seed"]) == 1234
        and int(row.get("gold_count", 0)) in {7, 8, 9, 10}
        and bool(row.get("trace_parse", {}).get("parser", {}).get("trace_one_to_one"))
    ]
    if not candidates:
        raise RuntimeError("No canonical one-to-one smoke row is available")
    row = sorted(candidates, key=lambda value: int(value["gold_count"]))[-1]
    count = int(row["gold_count"])
    model, tokenizer, adapter = load_registered_model(
        resolve_model_spec(args.model),
        cache_dir=args.cache_dir,
        device_map="auto",
        torch_dtype="bfloat16",
        attention_backend="sdpa",
    )
    patch_layer = {"Qwen3-8B": 19, "Gemma4-E4B": 16}[args.model]
    head_arms = [
        {"condition": "clean", "repeat": 0, "heads": []},
        {"condition": "selected_bank", "repeat": 0, "heads": [[0, 0]]},
        {
            "condition": "layer_matched_random",
            "repeat": 1,
            "heads": [[0, 1]],
        },
        {
            "condition": "layer_matched_random",
            "repeat": 2,
            "heads": [[0, 2]],
        },
        {
            "condition": "layer_matched_random",
            "repeat": 3,
            "heads": [[0, 3]],
        },
    ]
    results = run_full_state_patch_head_readout_factorial_trials(
        model,
        tokenizer,
        adapter,
        row,
        receiver_occurrence=count,
        donor_occurrence=count - 1,
        layer=patch_layer,
        geometry="suffix8",
        layer_mode="cumulative_clamp",
        head_arms=head_arms,
        run_greedy=False,
        max_new_tokens=1,
    )
    assert len(results) == 10, len(results)
    assert {value["patch_condition"] for value in results} == {
        "self_patch",
        "full_donor_patch",
    }
    assert {value["head_condition"] for value in results} == {
        "clean",
        "selected_bank",
        "layer_matched_random",
    }
    assert all(value["receiver_is_terminal"] for value in results)
    assert not any(value["later_trace_self_correction_possible"] for value in results)
    print(
        json.dumps(
            {
                "status": "PASS",
                "model_label": args.model,
                "request_id": row["request_id"],
                "seed": int(row["seed"]),
                "gold_count": count,
                "rows": len(results),
                "patch_layer": patch_layer,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
