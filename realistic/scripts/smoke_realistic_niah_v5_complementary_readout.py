#!/usr/bin/env python3
"""One-row GPU smoke for Qwen complementary relay/source readout."""

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
    run_terminal_state_complementary_readout_trials,
)
from realistic_niah_v5.pipeline import read_jsonl, registered_records  # noqa: E402
from realistic_niah_v5.spec import V5Config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    args = parser.parse_args()

    config = V5Config.load(ROOT / "configs" / "realistic_niah_v5.json")
    candidates = [
        row
        for row in registered_records(
            read_jsonl(args.generations), config, model_label="Qwen3-8B"
        )
        if int(row["seed"]) == 1234
        and int(row.get("gold_count", 0)) in {7, 8, 9, 10}
        and bool(row.get("trace_parse", {}).get("parser", {}).get("trace_one_to_one"))
    ]
    if not candidates:
        raise RuntimeError("No canonical Qwen one-to-one smoke row is available")
    row = sorted(candidates, key=lambda value: int(value["gold_count"]))[-1]
    count = int(row["gold_count"])
    model, tokenizer, adapter = load_registered_model(
        resolve_model_spec("Qwen3-8B"),
        cache_dir=args.cache_dir,
        device_map="auto",
        torch_dtype="bfloat16",
        attention_backend="sdpa",
    )
    results = run_terminal_state_complementary_readout_trials(
        model,
        tokenizer,
        adapter,
        row,
        receiver_occurrence=count,
        donor_occurrence=count - 1,
        source_layer=19,
        relay_layer=26,
        geometry="suffix8",
        run_greedy=False,
        max_new_tokens=1,
    )
    assert len(results) == 12, len(results)
    assert {value["patch_condition"] for value in results} == {
        "self_patch",
        "full_donor_patch",
    }
    assert {value["relay_condition"] for value in results} == {
        "natural_relay",
        "post_terminal_suffix_clean_reset",
    }
    assert {value["mask_condition"] for value in results} == {
        "clean",
        "block_trace_items",
        "block_trace_items_matched_control",
    }
    print(
        json.dumps(
            {
                "status": "PASS",
                "model_label": "Qwen3-8B",
                "request_id": row["request_id"],
                "seed": int(row["seed"]),
                "gold_count": count,
                "rows": len(results),
                "source_layer": 19,
                "relay_layer": 26,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
