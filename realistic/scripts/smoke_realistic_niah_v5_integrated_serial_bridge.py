#!/usr/bin/env python3
"""One-row GPU smoke for the frozen integrated serial bridge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v4.modeling import load_registered_model  # noqa: E402
from realistic_niah_v4.spec import resolve_model_spec  # noqa: E402
from realistic_niah_v5.integrated_bridge import (  # noqa: E402
    run_integrated_serial_bridge_trials,
)
from realistic_niah_v5.parsing import (  # noqa: E402
    find_trace_count_sequence,
    gold_records,
    infer_model_family,
    raw_output_text,
)
from realistic_niah_v5.pipeline import read_jsonl, registered_records  # noqa: E402
from realistic_niah_v5.spec import V5Config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["Qwen3-8B", "Gemma4-E4B"], required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--bank-plan", type=Path, required=True)
    parser.add_argument("--anchor-registry", type=Path, required=True)
    parser.add_argument("--request-id")
    parser.add_argument(
        "--write-window",
        choices=["exact_query", "query_through_trace"],
        default="exact_query",
    )
    parser.add_argument(
        "--bridge-design",
        choices=["transfer", "restoration"],
        default="transfer",
    )
    parser.add_argument(
        "--geometry",
        choices=["suffix8", "full_span"],
        default="suffix8",
    )
    parser.add_argument("--run-greedy", action="store_true")
    args = parser.parse_args()

    config = V5Config.load(ROOT / "configs" / "realistic_niah_v5.json")
    candidates = []
    for row in registered_records(
        read_jsonl(args.generations), config, model_label=args.model
    ):
        if args.request_id is None:
            if int(row["seed"]) != 1234 or int(row.get("gold_count", 0)) not in {
                7,
                8,
                9,
                10,
            }:
                continue
        elif str(row["request_id"]) != str(args.request_id):
            continue
        parsed = find_trace_count_sequence(
            raw_output_text(row),
            model_family=infer_model_family(row),
            gold_records=gold_records(row),
        )
        if parsed.trace_one_to_one:
            candidates.append(row)
    registry = {
        str(value["request_id"]): value for value in read_jsonl(args.anchor_registry)
    }
    candidates = [
        row for row in candidates if str(row["request_id"]) in registry
    ]
    if not candidates:
        raise RuntimeError("No canonical one-to-one integrated smoke row is available")
    row = sorted(candidates, key=lambda value: int(value["gold_count"]))[-1]
    plan = pd.read_csv(args.bank_plan)
    plan = plan.loc[plan["model_label"].astype(str).eq(args.model)]
    banks = [
        {"condition": "clean", "repeat": 0, "heads": [], "bank_sha256": "clean"}
    ]
    for bank in plan.sort_values(["condition", "repeat"]).itertuples(index=False):
        banks.append(
            {
                "condition": str(bank.condition),
                "repeat": int(bank.repeat),
                "heads": json.loads(str(bank.heads)),
                "bank_sha256": str(bank.bank_sha256),
            }
        )
    model, tokenizer, adapter = load_registered_model(
        resolve_model_spec(args.model),
        cache_dir=args.cache_dir,
        device_map="auto",
        torch_dtype="bfloat16",
        attention_backend="sdpa",
    )
    patch_layers = tuple(range(19, 26)) if args.model == "Qwen3-8B" else tuple(range(16, 42))
    results = run_integrated_serial_bridge_trials(
        model,
        tokenizer,
        adapter,
        row,
        banks=banks,
        targeted_site=registry[str(row["request_id"])],
        patch_layers=patch_layers,
        model_label=args.model,
        geometry=str(args.geometry),
        relay_layer=26 if args.model == "Qwen3-8B" else None,
        write_window=str(args.write_window),
        bridge_design=str(args.bridge_design),
        run_greedy=bool(args.run_greedy),
        max_new_tokens=1,
    )
    assert len(results) == (30 if args.bridge_design == "restoration" else 15), len(
        results
    )
    assert {value["write_condition"] for value in results} == {
        "clean",
        "selected_bank",
        "layer_matched_random",
    }
    assert {value["readout_condition"] for value in results} == {
        "natural",
        "matched_control",
        "cut",
    }
    if args.bridge_design == "restoration":
        assert {value["mediator_condition"] for value in results} == {
            "self_state",
            "clean_state_restore",
        }
        assert {value["receiver_write_condition"] for value in results} == {
            "clean",
            "selected_bank",
            "layer_matched_random",
        }
        assert sum(bool(value["greedy_generation_run"]) for value in results) == (
            1 if args.run_greedy else 0
        )
    assert all(value["status"] == "ok" for value in results)
    print(
        json.dumps(
            {
                "status": "PASS",
                "model_label": args.model,
                "request_id": row["request_id"],
                "seed": int(row["seed"]),
                "gold_count": int(row["gold_count"]),
                "rows": len(results),
                "bridge_design": str(args.bridge_design),
                "geometry": str(args.geometry),
                "patch_layers": list(patch_layers),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
