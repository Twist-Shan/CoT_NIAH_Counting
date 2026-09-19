#!/usr/bin/env python3
"""Build audited V5-kernel-compatible native-loop bank/routing files for V6."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _first_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Non-object source-write row in {path}")
                return value
    raise ValueError(f"Empty source-write shard: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument(
        "--prompt-mode",
        choices=("enumeration_index", "enumeration_bullet"),
        required=True,
    )
    parser.add_argument("--anchor-role", choices=("post_marker", "p0_item_end"), required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--bank-plan", type=Path, required=True)
    parser.add_argument("--source-writes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    expected = {
        "status": "DISCOVERY_FROZEN_CHOICE",
        "model_label": str(args.model),
        "prompt_mode": str(args.prompt_mode),
        "selection_split": "discovery",
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise ValueError(
                f"Native-loop selection {key} mismatch: "
                f"expected {value!r}, got {selection.get(key)!r}"
            )
    if _sha256(args.bank_plan) != str(selection["frozen_plan_sha256"]):
        raise ValueError("Native-loop bank plan is not the selected V6 plan")
    selected_k = int(selection["selected_k"])
    with args.bank_plan.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("model_label")) == str(args.model)
            and str(row.get("condition")) == "selected_bank"
        ]
    if len(rows) != 1:
        raise ValueError("Native-loop contract requires one selected-bank row")
    row = rows[0]
    heads = [[int(layer), int(head)] for layer, head in json.loads(row["heads"])]
    if len(heads) != selected_k or len({tuple(value) for value in heads}) != selected_k:
        raise ValueError("Native-loop selected-bank membership changed")
    bank_sha = str(row["bank_sha256"])
    if hashlib.sha256(str(row["heads"]).encode("utf-8")).hexdigest() != bank_sha:
        raise ValueError("Native-loop selected-bank hash changed")

    grammars: set[str] = set()
    source_shards = sorted((args.source_writes / "shards").glob("*.jsonl"))
    if not source_shards:
        raise FileNotFoundError("Native-loop contract found no source-write shards")
    for path in source_shards:
        source = _first_row(path)
        roles = {
            str(value)
            for value in source.get("anchor_roles", [source.get("anchor_role")])
            if value is not None
        }
        if str(args.anchor_role) not in roles:
            continue
        grammar = str(source.get("grammar_pair", "")).rsplit(" -> ", 1)[-1]
        if grammar:
            grammars.add(grammar)
    if not grammars:
        raise ValueError("Native-loop contract resolved no grammar routes")

    compatibility_selection = {
        "schema_version": "realistic_niah_v6_native_loop_selection_compat_v1",
        "status": "DISCOVERY_FROZEN_CHOICE",
        "model_label": str(args.model),
        "prompt_mode": str(args.prompt_mode),
        "development_selection": {
            "primary_bank_heads": heads,
            "primary_bank_size": selected_k,
            "primary_bank_sha256": bank_sha,
            "selection_rank_used": False,
            "source_v6_selection_sha256": _sha256(args.selection),
            "source_v6_bank_plan_sha256": _sha256(args.bank_plan),
        },
    }
    compatibility_routing = {
        "schema_version": "realistic_niah_v6_native_loop_routing_compat_v1",
        "status": "FROZEN_OUTCOME_BLIND",
        "model_label": str(args.model),
        "prompt_mode": str(args.prompt_mode),
        "policy_id": (
            f"v6_{args.prompt_mode}_{args.anchor_role}_single_role_v1"
        ),
        "head_bank": {
            "selected_bank_sha256": bank_sha,
            "selected_bank_size": selected_k,
        },
        "routes": {
            grammar: {"required": [str(args.anchor_role)], "optional": []}
            for grammar in sorted(grammars)
        },
        "route_selection_inputs": [
            "prompt_mode",
            "compiled_target_grammar_class",
            "registered_anchor_availability",
        ],
        "intervention_outcomes_read": False,
        "selection_rank_used": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    selection_path = args.output / "targeted_selection_compat.json"
    routing_path = args.output / "anchor_routing_compat.json"
    _atomic_json(selection_path, compatibility_selection)
    _atomic_json(routing_path, compatibility_routing)
    _atomic_json(
        args.output / "manifest.json",
        {
            "schema_version": "realistic_niah_v6_native_loop_contract_v1",
            "status": "FROZEN_OUTCOME_BLIND",
            "model_label": str(args.model),
            "prompt_mode": str(args.prompt_mode),
            "anchor_role": str(args.anchor_role),
            "selected_k": selected_k,
            "selected_bank_sha256": bank_sha,
            "grammar_routes": sorted(grammars),
            "source_selection": str(args.selection.resolve()),
            "source_selection_sha256": _sha256(args.selection),
            "source_bank_plan": str(args.bank_plan.resolve()),
            "source_bank_plan_sha256": _sha256(args.bank_plan),
            "source_writes": str(args.source_writes.resolve()),
            "source_manifest_sha256": _sha256(args.source_writes / "manifest.json"),
            "targeted_selection_compat_sha256": _sha256(selection_path),
            "anchor_routing_compat_sha256": _sha256(routing_path),
            "intervention_outcomes_read": False,
            "selection_rank_used": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "selected_k": selected_k,
                "grammar_routes": sorted(grammars),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
