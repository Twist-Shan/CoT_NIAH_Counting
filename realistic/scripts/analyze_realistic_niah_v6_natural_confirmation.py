#!/usr/bin/env python3
"""Evaluate discovery-frozen natural patch layers on V6 confirmation seeds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from analyze_realistic_niah_v5_natural_patch_scope_layer_sweep import (  # noqa: E402
    _baseline_rows,
    _scope_cells,
    _summarize,
)
from realistic_niah_v6.pipeline import sha256_file  # noqa: E402
from realistic_niah_v6.suite import validate_confirmation_freeze  # noqa: E402


SCHEMA_VERSION = "realistic_niah_v6_natural_patch_confirmation_v1"


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--discovery-selection", type=Path, required=True)
    parser.add_argument("--confirmation-freeze", type=Path, required=True)
    parser.add_argument(
        "--prompt-mode",
        choices=("enumeration_index", "enumeration_bullet"),
        required=True,
    )
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    freeze = validate_confirmation_freeze(
        args.confirmation_freeze,
        prompt_mode=str(args.prompt_mode),
        model_label=str(args.model),
    )
    discovery = json.loads(args.discovery_selection.read_text(encoding="utf-8"))
    if discovery.get("selection_split") != "discovery":
        raise ValueError("Natural patch layer selection is not discovery-only")
    selections = {str(row["scope"]): row for row in discovery.get("scopes", [])}
    if set(selections) != {"item_end_w1", "event_tail_w4", "item_span"}:
        raise ValueError("Natural patch discovery scope registry changed")

    baselines = _baseline_rows(args.root)
    results = []
    for scope_name, selection in sorted(selections.items()):
        directory = args.root / scope_name
        selected_layer = selection.get("selected_layer")
        if selected_layer is None:
            negative = directory / "negative_skip.json"
            value = json.loads(negative.read_text(encoding="utf-8"))
            if value.get("status") != (
                "DISCOVERY_NEGATIVE_RETAINED_NO_CONFIRMATION_LAYER"
            ):
                raise ValueError(f"Natural negative skip changed for {scope_name}")
            results.append(
                {
                    "scope": scope_name,
                    "status": "DISCOVERY_NEGATIVE_RETAINED",
                    "selected_layer": None,
                    "confirmation_evaluated": False,
                    "negative_result_retained": True,
                    "negative_skip_sha256": sha256_file(negative),
                }
            )
            continue
        cells = _scope_cells(directory, baselines)
        observed_layers = {int(cell["layer"]) for cell in cells}
        if observed_layers != {int(selected_layer)}:
            raise ValueError(
                f"Natural confirmation scope {scope_name} changed the frozen "
                f"layer: expected={selected_layer}, observed={sorted(observed_layers)}"
            )
        directions = {
            direction: _summarize(
                [cell for cell in cells if cell["direction"] == direction]
            )
            for direction in ("forward_skip", "backward_rewind")
        }
        results.append(
            {
                "scope": scope_name,
                "status": "CONFIRMATION_EVALUATED_FROZEN_LAYER",
                "selected_layer": int(selected_layer),
                "confirmation_evaluated": True,
                "summary": _summarize(cells),
                "directions": directions,
                "cell_count": len(cells),
                "negative_result_retained": True,
            }
        )

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "CONFIRMATION_COMPLETE",
        "model_label": str(args.model),
        "prompt_mode": str(args.prompt_mode),
        "scopes": results,
        "discovery_selection": str(args.discovery_selection.resolve()),
        "discovery_selection_sha256": sha256_file(args.discovery_selection),
        "confirmation_freeze": str(args.confirmation_freeze.resolve()),
        "confirmation_freeze_sha256": sha256_file(args.confirmation_freeze),
        "freeze_contract_sha256": str(freeze["freeze_sha256"]),
        "confirmation_used_for_selection": False,
        "layer_reselected": False,
        "negative_results_retained": True,
        "statistical_identity": "true_source_seed",
        "seed_aliasing": False,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
