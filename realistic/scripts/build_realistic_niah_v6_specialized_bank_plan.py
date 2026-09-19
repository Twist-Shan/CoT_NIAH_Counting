#!/usr/bin/env python3
"""Build the outcome-blind bank plan consumed by V6 specialized assays.

The targeted-retrieval dose experiment freezes both K and the selected head
bank.  A stratified downstream assay must capture strictly after every
intervened layer.  The report-matched Qwen K=128 plan uses global-random
controls, and a few of those controls can occupy the final transformer layer
even though the selected bank ends one layer earlier.  Such a plan has no
legal post-intervention capture layer.

This builder makes the minimal structural repair: it keeps the selected bank
byte-for-byte identical and deterministically replaces only global-random
heads at or above the selected bank's capture layer.  Replacements are drawn
from the model's complete head-identity universe, excluding the selected bank
and the retained members of that random bank.  No behavior, specialized, or
confirmation outcome is used.  Layer-matched plans, and already reachable
global plans, are copied byte-for-byte.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "realistic_niah_v6_specialized_bank_plan_adapter_v1"
SELECTION_SCHEMA_VERSION = "realistic_niah_v6_targeted_retrieval_selection_v1"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        return fieldnames, [dict(row) for row in reader]


def _heads(row: Mapping[str, Any], *, bank_size: int) -> list[tuple[int, int]]:
    serialized = str(row.get("heads", ""))
    raw = json.loads(serialized)
    heads = [(int(layer), int(head)) for layer, head in raw]
    if len(heads) != bank_size or len(set(heads)) != bank_size:
        raise ValueError("Specialized source plan has an invalid head bank")
    if hashlib.sha256(serialized.encode("utf-8")).hexdigest() != str(
        row.get("bank_sha256", "")
    ):
        raise ValueError("Specialized source plan bank hash changed")
    return heads


def _identity_universe(
    ranking_path: Path, *, model_label: str
) -> tuple[set[tuple[int, int]], int, int, int]:
    fieldnames, rows = _read_csv(ranking_path)
    if not {"model_label", "layer", "head"} <= set(fieldnames):
        raise ValueError("Head-identity source lacks model_label/layer/head")
    identities = [
        (int(row["layer"]), int(row["head"]))
        for row in rows
        if str(row.get("model_label")) == model_label
    ]
    if not identities or len(identities) != len(set(identities)):
        raise ValueError("Head-identity source is empty or contains duplicates")
    maximum_layer = max(layer for layer, _head in identities)
    maximum_head = max(head for _layer, head in identities)
    expected = {
        (layer, head)
        for layer in range(maximum_layer + 1)
        for head in range(maximum_head + 1)
    }
    observed = set(identities)
    if observed != expected:
        raise ValueError("Head-identity source is not a complete rectangular model")
    return observed, maximum_layer + 1, maximum_head + 1, len(identities)


def _write_csv(
    path: Path, *, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_specialized_bank_plan(
    *,
    selection_path: Path,
    source_plan_path: Path,
    head_universe_path: Path,
    model_label: str,
    prompt_mode: str,
    output: Path,
) -> dict[str, Any]:
    selection_path = selection_path.resolve()
    source_plan_path = source_plan_path.resolve()
    head_universe_path = head_universe_path.resolve()
    output = output.resolve()
    for path in (selection_path, source_plan_path, head_universe_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    expected_selection = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": "DISCOVERY_FROZEN_CHOICE",
        "model_label": model_label,
        "prompt_mode": prompt_mode,
        "selection_split": "discovery",
    }
    for key, expected in expected_selection.items():
        if selection.get(key) != expected:
            raise ValueError(
                f"Specialized bank selection {key} mismatch: "
                f"expected {expected!r}, got {selection.get(key)!r}"
            )
    if selection.get("dose_argmax_used_for_downstream_bank") is not True:
        raise ValueError("Specialized bank did not come from the frozen dose argmax")
    selected_k = int(selection["selected_k"])
    if int(selection.get("dose_argmax_k", -1)) != selected_k:
        raise ValueError("Specialized selected K and dose argmax disagree")
    random_condition = str(selection.get("selected_random_condition"))
    if random_condition not in {"layer_matched_random", "global_random"}:
        raise ValueError("Specialized selection has an invalid control family")

    source_plan_sha256 = _sha256_file(source_plan_path)
    if source_plan_sha256 != str(selection.get("frozen_plan_sha256", "")):
        raise ValueError("Specialized source plan is not the discovery-frozen plan")
    fieldnames, rows = _read_csv(source_plan_path)
    required_fields = {
        "model_label",
        "condition",
        "repeat",
        "bank_size",
        "bank_sha256",
        "heads",
    }
    if not required_fields <= set(fieldnames):
        raise ValueError("Specialized source plan lacks required fields")
    model_rows = [
        row for row in rows if str(row.get("model_label")) == model_label
    ]
    identities = Counter(
        (str(row.get("condition")), int(row.get("repeat", 0)))
        for row in model_rows
    )
    expected_identities = Counter(
        {
            ("selected_bank", 0): 1,
            (random_condition, 1): 1,
            (random_condition, 2): 1,
            (random_condition, 3): 1,
        }
    )
    if identities != expected_identities:
        raise ValueError("Specialized source plan does not contain the four frozen arms")
    if any(int(row.get("bank_size", -1)) != selected_k for row in model_rows):
        raise ValueError("Specialized source plan mixes bank sizes")

    selected_row = next(
        row for row in model_rows if str(row["condition"]) == "selected_bank"
    )
    selected_heads = _heads(selected_row, bank_size=selected_k)
    selected_set = set(selected_heads)
    selected_bank_sha256 = str(selected_row["bank_sha256"])
    selected_max_layer = max(layer for layer, _head in selected_heads)

    universe, num_layers, num_heads, universe_count = _identity_universe(
        head_universe_path, model_label=model_label
    )
    if not selected_set <= universe:
        raise ValueError("Frozen selected bank escapes the model head universe")
    capture_start_layer = selected_max_layer + 1
    if capture_start_layer >= num_layers:
        raise ValueError(
            "Frozen selected bank reaches the final layer; no downstream capture "
            "layer exists without changing treatment heads"
        )

    source_random_hashes: dict[str, str] = {}
    output_random_hashes: dict[str, str] = {}
    replacements_by_repeat: dict[str, list[dict[str, list[int]]]] = {}
    changed = False
    source_random_max_layer = -1
    output_random_max_layer = -1
    output_rows = [dict(row) for row in rows]
    for row in output_rows:
        if (
            str(row.get("model_label")) != model_label
            or str(row.get("condition")) != random_condition
        ):
            continue
        repeat = int(row["repeat"])
        source_heads = _heads(row, bank_size=selected_k)
        source_set = set(source_heads)
        if source_set & selected_set:
            raise ValueError("Frozen random bank overlaps the selected treatment")
        source_random_max_layer = max(
            source_random_max_layer,
            max(layer for layer, _head in source_heads),
        )
        invalid = [
            head for head in source_heads if int(head[0]) >= capture_start_layer
        ]
        retained = [
            head for head in source_heads if int(head[0]) < capture_start_layer
        ]
        replacement_rows: list[dict[str, list[int]]] = []
        output_heads = list(source_heads)
        if invalid:
            if random_condition != "global_random":
                raise ValueError(
                    "A layer-matched control exceeds the selected capture boundary"
                )
            candidates = sorted(
                head
                for head in universe
                if head[0] < capture_start_layer
                and head not in selected_set
                and head not in set(retained)
            )
            if len(candidates) < len(invalid):
                raise ValueError("Insufficient capture-reachable global controls")
            deterministic_seed_material = "|".join(
                (
                    SCHEMA_VERSION,
                    _sha256_file(selection_path),
                    source_plan_sha256,
                    model_label,
                    prompt_mode,
                    str(selected_k),
                    str(repeat),
                )
            )
            deterministic_seed_sha256 = hashlib.sha256(
                deterministic_seed_material.encode("utf-8")
            ).hexdigest()
            generator = random.Random(int(deterministic_seed_sha256, 16))
            replacements = generator.sample(candidates, len(invalid))
            output_heads = sorted([*retained, *replacements])
            replacement_rows = [
                {"from": list(old), "to": list(new)}
                for old, new in zip(sorted(invalid), sorted(replacements))
            ]
            changed = True
        if len(output_heads) != selected_k or len(set(output_heads)) != selected_k:
            raise RuntimeError("Specialized control adaptation changed bank cardinality")
        if set(output_heads) & selected_set:
            raise RuntimeError("Specialized control adaptation overlaps treatment")
        if max(layer for layer, _head in output_heads) >= capture_start_layer:
            raise RuntimeError("Specialized control adaptation is not capture-reachable")
        serialized = json.dumps([list(head) for head in output_heads])
        row["heads"] = serialized
        row["bank_sha256"] = hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()
        source_random_hashes[str(repeat)] = str(
            next(
                source["bank_sha256"]
                for source in model_rows
                if str(source["condition"]) == random_condition
                and int(source["repeat"]) == repeat
            )
        )
        output_random_hashes[str(repeat)] = str(row["bank_sha256"])
        replacements_by_repeat[str(repeat)] = replacement_rows
        output_random_max_layer = max(
            output_random_max_layer,
            max(layer for layer, _head in output_heads),
        )

    if len(set(output_random_hashes.values())) != 3:
        raise RuntimeError("Specialized random control banks are not distinct")
    selected_after = next(
        row
        for row in output_rows
        if str(row.get("model_label")) == model_label
        and str(row.get("condition")) == "selected_bank"
    )
    if selected_after != selected_row:
        raise RuntimeError("Specialized adapter changed the selected treatment row")

    output_plan = output / "retrieval_anchor_bank_plan.csv"
    if changed:
        _write_csv(output_plan, fieldnames=fieldnames, rows=output_rows)
    else:
        _atomic_bytes(output_plan, source_plan_path.read_bytes())
    output_plan_sha256 = _sha256_file(output_plan)
    status = (
        "PASS_CAPTURE_REACHABLE_GLOBAL_CONTROL_ADAPTER"
        if changed
        else "PASS_SOURCE_PLAN_UNCHANGED"
    )
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "model_label": model_label,
        "prompt_mode": prompt_mode,
        "selected_k": selected_k,
        "selected_random_condition": random_condition,
        "selection_registry": str(selection_path),
        "selection_registry_sha256": _sha256_file(selection_path),
        "source_plan": str(source_plan_path),
        "source_plan_sha256": source_plan_sha256,
        "head_identity_universe": str(head_universe_path),
        "head_identity_universe_sha256": _sha256_file(head_universe_path),
        "head_identity_universe_count": universe_count,
        "model_num_layers": num_layers,
        "model_num_heads_per_layer": num_heads,
        "output_plan": str(output_plan),
        "output_plan_sha256": output_plan_sha256,
        "selected_bank_sha256": selected_bank_sha256,
        "selected_max_layer": selected_max_layer,
        "capture_start_layer": capture_start_layer,
        "source_random_max_layer": source_random_max_layer,
        "output_random_max_layer": output_random_max_layer,
        "source_random_bank_sha256_by_repeat": source_random_hashes,
        "output_random_bank_sha256_by_repeat": output_random_hashes,
        "replacements_by_repeat": replacements_by_repeat,
        "replacement_count": sum(
            len(values) for values in replacements_by_repeat.values()
        ),
        "random_controls_changed": changed,
        "selected_treatment_row_unchanged": True,
        "selected_treatment_heads_unchanged": True,
        "selected_treatment_bank_sha256_unchanged": True,
        "bank_size_unchanged": True,
        "random_control_family_unchanged": True,
        "adapter_rule": (
            "replace only global-random heads at or above max(selected layer)+1; "
            "sample deterministically from the capture-reachable complement"
        ),
        "behavior_outcomes_used_to_construct_controls": False,
        "specialized_outcomes_used_to_construct_controls": False,
        "confirmation_outcomes_used_to_construct_controls": False,
        "sample_failure": False,
        "seed_replacement_triggered": False,
        "v5_source_files_modified": False,
    }
    audit_path = output / "specialized_bank_plan_audit.json"
    _atomic_json(audit_path, audit)
    _atomic_bytes(output / "specialized_bank_plan.COMPLETE", b"PASS\n")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--head-universe", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt-mode",
        choices=("enumeration_index", "enumeration_bullet"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_specialized_bank_plan(
        selection_path=args.selection,
        source_plan_path=args.source_plan,
        head_universe_path=args.head_universe,
        model_label=args.model,
        prompt_mode=args.prompt_mode,
        output=args.output,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
