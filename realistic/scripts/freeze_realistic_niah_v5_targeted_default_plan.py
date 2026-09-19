#!/usr/bin/env python3
"""Freeze the prospective model-specific targeted-retrieval default plan."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_realistic_niah_v5_targeted_count_plan import build_plan


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def freeze(
    *, root: Path, defaults_path: Path, model: str, output: Path
) -> dict[str, Any]:
    root = root.resolve()
    defaults_path = defaults_path.resolve()
    defaults = _load(defaults_path)
    if defaults.get("status") != "FROZEN":
        raise ValueError("Prospective targeted defaults are not frozen")
    protocol = defaults["protocol"]
    if protocol.get("selection_rank_used") is not False:
        raise ValueError("Prospective targeted defaults allow selection_rank")
    model_spec = defaults["models"][model]
    selection_path = _resolve(root, str(model_spec["selection"]))
    routing_path = _resolve(root, str(model_spec["routing"]))
    selection = _load(selection_path)
    routing = _load(routing_path)
    if str(selection["model_label"]) != model:
        raise ValueError("Selection model mismatch")
    if selection.get("prospective_only") is not True:
        raise ValueError("Selection is not marked prospective-only")
    expected_k = int(model_spec["bank_size"])
    expected_sha = str(model_spec["selected_bank_sha256"])
    if int(selection["development_selection"]["primary_bank_size"]) != expected_k:
        raise ValueError("Selection/default bank-size mismatch")
    if str(selection["development_selection"]["primary_bank_sha256"]) != expected_sha:
        raise ValueError("Selection/default bank hash mismatch")
    if int(routing["head_bank"]["bank_size"]) != expected_k:
        raise ValueError("Routing/default bank-size mismatch")
    if str(routing["head_bank"]["selected_bank_sha256"]) != expected_sha:
        raise ValueError("Routing/default bank hash mismatch")

    frame = build_plan(
        selection,
        heads_per_layer=int(model_spec["heads_per_layer"]),
        random_repeats=3,
        random_seed=20260821,
    )
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    payload = buffer.getvalue().encode("utf-8")
    plan_sha = hashlib.sha256(payload).hexdigest()
    output = output.resolve()
    if output.exists():
        if output.read_bytes() != payload:
            raise FileExistsError(
                f"Refusing to replace mismatched frozen plan: {output}"
            )
        status = "REUSED_IDENTICAL"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(output)
        status = "FROZEN_NEW"
    audit = {
        "schema_version": "realistic_niah_v5_targeted_default_plan_audit_v1",
        "status": status,
        "model_label": model,
        "bank_size": expected_k,
        "selected_bank_sha256": expected_sha,
        "plan_sha256": plan_sha,
        "selection_sha256": _sha(selection_path),
        "routing_sha256": _sha(routing_path),
        "defaults_sha256": _sha(defaults_path),
        "routing": str(routing_path),
        "outcome_blind": True,
        "selection_rank_used": False,
        "historical_artifacts_modified": False,
    }
    audit_path = output.with_suffix(".audit.json")
    temporary = audit_path.with_name(f".{audit_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(audit_path)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--defaults", type=Path, required=True)
    parser.add_argument("--model", choices=["Qwen3-8B", "Gemma4-E4B"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(
                root=args.root,
                defaults_path=args.defaults,
                model=str(args.model),
                output=args.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
