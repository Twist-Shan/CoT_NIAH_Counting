#!/usr/bin/env python3
"""Fail-fast consistency checks for the frozen hybrid-localizer design."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qwen-routing", type=Path, required=True)
    parser.add_argument("--gemma-routing", type=Path, required=True)
    args = parser.parse_args()

    spec = _read(args.spec)
    config = _read(args.config)
    code_root = args.spec.resolve().parents[1]
    contract = spec["scientific_contract"]
    if contract["intervention_start_anchor_role"] != "p0_item_end":
        raise AssertionError("The registered intervention must start at exact P0")
    if int(contract["decode_head_ablation_steps"]) != -1:
        raise AssertionError("The registered intervention must persist to decode end")
    repeats = int(contract["random_control_repeats"])
    if repeats != 3 or int(config["causal_random_controls"]) != repeats:
        raise AssertionError("Frozen spec and runner must both use three controls")

    routings = {
        "Qwen3-8B": _read(args.qwen_routing),
        "Gemma4-E4B": _read(args.gemma_routing),
    }
    summary = {}
    for model, model_spec in spec["models"].items():
        grammar_roles = model_spec["grammars"]
        if not grammar_roles:
            raise AssertionError(f"{model} has no registered grammars")
        if set(grammar_roles.values()) - {"p0_item_end", "post_marker"}:
            raise AssertionError(f"{model} declares an unregistered selection role")
        doses = list(map(int, model_spec["doses"]))
        if doses != sorted(set(doses)):
            raise AssertionError(f"{model} doses must be sorted and unique")
        if int(model_spec["primary_bank_size"]) != max(doses):
            raise AssertionError(f"{model} primary bank must be the maximum dose")
        if set(map(int, model_spec["control_policy"])) != set(doses):
            raise AssertionError(f"{model} control policy does not cover every dose")
        supplement = model_spec["supplement_execution"]
        run_grammars = set(supplement["run_grammars"])
        reuse_grammars = set(supplement["reuse_grammars"])
        if run_grammars & reuse_grammars or run_grammars | reuse_grammars != set(
            grammar_roles
        ):
            raise AssertionError(
                f"{model} supplement run/reuse grammars do not partition the panel"
            )
        if run_grammars != {
            grammar for grammar, role in grammar_roles.items() if role == "post_marker"
        }:
            raise AssertionError(
                f"{model} supplement must run exactly the P2-ranked grammars"
            )
        for key in ("prior_full_completion", "prior_dose_completion"):
            path = code_root / supplement[key]
            expected_sha = supplement[f"{key}_sha256"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
                raise AssertionError(f"{model} frozen prior result drifted: {path}")
        routing = routings[model]
        missing_routes = set(grammar_roles) - set(routing["routes"])
        if missing_routes:
            raise AssertionError(f"{model} routing misses {sorted(missing_routes)}")
        for grammar, route in routing["routes"].items():
            if route.get("required") != ["p0_item_end"] or route.get("optional") != []:
                raise AssertionError(
                    f"{model}/{grammar} behavior route is not exact-P0-only"
                )
        summary[model] = {
            "grammars": len(grammar_roles),
            "doses": doses,
            "registry_rows": int(model_spec["registry_rows"]),
            "registry_sha256": model_spec["registry_sha256"],
        }
    sidecar = spec["registered_sidecar"]
    if sidecar["selection_anchor_role"] != "p0_item_end":
        raise AssertionError("Sidecar ranking must be P0")
    if sidecar["layer_profile_reference_anchor_role"] != "post_marker":
        raise AssertionError("Sidecar layer-profile reference must be P2")
    p0_ranking = code_root / sidecar["p0_ranking"]
    if hashlib.sha256(p0_ranking.read_bytes()).hexdigest() != sidecar[
        "p0_ranking_sha256"
    ]:
        raise AssertionError("Frozen P0 sidecar ranking drifted")
    payload = {
        "schema_version": "realistic_niah_v5_native_hybrid_spec_audit_v1",
        "status": "PASS",
        "spec_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
        "models": summary,
        "intervention_start_anchor_role": contract[
            "intervention_start_anchor_role"
        ],
        "decode_head_ablation_steps": contract["decode_head_ablation_steps"],
        "random_control_repeats": repeats,
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
