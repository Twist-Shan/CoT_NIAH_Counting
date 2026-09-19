#!/usr/bin/env python3
"""Analyze the V6 specialized discovery/confirmation assays without seed aliasing.

The frozen V5 numerical analyzers are reused verbatim.  Two of them imported
the historical 1234..1263 seed constants at module import time; this wrapper
rebinds only those membership constants to the true source seeds recorded in
the V6 replacement registries.  Fixed panel membership remains the original
``analysis_slot_seed`` and every statistical aggregation remains keyed by the
true source ``seed``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v6.pipeline import read_jsonl, sha256_file  # noqa: E402
from realistic_niah_v6.spec import V6Config  # noqa: E402
from realistic_niah_v6.suite import validate_confirmation_freeze  # noqa: E402
from realistic_niah_v6.kernel import install_v6_specialized_geometry  # noqa: E402
from scripts import analyze_realistic_niah_v5_targeted_counter_ncc as ncc  # noqa: E402
from scripts import analyze_realistic_niah_v5_targeted_counter_write as write  # noqa: E402


ANSWER_CONDITIONS = {
    "clean",
    "prompt_all_blank",
    "prompt_records_blank",
    "trace_all_blank",
    "prompt_and_trace_blank",
}
TARGETING_CONDITIONS = {
    "clean",
    "early_half_trace_blank",
    "cumulative_trace_blank",
    "recent_transition_blank",
    "full_trace_blank",
    "early_half_trace_matched_control",
    "cumulative_trace_matched_control",
    "recent_transition_matched_control",
    "full_trace_matched_control",
}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _registry_contract(
    path: Path,
    *,
    config: V6Config,
    model_label: str,
    split: str,
) -> dict[str, Any]:
    expected_slots = tuple(
        map(
            int,
            config.discovery_seeds
            if split == "discovery"
            else config.confirmation_seeds,
        )
    )
    rows = [
        dict(row)
        for row in read_jsonl(path)
        if str(row.get("model_label")) == model_label
        and str(row.get("split")) == split
        and int(row.get("gold_count", -1)) == 10
    ]
    rows.sort(key=lambda row: int(row["analysis_slot_seed"]))
    slots = tuple(int(row["analysis_slot_seed"]) for row in rows)
    if slots != tuple(sorted(expected_slots)):
        raise ValueError(
            f"{split} specialized registry changed the fixed N=10 slots: "
            f"expected={sorted(expected_slots)} observed={list(slots)}"
        )
    sources = tuple(int(row["source_seed"]) for row in rows)
    requests = tuple(str(row["source_request_id"]) for row in rows)
    if len(set(sources)) != len(sources):
        raise ValueError(f"{split} specialized registry reuses a source seed")
    if len(set(requests)) != len(requests):
        raise ValueError(f"{split} specialized registry reuses a source request")
    for row in rows:
        slot = int(row["analysis_slot_seed"])
        source = int(row["source_seed"])
        if bool(row["replacement_applied"]) != (slot != source):
            raise ValueError(f"{split} specialized replacement flag changed")
        if bool(row.get("intervention_outcomes_read", False)):
            raise ValueError(f"{split} replacement inspected an intervention outcome")
    return {
        "split": split,
        "registry": str(path.resolve()),
        "registry_sha256": sha256_file(path),
        "analysis_slots": list(slots),
        "true_source_seeds": list(sources),
        "source_request_ids": list(requests),
        "slot_to_true_source_seed": {
            str(slot): source for slot, source in zip(slots, sources)
        },
        "replacement_count": sum(slot != source for slot, source in zip(slots, sources)),
    }


def _audit_seed_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    label: str,
) -> None:
    materialized = list(rows)
    expected = set(map(int, contract["true_source_seeds"]))
    observed = {int(row["seed"]) for row in materialized}
    if observed != expected:
        raise ValueError(
            f"{label} true-source seed mismatch: "
            f"expected={sorted(expected)} observed={sorted(observed)}"
        )
    if not materialized:
        raise ValueError(f"{label} contains no rows")


def _audit_specialized_inputs(
    discovery_root: Path,
    confirmation_root: Path,
    *,
    discovery: Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for phase, root, contract in (
        ("discovery", discovery_root, discovery),
        ("confirmation", confirmation_root, confirmation),
    ):
        write_frame = write._read(root / "targeted_counter_write")
        _audit_seed_rows(
            write_frame.to_dict("records"),
            contract=contract,
            label=f"{phase} targeted-counter-write",
        )

        stratified_rows, stratified_plan = _load_npz_metadata(
            root / "stratified_ncc", "teacher_forced_stratified_targeted_counter_ncc"
        )
        _audit_seed_rows(
            stratified_rows,
            contract=contract,
            label=f"{phase} stratified NCC",
        )
        if set(map(int, stratified_plan.get("seeds", ()))) != set(
            map(int, contract["true_source_seeds"])
        ):
            raise ValueError(f"{phase} stratified NCC row plan aliases a seed")

        logit_rows = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((root / "direct_count_logit_margin/shards").glob("*.json"))
        ]
        _audit_seed_rows(
            logit_rows,
            contract=contract,
            label=f"{phase} direct count-logit margin",
        )

        ncc_rows = ncc._load(root / "count_geometry_ncc")
        _audit_seed_rows(
            (row["metadata"] for row in ncc_rows),
            contract=contract,
            label=f"{phase} count-geometry NCC",
        )

        terminal_rows = _read_jsonl_shards(root / "terminal_state_bridge")
        _audit_seed_rows(
            terminal_rows,
            contract=contract,
            label=f"{phase} terminal-state bridge",
        )

        token_audits = {}
        for name, conditions in (
            ("token_ablation_answer", ANSWER_CONDITIONS),
            ("token_ablation_targeting", TARGETING_CONDITIONS),
        ):
            rows = _read_jsonl_shards(root / name)
            _audit_seed_rows(rows, contract=contract, label=f"{phase} {name}")
            bad = [row for row in rows if str(row.get("status", "ok")) != "ok"]
            if bad:
                raise ValueError(
                    f"{phase} {name} contains {len(bad)} excluded/error rows; "
                    "resume the assay rather than silently reducing n"
                )
            observed_conditions = {str(row["condition"]) for row in rows}
            if observed_conditions != conditions:
                raise ValueError(
                    f"{phase} {name} factorial changed: "
                    f"expected={sorted(conditions)} observed={sorted(observed_conditions)}"
                )
            by_seed = pd.DataFrame(rows).groupby("seed")["condition"].nunique()
            if not by_seed.eq(len(conditions)).all():
                raise ValueError(f"{phase} {name} has an incomplete seed factorial")
            token_audits[name] = {
                "row_count": len(rows),
                "condition_count": len(conditions),
                "all_rows_status_ok": True,
            }

        audit[phase] = {
            "true_source_seed_count": len(contract["true_source_seeds"]),
            "targeted_counter_write_rows": len(write_frame),
            "stratified_ncc_shards": len(stratified_rows),
            "direct_count_logit_margin_shards": len(logit_rows),
            "count_geometry_ncc_shards": len(ncc_rows),
            "terminal_state_bridge_rows": len(terminal_rows),
            "token_ablation": token_audits,
        }
    return audit


def _load_npz_metadata(root: Path, experiment_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    plan = json.loads((root / "frozen_row_plan.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "shards").glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
        if str(metadata.get("experiment_id")) != experiment_id:
            raise ValueError(f"Unexpected experiment under {root}: {metadata.get('experiment_id')}")
        rows.append(metadata)
    if not rows:
        raise FileNotFoundError(f"No NPZ shards under {root}")
    return rows, plan


def _read_jsonl_shards(root: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for path in sorted((root / "shards").glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise FileNotFoundError(f"No JSONL shards under {root}")
    return rows


def _run_legacy(script: str, arguments: Sequence[str]) -> dict[str, Any]:
    command = [sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "script": script,
        "script_sha256": sha256_file(ROOT / "scripts" / script),
        "argv": command[2:],
        "stdout_tail": completed.stdout[-4000:],
    }


def _analyze_counter_write(
    discovery_root: Path,
    confirmation_root: Path,
    output: Path,
    discovery: Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> dict[str, Any]:
    write.COUNT_STREAM_DISCOVERY_SEEDS = tuple(discovery["true_source_seeds"])
    write.COUNT_STREAM_CONFIRMATION_SEEDS = tuple(confirmation["true_source_seeds"])
    results = {}
    for phase, root in (
        ("discovery", discovery_root),
        ("confirmation", confirmation_root),
    ):
        active = output / phase
        effects, claim = write.analyze(
            write._read(root / "targeted_counter_write"),
            phase=phase,
            random_seed=20260822,
        )
        write._atomic_csv(active / "seed_effects.csv", effects)
        write._atomic_json(active / "claim_gates.json", claim)
        write._atomic_json(
            active / "audit.json",
            {
                "status": "PASS",
                "seed_count": int(claim["seed_count"]),
                "conditions_per_seed": 7,
                "teacher_forced_trace_tokens": True,
                "selection_rank_used": False,
                "v6_seed_membership_adapter": "true_source_seed_from_registry",
            },
        )
        results[phase] = claim
    return results


def _analyze_count_ncc(
    discovery_root: Path,
    confirmation_root: Path,
    output: Path,
    discovery: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    timing: str,
) -> dict[str, Any]:
    ncc.COUNT_STREAM_DISCOVERY_SEEDS = tuple(discovery["true_source_seeds"])
    ncc.COUNT_STREAM_CONFIRMATION_SEEDS = tuple(confirmation["true_source_seeds"])
    # The legacy report pooled its two native numbered-list timing strata.
    # Each V6 enumeration mode has exactly one honest, mode-native stratum.
    ncc.TIMINGS = (str(timing),)
    layer_metrics, effects, predictions, result = ncc.analyze(
        ncc._load(discovery_root / "count_geometry_ncc"),
        ncc._load(confirmation_root / "count_geometry_ncc"),
    )
    output.mkdir(parents=True, exist_ok=True)
    layer_metrics.to_csv(output / "layer_metrics.csv", index=False)
    effects.to_csv(output / "seed_effects.csv", index=False)
    predictions.to_csv(output / "confirmation_predictions.csv", index=False)
    ncc._atomic_json(output / "claim_gates.json", result)
    ncc._atomic_json(
        output / "audit.json",
        {
            "status": "PASS",
            "discovery_seed_count": len(discovery["true_source_seeds"]),
            "confirmation_seed_count": len(confirmation["true_source_seeds"]),
            "confirmation_used_for_fit_or_layer_selection": False,
            "outcome_blind": True,
            "selection_rank_used": False,
            "v6_seed_membership_adapter": "true_source_seed_from_registry",
        },
    )
    return result


def _analyze_stratified(
    discovery_root: Path,
    confirmation_root: Path,
    output: Path,
    *,
    timing: str,
) -> dict[str, Any]:
    from scripts import analyze_realistic_niah_v5_stratified_targeted_counter_ncc as module

    discovery, discovery_plan = module._load(discovery_root / "stratified_ncc")
    confirmation, confirmation_plan = module._load(
        confirmation_root / "stratified_ncc"
    )
    layer_metrics, effects, predictions, result = module.analyze(
        discovery,
        discovery_plan,
        confirmation,
        confirmation_plan,
        timing=timing,
    )
    output.mkdir(parents=True, exist_ok=True)
    layer_metrics.to_csv(output / "layer_metrics.csv", index=False)
    effects.to_csv(output / "seed_effects.csv", index=False)
    predictions.to_csv(output / "confirmation_predictions.csv", index=False)
    module._atomic_json(output / "claim_gates.json", result)
    module._atomic_json(
        output / "audit.json",
        {
            "status": "PASS",
            "model_label": result["model_label"],
            "timing_branch": timing,
            "development_seed_count": len(discovery),
            "confirmation_seed_count": len(confirmation),
            "confirmation_used_for_fit_or_layer_selection": False,
            "capture_layer_rule": "strictly_above_all_ablated_head_layers",
            "outcome_blind": True,
            "selection_rank_used": False,
        },
    )
    return result


def _analyze_logit_margin(
    discovery_root: Path,
    confirmation_root: Path,
    output: Path,
    *,
    timing: str,
) -> dict[str, Any]:
    from scripts import analyze_realistic_niah_v5_targeted_counter_logit_margin as module

    discovery, discovery_plan = module._load(
        discovery_root / "direct_count_logit_margin"
    )
    confirmation, confirmation_plan = module._load(
        confirmation_root / "direct_count_logit_margin"
    )
    long, effects, claim = module.analyze(
        discovery,
        discovery_plan,
        confirmation,
        confirmation_plan,
        timing=timing,
    )
    output.mkdir(parents=True, exist_ok=True)
    long.to_csv(output / "condition_margins.csv", index=False)
    effects.to_csv(output / "seed_effects.csv", index=False)
    module._atomic_json(output / "claim_gates.json", claim)
    module._atomic_json(
        output / "audit.json",
        {
            "status": "PASS",
            "model_label": claim["model_label"],
            "timing_branch": timing,
            "discovery_seed_count": len(discovery),
            "confirmation_seed_count": len(confirmation),
            "confirmation_used_for_registration": False,
            "no_decoder_fit_or_layer_selection": True,
            "outcome_blind": True,
            "selection_rank_used": False,
        },
    )
    return claim


def _analysis_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {"analysis_manifest.json", "confirmation_analysis.COMPLETE"}
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-config", type=Path, required=True)
    parser.add_argument("--confirmation-freeze", type=Path, required=True)
    parser.add_argument("--model", choices=("Qwen3-8B", "Gemma4-E4B"), required=True)
    parser.add_argument(
        "--timing", choices=("rank_before_city", "structural_item_end"), required=True
    )
    parser.add_argument("--discovery-root", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--discovery-registry", type=Path, required=True)
    parser.add_argument("--confirmation-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = V6Config.load(args.v6_config)
    validate_confirmation_freeze(
        args.confirmation_freeze,
        prompt_mode=config.prompt_mode,
        model_label=args.model,
    )
    geometry_audit = install_v6_specialized_geometry(config.prompt_mode)
    discovery = _registry_contract(
        args.discovery_registry,
        config=config,
        model_label=args.model,
        split="discovery",
    )
    confirmation = _registry_contract(
        args.confirmation_registry,
        config=config,
        model_label=args.model,
        split="confirmation",
    )
    preflight = _audit_specialized_inputs(
        args.discovery_root,
        args.confirmation_root,
        discovery=discovery,
        confirmation=confirmation,
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    numerical: dict[str, Any] = {}
    numerical["targeted_counter_write"] = _analyze_counter_write(
        args.discovery_root,
        args.confirmation_root,
        output / "targeted_counter_write",
        discovery,
        confirmation,
    )

    numerical["stratified_ncc"] = _analyze_stratified(
        args.discovery_root,
        args.confirmation_root,
        output / "stratified_ncc",
        timing=args.timing,
    )
    numerical["direct_count_logit_margin"] = _analyze_logit_margin(
        args.discovery_root,
        args.confirmation_root,
        output / "direct_count_logit_margin",
        timing=args.timing,
    )
    numerical["count_geometry_ncc"] = _analyze_count_ncc(
        args.discovery_root,
        args.confirmation_root,
        output / "count_geometry_ncc",
        discovery,
        confirmation,
        args.timing,
    )
    legacy_calls = []
    for phase, root in (
        ("discovery", args.discovery_root),
        ("confirmation", args.confirmation_root),
    ):
        legacy_calls.append(
            _run_legacy(
                "analyze_realistic_niah_v5_terminal_token_state_bridge.py",
                (
                    "--input", root / "terminal_state_bridge",
                    "--phase", phase,
                    "--bootstrap-samples", "10000",
                    "--random-seed", "20260821",
                    "--output", output / "terminal_state_bridge" / phase,
                ),
            )
        )
        for name in ("token_ablation_answer", "token_ablation_targeting"):
            legacy_calls.append(
                _run_legacy(
                    "analyze_realistic_niah_v5_token_level_ablation.py",
                    (
                        "--input", root / name,
                        "--output", output / name / phase,
                    ),
                )
            )
            token_audit = json.loads(
                (output / name / phase / "analysis_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            if (
                int(token_audit["seed_count"])
                != len(
                    discovery["true_source_seeds"]
                    if phase == "discovery"
                    else confirmation["true_source_seeds"]
                )
                or int(token_audit["excluded_rows"]) != 0
            ):
                raise ValueError(f"{phase} {name} analysis silently changed n")

    manifest = {
        "schema_version": "realistic_niah_v6_specialized_confirmation_analysis_v1",
        "status": "PASS",
        "model_label": args.model,
        "prompt_mode": config.prompt_mode,
        "mode_timing_stratum": args.timing,
        "confirmation_freeze": str(args.confirmation_freeze.resolve()),
        "confirmation_freeze_sha256": sha256_file(args.confirmation_freeze),
        "panel_membership_identity": "analysis_slot_seed",
        "statistical_identity": "true_source_seed",
        "seed_aliasing": False,
        "confirmation_used_for_fit_or_layer_selection": False,
        "intervention_outcomes_used_for_replacement": False,
        "discovery_contract": discovery,
        "confirmation_contract": confirmation,
        "raw_preflight": preflight,
        "specialized_geometry_adapter": geometry_audit,
        "membership_adapted_v5_analyzers": [
            "analyze_realistic_niah_v5_targeted_counter_write.py",
            "analyze_realistic_niah_v5_targeted_counter_ncc.py",
        ],
        "membership_adapter_scope": (
            "expected-seed validation constants plus the single honest V6 "
            "mode-timing registry; no numerical calculation, condition, layer, "
            "endpoint fit, selection rule, or aggregation changed"
        ),
        "legacy_analysis_calls": legacy_calls,
        "numerical_kernel_summaries": numerical,
    }
    manifest["analysis_files_sha256"] = _analysis_hashes(output)
    _atomic_json(output / "analysis_manifest.json", manifest)
    _atomic_text(output / "confirmation_analysis.COMPLETE", "PASS\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "model_label": args.model,
                "prompt_mode": config.prompt_mode,
                "discovery_seed_count": len(discovery["true_source_seeds"]),
                "confirmation_seed_count": len(confirmation["true_source_seeds"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
