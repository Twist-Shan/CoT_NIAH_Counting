#!/usr/bin/env python3
"""Freeze every discovery-selected V6 choice before opening confirmation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from realistic_niah_v6.pipeline import sha256_file  # noqa: E402
from realistic_niah_v6.spec import MODEL_LABELS, V6Config  # noqa: E402
from realistic_niah_v6.suite import (  # noqa: E402
    discovery_ledger_template,
    freeze_confirmation,
    validate_confirmation_freeze,
)


SCHEMA_VERSION = "realistic_niah_v6_discovery_freeze_builder_v1"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing discovery freeze artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Discovery artifact is not one JSON object: {path}")
    return value


def _require_pass_marker(path: Path) -> None:
    if not path.is_file() or path.read_text(encoding="utf-8").strip() != "PASS":
        raise ValueError(f"Discovery completion marker is absent or not PASS: {path}")


def _cell(
    template: Mapping[str, Any],
    *,
    choice: Any,
    artifacts: list[Path],
    negative_result_retained: bool = True,
) -> dict[str, Any]:
    for path in artifacts:
        if not path.is_file():
            raise FileNotFoundError(f"Frozen discovery artifact is absent: {path}")
    if choice is None:
        raise ValueError("A discovery freeze cell must have an explicit choice")
    return {
        **dict(template),
        "status": "FROZEN",
        "choice": choice,
        "negative_result_retained": bool(negative_result_retained),
        "artifact_paths": [str(path.resolve()) for path in artifacts],
    }


def build_freeze(
    *,
    config_path: Path,
    mechanism_config: Path,
    model_label: str,
    model_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = V6Config.load(config_path)
    if model_label not in config.model_labels:
        raise ValueError(f"Model {model_label!r} is outside the V6 config")

    discovery_registry = model_root / "replacement/discovery/selected_cells.jsonl"
    broad_registry = (
        model_root / "replacement/discovery_broad_k/selected_cells.jsonl"
    )
    native_loop_registry = (
        model_root / "replacement/discovery_native_loop/selected_cells.jsonl"
    )
    native_loop_registry_manifest = (
        model_root / "replacement/discovery_native_loop/manifest.json"
    )
    targeted_root = model_root / "causal/targeted_retrieval/discovery_formal"
    count_root = model_root / "count_stream/discovery_formal"
    specialized_root = model_root / "causal/specialized/discovery_formal"
    report_root = model_root / "causal/report_tail/discovery_formal"

    for marker in (
        model_root / "discovery-foundation-resolved.COMPLETE",
        model_root / "replacement/discovery/discovery.COMPLETE",
        model_root
        / "replacement/discovery_broad_k/k_selection_discovery.COMPLETE",
        targeted_root / "all.COMPLETE",
        count_root / "stage1.COMPLETE",
        specialized_root / "discovery.COMPLETE",
        report_root / "discovery.COMPLETE",
    ):
        _require_pass_marker(marker)

    representation = model_root / "representation/formal/v6_adapter_manifest.json"
    selection_path = targeted_root / "analysis/selection.json"
    count_stage_path = count_root / "stage1_complete.json"
    basis_path = count_root / "running_basis.npz"
    basis_manifest = count_root / "running_basis.json"
    trace_analysis = count_root / "trace_patch_analysis/manifest.json"
    specialized_complete = specialized_root / "specialized_discovery_complete.json"
    counter_write = specialized_root / "targeted_counter_write/manifest.json"
    token_answer = specialized_root / "token_ablation_answer/worker_00_manifest.json"
    terminal_bridge = specialized_root / "terminal_state_bridge/manifest.json"
    stratified_ncc = specialized_root / "stratified_ncc/manifest.json"
    direct_margin = specialized_root / "direct_count_logit_margin/manifest.json"
    count_ncc = specialized_root / "count_geometry_ncc/manifest.json"
    natural_analysis = (
        report_root / "natural_layer_sweep/layer_sweep_analysis.json"
    )
    native_contract = report_root / "native_loop/contract/manifest.json"
    native_gates = report_root / "native_loop/analysis/claim_gates.json"
    restoration_analysis = report_root / "restoration/analysis/manifest.json"

    selection = _read(selection_path)
    if selection.get("status") != "DISCOVERY_FROZEN_CHOICE":
        raise ValueError("Targeted-retrieval K is not discovery-frozen")
    if selection.get("model_label") != model_label:
        raise ValueError("Targeted-retrieval selection has the wrong model")
    if selection.get("prompt_mode") != config.prompt_mode:
        raise ValueError("Targeted-retrieval selection has the wrong prompt mode")
    selected_k = int(selection["selected_k"])
    selected_plan = (
        targeted_root
        / "plans"
        / f"k{selected_k}"
        / "retrieval_anchor_bank_plan.csv"
    )
    if not selected_plan.is_file():
        raise FileNotFoundError(f"Selected targeted bank plan is absent: {selected_plan}")
    if sha256_file(selected_plan) != str(selection["frozen_plan_sha256"]):
        raise ValueError("Selected targeted bank plan changed after discovery")

    count_stage = _read(count_stage_path)
    if count_stage.get("status") != "DISCOVERY_COMPLETE":
        raise ValueError("Count-stream discovery stage is not complete")
    natural = _read(natural_analysis)
    if natural.get("selection_split") != "discovery":
        raise ValueError("Natural scope/layer analysis is not discovery-only")
    natural_choices = {
        str(row["scope"]): {
            "status": str(row["status"]),
            "selected_layer": row.get("selected_layer"),
            "negative_result_retained": bool(
                row.get("negative_result_retained", False)
            ),
        }
        for row in natural.get("scopes", [])
    }
    if not natural_choices:
        raise ValueError("Natural scope/layer analysis has no frozen scopes")

    ledger = discovery_ledger_template(
        prompt_mode=config.prompt_mode,
        model_label=model_label,
    )
    cells = ledger["experiments"]
    bank_choice = {
        "selected_k": selected_k,
        "selection_status": str(selection["status"]),
        "selection_rule": str(selection.get("selection_rule", "")),
        "selected_plan_sha256": sha256_file(selected_plan),
        "confirmation_may_not_reselect_k": True,
    }
    fixed_layer_choice = {
        "source_layer": int(
            _read(counter_write).get(
                "source_layer",
                19 if model_label == "Qwen3-8B" else 16,
            )
        ),
        "selected_k": selected_k,
        "discovery_bank_reused_unchanged": True,
    }

    cells["layerwise_representation"] = _cell(
        cells["layerwise_representation"],
        choice={
            "fit_split": "discovery",
            "evaluation_split": "confirmation",
            "layer_policy": "retain_complete_registered_layer_sweep",
            "post_confirmation_layer_reselection_forbidden": True,
        },
        artifacts=[representation, discovery_registry],
    )
    natural_artifacts = [natural_analysis]
    cells["trace_scope_layer_sweep"] = _cell(
        cells["trace_scope_layer_sweep"],
        choice={"scope_to_discovery_selected_layer": natural_choices},
        artifacts=natural_artifacts,
    )
    cells["visible_progress_positive_control"] = _cell(
        cells["visible_progress_positive_control"],
        choice={
            "scope_to_discovery_selected_layer": natural_choices,
            "same_frozen_assay_for_confirmation": True,
        },
        artifacts=natural_artifacts,
    )
    cells["targeted_retrieval_bank"] = _cell(
        cells["targeted_retrieval_bank"],
        choice=bank_choice,
        artifacts=[selection_path, selected_plan],
    )
    for experiment_id in (
        "targeted_query_to_carrier",
        "carrier_to_commit_restore",
    ):
        cells[experiment_id] = _cell(
            cells[experiment_id],
            choice=fixed_layer_choice,
            artifacts=[selection_path, selected_plan, counter_write],
        )
    cells["progress_state_to_successor"] = _cell(
        cells["progress_state_to_successor"],
        choice={
            "source_layer": int(count_stage["source_layer"]),
            "readout_layers": list(count_stage["readout_layers"]),
            "trace_pair_sampling_rule": "registered_sparse_panel",
            "basis_fit_split": "discovery",
        },
        artifacts=[count_stage_path, basis_path, basis_manifest, trace_analysis],
    )
    cells["answer_token_source_ablation"] = _cell(
        cells["answer_token_source_ablation"],
        choice={
            **bank_choice,
            "conditions_and_controls": "registered_token_blank_factorial",
        },
        artifacts=[selection_path, selected_plan, token_answer],
    )
    cells["terminal_state_bridge"] = _cell(
        cells["terminal_state_bridge"],
        choice={
            **fixed_layer_choice,
            "conditions": "registered_terminal_token_state_conditions",
        },
        artifacts=[terminal_bridge],
    )
    cells["timing_stratified_ncc"] = _cell(
        cells["timing_stratified_ncc"],
        choice={
            **fixed_layer_choice,
            "timing_stratum": (
                "rank_before_city"
                if config.prompt_mode == "enumeration_index"
                else "structural_item_end"
            ),
            "fit_split": "discovery",
        },
        artifacts=[stratified_ncc, specialized_complete],
    )
    cells["direct_count_output_margin"] = _cell(
        cells["direct_count_output_margin"],
        choice={
            **fixed_layer_choice,
            "decoder_endpoint": "gold_vs_best_wrong_autoregressive_sequence_margin",
        },
        artifacts=[direct_margin],
    )
    cells["count_geometry_ncc"] = _cell(
        cells["count_geometry_ncc"],
        choice={
            **fixed_layer_choice,
            "centroid_fit_split": "discovery",
            "confirmation_refit_forbidden": True,
        },
        artifacts=[count_ncc],
    )
    cells["commit_to_query_patch"] = _cell(
        cells["commit_to_query_patch"],
        choice={
            **fixed_layer_choice,
            "native_loop_offsets": [-3, -2, -1, 1, 2, 3],
            "basis_fit_split": "discovery",
        },
        artifacts=[
            native_contract,
            native_gates,
            basis_path,
            basis_manifest,
            native_loop_registry,
            native_loop_registry_manifest,
        ],
    )
    cells["format_specific_source_scrub_restore"] = _cell(
        cells["format_specific_source_scrub_restore"],
        choice={
            **fixed_layer_choice,
            "conditions": "registered_full_source_scrub_restore_factorial",
        },
        artifacts=[restoration_analysis],
    )

    ledger["status"] = "DISCOVERY_FROZEN"
    ledger["confirmation_outcomes_read"] = False
    ledger["builder_schema_version"] = SCHEMA_VERSION
    ledger["cohort_registries"] = {
        "cell_resolved": str(discovery_registry.resolve()),
        "coherent_broad_k": str(broad_registry.resolve()),
        "coherent_native_loop": str(native_loop_registry.resolve()),
    }
    ledger["seed_identity_contract"] = {
        "panel_membership_identity": "analysis_slot_seed",
        "statistical_identity": "true_source_seed",
        "seed_aliasing": False,
    }

    artifact_paths = sorted(
        {
            Path(path).resolve()
            for cell in cells.values()
            for path in cell["artifact_paths"]
        },
        key=str,
    )
    freeze = freeze_confirmation(
        prompt_mode=config.prompt_mode,
        model_label=model_label,
        discovery_ledger=ledger,
        artifact_paths=artifact_paths,
    )
    validate_confirmation_freeze(
        freeze,
        prompt_mode=config.prompt_mode,
        model_label=model_label,
    )

    mechanism = _read(mechanism_config)
    if mechanism.get("schema_version") != "realistic_niah_v6_count_stream_v1":
        raise ValueError("Mechanism config has the wrong V6 schema")
    if mechanism.get("status") != "development_only":
        raise ValueError("Only a development-only mechanism config may be frozen")
    if config.prompt_mode not in str(mechanism.get("experiment_id", "")):
        raise ValueError("Mechanism config has the wrong enumeration mode")
    # Keep this file loadable by the frozen NativeCountMechanismSpec.  Freeze
    # hashes live in the adjacent immutable confirmation manifest/audit rather
    # than as unrecognized numerical-kernel config fields.
    frozen_mechanism = {**mechanism, "status": "frozen_confirmation"}
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_DISCOVERY_LOCKED_BEFORE_CONFIRMATION",
        "model_label": model_label,
        "prompt_mode": config.prompt_mode,
        "experiment_count": len(cells),
        "artifact_count": len(artifact_paths),
        "selected_k": selected_k,
        "freeze_sha256": str(freeze["freeze_sha256"]),
        "confirmation_outcomes_read": False,
        "negative_results_retained": True,
        "seed_aliasing": False,
    }
    return ledger, freeze, frozen_mechanism, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-config", type=Path, required=True)
    parser.add_argument("--mechanism-config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_LABELS, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ledger, freeze, mechanism, audit = build_freeze(
        config_path=args.v6_config.resolve(),
        mechanism_config=args.mechanism_config.resolve(),
        model_label=str(args.model),
        model_root=args.model_root.resolve(),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output / "discovery_ledger.json", ledger)
    _atomic_json(args.output / "confirmation_freeze.json", freeze)
    _atomic_json(args.output / "mechanism_frozen_confirmation.json", mechanism)
    _atomic_json(args.output / "freeze_audit.json", audit)
    (args.output / "freeze.COMPLETE").write_text("PASS\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
