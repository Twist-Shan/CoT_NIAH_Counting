from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .spec import (
    CONFIRMATION_SEEDS,
    COUNTS,
    DISCOVERY_SEEDS,
    MODEL_LABELS,
    PROMPT_MODES,
)
from .suite import EXPERIMENTS, suite_document, validate_confirmation_freeze


COMPLETION_SCHEMA_VERSION = "realistic_niah_v6_suite_completion_audit_v4_native_aligned"


TARGETED_REPORT_GRIDS = {
    "Qwen3-8B": (32, 64, 80, 96, 112, 128),
    "Gemma4-E4B": (1, 2, 4, 6, 8),
}
TARGETED_REPORT_REFERENCE_K = {"Qwen3-8B": 128, "Gemma4-E4B": 6}
TARGETED_RANDOM_CONDITION = {
    "Qwen3-8B": {
        32: "layer_matched_random",
        64: "layer_matched_random",
        80: "layer_matched_random",
        96: "layer_matched_random",
        112: "layer_matched_random",
        128: "global_random",
    },
    "Gemma4-E4B": {
        1: "layer_matched_random",
        2: "layer_matched_random",
        4: "layer_matched_random",
        6: "layer_matched_random",
        8: "layer_matched_random",
    },
}


DISCOVERY_COMPLETION_FILES = {
    "targeted_retrieval": (
        "causal/targeted_retrieval/discovery_formal/all.COMPLETE",
        None,
    ),
    "count_stream": (
        "count_stream/discovery_formal/stage1.COMPLETE",
        "count_stream/discovery_formal/stage1_complete.json",
    ),
    "specialized": (
        "causal/specialized/discovery_formal/discovery.COMPLETE",
        "causal/specialized/discovery_formal/specialized_discovery_complete.json",
    ),
    "report_tail": (
        "causal/report_tail/discovery_formal/discovery.COMPLETE",
        "causal/report_tail/discovery_formal/report_tail_discovery_complete.json",
    ),
}


CONFIRMATION_COMPLETION_FILES = {
    "foundation": (
        "freeze/confirmation-foundation.COMPLETE",
        "freeze/confirmation_foundation_complete.json",
    ),
    "targeted_retrieval": (
        "causal/targeted_retrieval/confirmation_formal/confirmation.COMPLETE",
        "causal/targeted_retrieval/confirmation_formal/confirmation_complete.json",
    ),
    "count_stream": (
        "count_stream/confirmation_formal/confirmation.COMPLETE",
        "count_stream/confirmation_formal/confirmation_complete.json",
    ),
    "specialized": (
        "causal/specialized/confirmation_formal/confirmation.COMPLETE",
        "causal/specialized/confirmation_formal/specialized_confirmation_complete.json",
    ),
    "report_tail": (
        "causal/report_tail/confirmation_formal/confirmation.COMPLETE",
        "causal/report_tail/confirmation_formal/report_tail_confirmation_complete.json",
    ),
}


# Each entry is the minimal report-frame-specific evidence beyond the common
# cell completion and replacement audits.  The first tuple is discovery; the
# second is confirmation.  Design-only frames intentionally have no held-out
# result requirement in the registry.
FRAME_EVIDENCE: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "behavior_and_parser_baseline": (
        ("generation/manifest_discovery.json", "replacement/discovery/manifest.json"),
        ("generation/manifest_confirmation.json", "replacement/confirmation/manifest.json"),
    ),
    "layerwise_representation": (
        # The already-frozen generic discovery sweep is retained only as
        # historical localization provenance.  Primary confirmation evidence
        # is the post-capture Native-aligned two-endpoint analysis below.
        ("representation/formal/v6_adapter_manifest.json",),
        ("representation/native_aligned/cell_manifest.json",),
    ),
    "paired_causal_estimands": (
        ("causal/targeted_retrieval/discovery_formal/analysis/selection.json",),
        (),
    ),
    "trace_scope_layer_sweep": (
        ("causal/report_tail/discovery_formal/natural_layer_sweep/layer_sweep_analysis.json",),
        ("causal/report_tail/confirmation_formal/natural_selected/confirmation_analysis.json",),
    ),
    "targeted_retrieval_bank": (
        ("causal/targeted_retrieval/discovery_formal/analysis/selection.json",),
        ("causal/targeted_retrieval/confirmation_formal/analysis.json",),
    ),
    "seed_equal_sampling_contract": (
        ("replacement/discovery/selected_cells.jsonl",),
        (),
    ),
    "targeted_query_to_carrier": (
        ("causal/specialized/discovery_formal/targeted_counter_write/manifest.json",),
        ("causal/specialized/confirmation_analysis/analysis_manifest.json",),
    ),
    "carrier_to_commit_restore": (
        ("causal/specialized/discovery_formal/targeted_counter_write/manifest.json",),
        ("causal/specialized/confirmation_analysis/analysis_manifest.json",),
    ),
    "progress_state_to_successor": (
        ("count_stream/discovery_formal/trace_patch_analysis/manifest.json",),
        ("count_stream/confirmation_formal/trace_patch_analysis/manifest.json",),
    ),
    "answer_token_source_ablation": (
        ("causal/specialized/discovery_formal/token_ablation_answer/worker_00_manifest.json",),
        ("causal/specialized/confirmation_analysis/analysis_manifest.json",),
    ),
    "terminal_state_bridge": (
        ("causal/specialized/discovery_formal/terminal_state_bridge/manifest.json",),
        ("causal/specialized/confirmation_analysis/analysis_manifest.json",),
    ),
    "timing_stratified_ncc": (
        ("causal/specialized/discovery_formal/stratified_ncc/manifest.json",),
        ("causal/specialized/confirmation_analysis/analysis_manifest.json",),
    ),
    "direct_count_output_margin": (
        ("causal/specialized/discovery_formal/direct_count_logit_margin/manifest.json",),
        ("causal/specialized/confirmation_analysis/analysis_manifest.json",),
    ),
    "count_geometry_ncc": (
        ("causal/specialized/discovery_formal/count_geometry_ncc/manifest.json",),
        ("causal/specialized/confirmation_analysis/analysis_manifest.json",),
    ),
    "layer_timing_diagnostic": (
        ("causal/specialized/discovery_formal/stratified_ncc/manifest.json",),
        (),
    ),
    "visible_progress_positive_control": (
        ("causal/report_tail/discovery_formal/natural_layer_sweep/layer_sweep_analysis.json",),
        ("causal/report_tail/confirmation_formal/natural_selected/confirmation_analysis.json",),
    ),
    "commit_to_query_patch": (
        ("causal/report_tail/discovery_formal/native_loop/analysis/claim_gates.json",),
        ("causal/report_tail/confirmation_formal/native_loop/analysis/claim_gates.json",),
    ),
    "single_seed_walkthrough": (
        (),
        ("causal/report_tail/confirmation_formal/single_seed_walkthrough/analysis/walkthrough_complete.json",),
    ),
    "format_specific_source_scrub_restore": (
        ("causal/report_tail/discovery_formal/restoration/analysis/manifest.json",),
        ("causal/report_tail/confirmation_formal/restoration/analysis/manifest.json",),
    ),
    "scrub_coverage_and_cross_mode_audit": (
        ("suite_audit.json", "replacement/discovery/manifest.json"),
        (),
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def _ordinary_failed_reserve_attempts(
    attempts: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return failed generated reserve candidates across ledger label versions."""
    return [
        row
        for row in attempts
        if row.get("candidate_kind") in {"replacement", "reserve"}
        and not bool(row.get("eligible"))
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _require_pass(path: Path) -> None:
    if not path.is_file() or path.read_text(encoding="utf-8").strip() != "PASS":
        raise ValueError(f"Completion marker is absent or not PASS: {path}")


def _assert_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing hashed {label}: {path}")
    observed = _sha256(path)
    if observed != str(expected):
        raise ValueError(
            f"Hash mismatch for {label}: {path}; expected={expected}, observed={observed}"
        )


def _resolve_coherent_policy_lineage(
    manifest: Mapping[str, Any],
    *,
    panel_kind: str,
    manifest_path: Path,
) -> tuple[Path, str]:
    """Resolve current or legacy panel-specific policy lineage fail-closed."""
    if panel_kind == "broad":
        specific_prefix = "coherent_broad_policy"
    elif panel_kind == "native_loop":
        specific_prefix = "coherent_native_loop_policy"
    else:
        raise ValueError(f"Unknown coherent panel kind: {panel_kind}")

    generic_path = str(manifest.get("coherent_policy", "")).strip()
    generic_hash = str(manifest.get("coherent_policy_sha256", "")).strip()
    specific_path = str(manifest.get(specific_prefix, "")).strip()
    specific_hash = str(manifest.get(f"{specific_prefix}_sha256", "")).strip()

    if generic_path and specific_path and generic_path != specific_path:
        raise ValueError(f"Coherent panel policy paths disagree: {manifest_path}")
    if generic_hash and specific_hash and generic_hash != specific_hash:
        raise ValueError(f"Coherent panel policy hashes disagree: {manifest_path}")

    resolved_path = generic_path or specific_path
    resolved_hash = generic_hash or specific_hash
    if not resolved_path or not resolved_hash:
        raise ValueError(f"Coherent panel has no hashed policy lineage: {manifest_path}")
    return Path(resolved_path), resolved_hash


_FOUNDATION_RECOVERY_PATHS = {
    "formal_capture": "capture/formal/capture_manifest.json",
    "formal_capture_adapter": "capture/formal/v6_adapter_manifest.json",
    "all_capture": "capture/all_sample/capture_manifest.json",
    "all_capture_adapter": "capture/all_sample/v6_adapter_manifest.json",
    "formal_attention": "attention/discovery_formal.manifest.json",
    "all_attention": "attention/discovery_all_sample.manifest.json",
    "formal_answer_query": "attention/discovery_answer_query_formal.manifest.json",
}
_RESOLVED_FOUNDATION_SUPERSEDES = {
    "formal_capture",
    "formal_capture_adapter",
    "formal_attention",
    "formal_answer_query",
}


def _audit_foundation_recovery_evidence(
    model_root: Path,
    *,
    prompt_mode: str,
    model_label: str,
    recovery_path: Path,
    validated: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Audit historical marker evidence and its designed resolved-cohort successors."""
    registry_path = model_root / "replacement/discovery/selected_cells.jsonl"
    registry_hash = _sha256(registry_path)
    registry_rows = len(_read_jsonl(registry_path))
    states: dict[str, dict[str, Any]] = {}

    for name, artifact in validated.items():
        if name not in _FOUNDATION_RECOVERY_PATHS or not isinstance(artifact, Mapping):
            raise ValueError(f"Malformed marker-recovery artifact {name}: {recovery_path}")
        current_path = Path(str(artifact.get("path", "")))
        canonical_path = model_root / _FOUNDATION_RECOVERY_PATHS[name]
        if current_path.resolve() != canonical_path.resolve():
            raise ValueError(
                f"Foundation marker recovery path changed for {name}: {recovery_path}"
            )
        expected_hash = str(artifact.get("sha256", ""))
        if not current_path.is_file() or not expected_hash:
            raise ValueError(
                f"Foundation marker recovery has incomplete evidence for {name}: "
                f"{recovery_path}"
            )
        current_hash = _sha256(current_path)
        if current_hash == expected_hash:
            states[name] = {
                "path": str(current_path.resolve()),
                "historical_sha256": expected_hash,
                "current_sha256": current_hash,
                "status": "PASS_HISTORICAL_HASH_RETAINED",
            }
            continue

        if name not in _RESOLVED_FOUNDATION_SUPERSEDES:
            raise ValueError(
                f"Unexpected foundation recovery evidence hash change for {name}: "
                f"{current_path}; expected={expected_hash}, observed={current_hash}"
            )
        _require_pass(model_root / "discovery-foundation-resolved.COMPLETE")
        if current_path.stat().st_mtime_ns < recovery_path.stat().st_mtime_ns:
            raise ValueError(
                f"Resolved foundation artifact predates marker recovery: {current_path}"
            )

        value = _read_json(current_path)
        if name == "formal_capture":
            split_counts = value.get("split_trajectory_counts")
            if (
                int(value.get("input_rows", -1)) != registry_rows
                or int(value.get("rows", -1)) != registry_rows
                or int(value.get("excluded_rows", -1)) != 0
                or not isinstance(split_counts, Mapping)
                or int(split_counts.get("discovery", -1)) != registry_rows
                or int(split_counts.get("confirmation", -1)) != 0
            ):
                raise ValueError(
                    f"Resolved formal capture does not match discovery registry: "
                    f"{current_path}"
                )
        else:
            _validate_common_identity(
                value,
                prompt_mode=prompt_mode,
                model_label=model_label,
            )
            declared_registry = Path(str(value.get("cohort_registry", "")))
            if (
                declared_registry.resolve() != registry_path.resolve()
                or value.get("cohort_registry_sha256") != registry_hash
                or value.get("formal_cohort") is not True
            ):
                raise ValueError(
                    f"Resolved foundation lineage changed for {name}: {current_path}"
                )
            if name == "formal_capture_adapter":
                if value.get("run_status") != "COMPLETE" or value.get("status") != "INSTALLED":
                    raise ValueError(f"Resolved capture adapter is incomplete: {current_path}")
            elif (
                value.get("seed_role") != "discovery"
                or int(value.get("requests", -1)) != registry_rows
            ):
                raise ValueError(
                    f"Resolved attention manifest is incomplete: {current_path}"
                )

        states[name] = {
            "path": str(current_path.resolve()),
            "historical_sha256": expected_hash,
            "current_sha256": current_hash,
            "resolved_registry": str(registry_path.resolve()),
            "resolved_registry_sha256": registry_hash,
            "status": "PASS_SUPERSEDED_BY_RESOLVED_FOUNDATION",
        }
    return states


def _verify_named_artifacts(
    manifest: Mapping[str, Any],
    *,
    keys: Iterable[str] = ("outputs", "artifacts", "manifests"),
) -> int:
    verified = 0
    for key in keys:
        values = manifest.get(key)
        if not isinstance(values, Mapping):
            continue
        for name, raw in values.items():
            if not isinstance(raw, Mapping) or "path" not in raw or "sha256" not in raw:
                continue
            _assert_hash(Path(str(raw["path"])), str(raw["sha256"]), label=f"{key}.{name}")
            verified += 1
    analysis_manifest = manifest.get("analysis_manifest")
    if isinstance(analysis_manifest, Mapping) and {
        "path",
        "sha256",
    } <= set(analysis_manifest):
        _assert_hash(
            Path(str(analysis_manifest["path"])),
            str(analysis_manifest["sha256"]),
            label="analysis_manifest",
        )
        verified += 1
    return verified


def _validate_common_identity(
    value: Mapping[str, Any], *, prompt_mode: str, model_label: str
) -> None:
    if value.get("prompt_mode") not in {None, prompt_mode}:
        raise ValueError("Completion manifest prompt mode changed")
    if value.get("model_label") not in {None, model_label}:
        raise ValueError("Completion manifest model label changed")
    if value.get("seed_aliasing") is True:
        raise ValueError("Completion manifest reports seed aliasing")
    if value.get("intervention_outcomes_used_to_replace_rows") is True:
        raise ValueError("Intervention outcomes were used for replacement")
    if value.get("confirmation_used_for_selection") is True:
        raise ValueError("Confirmation outcomes were used for selection")


def audit_targeted_retrieval_selection(
    model_root: Path, *, prompt_mode: str, model_label: str
) -> dict[str, Any]:
    """Recompute discovery K selection and verify confirmation never reselected it."""

    discovery_root = model_root / "causal/targeted_retrieval/discovery_formal"
    selection_path = discovery_root / "analysis/selection.json"
    selection = _read_json(selection_path)
    required = {
        "schema_version": "realistic_niah_v6_targeted_retrieval_selection_v1",
        "status": "DISCOVERY_FROZEN_CHOICE",
        "model_label": model_label,
        "prompt_mode": prompt_mode,
        "selection_split": "discovery",
        "selected_by_v6_discovery_dose_rule": True,
        "dose_argmax_used_for_downstream_bank": True,
        "report_reference_k_is_audit_reference_not_forced_choice": True,
    }
    for key, expected in required.items():
        if selection.get(key) != expected:
            raise ValueError(
                f"Targeted selection {key} changed at {selection_path}: "
                f"expected {expected!r}, got {selection.get(key)!r}"
            )
    expected_grid = TARGETED_REPORT_GRIDS[model_label]
    observed_grid = tuple(int(value) for value in selection.get("bank_grid", ()))
    if observed_grid != expected_grid:
        raise ValueError(f"Targeted report-matched dose grid changed: {observed_grid}")
    expected_random = TARGETED_RANDOM_CONDITION[model_label]
    observed_random = {
        int(key): str(value)
        for key, value in selection.get("random_condition_by_k", {}).items()
    }
    if observed_random != expected_random:
        raise ValueError("Targeted random-control family map changed")
    if int(selection.get("report_reference_k", -1)) != int(
        TARGETED_REPORT_REFERENCE_K[model_label]
    ):
        raise ValueError("Targeted report-reference K changed")

    report_contract_path = Path(str(selection.get("report_contract", "")))
    _assert_hash(
        report_contract_path,
        str(selection.get("report_contract_sha256", "")),
        label="targeted report contract",
    )
    report_contract = _read_json(report_contract_path)
    if report_contract.get("status") != "FROZEN_OUTCOME_BLIND_PROTOCOL_CORRECTION":
        raise ValueError("Targeted report contract is not outcome-blind frozen")
    report_model = report_contract.get("models", {}).get(model_label, {})
    if tuple(int(value) for value in report_model.get("bank_grid", ())) != expected_grid:
        raise ValueError("Targeted selection and report-contract grids disagree")
    contract_random = {
        int(key): str(value)
        for key, value in report_model.get("random_condition_by_k", {}).items()
    }
    if contract_random != expected_random:
        raise ValueError("Targeted selection and report-contract controls disagree")

    dose_path = Path(str(selection.get("dose_response", "")))
    _assert_hash(
        dose_path,
        str(selection.get("dose_response_sha256", "")),
        label="targeted discovery dose response",
    )
    with dose_path.open("r", encoding="utf-8", newline="") as handle:
        dose_rows = [dict(row) for row in csv.DictReader(handle)]
    if tuple(int(row["bank_size"]) for row in dose_rows) != expected_grid:
        raise ValueError("Targeted discovery dose-response rows changed")
    for row in dose_rows:
        bank_size = int(row["bank_size"])
        if str(row.get("split")) != "discovery":
            raise ValueError("Targeted K selection read a non-discovery dose")
        if str(row.get("random_condition")) != expected_random[bank_size]:
            raise ValueError("Targeted dose row has the wrong random-control family")
    dose_argmax = min(
        dose_rows,
        key=lambda row: (
            -float(row["seed_equal_selected_minus_random_failure"]),
            int(row["bank_size"]),
        ),
    )
    selected_k = int(selection["selected_k"])
    if selected_k != int(dose_argmax["bank_size"]):
        raise ValueError("Targeted selected K is not the discovery dose argmax")
    if int(selection.get("dose_argmax_k", -1)) != selected_k:
        raise ValueError("Targeted selection and recorded dose argmax disagree")
    if float(selection.get("selected_effect")) != float(
        dose_argmax["seed_equal_selected_minus_random_failure"]
    ):
        raise ValueError("Targeted selected effect changed after discovery")
    selected_random = expected_random[selected_k]
    if str(selection.get("selected_random_condition")) != selected_random:
        raise ValueError("Targeted selected K/control-family pairing changed")

    plan_path = Path(str(selection.get("frozen_plan", "")))
    _assert_hash(
        plan_path,
        str(selection.get("frozen_plan_sha256", "")),
        label="targeted discovery-frozen bank plan",
    )
    with plan_path.open("r", encoding="utf-8", newline="") as handle:
        plan_rows = [
            dict(row)
            for row in csv.DictReader(handle)
            if str(row.get("model_label")) == model_label
        ]
    identities = Counter(
        (str(row.get("condition")), int(row.get("repeat", 0)))
        for row in plan_rows
    )
    expected_identities = Counter(
        {
            ("selected_bank", 0): 1,
            (selected_random, 1): 1,
            (selected_random, 2): 1,
            (selected_random, 3): 1,
        }
    )
    if identities != expected_identities:
        raise ValueError("Targeted discovery-frozen plan arms changed")
    for row in plan_rows:
        if int(row["bank_size"]) != selected_k:
            raise ValueError("Targeted discovery-frozen plan contains another K")
        serialized = str(row["heads"])
        heads = [(int(layer), int(head)) for layer, head in json.loads(serialized)]
        if len(heads) != selected_k or len(set(heads)) != selected_k:
            raise ValueError("Targeted discovery-frozen plan head bank changed")
        if hashlib.sha256(serialized.encode("utf-8")).hexdigest() != str(
            row["bank_sha256"]
        ):
            raise ValueError("Targeted discovery-frozen plan bank hash changed")

    confirmation_path = (
        model_root / "causal/targeted_retrieval/confirmation_formal/analysis.json"
    )
    confirmation = _read_json(confirmation_path)
    if confirmation.get("status") != "CONFIRMATION_EVALUATED_FROZEN_K":
        raise ValueError("Targeted confirmation analysis is incomplete")
    if (
        confirmation.get("model_label") != model_label
        or confirmation.get("prompt_mode") != prompt_mode
        or int(confirmation.get("selected_k", -1)) != selected_k
        or confirmation.get("selected_random_condition") != selected_random
        or confirmation.get("confirmation_used_for_selection") is not False
        or confirmation.get("bank_size_reselected") is not False
        or confirmation.get("confirmation_reselected_k") is not False
    ):
        raise ValueError("Targeted confirmation changed the discovery-frozen choice")
    if str(confirmation.get("selection_sha256")) != _sha256(selection_path):
        raise ValueError("Targeted confirmation used another discovery selection")
    if str(confirmation.get("report_contract_sha256")) != _sha256(
        report_contract_path
    ):
        raise ValueError("Targeted confirmation used another report contract")
    if str(confirmation.get("discovery_dose_response_sha256")) != _sha256(
        dose_path
    ):
        raise ValueError("Targeted confirmation used another discovery dose table")
    result = confirmation.get("result")
    if not isinstance(result, Mapping) or (
        int(result.get("bank_size", -1)) != selected_k
        or str(result.get("random_condition")) != selected_random
        or str(result.get("split")) != "confirmation"
    ):
        raise ValueError("Targeted confirmation result has the wrong frozen arm")
    return {
        "status": "PASS_DISCOVERY_SELECTED_CONFIRMATION_FROZEN_K",
        "selected_k": selected_k,
        "selected_random_condition": selected_random,
        "selection_rule": (
            "maximize discovery seed-equal selected-minus-three-random failure; "
            "exact tie smaller K"
        ),
        "report_reference_k": TARGETED_REPORT_REFERENCE_K[model_label],
        "report_reference_k_forced": False,
        "effective_bank_grid": list(expected_grid),
        "selection_sha256": _sha256(selection_path),
        "dose_response_sha256": _sha256(dose_path),
        "frozen_plan_sha256": _sha256(plan_path),
        "report_contract_sha256": _sha256(report_contract_path),
        "confirmation_analysis_sha256": _sha256(confirmation_path),
        "confirmation_reselected_k": False,
    }


def audit_specialized_bank_plan_adapter(
    model_root: Path,
    *,
    prompt_mode: str,
    model_label: str,
    targeted_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the selected-treatment-preserving downstream bank geometry."""

    discovery_complete_path = (
        model_root
        / "causal/specialized/discovery_formal/specialized_discovery_complete.json"
    )
    confirmation_complete_path = (
        model_root
        / "causal/specialized/confirmation_formal/specialized_confirmation_complete.json"
    )
    discovery_complete = _read_json(discovery_complete_path)
    confirmation_complete = _read_json(confirmation_complete_path)
    discovery_adapter_ref = discovery_complete.get(
        "specialized_bank_plan_adapter"
    )
    confirmation_adapter_ref = confirmation_complete.get(
        "specialized_bank_plan_adapter"
    )
    if not isinstance(discovery_adapter_ref, Mapping) or not isinstance(
        confirmation_adapter_ref, Mapping
    ):
        raise ValueError("Specialized completion lacks its bank-plan adapter")
    adapter_path = Path(str(discovery_adapter_ref.get("path", "")))
    adapter_sha256 = str(discovery_adapter_ref.get("sha256", ""))
    _assert_hash(adapter_path, adapter_sha256, label="specialized bank-plan adapter")
    if (
        str(confirmation_adapter_ref.get("path", ""))
        != str(discovery_adapter_ref.get("path", ""))
        or str(confirmation_adapter_ref.get("sha256", "")) != adapter_sha256
    ):
        raise ValueError("Confirmation did not reuse the frozen specialized plan")
    adapter = _read_json(adapter_path)
    selection_path = (
        model_root
        / "causal/targeted_retrieval/discovery_formal/analysis/selection.json"
    )
    selection = _read_json(selection_path)
    expected = {
        "schema_version": "realistic_niah_v6_specialized_bank_plan_adapter_v1",
        "model_label": model_label,
        "prompt_mode": prompt_mode,
        "selected_k": int(targeted_selection["selected_k"]),
        "selected_random_condition": str(
            targeted_selection["selected_random_condition"]
        ),
        "selection_registry_sha256": _sha256(selection_path),
        "source_plan_sha256": str(selection["frozen_plan_sha256"]),
        "selected_treatment_row_unchanged": True,
        "selected_treatment_heads_unchanged": True,
        "selected_treatment_bank_sha256_unchanged": True,
        "bank_size_unchanged": True,
        "random_control_family_unchanged": True,
        "behavior_outcomes_used_to_construct_controls": False,
        "specialized_outcomes_used_to_construct_controls": False,
        "confirmation_outcomes_used_to_construct_controls": False,
        "sample_failure": False,
        "seed_replacement_triggered": False,
        "v5_source_files_modified": False,
    }
    for key, expected_value in expected.items():
        if adapter.get(key) != expected_value:
            raise ValueError(
                f"Specialized bank-plan audit {key} changed: expected "
                f"{expected_value!r}, got {adapter.get(key)!r}"
            )
    source_plan = Path(str(adapter.get("source_plan", "")))
    output_plan = Path(str(adapter.get("output_plan", "")))
    _assert_hash(
        source_plan,
        str(adapter["source_plan_sha256"]),
        label="specialized source bank plan",
    )
    _assert_hash(
        output_plan,
        str(adapter["output_plan_sha256"]),
        label="specialized effective bank plan",
    )
    selected_k = int(adapter["selected_k"])
    random_condition = str(adapter["selected_random_condition"])

    def bank_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [
                dict(row)
                for row in csv.DictReader(handle)
                if str(row.get("model_label")) == model_label
            ]
        result: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            identity = (str(row["condition"]), int(row["repeat"]))
            heads = [
                (int(layer), int(head))
                for layer, head in json.loads(str(row["heads"]))
            ]
            if len(heads) != selected_k or len(set(heads)) != selected_k:
                raise ValueError("Specialized effective bank cardinality changed")
            result[identity] = {**row, "parsed_heads": heads}
        return result

    source_rows = bank_rows(source_plan)
    output_rows = bank_rows(output_plan)
    selected_identity = ("selected_bank", 0)
    if source_rows[selected_identity] != output_rows[selected_identity]:
        raise ValueError("Specialized effective plan changed selected treatment")
    random_identities = [(random_condition, repeat) for repeat in (1, 2, 3)]
    if set(output_rows) != {selected_identity, *random_identities}:
        raise ValueError("Specialized effective plan arms changed")
    selected_max = max(
        layer for layer, _head in output_rows[selected_identity]["parsed_heads"]
    )
    random_max = max(
        layer
        for identity in random_identities
        for layer, _head in output_rows[identity]["parsed_heads"]
    )
    if (
        int(adapter.get("selected_max_layer", -1)) != selected_max
        or int(adapter.get("capture_start_layer", -1)) != selected_max + 1
        or int(adapter.get("output_random_max_layer", -1)) != random_max
        or random_max >= selected_max + 1
    ):
        raise ValueError("Specialized bank plan is not causally capture-reachable")
    changed = _sha256(source_plan) != _sha256(output_plan)
    if bool(adapter.get("random_controls_changed")) != changed:
        raise ValueError("Specialized bank-plan change flag is incorrect")
    expected_status = (
        "PASS_CAPTURE_REACHABLE_GLOBAL_CONTROL_ADAPTER"
        if changed
        else "PASS_SOURCE_PLAN_UNCHANGED"
    )
    if adapter.get("status") != expected_status:
        raise ValueError("Specialized bank-plan adapter status changed")
    if changed and random_condition != "global_random":
        raise ValueError("Layer-matched controls were unexpectedly rewritten")
    return {
        "status": expected_status,
        "selected_k": selected_k,
        "selected_random_condition": random_condition,
        "selected_treatment_heads_changed": False,
        "random_control_heads_changed": changed,
        "replacement_count": int(adapter.get("replacement_count", 0)),
        "capture_start_layer": selected_max + 1,
        "output_random_max_layer": random_max,
        "adapter_sha256": adapter_sha256,
        "confirmation_reused_discovery_adapter": True,
        "sample_failure": False,
        "seed_replacement_triggered": False,
    }


def audit_resolved_cell_registry(
    model_root: Path,
    *,
    prompt_mode: str,
    model_label: str,
    split: str,
) -> dict[str, Any]:
    if split == "discovery":
        slots = DISCOVERY_SEEDS
    elif split == "confirmation":
        slots = CONFIRMATION_SEEDS
    else:
        raise ValueError(f"Unknown V6 split: {split}")
    root = model_root / "replacement" / split
    manifest_path = root / "manifest.json"
    selected_path = root / "selected_cells.jsonl"
    mapping_path = root / "replacement_mapping.jsonl"
    attempts_path = root / "attempt_ledger.jsonl"
    manifest = _read_json(manifest_path)
    rows = _read_jsonl(selected_path)
    mappings = _read_jsonl(mapping_path)
    attempts = _read_jsonl(attempts_path)

    if manifest.get("status") != "PASS_STRICT_QUOTA_FILLED":
        raise ValueError(f"Strict replacement quota is not PASS: {manifest_path}")
    if manifest.get("prompt_mode") != prompt_mode or manifest.get("model_label") != model_label:
        raise ValueError(f"Resolved registry identity changed: {selected_path}")
    if manifest.get("seed_role") != split:
        raise ValueError(f"Resolved registry split changed: {selected_path}")
    if any(
        manifest.get(field) is not False
        for field in (
            "hidden_states_read",
            "attention_scores_read",
            "head_ranks_read",
            "intervention_outcomes_read",
        )
    ):
        raise ValueError(f"Replacement read a forbidden outcome: {manifest_path}")
    output_hashes = manifest.get("outputs")
    if not isinstance(output_hashes, Mapping):
        raise ValueError(f"Replacement manifest has no output hashes: {manifest_path}")
    for filename, path in (
        ("selected_cells.jsonl", selected_path),
        ("replacement_mapping.jsonl", mapping_path),
        ("attempt_ledger.jsonl", attempts_path),
    ):
        _assert_hash(path, str(output_hashes.get(filename, "")), label=filename)

    expected_keys = {(count, slot) for count in COUNTS for slot in slots}
    observed_keys = [
        (int(row["gold_count"]), int(row["analysis_slot_seed"])) for row in rows
    ]
    if set(observed_keys) != expected_keys or len(observed_keys) != len(set(observed_keys)):
        raise ValueError(f"Resolved fixed-slot panel changed: {selected_path}")
    if len(rows) != len(expected_keys):
        raise ValueError(f"Resolved registry row count changed: {selected_path}")

    selected_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["gold_count"]), int(row["analysis_slot_seed"]))
        selected_by_key[key] = row
        if row.get("prompt_mode") != prompt_mode or row.get("model_label") != model_label:
            raise ValueError(f"Resolved row identity changed: {selected_path}")
        if row.get("split") != split or row.get("intervention_outcomes_read") is not False:
            raise ValueError(f"Resolved row violates split/outcome contract: {selected_path}")
        source_seed = int(row["source_seed"])
        replaced = bool(row["replacement_applied"])
        if replaced != (source_seed != int(row["analysis_slot_seed"])):
            raise ValueError(f"Replacement flag disagrees with source identity: {selected_path}")
    for count in COUNTS:
        source_seeds = [
            int(row["source_seed"]) for row in rows if int(row["gold_count"]) == count
        ]
        if len(source_seeds) != len(set(source_seeds)):
            raise ValueError(f"A resolved count cell reuses a source seed: {selected_path}")

    mapping_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in mappings:
        key = (int(row["gold_count"]), int(row["analysis_slot_seed"]))
        if key in mapping_by_key:
            raise ValueError(f"Duplicate replacement mapping: {mapping_path}")
        mapping_by_key[key] = row
        selected = selected_by_key.get(key)
        if selected is None or not bool(selected["replacement_applied"]):
            raise ValueError(f"Replacement mapping does not name a replaced slot: {mapping_path}")
        if int(row["original_seed"]) != key[1]:
            raise ValueError(f"Replacement original seed changed: {mapping_path}")
        if int(row["replacement_seed"]) != int(selected["source_seed"]):
            raise ValueError(f"Replacement mapping source seed changed: {mapping_path}")
        if row.get("intervention_outcomes_read") is not False:
            raise ValueError(f"Replacement mapping read an intervention outcome: {mapping_path}")
    replaced_keys = {
        key for key, row in selected_by_key.items() if bool(row["replacement_applied"])
    }
    if set(mapping_by_key) != replaced_keys:
        raise ValueError(f"Replacement mappings do not exactly cover replaced slots: {mapping_path}")
    if int(manifest.get("replacement_count", -1)) != len(mappings):
        raise ValueError(f"Replacement count changed: {manifest_path}")

    attempts_by_cell: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        if row.get("intervention_outcomes_read") is not False:
            raise ValueError(f"Replacement attempt read an intervention outcome: {attempts_path}")
        attempts_by_cell[(int(row["gold_count"]), int(row["seed"]))].append(row)
    candidate_kinds = {str(row.get("candidate_kind")) for row in attempts}
    if not candidate_kinds <= {"original", "replacement", "reserve"}:
        raise ValueError(
            f"Replacement attempt ledger has an unknown candidate kind: {attempts_path}"
        )
    # ``replacement`` is the canonical writer label.  ``reserve`` is accepted
    # for compatibility with the policy wording and early V6 ledgers.  These
    # are generated reserve samples that failed runtime or the fresh strict
    # parser; they must never disappear from the final audit.
    failed_reserve_attempts = _ordinary_failed_reserve_attempts(attempts)
    for mapping in mappings:
        count = int(mapping["gold_count"])
        seed = int(mapping["replacement_seed"])
        selected_attempts = [
            row for row in attempts_by_cell[(count, seed)] if bool(row.get("selected"))
        ]
        if len(selected_attempts) != 1 or not bool(selected_attempts[0].get("eligible")):
            raise ValueError(f"Selected replacement has no unique eligible attempt: {mapping_path}")

    failures = [
        {
            "prompt_mode": prompt_mode,
            "model_label": model_label,
            "split": split,
            "gold_count": int(row["gold_count"]),
            "analysis_slot_seed": int(row["analysis_slot_seed"]),
            "original_seed": int(row["original_seed"]),
            "failure_reasons": list(row.get("original_failure_reasons", [])),
            "replacement_seed": int(row["replacement_seed"]),
            "replacement_candidate_rank": int(row["replacement_candidate_rank"]),
        }
        for row in mappings
    ]
    return {
        "status": "PASS_STRICT_FIXED_QUOTA",
        "split": split,
        "fixed_slot_count": len(slots),
        "cell_count": len(rows),
        "replacement_count": len(mappings),
        "failed_reserve_attempt_count": len(failed_reserve_attempts),
        "selected_cells_sha256": _sha256(selected_path),
        "mapping_sha256": _sha256(mapping_path),
        "attempt_ledger_sha256": _sha256(attempts_path),
        "failures": failures,
        "failed_reserve_attempts": failed_reserve_attempts,
    }


def audit_coherent_registry(
    model_root: Path,
    *,
    prompt_mode: str,
    model_label: str,
    split: str,
    panel_kind: str,
) -> dict[str, Any]:
    if split not in {"discovery", "confirmation"}:
        raise ValueError(f"Unknown V6 split: {split}")
    if panel_kind == "broad":
        directory = "discovery_broad_k" if split == "discovery" else "confirmation_broad"
        expected_status = "PASS_TRUE_SOURCE_COHERENT_BROAD_PANEL"
    elif panel_kind == "native_loop":
        directory = f"{split}_native_loop"
        expected_status = "PASS_TRUE_SOURCE_COHERENT_NATIVE_LOOP_PANEL"
    else:
        raise ValueError(f"Unknown coherent panel kind: {panel_kind}")
    root = model_root / "replacement" / directory
    manifest_path = root / "manifest.json"
    selected_path = root / "selected_cells.jsonl"
    mapping_path = root / "coherent_mapping.jsonl"
    attempts_path = root / "attempt_ledger.jsonl"
    manifest = _read_json(manifest_path)
    rows = _read_jsonl(selected_path)
    mappings = _read_jsonl(mapping_path)
    attempts = _read_jsonl(attempts_path)
    if manifest.get("status") != expected_status:
        raise ValueError(f"Coherent panel is not PASS: {manifest_path}")
    if manifest.get("prompt_mode") != prompt_mode or manifest.get("model_label") != model_label:
        raise ValueError(f"Coherent panel identity changed: {manifest_path}")
    if manifest.get("seed_role") != split:
        raise ValueError(f"Coherent panel split changed: {manifest_path}")
    if any(
        manifest.get(field) is not False
        for field in (
            "hidden_states_read",
            "attention_scores_read",
            "head_ranks_read",
            "intervention_outcomes_read",
        )
    ):
        raise ValueError(f"Coherent replacement read a forbidden outcome: {manifest_path}")
    output_hashes = manifest.get("outputs")
    if not isinstance(output_hashes, Mapping):
        raise ValueError(f"Coherent panel has no output hashes: {manifest_path}")
    for filename, path in (
        ("selected_cells.jsonl", selected_path),
        ("coherent_mapping.jsonl", mapping_path),
        ("attempt_ledger.jsonl", attempts_path),
    ):
        _assert_hash(path, str(output_hashes.get(filename, "")), label=filename)
    replacement_policy_path = Path(str(manifest.get("replacement_policy", "")))
    coherent_policy_path, coherent_policy_sha256 = _resolve_coherent_policy_lineage(
        manifest,
        panel_kind=panel_kind,
        manifest_path=manifest_path,
    )
    replacement_stimuli_path = Path(str(manifest.get("replacement_stimuli", "")))
    _assert_hash(
        replacement_policy_path,
        str(manifest.get("replacement_policy_sha256", "")),
        label="coherent replacement policy",
    )
    _assert_hash(
        coherent_policy_path,
        coherent_policy_sha256,
        label="coherent panel policy",
    )
    _assert_hash(
        replacement_stimuli_path,
        str(manifest.get("replacement_stimuli_sha256", "")),
        label="coherent replacement stimuli",
    )
    _assert_hash(
        replacement_stimuli_path.parent / "manifest.json",
        str(manifest.get("pool_manifest_sha256", "")),
        label="coherent replacement pool manifest",
    )

    required_raw = manifest.get("required_counts_by_slot")
    if not isinstance(required_raw, Mapping) or not required_raw:
        raise ValueError(f"Coherent panel has no required counts: {manifest_path}")
    required = {
        int(slot): tuple(map(int, counts)) for slot, counts in required_raw.items()
    }
    by_key = {
        (int(row["analysis_slot_seed"]), int(row["gold_count"])): row for row in rows
    }
    if len(by_key) != len(rows):
        raise ValueError(f"Coherent registry has duplicate fixed cells: {selected_path}")
    source_by_slot: dict[int, int] = {}
    for slot, counts in required.items():
        source_values = {
            int(by_key[(slot, count)]["source_seed"])
            for count in counts
            if (slot, count) in by_key
        }
        if len(source_values) != 1 or any((slot, count) not in by_key for count in counts):
            raise ValueError(f"Coherent slot {slot} mixes or misses source seeds: {selected_path}")
        source_by_slot[slot] = next(iter(source_values))
    if len(set(source_by_slot.values())) != len(source_by_slot):
        raise ValueError(f"Two coherent slots share a true source seed: {selected_path}")

    mapping_by_slot = {int(row["analysis_slot_seed"]): row for row in mappings}
    if len(mapping_by_slot) != len(mappings):
        raise ValueError(f"Duplicate coherent mappings: {mapping_path}")
    affected = {int(value) for value in manifest.get("affected_slots", [])}
    if set(mapping_by_slot) != affected:
        raise ValueError(f"Coherent mappings do not exactly cover affected slots: {mapping_path}")
    for slot, mapping in mapping_by_slot.items():
        if int(mapping["replacement_seed"]) != source_by_slot[slot]:
            raise ValueError(f"Coherent mapping source identity changed: {mapping_path}")
        failed_counts = {
            int(value["gold_count"]) for value in mapping.get("original_failed_cells", [])
        }
        displaced = set(
            map(int, mapping.get("successful_original_cells_replaced_for_seed_coherence", []))
        )
        if failed_counts | displaced != set(required[slot]) or failed_counts & displaced:
            raise ValueError(f"Coherent displaced-cell audit is incomplete: {mapping_path}")
        if mapping.get("intervention_outcomes_read") is not False:
            raise ValueError(f"Coherent mapping read an intervention outcome: {mapping_path}")
    if int(manifest.get("replacement_seed_count", -1)) != len(mappings):
        raise ValueError(f"Coherent replacement count changed: {manifest_path}")
    if any(row.get("intervention_outcomes_read") is not False for row in attempts):
        raise ValueError(f"Coherent attempt ledger read an intervention outcome: {attempts_path}")
    pending_attempts = [row for row in attempts if bool(row.get("pending_generation"))]
    if pending_attempts:
        raise ValueError(f"Final coherent attempt ledger still has pending generation: {attempts_path}")
    selected_attempts = [row for row in attempts if bool(row.get("selected"))]
    selected_by_slot = Counter(int(row["analysis_slot_seed"]) for row in selected_attempts)
    if selected_by_slot != Counter({slot: 1 for slot in mapping_by_slot}):
        raise ValueError(
            f"Coherent attempt ledger does not contain one selected source per mapping: {attempts_path}"
        )
    for row in selected_attempts:
        mapping = mapping_by_slot[int(row["analysis_slot_seed"])]
        if int(row["candidate_seed"]) != int(mapping["replacement_seed"]):
            raise ValueError(
                f"Selected coherent attempt disagrees with its mapping: {attempts_path}"
            )

    candidate_rejections = [
        {
            "prompt_mode": prompt_mode,
            "model_label": model_label,
            "split": split,
            "panel_kind": panel_kind,
            **row,
        }
        for row in attempts
        if not bool(row.get("selected"))
    ]
    failed_reserve_attempts = [
        row
        for row in candidate_rejections
        if any(
            reason != "source_request_already_used_outside_coherent_panel"
            for reason in row.get("failure_reasons", [])
        )
    ]
    non_failure_candidate_rejections = [
        row for row in candidate_rejections if row not in failed_reserve_attempts
    ]

    replacements = [
        {
            "prompt_mode": prompt_mode,
            "model_label": model_label,
            "split": split,
            "panel_kind": panel_kind,
            "analysis_slot_seed": int(row["analysis_slot_seed"]),
            "original_seed": int(row["original_seed"]),
            "replacement_seed": int(row["replacement_seed"]),
            "replacement_candidate_rank": int(row["replacement_candidate_rank"]),
            "required_counts": list(map(int, row["required_counts"])),
            "original_failed_cells": list(row.get("original_failed_cells", [])),
            "successful_original_cells_replaced_for_seed_coherence": list(
                map(
                    int,
                    row.get(
                        "successful_original_cells_replaced_for_seed_coherence", []
                    ),
                )
            ),
        }
        for row in mappings
    ]
    return {
        "status": expected_status,
        "split": split,
        "panel_kind": panel_kind,
        "analysis_slot_count": len(required),
        "required_panel_cell_count": sum(map(len, required.values())),
        "replacement_trajectory_count": len(mappings),
        "selected_cells_sha256": _sha256(selected_path),
        "mapping_sha256": _sha256(mapping_path),
        "attempt_ledger_sha256": _sha256(attempts_path),
        "replacement_policy": str(replacement_policy_path),
        "replacement_policy_sha256": str(
            manifest["replacement_policy_sha256"]
        ),
        "coherent_policy": str(coherent_policy_path),
        "coherent_policy_sha256": coherent_policy_sha256,
        "replacement_stimuli_sha256": str(
            manifest["replacement_stimuli_sha256"]
        ),
        "pool_manifest_sha256": str(manifest["pool_manifest_sha256"]),
        "replacements": replacements,
        "failed_reserve_attempt_count": len(failed_reserve_attempts),
        "failed_reserve_attempts": failed_reserve_attempts,
        "non_failure_candidate_rejection_count": len(non_failure_candidate_rejections),
        "non_failure_candidate_rejections": non_failure_candidate_rejections,
    }


def _audit_phase_completions(
    model_root: Path,
    *,
    prompt_mode: str,
    model_label: str,
    entries: Mapping[str, tuple[str, str | None]],
    phase: str,
) -> dict[str, Any]:
    manifests: dict[str, Any] = {}
    hashes_verified = 0
    for name, (marker_rel, manifest_rel) in entries.items():
        _require_pass(model_root / marker_rel)
        if manifest_rel is None:
            manifests[name] = {"marker": marker_rel, "status": "PASS"}
            continue
        path = model_root / manifest_rel
        value = _read_json(path)
        _validate_common_identity(value, prompt_mode=prompt_mode, model_label=model_label)
        if value.get("status") not in {
            "DISCOVERY_COMPLETE",
            "CONFIRMATION_COMPLETE",
            "CONFIRMATION_FOUNDATION_COMPLETE",
        }:
            raise ValueError(f"Unexpected {phase} completion status: {path}")
        hashes_verified += _verify_named_artifacts(value)
        manifests[name] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "status": value["status"],
        }
    return {
        "status": "PASS",
        "phase": phase,
        "completion_count": len(entries),
        "referenced_hashes_verified": hashes_verified,
        "manifests": manifests,
    }


def _audit_frame_coverage(model_root: Path) -> list[dict[str, Any]]:
    suite = {row.experiment_id: row for row in EXPERIMENTS}
    if set(FRAME_EVIDENCE) != set(suite):
        raise ValueError("Frame-evidence registry no longer matches the 20-frame suite")
    rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        discovery_rel, confirmation_rel = FRAME_EVIDENCE[experiment.experiment_id]
        discovery = [model_root / path for path in discovery_rel]
        confirmation = [model_root / path for path in confirmation_rel]
        missing = [str(path) for path in (*discovery, *confirmation) if not path.is_file()]
        if experiment.confirmation_required and not confirmation:
            raise ValueError(
                f"Confirmation-required frame has no evidence route: {experiment.experiment_id}"
            )
        if missing:
            raise FileNotFoundError(
                f"Missing frame evidence for {experiment.experiment_id}: {missing}"
            )
        rows.append(
            {
                "report_frame": experiment.report_frame,
                "experiment_id": experiment.experiment_id,
                "confirmation_required": experiment.confirmation_required,
                "discovery_evidence": [
                    {"path": str(path.resolve()), "sha256": _sha256(path)}
                    for path in discovery
                ],
                "confirmation_evidence": [
                    {"path": str(path.resolve()), "sha256": _sha256(path)}
                    for path in confirmation
                ],
                "status": "PASS",
            }
        )
    return rows


def audit_model_mode_cell(
    run_root: str | Path, *, prompt_mode: str, model_label: str
) -> dict[str, Any]:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unknown V6 prompt mode: {prompt_mode}")
    if model_label not in MODEL_LABELS:
        raise ValueError(f"Unknown V6 model label: {model_label}")
    run_root = Path(run_root).resolve()
    model_root = run_root / prompt_mode / model_label
    for marker in (
        "discovery-generate.COMPLETE",
        "discovery-foundation.COMPLETE",
        "discovery-supplement.COMPLETE",
        "discovery-foundation-resolved.COMPLETE",
        "replacement/discovery/discovery.COMPLETE",
        "replacement/discovery_broad_k/k_selection_discovery.COMPLETE",
        "suite-audit.COMPLETE",
        "freeze/freeze.COMPLETE",
        "confirmation-generate.COMPLETE",
        "confirmation-supplement.COMPLETE",
        "confirmation-foundation-resolved.COMPLETE",
        "freeze/full-confirmation.COMPLETE",
    ):
        _require_pass(model_root / marker)

    preflight = _read_json(model_root / "preflight.json")
    suite_audit = _read_json(model_root / "suite_audit.json")
    if preflight.get("status") != "PASS" or preflight.get("prompt_mode") not in {
        None,
        prompt_mode,
    }:
        raise ValueError(f"V6 preflight is not PASS: {model_root / 'preflight.json'}")
    if suite_audit.get("status") != "PASS" or suite_audit.get("prompt_mode") != prompt_mode:
        raise ValueError(f"V6 20-frame suite audit is not PASS: {model_root / 'suite_audit.json'}")

    discovery = _audit_phase_completions(
        model_root,
        prompt_mode=prompt_mode,
        model_label=model_label,
        entries=DISCOVERY_COMPLETION_FILES,
        phase="discovery",
    )
    confirmation = _audit_phase_completions(
        model_root,
        prompt_mode=prompt_mode,
        model_label=model_label,
        entries=CONFIRMATION_COMPLETION_FILES,
        phase="confirmation",
    )
    freeze_path = model_root / "freeze/confirmation_freeze.json"
    freeze = validate_confirmation_freeze(
        freeze_path,
        prompt_mode=prompt_mode,
        model_label=model_label,
        verify_artifacts=True,
    )
    cell_replacements = {
        split: audit_resolved_cell_registry(
            model_root,
            prompt_mode=prompt_mode,
            model_label=model_label,
            split=split,
        )
        for split in ("discovery", "confirmation")
    }
    coherent = {
        f"{panel_kind}_{split}": audit_coherent_registry(
            model_root,
            prompt_mode=prompt_mode,
            model_label=model_label,
            split=split,
            panel_kind=panel_kind,
        )
        for panel_kind in ("broad", "native_loop")
        for split in ("discovery", "confirmation")
    }
    targeted_retrieval_selection = audit_targeted_retrieval_selection(
        model_root,
        prompt_mode=prompt_mode,
        model_label=model_label,
    )
    specialized_bank_plan_adapter = audit_specialized_bank_plan_adapter(
        model_root,
        prompt_mode=prompt_mode,
        model_label=model_label,
        targeted_selection=targeted_retrieval_selection,
    )
    frame_coverage = _audit_frame_coverage(model_root)
    infrastructure_recoveries: list[dict[str, Any]] = []
    recovery = model_root / "foundation_marker_recovery_audit.json"
    if recovery.is_file():
        value = _read_json(recovery)
        if not str(value.get("status", "")).startswith("PASS_"):
            raise ValueError(f"Foundation marker recovery is not PASS: {recovery}")
        if (
            value.get("prompt_mode") != prompt_mode
            or value.get("model_label") != model_label
            or value.get("model_outputs_recomputed") is not False
            or value.get("seed_failure_recorded") is not False
        ):
            raise ValueError(f"Foundation marker recovery changed scientific state: {recovery}")
        validated = value.get("validated_files")
        if not isinstance(validated, Mapping) or not validated:
            raise ValueError(f"Foundation marker recovery has no hashed evidence: {recovery}")
        evidence_states = _audit_foundation_recovery_evidence(
            model_root,
            prompt_mode=prompt_mode,
            model_label=model_label,
            recovery_path=recovery,
            validated=validated,
        )
        infrastructure_recoveries.append(
            {
                "path": str(recovery.resolve()),
                "sha256": _sha256(recovery),
                "status": value.get("status"),
                "sample_failure": False,
                "details": {**value, "validated_file_states": evidence_states},
            }
        )
    for recovery in sorted((model_root / "quarantine").glob("*.recovery.json")):
        value = _read_json(recovery)
        if not str(value.get("status", "")).startswith("PASS_"):
            raise ValueError(f"Quarantine recovery is not PASS: {recovery}")
        if value.get("sample_failure") is not False:
            raise ValueError(f"Quarantine recovery is not labeled infrastructure-only: {recovery}")
        if value.get("deletion_performed") is not False:
            raise ValueError(f"Quarantine recovery deleted artifacts: {recovery}")
        infrastructure_recoveries.append(
            {
                "path": str(recovery.resolve()),
                "sha256": _sha256(recovery),
                "status": value.get("status"),
                "sample_failure": False,
                "details": value,
            }
        )
    return {
        "status": "PASS_FULL_MODEL_MODE_CELL",
        "prompt_mode": prompt_mode,
        "model_label": model_label,
        "model_root": str(model_root),
        "discovery": discovery,
        "confirmation": confirmation,
        "confirmation_freeze": {
            "path": str(freeze_path.resolve()),
            "file_sha256": _sha256(freeze_path),
            "content_freeze_sha256": freeze["freeze_sha256"],
            "confirmation_outcomes_read_before_freeze": False,
        },
        "cell_replacements": cell_replacements,
        "coherent_replacements": coherent,
        "targeted_retrieval_selection": targeted_retrieval_selection,
        "specialized_bank_plan_adapter": specialized_bank_plan_adapter,
        "frame_coverage": frame_coverage,
        "infrastructure_recoveries": infrastructure_recoveries,
    }


def audit_native_aligned_representation(run_root: str | Path) -> dict[str, Any]:
    """Verify the shared two-endpoint analysis and all four cell projections."""

    run_root = Path(run_root).resolve()
    analysis_root = run_root / "native_aligned_representation"
    _require_pass(analysis_root / "COMPLETE")
    manifest_path = analysis_root / "analysis_manifest.json"
    manifest = _read_json(manifest_path)
    expected_schema = "realistic_niah_v6_native_aligned_representation_v1"
    if manifest.get("schema_version") != expected_schema:
        raise ValueError("Native-aligned representation schema changed")
    if manifest.get("status") != "PASS_NATIVE_ANALYSIS_PATH_ALIGNED":
        raise ValueError("Native-aligned representation is not PASS")
    if (
        manifest.get("selection_split") != "discovery"
        or manifest.get("evaluation_split") != "confirmation"
        or manifest.get("confirmation_used_for_selection") is not False
    ):
        raise ValueError("Native-aligned discovery/confirmation firewall changed")
    verified = _verify_named_artifacts(manifest, keys=("outputs",))
    contract = manifest.get("alignment_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Native-aligned manifest has no frozen contract")
    _assert_hash(
        Path(str(contract.get("path", ""))),
        str(contract.get("sha256", "")),
        label="Native-aligned analysis contract",
    )
    verified += 1

    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("Native-aligned manifest has no input registry")
    expected_cells = {
        f"{mode}|{model}" for mode in PROMPT_MODES for model in MODEL_LABELS
    }
    if set(map(str, inputs)) != expected_cells:
        raise ValueError("Native-aligned input registry does not contain four cells")
    for cell, entry in inputs.items():
        if not isinstance(entry, Mapping) or int(entry.get("replacement_rows", -1)) != 0:
            raise ValueError(f"Native-aligned input contains replacement rows: {cell}")
        for name in ("capture_index", "capture_adapter"):
            artifact = entry.get(name)
            if not isinstance(artifact, Mapping):
                raise ValueError(f"Missing {cell}/{name} input artifact")
            _assert_hash(
                Path(str(artifact.get("path", ""))),
                str(artifact.get("sha256", "")),
                label=f"Native-aligned {cell}/{name}",
            )
            verified += 1

    audit_path = analysis_root / "alignment_audit.json"
    alignment = _read_json(audit_path)
    if (
        alignment.get("status") != "PASS_NATIVE_ANALYSIS_PATH_ALIGNED"
        or alignment.get("analysis_population")
        != "original_registered_all_sample_panel"
        or alignment.get("replacement_rows_allowed") is not False
        or alignment.get("running_index", {}).get("site_kind") != "item_end"
        or alignment.get("running_index", {}).get(
            "exact_four_cell_common_support"
        )
        is not True
        or alignment.get("final_count", {}).get("site_kind")
        != "answer_query_v3"
        or alignment.get("final_count", {}).get("exact_full_registered_panel")
        is not True
        or int(alignment.get("final_count", {}).get("trajectory_rows_per_cell", -1))
        != 300
    ):
        raise ValueError("Native-aligned endpoint or population contract changed")

    cell_manifests = manifest.get("cell_manifests")
    if not isinstance(cell_manifests, Mapping) or set(map(str, cell_manifests)) != expected_cells:
        raise ValueError("Native-aligned analysis has no exact four-cell manifest set")
    for cell, artifact in cell_manifests.items():
        if not isinstance(artifact, Mapping):
            raise ValueError(f"Invalid Native-aligned cell manifest record: {cell}")
        path = Path(str(artifact.get("path", "")))
        _assert_hash(path, str(artifact.get("sha256", "")), label=f"{cell} cell manifest")
        verified += 1
        value = _read_json(path)
        if (
            value.get("status") != "PASS_NATIVE_ANALYSIS_PATH_ALIGNED"
            or value.get("analysis_population")
            != "original_registered_all_sample_panel"
            or value.get("exact_four_cell_sample_alignment") is not True
            or value.get("confirmation_used_for_selection") is not False
            or int(value.get("replacement_rows", -1)) != 0
        ):
            raise ValueError(f"Native-aligned cell contract changed: {cell}")
        verified += _verify_named_artifacts(value, keys=("artifacts",))

    return {
        "status": "PASS_NATIVE_ANALYSIS_PATH_ALIGNED",
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "alignment_audit": {"path": str(audit_path), "sha256": _sha256(audit_path)},
        "verified_artifact_hashes": verified,
        "analysis_population": "original_registered_all_sample_panel",
        "running_endpoint": "item_end exact four-cell common support",
        "final_endpoint": "answer_query_v3 exact full 300-trajectory panel",
        "confirmation_used_for_selection": False,
        "legacy_generic_confirmation_scan_required": False,
    }


def audit_full_suite(run_root: str | Path) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    suite = suite_document()
    native_aligned_representation = audit_native_aligned_representation(run_root)
    cells = [
        audit_model_mode_cell(run_root, prompt_mode=mode, model_label=model)
        for mode in PROMPT_MODES
        for model in MODEL_LABELS
    ]
    frame_counts = Counter(
        int(frame["report_frame"])
        for cell in cells
        for frame in cell["frame_coverage"]
        if frame["status"] == "PASS"
    )
    expected_frame_counts = {frame: len(PROMPT_MODES) * len(MODEL_LABELS) for frame in range(1, 21)}
    if dict(sorted(frame_counts.items())) != expected_frame_counts:
        raise ValueError(f"20-frame coverage is incomplete: {dict(frame_counts)}")

    ordinary_failures = [
        event
        for cell in cells
        for split in ("discovery", "confirmation")
        for event in cell["cell_replacements"][split]["failures"]
    ]
    ordinary_failed_reserve_attempts = [
        {
            "prompt_mode": cell["prompt_mode"],
            "model_label": cell["model_label"],
            "split": split,
            "panel_kind": "ordinary_cell",
            **attempt,
        }
        for cell in cells
        for split in ("discovery", "confirmation")
        for attempt in cell["cell_replacements"][split]["failed_reserve_attempts"]
    ]
    coherent_failed_reserve_attempts = [
        attempt
        for cell in cells
        for audit in cell["coherent_replacements"].values()
        for attempt in audit["failed_reserve_attempts"]
    ]
    coherent_non_failure_candidate_rejections = [
        attempt
        for cell in cells
        for audit in cell["coherent_replacements"].values()
        for attempt in audit["non_failure_candidate_rejections"]
    ]
    failed_reserve_attempts = (
        ordinary_failed_reserve_attempts + coherent_failed_reserve_attempts
    )
    coherent_replacements = [
        event
        for cell in cells
        for audit in cell["coherent_replacements"].values()
        for event in audit["replacements"]
    ]
    infrastructure_recoveries = [
        event for cell in cells for event in cell["infrastructure_recoveries"]
    ]
    pool_exhaustion_amendments = [
        {
            "prompt_mode": cell["prompt_mode"],
            "model_label": cell["model_label"],
            "coherent_panel": name,
            "replacement_policy": audit["replacement_policy"],
            "replacement_policy_sha256": audit["replacement_policy_sha256"],
            "coherent_policy": audit["coherent_policy"],
            "coherent_policy_sha256": audit["coherent_policy_sha256"],
            "replacement_stimuli_sha256": audit[
                "replacement_stimuli_sha256"
            ],
            "pool_manifest_sha256": audit["pool_manifest_sha256"],
            "intervention_outcomes_read": False,
        }
        for cell in cells
        for name, audit in cell["coherent_replacements"].items()
        if "amendment" in Path(audit["replacement_policy"]).stem
    ]
    payload: dict[str, Any] = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "status": "PASS_FULL_V6_ENUMERATION_SUITE",
        "run_root": str(run_root),
        "suite_registry_sha256": hashlib.sha256(
            json.dumps(suite, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "models": list(MODEL_LABELS),
        "prompt_modes": list(PROMPT_MODES),
        "model_mode_cell_count": len(cells),
        "report_frame_count": 20,
        "report_frame_pass_counts": {
            str(frame): count for frame, count in sorted(frame_counts.items())
        },
        "discovery_confirmation_firewall": "PASS_HASH_LOCKED_BEFORE_CONFIRMATION",
        "panel_membership_identity": "analysis_slot_seed",
        "statistical_identity": "true_source_seed",
        "seed_aliasing": False,
        "native_aligned_representation": native_aligned_representation,
        "replacement_policy": {
            "ordinary_cell_failure_count": len(ordinary_failures),
            "failed_reserve_attempt_count": len(failed_reserve_attempts),
            "coherent_replacement_trajectory_count": len(coherent_replacements),
            "ordinary_failures": ordinary_failures,
            "failed_reserve_attempts": failed_reserve_attempts,
            "ordinary_failed_reserve_attempts": ordinary_failed_reserve_attempts,
            "coherent_failed_reserve_attempts": coherent_failed_reserve_attempts,
            "coherent_non_failure_candidate_rejections": (
                coherent_non_failure_candidate_rejections
            ),
            "coherent_non_failure_candidate_rejection_count": len(
                coherent_non_failure_candidate_rejections
            ),
            "coherent_replacements": coherent_replacements,
            "all_failures_reported": True,
            "all_seed_attempts_accounted_for": True,
            "silent_sample_exclusion": False,
            "negative_experimental_results_trigger_replacement": False,
        },
        "protocol_amendments": {
            "pool_exhaustion_amendment_count": len(
                pool_exhaustion_amendments
            ),
            "pool_exhaustion_amendments": pool_exhaustion_amendments,
            "confirmation_contract_changed": False,
            "selected_k_changed": False,
            "intervention_outcomes_read": False,
        },
        "infrastructure_recoveries": infrastructure_recoveries,
        "cells": cells,
    }
    payload["audit_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload
