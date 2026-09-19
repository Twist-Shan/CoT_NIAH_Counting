from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .spec import CONFIRMATION_SEEDS, DISCOVERY_SEEDS, MODEL_LABELS, PROMPT_MODES


SUITE_SCHEMA_VERSION = "realistic_niah_v6_enumeration_suite_v1"
FREEZE_SCHEMA_VERSION = "realistic_niah_v6_confirmation_freeze_v1"


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    report_frame: int
    section: str
    claim_role: str
    kernel_entrypoint: str
    primary_estimand: str
    controls: tuple[str, ...]
    discovery_selection: bool
    confirmation_required: bool
    full_layer_sweep: bool = False
    full_head_sweep: bool = False
    # These flags describe an actually materialized parallel view.  They are
    # not universal requirements: Native-thinking uses endpoint-specific
    # populations rather than duplicating every frame as correct-only/all-row.
    correct_only_sensitivity: bool = False
    all_sample_analysis: bool = False


EXPERIMENTS = (
    ExperimentSpec(
        "behavior_and_parser_baseline", 1, "baseline", "define_population",
        "scripts/run_realistic_niah_v6.py generate|audit-generations",
        "exact count, strict format, ordered pair precision/recall",
        ("all generations", "strict-format cohort"), False, True,
        correct_only_sensitivity=True,
        all_sample_analysis=True,
    ),
    ExperimentSpec(
        "layerwise_representation", 2, "representation", "localization",
        "scripts/run_realistic_niah_v6.py capture|representation",
        "held-out balanced accuracy and count regression",
        ("discovery-only fit", "nearest-centroid", "linear probe"), True, True,
        full_layer_sweep=True,
        all_sample_analysis=True,
    ),
    ExperimentSpec(
        "paired_causal_estimands", 3, "design", "estimand_contract",
        "scripts/run_realistic_niah_v6_causal.py causal-plan|causal-patch",
        "paired donor-vs-receiver route and attention effects",
        ("self patch", "norm-matched orthogonal patch"), False, False,
    ),
    ExperimentSpec(
        "trace_scope_layer_sweep", 4, "formation", "minimal_state_scope",
        "scripts/run_realistic_niah_v6_kernel.py natural-aligned-progress",
        "paired successor log-odds across endpoint/tail/full-item scopes",
        ("self patch", "endpoint", "suffix4", "full span"), True, True,
        full_layer_sweep=True,
    ),
    ExperimentSpec(
        "targeted_retrieval_bank", 5, "retrieval", "necessity",
        "scripts/run_realistic_niah_v6.py attention + scripts/run_realistic_niah_v6_causal.py causal-heads",
        "selected-minus-layer-matched-random next-city failure",
        ("clean", "selected bank", "3 layer-matched random banks"), True, True,
        full_head_sweep=True,
    ),
    ExperimentSpec(
        "seed_equal_sampling_contract", 6, "design", "independence",
        "src/realistic_niah_v6/suite.py",
        "seed-first equal-weight aggregation",
        ("20 discovery seeds", "10 disjoint confirmation seeds"), False, False,
    ),
    ExperimentSpec(
        "targeted_query_to_carrier", 7, "write", "local_transport",
        "scripts/run_realistic_niah_v6_kernel.py targeted-counter-write",
        "carrier hidden-state RMS damage after query-local bank mask",
        ("clean", "selected bank", "3 layer-matched random banks"), True, True,
        full_layer_sweep=True,
    ),
    ExperimentSpec(
        "carrier_to_commit_restore", 8, "write", "state_execution",
        "scripts/run_realistic_niah_v6_kernel.py targeted-counter-write",
        "commit-distance recovery under clean-carrier cumulative clamp",
        ("selected mask", "selected mask + clean carrier", "random mask"), True, True,
    ),
    ExperimentSpec(
        "progress_state_to_successor", 9, "write", "next_item_control",
        "scripts/run_realistic_niah_v6_count_stream.py trace-patch",
        "route, attention, argmax, and first-city donor transfer",
        ("self patch", "full donor", "progress subspace", "orthogonal"), True, True,
        full_layer_sweep=True,
    ),
    ExperimentSpec(
        "answer_token_source_ablation", 10, "answer", "source_necessity",
        "scripts/run_realistic_niah_v6_kernel.py token-level-ablation",
        "clean-minus-blank exact count and correct-count margin",
        ("trace blank", "prompt-record blank", "length-matched ordinary blank"), True, True,
    ),
    ExperimentSpec(
        "terminal_state_bridge", 11, "answer", "terminal_transport",
        "scripts/run_realistic_niah_v6_kernel.py terminal-token-state-bridge",
        "semantic restoration and matched-state specificity margins",
        ("uninformative", "semantic restore", "matched ordinary restore"), True, True,
    ),
    ExperimentSpec(
        "timing_stratified_ncc", 12, "write", "geometry_direction",
        "scripts/run_realistic_niah_v6_kernel.py stratified-targeted-counter-ncc",
        "correct-centroid margin loss in grammar-timed strata",
        ("clean", "selected bank", "3 matched random banks"), True, True,
        full_layer_sweep=True,
    ),
    ExperimentSpec(
        "direct_count_output_margin", 13, "answer", "decoder_effect",
        "scripts/run_realistic_niah_v6_kernel.py targeted-counter-logit-margin",
        "gold-vs-best-wrong autoregressive sequence margin loss",
        ("clean", "selected bank", "3 matched random banks"), True, True,
    ),
    ExperimentSpec(
        "count_geometry_ncc", 14, "representation", "count_specificity",
        "scripts/run_realistic_niah_v6_kernel.py targeted-counter-ncc",
        "masked-state displacement relative to discovery centroids",
        ("clean", "selected bank", "3 matched random banks"), True, True,
        full_layer_sweep=True,
    ),
    ExperimentSpec(
        "layer_timing_diagnostic", 15, "robustness", "causal_timing",
        "scripts/run_realistic_niah_v6_kernel.py analyze-stratified-targeted-counter-ncc",
        "layer-by-timing sign and reachability profile",
        ("pre-causal-layer sanity window", "post-bank propagation window"), False, False,
        full_layer_sweep=True,
    ),
    ExperimentSpec(
        "visible_progress_positive_control", 16, "control", "assay_calibration",
        "scripts/run_realistic_niah_v6_kernel.py natural-aligned-progress",
        "successor routing under an explicit visible progress scaffold",
        ("same assay", "same seeds", "externally anchored mid-layer"), True, True,
        full_layer_sweep=True,
    ),
    ExperimentSpec(
        "commit_to_query_patch", 17, "integrated", "loop_edge",
        "scripts/run_realistic_niah_v6_count_stream.py p0-native-loop",
        "donor-successor minus receiver-successor bank attention",
        ("self", "full donor", "count subspace", "orthogonal"), True, True,
    ),
    ExperimentSpec(
        "single_seed_walkthrough", 18, "illustration", "strict_sufficiency_sanity",
        "scripts/run_realistic_niah_v6_kernel.py single-seed-walkthrough",
        "expected answer count after isolated state restoration",
        ("fully scrubbed", "full item", "carrier", "ordinary state"), False, False,
    ),
    ExperimentSpec(
        "format_specific_source_scrub_restore", 19, "robustness", "state_sufficiency",
        "scripts/run_realistic_niah_v6_count_stream.py restoration",
        "target-count margin gain after same-position item-state restoration",
        ("clean", "fully uninformative", "semantic restore", "ordinary restore"), True, True,
    ),
    ExperimentSpec(
        "scrub_coverage_and_cross_mode_audit", 20, "robustness", "confound_audit",
        "scripts/run_realistic_niah_v6.py suite-audit",
        "information-source coverage, topology preservation, and mode contrast",
        ("V1 partial scrub", "V2 full-source scrub", "index vs bullet"), False, False,
    ),
)


# One auditable map from the 20 implementation frames back to the final
# Native-thinking report hierarchy.  ``analysis_population`` is deliberately
# explicit because a universal correct-only/all-sample duplication is not the
# Native-thinking analysis path.
NATIVE_ANALYSIS_PATH: dict[str, dict[str, str]] = {
    "behavior_and_parser_baseline": {
        "native_report_section": "1_behavior",
        "analysis_population": "original_all_sample_plus_resolved_strict_baseline",
        "evidence_tier": "primary",
    },
    "layerwise_representation": {
        "native_report_section": "2_representation_measurement",
        "analysis_population": "original_item_end_exact_four_cell_common_support_plus_answer_query_v3_full300",
        "evidence_tier": "primary",
    },
    "paired_causal_estimands": {
        "native_report_section": "appendix_design_and_walkthrough",
        "analysis_population": "design_only",
        "evidence_tier": "design",
    },
    "trace_scope_layer_sweep": {
        "native_report_section": "3_counter_state",
        "analysis_population": "resolved_strict_cellwise_discovery_confirmation",
        "evidence_tier": "supporting",
    },
    "targeted_retrieval_bank": {
        "native_report_section": "4_targeted_retrieval",
        "analysis_population": "resolved_strict_cellwise_discovery_confirmation",
        "evidence_tier": "primary",
    },
    "seed_equal_sampling_contract": {
        "native_report_section": "appendix_design_and_walkthrough",
        "analysis_population": "design_only",
        "evidence_tier": "design",
    },
    "targeted_query_to_carrier": {
        "native_report_section": "5_counter_update",
        "analysis_population": "resolved_strict_cellwise_discovery_confirmation",
        "evidence_tier": "primary",
    },
    "carrier_to_commit_restore": {
        "native_report_section": "5_counter_update",
        "analysis_population": "resolved_strict_cellwise_discovery_confirmation",
        "evidence_tier": "primary",
    },
    "progress_state_to_successor": {
        "native_report_section": "5_counter_update",
        "analysis_population": "resolved_strict_cellwise_discovery_confirmation",
        "evidence_tier": "primary",
    },
    "answer_token_source_ablation": {
        "native_report_section": "6_terminal_readout",
        "analysis_population": "resolved_strict_cellwise_discovery_confirmation",
        "evidence_tier": "primary",
    },
    "terminal_state_bridge": {
        "native_report_section": "6_terminal_readout",
        "analysis_population": "resolved_strict_cellwise_discovery_confirmation",
        "evidence_tier": "primary",
    },
    "timing_stratified_ncc": {
        "native_report_section": "3_counter_state",
        "analysis_population": "resolved_strict_cellwise_discovery_confirmation",
        "evidence_tier": "diagnostic",
    },
    "direct_count_output_margin": {
        "native_report_section": "6_terminal_readout",
        "analysis_population": "resolved_strict_cellwise_discovery_confirmation",
        "evidence_tier": "supporting",
    },
    "count_geometry_ncc": {
        "native_report_section": "3_counter_state",
        "analysis_population": "resolved_strict_cellwise_discovery_confirmation",
        "evidence_tier": "supporting",
    },
    "layer_timing_diagnostic": {
        "native_report_section": "3_counter_state",
        "analysis_population": "discovery_diagnostic_only",
        "evidence_tier": "diagnostic",
    },
    "visible_progress_positive_control": {
        "native_report_section": "3_counter_state",
        "analysis_population": "resolved_strict_cellwise_discovery_confirmation",
        "evidence_tier": "positive_control",
    },
    "commit_to_query_patch": {
        "native_report_section": "7_evidence_synthesis",
        "analysis_population": "resolved_strict_coherent_trajectory_discovery_confirmation",
        "evidence_tier": "synthesis",
    },
    "single_seed_walkthrough": {
        "native_report_section": "appendix_design_and_walkthrough",
        "analysis_population": "resolved_strict_confirmation_illustration",
        "evidence_tier": "illustration",
    },
    "format_specific_source_scrub_restore": {
        "native_report_section": "8_extension_audit",
        "analysis_population": "resolved_strict_cellwise_discovery_confirmation",
        "evidence_tier": "robustness",
    },
    "scrub_coverage_and_cross_mode_audit": {
        "native_report_section": "8_extension_audit",
        "analysis_population": "audit_ledgers_and_source_identity",
        "evidence_tier": "audit",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_experiment_registry(
    experiments: Iterable[ExperimentSpec] = EXPERIMENTS,
) -> dict[str, Any]:
    rows = tuple(experiments)
    ids = [row.experiment_id for row in rows]
    frames = [row.report_frame for row in rows]
    errors = []
    if len(rows) != 20:
        errors.append(f"expected 20 report-aligned experiments, found {len(rows)}")
    if len(ids) != len(set(ids)):
        errors.append("experiment IDs are not unique")
    if sorted(frames) != list(range(1, 21)):
        errors.append("report frames must be exactly 1..20")
    if set(NATIVE_ANALYSIS_PATH) != set(ids):
        errors.append("Native-thinking analysis-path map must cover every experiment")
    representation = next(
        (row for row in rows if row.experiment_id == "layerwise_representation"),
        None,
    )
    if representation is None or not representation.all_sample_analysis or (
        representation.correct_only_sensitivity
    ):
        errors.append(
            "representation must use the original aligned all-sample panel only"
        )
    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "experiment_count": len(rows),
        "experiment_ids": ids,
        "report_frames": frames,
        "modes": list(PROMPT_MODES),
        "models": list(MODEL_LABELS),
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
    }


def suite_document() -> dict[str, Any]:
    audit = validate_experiment_registry()
    if audit["status"] != "PASS":
        raise ValueError(f"Invalid V6 suite registry: {audit['errors']}")
    return {
        **audit,
        "experiments": [
            {**asdict(row), **NATIVE_ANALYSIS_PATH[row.experiment_id]}
            for row in EXPERIMENTS
        ],
        "native_report_path": NATIVE_ANALYSIS_PATH,
        "evidence_policy": {
            "selection": "discovery only",
            "confirmation": "read once after a hash-locked freeze manifest",
            "independent_unit": "seed-level trajectory",
            "aggregation": "within-seed contrasts, then equal seed weighting",
            "multiplicity": "report all registered contrasts; familywise adjustment where already used by the V5 kernel",
            "negative_results": "retained; never trigger replacement confirmation seeds",
            "population_rule": (
                "use the Native-thinking endpoint-specific population; never "
                "manufacture correct-only/all-sample duplicates for every frame"
            ),
            "direct_grammar_contrast": (
                "original source-identical rows only; resolved replacements remain "
                "cellwise replication evidence unless source identity is exact"
            ),
        },
    }


def discovery_ledger_template(*, prompt_mode: str, model_label: str) -> dict[str, Any]:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unknown V6 prompt mode: {prompt_mode}")
    if model_label not in MODEL_LABELS:
        raise ValueError(f"Unknown V6 model: {model_label}")
    registered = [
        row
        for row in EXPERIMENTS
        if row.discovery_selection and row.confirmation_required
    ]
    return {
        "schema_version": "realistic_niah_v6_discovery_ledger_v1",
        "status": "DISCOVERY_OPEN",
        "prompt_mode": prompt_mode,
        "model_label": model_label,
        "confirmation_outcomes_read": False,
        "experiments": {
            row.experiment_id: {
                "status": "PENDING",
                "report_frame": row.report_frame,
                "selection_rule": row.primary_estimand,
                "choice": None,
                "negative_result_retained": None,
                "artifact_paths": [],
            }
            for row in registered
        },
    }


def freeze_confirmation(
    *,
    prompt_mode: str,
    model_label: str,
    discovery_ledger: Mapping[str, Any],
    artifact_paths: Iterable[str | Path],
) -> dict[str, Any]:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unknown V6 prompt mode: {prompt_mode}")
    if model_label not in MODEL_LABELS:
        raise ValueError(f"Unknown V6 model: {model_label}")
    if discovery_ledger.get("schema_version") != "realistic_niah_v6_discovery_ledger_v1":
        raise ValueError("Discovery ledger has the wrong schema")
    if discovery_ledger.get("status") != "DISCOVERY_FROZEN":
        raise ValueError("Discovery ledger status must be DISCOVERY_FROZEN")
    if discovery_ledger.get("prompt_mode") != prompt_mode:
        raise ValueError("Discovery ledger prompt mode mismatch")
    if discovery_ledger.get("model_label") != model_label:
        raise ValueError("Discovery ledger model mismatch")
    if discovery_ledger.get("confirmation_outcomes_read") is not False:
        raise ValueError("Discovery ledger says confirmation outcomes were read")
    expected = {
        row.experiment_id
        for row in EXPERIMENTS
        if row.discovery_selection and row.confirmation_required
    }
    cells = discovery_ledger.get("experiments")
    if not isinstance(cells, Mapping):
        raise ValueError("Discovery ledger must contain an experiments mapping")
    missing = sorted(expected - set(map(str, cells)))
    if missing:
        raise ValueError(f"Discovery ledger is missing frozen cells: {missing}")
    invalid = {
        experiment_id: cells[experiment_id]
        for experiment_id in sorted(expected)
        if not isinstance(cells[experiment_id], Mapping)
        or cells[experiment_id].get("status") not in {"FROZEN", "NEGATIVE_FROZEN"}
    }
    if invalid:
        raise ValueError(f"Discovery cells are not frozen: {sorted(invalid)}")
    incomplete = []
    registered_cell_artifacts: set[Path] = set()
    for experiment_id in sorted(expected):
        cell = cells[experiment_id]
        if cell.get("choice") is None:
            incomplete.append(f"{experiment_id}:choice")
        if not isinstance(cell.get("negative_result_retained"), bool):
            incomplete.append(f"{experiment_id}:negative_result_retained")
        if (
            cell.get("status") == "NEGATIVE_FROZEN"
            and cell.get("negative_result_retained") is not True
        ):
            incomplete.append(f"{experiment_id}:negative_result_retained_true")
        cell_paths = cell.get("artifact_paths")
        if not isinstance(cell_paths, list) or not cell_paths:
            incomplete.append(f"{experiment_id}:artifact_paths")
            continue
        registered_cell_artifacts.update(Path(path).resolve() for path in cell_paths)
    if incomplete:
        raise ValueError(f"Discovery cells are incomplete: {incomplete}")

    resolved = list(dict.fromkeys(Path(path).resolve() for path in artifact_paths))
    if not resolved:
        raise ValueError("Confirmation freeze requires discovery artifacts")
    if set(resolved) != registered_cell_artifacts:
        missing_from_cli = sorted(map(str, registered_cell_artifacts - set(resolved)))
        unregistered = sorted(map(str, set(resolved) - registered_cell_artifacts))
        raise ValueError(
            "Frozen artifact list disagrees with the discovery ledger: "
            f"missing_from_cli={missing_from_cli}, unregistered={unregistered}"
        )
    absent = [str(path) for path in resolved if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Frozen discovery artifacts are absent: {absent}")
    artifact_hashes = {str(path): _sha256(path) for path in resolved}
    payload = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "status": "CONFIRMATION_FROZEN",
        "prompt_mode": prompt_mode,
        "model_label": model_label,
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "frozen_experiments": {
            experiment_id: dict(cells[experiment_id])
            for experiment_id in sorted(expected)
        },
        "artifact_sha256": artifact_hashes,
        "confirmation_outcomes_read": False,
    }
    payload["freeze_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def validate_confirmation_freeze(
    source: str | Path | Mapping[str, Any],
    *,
    prompt_mode: str,
    model_label: str,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    """Validate one immutable V6 discovery-to-confirmation handoff."""

    if isinstance(source, Mapping):
        value = dict(source)
    else:
        value = json.loads(Path(source).read_text(encoding="utf-8"))
    if value.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise ValueError("Confirmation freeze has the wrong schema")
    if value.get("status") != "CONFIRMATION_FROZEN":
        raise ValueError("Confirmation freeze is not active")
    if value.get("prompt_mode") != prompt_mode:
        raise ValueError("Confirmation freeze prompt mode mismatch")
    if value.get("model_label") != model_label:
        raise ValueError("Confirmation freeze model mismatch")
    if tuple(value.get("discovery_seeds", ())) != DISCOVERY_SEEDS:
        raise ValueError("Confirmation freeze discovery seeds changed")
    if tuple(value.get("confirmation_seeds", ())) != CONFIRMATION_SEEDS:
        raise ValueError("Confirmation freeze confirmation seeds changed")
    if value.get("confirmation_outcomes_read") is not False:
        raise ValueError("Confirmation outcomes were read before the freeze")

    claimed = str(value.get("freeze_sha256", ""))
    unhashed = dict(value)
    unhashed.pop("freeze_sha256", None)
    observed = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if claimed != observed:
        raise ValueError("Confirmation freeze content hash is invalid")

    expected = {
        row.experiment_id
        for row in EXPERIMENTS
        if row.discovery_selection and row.confirmation_required
    }
    frozen = value.get("frozen_experiments")
    if not isinstance(frozen, Mapping) or set(map(str, frozen)) != expected:
        raise ValueError("Confirmation freeze experiment registry changed")
    registered_artifacts: set[Path] = set()
    for experiment_id in sorted(expected):
        cell = frozen[experiment_id]
        if not isinstance(cell, Mapping):
            raise ValueError(f"Frozen experiment is invalid: {experiment_id}")
        if cell.get("status") not in {"FROZEN", "NEGATIVE_FROZEN"}:
            raise ValueError(f"Frozen experiment status changed: {experiment_id}")
        if cell.get("choice") is None:
            raise ValueError(f"Frozen experiment has no explicit choice: {experiment_id}")
        if not isinstance(cell.get("negative_result_retained"), bool):
            raise ValueError(
                f"Frozen experiment has no negative-result decision: {experiment_id}"
            )
        if (
            cell.get("status") == "NEGATIVE_FROZEN"
            and cell.get("negative_result_retained") is not True
        ):
            raise ValueError(
                f"Frozen negative result was not retained: {experiment_id}"
            )
        cell_paths = cell.get("artifact_paths")
        if not isinstance(cell_paths, list) or not cell_paths:
            raise ValueError(f"Frozen experiment has no artifacts: {experiment_id}")
        registered_artifacts.update(Path(path).resolve() for path in cell_paths)

    artifacts = value.get("artifact_sha256")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("Confirmation freeze has no hashed discovery artifacts")
    hashed_artifacts = {Path(path).resolve() for path in artifacts}
    if hashed_artifacts != registered_artifacts:
        raise ValueError("Confirmation freeze artifact registry changed")
    if verify_artifacts:
        for raw_path, expected_hash in artifacts.items():
            artifact = Path(raw_path)
            if not artifact.is_file() or _sha256(artifact) != str(expected_hash):
                raise ValueError(f"Frozen discovery artifact changed: {artifact}")
    return value
