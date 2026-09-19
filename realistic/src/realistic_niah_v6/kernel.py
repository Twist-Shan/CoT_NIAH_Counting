from __future__ import annotations

import csv
import hashlib
import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .encoding import (
    StructuredTraceEncoding,
    build_structured_causal_encoding,
    build_structured_trace_encoding,
)
from .parsing import (
    PARSER_IMPLEMENTATION,
    PARSER_SCHEMA_VERSION,
    PARSER_SELECTION_RULE,
    SITE_SCHEMA_VERSION,
    parse_trace_record,
)


KERNEL_ADAPTER_SCHEMA_VERSION = "realistic_niah_v6_kernel_adapter_v1"
SELECTED_BANK_CONTRACT_SCHEMA_VERSION = (
    "realistic_niah_v6_selected_bank_contract_v2"
)
SPECIALIZED_BANK_PLAN_ADAPTER_SCHEMA_VERSION = (
    "realistic_niah_v6_specialized_bank_plan_adapter_v1"
)
SPECIALIZED_GEOMETRY_SCHEMA_VERSION = (
    "realistic_niah_v6_specialized_enumeration_geometry_v1"
)
COUNT_STREAM_SCHEMA_VERSION = "realistic_niah_v6_count_stream_v1"
CAPTURE_SCHEMA_VERSION = "realistic_niah_v6_trace_capture_v1"
ATTENTION_SCHEMA_VERSION = "realistic_niah_v6_mechanism_attention_v1"
REPRESENTATION_SCHEMA_VERSION = "realistic_niah_v6_representation_v1"


_ENCODING_CONSUMERS = (
    "realistic_niah_v5",
    "realistic_niah_v5.capture",
    "realistic_niah_v5.causal",
    "realistic_niah_v5.causal_sites",
    "realistic_niah_v5.count_stream",
    "realistic_niah_v5.counting_mechanism_transfer",
    "realistic_niah_v5.native_loop",
    "realistic_niah_v5.pre_city",
    "realistic_niah_v5.pipeline",
    "realistic_niah_v5.token_level_ablation",
)

_SCHEMA_MODULES = (
    "realistic_niah_v5.capture",
    "realistic_niah_v5.causal",
    "realistic_niah_v5.count_stream",
    "realistic_niah_v5.counter_write_site",
    "realistic_niah_v5.counting_mechanism_transfer",
    "realistic_niah_v5.native_loop",
    "realistic_niah_v5.pre_city",
    "realistic_niah_v5.representation",
    "realistic_niah_v5.response_reference",
    "realistic_niah_v5.token_level_ablation",
    "realistic_niah_v5.unified_carrier_transition",
)


SELECTED_BANK_CONTRACT_TARGETS = frozenset(
    {
        "targeted-counter-write",
        "stratified-targeted-counter-ncc",
        "targeted-counter-logit-margin",
        "targeted-counter-ncc",
        "token-level-ablation",
    }
)


MODE_TIMING_STRATA = {
    "enumeration_index": "rank_before_city",
    "enumeration_bullet": "structural_item_end",
}


def install_v6_specialized_geometry(prompt_mode: str) -> dict[str, Any]:
    """Install the mode-native carrier geometry used by report follow-ups.

    Indexed enumeration already compiles to the explicit
    ``structural_explicit_rank_before_city`` grammar accepted by the frozen
    kernels.  Bullet enumeration has an invariant bullet marker rather than a
    numerical rank, so calling it ``rank_after_city`` would be scientifically
    false.  For that mode only, this adapter registers a distinct
    ``structural_item_end`` stratum whose carrier/readout is the retrieved-city
    through item-commit tail.  The targeted query, model hooks, bank masks,
    layers, controls, and numerical estimands are unchanged.
    """

    mode = str(prompt_mode)
    if mode not in MODE_TIMING_STRATA:
        raise ValueError(f"Unsupported V6 specialized prompt mode: {mode}")
    timing = MODE_TIMING_STRATA[mode]
    audit: dict[str, Any] = {
        "schema_version": SPECIALIZED_GEOMETRY_SCHEMA_VERSION,
        "prompt_mode": mode,
        "mode_timing_stratum": timing,
        "process_local_patch": False,
        "v5_source_files_modified": False,
        "patched_symbols": [],
    }
    if mode == "enumeration_index":
        audit.update(
            {
                "status": "NATIVE_EXPLICIT_RANK_BEFORE_CITY",
                "carrier_rule": "retrieved_city_through_item_commit",
                "marker_interpretation": "ordinary_index_is_explicit_progress_marker",
            }
        )
        return audit

    terminal = importlib.import_module("realistic_niah_v5.terminal_token_state")
    counter_write = importlib.import_module("realistic_niah_v5.targeted_counter_write")
    counter_ncc = importlib.import_module("realistic_niah_v5.targeted_counter_ncc")
    stratified = importlib.import_module(
        "realistic_niah_v5.stratified_targeted_counter_ncc"
    )
    logit_margin = importlib.import_module(
        "realistic_niah_v5.targeted_counter_logit_margin"
    )
    walkthrough = importlib.import_module(
        "realistic_niah_v5.single_seed_walkthrough"
    )

    if not hasattr(terminal, "_v6_original_grammar_timed_geometry_positions"):
        terminal._v6_original_grammar_timed_geometry_positions = (  # type: ignore[attr-defined]
            terminal._grammar_timed_geometry_positions
        )
    if not hasattr(counter_ncc, "_v6_original_transition_carrier_positions"):
        counter_ncc._v6_original_transition_carrier_positions = (  # type: ignore[attr-defined]
            counter_ncc.transition_carrier_positions
        )
    if not hasattr(stratified, "_v6_original_grammar_timing"):
        stratified._v6_original_grammar_timing = stratified.grammar_timing  # type: ignore[attr-defined]
    if not hasattr(walkthrough, "_v6_original_occurrence_counter_geometry"):
        walkthrough._v6_original_occurrence_counter_geometry = (  # type: ignore[attr-defined]
            walkthrough.occurrence_counter_geometry
        )

    original_geometry = terminal._v6_original_grammar_timed_geometry_positions  # type: ignore[attr-defined]
    original_carrier = counter_ncc._v6_original_transition_carrier_positions  # type: ignore[attr-defined]
    original_timing = stratified._v6_original_grammar_timing  # type: ignore[attr-defined]
    original_walkthrough_geometry = (  # type: ignore[attr-defined]
        walkthrough._v6_original_occurrence_counter_geometry
    )

    def structural_geometry(
        registry: Any,
        terminal_event: dict[str, Any],
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
        grammar_class = str(terminal_event.get("grammar_class", ""))
        if grammar_class != "structural_invariant_bullet":
            return original_geometry(registry, terminal_event)
        terminal_positions = tuple(registry.positions("terminal_trace_item"))
        terminal_set = set(terminal_positions)
        if not terminal_positions:
            raise ValueError("Bullet carrier has an empty terminal item")
        sites = terminal_event.get("sites", {})
        marker_positions = terminal._site_positions(
            sites.get("rank_evidence_core_span"),
            role="rank_evidence_core_span",
        )
        city_positions = terminal._site_positions(
            sites.get("city_target_span"), role="city_target_span"
        )
        commit_positions = terminal._site_positions(
            sites.get("post_update_commit_state"),
            role="post_update_commit_state",
        )
        update_start = int(city_positions[0])
        update_end = int(commit_positions[-1]) + 1
        if not update_start < update_end:
            raise ValueError("Bullet city-to-commit carrier is reversed")
        geometries = {
            "full_item": terminal_positions,
            "marker_core": tuple(marker_positions),
            "retrieved_city": tuple(city_positions),
            "grammar_terminal_update": tuple(range(update_start, update_end)),
            "boundary_commit": tuple(commit_positions),
        }
        if tuple(geometries) != terminal.REGISTERED_GRAMMAR_SPAN_DECOMPOSITION_GEOMETRIES:
            raise RuntimeError("Bullet grammar-span geometry order changed")
        for geometry, positions in geometries.items():
            if not positions or not set(positions) <= terminal_set:
                raise ValueError(
                    f"Bullet grammar geometry {geometry} escapes the terminal item"
                )
            if max(positions) >= int(registry.query_position):
                raise ValueError(
                    f"Bullet grammar geometry {geometry} reaches the answer query"
                )
        return geometries, {
            "terminal_grammar_class": grammar_class,
            "grammar_timing_stratum": "structural_item_end",
            "grammar_terminal_component": "city_target_span",
            "grammar_terminal_update_start": update_start,
            "grammar_terminal_update_end": update_end,
            "invariant_bullet_not_interpreted_as_numeric_rank": True,
        }

    def structural_carrier(
        registry: Any,
        event: dict[str, Any],
        *,
        occurrence: int,
    ) -> tuple[tuple[int, ...], str, str]:
        grammar_class = str(event.get("grammar_class", ""))
        if grammar_class != "structural_invariant_bullet":
            return original_carrier(registry, event, occurrence=occurrence)
        index = int(occurrence) - 1
        if not 0 <= index < len(registry.trace_items):
            raise ValueError("Bullet carrier occurrence is outside the trace")
        item_start, item_end = (int(value) for value in registry.trace_items[index])
        sites = event.get("sites", {})
        city = terminal._site_positions(
            sites.get("city_target_span"), role="city_target_span"
        )
        commit = terminal._site_positions(
            sites.get("post_update_commit_state"),
            role="post_update_commit_state",
        )
        positions = tuple(range(int(city[0]), int(commit[-1]) + 1))
        if not positions or not set(positions) <= set(range(item_start, item_end)):
            raise ValueError("Bullet city-to-commit carrier escapes its trace item")
        return positions, "city_to_commit_tail", "structural_item_end"

    def structural_timing(event: dict[str, Any]) -> str | None:
        if str(event.get("grammar_class", "")) == "structural_invariant_bullet":
            return "structural_item_end"
        return original_timing(event)

    def structural_walkthrough_geometry(
        registry: Any,
        event: dict[str, Any],
        occurrence: int,
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
        grammar_class = str(event.get("grammar_class", ""))
        if grammar_class != "structural_invariant_bullet":
            return original_walkthrough_geometry(registry, event, occurrence)
        index = int(occurrence) - 1
        if not 0 <= index < len(registry.trace_items):
            raise ValueError("Bullet walkthrough occurrence is outside the trace")
        item_start, item_end = (int(value) for value in registry.trace_items[index])
        full_item = tuple(range(item_start, item_end))
        carrier, component, timing_stratum = structural_carrier(
            registry, event, occurrence=occurrence
        )
        for name, positions in (
            ("full_item", full_item),
            ("counter_carrier", carrier),
        ):
            if not positions or not set(positions) <= set(full_item):
                raise ValueError(
                    f"Bullet walkthrough {name} escapes occurrence {occurrence}"
                )
            if max(positions) >= int(registry.query_position):
                raise ValueError(f"Bullet walkthrough {name} reaches the answer query")
        return {
            "full_item": full_item,
            "counter_carrier": carrier,
        }, {
            "occurrence_grammar_class": grammar_class,
            "grammar_timing_stratum": timing_stratum,
            "counter_carrier_component": component,
            "item_span": [item_start, item_end],
            "invariant_bullet_not_interpreted_as_numeric_rank": True,
        }

    terminal._grammar_timed_geometry_positions = structural_geometry
    counter_write._grammar_timed_geometry_positions = structural_geometry
    counter_ncc.transition_carrier_positions = structural_carrier
    stratified.grammar_timing = structural_timing
    stratified.STRATIFIED_NCC_ENDPOINTS["structural_item_end"] = (
        "city_to_commit",
    )
    logit_margin.grammar_timing = structural_timing
    logit_margin.LOGIT_MARGIN_ENDPOINTS["structural_item_end"] = (
        "final_answer_sequence_margin",
    )
    walkthrough.occurrence_counter_geometry = structural_walkthrough_geometry
    terminal.REGISTERED_GRAMMAR_TIMING_STRATA = tuple(
        dict.fromkeys(
            (*terminal.REGISTERED_GRAMMAR_TIMING_STRATA, "structural_item_end")
        )
    )
    audit.update(
        {
            "status": "STRUCTURAL_ITEM_END_INSTALLED",
            "process_local_patch": True,
            "carrier_rule": "retrieved_city_through_item_commit",
            "marker_interpretation": (
                "invariant bullet is a format shell, never numeric count evidence"
            ),
            "patched_symbols": [
                "realistic_niah_v5.terminal_token_state._grammar_timed_geometry_positions",
                "realistic_niah_v5.targeted_counter_write._grammar_timed_geometry_positions",
                "realistic_niah_v5.targeted_counter_ncc.transition_carrier_positions",
                "realistic_niah_v5.stratified_targeted_counter_ncc.grammar_timing",
                "realistic_niah_v5.stratified_targeted_counter_ncc.STRATIFIED_NCC_ENDPOINTS",
                "realistic_niah_v5.targeted_counter_logit_margin.grammar_timing",
                "realistic_niah_v5.targeted_counter_logit_margin.LOGIT_MARGIN_ENDPOINTS",
                "realistic_niah_v5.single_seed_walkthrough.occurrence_counter_geometry",
            ],
        }
    )
    return audit


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_factorial_banks(
    banks: Sequence[Mapping[str, Any]],
    *,
    selected_size: int,
    random_condition: str,
) -> list[dict[str, Any]]:
    """Validate and order one clean/selected/three-random bank factorial.

    V5's report follow-ups hard-code ``layer_matched_random``.  The cited
    Native-thinking Qwen K=128 source artifact instead registers
    ``global_random``.  This validator keeps that distinction explicit while
    enforcing every invariant shared by the two control families.
    """

    if random_condition not in {"layer_matched_random", "global_random"}:
        raise ValueError(f"Unsupported V6 random-control family: {random_condition}")
    normalized: list[dict[str, Any]] = []
    identities: Counter[tuple[str, int]] = Counter()
    for raw in banks:
        condition = str(raw.get("condition"))
        repeat = int(raw.get("repeat", 0))
        heads = tuple((int(layer), int(head)) for layer, head in raw.get("heads", ()))
        if len(heads) != len(set(heads)):
            raise ValueError(f"V6 bank {condition}:r{repeat} contains duplicate heads")
        identities[(condition, repeat)] += 1
        normalized.append(
            {
                "condition": condition,
                "repeat": repeat,
                "heads": [list(value) for value in heads],
                "bank_sha256": str(raw.get("bank_sha256", "clean")),
            }
        )
    expected = Counter(
        {
            ("clean", 0): 1,
            ("selected_bank", 0): 1,
            (random_condition, 1): 1,
            (random_condition, 2): 1,
            (random_condition, 3): 1,
        }
    )
    if identities != expected:
        raise ValueError(
            "V6 selected-bank factorial changed: "
            f"expected {dict(expected)}, got {dict(identities)}"
        )
    by_identity = {
        (row["condition"], int(row["repeat"])): row for row in normalized
    }
    clean = by_identity[("clean", 0)]
    selected = by_identity[("selected_bank", 0)]
    randoms = [by_identity[(random_condition, repeat)] for repeat in (1, 2, 3)]
    if clean["heads"]:
        raise ValueError("V6 clean bank unexpectedly contains heads")
    if len(selected["heads"]) != int(selected_size):
        raise ValueError("V6 selected-bank size changed")
    if any(len(row["heads"]) != int(selected_size) for row in randoms):
        raise ValueError("V6 random-control bank size changed")
    selected_heads = {tuple(value) for value in selected["heads"]}
    selected_layers = Counter(int(layer) for layer, _head in selected_heads)
    for row in randoms:
        random_heads = {tuple(value) for value in row["heads"]}
        if selected_heads & random_heads:
            raise ValueError("V6 random-control bank overlaps the selected bank")
        if random_condition == "layer_matched_random" and Counter(
            int(layer) for layer, _head in random_heads
        ) != selected_layers:
            raise ValueError("V6 layer-matched random bank changed its layer profile")
    nonclean_hashes = [
        str(row["bank_sha256"]) for row in (selected, *randoms)
    ]
    if len(set(nonclean_hashes)) != len(nonclean_hashes):
        raise ValueError("V6 selected/random bank hashes are not distinct")
    return [clean, selected, *randoms]


def _load_v6_selected_bank_plan(
    path: str | Path,
    *,
    model_label: str,
    selected_size: int,
    random_condition: str,
) -> list[dict[str, Any]]:
    """Load one hash-audited V6 plan without relying on V5 label defaults."""

    source = Path(path).resolve()
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        required = {
            "model_label",
            "condition",
            "repeat",
            "bank_size",
            "bank_sha256",
            "heads",
        }
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"V6 selected-bank plan lacks {missing}")
        if "selection_rank" in fieldnames:
            raise ValueError("V6 selected-bank plan forbids selection_rank")
        rows = [
            dict(row)
            for row in reader
            if str(row.get("model_label")) == str(model_label)
        ]
    plan_banks: list[dict[str, Any]] = []
    for row in rows:
        if int(row["bank_size"]) != int(selected_size):
            raise ValueError("V6 selected-bank plan contains a different K")
        serialized = str(row["heads"])
        heads = [(int(layer), int(head)) for layer, head in json.loads(serialized)]
        if len(heads) != int(selected_size) or len(set(heads)) != len(heads):
            raise ValueError("V6 selected-bank plan head count or uniqueness changed")
        if hashlib.sha256(serialized.encode("utf-8")).hexdigest() != str(
            row["bank_sha256"]
        ):
            raise ValueError("V6 selected-bank plan bank hash changed")
        plan_banks.append(
            {
                "condition": str(row["condition"]),
                "repeat": int(row["repeat"]),
                "heads": [list(value) for value in heads],
                "bank_sha256": str(row["bank_sha256"]),
            }
        )
    ordered = _normalized_factorial_banks(
        [
            {
                "condition": "clean",
                "repeat": 0,
                "heads": [],
                "bank_sha256": "clean",
            },
            *plan_banks,
        ],
        selected_size=int(selected_size),
        random_condition=random_condition,
    )
    clean, selected, *randoms = ordered
    # Match the V5 bridge's lexical plan order so numerical runner row order is
    # unchanged: clean, three random controls, selected bank.
    return [clean, *randoms, selected]


def _validate_specialized_bank_plan_adapter(
    *,
    selection_source: Path,
    frozen_plan_source: Path,
    plan_source: Path,
    selection: Mapping[str, Any],
    model_label: str,
    prompt_mode: str,
    selected_k: int,
    random_condition: str,
    frozen_banks: Sequence[Mapping[str, Any]],
    observed_banks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate an optional outcome-blind control-only geometry adapter."""

    frozen_sha256 = _sha256_file(frozen_plan_source)
    observed_sha256 = _sha256_file(plan_source)
    selected_frozen = next(
        bank for bank in frozen_banks if bank["condition"] == "selected_bank"
    )
    selected_observed = next(
        bank for bank in observed_banks if bank["condition"] == "selected_bank"
    )
    if selected_observed != selected_frozen:
        raise ValueError(
            "Specialized bank plan changed the discovery-frozen treatment heads"
        )
    changed = observed_sha256 != frozen_sha256
    audit_path = plan_source.parent / "specialized_bank_plan_audit.json"
    if not changed and not audit_path.is_file():
        return {
            "status": "NOT_APPLIED_DISCOVERY_FROZEN_PLAN",
            "adapter_manifest": None,
            "source_plan_sha256": frozen_sha256,
            "output_plan_sha256": observed_sha256,
            "selected_treatment_heads_changed": False,
            "random_control_heads_changed": False,
        }
    if not audit_path.is_file():
        raise FileNotFoundError(
            "A non-frozen specialized plan requires specialized_bank_plan_audit.json"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": SPECIALIZED_BANK_PLAN_ADAPTER_SCHEMA_VERSION,
        "model_label": model_label,
        "prompt_mode": prompt_mode,
        "selected_k": selected_k,
        "selected_random_condition": random_condition,
        "selection_registry_sha256": _sha256_file(selection_source),
        "source_plan_sha256": frozen_sha256,
        "output_plan_sha256": observed_sha256,
        "selected_bank_sha256": str(selected_frozen["bank_sha256"]),
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
        if audit.get(key) != expected_value:
            raise ValueError(
                f"Specialized bank adapter {key} mismatch: expected "
                f"{expected_value!r}, got {audit.get(key)!r}"
            )
    if str(selection.get("frozen_plan_sha256")) != frozen_sha256:
        raise ValueError("Specialized adapter source is not the selected frozen plan")
    random_banks = [
        bank for bank in observed_banks if bank["condition"] == random_condition
    ]
    observed_random_max = max(
        int(layer)
        for bank in random_banks
        for layer, _head in bank["heads"]
    )
    selected_max = max(
        int(layer) for layer, _head in selected_observed["heads"]
    )
    if int(audit.get("selected_max_layer", -1)) != selected_max:
        raise ValueError("Specialized adapter selected-layer boundary changed")
    if int(audit.get("capture_start_layer", -1)) != selected_max + 1:
        raise ValueError("Specialized adapter capture start changed")
    if int(audit.get("output_random_max_layer", -1)) != observed_random_max:
        raise ValueError("Specialized adapter random-layer audit changed")
    if observed_random_max >= selected_max + 1:
        raise ValueError("Specialized random controls are not capture-reachable")
    expected_status = (
        "PASS_CAPTURE_REACHABLE_GLOBAL_CONTROL_ADAPTER"
        if changed
        else "PASS_SOURCE_PLAN_UNCHANGED"
    )
    if audit.get("status") != expected_status:
        raise ValueError("Specialized bank adapter status disagrees with plan hash")
    if bool(audit.get("random_controls_changed")) != changed:
        raise ValueError("Specialized random-control change flag is incorrect")
    if changed and random_condition != "global_random":
        raise ValueError("Only global-random controls may receive geometry repair")
    return {
        "status": str(audit["status"]),
        "adapter_manifest": str(audit_path),
        "adapter_manifest_sha256": _sha256_file(audit_path),
        "source_plan": str(frozen_plan_source),
        "source_plan_sha256": frozen_sha256,
        "output_plan": str(plan_source),
        "output_plan_sha256": observed_sha256,
        "selected_treatment_heads_changed": False,
        "random_control_heads_changed": changed,
        "replacement_count": int(audit.get("replacement_count", 0)),
        "capture_start_layer": selected_max + 1,
        "output_random_max_layer": observed_random_max,
    }


def install_v6_selected_bank_contract(
    selection_path: str | Path,
    bank_plan_path: str | Path,
    *,
    config: Any,
    model_label: str,
) -> dict[str, Any]:
    """Install one discovery-frozen K without editing a V5 kernel source.

    The report kernels intentionally retain their original Qwen-128/Gemma-6
    defaults and assume a layer-matched random label. Enumeration modes rerun
    the registered discovery dose grid, so this adapter validates the frozen
    choice and exact four-arm plan before installing process-local size and
    label compatibility. Head identities, source layers, hooks, numerical
    interventions, and every V5 source file remain unchanged.
    """

    selection_source = Path(selection_path).resolve()
    plan_source = Path(bank_plan_path).resolve()
    selection = json.loads(selection_source.read_text(encoding="utf-8"))
    required_selection = {
        "schema_version": "realistic_niah_v6_targeted_retrieval_selection_v1",
        "status": "DISCOVERY_FROZEN_CHOICE",
        "model_label": str(model_label),
        "prompt_mode": str(config.prompt_mode),
        "selection_split": "discovery",
    }
    for key, expected in required_selection.items():
        if selection.get(key) != expected:
            raise ValueError(
                f"Selected-bank registry {key} mismatch: "
                f"expected {expected!r}, got {selection.get(key)!r}"
            )
    selected_k = int(selection["selected_k"])
    if selection.get("selected_by_v6_discovery_dose_rule") is not True:
        raise ValueError("Selected-bank registry did not use the discovery dose rule")
    if selection.get("dose_argmax_used_for_downstream_bank") is not True:
        raise ValueError("Selected-bank registry did not freeze the discovery argmax")
    if int(selection.get("dose_argmax_k", -1)) != selected_k:
        raise ValueError("Selected-bank K and discovery dose argmax disagree")
    random_condition = str(selection.get("selected_random_condition"))
    if random_condition not in {"layer_matched_random", "global_random"}:
        raise ValueError("Selected-bank registry has an invalid random control family")
    base_config_grid = tuple(
        int(value) for value in config.targeted_bank_grid(model_label)
    )
    effective_grid = tuple(int(value) for value in selection.get("bank_grid", ()))
    if not effective_grid or len(effective_grid) != len(set(effective_grid)):
        raise ValueError("Selected-bank registry has an invalid effective dose grid")
    if selected_k not in effective_grid:
        raise ValueError(
            f"Discovery-selected K={selected_k} lies outside effective "
            f"{model_label} grid {effective_grid}"
        )
    if not set(effective_grid) <= set(base_config_grid):
        raise ValueError("Targeted subprotocol grid is not contained in the base config")
    report_contract_path = Path(str(selection.get("report_contract", ""))).resolve()
    if not report_contract_path.is_file():
        raise FileNotFoundError("Selected-bank registry report contract is unavailable")
    report_contract_sha256 = _sha256_file(report_contract_path)
    if report_contract_sha256 != str(selection.get("report_contract_sha256")):
        raise ValueError("Selected-bank report contract hash changed")
    report_contract = json.loads(report_contract_path.read_text(encoding="utf-8"))
    if report_contract.get("status") != "FROZEN_OUTCOME_BLIND_PROTOCOL_CORRECTION":
        raise ValueError("Selected-bank report contract is not outcome-blind frozen")
    report_model = report_contract.get("models", {}).get(model_label, {})
    if tuple(int(value) for value in report_model.get("bank_grid", ())) != effective_grid:
        raise ValueError("Selected-bank effective grid disagrees with its report contract")
    report_conditions = report_model.get("random_condition_by_k", {})
    if str(report_conditions.get(str(selected_k))) != random_condition:
        raise ValueError("Selected-bank random family disagrees with its report contract")
    frozen_plan_source = Path(str(selection.get("frozen_plan", ""))).resolve()
    if not frozen_plan_source.is_file():
        raise FileNotFoundError(
            "Discovery-frozen selected bank plan is unavailable"
        )
    frozen_plan_sha256 = _sha256_file(frozen_plan_source)
    if frozen_plan_sha256 != str(selection.get("frozen_plan_sha256")):
        raise ValueError("Discovery-frozen selected bank plan changed")
    frozen_banks = _load_v6_selected_bank_plan(
        frozen_plan_source,
        model_label=model_label,
        selected_size=selected_k,
        random_condition=random_condition,
    )
    observed_plan_sha256 = _sha256_file(plan_source)
    observed_banks = _load_v6_selected_bank_plan(
        plan_source,
        model_label=model_label,
        selected_size=selected_k,
        random_condition=random_condition,
    )
    specialized_plan_adapter = _validate_specialized_bank_plan_adapter(
        selection_source=selection_source,
        frozen_plan_source=frozen_plan_source,
        plan_source=plan_source,
        selection=selection,
        model_label=model_label,
        prompt_mode=str(config.prompt_mode),
        selected_k=selected_k,
        random_condition=random_condition,
        frozen_banks=frozen_banks,
        observed_banks=observed_banks,
    )

    bridge = importlib.import_module(
        "scripts.run_realistic_niah_v5_generated_suffix_state_bridge"
    )
    if model_label not in bridge.MODEL_CONTRACTS:
        raise ValueError(f"No report-kernel contract for {model_label}")
    if not hasattr(bridge, "_v6_original_model_contracts"):
        bridge._v6_original_model_contracts = {  # type: ignore[attr-defined]
            key: dict(value) for key, value in bridge.MODEL_CONTRACTS.items()
        }
    original = dict(bridge._v6_original_model_contracts[model_label])  # type: ignore[attr-defined]
    patched = dict(original)
    patched["bank_size"] = selected_k
    bridge.MODEL_CONTRACTS[model_label] = patched

    targeted_write = importlib.import_module(
        "realistic_niah_v5.targeted_counter_write"
    )
    targeted_ncc = importlib.import_module("realistic_niah_v5.targeted_counter_ncc")
    stratified_ncc = importlib.import_module(
        "realistic_niah_v5.stratified_targeted_counter_ncc"
    )
    logit_margin = importlib.import_module(
        "realistic_niah_v5.targeted_counter_logit_margin"
    )
    if not hasattr(targeted_write, "_v6_original_model_contracts"):
        targeted_write._v6_original_model_contracts = {  # type: ignore[attr-defined]
            key: dict(value) for key, value in targeted_write._MODEL_CONTRACTS.items()
        }
    write_contract = dict(
        targeted_write._v6_original_model_contracts[model_label]  # type: ignore[attr-defined]
    )
    if int(write_contract["source_layer"]) != int(original["source_layer"]):
        raise ValueError("V5 selected-bank consumers disagree on source layer")
    write_contract["bank_size"] = selected_k
    targeted_write._MODEL_CONTRACTS[model_label] = write_contract

    original_symbols = (
        (bridge, "_load_banks"),
        (targeted_write, "run_targeted_counter_write_trials"),
        (targeted_ncc, "_normalize_banks"),
        (stratified_ncc, "_validate_causal_reach"),
        (logit_margin, "_validate_factorial_banks"),
    )
    for module, name in original_symbols:
        original_name = f"_v6_original_{name.lstrip('_')}"
        if not hasattr(module, original_name):
            setattr(module, original_name, getattr(module, name))

    def v6_load_banks(path: Path, *, model_label: str) -> list[dict[str, Any]]:
        if str(model_label) != str(selection["model_label"]):
            raise ValueError("V6 bank loader received a different model")
        return _load_v6_selected_bank_plan(
            path,
            model_label=model_label,
            selected_size=selected_k,
            random_condition=random_condition,
        )

    def v6_normalize_banks(
        adapter: Any,
        banks: Sequence[Mapping[str, Any]],
        *,
        selected_size: int,
    ) -> list[dict[str, Any]]:
        if int(selected_size) != selected_k:
            raise ValueError("V6 NCC runner received a different selected K")
        normalized = _normalized_factorial_banks(
            banks,
            selected_size=selected_k,
            random_condition=random_condition,
        )
        for bank in normalized:
            targeted_ncc._validated_heads(
                adapter,
                tuple((int(layer), int(head)) for layer, head in bank["heads"]),
            )
        return normalized

    original_counter_write = targeted_write._v6_original_run_targeted_counter_write_trials  # type: ignore[attr-defined]

    def v6_run_targeted_counter_write_trials(
        *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        supplied = kwargs.get("banks")
        if supplied is None:
            raise ValueError("V6 targeted-counter-write requires keyword banks")
        _normalized_factorial_banks(
            supplied,
            selected_size=selected_k,
            random_condition=random_condition,
        )
        label_adapter = random_condition == "global_random"
        if label_adapter:
            kwargs = dict(kwargs)
            kwargs["banks"] = [
                {
                    **dict(bank),
                    "condition": (
                        "layer_matched_random"
                        if str(bank.get("condition")) == "global_random"
                        else str(bank.get("condition"))
                    ),
                }
                for bank in supplied
            ]
        results = original_counter_write(*args, **kwargs)
        for row in results:
            if label_adapter and str(row.get("receiver_bank_condition")) == (
                "layer_matched_random"
            ):
                row["receiver_bank_condition"] = "global_random"
            row.update(
                {
                    "v6_selected_random_condition": random_condition,
                    "v6_random_control_label_adapter_applied": label_adapter,
                    "v6_v5_numerical_intervention_unchanged": True,
                }
            )
        return results

    original_causal_reach = stratified_ncc._v6_original_validate_causal_reach  # type: ignore[attr-defined]

    def v6_validate_causal_reach(
        banks: Sequence[Mapping[str, Any]], *, capture_start_layer: int
    ) -> int:
        if random_condition == "layer_matched_random":
            return int(
                original_causal_reach(
                    banks, capture_start_layer=int(capture_start_layer)
                )
            )
        normalized = _normalized_factorial_banks(
            banks,
            selected_size=selected_k,
            random_condition=random_condition,
        )
        maximum = max(
            int(layer)
            for bank in normalized
            for layer, _head in bank["heads"]
        )
        if int(capture_start_layer) != maximum + 1:
            raise ValueError(
                "Global-control stratified NCC must start exactly one layer "
                "above every ablated selected or random head"
            )
        return maximum

    original_factorial_validation = logit_margin._v6_original_validate_factorial_banks  # type: ignore[attr-defined]

    def v6_validate_factorial_banks(
        banks: Sequence[Mapping[str, Any]],
    ) -> None:
        if random_condition == "layer_matched_random":
            original_factorial_validation(banks)
            return
        _normalized_factorial_banks(
            banks,
            selected_size=selected_k,
            random_condition=random_condition,
        )

    bridge._load_banks = v6_load_banks
    targeted_write.run_targeted_counter_write_trials = (
        v6_run_targeted_counter_write_trials
    )
    targeted_ncc._normalize_banks = v6_normalize_banks
    # These modules imported the symbol directly, so patch their bound globals
    # as well as the defining module.
    stratified_ncc._normalize_banks = v6_normalize_banks
    logit_margin._normalize_banks = v6_normalize_banks
    stratified_ncc._validate_causal_reach = v6_validate_causal_reach
    logit_margin._validate_factorial_banks = v6_validate_factorial_banks

    patched_symbols = [
        f"scripts.run_realistic_niah_v5_generated_suffix_state_bridge.MODEL_CONTRACTS[{model_label!r}]['bank_size']",
        "scripts.run_realistic_niah_v5_generated_suffix_state_bridge._load_banks",
        f"realistic_niah_v5.targeted_counter_write._MODEL_CONTRACTS[{model_label!r}]['bank_size']",
        "realistic_niah_v5.targeted_counter_write.run_targeted_counter_write_trials",
        "realistic_niah_v5.targeted_counter_ncc._normalize_banks",
        "realistic_niah_v5.stratified_targeted_counter_ncc._normalize_banks",
        "realistic_niah_v5.stratified_targeted_counter_ncc._validate_causal_reach",
        "realistic_niah_v5.targeted_counter_logit_margin._normalize_banks",
        "realistic_niah_v5.targeted_counter_logit_margin._validate_factorial_banks",
    ]
    return {
        "schema_version": SELECTED_BANK_CONTRACT_SCHEMA_VERSION,
        "status": "INSTALLED",
        "model_label": model_label,
        "prompt_mode": str(config.prompt_mode),
        "selection_split": "discovery",
        "selection_registry": str(selection_source),
        "selection_registry_sha256": _sha256_file(selection_source),
        "discovery_frozen_bank_plan": str(frozen_plan_source),
        "discovery_frozen_bank_plan_sha256": frozen_plan_sha256,
        "bank_plan": str(plan_source),
        "bank_plan_sha256": observed_plan_sha256,
        "base_config_bank_grid": list(base_config_grid),
        "effective_registered_bank_grid": list(effective_grid),
        "report_matched_bank_grid": list(effective_grid),
        "report_contract": str(report_contract_path),
        "report_contract_sha256": report_contract_sha256,
        "legacy_report_reference_bank_size": int(original["bank_size"]),
        "selected_bank_size": selected_k,
        "selected_random_condition": random_condition,
        "source_layer_preserved": int(original["source_layer"]),
        "v5_source_files_modified": False,
        "process_local_patch": True,
        "patched_symbols": patched_symbols,
        "random_control_kernel_adapter": (
            "v6_process_local_global_random_label_compatibility"
            if random_condition == "global_random"
            else "native_v5_layer_matched_random_factorial"
        ),
        "specialized_bank_plan_adapter": specialized_plan_adapter,
        "head_identities_changed": bool(
            specialized_plan_adapter["random_control_heads_changed"]
        ),
        "selected_head_identities_changed": False,
        "random_control_head_identities_changed": bool(
            specialized_plan_adapter["random_control_heads_changed"]
        ),
        "numerical_intervention_implementation_changed": False,
    }


def _replace_schema_globals(module: Any) -> list[str]:
    changed: list[str] = []
    for name, value in tuple(vars(module).items()):
        if not name.endswith("SCHEMA_VERSION") or not isinstance(value, str):
            continue
        if "realistic_niah_v5" not in value:
            continue
        setattr(module, name, value.replace("realistic_niah_v5", "realistic_niah_v6"))
        changed.append(f"{module.__name__}.{name}")
    return changed


def install_v6_kernel_adapters() -> dict[str, Any]:
    """Install process-local V6 adapters around frozen V5 numerical kernels.

    No V5 source file or artifact is modified.  The adapter replaces only
    prompt/container-sensitive encoding functions, parser provenance, and
    output schema labels in the current Python process.  Numerical model hooks,
    head interventions, patch geometry, bootstrap code, and estimands remain
    the audited kernels used by the Native-thinking report.
    """

    patched: list[str] = []
    for module_name in _ENCODING_CONSUMERS:
        module = importlib.import_module(module_name)
        if hasattr(module, "NativeTraceEncoding"):
            module.NativeTraceEncoding = StructuredTraceEncoding
            patched.append(f"{module_name}.NativeTraceEncoding")
        if hasattr(module, "build_native_trace_encoding"):
            module.build_native_trace_encoding = build_structured_trace_encoding
            patched.append(f"{module_name}.build_native_trace_encoding")
        if hasattr(module, "build_native_causal_encoding"):
            module.build_native_causal_encoding = build_structured_causal_encoding
            patched.append(f"{module_name}.build_native_causal_encoding")
        if hasattr(module, "parse_trace_record"):
            module.parse_trace_record = parse_trace_record
            patched.append(f"{module_name}.parse_trace_record")

    # The late V5 trace-patch/native-loop kernels consume the registry-shaped
    # progress fields (from_occurrence, fixed target-token span, grammar pair),
    # but the public dispatcher still routes progress_transition through its
    # older marker-only compiler.  Structured index/bullet traces must use the
    # same grammar-aware compiler already used by retrieval localization.
    causal = importlib.import_module("realistic_niah_v5.causal")
    if not hasattr(causal, "_v6_original_mechanism_continuations"):
        causal._v6_original_mechanism_continuations = (  # type: ignore[attr-defined]
            causal.mechanism_continuations
        )
    original_mechanism_continuations = (
        causal._v6_original_mechanism_continuations  # type: ignore[attr-defined]
    )

    def structured_mechanism_continuations(
        row: Mapping[str, Any],
        tokenizer: Any,
        *,
        mechanism: str,
        boundary_policy: str = "strict_registered",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if mechanism == "progress_transition":
            if boundary_policy != "strict_registered":
                raise ValueError(
                    "V6 structured progress transitions require the registered "
                    "grammar boundary policy"
                )
            return causal._registered_mechanism_continuations(
                row, tokenizer, mechanism=mechanism
            )
        return original_mechanism_continuations(
            row,
            tokenizer,
            mechanism=mechanism,
            boundary_policy=boundary_policy,
        )

    causal.mechanism_continuations = structured_mechanism_continuations
    patched.append("realistic_niah_v5.causal.mechanism_continuations")
    for module_name in (
        "realistic_niah_v5.count_stream",
        "realistic_niah_v5.native_loop",
    ):
        module = importlib.import_module(module_name)
        module.mechanism_continuations = structured_mechanism_continuations
        patched.append(f"{module_name}.mechanism_continuations")

    capture = importlib.import_module("realistic_niah_v5.capture")
    capture.CAPTURE_SCHEMA_VERSION = CAPTURE_SCHEMA_VERSION
    capture.ATTENTION_SCHEMA_VERSION = ATTENTION_SCHEMA_VERSION
    capture.NO_ALIGNED_TRACE_SITES_REASON = "no_aligned_registered_v6_trace_sites"
    capture.PARSER_IMPLEMENTATION = PARSER_IMPLEMENTATION
    capture.PARSER_SCHEMA_VERSION = PARSER_SCHEMA_VERSION
    capture.PARSER_SELECTION_RULE = PARSER_SELECTION_RULE
    capture.SITE_SCHEMA_VERSION = SITE_SCHEMA_VERSION
    capture.parse_trace_record = parse_trace_record
    patched.extend(
        [
            "realistic_niah_v5.capture.CAPTURE_SCHEMA_VERSION",
            "realistic_niah_v5.capture.ATTENTION_SCHEMA_VERSION",
            "realistic_niah_v5.capture.parse_trace_record",
        ]
    )

    representation = importlib.import_module("realistic_niah_v5.representation")
    representation.REPRESENTATION_SCHEMA_VERSION = REPRESENTATION_SCHEMA_VERSION
    patched.append("realistic_niah_v5.representation.REPRESENTATION_SCHEMA_VERSION")

    count_stream = importlib.import_module("realistic_niah_v5.count_stream")
    count_stream.COUNT_STREAM_SCHEMA_VERSION = COUNT_STREAM_SCHEMA_VERSION
    patched.append("realistic_niah_v5.count_stream.COUNT_STREAM_SCHEMA_VERSION")

    for module_name in _SCHEMA_MODULES:
        module = importlib.import_module(module_name)
        patched.extend(_replace_schema_globals(module))

    return {
        "schema_version": KERNEL_ADAPTER_SCHEMA_VERSION,
        "status": "INSTALLED",
        "source_kernel_family": "realistic_niah_v5",
        "target_protocol_family": "realistic_niah_v6",
        "v5_source_files_modified": False,
        "structured_progress_transition_adapter": {
            "status": "INSTALLED",
            "mechanism": "progress_transition",
            "boundary_policy": "strict_registered",
            "compiler": (
                "realistic_niah_v5.causal._registered_mechanism_continuations"
            ),
            "numerical_patch_implementation_changed": False,
        },
        "patched_process_globals": sorted(set(patched)),
    }
