from __future__ import annotations

import json
from collections import Counter
from typing import Any, Mapping, Sequence

from realistic_niah_v5.count_stream import (
    build_sparse_trace_patch_sample_plan as _v5_sparse_trace_patch_plan,
    valid_trace_patch_receivers,
)


COUNT_STREAM_PANEL_ADAPTER_SCHEMA_VERSION = (
    "realistic_niah_v6_count_stream_panel_adapter_v1"
)


def build_v6_sparse_trace_patch_sample_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    model_label: str,
    donor_offsets: Sequence[int],
    seeds_per_cell: int,
    sampling_seed: int,
    include_count2_terminal_panel: bool = True,
    candidate_counts: Sequence[int] = tuple(range(1, 11)),
):
    """Preserve equal cell size when a strict V6 cohort has missing rows.

    The V5 design requests ten seeds in every count/offset cell. A structured
    enumeration mode can have fewer than ten parser-eligible rows at one count
    because formatting/count failures are retained rather than replaced. In
    that case all cells use the same largest feasible size, keeping cell
    weighting balanced and making the shortfall explicit in every plan row.
    No intervention outcome enters this calculation.
    """

    requested = int(seeds_per_cell)
    if requested < 1:
        raise ValueError("seeds_per_cell must be positive")
    required_counts = {
        int(count)
        for count in candidate_counts
        if any(
            valid_trace_patch_receivers(int(count), int(offset))
            for offset in donor_offsets
        )
    }
    if include_count2_terminal_panel:
        required_counts.add(2)
    coverage = Counter(
        int(row["gold_count"])
        for row in rows
        if str(row.get("model_label", model_label)) == str(model_label)
        and int(row["gold_count"]) in required_counts
    )
    missing = sorted(required_counts - set(coverage))
    if missing:
        raise ValueError(f"V6 trace-patch panel has no rows for counts {missing}")
    effective = min(requested, min(coverage[count] for count in required_counts))
    minimum_allowed = max(2, (requested + 1) // 2)
    if effective < minimum_allowed:
        raise ValueError(
            "V6 strict trace-patch coverage is below the registered half-panel "
            f"floor: requested={requested}, effective={effective}, "
            f"coverage={dict(sorted(coverage.items()))}"
        )
    plan = _v5_sparse_trace_patch_plan(
        rows,
        model_label=model_label,
        donor_offsets=donor_offsets,
        seeds_per_cell=effective,
        sampling_seed=sampling_seed,
        include_count2_terminal_panel=include_count2_terminal_panel,
        candidate_counts=candidate_counts,
    )
    plan["v6_requested_seeds_per_cell"] = requested
    plan["v6_effective_seeds_per_cell"] = effective
    plan["v6_parser_eligible_count_coverage_json"] = json.dumps(
        dict(sorted(coverage.items())), sort_keys=True
    )
    plan["v6_cell_shortfall_adapted"] = bool(effective < requested)
    plan["v6_cell_shortfall_policy"] = (
        "uniform_largest_feasible_strict_cohort_size_no_intervention_outcomes"
    )
    return plan


def install_v6_count_stream_panel_adapter() -> dict[str, Any]:
    import realistic_niah_v5.count_stream as legacy

    legacy.build_sparse_trace_patch_sample_plan = (
        build_v6_sparse_trace_patch_sample_plan
    )
    return {
        "schema_version": COUNT_STREAM_PANEL_ADAPTER_SCHEMA_VERSION,
        "status": "INSTALLED",
        "policy": (
            "ten seeds/cell when feasible; otherwise use the largest common "
            "strict-parser-eligible size across all registered cells, with a "
            "minimum half-panel floor"
        ),
        "selection_inputs": ["model_label", "gold_count", "parser eligibility"],
        "intervention_outcomes_read": False,
        "v5_source_files_modified": False,
    }
