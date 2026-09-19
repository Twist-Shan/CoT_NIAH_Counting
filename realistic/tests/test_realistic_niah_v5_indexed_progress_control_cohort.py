from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "assemble_realistic_niah_v5_indexed_progress_control_cohort.py"
SPEC = importlib.util.spec_from_file_location("indexed_progress_control_cohort", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(model: str) -> dict[str, object]:
    gold = [
        {"city": f"City{index}", "score": 50 + index, "slot_index": index}
        for index in range(1, 11)
    ]
    if model == "Qwen3-8B":
        items = [
            f"{index}. City{index} - {50 + index}" for index in range(1, 11)
        ]
        marker_kind = "indexed"
    else:
        items = [
            f"*   Record {index}: (City{index}, {50 + index})"
            for index in range(1, 11)
        ]
        marker_kind = "inline_count"
    raw = "\n".join(items) + "\nTotal: 10"
    starts = []
    ends = []
    cursor = 0
    for item in items:
        starts.append(cursor)
        ends.append(cursor + len(item))
        cursor += len(item) + 1
    return {
        "model_label": model,
        "gold_count": 10,
        "gold_records": gold,
        "raw_output_text": raw,
        "trace_parse": {
            "parser": {
                "item_count": 10,
                "trace_one_to_one": True,
                "marker_kind": marker_kind,
                "item_markers": list(range(1, 11)),
                "item_gold_cities": [f"City{index}" for index in range(1, 11)],
                "item_start_chars": starts,
                "item_end_chars": ends,
            }
        },
    }


def test_exact_qwen_surface_passes() -> None:
    audit = MODULE.audit_row(_row("Qwen3-8B"), model_label="Qwen3-8B")
    assert audit["primary_eligible_indexed_positive_control"] is True
    assert audit["grammar_class"] == "adjacent_rank_before_city"


def test_exact_gemma_surface_passes() -> None:
    audit = MODULE.audit_row(_row("Gemma4-E4B"), model_label="Gemma4-E4B")
    assert audit["primary_eligible_indexed_positive_control"] is True
    assert audit["grammar_class"] == "same_unit_rank_before_city"


def test_surface_drift_fails_without_resampling() -> None:
    row = _row("Qwen3-8B")
    row["raw_output_text"] = str(row["raw_output_text"]).replace(
        "5. City5 - 55", "5) City5 - 55"
    )
    audit = MODULE.audit_row(row, model_label="Qwen3-8B")
    assert audit["primary_eligible_indexed_positive_control"] is False
    assert "surface_template_mismatch" in audit["reasons"]
