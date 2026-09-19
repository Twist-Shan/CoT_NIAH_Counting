from __future__ import annotations

import importlib.util
from pathlib import Path

from realistic_niah_v5.counting_mechanism_transfer import (
    _first_pass_metadata_keys,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_realistic_niah_v5_gemma_prompt_conditioned_noindex.py"
SPEC = importlib.util.spec_from_file_location("gemma_found_noindex", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(*, total: int = 10, extra: str = "") -> dict:
    records = [
        {"city": f"City {index}", "score": 50 + index}
        for index in range(1, 11)
    ]
    lines = [
        f"FOUND: {record['city']} | score {record['score']}" for record in records
    ]
    raw = "\n".join(lines) + f"\n{extra}Total: {total}<turn|>"
    return {
        "seed": 1234,
        "gold_count": 10,
        "gold_records": records,
        "raw_output_text": raw,
        "split": "discovery",
    }


def test_exact_found_grammar_is_prefix_clean_even_if_final_total_is_wrong() -> None:
    audit = MODULE.audit_prompt_conditioned_noindex_row(_row(total=9))
    assert audit["status"] == "PASS"
    assert audit["primary_eligible_prompt_conditioned_noindex"] is True
    assert audit["primary_eligible_prefix_clean"] is True
    assert audit["terminal_total_correct"] is False
    assert audit["terminal_total_correctness_used_for_selection"] is False
    assert len(audit["first_occurrences"]) == 10
    assert audit["t_star_char"] == audit["first_occurrences"][-1]["char_end"]


def test_found_grammar_rejects_prose_or_numbered_marker() -> None:
    prose = MODULE.audit_prompt_conditioned_noindex_row(
        _row(extra="I found all records.\n")
    )
    assert prose["status"] == "FAIL"
    assert "non_found_reasoning_text" in prose["reasons"]

    numbered = _row()
    numbered["raw_output_text"] = numbered["raw_output_text"].replace(
        "FOUND: City 1", "FOUND 1: City 1", 1
    )
    audit = MODULE.audit_prompt_conditioned_noindex_row(numbered)
    assert audit["status"] == "FAIL"
    assert "found_line_count_mismatch" in audit["reasons"]


def test_found_grammar_allows_nonindex_payload_variation() -> None:
    row = _row()
    row["raw_output_text"] = row["raw_output_text"].replace(
        "FOUND: City 1 | score 51",
        "FOUND: 2024 city score audit, City 1 received a score of 51. | score 51",
        1,
    )
    audit = MODULE.audit_prompt_conditioned_noindex_row(row)
    assert audit["status"] == "PASS"
    assert audit["exact_city_pipe_score_payload_count"] == 9


def test_prompt_conditioned_population_has_distinct_metadata_route() -> None:
    row = _row()
    row[MODULE.AUDIT_KEY] = MODULE.audit_prompt_conditioned_noindex_row(row)
    row[MODULE.COHORT_KEY] = {
        "selection_population": MODULE.SELECTION_POPULATION,
        "split": "discovery",
    }
    audit_key, cohort_key = _first_pass_metadata_keys(
        row,
        selection_population=MODULE.SELECTION_POPULATION,
        eligibility_field="primary_eligible_prompt_conditioned_noindex",
    )
    assert audit_key == MODULE.AUDIT_KEY
    assert cohort_key == MODULE.COHORT_KEY


def test_prompt_rewrite_is_explicitly_non_native() -> None:
    original = "prefix\n" + MODULE._ORIGINAL_TASK_TAIL
    rewritten = MODULE.rewrite_task_tail(original)
    assert rewritten.startswith("prefix\n")
    assert "FOUND: <city> | score <score>" in rewritten
    assert MODULE.ASSISTANT_PREFIX == "FOUND: "
    assert rewritten != original
