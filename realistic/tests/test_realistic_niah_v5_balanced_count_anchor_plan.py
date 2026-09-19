from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_realistic_niah_v5_balanced_count_anchor_plan.py"
SPEC = importlib.util.spec_from_file_location("balanced_count_plan", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _registry(path: Path) -> None:
    rows = []
    for seed in range(1234, 1264):
        for count in range(2, 11):
            if (seed, count) in {(1234, 2), (1241, 3), (1258, 9)}:
                continue
            rows.append(
                {
                    "anchor_equivalence_id": f"a-{seed}-{count}",
                    "anchor_roles": ["adjacent_rank_after_city"],
                    "from_occurrence": count - 1,
                    "gold_count": count,
                    "request_id": f"r-{seed}-{count}",
                    "seed": seed,
                    "target_grammar_class": "test",
                    "target_retrieval_surface_variant": "test",
                    "to_occurrence": count,
                }
            )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_balanced_count_plan_has_exact_20d10c_quotas(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _registry(source)
    rows, manifest = MODULE.build_plan(
        source, counts=(2, 3, 4, 5, 6), panel_id="count2_6"
    )
    assert len(rows) == 30
    discovery = [row for row in rows if row["selection_phase"] == "discovery"]
    confirmation = [row for row in rows if row["selection_phase"] == "confirmation"]
    assert Counter(row["gold_count"] for row in discovery) == {
        2: 4,
        3: 4,
        4: 4,
        5: 4,
        6: 4,
    }
    assert Counter(row["gold_count"] for row in confirmation) == {
        2: 2,
        3: 2,
        4: 2,
        5: 2,
        6: 2,
    }
    assert len({row["seed"] for row in rows}) == 30
    assert manifest["outcome_blind"] is True
    assert manifest["selection_rank_used"] is False


def test_balanced_count_plan_rejects_outcome_fields(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "seed": 1234,
                "gold_count": 2,
                "request_id": "r",
                "expected_count_utility": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        MODULE.build_plan(source, counts=(2, 3), panel_id="bad")
    except ValueError as error:
        assert "forbidden fields" in str(error)
    else:
        raise AssertionError("Outcome-bearing registry was accepted")
