from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from realistic_niah_v5.integrated_bridge import (
    _post_query_receiver_positions,
    _targeted_anchor_role,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = (
    ROOT / "scripts" / "analyze_realistic_niah_v5_integrated_mediator_restoration.py"
)
SPEC = importlib.util.spec_from_file_location("mediator_restoration_analyzer", ANALYZER_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)

REPORT_PATH = ROOT / "scripts" / "build_v5_native_count_chain_report.py"
REPORT_SPEC = importlib.util.spec_from_file_location("native_count_chain_report", REPORT_PATH)
assert REPORT_SPEC is not None and REPORT_SPEC.loader is not None
REPORT = importlib.util.module_from_spec(REPORT_SPEC)
REPORT_SPEC.loader.exec_module(REPORT)


def _synthetic_trials(seed_count: int, *, role: str) -> pd.DataFrame:
    margins = {
        "natural": {
            "clean": (10.0, 10.0),
            "selected_bank": (2.0, 8.0),
            "layer_matched_random": (7.0, 7.5),
        },
        "matched_control": {
            "clean": (9.5, 9.5),
            "selected_bank": (2.0, 8.0),
            "layer_matched_random": (7.0, 7.5),
        },
        "cut": {
            "clean": (1.0, 1.0),
            "selected_bank": (1.0, 1.1),
            "layer_matched_random": (1.0, 1.02),
        },
    }
    rows = []
    receivers = [
        ("clean", 0),
        ("selected_bank", 0),
        ("layer_matched_random", 1),
        ("layer_matched_random", 2),
        ("layer_matched_random", 3),
    ]
    for seed in range(100, 100 + seed_count):
        for receiver, repeat in receivers:
            for mediator_index, mediator in enumerate(
                ("self_state", "clean_state_restore")
            ):
                for readout in ("natural", "matched_control", "cut"):
                    rows.append(
                        {
                            "request_id": f"request-{seed}",
                            "model_label": "Qwen3-8B",
                            "seed": seed,
                            "gold_count": 6,
                            "mechanism_split": role,
                            "bridge_design": "restoration",
                            "receiver_write_condition": receiver,
                            "receiver_write_repeat": repeat,
                            "mediator_condition": mediator,
                            "mediator_state_source": (
                                receiver if mediator == "self_state" else "clean"
                            ),
                            "readout_condition": readout,
                            "greedy_generation_run": (
                                receiver == "clean"
                                and mediator == "self_state"
                                and readout == "natural"
                            ),
                            "status": "ok",
                            "correct_count_margin": margins[readout][receiver][
                                mediator_index
                            ],
                            "exact_count": receiver == "clean",
                        }
                    )
    return pd.DataFrame(rows)


def test_mediator_restoration_discovery_passes_frozen_gates() -> None:
    effects, claims, audit = ANALYZER.analyze(
        _synthetic_trials(20, role="development"),
        phase="discovery",
        bootstrap_samples=500,
        random_seed=7,
    )
    assert len(effects) == 20
    assert claims["integrated_mediator_restoration_pass"] is True
    assert claims["gates"]["restoration_is_targeted_specific"]["ci_low"] > 0
    assert claims["gates"]["cut_restoration_residual_equivalence"]["ci_high"] < 0.20
    assert audit["seed_count"] == 20
    assert audit["applicable_seed_count"] == 20
    assert audit["selection_rank_used"] is False


def test_mediator_restoration_confirmation_requires_ten_seeds() -> None:
    _effects, claims, audit = ANALYZER.analyze(
        _synthetic_trials(10, role="confirmation"),
        phase="confirmation",
        bootstrap_samples=500,
        random_seed=11,
    )
    assert claims["confirmatory"] is True
    assert claims["integrated_mediator_restoration_pass"] is True
    assert audit["seed_count"] == 10
    assert audit["applicable_seed_count"] == 10


def test_mediator_restoration_rejects_seed_with_only_not_applicable_samples() -> None:
    trials = _synthetic_trials(20, role="development")
    trials.loc[trials["seed"].eq(100), "status"] = "not_applicable"
    try:
        ANALYZER.analyze(
            trials,
            phase="discovery",
            bootstrap_samples=50,
            random_seed=3,
        )
    except ValueError as exc:
        assert "effective seed count" in str(exc)
        assert "100" in str(exc)
    else:
        raise AssertionError("Every frozen seed must have an applicable sample")


def test_mediator_restoration_rejects_selection_rank() -> None:
    trials = _synthetic_trials(20, role="development")
    trials["selection_rank"] = 1
    try:
        ANALYZER.analyze(
            trials,
            phase="discovery",
            bootstrap_samples=50,
            random_seed=1,
        )
    except ValueError as exc:
        assert "selection_rank" in str(exc)
    else:
        raise AssertionError("selection_rank must be rejected")


def test_report_uses_confirmed_mediator_geometry() -> None:
    assert REPORT._mediator_label(
        "Qwen3-8B", {"mediator_geometry": "suffix8"}
    ) == "terminal suffix8 full state · L19–25"
    assert REPORT._mediator_label(
        "Gemma4-E4B", {"mediator_geometry": "full_span"}
    ) == "terminal full-trace-item hidden state · L16–41"


def test_integrated_geometry_marks_noncausal_span_not_applicable() -> None:
    assert _post_query_receiver_positions(5, (5, 6, 7)) == (6, 7)
    for query, positions in ((6, (5, 6, 7)), (5, (4, 5))):
        try:
            _post_query_receiver_positions(query, positions)
        except ValueError as exc:
            assert "not applicable" in str(exc).lower()
        else:
            raise AssertionError("Noncausal integrated geometry must be excluded")


def test_integrated_bridge_records_model_specific_targeted_anchor_role() -> None:
    assert _targeted_anchor_role("Qwen3-8B") == "post_marker"
    assert _targeted_anchor_role("Gemma4-E4B") == "p0_item_end"
