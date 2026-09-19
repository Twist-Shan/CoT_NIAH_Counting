from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = ROOT / "scripts" / "analyze_realistic_niah_v5_integrated_serial_bridge.py"
SPEC = importlib.util.spec_from_file_location("integrated_bridge_analyzer", ANALYZER_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def _synthetic_trials(seed_count: int, *, role: str) -> pd.DataFrame:
    rows = []
    values = {
        "natural": {"clean": 10.0, "selected_bank": 4.0, "random": 8.0},
        "matched_control": {
            "clean": 10.0,
            "selected_bank": 4.5,
            "random": 8.5,
        },
        "cut": {"clean": 2.0, "selected_bank": 2.0, "random": 2.1},
    }
    for seed in range(100, 100 + seed_count):
        for readout, cell in values.items():
            rows.append(
                {
                    "request_id": f"request-{seed}",
                    "model_label": "Qwen3-8B",
                    "seed": seed,
                    "gold_count": 6,
                    "mechanism_split": role,
                    "write_condition": "clean",
                    "write_repeat": 0,
                    "readout_condition": readout,
                    "status": "ok",
                    "correct_count_margin": cell["clean"],
                    "exact_count": True,
                }
            )
            rows.append(
                {
                    "request_id": f"request-{seed}",
                    "model_label": "Qwen3-8B",
                    "seed": seed,
                    "gold_count": 6,
                    "mechanism_split": role,
                    "write_condition": "selected_bank",
                    "write_repeat": 0,
                    "readout_condition": readout,
                    "status": "ok",
                    "correct_count_margin": cell["selected_bank"],
                    "exact_count": False,
                }
            )
            for repeat in (1, 2, 3):
                rows.append(
                    {
                        "request_id": f"request-{seed}",
                        "model_label": "Qwen3-8B",
                        "seed": seed,
                        "gold_count": 6,
                        "mechanism_split": role,
                        "write_condition": "layer_matched_random",
                        "write_repeat": repeat,
                        "readout_condition": readout,
                        "status": "ok",
                        "correct_count_margin": cell["random"],
                        "exact_count": True,
                    }
                )
    return pd.DataFrame(rows)


def test_integrated_bridge_discovery_gates_and_contract() -> None:
    effects, claims, audit = ANALYZER.analyze(
        _synthetic_trials(20, role="development"),
        phase="discovery",
        bootstrap_samples=500,
        random_seed=7,
    )
    assert len(effects) == 20
    assert claims["integrated_serial_bridge_pass"] is True
    assert claims["gates"]["cut_residual_equivalence"]["ci_high"] < 0.20
    assert audit["seed_count"] == 20
    assert audit["selection_rank_used"] is False


def test_integrated_bridge_confirmation_requires_ten_seeds() -> None:
    _effects, claims, audit = ANALYZER.analyze(
        _synthetic_trials(10, role="confirmation"),
        phase="confirmation",
        bootstrap_samples=500,
        random_seed=9,
    )
    assert claims["confirmatory"] is True
    assert claims["integrated_serial_bridge_pass"] is True
    assert audit["seed_count"] == 10


def test_integrated_bridge_rejects_selection_rank() -> None:
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
