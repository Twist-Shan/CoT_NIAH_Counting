from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_realistic_niah_v5_targeted_counter_write.py"
SPEC = importlib.util.spec_from_file_location("targeted_counter_write_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_teacher_forced_counter_write_gate_uses_restoration_and_position_control() -> None:
    values = {
        "clean": (0.0, 0.0),
        "selected_mask": (2.0, 3.0),
        "random_mask_r1": (1.0, 1.0),
        "random_mask_r2": (1.1, 1.1),
        "random_mask_r3": (0.9, 0.9),
        "selected_mask_clean_carrier_restore": (0.0, 0.5),
        "selected_mask_matched_position_state_control": (2.5, 2.5),
    }
    rows = []
    for seed in range(1234, 1254):
        timing = "rank_after_city" if seed % 2 == 0 else "rank_before_city"
        for condition, (carrier, boundary) in values.items():
            rows.append(
                {
                    "experiment_id": "teacher_forced_targeted_counter_write",
                    "seed": seed,
                    "condition": condition,
                    "selection_rank_used": False,
                    "teacher_forced_trace_tokens": True,
                    "grammar_timing_stratum": timing,
                    "carrier_state_rms_distance_mean_downstream": carrier,
                    "boundary_state_rms_distance_to_clean_final": boundary,
                }
            )
    effects, result = MODULE.analyze(
        pd.DataFrame(rows), phase="discovery", random_seed=3
    )
    assert len(effects) == 20
    assert result["grammar_timing_counts"] == {
        "rank_after_city": 10,
        "rank_before_city": 10,
    }
    assert result["targeted_counter_write_directional_pass"] is True
    assert result["targeted_counter_write_strong_gate_pass"] is True
