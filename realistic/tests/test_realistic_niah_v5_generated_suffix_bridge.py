from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

from realistic_niah_v5.generated_suffix_bridge import (
    REGISTERED_GENERATED_SUFFIX_STATE_CONDITIONS,
    REGISTERED_STATE_PATCH_GEOMETRIES,
    _causal_terminal_suffix_positions,
    _replace_fixed_suffix,
    _token_accuracy,
)


ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = (
    ROOT / "scripts" / "analyze_realistic_niah_v5_generated_suffix_state_bridge.py"
)
SPEC = importlib.util.spec_from_file_location("generated_suffix_analyzer", ANALYZER_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


@dataclass(frozen=True)
class _Encoding:
    input_ids: tuple[int, ...]

    @property
    def sequence_length(self) -> int:
        return len(self.input_ids)


def test_fixed_suffix_replay_preserves_alignment() -> None:
    encoding = _Encoding((1, 2, 3, 4, 5))
    replay = _replace_fixed_suffix(encoding, start=1, stop=4, token_ids=(8, 7, 6))
    assert replay.input_ids == (1, 8, 7, 6, 5)
    assert replay.sequence_length == encoding.sequence_length


def test_token_accuracy_can_focus_on_terminal_nonmarkers() -> None:
    assert _token_accuracy((1, 2, 3, 4), (1, 9, 3, 8)) == 0.5
    assert _token_accuracy((1, 2, 3, 4), (1, 9, 3, 8), (0, 2)) == 1.0


def test_causal_terminal_suffix_intersects_item_with_post_query_tokens() -> None:
    assert _causal_terminal_suffix_positions(
        terminal_start=10, terminal_stop=16, replay_start=8
    ) == tuple(range(10, 16))
    assert _causal_terminal_suffix_positions(
        terminal_start=10, terminal_stop=16, replay_start=13
    ) == (13, 14, 15)


def _row(
    *,
    seed: int,
    condition: str,
    repeat: int,
    utility: float,
    suffix_accuracy: float,
    nonmarker_accuracy: float,
    exact_suffix: bool,
) -> dict[str, object]:
    if condition.startswith("selected_generated"):
        receiver_generation_condition = "selected_bank"
    elif condition.startswith("layer_matched_random"):
        receiver_generation_condition = "layer_matched_random"
    else:
        receiver_generation_condition = "clean"
    return {
        "model_label": "Gemma4-E4B",
        "seed": seed,
        "gold_count": 8,
        "request_id": f"request-{seed}",
        "condition": condition,
        "receiver_generation_repeat": repeat,
        "receiver_generation_condition": receiver_generation_condition,
        "receiver_heads": (
            []
            if receiver_generation_condition == "clean"
            else [[layer, 0] for layer in range(6)]
        ),
        "selection_rank_used": False,
        "outcome_blind": True,
        "teacher_forced_terminal_suffix": False,
        "post_terminal_suffix_teacher_forced": True,
        "state_patch_geometry": "terminal_span",
        "state_patch_excludes_answer_query": True,
        "row_plan_sha256": "plan",
        "free_running_token_budget": 6,
        "generated_suffix_token_count": 6,
        "reference_suffix_exact": exact_suffix,
        "reference_suffix_token_accuracy": suffix_accuracy,
        "reference_terminal_nonmarker_token_accuracy": nonmarker_accuracy,
        "early_eos_generated": False,
        "expected_count_utility": utility,
        "correct_count_probability": utility + 2.0,
        "correct_count_margin": utility * 10.0,
        "exact_count": utility > -0.5,
    }


def test_generated_suffix_analysis_closes_synthetic_serial_chain(tmp_path: Path) -> None:
    root = tmp_path / "discovery"
    shards = root / "shards"
    shards.mkdir(parents=True)
    (root / "frozen_row_plan.json").write_text(
        json.dumps({"plan_sha256": "plan"}), encoding="utf-8"
    )
    for seed in range(1234, 1254):
        rows = [
            _row(
                seed=seed,
                condition="clean_generated_self_state",
                repeat=0,
                utility=0.0,
                suffix_accuracy=1.0,
                nonmarker_accuracy=1.0,
                exact_suffix=True,
            ),
            _row(
                seed=seed,
                condition="selected_generated_self_state",
                repeat=0,
                utility=-1.0,
                suffix_accuracy=0.5,
                nonmarker_accuracy=0.4,
                exact_suffix=False,
            ),
            _row(
                seed=seed,
                condition="selected_generated_clean_state_restore",
                repeat=0,
                utility=-0.1,
                suffix_accuracy=0.5,
                nonmarker_accuracy=0.4,
                exact_suffix=False,
            ),
            _row(
                seed=seed,
                condition="clean_generated_selected_state_occlusion",
                repeat=0,
                utility=-0.8,
                suffix_accuracy=1.0,
                nonmarker_accuracy=1.0,
                exact_suffix=True,
            ),
        ]
        for repeat in (1, 2, 3):
            rows.extend(
                [
                    _row(
                        seed=seed,
                        condition="layer_matched_random_generated_self_state",
                        repeat=repeat,
                        utility=-0.2,
                        suffix_accuracy=0.9,
                        nonmarker_accuracy=0.9,
                        exact_suffix=True,
                    ),
                    _row(
                        seed=seed,
                        condition=(
                            "layer_matched_random_generated_clean_state_restore"
                        ),
                        repeat=repeat,
                        utility=-0.1,
                        suffix_accuracy=0.9,
                        nonmarker_accuracy=0.9,
                        exact_suffix=True,
                    ),
                ]
            )
        (shards / f"seed-{seed}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    effects, claims, audit = ANALYZER.analyze(
        root,
        phase="discovery",
        bootstrap_samples=500,
        random_seed=7,
    )
    assert len(effects) == 20
    assert claims["generated_suffix_state_bridge_pass"] is True
    assert len(claims["primary_estimands"]) == 4
    assert audit["teacher_forced_terminal_suffix"] is False
    assert set(REGISTERED_GENERATED_SUFFIX_STATE_CONDITIONS) == {
        "clean_generated_self_state",
        "selected_generated_self_state",
        "layer_matched_random_generated_self_state",
        "selected_generated_clean_state_restore",
        "layer_matched_random_generated_clean_state_restore",
        "clean_generated_selected_state_occlusion",
    }
    assert set(REGISTERED_STATE_PATCH_GEOMETRIES) == {
        "terminal_span",
        "generated_suffix_span",
        "terminal_prefix_span",
    }
    assert claims["model_label"] == "Gemma4-E4B"
    assert claims["state_patch_geometry"] == "terminal_span"
