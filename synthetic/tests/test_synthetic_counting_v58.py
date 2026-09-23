from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import pytest

from synthetic_counting_v20.config import config_from_dict
from synthetic_counting_v20.data import V20Example, V20Vocab, character_token, render_v20
from synthetic_counting_v20.model import build_model
from synthetic_counting_v57.config import preset_config as preset_v57
from synthetic_counting_v58.config import preset_config as preset_v58
from synthetic_counting_v58.behavior_gate import GATE_THRESHOLDS
from synthetic_counting_v58.confirmation import (
    select_disjoint_balanced,
    summarize_confirmation,
)
from synthetic_counting_v58.preflight import run_preflight


def _example() -> V20Example:
    marker = character_token("a")
    return V20Example(
        example_kind="counting_task",
        seq_tokens=[marker, marker, marker],
        corpus_region="train",
        corpus_start=0,
        corpus_end=3,
        prompt_sha256="count-3",
        set_id="set_000",
        needle_characters=("a", "b", "c"),
        rendered_set_order=("a", "b", "c"),
        needle_positions=(0, 1, 2),
        needle_markers=(marker, marker, marker),
        count=3,
        per_character_counts=(3, 0, 0),
    )


def test_v58_changes_only_v57_parallel_width_factor() -> None:
    baseline = preset_v57("main", device="cpu")
    candidate = preset_v58("main", device="cpu")
    baseline_values = asdict(baseline)
    changed = {
        key
        for key, value in asdict(candidate).items()
        if baseline_values.get(key) != value
    }
    assert changed == {"version", "n_head", "n_embd", "n_inner"}
    assert candidate.n_layer == baseline.n_layer == 4
    assert (baseline.n_head, baseline.n_embd, baseline.n_inner) == (6, 384, 1536)
    assert (candidate.n_head, candidate.n_embd, candidate.n_inner) == (8, 512, 2048)
    assert baseline.n_embd // baseline.n_head == candidate.n_embd // candidate.n_head == 64
    assert candidate.batch_size == baseline.batch_size == 128
    assert candidate.task_output_structure_weight == 16.0
    assert config_from_dict(candidate.to_dict()) == candidate


def test_v58_adds_parallel_capacity_without_serial_depth() -> None:
    old = preset_v57("debug", device="cpu")
    new = preset_v58("debug", device="cpu")
    vocab = V20Vocab.build(new, "abc xyz\n")
    old_model = build_model(old, vocab, device="cpu")
    new_model = build_model(new, vocab, device="cpu")
    assert len(old_model.layers) == len(new_model.layers) == 4
    assert new_model.parameter_count() > old_model.parameter_count()


def test_v58_trace_serialization_is_identical_to_v57() -> None:
    old = preset_v57("debug", device="cpu")
    new = preset_v58("debug", device="cpu")
    old_vocab = V20Vocab.build(old, "abc xyz\n")
    new_vocab = V20Vocab.build(new, "abc xyz\n")
    assert render_v20(_example(), old_vocab, "thinking").tokens == render_v20(
        _example(), new_vocab, "thinking"
    ).tokens


def test_v58_preflight_is_exposed() -> None:
    assert callable(run_preflight)


def test_v58_gate_is_comparative_and_count_uniform() -> None:
    assert GATE_THRESHOLDS == {
        "thinking_accuracy_min": 0.75,
        "thinking_min_count_accuracy_min": 0.70,
        "thinking_count_spread_max": 0.20,
        "thinking_minus_nonthinking_gap_min": 0.30,
    }


def _confirmation_example(count: int, start: int, set_id: str = "set_000") -> V20Example:
    marker = character_token("a")
    return V20Example(
        example_kind="counting_task",
        seq_tokens=[marker] * count,
        corpus_region="test",
        corpus_start=start,
        corpus_end=start + count,
        prompt_sha256=f"{set_id}-{start}",
        set_id=set_id,
        needle_characters=("a", "b", "c"),
        rendered_set_order=("a", "b", "c"),
        needle_positions=tuple(range(count)),
        needle_markers=(marker,) * count,
        count=count,
        per_character_counts=(count, 0, 0),
    )


def test_v58_confirmation_selection_is_balanced_and_disjoint() -> None:
    excluded = [_confirmation_example(count, count * 100) for count in (1, 2)]
    candidates = [
        _confirmation_example(count, count * 100 + offset)
        for count in (1, 2)
        for offset in range(3)
    ]
    selected = select_disjoint_balanced(
        candidates, excluded, count_max=2, examples_per_count=2
    )
    assert [example.count for example in selected] == [1, 1, 2, 2]
    assert {(example.set_id, example.corpus_start) for example in selected}.isdisjoint(
        {(example.set_id, example.corpus_start) for example in excluded}
    )


def test_v58_confirmation_summary_applies_revised_gate() -> None:
    rows = []
    thinking_correct = {count: 8 if count < 10 else 9 for count in range(1, 11)}
    for mode in ("nonthinking", "thinking"):
        for count in range(1, 11):
            correct = 2 if mode == "nonthinking" else thinking_correct[count]
            for row_id in range(10):
                rows.append(
                    {
                        "mode": mode,
                        "count": count,
                        "row_id": row_id,
                        "ar_accuracy": float(row_id < correct),
                        "ar_answered": 1.0,
                        "trace_exact": 0.5 if mode == "thinking" else float("nan"),
                        "trace_ordered_marker_accuracy": (
                            0.9 if mode == "thinking" else float("nan")
                        ),
                        "trace_marker_count_accuracy": 0.8 if mode == "thinking" else float("nan"),
                        "trace_format_valid": 1.0 if mode == "thinking" else float("nan"),
                        "trace_closed": 1.0 if mode == "thinking" else float("nan"),
                        "trace_delimiter_count_accuracy": (
                            0.8 if mode == "thinking" else float("nan")
                        ),
                    }
                )
    summary, by_count, gate = summarize_confirmation(pd.DataFrame(rows))
    assert len(summary) == 2
    assert len(by_count) == 20
    assert gate["passed"] is True
    assert gate["metrics"]["thinking_accuracy"] == pytest.approx(0.81)
    assert gate["metrics"]["thinking_min_count_accuracy"] == pytest.approx(0.8)
    assert gate["metrics"]["thinking_count_spread"] == pytest.approx(0.1)
    assert gate["metrics"]["thinking_minus_nonthinking_gap"] == pytest.approx(0.61)
