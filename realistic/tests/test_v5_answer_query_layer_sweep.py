from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_analyzer():
    path = REPO_ROOT / "scripts" / "analyze_v5_answer_query_layer_sweep.py"
    spec = importlib.util.spec_from_file_location("answer_query_layer_analyzer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    pairs = [
        {
            "pair_id": "p1",
            "model_label": "Qwen3-8B",
            "seed": 1254,
            "receiver_count": 3,
            "donor_count": 4,
            "pair_direction": "higher_to_lower",
        },
        {
            "pair_id": "p2",
            "model_label": "Qwen3-8B",
            "seed": 1255,
            "receiver_count": 7,
            "donor_count": 6,
            "pair_direction": "lower_to_higher",
        },
    ]
    trials: list[dict] = []
    for layer in (0, 5):
        for pair in pairs:
            receiver = pair["receiver_count"]
            donor = pair["donor_count"]
            full_prediction = receiver if layer == 0 else donor
            trials.extend(
                [
                    {
                        "pair_id": pair["pair_id"],
                        "layer": layer,
                        "condition": "self_patch",
                        "prediction": receiver,
                        "completion_text_raw": str(receiver),
                        "generated_token_count": 1,
                    },
                    {
                        "pair_id": pair["pair_id"],
                        "layer": layer,
                        "condition": "full_donor_patch",
                        "prediction": full_prediction,
                        "completion_text_raw": str(full_prediction),
                        "generated_token_count": 1,
                    },
                ]
            )
    pairs_path = tmp_path / "pairs.jsonl"
    trials_path = tmp_path / "trials.jsonl"
    _write_jsonl(pairs_path, pairs)
    _write_jsonl(trials_path, trials)
    return trials_path, pairs_path


def test_analyzer_requires_and_summarizes_full_preregistered_layer_grid(
    tmp_path: Path,
) -> None:
    analyzer = _load_analyzer()
    trials, pairs = _fixture(tmp_path)
    output = tmp_path / "analysis"
    audit = analyzer.analyze(
        trials,
        pairs,
        output,
        expected_layers=[0, 5],
    )

    layers = pd.read_csv(output / "layer_effects.csv").set_index("layer")
    assert layers.loc[0, "full_donor_adoption"] == 0.0
    assert layers.loc[5, "full_donor_adoption"] == 1.0
    assert audit["layers"] == [0, 5]
    assert audit["completed_pair_layer_cells"] == 4
    assert audit["descriptive_onset_layer"] == {"Qwen3-8B": 5}


def test_analyzer_rejects_silently_missing_entire_layer(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    trials, pairs = _fixture(tmp_path)
    rows = [
        row
        for row in (
            json.loads(line)
            for line in trials.read_text(encoding="utf-8").splitlines()
        )
        if row["layer"] == 0
    ]
    _write_jsonl(trials, rows)

    with pytest.raises(ValueError, match="preregistered grid"):
        analyzer.analyze(
            trials,
            pairs,
            tmp_path / "analysis",
            expected_layers=[0, 5],
        )


def test_analyzer_rejects_self_patch_regeneration_failure(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    trials, pairs = _fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in trials.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["prediction"] = 9
    _write_jsonl(trials, rows)

    with pytest.raises(ValueError, match="Self patch did not reproduce"):
        analyzer.analyze(
            trials,
            pairs,
            tmp_path / "analysis",
            expected_layers=[0, 5],
        )
