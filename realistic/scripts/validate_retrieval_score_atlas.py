from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == (
        "realistic_niah_v4_4_2_retrieval_score_atlas_v1"
    )
    assert payload["score_definitions"]["nonthinking"]["name"] == (
        "broad retrieval primary"
    )
    assert payload["score_definitions"]["native_thinking"]["name"] == (
        "targeted retrieval lift"
    )
    expected = {
        "Qwen3-8B": (36, 32),
        "Gemma4-E4B": (42, 8),
    }
    for model, (expected_layers, expected_heads) in expected.items():
        for mode in ("nonthinking", "native_thinking"):
            mode_data = payload["models"][model]["modes"][mode]
            assert len(mode_data["layers"]) == expected_layers
            assert mode_data["heads"] == expected_heads
            for condition in ("cue_present", "cue_absent"):
                condition_data = mode_data["conditions"][condition]
                assert condition_data["samples"] == 100
                score = condition_data["layer_head_score"]
                assert len(score) == expected_layers
                assert all(len(row) == expected_heads for row in score)
                valid = condition_data["valid_samples_by_layer"]
                assert len(valid) == expected_layers
                if mode == "nonthinking":
                    assert set(valid) == {100}
                    assert all(value is not None for row in score for value in row)
    gemma_native = payload["models"]["Gemma4-E4B"]["modes"][
        "native_thinking"
    ]
    gemma_native_valid = [
        value
        for condition in gemma_native["conditions"].values()
        for value in condition["valid_samples_by_layer"]
    ]
    print(
        json.dumps(
            {
                "validated": str(path),
                "bytes": path.stat().st_size,
                "models": sorted(payload["models"]),
                "gemma_native_valid_range": [
                    min(gemma_native_valid),
                    max(gemma_native_valid),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    arguments = parser.parse_args()
    main(arguments.path)
