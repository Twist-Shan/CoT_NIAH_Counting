from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from realistic_niah_v5.cross_mode_geometry import ModeDataset
from realistic_niah_v6 import native_aligned_representation as aligned
from realistic_niah_v6.completion import audit_native_aligned_representation
from realistic_niah_v6.spec import (
    CONFIRMATION_SEEDS,
    COUNTS,
    DISCOVERY_SEEDS,
    MODEL_LABELS,
    PROMPT_MODES,
)


ROOT = Path(__file__).resolve().parents[1]


def _cell_from_index(path: str | Path) -> tuple[str, str]:
    value = Path(path)
    for prompt_mode in PROMPT_MODES:
        for model_label in MODEL_LABELS:
            if prompt_mode in value.parts and model_label in value.parts:
                return prompt_mode, model_label
    raise AssertionError(path)


def _datasets() -> tuple[
    dict[tuple[str, str], ModeDataset],
    dict[tuple[str, str], ModeDataset],
]:
    running: dict[tuple[str, str], ModeDataset] = {}
    final: dict[tuple[str, str], ModeDataset] = {}
    all_seeds = (*DISCOVERY_SEEDS, *CONFIRMATION_SEEDS)
    hidden = 18
    for cell_index, cell in enumerate(
        (mode, model) for mode in PROMPT_MODES for model in MODEL_LABELS
    ):
        prompt_mode, model_label = cell
        running_metadata: list[dict[str, object]] = []
        final_metadata: list[dict[str, object]] = []
        running_clear: list[np.ndarray] = []
        final_clear: list[np.ndarray] = []
        rng = np.random.default_rng(100 + cell_index)
        for seed in all_seeds:
            split = "discovery" if seed in DISCOVERY_SEEDS else "confirmation"
            for gold_count in COUNTS:
                stimulus_id = f"v4.4/seed-{seed}/count-{gold_count}"
                for occurrence in range(1, gold_count + 1):
                    running_metadata.append(
                        {
                            "split": split,
                            "seed": seed,
                            "gold_count": gold_count,
                            "occurrence": occurrence,
                            "stimulus_id": stimulus_id,
                            "marker_kind": (
                                "indexed"
                                if prompt_mode == "enumeration_index"
                                else "bullet"
                            ),
                        }
                    )
                    state = np.zeros(hidden, dtype=np.float32)
                    state[occurrence - 1] = 4.0
                    state[10] = (seed - 1248.5) / 20.0
                    state += rng.normal(0.0, 0.04, hidden).astype(np.float32)
                    running_clear.append(state)
                final_metadata.append(
                    {
                        "split": split,
                        "seed": seed,
                        "gold_count": gold_count,
                        "occurrence": gold_count,
                        "stimulus_id": stimulus_id,
                        "marker_kind": (
                            "indexed"
                            if prompt_mode == "enumeration_index"
                            else "bullet"
                        ),
                    }
                )
                state = np.zeros(hidden, dtype=np.float32)
                state[gold_count - 1] = 4.0
                state[11] = (seed - 1248.5) / 20.0
                state += rng.normal(0.0, 0.04, hidden).astype(np.float32)
                final_clear.append(state)
        running_clear_array = np.stack(running_clear)
        final_clear_array = np.stack(final_clear)
        running_noise = rng.normal(
            0.0, 1.0, running_clear_array.shape
        ).astype(np.float32)
        final_noise = rng.normal(0.0, 1.0, final_clear_array.shape).astype(np.float32)
        running[cell] = ModeDataset(
            mode="native_thinking",
            model_label=model_label,
            metadata=pd.DataFrame(running_metadata),
            states_by_layer={0: running_noise, 1: running_clear_array},
        )
        final[cell] = ModeDataset(
            mode="native_thinking",
            model_label=model_label,
            metadata=pd.DataFrame(final_metadata),
            states_by_layer={0: final_noise, 1: final_clear_array},
        )
        running[cell].validate()
        final[cell].validate()
    return running, final


def _write_capture_registries(run_root: Path) -> None:
    for prompt_mode in PROMPT_MODES:
        for model_label in MODEL_LABELS:
            capture = (
                run_root
                / prompt_mode
                / model_label
                / "capture"
                / "confirmation_all_sample"
            )
            capture.mkdir(parents=True)
            rows = []
            for seed in (*DISCOVERY_SEEDS, *CONFIRMATION_SEEDS):
                split = (
                    "discovery" if seed in DISCOVERY_SEEDS else "confirmation"
                )
                for gold_count in COUNTS:
                    rows.append(
                        {
                            "request_id": (
                                f"{prompt_mode}/{model_label}/{seed}/{gold_count}"
                            ),
                            "stimulus_id": (
                                f"v4.4/seed-{seed}/count-{gold_count}"
                            ),
                            "split": split,
                            "seed": seed,
                            "gold_count": gold_count,
                            "model_label": model_label,
                            "prompt_mode": prompt_mode,
                        }
                    )
            (capture / "capture_index.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            (capture / "v6_adapter_manifest.json").write_text(
                json.dumps(
                    {
                        "run_status": "COMPLETE",
                        "formal_cohort": False,
                        "prompt_mode": prompt_mode,
                        "model_label": model_label,
                    }
                )
                + "\n",
                encoding="utf-8",
            )


def _install_loaders(
    monkeypatch: pytest.MonkeyPatch,
    running: dict[tuple[str, str], ModeDataset],
    final: dict[tuple[str, str], ModeDataset],
) -> None:
    monkeypatch.setattr(
        aligned,
        "load_native_thinking_capture",
        lambda path, **_kwargs: running[_cell_from_index(path)],
    )
    monkeypatch.setattr(
        aligned,
        "load_native_thinking_final_count",
        lambda path: final[_cell_from_index(path)],
    )


def test_native_aligned_representation_uses_two_exact_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    output = run_root / "native_aligned_representation"
    _write_capture_registries(run_root)
    running, final = _datasets()
    _install_loaders(monkeypatch, running, final)

    paths = aligned.analyze_native_aligned_representation(
        run_root=run_root,
        output_dir=output,
        contract_path=(
            ROOT / "configs/realistic_niah_v6_native_analysis_alignment_v1.json"
        ),
        project_root=ROOT,
        command="pytest synthetic native-aligned",
    )

    assert paths["complete"].read_text(encoding="utf-8") == "PASS\n"
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    assert audit["status"] == "PASS_NATIVE_ANALYSIS_PATH_ALIGNED"
    assert audit["running_index"]["site_kind"] == "item_end"
    assert audit["running_index"]["common_state_rows"] == 1650
    assert audit["running_index"]["common_trajectory_cells"] == 300
    assert audit["final_count"]["site_kind"] == "answer_query_v3"
    assert audit["final_count"]["trajectory_rows_per_cell"] == 300
    assert audit["replacement_rows_allowed"] is False

    running_candidates = pd.read_csv(paths["running_candidates"])
    final_candidates = pd.read_csv(paths["final_candidates"])
    contrasts = pd.read_csv(paths["grammar_contrasts"])
    assert len(running_candidates) == 8
    assert len(final_candidates) == 8
    assert len(contrasts) == 4
    assert set(running_candidates["token_site"]) == {"item_end"}
    assert set(final_candidates["token_site"]) == {"answer_query_v3"}
    assert running_candidates["exact_four_cell_sample_alignment"].all()
    assert set(contrasts["contrast"]) == {
        "enumeration_bullet_minus_enumeration_index"
    }
    for prompt_mode in PROMPT_MODES:
        for model_label in MODEL_LABELS:
            manifest = (
                run_root
                / prompt_mode
                / model_label
                / "representation"
                / "native_aligned"
                / "cell_manifest.json"
            )
            value = json.loads(manifest.read_text(encoding="utf-8"))
            assert value["status"] == "PASS_NATIVE_ANALYSIS_PATH_ALIGNED"
            assert value["confirmation_used_for_selection"] is False
            assert value["replacement_rows"] == 0
    completion = audit_native_aligned_representation(run_root)
    assert completion["status"] == "PASS_NATIVE_ANALYSIS_PATH_ALIGNED"
    assert completion["verified_artifact_hashes"] >= 20


def test_native_aligned_representation_rejects_cross_grammar_stimulus_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    _write_capture_registries(run_root)
    running, final = _datasets()
    broken = running[("enumeration_bullet", "Qwen3-8B")]
    broken.metadata.loc[0, "stimulus_id"] = "different/source"
    _install_loaders(monkeypatch, running, final)

    with pytest.raises(ValueError, match="Stimulus identity differs"):
        aligned.analyze_native_aligned_representation(
            run_root=run_root,
            output_dir=run_root / "native_aligned_representation",
            contract_path=(
                ROOT
                / "configs/realistic_niah_v6_native_analysis_alignment_v1.json"
            ),
            project_root=ROOT,
        )
