from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _design() -> dict:
    return {
        "rank": 3,
        "confirmation_seeds": [1, 2, 3],
        "prompt_removal": {
            "counts": [2, 3],
            "layers": {"Tiny": [0, 1]},
            "realized_norm_relative_tolerance": 0.025,
        },
        "answer_query_removal": {
            "counts": [2, 3],
            "layers": {"Tiny": [0, 1]},
            "realized_norm_relative_tolerance": 0.025,
        },
        "answer_transport": {
            "pairs": [[1, 2], [2, 1]],
            "boundaries": {"Tiny": [[0, 1], [1, 2]]},
            "primary_endpoint": "target_donor_fraction",
            "realized_norm_relative_tolerance": 0.025,
        },
        "map_causal_link": {
            "role": "answer_query",
            "rank": 3,
            "stable_cv_centroid_r2_min": 0.9,
            "stable_bootstrap_map_relative_frobenius_median_max": 0.1,
            "primary_contrast": "aligned_dose_1_minus_orthogonal",
            "primary_metric": "target_donor_fraction",
            "primary_estimand": "stable minus unstable",
        },
        "multiplicity": {
            "prompt_removal": "Holm across registered layers",
            "answer_query_removal": "Holm across registered layers",
            "answer_transport": "Holm across registered boundaries",
            "map_causal_link": "Holm across six tests",
        },
    }


def test_prompt_removal_analysis_accepts_a_complete_registered_grid(
    tmp_path: Path, monkeypatch
) -> None:
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(_design()), encoding="utf-8")
    rows = []
    for seed in (1, 2, 3):
        for count in (2, 3):
            for layer in (0, 1):
                for condition in (
                    "actual_rank3_remove",
                    "actual_normmatched_orthogonal",
                ):
                    candidate = condition == "actual_rank3_remove"
                    rows.append(
                        {
                            "model_label": "Tiny",
                            "seed": seed,
                            "gold_count": count,
                            "layer": layer,
                            "normalized_depth": float(layer),
                            "condition": condition,
                            "prediction": count + int(candidate),
                            "correct": int(not candidate),
                            "absolute_error": int(candidate),
                            "signed_error": int(candidate),
                            "clean_prediction": count,
                            "clean_correct": 1,
                            "clean_absolute_error": 0,
                            "removed_fro_norm": 2.0,
                            "target_removed_fro_norm": 2.0,
                            "norm_ratio": 1.0,
                            "completion": str(count),
                            "runtime_seconds": 0.1,
                        }
                    )
    input_path = tmp_path / "layerwise_prompt_removal_detail.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    output = tmp_path / "prompt-analysis"
    module = _load_script("analyze_v446_layerwise_prompt_removal")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module.__file__,
            str(input_path),
            "--design-config",
            str(design_path),
            "--output",
            str(output),
            "--bootstraps",
            "100",
        ],
    )
    module.main()

    statistics = pd.read_csv(output / "layerwise_prompt_removal_statistics.csv")
    assert len(statistics) == 8
    assert set(statistics["mean_effect"]) == {1.0}
    audit = json.loads((output / "analysis_audit.json").read_text())
    assert audit["status"] == "PASS"


def test_answer_query_removal_analysis_uses_separate_design_and_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(_design()), encoding="utf-8")
    rows = []
    for seed in (1, 2, 3):
        for count in (2, 3):
            for layer in (0, 1):
                for condition in (
                    "actual_rank3_remove",
                    "actual_normmatched_orthogonal",
                ):
                    candidate = condition == "actual_rank3_remove"
                    rows.append(
                        {
                            "model_label": "Tiny",
                            "seed": seed,
                            "gold_count": count,
                            "layer": layer,
                            "normalized_depth": float(layer),
                            "condition": condition,
                            "prediction": count + int(candidate),
                            "correct": int(not candidate),
                            "absolute_error": int(candidate),
                            "signed_error": int(candidate),
                            "clean_prediction": count,
                            "clean_correct": 1,
                            "clean_absolute_error": 0,
                            "removed_fro_norm": 2.0,
                            "target_removed_fro_norm": 2.0,
                            "norm_ratio": 1.0,
                            "completion": str(count),
                            "runtime_seconds": 0.1,
                        }
                    )
    input_path = tmp_path / "layerwise_answer_query_removal_detail.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    output = tmp_path / "answer-query-analysis"
    module = _load_script("analyze_v446_layerwise_prompt_removal")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module.__file__,
            str(input_path),
            "--design-config",
            str(design_path),
            "--support-role",
            "answer_query",
            "--output",
            str(output),
            "--bootstraps",
            "100",
        ],
    )
    module.main()

    statistics = pd.read_csv(
        output / "layerwise_answer_query_removal_statistics.csv"
    )
    assert len(statistics) == 8
    assert set(statistics["mean_effect"]) == {1.0}
    damage = pd.read_csv(
        output / "layerwise_answer_query_removal_damage_statistics.csv"
    )
    assert len(damage) == 8
    assert set(
        damage.loc[
            damage["endpoint"] == "candidate_absolute_error_damage",
            "mean_damage",
        ]
    ) == {1.0}
    assert set(
        damage.loc[
            damage["endpoint"] == "control_absolute_error_damage",
            "mean_damage",
        ]
    ) == {0.0}
    audit = json.loads((output / "analysis_audit.json").read_text())
    assert audit["status"] == "PASS"
    assert audit["support_role"] == "answer_query"
    assert audit["damage_statistics_rows"] == 8


def test_transport_analysis_accepts_a_complete_registered_grid(
    tmp_path: Path, monkeypatch
) -> None:
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(_design()), encoding="utf-8")
    rows = []
    condition_values = {
        "aligned_dose_1": 0.4,
        "aligned_dose_2": 0.8,
        "matched_orthogonal": 0.0,
    }
    for seed in (1, 2, 3):
        for receiver, donor in ((1, 2), (2, 1)):
            for source, target in ((0, 1), (1, 2)):
                for condition, effect in condition_values.items():
                    dose = 2.0 if condition == "aligned_dose_2" else 1.0
                    planned_dose = (
                        dose / 1.038 if condition == "aligned_dose_2" else dose
                    )
                    rows.append(
                        {
                            "model_label": "Tiny",
                            "seed": seed,
                            "receiver_count": receiver,
                            "donor_count": donor,
                            "support": "answer_query_relay",
                            "condition": condition,
                            "source_layer": source,
                            "target_layer": target,
                            "normalized_depth": target / 2,
                            "replacement_delta_norm": dose,
                            "planned_replacement_delta_norm": planned_dose,
                            "aligned_dose_1_norm": 1.0,
                            "realized_norm_ratio_to_aligned_dose_1": dose,
                            "clean_donor_log_odds": -1.0,
                            "condition_donor_log_odds": -1.0 + effect,
                            "donor_log_odds_gain": effect,
                            "target_donor_fraction": effect,
                            "argmax_token_changed": 0,
                            "geometry_discovery_centroid_r2": 0.9,
                            "runtime_seconds": 0.1,
                        }
                    )
    input_path = tmp_path / "layerwise_transport_patch.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)
    output = tmp_path / "transport-analysis"
    module = _load_script("analyze_v446_layerwise_transport_patch")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module.__file__,
            str(input_path),
            "--design-config",
            str(design_path),
            "--output",
            str(output),
            "--bootstraps",
            "100",
        ],
    )
    module.main()

    statistics = pd.read_csv(output / "layerwise_transport_statistics.csv")
    assert len(statistics) == 12
    primary = statistics[
        statistics["contrast"] == "aligned_dose_1_minus_orthogonal"
    ]
    assert set(primary["mean_contrast"].round(8)) == {0.4}
    audit = json.loads((output / "analysis_audit.json").read_text())
    assert audit["status"] == "PASS"
    assert audit["max_planned_norm_ratio_error_diagnostic"] > 0.025


def test_transport_control_matches_the_realized_bfloat16_norm() -> None:
    module = _load_script("run_v446_layerwise_transport_patch")
    generator = torch.Generator().manual_seed(452)
    base = torch.randn(4096, generator=generator).to(torch.bfloat16).float()
    aligned_direction = torch.randn(4096, generator=generator)
    aligned_direction *= 0.8 / torch.linalg.vector_norm(aligned_direction)
    control_direction = torch.randn(4096, generator=generator)

    _, _, aligned_norm = module.quantized_additive_replacement(
        base, aligned_direction, dtype=torch.bfloat16
    )
    replacement, realized, control_norm = module.closest_quantized_direction(
        base,
        control_direction,
        aligned_norm,
        dtype=torch.bfloat16,
    )

    assert replacement.dtype == torch.bfloat16
    assert torch.linalg.vector_norm(realized).item() == control_norm
    assert control_norm / aligned_norm == pytest.approx(1.0, rel=0.025)


def test_transport_aligned_doses_match_small_planned_bfloat16_norms() -> None:
    module = _load_script("run_v446_layerwise_transport_patch")
    generator = torch.Generator().manual_seed(453)
    base = torch.randn(4096, generator=generator).to(torch.bfloat16).float()
    direction = torch.randn(4096, generator=generator)
    direction *= 6.6e-4 / torch.linalg.vector_norm(direction)
    planned_norm = float(torch.linalg.vector_norm(direction))

    dose1, dose2, _ = module.quantized_aligned_doses(
        base,
        direction,
        planned_norm,
        dtype=torch.bfloat16,
        realized_norm_tolerance=0.025,
    )
    dose1_norm = dose1[2]
    dose2_norm = dose2[2]

    assert dose1_norm / planned_norm == pytest.approx(1.0, rel=0.025)
    assert dose2_norm / (2.0 * planned_norm) == pytest.approx(1.0, rel=0.025)
    assert dose2_norm / dose1_norm == pytest.approx(2.0, rel=0.025)


def test_transport_aligned_doses_prioritize_realized_ratio_when_needed(
    monkeypatch,
) -> None:
    module = _load_script("run_v446_layerwise_transport_patch")
    realized_norms = iter((1.0, 1.9, 2.0))

    def fake_closest(base, direction, target_norm, *, dtype):
        del direction, target_norm, dtype
        return base, base, next(realized_norms)

    monkeypatch.setattr(module, "closest_quantized_direction", fake_closest)
    base = torch.zeros(4)
    dose1, dose2, used_paired_target = module.quantized_aligned_doses(
        base,
        torch.ones(4),
        1.0,
        dtype=torch.bfloat16,
        realized_norm_tolerance=0.025,
    )

    assert used_paired_target
    assert dose1[2] == 1.0
    assert dose2[2] / dose1[2] == pytest.approx(2.0, rel=0.025)


def test_map_causal_link_uses_frozen_stability_regimes(
    tmp_path: Path, monkeypatch
) -> None:
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(_design()), encoding="utf-8")
    map_analysis = tmp_path / "maps"
    transport_analysis = tmp_path / "transport"
    map_analysis.mkdir()
    transport_analysis.mkdir()
    (map_analysis / "analysis_audit.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    (transport_analysis / "analysis_audit.json").write_text(
        json.dumps({"status": "PASS"}), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "model_label": "Tiny",
                "role": "answer_query",
                "rank": 3,
                "source_layer": 0,
                "target_layer": 1,
                "cv_centroid_r2": 0.5,
                "bootstrap_map_relative_frobenius_median": 0.2,
                "bootstrap_rotation_geodesic_degrees_median": 12.0,
                "subspace_principal_angle_max_degrees": 70.0,
                "full_operator_cosine_to_next": 0.1,
                "full_operator_relative_drift_to_next": 1.4,
            },
            {
                "model_label": "Tiny",
                "role": "answer_query",
                "rank": 3,
                "source_layer": 1,
                "target_layer": 2,
                "cv_centroid_r2": 0.95,
                "bootstrap_map_relative_frobenius_median": 0.05,
                "bootstrap_rotation_geodesic_degrees_median": 2.0,
                "subspace_principal_angle_max_degrees": 30.0,
                "full_operator_cosine_to_next": 0.8,
                "full_operator_relative_drift_to_next": 0.5,
            },
        ]
    ).to_csv(map_analysis / "layerwise_linear_map_summary.csv", index=False)
    effect_rows = []
    contrasts = (
        "aligned_dose_1_minus_orthogonal",
        "aligned_dose_2_minus_orthogonal",
        "dose_2_minus_dose_1",
    )
    for seed in (1, 2, 3):
        for source, target, effect in ((0, 1, 0.0), (1, 2, 1.0)):
            for contrast in contrasts:
                for metric in ("target_donor_fraction", "donor_log_odds_gain"):
                    effect_rows.append(
                        {
                            "model_label": "Tiny",
                            "source_layer": source,
                            "target_layer": target,
                            "normalized_depth": target / 2,
                            "contrast": contrast,
                            "metric": metric,
                            "seed": seed,
                            "effect": effect,
                            "pairs_per_seed": 2,
                        }
                    )
    pd.DataFrame(effect_rows).to_csv(
        transport_analysis / "layerwise_transport_seed_effects.csv", index=False
    )
    output = tmp_path / "link"
    module = _load_script("analyze_v446_map_causal_link")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            module.__file__,
            "--map-analysis",
            str(map_analysis),
            "--transport-analysis",
            str(transport_analysis),
            "--design-config",
            str(design_path),
            "--output",
            str(output),
            "--bootstraps",
            "100",
        ],
    )
    module.main()

    tests = pd.read_csv(output / "stable_minus_unstable_tests.csv")
    assert len(tests) == 6
    assert set(tests["mean_stable_minus_unstable"]) == {1.0}
    assert set(tests["stable_boundaries"]) == {1}
    correlations = pd.read_csv(output / "boundary_spearman_descriptive.csv")
    assert set(correlations["predictor"]) == {
        "cv_centroid_r2",
        "bootstrap_map_relative_frobenius_median",
        "bootstrap_rotation_geodesic_degrees_median",
        "subspace_principal_angle_max_degrees",
        "full_operator_cosine_to_next",
        "full_operator_relative_drift_to_next",
    }
    assert json.loads((output / "analysis_audit.json").read_text())["status"] == "PASS"
