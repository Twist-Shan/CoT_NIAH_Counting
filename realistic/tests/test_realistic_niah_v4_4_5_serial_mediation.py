from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.analyze_realistic_niah_v4_4_5_serial_mediation import (
    cluster_bootstrap_mean,
    unit_effects,
)
from realistic_niah_v4.layerwise_removal import PromptRemovalGeometry
from realistic_niah_v4_4_5.serial_mediation import (
    SERIAL_ARMS,
    late_answer_path_hook,
    load_answer_geometry,
    retrieval_path_hook,
    serial_arm_map,
    validate_serial_registry,
)


class IdentityAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.o_proj = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.o_proj.weight.copy_(torch.eye(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.o_proj(value)


def fake_adapter(width: int = 8) -> tuple[SimpleNamespace, IdentityAttention, nn.Identity]:
    attention = IdentityAttention(width)
    layer = nn.Identity()
    adapter = SimpleNamespace(
        output_projections=[attention.o_proj],
        attentions=[attention],
        layers=[layer],
        head_dims=[4],
        num_heads=[2],
        num_layers=1,
    )
    return adapter, attention, layer


def test_frozen_arm_registry_and_order() -> None:
    names = tuple(arm.name for arm in SERIAL_ARMS)
    assert len(names) == 11
    assert len(set(names)) == 11
    assert serial_arm_map()["S_Raligned_Torth"].retrieval == "aligned"
    assert serial_arm_map()["S_Raligned_Torth"].late == "orthogonal"
    validate_serial_registry(names, source=8, retrieval=23, late=29)


def test_load_answer_geometry_uses_only_discovery_rows(tmp_path) -> None:
    rng = np.random.default_rng(20260814)
    rows = []
    counts = []
    seeds = []
    splits = []
    for seed in (1234, 1235, 1254):
        for count in range(1, 11):
            for repeat in range(3):
                signal = np.asarray(
                    [count, count**2 / 10, count**3 / 100], dtype=np.float64
                )
                nuisance = rng.normal(size=5) + repeat * 0.01
                rows.append(np.concatenate((signal, nuisance)))
                counts.append(count)
                seeds.append(seed)
                splits.append("discovery" if seed < 1254 else "confirmation")
    path = tmp_path / "answer.npz"
    np.savez_compressed(
        path,
        states=np.stack(rows),
        count=np.asarray(counts),
        seed=np.asarray(seeds),
        split=np.asarray(splits),
    )
    geometry = load_answer_geometry(
        path, discovery_seeds=(1234, 1235), rank=3
    )
    assert geometry.basis.shape == (8, 3)
    assert geometry.control_basis.shape == (8, 3)
    assert geometry.classes.tolist() == list(range(1, 11))
    assert np.linalg.norm(geometry.basis.T @ geometry.control_basis) < 1e-7


def test_retrieval_aligned_hook_removes_frozen_coordinates() -> None:
    adapter, attention, _layer = fake_adapter()
    encoding = SimpleNamespace(sequence_length=5, query_position=4)
    basis = torch.eye(8)[:3]
    value = torch.arange(40, dtype=torch.float32).reshape(1, 5, 8) / 10
    with retrieval_path_hook(
        adapter,
        encoding,
        layer=0,
        heads=(0, 1),
        mean=torch.zeros(8),
        basis=basis,
        control_direction=torch.nn.functional.normalize(torch.eye(8)[3], dim=0),
        mode="aligned",
    ) as audit:
        output = attention(value)
    assert audit["applications"] == 1
    assert np.allclose(audit["coordinates_after"], [0.0, 0.0, 0.0], atol=1e-6)
    assert torch.allclose(output[0, 4, :3], torch.zeros(3), atol=1e-6)
    assert abs(audit["norm_ratio"] - 1.0) < 1e-6


def test_retrieval_bank_reconstruction_does_not_reenter_projection_hooks() -> None:
    adapter, attention, _layer = fake_adapter()
    encoding = SimpleNamespace(sequence_length=5, query_position=4)
    basis = torch.eye(8)[:3]
    value = torch.arange(40, dtype=torch.float32).reshape(1, 5, 8) / 10
    observed_calls = 0

    def full_prompt_only_hook(_module, args):
        nonlocal observed_calls
        observed_calls += 1
        if args[0].shape[1] != encoding.sequence_length:
            raise RuntimeError("capture hook saw a recursive projection call")

    capture_handle = attention.o_proj.register_forward_pre_hook(full_prompt_only_hook)
    try:
        with retrieval_path_hook(
            adapter,
            encoding,
            layer=0,
            heads=(0, 1),
            mean=torch.zeros(8),
            basis=basis,
            control_direction=torch.nn.functional.normalize(torch.eye(8)[3], dim=0),
            mode=None,
        ) as audit:
            attention(value)
    finally:
        capture_handle.remove()

    assert observed_calls == 1
    assert audit["applications"] == 1


def test_retrieval_interventions_move_cpu_geometry_to_model_device() -> None:
    if not torch.cuda.is_available():
        return
    encoding = SimpleNamespace(sequence_length=5, query_position=4)
    value = (
        torch.arange(40, dtype=torch.float32, device="cuda").reshape(1, 5, 8)
        / 10
    )
    for mode in ("aligned", "orthogonal"):
        adapter, attention, _layer = fake_adapter()
        attention.to("cuda")
        with retrieval_path_hook(
            adapter,
            encoding,
            layer=0,
            heads=(0, 1),
            mean=torch.zeros(8, device="cpu"),
            basis=torch.eye(8, device="cpu")[:3],
            control_direction=torch.nn.functional.normalize(
                torch.eye(8, device="cpu")[3], dim=0
            ),
            mode=mode,
        ) as audit:
            output = attention(value)
        assert output.device.type == "cuda"
        assert audit["applications"] == 1
        assert abs(audit["norm_ratio"] - 1.0) < 1e-3


def test_late_aligned_hook_removes_count_coordinate() -> None:
    adapter, _attention, layer = fake_adapter()
    encoding = SimpleNamespace(sequence_length=5, query_position=4)
    basis = np.eye(8, dtype=np.float64)[:, :3]
    control = np.eye(8, dtype=np.float64)[:, 3:6]
    centroids = np.zeros((10, 8), dtype=np.float64)
    geometry = PromptRemovalGeometry(
        classes=np.arange(1, 11),
        centroids=centroids,
        basis=basis,
        control_basis=control,
        centroid_variance_capture=1.0,
    )
    value = torch.arange(40, dtype=torch.float32).reshape(1, 5, 8) / 10
    with late_answer_path_hook(
        adapter,
        encoding,
        layer=0,
        geometry=geometry,
        mode="aligned",
    ) as audit:
        output = layer(value)
    assert audit["applications"] == 1
    assert np.allclose(audit["coordinates_after"], [0.0, 0.0, 0.0], atol=1e-6)
    assert torch.allclose(output[0, 4, :3], torch.zeros(3), atol=1e-6)
    assert abs(audit["norm_ratio"] - 1.0) < 1e-6


def test_serial_effect_definitions_match_preregistered_example() -> None:
    expected_counts = {
        "C": 2.0,
        "O": 3.0,
        "S": 6.0,
        "S_Rorth": 5.8,
        "S_Raligned": 4.8,
        "S_Torth": 5.5,
        "S_Taligned": 4.5,
        "S_Rorth_Torth": 5.5,
        "S_Raligned_Torth": 4.5,
        "S_Rorth_Taligned": 5.0,
        "S_Raligned_Taligned": 4.7,
    }
    rows = []
    for arm, expected in expected_counts.items():
        retrieval_mode = (
            "aligned" if "Raligned" in arm else "orthogonal" if "Rorth" in arm else "none"
        )
        rows.append(
            {
                "model_label": "Qwen3-8B",
                "seed": 1254,
                "gold_count": 8,
                "arm": arm,
                "expected_count": expected,
                "strict_absolute_error": abs(round(expected) - 8),
                "retrieval_coordinates_before": [1.0, 2.0, 3.0],
                "late_coordinates_before": (
                    [0.5, 0.0, 0.0]
                    if retrieval_mode == "aligned"
                    else [1.0, 0.0, 0.0]
                ),
                "retrieval_bank_broad_score_mean": 0.4 if arm.startswith("S") else 0.2,
            }
        )
    effects = unit_effects(pd.DataFrame(rows))
    assert np.isclose(effects["source_repair"], 3.0)
    assert np.isclose(effects["retrieval_mediation"], 1.0)
    assert np.isclose(effects["late_mediation"], 1.0)
    assert np.isclose(effects["joint_interaction"], -0.7)
    assert np.isclose(effects["source_broad_score_change"], 0.2)
    assert effects["late_to_retrieval_invariance_max_abs"] == 0.0


def test_serial_inference_uses_equal_weight_seed_clusters() -> None:
    summary = cluster_bootstrap_mean(
        np.asarray([0.0, 0.0, 0.0, 0.0, 10.0, 20.0]),
        np.asarray([1, 1, 1, 1, 2, 3]),
        draws=2000,
        seed=17,
    )
    assert summary["mean"] == 10.0
    assert summary["mean"] != np.mean([0.0, 0.0, 0.0, 0.0, 10.0, 20.0])
    assert summary["units"] == 3
    assert summary["independent_seed_clusters"] == 3
    assert summary["paired_seed_count_units"] == 6
    assert summary["finite_rows_per_cluster_min"] == 1
    assert summary["finite_rows_per_cluster_max"] == 4
    assert summary["bootstrap_method"] == "seed-cluster percentile bootstrap"


def test_serial_seed_sign_flip_is_exact_and_deterministic() -> None:
    values = np.repeat(np.arange(1.0, 11.0), 10)
    seeds = np.repeat(np.arange(1254, 1264), 10)
    first = cluster_bootstrap_mean(values, seeds, draws=500, seed=20260829)
    second = cluster_bootstrap_mean(values, seeds, draws=500, seed=20260829)
    assert first == second
    assert first["units"] == 10
    assert first["paired_seed_count_units"] == 100
    assert first["finite_rows_per_cluster_min"] == 10
    assert first["finite_rows_per_cluster_max"] == 10
    assert first["exact_sign_flip_permutations"] == 1024
    assert first["exact_sign_flip_p_two_sided"] == 2 / 1024
