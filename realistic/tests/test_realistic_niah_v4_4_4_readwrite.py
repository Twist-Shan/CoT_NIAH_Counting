from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from realistic_niah_v4.prompts import PromptEncoding, TokenSpan
from realistic_niah_v4_4_3.interventions import QueryBundle
from realistic_niah_v4_4_4.readwrite_analysis import (
    _json_records,
    _markdown_report,
    build_seed_metrics,
    primary_decision,
    summarize_seed_metrics,
)
from realistic_niah_v4_4_4.readwrite import (
    fit_count_intercept_and_step,
    full_set_delta_from_bundles,
    read_component_output_diagnostics,
    shapley_read_decomposition,
    stable_position_partition,
    write_central_difference_diagnostics,
)
from realistic_niah_v4_4_4.readwrite_spec import V444ReadWriteConfig
from realistic_niah_v4_4_4.readwrite_pipeline import _attention_cache_diagnostics


def _load_runner_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_realistic_niah_v4_4_4_readwrite.py"
    )
    spec = importlib.util.spec_from_file_location("v444_readwrite_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the read/write supplement runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _adapter() -> SimpleNamespace:
    projection = torch.nn.Linear(8, 3, bias=False)
    with torch.no_grad():
        projection.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5],
                    [0.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0],
                    [0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
                ]
            )
        )
    return SimpleNamespace(
        head_dims={28: 2},
        num_heads={28: 4},
        output_projections={28: projection},
        num_layers=36,
    )


def _bundle(values: torch.Tensor, alpha: torch.Tensor) -> QueryBundle:
    heads = []
    for head in range(4):
        kv = head // 2
        selected = values[:, kv * 2 : kv * 2 + 2]
        heads.append(torch.einsum("k,kd->d", alpha[head], selected))
    return QueryBundle(
        logits=torch.zeros(12),
        candidate_log_scores={count: float(-count) for count in range(1, 11)},
        z_by_layer={28: torch.cat(heads)},
        value_by_layer={28: values},
        attention_output_by_layer={28: torch.zeros(3)},
        alpha_by_layer={28: alpha},
        alpha_key_start_by_layer={28: 0},
        attention_cache_candidate_logit_max_abs_delta=0.0,
        attention_cache_candidate_centered_logit_max_abs_delta=0.0,
    )


def _encoding() -> PromptEncoding:
    slots = (
        TokenSpan(0, 1, 2, True, "needle", 1, 1),
        TokenSpan(1, 3, 4, False, "negative", 1, 1),
    )
    return PromptEncoding(
        stimulus_id="x",
        design_variant="v4.4",
        seed=1,
        split="confirmation",
        count=1,
        model_label="Qwen3-8B",
        answer_format="numeric",
        text="x",
        generation_prompt="x",
        input_ids=(1, 2, 3, 4, 5, 6, 7, 8),
        attention_mask=(1,) * 8,
        query_position=7,
        slot_spans=slots,
        needle_spans=(slots[0],),
        hard_negative_spans=(slots[1],),
        count_candidate_texts=tuple((count, str(count)) for count in range(1, 11)),
        count_candidate_answer_token_ids=tuple(
            (count, (count,)) for count in range(1, 11)
        ),
        count_candidate_token_ids=tuple(
            (count, (count, 0)) for count in range(1, 11)
        ),
    )


def _alpha() -> torch.Tensor:
    return torch.tensor(
        [
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20, 0.20],
            [0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
            [0.10, 0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
            [0.10, 0.10, 0.20, 0.10, 0.10, 0.10, 0.10, 0.20],
        ],
        dtype=torch.float32,
    )


def test_config_and_position_partition_are_frozen_and_exhaustive() -> None:
    V444ReadWriteConfig().validate()
    groups = stable_position_partition(_encoding(), tail_width=3)
    names = (
        "slot_tokens",
        "pre_query_non_slot_early",
        "pre_query_non_slot_tail",
        "answer_query_self",
    )
    flat = [position for name in names for position in groups[name]]
    assert len(flat) == len(set(flat)) == 8
    assert set(flat) == set(groups["all_positions"]) == set(range(8))
    assert groups["slot_tokens"] == (1, 3)
    assert groups["answer_query_self"] == (7,)


def test_attention_cache_logit_drift_is_recorded_but_not_a_hard_failure() -> None:
    bundle = SimpleNamespace(
        attention_cache_candidate_logit_max_abs_delta=0.75,
        attention_cache_candidate_centered_logit_max_abs_delta=0.5833358764648438,
    )
    diagnostics = _attention_cache_diagnostics(bundle, reference_tolerance=0.5)
    assert diagnostics["attention_cache_candidate_logit_max_abs_delta"] == 0.75
    assert diagnostics[
        "attention_cache_candidate_centered_logit_max_abs_delta"
    ] == pytest.approx(0.5833358764648438)
    assert diagnostics["attention_cache_reference_tolerance_exceeded"] is True

    bundle.attention_cache_candidate_logit_max_abs_delta = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        _attention_cache_diagnostics(bundle, reference_tolerance=0.5)


def test_base_snapshot_rejects_changes_to_frozen_v444_files(tmp_path: Path) -> None:
    runner = _load_runner_module()
    protected = (
        Path("resolved_config.json"),
        Path("dataset/stimuli.jsonl"),
        Path("models/Qwen3-8B/directions/complete.json"),
        Path("models/Qwen3-8B/center_controls/complete.json"),
        Path("models/Qwen3-8B/center_controls/selection.json"),
        Path("models/Qwen3-8B/center_controls/artifacts.pt"),
        Path("models/Qwen3-8B/smoke/complete.json"),
        Path("models/Qwen3-8B/confirmation/complete.json"),
    )
    for offset, relative in enumerate(protected):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"frozen-{offset}".encode())
    runner._register_base_snapshot(tmp_path, "Qwen3-8B")
    assert (tmp_path / "v4_4_4_read_write_base_snapshot.json").is_file()
    (tmp_path / "resolved_config.json").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="base artifacts changed"):
        runner._register_base_snapshot(tmp_path, "Qwen3-8B")


def test_shapley_decomposition_has_exact_closure_and_reconstructs_full_z() -> None:
    adapter = _adapter()
    receiver_values = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10
    donor_values = receiver_values + torch.tensor([0.2, -0.1, 0.3, -0.2])
    receiver_alpha = _alpha()
    donor_alpha = torch.roll(receiver_alpha, shifts=1, dims=1)
    receiver = _bundle(receiver_values, receiver_alpha)
    donor = _bundle(donor_values, donor_alpha)
    result = shapley_read_decomposition(
        receiver,
        donor,
        adapter,
        layer=28,
        heads=(0, 1),
        positions=tuple(range(8)),
    )
    assert result["closure_relative_l2"] < 1e-6
    assert torch.allclose(result["full"], result["value"] + result["routing"])
    exact = full_set_delta_from_bundles(
        receiver, donor, adapter, layer=28, heads=(0, 1)
    )
    assert torch.allclose(result["full"], exact, atol=1e-6)


def test_anchored_shapley_closes_onto_captured_pre_o_endpoints() -> None:
    adapter = _adapter()
    values = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10
    receiver = _bundle(values, _alpha())
    donor = _bundle(values + 0.1, torch.roll(_alpha(), shifts=1, dims=1))
    receiver = replace(
        receiver,
        z_by_layer={28: receiver.z_by_layer[28] + 0.01},
    )
    donor = replace(
        donor,
        z_by_layer={28: donor.z_by_layer[28] - 0.02},
    )
    result = shapley_read_decomposition(
        receiver,
        donor,
        adapter,
        layer=28,
        heads=(0, 1),
        positions=tuple(range(8)),
        anchor_to_captured_endpoints=True,
    )
    exact = full_set_delta_from_bundles(
        receiver, donor, adapter, layer=28, heads=(0, 1)
    )
    assert torch.allclose(result["full"], exact, atol=1e-6)
    assert torch.allclose(
        result["value"] + result["routing"], exact, atol=1e-6
    )
    assert result["receiver_endpoint_reconstruction_relative_l2"] > 0
    assert result["donor_endpoint_reconstruction_relative_l2"] > 0


def test_shapley_separates_pure_value_and_pure_routing_changes() -> None:
    adapter = _adapter()
    values = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10
    alpha = _alpha()
    receiver = _bundle(values, alpha)
    value_donor = _bundle(values + 0.25, alpha)
    value_result = shapley_read_decomposition(
        receiver,
        value_donor,
        adapter,
        layer=28,
        heads=(0, 1),
        positions=tuple(range(8)),
    )
    assert torch.linalg.vector_norm(value_result["routing"]) < 1e-7
    route_donor = _bundle(values, torch.roll(alpha, shifts=1, dims=1))
    route_result = shapley_read_decomposition(
        receiver,
        route_donor,
        adapter,
        layer=28,
        heads=(0, 1),
        positions=tuple(range(8)),
    )
    assert torch.linalg.vector_norm(route_result["value"]) < 1e-7


def test_read_output_diagnostics_are_finite() -> None:
    adapter = _adapter()
    values = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10
    receiver = _bundle(values, _alpha())
    donor = _bundle(values + 0.1, torch.roll(_alpha(), shifts=1, dims=1))
    result = shapley_read_decomposition(
        receiver,
        donor,
        adapter,
        layer=28,
        heads=(0, 1),
        positions=tuple(range(8)),
    )
    diagnostics = read_component_output_diagnostics(
        adapter,
        layer=28,
        heads=(0, 1),
        components=result,
        global_z_count_steps=torch.tensor([[1.0, 0.0], [0.5, 0.5]]),
        count_gap=5,
    )
    assert diagnostics["closure_relative_l2"] < 1e-6
    assert all(math_value == pytest.approx(math_value) for math_value in diagnostics.values())


def test_count_axis_fit_and_write_central_difference() -> None:
    counts = (1, 2, 3, 4)
    step = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    intercept = torch.tensor([[0.5, -1.0], [2.0, 0.0]])
    states = torch.stack([intercept + count * step for count in counts])
    fitted_intercept, fitted_step = fit_count_intercept_and_step(states, counts)
    assert torch.allclose(fitted_intercept, intercept, atol=1e-6)
    assert torch.allclose(fitted_step, step, atol=1e-6)

    downstream = {28: torch.tensor([1.0, 0.0]), 29: torch.tensor([0.0, 2.0])}
    plus = {28: torch.tensor([1.0, 0.0]), 29: torch.tensor([0.0, 4.0])}
    minus = {28: torch.tensor([-1.0, 0.0]), 29: torch.tensor([0.0, -4.0])}
    rows = write_central_difference_diagnostics(
        plus, minus, beta=1.0, downstream_steps=downstream
    )
    assert rows[0]["natural_count_axis_coefficient"] == pytest.approx(1.0)
    assert rows[1]["natural_count_axis_coefficient"] == pytest.approx(2.0)
    assert all(row["natural_count_axis_cosine"] == pytest.approx(1.0) for row in rows)


def test_synthetic_analysis_recovers_value_dominant_read_and_write() -> None:
    config = V444ReadWriteConfig(
        evaluation_seeds=tuple(range(20)),
        donor_pairs=((1, 6),),
        downstream_layers=(28, 29),
        write_counts=(3,),
        bootstrap_repetitions=100,
    )
    natural_rows = []
    read_rows = []
    read_trace_rows = []
    write_rows = []
    for seed in config.evaluation_seeds:
        natural_rows.extend(
            {
                "seed": seed,
                "gold_count": count,
                "layer": layer,
                "baseline_correct": True,
            }
            for count in config.counts
            for layer in config.downstream_layers
        )
        for component, transport, blocked, control in (
            ("full", 0.30, None, None),
            ("value", 0.25, 0.05, 0.24),
            ("routing", 0.0, 0.0, 0.0),
        ):
            interventions = [("component_patch", transport)]
            if component != "full":
                interventions.extend(
                    [
                        ("component_patch_plus_natural_axis_block", blocked),
                        ("component_patch_plus_orthogonal_control", control),
                    ]
                )
            for intervention, observed in interventions:
                read_rows.append(
                    {
                        "seed": seed,
                        "receiver_count": 1,
                        "donor_count": 6,
                        "component": component,
                        "intervention": intervention,
                        "continuous_normalized_transport": observed,
                        "component_mechanical_transport": transport,
                        "baseline_predicted_count": 1,
                    }
                )
            for layer in config.downstream_layers:
                read_trace_rows.append(
                    {
                        "seed": seed,
                        "receiver_count": 1,
                        "donor_count": 6,
                        "component": component,
                        "layer": layer,
                        "downstream_count_axis_coefficient": transport * 5,
                    }
                )
        for intervention, sign, behavior_scale, residual_scale in (
            ("natural_plus", 1.0, 0.20, 0.50),
            ("natural_minus", -1.0, 0.20, 0.50),
            ("orthogonal_plus", 1.0, 0.01, 0.01),
            ("orthogonal_minus", -1.0, 0.01, 0.01),
        ):
            for layer in config.downstream_layers:
                write_rows.append(
                    {
                        "seed": seed,
                        "gold_count": 3,
                        "intervention": intervention,
                        "layer": layer,
                        "signed_beta": sign * config.write_beta,
                        "delta_expected_count": (
                            sign * behavior_scale * config.write_beta
                        ),
                        "downstream_count_axis_coefficient": (
                            sign * residual_scale * config.write_beta
                        ),
                    }
                )
    seed_metrics = build_seed_metrics(
        pd.DataFrame(natural_rows),
        pd.DataFrame(read_rows),
        pd.DataFrame(read_trace_rows),
        pd.DataFrame(write_rows),
        config=config,
    )
    summary = summarize_seed_metrics(seed_metrics, config=config)
    decision = primary_decision(summary, config=config)
    assert decision["read_mode"]["classification"] == "value_dominant"
    assert decision["write_propagation"]["supported"] is True
    assert decision["serial_read_write_supported"] is True
    report = _markdown_report(
        {
            "primary_decision": decision,
            "summary": _json_records(summary),
            "audit": {"all_checks_pass": True, "check_count": 7},
        },
        config=config,
    )
    assert "读取模式为 **value_dominant**" in report
    assert "| L29 |" in report
