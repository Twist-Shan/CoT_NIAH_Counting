from pathlib import Path

import matplotlib.figure
import torch

from dataset_generation.qk_hook_attention.outlier_analysis import (
    _layer_colors_from_measurement_rows,
    analyze_massive_activations,
    join_outlier_attention,
    plot_qk_outlier_figures,
)


def test_plot_qk_outlier_figures_plots_all_layers_and_saves_tables(
    tmp_path: Path, monkeypatch
) -> None:
    run_dir = tmp_path
    (run_dir / "tables").mkdir()
    (run_dir / "tables" / "inputs_0_measurements.csv").write_text(
        "position,layer,layer_color,relative_norm_diff,cosine_similarity\n"
        "0,2,tab:orange,0.1,0.9\n"
        "1,2,tab:orange,0.2,0.8\n"
        "0,4,tab:purple,0.3,0.7\n"
        "1,4,tab:purple,0.4,0.6\n",
        encoding="utf-8",
    )
    for layer in [2, 4]:
        norm_dir = run_dir / "tensors" / "massive_activations" / "input_0"
        norm_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "norm_ratio_to_median": torch.tensor([float(layer), float(layer + 1)]),
                "linf_norm_ratio_to_median": torch.tensor(
                    [float(layer + 10), float(layer + 11)]
                ),
            },
            norm_dir / f"hidden_norms_layer_{layer:02d}.pt",
        )
        stats_dir = run_dir / "tensors" / "attention_stats" / "input_0"
        stats_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "received_uniform_ratio": torch.tensor(
                    [float(layer + 2), float(layer + 3)]
                )
            },
            stats_dir / f"attention_stats_layer_{layer:02d}_head_00.pt",
        )

    saved_figures = []
    original_savefig = matplotlib.figure.Figure.savefig

    def spy_savefig(self, *args, **kwargs):
        saved_figures.append(self)
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy_savefig)

    paths = plot_qk_outlier_figures(run_dir=run_dir, example_indices=[0], layers=[2, 4])

    assert paths == [run_dir / "figures" / "inputs_0_qk_outliers.png"]
    assert paths[0].exists()
    assert len(saved_figures) == 1
    axes = saved_figures[0].axes
    assert len(axes) == 5
    assert [
        line.get_label() for line in axes[2].lines if line.get_linestyle() != "--"
    ] == ["layer 2", "layer 4"]
    assert [
        line.get_label() for line in axes[3].lines if line.get_linestyle() != "--"
    ] == ["layer 2", "layer 4"]
    assert axes[2].get_yscale() == "log"
    assert axes[3].get_yscale() == "log"
    assert axes[4].get_yscale() == "log"
    for ax in axes:
        solid_lines = [line for line in ax.lines if line.get_linestyle() != "--"]
        dashed_lines = [line for line in ax.lines if line.get_linestyle() == "--"]
        assert all(tuple(line.get_xdata()) == (0, 1) for line in solid_lines)
        assert sorted(int(line.get_xdata()[0]) for line in dashed_lines) == [0, 1]
        assert all(tuple(line.get_xdata()) in {(0, 0), (1, 1)} for line in dashed_lines)

    table_path = run_dir / "tables" / "inputs_0_qk_outliers.csv"
    text = table_path.read_text(encoding="utf-8")
    assert (
        "position,layer,hidden_norm_ratio_to_median,hidden_linf_norm_ratio_to_median,max_received_uniform_ratio"
        in text
    )
    assert "0,2,2.0,12.0,4.0" in text
    assert "0,4,4.0,14.0,6.0" in text


def test_plot_qk_outlier_figures_respects_configured_output_dirs(
    tmp_path: Path,
    capsys,
) -> None:
    run_dir = tmp_path
    (run_dir / "custom_tables").mkdir()
    (run_dir / "custom_tables" / "inputs_0_measurements.csv").write_text(
        "position,layer,layer_color,relative_norm_diff,cosine_similarity\n"
        "0,2,tab:orange,0.1,0.9\n",
        encoding="utf-8",
    )
    norm_dir = run_dir / "tensors" / "massive_activations" / "input_0"
    norm_dir.mkdir(parents=True)
    torch.save(
        {
            "norm_ratio_to_median": torch.tensor([2.0]),
            "linf_norm_ratio_to_median": torch.tensor([12.0]),
        },
        norm_dir / "hidden_norms_layer_02.pt",
    )

    paths = plot_qk_outlier_figures(
        run_dir=run_dir,
        example_indices=[0],
        layers=[2],
        figures_dir="custom_figures",
        tables_dir="custom_tables",
    )

    assert paths == [run_dir / "custom_figures" / "inputs_0_qk_outliers.png"]
    captured = capsys.readouterr()
    assert f"[qk-outlier] saved figure={paths[0]}" in captured.out
    assert paths[0].exists()
    assert (run_dir / "custom_tables" / "inputs_0_qk_outliers.csv").exists()
    assert not (run_dir / "figures" / "inputs_0_qk_outliers.png").exists()
    assert not (run_dir / "tables" / "inputs_0_qk_outliers.csv").exists()


def test_plot_qk_outlier_figures_uses_uncontrolled_final_input_axis(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path
    (run_dir / "tables").mkdir()
    (run_dir / "tables" / "inputs_0_measurements.csv").write_text(
        "position,layer,layer_color,relative_norm_diff,cosine_similarity\n"
        "0,2,tab:orange,0.1,0.9\n"
        "1,2,tab:orange,0.2,0.8\n"
        "2,2,tab:orange,0.3,0.7\n",
        encoding="utf-8",
    )
    tensors_dir = run_dir / "tensors"
    tensors_dir.mkdir()
    norm_dir = tensors_dir / "massive_activations" / "input_0"
    norm_dir.mkdir(parents=True)
    torch.save(
        {
            "norm_ratio_to_median": torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0]),
            "linf_norm_ratio_to_median": torch.tensor([20.0, 21.0, 22.0, 23.0, 24.0]),
        },
        norm_dir / "hidden_norms_layer_02.pt",
    )
    stats_dir = tensors_dir / "attention_stats" / "input_0"
    stats_dir.mkdir(parents=True)
    torch.save(
        {"received_uniform_ratio": torch.tensor([30.0, 31.0, 32.0, 33.0, 34.0])},
        stats_dir / "attention_stats_layer_02_head_00.pt",
    )

    plot_qk_outlier_figures(run_dir=run_dir, example_indices=[0], layers=[2])

    text = (run_dir / "tables" / "inputs_0_qk_outliers.csv").read_text(encoding="utf-8")
    assert "0,2,10.0,20.0,30.0" in text
    assert "1,2,11.0,21.0,31.0" in text
    assert "2,2,12.0,22.0,32.0" in text
    assert "3,2,13.0,23.0,33.0" in text
    assert "4,2,14.0,24.0,34.0" in text


def test_analyze_massive_activations_writes_outside_needle_report(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path
    tensors_dir = run_dir / "tensors"
    tensors_dir.mkdir(parents=True)
    hidden = torch.ones(1, 6, 1)
    hidden[0, :, 0] = torch.tensor([1.0, 100.0, 100.0, 1.0, 20.0, 10.0])
    torch.save(
        {
            "hidden": hidden,
            "hidden_control": hidden,
            "sample_idx": 0,
            "layers": [2],
            "stored_layers": [2],
            "expanded_needle_segments": [
                {"needle_id": "N1", "start": 1, "end": 3, "expanded_end": 4}
            ],
        },
        tensors_dir / "hidden_inputs_0.pt",
    )

    analyze_massive_activations(
        run_dir=run_dir, layers=[2], threshold=10.0, top_k=2, n_edge=0, n_after_needle=1
    )

    outside_text = (
        run_dir / "tables" / "massive_tokens_outside_needles.txt"
    ).read_text(encoding="utf-8")
    assert "pos=1" not in outside_text
    assert "pos=2" not in outside_text
    assert "pos=3" not in outside_text
    assert "pos=4" in outside_text
    assert "pos=5" in outside_text
    outside_csv = (
        run_dir / "tables" / "massive_tokens_outside_needles_all.csv"
    ).read_text(encoding="utf-8")
    assert "median_norm_outside_needles" in outside_csv


def test_analyze_massive_activations_handles_original_layer_ids_for_stored_subset(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path
    tensors_dir = run_dir / "tensors"
    tensors_dir.mkdir(parents=True)
    hidden = torch.ones(2, 3, 4)
    hidden[0, :, :] *= 2
    hidden[1, :, :] *= 4
    torch.save(
        {
            "hidden": hidden,
            "hidden_control": hidden,
            "sample_idx": 0,
            "layers": [2, 4],
            "stored_layers": [2, 4],
        },
        tensors_dir / "hidden_inputs_0.pt",
    )

    analyze_massive_activations(
        run_dir=run_dir,
        layers=[2, 4],
        threshold=10.0,
        top_k=0,
        n_edge=0,
        n_after_needle=0,
    )

    assert (
        tensors_dir / "massive_activations" / "input_0" / "hidden_norms_layer_02.pt"
    ).exists()
    assert (
        tensors_dir / "massive_activations" / "input_0" / "hidden_norms_layer_04.pt"
    ).exists()


def test_layer_colors_prefer_measurement_table_colors() -> None:
    rows = [
        {"layer": "2", "layer_color": "tab:orange"},
        {"layer": "4", "layer_color": "tab:purple"},
    ]

    colors = _layer_colors_from_measurement_rows(
        rows, [2, 4, 6], ["fallback0", "fallback1", "fallback2"]
    )

    assert colors[2] == "tab:orange"
    assert colors[4] == "tab:purple"
    assert colors[6] == "fallback2"


def test_analyze_massive_activations_uses_hidden_input_ids_not_mismatched_qk_cache(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path
    tensors_dir = run_dir / "tensors"
    tensors_dir.mkdir(parents=True)
    hidden = torch.ones(1, 3, 1)
    hidden[0, :, 0] = torch.tensor([1.0, 100.0, 2.0])
    torch.save(
        {
            "hidden": hidden,
            "hidden_control": hidden,
            "sample_idx": 0,
            "layers": [2],
            "stored_layers": [2],
            "input_ids": [101, 202, 303],
        },
        tensors_dir / "hidden_inputs_0.pt",
    )

    cache_dir = tensors_dir / "qk_cache" / "input_0"
    cache_dir.mkdir(parents=True)
    torch.save(torch.tensor([[999, 888, 777]]), cache_dir / "input_ids.pt")
    (cache_dir / "tokens.json").write_text(
        '["wrong0", "wrong1", "wrong2"]', encoding="utf-8"
    )
    (cache_dir / "metadata.json").write_text(
        '{"special_token_ids": []}', encoding="utf-8"
    )
    (cache_dir / "analysis_spec.json").write_text("{}", encoding="utf-8")

    all_events, _, _ = analyze_massive_activations(
        run_dir=run_dir,
        layers=[2],
        threshold=10.0,
        top_k=1,
        n_edge=0,
        n_after_needle=0,
    )

    event = next(row for row in all_events if row["position"] == 1)
    assert event["token_id"] == 202
    assert event["token"] == "<id:202>"
    assert "wrong1" not in (run_dir / "tables" / "massive_tokens.txt").read_text(
        encoding="utf-8"
    )


def test_analyze_massive_activations_falls_back_to_model_input_ids_table(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path
    tensors_dir = run_dir / "tensors"
    tensors_dir.mkdir(parents=True)
    hidden = torch.ones(1, 2, 1)
    hidden[0, :, 0] = torch.tensor([1.0, 50.0])
    torch.save(
        {
            "hidden": hidden,
            "hidden_control": hidden,
            "sample_idx": 7,
            "layers": [2],
            "stored_layers": [2],
        },
        tensors_dir / "hidden_inputs_7.pt",
    )
    tables_dir = run_dir / "tables"
    tables_dir.mkdir()
    (tables_dir / "model_input_ids.txt").write_text(
        "Example ID 7\n"
        "uncontrolled input ids\n"
        "111 222\n"
        "controlled input ids\n"
        "111 333\n",
        encoding="utf-8",
    )

    all_events, _, _ = analyze_massive_activations(
        run_dir=run_dir,
        layers=[2],
        threshold=10.0,
        top_k=1,
        n_edge=0,
        n_after_needle=0,
    )

    event = next(row for row in all_events if row["position"] == 1)
    assert event["token_id"] == 222
    assert event["token"] == "<id:222>"


def test_join_outlier_attention_handles_short_attention_stats(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path
    stats_dir = run_dir / "tensors" / "attention_stats" / "input_0"
    stats_dir.mkdir(parents=True)
    torch.save(
        {
            "head": 0,
            "received_mean": torch.tensor([0.1, 0.2]),
            "received_sum": torch.tensor([1.0, 2.0]),
            "received_count": torch.tensor([1, 1]),
            "uniform_baseline": torch.tensor([0.5, 0.5]),
            "received_uniform_ratio": torch.tensor([0.2, 0.4]),
        },
        stats_dir / "attention_stats_layer_02_head_00.pt",
    )

    joined, overlap = join_outlier_attention(
        run_dir=run_dir,
        massive_events=[
            {
                "example_idx": 0,
                "layer": 2,
                "position": 3,
                "norm_ratio_to_median": 10.0,
            }
        ],
        sink_rows=[],
    )

    assert overlap[0]["massive_event_count"] == 1
    assert overlap[0]["massive_head_join_count"] == 1
    assert joined == [
        {
            "example_idx": 0,
            "layer": 2,
            "position": 3,
            "norm_ratio_to_median": 10.0,
            "is_topk_sink": False,
            "attention_stats_missing_reason": "position_outside_attention_stats",
            "attention_stats_seq_len": 2,
        }
    ]


def test_ensure_qk_cache_uses_parent_run_cot_inputs_for_ablation_example(
    tmp_path: Path, monkeypatch
) -> None:
    from dataset_generation.dynamic_niah_v2 import DynamicNiahV2Config
    from dataset_generation.qk_hook_attention import outlier_analysis as outlier

    root_run = tmp_path / "run"
    example_run = root_run / "ablation_examples" / "example_id_0"
    (root_run / "tensors").mkdir(parents=True)
    cot_input_ids = torch.tensor([[11, 22, 33, 44]], dtype=torch.long)
    torch.save({"input_ids": cot_input_ids}, root_run / "tensors" / "inputs_cot_0.pt")
    captured = {}

    def fake_capture_qk_cache(**kwargs):
        captured.update(kwargs)
        out_dir = kwargs["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(kwargs["input_ids_override"], out_dir / "input_ids.pt")
        torch.save(
            torch.ones_like(kwargs["input_ids_override"]), out_dir / "attention_mask.pt"
        )
        torch.save(
            torch.arange(kwargs["input_ids_override"].shape[1]).view(1, -1),
            out_dir / "position_ids.pt",
        )
        torch.save(torch.zeros(1, 4, 4), out_dir / "layer_02_q_raw.pt")
        torch.save(torch.zeros(1, 4, 4), out_dir / "layer_02_k_raw.pt")
        torch.save({}, out_dir / "layer_02_qk_norms.pt")
        return {
            "prompt": "prompt",
            "model_text": "prompt",
            "markers": {},
            "analysis_spec": {},
            "tokens": ["a", "b", "c", "d"],
            "metadata": {"seq_len": 4},
            "model_elapsed_seconds": 0.0,
        }

    monkeypatch.setattr(
        outlier, "build_uncontrolled_prompt_text", lambda *args, **kwargs: "prompt"
    )
    monkeypatch.setattr(outlier, "capture_qk_cache", fake_capture_qk_cache)

    cache_dir = outlier.ensure_qk_cache(
        run_dir=example_run,
        example_idx=0,
        row={"needles": []},
        cfg=DynamicNiahV2Config(analyze_reasoning_tokens=True),
        analysis_cfg=outlier.QKOutlierAnalysisConfig(model="fake", layers=[2]),
        model=object(),
        tokenizer=object(),
    )

    assert cache_dir == example_run / "tensors" / "qk_cache" / "input_0"
    assert torch.equal(captured["input_ids_override"], cot_input_ids)
