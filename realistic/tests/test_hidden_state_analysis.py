import json
from pathlib import Path

import torch

from dataset_generation.hidden_state_analysis import (
    _scatter_non_outlier_masks,
    build_outside_segments_mask,
    build_prompt_needle_spans,
    build_uncontrolled_needle_insertions,
    compute_needle_sensitive_tokens,
    expand_needle_segments,
    compare_hidden_states,
    compute_alignment_offset,
    compute_pca_projection_2d,
    fit_pca_projections_from_records,
    load_hidden_state_records,
    normalize_hidden_tensor,
    plot_pca_projection,
    plot_pca_trajectory,
    plot_measurements,
    plot_saved_hidden_pca,
    project_hidden_states_with_pca,
    prune_large_pt_files,
    project_hidden_trajectories_pca,
    save_hidden_states,
    save_model_input_ids_table,
    save_needle_sensitive_outputs,
    split_pca_train_test_records,
)


def test_compute_alignment_offset_detects_shift() -> None:
    inputs = torch.tensor([[10, 11, 12, 13, 14, 15]])
    inputs_control = torch.tensor([[99, 98, 10, 11, 12, 13, 14, 15]])
    offset = compute_alignment_offset(
        inputs, inputs_control, insertion_position=0, max_search_offset=4
    )
    assert offset == 2


def test_compare_hidden_states_shapes_and_ranges() -> None:
    hidden = torch.randn(4, 12, 6)
    hidden_control = hidden.clone()
    hidden_control[:, 5:, :] += 0.02

    out = compare_hidden_states(
        hidden, hidden_control, insertion_position=5, offset=0, layer_indices=[1, 3]
    )

    assert out["relative_norm_diff"].shape == (2, 12)
    assert out["cosine_similarity"].shape == (2, 12)
    assert out["positions"].tolist() == list(range(12))
    assert torch.isfinite(out["relative_norm_diff"]).all()
    assert (out["cosine_similarity"] >= -1.000001).all()
    assert (out["cosine_similarity"] <= 1.000001).all()


def test_compare_hidden_states_returns_cpu_tensors() -> None:
    hidden = torch.randn(2, 6, 4)
    hidden_control = hidden.clone()

    out = compare_hidden_states(
        hidden, hidden_control, insertion_position=2, offset=0, layer_indices=[0]
    )

    assert out["relative_norm_diff"].device.type == "cpu"
    assert out["cosine_similarity"].device.type == "cpu"
    assert out["positions"].device.type == "cpu"
    assert out["control_positions"].device.type == "cpu"


def test_compare_hidden_states_accepts_cuda_tensors_when_available() -> None:
    if not torch.cuda.is_available():
        return
    hidden = torch.randn(2, 6, 4, device="cuda")
    hidden_control = hidden.clone()

    out = compare_hidden_states(
        hidden, hidden_control, insertion_position=2, offset=0, layer_indices=[0]
    )

    assert torch.allclose(out["relative_norm_diff"], torch.zeros(1, 6))
    assert out["relative_norm_diff"].device.type == "cpu"


def test_compare_hidden_states_applies_offset_on_uncontrolled_axis() -> None:
    hidden = torch.randn(2, 6, 4)
    hidden_control = torch.randn(2, 8, 4)
    hidden_control[:, 0, :] = hidden[:, 0, :]
    hidden_control[:, 3:, :] = hidden[:, 1:, :]

    out = compare_hidden_states(
        hidden, hidden_control, insertion_position=1, offset=2, layer_indices=[0]
    )

    assert out["positions"].tolist() == list(range(6))
    assert out["control_positions"].tolist() == [0, 3, 4, 5, 6, 7]
    assert torch.isfinite(out["relative_norm_diff"]).all()
    assert torch.allclose(out["relative_norm_diff"], torch.zeros(1, 6))
    assert torch.allclose(out["cosine_similarity"], torch.ones(1, 6), atol=1e-6)


def test_compute_pca_projection_2d_shapes_and_finite_values() -> None:
    hidden_layer = torch.randn(10, 5)

    projection, mean = compute_pca_projection_2d(hidden_layer)

    assert projection.shape == (5, 2)
    assert mean.shape == (5,)
    assert torch.isfinite(projection).all()
    assert torch.isfinite(mean).all()


def test_project_hidden_trajectories_pca_applies_alignment_offset() -> None:
    hidden = torch.randn(3, 8, 5)
    hidden_control = torch.randn(3, 10, 5)
    hidden_control[:, 2:, :] = hidden

    projected = project_hidden_trajectories_pca(
        hidden, hidden_control, layer_idx=1, insertion_position=0, offset=2
    )

    assert projected["normal"].shape == (8, 2)
    assert projected["control"].shape == (8, 2)
    assert projected["positions"].tolist() == list(range(8))
    assert torch.isfinite(projected["normal"]).all()
    assert torch.isfinite(projected["control"]).all()


def test_plot_pca_trajectory_writes_expected_filename(tmp_path: Path) -> None:
    hidden = torch.randn(3, 12, 6)
    hidden_control = hidden.clone()
    hidden_control[:, 4:, :] += 0.01

    out_path = plot_pca_trajectory(
        hidden,
        hidden_control,
        layer_idx=1,
        insertion_position=4,
        offset=0,
        output_dir=tmp_path,
        sample_idx=0,
    )

    assert out_path == tmp_path / "PCA_layer_1_inputs_0.png"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_normalize_hidden_tensor_accepts_optional_batch_dimension() -> None:
    hidden = torch.randn(1, 3, 7, 5)

    normalized = normalize_hidden_tensor(hidden)

    assert normalized.shape == (3, 7, 5)
    assert normalized.dtype == torch.float32


def test_normalize_hidden_tensor_rejects_multi_batch() -> None:
    hidden = torch.randn(2, 3, 7, 5)

    try:
        normalize_hidden_tensor(hidden)
    except ValueError as exc:
        assert "batch_size=1" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_save_and_load_hidden_state_records_round_trip(tmp_path: Path) -> None:
    hidden = torch.randn(1, 3, 8, 5)
    hidden_control = torch.randn(3, 9, 5)

    out_path = save_hidden_states(
        hidden,
        hidden_control,
        tmp_path,
        4,
        layers=[0, 1],
        insertion_position=2,
        offset=1,
        pca_start_position=3,
    )
    records = load_hidden_state_records(tmp_path)

    assert out_path == tmp_path / "hidden_inputs_4.pt"
    assert len(records) == 1
    assert records[0]["sample_idx"] == 4
    assert records[0]["hidden"].shape == (2, 8, 5)
    assert records[0]["hidden_control"].shape == (2, 9, 5)
    assert records[0]["stored_layers"] == [0, 1]
    assert records[0]["insertion_position"] == 2
    assert records[0]["offset"] == 1
    assert records[0]["pca_start_position"] == 3


def test_split_pca_train_test_records_reserves_first_five() -> None:
    records = [{"sample_idx": i, "hidden": torch.randn(2, 4, 3)} for i in range(8)]

    train, test = split_pca_train_test_records(records, test_count=5)

    assert [r["sample_idx"] for r in train] == [5, 6, 7]
    assert [r["sample_idx"] for r in test] == [0, 1, 2, 3, 4]


def test_fit_pca_projections_from_records_uses_train_records_only() -> None:
    records = []
    for i in range(3):
        hidden = torch.randn(2, 10, 4)
        hidden[:, -1, :] = 10_000.0
        records.append(
            {"sample_idx": i, "hidden": hidden, "hidden_control": hidden.clone()}
        )

    projections = fit_pca_projections_from_records(
        records, [0, 1], filter_top_frac=0.10
    )

    assert set(projections) == {0, 1}
    for params in projections.values():
        projection = params["projection"]
        assert projection.shape == (4, 2)
        assert params["mean"].shape == (4,)
        assert torch.isfinite(projection).all()
        gram = projection.T @ projection
        assert torch.allclose(gram, torch.eye(2), atol=1e-4)


def test_fit_pca_projections_from_records_excludes_pre_pca_start_positions() -> None:
    records = []
    for i, post_values in enumerate(([1.0, 3.0], [5.0, 7.0])):
        hidden = torch.zeros(2, 4, 3)
        hidden[:, :2, 0] = 1_000.0
        hidden[0, 2:, 0] = torch.tensor(post_values)
        hidden[1, 2:, 1] = torch.tensor(post_values)
        records.append(
            {
                "sample_idx": i,
                "hidden": hidden,
                "hidden_control": hidden.clone(),
                "pca_start_position": 2,
            }
        )

    projections = fit_pca_projections_from_records(records, [0, 1], filter_top_frac=0.0)

    assert torch.allclose(projections[0]["mean"], torch.tensor([4.0, 0.0, 0.0]))
    assert torch.allclose(projections[1]["mean"], torch.tensor([0.0, 4.0, 0.0]))


def test_scatter_non_outlier_masks_exclude_far_projected_points() -> None:
    normal = torch.stack([torch.tensor([float(i), 0.0]) for i in range(10)])
    control = torch.stack(
        [torch.tensor([float(i), 0.0]) for i in range(10, 20)]
        + [torch.tensor([1_000.0, 0.0])]
    )

    normal_keep, control_keep, outlier_count, total_count = _scatter_non_outlier_masks(
        normal, control
    )

    assert normal_keep.tolist() == [True] * 10
    assert control_keep.tolist() == [True] * 10 + [False]
    assert outlier_count == 1
    assert total_count == 21


def test_project_hidden_states_with_train_projection_starts_at_requested_position() -> (
    None
):
    hidden = torch.randn(3, 8, 5)
    hidden_control = torch.randn(3, 10, 5)
    projection = torch.eye(5, 2)
    mean = torch.zeros(5)

    projected = project_hidden_states_with_pca(
        hidden,
        hidden_control,
        layer_idx=1,
        projection=projection,
        mean=mean,
        start_position=3,
    )

    assert projected["normal"].shape == (5, 2)
    assert projected["control"].shape == (5, 2)
    assert projected["positions"].tolist() == list(range(3, 8))
    assert projected["control_positions"].tolist() == list(range(3, 8))


def test_project_hidden_states_with_train_projection_applies_alignment_offset() -> None:
    hidden = torch.randn(2, 6, 4)
    hidden_control = torch.randn(2, 8, 4)
    hidden_control[:, 3:, :] = hidden[:, 1:, :]
    projection = torch.eye(4, 2)
    mean = torch.zeros(4)

    projected = project_hidden_states_with_pca(
        hidden,
        hidden_control,
        layer_idx=1,
        projection=projection,
        mean=mean,
        start_position=1,
        insertion_position=1,
        offset=2,
    )

    assert projected["normal"].shape == (5, 2)
    assert projected["control"].shape == (5, 2)
    assert projected["positions"].tolist() == list(range(1, 6))
    assert projected["control_positions"].tolist() == list(range(3, 8))
    assert torch.equal(projected["normal"], projected["control"])


def test_plot_pca_projection_writes_expected_filename(tmp_path: Path) -> None:
    projected = {
        "normal": torch.randn(8, 2),
        "control": torch.randn(9, 2),
        "positions": torch.arange(8),
        "control_positions": torch.arange(9),
    }

    out_path = plot_pca_projection(projected, tmp_path, sample_idx=7, layer_idx=2)

    assert out_path == tmp_path / "PCA_layer_2_inputs_7.png"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_saved_hidden_pca_writes_five_test_examples(tmp_path: Path) -> None:
    records = []
    for i in range(7):
        hidden = torch.randn(2, 12, 5)
        hidden_control = torch.randn(2, 13, 5)
        records.append(
            {"sample_idx": i, "hidden": hidden, "hidden_control": hidden_control}
        )

    paths = plot_saved_hidden_pca(records, tmp_path, [1], test_count=5)

    assert len(paths) == 5
    assert paths[0] == tmp_path / "PCA_layer_1_inputs_0.png"
    assert all(path.exists() for path in paths)


def test_plot_saved_hidden_pca_uses_record_pca_start_position(tmp_path: Path) -> None:
    records = []
    for i in range(7):
        hidden = torch.randn(2, 12, 5)
        hidden_control = torch.randn(2, 13, 5)
        records.append(
            {
                "sample_idx": i,
                "hidden": hidden,
                "hidden_control": hidden_control,
                "pca_start_position": 4,
            }
        )

    projected = project_hidden_states_with_pca(
        records[-1]["hidden"],
        records[-1]["hidden_control"],
        layer_idx=1,
        projection=torch.eye(5, 2),
        mean=torch.zeros(5),
        start_position=records[-1]["pca_start_position"],
    )

    assert projected["positions"].tolist()[0] == 4
    assert projected["control_positions"].tolist()[0] == 4


def test_prune_large_pt_files_deletes_only_oversized_pt(tmp_path: Path) -> None:
    small = tmp_path / "small.pt"
    large = tmp_path / "large.pt"
    txt = tmp_path / "large.txt"
    small.write_bytes(b"12")
    large.write_bytes(b"12345")
    txt.write_bytes(b"12345")

    deleted = prune_large_pt_files(tmp_path, max_bytes=4, delete=True)

    assert deleted == [large]
    assert small.exists()
    assert not large.exists()
    assert txt.exists()


def test_plot_measurements_saves_plot_table_in_run_tables(tmp_path: Path) -> None:
    figures_dir = tmp_path / "figures"
    measurements = {
        "layers": torch.tensor([2, 4]),
        "relative_norm_diff": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        "cosine_similarity": torch.tensor([[0.9, 0.8], [0.7, 0.6]]),
    }

    out_path = plot_measurements(measurements, figures_dir, sample_idx=3)

    table_path = tmp_path / "tables" / "inputs_3_measurements.csv"
    assert out_path == figures_dir / "inputs_3.png"
    assert out_path.exists()
    assert table_path.exists()
    text = table_path.read_text(encoding="utf-8")
    assert "position,layer,layer_color,relative_norm_diff,cosine_similarity" in text
    assert "1,4,#ff7f0e,0.4000000059604645,0.6000000238418579" in text


def test_build_uncontrolled_needle_insertions_uses_original_needle_lengths() -> None:
    realized = [
        {
            "needle_id": "N1",
            "requested_position": 0,
            "final_position": 0,
            "tokens": [900],
            "decoded_text": "control",
            "is_control": True,
            "inserted_from": "control",
        },
        {
            "needle_id": "N2",
            "requested_position": 10,
            "final_position": 11,
            "tokens": [20, 21],
            "decoded_text": "needle two",
            "is_control": False,
            "inserted_from": "needle",
        },
    ]
    needles = [
        {
            "needle_id": "N1",
            "tokens": [10, 11, 12],
            "decoded_text": "needle one",
            "is_control": True,
        },
        {
            "needle_id": "N2",
            "tokens": [20, 21],
            "decoded_text": "needle two",
            "is_control": False,
        },
    ]

    restored = build_uncontrolled_needle_insertions(realized, needles)

    assert [item["final_position"] for item in restored] == [0, 13]
    assert [item["tokens"] for item in restored] == [[10, 11, 12], [20, 21]]
    assert restored[0]["decoded_text"] == "needle one"
    assert restored[0]["is_control"] is True
    assert restored[0]["inserted_from"] == "needle"


def test_build_prompt_needle_spans_uses_final_model_input_positions() -> None:
    input_ids = [100, 101, 10, 11, 102, 20, 21, 22, 103]
    realized = [
        {"needle_id": "N1", "final_position": 0, "tokens": [10, 11]},
        {"needle_id": "N2", "final_position": 4, "tokens": [20, 21, 22]},
    ]

    spans = build_prompt_needle_spans(input_ids, realized)

    assert spans == [
        {
            "needle_id": "N1",
            "start": 2,
            "end": 4,
            "length": 2,
            "context_final_position": 0,
            "is_control": False,
            "inserted_from": None,
            "decoded_text": None,
        },
        {
            "needle_id": "N2",
            "start": 5,
            "end": 8,
            "length": 3,
            "context_final_position": 4,
            "is_control": False,
            "inserted_from": None,
            "decoded_text": None,
        },
    ]


def test_build_prompt_needle_spans_approximates_retokenized_later_needles() -> None:
    input_ids = [100, 101, 10, 11, 102, 200, 201, 202, 103]
    realized = [
        {"needle_id": "N1", "final_position": 0, "tokens": [10, 11]},
        {"needle_id": "N2", "final_position": 4, "tokens": [20, 21, 22]},
    ]

    spans = build_prompt_needle_spans(input_ids, realized)

    assert len(spans) == 2
    assert spans[0]["needle_id"] == "N1"
    assert spans[0]["start"] == 2
    assert spans[0]["end"] == 4
    assert spans[1]["needle_id"] == "N2"
    assert spans[1]["start"] == 6
    assert spans[1]["end"] == 9


def test_build_prompt_needle_spans_uses_prompt_text_offsets_when_tokens_retokenize() -> (
    None
):
    prompt_text = (
        "prefix Alpha City scored 91 points. middle Beta Town scored 88 points. suffix"
    )
    token_offsets = [
        (0, 6),
        (7, 12),
        (13, 17),
        (18, 24),
        (25, 27),
        (28, 35),
        (36, 42),
        (43, 49),
        (50, 55),
        (56, 62),
        (63, 65),
        (66, 73),
    ]
    input_ids = list(range(len(token_offsets)))
    realized = [
        {
            "needle_id": "N1",
            "final_position": 0,
            "tokens": [1000, 1001],
            "decoded_text": "Alpha City scored 91 points.",
        },
        {
            "needle_id": "N2",
            "final_position": 4,
            "tokens": [1002, 1003],
            "decoded_text": "Beta Town scored 88 points.",
        },
    ]

    spans = build_prompt_needle_spans(
        input_ids, realized, prompt_text=prompt_text, token_offsets=token_offsets
    )

    assert [(span["needle_id"], span["start"], span["end"]) for span in spans] == [
        ("N1", 1, 6),
        ("N2", 7, 12),
    ]


def test_expanded_needle_segments_and_mask_exclude_following_tokens() -> None:
    spans = [{"needle_id": "N1", "start": 3, "end": 5}]

    expanded = expand_needle_segments(spans, sequence_length=10, expansion=2)
    mask = build_outside_segments_mask(10, expanded)

    assert expanded[0]["expanded_end"] == 7
    assert mask.tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_compute_needle_sensitive_tokens_sorts_and_decodes_outside_segments() -> None:
    hidden = torch.zeros(2, 6, 3)
    hidden_control = torch.zeros(2, 7, 3)
    input_ids = [10, 11, 12, 13, 14, 15]
    # Layer 1 candidates after masking are positions 0, 1, 4, 5. Cosines are
    # 1.0, 0.0, -1.0, and approximately 0.707, respectively.
    hidden[1, 0] = torch.tensor([1.0, 0.0, 0.0])
    hidden[1, 1] = torch.tensor([1.0, 0.0, 0.0])
    hidden[1, 4] = torch.tensor([1.0, 0.0, 0.0])
    hidden[1, 5] = torch.tensor([1.0, 1.0, 0.0])
    hidden_control[1, 0] = torch.tensor([1.0, 0.0, 0.0])
    hidden_control[1, 1] = torch.tensor([0.0, 1.0, 0.0])
    hidden_control[1, 4] = torch.tensor([-1.0, 0.0, 0.0])
    hidden_control[1, 6] = torch.tensor([1.0, 0.0, 0.0])

    result = compute_needle_sensitive_tokens(
        hidden,
        hidden_control,
        input_ids,
        layer_indices=[1],
        layer_labels=[12],
        expanded_segments=[{"start": 2, "expanded_end": 4}],
        insertion_position=5,
        offset=1,
        top_m=3,
        decode_token=lambda ids: f"tok-{ids[0]}",
    )

    tokens = result[0]["tokens"]
    assert result[0]["layer"] == 12
    assert [token["position"] for token in tokens] == [4, 1, 5]
    assert [token["control_position"] for token in tokens] == [4, 1, 6]
    assert [token["token"] for token in tokens] == ["tok-14", "tok-11", "tok-15"]
    assert tokens[0]["cosine_similarity"] < tokens[1]["cosine_similarity"]
    assert tokens[1]["cosine_similarity"] < tokens[2]["cosine_similarity"]


def test_save_model_input_ids_table_writes_requested_format(tmp_path: Path) -> None:
    path = save_model_input_ids_table(
        [
            {
                "sample_idx": 0,
                "uncontrolled_input_ids": [1, 2, 3],
                "controlled_input_ids": [1, 9, 3],
            },
            {
                "sample_idx": 1,
                "uncontrolled_input_ids": [4],
                "controlled_input_ids": [5],
            },
        ],
        tmp_path / "tables",
    )

    text = path.read_text(encoding="utf-8")
    assert path == tmp_path / "tables" / "model_input_ids.txt"
    assert (
        "Example ID 0\nuncontrolled input ids\n1 2 3\ncontrolled input ids\n1 9 3"
        in text
    )
    assert "Example ID 1\nuncontrolled input ids\n4\ncontrolled input ids\n5" in text


def test_save_needle_sensitive_outputs_groups_by_layer(tmp_path: Path) -> None:
    records = [
        {
            "sample_idx": 0,
            "layers": [
                {
                    "layer": 2,
                    "tokens": [
                        {
                            "token": "a",
                            "token_id": 1,
                            "position": 4,
                            "control_position": 4,
                            "cosine_similarity": 0.3,
                        }
                    ],
                }
            ],
        },
        {
            "sample_idx": 1,
            "layers": [
                {
                    "layer": 2,
                    "tokens": [
                        {
                            "token": "b",
                            "token_id": 2,
                            "position": 5,
                            "control_position": 6,
                            "cosine_similarity": 0.1,
                        }
                    ],
                }
            ],
        },
    ]

    json_path, txt_path = save_needle_sensitive_outputs(records, tmp_path / "tables")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert [row["token"] for row in payload["by_layer"]["2"]] == ["b", "a"]
    assert "Layer 2" in txt_path.read_text(encoding="utf-8")
