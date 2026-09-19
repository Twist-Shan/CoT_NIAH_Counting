import json
import zipfile
from pathlib import Path

from single_example.single_example_analysis import (
    DEFAULT_DATASET_PATH,
    cleanup_large_tensor_artifacts,
    build_single_example_config,
    create_single_example_run,
    load_jsonl_example,
    locate_uncontrolled_needle_segments,
    prepare_single_example_dataset,
    resolve_niah_example_dataset_path,
    run_single_example_qk_outlier_analysis,
    save_input_metadata,
    zip_single_example_results,
)
from dataset_generation.hidden_state_analysis import (
    build_uncontrolled_needle_insertions,
)


def test_resolve_niah_example_dataset_path_defaults_to_root_dataset() -> None:
    dataset_path = resolve_niah_example_dataset_path()

    assert dataset_path == DEFAULT_DATASET_PATH
    assert dataset_path.exists()


def test_resolve_niah_example_dataset_path_uses_named_run_folder() -> None:
    dataset_path = resolve_niah_example_dataset_path(
        "Qwen_Qwen3-8B_task-count_avg_prompt-vanilla_len-1000_needles-3"
    )

    assert dataset_path == Path(
        "data/niah-example/"
        "Qwen_Qwen3-8B_task-count_avg_prompt-vanilla_len-1000_needles-3/"
        "dynamic_niah_v2.jsonl"
    )
    assert dataset_path.exists()


def test_resolve_niah_example_dataset_path_errors_for_missing_run() -> None:
    missing_run = "does-not-exist"

    try:
        resolve_niah_example_dataset_path(missing_run)
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing dataset run")

    assert missing_run in message
    assert "dynamic_niah_v2.jsonl" in message


def test_build_single_example_config_uses_count_avg_alternative_dataset() -> None:
    dataset_path = resolve_niah_example_dataset_path(
        "Qwen_Qwen3-8B_task-count_avg_prompt-vanilla_len-1000_needles-3"
    )
    row, rows = load_jsonl_example(dataset_path, 0)

    cfg = build_single_example_config(
        row, dataset_path=dataset_path, model_name="Qwen/Qwen3-8B"
    )

    assert len(rows) == 20
    assert cfg.task_type == "count_avg"
    assert cfg.prompt_style == "vanilla"
    assert cfg.insertion_positions == (100, 200, 400)
    assert cfg.num_needles == len(row["needles"]) == 3


def test_load_jsonl_example_uses_zero_based_index() -> None:
    row, rows = load_jsonl_example(DEFAULT_DATASET_PATH, 0)

    assert len(rows) >= 1
    assert row["id"] == rows[0]["id"]
    assert "messages" in row
    assert "uncontrolled_messages" in row


def test_prepare_single_example_dataset_writes_generate_data_and_metadata(
    tmp_path: Path,
) -> None:
    row, _ = load_jsonl_example(DEFAULT_DATASET_PATH, 0)
    paths = create_single_example_run(
        row=row,
        example_id=0,
        model_name="Qwen/Qwen3-8B",
        run_root=tmp_path,
        user_run_name="run_single_example_test",
    )

    dataset_copy, cfg = prepare_single_example_dataset(
        row=row,
        example_id=0,
        dataset_path=DEFAULT_DATASET_PATH,
        paths=paths,
        model_name="Qwen/Qwen3-8B",
    )

    assert dataset_copy == paths.generate_data_dir / "dynamic_niah_v2.jsonl"
    assert dataset_copy.exists()
    assert json.loads(dataset_copy.read_text(encoding="utf-8"))["id"] == row["id"]
    assert cfg.num_examples == 1
    assert cfg.data_save_path == str(dataset_copy)
    assert paths.metadata_path.exists()
    assert paths.analyze_config_path.exists()
    metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    assert metadata["resolved_config"]["data_save_path"] == str(dataset_copy)


def test_locate_uncontrolled_needle_segments_validates_three_consecutive_spans() -> (
    None
):
    row, _ = load_jsonl_example(DEFAULT_DATASET_PATH, 0)
    insertions = build_uncontrolled_needle_insertions(
        row["realized_insertions"], row["needles"]
    )
    input_ids: list[int] = [101, 102]
    expected_starts: list[int] = []
    for insertion in insertions:
        expected_starts.append(len(input_ids))
        input_ids.extend(int(token) for token in insertion["tokens"])
        input_ids.extend([9000 + len(expected_starts)])

    segments = locate_uncontrolled_needle_segments(
        row=row,
        uncontrolled_input_ids=input_ids,
        expected_num_needles=3,
    )

    assert [segment["start"] for segment in segments] == expected_starts
    assert len(segments) == 3
    for segment in segments:
        assert segment["positions"] == list(range(segment["start"], segment["end"]))
        assert segment["length"] == segment["end"] - segment["start"]


def test_save_input_metadata_and_cleanup_large_tensor_artifacts(tmp_path: Path) -> None:
    row, _ = load_jsonl_example(DEFAULT_DATASET_PATH, 0)
    paths = create_single_example_run(
        row=row,
        example_id=2,
        model_name="Qwen/Qwen3-8B",
        run_root=tmp_path,
        user_run_name="run_cleanup_test",
    )
    hidden_path = paths.tensors_dir / "hidden_inputs_2.pt"
    hidden_path.write_bytes(b"hidden")
    representation_tensors_dir = paths.tensors_dir / "ablation_representation"
    representation_tensors_dir.mkdir()
    stats_path = representation_tensors_dir / "hidden_state_distribution_stats.pt"
    stats_path.write_bytes(b"stats")
    unablated_path = representation_tensors_dir / "hidden_states_unablated_2.pt"
    unablated_path.write_bytes(b"unablated")
    qk_cache = paths.tensors_dir / "qk_cache"
    qk_cache.mkdir()
    (qk_cache / "layer.pt").write_bytes(b"qk")

    metadata_path = save_input_metadata(
        path=paths.generate_data_dir / "inputs_2.json",
        example_id=2,
        row=row,
        model_name="Qwen/Qwen3-8B",
        uncontrolled_input_ids=[1, 2, 3],
        controlled_input_ids=[4, 5],
        needle_segments=[{"needle_id": "N1", "start": 1, "end": 2, "positions": [1]}],
    )
    removed = cleanup_large_tensor_artifacts(paths, example_id=2)

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "single_example_inputs_v1"
    assert payload["uncontrolled_input_ids"] == [1, 2, 3]
    assert hidden_path in removed
    assert stats_path in removed
    assert unablated_path in removed
    assert qk_cache in removed
    assert not hidden_path.exists()
    assert not stats_path.exists()
    assert not unablated_path.exists()
    assert not qk_cache.exists()


def test_zip_single_example_results_removes_representation_intermediates(
    tmp_path: Path,
) -> None:
    row, _ = load_jsonl_example(DEFAULT_DATASET_PATH, 0)
    paths = create_single_example_run(
        row=row,
        example_id=0,
        model_name="Qwen/Qwen3-8B",
        run_root=tmp_path,
        user_run_name="run_zip_cleanup_test",
    )
    keep_path = paths.tables_dir / "summary.json"
    keep_path.write_text("{}", encoding="utf-8")
    representation_tensors_dir = paths.tensors_dir / "ablation_representation"
    representation_tensors_dir.mkdir()
    stats_path = representation_tensors_dir / "hidden_state_distribution_stats.pt"
    stats_path.write_bytes(b"stats")
    unablated_path = representation_tensors_dir / "hidden_states_unablated_0.pt"
    unablated_path.write_bytes(b"unablated")

    archive_path = zip_single_example_results(paths=paths, results_path=tmp_path / "out")

    assert not stats_path.exists()
    assert not unablated_path.exists()
    with zipfile.ZipFile(archive_path) as zf:
        archived_files = {name for name in zf.namelist() if not name.endswith("/")}
    assert "run_zip_cleanup_test/tables/summary.json" in archived_files
    assert (
        "run_zip_cleanup_test/tensors/ablation_representation/"
        "hidden_state_distribution_stats.pt"
        not in archived_files
    )
    assert (
        "run_zip_cleanup_test/tensors/ablation_representation/"
        "hidden_states_unablated_0.pt"
        not in archived_files
    )


def test_run_single_example_qk_outlier_analysis_preserves_selected_example_id(
    tmp_path: Path, monkeypatch
) -> None:
    row, _ = load_jsonl_example(DEFAULT_DATASET_PATH, 0)
    paths = create_single_example_run(
        row=row,
        example_id=1,
        model_name="Qwen/Qwen3-8B",
        run_root=tmp_path,
        user_run_name="run_qk_id_test",
    )
    captured: dict[str, object] = {}

    def fake_run_qk_outlier_analysis(**kwargs):
        captured.update(kwargs)
        return {"figure_paths": [str(paths.figures_dir / "inputs_1_qk_outliers.png")]}

    monkeypatch.setattr(
        "single_example.single_example_analysis.run_qk_outlier_analysis",
        fake_run_qk_outlier_analysis,
    )

    summary = run_single_example_qk_outlier_analysis(
        paths=paths,
        layers=[4, 8],
        example_id=1,
        model_name="Qwen/Qwen3-8B",
        repo_root=tmp_path,
    )

    assert captured["example_indices"] == [1]
    assert captured["layers"] == [4, 8]
    assert summary["figure_paths"] == [
        str(paths.figures_dir / "inputs_1_qk_outliers.png")
    ]
