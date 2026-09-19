from datetime import datetime
from pathlib import Path
import zipfile

from dataset_generation import run_utils
from dataset_generation.run_utils import (
    archive_directory,
    build_run_name,
    create_run_paths,
    is_google_drive_path,
    tee_output,
)


def test_build_run_name_sanitizes_model_and_params() -> None:
    name = build_run_name(
        model_name="Qwen/Qwen3-8B",
        params={"task": "argmax", "prompt": "easier", "len": 1000},
        start_time=datetime(2026, 5, 28, 14, 30, 15),
    )

    assert (
        name == "run_20260528_143015_Qwen_Qwen3-8B_task-argmax_prompt-easier_len-1000"
    )
    assert "/" not in name


def test_create_run_paths_uses_tensors_directory(tmp_path) -> None:
    paths = create_run_paths(
        results_root=tmp_path,
        model_name="simple",
        params={"task": "argmax"},
        start_time=datetime(2026, 5, 28, 14, 30, 15),
    )

    assert paths.run_dir.exists()
    assert paths.figures_dir.exists()
    assert paths.tensors_dir.exists()
    assert paths.tensors_dir.name == "tensors"
    assert paths.generate_data_dir.exists()
    assert paths.logs_path.name == "logs.txt"


def test_tee_output_writes_stdout_and_stderr(tmp_path, capsys) -> None:
    log_path = tmp_path / "logs.txt"
    with tee_output(log_path):
        print("hello log")

    captured = capsys.readouterr()
    assert "hello log" in captured.out
    assert "hello log" in log_path.read_text(encoding="utf-8")


def test_create_run_paths_includes_prediction_outputs(tmp_path) -> None:
    paths = create_run_paths(
        results_root=tmp_path,
        model_name="simple",
        params={"task": "count_avg"},
        start_time=datetime(2026, 5, 28, 14, 30, 15),
    )

    assert paths.tables_dir.exists()
    assert paths.predictions_path == paths.tables_dir / "predictions.jsonl"
    assert paths.metrics_path == paths.tables_dir / "metrics.json"


def test_is_google_drive_path_detects_content_drive() -> None:
    assert is_google_drive_path("/content/drive/MyDrive/results")
    assert not is_google_drive_path("/content/dataset_generation_results")


def test_archive_directory_moves_one_zip(tmp_path) -> None:
    source = tmp_path / "run"
    nested = source / "tables"
    nested.mkdir(parents=True)
    (source / "logs.txt").write_text("hello", encoding="utf-8")
    (nested / "metrics.json").write_text("{}", encoding="utf-8")
    dest = tmp_path / "drive"

    archive = archive_directory(source, dest)

    assert archive == dest / "run.zip"
    assert archive.exists()
    assert not (source.parent / "_archives" / "run.zip").exists()
    assert sorted(p.name for p in dest.iterdir()) == ["run.zip"]
    with zipfile.ZipFile(archive) as zf:
        assert sorted(name for name in zf.namelist() if not name.endswith("/")) == [
            "run/logs.txt",
            "run/tables/metrics.json",
        ]


def test_archive_directory_can_zip_contents_directly(tmp_path) -> None:
    source = tmp_path / "run"
    nested = source / "tables"
    nested.mkdir(parents=True)
    (source / "logs.txt").write_text("hello", encoding="utf-8")
    (nested / "metrics.json").write_text("{}", encoding="utf-8")
    dest = tmp_path / "results" / "run"

    archive = archive_directory(
        source, dest, archive_name="run", include_source_dir=False
    )

    assert archive == dest / "run.zip"
    assert archive.exists()
    with zipfile.ZipFile(archive) as zf:
        assert sorted(name for name in zf.namelist() if not name.endswith("/")) == [
            "logs.txt",
            "tables/metrics.json",
        ]


def test_create_run_paths_localizes_relative_colab_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(run_utils, "is_colab_runtime", lambda: True)
    monkeypatch.setattr(run_utils, "is_local_content_path", lambda path: False)
    local_root = tmp_path / "content_runtime"
    monkeypatch.setattr(run_utils, "COLAB_LOCAL_RUNTIME_ROOT", local_root)

    paths = run_utils.create_run_paths(
        results_root="results/colab_hidden_states",
        model_name="simple",
        params={"task": "argmax"},
        start_time=datetime(2026, 5, 28, 14, 30, 15),
    )

    assert paths.run_dir.is_relative_to(local_root)
    assert paths.run_dir.exists()
