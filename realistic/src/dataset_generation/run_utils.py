from __future__ import annotations

import contextlib
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, TextIO

GOOGLE_DRIVE_ROOT = Path("/content/drive")
COLAB_LOCAL_RUNTIME_ROOT = Path(
    os.environ.get(
        "DATASET_GENERATION_LOCAL_RUNTIME_ROOT", "/content/dataset_generation_runtime"
    )
)


def _absolute_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def is_colab_runtime() -> bool:
    """Return True when running inside a Colab-style /content runtime."""
    return Path("/content").exists() and _is_relative_to(
        _absolute_path(Path.cwd()), Path("/content")
    )


def is_google_drive_path(path: str | Path) -> bool:
    """Return True when a path resolves under Colab's mounted Google Drive."""
    return _is_relative_to(_absolute_path(path), GOOGLE_DRIVE_ROOT)


def is_local_content_path(path: str | Path) -> bool:
    """Return True when a path is local to /content and not under /content/drive."""
    absolute = _absolute_path(path)
    return _is_relative_to(absolute, Path("/content")) and not _is_relative_to(
        absolute, GOOGLE_DRIVE_ROOT
    )


def localize_runtime_path(
    path: str | Path, *, local_root: str | Path | None = None
) -> Path:
    """Map Colab runtime outputs to /content instead of a Drive-backed location.

    Relative paths are unsafe when the notebook checkout lives in /content/drive, because
    they would write under Drive. In Colab we map any non-/content-local path beneath
    DATASET_GENERATION_LOCAL_RUNTIME_ROOT (default: /content/dataset_generation_runtime).
    Outside Colab this returns the input path unchanged.
    """
    p = Path(path).expanduser()
    if not is_colab_runtime() or is_local_content_path(p):
        return p
    root = Path(local_root or COLAB_LOCAL_RUNTIME_ROOT).expanduser()
    if p.is_absolute():
        name = slugify(str(p).lstrip("/"), max_length=160)
        return root / "absolute_paths" / name
    return root / p


def archive_directory(
    source_dir: str | Path,
    destination_dir: str | Path,
    *,
    archive_format: str = "zip",
    archive_name: str | None = None,
    staging_dir: str | Path | None = None,
    include_source_dir: bool = True,
) -> Path:
    """Create one archive locally, then move that single file to destination_dir.

    This is intended for Colab + Google Drive workflows: thousands of runtime files
    remain under /content, and Drive only sees the final .zip or .tar.gz.
    """
    source = Path(source_dir)
    dest = Path(destination_dir)
    if not source.is_dir():
        raise NotADirectoryError(f"source_dir must be an existing directory: {source}")
    if archive_format not in {"zip", "gztar"}:
        raise ValueError("archive_format must be 'zip' or 'gztar'")

    stem = archive_name or source.name
    stage = (
        Path(staging_dir) if staging_dir is not None else source.parent / "_archives"
    )
    stage.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    base_name = stage / stem
    if include_source_dir:
        local_archive = Path(
            shutil.make_archive(
                str(base_name),
                archive_format,
                root_dir=source.parent,
                base_dir=source.name,
            )
        )
    else:
        local_archive = Path(
            shutil.make_archive(str(base_name), archive_format, root_dir=source)
        )
    final_path = dest / local_archive.name
    if final_path.exists():
        final_path.unlink()
    shutil.move(str(local_archive), final_path)
    return final_path


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    figures_dir: Path
    tensors_dir: Path
    generate_data_dir: Path
    tables_dir: Path
    logs_path: Path
    analyze_config_path: Path
    metadata_path: Path
    predictions_path: Path
    metrics_path: Path


class _TeeStream:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(
            getattr(stream, "isatty", lambda: False)() for stream in self.streams
        )


def slugify(value: Any, *, max_length: int = 80) -> str:
    text = str(value).strip()
    text = text.replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-_.")
    if not text:
        text = "none"
    return text[:max_length].strip("-_.") or "none"


def build_run_name(
    *,
    model_name: str,
    params: Mapping[str, Any],
    start_time: datetime | None = None,
) -> str:
    start = start_time or datetime.now()
    timestamp = start.strftime("%Y%m%d_%H%M%S")
    model_slug = slugify(model_name, max_length=80)
    param_parts = [
        f"{slugify(k, max_length=24)}-{slugify(v, max_length=40)}"
        for k, v in params.items()
    ]
    suffix = "_".join(param_parts)
    base = f"run_{timestamp}_{model_slug}"
    return f"{base}_{suffix}" if suffix else base


def create_run_paths(
    *,
    results_root: str | Path = "results",
    model_name: str,
    params: Mapping[str, Any],
    run_dir: str | Path | None = None,
    run_name: str | None = None,
    start_time: datetime | None = None,
) -> RunPaths:
    root = localize_runtime_path(results_root)
    if run_dir is None:
        resolved_run_name = run_name or build_run_name(
            model_name=model_name, params=params, start_time=start_time
        )
        run_dir_path = root / resolved_run_name
    else:
        run_dir_path = localize_runtime_path(run_dir)

    paths = RunPaths(
        run_dir=run_dir_path,
        figures_dir=run_dir_path / "figures",
        tensors_dir=run_dir_path / "tensors",
        generate_data_dir=run_dir_path / "generate_data",
        tables_dir=run_dir_path / "tables",
        logs_path=run_dir_path / "logs.txt",
        analyze_config_path=run_dir_path / "analyze_hidden_states_config.json",
        metadata_path=run_dir_path / "run_metadata.json",
        predictions_path=run_dir_path / "tables" / "predictions.jsonl",
        metrics_path=run_dir_path / "tables" / "metrics.json",
    )
    for path in (
        paths.run_dir,
        paths.figures_dir,
        paths.tensors_dir,
        paths.generate_data_dir,
        paths.tables_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return paths


@contextlib.contextmanager
def tee_output(logs_path: str | Path) -> Iterator[None]:
    path = Path(logs_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log_file:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = _TeeStream(old_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = _TeeStream(old_stderr, log_file)  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
