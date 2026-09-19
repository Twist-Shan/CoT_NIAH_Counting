from __future__ import annotations

from pathlib import Path
from typing import Iterable

from synthetic_counting_v20.pipeline import run_v20_pipeline

from .config import V21Config


def run_v21_pipeline(
    cfg: V21Config,
    *,
    stage: str | Iterable[str] = "all",
    out_root: str | Path = "runs/synthetic_counting_v21",
    run_name: str | None = None,
    checkpoint_sync_root: str | Path | None = None,
    skip_completed: bool = True,
) -> Path:
    if cfg.version != "v21" or cfg.count_tokenization != "digitwise":
        raise ValueError("run_v21_pipeline requires the v21 digit-wise config")
    return run_v20_pipeline(
        cfg,
        stage=stage,
        out_root=out_root,
        run_name=run_name,
        checkpoint_sync_root=checkpoint_sync_root,
        skip_completed=skip_completed,
    )


__all__ = ["run_v21_pipeline"]
