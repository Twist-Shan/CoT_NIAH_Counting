from __future__ import annotations

from pathlib import Path

from synthetic_counting_v24_8.readout_tail import CandidateSpec, run_readout_tail


def run_v26_1_calibration(
    source_run: str | Path,
    output_dir: str | Path,
    *,
    device: str,
    batch_size: int = 128,
    eval_every: int = 100,
    validation_per_count: int = 10,
    seed: int = 2478,
    candidates: tuple[CandidateSpec, ...] | None = None,
) -> Path:
    """Calibrate both native number heads without changing either backbone."""

    return run_readout_tail(
        source_run,
        output_dir,
        device=device,
        batch_size=batch_size,
        eval_every=eval_every,
        validation_per_count=validation_per_count,
        seed=seed,
        candidates=candidates,
        experiment="v26.1",
        expected_source_version="v26",
    )


__all__ = ["run_v26_1_calibration"]
