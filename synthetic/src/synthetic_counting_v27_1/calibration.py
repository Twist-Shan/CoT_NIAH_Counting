from __future__ import annotations

from pathlib import Path

from synthetic_counting_v24_8.readout_tail import CandidateSpec, run_readout_tail


def default_v27_1_candidates() -> tuple[CandidateSpec, ...]:
    """Pre-registered trace-safety sweep at v27's selected learning rate."""

    return (
        CandidateSpec("trace_safe_w0p1", 3e-4, 600, 25, 0.1),
        CandidateSpec("trace_safe_w0p3", 3e-4, 600, 25, 0.3),
        CandidateSpec("trace_safe_w1", 3e-4, 600, 25, 1.0),
    )


def run_v27_1_calibration(
    source_run: str | Path,
    output_dir: str | Path,
    *,
    device: str,
    batch_size: int = 128,
    eval_every: int = 50,
    validation_per_count: int = 10,
    seed: int = 2478,
    candidates: tuple[CandidateSpec, ...] | None = None,
) -> Path:
    """Calibrate v24.3 count rows while suppressing premature trace answers."""

    return run_readout_tail(
        source_run,
        output_dir,
        device=device,
        batch_size=batch_size,
        eval_every=eval_every,
        validation_per_count=validation_per_count,
        seed=seed,
        candidates=candidates or default_v27_1_candidates(),
        experiment="v27.1",
        expected_source_version="v24.3",
        readout_mode="tied_unembedding",
    )


__all__ = ["default_v27_1_candidates", "run_v27_1_calibration"]
