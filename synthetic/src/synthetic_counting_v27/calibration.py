from __future__ import annotations

from pathlib import Path

from synthetic_counting_v24_8.readout_tail import CandidateSpec, run_readout_tail


def run_v27_calibration(
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
    """Calibrate the tied atomic-number rows of both v24.3 models.

    The source backbone, trace grammar, and tied parameterization are kept
    intact.  Because atomic answer tokens never occur in the causal prefix of
    an answer query, the selected shared rows receive unembedding-only
    gradients from the final-count objective.
    """

    return run_readout_tail(
        source_run,
        output_dir,
        device=device,
        batch_size=batch_size,
        eval_every=eval_every,
        validation_per_count=validation_per_count,
        seed=seed,
        candidates=candidates,
        experiment="v27",
        expected_source_version="v24.3",
        readout_mode="tied_unembedding",
    )


__all__ = ["run_v27_calibration"]
