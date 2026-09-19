"""v24.8: native LM-head readout calibration on top of v24.7."""

from .readout_tail import (
    CandidateSpec,
    GateSummary,
    default_candidate_specs,
    summarize_gate,
)

__all__ = [
    "CandidateSpec",
    "GateSummary",
    "default_candidate_specs",
    "summarize_gate",
]
