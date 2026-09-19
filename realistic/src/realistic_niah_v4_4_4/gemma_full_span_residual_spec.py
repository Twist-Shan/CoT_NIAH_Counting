from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .gemma_residual_spec import GemmaResidualConfig


SCHEMA_VERSION = "realistic_niah_v4_4_4_gemma_full_span_residual_k_sweep_v2"
PROTOCOL_VERSION = "realistic_niah_v4_4_4_gemma_full_span_ranked_residual_path_v2"
RANKING_SHA256 = "52646c9bec4d90336f469c46aecd8ffbaf54008648f7aef45f56215984744de7"
RANKING_FILTER = (
    "broad_aggregation full-span registry restricted to source layers below "
    "the first candidate residual mediator"
)


@dataclass(frozen=True)
class GemmaFullSpanResidualConfig(GemmaResidualConfig):
    """Generic, hash-locked Gemma residual path for a geometric full-span K grid."""

    schema_version: str = SCHEMA_VERSION
    protocol_version: str = PROTOCOL_VERSION
    mechanism_variant: str = "fullspan_k2"
    top_n: int = 2
    ranking_registry_sha256: str = RANKING_SHA256
    ranking_filter: str = RANKING_FILTER

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Unexpected Gemma full-span residual schema")
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("Unexpected Gemma full-span residual protocol")
        if self.model_label != "Gemma4-E4B":
            raise ValueError("Full-span residual sweep is model-locked to Gemma4-E4B")
        if self.top_n not in {1, 2, 4, 8, 16}:
            raise ValueError("Gemma residual K must be one of 1,2,4,8,16")
        if self.mechanism_variant != f"fullspan_k{self.top_n}":
            raise ValueError("mechanism_variant and top_n disagree")
        if self.ranking_registry_sha256 != RANKING_SHA256:
            raise ValueError("Gemma full-span ranking registry hash changed")
        if self.ranking_filter != RANKING_FILTER:
            raise ValueError("Gemma source-layer ranking filter changed")
        if len(self.candidate_sites) != self.top_n:
            raise ValueError("Candidate site count does not equal top_n")
        if len(set(self.candidate_sites)) != self.top_n:
            raise ValueError("Candidate sites must be unique")
        if len(self.matched_control_sets) != 3:
            raise ValueError("Exactly three layer-matched controls are required")
        expected_layers = Counter(site.layer for site in self.candidate_sites)
        for site_set in (self.candidate_sites, *self.matched_control_sets):
            if len(site_set) != self.top_n or len(set(site_set)) != self.top_n:
                raise ValueError("Every source set must contain top_n unique sites")
            if Counter(site.layer for site in site_set) != expected_layers:
                raise ValueError("A matched control changed the candidate layer composition")
            if any(not 0 <= site.head < 8 for site in site_set):
                raise ValueError("Gemma source heads must lie in [0,8)")
        if self.matched_control_sampling_seed != 4500 + self.top_n:
            raise ValueError("Matched-control rotation seed changed")
        if self.discovery_seeds != tuple(range(1630, 1640)):
            raise ValueError("Full-span residual discovery seeds changed")
        if self.confirmation_seeds != tuple(range(1640, 1660)):
            raise ValueError("Full-span residual confirmation seeds changed")
        if set(self.discovery_seeds) & set(self.confirmation_seeds):
            raise ValueError("Discovery and confirmation seeds overlap")
        if self.counts != tuple(range(1, 11)):
            raise ValueError("Counts must be exactly 1 through 10")
        if set(self.fit_counts) | set(self.heldout_counts) != set(self.counts):
            raise ValueError("Fit and held-out counts do not cover the count grid")
        if set(self.fit_counts) & set(self.heldout_counts):
            raise ValueError("Fit and held-out counts overlap")
        if not self.donor_pairs or any(
            receiver >= donor
            or receiver not in self.counts
            or donor not in self.counts
            for receiver, donor in self.donor_pairs
        ):
            raise ValueError("Residual donor pairs are invalid")
        if self.candidate_mediator_layers != (36, 37, 38, 39, 40):
            raise ValueError("Residual mediator registry changed")
        if max(site.layer for site in self.candidate_sites) >= min(
            self.candidate_mediator_layers
        ):
            raise ValueError("Every ranked source site must precede the mediator registry")
        if self.terminal_trace_layer != 41:
            raise ValueError("Terminal trace layer changed")
        if not 0 < self.primary_alpha <= 0.025:
            raise ValueError("Primary alpha must lie in (0,.025]")
        if self.bootstrap_repetitions < 10_000:
            raise ValueError("Full-span residual sweep needs 10,000 bootstraps")
        if not self.require_clean_necessity:
            raise ValueError("Full-span residual sweep requires clean-run necessity")
        if self.selection_status != "frozen_full_span_k_sweep_before_outcomes":
            raise ValueError("Selection status changed")
        if not self.selection_source.startswith("hash-locked full-span broad-aggregation"):
            raise ValueError("Selection source no longer identifies full-span ranking")
        if self.write_raw_attention_rows or self.write_full_hidden_states:
            raise ValueError("Raw tensors must not be persisted")
        if not self.atomic_shards:
            raise ValueError("Atomic shards are required")
