from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from realistic_niah_v4.modeling import DecoderAdapter
from realistic_niah_v4.prompts import PromptEncoding
from realistic_niah_v4_4_3.interventions import QueryBundle, head_output_from_z, head_z

from .gemma_cross_layer_spec import FrozenSite
from .interventions import _orthogonal_equal_output_norm_delta
from .upstream_path import PathCausalOutput, run_path_intervention


def site_label(site: FrozenSite) -> str:
    return f"L{int(site.layer)}H{int(site.head)}"


def site_set_label(sites: Sequence[FrozenSite]) -> str:
    return ",".join(site_label(site) for site in sites)


def site_set_id(role: str, sites: Sequence[FrozenSite]) -> str:
    return f"{role}_" + "_".join(site_label(site) for site in sites)


def frozen_selection(
    candidate: Sequence[FrozenSite],
    controls: Sequence[Sequence[FrozenSite]],
) -> dict[str, Any]:
    candidate_entry = {
        "set_id": site_set_id("candidate_core", candidate),
        "set_role": "candidate_core",
        "sites": [site.to_list() for site in candidate],
        "heads": [site_label(site) for site in candidate],
    }
    control_entries = [
        {
            "set_id": site_set_id("matched_control", sites),
            "set_role": "matched_control",
            "sites": [site.to_list() for site in sites],
            "heads": [site_label(site) for site in sites],
        }
        for sites in controls
    ]
    payload = {
        "schema_version": "realistic_niah_v4_4_4_cross_layer_selection_v1",
        "candidate": candidate_entry,
        "matched_controls": control_entries,
        "factorial_components": [],
        "registered_nested_sets": [],
        "selection_uses_causal_outcomes": False,
        "selection_source": (
            "frozen correct-only K2 ranked set and three layer-matched random sets"
        ),
    }
    encoded = repr(payload).encode("utf-8")
    payload["selection_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def unique_sites(site_sets: Sequence[Sequence[FrozenSite]]) -> tuple[FrozenSite, ...]:
    return tuple(sorted({site for sites in site_sets for site in sites}))


def stacked_site_z(
    bundle: QueryBundle,
    adapter: DecoderAdapter,
    sites: Sequence[FrozenSite],
) -> tuple[torch.Tensor, ...]:
    return tuple(
        head_z(
            bundle,
            adapter,
            layer=int(site.layer),
            head=int(site.head),
        )
        .detach()
        .float()
        .cpu()
        for site in sites
    )


def fit_site_intercept_and_slope(
    values: Sequence[torch.Tensor], counts: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.stack([value.detach().float().cpu() for value in values], dim=0)
    count_tensor = torch.as_tensor(counts, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[0] != len(count_tensor):
        raise ValueError("Site fit received incompatible values/counts")
    centered = count_tensor - count_tensor.mean()
    denominator = torch.sum(centered.square())
    if float(denominator) <= 0:
        raise ValueError("Site fit needs at least two distinct counts")
    slope = torch.einsum("n,nd->d", centered, tensor) / denominator
    intercept = tensor.mean(dim=0) - count_tensor.mean() * slope
    return intercept, slope


def site_output(
    adapter: DecoderAdapter, site: FrozenSite, z: torch.Tensor
) -> torch.Tensor:
    return head_output_from_z(
        adapter,
        layer=int(site.layer),
        head=int(site.head),
        z=z.detach().float().cpu(),
    )


def joint_output(
    adapter: DecoderAdapter,
    sites: Sequence[FrozenSite],
    values: Mapping[FrozenSite, torch.Tensor],
) -> torch.Tensor:
    outputs = [site_output(adapter, site, values[site]) for site in sites]
    if not outputs:
        raise ValueError("A joint output needs at least one site")
    return torch.stack(outputs, dim=0).sum(dim=0)


def joint_natural_carrier(
    adapter: DecoderAdapter,
    bundle: QueryBundle,
    *,
    sites: Sequence[FrozenSite],
    centers: Mapping[FrozenSite, torch.Tensor],
    slopes: Mapping[FrozenSite, torch.Tensor],
) -> dict[str, float]:
    actual = {
        site: head_z(
            bundle, adapter, layer=int(site.layer), head=int(site.head)
        ).detach().float().cpu()
        for site in sites
    }
    centered = {site: actual[site] - centers[site] for site in sites}
    centered_output = joint_output(adapter, sites, centered)
    step_output = joint_output(adapter, sites, slopes)
    step_norm = torch.linalg.vector_norm(step_output)
    if not torch.isfinite(step_norm) or float(step_norm) <= 1e-8:
        raise RuntimeError("Cross-layer natural output step is degenerate")
    unit = step_output / step_norm
    projection = float(torch.dot(centered_output, unit))
    return {
        "natural_carrier_output_projection": projection,
        "natural_carrier_coefficient": projection / float(step_norm),
        "natural_centered_set_output_norm": float(
            torch.linalg.vector_norm(centered_output)
        ),
        "natural_output_step_norm": float(step_norm),
    }


def _local_axis_delta(
    adapter: DecoderAdapter,
    *,
    site: FrozenSite,
    source_z: torch.Tensor,
    reference_z: torch.Tensor,
    slope: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    step_output = site_output(adapter, site, slope)
    step_norm = torch.linalg.vector_norm(step_output)
    if not torch.isfinite(step_norm) or float(step_norm) <= 1e-8:
        raise RuntimeError(f"Natural step is degenerate at {site_label(site)}")
    source_output = site_output(adapter, site, source_z - reference_z)
    coefficient = float(torch.dot(source_output, step_output / step_norm))
    return -coefficient / float(step_norm) * slope.detach().float().cpu(), coefficient


def joint_centered_removal_and_control(
    adapter: DecoderAdapter,
    bundle: QueryBundle,
    *,
    sites: Sequence[FrozenSite],
    centers: Mapping[FrozenSite, torch.Tensor],
    slopes: Mapping[FrozenSite, torch.Tensor],
    label: str,
) -> tuple[
    dict[tuple[int, int], torch.Tensor],
    dict[tuple[int, int], torch.Tensor],
    dict[str, float],
]:
    removal: dict[tuple[int, int], torch.Tensor] = {}
    control: dict[tuple[int, int], torch.Tensor] = {}
    removal_norms = []
    control_norms = []
    coefficients = []
    for site in sites:
        actual = head_z(
            bundle, adapter, layer=int(site.layer), head=int(site.head)
        ).detach().float().cpu()
        delta, coefficient = _local_axis_delta(
            adapter,
            site=site,
            source_z=actual,
            reference_z=centers[site],
            slope=slopes[site],
        )
        local_control, local_output = _orthogonal_equal_output_norm_delta(
            adapter,
            layer=int(site.layer),
            heads=(int(site.head),),
            z_count_steps=slopes[site][None, :],
            target_output_norm=abs(coefficient),
            label=f"{label}:{site_label(site)}",
        )
        key = (int(site.layer), int(site.head))
        removal[key] = delta
        control[key] = local_control[0]
        removal_norms.append(
            float(torch.linalg.vector_norm(site_output(adapter, site, delta)))
        )
        control_norms.append(float(torch.linalg.vector_norm(local_output)))
        coefficients.append(coefficient)
    return removal, control, {
        "joint_axis_coefficient_l1": float(sum(abs(value) for value in coefficients)),
        "joint_removal_output_norm_l1": float(sum(removal_norms)),
        "joint_control_output_norm_l1": float(sum(control_norms)),
        "joint_output_norm_mismatch_max_abs": float(
            max(abs(left - right) for left, right in zip(removal_norms, control_norms))
        ),
    }


def joint_donor_patch_conditions(
    adapter: DecoderAdapter,
    receiver: QueryBundle,
    donor: QueryBundle,
    *,
    sites: Sequence[FrozenSite],
    slopes: Mapping[FrozenSite, torch.Tensor],
    label: str,
) -> tuple[
    dict[tuple[int, int], torch.Tensor],
    dict[tuple[int, int], torch.Tensor],
    dict[tuple[int, int], torch.Tensor],
    dict[str, float],
]:
    patch: dict[tuple[int, int], torch.Tensor] = {}
    blocked: dict[tuple[int, int], torch.Tensor] = {}
    controlled: dict[tuple[int, int], torch.Tensor] = {}
    removed_norms = []
    control_norms = []
    for site in sites:
        receiver_z = head_z(
            receiver, adapter, layer=int(site.layer), head=int(site.head)
        ).detach().float().cpu()
        donor_z = head_z(
            donor, adapter, layer=int(site.layer), head=int(site.head)
        ).detach().float().cpu()
        block, coefficient = _local_axis_delta(
            adapter,
            site=site,
            source_z=donor_z,
            reference_z=receiver_z,
            slope=slopes[site],
        )
        local_control, local_output = _orthogonal_equal_output_norm_delta(
            adapter,
            layer=int(site.layer),
            heads=(int(site.head),),
            z_count_steps=slopes[site][None, :],
            target_output_norm=abs(coefficient),
            label=f"{label}:{site_label(site)}",
        )
        key = (int(site.layer), int(site.head))
        patch[key] = donor_z
        blocked[key] = donor_z + block
        controlled[key] = donor_z + local_control[0]
        removed_norms.append(
            float(torch.linalg.vector_norm(site_output(adapter, site, block)))
        )
        control_norms.append(float(torch.linalg.vector_norm(local_output)))
    return patch, blocked, controlled, {
        "joint_patch_axis_removed_output_norm_l1": float(sum(removed_norms)),
        "joint_patch_control_output_norm_l1": float(sum(control_norms)),
        "joint_patch_norm_mismatch_max_abs": float(
            max(abs(left - right) for left, right in zip(removed_norms, control_norms))
        ),
    }


@torch.inference_mode()
def run_joint_query_intervention(
    model: nn.Module,
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    terminal_layer: int,
    deltas: Mapping[tuple[int, int], torch.Tensor] | None = None,
    replacements: Mapping[tuple[int, int], torch.Tensor] | None = None,
) -> PathCausalOutput:
    return run_path_intervention(
        model,
        adapter,
        encoding,
        mediator_layer=int(terminal_layer),
        query_deltas=deltas,
        query_replacements=replacements,
    )
