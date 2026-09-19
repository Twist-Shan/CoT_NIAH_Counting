from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


HookFn = Callable[[str, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class AdditiveSteeringSpec:
    hook_name: str
    positions: list[int]
    direction: torch.Tensor
    alpha: float = 1.0
    scale: float = 1.0


@dataclass(frozen=True)
class ReplacementPatchSpec:
    hook_name: str
    position: int
    donor_hidden: torch.Tensor


def make_additive_hook(spec: AdditiveSteeringSpec) -> HookFn:
    direction = spec.direction

    def hook(name: str, hidden: torch.Tensor) -> torch.Tensor:
        if name != spec.hook_name:
            return hidden
        out = hidden.clone()
        vec = direction.to(device=out.device, dtype=out.dtype) * float(spec.alpha) * float(spec.scale)
        for pos in spec.positions:
            if 0 <= int(pos) < out.size(1):
                out[:, int(pos), :] = out[:, int(pos), :] + vec
        return out

    return hook


def make_replacement_hook(spec: ReplacementPatchSpec) -> HookFn:
    donor = spec.donor_hidden

    def hook(name: str, hidden: torch.Tensor) -> torch.Tensor:
        if name != spec.hook_name or not (0 <= int(spec.position) < hidden.size(1)):
            return hidden
        out = hidden.clone()
        donor_vec = donor.to(device=out.device, dtype=out.dtype).view(-1)
        out[:, int(spec.position), :] = donor_vec
        return out

    return hook


def compose_hooks(*hooks: HookFn | None) -> HookFn | None:
    active = [hook for hook in hooks if hook is not None]
    if not active:
        return None

    def composed(name: str, hidden: torch.Tensor) -> torch.Tensor:
        out = hidden
        for hook in active:
            out = hook(name, out)
        return out

    return composed
