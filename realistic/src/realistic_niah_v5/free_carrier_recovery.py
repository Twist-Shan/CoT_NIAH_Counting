"""Fixed-schedule carrier intervention during an otherwise free continuation."""
from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Mapping, Sequence
import torch
from realistic_niah_v4.modeling import _replace_output_tensor, _tensor_from_output


def match_vector_norms(control: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if control.shape != reference.shape or control.ndim != 2:
        raise ValueError('Expected equal [positions, hidden] tensors')
    source = control.float()
    source_norm = source.norm(dim=-1, keepdim=True)
    target_norm = reference.float().norm(dim=-1, keepdim=True)
    if not torch.isfinite(source).all() or not torch.isfinite(reference).all() or (source_norm <= 0).any():
        raise ValueError('Non-finite or zero matched-control state')
    return (source * (target_norm / source_norm)).to(reference.dtype)


@contextmanager
def scheduled_carrier_clamp(adapter: Any, *, prefix_length: int,
                            positions: Sequence[int], replacements: Mapping[int, torch.Tensor]):
    """Patch post-block states at frozen absolute positions in cached decoding.

    A missing future position after EOS is recorded, not treated as exclusion.
    Full prefill is followed by single-token cached forwards; any other shape
    is an error. Each decoder layer maintains an independent absolute cursor.
    """
    positions = tuple(int(p) for p in positions)
    if not positions or tuple(sorted(set(positions))) != positions or positions[0] < prefix_length:
        raise ValueError('Carrier positions must be unique, increasing and after the prefix')
    applications = {int(layer): [] for layer in replacements}
    cursors = {int(layer): 0 for layer in replacements}
    realized = {int(layer): [] for layer in replacements}
    handles = []
    audit = {'applications': applications, 'realized_rms': realized,
             'positions': list(positions), 'prefix_length': prefix_length}
    try:
        for raw_layer, states in sorted(replacements.items()):
            layer = int(raw_layer)
            if states.ndim != 2 or len(states) != len(positions) or not torch.isfinite(states).all():
                raise ValueError('Invalid carrier replacement tensor')
            def hook(_module, _args, output, *, layer=layer, states=states):
                hidden = _tensor_from_output(output)
                if hidden.ndim != 3 or hidden.shape[0] != 1:
                    raise RuntimeError('Expected single-batch decoder output')
                width = int(hidden.shape[1])
                start = cursors[layer]
                if (start == 0 and width != prefix_length) or (start > 0 and width != 1):
                    raise RuntimeError('Free generation changed the prefill/cached-decode contract')
                cursors[layer] += width
                active = [(i, p-start) for i,p in enumerate(positions) if start <= p < start+width]
                if not active:
                    return output
                patched = hidden.clone()
                for index, local in active:
                    target = states[index].to(device=hidden.device, dtype=hidden.dtype)
                    if target.shape != hidden[0,local].shape:
                        raise RuntimeError('Carrier hidden dimension changed')
                    delta = (hidden[0,local].float()-target.float()).square().mean().sqrt()
                    realized[layer].append(float(delta.cpu()))
                    patched[0,local] = target
                    applications[layer].append(positions[index])
                return _replace_output_tensor(output, patched)
            handles.append(adapter.layers[layer].register_forward_hook(hook))
        yield audit
    finally:
        for handle in handles:
            handle.remove()
        audit['visited_through_position'] = {layer:cursor-1 for layer,cursor in cursors.items()}
        audit['missing_positions'] = {layer:[p for p in positions if p not in seen]
                                      for layer,seen in applications.items()}
        for layer,seen in applications.items():
            if len(seen) != len(set(seen)):
                raise RuntimeError(f'Carrier replacement applied twice at layer {layer}')
