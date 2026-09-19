from __future__ import annotations

from typing import Any
import sys
import importlib.util as importlib_util

import torch
import torch.nn.functional as F

for _optional_module in ("sklearn", "scipy"):
    _module = sys.modules.get(_optional_module)
    if _module is not None and getattr(_module, "__spec__", None) is None:
        del sys.modules[_optional_module]

_original_find_spec = importlib_util.find_spec


def _find_spec_without_sklearn(name: str, *args: Any, **kwargs: Any):
    if name in {"sklearn", "scipy"} or name.startswith(("sklearn.", "scipy.")):
        return None
    return _original_find_spec(name, *args, **kwargs)


importlib_util.find_spec = _find_spec_without_sklearn
try:
    from transformers import GPT2Config, GPT2LMHeadModel
finally:
    importlib_util.find_spec = _original_find_spec


def make_gpt2_config(config: dict[str, Any]) -> GPT2Config:
    cfg = dict(config)
    vocab_size = int(cfg.pop("vocab_size"))
    bos_token_id = int(cfg.pop("bos_token_id"))
    eos_token_id = int(cfg.pop("eos_token_id"))
    pad_token_id = int(cfg.pop("pad_token_id", eos_token_id))
    cfg.setdefault("attn_implementation", "eager")
    return GPT2Config(
        vocab_size=vocab_size,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        **cfg,
    )


def make_model(config: dict[str, Any], device: str | torch.device) -> GPT2LMHeadModel:
    """Build a v2-style random-init GPT-2 LM with learned absolute positions."""

    model = GPT2LMHeadModel(make_gpt2_config(config))
    return model.to(device)


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab_size = shift_logits.size(-1)
    ce = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(shift_labels)
    active = shift_labels.ne(ignore_index)
    loss = (ce * active).sum() / active.sum().clamp_min(1)
    return loss, ce.detach()
