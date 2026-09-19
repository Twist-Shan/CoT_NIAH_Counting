from __future__ import annotations

import importlib.util as importlib_util
import sys
from typing import Any

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

from .config import V6Config
from .vocab import Vocab


def gpt2_config_dict(cfg: V6Config, vocab: Vocab) -> dict[str, Any]:
    return {
        "vocab_size": len(vocab.id_to_token),
        "bos_token_id": vocab.bos_id,
        "eos_token_id": vocab.eos_id,
        "pad_token_id": vocab.pad_id,
        "n_layer": int(cfg.model.n_layer),
        "n_head": int(cfg.model.n_head),
        "n_embd": int(cfg.model.n_embd),
        "n_positions": int(cfg.model.n_positions),
        "n_ctx": int(cfg.model.n_positions),
        "activation_function": str(cfg.model.activation_function),
        "resid_pdrop": float(cfg.model.resid_pdrop),
        "embd_pdrop": float(cfg.model.embd_pdrop),
        "attn_pdrop": float(cfg.model.attn_pdrop),
        "tie_word_embeddings": bool(cfg.model.tie_word_embeddings),
        "attn_implementation": "eager",
    }


def make_model(cfg: V6Config, vocab: Vocab, device: str | torch.device) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel(GPT2Config(**gpt2_config_dict(cfg, vocab)))
    return model.to(device)


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def count_logits(logits_at_pos: torch.Tensor, vocab: Vocab) -> torch.Tensor:
    return logits_at_pos[..., torch.tensor(vocab.count_ids, device=logits_at_pos.device)]


def save_pretrained_checkpoint(model: GPT2LMHeadModel, path, metadata: dict[str, Any] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    if metadata is not None:
        import json

        (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

