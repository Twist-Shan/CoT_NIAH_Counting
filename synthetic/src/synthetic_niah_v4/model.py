from __future__ import annotations

import sys
import importlib.util as importlib_util
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
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

from .hooks import HookFn
from .vocab import Vocab


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    hidden_states: dict[str, torch.Tensor] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


class HookedGPT2LMHeadModel(nn.Module):
    """GPT2LMHeadModel wrapper with named residual-stream hooks.

    The underlying architecture is HuggingFace GPT-2 with learned absolute
    positional embeddings. Hooks are registered only for a single forward call.
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel(config)

    @property
    def config(self) -> GPT2Config:
        return self.gpt2.config

    def _hooked_forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        output_hidden_states: bool,
        output_attentions: bool,
        hook_fn: HookFn | None,
    ) -> ModelOutput:
        captured: dict[str, torch.Tensor] = {}

        def apply(name: str, hidden: torch.Tensor) -> torch.Tensor:
            out = hook_fn(name, hidden) if hook_fn is not None else hidden
            if output_hidden_states:
                captured[name] = out.detach().clone()
            return out

        handles = []
        try:
            handles.append(
                self.gpt2.transformer.drop.register_forward_hook(
                    lambda _module, _inputs, output: apply("embed", output)
                )
            )
            for layer_idx, block in enumerate(self.gpt2.transformer.h):
                def pre_hook(_module, inputs, layer_idx=layer_idx):
                    if not inputs:
                        return inputs
                    hidden = inputs[0]
                    if isinstance(hidden, tuple):
                        if not hidden:
                            return inputs
                        patched = (apply(f"resid_pre_layer_{layer_idx}", hidden[0]), *hidden[1:])
                        return (patched, *inputs[1:])
                    return (apply(f"resid_pre_layer_{layer_idx}", hidden), *inputs[1:])

                handles.append(
                    block.register_forward_pre_hook(pre_hook)
                )

                def post_hook(_module, _inputs, output, layer_idx=layer_idx):
                    if isinstance(output, tuple):
                        hidden = apply(f"resid_post_layer_{layer_idx}", output[0])
                        return (hidden, *output[1:])
                    return apply(f"resid_post_layer_{layer_idx}", output)

                handles.append(block.register_forward_hook(post_hook))
            handles.append(
                self.gpt2.transformer.ln_f.register_forward_hook(
                    lambda _module, _inputs, output: apply("final_norm", output)
                )
            )
            out = self.gpt2(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                use_cache=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        return ModelOutput(
            logits=out.logits,
            loss=out.loss,
            hidden_states=captured if output_hidden_states else None,
            attentions=out.attentions,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        hook_fn: HookFn | None = None,
    ) -> ModelOutput:
        return self._hooked_forward(input_ids, labels, attention_mask, output_hidden_states, output_attentions, hook_fn)


def model_config_dict(cfg: Any, vocab: Vocab | int) -> dict[str, Any]:
    model = cfg.model if hasattr(cfg, "model") else cfg
    vocab_obj = Vocab.build() if isinstance(vocab, int) else vocab
    vocab_size = int(vocab) if isinstance(vocab, int) else len(vocab_obj.id_to_token)
    return {
        "vocab_size": vocab_size,
        "bos_token_id": vocab_obj.bos_id,
        "eos_token_id": vocab_obj.eos_id,
        "pad_token_id": vocab_obj.pad_id,
        "n_layer": int(model.n_layer),
        "n_head": int(model.n_head),
        "n_embd": int(model.n_embd),
        "n_positions": int(model.n_positions),
        "n_ctx": int(model.n_positions),
        "activation_function": str(model.activation_function),
        "resid_pdrop": float(model.resid_pdrop),
        "embd_pdrop": float(model.embd_pdrop),
        "attn_pdrop": float(model.attn_pdrop),
    }


def make_model(cfg: Any, vocab: Vocab | int, device: str | torch.device) -> HookedGPT2LMHeadModel:
    gpt2_config = GPT2Config(**model_config_dict(cfg, vocab))
    return HookedGPT2LMHeadModel(gpt2_config).to(device)


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def count_logits(logits_at_pos: torch.Tensor, count_ids: list[int]) -> torch.Tensor:
    return logits_at_pos[..., torch.tensor(count_ids, device=logits_at_pos.device)]
