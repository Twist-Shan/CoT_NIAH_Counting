"""RoPE causal Transformer with a fast SDPA training path.

The module keeps the v10/v16.2 component names (``attention.output``,
``mlp[2]`` and residual block hooks), so the existing causal patching code can
be reused.  Explicit attention weights and head masks fall back to the exact
manual implementation; ordinary training uses PyTorch SDPA/Flash Attention.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from .config import V20Config
from .data import V20Vocab


def _apply_rope(tensor: torch.Tensor, *, base: float) -> torch.Tensor:
    _, _, length, width = tensor.shape
    if width % 2:
        raise ValueError("RoPE requires an even head width")
    positions = torch.arange(length, device=tensor.device, dtype=torch.float32)
    inverse = 1.0 / (
        float(base)
        ** (torch.arange(0, width, 2, device=tensor.device, dtype=torch.float32) / width)
    )
    angles = torch.outer(positions, inverse).to(tensor.dtype)
    cosine, sine = angles.cos()[None, None], angles.sin()[None, None]
    even, odd = tensor[..., 0::2], tensor[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2)


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    attentions: tuple[torch.Tensor, ...] | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: V20Config):
        super().__init__()
        self.n_head = int(cfg.n_head)
        self.head_dim = int(cfg.n_embd // cfg.n_head)
        self.rope_base = float(cfg.rope_base)
        self.use_sdpa = bool(cfg.use_sdpa)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.output = nn.Linear(cfg.n_embd, cfg.n_embd)
        # Analysis-only callback used by causal mediation experiments.  The
        # ordinary training path leaves this as ``None`` and therefore retains
        # SDPA/Flash Attention.  A callback receives (query, key, value,
        # weights) after softmax and returns replacement (value, weights).
        self.intervention = None

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        output_attentions: bool,
        head_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).view(batch, length, 3, self.n_head, self.head_dim)
        query, key, value = (part.transpose(1, 2) for part in qkv.unbind(2))
        query = _apply_rope(query, base=self.rope_base)
        key = _apply_rope(key, base=self.rope_base)

        fast_path = (
            self.use_sdpa
            and not output_attentions
            and head_mask is None
            and self.intervention is None
        )
        if fast_path and (attention_mask is None or bool(attention_mask.bool().all())):
            context = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=True
            )
            weights = None
        elif fast_path:
            causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
            allowed = causal[None, None] & attention_mask[:, None, None, :].bool()
            context = F.scaled_dot_product_attention(
                query, key, value, attn_mask=allowed, dropout_p=0.0, is_causal=False
            )
            weights = None
        else:
            scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
            causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).triu(1)
            scores = scores.masked_fill(causal[None, None], torch.finfo(scores.dtype).min)
            if attention_mask is not None:
                scores = scores.masked_fill(
                    attention_mask[:, None, None, :].eq(0), torch.finfo(scores.dtype).min
                )
            weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
            if self.intervention is not None:
                value, weights = self.intervention(query, key, value, weights)
            if head_mask is not None:
                weights = weights * head_mask[None, :, None, None].to(weights)
            context = weights @ value

        projected = self.output(context.transpose(1, 2).contiguous().view(batch, length, width))
        return projected, weights if output_attentions else None


class TransformerLayer(nn.Module):
    def __init__(self, cfg: V20Config):
        super().__init__()
        self.ln_attention = nn.LayerNorm(cfg.n_embd)
        self.attention = CausalSelfAttention(cfg)
        self.ln_mlp = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_inner),
            nn.GELU(approximate="tanh"),
            nn.Linear(cfg.n_inner, cfg.n_embd),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        output_attentions: bool,
        head_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attention_output, weights = self.attention(
            self.ln_attention(hidden),
            attention_mask,
            output_attentions=output_attentions,
            head_mask=head_mask,
        )
        hidden = hidden + attention_output
        hidden = hidden + self.mlp(self.ln_mlp(hidden))
        return hidden, weights


class TinyPositionCausalLM(nn.Module):
    def __init__(self, cfg: V20Config, vocab: V20Vocab):
        super().__init__()
        self.position_encoding = "rope"
        self.token_embedding = nn.Embedding(len(vocab.id_to_token), cfg.n_embd)
        self.lm_head = (
            None
            if cfg.tie_word_embeddings
            else nn.Linear(cfg.n_embd, len(vocab.id_to_token), bias=False)
        )
        self.count_lm_head = (
            nn.Linear(cfg.n_embd, len(vocab.number_ids), bias=False)
            if cfg.untie_atomic_count_readout
            else None
        )
        self.register_buffer(
            "_atomic_count_ids",
            torch.tensor(vocab.number_ids, dtype=torch.long),
            persistent=False,
        )
        self.position_embedding = None
        self.layers = nn.ModuleList(TransformerLayer(cfg) for _ in range(cfg.n_layer))
        self.final_norm = nn.LayerNorm(cfg.n_embd)
        self.config = SimpleNamespace(
            vocab_size=len(vocab.id_to_token),
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            n_positions=cfg.n_positions,
            position_encoding="rope",
            rope_base=cfg.rope_base,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
            tie_word_embeddings=cfg.tie_word_embeddings,
            untie_atomic_count_readout=cfg.untie_atomic_count_readout,
        )

    @property
    def unembedding_weight(self) -> torch.Tensor:
        weight = (
            self.token_embedding.weight
            if self.lm_head is None
            else self.lm_head.weight
        )
        if self.count_lm_head is None:
            return weight
        # ``index_copy`` gives the count rows only to ``count_lm_head`` during
        # backpropagation; the corresponding input-embedding rows do not
        # receive answer-unembedding gradients.  All other rows remain tied.
        return weight.index_copy(
            0,
            self._atomic_count_ids,
            self.count_lm_head.weight,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        head_mask: torch.Tensor | None = None,
        **_unused,
    ) -> CausalLMOutput:
        _, length = input_ids.shape
        if length > self.config.n_positions:
            raise ValueError(f"input length {length} exceeds n_positions={self.config.n_positions}")
        want_attention = self.config.output_attentions if output_attentions is None else output_attentions
        want_hidden = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        hidden = self.token_embedding(input_ids)
        hidden_states = [hidden] if want_hidden else None
        attentions = [] if want_attention else None
        for layer_index, layer in enumerate(self.layers):
            layer_mask = None if head_mask is None else head_mask[layer_index]
            hidden, weights = layer(
                hidden,
                attention_mask,
                output_attentions=want_attention,
                head_mask=layer_mask,
            )
            if hidden_states is not None:
                hidden_states.append(hidden)
            if attentions is not None and weights is not None:
                attentions.append(weights)
        logits = F.linear(self.final_norm(hidden), self.unembedding_weight)
        return CausalLMOutput(
            logits,
            tuple(attentions) if attentions is not None else None,
            tuple(hidden_states) if hidden_states is not None else None,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def initialize_model(model: TinyPositionCausalLM, seed: int) -> None:
    """Name-seeded initialization keeps v20/v21 shared weights exactly paired."""

    with torch.no_grad():
        for name, module in model.named_modules():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_stable_seed(seed, name))
            if isinstance(module, nn.Linear):
                module.weight.normal_(0.0, 0.02, generator=generator)
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.normal_(0.0, 0.02, generator=generator)
            elif isinstance(module, nn.LayerNorm):
                module.weight.fill_(1.0)
                module.bias.zero_()
        # Untying is an optimization constraint change, not a step-zero
        # function change: both variants begin with exactly the same logits.
        if model.lm_head is not None:
            model.lm_head.weight.copy_(model.token_embedding.weight)
        if model.count_lm_head is not None:
            model.count_lm_head.weight.copy_(
                model.token_embedding.weight.index_select(
                    0, model._atomic_count_ids
                )
            )


def build_model(
    cfg: V20Config,
    vocab: V20Vocab,
    position_encoding: str = "rope",
    device: str | torch.device | None = None,
) -> TinyPositionCausalLM:
    if position_encoding != "rope":
        raise ValueError("v20/v21 are RoPE-only experiments")
    model = TinyPositionCausalLM(cfg, vocab)
    initialize_model(model, cfg.seed)
    return model.to(device or cfg.device)


__all__ = ["CausalLMOutput", "TinyPositionCausalLM", "build_model", "initialize_model"]
