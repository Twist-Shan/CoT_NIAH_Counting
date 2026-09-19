from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from .config import ExperimentConfig, SUPPORTED_POSITION_ENCODINGS
from .data import Vocab


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    attentions: tuple[torch.Tensor, ...] | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


def _apply_rope(tensor: torch.Tensor, *, base: float = 10_000.0) -> torch.Tensor:
    """Apply standard interleaved rotary position embeddings to B,H,T,D."""

    _, _, length, head_dim = tensor.shape
    if head_dim % 2:
        raise ValueError("RoPE requires an even attention head dimension")
    positions = torch.arange(length, device=tensor.device, dtype=torch.float32)
    inverse = 1.0 / (
        float(base)
        ** (torch.arange(0, head_dim, 2, device=tensor.device, dtype=torch.float32) / head_dim)
    )
    angles = torch.outer(positions, inverse).to(dtype=tensor.dtype)
    cosine = angles.cos()[None, None]
    sine = angles.sin()[None, None]
    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]
    rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ExperimentConfig, position_encoding: str):
        super().__init__()
        self.n_head = int(cfg.n_head)
        self.head_dim = int(cfg.n_embd // cfg.n_head)
        self.position_encoding = position_encoding
        self.rope_base = float(cfg.rope_base)
        self.max_relative_distance = int(cfg.max_relative_distance)
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.output = nn.Linear(cfg.n_embd, cfg.n_embd)
        if position_encoding == "rpe":
            self.relative_bias = nn.Parameter(
                torch.zeros(cfg.n_head, self.max_relative_distance + 1)
            )
        else:
            self.register_parameter("relative_bias", None)

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
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if self.position_encoding == "rope":
            query = _apply_rope(query, base=self.rope_base)
            key = _apply_rope(key, base=self.rope_base)

        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        if self.relative_bias is not None:
            query_positions = torch.arange(length, device=hidden.device)[:, None]
            key_positions = torch.arange(length, device=hidden.device)[None, :]
            distances = (query_positions - key_positions).clamp(0, self.max_relative_distance)
            scores = scores + self.relative_bias[:, distances][None].to(dtype=scores.dtype)

        causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).triu(1)
        scores = scores.masked_fill(causal[None, None], torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            invalid_keys = attention_mask[:, None, None, :].eq(0)
            scores = scores.masked_fill(invalid_keys, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        if head_mask is not None:
            weights = weights * head_mask[None, :, None, None].to(weights)
        context = weights @ value
        context = context.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(context), weights if output_attentions else None


class TransformerLayer(nn.Module):
    def __init__(self, cfg: ExperimentConfig, position_encoding: str):
        super().__init__()
        self.ln_attention = nn.LayerNorm(cfg.n_embd)
        self.attention = CausalSelfAttention(cfg, position_encoding)
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
    """A small causal Transformer whose only variant is positional encoding."""

    def __init__(self, cfg: ExperimentConfig, vocab: Vocab, position_encoding: str):
        super().__init__()
        if position_encoding not in SUPPORTED_POSITION_ENCODINGS:
            raise ValueError(f"unsupported position encoding {position_encoding!r}")
        self.position_encoding = position_encoding
        self.token_embedding = nn.Embedding(len(vocab.id_to_token), cfg.n_embd)
        self.position_embedding = (
            nn.Embedding(cfg.n_positions, cfg.n_embd) if position_encoding == "ape" else None
        )
        self.layers = nn.ModuleList(
            [TransformerLayer(cfg, position_encoding) for _ in range(cfg.n_layer)]
        )
        self.final_norm = nn.LayerNorm(cfg.n_embd)
        self.config = SimpleNamespace(
            vocab_size=len(vocab.id_to_token),
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            n_positions=cfg.n_positions,
            position_encoding=position_encoding,
            rope_base=cfg.rope_base,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

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
        batch, length = input_ids.shape
        if length > self.config.n_positions:
            raise ValueError(
                f"input length {length} exceeds n_positions={self.config.n_positions}"
            )
        want_attention = self.config.output_attentions if output_attentions is None else output_attentions
        want_hidden = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        hidden = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = torch.arange(length, device=input_ids.device)
            hidden = hidden + self.position_embedding(positions)[None]

        hidden_states: list[torch.Tensor] | None = [hidden] if want_hidden else None
        attentions: list[torch.Tensor] | None = [] if want_attention else None
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
        normalized = self.final_norm(hidden)
        logits = torch.nn.functional.linear(normalized, self.token_embedding.weight)
        return CausalLMOutput(
            logits=logits,
            attentions=tuple(attentions) if attentions is not None else None,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_model(
    cfg: ExperimentConfig,
    vocab: Vocab,
    position_encoding: str,
    device: str | torch.device | None = None,
) -> TinyPositionCausalLM:
    model = TinyPositionCausalLM(cfg, vocab, position_encoding)
    return model.to(device or cfg.device)
