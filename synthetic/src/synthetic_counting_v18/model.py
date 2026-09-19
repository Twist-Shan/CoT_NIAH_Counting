from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import ReferenceConfig


def apply_rope(tensor: torch.Tensor, base: float) -> torch.Tensor:
    _, _, length, width = tensor.shape
    positions = torch.arange(length, device=tensor.device, dtype=torch.float32)
    frequencies = 1.0 / (
        float(base)
        ** (torch.arange(0, width, 2, device=tensor.device, dtype=torch.float32) / width)
    )
    angles = torch.outer(positions, frequencies).to(tensor.dtype)
    cosine, sine = angles.cos()[None, None], angles.sin()[None, None]
    even, odd = tensor[..., 0::2], tensor[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ReferenceConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.rope_base = cfg.rope_base
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(
        self,
        hidden: torch.Tensor,
        valid: torch.Tensor | None,
        *,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).view(batch, length, 3, self.n_head, self.head_dim)
        query, key, value = (part.transpose(1, 2) for part in qkv.unbind(2))
        query = apply_rope(query, self.rope_base)
        key = apply_rope(key, self.rope_base)
        if output_attentions:
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
            allowed = causal[None, None]
            if valid is not None:
                allowed = allowed & valid[:, None, None, :].bool()
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores.float(), dim=-1).to(query.dtype)
            context = torch.matmul(weights, value)
        elif valid is None or bool(valid.all()):
            weights = None
            context = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        else:
            weights = None
            causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
            allowed = causal[None, None] & valid[:, None, None, :].bool()
            bias = torch.zeros(batch, 1, length, length, device=hidden.device, dtype=query.dtype)
            bias.masked_fill_(~allowed, torch.finfo(query.dtype).min)
            context = F.scaled_dot_product_attention(query, key, value, attn_mask=bias)
        projected = self.proj(context.transpose(1, 2).contiguous().view(batch, length, width))
        return projected, weights


class Block(nn.Module):
    def __init__(self, cfg: ReferenceConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attention = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_inner),
            nn.GELU(),
            nn.Linear(cfg.n_inner, cfg.n_embd),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        valid: torch.Tensor | None,
        *,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attention_output, weights = self.attention(
            self.ln1(hidden),
            valid,
            output_attentions=output_attentions,
        )
        hidden = hidden + attention_output
        return hidden + self.mlp(self.ln2(hidden)), weights


@dataclass
class ModelOutput:
    logits: torch.Tensor
    attentions: tuple[torch.Tensor, ...] | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


class ReferenceTransformer(nn.Module):
    def __init__(self, cfg: ReferenceConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.final_norm = nn.LayerNorm(cfg.n_embd)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        hidden = self.embedding(input_ids)
        hidden_states = [hidden] if output_hidden_states else None
        attentions = [] if output_attentions else None
        for layer in self.layers:
            hidden, weights = layer(
                hidden,
                attention_mask,
                output_attentions=output_attentions,
            )
            if hidden_states is not None:
                hidden_states.append(hidden)
            if attentions is not None and weights is not None:
                attentions.append(weights)
        logits = F.linear(self.final_norm(hidden), self.embedding.weight)
        return ModelOutput(
            logits,
            tuple(attentions) if attentions is not None else None,
            tuple(hidden_states) if hidden_states is not None else None,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
