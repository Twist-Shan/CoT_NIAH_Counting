from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelOutput:
    logits: torch.Tensor
    hidden_states: list[torch.Tensor] | None = None
    attentions: list[torch.Tensor] | None = None


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def build_rope_cache(seq_len: int, head_dim: int, device: torch.device, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos[:, :, : x.size(2), :]) + (rotate_half(x) * sin[:, :, : x.size(2), :])


class RoPECausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        output_attentions: bool = False,
        ablate_heads: set[tuple[int, int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = build_rope_cache(seq_len, self.head_dim, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask[None, None, :, :], torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        y = probs @ v
        if ablate_heads:
            for target_layer, target_head in ablate_heads:
                if int(target_layer) == layer_idx and 0 <= int(target_head) < self.n_heads:
                    y[:, int(target_head), :, :] = 0.0
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.out(y), probs if output_attentions else None


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RoPECausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        output_attentions: bool = False,
        ablate_heads: set[tuple[int, int]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, probs = self.attn(self.ln1(x), layer_idx, output_attentions, ablate_heads)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, probs


class TinyRoPETransformer(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = dict(config)
        self.token_embed = nn.Embedding(config["vocab_size"], config["d_model"])
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config["d_model"], config["n_heads"], config["d_mlp"], config.get("dropout", 0.0))
                for _ in range(config["n_layers"])
            ]
        )
        self.ln_f = nn.LayerNorm(config["d_model"])
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)
        self.context_len = int(config.get("context_len", 2048))
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        ablate_heads: set[tuple[int, int]] | None = None,
    ) -> ModelOutput:
        if input_ids.size(1) > self.context_len:
            raise ValueError(f"Sequence length {input_ids.size(1)} exceeds context_len={self.context_len}.")
        x = self.token_embed(input_ids)
        hidden_states: list[torch.Tensor] | None = [x] if output_hidden_states else None
        attentions: list[torch.Tensor] | None = [] if output_attentions else None
        for layer_idx, block in enumerate(self.blocks):
            x, probs = block(x, layer_idx, output_attentions, ablate_heads)
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions:
                attentions.append(probs)
        logits = self.lm_head(self.ln_f(x))
        return ModelOutput(logits=logits, hidden_states=hidden_states, attentions=attentions)


def make_model(config: dict[str, Any], device: str | torch.device) -> TinyRoPETransformer:
    return TinyRoPETransformer(config).to(device)


def weighted_next_token_loss(logits: torch.Tensor, input_ids: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ce = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).view(input_ids.size(0), -1)
    usable_weights = weights[:, :-1].to(ce.device)
    denom = usable_weights.sum().clamp_min(1.0)
    return (ce * usable_weights).sum() / denom, ce.detach()


def numeric_logits(logits_at_pos: torch.Tensor, number_ids: list[int]) -> torch.Tensor:
    return logits_at_pos[..., torch.tensor(number_ids, device=logits_at_pos.device)]
