#!/usr/bin/env python3
"""
Offline attention statistics from Qwen3 q_proj/k_proj caches.

Loads raw q_proj/k_proj outputs captured by capture_qk_qwen3.py, reconstructs
Qwen3's per-head post-q_norm/post-k_norm/post-RoPE Q/K, and computes attention
statistics for one layer and one query head without materializing the full
[heads, seq, seq] attention tensor by default. For short-context validation,
it can also save the full [seq, seq] matrix for the selected head.

Target: Qwen/Qwen3-8B text-only causal LM, batch size 1.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


COMPUTE_DTYPES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def load_json(path: Path) -> Dict:
    """
    Load a UTF-8 JSON file and return it as a Python dictionary.
    
    Used for ``metadata.json``, ``analysis_spec.json``, and result configuration
    files produced by the capture script.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def _consolidated_qk_metadata(cache_dir: Path) -> Dict:
    manifest_path = cache_dir.parent / "qk_cache_metadata.json"
    if not manifest_path.exists():
        return {}
    manifest = load_json(manifest_path)
    examples = manifest.get("examples", {})
    if isinstance(examples, dict):
        item = examples.get(cache_dir.name, {})
        return dict(item) if isinstance(item, dict) else {}
    if isinstance(examples, list):
        for item in examples:
            if isinstance(item, dict) and item.get("cache_dir") == cache_dir.name:
                return dict(item)
    return {}


def load_cache_metadata(cache_dir: Path) -> Dict:
    """Load per-cache metadata from a legacy file or consolidated parent manifest."""

    legacy_path = cache_dir / "metadata.json"
    if legacy_path.exists():
        return load_json(legacy_path)
    consolidated = _consolidated_qk_metadata(cache_dir)
    metadata = consolidated.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    raise FileNotFoundError(f"Missing Q/K metadata for {cache_dir}")


def load_cache_analysis_spec(cache_dir: Path) -> Dict:
    legacy_path = cache_dir / "analysis_spec.json"
    if legacy_path.exists():
        return load_json(legacy_path)
    consolidated = _consolidated_qk_metadata(cache_dir)
    spec = consolidated.get("analysis_spec")
    return spec if isinstance(spec, dict) else {}


def load_cache_tokens(cache_dir: Path) -> list[str] | None:
    legacy_path = cache_dir / "tokens.json"
    if legacy_path.exists():
        return json.loads(legacy_path.read_text(encoding="utf-8"))
    consolidated = _consolidated_qk_metadata(cache_dir)
    tokens = consolidated.get("tokens")
    return [str(x) for x in tokens] if isinstance(tokens, list) else None


def load_tensor(path: Path):
    """
    Load a PyTorch tensor-like artifact from disk onto CPU.
    
    The capture path normally saves tensors directly, but this helper also
    accepts a dictionary containing a ``"tensor"`` key for compatibility with
    slightly richer save formats.
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "tensor" in obj:
        return obj["tensor"]
    return obj


def resolve_index(idx: Optional[int], T: int, *, end_exclusive: bool = False) -> int:
    """
    Convert user-facing positive or negative indices into absolute token positions.
    
    Negative indices follow Python-like conventions. For exclusive range ends,
    ``-1`` is interpreted as including the final token, so the resolved value is
    one past the final position.
    """
    if idx is None:
        return T if end_exclusive else T - 1
    idx = int(idx)
    if idx < 0:
        return T + idx + (1 if end_exclusive else 0)
    return idx


def clamp_interval(start: int, end: int, T: int) -> Tuple[int, int]:
    """
    Clamp a half-open interval to the valid token range ``[0, T]``.
    
    If the end falls before the start after clamping, the function returns an
    empty interval by setting ``end == start``.
    """
    start = max(0, min(T, int(start)))
    end = max(0, min(T, int(end)))
    if end < start:
        end = start
    return start, end


def normalize_intervals(raw, T: int) -> List[Tuple[int, int]]:
    """
    Normalize one or more span specifications into valid half-open intervals.
    
    Accepts either a single interval ``[start, end]`` or a list of intervals.
    Invalid or empty intervals are dropped after clamping to the sequence length.
    """
    intervals: List[Tuple[int, int]] = []
    if raw is None:
        return intervals
    if isinstance(raw, (list, tuple)) and len(raw) == 2 and all(isinstance(x, int) for x in raw):
        raw = [raw]
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Bad interval {item!r}; expected [start, end].")
        s, e = clamp_interval(int(item[0]), int(item[1]), T)
        if e > s:
            intervals.append((s, e))
    return intervals


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Implement the half-dimension rotation used by Qwen/Llama-style RoPE.
    
    Given ``[..., D]``, it splits the final dimension into two halves and returns
    ``[-second_half, first_half]`` so that ``x*cos + rotate_half(x)*sin`` applies
    the rotary position embedding.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def qwen3_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Apply Qwen3 RMSNorm over the last dimension.
    
    This mirrors the attention-layer ``q_norm`` and ``k_norm`` modules. It is
    needed because the captured q_proj/k_proj outputs are raw projections, while
    Qwen3 attention uses normalized Q/K before RoPE.
    """
    # Mirrors Qwen3RMSNorm over the last dimension.
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    y = x_float * torch.rsqrt(variance + eps)
    return (y * weight.float()).to(dtype=x.dtype)


def get_rope_theta(config: Dict) -> float:
    """
    Read the RoPE base theta from the saved model config.
    
    Qwen3 stores RoPE settings in ``rope_parameters`` and/or top-level config
    fields depending on Transformers version. The released Qwen3-8B default is
    handled here, with a conservative fallback.
    """
    rope_parameters = config.get("rope_parameters") or {}
    if "rope_theta" in rope_parameters:
        return float(rope_parameters["rope_theta"])
    if "rope_theta" in config:
        return float(config["rope_theta"])
    return 1_000_000.0


def assert_default_rope(config: Dict) -> None:
    """
    Reject unsupported RoPE scaling modes before reconstruction begins.

    This analyzer intentionally implements the released default Qwen3 RoPE only.
    If a checkpoint enables YaRN, dynamic scaling, or another long-context RoPE
    variant, extend this code to call the exact Hugging Face rotary embedding.
    """
    rope_parameters = config.get("rope_parameters") or {}
    rope_type = rope_parameters.get("rope_type", "default")
    rope_scaling = config.get("rope_scaling", None)
    if rope_type not in (None, "default"):
        raise NotImplementedError(f"Only default Qwen3 RoPE is implemented here; got rope_type={rope_type!r}")
    if rope_scaling not in (None, "null"):
        raise NotImplementedError(
            f"Only Qwen3 default RoPE is implemented here; got rope_scaling={rope_scaling!r}. "
            "For YaRN/long-context scaling, compute cos/sin with the exact Hugging Face rotary_emb."
        )


def assert_supported_attention_masking(config: Dict, layer: int) -> None:
    """Reject attention masking modes that this offline analyzer does not implement.

    Qwen3 configs can carry a global ``use_sliding_window`` flag while also
    describing each layer in ``layer_types``. When per-layer metadata is present,
    only layers explicitly marked ``sliding_attention`` require a sliding-window
    mask. Older configs without ``layer_types`` cannot identify full-attention
    exceptions, so a global sliding-window setting is conservatively rejected.
    """
    layer_types = config.get("layer_types") or []
    layer_type = layer_types[layer] if 0 <= layer < len(layer_types) else None
    has_window = config.get("sliding_window") not in (None, "null")
    if not has_window:
        return

    if layer_type == "sliding_attention" or (not layer_types and bool(config.get("use_sliding_window"))):
        raise NotImplementedError(
            "This offline analyzer currently supports full causal attention only; "
            f"layer {layer} appears to use sliding-window attention "
            f"(layer_type={layer_type!r}, sliding_window={config.get('sliding_window')!r})."
        )


def qwen3_default_rope_cos_sin(
    *,
    position_ids: torch.Tensor,  # [T]
    head_dim: int,
    rope_theta: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute default Qwen3 RoPE cosine and sine tables for all token positions.
    
    The returned tensors have shape ``[T, head_dim]`` and can be applied to a
    single selected head with ``q*cos + rotate_half(q)*sin``. This assumes the
    standard non-scaled RoPE used by the released Qwen/Qwen3-8B config.
    """
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    pos = position_ids.to(device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [T, D/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [T, D]
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def reconstruct_single_head_qk(
    *,
    cache_dir: Path,
    layer: int,
    head: int,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Load raw Q/K caches and reconstruct post-q_norm/post-RoPE Q/K for one head.
    
    The function maps a query head to its corresponding KV head under Qwen3 GQA,
    applies the saved q_norm/k_norm parameters, applies default Qwen3 RoPE, and
    returns ``q`` and ``k`` with shape ``[seq_len, head_dim]`` for offline stats.
    """
    meta = load_cache_metadata(cache_dir)
    config = meta["model_config"]
    assert_default_rope(config)
    assert_supported_attention_masking(config, layer)

    Hq = int(config["num_attention_heads"])
    Hkv = int(config["num_key_value_heads"])
    D = int(config.get("head_dim") or int(config["hidden_size"]) // Hq)
    T = int(meta["seq_len"])
    if Hq <= 0 or Hkv <= 0 or Hq % Hkv != 0:
        raise ValueError(f"Expected num_attention_heads to be divisible by num_key_value_heads; got {Hq=} {Hkv=}")
    if not (0 <= head < Hq):
        raise ValueError(f"head must be in [0, {Hq}); got {head}")
    n_rep = Hq // Hkv
    kv_head = head // n_rep

    q_raw = load_tensor(cache_dir / f"layer_{layer:02d}_q_raw.pt")
    k_raw = load_tensor(cache_dir / f"layer_{layer:02d}_k_raw.pt")
    if q_raw.shape[0] != 1 or k_raw.shape[0] != 1:
        raise ValueError("This simple implementation assumes batch size 1.")
    if int(q_raw.shape[1]) != T or int(k_raw.shape[1]) != T:
        raise ValueError("Q/K cache sequence length does not match metadata.")
    expected_q_width = Hq * D
    expected_k_width = Hkv * D
    if int(q_raw.shape[-1]) != expected_q_width or int(k_raw.shape[-1]) != expected_k_width:
        raise ValueError(
            "Q/K cache hidden dimensions do not match metadata: "
            f"q_raw.shape[-1]={int(q_raw.shape[-1])} expected {expected_q_width}, "
            f"k_raw.shape[-1]={int(k_raw.shape[-1])} expected {expected_k_width}."
        )

    q_head_raw = q_raw.view(1, T, Hq, D)[0, :, head, :].to(device=device, dtype=compute_dtype)
    k_head_raw = k_raw.view(1, T, Hkv, D)[0, :, kv_head, :].to(device=device, dtype=compute_dtype)
    del q_raw, k_raw

    norms = torch.load(cache_dir / f"layer_{layer:02d}_qk_norms.pt", map_location="cpu")
    q_weight = norms["q_norm_weight"].to(device=device, dtype=compute_dtype)
    k_weight = norms["k_norm_weight"].to(device=device, dtype=compute_dtype)
    q = qwen3_rmsnorm(q_head_raw, q_weight, float(norms["q_norm_eps"]))
    k = qwen3_rmsnorm(k_head_raw, k_weight, float(norms["k_norm_eps"]))
    del q_head_raw, k_head_raw

    position_ids = load_tensor(cache_dir / "position_ids.pt")[0].to(device=device)
    cos, sin = qwen3_default_rope_cos_sin(
        position_ids=position_ids,
        head_dim=D,
        rope_theta=get_rope_theta(config),
        dtype=compute_dtype,
        device=device,
    )
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)

    info = {
        "seq_len": T,
        "head_dim": D,
        "num_attention_heads": Hq,
        "num_key_value_heads": Hkv,
        "num_key_value_groups": n_rep,
        "kv_head": kv_head,
        "scaling": D ** -0.5,
        "config": config,
        "metadata": meta,
    }
    return q, k, info


def block_span_mask(intervals: Sequence[Tuple[int, int]], ks: int, ke: int, device: torch.device) -> torch.Tensor:
    """
    Build a boolean mask for the part of named spans that overlaps one key block.
    
    During blockwise analysis, keys are processed as ``[ks, ke)`` slices. This
    helper marks which key positions in the current block belong to a target
    span such as a needle, BOS token, or special-token set.
    """
    mask = torch.zeros(ke - ks, dtype=torch.bool, device=device)
    for s, e in intervals:
        lo = max(s, ks)
        hi = min(e, ke)
        if hi > lo:
            mask[lo - ks : hi - ks] = True
    return mask


@torch.no_grad()
def compute_query_block_stats(
    *,
    q_block: torch.Tensor,  # [M, D]
    k: torch.Tensor,  # [T, D]
    query_positions: torch.Tensor,  # [M], absolute token indices
    spans: Dict[str, List[Tuple[int, int]]],
    key_padding_mask: Optional[torch.Tensor],  # [T] bool or None
    scaling: float,
    key_block_size: int,
    topk: int = 0,
    local_windows: Sequence[int] = (),
    return_rows: bool = False,
) -> Dict:
    """
    Compute exact attention statistics for a block of selected query positions.
    
    This routine never materializes a full ``[T, T]`` attention matrix. It uses a
    two-pass log-sum-exp softmax over key blocks: first row maxima/top-k logits,
    then denominators, entropy, span mass, local-window mass, optional top-k
    probabilities, and optional selected full rows.
    """
    device = q_block.device
    T = k.shape[0]
    M = q_block.shape[0]
    if key_block_size <= 0:
        raise ValueError(f"key_block_size must be positive; got {key_block_size}")
    if topk < 0:
        raise ValueError(f"topk must be non-negative; got {topk}")
    query_positions = query_positions.to(device=device, dtype=torch.long)

    row_max = torch.full((M,), -torch.inf, dtype=torch.float32, device=device)
    if topk > 0:
        k_eff = min(int(topk), T)
        top_vals = torch.full((M, k_eff), -torch.inf, dtype=torch.float32, device=device)
        top_idx = torch.full((M, k_eff), -1, dtype=torch.long, device=device)
    else:
        top_vals = top_idx = None

    # Pass 1: stable row max and top-k logits.
    for ks in range(0, T, key_block_size):
        ke = min(ks + key_block_size, T)
        key_positions = torch.arange(ks, ke, device=device, dtype=torch.long)
        scores = (q_block.float() @ k[ks:ke].float().T) * float(scaling)  # [M, Bk]
        allowed = key_positions[None, :] <= query_positions[:, None]
        if key_padding_mask is not None:
            allowed = allowed & key_padding_mask[ks:ke][None, :]
        scores = scores.masked_fill(~allowed, -torch.inf)
        row_max = torch.maximum(row_max, scores.max(dim=-1).values)

        if topk > 0:
            candidate_vals = torch.cat([top_vals, scores], dim=-1)
            candidate_idx = torch.cat(
                [top_idx, key_positions[None, :].expand(M, -1)],
                dim=-1,
            )
            new_vals, gather_pos = torch.topk(candidate_vals, k=top_vals.shape[1], dim=-1)
            top_vals = new_vals
            top_idx = candidate_idx.gather(dim=-1, index=gather_pos)

    has_valid_key = torch.isfinite(row_max)
    row_max_safe = torch.where(has_valid_key, row_max, torch.zeros_like(row_max))

    denom = torch.zeros((M,), dtype=torch.float32, device=device)
    weighted_score_sum = torch.zeros((M,), dtype=torch.float32, device=device)
    span_numer = {name: torch.zeros((M,), dtype=torch.float32, device=device) for name in spans}
    local_numer = {int(w): torch.zeros((M,), dtype=torch.float32, device=device) for w in local_windows}
    rows = torch.empty((M, T), dtype=torch.float32, device="cpu") if return_rows else None

    # Pass 2: denominator and statistics.
    for ks in range(0, T, key_block_size):
        ke = min(ks + key_block_size, T)
        key_positions = torch.arange(ks, ke, device=device, dtype=torch.long)
        scores = (q_block.float() @ k[ks:ke].float().T) * float(scaling)
        allowed = key_positions[None, :] <= query_positions[:, None]
        if key_padding_mask is not None:
            allowed = allowed & key_padding_mask[ks:ke][None, :]
        scores = scores.masked_fill(~allowed, -torch.inf)
        safe_scores = scores.masked_fill(~allowed, 0.0)
        weights = torch.exp(scores - row_max_safe[:, None]).masked_fill(~allowed, 0.0)

        denom += weights.sum(dim=-1)
        weighted_score_sum += (weights * safe_scores).sum(dim=-1)

        for name, intervals in spans.items():
            smask = block_span_mask(intervals, ks, ke, device)
            if smask.any():
                span_numer[name] += weights[:, smask].sum(dim=-1)

        for w in local_numer:
            local_start = torch.clamp(query_positions - int(w) + 1, min=0)
            lmask = (key_positions[None, :] >= local_start[:, None]) & (
                key_positions[None, :] <= query_positions[:, None]
            )
            local_numer[w] += weights.masked_fill(~lmask, 0.0).sum(dim=-1)

        if rows is not None:
            # We cannot fill normalized rows until final denom is known, so write
            # unnormalized weights now and normalize after the loop.
            rows[:, ks:ke] = weights.cpu()

    denom_safe = denom.clamp_min(1e-30)
    entropy = row_max_safe + torch.log(denom_safe) - (weighted_score_sum / denom_safe)
    entropy = torch.where(has_valid_key, entropy, torch.full_like(entropy, torch.nan))
    span_mass = {name: (num / denom_safe).detach().cpu() for name, num in span_numer.items()}
    local_mass = {str(w): (num / denom_safe).detach().cpu() for w, num in local_numer.items()}

    out = {
        "entropy": entropy.detach().cpu(),
        "span_mass": span_mass,
        "local_mass": local_mass,
    }

    if topk > 0:
        top_probs = torch.exp(top_vals - row_max_safe[:, None]) / denom_safe[:, None]
        top_probs = torch.where(torch.isfinite(top_vals) & has_valid_key[:, None], top_probs, torch.zeros_like(top_probs))
        out["topk_values"] = top_vals.detach().cpu()
        out["topk_indices"] = top_idx.detach().cpu()
        out["topk_probs"] = top_probs.detach().cpu()

    if rows is not None:
        rows = rows / denom_safe.detach().cpu()[:, None]
        out["rows"] = rows

    return out


@torch.no_grad()
def compute_full_attention_matrix(
    *,
    q: torch.Tensor,  # [T, D]
    k: torch.Tensor,  # [T, D]
    key_padding_mask: Optional[torch.Tensor],  # [T] bool or None
    scaling: float,
) -> torch.Tensor:
    """
    Materialize the exact causal attention matrix for one reconstructed head.

    This is intentionally separate from the blockwise statistic path because it
    allocates a full ``[seq_len, seq_len]`` matrix. It is useful for
    short-context validation against Hugging Face ``output_attentions=True`` and
    should not be used for long-context runs. The result is returned on CPU in
    ``float32`` with invalid/future key positions set to probability zero.
    """
    if q.ndim != 2 or k.ndim != 2:
        raise ValueError(f"Expected q and k to be rank-2 [T, D] tensors; got {q.shape=} {k.shape=}")
    if q.shape != k.shape:
        raise ValueError(f"Expected q and k to have matching shapes; got {q.shape=} {k.shape=}")

    device = q.device
    T = int(q.shape[0])
    logits = (q.float() @ k.float().T) * float(scaling)
    positions = torch.arange(T, device=device, dtype=torch.long)
    allowed = positions[None, :] <= positions[:, None]
    if key_padding_mask is not None:
        if key_padding_mask.shape[-1] != T:
            raise ValueError(
                f"key_padding_mask length must match sequence length {T}; got {tuple(key_padding_mask.shape)}"
            )
        allowed = allowed & key_padding_mask.to(device=device, dtype=torch.bool)[None, :]
    logits = logits.masked_fill(~allowed, -torch.inf)

    # Softmax returns NaN for rows with no valid keys. That should not occur for
    # normal causal LM inputs, but keep the saved artifact finite for easier
    # notebook validation.
    probs = torch.softmax(logits, dim=-1).masked_fill(~allowed, 0.0)
    return torch.nan_to_num(probs, nan=0.0).detach().cpu().float()


def save_full_attention_matrix(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    info: Dict,
    cache_dir: Path,
    layer: int,
    head: int,
) -> Path:
    """Save the full reconstructed causal attention matrix for one layer/head."""
    attention_mask_path = cache_dir / "attention_mask.pt"
    if attention_mask_path.exists():
        key_padding_mask = load_tensor(attention_mask_path)[0].to(device=q.device).bool()
    else:
        key_padding_mask = None

    matrix = compute_full_attention_matrix(
        q=q,
        k=k,
        key_padding_mask=key_padding_mask,
        scaling=float(info["scaling"]),
    )
    matrix_path = cache_dir / f"attention_matrix_layer_{layer:02d}_head_{head:02d}.pt"
    torch.save(
        {
            "layer": int(layer),
            "head": int(head),
            "kv_head": int(info["kv_head"]),
            "matrix": matrix,
        },
        matrix_path,
    )
    return matrix_path


def normalize_query_positions(spec: Dict, T: int, metadata: Dict) -> Tuple[Dict[str, List[int]], List[int]]:
    """
    Resolve the analysis spec's query-position labels into absolute positions.
    
    Supports explicit integer positions, ``"last"``, and ``"auto_special"``.
    It returns both the label-to-position mapping and the sorted union of all
    query positions that need to be analyzed.
    """
    raw = spec.get("query_positions") or {}
    if not raw:
        raw = {"last": [T - 1]}
    normalized: Dict[str, List[int]] = {}
    all_positions = set()
    for label, vals in raw.items():
        if vals == "last":
            vals = [T - 1]
        elif vals == "auto_special":
            input_ids = load_tensor(Path(metadata["cache_dir"]) / "input_ids.pt")[0].tolist()
            special_ids = set(metadata.get("special_token_ids") or [])
            vals = [i for i, tok in enumerate(input_ids) if tok in special_ids]
        elif isinstance(vals, int):
            vals = [vals]
        elif vals is None:
            vals = []
        positions = []
        for v in vals:
            p = resolve_index(int(v), T)
            if 0 <= p < T:
                positions.append(p)
                all_positions.add(p)
        normalized[str(label)] = sorted(set(positions))
    return normalized, sorted(all_positions)


def normalize_spans(spec: Dict, T: int) -> Dict[str, List[Tuple[int, int]]]:
    """
    Resolve named span definitions from the analysis spec.
    
    Spans are returned as ``{name: [(start, end), ...]}`` after clamping and
    empty-interval removal. If no spans are provided, the BOS token span is used
    as a minimal default.
    """
    raw_spans = spec.get("spans") or {"bos": [[0, 1]]}
    spans: Dict[str, List[Tuple[int, int]]] = {}
    for name, raw in raw_spans.items():
        intervals = normalize_intervals(raw, T)
        if intervals:
            spans[str(name)] = intervals
    return spans


def token_repr(tokens: Optional[List[str]], input_ids: List[int], pos: int) -> Dict:
    """
    Create a JSON-friendly description of one token position.
    
    The output always contains the token position and token id. If token strings
    were saved by the capture script, it also includes the decoded tokenizer
    token for easier inspection of top-k attention results.
    """
    out = {"position": int(pos), "token_id": int(input_ids[pos])}
    if tokens is not None and 0 <= pos < len(tokens):
        out["token"] = tokens[pos]
    return out


def summarize_selected_rows(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    info: Dict,
    cache_dir: Path,
    spec: Dict,
    layer: int,
    head: int,
    key_block_size: int,
    topk: int,
    save_full_rows: bool,
) -> Tuple[Dict, Optional[Path]]:
    """
    Compute row-wise statistics for critical query tokens in one layer/head.
    
    This is the main path for last-token, special-token, and after-needle
    analysis. It loads the analysis spec, calls the blockwise stat routine, and
    converts tensors into a compact JSON-ready summary with optional saved rows.
    """
    T = int(info["seq_len"])
    metadata = info["metadata"]
    metadata = dict(metadata)
    metadata["cache_dir"] = str(cache_dir)
    query_by_label, all_q_positions = normalize_query_positions(spec, T, metadata)
    spans = normalize_spans(spec, T)
    local_windows = [int(w) for w in (spec.get("local_windows") or [])]

    attention_mask_path = cache_dir / "attention_mask.pt"
    if attention_mask_path.exists():
        key_padding_mask = load_tensor(attention_mask_path)[0].to(device=q.device).bool()
    else:
        key_padding_mask = None

    input_ids = load_tensor(cache_dir / "input_ids.pt")[0].tolist()
    tokens = load_cache_tokens(cache_dir)

    if not all_q_positions:
        return {"query_positions": query_by_label, "rows": {}}, None

    qpos_tensor = torch.tensor(all_q_positions, dtype=torch.long, device=q.device)
    q_block = q[qpos_tensor]
    stats = compute_query_block_stats(
        q_block=q_block,
        k=k,
        query_positions=qpos_tensor,
        spans=spans,
        key_padding_mask=key_padding_mask,
        scaling=float(info["scaling"]),
        key_block_size=key_block_size,
        topk=topk,
        local_windows=local_windows,
        return_rows=save_full_rows,
    )

    pos_to_row = {p: i for i, p in enumerate(all_q_positions)}
    rows_out: Dict[str, Dict] = {}
    for pos in all_q_positions:
        i = pos_to_row[pos]
        labels = [label for label, positions in query_by_label.items() if pos in positions]
        item = token_repr(tokens, input_ids, pos)
        item["labels"] = labels
        item["entropy_nats"] = float(stats["entropy"][i].item())
        item["span_mass"] = {name: float(vals[i].item()) for name, vals in stats["span_mass"].items()}
        item["local_window_mass"] = {name: float(vals[i].item()) for name, vals in stats["local_mass"].items()}
        if topk > 0:
            top_items = []
            vals = stats["topk_values"][i].tolist()
            idxs = stats["topk_indices"][i].tolist()
            probs = stats["topk_probs"][i].tolist()
            for logit, idx, prob in zip(vals, idxs, probs):
                if idx < 0 or not math.isfinite(float(logit)):
                    continue
                tok = token_repr(tokens, input_ids, int(idx))
                tok["logit"] = float(logit)
                tok["prob"] = float(prob)
                top_items.append(tok)
            item["topk"] = top_items
        rows_out[str(pos)] = item

    row_path = None
    if save_full_rows and "rows" in stats:
        row_path = cache_dir / f"attention_rows_layer_{layer:02d}_head_{head:02d}.pt"
        torch.save({"query_positions": all_q_positions, "rows": stats["rows"]}, row_path)

    return {
        "query_positions_by_label": query_by_label,
        "rows": rows_out,
    }, row_path


@torch.no_grad()
def average_window_received(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    info: Dict,
    cache_dir: Path,
    spec: Dict,
    key_block_size: int,
    query_block_size: int,
) -> Dict[str, Dict]:
    """
    Measure how much attention a target span receives from later query tokens.
    
    For each ``window_received`` entry, this function averages the normalized
    attention mass assigned to a target span over a query range such as all later
    tokens, the last N tokens, or tokens after a needle marker.
    """
    T = int(info["seq_len"])
    spans = normalize_spans(spec, T)
    attention_mask_path = cache_dir / "attention_mask.pt"
    if attention_mask_path.exists():
        key_padding_mask = load_tensor(attention_mask_path)[0].to(device=q.device).bool()
    else:
        key_padding_mask = None

    results: Dict[str, Dict] = {}
    entries = spec.get("window_received") or []
    for entry in entries:
        name = str(entry.get("name") or entry.get("span_name") or "window")
        if "span_name" in entry:
            span_name = str(entry["span_name"])
            if span_name not in spans:
                raise ValueError(f"window_received entry refers to unknown span_name={span_name!r}")
            intervals = spans[span_name]
        else:
            intervals = normalize_intervals(entry.get("span"), T)
        if not intervals:
            continue

        default_start = max(e for _, e in intervals)
        q_start = resolve_index(entry.get("query_start", default_start), T, end_exclusive=False)
        q_end = resolve_index(entry.get("query_end", None), T, end_exclusive=True)
        q_start, q_end = clamp_interval(q_start, q_end, T)
        if q_end <= q_start:
            results[name] = {
                "intervals": intervals,
                "query_range": [q_start, q_end],
                "num_queries": 0,
                "mean_mass": None,
                "sum_mass": 0.0,
            }
            continue

        q_positions_all = torch.arange(q_start, q_end, dtype=torch.long, device=q.device)
        if key_padding_mask is not None:
            q_positions_all = q_positions_all[key_padding_mask[q_positions_all]]

        total_mass = 0.0
        total_entropy = 0.0
        count = 0
        for qb in range(0, int(q_positions_all.numel()), query_block_size):
            qpos = q_positions_all[qb : qb + query_block_size]
            if qpos.numel() == 0:
                continue
            stats = compute_query_block_stats(
                q_block=q[qpos],
                k=k,
                query_positions=qpos,
                spans={"target": intervals},
                key_padding_mask=key_padding_mask,
                scaling=float(info["scaling"]),
                key_block_size=key_block_size,
                topk=0,
                local_windows=(),
                return_rows=False,
            )
            masses = stats["span_mass"]["target"]
            entropy = stats["entropy"]
            total_mass += float(masses.sum().item())
            total_entropy += float(entropy.sum().item())
            count += int(masses.numel())

        results[name] = {
            "intervals": [[s, e] for s, e in intervals],
            "query_range": [q_start, q_end],
            "num_queries": count,
            "mean_mass": (total_mass / count) if count else None,
            "sum_mass": total_mass,
            "mean_entropy_nats": (total_entropy / count) if count else None,
        }
    return results


def main() -> None:
    """
    Command-line entry point for offline analysis of one layer and one head.
    
    The function loads cached raw Q/K, reconstructs the selected head's Q/K,
    computes selected-row and window-received statistics, and writes a JSON
    report without rerunning a full Qwen3 forward pass.
    """
    parser = argparse.ArgumentParser(description="Analyze Qwen3 Q/K cache for one layer/head.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--spec-json", default=None, help="Defaults to cache_dir/analysis_spec.json")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute-dtype", default="fp32", choices=list(COMPUTE_DTYPES.keys()))
    parser.add_argument("--key-block-size", type=int, default=8192)
    parser.add_argument("--query-block-size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--save-full-rows", action="store_true")
    parser.add_argument(
        "--save-full-matrix",
        action="store_true",
        help="Materialize and save the full [seq_len, seq_len] attention matrix for the selected layer/head. Use only for short contexts.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if args.spec_json is not None:
        spec_path = Path(args.spec_json)
        spec = load_json(spec_path) if spec_path.exists() else {"query_positions": {"last": [-1]}, "spans": {"bos": [[0, 1]]}}
    else:
        spec = load_cache_analysis_spec(cache_dir)
        if not spec:
            spec = {"query_positions": {"last": [-1]}, "spans": {"bos": [[0, 1]]}}

    device = torch.device(args.device)
    compute_dtype = COMPUTE_DTYPES[args.compute_dtype]
    q, k, info = reconstruct_single_head_qk(
        cache_dir=cache_dir,
        layer=args.layer,
        head=args.head,
        device=device,
        compute_dtype=compute_dtype,
    )

    selected_rows, row_path = summarize_selected_rows(
        q=q,
        k=k,
        info=info,
        cache_dir=cache_dir,
        spec=spec,
        layer=args.layer,
        head=args.head,
        key_block_size=args.key_block_size,
        topk=args.topk,
        save_full_rows=args.save_full_rows,
    )

    received = average_window_received(
        q=q,
        k=k,
        info=info,
        cache_dir=cache_dir,
        spec=spec,
        key_block_size=args.key_block_size,
        query_block_size=args.query_block_size,
    )

    matrix_path = None
    if args.save_full_matrix:
        matrix_path = save_full_attention_matrix(
            q=q,
            k=k,
            info=info,
            cache_dir=cache_dir,
            layer=args.layer,
            head=args.head,
        )

    result = {
        "cache_dir": str(cache_dir),
        "layer": args.layer,
        "head": args.head,
        "kv_head": info["kv_head"],
        "num_key_value_groups": info["num_key_value_groups"],
        "seq_len": info["seq_len"],
        "head_dim": info["head_dim"],
        "selected_query_rows": selected_rows,
        "window_received": received,
        "full_rows_path": str(row_path) if row_path is not None else None,
        "full_matrix_path": str(matrix_path) if matrix_path is not None else None,
    }

    out_path = Path(args.out_json) if args.out_json is not None else cache_dir / f"stats_layer_{args.layer:02d}_head_{args.head:02d}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
