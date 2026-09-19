from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from .spec import V442Config


REGION_NAMES = (
    "cue",
    "needle_span",
    "needle_end",
    "passage",
    "question",
    "other_prompt",
    "trace",
    "final",
)


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first = value[..., : value.shape[-1] // 2]
    second = value[..., value.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def apply_saved_rope(
    value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply saved model-produced RoPE tables to [time, heads, dim] Q/K."""

    if value.ndim != 3 or cos.ndim != 2 or sin.shape != cos.shape:
        raise ValueError("Expected value [T,H,D] and cos/sin [T,R]")
    if value.shape[0] != cos.shape[0]:
        raise ValueError("RoPE time dimension does not match Q/K")
    rotated_width = int(cos.shape[-1])
    if rotated_width > value.shape[-1]:
        cos = cos[..., : value.shape[-1]]
        sin = sin[..., : value.shape[-1]]
        rotated_width = value.shape[-1]
    if rotated_width % 2 != 0:
        raise ValueError("RoPE width must be even")
    prefix = value[..., :rotated_width]
    cos_value = cos[:, None, :].to(device=value.device, dtype=value.dtype)
    sin_value = sin[:, None, :].to(device=value.device, dtype=value.dtype)
    rotated = prefix * cos_value + rotate_half(prefix) * sin_value
    if rotated_width == value.shape[-1]:
        return rotated
    return torch.cat((rotated, value[..., rotated_width:]), dim=-1)


def _interval_mask(length: int, intervals: Sequence[Sequence[int]]) -> torch.Tensor:
    mask = torch.zeros(length, dtype=torch.bool)
    for start, end in intervals:
        lo = max(0, min(length, int(start)))
        hi = max(lo, min(length, int(end)))
        mask[lo:hi] = True
    return mask


def _region_masks(manifest: dict[str, Any]) -> torch.Tensor:
    length = int(manifest["sequence_length"])
    prompt_len = int(manifest["prompt_token_count"])
    boundaries = manifest["boundaries"]
    trace_span = (
        prompt_len + int(boundaries["trace_start"]),
        prompt_len + int(boundaries["trace_end"]),
    )
    final_span = (
        prompt_len + int(boundaries["final_start"]),
        prompt_len + int(boundaries["final_end"]),
    )
    cue = [] if manifest.get("cue_span") is None else [manifest["cue_span"]]
    needle_spans = manifest.get("needle_spans", [])
    needle_ends = [[int(value), int(value) + 1] for value in manifest.get("needle_end_positions", [])]
    passage = [manifest["passage_span"]]
    question = [manifest["question_span"]]
    prompt_known = _interval_mask(length, cue + passage + question)
    prompt_mask = _interval_mask(length, [[0, prompt_len]])
    other_prompt = prompt_mask & ~prompt_known
    masks = [
        _interval_mask(length, cue),
        _interval_mask(length, needle_spans),
        _interval_mask(length, needle_ends),
        _interval_mask(length, passage),
        _interval_mask(length, question),
        other_prompt,
        _interval_mask(length, [trace_span]),
        _interval_mask(length, [final_span]),
    ]
    return torch.stack(masks, dim=-1)


def _bin_indices(size: int, bins: int) -> torch.Tensor:
    if size <= 0:
        return torch.empty(0, dtype=torch.long)
    positions = torch.arange(size, dtype=torch.long)
    return torch.clamp((positions * bins) // size, max=bins - 1)


def _softcap(scores: torch.Tensor, value: float | None) -> torch.Tensor:
    if value is None:
        return scores
    return torch.tanh(scores / float(value)) * float(value)


def _load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected tensor at {path}")
    return value


@torch.inference_mode()
def reconstruct_attention_summary(
    capture_dir: str | Path,
    *,
    config: V442Config,
    device: str | torch.device = "cpu",
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(capture_dir)
    manifest = json.loads((root / "capture_manifest.json").read_text(encoding="utf-8"))
    output_path = root / "attention_summary.pt"
    metadata_path = root / "attention_summary.json"
    if output_path.exists() and metadata_path.exists() and not overwrite:
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    compute_device = torch.device(device)
    compute_dtype = getattr(torch, config.attention_compute_dtype)
    query_positions = _load_tensor(root / "query_positions.pt").long()
    query_roles = [str(value) for value in manifest["query_roles"]]
    if len(query_roles) != len(query_positions):
        raise RuntimeError("query_roles/query_positions length mismatch")
    trace_row_indices = torch.tensor(
        [index for index, role in enumerate(query_roles) if role == "trace"],
        dtype=torch.long,
    )
    answer_query_indices = torch.tensor(
        [index for index, role in enumerate(query_roles) if role == "answer_query"],
        dtype=torch.long,
    )
    bins = int(config.attention_trace_bins)
    trace_query_bins = _bin_indices(len(trace_row_indices), bins)
    trace_key_start = int(manifest["prompt_token_count"]) + int(
        manifest["boundaries"]["trace_start"]
    )
    trace_key_end = int(manifest["prompt_token_count"]) + int(
        manifest["boundaries"]["trace_end"]
    )
    trace_key_bins = _bin_indices(max(0, trace_key_end - trace_key_start), bins)
    region_masks = _region_masks(manifest)

    layer_payloads: dict[int, dict[str, torch.Tensor]] = {}
    compact_rows: list[dict[str, Any]] = []
    for layer_row in manifest["layers"]:
        layer = int(layer_row["layer"])
        q = _load_tensor(root / f"layer_{layer:02d}_q_norm.pt").to(
            device=compute_device, dtype=compute_dtype
        )
        k = _load_tensor(root / str(layer_row["k_file"])).to(
            device=compute_device, dtype=compute_dtype
        )
        rope = torch.load(
            root / f"layer_{layer:02d}_rope.pt",
            map_location="cpu",
            weights_only=True,
        )
        cos = rope["cos"].to(device=compute_device, dtype=compute_dtype)
        sin = rope["sin"].to(device=compute_device, dtype=compute_dtype)
        q = apply_saved_rope(q, cos[query_positions.to(cos.device)], sin[query_positions.to(sin.device)])
        k = apply_saved_rope(k, cos, sin)
        num_queries, num_heads, head_dim = q.shape
        seq_len, num_kv_heads, k_head_dim = k.shape
        if head_dim != k_head_dim or num_heads % num_kv_heads != 0:
            raise RuntimeError(
                f"Layer {layer} incompatible Q/K shapes: q={tuple(q.shape)} k={tuple(k.shape)}"
            )
        scaling = layer_row.get("scaling")
        scaling = head_dim**-0.5 if scaling is None else float(scaling)
        softcap = layer_row.get("softcap")
        sliding_window = (
            int(layer_row["sliding_window"])
            if layer_row.get("is_sliding") and layer_row.get("sliding_window")
            else None
        )

        head_region_sum = torch.zeros(
            num_heads,
            len(REGION_NAMES),
            dtype=compute_dtype,
            device=compute_device,
        )
        head_trace_count = torch.zeros(
            num_heads, dtype=torch.long, device=compute_device
        )
        answer_region_sum = torch.zeros_like(head_region_sum)
        answer_last_region = torch.zeros_like(head_region_sum)
        answer_count = torch.zeros_like(head_trace_count)
        trace_map_sum = torch.zeros(
            bins,
            len(REGION_NAMES),
            dtype=compute_dtype,
            device=compute_device,
        )
        trace_map_count = torch.zeros(
            bins, dtype=torch.long, device=compute_device
        )
        trace_to_trace_sum = torch.zeros(
            bins, bins, dtype=compute_dtype, device=compute_device
        )
        entropy_sum = torch.zeros(
            num_heads, dtype=compute_dtype, device=compute_device
        )

        masks_device = region_masks.to(device=compute_device, dtype=compute_dtype)
        key_positions = torch.arange(seq_len, device=compute_device)
        trace_length = max(0, trace_key_end - trace_key_start)
        trace_bin_matrix = torch.zeros(
            trace_length, bins, device=compute_device, dtype=compute_dtype
        )
        if trace_key_end > trace_key_start:
            trace_bin_matrix[
                torch.arange(trace_length, device=compute_device),
                trace_key_bins.to(compute_device),
            ] = 1

        query_block = int(config.attention_query_block_size)
        groups = num_heads // num_kv_heads
        kv_head_indices = torch.arange(num_heads, device=compute_device) // groups
        # [H, D, S], shared by every query block in this layer.
        keys_by_head = k[:, kv_head_indices].permute(1, 2, 0).contiguous()
        query_trace_bins = torch.full(
            (num_queries,), -1, dtype=torch.long, device=compute_device
        )
        if len(trace_row_indices):
            query_trace_bins[trace_row_indices.to(compute_device)] = trace_query_bins.to(
                compute_device
            )
        answer_last_index = (
            int(answer_query_indices[-1]) if len(answer_query_indices) else None
        )
        for q_start in range(0, num_queries, query_block):
            q_end = min(num_queries, q_start + query_block)
            # [H, B, D] @ [H, D, S] -> [H, B, S]
            queries = q[q_start:q_end].permute(1, 0, 2)
            scores = torch.matmul(queries, keys_by_head) * scaling
            scores = _softcap(scores, softcap)
            absolute_queries = query_positions[q_start:q_end].to(compute_device)
            allowed = key_positions[None, :] <= absolute_queries[:, None]
            if sliding_window is not None:
                allowed &= key_positions[None, :] >= (
                    absolute_queries[:, None] - sliding_window + 1
                )
            scores = scores.masked_fill(~allowed[None, :, :], -torch.inf)
            probabilities = torch.softmax(scores.float(), dim=-1)
            region_mass = probabilities @ masks_device.float()  # [H, B, R]
            entropy = -(
                probabilities
                * torch.where(
                    probabilities > 0,
                    probabilities.log(),
                    torch.zeros_like(probabilities),
                )
            ).sum(dim=-1)  # [H, B]

            block_roles = query_roles[q_start:q_end]
            trace_local = torch.tensor(
                [role == "trace" for role in block_roles],
                dtype=torch.bool,
                device=compute_device,
            )
            answer_local = torch.tensor(
                [role == "answer_query" for role in block_roles],
                dtype=torch.bool,
                device=compute_device,
            )
            trace_count = int(trace_local.sum())
            if trace_count:
                selected_regions = region_mass[:, trace_local, :]
                head_region_sum += selected_regions.sum(dim=1)
                head_trace_count += trace_count
                entropy_sum += entropy[:, trace_local].sum(dim=1)
                selected_bins = query_trace_bins[q_start:q_end][trace_local]
                trace_map_sum.index_add_(
                    0, selected_bins, selected_regions.sum(dim=0)
                )
                trace_map_count.index_add_(
                    0,
                    selected_bins,
                    torch.full(
                        (trace_count,),
                        num_heads,
                        dtype=torch.long,
                        device=compute_device,
                    ),
                )
                if trace_length:
                    trace_key_mass = (
                        probabilities[
                            :, trace_local, trace_key_start:trace_key_end
                        ]
                        @ trace_bin_matrix.float()
                    )
                    trace_to_trace_sum.index_add_(
                        0, selected_bins, trace_key_mass.sum(dim=0)
                    )
            answer_tokens = int(answer_local.sum())
            if answer_tokens:
                answer_region_sum += region_mass[:, answer_local, :].sum(dim=1)
                answer_count += answer_tokens
            if answer_last_index is not None and q_start <= answer_last_index < q_end:
                answer_last_region.copy_(region_mass[:, answer_last_index - q_start, :])

        head_region_mean = (
            head_region_sum / head_trace_count.clamp_min(1)[:, None]
        ).cpu()
        answer_region_mean = (
            answer_region_sum / answer_count.clamp_min(1)[:, None]
        ).cpu()
        answer_last_region = answer_last_region.cpu()
        trace_map_mean = (
            trace_map_sum / trace_map_count.clamp_min(1)[:, None]
        ).cpu()
        # trace_map_count includes heads, so the same denominator applies to
        # each key-bin row accumulated for a query bin.
        trace_to_trace_mean = (
            trace_to_trace_sum / trace_map_count.clamp_min(1)[:, None]
        ).cpu()
        entropy_mean = (
            entropy_sum / head_trace_count.clamp_min(1)
        ).cpu()
        head_trace_count = head_trace_count.cpu()
        answer_count = answer_count.cpu()
        trace_map_count = trace_map_count.cpu()
        layer_payloads[layer] = {
            "head_region_mean": head_region_mean.float(),
            "answer_query_region_mean": answer_region_mean.float(),
            "answer_query_last_region": answer_last_region.float(),
            "trace_region_map": trace_map_mean.float(),
            "trace_to_trace_map": trace_to_trace_mean.float(),
            "head_trace_entropy_mean": entropy_mean.float(),
            "trace_query_bin_counts": trace_map_count,
        }
        for head in range(num_heads):
            row = {
                "layer": layer,
                "head": head,
                "trace_queries": int(head_trace_count[head]),
                "answer_query_tokens": int(answer_count[head]),
                "trace_entropy": float(entropy_mean[head]),
            }
            for index, name in enumerate(REGION_NAMES):
                row[f"trace_to_{name}"] = float(head_region_mean[head, index])
                row[f"answer_query_to_{name}"] = float(answer_region_mean[head, index])
                row[f"answer_query_last_to_{name}"] = float(
                    answer_last_region[head, index]
                )
            compact_rows.append(row)

    payload = {
        "schema_version": "realistic_niah_v4_4_2_attention_summary_v1",
        "region_names": REGION_NAMES,
        "trace_bins": bins,
        "layers": layer_payloads,
    }
    temporary = output_path.with_name(output_path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    csv_path = root / "attention_head_summary.csv.gz"
    csv_tmp = csv_path.with_name(csv_path.name + ".tmp")
    if compact_rows:
        with gzip.open(csv_tmp, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(compact_rows[0]))
            writer.writeheader()
            writer.writerows(compact_rows)
        csv_tmp.replace(csv_path)
    metadata = {
        "schema_version": "realistic_niah_v4_4_2_attention_summary_manifest_v1",
        "capture_manifest": str((root / "capture_manifest.json").resolve()),
        "attention_summary": str(output_path.resolve()),
        "head_summary": str(csv_path.resolve()),
        "layers": sorted(layer_payloads),
        "trace_bins": bins,
        "region_names": list(REGION_NAMES),
        "compute_device": str(compute_device),
        "compute_dtype": config.attention_compute_dtype,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def summarize_hidden_states(capture_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(capture_dir)
    manifest = json.loads((root / "capture_manifest.json").read_text(encoding="utf-8"))
    roles = [str(value) for value in manifest["query_roles"]]
    rows: list[dict[str, Any]] = []
    for layer_row in manifest["layers"]:
        layer = int(layer_row["layer"])
        hidden = _load_tensor(root / f"layer_{layer:02d}_hidden.pt").float()
        for role in ("trace", "answer_query", "answer"):
            indices = [index for index, value in enumerate(roles) if value == role]
            if not indices:
                continue
            selected = hidden[indices]
            norms = selected.norm(dim=-1)
            adjacent_cosine = None
            if len(selected) > 1:
                adjacent_cosine = float(
                    torch.nn.functional.cosine_similarity(
                        selected[:-1], selected[1:], dim=-1
                    ).mean()
                )
            rows.append(
                {
                    "layer": layer,
                    "role": role,
                    "tokens": len(indices),
                    "mean_norm": float(norms.mean()),
                    "std_norm": float(norms.std(unbiased=False)),
                    "adjacent_cosine": adjacent_cosine,
                }
            )
    return rows
