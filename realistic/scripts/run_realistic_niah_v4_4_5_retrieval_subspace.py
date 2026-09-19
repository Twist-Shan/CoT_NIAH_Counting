from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from realistic_niah_v4.behavior import parse_numeric_completion
from realistic_niah_v4.modeling import (
    DecoderAdapter,
    _bounded_logits_kwargs,
    _encoding_tensors,
    _is_prompt_prefill,
    _replace_output_tensor,
    _tensor_from_output,
    capture_post_block_states,
    generate_answer_completion,
    load_registered_model,
)
from realistic_niah_v4.prompts import PromptEncoding, render_v4_prompt
from realistic_niah_v4.spec import V4Config, resolve_model_spec
from realistic_niah_v4.stimuli import load_stimuli
from realistic_niah_v4_4_3.interventions import (
    _score_candidate_sequences,
    candidate_sequence_metrics,
)
from realistic_niah_v4_4_5.restoration import (
    build_corruption_plan,
    corrupt_encoding,
    residual_patch_hook,
    segment_positions,
)


SCHEMA = "realistic_niah_v4_4_5_retrieval_subspace_detail_v1"


def csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("At least one integer is required")
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def completed_keys(path: Path) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    return {
        (int(row["seed"]), int(row["gold_count"]), str(row["condition"]))
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def projection_result(projection: nn.Module, value: torch.Tensor) -> torch.Tensor:
    device = next(projection.parameters()).device
    dtype = next(projection.parameters()).dtype
    tensor = value.to(device=device, dtype=dtype).reshape(1, 1, -1)
    return projection(tensor)[0, 0].detach().float().cpu()


def bank_output(
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    z: torch.Tensor,
) -> torch.Tensor:
    projection = adapter.output_projections[int(layer)]
    width = int(adapter.head_dims[int(layer)])
    zeros = torch.zeros_like(z)
    base = projection_result(projection, zeros)
    values: list[torch.Tensor] = []
    for head in heads:
        selected = zeros.clone()
        start = int(head) * width
        selected[start : start + width] = z[start : start + width]
        values.append(projection_result(projection, selected) - base)
    return torch.stack(values).sum(dim=0)


def deterministic_orthogonal_in_bank_span(
    adapter: DecoderAdapter,
    *,
    layer: int,
    heads: Sequence[int],
    basis: torch.Tensor,
    random_seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
    width_total = int(adapter.num_heads[int(layer)] * adapter.head_dims[int(layer)])
    mask = torch.zeros(width_total, dtype=torch.float32)
    width = int(adapter.head_dims[int(layer)])
    for head in heads:
        start = int(head) * width
        mask[start : start + width] = 1
    for _ in range(64):
        z = torch.randn(width_total, generator=generator) * mask
        projection = adapter.output_projections[int(layer)]
        vector = projection_result(projection, z) - projection_result(
            projection, torch.zeros_like(z)
        )
        vector = vector - (vector @ basis.T) @ basis
        norm = torch.linalg.vector_norm(vector)
        if float(norm) > 1e-8:
            return vector / norm
    raise RuntimeError("Could not construct an orthogonal bank-span control")


@contextmanager
def retrieval_block_hook(
    adapter: DecoderAdapter,
    encoding: PromptEncoding,
    *,
    layer: int,
    heads: Sequence[int],
    mean: torch.Tensor,
    basis: torch.Tensor,
    control_direction: torch.Tensor,
    mode: str,
) -> Iterator[dict[str, Any]]:
    """Remove the centered rank-3 bank component or an equal-norm control."""

    if mode not in {"aligned", "orthogonal"}:
        raise ValueError("mode must be aligned or orthogonal")
    captured_z: dict[str, torch.Tensor | None] = {"value": None}
    audit: dict[str, Any] = {
        "applications": 0,
        "aligned_norms": [],
        "removed_norms": [],
        "orthogonality": [],
    }

    def pre_hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("Retrieval block saw no pre-O tensor")
        value = args[0]
        if not _is_prompt_prefill(value, encoding):
            return
        captured_z["value"] = (
            value[0, int(encoding.query_position)].detach().float().cpu()
        )

    def post_hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
        hidden = _tensor_from_output(output)
        if not _is_prompt_prefill(hidden, encoding):
            return output
        z = captured_z["value"]
        if z is None:
            raise RuntimeError("Post-O block did not receive its pre-O capture")
        bank = bank_output(adapter, layer=layer, heads=heads, z=z)
        centered = bank - mean
        aligned = (centered @ basis.T) @ basis
        aligned_norm = torch.linalg.vector_norm(aligned)
        if mode == "aligned":
            removed = aligned
        else:
            removed = control_direction * aligned_norm
        patched = hidden.clone()
        patched[:, int(encoding.query_position), :] -= removed.to(
            device=hidden.device, dtype=hidden.dtype
        )
        audit["applications"] += 1
        audit["aligned_norms"].append(float(aligned_norm))
        audit["removed_norms"].append(float(torch.linalg.vector_norm(removed)))
        audit["orthogonality"].append(
            float(torch.max(torch.abs(removed.float() @ basis.T)))
            if mode == "orthogonal"
            else 0.0
        )
        captured_z["value"] = None
        return _replace_output_tensor(output, patched)

    pre = adapter.output_projections[int(layer)].register_forward_pre_hook(pre_hook)
    post = adapter.attentions[int(layer)].register_forward_hook(post_hook)
    try:
        yield audit
    finally:
        pre.remove()
        post.remove()


def strict_fields(result: Mapping[str, Any], gold: int) -> dict[str, Any]:
    parsed = parse_numeric_completion(str(result.get("completion_text", "")))
    prediction = parsed.get("parsed_count")
    value = None if prediction is None else int(prediction)
    return {
        "strict_prediction": value,
        "strict_correct": bool(value == int(gold)),
        "strict_absolute_error": 10 if value is None else abs(value - int(gold)),
        "strict_completion": str(result.get("completion_text", "")),
        "strict_format_valid": bool(parsed.get("format_valid", False)),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    tokenizer: Any,
    encoding: PromptEncoding,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    input_ids, attention_mask = _encoding_tensors(model, encoding)
    prefill = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        **_bounded_logits_kwargs(model),
    )
    candidate = _score_candidate_sequences(model, encoding, prefill)
    strict = generate_answer_completion(
        model, tokenizer, encoding, max_new_tokens=int(max_new_tokens)
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {
        **candidate_sequence_metrics(candidate.candidate_log_scores, encoding),
        **strict_fields(strict, int(encoding.count)),
        "elapsed_seconds": float(time.perf_counter() - started),
        "max_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "max_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
        ),
    }


def take_positions(
    captured: torch.Tensor,
    capture_positions: Sequence[int],
    selected: Sequence[int],
) -> torch.Tensor:
    lookup = {int(position): index for index, position in enumerate(capture_positions)}
    return captured[[lookup[int(position)] for position in selected]]


def summarize_hook(audit: Mapping[str, Any]) -> dict[str, Any]:
    aligned = np.asarray(audit["aligned_norms"], dtype=float)
    removed = np.asarray(audit["removed_norms"], dtype=float)
    return {
        "retrieval_block_applications": int(audit["applications"]),
        "aligned_component_norm_mean": float(aligned.mean()),
        "removed_component_norm_mean": float(removed.mean()),
        "norm_match_max_abs_delta": float(np.max(np.abs(aligned - removed))),
        "orthogonality_max_abs_dot": float(max(audit["orthogonality"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("Qwen3-8B", "Gemma4-E4B"))
    parser.add_argument("--stimuli", required=True)
    parser.add_argument(
        "--stimuli-config", default="configs/realistic_niah_v4_4_5_stimuli.json"
    )
    parser.add_argument(
        "--experiment-config",
        default="configs/realistic_niah_v4_4_5_span_restoration.json",
    )
    parser.add_argument("--basis-file", required=True)
    parser.add_argument("--retrieval-layer", type=int, required=True)
    parser.add_argument("--source-patch-layer", type=int, required=True)
    parser.add_argument("--seeds", type=csv_ints, required=True)
    parser.add_argument("--counts", type=csv_ints, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    experiment_path = Path(args.experiment_config).resolve()
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    config = V4Config.from_json(args.stimuli_config)
    model_label = str(args.model)
    retrieval_layer = int(args.retrieval_layer)
    source_layer = int(args.source_patch_layer)
    registered_heads = [
        (int(layer), int(head))
        for layer, head in experiment["retrieval_heads"][model_label]
    ]
    heads = tuple(head for layer, head in registered_heads if layer == retrieval_layer)
    if not heads:
        raise ValueError(f"No frozen heads registered at L{retrieval_layer}")
    basis_file = Path(args.basis_file).resolve()
    bases = torch.load(basis_file, map_location="cpu", weights_only=True)
    basis_key = f"{model_label}.L{retrieval_layer}"
    if basis_key not in bases:
        raise KeyError(f"Basis file lacks {basis_key}")
    mean = bases[basis_key]["mean"].float()
    basis = bases[basis_key]["components"].float()
    if basis.shape[0] != 3:
        raise RuntimeError("Retrieval intervention requires a rank-3 basis")

    rows = {
        (int(row["seed"]), int(row["gold_count"])): row
        for row in load_stimuli(args.stimuli)
        if str(row.get("design_variant")) == "v4.4"
        and int(row["seed"]) in set(args.seeds)
        and int(row["gold_count"]) in set(args.counts)
    }
    expected = {(seed, count) for seed in args.seeds for count in args.counts}
    if set(rows) != expected:
        raise RuntimeError(f"Missing selected stimuli: {sorted(expected - set(rows))}")
    output = Path(args.output_dir).resolve() / model_label
    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "detail.jsonl"
    complete = completed_keys(detail_path)
    spec = resolve_model_spec(model_label)
    provenance = {
        "schema_version": "realistic_niah_v4_4_5_retrieval_subspace_run_v1",
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "model": model_label,
        "model_revision": spec.revision,
        "stimuli_sha256": sha256(Path(args.stimuli)),
        "basis_file": str(basis_file),
        "basis_sha256": sha256(basis_file),
        "basis_key": basis_key,
        "source_patch_layer": source_layer,
        "retrieval_layer": retrieval_layer,
        "heads": list(heads),
        "seeds": list(args.seeds),
        "counts": list(args.counts),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    (output / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    model, tokenizer, adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=experiment["model_torch_dtype"],
        attention_backend=experiment["attention_prefix_backend"],
    )
    control = deterministic_orthogonal_in_bank_span(
        adapter,
        layer=retrieval_layer,
        heads=heads,
        basis=basis,
        random_seed=20260813 + retrieval_layer,
    )
    if float(torch.max(torch.abs(control @ basis.T))) > 1e-4:
        raise RuntimeError("Constructed retrieval control is not orthogonal")

    for seed in args.seeds:
        for count in args.counts:
            encoding = render_v4_prompt(
                rows[(seed, count)],
                tokenizer=tokenizer,
                model_spec=spec,
                config=config,
                answer_format="numeric",
            )
            plan = build_corruption_plan(encoding)
            corrupted, changed = corrupt_encoding(
                encoding, plan, condition="needle_corrupt"
            )
            positions = segment_positions(plan, condition="needle")
            _logits, clean_states = capture_post_block_states(
                model,
                adapter,
                encoding,
                positions,
                layers=(source_layer,),
            )
            replacement = take_positions(clean_states[source_layer], positions, positions)
            for condition in (
                "clean_aligned_block",
                "clean_orthogonal_block",
                "restored_aligned_block",
                "restored_orthogonal_block",
            ):
                key = (int(seed), int(count), condition)
                if key in complete:
                    continue
                mode = "aligned" if "aligned" in condition else "orthogonal"
                current = corrupted if condition.startswith("restored") else encoding
                source_audit: dict[str, int] | None = None
                with retrieval_block_hook(
                    adapter,
                    current,
                    layer=retrieval_layer,
                    heads=heads,
                    mean=mean,
                    basis=basis,
                    control_direction=control,
                    mode=mode,
                ) as block_audit:
                    if condition.startswith("restored"):
                        with residual_patch_hook(
                            adapter,
                            current,
                            layer=source_layer,
                            positions=positions,
                            replacement=replacement,
                        ) as source_audit:
                            metrics = evaluate(
                                model,
                                tokenizer,
                                current,
                                max_new_tokens=experiment[
                                    "strict_generation_max_new_tokens"
                                ],
                            )
                    else:
                        metrics = evaluate(
                            model,
                            tokenizer,
                            current,
                            max_new_tokens=experiment[
                                "strict_generation_max_new_tokens"
                            ],
                        )
                if int(block_audit["applications"]) != 2:
                    raise RuntimeError("Retrieval block must apply to two prefills")
                if source_audit is not None and int(source_audit["count"]) != 2:
                    raise RuntimeError("Source restoration must apply to two prefills")
                row = {
                    "schema_version": SCHEMA,
                    "model_label": model_label,
                    "seed": int(seed),
                    "gold_count": int(count),
                    "condition": condition,
                    "block_mode": mode,
                    "source_patch_layer": source_layer,
                    "retrieval_layer": retrieval_layer,
                    "retrieval_heads": list(heads),
                    "source_restoration": bool(condition.startswith("restored")),
                    "source_patch_applications": (
                        0 if source_audit is None else int(source_audit["count"])
                    ),
                    "token_budget": int(plan.token_budget),
                    "changed_tokens": int(changed),
                    **summarize_hook(block_audit),
                    **metrics,
                }
                append_jsonl(detail_path, row)
                complete.add(key)
                print(
                    f"[retrieval-subspace] {model_label} seed={seed} N={count} "
                    f"{condition} E={metrics['expected_count']:.3f} "
                    f"strict={metrics['strict_prediction']}",
                    flush=True,
                )
    audit = {
        "status": "complete",
        "model": model_label,
        "rows": len(complete),
        "detail": str(detail_path),
    }
    (output / "complete.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
