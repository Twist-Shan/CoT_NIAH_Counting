from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from realistic_niah_v4.behavior import parse_numeric_completion
from realistic_niah_v4.modeling import (
    _is_prompt_prefill,
    _tensor_from_output,
    capture_post_block_states,
    generate_answer_completion,
    load_registered_model,
)
from realistic_niah_v4.prompts import PromptEncoding, render_v4_prompt
from realistic_niah_v4.spec import V4Config, resolve_model_spec
from realistic_niah_v4.stimuli import load_stimuli
from realistic_niah_v4_4_3.interventions import (
    candidate_sequence_metrics,
    capture_query_bundle,
)
from realistic_niah_v4_4_5.restoration import (
    active_broad_metrics,
    build_corruption_plan,
    corrupt_encoding,
    generate_answer_completion_from_prefill,
    residual_patch_hook,
    segment_positions,
)


SCHEMA = "realistic_niah_v4_4_5_span_restoration_detail_v1"


def csv_ints(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(item) for item in value.split(",") if item.strip())


def csv_strings(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_completed(path: Path) -> set[tuple[int, int, str, int]]:
    if not path.exists():
        return set()
    completed: set[tuple[int, int, str, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        completed.add(
            (
                int(row["seed"]),
                int(row["gold_count"]),
                str(row["condition"]),
                int(row["patch_layer"]),
            )
        )
    return completed


def strict_fields(result: Mapping[str, Any], gold: int) -> dict[str, Any]:
    labels = parse_numeric_completion(str(result.get("completion_text", "")))
    prediction = labels.get("parsed_count")
    parsed = None if prediction is None else int(prediction)
    return {
        "strict_prediction": parsed,
        "strict_correct": bool(parsed == int(gold)),
        "strict_absolute_error": (
            10 if parsed is None else abs(int(parsed) - int(gold))
        ),
        "strict_completion": str(result.get("completion_text", "")),
        "strict_format_valid": bool(labels.get("format_valid", False)),
    }


def selected_head_writes(
    adapter: Any,
    bundle: Any,
    heads: Sequence[tuple[int, int]],
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    grouped: dict[int, list[int]] = defaultdict(list)
    for layer, head in heads:
        grouped[int(layer)].append(int(head))
    for layer, layer_heads in grouped.items():
        z = bundle.z_by_layer[layer].float()
        width = int(adapter.head_dims[layer])
        projection = adapter.output_projections[layer]
        device = next(projection.parameters()).device
        dtype = next(projection.parameters()).dtype
        zero = torch.zeros((1, 1, z.numel()), dtype=dtype, device=device)
        base = projection(zero)[0, 0].detach().float().cpu()
        bank: list[torch.Tensor] = []
        for head in layer_heads:
            start = head * width
            end = start + width
            value = zero.clone()
            value[0, 0, start:end] = z[start:end].to(device=device, dtype=dtype)
            output = projection(value)[0, 0].detach().float().cpu() - base
            result[f"L{layer}H{head}.z"] = z[start:end].to(torch.float16)
            result[f"L{layer}H{head}.o"] = output.to(torch.float16)
            bank.append(output)
        result[f"L{layer}.bank_o"] = torch.stack(bank).sum(dim=0).to(torch.float16)
    return result


def capture_answer_states(
    adapter: Any,
    encoding: PromptEncoding,
    layers: Iterable[int],
) -> tuple[dict[int, torch.Tensor], list[Any]]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in sorted({int(value) for value in layers}):

        def hook(_module: Any, _args: tuple[Any, ...], output: Any, *, layer=layer):
            hidden = _tensor_from_output(output)
            if _is_prompt_prefill(hidden, encoding):
                captured[layer] = (
                    hidden[0, int(encoding.query_position)]
                    .detach()
                    .to(device="cpu", dtype=torch.float16)
                )

        handles.append(adapter.layers[layer].register_forward_hook(hook))
    return captured, handles


def save_payload_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def evaluate(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    encoding: PromptEncoding,
    *,
    retrieval_heads: Sequence[tuple[int, int]],
    answer_layers: Sequence[int],
    max_new_tokens: int,
    state_path: Path,
    cache_logit_tolerance: float,
    cache_probability_tv_tolerance: float,
    audit_cache_equivalence: bool,
    reuse_prefill_for_generation: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    retrieval_layers = sorted({int(layer) for layer, _head in retrieval_heads})
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    answer_states, handles = capture_answer_states(adapter, encoding, answer_layers)
    try:
        bundle = capture_query_bundle(
            model,
            adapter,
            encoding,
            layers=retrieval_layers,
            capture_attention=True,
            capture_values=False,
            cache_logit_tolerance=float(cache_logit_tolerance),
            cache_probability_total_variation_tolerance=float(
                cache_probability_tv_tolerance
            ),
            require_cache_candidate_argmax_agreement=True,
            audit_cache_equivalence=bool(audit_cache_equivalence),
            retain_prefill_output=bool(reuse_prefill_for_generation),
        )
    finally:
        for handle in handles:
            handle.remove()
    if reuse_prefill_for_generation:
        if bundle.reusable_prefill_output is None:
            raise RuntimeError("Reusable strict generation retained no prefill output")
        strict = generate_answer_completion_from_prefill(
            model,
            tokenizer,
            encoding,
            bundle.reusable_prefill_output,
            max_new_tokens=int(max_new_tokens),
        )
    else:
        strict = generate_answer_completion(
            model, tokenizer, encoding, max_new_tokens=int(max_new_tokens)
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metrics = candidate_sequence_metrics(bundle.candidate_log_scores, encoding)
    writes = selected_head_writes(adapter, bundle, retrieval_heads)
    payload: dict[str, Any] = {
        "answer_states": {str(key): value for key, value in answer_states.items()},
        "pre_o": {
            str(layer): value.to(torch.float16)
            for layer, value in bundle.z_by_layer.items()
        },
        "post_o_total": {
            str(layer): value.to(torch.float16)
            for layer, value in bundle.attention_output_by_layer.items()
        },
        "selected_head_writes": writes,
    }
    save_payload_atomic(state_path, payload)
    selected = {(int(layer), int(head)) for layer, head in retrieval_heads}
    broad_rows: list[dict[str, Any]] = []
    for layer in retrieval_layers:
        rows = active_broad_metrics(
            bundle.alpha_by_layer[layer],
            key_start=bundle.alpha_key_start_by_layer[layer],
            spans=encoding.needle_spans,
        )
        for row in rows:
            if (layer, int(row["head"])) not in selected:
                continue
            broad_rows.append(
                {
                    "layer": int(layer),
                    "head": int(row["head"]),
                    "needle_mass": float(row["needle_mass"]),
                    "coverage": float(row["coverage"]),
                    "broad_score": float(row["broad_score"]),
                    "span_masses": list(row["span_masses"]),
                }
            )
    return (
        {
            **metrics,
            **strict_fields(strict, int(encoding.count)),
            "elapsed_seconds": float(elapsed),
            "max_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else 0
            ),
            "max_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved())
                if torch.cuda.is_available()
                else 0
            ),
            "attention_cache_equivalence_audited": bool(
                bundle.attention_cache_equivalence_audited
            ),
            "attention_cache_candidate_logit_max_abs_delta": (
                float(bundle.attention_cache_candidate_logit_max_abs_delta)
                if bundle.attention_cache_equivalence_audited
                else None
            ),
            "attention_cache_candidate_centered_logit_max_abs_delta": (
                float(bundle.attention_cache_candidate_centered_logit_max_abs_delta)
                if bundle.attention_cache_equivalence_audited
                else None
            ),
            "attention_cache_candidate_argmax_agreement": (
                bool(bundle.attention_cache_candidate_argmax_agreement)
                if bundle.attention_cache_equivalence_audited
                else None
            ),
            "attention_cache_candidate_probability_total_variation": (
                float(bundle.attention_cache_candidate_probability_total_variation)
                if bundle.attention_cache_equivalence_audited
                else None
            ),
            "state_path": str(state_path),
            "strict_generation_reused_prefill": bool(reuse_prefill_for_generation),
        },
        broad_rows,
    )


def take_states(
    captured: torch.Tensor,
    captured_positions: Sequence[int],
    selected_positions: Sequence[int],
) -> torch.Tensor:
    lookup = {int(position): index for index, position in enumerate(captured_positions)}
    missing = sorted(set(int(value) for value in selected_positions) - set(lookup))
    if missing:
        raise RuntimeError(f"Captured state bank lacks positions: {missing}")
    return captured[[lookup[int(position)] for position in selected_positions]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run V4.4.5 layerwise clean-state restoration after token corruption."
    )
    parser.add_argument("--model", required=True, choices=("Qwen3-8B", "Gemma4-E4B"))
    parser.add_argument("--stimuli", required=True)
    parser.add_argument(
        "--stimuli-config", default="configs/realistic_niah_v4_4_5_stimuli.json"
    )
    parser.add_argument(
        "--experiment-config",
        default="configs/realistic_niah_v4_4_5_span_restoration.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seeds")
    parser.add_argument("--counts")
    parser.add_argument("--layers")
    parser.add_argument("--patch-kinds")
    parser.add_argument(
        "--cache-logit-tolerance",
        type=float,
        default=0.75,
        help=(
            "Maximum common-shift-invariant first-token candidate-logit delta "
            "between full and cached attention-reconstruction forwards."
        ),
    )
    parser.add_argument(
        "--cache-probability-tv-tolerance",
        type=float,
        default=0.10,
        help=(
            "Maximum total-variation distance between full and cached "
            "first-token candidate distributions; their argmax must also agree."
        ),
    )
    parser.add_argument(
        "--skip-cache-equivalence-audit",
        action="store_true",
        help=(
            "Use the original V4 cache-reconstructed answer-query attention "
            "rows without comparing their logits with the causal full forward."
        ),
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Capture selected baselines without any restoration interventions.",
    )
    parser.add_argument(
        "--reuse-prefill-for-generation",
        action="store_true",
        help=(
            "Clone the candidate-scoring KV branch and continue strict greedy "
            "generation from the retained patched full-prompt prefill."
        ),
    )
    parser.add_argument(
        "--baseline-conditions",
        default="clean,needle_corrupt,ordinary_corrupt",
        help="Comma-separated subset of clean,needle_corrupt,ordinary_corrupt.",
    )
    args = parser.parse_args()

    cache_probability_tv_tolerance = float(args.cache_probability_tv_tolerance)
    audit_cache_equivalence = not bool(args.skip_cache_equivalence_audit)

    experiment_path = Path(args.experiment_config).resolve()
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if experiment.get("schema_version") != "realistic_niah_v4_4_5_span_restoration_v1":
        raise ValueError("Unexpected V4.4.5 restoration schema")
    v4_config = V4Config.from_json(args.stimuli_config)
    model_label = str(args.model)
    seeds = csv_ints(args.seeds) or tuple(experiment["pilot_seeds"])
    counts = csv_ints(args.counts) or tuple(experiment["pilot_counts"])
    layers = csv_ints(args.layers) or tuple(experiment["pilot_layers"][model_label])
    patch_kinds = csv_strings(args.patch_kinds) or tuple(experiment["patch_kinds"])
    baseline_conditions = csv_strings(args.baseline_conditions) or ()
    allowed_baselines = {"clean", "needle_corrupt", "ordinary_corrupt"}
    if not baseline_conditions or not set(baseline_conditions).issubset(
        allowed_baselines
    ):
        raise ValueError("Invalid --baseline-conditions")
    allowed = {"needle_endpoint", "needle_full", "ordinary_full"}
    if not set(patch_kinds).issubset(allowed):
        raise ValueError(f"Unknown patch kinds: {sorted(set(patch_kinds) - allowed)}")
    retrieval_heads = tuple(
        (int(layer), int(head))
        for layer, head in experiment["retrieval_heads"][model_label]
    )
    output = Path(args.output_dir).resolve() / model_label
    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "detail.jsonl"
    broad_path = output / "broad_metrics.jsonl"
    completed = load_completed(detail_path)

    rows = {
        (int(row["seed"]), int(row["gold_count"])): row
        for row in load_stimuli(args.stimuli)
        if str(row.get("design_variant")) == "v4.4"
        and int(row["seed"]) in set(seeds)
        and int(row["gold_count"]) in set(counts)
    }
    expected = {(seed, count) for seed in seeds for count in counts}
    if set(rows) != expected:
        raise RuntimeError(f"Missing selected stimuli: {sorted(expected - set(rows))}")
    spec = resolve_model_spec(model_label)
    provenance = {
        "schema_version": "realistic_niah_v4_4_5_span_restoration_run_v1",
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "model": model_label,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "stimuli": str(Path(args.stimuli).resolve()),
        "stimuli_sha256": sha256(Path(args.stimuli)),
        "experiment_config": str(experiment_path),
        "experiment_config_sha256": sha256(experiment_path),
        "seeds": list(seeds),
        "counts": list(counts),
        "layers": list(layers),
        "patch_kinds": list(patch_kinds),
        "baseline_only": bool(args.baseline_only),
        "baseline_conditions": list(baseline_conditions),
        "cache_logit_tolerance": float(args.cache_logit_tolerance),
        "cache_equivalence_audit_enabled": audit_cache_equivalence,
        "cache_probability_tv_tolerance": (
            cache_probability_tv_tolerance if audit_cache_equivalence else None
        ),
        "cache_candidate_argmax_agreement_required": audit_cache_equivalence,
        "reuse_prefill_for_generation": bool(args.reuse_prefill_for_generation),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
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
    if max(layers) >= adapter.num_layers:
        raise ValueError("A requested patch layer is outside the model")
    answer_layers = tuple(range(adapter.num_layers))
    for seed in seeds:
        for count in counts:
            encoding = render_v4_prompt(
                rows[(seed, count)],
                tokenizer=tokenizer,
                model_spec=spec,
                config=v4_config,
                answer_format="numeric",
            )
            plan = build_corruption_plan(encoding)
            needle_corrupt, needle_changed = corrupt_encoding(
                encoding, plan, condition="needle_corrupt"
            )
            ordinary_corrupt, ordinary_changed = corrupt_encoding(
                encoding, plan, condition="ordinary_corrupt"
            )
            needle_full = segment_positions(plan, condition="needle")
            needle_endpoint = segment_positions(
                plan, condition="needle", endpoint_only=True
            )
            ordinary_full = segment_positions(plan, condition="ordinary")
            capture_positions = tuple(sorted(set(needle_full + ordinary_full)))
            donor_states: dict[int, Any] = {}
            if not args.baseline_only:
                _logits, donor_states = capture_post_block_states(
                    model,
                    adapter,
                    encoding,
                    capture_positions,
                    layers=layers,
                )
            baseline_specs = (
                ("clean", encoding, 0),
                ("needle_corrupt", needle_corrupt, needle_changed),
                ("ordinary_corrupt", ordinary_corrupt, ordinary_changed),
            )
            for condition, current, changed in baseline_specs:
                if condition not in baseline_conditions:
                    continue
                key = (seed, count, condition, -1)
                if key in completed:
                    continue
                relative = Path("states") / f"seed_{seed}_count_{count}_{condition}.pt"
                metrics, broad = evaluate(
                    model,
                    tokenizer,
                    adapter,
                    current,
                    retrieval_heads=retrieval_heads,
                    answer_layers=answer_layers,
                    max_new_tokens=experiment["strict_generation_max_new_tokens"],
                    state_path=output / relative,
                    cache_logit_tolerance=float(args.cache_logit_tolerance),
                    cache_probability_tv_tolerance=cache_probability_tv_tolerance,
                    audit_cache_equivalence=audit_cache_equivalence,
                    reuse_prefill_for_generation=bool(
                        args.reuse_prefill_for_generation
                    ),
                )
                common = {
                    "schema_version": SCHEMA,
                    "model_label": model_label,
                    "seed": int(seed),
                    "gold_count": int(count),
                    "condition": condition,
                    "patch_layer": -1,
                    "patch_kind": "none",
                    "token_budget": int(plan.token_budget),
                    "changed_tokens": int(changed),
                    "sequence_length": int(current.sequence_length),
                }
                for row in broad:
                    append_jsonl(broad_path, {**common, **row})
                append_jsonl(detail_path, {**common, **metrics, "state_path": relative.as_posix()})
                completed.add(key)
                print(
                    f"[v4.4.5 baseline] {model_label} seed={seed} N={count} "
                    f"{condition} E={metrics['expected_count']:.3f} "
                    f"strict={metrics['strict_prediction']}",
                    flush=True,
                )
            if args.baseline_only:
                continue
            for layer in layers:
                patch_specs = {
                    "needle_endpoint": (needle_corrupt, needle_endpoint),
                    "needle_full": (needle_corrupt, needle_full),
                    "ordinary_full": (ordinary_corrupt, ordinary_full),
                }
                for patch_kind in patch_kinds:
                    condition = f"restore_{patch_kind}"
                    key = (seed, count, condition, int(layer))
                    if key in completed:
                        continue
                    current, positions = patch_specs[patch_kind]
                    replacement = take_states(
                        donor_states[int(layer)], capture_positions, positions
                    )
                    relative = (
                        Path("states")
                        / f"seed_{seed}_count_{count}_{condition}_L{layer}.pt"
                    )
                    with residual_patch_hook(
                        adapter,
                        current,
                        layer=int(layer),
                        positions=positions,
                        replacement=replacement,
                    ) as applications:
                        metrics, broad = evaluate(
                            model,
                            tokenizer,
                            adapter,
                            current,
                            retrieval_heads=retrieval_heads,
                            answer_layers=answer_layers,
                            max_new_tokens=experiment[
                                "strict_generation_max_new_tokens"
                            ],
                            state_path=output / relative,
                            cache_logit_tolerance=float(args.cache_logit_tolerance),
                            cache_probability_tv_tolerance=(
                                cache_probability_tv_tolerance
                            ),
                            audit_cache_equivalence=audit_cache_equivalence,
                            reuse_prefill_for_generation=bool(
                                args.reuse_prefill_for_generation
                            ),
                        )
                    common = {
                        "schema_version": SCHEMA,
                        "model_label": model_label,
                        "seed": int(seed),
                        "gold_count": int(count),
                        "condition": condition,
                        "patch_layer": int(layer),
                        "patch_kind": patch_kind,
                        "token_budget": int(plan.token_budget),
                        "patched_positions": len(positions),
                        "changed_tokens": int(
                            needle_changed
                            if patch_kind.startswith("needle")
                            else ordinary_changed
                        ),
                        "sequence_length": int(current.sequence_length),
                        "patch_hook_applications": int(applications["count"]),
                    }
                    expected_applications = (
                        2 if args.reuse_prefill_for_generation else 3
                    )
                    if int(applications["count"]) != expected_applications:
                        raise RuntimeError(
                            "Patch hook application count disagrees with the selected "
                            "generation path: "
                            f"expected {expected_applications}, "
                            f"got {applications['count']}"
                        )
                    for row in broad:
                        append_jsonl(broad_path, {**common, **row})
                    append_jsonl(
                        detail_path,
                        {**common, **metrics, "state_path": relative.as_posix()},
                    )
                    completed.add(key)
                    print(
                        f"[v4.4.5 patch] {model_label} seed={seed} N={count} "
                        f"{patch_kind} L{layer} E={metrics['expected_count']:.3f} "
                        f"strict={metrics['strict_prediction']}",
                        flush=True,
                    )
    complete = {
        "status": "complete",
        "model": model_label,
        "rows": len(load_completed(detail_path)),
        "detail": str(detail_path),
        "broad_metrics": str(broad_path),
    }
    (output / "complete.json").write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(complete, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
