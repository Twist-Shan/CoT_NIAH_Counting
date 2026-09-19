from __future__ import annotations

"""Run the preregistered V4.4.5 same-forward partial serial mediation."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from realistic_niah_v4.behavior import parse_numeric_completion
from realistic_niah_v4.modeling import capture_post_block_states, load_registered_model
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
from realistic_niah_v4_4_5.serial_mediation import (
    deterministic_bank_orthogonal_direction,
    flatten_path_audits,
    late_answer_path_hook,
    load_answer_geometry,
    retrieval_path_hook,
    serial_arm_map,
    validate_serial_registry,
)


SCHEMA = "realistic_niah_v4_4_5_serial_mediation_detail_v1"


def csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("At least one integer is required")
    return result


def csv_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("At least one string is required")
    return result


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def completed_keys(path: Path) -> set[tuple[int, int, str]]:
    if not path.exists():
        return set()
    keys: set[tuple[int, int, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (int(row["seed"]), int(row["gold_count"]), str(row["arm"]))
        if key in keys:
            raise RuntimeError(f"Duplicate existing serial row: {key}")
        keys.add(key)
    return keys


def write_or_verify_json(path: Path, value: Mapping[str, Any]) -> None:
    serialized = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(value):
            raise RuntimeError(f"Existing provenance differs from this launch: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def strict_fields(result: Mapping[str, Any], gold: int) -> dict[str, Any]:
    labels = parse_numeric_completion(str(result.get("completion_text", "")))
    prediction = labels.get("parsed_count")
    value = None if prediction is None else int(prediction)
    return {
        "strict_prediction": value,
        "strict_correct": bool(value == int(gold)),
        "strict_absolute_error": 10 if value is None else abs(value - int(gold)),
        "strict_completion": str(result.get("completion_text", "")),
        "strict_format_valid": bool(labels.get("format_valid", False)),
    }


def take_positions(
    captured: torch.Tensor,
    captured_positions: Sequence[int],
    selected_positions: Sequence[int],
) -> torch.Tensor:
    lookup = {int(position): index for index, position in enumerate(captured_positions)}
    missing = sorted(set(int(value) for value in selected_positions) - set(lookup))
    if missing:
        raise RuntimeError(f"Clean state bank lacks positions: {missing}")
    return captured[[lookup[int(position)] for position in selected_positions]]


@torch.inference_mode()
def evaluate_arm(
    model: Any,
    tokenizer: Any,
    adapter: Any,
    encoding: PromptEncoding,
    *,
    retrieval_layer: int,
    retrieval_heads: Sequence[int],
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    bundle = capture_query_bundle(
        model,
        adapter,
        encoding,
        layers=(int(retrieval_layer),),
        capture_attention=True,
        capture_values=False,
        audit_cache_equivalence=False,
        retain_prefill_output=True,
    )
    if bundle.reusable_prefill_output is None:
        raise RuntimeError("Serial mediation retained no causal prefill")
    strict = generate_answer_completion_from_prefill(
        model,
        tokenizer,
        encoding,
        bundle.reusable_prefill_output,
        max_new_tokens=int(max_new_tokens),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    broad_all = active_broad_metrics(
        bundle.alpha_by_layer[int(retrieval_layer)],
        key_start=bundle.alpha_key_start_by_layer[int(retrieval_layer)],
        spans=encoding.needle_spans,
    )
    selected = set(int(head) for head in retrieval_heads)
    broad = [row for row in broad_all if int(row["head"]) in selected]
    if len(broad) != len(selected):
        raise RuntimeError("Frozen retrieval heads were not all present in attention rows")
    metrics = {
        **candidate_sequence_metrics(bundle.candidate_log_scores, encoding),
        **strict_fields(strict, int(encoding.count)),
        "elapsed_seconds": float(time.perf_counter() - started),
        "strict_generation_reused_causal_prefill": True,
        "attention_source": "cache_reconstructed_query_row_without_equivalence_comparison",
        "retrieval_bank_needle_mass_mean": float(
            np.mean([float(row["needle_mass"]) for row in broad])
        ),
        "retrieval_bank_coverage_mean": float(
            np.mean([float(row["coverage"]) for row in broad])
        ),
        "retrieval_bank_broad_score_mean": float(
            np.mean([float(row["broad_score"]) for row in broad])
        ),
        "max_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "max_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
        ),
    }
    return metrics, broad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("Qwen3-8B", "Gemma4-E4B"))
    parser.add_argument("--stimuli", required=True)
    parser.add_argument(
        "--stimuli-config", default="configs/realistic_niah_v4_4_5_stimuli.json"
    )
    parser.add_argument(
        "--experiment-config",
        default="configs/realistic_niah_v4_4_5_serial_mediation.json",
    )
    parser.add_argument("--retrieval-basis", required=True)
    parser.add_argument("--answer-packed-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=csv_ints)
    parser.add_argument("--counts", type=csv_ints)
    parser.add_argument("--arms", type=csv_strings)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    experiment_path = Path(args.experiment_config).resolve()
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if experiment.get("schema_version") != "realistic_niah_v4_4_5_serial_mediation_v1":
        raise ValueError("Unexpected serial-mediation schema")
    model_label = str(args.model)
    stages = experiment["stages"][model_label]
    source_layer = int(stages["source_layer"])
    retrieval_layer = int(stages["retrieval_layer"])
    late_layer = int(stages["late_answer_layer"])
    retrieval_heads = tuple(int(value) for value in stages["retrieval_heads"])
    validate_serial_registry(
        experiment["arms"],
        source=source_layer,
        retrieval=retrieval_layer,
        late=late_layer,
    )
    frozen_arms = serial_arm_map()
    selected_arms = tuple(args.arms or tuple(experiment["arms"]))
    unknown = sorted(set(selected_arms) - set(frozen_arms))
    if unknown:
        raise ValueError(f"Unknown serial arms: {unknown}")
    seeds = tuple(args.seeds or tuple(int(value) for value in experiment["confirmation_seeds"]))
    counts = tuple(args.counts or tuple(int(value) for value in experiment["counts"]))
    if not set(seeds).issubset(set(int(value) for value in experiment["confirmation_seeds"])):
        raise ValueError("Serial run may use only frozen confirmation seeds")
    if not set(counts).issubset(set(int(value) for value in experiment["counts"])):
        raise ValueError("A requested count is outside the frozen registry")

    stimuli_path = Path(args.stimuli).resolve()
    stimulus_hash = sha256(stimuli_path)
    if stimulus_hash != str(experiment["stimulus_sha256"]):
        raise RuntimeError(
            f"Frozen stimulus hash mismatch: {stimulus_hash} != {experiment['stimulus_sha256']}"
        )
    rows = {
        (int(row["seed"]), int(row["gold_count"])): row
        for row in load_stimuli(stimuli_path)
        if str(row.get("design_variant")) == "v4.4"
        and int(row["seed"]) in set(seeds)
        and int(row["gold_count"]) in set(counts)
    }
    expected_units = {(seed, count) for seed in seeds for count in counts}
    if set(rows) != expected_units:
        raise RuntimeError(f"Missing selected stimuli: {sorted(expected_units - set(rows))}")

    retrieval_basis_path = Path(args.retrieval_basis).resolve()
    retrieval_payload = torch.load(
        retrieval_basis_path, map_location="cpu", weights_only=True
    )
    retrieval_key = f"{model_label}.L{retrieval_layer}"
    if retrieval_key not in retrieval_payload:
        raise KeyError(f"Retrieval basis lacks {retrieval_key}")
    retrieval_mean = retrieval_payload[retrieval_key]["mean"].float()
    retrieval_basis = retrieval_payload[retrieval_key]["components"].float()
    if tuple(retrieval_basis.shape[:1]) != (int(experiment["rank"]),):
        raise RuntimeError("Retrieval basis has the wrong rank")

    answer_npz = (
        Path(args.answer_packed_root).resolve()
        / "layers"
        / f"{model_label}__answer_query__L{late_layer:02d}.npz"
    )
    answer_geometry = load_answer_geometry(
        answer_npz,
        discovery_seeds=experiment["discovery_seeds"],
        rank=int(experiment["rank"]),
    )
    config = V4Config.from_json(args.stimuli_config)
    spec = resolve_model_spec(model_label)
    output = Path(args.output_dir).resolve() / model_label
    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "detail.jsonl"
    broad_path = output / "broad_metrics.jsonl"
    completed = completed_keys(detail_path)
    expected_keys = {
        (int(seed), int(count), arm)
        for seed in seeds
        for count in counts
        for arm in selected_arms
    }
    if not completed.issubset(expected_keys):
        raise RuntimeError("Existing output contains keys outside this launch selection")
    provenance = {
        "schema_version": "realistic_niah_v4_4_5_serial_mediation_run_v1",
        "model": model_label,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "experiment_config": str(experiment_path),
        "experiment_config_sha256": sha256(experiment_path),
        "stimuli": str(stimuli_path),
        "stimuli_sha256": stimulus_hash,
        "retrieval_basis": str(retrieval_basis_path),
        "retrieval_basis_sha256": sha256(retrieval_basis_path),
        "retrieval_basis_key": retrieval_key,
        "answer_geometry_npz": str(answer_npz),
        "answer_geometry_npz_sha256": sha256(answer_npz),
        "answer_geometry_centroid_rank3_capture": float(
            answer_geometry.centroid_variance_capture
        ),
        "source_layer": source_layer,
        "retrieval_layer": retrieval_layer,
        "late_answer_layer": late_layer,
        "retrieval_heads": list(retrieval_heads),
        "seeds": list(seeds),
        "counts": list(counts),
        "arms": list(selected_arms),
        "attention_source": "cache_reconstruction_only_no_equivalence_comparison",
        "strict_generation": "reuse_same_causal_prefill_kv_cache",
        "torch_version": torch.__version__,
    }
    write_or_verify_json(output / "run_provenance.json", provenance)

    model, tokenizer, adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=experiment["model_torch_dtype"],
        attention_backend=experiment["attention_prefix_backend"],
    )
    if late_layer >= adapter.num_layers:
        raise ValueError("A frozen serial layer is outside the loaded model")
    retrieval_control = deterministic_bank_orthogonal_direction(
        adapter,
        layer=retrieval_layer,
        heads=retrieval_heads,
        basis=retrieval_basis,
        random_seed=int(experiment["retrieval_control_seed"]) + retrieval_layer,
    )
    prequant_orthogonality = float(
        torch.max(torch.abs(retrieval_control @ retrieval_basis.T))
    )
    if prequant_orthogonality > 1e-4:
        raise RuntimeError("Frozen retrieval control is not orthogonal before quantization")

    norm_tolerance = float(experiment["realized_norm_relative_tolerance"])
    orth_tolerance = float(experiment["orthogonality_max_abs_cosine_tolerance"])
    for seed in seeds:
        for count in counts:
            if all((seed, count, arm) in completed for arm in selected_arms):
                continue
            clean = render_v4_prompt(
                rows[(seed, count)],
                tokenizer=tokenizer,
                model_spec=spec,
                config=config,
                answer_format="numeric",
            )
            plan = build_corruption_plan(clean)
            corrupted, changed = corrupt_encoding(
                clean, plan, condition="needle_corrupt"
            )
            needle_positions = segment_positions(plan, condition="needle")
            ordinary_positions = segment_positions(plan, condition="ordinary")
            if len(needle_positions) != len(ordinary_positions):
                raise RuntimeError("Needle and ordinary controls differ in token budget")
            capture_positions = tuple(sorted(set(needle_positions + ordinary_positions)))
            _logits, clean_states = capture_post_block_states(
                model,
                adapter,
                clean,
                capture_positions,
                layers=(source_layer,),
            )
            replacements = {
                "needle": take_positions(
                    clean_states[source_layer], capture_positions, needle_positions
                ),
                "ordinary": take_positions(
                    clean_states[source_layer], capture_positions, ordinary_positions
                ),
            }
            for arm_name in selected_arms:
                key = (int(seed), int(count), arm_name)
                if key in completed:
                    continue
                arm = frozen_arms[arm_name]
                if arm.source == "none":
                    source_context = nullcontext({"count": 0})
                    source_positions: tuple[int, ...] = ()
                else:
                    source_positions = (
                        needle_positions if arm.source == "needle" else ordinary_positions
                    )
                    source_context = residual_patch_hook(
                        adapter,
                        corrupted,
                        layer=source_layer,
                        positions=source_positions,
                        replacement=replacements[arm.source],
                    )
                with source_context as source_audit:
                    with retrieval_path_hook(
                        adapter,
                        corrupted,
                        layer=retrieval_layer,
                        heads=retrieval_heads,
                        mean=retrieval_mean,
                        basis=retrieval_basis,
                        control_direction=retrieval_control,
                        mode=arm.retrieval,
                    ) as retrieval_audit:
                        with late_answer_path_hook(
                            adapter,
                            corrupted,
                            layer=late_layer,
                            geometry=answer_geometry,
                            mode=arm.late,
                        ) as late_audit:
                            metrics, broad = evaluate_arm(
                                model,
                                tokenizer,
                                adapter,
                                corrupted,
                                retrieval_layer=retrieval_layer,
                                retrieval_heads=retrieval_heads,
                                max_new_tokens=int(
                                    experiment["strict_generation_max_new_tokens"]
                                ),
                            )
                expected_source_applications = 0 if arm.source == "none" else 2
                if int(source_audit["count"]) != expected_source_applications:
                    raise RuntimeError(
                        f"{arm_name} source hook applied {source_audit['count']} times; "
                        f"expected {expected_source_applications}"
                    )
                if int(retrieval_audit["applications"]) != 1:
                    raise RuntimeError("Retrieval read/block must apply to one causal prefill")
                if int(late_audit["applications"]) != 1:
                    raise RuntimeError("Late read/block must apply to one causal prefill")
                for stage_name, audit, mode in (
                    ("retrieval", retrieval_audit, arm.retrieval),
                    ("late", late_audit, arm.late),
                ):
                    if mode is not None and abs(float(audit["norm_ratio"]) - 1.0) > norm_tolerance:
                        raise RuntimeError(f"{stage_name} realized norm match failed")
                    if mode == "orthogonal" and float(
                        audit["orthogonality_max_abs_cosine"]
                    ) > orth_tolerance:
                        raise RuntimeError(f"{stage_name} orthogonal control leaked into rank 3")
                common = {
                    "schema_version": SCHEMA,
                    "model_label": model_label,
                    "seed": int(seed),
                    "gold_count": int(count),
                    "arm": arm_name,
                    "source_mode": arm.source,
                    "retrieval_mode": arm.retrieval or "none",
                    "late_mode": arm.late or "none",
                    "source_layer": source_layer,
                    "retrieval_layer": retrieval_layer,
                    "late_answer_layer": late_layer,
                    "source_patch_positions": len(source_positions),
                    "source_patch_applications": int(source_audit["count"]),
                    "token_budget": int(plan.token_budget),
                    "changed_tokens": int(changed),
                    "sequence_length": int(corrupted.sequence_length),
                    **flatten_path_audits(retrieval_audit, late_audit),
                    **metrics,
                }
                for broad_row in broad:
                    append_jsonl(
                        broad_path,
                        {
                            "schema_version": "realistic_niah_v4_4_5_serial_mediation_broad_v1",
                            "model_label": model_label,
                            "seed": int(seed),
                            "gold_count": int(count),
                            "arm": arm_name,
                            "layer": retrieval_layer,
                            "head": int(broad_row["head"]),
                            "needle_mass": float(broad_row["needle_mass"]),
                            "coverage": float(broad_row["coverage"]),
                            "broad_score": float(broad_row["broad_score"]),
                            "span_masses": list(broad_row["span_masses"]),
                        },
                    )
                append_jsonl(detail_path, common)
                completed.add(key)
                print(
                    f"[serial] {model_label} seed={seed} N={count} arm={arm_name} "
                    f"E={metrics['expected_count']:.3f} strict={metrics['strict_prediction']}",
                    flush=True,
                )
    missing = sorted(expected_keys - completed)
    if missing:
        raise RuntimeError(f"Serial run ended with missing rows: {missing[:5]}")
    completion = {
        "status": "complete",
        "schema_version": "realistic_niah_v4_4_5_serial_mediation_complete_v1",
        "model": model_label,
        "rows": len(expected_keys),
        "unique_keys": len(completed),
        "seeds": list(seeds),
        "counts": list(counts),
        "arms": list(selected_arms),
        "detail": str(detail_path),
        "broad_metrics": str(broad_path),
    }
    write_or_verify_json(output / "complete.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
