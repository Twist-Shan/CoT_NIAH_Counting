from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from realistic_niah_v4.modeling import capture_post_block_states, load_registered_model
from realistic_niah_v4.prompts import PromptEncoding, render_v4_prompt
from realistic_niah_v4.spec import V4Config, resolve_model_spec
from realistic_niah_v4.stimuli import load_stimuli
from realistic_niah_v4_4_5.restoration import (
    build_corruption_plan,
    corrupt_encoding,
    segment_positions,
)


SCHEMA = "realistic_niah_v4_4_5_span_state_deformation_detail_v1"
UNIT_SCHEMA = "realistic_niah_v4_4_5_span_state_deformation_unit_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_sha256(encoding: PromptEncoding) -> str:
    payload = json.dumps(list(encoding.input_ids), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def select_positions(
    captured: torch.Tensor,
    captured_positions: Sequence[int],
    selected_positions: Sequence[int],
) -> torch.Tensor:
    lookup = {int(position): index for index, position in enumerate(captured_positions)}
    missing = sorted(set(int(value) for value in selected_positions) - set(lookup))
    if missing:
        raise RuntimeError(f"Captured state bank lacks positions: {missing}")
    return captured[[lookup[int(position)] for position in selected_positions]]


def deformation_metrics(clean: torch.Tensor, corrupt: torch.Tensor) -> dict[str, float]:
    if clean.shape != corrupt.shape or clean.ndim != 2:
        raise ValueError("clean/corrupt states must share [positions, hidden] shape")
    clean = clean.float()
    corrupt = corrupt.float()
    delta = clean - corrupt
    raw_rms = torch.sqrt(torch.mean(delta.square()))
    clean_rms = torch.sqrt(torch.mean(clean.square()))
    relative_rms = raw_rms / clean_rms.clamp_min(torch.finfo(torch.float32).eps)
    cosine_distance = 1.0 - F.cosine_similarity(clean, corrupt, dim=-1)
    token_rms = torch.sqrt(torch.mean(delta.square(), dim=-1))
    result = {
        "raw_rms_change": float(raw_rms.item()),
        "clean_rms": float(clean_rms.item()),
        "relative_rms_change": float(relative_rms.item()),
        "mean_cosine_distance": float(cosine_distance.mean().item()),
        "median_token_rms_change": float(token_rms.median().item()),
        "max_token_rms_change": float(token_rms.max().item()),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("A deformation metric is non-finite")
    return result


def capture(
    model: Any,
    adapter: Any,
    encoding: PromptEncoding,
    positions: Sequence[int],
    layers: Sequence[int],
) -> tuple[dict[int, torch.Tensor], float]:
    started = time.perf_counter()
    _logits, states = capture_post_block_states(
        model, adapter, encoding, positions, layers=layers
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return states, float(time.perf_counter() - started)


def audit_complete(
    detail_path: Path,
    units_path: Path,
    *,
    model_label: str,
    seeds: Sequence[int],
    counts: Sequence[int],
    layers: Sequence[int],
    stimulus_hash: str,
) -> dict[str, Any]:
    details = read_jsonl(detail_path)
    units = read_jsonl(units_path)
    detail_keys = [
        (str(row["model_label"]), int(row["seed"]), int(row["gold_count"]), int(row["layer"]))
        for row in details
    ]
    unit_keys = [
        (str(row["model_label"]), int(row["seed"]), int(row["gold_count"]))
        for row in units
    ]
    expected_detail = {
        (model_label, int(seed), int(count), int(layer))
        for seed in seeds
        for count in counts
        for layer in layers
    }
    expected_units = {
        (model_label, int(seed), int(count)) for seed in seeds for count in counts
    }
    finite = all(
        math.isfinite(float(row[field]))
        for row in details
        for field in (
            "needle_raw_rms_change",
            "ordinary_raw_rms_change",
            "raw_specificity",
            "needle_relative_rms_change",
            "ordinary_relative_rms_change",
            "relative_specificity",
            "needle_mean_cosine_distance",
            "ordinary_mean_cosine_distance",
            "cosine_specificity",
        )
    )
    audit = {
        "schema_version": "realistic_niah_v4_4_5_span_state_deformation_run_audit_v1",
        "status": "PASS",
        "model_label": model_label,
        "detail_rows": len(details),
        "expected_detail_rows": len(expected_detail),
        "unit_rows": len(units),
        "expected_unit_rows": len(expected_units),
        "unique_detail_keys": len(set(detail_keys)),
        "unique_unit_keys": len(set(unit_keys)),
        "detail_key_coverage_exact": set(detail_keys) == expected_detail,
        "unit_key_coverage_exact": set(unit_keys) == expected_units,
        "finite_metrics": finite,
        "token_budget_matched": all(
            int(row["needle_token_budget"]) == int(row["ordinary_token_budget"])
            and int(row["needle_token_budget"]) > 0
            for row in units
        ),
        "sequence_and_query_preserved": all(
            bool(row["sequence_length_preserved"])
            and bool(row["query_position_preserved"])
            for row in units
        ),
        "stimulus_sha256": stimulus_hash,
    }
    checks = [
        audit["detail_rows"] == audit["expected_detail_rows"],
        audit["unit_rows"] == audit["expected_unit_rows"],
        audit["unique_detail_keys"] == audit["expected_detail_rows"],
        audit["unique_unit_keys"] == audit["expected_unit_rows"],
        audit["detail_key_coverage_exact"],
        audit["unit_key_coverage_exact"],
        audit["finite_metrics"],
        audit["token_budget_matched"],
        audit["sequence_and_query_preserved"],
    ]
    if not all(checks):
        audit["status"] = "FAIL"
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture full-vector clean/corrupt span-state deformation by layer."
    )
    parser.add_argument("--model", required=True, choices=("Qwen3-8B", "Gemma4-E4B"))
    parser.add_argument("--stimuli", required=True)
    parser.add_argument(
        "--stimuli-config", default="configs/realistic_niah_v4_4_5_stimuli.json"
    )
    parser.add_argument(
        "--experiment-config",
        default="configs/realistic_niah_v4_4_5_span_state_deformation.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seeds")
    parser.add_argument("--counts")
    args = parser.parse_args()

    config_path = Path(args.experiment_config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "realistic_niah_v4_4_5_span_state_deformation_v1":
        raise ValueError("Unexpected span-state deformation config schema")
    model_label = str(args.model)
    selected_seeds = (
        tuple(int(value) for value in args.seeds.split(",") if value.strip())
        if args.seeds
        else tuple(int(value) for value in config["confirmation_seeds"])
    )
    selected_counts = (
        tuple(int(value) for value in args.counts.split(",") if value.strip())
        if args.counts
        else tuple(int(value) for value in config["counts"])
    )
    allowed_seeds = set(int(value) for value in config["confirmation_seeds"])
    allowed_counts = set(int(value) for value in config["counts"])
    if not set(selected_seeds).issubset(allowed_seeds):
        raise ValueError("Requested seeds leave the frozen confirmation cohort")
    if not set(selected_counts).issubset(allowed_counts):
        raise ValueError("Requested counts leave the frozen count range")
    layers = tuple(int(value) for value in config["layers"][model_label])

    stimuli_path = Path(args.stimuli).resolve()
    stimulus_hash = sha256(stimuli_path)
    if stimulus_hash != str(config["stimulus_sha256"]):
        raise RuntimeError(
            f"Frozen stimulus hash mismatch: {stimulus_hash} != {config['stimulus_sha256']}"
        )
    rows = {
        (int(row["seed"]), int(row["gold_count"])): row
        for row in load_stimuli(stimuli_path)
        if str(row.get("design_variant")) == "v4.4"
        and int(row["seed"]) in set(selected_seeds)
        and int(row["gold_count"]) in set(selected_counts)
    }
    expected_units = {
        (int(seed), int(count)) for seed in selected_seeds for count in selected_counts
    }
    if set(rows) != expected_units:
        raise RuntimeError(f"Missing selected stimuli: {sorted(expected_units - set(rows))}")

    output = Path(args.output_dir).resolve() / model_label
    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "detail.jsonl"
    units_path = output / "units.jsonl"
    existing_details = read_jsonl(detail_path)
    existing_units = read_jsonl(units_path)
    completed_layers: dict[tuple[int, int], set[int]] = {}
    for row in existing_details:
        completed_layers.setdefault(
            (int(row["seed"]), int(row["gold_count"])), set()
        ).add(int(row["layer"]))
    completed_units = {
        (int(row["seed"]), int(row["gold_count"])) for row in existing_units
    }

    spec = resolve_model_spec(model_label)
    provenance = {
        "schema_version": "realistic_niah_v4_4_5_span_state_deformation_run_v1",
        "model_label": model_label,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "stimuli": str(stimuli_path),
        "stimulus_sha256": stimulus_hash,
        "experiment_config": str(config_path),
        "experiment_config_sha256": sha256(config_path),
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "runtime_files": {
            "modeling": {
                "path": str(Path(capture_post_block_states.__code__.co_filename).resolve()),
                "sha256": sha256(
                    Path(capture_post_block_states.__code__.co_filename).resolve()
                ),
            },
            "restoration": {
                "path": str(Path(build_corruption_plan.__code__.co_filename).resolve()),
                "sha256": sha256(
                    Path(build_corruption_plan.__code__.co_filename).resolve()
                ),
            },
            "prompt_renderer": {
                "path": str(Path(render_v4_prompt.__code__.co_filename).resolve()),
                "sha256": sha256(Path(render_v4_prompt.__code__.co_filename).resolve()),
            },
            "stimulus_loader": {
                "path": str(Path(load_stimuli.__code__.co_filename).resolve()),
                "sha256": sha256(Path(load_stimuli.__code__.co_filename).resolve()),
            },
        },
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "seeds": list(selected_seeds),
        "counts": list(selected_counts),
        "layers": list(layers),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    v4_config = V4Config.from_json(args.stimuli_config)
    model, tokenizer, adapter = load_registered_model(
        spec,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=config["model_torch_dtype"],
        attention_backend=config["attention_backend"],
    )
    if tuple(range(adapter.num_layers)) != layers:
        raise RuntimeError(
            f"Frozen layer registry does not equal model layers: {layers} vs {adapter.num_layers}"
        )

    for seed in selected_seeds:
        for count in selected_counts:
            unit = (int(seed), int(count))
            missing_layers = set(layers) - completed_layers.get(unit, set())
            if not missing_layers and unit in completed_units:
                print(f"SKIP model={model_label} seed={seed} count={count} complete", flush=True)
                continue
            encoding = render_v4_prompt(
                rows[unit],
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
            needle_positions = segment_positions(plan, condition="needle")
            ordinary_positions = segment_positions(plan, condition="ordinary")
            if len(needle_positions) != len(ordinary_positions) or not needle_positions:
                raise RuntimeError("Needle/ordinary token budgets are not exactly matched")
            if set(needle_positions).intersection(ordinary_positions):
                raise RuntimeError("Needle and ordinary target positions overlap")
            capture_positions = tuple(sorted(set(needle_positions + ordinary_positions)))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            clean_states, clean_seconds = capture(
                model, adapter, encoding, capture_positions, layers
            )
            needle_states, needle_seconds = capture(
                model, adapter, needle_corrupt, needle_positions, layers
            )
            ordinary_states, ordinary_seconds = capture(
                model, adapter, ordinary_corrupt, ordinary_positions, layers
            )
            for layer in layers:
                if layer not in missing_layers:
                    continue
                clean_needle = select_positions(
                    clean_states[layer], capture_positions, needle_positions
                )
                clean_ordinary = select_positions(
                    clean_states[layer], capture_positions, ordinary_positions
                )
                needle = deformation_metrics(clean_needle, needle_states[layer])
                ordinary = deformation_metrics(clean_ordinary, ordinary_states[layer])
                append_jsonl(
                    detail_path,
                    {
                        "schema_version": SCHEMA,
                        "model_label": model_label,
                        "seed": int(seed),
                        "gold_count": int(count),
                        "layer": int(layer),
                        "needle_token_budget": len(needle_positions),
                        "ordinary_token_budget": len(ordinary_positions),
                        "hidden_width": int(clean_needle.shape[-1]),
                        "needle_raw_rms_change": needle["raw_rms_change"],
                        "ordinary_raw_rms_change": ordinary["raw_rms_change"],
                        "raw_specificity": needle["raw_rms_change"]
                        - ordinary["raw_rms_change"],
                        "needle_clean_rms": needle["clean_rms"],
                        "ordinary_clean_rms": ordinary["clean_rms"],
                        "needle_relative_rms_change": needle["relative_rms_change"],
                        "ordinary_relative_rms_change": ordinary["relative_rms_change"],
                        "relative_specificity": needle["relative_rms_change"]
                        - ordinary["relative_rms_change"],
                        "needle_mean_cosine_distance": needle["mean_cosine_distance"],
                        "ordinary_mean_cosine_distance": ordinary["mean_cosine_distance"],
                        "cosine_specificity": needle["mean_cosine_distance"]
                        - ordinary["mean_cosine_distance"],
                        "needle_median_token_rms_change": needle[
                            "median_token_rms_change"
                        ],
                        "ordinary_median_token_rms_change": ordinary[
                            "median_token_rms_change"
                        ],
                        "needle_max_token_rms_change": needle["max_token_rms_change"],
                        "ordinary_max_token_rms_change": ordinary["max_token_rms_change"],
                    },
                )
            if unit not in completed_units:
                append_jsonl(
                    units_path,
                    {
                        "schema_version": UNIT_SCHEMA,
                        "model_label": model_label,
                        "seed": int(seed),
                        "gold_count": int(count),
                        "sequence_length": int(encoding.sequence_length),
                        "query_position": int(encoding.query_position),
                        "needle_token_budget": len(needle_positions),
                        "ordinary_token_budget": len(ordinary_positions),
                        "needle_changed_tokens": int(needle_changed),
                        "ordinary_changed_tokens": int(ordinary_changed),
                        "sequence_length_preserved": bool(
                            encoding.sequence_length
                            == needle_corrupt.sequence_length
                            == ordinary_corrupt.sequence_length
                        ),
                        "query_position_preserved": bool(
                            encoding.query_position
                            == needle_corrupt.query_position
                            == ordinary_corrupt.query_position
                        ),
                        "clean_input_sha256": input_sha256(encoding),
                        "needle_corrupt_input_sha256": input_sha256(needle_corrupt),
                        "ordinary_corrupt_input_sha256": input_sha256(ordinary_corrupt),
                        "clean_seconds": clean_seconds,
                        "needle_corrupt_seconds": needle_seconds,
                        "ordinary_corrupt_seconds": ordinary_seconds,
                        "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated())
                        if torch.cuda.is_available()
                        else 0,
                        "max_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())
                        if torch.cuda.is_available()
                        else 0,
                    },
                )
            print(
                f"DONE model={model_label} seed={seed} count={count} "
                f"layers={len(layers)} seconds={clean_seconds + needle_seconds + ordinary_seconds:.2f}",
                flush=True,
            )
            del clean_states, needle_states, ordinary_states

    audit = audit_complete(
        detail_path,
        units_path,
        model_label=model_label,
        seeds=selected_seeds,
        counts=selected_counts,
        layers=layers,
        stimulus_hash=stimulus_hash,
    )
    (output / "complete.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if audit["status"] != "PASS":
        raise RuntimeError("Span-state deformation run audit failed")
    (output / ".RUN_COMPLETE").write_text("PASS\n", encoding="utf-8")


if __name__ == "__main__":
    main()
