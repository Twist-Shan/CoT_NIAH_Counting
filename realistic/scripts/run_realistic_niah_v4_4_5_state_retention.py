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

from realistic_niah_v4.modeling import capture_span_states, load_registered_model
from realistic_niah_v4.prompts import PromptEncoding, render_v4_prompt
from realistic_niah_v4.spec import V4Config, resolve_model_spec
from realistic_niah_v4.stimuli import load_stimuli
from realistic_niah_v4_4_5.restoration import (
    build_corruption_plan,
    corrupt_encoding,
    segment_positions,
)


LAYER_SCHEMA = "realistic_niah_v4_4_5_state_retention_layer_seed_v1"
OCCURRENCE_SCHEMA = "realistic_niah_v4_4_5_state_retention_occurrence_v1"
UNIT_SCHEMA = "realistic_niah_v4_4_5_state_retention_unit_v1"
SITES = ("span_end", "span_mean")


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


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def relative_change_metrics(
    clean: torch.Tensor, changed: torch.Tensor
) -> dict[str, Any]:
    if clean.shape != changed.shape or clean.ndim != 2:
        raise ValueError("clean/changed states must share [occurrence, hidden] shape")
    clean = clean.float()
    changed = changed.float()
    delta = changed - clean
    epsilon = torch.finfo(torch.float32).eps
    per_occurrence_l2 = torch.linalg.vector_norm(delta, dim=-1)
    per_occurrence_clean_l2 = torch.linalg.vector_norm(clean, dim=-1)
    per_occurrence_relative = per_occurrence_l2 / per_occurrence_clean_l2.clamp_min(
        epsilon
    )
    per_occurrence_cosine = 1.0 - F.cosine_similarity(clean, changed, dim=-1)
    raw_rms = torch.sqrt(torch.mean(delta.square()))
    clean_rms = torch.sqrt(torch.mean(clean.square()))
    result = {
        "raw_rms": float(raw_rms.item()),
        "clean_rms": float(clean_rms.item()),
        "relative_rms": float((raw_rms / clean_rms.clamp_min(epsilon)).item()),
        "mean_cosine_distance": float(per_occurrence_cosine.mean().item()),
        "per_occurrence_l2": per_occurrence_l2,
        "per_occurrence_relative_l2": per_occurrence_relative,
        "per_occurrence_cosine_distance": per_occurrence_cosine,
    }
    scalars = (
        result["raw_rms"],
        result["clean_rms"],
        result["relative_rms"],
        result["mean_cosine_distance"],
    )
    if not all(math.isfinite(float(value)) for value in scalars):
        raise RuntimeError("A state-change metric is non-finite")
    return result


def centroid_retention_metrics(
    states: torch.Tensor,
    centroids: torch.Tensor,
    radii: torch.Tensor,
) -> dict[str, Any]:
    if states.ndim != 2 or centroids.shape != states.shape:
        raise ValueError("states/centroids must share [running_index, hidden] shape")
    if radii.shape != (states.shape[0],):
        raise ValueError("radii must have one value per running-index centroid")
    if not bool(torch.all(torch.isfinite(states))) or not bool(
        torch.all(torch.isfinite(centroids))
    ):
        raise RuntimeError("Non-finite states or centroids")
    epsilon = torch.finfo(torch.float32).eps
    standardized = torch.cdist(states.float(), centroids.float()) / radii.float().clamp_min(
        epsilon
    ).unsqueeze(0)
    count = states.shape[0]
    indices = torch.arange(count)
    correct = standardized[indices, indices]
    wrong = standardized.clone()
    wrong[indices, indices] = torch.inf
    nearest_wrong = wrong.min(dim=1).values
    margin = nearest_wrong - correct
    prediction = standardized.argmin(dim=1)
    absolute_error = (prediction - indices).abs().float()
    return {
        "correct_distance": correct,
        "nearest_wrong_distance": nearest_wrong,
        "margin": margin,
        "prediction": prediction,
        "absolute_error": absolute_error,
        "accuracy": float((prediction == indices).float().mean().item()),
        "mad": float(absolute_error.mean().item()),
        "mean_correct_distance": float(correct.mean().item()),
        "mean_margin": float(margin.mean().item()),
    }


def layer_retention_rows(
    *,
    model_label: str,
    seed: int,
    layer: int,
    pooling: str,
    clean: torch.Tensor,
    needle: torch.Tensor,
    ordinary: torch.Tensor,
    centroids: torch.Tensor,
    radii: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clean_geometry = centroid_retention_metrics(clean, centroids, radii)
    needle_geometry = centroid_retention_metrics(needle, centroids, radii)
    ordinary_geometry = centroid_retention_metrics(ordinary, centroids, radii)
    needle_change = relative_change_metrics(clean, needle)
    ordinary_change = relative_change_metrics(clean, ordinary)

    layer_row = {
        "schema_version": LAYER_SCHEMA,
        "model_label": model_label,
        "seed": int(seed),
        "gold_count": 10,
        "layer": int(layer),
        "pooling": pooling,
        "running_index_count": int(clean.shape[0]),
        "hidden_width": int(clean.shape[1]),
        "clean_correct_distance": clean_geometry["mean_correct_distance"],
        "needle_correct_distance": needle_geometry["mean_correct_distance"],
        "ordinary_correct_distance": ordinary_geometry["mean_correct_distance"],
        "needle_correct_distance_damage": needle_geometry["mean_correct_distance"]
        - clean_geometry["mean_correct_distance"],
        "ordinary_correct_distance_damage": ordinary_geometry["mean_correct_distance"]
        - clean_geometry["mean_correct_distance"],
        "correct_distance_specificity": needle_geometry["mean_correct_distance"]
        - ordinary_geometry["mean_correct_distance"],
        "clean_margin": clean_geometry["mean_margin"],
        "needle_margin": needle_geometry["mean_margin"],
        "ordinary_margin": ordinary_geometry["mean_margin"],
        "needle_margin_damage": clean_geometry["mean_margin"]
        - needle_geometry["mean_margin"],
        "ordinary_margin_damage": clean_geometry["mean_margin"]
        - ordinary_geometry["mean_margin"],
        "margin_damage_specificity": ordinary_geometry["mean_margin"]
        - needle_geometry["mean_margin"],
        "clean_accuracy": clean_geometry["accuracy"],
        "needle_accuracy": needle_geometry["accuracy"],
        "ordinary_accuracy": ordinary_geometry["accuracy"],
        "needle_accuracy_damage": clean_geometry["accuracy"]
        - needle_geometry["accuracy"],
        "ordinary_accuracy_damage": clean_geometry["accuracy"]
        - ordinary_geometry["accuracy"],
        "accuracy_damage_specificity": ordinary_geometry["accuracy"]
        - needle_geometry["accuracy"],
        "clean_mad": clean_geometry["mad"],
        "needle_mad": needle_geometry["mad"],
        "ordinary_mad": ordinary_geometry["mad"],
        "mad_damage_specificity": needle_geometry["mad"]
        - ordinary_geometry["mad"],
        "needle_raw_rms": needle_change["raw_rms"],
        "ordinary_raw_rms": ordinary_change["raw_rms"],
        "raw_rms_specificity": needle_change["raw_rms"]
        - ordinary_change["raw_rms"],
        "needle_relative_rms": needle_change["relative_rms"],
        "ordinary_relative_rms": ordinary_change["relative_rms"],
        "relative_rms_specificity": needle_change["relative_rms"]
        - ordinary_change["relative_rms"],
        "needle_cosine_distance": needle_change["mean_cosine_distance"],
        "ordinary_cosine_distance": ordinary_change["mean_cosine_distance"],
        "cosine_specificity": needle_change["mean_cosine_distance"]
        - ordinary_change["mean_cosine_distance"],
    }
    if not all(
        math.isfinite(float(value))
        for key, value in layer_row.items()
        if key
        not in {
            "schema_version",
            "model_label",
            "pooling",
        }
    ):
        raise RuntimeError("A layer-level retention metric is non-finite")

    occurrence_rows: list[dict[str, Any]] = []
    for occurrence in range(clean.shape[0]):
        occurrence_rows.append(
            {
                "schema_version": OCCURRENCE_SCHEMA,
                "model_label": model_label,
                "seed": int(seed),
                "gold_count": 10,
                "layer": int(layer),
                "pooling": pooling,
                "running_index": occurrence + 1,
                "discovery_radius": float(radii[occurrence].item()),
                "clean_correct_distance": float(
                    clean_geometry["correct_distance"][occurrence].item()
                ),
                "needle_correct_distance": float(
                    needle_geometry["correct_distance"][occurrence].item()
                ),
                "ordinary_correct_distance": float(
                    ordinary_geometry["correct_distance"][occurrence].item()
                ),
                "clean_margin": float(clean_geometry["margin"][occurrence].item()),
                "needle_margin": float(needle_geometry["margin"][occurrence].item()),
                "ordinary_margin": float(
                    ordinary_geometry["margin"][occurrence].item()
                ),
                "clean_prediction": int(clean_geometry["prediction"][occurrence]) + 1,
                "needle_prediction": int(needle_geometry["prediction"][occurrence])
                + 1,
                "ordinary_prediction": int(
                    ordinary_geometry["prediction"][occurrence]
                )
                + 1,
                "needle_l2_change": float(
                    needle_change["per_occurrence_l2"][occurrence].item()
                ),
                "ordinary_l2_change": float(
                    ordinary_change["per_occurrence_l2"][occurrence].item()
                ),
                "needle_relative_l2_change": float(
                    needle_change["per_occurrence_relative_l2"][occurrence].item()
                ),
                "ordinary_relative_l2_change": float(
                    ordinary_change["per_occurrence_relative_l2"][occurrence].item()
                ),
                "needle_cosine_distance": float(
                    needle_change["per_occurrence_cosine_distance"][occurrence].item()
                ),
                "ordinary_cosine_distance": float(
                    ordinary_change["per_occurrence_cosine_distance"][occurrence].item()
                ),
            }
        )
    return layer_row, occurrence_rows


def capture(
    model: Any,
    adapter: Any,
    encoding: PromptEncoding,
    spans: Sequence[Any],
    layers: Sequence[int],
) -> tuple[dict[str, torch.Tensor], float]:
    started = time.perf_counter()
    states = capture_span_states(
        model,
        adapter,
        encoding,
        spans=spans,
        layers=layers,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return states, float(time.perf_counter() - started)


def build_discovery_bank(
    *,
    model: Any,
    adapter: Any,
    tokenizer: Any,
    spec: Any,
    v4_config: V4Config,
    stimuli: Mapping[tuple[int, int], Mapping[str, Any]],
    seeds: Sequence[int],
    layers: Sequence[int],
    output: Path,
    config_hash: str,
    stimulus_hash: str,
) -> dict[str, torch.Tensor]:
    bank_path = output / "discovery_centroids.pt"
    audit_path = output / "discovery_audit.json"
    if bank_path.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("status") == "PASS"
            and audit.get("experiment_config_sha256") == config_hash
            and audit.get("stimulus_sha256") == stimulus_hash
            and audit.get("centroid_bank_sha256") == sha256(bank_path)
        ):
            payload = torch.load(bank_path, map_location="cpu", weights_only=False)
            print("REUSE audited discovery centroid bank", flush=True)
            return payload
        raise RuntimeError("Existing discovery centroid bank failed provenance checks")

    sums: dict[str, torch.Tensor] = {}
    squared_norm_sums: dict[str, torch.Tensor] = {}
    unit_rows: list[dict[str, Any]] = []
    for seed in seeds:
        encoding = render_v4_prompt(
            stimuli[(int(seed), 10)],
            tokenizer=tokenizer,
            model_spec=spec,
            config=v4_config,
            answer_format="numeric",
        )
        if len(encoding.needle_spans) != 10:
            raise RuntimeError(f"Discovery seed {seed} does not have ten needle spans")
        states, elapsed = capture(
            model, adapter, encoding, encoding.needle_spans, layers
        )
        if tuple(int(value) for value in states["layer_indices"].tolist()) != tuple(
            layers
        ):
            raise RuntimeError("Discovery capture layer registry drifted")
        for site in SITES:
            value = states[site].double()
            if value.shape[:2] != (len(layers), 10):
                raise RuntimeError(f"Unexpected discovery {site} shape: {value.shape}")
            if site not in sums:
                sums[site] = torch.zeros_like(value)
                squared_norm_sums[site] = torch.zeros(
                    value.shape[:2], dtype=torch.float64
                )
            sums[site] += value
            squared_norm_sums[site] += value.square().sum(dim=-1)
        unit_rows.append(
            {
                "schema_version": "realistic_niah_v4_4_5_state_retention_discovery_unit_v1",
                "seed": int(seed),
                "gold_count": 10,
                "sequence_length": int(encoding.sequence_length),
                "query_position": int(encoding.query_position),
                "needle_spans": len(encoding.needle_spans),
                "input_sha256": input_sha256(encoding),
                "capture_seconds": elapsed,
            }
        )
        print(f"DISCOVERY seed={seed} seconds={elapsed:.2f}", flush=True)

    sample_count = len(seeds)
    if sample_count < 2:
        raise RuntimeError("At least two discovery seeds are required")
    payload: dict[str, Any] = {
        "schema_version": "realistic_niah_v4_4_5_state_retention_centroid_bank_v1",
        "layer_indices": torch.tensor(layers, dtype=torch.long),
        "running_indices": torch.arange(1, 11, dtype=torch.long),
        "discovery_seeds": torch.tensor(seeds, dtype=torch.long),
    }
    audit_sites: dict[str, Any] = {}
    for site in SITES:
        centroid = sums[site] / float(sample_count)
        centroid_squared_norm = centroid.square().sum(dim=-1)
        sse = squared_norm_sums[site] - sample_count * centroid_squared_norm
        radius = torch.sqrt((sse.clamp_min(0.0) / float(sample_count - 1)))
        if not bool(torch.all(torch.isfinite(centroid))) or not bool(
            torch.all(torch.isfinite(radius))
        ):
            raise RuntimeError(f"Non-finite discovery centroid/radius at {site}")
        if not bool(torch.all(radius > 0)):
            raise RuntimeError(f"Zero discovery radius at {site}")
        payload[f"{site}_centroids"] = centroid.float()
        payload[f"{site}_radii"] = radius.float()
        audit_sites[site] = {
            "centroid_shape": list(centroid.shape),
            "radius_shape": list(radius.shape),
            "radius_min": float(radius.min().item()),
            "radius_max": float(radius.max().item()),
            "finite": True,
        }

    temporary = bank_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(bank_path)
    write_jsonl(output / "discovery_units.jsonl", unit_rows)
    audit = {
        "schema_version": "realistic_niah_v4_4_5_state_retention_discovery_audit_v1",
        "status": "PASS",
        "discovery_seed_count": sample_count,
        "discovery_seeds": list(map(int, seeds)),
        "gold_count": 10,
        "running_indices": list(range(1, 11)),
        "layers": list(map(int, layers)),
        "pooling_sites": list(SITES),
        "sites": audit_sites,
        "experiment_config_sha256": config_hash,
        "stimulus_sha256": stimulus_hash,
        "centroid_bank_sha256": sha256(bank_path),
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def audit_complete(
    output: Path,
    *,
    model_label: str,
    seeds: Sequence[int],
    layers: Sequence[int],
    stimulus_hash: str,
    config_hash: str,
) -> dict[str, Any]:
    layer_rows = read_jsonl(output / "layer_seed_metrics.jsonl")
    occurrence_rows = read_jsonl(output / "occurrence_metrics.jsonl")
    unit_rows = read_jsonl(output / "confirmation_units.jsonl")
    expected_layer_keys = {
        (model_label, int(seed), int(layer), site)
        for seed in seeds
        for layer in layers
        for site in SITES
    }
    expected_occurrence_keys = {
        (model_label, int(seed), int(layer), site, running_index)
        for seed in seeds
        for layer in layers
        for site in SITES
        for running_index in range(1, 11)
    }
    layer_keys = [
        (
            str(row["model_label"]),
            int(row["seed"]),
            int(row["layer"]),
            str(row["pooling"]),
        )
        for row in layer_rows
    ]
    occurrence_keys = [
        (
            str(row["model_label"]),
            int(row["seed"]),
            int(row["layer"]),
            str(row["pooling"]),
            int(row["running_index"]),
        )
        for row in occurrence_rows
    ]
    expected_unit_keys = {(model_label, int(seed)) for seed in seeds}
    unit_keys = [(str(row["model_label"]), int(row["seed"])) for row in unit_rows]
    layer_numeric = (
        "clean_correct_distance",
        "needle_correct_distance",
        "ordinary_correct_distance",
        "correct_distance_specificity",
        "clean_margin",
        "needle_margin",
        "ordinary_margin",
        "margin_damage_specificity",
        "clean_accuracy",
        "needle_accuracy",
        "ordinary_accuracy",
        "accuracy_damage_specificity",
        "needle_relative_rms",
        "ordinary_relative_rms",
        "relative_rms_specificity",
        "needle_cosine_distance",
        "ordinary_cosine_distance",
        "cosine_specificity",
    )
    finite = all(
        math.isfinite(float(row[field]))
        for row in layer_rows
        for field in layer_numeric
    )
    discovery_audit = json.loads(
        (output / "discovery_audit.json").read_text(encoding="utf-8")
    )
    audit = {
        "schema_version": "realistic_niah_v4_4_5_state_retention_run_audit_v1",
        "status": "PASS",
        "model_label": model_label,
        "layer_rows": len(layer_rows),
        "expected_layer_rows": len(expected_layer_keys),
        "unique_layer_keys": len(set(layer_keys)),
        "exact_layer_key_coverage": set(layer_keys) == expected_layer_keys,
        "occurrence_rows": len(occurrence_rows),
        "expected_occurrence_rows": len(expected_occurrence_keys),
        "unique_occurrence_keys": len(set(occurrence_keys)),
        "exact_occurrence_key_coverage": set(occurrence_keys)
        == expected_occurrence_keys,
        "confirmation_unit_rows": len(unit_rows),
        "expected_confirmation_unit_rows": len(expected_unit_keys),
        "unique_confirmation_unit_keys": len(set(unit_keys)),
        "exact_confirmation_unit_coverage": set(unit_keys) == expected_unit_keys,
        "finite_metrics": finite,
        "token_budget_matched": all(
            int(row["needle_token_budget"]) == int(row["ordinary_token_budget"])
            and int(row["needle_token_budget"]) > 0
            for row in unit_rows
        ),
        "sequence_and_query_preserved": all(
            bool(row["sequence_length_preserved"])
            and bool(row["query_position_preserved"])
            for row in unit_rows
        ),
        "discovery_audit_status": discovery_audit.get("status"),
        "stimulus_sha256": stimulus_hash,
        "experiment_config_sha256": config_hash,
    }
    checks = (
        audit["layer_rows"] == audit["expected_layer_rows"],
        audit["unique_layer_keys"] == audit["expected_layer_rows"],
        audit["exact_layer_key_coverage"],
        audit["occurrence_rows"] == audit["expected_occurrence_rows"],
        audit["unique_occurrence_keys"] == audit["expected_occurrence_rows"],
        audit["exact_occurrence_key_coverage"],
        audit["confirmation_unit_rows"]
        == audit["expected_confirmation_unit_rows"],
        audit["unique_confirmation_unit_keys"]
        == audit["expected_confirmation_unit_rows"],
        audit["exact_confirmation_unit_coverage"],
        audit["finite_metrics"],
        audit["token_budget_matched"],
        audit["sequence_and_query_preserved"],
        audit["discovery_audit_status"] == "PASS",
    )
    if not all(checks):
        audit["status"] = "FAIL"
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure held-out retention of clean running-index centroids after "
            "needle versus token-budget-matched ordinary corruption."
        )
    )
    parser.add_argument("--model", required=True, choices=("Qwen3-8B", "Gemma4-E4B"))
    parser.add_argument("--stimuli", required=True)
    parser.add_argument(
        "--stimuli-config", default="configs/realistic_niah_v4_4_5_stimuli.json"
    )
    parser.add_argument(
        "--experiment-config",
        default="configs/realistic_niah_v4_4_5_state_retention.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    config_path = Path(args.experiment_config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "realistic_niah_v4_4_5_state_retention_v1":
        raise ValueError("Unexpected state-retention config schema")
    model_label = str(args.model)
    discovery_seeds = tuple(int(value) for value in config["discovery_seeds"])
    confirmation_seeds = tuple(int(value) for value in config["confirmation_seeds"])
    if set(discovery_seeds).intersection(confirmation_seeds):
        raise ValueError("Discovery and confirmation seeds overlap")
    if tuple(int(value) for value in config["running_indices"]) != tuple(range(1, 11)):
        raise ValueError("Running-index registry must be exactly 1..10")
    if int(config["gold_count"]) != 10:
        raise ValueError("This assay is frozen to final count 10")
    if tuple(config["pooling_sites"]) != SITES:
        raise ValueError("Pooling sites drifted from span_end/span_mean")
    layers = tuple(int(value) for value in config["layers"][model_label])

    stimuli_path = Path(args.stimuli).resolve()
    stimulus_hash = sha256(stimuli_path)
    config_hash = sha256(config_path)
    if stimulus_hash != str(config["stimulus_sha256"]):
        raise RuntimeError(
            f"Frozen stimulus hash mismatch: {stimulus_hash} != {config['stimulus_sha256']}"
        )
    all_seeds = set(discovery_seeds + confirmation_seeds)
    stimuli = {
        (int(row["seed"]), int(row["gold_count"])): row
        for row in load_stimuli(stimuli_path)
        if str(row.get("design_variant")) == "v4.4"
        and int(row["seed"]) in all_seeds
        and int(row["gold_count"]) == 10
    }
    expected_stimuli = {(seed, 10) for seed in all_seeds}
    if set(stimuli) != expected_stimuli:
        raise RuntimeError(
            f"Missing frozen count-10 stimuli: {sorted(expected_stimuli - set(stimuli))}"
        )

    output = Path(args.output_dir).resolve() / model_label
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "run_provenance.json"
    runner_path = Path(__file__).resolve()
    spec = resolve_model_spec(model_label)
    provenance = {
        "schema_version": "realistic_niah_v4_4_5_state_retention_run_v1",
        "model_label": model_label,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "stimuli": str(stimuli_path),
        "stimulus_sha256": stimulus_hash,
        "experiment_config": str(config_path),
        "experiment_config_sha256": config_hash,
        "runner_path": str(runner_path),
        "runner_sha256": sha256(runner_path),
        "runtime_files": {
            "modeling": {
                "path": str(Path(capture_span_states.__code__.co_filename).resolve()),
                "sha256": sha256(
                    Path(capture_span_states.__code__.co_filename).resolve()
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
        "discovery_seeds": list(discovery_seeds),
        "confirmation_seeds": list(confirmation_seeds),
        "gold_count": 10,
        "layers": list(layers),
        "pooling_sites": list(SITES),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        for key in ("model_label", "stimulus_sha256", "experiment_config_sha256"):
            if existing.get(key) != provenance.get(key):
                raise RuntimeError(f"Existing output provenance mismatch for {key}")
    provenance_path.write_text(
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

    discovery = build_discovery_bank(
        model=model,
        adapter=adapter,
        tokenizer=tokenizer,
        spec=spec,
        v4_config=v4_config,
        stimuli=stimuli,
        seeds=discovery_seeds,
        layers=layers,
        output=output,
        config_hash=config_hash,
        stimulus_hash=stimulus_hash,
    )

    layer_path = output / "layer_seed_metrics.jsonl"
    occurrence_path = output / "occurrence_metrics.jsonl"
    unit_path = output / "confirmation_units.jsonl"
    existing_layer_keys = {
        (
            int(row["seed"]),
            int(row["layer"]),
            str(row["pooling"]),
        )
        for row in read_jsonl(layer_path)
    }
    existing_occurrence_keys = {
        (
            int(row["seed"]),
            int(row["layer"]),
            str(row["pooling"]),
            int(row["running_index"]),
        )
        for row in read_jsonl(occurrence_path)
    }
    existing_unit_seeds = {int(row["seed"]) for row in read_jsonl(unit_path)}

    for seed in confirmation_seeds:
        expected_seed_layers = {
            (seed, layer, site) for layer in layers for site in SITES
        }
        expected_seed_occurrences = {
            (seed, layer, site, running_index)
            for layer in layers
            for site in SITES
            for running_index in range(1, 11)
        }
        if (
            expected_seed_layers.issubset(existing_layer_keys)
            and expected_seed_occurrences.issubset(existing_occurrence_keys)
            and seed in existing_unit_seeds
        ):
            print(f"SKIP confirmation seed={seed} complete", flush=True)
            continue

        encoding = render_v4_prompt(
            stimuli[(seed, 10)],
            tokenizer=tokenizer,
            model_spec=spec,
            config=v4_config,
            answer_format="numeric",
        )
        if len(encoding.needle_spans) != 10:
            raise RuntimeError(f"Confirmation seed {seed} lacks ten needle spans")
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

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        clean_states, clean_seconds = capture(
            model, adapter, encoding, encoding.needle_spans, layers
        )
        needle_states, needle_seconds = capture(
            model, adapter, needle_corrupt, encoding.needle_spans, layers
        )
        ordinary_states, ordinary_seconds = capture(
            model, adapter, ordinary_corrupt, encoding.needle_spans, layers
        )
        for captured in (clean_states, needle_states, ordinary_states):
            if tuple(int(value) for value in captured["layer_indices"].tolist()) != layers:
                raise RuntimeError("Confirmation capture layer registry drifted")

        for layer_offset, layer in enumerate(layers):
            for site in SITES:
                layer_key = (seed, layer, site)
                occurrence_keys = {
                    (seed, layer, site, running_index)
                    for running_index in range(1, 11)
                }
                if layer_key in existing_layer_keys and occurrence_keys.issubset(
                    existing_occurrence_keys
                ):
                    continue
                layer_row, occurrence_rows = layer_retention_rows(
                    model_label=model_label,
                    seed=seed,
                    layer=layer,
                    pooling=site,
                    clean=clean_states[site][layer_offset],
                    needle=needle_states[site][layer_offset],
                    ordinary=ordinary_states[site][layer_offset],
                    centroids=discovery[f"{site}_centroids"][layer_offset],
                    radii=discovery[f"{site}_radii"][layer_offset],
                )
                if layer_key not in existing_layer_keys:
                    append_jsonl(layer_path, layer_row)
                    existing_layer_keys.add(layer_key)
                for occurrence_row in occurrence_rows:
                    occurrence_key = (
                        seed,
                        layer,
                        site,
                        int(occurrence_row["running_index"]),
                    )
                    if occurrence_key not in existing_occurrence_keys:
                        append_jsonl(occurrence_path, occurrence_row)
                        existing_occurrence_keys.add(occurrence_key)

        if seed not in existing_unit_seeds:
            append_jsonl(
                unit_path,
                {
                    "schema_version": UNIT_SCHEMA,
                    "model_label": model_label,
                    "seed": int(seed),
                    "gold_count": 10,
                    "sequence_length": int(encoding.sequence_length),
                    "query_position": int(encoding.query_position),
                    "needle_span_count": len(encoding.needle_spans),
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
            existing_unit_seeds.add(seed)
        print(
            f"CONFIRMATION seed={seed} layers={len(layers)} "
            f"seconds={clean_seconds + needle_seconds + ordinary_seconds:.2f}",
            flush=True,
        )
        del clean_states, needle_states, ordinary_states

    audit = audit_complete(
        output,
        model_label=model_label,
        seeds=confirmation_seeds,
        layers=layers,
        stimulus_hash=stimulus_hash,
        config_hash=config_hash,
    )
    (output / "complete.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if audit["status"] != "PASS":
        raise RuntimeError("State-retention run audit failed")
    (output / ".RUN_COMPLETE").write_text("PASS\n", encoding="utf-8")


if __name__ == "__main__":
    main()
