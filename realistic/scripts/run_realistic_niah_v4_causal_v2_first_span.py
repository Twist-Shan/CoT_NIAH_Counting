from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from realistic_niah_v4.behavior import capture_generation_labels
from realistic_niah_v4.causal_generation import (
    causal_v2_prompt_span_alignment_table,
    compare_head_ablation_v2_to_random,
    load_generation_labels,
    run_generation_head_ablation_v2,
    run_generation_residual_patching_v2,
    summarize_generation_head_ablation_v2,
    summarize_generation_residual_patching_v2,
)
from realistic_niah_v4.causal_v2 import (
    CausalV2Design,
    confirmation_statistics,
    head_ablation_confirmation_seed_effects,
    head_ablation_confirmation_statistics,
    head_phenotype_scores,
    load_head_phenotype_registry,
    stable_layer_k_conditions,
    write_head_phenotype_registry,
)
from realistic_niah_v4.correct_interventions import summarize_ablation_n_diagnostics
from realistic_niah_v4.correct_only_slices import clean_correct_ablation_rows
from realistic_niah_v4.geometric_steering import (
    LayerSetSteeringPlan,
    capture_query_residual_shard,
    centroid_geometry_tables,
    fit_count_centroids,
    load_centroid_bundle,
    run_generation_layer_set_centroid_delta,
    save_centroid_bundle,
)
from realistic_niah_v4.modeling import load_registered_model, load_registered_tokenizer
from realistic_niah_v4.prompts import PromptEncoding, render_v4_prompt
from realistic_niah_v4.spec import V4Config, resolve_model_spec
from realistic_niah_v4.stimuli import load_stimuli
from first_span_m10_generation import run_generation_head_ablation_v2_m10


GPU_STAGES = {
    "baseline",
    "ablation",
    "prompt-patching",
    "answer-patching",
    "steering-centroids",
    "steering",
}

ATTENTION_PHENOTYPE_COLUMNS = (
    "stimulus_id",
    "design_variant",
    "model_label",
    "seed",
    "split",
    "count",
    "layer",
    "head",
    "broad_mass",
    "broad_coverage",
    "broad_primary",
    "needle_span_masses",
)


def _json_default(value: Any) -> Any:
    """Convert NumPy scalar metadata without weakening JSON type checks."""

    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_normalized_text(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv_gzip_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read_attention_csv(
    path: Path,
    *,
    model_label: str,
) -> pd.DataFrame:
    """Read only causal-v2 phenotype columns and the requested model rows."""

    frames: list[pd.DataFrame] = []
    try:
        chunks = pd.read_csv(
            path,
            compression="infer",
            usecols=list(ATTENTION_PHENOTYPE_COLUMNS),
            chunksize=100_000,
        )
        for chunk in chunks:
            selected = chunk[
                chunk["model_label"].astype(str).eq(str(model_label))
                & chunk["design_variant"].astype(str).eq("v4.4")
                & chunk["split"].astype(str).eq("discovery")
            ].copy()
            if not selected.empty:
                frames.append(selected)
    except ValueError as error:
        raise ValueError(
            f"Attention source {path} lacks the raw full-span phenotype "
            f"columns {list(ATTENTION_PHENOTYPE_COLUMNS)}. Use the original "
            "attention/capture directory or attention_capture_index.jsonl, "
            "not pooling_head_detail.csv.gz."
        ) from error
    if not frames:
        raise ValueError(
            f"Attention source {path} has no {model_label}/v4.4/discovery rows"
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def _resolve_attention_capture_index(source: Path, model_label: str) -> Path | None:
    if source.is_file() and source.name == "attention_capture_index.jsonl":
        return source
    if not source.is_dir():
        return None
    candidates = (
        source / "attention_capture_index.jsonl",
        source
        / model_label
        / "numeric"
        / "attention"
        / "capture"
        / "attention_capture_index.jsonl",
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1 and existing[0].resolve() != existing[1].resolve():
        raise RuntimeError(
            f"Attention source is ambiguous for {model_label}: {existing}"
        )
    return existing[0] if existing else None


def _load_attention_phenotype_source(
    source: str | Path,
    *,
    model_label: str,
    expected_seeds: Sequence[int],
    expected_counts: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load raw full-span discovery metrics from a CSV, index, capture, or run root."""

    source_path = Path(source).resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    capture_index = _resolve_attention_capture_index(source_path, model_label)
    file_records: list[dict[str, Any]] = []
    if capture_index is not None:
        index_rows = [
            json.loads(line)
            for line in capture_index.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        selected_index = [
            row
            for row in index_rows
            if str(row.get("model_label")) == str(model_label)
            and str(row.get("design_variant")) == "v4.4"
            and str(row.get("split")) == "discovery"
            and int(row.get("seed", -1)) in {int(value) for value in expected_seeds}
            and int(row.get("count", -1)) in {int(value) for value in expected_counts}
        ]
        selected_index.sort(key=lambda row: (int(row["seed"]), int(row["count"])))
        frames: list[pd.DataFrame] = []
        for row in selected_index:
            shard_path = (capture_index.parent / str(row["shard_path"])).resolve()
            if not shard_path.is_file():
                raise FileNotFoundError(shard_path)
            frames.append(_read_attention_csv(shard_path, model_label=model_label))
            file_records.append(
                {
                    "stimulus_id": str(row["stimulus_id"]),
                    "relative_shard_path": str(row["shard_path"]),
                    "bytes": int(shard_path.stat().st_size),
                    "sha256": _sha256_file(shard_path),
                }
            )
        if not frames:
            raise ValueError(
                f"Attention capture index {capture_index} has no registered "
                f"{model_label}/v4.4/discovery shards"
            )
        detail = pd.concat(frames, ignore_index=True, sort=False)
        source_kind = "attention_capture_index"
        resolved_source = capture_index
        registered_suffix = (
            str(model_label),
            "numeric",
            "attention",
            "capture",
            "attention_capture_index.jsonl",
        )
        portable_source = (
            "/".join(registered_suffix)
            if tuple(capture_index.parts[-len(registered_suffix) :])
            == registered_suffix
            else None
        )
        source_index_sha256 = _sha256_file(capture_index)
    elif source_path.is_file():
        detail = _read_attention_csv(source_path, model_label=model_label)
        file_records.append(
            {
                "relative_shard_path": source_path.name,
                "bytes": int(source_path.stat().st_size),
                "sha256": _sha256_file(source_path),
            }
        )
        source_kind = "consolidated_attention_csv"
        resolved_source = source_path
        portable_source = None
        source_index_sha256 = None
    else:
        raise ValueError(
            f"Could not resolve an attention capture index below {source_path} "
            f"for {model_label}"
        )

    expected_grid = {
        (int(seed), int(count)) for seed in expected_seeds for count in expected_counts
    }
    observed_grid = set(
        zip(
            pd.to_numeric(detail["seed"], errors="raise").astype(int),
            pd.to_numeric(detail["count"], errors="raise").astype(int),
        )
    )
    if observed_grid != expected_grid:
        missing = sorted(expected_grid - observed_grid)
        extra = sorted(observed_grid - expected_grid)
        raise RuntimeError(
            f"Attention discovery grid mismatch: missing={missing[:20]}, "
            f"extra={extra[:20]}"
        )
    identity = ["stimulus_id", "layer", "head"]
    if detail.duplicated(identity).any():
        raise RuntimeError(
            "Attention phenotype source has duplicate stimulus/head rows"
        )
    heads_per_prompt = detail.groupby(["seed", "count"], sort=False).size()
    if heads_per_prompt.nunique() != 1:
        raise RuntimeError(
            "Attention phenotype source does not contain the same head grid for "
            "every discovery prompt"
        )
    fingerprint_payload = {
        "model_label": str(model_label),
        "files": [
            {key: row[key] for key in sorted(row) if key != "bytes"}
            for row in file_records
        ],
    }
    manifest = {
        "schema_version": "realistic_niah_v4_causal_v2_attention_source_v1",
        "source_kind": source_kind,
        "resolved_source": str(resolved_source),
        "portable_run_relative_source": portable_source,
        "source_index_sha256": source_index_sha256,
        "model_label": str(model_label),
        "design_variant": "v4.4",
        "split": "discovery",
        "seeds": sorted(int(value) for value in expected_seeds),
        "counts": sorted(int(value) for value in expected_counts),
        "prompt_count": len(observed_grid),
        "rows": int(len(detail)),
        "heads_per_prompt": int(heads_per_prompt.iloc[0]),
        "file_count": len(file_records),
        "files": file_records,
        "aggregate_sha256": hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    return detail, manifest


def _git_head(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _runtime_payload(repo_root: Path) -> dict[str, Any]:
    return {
        "unix_time": time.time(),
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "git_head": _git_head(repo_root),
        "pid": os.getpid(),
        "argv": sys.argv,
    }


def _implementation_fingerprint(repo_root: Path) -> str:
    relative_paths = (
        "scripts/run_realistic_niah_v4_causal_v2.py",
        "src/realistic_niah_v4/attention.py",
        "src/realistic_niah_v4/behavior.py",
        "src/realistic_niah_v4/causal_generation.py",
        "src/realistic_niah_v4/causal_v2.py",
        "src/realistic_niah_v4/correct_interventions.py",
        "src/realistic_niah_v4/correct_only_slices.py",
        "src/realistic_niah_v4/geometric_steering.py",
        "src/realistic_niah_v4/modeling.py",
        "src/realistic_niah_v4/prompts.py",
        "src/realistic_niah_v4/stimuli.py",
    )
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _stage_root(
    causal_root: Path,
    *,
    family: str,
    phase: str,
    design: dict[str, Any],
) -> tuple[Path, str]:
    design_hash = _json_hash(design)
    root = causal_root / family / f"{phase}_{design_hash}"
    root.mkdir(parents=True, exist_ok=True)
    design_path = root / "design.json"
    if design_path.exists():
        observed = json.loads(design_path.read_text(encoding="utf-8"))
        if observed != design:
            raise RuntimeError(f"Existing design mismatch: {design_path}")
    else:
        _write_json_atomic(design_path, design)
    return root, design_hash


def _select_rows(
    stimuli_path: Path,
    *,
    seeds: Sequence[int],
    counts: Sequence[int],
) -> list[dict[str, Any]]:
    seed_set = {int(value) for value in seeds}
    count_set = {int(value) for value in counts}
    rows = [
        row
        for row in load_stimuli(stimuli_path)
        if str(row.get("design_variant")) == "v4.4"
        and int(row["seed"]) in seed_set
        and int(row["gold_count"]) in count_set
    ]
    expected = len(seed_set) * len(count_set)
    if len(rows) != expected:
        raise ValueError(
            f"Causal-v2 stimulus selection has {len(rows)} rows; expected {expected}"
        )
    keys = {(int(row["seed"]), int(row["gold_count"])) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("Causal-v2 stimulus selection contains duplicate cells")
    return rows


def _render(
    rows: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    model_label: str,
    config: V4Config,
) -> list[PromptEncoding]:
    model_spec = resolve_model_spec(model_label)
    return [
        render_v4_prompt(
            row,
            tokenizer=tokenizer,
            model_spec=model_spec,
            config=config,
            answer_format="numeric",
        )
        for row in rows
    ]


def _load_model(
    *,
    model_label: str,
    config: V4Config,
    cache_dir: str | None,
    device_map: str,
) -> tuple[Any, Any, Any]:
    return load_registered_model(
        resolve_model_spec(model_label),
        cache_dir=cache_dir,
        device_map=device_map,
        torch_dtype=config.model_torch_dtype,
        attention_backend=config.attention_prefix_backend,
    )


def _load_shards(index_rows: Sequence[dict[str, Any]], root: Path) -> pd.DataFrame:
    if not index_rows:
        raise ValueError("No causal-v2 shards were indexed")
    frames = [
        pd.read_csv(root / str(row["shard_path"]), compression="gzip")
        for row in index_rows
    ]
    return pd.concat(frames, ignore_index=True)


def _phase_seeds(design: CausalV2Design, phase: str) -> tuple[int, ...]:
    if phase == "screen":
        return design.screen_seeds
    if phase == "confirmation":
        return design.confirmation_seeds
    raise ValueError(f"Unknown causal-v2 phase: {phase}")


def _selection_filter(
    path: Path,
    *,
    family: str,
    model_label: str,
) -> tuple[set[tuple[str, str, int, int]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("family") != family or payload.get("selection_split") != "screen":
        raise ValueError(f"Selection manifest does not match {family}")
    selected = payload.get("selected")
    if not isinstance(selected, list):
        raise ValueError("Selection manifest has no selected condition list")
    result = {
        (
            str(item["site"]),
            str(item["patch_protocol"]),
            int(item["start_layer"]),
            int(item["k"]),
        )
        for item in selected
        if str(item["model_label"]) == str(model_label)
    }
    if not result:
        raise ValueError(f"No stable {family} conditions for {model_label}")
    return result, payload


def _steering_plans_from_selection(
    path: Path,
    *,
    model_label: str,
    k: int,
) -> tuple[LayerSetSteeringPlan, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("family") != "steering"
        or payload.get("selection_split") != "screen"
    ):
        raise ValueError("Steering confirmation requires a screen selection manifest")
    layers = sorted(
        {
            int(str(item["layer_set"]))
            for item in payload.get("selected", [])
            if str(item["model_label"]) == str(model_label)
            and int(item["k"]) == int(k)
            and str(item["steering_protocol"]) == "single_layer"
            and "+" not in str(item["layer_set"])
        }
    )
    if not layers:
        return ()
    plans = [LayerSetSteeringPlan(layers=(layer,), alpha=1.0) for layer in layers]
    if len(layers) >= 2:
        plans.append(LayerSetSteeringPlan(layers=tuple(layers), alpha=1.0))
    return tuple(plans)


def _condition_key_series(frame: pd.DataFrame, family: str) -> pd.Series:
    if family in {"prompt_patching", "answer_patching"}:
        required = {"site", "patch_protocol", "start_layer", "k"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{family} detail is missing condition keys: {missing}")
        return pd.Series(
            list(
                zip(
                    frame["site"].astype(str),
                    frame["patch_protocol"].astype(str),
                    pd.to_numeric(frame["start_layer"], errors="raise").astype(int),
                    pd.to_numeric(frame["k"], errors="raise").astype(int),
                )
            ),
            index=frame.index,
            dtype=object,
        )
    if family == "steering":
        required = {"steering_protocol", "layer_set", "k"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"steering detail is missing condition keys: {missing}")
        return pd.Series(
            list(
                zip(
                    frame["steering_protocol"].astype(str),
                    frame["layer_set"].astype(str),
                    pd.to_numeric(frame["k"], errors="raise").astype(int),
                )
            ),
            index=frame.index,
            dtype=object,
        )
    raise KeyError(f"Unknown causal-v2 family: {family}")


def _selected_confirmation_detail(
    screen: pd.DataFrame,
    confirmation: pd.DataFrame,
    *,
    family: str,
    selection_payload: dict[str, Any],
    model_label: str,
    design: CausalV2Design,
    strict_seed_sets: bool = True,
) -> pd.DataFrame:
    """Keep only preregistered stable conditions plus frozen steering multis.

    Screen-only conditions that failed the stability rule are never allowed
    into the final confirmation statistics.  Selected singleton conditions
    combine five screen and five held-out seeds.  A steering multi-layer plan
    is constructed only after selection and therefore remains explicitly a
    five-held-out-seed estimate.
    """

    if selection_payload.get("family") != family:
        raise ValueError("Selection manifest family does not match confirmation")
    if selection_payload.get("selection_split") != "screen":
        raise ValueError("Confirmation requires a screen selection manifest")
    selected = [
        item
        for item in selection_payload.get("selected", [])
        if str(item.get("model_label")) == str(model_label)
    ]
    if not selected:
        raise ValueError(f"No selected {family} conditions for {model_label}")

    if family in {"prompt_patching", "answer_patching"}:
        selected_keys = {
            (
                str(item["site"]),
                str(item["patch_protocol"]),
                int(item["start_layer"]),
                int(item["k"]),
            )
            for item in selected
        }
        screen_keys = _condition_key_series(screen, family)
        confirmation_keys = _condition_key_series(confirmation, family)
        kept_screen = screen[screen_keys.isin(selected_keys)].copy()
        kept_confirmation = confirmation[confirmation_keys.isin(selected_keys)].copy()
        expected_confirmation_keys = selected_keys
    elif family == "steering":
        selected_keys = {
            (
                str(item["steering_protocol"]),
                str(item["layer_set"]),
                int(item["k"]),
            )
            for item in selected
        }
        if any(protocol != "single_layer" for protocol, _layer, _k in selected_keys):
            raise ValueError("Steering screen may select only single-layer conditions")
        screen_keys = _condition_key_series(screen, family)
        kept_screen = screen[screen_keys.isin(selected_keys)].copy()
        layers_by_k: dict[int, list[int]] = {}
        for _protocol, layer_set, k in selected_keys:
            layers_by_k.setdefault(int(k), []).append(int(layer_set))
        multi_keys = {
            ("multi_layer", "+".join(str(layer) for layer in sorted(layers)), int(k))
            for k, layers in layers_by_k.items()
            if len(set(layers)) >= 2
        }
        expected_confirmation_keys = selected_keys | multi_keys
        confirmation_keys = _condition_key_series(confirmation, family)
        kept_confirmation = confirmation[
            confirmation_keys.isin(expected_confirmation_keys)
        ].copy()
    else:
        raise KeyError(f"Unknown causal-v2 family: {family}")

    observed_screen_keys = set(_condition_key_series(kept_screen, family).tolist())
    if observed_screen_keys != selected_keys:
        raise ValueError(
            "Selected screen conditions are incomplete: "
            f"expected={sorted(selected_keys)}, observed={sorted(observed_screen_keys)}"
        )
    observed_confirmation_keys = set(
        _condition_key_series(kept_confirmation, family).tolist()
    )
    if observed_confirmation_keys != expected_confirmation_keys:
        raise ValueError(
            "Held-out confirmation conditions are incomplete: "
            f"expected={sorted(expected_confirmation_keys)}, "
            f"observed={sorted(observed_confirmation_keys)}"
        )

    screen_seed_set = set(
        pd.to_numeric(kept_screen["seed"], errors="raise").astype(int)
    )
    confirmation_seed_set = set(
        pd.to_numeric(kept_confirmation["seed"], errors="raise").astype(int)
    )
    if screen_seed_set & confirmation_seed_set:
        raise ValueError("Screen and held-out confirmation seeds overlap")
    if strict_seed_sets:
        if screen_seed_set != set(design.screen_seeds):
            raise ValueError(
                f"Screen seed set changed: {sorted(screen_seed_set)} != "
                f"{list(design.screen_seeds)}"
            )
        if confirmation_seed_set != set(design.confirmation_seeds):
            raise ValueError(
                "Held-out confirmation seed set changed: "
                f"{sorted(confirmation_seed_set)} != {list(design.confirmation_seeds)}"
            )

    kept_screen["evidence_split"] = "screen"
    kept_confirmation["evidence_split"] = "held_out_confirmation"
    return pd.concat([kept_screen, kept_confirmation], ignore_index=True, sort=False)


def _base_design(
    *,
    family: str,
    phase: str,
    model_label: str,
    stimuli_path: Path,
    base_config_path: Path,
    causal_config_path: Path,
    profile: str,
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "realistic_niah_v4_causal_v2_run_design_v1",
        "family": family,
        "phase": phase,
        "profile": profile,
        "model_label": model_label,
        "answer_format": "numeric",
        "design_variant": "v4.4",
        "stimuli_sha256": _sha256_file(stimuli_path),
        "base_config_sha256": _sha256_file(base_config_path),
        "causal_config_sha256": _sha256_file(causal_config_path),
        "implementation_sha256": _implementation_fingerprint(repo_root),
        "behavior_metric": "strict_greedy_complete_numeric_generation",
    }


def _baseline_stage(
    args: argparse.Namespace, context: dict[str, Any]
) -> dict[str, Any]:
    design: CausalV2Design = context["design"]
    config: V4Config = context["config"]
    run_root: Path = context["run_root"]
    causal_root: Path = context["causal_root"]
    model_label: str = args.model
    base_behavior = (
        Path(args.base_behavior_labels).resolve()
        if args.base_behavior_labels
        else run_root
        / model_label
        / "numeric"
        / "behavior"
        / "capture"
        / "generation_labels.csv"
    )
    if not base_behavior.is_file():
        raise FileNotFoundError(base_behavior)
    stage_design = {
        **_base_design(
            family="baseline",
            phase="all",
            model_label=model_label,
            stimuli_path=context["stimuli_path"],
            base_config_path=context["base_config_path"],
            causal_config_path=context["causal_config_path"],
            profile=context["profile"],
            repo_root=context["repo_root"],
        ),
        "seeds": list(
            sorted(
                set(design.centroid_fit_seeds)
                | set(design.screen_seeds)
                | set(design.confirmation_seeds)
            )
        ),
        "generated_counts": [0],
        "valid_counts": list(design.valid_counts),
        "base_behavior_sha256": _sha256_file(base_behavior),
        "max_new_tokens": int(args.generation_max_new_tokens),
    }
    stage_root, design_hash = _stage_root(
        causal_root, family="baseline", phase="all", design=stage_design
    )
    merged_path = stage_root / "generation_labels.csv"
    if merged_path.exists() and not args.overwrite:
        merged = pd.read_csv(merged_path)
    else:
        model, tokenizer, _adapter = _load_model(
            model_label=model_label,
            config=config,
            cache_dir=args.cache_dir,
            device_map=args.device_map,
        )
        seeds = tuple(stage_design["seeds"])
        zero_rows = _select_rows(context["stimuli_path"], seeds=seeds, counts=(0,))
        zero_encodings = _render(
            zero_rows,
            tokenizer=tokenizer,
            model_label=model_label,
            config=config,
        )
        zero_outputs = capture_generation_labels(
            model,
            tokenizer,
            zero_encodings,
            output_dir=stage_root / "zero_capture",
            valid_counts=design.valid_counts,
            max_new_tokens=int(args.generation_max_new_tokens),
            overwrite=args.overwrite,
        )
        zero = pd.read_csv(zero_outputs["labels"])
        base = pd.read_csv(base_behavior)
        base = base[
            base["model_label"].astype(str).eq(model_label)
            & base["design_variant"].astype(str).eq("v4.4")
            & base["seed"].astype(int).isin(seeds)
            & base["gold_count"].astype(int).between(1, 10)
        ].copy()
        merged = pd.concat([zero, base], ignore_index=True, sort=False)
        expected = len(seeds) * len(design.valid_counts)
        if len(merged) != expected or merged["stimulus_id"].duplicated().any():
            raise RuntimeError(
                f"Merged causal-v2 baseline has {len(merged)} rows; expected {expected}"
            )
        _write_csv_atomic(merged, merged_path)
    _write_json_atomic(
        stage_root / "complete.json",
        {
            "status": "complete",
            "design_hash": design_hash,
            "rows": int(len(merged)),
            "counts": sorted(pd.to_numeric(merged["gold_count"]).astype(int).unique()),
        },
    )
    return {
        "stage_root": str(stage_root),
        "labels": str(merged_path),
        "rows": len(merged),
    }


def _prompt_alignment_stage(
    args: argparse.Namespace, context: dict[str, Any]
) -> dict[str, Any]:
    """Verify and record every formal donor-to-receiver full-span mapping.

    V4 was length-matched with the canonical Qwen tokenizer. This tokenizer-
    only preflight therefore runs for each evaluated model before any expensive
    causal generation. Equal-length spans retain positionwise identity; unequal
    spans use the registered monotonic endpoint-preserving nearest-neighbor map.
    The stage fails if any changed slot cannot support that frozen policy.
    """

    design: CausalV2Design = context["design"]
    config: V4Config = context["config"]
    model_spec = resolve_model_spec(args.model)
    evaluation_seeds = (
        design.screen_seeds[:1]
        if args.smoke
        else tuple((*design.screen_seeds, *design.confirmation_seeds))
    )
    pairs = design.directed_pairs[:1] if args.smoke else design.directed_pairs
    counts = tuple(sorted({value for pair in pairs for value in pair}))
    stage_design = {
        **_base_design(
            family="prompt_span_alignment",
            phase="preflight",
            model_label=args.model,
            stimuli_path=context["stimuli_path"],
            base_config_path=context["base_config_path"],
            causal_config_path=context["causal_config_path"],
            profile=context["profile"],
            repo_root=context["repo_root"],
        ),
        "evaluation_seeds": list(evaluation_seeds),
        "directed_count_pairs": [list(pair) for pair in pairs],
        "alignment_policy": design.prompt_full_span_alignment,
        "tokenizer_model_id": model_spec.model_id,
        "tokenizer_revision": model_spec.revision,
    }
    stage_root, design_hash = _stage_root(
        context["causal_root"],
        family="prompt_span_alignment",
        phase="preflight",
        design=stage_design,
    )
    tokenizer = load_registered_tokenizer(model_spec, cache_dir=args.cache_dir)
    rows = _select_rows(context["stimuli_path"], seeds=evaluation_seeds, counts=counts)
    encodings = _render(
        rows, tokenizer=tokenizer, model_label=args.model, config=config
    )
    alignment = causal_v2_prompt_span_alignment_table(
        encodings,
        count_pairs=pairs,
        evaluation_seeds=evaluation_seeds,
        alignment_policy=design.prompt_full_span_alignment,
    )
    alignment_path = stage_root / "prompt_full_span_alignment.csv"
    _write_csv_atomic(alignment, alignment_path)
    unsupported = alignment[~alignment["mapping_supported"].astype(bool)]
    exact = alignment["exact_model_token_alignment"].astype(bool)
    if unsupported.empty:
        _write_json_atomic(
            stage_root / "complete.json",
            {
                "status": "complete",
                "design_hash": design_hash,
                "rows": int(len(alignment)),
                "exact_rows": int(exact.sum()),
                "remapped_rows": int((~exact).sum()),
                "unsupported_rows": 0,
                "max_absolute_model_token_length_delta": int(
                    alignment["absolute_model_token_length_delta"].max()
                ),
                "alignment_policy": design.prompt_full_span_alignment,
            },
        )
    else:
        examples = unsupported.head(20).to_dict(orient="records")
        raise RuntimeError(
            f"{args.model} cannot execute the registered full-span mapping: "
            f"{len(unsupported)} changed slots are unsupported; examples={examples}"
        )
    return {
        "stage_root": str(stage_root),
        "alignment": str(alignment_path),
        "rows": int(len(alignment)),
        "exact_rows": int(exact.sum()),
        "remapped_rows": int((~exact).sum()),
        "unsupported_rows": 0,
    }


def _head_ranking_stage(
    args: argparse.Namespace, context: dict[str, Any]
) -> dict[str, Any]:
    if not args.attention_source:
        raise ValueError("head-rankings requires --attention-source")
    design: CausalV2Design = context["design"]
    attention_source = (
        context["run_root"]
        if str(args.attention_source).strip().upper() == "AUTO"
        else Path(args.attention_source)
    )
    detail, source_manifest = _load_attention_phenotype_source(
        attention_source,
        model_label=args.model,
        expected_seeds=design.centroid_fit_seeds,
        expected_counts=tuple(range(1, 11)),
    )
    scores = head_phenotype_scores(detail)
    if scores.empty:
        raise ValueError(f"Attention detail has no rows for {args.model}")
    stage_design = {
        **_base_design(
            family="head_rankings",
            phase="discovery",
            model_label=args.model,
            stimuli_path=context["stimuli_path"],
            base_config_path=context["base_config_path"],
            causal_config_path=context["causal_config_path"],
            profile=context["profile"],
            repo_root=context["repo_root"],
        ),
        "attention_source_aggregate_sha256": source_manifest["aggregate_sha256"],
        "attention_source_index_sha256": source_manifest["source_index_sha256"],
        "attention_source_file_count": source_manifest["file_count"],
        "attention_source_prompt_count": source_manifest["prompt_count"],
        "selection_split": "discovery",
        "top_n": 32,
        "broad_metric": "full_span_mass_times_normalized_effective_needle_count",
        "first_locator_metric": "first_span_mass_minus_mean_other_span_mass",
    }
    output, design_hash = _stage_root(
        context["causal_root"],
        family="head_rankings",
        phase="discovery",
        design=stage_design,
    )
    _write_json_atomic(output / "attention_source_manifest.json", source_manifest)
    paths = write_head_phenotype_registry(scores, output_dir=output, top_n=32)
    _write_json_atomic(
        output / "complete.json",
        {
            "status": "complete",
            "design_hash": design_hash,
            "heads_scored": int(len(scores)),
        },
    )
    return {
        "stage_root": str(output),
        **{key: str(value) for key, value in paths.items()},
    }


def _baseline_labels_path(args: argparse.Namespace, context: dict[str, Any]) -> Path:
    if args.baseline_labels:
        path = Path(args.baseline_labels).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(
        (context["causal_root"] / "baseline").glob("all_*/generation_labels.csv")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one causal-v2 baseline; pass --baseline-labels explicitly"
        )
    return candidates[0]


def _ablation_confirmation_plan(
    path: Path,
    *,
    model_label: str,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    legacy_schema = (
        "realistic_niah_v4_causal_v2_ablation_confirmation_selection_v1"
    )
    extrapolation_schema = (
        "realistic_niah_v4_4_ablation_seed_extrapolation_selection_v1"
    )
    schema = payload.get("schema_version")
    if schema not in {legacy_schema, extrapolation_schema}:
        raise ValueError("Unexpected ablation-confirmation selection schema")
    expected_status = (
        "frozen_before_confirmation"
        if schema == legacy_schema
        else "frozen_before_seed_extrapolation"
    )
    if payload.get("selection_status") != expected_status:
        raise ValueError("Ablation-confirmation selection was not frozen in advance")
    rule = payload.get("selection_rule")
    if not isinstance(rule, dict) or not bool(
        rule.get("applied_separately_per_model")
    ):
        raise ValueError("Ablation top-k must be selected separately per model")
    source = payload.get("discovery_source")
    if not isinstance(source, dict):
        raise ValueError("Ablation-confirmation selection lacks discovery provenance")
    source_table = repo_root / str(source.get("table", ""))
    if not source_table.is_file():
        raise FileNotFoundError(source_table)
    expected_sha = str(source.get("table_sha256", "")).lower()
    if _sha256_normalized_text(source_table).lower() != expected_sha:
        raise ValueError("Discovery ablation table hash no longer matches selection")
    confirmation = payload.get("confirmation_design")
    if not isinstance(confirmation, dict):
        raise ValueError("Selection lacks confirmation_design")
    seeds = tuple(int(value) for value in confirmation.get("seeds", ()))
    counts = tuple(int(value) for value in confirmation.get("counts", ()))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Ablation confirmation seeds must be unique")
    if schema == legacy_schema:
        discovery_seeds = {
            int(value) for value in source.get("screen_seeds", ())
        }
        if len(seeds) != 10:
            raise ValueError("Formal ablation confirmation requires ten new seeds")
        if discovery_seeds.intersection(seeds):
            raise ValueError("Ablation confirmation seeds overlap discovery seeds")
        if counts != (7, 8, 9, 10):
            raise ValueError("Ablation confirmation must retain counts 7,8,9,10")
    else:
        if len(seeds) != 20:
            raise ValueError("Seed extrapolation requires exactly twenty new seeds")
        prior_seed_end = int(source.get("prior_seed_end_inclusive", -1))
        if not seeds or min(seeds) <= prior_seed_end:
            raise ValueError("Seed extrapolation overlaps a previously used seed range")
        if seeds != tuple(range(seeds[0], seeds[0] + len(seeds))):
            raise ValueError("Seed extrapolation must use one contiguous frozen suffix")
        if counts != (1, 2, 3, 4, 5):
            raise ValueError("Seed extrapolation must retain counts 1,2,3,4,5")
    if int(confirmation.get("random_replicates", 0)) != 3:
        raise ValueError("Ablation confirmation requires three random replicates")
    if confirmation.get("random_baseline") != (
        "all_heads_in_matched_layers_without_replacement_overlap_allowed"
    ):
        raise ValueError("Ablation confirmation random baseline changed")
    if confirmation.get("scope") != "answer_query":
        raise ValueError("Ablation confirmation scope must remain answer_query")
    models = payload.get("models")
    if not isinstance(models, dict) or model_label not in models:
        raise ValueError(f"Selection has no frozen plan for {model_label}")
    plan = models[model_label]
    if not isinstance(plan, dict):
        raise ValueError(f"Invalid frozen plan for {model_label}")
    if plan.get("head_bank") not in {"broad_aggregation", "first_locator"}:
        raise ValueError(
            "Frozen confirmation bank must be broad_aggregation or first_locator"
        )
    if schema == legacy_schema:
        top_ns = (int(plan.get("top_n", 0)),)
    else:
        top_ns = tuple(int(value) for value in plan.get("top_ns", ()))
        if not top_ns or len(set(top_ns)) != len(top_ns):
            raise ValueError(
                "Seed extrapolation requires one or more distinct frozen top_n values"
            )
        if top_ns != tuple(sorted(top_ns)):
            raise ValueError("Seed extrapolation top_n values must be increasing")
    if any(not 1 <= top_n <= 32 for top_n in top_ns):
        raise ValueError("Frozen confirmation top_n must lie in [1,32]")
    return payload, {
        **plan,
        "top_ns": top_ns,
        "seeds": seeds,
        "counts": counts,
        "random_replicates": int(confirmation["random_replicates"]),
        "random_baseline": str(confirmation["random_baseline"]),
        "selection_status": expected_status,
        "evidence_split": (
            "independent_ablation_confirmation"
            if schema == legacy_schema
            else "independent_seed_extrapolation"
        ),
        "emit_dual_population": (
            schema == extrapolation_schema
            and str(plan.get("head_bank")) == "broad_aggregation"
        ),
    }


def _ablation_stage(
    args: argparse.Namespace, context: dict[str, Any]
) -> dict[str, Any]:
    design: CausalV2Design = context["design"]
    config: V4Config = context["config"]
    ranking_path = Path(args.head_rankings).resolve() if args.head_rankings else None
    if ranking_path is None or not ranking_path.is_file():
        raise ValueError("ablation requires --head-rankings")
    rankings = load_head_phenotype_registry(ranking_path)
    phase = args.phase
    selection_path: Path | None = None
    selection_sha: str | None = None
    plan: dict[str, Any] | None = None
    baseline_path: Path | None = None
    baseline_labels: dict[str, dict[str, Any]] | None = None
    if phase == "screen":
        baseline_path = _baseline_labels_path(args, context)
        baseline_labels = load_generation_labels(baseline_path)
        seeds = design.screen_seeds[:1] if args.smoke else design.screen_seeds
        counts = design.ablation_counts[:1] if args.smoke else design.ablation_counts
        top_ns = tuple(range(1, 3)) if args.smoke else design.ablation_top_ns
        random_replicates = int(design.ablation_random_replicates)
        random_baseline = str(design.ablation_random_baseline)
    elif phase == "confirmation":
        if not args.selection_json:
            raise ValueError("ablation confirmation requires --selection-json")
        selection_path = Path(args.selection_json).resolve()
        if not selection_path.is_file():
            raise FileNotFoundError(selection_path)
        _selection, plan = _ablation_confirmation_plan(
            selection_path,
            model_label=args.model,
            repo_root=context["repo_root"],
        )
        if tuple(int(value) for value in config.seeds) != tuple(plan["seeds"]):
            raise ValueError(
                "Supplemental base config seeds do not match frozen confirmation"
            )
        seeds = tuple(plan["seeds"][:1]) if args.smoke else tuple(plan["seeds"])
        counts = tuple(plan["counts"][:1]) if args.smoke else tuple(plan["counts"])
        top_ns = tuple(int(value) for value in plan["top_ns"])
        random_replicates = int(plan["random_replicates"])
        random_baseline = str(plan["random_baseline"])
        rankings = {str(plan["head_bank"]): rankings[str(plan["head_bank"])]}
        selection_sha = _sha256_file(selection_path)
    else:
        raise ValueError(f"Unknown ablation phase: {phase}")
    # Discovery always uses all 1..32 and both banks. Confirmation evaluates
    # only the frozen model-specific dose or doses without any reselection.
    m10_control_json = os.environ.get("V44_M10_CONTROL_JSON")
    m10_control_path = Path(m10_control_json).resolve() if m10_control_json else None
    if m10_control_path is not None and not m10_control_path.is_file():
        raise FileNotFoundError(m10_control_path)
    stage_design = {
        **_base_design(
            family="answer_query_head_ablation",
            phase=phase,
            model_label=args.model,
            stimuli_path=context["stimuli_path"],
            base_config_path=context["base_config_path"],
            causal_config_path=context["causal_config_path"],
            profile=context["profile"],
            repo_root=context["repo_root"],
        ),
        "seeds": list(seeds),
        "counts": list(counts),
        "head_banks": sorted(rankings),
        "top_ns": list(top_ns),
        "random_replicates": random_replicates,
        "scope": design.ablation_scope,
        "random_baseline": random_baseline,
        "rankings_sha256": _sha256_file(ranking_path),
    }
    if m10_control_path is not None:
        stage_design.update(
            {
                "control_family": "layer_and_discovery_M10_matched",
                "m10_control_sha256": _sha256_file(m10_control_path),
            }
        )
    if phase == "screen" and baseline_path is not None:
        stage_design["baseline_labels_sha256"] = _sha256_file(baseline_path)
    elif phase == "confirmation" and plan is not None:
        stage_design.update(
            {
                "baseline_source": (
                    "fresh_clean_generation_on_independent_confirmation_stimuli"
                ),
                "selection_sha256": selection_sha,
                "frozen_model_specific_top_ns": list(top_ns),
                "selection_status": str(plan["selection_status"]),
            }
        )
        if len(top_ns) == 1:
            stage_design["frozen_model_specific_top_n"] = int(top_ns[0])
    stage_root, design_hash = _stage_root(
        context["causal_root"],
        family="answer_query_head_ablation",
        phase=phase,
        design=stage_design,
    )
    model, tokenizer, adapter = _load_model(
        model_label=args.model,
        config=config,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
    )
    rows = _select_rows(context["stimuli_path"], seeds=seeds, counts=counts)
    encodings = _render(
        rows, tokenizer=tokenizer, model_label=args.model, config=config
    )
    if phase == "confirmation":
        clean_outputs = capture_generation_labels(
            model,
            tokenizer,
            encodings,
            output_dir=stage_root / "clean_baseline",
            valid_counts=design.valid_counts,
            max_new_tokens=int(args.generation_max_new_tokens),
            overwrite=args.overwrite,
        )
        baseline_path = clean_outputs["labels"]
        baseline_labels = load_generation_labels(baseline_path)
    if baseline_labels is None or baseline_path is None:
        raise RuntimeError("Head ablation has no clean baseline labels")
    capture_root = stage_root / "capture"
    index_rows: list[dict[str, Any]] = []
    for encoding in encodings:
        relative = Path("shards") / f"seed{encoding.seed}" / f"N{encoding.count}.csv.gz"
        shard = capture_root / relative
        if shard.exists() and not args.overwrite:
            frame = pd.read_csv(shard, compression="gzip")
        else:
            runner = (
                run_generation_head_ablation_v2_m10
                if m10_control_path is not None
                else run_generation_head_ablation_v2
            )
            runner_kwargs = {}
            if m10_control_path is not None:
                runner_kwargs["control_json"] = m10_control_path
            frame = runner(
                model,
                tokenizer,
                adapter,
                [encoding],
                baseline_labels=baseline_labels,
                rankings=rankings,
                top_ns=top_ns,
                random_replicates=random_replicates,
                require_full_sweep=phase == "screen" and not args.smoke,
                valid_counts=design.valid_counts,
                max_new_tokens=int(args.generation_max_new_tokens),
                **runner_kwargs,
            )
            frame["behavior_metric"] = stage_design["behavior_metric"]
            if phase == "confirmation":
                frame["evidence_split"] = str(plan["evidence_split"])
            _write_csv_gzip_atomic(frame, shard)
        index_rows.append(
            {
                "seed": int(encoding.seed),
                "count": int(encoding.count),
                "rows": int(len(frame)),
                "shard_path": relative.as_posix(),
                "sha256": _sha256_file(shard),
            }
        )
    _write_jsonl_atomic(capture_root / "capture_index.jsonl", index_rows)
    detail = _load_shards(index_rows, capture_root)
    detail_path = stage_root / "detail.csv.gz"
    _write_csv_gzip_atomic(detail, detail_path)
    summary_path = stage_root / "summary.csv"
    _write_csv_atomic(summarize_generation_head_ablation_v2(detail), summary_path)
    comparison_path = stage_root / "ranked_vs_layer_matched_random.csv"
    _write_csv_atomic(
        compare_head_ablation_v2_to_random(
            detail, bootstrap_repetitions=design.bootstrap_repetitions
        ),
        comparison_path,
    )
    seed_effects_path: Path | None = None
    statistics_path: Path | None = None
    clean_correct_path: Path | None = None
    dual_population_path: Path | None = None
    if phase == "confirmation":
        analysis_root = stage_root / "analysis"
        seed_effects = head_ablation_confirmation_seed_effects(detail)
        statistics = head_ablation_confirmation_statistics(
            seed_effects,
            bootstrap_repetitions=design.bootstrap_repetitions,
        )
        seed_effects_path = analysis_root / "head_ablation_seed_effects.csv"
        statistics_path = (
            analysis_root / "head_ablation_confirmation_statistics.csv"
        )
        _write_csv_atomic(seed_effects, seed_effects_path)
        _write_csv_atomic(statistics, statistics_path)
        if plan is not None and bool(plan.get("emit_dual_population")):
            clean_correct_detail = clean_correct_ablation_rows(detail)
            clean_correct_path = (
                analysis_root / "detail.clean_correct.seed_extrapolation.csv.gz"
            )
            _write_csv_gzip_atomic(clean_correct_detail, clean_correct_path)
            all_population = summarize_ablation_n_diagnostics(
                detail,
                population="all_examples_signed",
                bootstrap_repetitions=design.bootstrap_repetitions,
            )
            correct_population = summarize_ablation_n_diagnostics(
                detail,
                population="clean_correct_only",
                bootstrap_repetitions=design.bootstrap_repetitions,
            )
            for population_frame in (all_population, correct_population):
                population_frame["selection_status"] = str(
                    plan["selection_status"]
                )
                population_frame["evidence_split"] = str(plan["evidence_split"])
            dual_population_path = (
                analysis_root / "dual_population_seed_extrapolation_summary.csv"
            )
            _write_csv_atomic(
                pd.concat(
                    [all_population, correct_population],
                    ignore_index=True,
                    sort=False,
                ),
                dual_population_path,
            )
    completion_payload: dict[str, Any] = {
        "status": "complete",
        "design_hash": design_hash,
        "rows": len(detail),
    }
    if phase == "confirmation":
        completion_payload.update(
            {
                "phase": phase,
                "baseline_rows": len(baseline_labels),
                "seeds": list(seeds),
                "counts": list(counts),
                "head_banks": sorted(rankings),
                "top_ns": list(top_ns),
                "selection_sha256": selection_sha,
                "baseline_labels_sha256": _sha256_file(baseline_path),
            }
        )
        if clean_correct_path is not None:
            completion_payload["clean_correct_rows"] = int(len(clean_correct_detail))
    _write_json_atomic(
        stage_root / "complete.json",
        completion_payload,
    )
    result = {
        "stage_root": str(stage_root),
        "detail": str(detail_path),
        "summary": str(summary_path),
        "comparison": str(comparison_path),
        "rows": len(detail),
    }
    if seed_effects_path is not None and statistics_path is not None:
        result["seed_effects"] = str(seed_effects_path)
        result["confirmation_statistics"] = str(statistics_path)
    if clean_correct_path is not None and dual_population_path is not None:
        result["clean_correct_detail"] = str(clean_correct_path)
        result["dual_population_summary"] = str(dual_population_path)
    return result


def _patching_stage(
    args: argparse.Namespace, context: dict[str, Any]
) -> dict[str, Any]:
    design: CausalV2Design = context["design"]
    config: V4Config = context["config"]
    phase = args.phase
    seeds = _phase_seeds(design, phase)
    evaluation_seeds = seeds[:1] if args.smoke else seeds
    source_seeds = seeds[:2] if args.smoke else seeds
    pairs = design.directed_pairs[:1] if args.smoke else design.directed_pairs
    baseline_path = _baseline_labels_path(args, context)
    baseline_labels = load_generation_labels(baseline_path)
    if args.stage == "prompt-patching":
        family = "prompt_patching"
        sites = ("toggled_needle_end", "toggled_needle_span")
        controls = ("donor_transport", "self_patch")
    else:
        family = "answer_patching"
        sites = ("answer_query",)
        controls = ("donor_transport", "self_patch", "same_count_seed")
    condition_filter: set[tuple[str, str, int, int]] | None = None
    selection_sha: str | None = None
    if phase == "confirmation":
        if not args.selection_json:
            raise ValueError("patching confirmation requires --selection-json")
        selection_path = Path(args.selection_json).resolve()
        condition_filter, _payload = _selection_filter(
            selection_path, family=family, model_label=args.model
        )
        selection_sha = _sha256_file(selection_path)
        pairs = tuple(
            pair
            for pair in pairs
            if abs(int(pair[1]) - int(pair[0])) in {key[3] for key in condition_filter}
        )
    stage_design = {
        **_base_design(
            family=family,
            phase=phase,
            model_label=args.model,
            stimuli_path=context["stimuli_path"],
            base_config_path=context["base_config_path"],
            causal_config_path=context["causal_config_path"],
            profile=context["profile"],
            repo_root=context["repo_root"],
        ),
        "evaluation_seeds": list(evaluation_seeds),
        "source_seed_pool": list(source_seeds),
        "directed_count_pairs": [list(pair) for pair in pairs],
        "sites": list(sites),
        "protocols": list(design.patch_protocols),
        "controls": list(controls),
        "layers": "all_decoder_layers",
        "prompt_full_span_alignment": design.prompt_full_span_alignment,
        "answer_multi_layer_protocol": design.answer_multi_layer_protocol,
        "control_compute_reuse": {
            "self_patch": "reuse_baseline_after_executed_identity_preflight",
            "answer_same_count_seed": ("cache_by_receiver_source_site_protocol_layer"),
            "logical_rows_preserved": True,
        },
        "transport_metric_version": design.transport_metric_version,
        "invalid_policy": design.invalid_policy,
        "selection_json_sha256": selection_sha,
        "baseline_labels_sha256": _sha256_file(baseline_path),
    }
    stage_root, design_hash = _stage_root(
        context["causal_root"], family=family, phase=phase, design=stage_design
    )
    model, tokenizer, adapter = _load_model(
        model_label=args.model,
        config=config,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
    )
    layers = (
        (0, adapter.num_layers - 1) if args.smoke else tuple(range(adapter.num_layers))
    )
    identity_layers = tuple(
        sorted({0, adapter.num_layers // 2, adapter.num_layers - 1})
    )
    identity_pair = max(
        pairs,
        key=lambda pair: (
            abs(int(pair[1]) - int(pair[0])),
            -int(pair[0]),
            int(pair[1]),
        ),
    )
    ordered_pairs = (identity_pair,) + tuple(
        pair for pair in pairs if pair != identity_pair
    )
    identity_execution_filter = {
        (
            int(evaluation_seeds[0]),
            int(identity_pair[0]),
            int(identity_pair[1]),
            str(site),
            str(protocol),
            int(layer),
        )
        for site in sites
        for protocol in design.patch_protocols
        for layer in identity_layers
    }
    shared_completion_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    selected_counts = tuple(sorted({value for pair in pairs for value in pair}))
    rows = _select_rows(
        context["stimuli_path"], seeds=source_seeds, counts=selected_counts
    )
    encodings = _render(
        rows, tokenizer=tokenizer, model_label=args.model, config=config
    )
    if family == "prompt_patching":
        alignment = causal_v2_prompt_span_alignment_table(
            encodings,
            count_pairs=pairs,
            evaluation_seeds=evaluation_seeds,
            alignment_policy=design.prompt_full_span_alignment,
        )
        alignment_path = stage_root / "prompt_full_span_alignment.csv"
        _write_csv_atomic(alignment, alignment_path)
        unsupported = alignment[~alignment["mapping_supported"].astype(bool)]
        if not args.smoke and not unsupported.empty:
            examples = unsupported.head(20).to_dict(orient="records")
            raise RuntimeError(
                "Formal full-span prompt patching found unsupported token-span "
                f"mappings; unsupported={len(unsupported)}, examples={examples}"
            )
    capture_root = stage_root / "capture"
    index_rows: list[dict[str, Any]] = []
    for seed in evaluation_seeds:
        for receiver_count, donor_count in ordered_pairs:
            relative = (
                Path("shards")
                / f"seed{seed}"
                / f"N{receiver_count}_to_N{donor_count}.csv.gz"
            )
            shard = capture_root / relative
            if shard.exists() and not args.overwrite:
                frame = pd.read_csv(shard, compression="gzip")
            else:
                frame = run_generation_residual_patching_v2(
                    model,
                    tokenizer,
                    adapter,
                    encodings,
                    baseline_labels=baseline_labels,
                    count_pairs=((receiver_count, donor_count),),
                    start_layers=layers,
                    sites=sites,
                    protocols=design.patch_protocols,
                    control_conditions=controls,
                    condition_filter=condition_filter,
                    evaluation_seeds=(seed,),
                    shared_completion_cache=shared_completion_cache,
                    identity_execution_filter=identity_execution_filter,
                    valid_counts=design.valid_counts,
                    max_new_tokens=int(args.generation_max_new_tokens),
                    full_span_alignment_policy=(design.prompt_full_span_alignment),
                )
                frame["phase"] = phase
                frame["behavior_metric"] = stage_design["behavior_metric"]
                _write_csv_gzip_atomic(frame, shard)
            index_rows.append(
                {
                    "seed": int(seed),
                    "receiver_count": int(receiver_count),
                    "donor_count": int(donor_count),
                    "k": abs(int(donor_count) - int(receiver_count)),
                    "rows": int(len(frame)),
                    "successful_rows": int(frame["status"].astype(str).eq("ok").sum()),
                    "skipped_rows": int(frame["status"].astype(str).ne("ok").sum()),
                    "shard_path": relative.as_posix(),
                    "sha256": _sha256_file(shard),
                }
            )
    _write_jsonl_atomic(capture_root / "capture_index.jsonl", index_rows)
    detail = _load_shards(index_rows, capture_root)
    skipped = detail[detail["status"].astype(str).ne("ok")]
    if not args.smoke and not skipped.empty:
        reasons = skipped["skip_reason"].astype(str).value_counts().to_dict()
        raise RuntimeError(
            "Formal causal-v2 patching encountered non-executable full-span "
            f"conditions; refusing a partial result: {reasons}"
        )
    detail_path = stage_root / "detail.csv.gz"
    _write_csv_gzip_atomic(detail, detail_path)
    summary_path = stage_root / "summary.csv"
    _write_csv_atomic(summarize_generation_residual_patching_v2(detail), summary_path)
    compute_reuse_path = stage_root / "compute_reuse_summary.csv"
    compute_reuse = (
        detail.groupby(
            ["condition", "generation_executed", "generation_reuse_mode"],
            as_index=False,
            dropna=False,
        )
        .agg(logical_rows=("stimulus_id", "size"))
        .sort_values(["condition", "generation_executed", "generation_reuse_mode"])
    )
    _write_csv_atomic(compute_reuse, compute_reuse_path)
    generation_executed = (
        detail["generation_executed"].astype(str).str.lower().eq("true")
    )
    _write_json_atomic(
        stage_root / "complete.json",
        {
            "status": "complete",
            "design_hash": design_hash,
            "rows": len(detail),
            "executed_generation_rows": int(generation_executed.sum()),
            "reused_logical_rows": int((~generation_executed).sum()),
            "successful_rows": int(detail["status"].astype(str).eq("ok").sum()),
            "skipped_rows": int(detail["status"].astype(str).ne("ok").sum()),
        },
    )
    return {
        "stage_root": str(stage_root),
        "detail": str(detail_path),
        "summary": str(summary_path),
        "compute_reuse_summary": str(compute_reuse_path),
        "rows": len(detail),
        "executed_generation_rows": int(generation_executed.sum()),
    }


def _steering_centroids_stage(
    args: argparse.Namespace, context: dict[str, Any]
) -> dict[str, Any]:
    design: CausalV2Design = context["design"]
    config: V4Config = context["config"]
    seeds = design.centroid_fit_seeds[:1] if args.smoke else design.centroid_fit_seeds
    counts = design.valid_counts[:2] if args.smoke else design.valid_counts
    baseline_path = _baseline_labels_path(args, context)
    stage_design = {
        **_base_design(
            family="steering_centroids",
            phase="fit",
            model_label=args.model,
            stimuli_path=context["stimuli_path"],
            base_config_path=context["base_config_path"],
            causal_config_path=context["causal_config_path"],
            profile=context["profile"],
            repo_root=context["repo_root"],
        ),
        "fit_seeds": list(seeds),
        "counts": list(counts),
        "layers": "all_decoder_layers",
        "save_dtype": config.hidden_save_dtype,
        "baseline_labels_sha256": _sha256_file(baseline_path),
    }
    stage_root, design_hash = _stage_root(
        context["causal_root"],
        family="steering_centroids",
        phase="fit",
        design=stage_design,
    )
    model, tokenizer, adapter = _load_model(
        model_label=args.model,
        config=config,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
    )
    layers = (
        (0, adapter.num_layers - 1) if args.smoke else tuple(range(adapter.num_layers))
    )
    rows = _select_rows(context["stimuli_path"], seeds=seeds, counts=counts)
    encodings = _render(
        rows, tokenizer=tokenizer, model_label=args.model, config=config
    )
    capture_root = stage_root / "capture"
    index_rows: list[dict[str, Any]] = []
    for encoding in encodings:
        relative = Path("shards") / f"seed{encoding.seed}" / f"N{encoding.count}.npz"
        metadata = capture_query_residual_shard(
            model,
            adapter,
            encoding,
            layers=layers,
            path=capture_root / relative,
            save_dtype=config.hidden_save_dtype,
            overwrite=args.overwrite,
        )
        index_rows.append(
            {
                **metadata,
                "shard_path": relative.as_posix(),
                "sha256": _sha256_file(capture_root / relative),
            }
        )
    _write_jsonl_atomic(capture_root / "capture_index.jsonl", index_rows)
    centroids = fit_count_centroids(
        index_rows,
        capture_root=capture_root,
        variants=("v4.4",),
        layers=layers,
        counts=counts,
        discovery_seeds=seeds,
    )
    centroid_path = save_centroid_bundle(centroids, stage_root / "centroids.npz")
    geometry, adjacent = centroid_geometry_tables(centroids)
    _write_csv_atomic(geometry, stage_root / "centroid_geometry_summary.csv")
    _write_csv_atomic(adjacent, stage_root / "centroid_adjacent_steps.csv")
    _write_json_atomic(
        stage_root / "complete.json",
        {"status": "complete", "design_hash": design_hash, "captures": len(index_rows)},
    )
    return {
        "stage_root": str(stage_root),
        "centroids": str(centroid_path),
        "captures": len(index_rows),
    }


def _steering_summary(detail: pd.DataFrame) -> pd.DataFrame:
    groups = [
        "model_label",
        "steering_protocol",
        "layer_set",
        "k",
        "target_direction",
        "condition",
        "baseline_outcome",
    ]
    return (
        detail.groupby(groups, as_index=False, dropna=False)
        .agg(
            examples=("receiver_stimulus_id", "size"),
            seeds=("seed", "nunique"),
            patched_valid_rate=("patched_format_valid", "mean"),
            prediction_changed_rate=("prediction_changed", "mean"),
            mean_normalized_transport=("normalized_transport", "mean"),
            mean_strict_normalized_transport=("strict_normalized_transport", "mean"),
            mean_target_conformity=("target_conformity", "mean"),
            target_hit_rate=("strict_target_hit", "mean"),
        )
        .sort_values(groups)
        .reset_index(drop=True)
    )


def _steering_stage(
    args: argparse.Namespace, context: dict[str, Any]
) -> dict[str, Any]:
    design: CausalV2Design = context["design"]
    config: V4Config = context["config"]
    if not args.centroids:
        raise ValueError("steering requires --centroids")
    centroid_path = Path(args.centroids).resolve()
    centroids = load_centroid_bundle(centroid_path)
    phase = args.phase
    seeds = _phase_seeds(design, phase)
    evaluation_seeds = seeds[:1] if args.smoke else seeds
    pairs = design.directed_pairs[:1] if args.smoke else design.directed_pairs
    baseline_path = _baseline_labels_path(args, context)
    baseline_labels = load_generation_labels(baseline_path)
    selection_sha: str | None = None
    selection_path: Path | None = None
    if phase == "confirmation":
        if not args.selection_json:
            raise ValueError("steering confirmation requires --selection-json")
        selection_path = Path(args.selection_json).resolve()
        selection_sha = _sha256_file(selection_path)
        stable_k = {
            int(item["k"])
            for item in json.loads(selection_path.read_text(encoding="utf-8")).get(
                "selected", []
            )
            if str(item["model_label"]) == args.model
        }
        pairs = tuple(pair for pair in pairs if abs(pair[1] - pair[0]) in stable_k)
        if not pairs:
            raise ValueError(f"No stable steering k values for {args.model}")
    stage_design = {
        **_base_design(
            family="steering",
            phase=phase,
            model_label=args.model,
            stimuli_path=context["stimuli_path"],
            base_config_path=context["base_config_path"],
            causal_config_path=context["causal_config_path"],
            profile=context["profile"],
            repo_root=context["repo_root"],
        ),
        "seeds": list(evaluation_seeds),
        "directed_count_pairs": [list(pair) for pair in pairs],
        "single_layer_sweep": phase == "screen",
        "confirmation_includes_frozen_multi_layer_plan": phase == "confirmation",
        "alpha": design.steering_alpha,
        "random_replicates": design.steering_random_replicates,
        "transport_metric_version": design.transport_metric_version,
        "invalid_policy": design.invalid_policy,
        "centroids_sha256": _sha256_file(centroid_path),
        "selection_json_sha256": selection_sha,
        "baseline_labels_sha256": _sha256_file(baseline_path),
    }
    stage_root, design_hash = _stage_root(
        context["causal_root"], family="steering", phase=phase, design=stage_design
    )
    model, tokenizer, adapter = _load_model(
        model_label=args.model,
        config=config,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
    )
    layers = (
        (0, adapter.num_layers - 1) if args.smoke else tuple(range(adapter.num_layers))
    )
    selected_counts = tuple(sorted({value for pair in pairs for value in pair}))
    rows = _select_rows(
        context["stimuli_path"], seeds=evaluation_seeds, counts=selected_counts
    )
    encodings = _render(
        rows, tokenizer=tokenizer, model_label=args.model, config=config
    )
    by_seed: dict[int, list[PromptEncoding]] = {}
    for encoding in encodings:
        by_seed.setdefault(int(encoding.seed), []).append(encoding)
    capture_root = stage_root / "capture"
    index_rows: list[dict[str, Any]] = []
    for seed in evaluation_seeds:
        for receiver_count, target_count in pairs:
            k = abs(target_count - receiver_count)
            if phase == "screen":
                plans = tuple(
                    LayerSetSteeringPlan(layers=(layer,), alpha=design.steering_alpha)
                    for layer in layers
                )
            else:
                assert selection_path is not None
                plans = _steering_plans_from_selection(
                    selection_path, model_label=args.model, k=k
                )
                if not plans:
                    continue
            relative = (
                Path("shards")
                / f"seed{seed}"
                / f"N{receiver_count}_to_N{target_count}.csv.gz"
            )
            shard = capture_root / relative
            if shard.exists() and not args.overwrite:
                frame = pd.read_csv(shard, compression="gzip")
            else:
                frame = run_generation_layer_set_centroid_delta(
                    model,
                    tokenizer,
                    adapter,
                    by_seed[int(seed)],
                    baseline_labels=baseline_labels,
                    centroids=centroids,
                    count_pairs=((receiver_count, target_count),),
                    plans=plans,
                    random_replicates=design.steering_random_replicates,
                    valid_counts=design.valid_counts,
                    max_new_tokens=int(args.generation_max_new_tokens),
                )
                frame["phase"] = phase
                frame["behavior_metric"] = stage_design["behavior_metric"]
                _write_csv_gzip_atomic(frame, shard)
            index_rows.append(
                {
                    "seed": int(seed),
                    "receiver_count": int(receiver_count),
                    "target_count": int(target_count),
                    "k": int(k),
                    "plans": len(plans),
                    "rows": len(frame),
                    "shard_path": relative.as_posix(),
                    "sha256": _sha256_file(shard),
                }
            )
    _write_jsonl_atomic(capture_root / "capture_index.jsonl", index_rows)
    detail = _load_shards(index_rows, capture_root)
    detail_path = stage_root / "detail.csv.gz"
    _write_csv_gzip_atomic(detail, detail_path)
    summary_path = stage_root / "summary.csv"
    _write_csv_atomic(_steering_summary(detail), summary_path)
    _write_json_atomic(
        stage_root / "complete.json",
        {"status": "complete", "design_hash": design_hash, "rows": len(detail)},
    )
    return {
        "stage_root": str(stage_root),
        "detail": str(detail_path),
        "summary": str(summary_path),
        "rows": len(detail),
    }


def _selection_stage(
    args: argparse.Namespace, context: dict[str, Any]
) -> dict[str, Any]:
    if not args.detail or not args.family:
        raise ValueError("select requires --detail and --family")
    detail_path = Path(args.detail).resolve()
    detail = pd.read_csv(detail_path, compression="infer")
    scores, manifest = stable_layer_k_conditions(
        detail,
        family=args.family,
        design=context["design"],
    )
    manifest.update(
        {
            "model_label": args.model,
            "screen_detail_sha256": _sha256_file(detail_path),
        }
    )
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else detail_path.parent / "selection"
    )
    output.mkdir(parents=True, exist_ok=True)
    scores_path = output / f"{args.family}_stability_scores.csv"
    manifest_path = output / f"{args.family}_selection.json"
    _write_csv_atomic(scores, scores_path)
    _write_json_atomic(manifest_path, manifest)
    return {
        "scores": str(scores_path),
        "selection": str(manifest_path),
        "selected_conditions": int(manifest["selected_condition_count"]),
    }


def _confirmation_stats_stage(
    args: argparse.Namespace, context: dict[str, Any]
) -> dict[str, Any]:
    if (
        not args.screen_detail
        or not args.confirmation_detail
        or not args.family
        or not args.selection_json
    ):
        raise ValueError(
            "confirmation-stats requires --screen-detail, --confirmation-detail, "
            "--selection-json, and --family"
        )
    screen_path = Path(args.screen_detail).resolve()
    confirmation_path = Path(args.confirmation_detail).resolve()
    selection_path = Path(args.selection_json).resolve()
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    detail = _selected_confirmation_detail(
        pd.read_csv(screen_path, compression="infer"),
        pd.read_csv(confirmation_path, compression="infer"),
        family=args.family,
        selection_payload=selection_payload,
        model_label=args.model,
        design=context["design"],
        strict_seed_sets=not args.smoke,
    )
    statistics = confirmation_statistics(
        detail,
        family=args.family,
        bootstrap_repetitions=context["design"].bootstrap_repetitions,
    )
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else confirmation_path.parent / "analysis"
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{args.family}_confirmation_statistics.csv"
    _write_csv_atomic(statistics, path)
    provenance = output / f"{args.family}_confirmation_inputs.json"
    _write_json_atomic(
        provenance,
        {
            "schema_version": "realistic_niah_v4_causal_v2_confirmation_inputs_v1",
            "family": args.family,
            "model_label": args.model,
            "screen_detail_sha256": _sha256_file(screen_path),
            "confirmation_detail_sha256": _sha256_file(confirmation_path),
            "selection_json_sha256": _sha256_file(selection_path),
            "input_rows_after_selection": int(len(detail)),
            "screen_seeds": sorted(
                detail.loc[detail["evidence_split"].eq("screen"), "seed"]
                .astype(int)
                .unique()
                .tolist()
            ),
            "held_out_confirmation_seeds": sorted(
                detail.loc[detail["evidence_split"].eq("held_out_confirmation"), "seed"]
                .astype(int)
                .unique()
                .tolist()
            ),
        },
    )
    return {
        "statistics": str(path),
        "provenance": str(provenance),
        "rows": len(statistics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run restartable V4.4-only causal-v2 experiments."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "prompt-alignment",
            "baseline",
            "head-rankings",
            "ablation",
            "prompt-patching",
            "answer-patching",
            "steering-centroids",
            "steering",
            "select",
            "confirmation-stats",
        ),
    )
    parser.add_argument("--phase", choices=("screen", "confirmation"), default="screen")
    parser.add_argument(
        "--family", choices=("prompt_patching", "answer_patching", "steering")
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/realistic_niah_v4.json")
    parser.add_argument(
        "--causal-config", default="configs/realistic_niah_v4_causal_v2.json"
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--base-behavior-labels")
    parser.add_argument("--baseline-labels")
    parser.add_argument(
        "--attention-source",
        "--attention-detail",
        dest="attention_source",
        help=(
            "Raw full-span attention CSV, capture index/directory, run root, or "
            "AUTO. --attention-detail is retained as a compatibility alias."
        ),
    )
    parser.add_argument("--head-rankings")
    parser.add_argument("--centroids")
    parser.add_argument("--selection-json")
    parser.add_argument("--detail")
    parser.add_argument("--screen-detail")
    parser.add_argument("--confirmation-detail")
    parser.add_argument("--output-dir")
    parser.add_argument("--generation-max-new-tokens", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_config_path = Path(args.config).resolve()
    causal_config_path = Path(args.causal_config).resolve()
    stimuli_path = Path(args.stimuli).resolve()
    run_root = Path(args.run_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    if not stimuli_path.is_file():
        raise FileNotFoundError(stimuli_path)
    config = V4Config.from_json(base_config_path)
    design = CausalV2Design.from_json(causal_config_path)
    model_spec = resolve_model_spec(args.model)
    causal_root = run_root / model_spec.label / "numeric" / "causal_v2"
    causal_root.mkdir(parents=True, exist_ok=True)
    context = {
        "config": config,
        "design": design,
        "base_config_path": base_config_path,
        "causal_config_path": causal_config_path,
        "stimuli_path": stimuli_path,
        "run_root": run_root,
        "causal_root": causal_root,
        "repo_root": repo_root,
        "profile": "smoke" if args.smoke else "formal",
    }
    _write_json_atomic(causal_root / "last_runtime.json", _runtime_payload(repo_root))
    started = time.perf_counter()
    if args.stage == "prompt-alignment":
        outputs = _prompt_alignment_stage(args, context)
    elif args.stage == "baseline":
        outputs = _baseline_stage(args, context)
    elif args.stage == "head-rankings":
        outputs = _head_ranking_stage(args, context)
    elif args.stage == "ablation":
        outputs = _ablation_stage(args, context)
    elif args.stage in {"prompt-patching", "answer-patching"}:
        outputs = _patching_stage(args, context)
    elif args.stage == "steering-centroids":
        outputs = _steering_centroids_stage(args, context)
    elif args.stage == "steering":
        outputs = _steering_stage(args, context)
    elif args.stage == "select":
        outputs = _selection_stage(args, context)
    else:
        outputs = _confirmation_stats_stage(args, context)
    outputs["elapsed_seconds"] = float(time.perf_counter() - started)
    outputs["stage"] = args.stage
    outputs["phase"] = args.phase
    outputs["profile"] = context["profile"]
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
