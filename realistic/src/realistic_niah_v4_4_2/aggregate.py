from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


PAIR_SPECS = (
    (
        "cue_present_mode_effect",
        ("nonthinking", "cue_present"),
        ("native_thinking", "cue_present"),
    ),
    (
        "native_cue_effect",
        ("native_thinking", "cue_present"),
        ("native_thinking", "cue_absent"),
    ),
    (
        "cue_absent_mode_effect",
        ("nonthinking", "cue_absent"),
        ("native_thinking", "cue_absent"),
    ),
)

LEGACY_MODE_COMPARISON = "cue_present_mode_effect_legacy"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _legacy_artifact(root: Path, model: str, artifact: str) -> Path | None:
    reference = root / "legacy_v4_4_reference.json"
    if not reference.exists():
        return None
    matches = [
        Path(row["path"])
        for row in _read_json(reference)["artifacts"]
        if row["model_label"] == model and row["artifact"] == artifact
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Duplicate legacy artifact: {model}/{artifact}")
    return matches[0] if matches else None


def _capture_lookup(root: Path) -> dict[tuple[str, str, str, str], Path]:
    result: dict[tuple[str, str, str, str], Path] = {}
    for path in sorted(root.glob("conditions/**/capture/capture_manifest.json")):
        row = _read_json(path)
        key = (
            str(row["model_label"]),
            str(row["stimulus_id"]),
            str(row["mode"]),
            str(row["prompt_variant"]),
        )
        if key in result:
            raise RuntimeError(f"Duplicate capture key: {key}")
        result[key] = path.parent
    return result


def _role_vector(
    capture_dir: Path,
    manifest: dict[str, Any],
    *,
    layer: int,
    role: str,
) -> torch.Tensor | None:
    roles = [str(value) for value in manifest["query_roles"]]
    indices = [index for index, value in enumerate(roles) if value == role]
    if not indices:
        return None
    hidden = torch.load(
        capture_dir / f"layer_{layer:02d}_hidden.pt",
        map_location="cpu",
        weights_only=True,
    ).float()
    selected = hidden[indices]
    if role == "trace_mean":
        return selected.mean(dim=0)
    return selected[-1]


def _hidden_vector(
    capture_dir: Path,
    manifest: dict[str, Any],
    *,
    layer: int,
    site: str,
) -> torch.Tensor | None:
    if site == "trace_mean":
        roles = [str(value) for value in manifest["query_roles"]]
        indices = [index for index, value in enumerate(roles) if value == "trace"]
        if not indices:
            return None
        hidden = torch.load(
            capture_dir / f"layer_{layer:02d}_hidden.pt",
            map_location="cpu",
            weights_only=True,
        ).float()
        return hidden[indices].mean(dim=0)
    return _role_vector(capture_dir, manifest, layer=layer, role=site)


def hidden_paired_effects(root: str | Path) -> pd.DataFrame:
    output = Path(root)
    lookup = _capture_lookup(output)
    rows: list[dict[str, Any]] = []
    identities = sorted({(key[0], key[1]) for key in lookup})
    for comparison, left_condition, right_condition in PAIR_SPECS:
        sites = (
            ("trace_mean", "trace", "answer_query")
            if comparison == "native_cue_effect"
            else ("answer_query",)
        )
        for model, stimulus_id in identities:
            left_key = (model, stimulus_id, *left_condition)
            right_key = (model, stimulus_id, *right_condition)
            if left_key not in lookup or right_key not in lookup:
                continue
            left_dir, right_dir = lookup[left_key], lookup[right_key]
            left_manifest = _read_json(left_dir / "capture_manifest.json")
            right_manifest = _read_json(right_dir / "capture_manifest.json")
            left_layers = {int(value["layer"]) for value in left_manifest["layers"]}
            right_layers = {int(value["layer"]) for value in right_manifest["layers"]}
            for layer in sorted(left_layers & right_layers):
                for site in sites:
                    left = _hidden_vector(
                        left_dir, left_manifest, layer=layer, site=site
                    )
                    right = _hidden_vector(
                        right_dir, right_manifest, layer=layer, site=site
                    )
                    if left is None or right is None:
                        continue
                    delta = right - left
                    rows.append(
                        {
                            "comparison": comparison,
                            "model_label": model,
                            "stimulus_id": stimulus_id,
                            "seed": int(left_manifest["seed"]),
                            "split": str(left_manifest["split"]),
                            "gold_count": int(left_manifest["gold_count"]),
                            "layer": layer,
                            "site": site,
                            "cosine": float(F.cosine_similarity(left[None], right[None])),
                            "l2_delta": float(delta.norm()),
                            "relative_l2_delta": float(
                                delta.norm() / left.norm().clamp_min(1e-12)
                            ),
                            "norm_delta": float(right.norm() - left.norm()),
                        }
                    )

    # The already-completed V4.4 cue-present/non-thinking answer-query capture
    # is attached by path.  It is never copied or recomputed.
    native = {
        (key[0], key[1]): value
        for key, value in lookup.items()
        if key[2:] == ("native_thinking", "cue_present")
    }
    for model in sorted({key[0] for key in native}):
        index_path = _legacy_artifact(
            output,
            model,
            "representation/answer_query_all_layers_v1/capture_index.jsonl",
        )
        if index_path is None:
            continue
        for record in _read_jsonl(index_path):
            if str(record.get("design_variant")) != "v4.4":
                continue
            stimulus_id = str(record["stimulus_id"])
            capture_dir = native.get((model, stimulus_id))
            if capture_dir is None:
                continue
            manifest = _read_json(capture_dir / "capture_manifest.json")
            with np.load(index_path.parent / str(record["shard_path"]), allow_pickle=False) as saved:
                legacy_layers = [int(value) for value in saved["layer_indices"]]
                legacy_states = torch.from_numpy(saved["query_states"]).float()
            native_layers = {int(value["layer"]) for value in manifest["layers"]}
            for legacy_index, layer in enumerate(legacy_layers):
                if layer not in native_layers:
                    continue
                left = legacy_states[legacy_index]
                right = _hidden_vector(
                    capture_dir, manifest, layer=layer, site="answer_query"
                )
                if right is None or left.shape != right.shape:
                    continue
                delta = right - left
                rows.append(
                    {
                        "comparison": LEGACY_MODE_COMPARISON,
                        "model_label": model,
                        "stimulus_id": stimulus_id,
                        "seed": int(manifest["seed"]),
                        "split": str(manifest["split"]),
                        "gold_count": int(manifest["gold_count"]),
                        "layer": layer,
                        "site": "answer_query",
                        "cosine": float(F.cosine_similarity(left[None], right[None])),
                        "l2_delta": float(delta.norm()),
                        "relative_l2_delta": float(
                            delta.norm() / left.norm().clamp_min(1e-12)
                        ),
                        "norm_delta": float(right.norm() - left.norm()),
                    }
                )
    return pd.DataFrame(rows)


def attention_paired_effects(root: str | Path) -> pd.DataFrame:
    output = Path(root)
    lookup = _capture_lookup(output)
    rows: list[dict[str, Any]] = []
    identities = sorted({(key[0], key[1]) for key in lookup})
    for comparison, left_condition, right_condition in PAIR_SPECS:
        for model, stimulus_id in identities:
            left_key = (model, stimulus_id, *left_condition)
            right_key = (model, stimulus_id, *right_condition)
            if left_key not in lookup or right_key not in lookup:
                continue
            left_dir, right_dir = lookup[left_key], lookup[right_key]
            if not (left_dir / "attention_summary.pt").exists() or not (
                right_dir / "attention_summary.pt"
            ).exists():
                continue
            left = torch.load(
                left_dir / "attention_summary.pt",
                map_location="cpu",
                weights_only=True,
            )
            right = torch.load(
                right_dir / "attention_summary.pt",
                map_location="cpu",
                weights_only=True,
            )
            if tuple(left["region_names"]) != tuple(right["region_names"]):
                raise RuntimeError("Attention region schema changed across conditions")
            left_manifest = _read_json(left_dir / "capture_manifest.json")
            common_layers = sorted(set(left["layers"]) & set(right["layers"]))
            field = (
                "head_region_mean"
                if comparison == "native_cue_effect"
                else "answer_query_last_region"
            )
            for layer in common_layers:
                left_values = left["layers"][layer][field].float()
                right_values = right["layers"][layer][field].float()
                if left_values.shape != right_values.shape:
                    raise RuntimeError(
                        f"Attention shape changed at {model}/{stimulus_id}/L{layer}"
                    )
                for head in range(left_values.shape[0]):
                    for region_index, region in enumerate(left["region_names"]):
                        rows.append(
                            {
                                "comparison": comparison,
                                "model_label": model,
                                "stimulus_id": stimulus_id,
                                "seed": int(left_manifest["seed"]),
                                "split": str(left_manifest["split"]),
                                "gold_count": int(left_manifest["gold_count"]),
                                "layer": int(layer),
                                "head": head,
                                "site": (
                                    "trace" if field == "head_region_mean" else "answer_query"
                                ),
                                "region": str(region),
                                "left_mass": float(left_values[head, region_index]),
                                "right_mass": float(right_values[head, region_index]),
                                "mass_delta": float(
                                    right_values[head, region_index]
                                    - left_values[head, region_index]
                                ),
                            }
                        )

    # Legacy broad_mass is the exact answer-query mass over all active needle
    # spans, matching the V4.4.2 `needle_span` region definition.
    native_attention_cache: dict[Path, dict[str, Any]] = {}
    for model in sorted({key[0] for key in lookup}):
        detail_path = _legacy_artifact(
            output, model, "attention/analysis/attention_head_detail.csv"
        )
        if detail_path is None:
            continue
        detail = pd.read_csv(detail_path)
        detail = detail[detail["design_variant"].astype(str) == "v4.4"]
        for legacy_row in detail.to_dict("records"):
            stimulus_id = str(legacy_row["stimulus_id"])
            capture_dir = lookup.get(
                (model, stimulus_id, "native_thinking", "cue_present")
            )
            if capture_dir is None or not (capture_dir / "attention_summary.pt").exists():
                continue
            if capture_dir not in native_attention_cache:
                native_attention_cache[capture_dir] = torch.load(
                    capture_dir / "attention_summary.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            native_summary = native_attention_cache[capture_dir]
            layer = int(legacy_row["layer"])
            head = int(legacy_row["head"])
            if layer not in native_summary["layers"]:
                continue
            region_index = tuple(native_summary["region_names"]).index("needle_span")
            values = native_summary["layers"][layer]["answer_query_last_region"]
            if head >= values.shape[0]:
                continue
            right_mass = float(values[head, region_index])
            left_mass = float(legacy_row["broad_mass"])
            rows.append(
                {
                    "comparison": LEGACY_MODE_COMPARISON,
                    "model_label": model,
                    "stimulus_id": stimulus_id,
                    "seed": int(legacy_row["seed"]),
                    "split": str(legacy_row["split"]),
                    "gold_count": int(legacy_row["count"]),
                    "layer": layer,
                    "head": head,
                    "site": "answer_query",
                    "region": "needle_span",
                    "left_mass": left_mass,
                    "right_mass": right_mass,
                    "mass_delta": right_mass - left_mass,
                }
            )
    return pd.DataFrame(rows)


def behavior_table(root: str | Path) -> pd.DataFrame:
    output = Path(root)
    rows: list[dict[str, Any]] = []
    selected_stimuli: set[tuple[str, str]] = set()
    for path in sorted(output.glob("conditions/**/generation.json")):
        value = _read_json(path)
        selected_stimuli.add((str(value["model_label"]), str(value["stimulus_id"])))
        rows.append(
            {
                "source": "v4_4_2",
                "model_label": value["model_label"],
                "mode": value["mode"],
                "prompt_variant": value["prompt_variant"],
                "stimulus_id": value["stimulus_id"],
                "seed": value["seed"],
                "split": value["split"],
                "gold_count": value["gold_count"],
                "parsed_count": value.get("parsed_count"),
                "exact_count": value.get("exact_count"),
                "trace_tokens": value["boundaries"]["trace_end"]
                - value["boundaries"]["trace_start"],
                "generated_tokens": value["generated_token_count"],
                "truncated": value["generation_truncated"],
                "boundary_status": value["boundaries"]["boundary_status"],
            }
        )
    legacy_path = output / "legacy_v4_4_reference.json"
    if legacy_path.exists():
        legacy = _read_json(legacy_path)
        for artifact in legacy["artifacts"]:
            if artifact["artifact"] != "behavior/capture/generation_labels.csv":
                continue
            frame = pd.read_csv(artifact["path"])
            frame = frame[frame["design_variant"].astype(str) == "v4.4"]
            for value in frame.to_dict("records"):
                if (
                    artifact["model_label"],
                    str(value["stimulus_id"]),
                ) not in selected_stimuli:
                    continue
                rows.append(
                    {
                        "source": "legacy_v4_4_reference",
                        "model_label": artifact["model_label"],
                        "mode": "nonthinking",
                        "prompt_variant": "cue_present",
                        "stimulus_id": value["stimulus_id"],
                        "seed": int(value["seed"]),
                        "split": value["split"],
                        "gold_count": int(value["gold_count"]),
                        "parsed_count": value.get("parsed_count"),
                        "exact_count": value.get("is_correct"),
                        "trace_tokens": 0,
                        "generated_tokens": value.get("generated_token_count"),
                        "truncated": value.get("generation_truncated"),
                        "boundary_status": "legacy_nonthinking",
                    }
                )
    return pd.DataFrame(rows)


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def aggregate_run(root: str | Path) -> dict[str, str]:
    output = Path(root)
    tables = output / "analysis" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    behavior = behavior_table(output)
    hidden = hidden_paired_effects(output)
    attention = attention_paired_effects(output)
    paths = {
        "behavior": tables / "behavior.csv.gz",
        "hidden_paired_effects": tables / "hidden_paired_effects.csv.gz",
        "attention_paired_effects": tables / "attention_paired_effects.csv.gz",
    }
    _write_frame(behavior, paths["behavior"])
    _write_frame(hidden, paths["hidden_paired_effects"])
    _write_frame(attention, paths["attention_paired_effects"])

    behavior_summary = pd.DataFrame()
    if not behavior.empty:
        behavior["exact_count_bool"] = behavior["exact_count"].map(
            lambda value: str(value).strip().lower() in {"1", "true", "yes"}
        )
        behavior_summary = (
            behavior.groupby(
                ["model_label", "mode", "prompt_variant"],
                as_index=False,
                dropna=False,
            )
            .agg(
                requests=("stimulus_id", "count"),
                exact_accuracy=("exact_count_bool", "mean"),
                mean_trace_tokens=("trace_tokens", "mean"),
                truncation_rate=("truncated", "mean"),
            )
        )
    behavior_summary_path = tables / "behavior_summary.csv"
    behavior_summary.to_csv(behavior_summary_path, index=False)
    report_path = output / "analysis" / "realistic_niah_v4_4_2_report.md"
    report_path.write_text(
        "# Realistic NIAH V4.4.2 trace representation report\n\n"
        "This report is generated from the append-only V4.4.2 filestream. "
        "All eight model/mode/prompt buckets are generated in this run.\n\n"
        "## Registered comparisons\n\n"
        "- `native_cue_effect`: cue-absent minus cue-present within native thinking.\n"
        "- `cue_absent_mode_effect`: native-thinking minus non-thinking under cue absence.\n"
        "- `cue_present_mode_effect`: native-thinking minus non-thinking under "
        "the cue-present prompt at the final `Total:` answer-query.\n\n"
        "## Machine-readable tables\n\n"
        f"- `{paths['behavior'].name}`\n"
        f"- `{paths['hidden_paired_effects'].name}`\n"
        f"- `{paths['attention_paired_effects'].name}`\n"
        f"- `{behavior_summary_path.name}`\n",
        encoding="utf-8",
    )
    result = {key: str(value.resolve()) for key, value in paths.items()}
    result["behavior_summary"] = str(behavior_summary_path.resolve())
    result["report"] = str(report_path.resolve())
    return result
