from __future__ import annotations

"""Pack the frozen V4.4 prompt/answer captures into CPU-analysis shards.

The source run is strictly read-only.  One compact NPZ is produced for every
model x role x layer, with seed-grouped metadata and answer correctness joined
from the frozen behavior labels.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np


MODELS = ("Qwen3-8B", "Gemma4-E4B")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_array(values: list[object]) -> np.ndarray:
    strings = ["" if value is None else str(value) for value in values]
    width = max(1, max(map(len, strings), default=1))
    return np.asarray(strings, dtype=f"<U{width}")


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_behavior(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["design_variant"] == "v4.4"]
    result = {row["stimulus_id"]: row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate behavior stimulus_id in {path}")
    return result


def answer_metadata(row: dict[str, object], behavior: dict[str, dict[str, str]]) -> dict[str, object]:
    sample_id = str(row["stimulus_id"])
    label = behavior.get(sample_id)
    if label is None:
        raise KeyError(f"missing behavior row for {sample_id}")
    return {
        "sample_id": sample_id,
        "seed": int(row["seed"]),
        "split": str(row["split"]),
        "count": int(label["gold_count"]),
        "correct": bool_value(label["is_correct"]),
        "prediction": int(label["parsed_count"]) if label["parsed_count"] else -1,
        "count_error": int(label["count_error"]) if label["count_error"] else 0,
        "absolute_deviation": abs(int(label["count_error"])) if label["count_error"] else 0,
        "outcome_group": label["outcome_group"],
        "sequence_length": int(row["sequence_length"]),
        "query_position": int(row["query_position"]),
    }


def prompt_metadata(row: dict[str, object], count: int, behavior: dict[str, dict[str, str]]) -> dict[str, object]:
    sample_id = f"V4_4_T10000_N{count}_seed{int(row['seed'])}"
    label = behavior.get(sample_id)
    if label is None:
        raise KeyError(f"missing behavior row for prompt endpoint {sample_id}")
    return {
        "sample_id": sample_id,
        "seed": int(row["seed"]),
        "split": str(row["split"]),
        "count": count,
        "running_index": count,
        "correct": bool_value(label["is_correct"]),
        "prediction": int(label["parsed_count"]) if label["parsed_count"] else -1,
        "count_error": int(label["count_error"]) if label["count_error"] else 0,
        "absolute_deviation": abs(int(label["count_error"])) if label["count_error"] else 0,
        "outcome_group": label["outcome_group"],
        "sequence_length": int(row["sequence_length"]),
        "query_position": int(row["query_position"]),
    }


def save_layer(path: Path, states: list[np.ndarray], metadata: list[dict[str, object]]) -> None:
    keys = sorted(metadata[0])
    if any(sorted(row) != keys for row in metadata):
        raise RuntimeError("metadata schema changed within a layer")
    arrays: dict[str, np.ndarray] = {"states": np.stack(states).astype(np.float16)}
    for key in keys:
        values = [row[key] for row in metadata]
        if key in {"sample_id", "split", "outcome_group"}:
            arrays[key] = text_array(values)
        elif key == "correct":
            arrays[key] = np.asarray(values, dtype=np.bool_)
        else:
            arrays[key] = np.asarray(values, dtype=np.int64)
    np.savez_compressed(path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()

    started = time.perf_counter()
    source = args.source_run.resolve()
    output = args.output.resolve()
    shard_output = output / "layers"
    shard_output.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []

    for model in args.models:
        model_root = source / model / "numeric"
        behavior_path = model_root / "behavior/capture/generation_labels.csv"
        behavior = load_behavior(behavior_path)
        sources = {
            "prompt_running": model_root / "representation/capture/capture_index.jsonl",
            "answer_query": model_root / "representation/answer_query_all_layers_v1/capture_index.jsonl",
        }
        for role, index_path in sources.items():
            index_rows = [row for row in read_jsonl(index_path) if row.get("design_variant") == "v4.4"]
            if not index_rows:
                raise RuntimeError(f"no V4.4 rows in {index_path}")
            capture_root = index_path.parent
            layer_indices: np.ndarray | None = None
            per_layer_states: dict[int, list[np.ndarray]] = {}
            per_layer_metadata: dict[int, list[dict[str, object]]] = {}
            for row in index_rows:
                shard = capture_root / str(row["shard_path"])
                with np.load(shard, allow_pickle=False) as payload:
                    current_layers = np.asarray(payload["layer_indices"], dtype=np.int64)
                    if layer_indices is None:
                        layer_indices = current_layers
                        per_layer_states = {int(layer): [] for layer in current_layers}
                        per_layer_metadata = {int(layer): [] for layer in current_layers}
                    elif not np.array_equal(layer_indices, current_layers):
                        raise RuntimeError(f"layer mismatch in {shard}")
                    if role == "prompt_running":
                        captured = np.asarray(payload["span_end"], dtype=np.float16)
                        if captured.shape[1] != 10:
                            raise RuntimeError(f"expected 10 prompt endpoints, got {captured.shape}")
                        metas = [prompt_metadata(row, count, behavior) for count in range(1, 11)]
                        for layer_offset, layer in enumerate(current_layers):
                            per_layer_states[int(layer)].extend(captured[layer_offset])
                            per_layer_metadata[int(layer)].extend(metas)
                    else:
                        captured = np.asarray(payload["query_states"], dtype=np.float16)
                        meta = answer_metadata(row, behavior)
                        for layer_offset, layer in enumerate(current_layers):
                            per_layer_states[int(layer)].append(captured[layer_offset])
                            per_layer_metadata[int(layer)].append(meta)
                provenance.append({"model_label": model, "role": role, "source": str(shard)})
            assert layer_indices is not None
            for layer in layer_indices:
                layer = int(layer)
                filename = f"{model}__{role}__L{layer:02d}.npz"
                path = shard_output / filename
                save_layer(path, per_layer_states[layer], per_layer_metadata[layer])
                manifest_rows.append({
                    "model_label": model,
                    "role": role,
                    "layer": layer,
                    "path": str(path.relative_to(output).as_posix()),
                    "rows": len(per_layer_metadata[layer]),
                    "hidden_size": int(per_layer_states[layer][0].shape[-1]),
                })
        provenance.extend([
            {"model_label": model, "role": "behavior", "source": str(behavior_path), "sha256": sha256(behavior_path)},
            *[{"model_label": model, "role": role, "source": str(path), "sha256": sha256(path)} for role, path in sources.items()],
        ])

    manifest = {
        "schema_version": "realistic_niah_v4_4_counter_channel_manifest_v1",
        "source_run": str(source),
        "design_variant": "v4.4",
        "prompt_pooling": "span_end",
        "datasets": manifest_rows,
    }
    (output / "layer_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    audit = {
        "schema_version": "realistic_niah_v4_4_counter_channel_pack_audit_v1",
        "source_read_only": True,
        "datasets": len(manifest_rows),
        "rows_by_model_role": {
            f"{model}/{role}": sorted({int(row["rows"]) for row in manifest_rows if row["model_label"] == model and row["role"] == role})
            for model in args.models for role in ("prompt_running", "answer_query")
        },
        "provenance": provenance,
        "elapsed_seconds": time.perf_counter() - started,
        "status": "PASS",
    }
    (output / "pack_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
