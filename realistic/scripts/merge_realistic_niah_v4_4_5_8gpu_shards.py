from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


BASELINES = ("clean", "needle_corrupt", "ordinary_corrupt")
PATCHES = (
    "restore_needle_endpoint",
    "restore_needle_full",
    "restore_ordinary_full",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def detail_key(row: dict[str, Any]) -> tuple[int, int, str, int]:
    return (
        int(row["seed"]),
        int(row["gold_count"]),
        str(row["condition"]),
        int(row["patch_layer"]),
    )


def expected_keys(seeds: list[int], layers: list[int]) -> set[tuple[int, int, str, int]]:
    keys = {
        (seed, count, condition, -1)
        for seed in seeds
        for count in range(1, 11)
        for condition in BASELINES
    }
    keys.update(
        (seed, count, condition, layer)
        for seed in seeds
        for count in range(1, 11)
        for layer in layers
        for condition in PATCHES
    )
    return keys


def link_state(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise RuntimeError(f"State collision: {destination}")
    os.link(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-name", default="canonical_merged")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    root = Path(manifest["run_root"])
    final = root / args.output_name
    partial = root / f"{args.output_name}.partial"
    if final.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite {final} or {partial}")
    partial.mkdir(parents=True)

    audit_models: dict[str, Any] = {}
    try:
        for model in ("Qwen3-8B", "Gemma4-E4B"):
            shards = [item for item in manifest["shards"] if item["model"] == model]
            model_root = partial / model
            model_root.mkdir()
            rows_by_key: dict[tuple[int, int, str, int], dict[str, Any]] = {}
            expected: set[tuple[int, int, str, int]] = set()
            state_sources: dict[str, Path] = {}
            for shard in shards:
                source_root = root / "shards" / shard["label"] / model
                rows = read_jsonl(source_root / "detail.jsonl")
                shard_expected = expected_keys(shard["seeds"], shard["layers"])
                shard_keys = {detail_key(row) for row in rows}
                if shard_keys != shard_expected or len(rows) != len(shard_keys):
                    raise RuntimeError(f"Detail coverage failed for {shard['label']}")
                if expected & shard_keys:
                    raise RuntimeError(f"Cross-shard detail overlap at {shard['label']}")
                expected.update(shard_expected)
                for row in rows:
                    key = detail_key(row)
                    rows_by_key[key] = row
                    relative = str(row["state_path"])
                    source = source_root / relative
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    previous = state_sources.setdefault(relative, source)
                    if previous != source:
                        raise RuntimeError(f"State-name collision: {relative}")

            if set(rows_by_key) != expected:
                raise RuntimeError(f"Merged detail coverage failed for {model}")
            ordered = [rows_by_key[key] for key in sorted(rows_by_key)]
            with (model_root / "detail.jsonl").open("w", encoding="utf-8") as handle:
                for row in ordered:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            for relative, source in state_sources.items():
                link_state(source, model_root / relative)

            expected_heads = 32 if model == "Qwen3-8B" else 8
            broad_count = 0
            broad_keys: set[tuple[int, int, str, int, int, int]] = set()
            with (model_root / "broad_metrics.jsonl").open(
                "w", encoding="utf-8"
            ) as output:
                for shard in shards:
                    source = root / "shards" / shard["label"] / model / "broad_metrics.jsonl"
                    with source.open(encoding="utf-8") as handle:
                        for line in handle:
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            key = (
                                int(row["seed"]),
                                int(row["gold_count"]),
                                str(row["condition"]),
                                int(row["patch_layer"]),
                                int(row["layer"]),
                                int(row["head"]),
                            )
                            if key in broad_keys:
                                raise RuntimeError(f"Duplicate broad key: {key}")
                            broad_keys.add(key)
                            broad_count += 1
                            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            expected_broad = len(ordered) * expected_heads
            if broad_count != expected_broad:
                raise RuntimeError(
                    f"{model} broad rows {broad_count} != {expected_broad}"
                )
            complete = {
                "status": "complete",
                "model": model,
                "detail_rows": len(ordered),
                "unique_detail_keys": len(rows_by_key),
                "state_files": len(state_sources),
                "broad_rows": broad_count,
                "unique_broad_keys": len(broad_keys),
            }
            (model_root / "complete.json").write_text(
                json.dumps(complete, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            audit_models[model] = complete
        audit = {
            "schema_version": "realistic_niah_v4_4_5_8gpu_merge_audit_v1",
            "stimulus_sha256": manifest["stimulus_sha256"],
            "models": audit_models,
            "status": "PASS",
        }
        (partial / "merge_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        partial.rename(final)
        print(json.dumps(audit, indent=2, sort_keys=True))
    except BaseException:
        (partial / "MERGE_FAILED").write_text("See exception output.\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
