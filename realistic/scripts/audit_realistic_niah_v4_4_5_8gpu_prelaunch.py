from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--allow-resume",
        action="store_true",
        help=(
            "Accept an audited partial result between the frozen preseed and "
            "final expected row counts instead of requiring the initial preseed "
            "count exactly."
        ),
    )
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    root = Path(manifest["run_root"])
    reports = []
    gpu_ids = set()
    all_model_seeds: dict[str, set[int]] = {}
    passed = True
    for shard in manifest["shards"]:
        label = str(shard["label"])
        model = str(shard["model"])
        gpu = int(shard["gpu"])
        seeds = {int(value) for value in shard["seeds"]}
        layers = {int(value) for value in shard["layers"]}
        if gpu in gpu_ids:
            passed = False
        gpu_ids.add(gpu)
        overlap = all_model_seeds.setdefault(model, set()) & seeds
        if overlap:
            passed = False
        all_model_seeds[model].update(seeds)
        detail = root / "shards" / label / model / "detail.jsonl"
        rows = read_rows(detail)
        keys = {
            (
                int(row["seed"]),
                int(row["gold_count"]),
                str(row["condition"]),
                int(row["patch_layer"]),
            )
            for row in rows
        }
        expected_keys = {
            (seed, count, condition, layer)
            for seed in seeds
            for count in range(1, 11)
            for condition, layer in (
                [("clean", -1), ("needle_corrupt", -1), ("ordinary_corrupt", -1)]
                + [
                    (f"restore_{patch_kind}", patch_layer)
                    for patch_layer in sorted(layers)
                    for patch_kind in (
                        "needle_endpoint",
                        "needle_full",
                        "ordinary_full",
                    )
                ]
            )
        }
        row_seeds = {int(row["seed"]) for row in rows}
        model_root = detail.parent
        missing_states = [
            str(row["state_path"])
            for row in rows
            if not (model_root / str(row["state_path"])).is_file()
        ]
        broad_path = model_root / "broad_metrics.jsonl"
        broad_rows = read_rows(broad_path)
        broad_keys = {
            (
                int(row["seed"]),
                int(row["gold_count"]),
                str(row["condition"]),
                int(row["patch_layer"]),
                int(row["layer"]),
                int(row["head"]),
            )
            for row in broad_rows
        }
        expected_heads = 32 if model == "Qwen3-8B" else 8
        expected_rows = int(shard["expected_rows"])
        manifest_preseed_rows = int(shard["preseed_rows"])
        row_count_valid = (
            manifest_preseed_rows <= len(rows) <= expected_rows
            if args.allow_resume
            else len(rows) == manifest_preseed_rows
        )
        shard_pass = (
            len(rows) == len(keys)
            and row_seeds.issubset(seeds)
            and row_count_valid
            and len(expected_keys) == expected_rows
            and keys.issubset(expected_keys)
            and not missing_states
            and len(broad_rows) == len(rows) * expected_heads
            and len(broad_rows) == len(broad_keys)
        )
        passed = passed and shard_pass
        reports.append(
            {
                "label": label,
                "model": model,
                "gpu": gpu,
                "seeds": sorted(seeds),
                "current_rows": len(rows),
                "manifest_preseed_rows": manifest_preseed_rows,
                "expected_rows": expected_rows,
                "unique_keys": len(keys),
                "missing_state_files": len(missing_states),
                "broad_rows": len(broad_rows),
                "unique_broad_keys": len(broad_keys),
                "status": "PASS" if shard_pass else "FAIL",
            }
        )
    expected_panel = set(range(1234, 1264))
    for seeds in all_model_seeds.values():
        passed = passed and seeds == expected_panel
    passed = passed and gpu_ids == set(range(8))
    result = {
        "schema_version": "realistic_niah_v4_4_5_8gpu_prelaunch_audit_v2",
        "allow_resume": bool(args.allow_resume),
        "shards": reports,
        "status": "PASS" if passed else "FAIL",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
