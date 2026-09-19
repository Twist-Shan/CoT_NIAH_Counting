from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


EXPECTED_MASS = "sum of answer-query attention over every token in each active needle span"
EXPECTED_SCORE = "mean(broad_mass * exp(entropy(per-needle mass))/needle_count)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(registry_path: Path, scores_path: Path, configs: list[Path]) -> dict[str, object]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("mass_definition") != EXPECTED_MASS:
        raise ValueError("Registry does not use full-span literal mass")
    if registry.get("broad_aggregation_definition") != EXPECTED_SCORE:
        raise ValueError("Registry broad score changed")
    frame = pd.read_csv(scores_path)
    eligible = frame[pd.to_numeric(frame["layer"], errors="raise").astype(int) < 36]
    eligible = eligible.sort_values(
        ["broad_aggregation_score", "layer", "head"],
        ascending=[False, True, True],
    )
    ordered = [
        [int(row["layer"]), int(row["head"])]
        for row in eligible.to_dict(orient="records")
    ]
    rows: list[dict[str, object]] = []
    for path in configs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        top_n = int(payload["top_n"])
        candidate = [[int(x) for x in site] for site in payload["candidate_sites"]]
        if candidate != ordered[:top_n]:
            raise ValueError(f"{path.name} is not the eligible full-span prefix")
        expected_controls = [
            [[layer, (head + offset) % 8] for layer, head in candidate]
            for offset in (1, 2, 3)
        ]
        controls = [
            [[int(x) for x in site] for site in site_set]
            for site_set in payload["matched_control_sets"]
        ]
        if controls != expected_controls:
            raise ValueError(f"{path.name} controls are not the frozen cyclic rotations")
        rows.append(
            {
                "config": str(path),
                "config_sha256": sha256(path),
                "top_n": top_n,
                "candidate_sites": candidate,
                "matched_controls": controls,
            }
        )
    return {
        "schema_version": "realistic_niah_v4_4_4_gemma_full_span_config_audit_v1",
        "status": "passed",
        "registry": str(registry_path),
        "registry_sha256": sha256(registry_path),
        "scores": str(scores_path),
        "scores_sha256": sha256(scores_path),
        "ranking_definition": {"mass": EXPECTED_MASS, "score": EXPECTED_SCORE},
        "temporal_filter": "layer < 36",
        "configs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = verify(
        Path(args.registry).resolve(),
        Path(args.scores).resolve(),
        [Path(value).resolve() for value in args.config],
    )
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
