from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate a frozen Top-K broad-head restoration response from the "
            "audited per-head span-restoration summary."
        )
    )
    parser.add_argument("--broad-summary", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.broad_summary.resolve()
    config_path = args.experiment_config.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    registered = {
        (int(layer), int(head))
        for layer, head in config["retrieval_heads"][str(args.model)]
    }
    if not registered:
        raise RuntimeError("The frozen head registry is empty")

    frame = pd.read_csv(source)
    required_columns = {
        "model_label",
        "patch_kind",
        "patch_layer",
        "layer",
        "head",
        "rows",
        "seeds",
        "mean_delta_needle_mass",
        "mean_delta_broad_score",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise RuntimeError(f"Broad summary lacks columns: {missing_columns}")

    frame = frame[frame["model_label"].eq(str(args.model))].copy()
    frame = frame[
        frame.apply(
            lambda row: (int(row["layer"]), int(row["head"])) in registered,
            axis=1,
        )
    ].copy()
    frame = frame[frame["patch_kind"].isin(("needle_full", "ordinary_full"))]
    if frame.empty:
        raise RuntimeError("No registered-head restoration rows were found")
    if set(zip(frame["layer"].astype(int), frame["head"].astype(int))) != registered:
        raise RuntimeError("The source summary does not cover the exact frozen registry")
    if not frame["rows"].astype(int).eq(300).all():
        raise RuntimeError("Expected 300 canonical prompts per head/layer response")
    if not frame["seeds"].astype(int).eq(30).all():
        raise RuntimeError("Expected all 30 canonical seeds per head/layer response")

    grouped = (
        frame.groupby(["patch_layer", "patch_kind"], as_index=False)
        .agg(
            heads=("head", "size"),
            mean_delta_needle_mass=("mean_delta_needle_mass", "mean"),
            mean_delta_broad_score=("mean_delta_broad_score", "mean"),
        )
        .sort_values(["patch_layer", "patch_kind"])
    )
    expected_layers = set(int(value) for value in config["pilot_layers"][str(args.model)])
    observed_layers = set(grouped["patch_layer"].astype(int))
    if observed_layers != expected_layers:
        raise RuntimeError(
            f"Layer coverage mismatch: observed={sorted(observed_layers)} "
            f"expected={sorted(expected_layers)}"
        )
    if not grouped["heads"].astype(int).eq(len(registered)).all():
        raise RuntimeError("A layer/condition does not contain the exact frozen head count")

    pivot = grouped.pivot(
        index="patch_layer",
        columns="patch_kind",
        values=["mean_delta_needle_mass", "mean_delta_broad_score"],
    )
    rows: list[dict[str, Any]] = []
    for layer in sorted(expected_layers):
        needle_mass = float(pivot.loc[layer, ("mean_delta_needle_mass", "needle_full")])
        ordinary_mass = float(
            pivot.loc[layer, ("mean_delta_needle_mass", "ordinary_full")]
        )
        needle_broad = float(
            pivot.loc[layer, ("mean_delta_broad_score", "needle_full")]
        )
        ordinary_broad = float(
            pivot.loc[layer, ("mean_delta_broad_score", "ordinary_full")]
        )
        rows.append(
            {
                "model_label": str(args.model),
                "patch_layer": int(layer),
                "frozen_heads": len(registered),
                "seeds": 30,
                "counts": "1-10",
                "needle_delta_mass": needle_mass,
                "ordinary_delta_mass": ordinary_mass,
                "mass_specificity": needle_mass - ordinary_mass,
                "needle_delta_broad_score": needle_broad,
                "ordinary_delta_broad_score": ordinary_broad,
                "broad_specificity": needle_broad - ordinary_broad,
            }
        )

    result = pd.DataFrame(rows)
    result_path = output / "attention_response_topk.csv"
    result.to_csv(result_path, index=False, float_format="%.9f")
    audit = {
        "status": "PASS",
        "schema_version": "realistic_niah_v4_4_5_topk_attention_response_v1",
        "model": str(args.model),
        "frozen_heads": [
            {"layer": layer, "head": head} for layer, head in sorted(registered)
        ],
        "head_count": len(registered),
        "layers": len(result),
        "seeds": 30,
        "counts": list(range(1, 11)),
        "source_broad_summary": str(source),
        "source_broad_summary_sha256": sha256(source),
        "experiment_config": str(config_path),
        "experiment_config_sha256": sha256(config_path),
        "output": str(result_path),
        "output_sha256": sha256(result_path),
        "estimand": (
            "mean_over_frozen_heads[(needle_full-needle_corrupt)-"
            "(ordinary_full-ordinary_corrupt)]"
        ),
    }
    (output / "attention_response_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
