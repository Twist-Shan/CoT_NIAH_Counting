#!/usr/bin/env python3
"""Analyze city/flower/animal answer-endpoint geometry on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from realistic_niah_v5.domain_transfer_geometry import (  # noqa: E402
    CONFIRMATION_SEEDS,
    DISCOVERY_SEEDS,
    SCHEMA_VERSION,
    capture_audit,
    city_anchored_pca3,
    combine_city_and_transfer,
    evaluate_frozen_layer,
    flatten_dimension_rows,
    load_city_answer_endpoints,
    load_transfer_answer_endpoints,
    select_layer,
    subset_by_seeds,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _index_paths(
    model: str,
    *,
    nonthinking_city_root: Path,
    native_city_root: Path,
    transfer_root: Path,
) -> dict[str, tuple[Path, Path]]:
    return {
        "non_thinking": (
            nonthinking_city_root
            / model
            / "numeric"
            / "representation"
            / "answer_query_all_layers_v1"
            / "capture_index.jsonl",
            transfer_root / "nonthinking" / model / "capture_index.jsonl",
        ),
        "native_thinking": (
            native_city_root / model / "capture_index.jsonl",
            transfer_root / "native" / model / "capture_index.jsonl",
        ),
    }


def analyze(
    *,
    nonthinking_city_root: Path,
    native_city_root: Path,
    transfer_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    layer_rows: list[dict[str, Any]] = []
    inputs: list[Path] = []
    site_catalogs: dict[str, dict[str, str]] = {}
    for model in MODELS:
        results[model] = {}
        for mode, (city_index, transfer_index) in _index_paths(
            model,
            nonthinking_city_root=nonthinking_city_root,
            native_city_root=native_city_root,
            transfer_root=transfer_root,
        ).items():
            city = load_city_answer_endpoints(city_index, mode=mode)
            transfer = load_transfer_answer_endpoints(transfer_index, mode=mode)
            panel = combine_city_and_transfer(
                subset_by_seeds(city, CONFIRMATION_SEEDS), transfer
            )
            selected_layer, sweep = select_layer(
                city, selection_seeds=DISCOVERY_SEEDS
            )
            metrics = evaluate_frozen_layer(
                panel,
                training_dataset=city,
                layer=selected_layer,
                selection_seeds=DISCOVERY_SEEDS,
                evaluation_seeds=CONFIRMATION_SEEDS,
            )
            visualization = city_anchored_pca3(
                panel,
                training_dataset=city,
                layer=selected_layer,
                selection_seeds=DISCOVERY_SEEDS,
            )
            audit = capture_audit(panel)
            site_index = transfer_index.parent / "site_index.jsonl"
            site_manifest = transfer_index.parent / "site_index_manifest.json"
            if not site_index.is_file() or not site_manifest.is_file():
                raise FileNotFoundError(
                    f"Missing reusable running/answer site catalog beside "
                    f"{transfer_index}"
                )
            results[model][mode] = {
                "selected_layer": selected_layer,
                "selection_rule": (
                    "maximize mean of Logistic and nearest-centroid balanced "
                    "accuracy under five-fold seed-grouped CV on the twenty city "
                    "discovery seeds; ties prefer the earlier layer"
                ),
                "metrics": metrics,
                "visualization": visualization,
                "audit": audit,
            }
            layer_rows.extend(sweep)
            inputs.extend(
                [
                    city_index.resolve(),
                    transfer_index.resolve(),
                    site_index.resolve(),
                    site_manifest.resolve(),
                ]
            )
            site_catalogs[f"{model}/{mode}"] = {
                "site_index": str(site_index.resolve()),
                "site_index_manifest": str(site_manifest.resolve()),
            }
            print(
                f"[domain-transfer geometry] {model} {mode} "
                f"selected=L{selected_layer}",
                flush=True,
            )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "design": {
            "domains": ["city", "flower", "animal"],
            "counts": list(range(1, 11)),
            "city_discovery_seeds": list(DISCOVERY_SEEDS),
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
            "layer_selection_seeds": list(DISCOVERY_SEEDS),
            "layer_selection_domain": "city",
            "evaluation_seeds": list(CONFIRMATION_SEEDS),
            "rows_per_model_mode": 300,
            "answer_endpoint": {
                "non_thinking": "prompt-final Total: colon",
                "native_thinking": "answer_query_v3 immediately before numeric answer",
            },
            "running_endpoint_saved_not_analyzed_here": {
                "non_thinking": "each active prompt record span end",
                "native_thinking": "each parser-observed thinking item end; ragged and duplicates retained",
            },
        },
        "preserved_transfer_site_catalogs": site_catalogs,
        "models": results,
        "inputs": {str(path): _sha256(path) for path in sorted(set(inputs), key=str)},
    }
    output_root.mkdir(parents=True, exist_ok=True)
    payload_path = output_root / "report_payload.json"
    _atomic_json(payload_path, payload)
    pd.DataFrame(layer_rows).to_csv(output_root / "layer_selection_sweep.csv", index=False)
    pd.DataFrame(list(flatten_dimension_rows(results))).to_csv(
        output_root / "pca_dimension_sweep.csv", index=False
    )
    _atomic_json(
        output_root / "analysis_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "payload": str(payload_path.resolve()),
            "payload_sha256": _sha256(payload_path),
            "layer_selection_sweep": str(
                (output_root / "layer_selection_sweep.csv").resolve()
            ),
            "pca_dimension_sweep": str(
                (output_root / "pca_dimension_sweep.csv").resolve()
            ),
        },
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nonthinking-city-root", type=Path, required=True)
    parser.add_argument("--native-city-root", type=Path, required=True)
    parser.add_argument("--transfer-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    analyze(
        nonthinking_city_root=args.nonthinking_city_root,
        native_city_root=args.native_city_root,
        transfer_root=args.transfer_root,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
