#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.causal import fit_centroid_subspace
from realistic_niah_v5.parsing import parse_trace_record
from realistic_niah_v5.representation import cohort_mask, load_capture_dataset


SCHEMA = "realistic_niah_v5_answer_execution_plan_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL {path}:{line_number}: {error}"
                ) from error
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discovery_seed_cv_mae(
    states: np.ndarray, labels: np.ndarray, seeds: np.ndarray
) -> tuple[float, int]:
    predictions = []
    targets = []
    folds = 0
    for seed in np.unique(seeds):
        train = seeds != seed
        test = seeds == seed
        if train.sum() < 2 or test.sum() < 1:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(states[train], labels[train])
        predictions.extend(model.predict(states[test]).tolist())
        targets.extend(labels[test].tolist())
        folds += 1
    if not predictions:
        raise ValueError("No discovery seed-CV folds were estimable")
    return float(mean_absolute_error(targets, predictions)), folds


def build(
    capture_index: Path,
    generations: Path,
    model_label: str,
    output_dir: Path,
    *,
    site_kind: str,
    site_id: str,
    rank: int,
) -> dict[str, Any]:
    dataset = load_capture_dataset(capture_index, site_kinds=[site_kind])
    metadata = dataset.metadata
    mask = cohort_mask(metadata, "one_to_one")
    mask &= metadata["split"].astype(str).eq("discovery").to_numpy()
    mask &= metadata["site_id"].astype(str).eq(site_id).to_numpy()
    layer_rows = []
    for layer in sorted(metadata.loc[mask, "layer"].astype(int).unique()):
        layer_mask = mask & metadata["layer"].astype(int).eq(layer).to_numpy()
        states = dataset.states[layer_mask]
        labels = metadata.loc[layer_mask, "gold_count"].to_numpy(dtype=float)
        seeds = metadata.loc[layer_mask, "seed"].to_numpy(dtype=int)
        mae, folds = discovery_seed_cv_mae(states, labels, seeds)
        layer_rows.append(
            {
                "layer": int(layer),
                "discovery_seed_cv_mae": mae,
                "discovery_seed_folds": int(folds),
                "discovery_observations": int(len(states)),
                "discovery_labels": sorted(int(value) for value in np.unique(labels)),
            }
        )
    if not layer_rows:
        raise ValueError("No discovery answer-query layers were available")
    selected = min(
        layer_rows,
        key=lambda row: (row["discovery_seed_cv_mae"], row["layer"]),
    )
    selected_layer = int(selected["layer"])
    layer_mask = mask & metadata["layer"].astype(int).eq(selected_layer).to_numpy()
    fit_states = dataset.states[layer_mask]
    fit_labels = metadata.loc[layer_mask, "gold_count"].to_numpy(dtype=int)
    center, basis = fit_centroid_subspace(fit_states, fit_labels, rank=rank)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_site_id = str(site_id).replace("/", "__")
    basis_path = output_dir / f"{model_label}__{safe_site_id}_basis.npz"
    temporary_basis = basis_path.with_name(basis_path.name + ".tmp")
    with temporary_basis.open("wb") as handle:
        np.savez(
            handle,
            **{
                f"center_L{selected_layer}": center,
                f"basis_L{selected_layer}": basis,
            },
        )
    temporary_basis.replace(basis_path)

    rows = [
        row
        for row in read_jsonl(generations)
        if str(row.get("model_label", row.get("model"))) == model_label
    ]
    eligible = []
    excluded_incorrect = []
    for row in rows:
        parsed = parse_trace_record(row)
        parser = parsed["parser"]
        if str(row.get("split")) != "confirmation":
            continue
        if not bool(parser.get("trace_one_to_one")):
            continue
        if not bool(parsed.get("exact_count")):
            excluded_incorrect.append(
                str(row.get("request_id", row.get("stimulus_id")))
            )
            continue
        eligible.append((row, parsed))
    by_seed: dict[int, dict[int, tuple[dict[str, Any], dict[str, Any]]]] = {}
    for row, parsed in eligible:
        seed = int(row["seed"])
        count = int(parsed["gold_count"])
        if count in by_seed.setdefault(seed, {}):
            raise ValueError(f"Duplicate confirmation seed/count: {seed}/{count}")
        by_seed[seed][count] = (row, parsed)
    pairs = []
    for seed, by_count in sorted(by_seed.items()):
        counts = sorted(by_count)
        for receiver_count in counts:
            lower = [value for value in counts if value < receiver_count]
            higher = [value for value in counts if value > receiver_count]
            donors = []
            if lower:
                donors.append((max(lower), "same_seed_nearest_lower"))
            if higher:
                donors.append((min(higher), "same_seed_nearest_higher"))
            receiver_row, receiver_parsed = by_count[receiver_count]
            receiver_id = str(
                receiver_row.get("request_id", receiver_row.get("stimulus_id"))
            )
            for donor_count, donor_role in donors:
                donor_row, donor_parsed = by_count[donor_count]
                donor_id = str(donor_row.get("request_id", donor_row.get("stimulus_id")))
                pairs.append(
                    {
                        "schema_version": SCHEMA,
                        "pair_id": (
                            f"{model_label}__seed{seed}__R{receiver_count}__D{donor_count}"
                        ),
                        "model_label": model_label,
                        "seed": seed,
                        "split": "confirmation",
                        "receiver_request_id": receiver_id,
                        "donor_request_id": donor_id,
                        "receiver_count": receiver_count,
                        "donor_count": donor_count,
                        "receiver_site_id": site_id,
                        "donor_site_id": site_id,
                        "layer": selected_layer,
                        "donor_role": donor_role,
                        "pair_direction": (
                            "higher_to_lower"
                            if donor_count > receiver_count
                            else "lower_to_higher"
                        ),
                        "receiver_exact_count": bool(receiver_parsed["exact_count"]),
                        "donor_exact_count": bool(donor_parsed["exact_count"]),
                        "pair_eligibility": (
                            "receiver_and_donor_baseline_final_answer_exact"
                        ),
                        "selection_split": "discovery",
                        "confirmation_used_for_layer_selection": False,
                    }
                )
    pair_path = output_dir / f"{model_label}__answer_execution_pairs.jsonl"
    write_jsonl(pair_path, pairs)
    layer_path = output_dir / f"{model_label}__answer_execution_layer_selection.json"
    layer_payload = {
        "schema_version": SCHEMA,
        "model_label": model_label,
        "site_kind": site_kind,
        "site_id": site_id,
        "selection_split": "discovery",
        "confirmation_used_for_selection": False,
        "selection_metric": "leave-one-seed-out ridge MAE",
        "selected_layer": selected_layer,
        "selected_rank": int(basis.shape[1]),
        "layers": layer_rows,
    }
    layer_path.write_text(
        json.dumps(layer_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = {
        "schema_version": SCHEMA,
        "model_label": model_label,
        "capture_index": str(capture_index.resolve()),
        "capture_index_sha256": sha256(capture_index),
        "generations": str(generations.resolve()),
        "generations_sha256": sha256(generations),
        "site_id": site_id,
        "query_definition": (
            "literal baseline token immediately before the first numeric "
            "answer token"
        ),
        "selection_split": "discovery",
        "confirmation_used_for_selection": False,
        "selected_layer": selected_layer,
        "basis_rank": int(basis.shape[1]),
        "confirmation_pairs": len(pairs),
        "confirmation_seeds": sorted(by_seed),
        "eligible_clean_correct_rows": len(eligible),
        "excluded_incorrect_one_to_one_rows": len(excluded_incorrect),
        "excluded_incorrect_request_ids": sorted(excluded_incorrect),
        "pair_policy": (
            "within-seed nearest available lower and higher count among strict "
            "one-to-one rows whose baseline final numeric answer is exact"
        ),
        "pair_eligibility": (
            "both receiver and donor baseline final answers must equal gold"
        ),
        "controls": [
            "self_patch",
            "full_donor_patch",
            "projected_donor_patch",
            "orthogonal_norm_matched",
        ],
        "outputs": {
            "basis": str(basis_path.resolve()),
            "basis_sha256": sha256(basis_path),
            "pairs": str(pair_path.resolve()),
            "pairs_sha256": sha256(pair_path),
            "layer_selection": str(layer_path.resolve()),
            "layer_selection_sha256": sha256(layer_path),
        },
    }
    audit_path = output_dir / f"{model_label}__answer_execution_plan_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site-kind", default="answer_query_v3")
    parser.add_argument("--site-id", default="answer_query_v3")
    parser.add_argument("--rank", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.capture_index,
                args.generations,
                args.model,
                args.output_dir,
                site_kind=args.site_kind,
                site_id=args.site_id,
                rank=args.rank,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
