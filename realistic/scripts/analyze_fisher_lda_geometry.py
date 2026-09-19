#!/usr/bin/env python3
"""Build discovery-fitted Fisher/LDA3 views for the registered NIAH endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v5.cross_mode_geometry import (  # noqa: E402
    CLASSES,
    ModeDataset,
    load_native_thinking_capture,
    load_non_thinking_capture,
)
from realistic_niah_v5.dual_endpoint_geometry import (  # noqa: E402
    load_native_thinking_final_count,
    load_non_thinking_final_count,
)
from realistic_niah_v5.fisher_lda_geometry import (  # noqa: E402
    discovery_fitted_fisher_lda3,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")
SCHEMA_VERSION = "niah_discovery_fitted_fisher_lda3_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--non-thinking-root",
        type=Path,
        default=ROOT / "work/nonthinking_v44_geometry_300_150_136_166_78",
    )
    parser.add_argument(
        "--native-root",
        type=Path,
        default=ROOT / "work/v5_geometry_full_panel",
    )
    parser.add_argument(
        "--dual-endpoint-root",
        type=Path,
        default=ROOT / "reports/v5_dual_endpoint_geometry_full300",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/fisher_lda_geometry",
    )
    parser.add_argument("--pca-dim", type=int, default=16)
    parser.add_argument("--relative-ridge", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_selected(path: Path, mode: str) -> dict[str, Any]:
    frame = pd.read_csv(path)
    rows = frame.loc[frame["mode"].astype(str).eq(mode)]
    if "analysis_group" in rows.columns:
        expected = "all_traces" if str(rows.iloc[0]["endpoint"]) == "running_index" else "all_counts"
        rows = rows.loc[rows["analysis_group"].astype(str).eq(expected)]
    if len(rows) != 1:
        raise ValueError(f"Expected one selected row for {path}/{mode}; got {len(rows)}")
    return rows.iloc[0].to_dict()


def dataset_specs(
    non_root: Path, native_root: Path, model: str
) -> list[tuple[str, str, Path, Callable[[], ModeDataset]]]:
    non_representation = non_root / model / "numeric/representation"
    non_running = non_representation / "capture/capture_index.jsonl"
    non_final = non_representation / "answer_query_all_layers_v1/capture_index.jsonl"
    native_running = native_root / "running" / model / "capture_index.jsonl"
    native_final = native_root / "final" / model / "capture_index.jsonl"
    return [
        (
            "running_index",
            "non_thinking",
            non_running,
            lambda: load_non_thinking_capture(non_running, pooling="span_end"),
        ),
        (
            "running_index",
            "native_thinking",
            native_running,
            lambda: load_native_thinking_capture(
                native_running, site_kind="item_end", cohort="parser_hit"
            ),
        ),
        (
            "final_count",
            "non_thinking",
            non_final,
            lambda: load_non_thinking_final_count(non_final),
        ),
        (
            "final_count",
            "native_thinking",
            native_final,
            lambda: load_native_thinking_final_count(native_final),
        ),
    ]


def main() -> None:
    args = parse_args()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "models": {},
    }
    selected_inputs: list[Path] = []
    capture_inputs: list[Path] = []
    summary_rows: list[dict[str, Any]] = []
    for model in MODELS:
        directory = args.dual_endpoint_root / model / "pca16_whiten"
        selected_paths = {
            "running_index": directory / "running_index_selected.csv",
            "final_count": directory / "final_count_selected.csv",
        }
        selected_inputs.extend(selected_paths.values())
        payload["models"][model] = {}
        for endpoint, mode, capture_index, loader in dataset_specs(
            args.non_thinking_root, args.native_root, model
        ):
            selected = read_selected(selected_paths[endpoint], mode)
            dataset = loader()
            layer = int(selected["layer"])
            if layer not in dataset.states_by_layer:
                raise ValueError(f"{model}/{endpoint}/{mode} lacks selected L{layer}")
            geometry = discovery_fitted_fisher_lda3(
                dataset.states_by_layer[layer],
                dataset.metadata,
                CLASSES,
                pca_dim=args.pca_dim,
                relative_ridge=args.relative_ridge,
                random_state=args.random_state,
            )
            geometry.update(
                {
                    "model_label": model,
                    "endpoint": endpoint,
                    "mode": mode,
                    "token_site": str(selected["token_site"]),
                    "selected_layer": layer,
                    "layer_selection": (
                        "grouped discovery-only mean Logistic/NCC balanced accuracy"
                    ),
                    "held_out": {
                        "logistic_balanced_accuracy": float(
                            selected["confirmation_logistic_balanced_accuracy"]
                        ),
                        "ncc_balanced_accuracy": float(
                            selected["confirmation_ncc_balanced_accuracy"]
                        ),
                        "rows": int(selected["confirmation_rows"]),
                        "support_min": int(selected["confirmation_support_min"]),
                        "support_max": int(selected["confirmation_support_max"]),
                    },
                }
            )
            payload["models"][model].setdefault(endpoint, {})[mode] = geometry
            summary_rows.append(
                {
                    "model_label": model,
                    "endpoint": endpoint,
                    "mode": mode,
                    "token_site": str(selected["token_site"]),
                    "selected_layer": layer,
                    "confirmation_logistic_balanced_accuracy": geometry["held_out"][
                        "logistic_balanced_accuracy"
                    ],
                    "confirmation_ncc_balanced_accuracy": geometry["held_out"][
                        "ncc_balanced_accuracy"
                    ],
                    "top3_fisher_trace_fraction": geometry["fit"][
                        "top3_fisher_trace_fraction"
                    ],
                    "discovery_lda3_silhouette": geometry["metrics"][
                        "discovery_lda3_class_balanced_silhouette"
                    ],
                    "confirmation_lda3_silhouette": geometry["metrics"][
                        "confirmation_lda3_class_balanced_silhouette"
                    ],
                    "confirmation_lda3_radius_gap_ratio": geometry["metrics"][
                        "confirmation_lda3_radius_gap_ratio"
                    ],
                }
            )
            capture_inputs.append(capture_index)
            print(
                model,
                endpoint,
                mode,
                f"L{layer}",
                f"top3={geometry['fit']['top3_fisher_trace_fraction']:.3f}",
                f"C-sil={geometry['metrics']['confirmation_lda3_class_balanced_silhouette']:.3f}",
                f"C-radius/gap={geometry['metrics']['confirmation_lda3_radius_gap_ratio']:.3f}",
                flush=True,
            )

    args.output.mkdir(parents=True, exist_ok=True)
    payload_path = args.output / "geometry_payload.json"
    summary_path = args.output / "summary.csv"
    audit_path = args.output / "audit.json"
    payload_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classes": list(CLASSES),
        "pca_dim": int(args.pca_dim),
        "relative_covariance_ridge": float(args.relative_ridge),
        "random_state": int(args.random_state),
        "fit_split": "discovery only",
        "display_default": "confirmation only",
        "selection": (
            "reuse each model x endpoint x mode layer selected by grouped "
            "discovery-only mean Logistic/NCC balanced accuracy"
        ),
        "interpretation_guardrail": (
            "Fisher/LDA3 is supervised and maximizes discovery class separation. "
            "It is a classifier-aligned diagnostic, not an unsupervised geometry "
            "view; only frozen confirmation structure is interpreted."
        ),
        "inputs": {
            str(path.resolve()): sha256(path)
            for path in sorted(set(selected_inputs + capture_inputs), key=str)
        },
        "outputs": {
            str(payload_path.resolve()): sha256(payload_path),
            str(summary_path.resolve()): sha256(summary_path),
        },
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
