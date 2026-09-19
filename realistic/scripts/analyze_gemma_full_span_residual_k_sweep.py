from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


K_VALUES = (1, 2, 4, 8, 16)


def holm(values: list[float]) -> list[float]:
    raw = np.asarray(values, dtype=float)
    result = np.full(len(raw), np.nan)
    order = np.argsort(raw)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float(raw[index]) * (len(order) - rank)))
        result[index] = running
    return result.tolist()


def analyze(namespace: Path) -> dict[str, object]:
    result_rows: list[dict[str, object]] = []
    global_rows: list[dict[str, object]] = []
    for k in K_VALUES:
        root = namespace / f"run_20260805_gemma_full_span_residual_k{k}"
        path = root / "analysis" / "realistic_niah_v4_4_4_residual_analysis.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        decision = payload["primary_decision"]
        global_rows.append(
            {
                "top_n": k,
                "selected_mediator_layer": int(payload["selected_mediator_layer"]),
                "full_residual_count_path_support": bool(
                    decision["full_residual_count_path_support"]
                ),
                "global_intersection_union_p": float(
                    decision["global_intersection_union_p"]
                ),
                "audit_all_checks_pass": bool(payload["audit"]["all_checks_pass"]),
                "run_root": str(root),
            }
        )
        for family, family_payload in decision["families"].items():
            for component in family_payload["components"]:
                result_rows.append(
                    {
                        "top_n": k,
                        "selected_mediator_layer": int(payload["selected_mediator_layer"]),
                        "family": family,
                        "role": component["role"],
                        "mean": float(component["mean"]),
                        "ci95_low": float(component["ci95_low"]),
                        "ci95_high": float(component["ci95_high"]),
                        "p": float(component["p"]),
                        "passes_ci": bool(component["passes_ci"]),
                        "passes_p": bool(component["passes_p"]),
                    }
                )
    global_frame = pd.DataFrame(global_rows).sort_values("top_n").reset_index(drop=True)
    global_frame["global_iut_holm_across_five_k"] = holm(
        global_frame["global_intersection_union_p"].tolist()
    )
    global_frame["supports_after_k_holm_0_025"] = (
        global_frame["full_residual_count_path_support"]
        & (global_frame["global_iut_holm_across_five_k"] <= 0.025)
    )
    endpoint_frame = pd.DataFrame(result_rows).sort_values(
        ["top_n", "family", "role"]
    )
    global_path = namespace / "full_span_residual_k_sweep_global.csv"
    endpoint_path = namespace / "full_span_residual_k_sweep_endpoints.csv"
    global_frame.to_csv(global_path, index=False)
    endpoint_frame.to_csv(endpoint_path, index=False)
    summary = {
        "schema_version": "realistic_niah_v4_4_4_gemma_full_span_residual_k_sweep_analysis_v1",
        "status": "complete",
        "top_ns": list(K_VALUES),
        "ranking": "hash-locked full-span broad-aggregation prefix with layer < 36 temporal filter",
        "multiplicity": "Holm across five K-level global intersection-union p-values",
        "global": global_frame.to_dict(orient="records"),
        "outputs": {"global": str(global_path), "endpoints": str(endpoint_path)},
    }
    summary_path = namespace / "full_span_residual_k_sweep_analysis.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(Path(args.namespace).resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
