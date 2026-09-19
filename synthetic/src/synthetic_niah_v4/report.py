from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def _table_md(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "_No rows produced._"
    return df.head(max_rows).to_string(index=False)


def _img(path: str) -> str:
    return f'<p><img src="{html.escape(path)}" style="max-width: 900px; width: 100%;"></p>'


def steering_conclusion(steering: pd.DataFrame, patching: pd.DataFrame) -> str:
    if not steering.empty:
        main = steering[steering["control_type"].eq("none")]
        if not main.empty:
            best = main["monotonicity_score"].abs().max()
            shift = main["mean_count_shift"].abs().max()
            if pd.notna(best) and best >= 0.7 and shift >= 0.25:
                return "Strong steering: a tested direction produced a monotonic count shift under the debug grid."
    if not patching.empty and patching["patched_moves_toward_donor"].astype(bool).mean() > 0.5:
        return "Patching but weak steering: interchange patching moved predictions more consistently than linear steering."
    return "Probe-only: count information may be linearly decodable, but causal steering/patching remains weak under these interventions."


def generate_report(run_dir: Path, cfg: object | None = None) -> tuple[Path, Path]:
    tables = run_dir / "tables"
    behavior = _read(tables / "behavior_eval.csv")
    probes = _read(tables / "probe_results.csv")
    residual = _read(tables / "probe_residualized.csv")
    directions = _read(tables / "direction_metrics.csv")
    steering = _read(tables / "steering_results.csv")
    patching = _read(tables / "interchange_patching_results.csv")
    conclusion = steering_conclusion(steering, patching)
    cfg_text = json.dumps(cfg.to_dict(), indent=2) if cfg is not None and hasattr(cfg, "to_dict") else "{}"
    md = f"""# Synthetic NIAH Counting v4 Report

## Setup

v4 uses the v2-style symbolic marker-trace task with HuggingFace `GPT2LMHeadModel`, random initialization, and learned absolute positional embeddings.

```json
{cfg_text}
```

## Behavioral Accuracy

{_table_md(behavior.groupby(["model_type", "eval_mode"], as_index=False)["final_accuracy"].mean() if not behavior.empty else behavior)}

## Probe Results

{_table_md(probes)}

## Residualized Probe Results

{_table_md(residual)}

## Direction Diagnostics

{_table_md(directions)}

## Steering

{_table_md(steering)}

## Interchange Patching

{_table_md(patching)}

## Conclusion

{conclusion}

Cautious interpretation: probe scores indicate linearly decodable count information only. The phrase count-vector is warranted only when monotonic steering or patching succeeds.
"""
    report_md = run_dir / "report.md"
    report_html = run_dir / "report.html"
    report_md.write_text(md, encoding="utf-8")
    sections = html.escape(md).replace("\n", "<br>\n")
    figures = "\n".join(
        _img(f"figures/{name}")
        for name in [
            "probe_acc_by_layer_anchor.png",
            "probe_r2_by_layer_anchor.png",
            "probe_minus_baseline_heatmap.png",
            "direction_cosine_heatmap.png",
            "projection_by_count.png",
            "steering_heatmap_anchor_layer.png",
            "steering_dose_response_top_configs.png",
            "steering_controls.png",
            "interchange_patch_matrix.png",
            "input_geometry_projection_trajectories.png",
        ]
    )
    report_html.write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>v4 report</title></head><body><pre>{sections}</pre>{figures}</body></html>",
        encoding="utf-8",
    )
    return report_html, report_md
