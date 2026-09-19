#!/usr/bin/env python3
"""Build the Native-thinking mechanism report.

The report follows the section grammar of the Non-thinking report while keeping
Native-specific causal edges and unresolved boundaries explicit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from build_v5_native_thinking_report_final import (
    answer_source_rerouting_svg,
    attention_example_switcher,
    head_map_svg,
    load_attention_examples,
    load_representation,
    load_token_ablation_evidence,
    token_source_ablation_svg,
)


MODELS = ("Qwen3-8B", "Gemma4-E4B")
SHORT = {"Qwen3-8B": "Qwen", "Gemma4-E4B": "Gemma"}
COLORS = {"Qwen3-8B": "#0f766e", "Gemma4-E4B": "#7c3aed"}
ANSWER_QUERY_LAYER_GRID = {
    "Qwen3-8B": [0, 5, 10, 15, 20, 25, 30, 35],
    "Gemma4-E4B": [0, 6, 12, 18, 23, 29, 35, 41],
}
ANSWER_QUERY_PAIR_COUNT = {"Qwen3-8B": 40, "Gemma4-E4B": 40}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _csv_truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _seed_equal_csv_mean(
    rows: Sequence[Mapping[str, Any]], column: str
) -> float:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        raw = row.get(column)
        if raw in {None, ""}:
            continue
        text = str(raw).strip().lower()
        value = float(_csv_truth(raw)) if text in {"true", "false"} else float(raw)
        grouped.setdefault(int(row["seed"]), []).append(value)
    require(grouped, f"No confirmation rows for {column}")
    return float(
        np.mean([float(np.mean(values)) for values in grouped.values()])
    )


def replace_with_aligned_targeting(
    evidence: dict[str, Any], aligned_run_root: Path, model: str
) -> dict[str, Any]:
    """Replace the legacy asymmetric targeting panel with exact-common support."""

    analysis = aligned_run_root / model / "next_city_token_blank" / "analysis"
    audit = read_json(analysis / "analysis_audit.json")
    require(audit.get("status") == "PASS", f"{model} aligned token audit failed")
    confirmation_seeds = set(range(1254, 1264))
    detail = [
        row
        for row in read_csv(analysis / "token_level_detail.csv")
        if int(row["seed"]) in confirmation_seeds and row.get("status") == "ok"
    ]
    matched = [
        row
        for row in read_csv(analysis / "matched_control_specificity.csv")
        if int(row["seed"]) in confirmation_seeds
    ]
    direct_conditions = (
        "clean",
        "early_half_trace_blank",
        "cumulative_trace_blank",
        "recent_transition_blank",
        "full_trace_blank",
    )
    retrieval_metric = (
        "target_city_retrieved"
        if detail and "target_city_retrieved" in detail[0]
        else "target_city_teacher_forced_exact"
    )
    require(
        detail and retrieval_metric in detail[0],
        f"{model} aligned targeting lacks a registered exact-city endpoint",
    )
    specificity_retrieval_metric = f"specificity__{retrieval_metric}"
    require(
        matched and specificity_retrieval_metric in matched[0],
        f"{model} aligned targeting lacks matched-control exact-city specificity",
    )
    targeting: dict[str, dict[str, Any]] = {}
    for condition in direct_conditions:
        rows = [row for row in detail if row["condition"] == condition]
        require(
            len(rows) == 10 and {int(row["seed"]) for row in rows} == confirmation_seeds,
            f"{model} aligned targeting {condition} is not exact 10-seed support",
        )
        targeting[condition] = {
            "rows": len(rows),
            "retrieved": sum(_csv_truth(row[retrieval_metric]) for row in rows),
            "retrieval_rate": _seed_equal_csv_mean(rows, retrieval_metric),
            "target_share": _seed_equal_csv_mean(
                rows, "bank_target_attention_share_of_gold_mass"
            ),
            "target_top1": _seed_equal_csv_mean(rows, "bank_target_top1_fraction"),
            "target_city_logp": _seed_equal_csv_mean(
                rows, "target_city_log_probability"
            ),
        }
    specificity: dict[str, dict[str, float]] = {}
    for condition in direct_conditions[1:]:
        rows = [row for row in matched if row["treatment"] == condition]
        require(
            {int(row["seed"]) for row in rows} == confirmation_seeds,
            f"{model} aligned matched control {condition} lost a seed",
        )
        specificity[condition] = {
            "retrieval": _seed_equal_csv_mean(
                rows, specificity_retrieval_metric
            ),
            "target_share": _seed_equal_csv_mean(
                rows,
                "specificity__bank_target_attention_share_of_gold_mass",
            ),
            "target_city_logp": _seed_equal_csv_mean(
                rows, "specificity__target_city_log_probability"
            ),
        }
    evidence = dict(evidence)
    evidence["target_registered_seed_count"] = 10
    evidence["target_request_count"] = 10
    evidence["target_exact_endpoint"] = retrieval_metric
    evidence["targeting"] = targeting
    evidence["specificity"] = specificity
    evidence["input_files"] = [
        *evidence["input_files"],
        analysis / "analysis_audit.json",
        analysis / "token_level_detail.csv",
        analysis / "matched_control_specificity.csv",
    ]
    return evidence


def load_all_layer_native_geometry(
    comparison_report: Path,
    running_defaults: Mapping[str, int],
    final_defaults: Mapping[str, int],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Extract all-layer native PCA3 coordinates from the audited geometry report.

    That report fits StandardScaler + PCA3 separately at every layer using only
    discovery states.  We retain confirmation rows only for the mechanism report.
    """

    prefix = "const DUAL="
    dual: dict[str, Any] | None = None
    with comparison_report.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(prefix):
                require(line.rstrip().endswith(";"), "Malformed DUAL geometry payload")
                dual = json.loads(line.rstrip()[len(prefix) : -1])
                break
    require(dual is not None, "Geometry comparison report lacks DUAL payload")

    result: dict[str, dict[str, dict[str, Any]]] = {
        "running": {},
        "final": {},
    }
    for model in MODELS:
        for endpoint, panel_name, defaults in (
            ("running", "running_native", running_defaults),
            ("final", "final_native", final_defaults),
        ):
            panel = dual[model]["panels"][panel_name]
            layer_payload: dict[str, Any] = {}
            discovery_rows = confirmation_rows = None
            for layer_text, block in panel["coordinates"].items():
                source_rows = list(block["points"])
                current_discovery = sum(row[0] == "discovery" for row in source_rows)
                current_confirmation = sum(row[0] == "confirmation" for row in source_rows)
                if discovery_rows is None:
                    discovery_rows = current_discovery
                    confirmation_rows = current_confirmation
                require(
                    (current_discovery, current_confirmation)
                    == (discovery_rows, confirmation_rows),
                    f"{model}/{endpoint}: layerwise state support changed",
                )
                confirmation = [row for row in source_rows if row[0] == "confirmation"]
                rows = [
                    [
                        int(row[1]),
                        int(row[2] if endpoint == "running" else row[6]),
                        float(row[3]),
                        float(row[4]),
                        float(row[5]),
                    ]
                    for row in confirmation
                ]
                require(rows, f"{model}/{endpoint}/L{layer_text}: no confirmation rows")
                layer_payload[str(int(layer_text))] = {
                    "evr": [float(value) for value in block["evr"]],
                    "rows": rows,
                }
            layers = sorted(int(layer) for layer in layer_payload)
            default_layer = int(defaults[model])
            require(
                default_layer in layers,
                f"{model}/{endpoint}: frozen display L{default_layer} unavailable",
            )
            result[endpoint][model] = {
                "default_layer": default_layer,
                "layers": layer_payload,
                "discovery_rows": int(discovery_rows or 0),
                "confirmation_rows": int(confirmation_rows or 0),
                "token_site": str(panel["token_site"]),
            }
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def inline_standalone_svg(path: Path) -> str:
    """Embed a generated SVG and make its namespace explicit for HTML delivery."""
    markup = path.read_text(encoding="utf-8")
    svg_start = markup.find("<svg")
    if svg_start < 0:
        raise ValueError(f"Expected SVG root in {path}")
    # Matplotlib exports include an XML declaration and a DOCTYPE.  Those are
    # valid in a standalone file but invalid when nested inside an HTML body.
    markup = markup[svg_start:]
    markup = "\n".join(line.rstrip() for line in markup.splitlines())
    if 'xmlns="http://www.w3.org/2000/svg"' not in markup[:512]:
        markup = markup.replace(
            "<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1
        )
    return markup


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def effect(rows: Iterable[Mapping[str, Any]], estimand: str) -> Mapping[str, Any]:
    matches = [row for row in rows if row.get("estimand") == estimand]
    require(len(matches) == 1, f"Expected one estimand {estimand}, got {len(matches)}")
    return matches[0]


def ci(row: Mapping[str, Any], digits: int = 3) -> str:
    return (
        f"{float(row['mean_effect']):+.{digits}f} "
        f"[{float(row['ci_low']):+.{digits}f}, {float(row['ci_high']):+.{digits}f}]"
    )


def table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    class_name: str = "",
) -> str:
    head = "".join(f"<th>{esc(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in rows
    )
    classes = "table-wrap" + (f" {class_name}" if class_name else "")
    return f'<div class="{classes}"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def extract_css(path: Path) -> str:
    match = re.search(r"<style>(.*?)</style>", path.read_text(encoding="utf-8"), re.S)
    require(match is not None, f"No CSS block in {path}")
    return match.group(1)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def selected_row(analysis: Mapping[str, Any], k: int) -> Mapping[str, Any]:
    rows = [
        row
        for row in analysis["overall"]
        if row["scope"] == "all_registered_grammars" and int(row["bank_size"]) == k
    ]
    require(len(rows) == 1, f"Missing all-grammar K={k} row")
    return rows[0]


def load_p0_head_score_rows(
    atlas_root: Path, model: str, *, grammar: str = "all"
) -> list[dict[str, str]]:
    """Adapt the frozen P0 ranking CSV to the report's layer-by-head renderer."""

    rows = [
        row
        for row in read_csv(atlas_root / "p0_targeted_retrieval_head_scores.csv")
        if row["model_label"] == model and row["grammar"] == grammar
    ]
    require(rows, f"Missing P0 head scores for {model}/{grammar}")
    return [
        {
            **row,
            "discovery_rank": row["rank"],
            "discovery_selection_value": row["score"],
        }
        for row in rows
    ]


def load_layerwise_representation(
    causal_root: Path, dual_root: Path
) -> dict[str, dict[str, list[dict[str, str]]]]:
    """Load frozen held-out probe curves without performing layer selection."""

    running_rows = read_csv(causal_root / "site_layer_candidates.csv")
    result: dict[str, dict[str, list[dict[str, str]]]] = {}
    for model in MODELS:
        running = [
            row
            for row in running_rows
            if row["model_label"] == model and row["site_kind"] == "item_end"
        ]
        final = [
            row
            for row in read_csv(
                dual_root
                / model
                / "pca16_whiten"
                / "final_count_candidate_metrics.csv"
            )
            if row["model_label"] == model
            and row["mode"] == "native_thinking"
            and row["endpoint"] == "final_count"
        ]
        running.sort(key=lambda row: int(row["layer"]))
        final.sort(key=lambda row: int(row["layer"]))
        require(running and final, f"Missing layerwise representation rows for {model}")
        result[model] = {"running": running, "final": final}
    return result


def layerwise_representation_svg(
    curves: Mapping[str, Mapping[str, Sequence[Mapping[str, str]]]],
    selected: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> str:
    """Four-panel probe plot aligned with the Non-thinking report grammar."""

    width, height = 1040, 620
    colors = {"logistic": "#0f766e", "ncc": "#d97706"}
    parts = [
        f'<svg class="paper-chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="layer-probe-title layer-probe-desc">',
        '<title id="layer-probe-title">Layerwise Native-thinking count decodability</title>',
        '<desc id="layer-probe-desc">Four panels show held-out balanced accuracy by decoder layer for running item-end and final answer-query states.</desc>',
    ]
    for row_index, endpoint in enumerate(("running", "final")):
        for col_index, model in enumerate(MODELS):
            rows = curves[model][endpoint]
            ox, oy = 58 + col_index * 500, 48 + row_index * 292
            plot_w, plot_h = 416, 190
            max_layer = max(int(row["layer"]) for row in rows)

            def sx(layer: int) -> float:
                return ox + layer / max(max_layer, 1) * plot_w

            def sy(value: float) -> float:
                return oy + (1.0 - value) * plot_h

            title = (
                "running commit · item_end"
                if endpoint == "running"
                else "final count · answer query"
            )
            parts.append(
                f'<text x="{ox}" y="{oy-18}" class="heat-title">'
                f'{esc(SHORT[model])} · {esc(title)}</text>'
            )
            parts.append(
                f'<rect x="{ox}" y="{oy}" width="{plot_w}" height="{plot_h}" class="plot-bg"/>'
            )
            for tick in (0.1, 0.4, 0.7, 1.0):
                y = sy(tick)
                parts.append(
                    f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+plot_w}" y2="{y:.1f}" class="grid"/>'
                )
                parts.append(
                    f'<text x="{ox-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:.1f}</text>'
                )
            chance_y = sy(0.1)
            parts.append(
                f'<line x1="{ox}" y1="{chance_y:.1f}" x2="{ox+plot_w}" y2="{chance_y:.1f}" '
                'stroke="#667085" stroke-dasharray="5 4"/>'
            )
            for metric, key in (
                ("logistic", "confirmation_logistic_balanced_accuracy"),
                ("ncc", "confirmation_ncc_balanced_accuracy"),
            ):
                points = " ".join(
                    f'{sx(int(row["layer"])):.1f},{sy(float(row[key])):.1f}'
                    for row in rows
                )
                parts.append(
                    f'<polyline points="{points}" class="series-line" stroke="{colors[metric]}"/>'
                )
            selected_layer = int(selected[model][endpoint]["layer"])
            selected_value = float(
                selected[model][endpoint]["confirmation_logistic_balanced_accuracy"]
            )
            parts.append(
                f'<circle cx="{sx(selected_layer):.1f}" cy="{sy(selected_value):.1f}" r="6" '
                'fill="#fff" stroke="#0f766e" stroke-width="3"/>'
            )
            parts.append(
                f'<text x="{sx(selected_layer)+8:.1f}" y="{sy(selected_value)-8:.1f}" class="chart-value">'
                f'L{selected_layer} · {selected_value:.2f}</text>'
            )
            for layer_tick in sorted({0, max_layer // 2, max_layer}):
                parts.append(
                    f'<text x="{sx(layer_tick):.1f}" y="{oy+plot_h+19}" text-anchor="middle" class="tick">{layer_tick}</text>'
                )
            if col_index == 0:
                parts.append(
                    f'<text transform="translate({ox-42} {oy+plot_h/2}) rotate(-90)" '
                    'text-anchor="middle" class="axis-label">confirmation balanced accuracy</text>'
                )
            parts.append(
                f'<text x="{ox+plot_w/2}" y="{oy+plot_h+38}" text-anchor="middle" class="axis-label">zero-based post-block layer</text>'
            )
    parts.extend(
        [
            '<line x1="372" y1="603" x2="398" y2="603" stroke="#0f766e" stroke-width="3"/>',
            '<text x="405" y="607" class="legend-label">L2 logistic</text>',
            '<line x1="500" y1="603" x2="526" y2="603" stroke="#d97706" stroke-width="3"/>',
            '<text x="533" y="607" class="legend-label">nearest centroid</text>',
            '<line x1="678" y1="603" x2="704" y2="603" stroke="#667085" stroke-dasharray="5 4"/>',
            '<text x="711" y="607" class="legend-label">10-class chance</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def load_running_manifold_points(
    band_root: Path,
) -> dict[str, list[dict[str, float]]]:
    result: dict[str, list[dict[str, float]]] = {}
    for model in MODELS:
        rows = read_csv(band_root / model / "confirmation_points.csv")
        points = [
            {
                "seed": float(row["seed"]),
                "label": float(row["occurrence"]),
                "x": float(row["pc1"]),
                "y": float(row["pc2"]),
                "z": float(row["pc3"]),
            }
            for row in rows
        ]
        require(points, f"Missing running-manifold points for {model}")
        result[model] = points
    return result


def load_final_manifold_points(
    capture_root: Path,
    representation: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[dict[str, list[dict[str, float]]], dict[str, dict[str, Any]]]:
    """Fit PCA on discovery answer-query states and project confirmation only."""

    result: dict[str, list[dict[str, float]]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        root = capture_root / "final" / model
        index = read_jsonl(root / "capture_index.jsonl")
        layer = int(representation[model]["final"]["layer"])
        discovery_states: list[np.ndarray] = []
        discovery_labels: list[float] = []
        confirmation_states: list[np.ndarray] = []
        confirmation_labels: list[float] = []
        confirmation_seeds: list[float] = []
        for row in index:
            archive = np.load(root / str(row["states_path"]))
            layers = archive["layer_indices"].astype(int)
            matches = np.flatnonzero(layers == layer)
            require(len(matches) == 1, f"{model} final capture misses L{layer}")
            states = archive["site_states"]
            require(states.shape[0] == 1, f"{model} final capture has multiple answer sites")
            vector = states[0, int(matches[0])].astype(np.float32)
            label = float(row["gold_count"])
            if row["split"] == "discovery":
                discovery_states.append(vector)
                discovery_labels.append(label)
            elif row["split"] == "confirmation":
                confirmation_states.append(vector)
                confirmation_labels.append(label)
                confirmation_seeds.append(float(row["seed"]))
        require(
            len(discovery_states) == 200 and len(confirmation_states) == 100,
            f"{model} final PCA split changed",
        )
        train = np.stack(discovery_states)
        test = np.stack(confirmation_states)
        center = train.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(train - center, full_matrices=False)
        components = vt[:3]
        train_coords = (train - center) @ components.T
        test_coords = (test - center) @ components.T
        labels = np.asarray(discovery_labels, dtype=np.float32)
        for axis in range(3):
            corr = np.corrcoef(train_coords[:, axis], labels)[0, 1]
            if np.isfinite(corr) and corr < 0:
                test_coords[:, axis] *= -1
        result[model] = [
            {
                "seed": seed,
                "label": label,
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
            }
            for seed, label, point in zip(
                confirmation_seeds, confirmation_labels, test_coords
            )
        ]
        variance_ratio = (_s[:3] ** 2) / max(float(np.sum(_s ** 2)), 1e-12)
        diagnostics[model] = {
            "discovery_rows": len(discovery_states),
            "confirmation_rows": len(confirmation_states),
            "explained_variance_ratio": [float(value) for value in variance_ratio],
            "explained_variance_ratio_sum": float(np.sum(variance_ratio)),
        }
    return result, diagnostics


def manifold_payload(
    running: Mapping[str, Sequence[Mapping[str, float]]],
    final: Mapping[str, Sequence[Mapping[str, float]]],
    running_layers: Mapping[str, int],
    final_layers: Mapping[str, int],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Serialize the two frozen display manifolds for the in-report 3D viewer."""

    payload: dict[str, dict[str, dict[str, Any]]] = {}
    for endpoint, source, layers in (
        ("running", running, running_layers),
        ("final", final, final_layers),
    ):
        payload[endpoint] = {}
        for model in MODELS:
            rows = [
                [
                    int(point["seed"]),
                    int(round(point["label"])),
                    round(float(point["x"]), 6),
                    round(float(point["y"]), 6),
                    round(float(point["z"]), 6),
                ]
                for point in source[model]
            ]
            require(rows, f"Missing {endpoint} 3D manifold points for {model}")
            payload[endpoint][model] = {
                "layer": int(layers[model]),
                "rows": rows,
            }
    return payload


def representation_manifold_svg(
    running: Mapping[str, Sequence[Mapping[str, float]]],
    final: Mapping[str, Sequence[Mapping[str, float]]],
) -> str:
    """Four-panel confirmation PCA plot with discovery-frozen axes."""

    palette = (
        "#0b4f6c", "#176b87", "#228b8d", "#3aa17e", "#66b56b",
        "#9abe55", "#d0bb42", "#e69b35", "#df6f32", "#c9413a",
    )
    width, height = 1040, 690
    parts = [
        f'<svg class="paper-chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="manifold-title manifold-desc">',
        '<title id="manifold-title">Native-thinking running and final count manifolds</title>',
        '<desc id="manifold-desc">Confirmation hidden states projected onto discovery-fitted first two principal components, colored by occurrence or final count.</desc>',
    ]
    panels = (("running", running), ("final", final))
    for row_index, (endpoint, source) in enumerate(panels):
        for col_index, model in enumerate(MODELS):
            points = list(source[model])
            ox, oy = 58 + col_index * 500, 48 + row_index * 292
            plot_w, plot_h = 416, 208
            xs = np.asarray([point["x"] for point in points])
            ys = np.asarray([point["y"] for point in points])
            x_pad = max(float(np.ptp(xs)) * 0.06, 1e-6)
            y_pad = max(float(np.ptp(ys)) * 0.06, 1e-6)
            x_min, x_max = float(xs.min() - x_pad), float(xs.max() + x_pad)
            y_min, y_max = float(ys.min() - y_pad), float(ys.max() + y_pad)

            def sx(value: float) -> float:
                return ox + (value - x_min) / max(x_max - x_min, 1e-9) * plot_w

            def sy(value: float) -> float:
                return oy + (y_max - value) / max(y_max - y_min, 1e-9) * plot_h

            layer_note = "display PCA" if endpoint == "running" else "answer-query PCA"
            parts.append(
                f'<text x="{ox}" y="{oy-18}" class="heat-title">{esc(SHORT[model])} · '
                f'{"running occurrence" if endpoint == "running" else "final gold count"} · {layer_note}</text>'
            )
            parts.append(
                f'<rect x="{ox}" y="{oy}" width="{plot_w}" height="{plot_h}" class="plot-bg"/>'
            )
            for point in points:
                label = max(1, min(10, int(round(point["label"]))))
                parts.append(
                    f'<circle cx="{sx(point["x"]):.1f}" cy="{sy(point["y"]):.1f}" r="3" '
                    f'fill="{palette[label-1]}" opacity=".32"/>'
                )
            centroids: list[tuple[int, float, float]] = []
            for label in range(1, 11):
                group = [point for point in points if int(round(point["label"])) == label]
                if not group:
                    continue
                cx = float(np.mean([point["x"] for point in group]))
                cy = float(np.mean([point["y"] for point in group]))
                centroids.append((label, cx, cy))
            polyline = " ".join(f"{sx(cx):.1f},{sy(cy):.1f}" for _, cx, cy in centroids)
            parts.append(
                f'<polyline points="{polyline}" fill="none" stroke="#344054" stroke-width="1.8" opacity=".7"/>'
            )
            for label, cx, cy in centroids:
                parts.append(
                    f'<circle cx="{sx(cx):.1f}" cy="{sy(cy):.1f}" r="7" fill="{palette[label-1]}" '
                    'stroke="#fff" stroke-width="1.5"/>'
                )
                parts.append(
                    f'<text x="{sx(cx):.1f}" y="{sy(cy)+3.2:.1f}" text-anchor="middle" '
                    'font-size="7.5" font-weight="800" fill="#fff">'
                    f'{label}</text>'
                )
            parts.append(
                f'<text x="{ox+plot_w/2}" y="{oy+plot_h+30}" text-anchor="middle" class="axis-label">PC1 (arbitrary PCA units)</text>'
            )
            parts.append(
                f'<text transform="translate({ox-38} {oy+plot_h/2}) rotate(-90)" text-anchor="middle" '
                'class="axis-label">PC2 (arbitrary PCA units)</text>'
            )
    for label in range(1, 11):
        x = 245 + (label - 1) * 58
        parts.append(f'<circle cx="{x}" cy="656" r="6" fill="{palette[label-1]}"/>')
        parts.append(f'<text x="{x+10}" y="660" class="legend-label">{label}</text>')
    parts.append('<text x="190" y="660" text-anchor="end" class="legend-label">occurrence / count</text>')
    parts.append("</svg>")
    return "".join(parts)


def split_gemma_current_head_map_svg(rows: Sequence[Mapping[str, str]]) -> str:
    """Render Gemma's tall 42x8 atlas as two large, legible layer panels."""

    layers = max(int(row["layer"]) for row in rows) + 1
    heads = max(int(row["head"]) for row in rows) + 1
    require((layers, heads) == (42, 8), "Unexpected Gemma layer/head geometry")
    by_head = {
        (int(row["layer"]), int(row["head"])): row for row in rows
    }
    scores = [max(0.0, float(row["discovery_selection_value"])) for row in rows]
    cap = max(float(np.quantile(scores, 0.99)), 1e-12)

    def color(value: float) -> str:
        t = max(0.0, min(1.0, value)) ** 0.5
        low, high = (241, 245, 249), (15, 118, 110)
        rgb = tuple(round(a + (b - a) * t) for a, b in zip(low, high))
        return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

    width, height = 1040, 485
    x_starts = (82, 495)
    layer_ranges = ((0, 21), (21, 42))
    y0, cw, ch = 72, 38, 16
    parts = [
        f'<svg class="head-map gemma-split-map" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="Gemma4-E4B exact-P0 current Top-6 layer by head attention atlas">',
        '<title>Gemma4-E4B exact-P0 target-attention atlas, current Top-6</title>',
    ]
    for (layer_start, layer_end), x0 in zip(layer_ranges, x_starts):
        parts.append(
            f'<text x="{x0}" y="28" class="heat-title">Layers {layer_start}–{layer_end-1}</text>'
        )
        for head in range(heads):
            parts.append(
                f'<text x="{x0 + (head + .5) * cw:.1f}" y="58" '
                f'text-anchor="middle" class="heat-x">H{head}</text>'
            )
        for local_layer, layer in enumerate(range(layer_start, layer_end)):
            y = y0 + local_layer * ch
            parts.append(
                f'<text x="{x0-10}" y="{y+12}" text-anchor="end" '
                f'class="heat-row">L{layer}</text>'
            )
            for head in range(heads):
                row = by_head[(layer, head)]
                score = max(0.0, float(row["discovery_selection_value"]))
                rank = int(row["discovery_rank"])
                selected = rank <= 6
                parts.append(
                    f'<rect x="{x0 + head*cw}" y="{y}" width="{cw}" height="{ch}" '
                    f'fill="{color(min(score/cap, 1.0))}" '
                    f'stroke="{"#b42318" if selected else "#ffffff"}" '
                    f'stroke-width="{1.35 if selected else .35}">'
                    f'<title>L{layer}H{head}; seed-event mean target-record attention mass='
                    f'{score:.6g}; frozen rank={rank}; selected={str(selected).lower()}</title></rect>'
                )
        parts.append(
            f'<text x="{x0 + heads*cw/2}" y="{height-42}" text-anchor="middle" '
            'class="axis-label">Attention head h</text>'
        )
    legend_x, legend_y = 900, 86
    for step in range(40):
        t = step / 39
        parts.append(
            f'<rect x="{legend_x}" y="{legend_y + (39-step)*6}" width="20" '
            f'height="6" fill="{color(t)}"/>'
        )
    parts.extend(
        [
            f'<text x="{legend_x+28}" y="{legend_y+10}" class="heat-x">≥99th pct.</text>',
            f'<text x="{legend_x+28}" y="{legend_y+236}" class="heat-x">0</text>',
            '<text transform="translate(23 240) rotate(-90)" text-anchor="middle" '
            'class="axis-label">Transformer layer ℓ</text>',
            '<rect x="884" y="372" width="18" height="14" fill="#fff" stroke="#b42318" stroke-width="1.5"/>',
            '<text x="912" y="384" class="heat-x">frozen Top-6</text>',
            '</svg>',
        ]
    )
    return "".join(parts)


def point_cloud_script(payload: Mapping[str, Any]) -> str:
    """Return a dependency-free interactive PC1-PC3 canvas viewer."""

    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return "const NATIVE_GEOMETRY=" + encoded + r""";
const COUNT_COLORS=['#0b4f6c','#176b87','#228b8d','#3aa17e','#66b56b','#9abe55','#d0bb42','#e69b35','#df6f32','#c9413a'];
const NATIVE_CLOUDS=[];
class NativePointCloud3D {
  constructor(canvasId, modelId, layerId, statsId, endpoint) {
    this.canvas=document.getElementById(canvasId);
    this.ctx=this.canvas.getContext('2d');
    this.model=document.getElementById(modelId);
    this.layer=document.getElementById(layerId);
    this.stats=document.getElementById(statsId);
    this.endpoint=endpoint;
    this.yaw=-0.58;
    this.pitch=0.34;
    this.dragging=false;
    this.last=null;
    this.model.addEventListener('change',()=>this.setModel());
    this.layer.addEventListener('change',()=>this.setLayer());
    this.canvas.addEventListener('pointerdown',(event)=>this.pointerDown(event));
    this.canvas.addEventListener('pointermove',(event)=>this.pointerMove(event));
    this.canvas.addEventListener('pointerup',(event)=>this.pointerUp(event));
    this.canvas.addEventListener('pointercancel',(event)=>this.pointerUp(event));
    this.canvas.addEventListener('dblclick',()=>this.resetView());
    new ResizeObserver(()=>this.resize()).observe(this.canvas);
    NATIVE_CLOUDS.push(this);
    this.setModel();
  }
  setModel() {
    this.modelData=NATIVE_GEOMETRY[this.endpoint][this.model.value];
    this.layer.innerHTML='';
    Object.keys(this.modelData.layers).map(Number).sort((a,b)=>a-b).forEach(layer=>{
      const option=document.createElement('option');
      option.value=String(layer);
      option.textContent='L'+layer+(layer===this.modelData.default_layer?' · frozen default':'');
      this.layer.appendChild(option);
    });
    this.layer.value=String(this.modelData.default_layer);
    this.setLayer();
  }
  setLayer() {
    this.data=this.modelData.layers[this.layer.value];
    this.rows=this.data.rows;
    this.prepare();
    const evr=100*this.data.evr.reduce((sum,value)=>sum+value,0);
    this.stats.textContent=this.modelData.token_site+' · L'+this.layer.value+' · '+this.modelData.discovery_rows+' discovery states fit StandardScaler/PCA3 · '+this.rows.length+' confirmation states shown · EVR₁₋₃ '+evr.toFixed(1)+'%';
    this.draw();
  }
  resetView() { NATIVE_CLOUDS.forEach(cloud=>{cloud.yaw=-0.58;cloud.pitch=0.34;cloud.draw();}); }
  pointerDown(event) {
    this.dragging=true; this.last=[event.clientX,event.clientY];
    this.canvas.setPointerCapture(event.pointerId);
  }
  pointerMove(event) {
    if(!this.dragging) return;
    const dx=event.clientX-this.last[0], dy=event.clientY-this.last[1];
    this.last=[event.clientX,event.clientY];
    NATIVE_CLOUDS.forEach(cloud=>{cloud.yaw+=dx*0.009;cloud.pitch=Math.max(-1.35,Math.min(1.35,cloud.pitch+dy*0.009));cloud.draw();});
  }
  pointerUp(event) {
    this.dragging=false; this.last=null;
    if(this.canvas.hasPointerCapture(event.pointerId)) this.canvas.releasePointerCapture(event.pointerId);
  }
  resize() {
    const rect=this.canvas.getBoundingClientRect();
    const dpr=Math.min(window.devicePixelRatio||1,2);
    const width=Math.max(320,Math.round(rect.width));
    const height=Math.max(400,Math.round(rect.height));
    if(this.canvas.width!==Math.round(width*dpr)||this.canvas.height!==Math.round(height*dpr)) {
      this.canvas.width=Math.round(width*dpr); this.canvas.height=Math.round(height*dpr);
    }
    this.ctx.setTransform(dpr,0,0,dpr,0,0);
    this.cssWidth=width; this.cssHeight=height;
    this.draw();
  }
  prepare() {
    const coords=[0,1,2].map(axis=>this.rows.map(row=>row[axis+2]));
    this.centers=coords.map(values=>(Math.min(...values)+Math.max(...values))/2);
    this.scales=coords.map(values=>Math.max(Math.max(...values)-Math.min(...values),1e-8)/2);
    this.points=this.rows.map(row=>({seed:row[0],label:row[1],v:[0,1,2].map(axis=>(row[axis+2]-this.centers[axis])/this.scales[axis])}));
    this.centroids=[];
    for(let label=1;label<=10;label++) {
      const group=this.points.filter(point=>point.label===label);
      if(!group.length) continue;
      this.centroids.push({label:label,v:[0,1,2].map(axis=>group.reduce((sum,point)=>sum+point.v[axis],0)/group.length)});
    }
  }
  rotate(v) {
    const cy=Math.cos(this.yaw), sy=Math.sin(this.yaw), cp=Math.cos(this.pitch), sp=Math.sin(this.pitch);
    const x=cy*v[0]+sy*v[2], z=-sy*v[0]+cy*v[2];
    return [x,cp*v[1]-sp*z,sp*v[1]+cp*z];
  }
  project(v) {
    const r=this.rotate(v), depth=3.4-r[2]*0.34, scale=Math.min(this.cssWidth,510)*0.78/depth;
    return {x:this.cssWidth*0.5+r[0]*scale,y:this.cssHeight*0.45-r[1]*scale,z:r[2],s:scale};
  }
  line(a,b,color,width,dash) {
    const pa=this.project(a),pb=this.project(b),ctx=this.ctx;
    ctx.save();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash||[]);
    ctx.beginPath();ctx.moveTo(pa.x,pa.y);ctx.lineTo(pb.x,pb.y);ctx.stroke();ctx.restore();
  }
  draw() {
    if(!this.ctx||!this.rows||!this.cssWidth) return;
    const ctx=this.ctx,w=this.cssWidth,h=this.cssHeight;
    ctx.clearRect(0,0,w,h);ctx.fillStyle='#fbfcfe';ctx.fillRect(0,0,w,h);
    const axes=[[[0,0,0],[1.18,0,0],'PC1'],[[0,0,0],[0,1.18,0],'PC2'],[[0,0,0],[0,0,1.18],'PC3']];
    axes.forEach(axis=>{this.line(axis[0],axis[1],'#98a2b3',1.25,[]);const p=this.project(axis[1]);ctx.fillStyle='#475467';ctx.font='700 12px system-ui';ctx.fillText(axis[2],p.x+5,p.y-5);});
    const projected=this.points.map(point=>Object.assign({},point,this.project(point.v))).sort((a,b)=>a.z-b.z);
    projected.forEach(point=>{ctx.beginPath();ctx.arc(point.x,point.y,3.1,0,Math.PI*2);ctx.fillStyle=COUNT_COLORS[point.label-1]+'70';ctx.fill();});
    const centroids=this.centroids.map(point=>Object.assign({},point,this.project(point.v))).sort((a,b)=>a.label-b.label);
    ctx.save();ctx.strokeStyle='#344054';ctx.lineWidth=1.7;ctx.globalAlpha=.75;ctx.beginPath();
    centroids.forEach((point,index)=>{if(index===0)ctx.moveTo(point.x,point.y);else ctx.lineTo(point.x,point.y);});ctx.stroke();ctx.restore();
    centroids.slice().sort((a,b)=>a.z-b.z).forEach(point=>{ctx.beginPath();ctx.arc(point.x,point.y,10.5,0,Math.PI*2);ctx.fillStyle=COUNT_COLORS[point.label-1];ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1.8;ctx.stroke();ctx.fillStyle='#fff';ctx.font='800 10px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(point.label),point.x,point.y+.4);});
    ctx.textAlign='left';ctx.textBaseline='alphabetic';ctx.font='12px system-ui';
    const legendWidth=Math.min(620,w-36), startX=(w-legendWidth)/2, step=legendWidth/10;
    for(let label=1;label<=10;label++){const x=startX+(label-.5)*step;ctx.fillStyle=COUNT_COLORS[label-1];ctx.beginPath();ctx.arc(x,h-24,5,0,Math.PI*2);ctx.fill();ctx.fillStyle='#475467';ctx.fillText(String(label),x+9,h-20);}
    ctx.fillStyle='#667085';ctx.font='11px system-ui';ctx.fillText(this.points.length+' confirmation states · drag to rotate · double-click to reset',16,22);
  }
}
new NativePointCloud3D('native-running-canvas','native-geometry-model','native-running-layer','native-running-stats','running');
new NativePointCloud3D('native-final-canvas','native-geometry-model','native-final-layer','native-final-stats','final');
document.getElementById('native-geometry-reset').addEventListener('click',()=>NATIVE_CLOUDS[0].resetView());
"""


def effect_small_multiples_svg(
    title: str,
    panels: Sequence[
        tuple[str, str, Sequence[tuple[str, Mapping[str, Any]]], float]
    ],
) -> str:
    """Draw independently scaled mean/95%-CI panels for heterogeneous estimands."""

    require(1 <= len(panels) <= 4, "Small-multiple figure supports one to four panels")
    width = 1040
    max_rows = max(len(rows) for _, _, rows, _ in panels)
    four_row_two_panel = len(panels) <= 2 and max_rows >= 4
    height = 390 if four_row_two_panel else (330 if len(panels) <= 2 else 600)
    legend_y = height - 20
    panel_w = 472
    panel_h = 285 if four_row_two_panel else 238
    origins = ((58, 68), (550, 68), (58, 338), (550, 338))
    parts = [
        f'<svg class="paper-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(title)}">',
        f'<title>{esc(title)}</title>',
    ]
    for panel_index, (panel_title, unit, rows, floor) in enumerate(panels):
        x0, y0 = origins[panel_index]
        values = [
            abs(float(row[key]))
            for _, row in rows
            for key in ("mean_effect", "ci_low", "ci_high")
        ]
        limit = max(max(values, default=0.0) * 1.16, float(floor))
        plot_left, plot_right = x0 + 130, x0 + panel_w - 18
        center = (plot_left + plot_right) / 2
        half = (plot_right - plot_left) / 2
        last_row_y = y0 + 35 + (len(rows) - 1) * 42
        zero_line_bottom = max(y0 + 150, last_row_y + 10)

        def sx(value: float) -> float:
            return center + max(-limit, min(limit, value)) / limit * half

        parts.extend(
            [
                f'<rect x="{x0}" y="{y0-30}" width="{panel_w}" height="{panel_h}" fill="#fbfcfe" stroke="#d0d5dd"/>',
                f'<text x="{x0+14}" y="{y0-7}" class="heat-title">{esc(panel_title)}</text>',
                f'<line x1="{center:.1f}" y1="{y0+12}" x2="{center:.1f}" y2="{zero_line_bottom}" stroke="#98a2b3" stroke-dasharray="4 4"/>',
            ]
        )
        for row_index, (label, row) in enumerate(rows):
            y = y0 + 35 + row_index * 42
            mean = float(row["mean_effect"])
            low = float(row["ci_low"])
            high = float(row["ci_high"])
            gate = bool(row.get("gate_pass", row.get("positive_95pct_ci", low > 0)))
            point_color = "#0f766e" if gate else "#d97706"
            parts.extend(
                [
                    f'<text x="{plot_left-10}" y="{y+4}" text-anchor="end" class="chart-axis">{esc(label)}</text>',
                    f'<line x1="{sx(low):.1f}" y1="{y}" x2="{sx(high):.1f}" y2="{y}" stroke="#475467" stroke-width="2"/>',
                    f'<line x1="{sx(low):.1f}" y1="{y-5}" x2="{sx(low):.1f}" y2="{y+5}" stroke="#475467"/>',
                    f'<line x1="{sx(high):.1f}" y1="{y-5}" x2="{sx(high):.1f}" y2="{y+5}" stroke="#475467"/>',
                    f'<circle cx="{sx(mean):.1f}" cy="{y}" r="5" fill="{point_color}"><title>{esc(label)}: mean={mean:+.6g}; 95% CI [{low:+.6g}, {high:+.6g}]</title></circle>',
                    f'<text x="{plot_right}" y="{y-8}" text-anchor="end" class="chart-value">{mean:+.3g}</text>',
                ]
            )
        tick_y = max(y0 + 177, last_row_y + 44)
        parts.extend(
            [
                f'<text x="{plot_left}" y="{tick_y}" text-anchor="middle" class="chart-axis">−{limit:.3g}</text>',
                f'<text x="{center}" y="{tick_y}" text-anchor="middle" class="chart-axis">0</text>',
                f'<text x="{plot_right}" y="{tick_y}" text-anchor="middle" class="chart-axis">+{limit:.3g}</text>',
                f'<text x="{center}" y="{tick_y+20}" text-anchor="middle" class="axis-label">{esc(unit)} · panel-specific scale</text>',
            ]
        )
    parts.extend(
        [
            f'<circle cx="390" cy="{legend_y}" r="5" fill="#0f766e"/><text x="401" y="{legend_y+4}" class="chart-axis">registered CI entirely supportive</text>',
            f'<circle cx="650" cy="{legend_y}" r="5" fill="#d97706"/><text x="661" y="{legend_y+4}" class="chart-axis">gate not met / interval touches zero</text>',
            '</svg>',
        ]
    )
    return "".join(parts)


def failed_control_svg() -> str:
    """Explain why the first single-seed scrub was not an interpretable null."""

    return """<svg class="paper-chart" viewBox="0 0 1040 330" role="img" aria-label="V1 versus V2 context-scrub control coverage">
<title>Why the V1 single-seed scrub was a failed control</title>
<defs><marker id="audit-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#667085"/></marker></defs>
<text x="40" y="34" class="heat-title">V1 · parsed-item-only scrub</text><rect x="40" y="55" width="260" height="62" rx="5" fill="#eef2f6" stroke="#98a2b3"/><text x="170" y="82" text-anchor="middle" class="chain-title">Prompt records blanked</text><text x="170" y="103" text-anchor="middle" class="chain-sub">registered record spans</text><line x1="300" y1="86" x2="350" y2="86" stroke="#667085" marker-end="url(#audit-arrow)"/><rect x="355" y="55" width="260" height="62" rx="5" fill="#eef2f6" stroke="#98a2b3"/><text x="485" y="82" text-anchor="middle" class="chain-title">Parsed trace items blanked</text><text x="485" y="103" text-anchor="middle" class="chain-sub">parser-observed list only</text><line x1="615" y1="86" x2="665" y2="86" stroke="#667085" marker-end="url(#audit-arrow)"/><rect x="670" y="55" width="330" height="62" rx="5" fill="#fff3e0" stroke="#d97706" stroke-width="2"/><text x="835" y="82" text-anchor="middle" class="chain-title">Trace tail remained visible</text><text x="835" y="103" text-anchor="middle" class="chain-sub">possible final-count leakage → null uninterpretable</text>
<text x="40" y="181" class="heat-title">V2 · complete source scrub</text><rect x="40" y="202" width="260" height="62" rx="5" fill="#e8f5f2" stroke="#0f766e"/><text x="170" y="229" text-anchor="middle" class="chain-title">Prompt records blanked</text><text x="170" y="250" text-anchor="middle" class="chain-sub">same-length ordinary tokens</text><line x1="300" y1="233" x2="350" y2="233" stroke="#667085" marker-end="url(#audit-arrow)"/><rect x="355" y="202" width="260" height="62" rx="5" fill="#e8f5f2" stroke="#0f766e"/><text x="485" y="229" text-anchor="middle" class="chain-title">Entire trace context blanked</text><text x="485" y="250" text-anchor="middle" class="chain-sub">items + tail + residual text</text><line x1="615" y1="233" x2="665" y2="233" stroke="#667085" marker-end="url(#audit-arrow)"/><rect x="670" y="202" width="330" height="62" rx="5" fill="#e8f5f2" stroke="#0f766e" stroke-width="2"/><text x="835" y="229" text-anchor="middle" class="chain-title">Uninformative baseline verified</text><text x="835" y="250" text-anchor="middle" class="chain-sub">single-item restoration can now be interpreted</text>
<text x="520" y="307" text-anchor="middle" class="axis-label">Only V2 enters the report's descriptive single-seed conclusion</text></svg>"""


def line_chart(
    panels: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> str:
    width, panel_w, height = 980, 410, 310
    parts = [
        f'<svg class="line-chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Selected and random targeted retrieval ablation dose response">'
    ]
    for panel_index, (label, rows) in enumerate(panels):
        x0 = 72 + panel_index * 490
        y0, plot_w, plot_h = 52, panel_w, 190
        ks = [int(row["bank_size"]) for row in rows]
        k_min, k_max = min(ks), max(ks)

        def sx(k: int) -> float:
            return x0 + (k - k_min) / max(k_max - k_min, 1) * plot_w

        def sy(value: float) -> float:
            return y0 + (1.0 - value) * plot_h

        parts.append(f'<text x="{x0}" y="25" class="heat-title">{esc(label)}</text>')
        parts.append(f'<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" class="plot-bg"/>')
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = sy(tick)
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+plot_w}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:.2f}</text>')
        for k in ks:
            x = sx(k)
            parts.append(f'<text x="{x:.1f}" y="{y0+plot_h+20}" text-anchor="middle" class="tick">{k}</text>')
        selected = " ".join(
            f"{sx(int(row['bank_size'])):.1f},{sy(float(row['selected_failure_rate'])):.1f}"
            for row in rows
        )
        random = " ".join(
            f"{sx(int(row['bank_size'])):.1f},{sy(float(row['random_failure_rate'])):.1f}"
            for row in rows
        )
        parts.append(f'<polyline points="{selected}" class="series-line" stroke="#0f766e"/>')
        parts.append(f'<polyline points="{random}" class="series-line" stroke="#98a2b3"/>')
        for row in rows:
            x = sx(int(row["bank_size"]))
            for key, color in (("selected_failure_rate", "#0f766e"), ("random_failure_rate", "#98a2b3")):
                parts.append(f'<circle cx="{x:.1f}" cy="{sy(float(row[key])):.1f}" r="4" fill="{color}" class="series-dot"/>')
        parts.append(f'<text x="{x0+plot_w/2}" y="{height-24}" text-anchor="middle" class="axis-label">Frozen bank size K</text>')
    parts.extend(
        [
            '<line x1="375" y1="280" x2="401" y2="280" stroke="#0f766e" stroke-width="3"/>',
            '<text x="407" y="284" class="legend-label">selected</text>',
            '<line x1="485" y1="280" x2="511" y2="280" stroke="#98a2b3" stroke-width="3"/>',
            '<text x="517" y="284" class="legend-label">layer-matched random</text>',
            '<text transform="translate(18 150) rotate(-90)" text-anchor="middle" class="axis-label">Retrieval failure rate</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def grouped_bars_svg(
    title: str,
    groups: Sequence[tuple[str, Mapping[str, float]]],
    *,
    maximum: float | None = None,
    suffix: str = "",
) -> str:
    values = [value for _, group in groups for value in group.values()]
    max_value = maximum if maximum is not None else max(max(values), 1e-9) * 1.12
    width, height = 900, 92 + len(groups) * 80
    x0, plot_w = 250, 560
    parts = [
        f'<svg class="bar-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
        f'<text x="{x0}" y="25" class="heat-title">{esc(title)}</text>',
    ]
    for group_index, (label, model_values) in enumerate(groups):
        y = 62 + group_index * 80
        parts.append(f'<text x="{x0-14}" y="{y+18}" text-anchor="end" class="bar-label">{esc(label)}</text>')
        for model_index, model in enumerate(MODELS):
            value = float(model_values[model])
            yy = y + model_index * 25
            bar_w = max(0.0, min(value / max_value, 1.0)) * plot_w
            parts.append(f'<rect x="{x0}" y="{yy}" width="{bar_w:.1f}" height="15" rx="2" fill="{COLORS[model]}" opacity=".88"/>')
            parts.append(f'<text x="{x0+bar_w+7:.1f}" y="{yy+12}" class="bar-value">{value:.3f}{esc(suffix)}</text>')
        parts.append(f'<text x="{x0+plot_w+18}" y="{y+12}" class="mini-model" fill="{COLORS[MODELS[0]]}">Q</text>')
        parts.append(f'<text x="{x0+plot_w+18}" y="{y+37}" class="mini-model" fill="{COLORS[MODELS[1]]}">G</text>')
    parts.append("</svg>")
    return "".join(parts)


def answer_query_layer_sweep_svg(
    curves: Mapping[str, Sequence[Mapping[str, str]]],
) -> str:
    """Draw confirmation donor-answer adoption over preregistered layer grids."""

    width, height = 1040, 360
    panel_w, plot_h = 420, 220
    parts = [
        f'<svg class="paper-chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="answer-patch-title answer-patch-desc">',
        '<title id="answer-patch-title">Native answer-query donor-state adoption by layer</title>',
        '<desc id="answer-patch-desc">Two panels show seed-equal donor-count adoption after replacing the receiver answer-query residual with a clean donor state. Error bars are seed-cluster bootstrap 95 percent intervals.</desc>',
    ]
    for panel_index, model in enumerate(MODELS):
        rows = sorted(curves[model], key=lambda row: int(row["layer"]))
        require(rows, f"Missing answer-query layer rows for {model}")
        ox, oy = 62 + panel_index * 500, 58
        max_layer = max(int(row["layer"]) for row in rows)

        def sx(layer: int) -> float:
            return ox + layer / max(max_layer, 1) * panel_w

        def sy(value: float) -> float:
            return oy + (1.0 - max(0.0, min(1.0, value))) * plot_h

        parts.extend(
            [
                f'<text x="{ox}" y="{oy-20}" class="heat-title">{esc(SHORT[model])} · full answer-state patch</text>',
                f'<rect x="{ox}" y="{oy}" width="{panel_w}" height="{plot_h}" class="plot-bg"/>',
            ]
        )
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = sy(tick)
            parts.extend(
                [
                    f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+panel_w}" y2="{y:.1f}" class="grid"/>',
                    f'<text x="{ox-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:.2f}</text>',
                ]
            )
        points = " ".join(
            f'{sx(int(row["layer"])):.1f},{sy(float(row["full_donor_adoption"])):.1f}'
            for row in rows
        )
        parts.append(
            f'<polyline points="{points}" class="series-line" stroke="{COLORS[model]}"/>'
        )
        for row in rows:
            layer = int(row["layer"])
            mean = float(row["full_donor_adoption"])
            low = float(row["full_donor_adoption_ci95_low"])
            high = float(row["full_donor_adoption_ci95_high"])
            x = sx(layer)
            parts.extend(
                [
                    f'<line x1="{x:.1f}" y1="{sy(high):.1f}" x2="{x:.1f}" y2="{sy(low):.1f}" stroke="#475467" stroke-width="1.5"/>',
                    f'<line x1="{x-4:.1f}" y1="{sy(high):.1f}" x2="{x+4:.1f}" y2="{sy(high):.1f}" stroke="#475467"/>',
                    f'<line x1="{x-4:.1f}" y1="{sy(low):.1f}" x2="{x+4:.1f}" y2="{sy(low):.1f}" stroke="#475467"/>',
                    f'<circle cx="{x:.1f}" cy="{sy(mean):.1f}" r="4.5" fill="{COLORS[model]}"><title>L{layer}: donor adoption={mean:.3f}; seed-bootstrap 95% CI [{low:.3f}, {high:.3f}]</title></circle>',
                    f'<text x="{x:.1f}" y="{oy+plot_h+19}" text-anchor="middle" class="tick">{layer}</text>',
                ]
            )
        parts.extend(
            [
                f'<text x="{ox+panel_w/2}" y="{oy+plot_h+42}" text-anchor="middle" class="axis-label">zero-based post-block layer</text>',
                f'<text transform="translate({ox-43} {oy+plot_h/2}) rotate(-90)" text-anchor="middle" class="axis-label">donor-count adoption</text>',
            ]
        )
    parts.extend(
        [
            f'<line x1="360" y1="338" x2="388" y2="338" stroke="{COLORS[MODELS[0]]}" stroke-width="3"/>',
            '<text x="396" y="342" class="legend-label">seed-equal mean</text>',
            '<line x1="545" y1="329" x2="545" y2="347" stroke="#475467" stroke-width="1.5"/>',
            '<text x="557" y="342" class="legend-label">seed-cluster bootstrap 95% CI</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def walkthrough_svg(walkthrough: Mapping[str, Mapping[str, Any]]) -> str:
    width, height = 980, 380
    parts = [
        f'<svg class="line-chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Single-seed expected-count paths under full context scrubbing and state restoration">'
    ]
    condition_style = {
        "full_item_restore": ("full item", "#0f766e"),
        "counter_carrier_restore": ("marker / tail carrier", "#7c3aed"),
        "counter_carrier_matched_control": ("matched ordinary state", "#98a2b3"),
    }
    for panel_index, model in enumerate(MODELS):
        data = walkthrough[model]
        x0 = 70 + panel_index * 490
        y0, plot_w, plot_h = 54, 400, 230

        def sx(k: int) -> float:
            return x0 + (k - 1) / 9 * plot_w

        def sy(value: float) -> float:
            return y0 + (10 - value) / 9 * plot_h

        parts.append(f'<text x="{x0}" y="25" class="heat-title">{esc(SHORT[model])} · seed {data["seed"]}</text>')
        parts.append(f'<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" class="plot-bg"/>')
        ideal = " ".join(f"{sx(k):.1f},{sy(k):.1f}" for k in range(1, 11))
        parts.append(f'<polyline points="{ideal}" fill="none" stroke="#d97706" stroke-width="1.5" stroke-dasharray="5 5"/>')
        for tick in (1, 4, 7, 10):
            y = sy(tick)
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+plot_w}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick}</text>')
        for k in range(1, 11):
            parts.append(f'<text x="{sx(k):.1f}" y="{y0+plot_h+20}" text-anchor="middle" class="tick">{k}</text>')
        for condition, (_, color) in condition_style.items():
            path = data["conditions"][condition]["expected_count_path"]
            points = " ".join(f"{sx(k):.1f},{sy(float(value)):.1f}" for k, value in enumerate(path, 1))
            parts.append(f'<polyline points="{points}" class="series-line" stroke="{color}"/>')
            for k, value in enumerate(path, 1):
                parts.append(f'<circle cx="{sx(k):.1f}" cy="{sy(float(value)):.1f}" r="3" fill="{color}"/>')
    legend_positions = (180, 370, 570)
    for x, (condition, (label, color)) in zip(legend_positions, condition_style.items()):
        parts.append(f'<line x1="{x}" y1="340" x2="{x+24}" y2="340" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x+30}" y="344" class="legend-label">{esc(label)}</text>')
    parts.append('<line x1="800" y1="340" x2="824" y2="340" stroke="#d97706" stroke-dasharray="5 5"/>')
    parts.append('<text x="830" y="344" class="legend-label">ideal y=k</text>')
    parts.append('<text transform="translate(18 170) rotate(-90)" text-anchor="middle" class="axis-label">Expected count</text>')
    parts.append('<text x="490" y="374" text-anchor="middle" class="axis-label">Restored item occurrence k</text>')
    parts.append("</svg>")
    return "".join(parts)


def indexed_progress_control_svg(
    analyses: Mapping[str, Mapping[str, Any]],
    active_confirmation_layers: Mapping[str, int],
) -> str:
    """Render discovery-only layer profiles for the explicit-index controls.

    Each model gets its own y scale because transition log-odds are not
    calibrated across tokenizers or models.  The frozen layer is selected from
    the dark seed-median curve; directional medians remain visible so a large
    one-sided effect cannot masquerade as a bidirectional progress transfer.
    """

    width, height = 1040, 365
    parts = [
        f'<svg class="paper-chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="indexed-layer-title indexed-layer-desc">',
        '<title id="indexed-layer-title">Explicit-index discovery layer sweeps</title>',
        '<desc id="indexed-layer-desc">Qwen and Gemma panels show the discovery paired donor-directed transition log-odds shift by post-block layer. The selected layer is frozen before confirmation.</desc>',
    ]
    series = (
        ("seed median", "median_seed_mean_effect", "#172033"),
        (
            "forward median",
            ("directions", "forward_skip", "median_paired_logodds_shift"),
            "#0f766e",
        ),
        (
            "backward median",
            ("directions", "backward_rewind", "median_paired_logodds_shift"),
            "#7c3aed",
        ),
    )

    def value(row: Mapping[str, Any], key: Any) -> float:
        if isinstance(key, tuple):
            active: Any = row
            for part in key:
                active = active[part]
            return float(active)
        return float(row[key])

    for column, model in enumerate(MODELS):
        scope_rows = [
            row for row in analyses[model]["scopes"] if row["scope"] == "item_span"
        ]
        require(len(scope_rows) == 1, f"{model}: expected one indexed item-span scope")
        scope = scope_rows[0]
        rows = sorted(scope["layer_summaries"], key=lambda row: int(row["layer"]))
        require(rows, f"{model}: indexed layer profile is empty")
        layers = [int(row["layer"]) for row in rows]
        values = [0.0]
        for _label, key, _color in series:
            values.extend(value(row, key) for row in rows)
        lower, upper = min(values), max(values)
        span = max(upper - lower, 1.0)
        lower -= 0.08 * span
        upper += 0.08 * span

        ox, oy = 62 + column * 510, 54
        plot_w, plot_h = 438, 232
        max_layer = max(layers)

        def sx(layer: int) -> float:
            return ox + layer / max(max_layer, 1) * plot_w

        def sy(metric: float) -> float:
            return oy + (upper - metric) / max(upper - lower, 1e-12) * plot_h

        grammar = (
            "k. City - score"
            if model == "Qwen3-8B"
            else "Record k: (City, score)"
        )
        parts.append(
            f'<text x="{ox}" y="{oy-26}" class="heat-title">'
            f'{esc(SHORT[model])} · {esc(grammar)}</text>'
        )
        parts.append(
            f'<rect x="{ox}" y="{oy}" width="{plot_w}" height="{plot_h}" class="plot-bg"/>'
        )
        for index in range(5):
            tick = lower + index * (upper - lower) / 4
            y = sy(tick)
            parts.append(
                f'<line x1="{ox}" y1="{y:.1f}" x2="{ox+plot_w}" y2="{y:.1f}" class="grid"/>'
            )
            parts.append(
                f'<text x="{ox-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:+.0f}</text>'
            )
        zero_y = sy(0.0)
        parts.append(
            f'<line x1="{ox}" y1="{zero_y:.1f}" x2="{ox+plot_w}" y2="{zero_y:.1f}" '
            'stroke="#667085" stroke-width="1.4" stroke-dasharray="5 4"/>'
        )
        for _label, key, color in series:
            points = " ".join(
                f'{sx(int(row["layer"])):.1f},{sy(value(row, key)):.1f}'
                for row in rows
            )
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.6"/>'
            )
        selected_layer = int(scope["selected_layer"])
        selected_rows = [row for row in rows if int(row["layer"]) == selected_layer]
        require(len(selected_rows) == 1, f"{model}: automatic indexed layer unavailable")
        selected_value = value(selected_rows[0], "median_seed_mean_effect")
        parts.append(
            f'<circle cx="{sx(selected_layer):.1f}" cy="{sy(selected_value):.1f}" r="6" '
            'fill="#fff" stroke="#172033" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{sx(selected_layer)+8:.1f}" y="{sy(selected_value)-10:.1f}" '
            f'class="chart-value">auto L{selected_layer}</text>'
        )
        active_layer = int(active_confirmation_layers[model])
        active_rows = [row for row in rows if int(row["layer"]) == active_layer]
        require(len(active_rows) == 1, f"{model}: active indexed layer unavailable")
        active_value = value(active_rows[0], "median_seed_mean_effect")
        active_x, active_y = sx(active_layer), sy(active_value)
        diamond = " ".join(
            (
                f"{active_x:.1f},{active_y-7:.1f}",
                f"{active_x+7:.1f},{active_y:.1f}",
                f"{active_x:.1f},{active_y+7:.1f}",
                f"{active_x-7:.1f},{active_y:.1f}",
            )
        )
        parts.append(
            f'<polygon points="{diamond}" fill="#d97706" stroke="#fff" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{active_x+9:.1f}" y="{active_y+18:.1f}" class="chart-value">'
            f'confirm L{active_layer} · {active_value:+.1f}</text>'
        )
        for layer_tick in sorted({0, max_layer // 2, max_layer}):
            parts.append(
                f'<text x="{sx(layer_tick):.1f}" y="{oy+plot_h+20}" text-anchor="middle" '
                f'class="tick">{layer_tick}</text>'
            )
        if column == 0:
            parts.append(
                f'<text transform="translate({ox-48} {oy+plot_h/2}) rotate(-90)" '
                'text-anchor="middle" class="axis-label">paired donor-directed log-odds shift</text>'
            )
        parts.append(
            f'<text x="{ox+plot_w/2}" y="{oy+plot_h+40}" text-anchor="middle" '
            'class="axis-label">zero-based post-block layer</text>'
        )
    legend_x = 288
    for index, (label, _key, color) in enumerate(series):
        x = legend_x + index * 175
        parts.append(
            f'<line x1="{x}" y1="348" x2="{x+26}" y2="348" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(f'<text x="{x+33}" y="352" class="legend-label">{esc(label)}</text>')
    parts.append('<polygon points="817,341 824,348 817,355 810,348" fill="#d97706"/>')
    parts.append('<text x="832" y="352" class="legend-label">external L16 confirmation anchor</text>')
    parts.append("</svg>")
    return "".join(parts)


def internal_counter_restoration_svg(
    occurrence_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    """Plot confirmation early-stop effects by restored occurrence k."""

    width, height = 1040, 390
    origins = ((70, 62), (565, 62))
    panel_w, panel_h = 400, 230
    panels = (
        ("A · Target-count margin gain", "mean target-margin gain", "mean_target_margin_gain"),
        ("B · Exact early-stop accuracy gain", "patched − scrub accuracy", "exact_gain"),
    )
    normalized: dict[str, list[dict[str, float]]] = {}
    for model, rows in occurrence_rows.items():
        normalized[model] = [
            {
                "k": float(row["target_occurrence"]),
                "mean_target_margin_gain": float(row["mean_target_margin_gain"]),
                "exact_gain": float(row["patched_exact_accuracy"])
                - float(row["baseline_exact_accuracy"]),
            }
            for row in rows
        ]
        require(
            [int(row["k"]) for row in normalized[model]] == list(range(2, 10)),
            f"{model} internal-counter occurrence support changed",
        )
    parts = [
        f'<svg class="paper-chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="No-running-index full-item restoration by early-stop target">'
    ]
    for panel_index, (title, y_label, key) in enumerate(panels):
        x0, y0 = origins[panel_index]
        values = [0.0] + [row[key] for rows in normalized.values() for row in rows]
        low, high = min(values), max(values)
        pad = max((high - low) * 0.16, 0.04 if key == "exact_gain" else 0.15)
        low, high = low - pad, high + pad

        def sx(k: float) -> float:
            return x0 + (k - 2.0) / 7.0 * panel_w

        def sy(value: float) -> float:
            return y0 + panel_h - (value - low) / (high - low) * panel_h

        parts.extend(
            [
                f'<text x="{x0}" y="30" class="heat-title">{esc(title)}</text>',
                f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#fbfcfe" stroke="#d0d5dd"/>',
                f'<line x1="{x0}" y1="{sy(0):.1f}" x2="{x0+panel_w}" y2="{sy(0):.1f}" stroke="#98a2b3" stroke-dasharray="4 4"/>',
            ]
        )
        for tick_index in range(5):
            value = low + tick_index * (high - low) / 4
            y = sy(value)
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+panel_w}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{value:+.2f}</text>')
        for k in range(2, 10):
            x = sx(k)
            parts.append(f'<text x="{x:.1f}" y="{y0+panel_h+20}" text-anchor="middle" class="tick">{k}</text>')
        for model in MODELS:
            color = COLORS[model]
            rows = normalized[model]
            points = " ".join(f"{sx(row['k']):.1f},{sy(row[key]):.1f}" for row in rows)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.6"/>')
            for row in rows:
                parts.append(
                    f'<circle cx="{sx(row["k"]):.1f}" cy="{sy(row[key]):.1f}" r="4" fill="{color}">'
                    f'<title>{esc(SHORT[model])}, k={int(row["k"])}: {row[key]:+.4f}</title></circle>'
                )
        parts.append(f'<text x="{x0+panel_w/2}" y="{height-48}" text-anchor="middle" class="axis-label">restored occurrence k</text>')
        parts.append(f'<text transform="translate({x0-54} {y0+panel_h/2}) rotate(-90)" text-anchor="middle" class="axis-label">{esc(y_label)}</text>')
    parts.extend(
        [
            '<line x1="380" y1="365" x2="405" y2="365" stroke="#0f766e" stroke-width="3"/><text x="412" y="369" class="legend-label">Qwen</text>',
            '<line x1="550" y1="365" x2="575" y2="365" stroke="#7c3aed" stroke-width="3"/><text x="582" y="369" class="legend-label">Gemma</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def chain_svg() -> str:
    stages = (
        ("Targeted retrieval", "read next record", "strong", "strong"),
        ("Grammar carrier", "write retrieved event", "controlled", "controlled"),
        ("Commit / event state", "content + progress", "controlled", "controlled"),
        ("Next-item routing", "state-guided successor", "natural", "simulated"),
        (
            "Terminal readout",
            "trace bridge + answer execution",
            "split",
            "split",
        ),
    )
    width, height, box_w, gap = 1080, 260, 170, 28
    x0 = 80
    parts = [f'<svg class="chain-figure" viewBox="0 0 {width} {height}" role="img" aria-label="Native-thinking causal chain by model">']
    for idx, (title, subtitle, _, _) in enumerate(stages):
        x = x0 + idx * (box_w + gap)
        parts.append(f'<rect x="{x}" y="35" width="{box_w}" height="68" rx="4" fill="#fff" stroke="#cfd6e2"/>')
        parts.append(f'<text x="{x+box_w/2}" y="61" text-anchor="middle" class="chain-title">{esc(title)}</text>')
        parts.append(f'<text x="{x+box_w/2}" y="82" text-anchor="middle" class="chain-sub">{esc(subtitle)}</text>')
        if idx < len(stages) - 1:
            parts.append(f'<path d="M{x+box_w} 69 H{x+box_w+gap-8}" stroke="#98a2b3" stroke-width="2"/>')
            parts.append(f'<path d="M{x+box_w+gap-14} 63 L{x+box_w+gap-8} 69 L{x+box_w+gap-14} 75" fill="none" stroke="#98a2b3" stroke-width="2"/>')
    for row_index, model in enumerate(MODELS):
        y = 142 + row_index * 48
        parts.append(f'<text x="{x0-8}" y="{y+16}" text-anchor="end" class="chain-model">{esc(SHORT[model])}</text>')
        for idx, (_, _, q_status, g_status) in enumerate(stages):
            status = q_status if model == MODELS[0] else g_status
            color = {
                "strong": "#0f766e",
                "natural": "#0f766e",
                "controlled": "#46758f",
                "qualified": "#46758f",
                "latent": "#9a4b00",
                "simulated": "#9a4b00",
                "conditional": "#7c3aed",
                "split": "#7c3aed",
                "not_run": "#667085",
            }[status]
            label = {
                "strong": "confirmed",
                "natural": "natural confirmed",
                "controlled": "controlled edge",
                "qualified": "legacy causal†",
                "latent": "latent score only",
                "simulated": "simulatively confirmed†",
                "conditional": "controlled only",
                "split": "partial relay + answer exec.",
                "not_run": "not run",
            }[status]
            x = x0 + idx * (box_w + gap)
            parts.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="29" rx="3" fill="{color}" opacity=".12" stroke="{color}"/>')
            parts.append(f'<text x="{x+box_w/2}" y="{y+19}" text-anchor="middle" class="chain-status" fill="{color}">{label}</text>')
    parts.append("</svg>")
    return "".join(parts)


def natural_progress_bridge_svg(summary: Mapping[str, Any]) -> str:
    """Show how one L16 intervention propagates across four ordered readouts."""

    values = (
        ("Δ route > 0", 20, 20),
        ("Δ attention > 0", 20, 20),
        ("donor argmax", int(round(20 * float(summary["patched_donor_argmax_rate"]))), 20),
        (
            "first city follows donor",
            int(summary["patched_first_known_city_donor_adoption_count"]),
            20,
        ),
    )
    width, height = 900, 390
    left, top, plot_w, plot_h = 92, 42, 750, 260
    bar_w = 105
    gap = (plot_w - len(values) * bar_w) / (len(values) + 1)
    parts = [
        f'<svg class="paper-chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="progress-bridge-title progress-bridge-desc">',
        '<title id="progress-bridge-title">Qwen L16 natural no-index progress-state bridge</title>',
        '<desc id="progress-bridge-desc">Four bars report the fraction of twenty paired cells with a donor-directed likelihood shift, donor-directed attention shift, donor successor candidate argmax, and donor-following first generated city.</desc>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfcfe" stroke="#d0d5dd"/>',
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + (1.0 - tick) * plot_h
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:.2f}</text>'
        )
    for index, (label, hits, total) in enumerate(values):
        value = hits / total
        x = left + gap + index * (bar_w + gap)
        y = top + (1.0 - value) * plot_h
        color = "#0f766e" if index < 2 else "#46758f"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{value*plot_h:.1f}" '
            f'fill="{color}" opacity=".88"><title>{esc(label)}: {hits}/{total}</title></rect>'
        )
        parts.append(
            f'<text x="{x+bar_w/2:.1f}" y="{y-9:.1f}" text-anchor="middle" class="chart-value">{hits}/{total}</text>'
        )
        words = label.split(" ")
        if len(words) > 2:
            first = " ".join(words[:2])
            second = " ".join(words[2:])
            parts.append(
                f'<text x="{x+bar_w/2:.1f}" y="{top+plot_h+24}" text-anchor="middle" class="tick">{esc(first)}</text>'
            )
            parts.append(
                f'<text x="{x+bar_w/2:.1f}" y="{top+plot_h+39}" text-anchor="middle" class="tick">{esc(second)}</text>'
            )
        else:
            parts.append(
                f'<text x="{x+bar_w/2:.1f}" y="{top+plot_h+28}" text-anchor="middle" class="tick">{esc(label)}</text>'
            )
    parts.append(
        f'<text transform="translate(25 {top+plot_h/2}) rotate(-90)" text-anchor="middle" class="axis-label">fraction of 20 paired cells</text>'
    )
    parts.append(
        f'<text x="{left+plot_w/2}" y="{height-18}" text-anchor="middle" class="axis-label">ordered readout after the same L16 item-span patch</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


TOP_LEVEL_SECTION_IDS = (
    "definitions",
    "summary",
    "design",
    "task",
    "representation",
    "retrieval",
    "write",
    "answer",
    "walkthrough",
    "comparison",
    "appendix",
    "audit",
)


def extract_top_level_sections(document: str) -> dict[str, str]:
    """Extract the report's original top-level sections before reordering them.

    The source template predates the current claim hierarchy.  Keeping the
    expensive, audited figures in that template and composing a new main path
    here makes the scientific reordering explicit without duplicating loaders.
    """

    starts: dict[str, int] = {}
    for section_id in TOP_LEVEL_SECTION_IDS:
        marker = f'<section id="{section_id}"'
        starts[section_id] = document.index(marker)
    ordered = sorted(starts.items(), key=lambda item: item[1])
    sections: dict[str, str] = {}
    for index, (section_id, start) in enumerate(ordered):
        end = ordered[index + 1][1] if index + 1 < len(ordered) else document.index("</main>", start)
        sections[section_id] = document[start:end].strip()
    return sections


def section_body(section: str) -> str:
    """Return the contents of a complete top-level <section> block."""

    start = section.index(">") + 1
    end = section.rfind("</section>")
    require(end > start, "Malformed top-level section")
    return section[start:end].strip()


def build_report(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    sample_alignment_audit = read_json(args.sample_alignment_audit)
    require(
        sample_alignment_audit.get("status") == "PASS"
        and sample_alignment_audit.get("evidence")
        and all(
            row.get("status") == "PASS"
            for row in sample_alignment_audit["evidence"].values()
        ),
        "Cross-model sample-alignment audit must pass before report construction",
    )
    output_alignment_audit = read_json(args.output_alignment_audit)
    require(
        output_alignment_audit.get("status") == "PASS"
        and output_alignment_audit.get("evidence")
        and all(
            row.get("status") == "PASS"
            for row in output_alignment_audit["evidence"].values()
        ),
        "Realized cross-model output-alignment audit must pass before report construction",
    )
    q_analysis = read_json(args.qwen_targeted_analysis)
    g_analysis = read_json(args.gemma_targeted_analysis)
    q_primary = selected_row(q_analysis, 128)
    g_primary = selected_row(g_analysis, 6)
    q_dose = [row for row in q_analysis["overall"] if row["scope"] == "all_registered_grammars"]
    g_dose = [row for row in g_analysis["overall"] if row["scope"] == "all_registered_grammars"]
    representation = load_representation(args.representation_root, args.dual_endpoint_root)
    layerwise_representation = load_layerwise_representation(
        args.representation_root, args.dual_endpoint_root
    )
    terminal_relay_summary = read_json(
        args.terminal_relay_common_support / "summary.json"
    )
    terminal_relay_pairs = read_csv(
        args.terminal_relay_common_support / "common_support_pairs.csv"
    )
    require(
        terminal_relay_summary.get("status") == "PASS"
        and terminal_relay_summary.get("phase") == "confirmation"
        and terminal_relay_summary.get("matching_keys")
        == ["seed", "gold_count", "donor_offset"]
        and int(terminal_relay_summary.get("common_pair_count_per_model", -1))
        == 86
        and int(terminal_relay_summary.get("common_seed_count", -1)) == 10
        and terminal_relay_summary.get("selection_uses_outcomes") is False
        and len(terminal_relay_pairs) == 86,
        "Terminal-relay common-support contract changed",
    )
    for model in MODELS:
        relay_model = terminal_relay_summary["models"][model]
        require(
            int(relay_model.get("common_pair_count", -1)) == 86
            and int(relay_model.get("seed_count", -1)) == 10
            and relay_model.get("pathway_primary_pass") is True
            and relay_model.get("answer_query_secondary_pass") is True,
            f"{model}: terminal-relay common-support gate changed",
        )
        for effect_name in (
            "patch_damage_natural",
            "specific_mediation__post_terminal_suffix",
            "specific_mediation__answer_query",
        ):
            require(
                float(relay_model["effects"][effect_name]["ci_low"]) > 0.0,
                f"{model}: terminal-relay {effect_name} no longer supports the path",
            )
    band_diagnostics = {
        model: read_json(args.band_diagnostic_root / model / "band_diagnostic.json")
        for model in MODELS
    }
    running_display_layers = {
        model: int(representation[model]["running"]["layer"])
        for model in MODELS
    }
    final_display_layers = {
        model: int(representation[model]["final"]["layer"]) for model in MODELS
    }
    geometry_3d = load_all_layer_native_geometry(
        args.geometry_comparison_report,
        running_display_layers,
        final_display_layers,
    )
    aligned_running_3d = read_json(args.running_pca3_coordinates)
    require(
        aligned_running_3d.get("status") == "PASS"
        and aligned_running_3d.get("exact_cross_model_sample_alignment") is True
        and aligned_running_3d.get("common_key_sha256")
        == sample_alignment_audit["evidence"][
            "running_representation_common_support"
        ]["key_multiset_sha256"],
        "Aligned running PCA3 payload does not match the frozen common support",
    )
    geometry_3d["running"] = {
        model: aligned_running_3d["models"][model] for model in MODELS
    }
    token_evidence = {
        model: replace_with_aligned_targeting(
            load_token_ablation_evidence(args.token_ablation_root / model, model),
            args.aligned_run_root,
            model,
        )
        for model in MODELS
    }
    attention_examples = load_attention_examples(args.atlas_root)
    qwen_attention_sum_svg = inline_standalone_svg(
        args.atlas_root
        / "Qwen3-8B_adjacent_rank_after_city_Top128_p0_attention_sum.svg"
    )
    gemma_attention_sum_svg = inline_standalone_svg(
        args.atlas_root
        / "Gemma4-E4B_adjacent_rank_after_city_Top6_p0_attention_sum.svg"
    )
    appendix_e_svgs = {
        "qwen_same_head": inline_standalone_svg(
            args.atlas_root / "Qwen3-8B_same_unit_rank_before_city_p0_head_map.svg"
        ),
        "gemma_same_head": inline_standalone_svg(
            args.atlas_root / "Gemma4-E4B_same_unit_rank_before_city_p0_head_map.svg"
        ),
        "qwen_same_ordinal": inline_standalone_svg(
            args.atlas_root
            / "Qwen3-8B_same_unit_rank_before_city_p0_needle_ordinal_by_head.svg"
        ),
        "gemma_same_ordinal": inline_standalone_svg(
            args.atlas_root
            / "Gemma4-E4B_same_unit_rank_before_city_p0_needle_ordinal_by_head.svg"
        ),
        "qwen_bullet_head": inline_standalone_svg(
            args.atlas_root / "Qwen3-8B_structural_invariant_bullet_p0_head_map.svg"
        ),
        "gemma_bullet_head": inline_standalone_svg(
            args.atlas_root / "Gemma4-E4B_structural_invariant_bullet_p0_head_map.svg"
        ),
        "qwen_bullet_ordinal": inline_standalone_svg(
            args.atlas_root
            / "Qwen3-8B_structural_invariant_bullet_p0_needle_ordinal_by_head.svg"
        ),
        "gemma_bullet_ordinal": inline_standalone_svg(
            args.atlas_root
            / "Gemma4-E4B_structural_invariant_bullet_p0_needle_ordinal_by_head.svg"
        ),
    }
    for example in attention_examples:
        if example["key"] == "gemma_top8_aggregate":
            example["label"] = (
                "Gemma4-E4B · ranked-head aggregate "
                "(columns 1–6 = current Top-6; 7–8 = K8 diagnostic)"
            )
            example["evidence"] = "exact P0 · 20 discovery seeds"
    qwen_head_map = head_map_svg(
        load_p0_head_score_rows(args.atlas_root, "Qwen3-8B"),
        "Qwen3-8B exact-P0 target-attention atlas · current Top-128",
        top_outline=128,
        value_field="discovery_selection_value",
        value_label="seed-event mean target-record attention mass",
    )
    gemma_head_map = split_gemma_current_head_map_svg(
        load_p0_head_score_rows(args.atlas_root, "Gemma4-E4B")
    )
    aligned_targeted = {
        model: {
            phase: read_json(
                args.aligned_run_root
                / model
                / "targeted_retrieval"
                / f"analysis_{phase}"
                / "claim_gates.json"
            )
            for phase in ("discovery", "confirmation")
        }
        for model in MODELS
    }
    write = {}
    for model in MODELS:
        phases = {
            phase: read_json(
                args.aligned_run_root
                / model
                / "targeted_counter_write"
                / f"analysis_{phase}"
                / "claim_gates.json"
            )
            for phase in ("discovery", "confirmation")
        }
        write[model] = {
            "status": "PASS",
            "discovery_seed_count": int(phases["discovery"]["seed_count"]),
            "confirmation_seed_count": int(phases["confirmation"]["seed_count"]),
            "outcome_blind": bool(
                phases["discovery"]["outcome_blind"]
                and phases["confirmation"]["outcome_blind"]
            ),
            "selection_rank_used": bool(
                phases["discovery"]["selection_rank_used"]
                or phases["confirmation"]["selection_rank_used"]
            ),
            **phases,
        }
    grammar_anchor_paths = {
        model: args.aligned_grammar_panel_root / f"{model}_anchor_panel.jsonl"
        for model in MODELS
    }
    grammar_anchors = {
        model: read_jsonl(path) for model, path in grammar_anchor_paths.items()
    }
    routed_anchors = {
        model: read_jsonl(
            args.aligned_routed_panel_root / f"{model}_anchor_panel.jsonl"
        )
        for model in MODELS
    }
    commit_query = {}
    grammar_span = {}
    for model in MODELS:
        commit_phases = {
            phase: read_json(
                args.aligned_run_root
                / model
                / "commit_state_to_targeted_query"
                / f"analysis_{phase}"
                / "claim_gates.json"
            )
            for phase in ("discovery", "confirmation")
        }
        commit_query[model] = {
            "status": "PASS",
            "discovery_seed_count": int(commit_phases["discovery"]["seed_count"]),
            "confirmation_seed_count": int(
                commit_phases["confirmation"]["seed_count"]
            ),
            "outcome_blind": bool(
                commit_phases["discovery"]["outcome_blind"]
                and commit_phases["confirmation"]["outcome_blind"]
            ),
            "selection_rank_used": bool(
                commit_phases["discovery"]["selection_rank_used"]
                or commit_phases["confirmation"]["selection_rank_used"]
            ),
            **commit_phases,
        }
        grammar_phases = {
            phase: read_json(
                args.aligned_run_root
                / model
                / "grammar_span"
                / f"analysis_{phase}"
                / "claim_gates.json"
            )
            for phase in ("discovery", "confirmation")
        }
        grammar_span[model] = {
            "status": "PASS",
            "discovery_seed_count": int(grammar_phases["discovery"]["seed_count"]),
            "confirmation_seed_count": int(
                grammar_phases["confirmation"]["seed_count"]
            ),
            "outcome_blind": bool(
                grammar_phases["discovery"]["outcome_blind"]
                and grammar_phases["confirmation"]["outcome_blind"]
            ),
            "selection_rank_used": bool(
                grammar_phases["discovery"]["selection_rank_used"]
                or grammar_phases["confirmation"]["selection_rank_used"]
            ),
            **grammar_phases,
        }
    answer_query_layer_rows: dict[str, list[dict[str, str]]] = {}
    answer_query_layer_audits: dict[str, dict[str, Any]] = {}
    answer_query_plan_audits: dict[str, dict[str, Any]] = {}
    answer_query_alignment_manifest = read_json(
        args.answer_query_plan_root / "alignment_manifest.json"
    )
    require(
        answer_query_alignment_manifest.get("status")
        == "FROZEN_OUTCOME_BLIND_EXACT_ALIGNMENT"
        and answer_query_alignment_manifest.get("exact_cross_model_key_equality")
        is True
        and int(
            answer_query_alignment_manifest.get("directed_pairs_per_model", -1)
        )
        == 40
        and answer_query_alignment_manifest.get(
            "pair_selection_uses_patch_outcome"
        )
        is False,
        "Answer-query common pair plan is not frozen and exactly aligned",
    )
    for model in MODELS:
        model_root = (
            args.answer_query_layer_root / model / "answer_query_layer_sweep"
        )
        pair_path = args.answer_query_plan_root / f"{model}_pairs.jsonl"
        trials_path = model_root / "trials.jsonl"
        rows = read_csv(model_root / "analysis" / "layer_effects.csv")
        analysis_audit = read_json(model_root / "analysis" / "audit.json")
        plan_audit = answer_query_alignment_manifest
        require(
            analysis_audit.get("status") == "passed"
            and plan_audit.get("status")
            == "FROZEN_OUTCOME_BLIND_EXACT_ALIGNMENT",
            f"{model}: answer-query layer sweep audit failed",
        )
        pair_digest = sha256(pair_path)
        pair_registry_digest = sha256_json(read_jsonl(pair_path))
        trials_digest = sha256(trials_path)
        require(
            plan_audit.get(
                "qwen_pairs_sha256"
                if model == "Qwen3-8B"
                else "gemma_pairs_sha256"
            )
            == pair_registry_digest
            and analysis_audit.get("pairs_sha256") == pair_digest,
            f"{model}: answer-query pair registry hash changed",
        )
        require(
            analysis_audit.get("trials_sha256") == trials_digest,
            f"{model}: answer-query trial file hash changed",
        )
        require(
            analysis_audit.get("conditions")
            == ["self_patch", "full_donor_patch"],
            f"{model}: answer-query layer conditions changed",
        )
        require(
            [int(value) for value in analysis_audit.get("layers", [])]
            == ANSWER_QUERY_LAYER_GRID[model]
            and [int(value) for value in plan_audit["layers"][model]]
            == ANSWER_QUERY_LAYER_GRID[model],
            f"{model}: answer-query layer grid changed",
        )
        require(
            int(analysis_audit.get("registered_pairs", -1))
            == ANSWER_QUERY_PAIR_COUNT[model]
            and int(plan_audit.get("directed_pairs_per_model", -1))
            == ANSWER_QUERY_PAIR_COUNT[model],
            f"{model}: answer-query pair count changed",
        )
        require(
            int(analysis_audit.get("completed_pair_layer_cells", -1))
            == ANSWER_QUERY_PAIR_COUNT[model] * len(ANSWER_QUERY_LAYER_GRID[model])
            and int(analysis_audit.get("completed_trials", -1))
            == 2 * ANSWER_QUERY_PAIR_COUNT[model] * len(ANSWER_QUERY_LAYER_GRID[model]),
            f"{model}: answer-query trial grid is incomplete",
        )
        eligible_seeds = [
            int(value)
            for value in plan_audit.get("confirmation_seeds", [])
        ]
        require(
            eligible_seeds == list(range(1254, 1264)),
            f"{model}: answer-query confirmation seed registry changed",
        )
        require(
            [int(row["layer"]) for row in rows]
            == ANSWER_QUERY_LAYER_GRID[model],
            f"{model}: answer-query layer table is not complete and ordered",
        )
        answer_query_layer_rows[model] = rows
        answer_query_layer_audits[model] = analysis_audit
        answer_query_plan_audits[model] = plan_audit

    historical_answer_rows = [
        row
        for row in read_csv(args.historical_causal_chain_statistics)
        if row.get("stage") == "answer_hidden_state_patch"
        and row.get("mechanism") == "answer_state_transport"
        and row.get("scope") == "overall"
    ]
    historical_answer_controls = {
        model: {
            metric: next(
                row
                for row in historical_answer_rows
                if row["model_label"] == model and row["metric"] == metric
            )
            for metric in (
                "full_donor_adoption",
                "projected_donor_adoption",
                "projected_adoption_specificity",
            )
        }
        for model in MODELS
    }
    q_free = read_json(args.snapshot_root / "targeted_counter_20260822" / "Qwen3-8B" / "targeted_counter_complete.json")
    gemma_query_mediation = {
        geometry: read_json(
            args.snapshot_root
            / f"gemma_query_mediation_{geometry}_discovery_claim_gates.json"
        )
        for geometry in ("endpoint", "suffix4", "suffix8")
    }
    walkthrough = {
        model: read_json(args.snapshot_root / "single_seed_walkthrough_20260822_v2" / model / "analysis" / "walkthrough_complete.json")
        for model in MODELS
    }
    ncc_supplement = {
        model: read_json(
            args.ncc_supplement_root / model / "ncc_analysis" / "claim_gates.json"
        )
        for model in MODELS
    }
    ncc_layerwise_diagnostic = {
        model: read_json(
            args.ncc_supplement_root
            / model
            / "ncc_analysis"
            / "layerwise_timing_diagnostic.json"
        )
        for model in MODELS
    }
    gemma_ncc_layerwise_diagnostic = ncc_layerwise_diagnostic["Gemma4-E4B"]
    stratified_ncc = {
        model: {
            timing: read_json(
                args.aligned_run_root
                / model
                / "stratified_ncc"
                / timing
                / "analysis"
                / "claim_gates.json"
            )
            for timing in ("rank_after_city", "rank_before_city")
        }
        for model in MODELS
    }
    stratified_ncc_synthesis = read_json(
        args.stratified_ncc_root
        / "synthesis"
        / "stratified_ncc_synthesis.json"
    )
    stratified_ncc_inputs = {
        model: read_json(
            args.stratified_ncc_root / model / "stratified_ncc_input_manifest.json"
        )
        for model in MODELS
    }
    stratified_ncc_site_audits = {
        model: {
            timing: read_json(
                args.stratified_ncc_root / model / f"{timing}_site_audit.json"
            )
            for timing in ("rank_after_city", "rank_before_city")
        }
        for model in MODELS
    }
    targeted_logit_margin = {
        model: {
            timing: read_json(
                args.aligned_run_root
                / model
                / "direct_count_logit_margin"
                / timing
                / "analysis"
                / "claim_gates.json"
            )
            for timing in ("rank_after_city", "rank_before_city")
        }
        for model in MODELS
    }
    targeted_logit_margin_synthesis = read_json(
        args.logit_margin_root
        / "synthesis"
        / "targeted_logit_margin_synthesis.json"
    )
    unnumbered_counter = {
        model: read_json(
            args.ncc_supplement_root
            / model
            / "unnumbered_analysis_confirmation"
            / "claim_gates.json"
        )
        for model in MODELS
    }
    unnumbered_occurrence = {
        model: read_csv(
            args.ncc_supplement_root
            / model
            / "unnumbered_analysis_confirmation"
            / "occurrence_metrics.csv"
        )
        for model in MODELS
    }
    patch_scope_layer = read_json(args.patch_scope_layer_sweep)
    patch_scope_frozen = read_json(args.patch_scope_frozen_confirmation)
    patch_scope_generation_audit = read_json(args.patch_scope_generation_audit)
    item_span_l16 = read_json(args.item_span_l16)
    item_span_l16_generation_audit = read_json(args.item_span_l16_generation_audit)
    historical_event_tail = read_json(args.historical_event_tail_confirmation)
    indexed_cohort_manifest = read_json(args.indexed_progress_cohort_manifest)
    indexed_progress_freeze = read_json(args.indexed_progress_freeze_manifest)
    indexed_progress_discovery = {
        model: read_json(
            args.indexed_progress_discovery_root
            / model
            / "indexed_progress_control"
            / "discovery_layer_sweep"
            / "layer_sweep_analysis.json"
        )
        for model in MODELS
    }
    indexed_progress_confirmation = {
        model: read_json(
            args.indexed_progress_confirmation_root
            / model
            / "indexed_progress_control"
            / "confirmation_analysis.json"
        )
        for model in MODELS
    }
    indexed_progress_generation_audit = read_json(
        args.indexed_progress_generation_audit
    )
    gemma_prompt_noindex_cohort = read_json(
        args.gemma_prompt_conditioned_noindex_cohort_manifest
    )
    gemma_prompt_noindex = read_json(
        args.gemma_prompt_conditioned_noindex_analysis
    )
    patch_scope_layer_svg = inline_standalone_svg(args.patch_scope_layer_plot)
    indexed_confirmation_layers = {
        model: int(indexed_progress_freeze["active_confirmation_layers"][model])
        for model in MODELS
    }
    indexed_progress_layer_svg = indexed_progress_control_svg(
        indexed_progress_discovery,
        indexed_confirmation_layers,
    )

    patch_scope_discovery = {
        row["scope"]: row for row in patch_scope_layer["scopes"]
    }
    patch_scope_confirmation = {
        row["scope"]: row for row in patch_scope_frozen["summaries"]
    }
    item_span_l16_summary = item_span_l16["summaries"][0]
    natural_progress_bridge = natural_progress_bridge_svg(item_span_l16_summary)
    qwen_noindex_discovery_seeds = sorted(
        {int(row["seed"]) for row in patch_scope_layer["cells"]}
    )
    qwen_noindex_confirmation_seeds = sorted(
        {int(row["seed"]) for row in patch_scope_frozen["cells"]}
    )
    require(
        len(qwen_noindex_discovery_seeds) == 20
        and len(qwen_noindex_confirmation_seeds) == 10
        and not set(qwen_noindex_discovery_seeds)
        & set(qwen_noindex_confirmation_seeds),
        "Qwen natural no-index 20/10 cohort changed",
    )
    historical_event_tail_summary = historical_event_tail["pooled_summary"]
    indexed_progress_selected = {
        model: next(
            row
            for row in indexed_progress_discovery[model]["scopes"]
            if row["scope"] == "item_span"
        )
        for model in MODELS
    }
    indexed_progress_active_discovery = {
        model: next(
            row
            for row in indexed_progress_selected[model]["layer_summaries"]
            if int(row["layer"]) == indexed_confirmation_layers[model]
        )
        for model in MODELS
    }
    indexed_progress_summary = {
        model: next(
            row
            for row in indexed_progress_confirmation[model]["summaries"]
            if row["split"] in {"confirmation", "confirmation10"}
            and row["scope"] == "item_span"
        )
        for model in MODELS
    }
    gemma_prompt_noindex_discovery = gemma_prompt_noindex["phases"]["discovery"]
    gemma_prompt_noindex_confirmation = gemma_prompt_noindex["phases"][
        "confirmation"
    ]
    gemma_prompt_noindex_confirm_pooled = gemma_prompt_noindex_confirmation[
        "pooled_across_k"
    ]

    require(q_analysis.get("status") == "PASS" and g_analysis.get("status") == "PASS", "Targeted analyses must PASS")
    require(
        gemma_prompt_noindex_cohort.get("status") == "PASS"
        and gemma_prompt_noindex_cohort.get("schema_version")
        == "realistic_niah_v5_gemma_prompt_conditioned_noindex_v3"
        and gemma_prompt_noindex_cohort.get("prompt_conditioned") is True
        and gemma_prompt_noindex_cohort.get("prompt_modified") is True
        and gemma_prompt_noindex_cohort.get(
            "fixed_marker_contains_count_information"
        )
        is False
        and gemma_prompt_noindex_cohort.get(
            "terminal_total_correctness_used_for_selection"
        )
        is False
        and gemma_prompt_noindex_cohort.get(
            "selection_independent_of_patch_outcomes"
        )
        is True
        and gemma_prompt_noindex_cohort.get(
            "spontaneous_natural_noindex_claim_allowed"
        )
        is False,
        "Gemma prompt-conditioned cohort claim boundary changed",
    )
    require(
        len(gemma_prompt_noindex_cohort["discovery_seeds"]) == 20
        and len(gemma_prompt_noindex_cohort["confirmation_seeds"]) == 10
        and not set(gemma_prompt_noindex_cohort["discovery_seeds"])
        & set(gemma_prompt_noindex_cohort["confirmation_seeds"])
        and int(gemma_prompt_noindex_cohort["scanned_seed_count"]) == 52
        and int(gemma_prompt_noindex_cohort["selected_seed_count"]) == 30,
        "Gemma prompt-conditioned 20/10 cohort changed",
    )
    require(
        gemma_prompt_noindex.get("status") == "PASS"
        and gemma_prompt_noindex.get("schema_version")
        == "gemma_prompt_conditioned_forward_analysis_v1"
        and gemma_prompt_noindex.get("model_label") == "Gemma4-E4B"
        and int(gemma_prompt_noindex.get("layer")) == 16
        and gemma_prompt_noindex.get("patch_scope") == "item_span"
        and gemma_prompt_noindex.get("direction")
        == "forward_only_k_to_k_plus_one"
        and gemma_prompt_noindex.get("claim_scope")
        == "prompt-conditioned no-index auxiliary only",
        "Gemma prompt-conditioned causal contract changed",
    )
    require(
        int(gemma_prompt_noindex_discovery["seed_count"]) == 20
        and int(gemma_prompt_noindex_discovery["pair_count_across_k"]) == 60
        and int(gemma_prompt_noindex_confirmation["seed_count"]) == 10
        and int(gemma_prompt_noindex_confirmation["pair_count_across_k"]) == 30
        and int(
            gemma_prompt_noindex_confirm_pooled["donor_argmax_patch"]["hits"]
        )
        == 30
        and int(
            gemma_prompt_noindex_confirm_pooled[
                "greedy_donor_adoption_patch"
            ]["hits"]
        )
        == 22
        and int(
            gemma_prompt_noindex_confirm_pooled["positive_logodds_gain"][
                "hits"
            ]
        )
        == 30,
        "Gemma prompt-conditioned confirmation result changed",
    )
    require(
        indexed_cohort_manifest.get("status")
        == "FROZEN_OUTCOME_BLIND_EXACT_ALIGNMENT"
        and indexed_cohort_manifest.get("exact_cross_model_key_equality") is True
        and indexed_cohort_manifest.get(
            "inner_enumeration_surface_matched_across_models"
        )
        is True
        and indexed_cohort_manifest.get("claim_role")
        == "controlled explicit-index positive control only"
        and indexed_cohort_manifest.get(
            "internal_counter_without_visible_index_claim_allowed"
        )
        is False,
        "Indexed cohort must remain an explicit-index positive control",
    )
    require(
        indexed_progress_freeze.get("status") == "FROZEN_BEFORE_CONFIRMATION"
        and indexed_progress_freeze.get("confirmation_outcomes_observed") is False
        and indexed_confirmation_layers
        == {"Qwen3-8B": 19, "Gemma4-E4B": 16}
        and indexed_progress_freeze.get("layer_rule")
        == (
            "use the model-specific item_end layer selected only from "
            "exact-common-support discovery representation geometry; do not "
            "select a layer from indexed-control outcomes"
        ),
        "Indexed confirmation must retain the representation-frozen layers",
    )
    require(
        len(indexed_cohort_manifest["discovery_seeds"]) == 20
        and len(indexed_cohort_manifest["confirmation_seeds"]) == 10
        and not set(indexed_cohort_manifest["discovery_seeds"])
        & set(indexed_cohort_manifest["confirmation_seeds"]),
        "Indexed exact-alignment 20/10 split changed",
    )
    for model in MODELS:
        cohort = indexed_cohort_manifest["models"][model]
        require(
            int(cohort["sample_count"]) == 30,
            f"{model}: indexed aligned sample count changed",
        )
        require(
            int(indexed_progress_selected[model]["selected_layer_summary"]["cell_count"])
            == 40
            and int(indexed_progress_selected[model]["selected_layer_summary"]["seed_count"])
            == 20,
            f"{model}: indexed discovery support changed",
        )
        require(
            int(indexed_progress_active_discovery[model]["cell_count"]) == 40
            and int(indexed_progress_active_discovery[model]["seed_count"]) == 20,
            f"{model}: representation-frozen indexed discovery support changed",
        )
        require(
            int(indexed_progress_summary[model]["cell_count"]) == 60
            and int(indexed_progress_summary[model]["seed_count"]) == 10,
            f"{model}: indexed confirmation support changed",
        )
        generation_audit = indexed_progress_generation_audit["models"][model]
        require(
            int(generation_audit["cell_count"]) == 60
            and int(generation_audit["patched_donor_adoption_count"])
            == int(
                indexed_progress_summary[model][
                    "patched_first_known_city_donor_adoption_count"
                ]
            )
            and int(generation_audit["receiver_donor_adoption_count"])
            == int(
                indexed_progress_summary[model][
                    "receiver_first_known_city_donor_adoption_count"
                ]
            )
            and int(generation_audit["adoption_after_first_80_chars_count"])
            == 0,
            f"{model}: indexed generation audit changed",
        )
        require(
            {
                int(cell["layer"])
                for cell in indexed_progress_confirmation[model]["cells"]
            }
            == {indexed_confirmation_layers[model]},
            f"{model}: indexed confirmation layer changed",
        )
        require(
            int(indexed_progress_freeze["layer_selection_source"][model]["layer"])
            == int(representation[model]["running"]["layer"]),
            f"{model}: indexed control layer no longer matches aligned representation freeze",
        )
    require(float(g_primary["selected_minus_random_failure_rate"]) > float(selected_row(g_analysis, 8)["selected_minus_random_failure_rate"]), "Gemma K6 must remain the frozen primary over K8")
    require(
        {
            scope: int(row["selected_layer"])
            for scope, row in patch_scope_discovery.items()
        }
        == {"event_tail_w4": 0, "item_end_w1": 26, "item_span": 0},
        "Patch-scope discovery layers changed",
    )
    require(
        {
            scope: int(row["cell_count"])
            for scope, row in patch_scope_confirmation.items()
        }
        == {"event_tail_w4": 60, "item_end_w1": 60, "item_span": 60},
        "Patch-scope confirmation support changed",
    )
    require(
        abs(
            float(
                patch_scope_confirmation["item_span"][
                    "patched_first_known_city_donor_adoption_rate"
                ]
            )
            - 43 / 60
        )
        < 1e-12,
        "Item-span held-out generation result changed",
    )
    require(
        patch_scope_generation_audit["summary"]["donor_adoption_count"] == 43
        and patch_scope_generation_audit["manual_review"][
            "recap_only_false_positive_count"
        ]
        == 0,
        "Item-span manual generation audit changed",
    )
    require(
        int(item_span_l16_summary["cell_count"]) == 20
        and abs(
            float(
                item_span_l16_summary[
                    "patched_first_known_city_donor_adoption_rate"
                ]
            )
            - 0.8
        )
        < 1e-12
        and item_span_l16_generation_audit["manual_review"][
            "recap_only_false_positive_count"
        ]
        == 0,
        "L16 contextual item-span robustness result changed",
    )
    require(
        int(historical_event_tail_summary["cell_count"]) == 60
        and float(historical_event_tail_summary["positive_logodds_shift_rate"])
        == 1.0,
        "Historical L16 event-tail result changed",
    )
    for model in MODELS:
        ncc = ncc_supplement[model]
        require(ncc.get("status") == "PASS", f"{model} NCC analysis not sealed")
        require(ncc.get("discovery_seed_count") == 20, f"{model} NCC discovery seed drift")
        require(ncc.get("confirmation_seed_count") == 10, f"{model} NCC confirmation seed drift")
        require(ncc.get("outcome_blind") is True, f"{model} NCC not outcome blind")
        require(ncc.get("selection_rank_used") is False, f"{model} NCC uses selection rank")
        counter = unnumbered_counter[model]
        require(counter.get("status") == "PASS", f"{model} unnumbered counter not sealed")
        require(counter.get("phase") == "confirmation", f"{model} unnumbered counter phase drift")
        require(counter.get("seed_count") == 10, f"{model} unnumbered counter seed drift")
        require(
            counter.get("trace_panel_kind") == "teacher_forced_unnumbered_gold_bullets",
            f"{model} unnumbered panel provenance drift",
        )
        require(counter.get("natural_generation_claim_allowed") is False, f"{model} counter claim overreach")
        require(len(unnumbered_occurrence[model]) == 8, f"{model} counter occurrence support drift")
        for evidence_name, evidence in (("write", write[model]), ("commit-query", commit_query[model])):
            require(evidence.get("status") == "PASS", f"{model} {evidence_name} not PASS")
            require(evidence.get("discovery_seed_count") == 20, f"{model} {evidence_name} discovery seed drift")
            require(evidence.get("confirmation_seed_count") == 10, f"{model} {evidence_name} confirmation seed drift")
            require(evidence.get("outcome_blind") is True, f"{model} {evidence_name} not outcome blind")
            require(evidence.get("selection_rank_used") is False, f"{model} {evidence_name} uses selection_rank")
        require(
            aligned_targeted[model]["discovery"].get("targeted_to_count_pass")
            is True
            and aligned_targeted[model]["confirmation"].get(
                "targeted_to_count_pass"
            )
            is True,
            f"{model} targeted-retrieval propagation gate failed",
        )
        require(
            write[model]["discovery"].get(
                "targeted_counter_write_directional_pass"
            )
            is True
            and write[model]["confirmation"].get(
                "targeted_counter_write_strong_gate_pass"
            )
            is True,
            f"{model} targeted-retrieval-to-write gate failed",
        )
        require(
            commit_query[model]["discovery"].get("confirmation_eligible")
            is True
            and commit_query[model]["confirmation"].get("analysis_status")
            == "PROSPECTIVE_CONFIRMATION"
            and commit_query[model]["confirmation"].get(
                "strong_direct_gate_pass"
            )
            is True,
            f"{model} commit-to-next-query confirmation gate failed",
        )
        anchors = grammar_anchors[model]
        require(
            len(anchors) == 28,
            f"{model} exact-common grammar anchor panel must contain 28 traces",
        )
        require(
            all(bool(row["grammar_span_outcome_blind"]) for row in anchors),
            f"{model} grammar anchor panel is not outcome blind",
        )
        require(
            not any(bool(row["grammar_span_selection_rank_used"]) for row in anchors),
            f"{model} grammar anchor panel uses selection rank",
        )
        for phase, role, expected_total, expected_by_timing in (
            (
                "discovery",
                "development",
                19,
                {"rank_after_city": 9, "rank_before_city": 10},
            ),
            (
                "confirmation",
                "confirmation",
                9,
                {"rank_after_city": 4, "rank_before_city": 5},
            ),
        ):
            active = [
                row
                for row in anchors
                if str(row["stratified_ncc_seed_role"]) == role
            ]
            require(
                len(active) == expected_total,
                f"{model} {phase} grammar anchor count changed",
            )
            for timing, expected_count in expected_by_timing.items():
                require(
                    sum(row["grammar_span_timing_stratum"] == timing for row in active)
                    == expected_count,
                    f"{model} {phase} grammar timing balance changed",
                )
        require(grammar_span[model].get("status") == "PASS", f"{model} grammar span not PASS")
        require(grammar_span[model].get("discovery_seed_count") == 19, f"{model} grammar span discovery drift")
        require(grammar_span[model].get("confirmation_seed_count") == 9, f"{model} grammar span confirmation drift")
        require(walkthrough[model].get("status") == "PASS", f"{model} walkthrough not PASS")
        require(walkthrough[model].get("case_study_not_inferential") is True, f"{model} walkthrough scope drift")
        require(walkthrough[model].get("case_selected_by_outcome") is False, f"{model} walkthrough selected by outcome")
        require(walkthrough[model].get("answer_query_patched") is False, f"{model} walkthrough patched answer query")
    for model in MODELS:
        diagnostic = ncc_layerwise_diagnostic[model]
        require(diagnostic.get("status") == "PASS", f"{model} NCC layerwise diagnostic not sealed")
        require(diagnostic.get("model_label") == model, f"{model} NCC diagnostic label drift")
        require(
            diagnostic.get("inferential_status") == "post_hoc_diagnostic_not_confirmatory",
            f"{model} NCC diagnostic must remain explicitly post-hoc",
        )
        require(
            diagnostic.get("discovery_seed_count") == 20
            and diagnostic.get("confirmation_seed_count") == 10,
            f"{model} NCC diagnostic seed contract changed",
        )
        require(
            diagnostic["phase_audit"]["discovery"]["final_timing_histogram"]
            == {"rank_after_city": 10, "rank_before_city": 10}
            and diagnostic["phase_audit"]["confirmation"]["final_timing_histogram"]
            == {"rank_after_city": 5, "rank_before_city": 5},
            f"{model} NCC timing balance changed",
        )
        require(
            diagnostic["frozen_result_reproduction"]["selected_layer_matches"] is True
            and diagnostic["frozen_result_reproduction"]["within_tolerance_0_002"] is True,
            f"{model} frozen NCC result did not reproduce",
        )
        require(
            diagnostic["interpretation"]["cross_model_effect_size_comparison_allowed"] is False,
            f"{model} NCC diagnostic overclaims cross-model comparability",
        )
    expected_stratified_counts = {
        "Qwen3-8B": {
            "rank_after_city": (9, 4),
            "rank_before_city": (10, 5),
        },
        "Gemma4-E4B": {
            "rank_after_city": (9, 4),
            "rank_before_city": (10, 5),
        },
    }
    for model in MODELS:
        for timing in ("rank_after_city", "rank_before_city"):
            result = stratified_ncc[model][timing]
            expected_dev, expected_confirmation = expected_stratified_counts[model][timing]
            require(result.get("status") == "PASS", f"{model}/{timing} stratified NCC not sealed")
            require(
                result.get("model_label") == model
                and result.get("timing_branch") == timing,
                f"{model}/{timing} stratified NCC label drift",
            )
            require(
                result.get("development_seed_count") == expected_dev
                and result.get("confirmation_seed_count") == expected_confirmation,
                f"{model}/{timing} stratified NCC seed drift",
            )
            require(
                result.get("confirmation_used_for_fit_or_layer_selection") is False
                and result.get("outcome_blind") is True
                and result.get("selection_rank_used") is False,
                f"{model}/{timing} stratified NCC fitting contract changed",
            )
            require(
                result.get("capture_layer_rule")
                == "strictly_above_all_ablated_head_layers",
                f"{model}/{timing} stratified NCC causal reach changed",
            )
    for model in MODELS:
        for timing in ("rank_after_city", "rank_before_city"):
            result = targeted_logit_margin[model][timing]
            expected_dev, expected_confirmation = expected_stratified_counts[model][
                timing
            ]
            endpoint = result["primary_endpoint_result"]
            require(
                result.get("status") == "PASS"
                and result.get("model_label") == model
                and result.get("timing_branch") == timing,
                f"{model}/{timing} direct margin label or status drift",
            )
            require(
                result.get("schema_version")
                == "realistic_niah_v5_targeted_counter_logit_margin_analysis_v1"
                and result.get("primary_endpoint")
                == "final_answer_sequence_margin"
                and result.get("candidate_answer_scoring")
                == "full_autoregressive_sequence_log_probability_1_to_10",
                f"{model}/{timing} direct margin endpoint changed",
            )
            require(
                result.get("conditions")
                == [
                    "clean",
                    "selected_mask",
                    "random_mask_r1",
                    "random_mask_r2",
                    "random_mask_r3",
                ],
                f"{model}/{timing} direct margin arms changed",
            )
            require(
                result.get("outcome_blind_panel") is True
                and result.get("selection_rank_used") is False
                and result.get("no_decoder_fit_or_layer_selection") is True
                and result.get("raw_ncc_centroids_used") is False
                and result.get("raw_margins_pooled_across_branches") is False,
                f"{model}/{timing} direct margin analysis contract changed",
            )
            require(
                result.get("margin_gate_registered_before_logit_outcome_inspection")
                is True
                and result.get("confirmation_used_for_registration") is False
                and result.get("confirmation_status")
                == "registered_existing_split_after_ncc_inspection",
                f"{model}/{timing} direct margin registration status changed",
            )
            require(
                endpoint["development"]["seed_count"] == expected_dev
                and endpoint["confirmation"]["seed_count"]
                == expected_confirmation,
                f"{model}/{timing} direct margin seed contract changed",
            )
            require(
                endpoint["readout_validity"]["pass"] is True
                and float(endpoint["confirmation"]["clean_accuracy"]) == 1.0
                and float(endpoint["confirmation"]["clean_mean_margin"]) > 0.0,
                f"{model}/{timing} direct output readout validity failed",
            )
    require(q_free.get("complete_strong_gate_pass") is False, "Qwen free-running negative boundary changed")
    for geometry, gates in gemma_query_mediation.items():
        require(gates.get("geometry") == geometry, f"Gemma mediation geometry mismatch: {geometry}")
        require(gates.get("phase") == "discovery", f"Gemma mediation phase mismatch: {geometry}")
        require(gates.get("seed_count") == 20, f"Gemma mediation seed drift: {geometry}")
        require(gates.get("selection_rank_used") is False, f"Gemma mediation rank leakage: {geometry}")
        require(gates.get("geometry_pass") is False, f"Gemma mediation null changed: {geometry}")

    write_effects: dict[str, dict[str, Mapping[str, Any]]] = {}
    query_effects: dict[str, dict[str, Mapping[str, Any]]] = {}
    terminal_effects: dict[str, dict[str, Mapping[str, Any]]] = {}
    terminal_selected_geometry: dict[str, str] = {}
    for model in MODELS:
        write_rows = write[model]["confirmation"]["all_estimands"]
        write_effects[model] = {
            key: effect(write_rows, key)
            for key in (
                "selected_carrier_deformation",
                "selected_carrier_deformation_specificity",
                "selected_boundary_deformation",
                "selected_boundary_deformation_specificity",
                "clean_carrier_restoration",
                "restoration_position_specificity",
            )
        }
        query_rows = commit_query[model]["confirmation"]["estimands"]
        query_effects[model] = {
            "self": effect(query_rows, "full_commit_targeted_attention_vs_self_distance_1"),
            "orthogonal": effect(query_rows, "full_commit_targeted_attention_vs_orthogonal_distance_1"),
        }
        terminal_selected_geometry[model] = str(
            grammar_span[model]["discovery"]["largest_split_geometry"]
        )
        terminal_rows = grammar_span[model]["confirmation"]["primary_estimands"]
        terminal_effects[model] = {
            row["contrast"]: row
            for row in terminal_rows
            if row["geometry"] == terminal_selected_geometry[model]
        }
        require(
            set(terminal_effects[model])
            == {"restoration", "matched_random_specificity"},
            f"{model} missing confirmation effects for discovery-selected terminal geometry",
        )

    rep_groups = [
        (
            "running commit · confirmation balanced accuracy",
            {model: float(representation[model]["running"]["confirmation_logistic_balanced_accuracy"]) for model in MODELS},
        ),
        (
            "answer query · confirmation balanced accuracy",
            {model: float(representation[model]["final"]["confirmation_logistic_balanced_accuracy"]) for model in MODELS},
        ),
    ]
    write_damage_panels = [
        (
            "A · Selected mask vs clean",
            "mean downstream carrier RMS distance",
            [
                (SHORT[model], write_effects[model]["selected_carrier_deformation"])
                for model in MODELS
            ],
            0.07,
        ),
        (
            "B · Selected mask vs 3× random masks",
            "additional carrier RMS distance",
            [
                (
                    SHORT[model],
                    write_effects[model]["selected_carrier_deformation_specificity"],
                )
                for model in MODELS
            ],
            0.03,
        ),
    ]
    ncc_primary = {
        model: effect(
            ncc_supplement[model]["all_estimands"],
            "selected_correct_centroid_margin_loss",
        )
        for model in MODELS
    }
    ncc_random = {
        model: effect(
            ncc_supplement[model]["all_estimands"],
            "random_mean_correct_centroid_margin_loss",
        )
        for model in MODELS
    }
    ncc_specificity = {
        model: effect(
            ncc_supplement[model]["all_estimands"],
            "selected_vs_random_margin_loss_specificity",
        )
        for model in MODELS
    }
    ncc_panels = [
        (
            "A · Selected mask corrupts count geometry",
            "loss of correct-centroid margin",
            [(SHORT[model], ncc_primary[model]) for model in MODELS],
            40.0,
        ),
        (
            "B · Selected effect beyond random masks",
            "selected-minus-random margin loss",
            [(SHORT[model], ncc_specificity[model]) for model in MODELS],
            40.0,
        ),
    ]
    ncc_condition = {
        model: {
            str(row["condition"]): row
            for row in ncc_supplement[model]["condition_metrics"]
        }
        for model in MODELS
    }
    stratified_primary = {
        model: {
            timing: stratified_ncc[model][timing]["primary_endpoint_result"]
            for timing in ("rank_after_city", "rank_before_city")
        }
        for model in MODELS
    }

    def stratified_plot_effect(
        model: str, timing: str, estimand_key: str
    ) -> dict[str, Any]:
        endpoint = stratified_primary[model][timing]
        value = dict(endpoint[estimand_key])
        value["gate_pass"] = bool(
            endpoint["readout_validity"]["pass"]
            and float(value["ci_low"]) > 0.0
        )
        return value

    stratified_ncc_panels = [
        (
            "A · Selected mask vs clean",
            "selected loss / discovery OOF SD",
            [
                (
                    f"{SHORT[model]} · {'City→rank' if timing == 'rank_after_city' else 'Rank→city'}",
                    stratified_plot_effect(
                        model, timing, "standardized_primary_estimand"
                    ),
                )
                for model in MODELS
                for timing in ("rank_after_city", "rank_before_city")
            ],
            0.01,
        ),
        (
            "B · Selected effect beyond random masks",
            "specificity / discovery OOF SD",
            [
                (
                    f"{SHORT[model]} · {'City→rank' if timing == 'rank_after_city' else 'Rank→city'}",
                    stratified_plot_effect(
                        model, timing, "standardized_specificity_estimand"
                    ),
                )
                for model in MODELS
                for timing in ("rank_after_city", "rank_before_city")
            ],
            0.01,
        ),
    ]
    stratified_status_text = {
        "NO_DIRECTIONAL_SPECIFIC_EVIDENCE": "未见 directional-specific damage",
        "UNINTERPRETABLE_MARGIN_SHIFT_READOUT_VALIDITY_FAILURE": (
            "margin shift 为正，但 clean readout validity 失败"
        ),
        "VALID_READOUT_DIRECTIONAL_SPECIFIC_EVIDENCE": (
            "readout 有效；方向与 specificity 为正，但至少一条主区间跨 0"
        ),
        "VALID_READOUT_INTERVAL_DIRECTIONAL_SPECIFIC_SUPPORT": (
            "readout 有效且两个区间均高于 0"
        ),
    }
    stratified_ncc_result_table = table(
        (
            "Model / timing",
            "Primary endpoint / layer",
            "Discovery / confirmation seeds",
            "OOF BA / clean confirmation exact",
            "Clean mean margin / validity",
            "Selected loss: raw / SD units [95% CI]",
            "Selected−random: raw / SD units [95% CI]",
            "Qualified result",
        ),
        (
            (
                f"{SHORT[model]} · {'City→rank' if timing == 'rank_after_city' else 'Rank→city'}",
                (
                    f"{stratified_ncc[model][timing]['primary_endpoint']} · "
                    f"L{int(stratified_primary[model][timing]['selected_layer'])}"
                ),
                (
                    f"{int(stratified_ncc[model][timing]['development_seed_count'])} / "
                    f"{int(stratified_ncc[model][timing]['confirmation_seed_count'])}"
                ),
                (
                    f"{float(stratified_primary[model][timing]['selected_layer_discovery_metrics']['grouped_oof_ncc_balanced_accuracy']):.3f} / "
                    f"{float(next(row for row in stratified_primary[model][timing]['condition_metrics'] if row['condition'] == 'clean')['exact_accuracy']):.3f}"
                ),
                (
                    f"{float(stratified_primary[model][timing]['readout_validity']['clean_confirmation_mean_correct_centroid_margin']):+.1f} / "
                    f"{'PASS' if stratified_primary[model][timing]['readout_validity']['pass'] else 'FAIL'}"
                ),
                (
                    f"{float(stratified_primary[model][timing]['raw_primary_estimand']['mean_effect']):+.2f} / "
                    f"{ci(stratified_primary[model][timing]['standardized_primary_estimand'], 4)}"
                ),
                (
                    f"{float(stratified_primary[model][timing]['raw_specificity_estimand']['mean_effect']):+.2f} / "
                    f"{ci(stratified_primary[model][timing]['standardized_specificity_estimand'], 4)}"
                ),
                stratified_status_text[
                    stratified_primary[model][timing]["ncc_effect_status"]
                ],
            )
            for model in MODELS
            for timing in ("rank_after_city", "rank_before_city")
        ),
    )
    stratified_ncc_bank_table = table(
        (
            "Model",
            "City→rank bank",
            "Rank→city bank",
            "Bank overlap",
            "Causally reachable readout layers",
        ),
        (
            (
                SHORT[model],
                (
                    f"Top-{int(stratified_ncc_inputs[model]['banks']['rank_after_city']['bank_size'])}; "
                    f"{stratified_ncc_inputs[model]['banks']['rank_after_city']['source_grammar']}; P0"
                ),
                (
                    f"Top-{int(stratified_ncc_inputs[model]['banks']['rank_before_city']['bank_size'])}; "
                    f"{stratified_ncc_inputs[model]['banks']['rank_before_city']['source_grammar']}; "
                    f"{'post-marker' if model == 'Qwen3-8B' else 'P0'}"
                ),
                (
                    f"{int(stratified_ncc_inputs[model]['selected_bank_overlap']['head_count'])}/"
                    f"{int(stratified_ncc_inputs[model]['banks']['rank_after_city']['bank_size'])}"
                ),
                "L35" if model == "Qwen3-8B" else "L30–L41（discovery 内选层）",
            )
            for model in MODELS
        ),
    )
    direct_margin_primary = {
        model: {
            timing: targeted_logit_margin[model][timing]["primary_endpoint_result"]
            for timing in ("rank_after_city", "rank_before_city")
        }
        for model in MODELS
    }
    direct_margin_local = {
        model: targeted_logit_margin[model]["rank_after_city"][
            "endpoint_results"
        ]["local_rank_adjacent_sequence_margin"]
        for model in MODELS
    }

    def direct_margin_plot_effect(
        model: str, timing: str, estimand_key: str
    ) -> dict[str, Any]:
        endpoint = direct_margin_primary[model][timing]
        value = dict(endpoint["confirmation"][estimand_key])
        value["gate_pass"] = bool(
            endpoint["readout_validity"]["pass"]
            and float(value["ci_low"]) > 0.0
        )
        return value

    direct_margin_panels = [
        (
            "A · Selected mask vs clean",
            "final count-margin loss",
            [
                (
                    f"{SHORT[model]} · {'City→rank' if timing == 'rank_after_city' else 'Rank→city'}",
                    direct_margin_plot_effect(
                        model, timing, "selected_margin_loss"
                    ),
                )
                for model in MODELS
                for timing in ("rank_after_city", "rank_before_city")
            ],
            0.6,
        ),
        (
            "B · Selected effect beyond random masks",
            "selected−random count-margin loss",
            [
                (
                    f"{SHORT[model]} · {'City→rank' if timing == 'rank_after_city' else 'Rank→city'}",
                    direct_margin_plot_effect(
                        model, timing, "selected_vs_random_specificity"
                    ),
                )
                for model in MODELS
                for timing in ("rank_after_city", "rank_before_city")
            ],
            0.6,
        ),
    ]
    direct_margin_status_text = {
        "NO_DIRECTIONAL_SPECIFIC_EVIDENCE": "无 specificity",
        "VALID_READOUT_DIRECTIONAL_SPECIFIC_EVIDENCE": "directional-specific†",
        "INTERVAL_CONFIRMED_DIRECTIONAL_SPECIFIC_SUPPORT": (
            "interval-confirmed"
        ),
        "VALID_READOUT_INTERVAL_DIRECTIONAL_SPECIFIC_SUPPORT": (
            "interval-confirmed"
        ),
    }
    direct_margin_result_table = table(
        (
            "Model / timing",
            "n: dev / conf",
            "Clean M / acc.",
            "Dev loss / spec.",
            "Conf. loss [95% CI]",
            "Conf. spec. [95% CI]",
            "Gate",
        ),
        (
            (
                f"{SHORT[model]} · {'City→rank' if timing == 'rank_after_city' else 'Rank→city'}",
                (
                    f"{int(direct_margin_primary[model][timing]['development']['seed_count'])} / "
                    f"{int(direct_margin_primary[model][timing]['confirmation']['seed_count'])}"
                ),
                (
                    f"{float(direct_margin_primary[model][timing]['confirmation']['clean_mean_margin']):+.3f} / "
                    f"{float(direct_margin_primary[model][timing]['confirmation']['clean_accuracy']):.3f}"
                ),
                (
                    f"{float(direct_margin_primary[model][timing]['development']['selected_margin_loss']['mean_effect']):+.3f} / "
                    f"{float(direct_margin_primary[model][timing]['development']['selected_vs_random_specificity']['mean_effect']):+.3f}"
                ),
                ci(
                    direct_margin_primary[model][timing]["confirmation"][
                        "selected_margin_loss"
                    ]
                ),
                ci(
                    direct_margin_primary[model][timing]["confirmation"][
                        "selected_vs_random_specificity"
                    ]
                ),
                direct_margin_status_text[
                    direct_margin_primary[model][timing]["effect_status"]
                ],
            )
            for model in MODELS
            for timing in ("rank_after_city", "rank_before_city")
        ),
    )
    ncc_result_table = table(
        (
            "Model",
            "Frozen layer",
            "Discovery grouped-OOF NCC BA",
            "Confirmation exact: clean → selected",
            "Selected margin loss",
            "Random mean loss",
            "Selected−random",
        ),
        (
            (
                SHORT[model],
                f"L{int(ncc_supplement[model]['selected_layer'])}",
                f"{float(ncc_supplement[model]['selected_layer_discovery_metrics']['mean_timing_oof_ncc_balanced_accuracy']):.3f}",
                f"{float(ncc_condition[model]['clean']['exact_accuracy']):.2f} → {float(ncc_condition[model]['selected_mask']['exact_accuracy']):.2f}",
                ci(ncc_primary[model]),
                ci(ncc_random[model]),
                ci(ncc_specificity[model]),
            )
            for model in MODELS
        ),
    )
    ncc_diagnostic_by_model_layer_timing = {
        model: {
            (int(row["layer"]), str(row["timing"])): row
            for row in ncc_layerwise_diagnostic[model]["layer_timing_rows"]
        }
        for model in MODELS
    }
    qwen_ncc_diagnostic_by_layer_timing = ncc_diagnostic_by_model_layer_timing["Qwen3-8B"]
    gemma_ncc_diagnostic_by_layer_timing = ncc_diagnostic_by_model_layer_timing["Gemma4-E4B"]

    def count_histogram_text(model: str) -> str:
        histogram = ncc_layerwise_diagnostic[model]["phase_audit"]["confirmation"][
            "gold_count_histogram"
        ]
        return " · ".join(
            f"N={count}×{histogram[count]}" for count in sorted(histogram, key=int)
        )

    ncc_correspondence_table = table(
        ("Audit dimension", "Qwen3-8B", "Gemma4-E4B", "Cross-model reading"),
        (
            (
                "Offline reproduction",
                (
                    f"L{int(ncc_layerwise_diagnostic['Qwen3-8B']['frozen_result_reproduction']['selected_layer_recomputed'])}; "
                    f"max |Δ|={float(ncc_layerwise_diagnostic['Qwen3-8B']['frozen_result_reproduction']['maximum_absolute_difference']):.6f}"
                ),
                (
                    f"L{int(ncc_layerwise_diagnostic['Gemma4-E4B']['frozen_result_reproduction']['selected_layer_recomputed'])}; "
                    f"max |Δ|={float(ncc_layerwise_diagnostic['Gemma4-E4B']['frozen_result_reproduction']['maximum_absolute_difference']):.6f}"
                ),
                "PASS：各模型冻结结果可复现",
            ),
            (
                "Seed/timing contract",
                "20 discovery（10/10）· 10 confirmation（5/5）",
                "20 discovery（10/10）· 10 confirmation（5/5）",
                "协议与 timing 配额对齐",
            ),
            (
                "Confirmation gold N",
                count_histogram_text("Qwen3-8B"),
                count_histogram_text("Gemma4-E4B"),
                "不匹配；不是跨模型 paired cohort",
            ),
            (
                "Frozen targeted bank",
                (
                    f"Top-{int(ncc_layerwise_diagnostic['Qwen3-8B']['bank_selection_contract']['bank_size'])}; "
                    f"{ncc_layerwise_diagnostic['Qwen3-8B']['bank_selection_contract']['selection_target_grammar_class']}; "
                    f"{ncc_layerwise_diagnostic['Qwen3-8B']['bank_selection_contract']['selection_anchor_role']}"
                ),
                (
                    f"Top-{int(ncc_layerwise_diagnostic['Gemma4-E4B']['bank_selection_contract']['bank_size'])}; "
                    f"{ncc_layerwise_diagnostic['Gemma4-E4B']['bank_selection_contract']['selection_target_grammar_class']}; "
                    f"{ncc_layerwise_diagnostic['Gemma4-E4B']['bank_selection_contract']['selection_anchor_role']}"
                ),
                "功能类比，但 bank 剂量与定位点不同",
            ),
            (
                "Frozen layer / causally active selected heads",
                (
                    f"L{int(ncc_layerwise_diagnostic['Qwen3-8B']['selection_audit']['original_pooled_timing_layer'])}; "
                    f"{int(ncc_layerwise_diagnostic['Qwen3-8B']['selection_audit']['original_pooled_timing_active_selected_heads'])}/"
                    f"{int(ncc_layerwise_diagnostic['Qwen3-8B']['bank_selection_contract']['bank_size'])}"
                ),
                (
                    f"L{int(ncc_layerwise_diagnostic['Gemma4-E4B']['selection_audit']['original_pooled_timing_layer'])}; "
                    f"{int(ncc_layerwise_diagnostic['Gemma4-E4B']['selection_audit']['original_pooled_timing_active_selected_heads'])}/"
                    f"{int(ncc_layerwise_diagnostic['Gemma4-E4B']['bank_selection_contract']['bank_size'])}"
                ),
                "有效 intervention dose 不对齐",
            ),
            (
                "NCC coordinates",
                "每 timing 独立 standardize + PCA-16 + centroids",
                "每 timing 独立 standardize + PCA-16 + centroids",
                "方法相同；raw margins 不同尺度",
            ),
        ),
    )
    ncc_frozen_timing_table = table(
        (
            "Model",
            "Frozen layer",
            "Timing",
            "Discovery OOF BA",
            "Clean margin mean",
            "Selected loss",
            "Selected−random",
            "Positive seeds",
        ),
        (
            (
                SHORT[model],
                f"L{int(ncc_supplement[model]['selected_layer'])}",
                "City→rank" if timing == "rank_after_city" else "Rank→city",
                f"{float(ncc_diagnostic_by_model_layer_timing[model][(int(ncc_supplement[model]['selected_layer']), timing)]['discovery_grouped_oof_balanced_accuracy']):.3f}",
                f"{float(ncc_diagnostic_by_model_layer_timing[model][(int(ncc_supplement[model]['selected_layer']), timing)]['clean_margin_mean']):.1f}",
                f"{float(ncc_diagnostic_by_model_layer_timing[model][(int(ncc_supplement[model]['selected_layer']), timing)]['selected_margin_loss_mean']):+.2f}",
                f"{float(ncc_diagnostic_by_model_layer_timing[model][(int(ncc_supplement[model]['selected_layer']), timing)]['selected_vs_random_specificity_mean']):+.2f}",
                f"{int(ncc_diagnostic_by_model_layer_timing[model][(int(ncc_supplement[model]['selected_layer']), timing)]['selected_margin_loss_positive_n'])}/5",
            )
            for model in MODELS
            for timing in ("rank_after_city", "rank_before_city")
        ),
    )
    qwen_ncc_l23_after = qwen_ncc_diagnostic_by_layer_timing[(23, "rank_after_city")]
    qwen_ncc_l23_before = qwen_ncc_diagnostic_by_layer_timing[(23, "rank_before_city")]
    gemma_ncc_diagnostic_layers = (16, 18, 24, 27, 28, 30, 34)
    gemma_ncc_causal_reach = {
        16: "mask 尚未可达",
        18: "L17 heads 首次可达",
        24: "仅 L17 heads",
        27: "仅 L17 heads；discovery-best",
        28: "仅 L17 heads；相邻稳定层",
        30: "L17+L29 首次可达",
        34: "full Top-6；冻结 pooled primary",
    }
    for layer in gemma_ncc_diagnostic_layers:
        for timing in ("rank_after_city", "rank_before_city"):
            require(
                (layer, timing) in gemma_ncc_diagnostic_by_layer_timing,
                f"Missing Gemma NCC diagnostic L{layer} {timing}",
            )
    gemma_ncc_layerwise_table = table(
        (
            "Layer",
            "query-mask causal reach",
            "City→rank: OOF BA / selected loss",
            "Rank→city: OOF BA / selected loss",
            "Rank→city selected−random",
            "Rank→city positive seeds",
        ),
        (
            (
                f"L{layer}",
                gemma_ncc_causal_reach[layer],
                (
                    f"{float(gemma_ncc_diagnostic_by_layer_timing[(layer, 'rank_after_city')]['discovery_grouped_oof_balanced_accuracy']):.3f} / "
                    f"{float(gemma_ncc_diagnostic_by_layer_timing[(layer, 'rank_after_city')]['selected_margin_loss_mean']):+.2f}"
                ),
                (
                    f"{float(gemma_ncc_diagnostic_by_layer_timing[(layer, 'rank_before_city')]['discovery_grouped_oof_balanced_accuracy']):.3f} / "
                    f"{float(gemma_ncc_diagnostic_by_layer_timing[(layer, 'rank_before_city')]['selected_margin_loss_mean']):+.2f}"
                ),
                f"{float(gemma_ncc_diagnostic_by_layer_timing[(layer, 'rank_before_city')]['selected_vs_random_specificity_mean']):+.2f}",
                f"{int(gemma_ncc_diagnostic_by_layer_timing[(layer, 'rank_before_city')]['selected_margin_loss_positive_n'])}/5",
            )
            for layer in gemma_ncc_diagnostic_layers
        ),
    )
    gemma_ncc_l27_before = gemma_ncc_diagnostic_by_layer_timing[(27, "rank_before_city")]
    gemma_ncc_l28_before = gemma_ncc_diagnostic_by_layer_timing[(28, "rank_before_city")]
    gemma_ncc_l34_after = gemma_ncc_diagnostic_by_layer_timing[(34, "rank_after_city")]
    gemma_ncc_l34_before = gemma_ncc_diagnostic_by_layer_timing[(34, "rank_before_city")]
    counter_margin = {
        model: effect(unnumbered_counter[model]["primary_estimands"], "target_margin_gain")
        for model in MODELS
    }
    counter_exact = {
        model: effect(unnumbered_counter[model]["primary_estimands"], "exact_accuracy_gain")
        for model in MODELS
    }
    counter_mae = {
        model: effect(unnumbered_counter[model]["primary_estimands"], "mae_reduction")
        for model in MODELS
    }
    counter_result_table = table(
        (
            "Model",
            "Frozen source layer",
            "Scrub → patched exact",
            "Exact gain",
            "MAE reduction",
            "Target-margin gain",
            "Old-HTML magnitude gate",
        ),
        (
            (
                SHORT[model],
                f"L{int(unnumbered_counter[model]['selected_layer'])}",
                f"{float(unnumbered_counter[model]['selected_layer_metrics']['baseline_exact_accuracy']):.3f} → {float(unnumbered_counter[model]['selected_layer_metrics']['patched_exact_accuracy']):.3f}",
                ci(counter_exact[model]),
                ci(counter_mae[model]),
                ci(counter_margin[model]),
                "PASS" if unnumbered_counter[model]["old_html_internal_counter_magnitude_pass"] else "not met",
            )
            for model in MODELS
        ),
    )
    write_mediation_panels = [
        (
            "A · Damage reaches the later commit",
            "final-boundary RMS distance from clean",
            [
                (SHORT[model], write_effects[model]["selected_boundary_deformation"])
                for model in MODELS
            ],
            0.09,
        ),
        (
            "B · Clean carrier rescues that damage",
            "reduction in final-boundary RMS distance",
            [
                (SHORT[model], write_effects[model]["clean_carrier_restoration"])
                for model in MODELS
            ],
            0.06,
        ),
    ]
    write_recovery_fraction = {
        model: float(write_effects[model]["clean_carrier_restoration"]["mean_effect"])
        / float(write_effects[model]["selected_boundary_deformation"]["mean_effect"])
        for model in MODELS
    }
    query_groups = [
        ("full commit vs self patch", {model: float(query_effects[model]["self"]["mean_effect"]) for model in MODELS}),
        ("full commit vs orthogonal", {model: float(query_effects[model]["orthogonal"]["mean_effect"]) for model in MODELS}),
    ]
    source_groups = [
        (
            "clean exact count",
            {model: float(token_evidence[model]["answer"]["trace_items"]["clean"]["exact_rate"]) for model in MODELS},
        ),
        (
            "trace blank exact count",
            {model: float(token_evidence[model]["answer"]["trace_items"]["trace_all_blank"]["exact_rate"]) for model in MODELS},
        ),
    ]
    terminal_groups = [
        (
            "discovery-selected semantic restoration vs corrupt",
            {
                model: float(terminal_effects[model]["restoration"]["mean_effect"])
                for model in MODELS
            },
        ),
        (
            "selected semantic state vs equal-token ordinary state",
            {
                model: float(
                    terminal_effects[model]["matched_random_specificity"]["mean_effect"]
                )
                for model in MODELS
            },
        ),
    ]
    answer_query_layer_plot = answer_query_layer_sweep_svg(answer_query_layer_rows)

    def probability_ci(row: Mapping[str, Any], metric: str) -> str:
        return (
            f"{float(row[metric]):.3f} "
            f"[{float(row[f'{metric}_ci95_low']):.3f}, "
            f"{float(row[f'{metric}_ci95_high']):.3f}]"
        )

    answer_query_final_rows = {
        model: answer_query_layer_rows[model][-1] for model in MODELS
    }
    answer_query_onsets = {
        model: answer_query_layer_audits[model]["descriptive_onset_layer"].get(model)
        for model in MODELS
    }
    answer_query_layer_table = table(
        ("Model", "Frozen layer grid", "Pairs / seed clusters", "Descriptive onset", "Final sampled layer"),
        (
            (
                SHORT[model],
                ", ".join(f"L{layer}" for layer in ANSWER_QUERY_LAYER_GRID[model]),
                f"{ANSWER_QUERY_PAIR_COUNT[model]} / 10",
                (
                    "none"
                    if answer_query_onsets[model] is None
                    else f"L{int(answer_query_onsets[model])}"
                ),
                probability_ci(answer_query_final_rows[model], "full_donor_adoption"),
            )
            for model in MODELS
        ),
        class_name="compact-table",
    )

    def relay_effect(model: str, effect_name: str) -> dict[str, Any]:
        row = terminal_relay_summary["models"][model]["effects"][effect_name]
        return {
            "mean_effect": float(row["estimate"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
            "gate_pass": float(row["ci_low"]) > 0.0,
        }

    terminal_relay_panels = [
        (
            "A · Earlier-state patch damage and remaining damage",
            "correct-count sequence-margin difference",
            [
                (
                    f"{SHORT[model]} · natural damage",
                    relay_effect(model, "patch_damage_natural"),
                )
                for model in MODELS
            ]
            + [
                (
                    f"{SHORT[model]} · after clean suffix reset",
                    relay_effect(model, "patch_damage__post_terminal_suffix"),
                )
                for model in MODELS
            ],
            0.25,
        ),
        (
            "B · Specific mediation by downstream relay state",
            "patch-by-clean-reset interaction",
            [
                (
                    f"{SHORT[model]} · post-terminal suffix",
                    relay_effect(model, "specific_mediation__post_terminal_suffix"),
                )
                for model in MODELS
            ]
            + [
                (
                    f"{SHORT[model]} · answer query only",
                    relay_effect(model, "specific_mediation__answer_query"),
                )
                for model in MODELS
            ],
            0.25,
        ),
    ]

    def relay_ci(model: str, effect_name: str) -> str:
        row = terminal_relay_summary["models"][model]["effects"][effect_name]
        return (
            f"{float(row['estimate']):+.3f} "
            f"[{float(row['ci_low']):+.3f}, {float(row['ci_high']):+.3f}]"
        )

    terminal_relay_table = table(
        (
            "Model",
            "Natural patch damage",
            "Suffix-specific mediation",
            "Answer-query-specific mediation",
            "Suffix explained fraction",
            "Residual after suffix reset",
        ),
        (
            (
                SHORT[model],
                relay_ci(model, "patch_damage_natural"),
                relay_ci(model, "specific_mediation__post_terminal_suffix"),
                relay_ci(model, "specific_mediation__answer_query"),
                (
                    f"{100 * float(terminal_relay_summary['models'][model]['effects']['post_terminal_suffix_explained_fraction']['estimate']):.1f}% "
                    f"[{100 * float(terminal_relay_summary['models'][model]['effects']['post_terminal_suffix_explained_fraction']['ci_low']):.1f}%, "
                    f"{100 * float(terminal_relay_summary['models'][model]['effects']['post_terminal_suffix_explained_fraction']['ci_high']):.1f}%]"
                ),
                relay_ci(model, "patch_damage__post_terminal_suffix"),
            )
            for model in MODELS
        ),
        class_name="compact-table",
    )
    mediation_panels = []
    for panel_title, estimand, floor in (
        ("A · Full-state city-log-odds effect", "full_state_effect_intact", 1.0),
        ("B · Selected-mask interaction", "full_selected_mask_interaction", 0.1),
        ("C · Selected-head pre-O restoration", "full_head_output_restore", 0.1),
        ("D · Selected vs random specificity", "full_selected_vs_random_specificity", 0.1),
    ):
        mediation_panels.append(
            (
                panel_title,
                "donor−receiver query city log-odds",
                [
                    (geometry, effect(gemma_query_mediation[geometry]["estimands"], estimand))
                    for geometry in ("endpoint", "suffix4", "suffix8")
                ],
                floor,
            )
        )
    q_free_rows = q_free["confirmation"]["all_estimands"]
    q_free_panels = [
        (
            "A · Candidate-distribution recovery",
            "reduction in TV distance",
            [("clean carrier restore", effect(q_free_rows, "selected_clean_state_distribution_recovery"))],
            1e-12,
        ),
        (
            "B · Expected-count recovery",
            "expected count change",
            [("clean carrier restore", effect(q_free_rows, "selected_clean_state_expected_count_recovery"))],
            1e-12,
        ),
        (
            "C · Correct-count margin recovery",
            "log-probability margin",
            [
                (
                    "clean carrier restore",
                    next(
                        row
                        for row in q_free_rows
                        if row["estimand"] == "selected_clean_state_restoration"
                        and row["outcome"] == "correct_count_margin"
                    ),
                )
            ],
            0.6,
        ),
        (
            "D · Greedy exact-count recovery",
            "paired exact-count rate",
            [
                (
                    "clean carrier restore",
                    next(
                        row
                        for row in q_free_rows
                        if row["estimand"] == "selected_clean_state_restoration"
                        and row["outcome"] == "exact_count"
                    ),
                )
            ],
            0.1,
        ),
    ]

    custom_css = """
.report-note{max-width:920px;color:#475467}.status-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:22px 0}.status-card{padding:18px;border:1px solid var(--line);background:#fbfcfe}.status-card h3{margin:0 0 8px}.status-card p{margin:6px 0;font-size:14px}.status-good{color:#075e58;font-weight:750}.status-open{color:#9a4b00;font-weight:750}.chain-figure{display:block;width:100%;height:auto;border:1px solid var(--line);background:#fbfcfe}.chain-title{fill:#172033;font-size:13px;font-weight:750}.chain-sub{fill:#667085;font-size:11px}.chain-model{fill:#344054;font-size:12px;font-weight:750}.chain-status{font-size:11px;font-weight:750}.mini-model{font-size:11px;font-weight:800}.metric-strip{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:20px 0}.metric{padding:15px;border-top:3px solid var(--teal);background:#f8fafc}.metric strong,.metric span{display:block}.metric strong{font-size:22px}.metric span{color:#667085;font-size:12px}.negative-result{padding:17px 19px;border-left:4px solid var(--amber);background:#fff8eb}.audit-list{font-size:12px;color:#667085;overflow-wrap:anywhere}.compact-table td,.compact-table th{padding:7px 8px}.walkthrough-callout{display:grid;grid-template-columns:1fr 1fr;gap:14px}.walkthrough-callout>div{padding:15px;border:1px solid var(--line);background:#fbfcfe}.edge-roadmap{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;margin:18px 0 28px;border:1px solid var(--line);background:var(--line)}.edge-roadmap>div{padding:16px 17px;background:#fbfcfe}.edge-roadmap strong,.edge-roadmap span{display:block}.edge-roadmap span{margin-bottom:5px;color:#0f766e;font:800 11px/1.3 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.condition-list{margin:14px 0 22px;border-top:1px solid var(--line)}.condition-row{display:grid;grid-template-columns:170px 1fr 1fr;gap:16px;padding:12px 4px;border-bottom:1px solid var(--line);font-size:13px;line-height:1.55}.condition-row strong{color:#172033}.condition-row span{color:#475467}.plain-language{margin:16px 0;padding:16px 18px;border:1px solid #b8d7d1;background:#f4fbf9}.plain-language strong{color:#075e58}.scope-compare{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:18px 0}.scope-compare>div{padding:16px 18px;border-top:3px solid #98a2b3;background:#f8fafc}.scope-compare>div:first-child{border-top-color:#0f766e}.scope-compare h4{margin:0 0 7px}.scope-compare p{margin:6px 0;font-size:13px}.appendix-e-index{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:14px 0 20px}.appendix-e-index div{padding:12px 14px;border-left:3px solid #46758f;background:#f5f8fb;font-size:12px;line-height:1.55}.appendix-e-index strong{display:block;color:#172033}.appendix-e-figure{scroll-margin-top:24px}.appendix-e-figure .attention-atlas-frame>svg{display:block;width:100%;height:auto;margin:0 auto}.appendix-e-figure .attention-atlas-frame>.ordinal-map{min-width:1100px}.appendix-e-proof{margin:10px 0;color:#475467;font-size:12px}
.definition-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:18px 0 28px}.definition{padding:15px 16px;border:1px solid var(--line);background:#fbfcfe}.definition dt{font-weight:800;color:#172033;margin-bottom:6px}.definition dd{margin:0;color:#475467;font-size:13px;line-height:1.62}.experiment-frame{margin:18px 0 26px;border:1px solid var(--line);background:#fff}.experiment-frame>div{padding:15px 18px;border-bottom:1px solid var(--line)}.experiment-frame>div:last-child{border-bottom:0}.experiment-label{display:inline-block;min-width:88px;color:#0f766e;font-size:11px;font-weight:850;letter-spacing:.07em;text-transform:uppercase}.formula{display:block;margin:9px 0 0;padding:15px 18px;background:#f5f8fb;border-left:3px solid #46758f;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px;overflow-x:auto}.figure-primer{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;margin:16px 0 8px;background:var(--line);border:1px solid var(--line)}.figure-primer>div{padding:13px 15px;background:#f8fafc;font-size:12px;line-height:1.55}.figure-primer strong{display:block;margin-bottom:4px;color:#172033}.paper-chart{display:block;width:100%;height:auto;border:1px solid var(--line);background:#fff}.three-d{margin:12px 0 0;border:1px solid var(--line);background:linear-gradient(#fbfcfe,#f5f8fb)}.three-d-head{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:13px 15px;border-bottom:1px solid var(--line)}.three-d-controls{display:flex;align-items:end;justify-content:flex-end;gap:10px;flex-wrap:wrap}.three-d-head label,.manifold-panel-head label{color:#475467;font-size:12px;font-weight:700}.three-d-head select,.three-d-head button,.manifold-panel-head select{display:block;margin-top:5px;padding:7px 9px;border:1px solid #b8c1cf;background:#fff;color:#172033;font:inherit}.three-d-head button{cursor:pointer}.manifold-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;background:var(--line)}.manifold-panel{min-width:0;padding:0;background:#fbfcfe;border:0}.manifold-panel-head{display:flex;align-items:end;justify-content:space-between;gap:12px;padding:12px 14px;border-bottom:1px solid var(--line)}.manifold-panel-head strong,.manifold-panel-head span{display:block}.manifold-panel-head span{margin-top:3px;color:#667085;font-size:12px}.three-d canvas{display:block;width:100%;height:455px;cursor:grab;touch-action:none}.three-d canvas:active{cursor:grabbing}.manifold-stats{min-height:45px;margin:0;padding:9px 13px;border-top:1px solid var(--line);color:#667085;font:11px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.attention-pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}.attention-pair figure{margin:0}.attention-pair img{display:block;width:100%;height:auto;border:1px solid var(--line);background:#fff}.attention-atlas-stack{display:grid;grid-template-columns:1fr;gap:26px;margin-top:12px}.attention-atlas-stack figure{margin:0;padding:16px;border:1px solid var(--line);background:#fff}.attention-atlas-frame{width:100%;overflow-x:auto}.attention-atlas-frame .head-map{display:block;width:100%;min-width:900px;height:auto;margin:0 auto}.attention-switcher{margin:14px 0;padding:16px;border:1px solid var(--line);background:#fbfcfe}.attention-select{display:block;max-width:680px;color:#344054;font-size:12px;font-weight:750}.attention-select select{display:block;width:100%;margin-top:7px;padding:9px 11px;border:1px solid #b8c1cf;background:#fff;color:#172033;font:inherit}.attention-example-panel{margin-top:16px}.attention-example-svg{overflow-x:auto}.attention-example-svg svg{display:block;width:100%;min-width:900px;height:auto;margin:0 auto}.map-meta{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:0 0 9px;color:#667085;font-size:12px}.map-meta strong{color:#172033;font-size:14px}.head-map{display:block;width:100%;height:auto;background:#fff}.term-note{font-size:12px;color:#667085}.qualification{padding:16px 18px;border-left:4px solid #0f766e;background:#f0f9f7}.appendix-block{margin-top:22px}.appendix-block summary{cursor:pointer;font-weight:800}.section-conclusion{margin-top:22px;padding:17px 19px;background:#eef7f5;border-left:4px solid #0f766e}.section-conclusion strong{color:#075e58}
.attention-pair svg{display:block;width:100%;height:auto;border:1px solid var(--line);background:#fff}
.core-claim{margin:24px 0;padding:22px 24px;border-top:4px solid #0f766e;background:#f0f9f7;font-size:17px;line-height:1.72}.core-claim strong{color:#075e58}.claim-tier-grid{display:grid;grid-template-columns:1.15fr 1fr 1fr;gap:1px;margin:20px 0;border:1px solid var(--line);background:var(--line)}.claim-tier-grid>div{padding:18px;background:#fff}.claim-tier-grid h3{margin:0 0 8px;font-size:15px}.claim-tier-grid p{margin:7px 0;color:#475467;font-size:13px}.claim-tier-grid>div:first-child{box-shadow:inset 0 3px #0f766e}.claim-tier-grid>div:nth-child(2){box-shadow:inset 0 3px #46758f}.claim-tier-grid>div:last-child{box-shadow:inset 0 3px #9a4b00}.scope-layer-figure{overflow-x:auto}.scope-layer-figure svg{display:block;width:100%;min-width:820px;height:auto;border:1px solid var(--line);background:#fff}.evidence-ladder{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;margin:18px 0;background:var(--line);border:1px solid var(--line)}.evidence-ladder>div{padding:15px 16px;background:#fbfcfe}.evidence-ladder span,.evidence-ladder strong{display:block}.evidence-ladder span{color:#0f766e;font:800 11px/1.3 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.evidence-ladder strong{margin:6px 0;font-size:15px}.evidence-ladder p{margin:0;color:#667085;font-size:12px}.evidence-ledger td:first-child{font-weight:750;color:#172033}.evidence-ledger td:last-child{color:#667085}.appendix-sequence{margin-top:20px}.appendix-sequence details{margin:12px 0;padding:0;border-top:1px solid var(--line)}.appendix-sequence summary{cursor:pointer;padding:13px 2px;font-weight:800}.appendix-sequence details>div{padding:0 2px 12px}.main-note{margin:16px 0;padding:14px 17px;border-left:3px solid #46758f;background:#f5f8fb;color:#344054;font-size:13px}.audit-badge{display:inline-block;margin:2px 5px 2px 0;padding:3px 7px;border:1px solid #b8d7d1;background:#f4fbf9;color:#075e58;font:750 11px/1.3 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.reading-contract{margin:22px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.contract-row{display:grid;grid-template-columns:150px 1fr;gap:18px;padding:13px 2px;border-bottom:1px solid var(--line)}.contract-row:last-child{border-bottom:0}.contract-row strong{color:#172033}.contract-row span{color:#475467;font-size:13px;line-height:1.65}.mirror-table td:nth-child(2){font-weight:750;color:#075e58}.mirror-table td:last-child{color:#667085}.subsection-conclusion{margin:16px 0 24px;padding:12px 15px;border-left:3px solid #46758f;background:#f5f8fb;color:#344054;font-size:13px;line-height:1.65}.subsection-conclusion strong{color:#244b62}.result-analysis{margin:16px 0}.result-analysis>p{margin:8px 0}.appendix-method{margin:12px 0 18px}.appendix-method p{margin:7px 0}.appendix-method strong{color:#172033}.completion-note{margin:14px 0;padding:13px 16px;border-left:3px solid #0f766e;background:#f4fbf9;color:#344054;font-size:13px}.figure-status{display:inline-block;margin-right:8px;padding:2px 7px;border:1px solid #b8c1cf;color:#475467;font:750 10px/1.3 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;text-transform:uppercase;letter-spacing:.04em}
.attention-atlas-stack figure,.appendix-e-figure{min-width:0;max-width:100%;box-sizing:border-box}.attention-atlas-frame{min-width:0;max-width:100%}.appendix-sequence details,.appendix-sequence details>div{min-width:0;max-width:100%}
.parser-contract{margin:30px 0 8px;padding-top:22px;border-top:1px solid var(--line);scroll-margin-top:72px}.parser-contract-head{display:grid;grid-template-columns:190px minmax(0,1fr);gap:22px;align-items:start;margin-bottom:14px}.parser-contract-kicker{color:#0f766e;font:850 11px/1.35 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;letter-spacing:.08em;text-transform:uppercase}.parser-contract-head h3{margin:4px 0 8px;font-size:22px}.parser-contract-head p{margin:0;color:#475467;font-size:13px;line-height:1.68}.parser-disclosure{margin:0;border-top:1px solid var(--line);background:#fff}.parser-disclosure:last-child{border-bottom:1px solid var(--line)}.parser-disclosure summary{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:14px;align-items:center;padding:15px 3px;cursor:pointer;color:#172033;font-weight:800;list-style:none}.parser-disclosure summary::-webkit-details-marker{display:none}.parser-disclosure summary::after{content:"+";display:grid;width:24px;height:24px;place-items:center;border:1px solid #b8c1cf;border-radius:50%;color:#0f766e;font:700 16px/1 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.parser-disclosure[open] summary::after{content:"−"}.parser-disclosure-body{padding:1px 3px 20px;color:#344054}.parser-disclosure-body>p:first-child{margin-top:2px}.parser-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 22px;margin:14px 0;border-top:1px solid var(--line)}.parser-grid>div{padding:13px 0;border-bottom:1px solid var(--line);font-size:13px;line-height:1.62}.parser-grid strong{display:block;margin-bottom:4px;color:#172033}.parser-flow{display:flex;align-items:stretch;gap:0;margin:15px 0;overflow-x:auto}.parser-flow>div{min-width:150px;flex:1;padding:13px 14px;border:1px solid var(--line);background:#f8fafc;font-size:12px;line-height:1.55}.parser-flow>span{display:grid;min-width:30px;place-items:center;color:#667085}.parser-code{display:block;margin:11px 0;padding:12px 14px;border-left:3px solid #46758f;background:#f5f8fb;font:12px/1.62 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;overflow-x:auto}.parser-warning{margin:14px 0 0;padding:13px 15px;border-left:3px solid var(--amber);background:#fff8eb;color:#7a3d00;font-size:12px;line-height:1.62}.parser-table td,.parser-table th{padding:8px 9px;font-size:12px;vertical-align:top}.parser-table td:first-child{font-weight:800;color:#172033}.parser-table code{white-space:normal}.parser-tag{display:inline-block;margin:2px 5px 2px 0;padding:3px 7px;border:1px solid #b8d7d1;background:#f4fbf9;color:#075e58;font:750 10px/1.3 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
@media print{.attention-pair{display:block}.attention-pair figure{break-inside:avoid-page;margin:0 0 20px}.attention-pair figure .head-map{width:auto;max-width:100%;max-height:620px;margin:0 auto}.attention-atlas-stack figure{break-inside:avoid-page}.attention-atlas-frame .head-map{min-width:0}.three-d{break-inside:avoid-page}.three-d canvas{height:430px}.formula{white-space:normal}.attention-switcher{break-inside:avoid-page}}
@media(max-width:900px){.manifold-grid{grid-template-columns:1fr}}
@media(max-width:820px){.status-grid,.walkthrough-callout,.metric-strip,.definition-grid,.attention-pair,.figure-primer,.edge-roadmap,.scope-compare,.appendix-e-index,.claim-tier-grid,.evidence-ladder{grid-template-columns:1fr}.contract-row{grid-template-columns:1fr;gap:4px}.condition-row{grid-template-columns:1fr;gap:4px}.three-d-head{align-items:flex-start;flex-direction:column}.three-d-controls{justify-content:flex-start}.three-d canvas{height:430px}.chain-figure{min-width:850px}.chain-scroll{overflow-x:auto}}
@media(max-width:820px){.parser-contract-head,.parser-grid{grid-template-columns:1fr}.parser-contract-head{gap:4px}.parser-flow{flex-direction:column;overflow-x:visible}.parser-flow>div{min-width:0}.parser-flow>span{min-width:0;min-height:26px;transform:rotate(90deg)}.parser-disclosure summary{padding:14px 0}.parser-disclosure-body{padding-left:0;padding-right:0}}
"""
    css = extract_css(args.reference_report) + custom_css
    generated = datetime.now(timezone.utc).isoformat()

    def targeted_gate_effect(model: str, gate_id: str) -> dict[str, Any]:
        gate = aligned_targeted[model]["confirmation"]["gates"][gate_id]
        require(
            int(gate["seed_count"]) == 10,
            f"{model}: aligned targeted-retrieval {gate_id} must use 10 confirmation seeds",
        )
        return {
            "mean_effect": float(gate["estimate"]),
            "ci_low": float(gate["ci_low"]),
            "ci_high": float(gate["ci_high"]),
            "gate_pass": bool(gate["pass"]),
        }

    targeted_effects = {
        model: {
            gate_id: targeted_gate_effect(model, gate_id)
            for gate_id in (
                "targeted_bank_changes_final_count",
                "retrieval_failure_propagates_to_count",
            )
        }
        for model in MODELS
    }
    targeted_panels = [
        (
            "A · Final-count failure",
            "selected − random probability",
            [
                (SHORT[model], targeted_effects[model]["targeted_bank_changes_final_count"])
                for model in MODELS
            ],
            0.1,
        ),
        (
            "B · Joint retrieval + count failure",
            "selected − random probability",
            [
                (SHORT[model], targeted_effects[model]["retrieval_failure_propagates_to_count"])
                for model in MODELS
            ],
            0.1,
        ),
    ]

    def probability_effect_ci(row: Mapping[str, Any]) -> str:
        return (
            f"{100 * float(row['mean_effect']):+.1f} pp "
            f"[{100 * float(row['ci_low']):+.1f}, {100 * float(row['ci_high']):+.1f}]"
        )

    targeted_table = table(
        (
            "Model",
            "Frozen bank",
            "Δ final-count failure",
            "Δ joint retrieval+count failure",
            "Clean final-count accuracy",
        ),
        (
            (
                SHORT[model],
                "Top-128" if model == "Qwen3-8B" else "Top-6",
                probability_effect_ci(
                    targeted_effects[model]["targeted_bank_changes_final_count"]
                ),
                probability_effect_ci(
                    targeted_effects[model]["retrieval_failure_propagates_to_count"]
                ),
                pct(
                    float(
                        aligned_targeted[model]["confirmation"]["gates"][
                            "clean_endpoint_adequacy"
                        ]["estimate"]
                    )
                ),
            )
            for model in MODELS
        ),
    )
    write_table = table(
        ("属于哪一步", "Qwen confirmation", "Gemma confirmation", "它单独回答什么"),
        (
            (
                "5.1 direct damage · selected mask → carrier",
                ci(write_effects["Qwen3-8B"]["selected_carrier_deformation"]),
                ci(write_effects["Gemma4-E4B"]["selected_carrier_deformation"]),
                "关掉 selected bank 后，query 之后的 carrier 是否离开 clean state",
            ),
            (
                "5.2 main rescue · clean carrier → commit",
                ci(write_effects["Qwen3-8B"]["clean_carrier_restoration"]),
                ci(write_effects["Gemma4-E4B"]["clean_carrier_restoration"]),
                "保持同一 head damage，只补回 clean carrier 能否把 commit 拉回 clean",
            ),
        ),
    )
    query_table = table(
        ("Endpoint", "Qwen confirmation", "Gemma confirmation", "Scope"),
        (
            (
                "full commit → next-query targeted attention (vs self)",
                ci(query_effects["Qwen3-8B"]["self"]),
                ci(query_effects["Gemma4-E4B"]["self"]),
                "direct routing endpoint",
            ),
            (
                "full commit → next-query targeted attention (vs orthogonal)",
                ci(query_effects["Qwen3-8B"]["orthogonal"]),
                ci(query_effects["Gemma4-E4B"]["orthogonal"]),
                "Qwen strong; Gemma local specificity attenuated",
            ),
        ),
    )
    query_sample_table = table(
        (
            "Model / phase",
            "独立样本",
            "每个 seed 的 donor–receiver pairs",
            "正式 pair 数",
            "patch layer / readout bank",
        ),
        (
            (
                f"{SHORT[model]} · {phase}",
                f"{20 if phase == 'discovery' else 10} seeds × 1 trace/seed",
                "d = −1, +1（每个 signed offset 1 pair）",
                str((20 if phase == "discovery" else 10) * 2),
                (
                    "L19 / frozen Top-128"
                    if model == "Qwen3-8B"
                    else "L16 / frozen Top-6"
                ),
            )
            for model in MODELS
            for phase in ("discovery", "confirmation")
        ),
    )
    query_arm_table = table(
        ("每个 donor–receiver pair 的 condition", "receiver commit 中写入什么", "该 arm 回答什么"),
        (
            ("Clean", "不调用 patch hook", "保留自然轨迹作审计参照；不进入正文两个主差分"),
            ("Self patch", "把 receiver 自己的完整 post-block vector 写回", "控制 hook、复制和写回本身"),
            ("Full donor patch", "把 donor commit 的完整 post-block vector 写到 receiver", "完整 commit state 是否足以重定向下一 query"),
            ("Count-subspace transplant", "只移植 donor−receiver delta 在 discovery-frozen count subspace 上的投影", "窄线性 count component 是否足够"),
            ("Norm-matched orthogonal patch", "写入与上述 count 投影等范数、但对该 subspace 正交的扰动", "同尺度的一般 state 扰动能否解释结果"),
        ),
    )
    def grammar_anchor_phase(model: str, phase: str) -> list[dict[str, Any]]:
        role = "development" if phase == "discovery" else "confirmation"
        return [
            row
            for row in grammar_anchors[model]
            if str(row["stratified_ncc_seed_role"]) == role
        ]

    def routed_anchor_phase(model: str, phase: str) -> list[dict[str, Any]]:
        role = "development" if phase == "discovery" else "confirmation"
        return [
            row
            for row in routed_anchors[model]
            if str(row["alignment_phase"]) == role
        ]

    def count_composition(rows: Sequence[Mapping[str, Any]]) -> str:
        counts = sorted({int(row["gold_count"]) for row in rows})
        return " · ".join(
            f"N={count} × {sum(int(row['gold_count']) == count for row in rows)}"
            for count in counts
        )

    write_sample_table = table(
        ("Model / phase", "独立 trace 样本", "实际 gold-count 组成", "grammar timing"),
        (
            (
                f"{SHORT[model]} · {phase}",
                f"{len(routed_anchor_phase(model, phase))} seeds × 1 trace/seed",
                count_composition(routed_anchor_phase(model, phase)),
                "; ".join(
                    f"{name} {int(count)}"
                    for name, count in sorted(
                        write[model][phase]["grammar_timing_counts"].items()
                    )
                ),
            )
            for model in MODELS
            for phase in ("discovery", "confirmation")
        ),
    )
    token_condition_table = table(
        ("图 5a panel / 横轴", "实际被置零的 token states", "为什么要放这个条件"),
        (
            ("左 · clean", "不置零", "给出自然 next-city retrieval 基线"),
            ("左 · early half", "当前 query 可见的已完成 items 中，最早 floor(k/2) 个 item 的 trace states", "检验较远的早期历史是否仍直接参与当前 routing"),
            ("左 · cumulative", "从 trace 起点到最近一个已完成 item 之前的全部 trace states；保留最近 transition", "把长期累计历史与最近一次局部更新分开"),
            ("左 · recent", "最近一个已完成 item 起点到当前 query 之前的 trace states；保留更早历史", "检验当前 query 是否主要由最近一次 transition 触发"),
            ("左 · full trace", "prompt 结束后到当前 query 之前的全部 trace states", "测量早期历史与最近 transition 合在一起的总必要性"),
            ("右 · clean", "不置零", "给出自然 greedy exact-count accuracy"),
            ("右 · prompt records", "只置零原 prompt 中 parser-registered needle / city records；保留其他 prompt 与完整 trace", "检验已有 trace 时，answer 是否仍必须回原始 records 重数"),
            ("右 · full trace", "置零 prompt 后生成的整段 trace；保留完整 prompt", "检验 answer 是否必须读取 trace 已写好的状态"),
            ("右 · full prompt + trace", "除 BOS/system boundary 与 answer query 外，置零完整 prompt 与完整 trace", "给出两类信息源同时移除的 floor / sanity control"),
        ),
    )
    terminal_sample_table = table(
        (
            "Model / phase",
            "独立 trace 样本",
            "实际 gold-count 组成",
            "grammar timing",
            "source layer / condition rows",
        ),
        (
            (
                f"{SHORT[model]} · {phase}",
                f"{len(grammar_anchor_phase(model, phase))} seeds × 1 trace/seed",
                count_composition(grammar_anchor_phase(model, phase)),
                (
                    "city→rank 9；rank→city 10"
                    if phase == "discovery"
                    else "city→rank 4；rank→city 5"
                ),
                (
                    f"L{19 if model == 'Qwen3-8B' else 16}→final block / "
                    f"{len(grammar_anchor_phase(model, phase)) * 12} rows"
                ),
            )
            for model in MODELS
            for phase in ("discovery", "confirmation")
        ),
    )
    terminal_geometry_table = table(
        ("Geometry", "最后一个 item 内的 receiver positions", "它区分的解释"),
        (
            ("Full item", "parser-registered 最后一个完整 trace item", "可恢复上界；不用于选择最小 carrier"),
            ("Marker core", "显式 rank / count marker 的 token span", "数字/序数 marker state 本身是否承载 terminal count"),
            ("Retrieved city", "最后一次检索得到的 city token span", "排除仅由 city 词汇内容驱动 answer 的解释"),
            ("Grammar-timed tail", "city→rank grammar: marker 到 commit；rank→city grammar: city 到 commit", "允许两种 grammar 在不同时间完成 terminal update"),
            ("Boundary commit", "parser 注册的 item-end commit token span", "最局部的提交边界是否已足够"),
        ),
    )
    terminal_arm_table = table(
        ("每条 trace 的 arm 组", "数量", "具体操作", "用于什么比较"),
        (
            ("Clean", "1", "原始 prompt + 原始 trace，不 patch", "自然参考"),
            ("Uninformative", "1", "每个 parsed trace item 都换成等长 prompt-background tokens", "所有 restoration 的共同 damaged baseline"),
            ("Semantic restore", "5", "分别把五种 geometry 的 clean same-position hidden states 写回", "restore − uninformative"),
            ("Matched-random restore", "5", "在相同 receiver positions 写入等 token、近深度、非 terminal-item 的 clean states", "semantic restore − matched random specificity"),
        ),
    )
    write_arm_table = table(
        ("Forward arm", "每条 trace 的次数", "在 query 做什么", "在 carrier 做什么", "用于哪一节"),
        (
            ("Clean", "1", "不 mask", "不 patch", "5.1/5.2 的 clean state 与距离参照"),
            ("Selected mask", "1", "mask frozen Top-128 / Top-6", "不 patch", "5.1 主 treatment；5.2 damaged baseline"),
            ("Layer-matched random mask", "3", "三套不同随机 bank；逐层 head 数与 selected bank 相同", "不 patch", "5.1 secondary identity control"),
            ("Selected mask + clean-carrier clamp", "1", "保持 selected mask", "逐层写回同一位置的 clean carrier", "5.2 主 restoration"),
        ),
    )
    query_distance_table = table(
        ("Distance |d|", "Qwen: full−orthogonal", "Gemma: full−orthogonal", "Registered role"),
        (
            (
                "1",
                ci(query_effects["Qwen3-8B"]["orthogonal"]),
                ci(query_effects["Gemma4-E4B"]["orthogonal"]),
                "exact-common-support primary",
            ),
        ),
    )
    walkthrough_table = table(
        ("Model", "Case", "Uninformative baseline", "Full-item path", "Carrier path", "Conclusion"),
        (
            (
                SHORT[model],
                f"seed {walkthrough[model]['seed']}, count 10",
                f"candidate={walkthrough[model]['baselines']['uninformative']['candidate_prediction']}; P(10)={walkthrough[model]['baselines']['uninformative']['gold_count_probability']:.3f}",
                f"exact {walkthrough[model]['conditions']['full_item_restore']['candidate_exact_path_count']}/10; r={walkthrough[model]['conditions']['full_item_restore']['expected_count_correlation_with_occurrence']:+.2f}",
                f"exact {walkthrough[model]['conditions']['counter_carrier_restore']['candidate_exact_path_count']}/10; r={walkthrough[model]['conditions']['counter_carrier_restore']['expected_count_correlation_with_occurrence']:+.2f}",
                "does not walk 1→10",
            )
            for model in MODELS
        ),
    )
    scope_labels = {
        "item_end_w1": "单 endpoint（1 token）",
        "event_tail_w4": "event tail（4 tokens）",
        "item_span": "endpoint-aligned item span",
    }
    patch_scope_discovery_table = table(
        (
            "Patch scope",
            "Frozen layer",
            "Median Δ donor log-odds",
            "Positive cells",
            "Mean patch norm",
            "Mean Δ / norm",
        ),
        (
            (
                scope_labels[scope],
                f"L{int(patch_scope_discovery[scope]['selected_layer'])}",
                f"{float(patch_scope_discovery[scope]['selected_layer_summary']['median_paired_logodds_shift']):+.2f}",
                f"{100*float(patch_scope_discovery[scope]['selected_layer_summary']['positive_shift_rate']):.1f}% / 40",
                f"{float(patch_scope_discovery[scope]['selected_layer_summary']['mean_patch_norm']):.2f}",
                f"{float(patch_scope_discovery[scope]['selected_layer_summary']['mean_logodds_shift_per_patch_norm']):.2f}",
            )
            for scope in ("item_end_w1", "event_tail_w4", "item_span")
        ),
    )
    patch_scope_confirmation_table = table(
        (
            "Frozen scope",
            "Median Δ log-odds",
            "Attention Δ > 0",
            "Donor argmax",
            "First known city → donor",
            "Seeds with ≥1 incremental transfer",
        ),
        (
            (
                f"{scope_labels[scope]} · L{int(patch_scope_discovery[scope]['selected_layer'])}",
                f"{float(patch_scope_confirmation[scope]['median_paired_logodds_shift']):+.2f}",
                f"{int(round(60*float(patch_scope_confirmation[scope]['positive_attention_shift_rate'])))}/60",
                f"{int(round(60*float(patch_scope_confirmation[scope]['patched_donor_argmax_rate'])))}/60",
                f"{int(round(60*float(patch_scope_confirmation[scope]['patched_first_known_city_donor_adoption_rate'])))}/60",
                f"{int(round(10*float(patch_scope_confirmation[scope]['seed_with_any_greedy_donor_adoption_rate'])))}/10",
            )
            for scope in ("item_end_w1", "event_tail_w4", "item_span")
        ),
    )
    extension_audit_table = table(
        ("Extension", "What was transplanted / intervened", "Observed result", "Role in this report"),
        (
            (
                "CountScope",
                "full-item state → one-placeholder receiver",
                "N=3: k=1/2/3 candidate 0.90/0.70/1.00; N=10: k=1–4 usable, k≥5 mostly fails",
                "supports readable local state; rejects a standalone context-invariant register",
            ),
            (
                "Continued counting",
                "source last-k states → target first-k; evaluate hop 1/2 and final",
                "N=3 hop 1 briefly 0.3–0.7; N=10 hop 1 ≤0.20; hop 2/final ≈0",
                "does not establish memoryless +1 recurrence",
            ),
            (
                "Geometry steering",
                "single-site +1 direction, all-layer scan; opposite/orthogonal controls",
                "peak L19: N=3 +0.622 [0.471, 0.767], N=10 +0.215 [0.107, 0.334]",
                "local causal geometry; post-hoc layer profile, not fresh confirmation",
            ),
            (
                "Separator dose",
                "collapse later events to first-event marker / closing / full-event states",
                "per-event slope: marker −0.125, closing −0.219, full event −0.690",
                "full event dominates marker; supports a distributed carrier",
            ),
            (
                "Maximum-count",
                "source last-k → target last-k; test max(Ns, Nt−k)",
                "donor-dominant candidate 0.13–0.30; target−k branch history-confounded",
                "no evidence for a general max operator",
            ),
            (
                "Marker K/V and operator scan",
                "K-only, V-only, layer bands; broad recurrence-operator family",
                "K/V 0.835, V 0.500, K 0.276; operator scan reset 97.08%, target +1 0.625%",
                "event-memory substrate is plausible; reset dominates explicit recurrence candidates",
            ),
        ),
        class_name="evidence-ledger",
    )

    def direction_count(
        summary: Mapping[str, Any], direction: str, count_field: str
    ) -> tuple[int, int]:
        rows = [
            row
            for row in summary["by_direction_k"]
            if row["direction"] == direction and int(row.get("cell_count", 0)) > 0
        ]
        return (
            sum(int(row[count_field]) for row in rows),
            sum(int(row["cell_count"]) for row in rows),
        )

    l0_item_forward = direction_count(
        patch_scope_confirmation["item_span"],
        "forward_skip",
        "patched_first_known_city_donor_adoption_count",
    )
    l0_item_backward = direction_count(
        patch_scope_confirmation["item_span"],
        "backward_rewind",
        "patched_first_known_city_donor_adoption_count",
    )
    l16_item_forward = direction_count(
        item_span_l16_summary,
        "forward_skip",
        "patched_first_known_city_donor_adoption_count",
    )
    l16_item_backward = direction_count(
        item_span_l16_summary,
        "backward_rewind",
        "patched_first_known_city_donor_adoption_count",
    )
    qwen_l0_forward_cells = [
        cell
        for cell in patch_scope_frozen["cells"]
        if cell["scope"] == "item_span"
        and cell["direction"] == "forward_skip"
    ]
    require(
        len(qwen_l0_forward_cells) == 30,
        "Qwen natural no-index forward comparison support changed",
    )
    qwen_l0_forward_argmax = sum(
        bool(cell["patched_donor_argmax"]) for cell in qwen_l0_forward_cells
    )
    qwen_l0_forward_attention = sum(
        float(cell["paired_attention_shift"]) > 0.0
        for cell in qwen_l0_forward_cells
    )
    qwen_l0_forward_logodds = sum(
        float(cell["paired_logodds_shift"]) > 0.0
        for cell in qwen_l0_forward_cells
    )
    qwen_l0_forward_self_donor = sum(
        int(cell["receiver_first_known_city_ordinal"])
        == int(cell["donor_occurrence_k"]) + 1
        for cell in qwen_l0_forward_cells
    )
    gemma_prompt_k_rows = []
    for group in sorted(
        gemma_prompt_noindex_confirmation["groups"].values(),
        key=lambda row: int(row["donor_occurrence"]),
    ):
        paired = group["paired_patch_minus_self"]
        gemma_prompt_k_rows.append(
            [
                f'{int(group["receiver_occurrence"])} ← {int(group["donor_occurrence"])}',
                f'{int(paired["donor_argmax_patch"]["hits"])}/10',
                f'{int(paired["greedy_donor_adoption_patch"]["hits"])}/10',
                f'{int(paired["greedy_donor_adoption_self"]["hits"])}/10',
                f'{float(paired["logodds_gain_patch_minus_self"]["mean"]):+.2f}',
                f'{int(paired["positive_attention_gain"]["hits"])}/10',
            ]
        )
    gemma_prompt_k_table = table(
        [
            "receiver ← donor",
            "Donor argmax",
            "Greedy donor",
            "Self donor",
            "Mean Δ log-odds",
            "Attention Δ > 0",
        ],
        gemma_prompt_k_rows,
        class_name="compact-table",
    )

    html_text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Native-thinking 计数机制：Internal counter 与定向检索</title><link rel="icon" href="data:,"><style>{css}</style></head>
<body><article class="page"><header><p class="eyebrow">Realistic CoT NiaH · Native-thinking mechanism</p>
<h1>Native-thinking 如何计数：Internal counter 驱动定向检索</h1>
<p class="dek">与 Non-thinking 报告使用同一逻辑骨架：先分开 representation 与 causal evidence，再沿 state formation、retrieval、write/control、terminal readout 组织证据。任何自然 no-index internal-counter / progress-controller 主张均严格限定于 Qwen3-8B 的自然 N=10 trace；Gemma 只作为显式-index、prompt-conditioned no-visible-index、targeted-retrieval 与 carrier 证据的跨模型参照。</p>
<div class="meta"><span>Qwen3-8B · frozen Top-128</span><span>Gemma4-E4B · frozen Top-6</span><span>formal: 20 discovery / 10 confirmation</span><span>generated {esc(generated)}</span></div></header>
<nav><a href="#summary">结论</a><a href="#baseline">1 基线</a><a href="#representation">2 表征</a><a href="#formation">3 State formation</a><a href="#retrieval">4 检索</a><a href="#write">5 写入与控制</a><a href="#answer">6 终端</a><a href="#ledger">7 证据表</a><a href="#extension-audit">8 扩展审计</a><a href="#limitations">9 边界</a><a href="#appendix">Appendix</a></nav>
<main>

<section id="definitions"><p class="eyebrow">00 · Definitions before claims</p><h2>先定义本文所有核心对象</h2>
<p class="lead">下面的名词在机制图之前统一定义。本文始终区分“信息可读”、“干预足够改变下游”与“自然运行必须使用”；三者不是同一命题。</p>
<dl class="definition-grid">
<div class="definition"><dt>Gold count <em>N</em> 与 occurrence <em>k</em></dt><dd><em>N</em>是 prompt 中真实 needle records 数，取 1–10。<em>k</em>是模型在 trace 中已完成的第 <em>k</em> 条记录。例如 <em>N</em>=6、<em>k</em>=3 表示总共应有 6 条，当前处理到第 3 条。</dd></div>
<div class="definition"><dt>Hidden state / residual stream</dt><dd>每一 token 在某个 transformer block 后的向量 <em>h</em><sub>ℓ,t</sub>。“state patch”是把 donor 的这个向量写到 receiver 的同层位置，而不改变可见 token。</dd></div>
<div class="definition"><dt>Query、attention head 与 head bank</dt><dd>Query 是当前 token 向先前 tokens 读取信息的位置。一枚 attention head 是一条局部读取通道；head bank 是事先冻结的多枚 heads 集合。Top-<em>K</em> 表示 bank 有 <em>K</em> 枚 heads，不表示单头必要。</dd></div>
<div class="definition"><dt>Raw attention mass 与 bank-summed mass</dt><dd>单头从 query <em>q</em> 指向 source span <em>R</em> 的 raw mass 是 <span class="math">A<sub>h</sub>(q,R)=Σ<sub>t∈R</sub>α<sub>h</sub>(q,t)</span>，范围 0–1。整 bank 的 mass 再对 heads 求和，因此可以大于 1；它不是概率，且不同 K 的绝对色值不可直接比较。</dd></div>
<div class="definition"><dt>Targeted retrieval</dt><dd>在一次 <em>k</em>→<em>k</em>+1 的 trace query 上，head bank 对 prompt 中第 <em>k</em>+1 条正确 city record 进行定向读取。它不是 answer-time 对全部 records 的 broad retrieval。</dd></div>
<div class="definition"><dt>Grammar carrier</dt><dd>检索之后、item commit 之前承载当前进度的 grammar-specific token states。Rank-after-city trace 使用显式 marker core；rank-before-city trace 使用 city-to-commit tail。</dd></div>
<div class="definition"><dt>Commit state (P0)</dt><dd>一条 trace item 完成时的 item-end post-block hidden state。它是“已经处理到 <em>k</em>”的提交位置，也是下一次 targeted query 的上游候选状态。</dd></div>
<div class="definition"><dt>Ablation、self control 与 matched control</dt><dd>Ablation 在指定 query 清零 selected heads 的 pre-O output。Self-patch 把 receiver 自己写回，控制 hook 本身。Layer-matched random 使用同层同数随机 heads。Orthogonal control 与 <em>donor−receiver 差向量在 frozen count 子空间中的投影</em>等范数，但其方向与该子空间正交；它并不与完整 donor−receiver 差向量等范数。</dd></div>
<div class="definition"><dt>Representation、probe 与 PCA manifold</dt><dd>Representation 问 hidden state 中是否可读出 count。Probe 是只在 discovery 拟合的简单分类器。PCA manifold 是 discovery-fitted 主成分上的可视化；confirmation 只被投影，不参与定轴。</dd></div>
<div class="definition"><dt>Effect、95% CI 与 distance <em>d</em></dt><dd>Effect 是同 seed、同 pair 内 treatment−control 的均值。95% CI 是 seed-level bootstrap 区间，作为稳定性记录。<em>d</em>=donor occurrence−receiver occurrence；正文主检验使用 |<em>d</em>|=1。</dd></div>
<div class="definition"><dt>Confirmed、confirmed† 与 controlled only</dt><dd><strong>Confirmed</strong>：冻结干预在 20-seed discovery 后，又在独立 10-seed confirmation 上复现。<strong>Confirmed†</strong>：正结果只来自受控 surrogate，或直接边虽复现但另一个更严格的局部控制仍未闭合；† 后会逐项写明限制。<strong>Controlled only</strong>：在所有后续可见 tokens 都固定相同的受控比较中，patch 会改变答案分数；但让模型自由继续生成时，还没有证明这一个 state 单独足以改变最终答案。它不是“没有效果”，而是“效果适用范围较窄”。Gemma 的自然 no-index surrogate 使用 †；其 exact-common commit→next-query 边在本版不再使用 †。</dd></div>
</dl>
<div class="section-conclusion"><strong>定义层面的结论。</strong> 后文的“confirmed”不意味唯一回路；“可解码”也不自动意味模型必须使用该线性方向。</div></section>

<section id="summary"><p class="eyebrow">Conclusion first</p><h2>一条 recurrent counting pathway 已经接上；natural end-to-end sufficiency 仍未证明</h2>
<p class="lead">两模型都支持同一类局部因果链：targeted heads 检索下一条 city，改变 grammar-specific marker/tail carrier；carrier 写入 commit state；commit state 再改变下一次 targeted query。终端 grammar state 在固定 suffix 的受控实验中能恢复 answer count margin，但把全部上下文抹掉后，仅恢复任一单 item 并不能让答案随 k 从 1 走到 10。</p>
<div class="figure-primer"><div><strong>图中画什么</strong>五个框是一次循环计数从检索到答案的候选阶段。</div><div><strong>怎么读</strong>每一列分别给 Qwen 与 Gemma 的最高证据级别；紫色表示仅在受控终端成立。</div><div><strong>简单例子</strong>完成第 3 条后，commit state 改变“下一次应读第 4 条”的 query routing。</div></div>
<div class="chain-scroll">{chain_svg()}</div>
<div class="status-grid"><div class="status-card"><h3>Qwen3-8B</h3><p class="status-good">recurrent loop：强 confirmation</p><p>Top-128 retrieval、carrier→commit、commit→next query 都有大效应。terminal grammar state 的局部受控 restoration 为正。</p><p class="status-open">仍开放：free-running answer count-margin 与全上下文擦除后的单点 sufficiency。</p></div>
<div class="status-card"><h3>Gemma4-E4B</h3><p class="status-good">recurrent direct edge：exact-common confirmation</p><p>Top-6 retrieval、carrier→commit 与 commit→next query 都使用和 Qwen 相同的 semantic pair multiset；full donor−self 为 {ci(query_effects['Gemma4-E4B']['self'])}，|d|=1 full−orthogonal 为 {ci(query_effects['Gemma4-E4B']['orthogonal'])}，两项 confirmation 区间都在 0 以上。</p><p class="status-open">边界：orthogonal 扰动只匹配 frozen count 投影的范数，不匹配完整 donor−receiver 差向量；因此仍未隔离唯一、低维或排他的 count component。</p></div></div>
<div class="claim"><strong>允许的主张。</strong> Native-thinking trace 中存在一条可重复干预的 recurrent counting pathway；它不是已证明唯一或排他的 counting circuit。</div></section>

<section id="design"><p class="eyebrow">01 · Experimental contract</p><h2>设计与判据</h2>
<p class="lead">正式因果实验固定使用 discovery seeds 1234–1253 与 confirmation seeds 1254–1263；所有正式 pair plan 都是 outcome-blind，且不使用 selection_rank。单 seed walkthrough 只作 case study。</p>
<div class="reading-protocol"><div class="protocol-step"><span class="protocol-no">Discovery</span><h3>定位与冻结</h3><p>20 seeds。选择层、head bank、geometry 与 primary estimand。</p></div><div class="protocol-step"><span class="protocol-no">Confirmation</span><h3>独立复现</h3><p>10 seeds。设计不因 partial outcomes 改动。</p></div><div class="protocol-step"><span class="protocol-no">Claim scope</span><h3>效应优先</h3><p>正文同时给 mean effect 与 95% CI，但机制判断首先看效应方向、大小和控制组。</p></div></div>
<p class="report-note">Head bank：Qwen Top-128；Gemma Top-6。Gemma K=6 的 selected−random retrieval failure 比 K=8 更大，因此最新主线固定 K=6；旧 K=8 仅属于历史实验。</p></section>

<section id="task"><p class="eyebrow">02 · Task and unit of analysis</p><h2>任务设定：模型一边在 trace 中找 city，一边维持当前计数</h2>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>先明确需要解释的行为单位：不仅是最后数字 <em>N</em>，还包括 trace 中每个 <em>k</em>→<em>k</em>+1 的局部检索。</div><div><span class="experiment-label">设定</span>每条 prompt 约 10k tokens，含 1–10 条真实 city records 与大量普通文本。Native-thinking 模型先生成 reasoning trace，再输出 total count。</div><div><span class="experiment-label">简单例子</span>Prompt 里有 6 条真实 records。Trace 已处理到 <em>k</em>=3 时，下一次 query 应读第 4 条 city；完成第 6 条后，answer query 应输出 6。</div><div><span class="experiment-label">分析单位</span>Representation 使用 seed–count 或 seed–occurrence state；causal patching 使用同 seed 内 donor–receiver pair，所有 treatment/control 在 pair 内配对。</div></div>
<div class="section-conclusion"><strong>本节结论。</strong> 要串起完整机制，必须同时解释“下一条读谁”和“已处理多少条”；只看 final exact accuracy 会混合多个故障源。</div></section>

<section id="representation"><p class="eyebrow">03 · Representation</p><h2>Trace commit 与 answer query 都含有可读的 count geometry</h2>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>定位 count information 何时出现、在 trace 和 answer 两个 endpoint 上是否可分。这一步只找候选状态，不声称自然因果使用。</div><div><span class="experiment-label">设定</span>每个 discovery fold 内单独拟合 standardization、PCA-16 与 L2 logistic probe；nearest-centroid (NCC) 作为不同决策规则的控制。选定层冻结后只在 10 confirmation seeds 评估。</div><div><span class="experiment-label">计算方法</span><span class="formula">BalancedAccuracy = (1/10) Σ<sub>c=1</sub><sup>10</sup> TP<sub>c</sub> / (TP<sub>c</sub> + FN<sub>c</sub>)</span>十个 count 类等权，因此 chance=0.10，不被不同 occurrence 样本数影响。</div><div><span class="experiment-label">简单例子</span>如果第 4 条完成后的 hidden states 在 held-out seeds 上仍靠近“4”的 discovery centroid，而远离“3/5”的 centroid，则该层含有可读的 running count。</div></div>

<h3>3.1 可解码性如何随层变化</h3>
<div class="figure-primer"><div><strong>图中画什么</strong>上排是 item-end running count，下排是 answer-query final count。</div><div><strong>坐标怎么读</strong>横轴是 zero-based post-block layer；纵轴是 confirmation balanced accuracy；灰虚线 0.10 是 chance。</div><div><strong>圆点是什么</strong>空心圆是 discovery 冻结的报告层，不是根据 confirmation 再挑的最高点。</div></div>
<figure><h3 class="figure-title">图 1a · Native count representation 的逐层 held-out readout</h3>{layerwise_representation_svg(layerwise_representation, representation)}<figcaption>每个 panel 的横轴是 zero-based transformer layer；纵轴是 10-class confirmation balanced accuracy。绿线=L2 logistic，橙线=nearest centroid，灰虚线=0.10 chance。Running panel 固定使用 exact causal item-end site；final panel 使用 answer query。折线是描述性层进程，不把最高 confirmation 点当作新选择。</figcaption></figure>

<h3>3.2 Count clouds 在低维中长什么样</h3>
<p><strong>先定义三维 PCA 图。</strong> PC1、PC2、PC3 是 discovery hidden-state covariance 中方差最大的三个彼此正交方向；它们不是三个预设的“count 维度”，也不是三个神经元。对 discovery 矩阵 <em>H</em><sub>D</sub> 去均值后取前三个右奇异向量 <em>V</em><sub>3</sub>，confirmation state 只做固定投影：</p>
<span class="formula">μ<sub>D,j</sub>, σ<sub>D,j</sub> = discovery mean and standard deviation of hidden dimension j<br>X<sub>D,j</sub> = (H<sub>D,j</sub>−μ<sub>D,j</sub>)/σ<sub>D,j</sub>, &nbsp; V<sub>3</sub> = top-3 right singular vectors of X<sub>D</sub><br>z<sub>confirm</sub> = ((h<sub>confirm</sub>−μ<sub>D</sub>)/σ<sub>D</sub>)V<sub>3</sub> = (PC1, PC2, PC3)</span>
<p>这条计算规则阻止 confirmation 信息参与定轴。图中的小点是 confirmation states；带数字的大点是各 occurrence/count 的 confirmation centroid；折线仅连接 1→10 的 centroid，帮助观察几何顺序，不进入 probe 或因果统计。分类器仍使用 PCA-16，因此 3D 云比 probe 有意丢弃更多信息。</p>
<div class="figure-primer"><div><strong>如何交互</strong>拖动点云改变观察角度；双击或按 Reset 恢复视角；模型下拉框切换 Qwen/Gemma。</div><div><strong>Layer 选择</strong>下拉框只显示 discovery 冻结的 report layer。逐层比较已在图 1a 完成，不允许凭 confirmation 云形状重新挑层。</div><div><strong>坐标和颜色</strong>三轴是任意 PCA 单位；颜色与 centroid 数字均表示 1–10。不同模型分别归一显示，不能跨模型比较坐标尺度。</div></div>

<figure><h3 class="figure-title">图 1b · Trace 与 answer 的逐层 PC1–PC3 confirmation manifold</h3>
<div class="three-d manifold-comparison"><div class="three-d-head"><div><strong>Native hidden-state geometry · trace / answer side by side</strong><div class="term-note">拖动任一 panel 会同步旋转两侧，便于比较几何形状；两个 endpoint 仍使用各自独立的 discovery-fitted PCA basis。</div></div><div class="three-d-controls"><label>模型<select id="native-geometry-model"><option>Qwen3-8B</option><option>Gemma4-E4B</option></select></label><button type="button" id="native-geometry-reset">Reset both views</button></div></div>
<div class="manifold-grid"><section class="manifold-panel"><div class="manifold-panel-head"><div><strong>Trace commit</strong><span>running occurrence <em>k</em></span></div><label>Layer<select id="native-running-layer" aria-label="Native running manifold layer"></select></label></div><canvas id="native-running-canvas" aria-label="Native-thinking item-end confirmation hidden states projected onto layer-specific discovery-fitted PC1 PC2 PC3. Drag to rotate.">浏览器需要 Canvas 支持。</canvas><p class="manifold-stats" id="native-running-stats"></p></section>
<section class="manifold-panel"><div class="manifold-panel-head"><div><strong>Answer query</strong><span>final gold count <em>N</em></span></div><label>Layer<select id="native-final-layer" aria-label="Native final manifold layer"></select></label></div><canvas id="native-final-canvas" aria-label="Native-thinking answer-query confirmation hidden states projected onto layer-specific discovery-fitted PC1 PC2 PC3. Drag to rotate.">浏览器需要 Canvas 支持。</canvas><p class="manifold-stats" id="native-final-stats"></p></section></div></div>
<figcaption>左 panel 取 trace item-end hidden states，颜色/数字是已完成 occurrence <em>k</em>；右 panel 取数字答案生成前的 <code>answer_query_v3</code>，颜色/数字是 gold count <em>N</em>。模型选择器同时切换两侧；两个 layer 选择器彼此独立，覆盖该模型的全部 post-block layers。每个 layer 都只用 discovery states 拟合 StandardScaler 与 PCA3，再显示 confirmation states；默认层分别为 running Qwen L{running_display_layers['Qwen3-8B']} / Gemma L{running_display_layers['Gemma4-E4B']}，answer Qwen L{final_display_layers['Qwen3-8B']} / Gemma L{final_display_layers['Gemma4-E4B']}。下方状态行实时报告 token site、样本数和前三 PC 的 discovery explained-variance ratio。</figcaption></figure>
<p><strong>如何严谨地使用 layer selector。</strong> 它用于检查 geometry 随深度如何演化，不参与任何 formal selection。默认层在 discovery 阶段冻结；图 1a 的 layerwise confirmation curves 是完整的数值汇总。用户手动切到另一层后看到的云只能作为描述性浏览，不能据此事后替换默认层或改变 causal experiment。</p>
<p><strong>替代解释与左右比较边界。</strong> Running endpoint 的 marker kind、boundary token、trace grammar 和 token identity 会共同改变 residual state，冻结诊断中也出现 grammar/token 相关双 band；因此左侧分簇不能单独证明“纯 count axis”。左右两侧使用不同 endpoint、不同 state 数和独立 PCA basis，只能比较有无有序结构，不能逐轴对齐 PC1 或比较 centroid 距离。更可靠的 evidence 是跨-seed PCA-16 readout 与后续配对 patching。</p>

<h3>3.3 冻结层的数值结果</h3>
<figure><h3 class="figure-title">图 1c · Frozen report layers 的 confirmation balanced accuracy</h3>{grouped_bars_svg('Frozen-decoder balanced accuracy', rep_groups, maximum=1.0)}<figcaption>Qwen running commit L{int(representation['Qwen3-8B']['running']['layer'])}={float(representation['Qwen3-8B']['running']['confirmation_logistic_balanced_accuracy']):.3f}，answer query L{int(representation['Qwen3-8B']['final']['layer'])}={float(representation['Qwen3-8B']['final']['confirmation_logistic_balanced_accuracy']):.3f}。Gemma running commit L{int(representation['Gemma4-E4B']['running']['layer'])}={float(representation['Gemma4-E4B']['running']['confirmation_logistic_balanced_accuracy']):.3f}，answer query L{int(representation['Gemma4-E4B']['final']['layer'])}={float(representation['Gemma4-E4B']['final']['confirmation_logistic_balanced_accuracy']):.3f}。Running commit 的层由 exact common-support discovery rows 冻结；横向条长是 balanced accuracy，从 0 到 1。数值来自 PCA-16 held-out probe，而不是前三 PC 的视觉可分性。</figcaption></figure>
<p><strong>结果分析。</strong> Qwen 的 answer-query count 几乎完全可读，Gemma 的 running commit 反而比 answer query 更强。这说明两模型不必在相同层或同一维度上保留 count；但两者都有可供后续因果实验锁定的 state。三维图补充的是几何直觉，图 1a/1d 的 held-out 数值才是 representation 主证据。</p>
<div class="section-conclusion"><strong>Experiment 3 结论。</strong> Count information 同时存在于循环中的 commit state 与末端 answer state；本节只建立 representation，后续 ablation/patching 再判断这些状态是否被实际使用。</div></section>

<section id="retrieval"><p class="eyebrow">04 · Targeted retrieval</p><h2>下一条 city 由 model-specific targeted head banks 定向检索</h2>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>确定“该读 prompt 中哪一条 city”是否由 discovery-localized heads 实际执行，而不只是 attention visualization。</div><div><span class="experiment-label">设定</span>在 exact next-city query 开始清零 selected bank 的 pre-O head slices，干预持续到首个可识别 city。每个 selected bank 与同层构成、同 K 的 random bank 配对。</div><div><span class="experiment-label">主终点</span><span class="formula">Failure(B) = 1[first generated semantic city ≠ registered next city]<br>Δ<sub>bank</sub> = mean Failure(selected) − mean Failure(layer-matched random)</span></div><div><span class="experiment-label">简单例子</span>当 trace 正从 3 走向 4，正确 target 是 prompt 第 4 条 Paris。如果只关 selected heads 后首 city 变成 Oslo，而关同数 random heads 仍是 Paris，该 pair 支持 selection-specific retrieval necessity。</div></div>

<h3>4.1 Heads 在哪里，attention 是否随 target 移动</h3>
<div class="figure-primer"><div><strong>左图</strong>横轴是 transition <em>k</em>→<em>k</em>+1，纵轴是 prompt record ordinal；颜色是 Top-K 合计 raw attention mass。</div><div><strong>正确图样</strong>亮带应随 <em>k</em>+1 沿对角线移动，而不是永远固定在第一条。</div><div><strong>证据边界</strong>这两张图是 descriptive localization；因果必要性来自下面 selected-vs-random ablation。</div></div>
<div class="attention-pair"><figure><h3 class="figure-title">图 2a · Qwen Top-128 单轨迹 targeted-attention map</h3>{qwen_attention_sum_svg}<figcaption>横轴为 P0 transition k→k+1，纵轴为真实 prompt record ordinal；颜色为 128 枚 heads 对该 record span 的 raw attention mass 之和，红框/红点标出正确 successor。跨模型不比较颜色绝对值。</figcaption></figure><figure><h3 class="figure-title">图 2b · Gemma Top-6 单轨迹 targeted-attention map</h3>{gemma_attention_sum_svg}<figcaption>横轴、纵轴、红色 target 标记与颜色定义同左图，但 bank 只有 6 枚 heads。对角 target pattern 说明少数 heads 可被多个 occurrence 重复使用；绝对色值不能与 Qwen Top-128 比较。</figcaption></figure></div>

<h3>4.2 单头、整 bank 与跨-seed attention pattern</h3>
<div class="figure-primer"><div><strong>单轨迹 panels</strong>横轴是 P0 transition；纵轴是带真实 city 名的 prompt records；红框或红点标出正确 <em>k</em>+1 target。</div><div><strong>跨-seed panel</strong>横轴是 discovery-ranked head；纵轴是正确 target ordinal；颜色是先在 seed 内平均、再对 20 discovery seeds 等权平均的 raw target mass。</div><div><strong>怎么使用</strong>用下拉框切换 Qwen/Gemma、单头/整 bank/跨-seed 版本。热图展示 routing 形状，不替代因果 ablation。</div></div>
<figure><h3 class="figure-title">图 2c · 可切换的 Native targeted-retrieval attention maps</h3>{attention_example_switcher(attention_examples)}<figcaption>单头和 bank-summed panels：横轴=P0 transition <em>k</em>→<em>k</em>+1，纵轴=prompt record region，颜色=raw attention mass，红框/红点=正确 next record；Top-K 合计可能大于 1。Ordinal×head panels：横轴=ranked head，纵轴=target ordinal，颜色=20 discovery seeds 的 seed-equal target mass。Gemma aggregate 保留八列以复现历史 atlas，其中前六列构成当前 Top-6，后两列仅是 K8 diagnostic；正式因果结论始终使用 Top-6。</figcaption></figure>

<h3>4.3 Head bank 在 layer×head 空间中的位置</h3>
<div class="figure-primer"><div><strong>坐标</strong>横轴是 head index <em>h</em>，纵轴是 zero-based layer ℓ；每个格子是一枚 head。</div><div><strong>颜色与边框</strong>颜色越深表示 exact-P0 query 对正确 next record 的 discovery seed-event mean raw mass 越大；红框是当前 frozen Top-K membership。</div><div><strong>为何要画</strong>它显示 bank 是集中在少数层还是跨层分布，也防止把 Top-K 误写成一个单层模块。</div></div>
<p>Atlas 的每格先在一个 event 内求 query 对正确 target-record span 的 attention mass，再在同一 seed 内平均 events，最后对 20 个 discovery seeds 等权平均。这样，一条 trace 中 occurrence 较多的 seed 不会因为事件数更多而自动获得更大权重。显示时对每个模型自己的分数取 99th percentile 作为色标上限，并对归一值开平方以显示弱但非零的 heads；红框 membership 使用未截断的冻结 rank。</p>
<div class="attention-atlas-stack"><figure><h3 class="figure-title">图 2d · Qwen Top-128 layer×head atlas（全宽）</h3><div class="attention-atlas-frame">{qwen_head_map}</div><figcaption>横轴=H0–H31，纵轴=L0–L35；每个格子是一枚 Qwen attention head。颜色是 20 discovery seeds 的 seed-equal exact-P0 target-record raw mass，经 Qwen 内部 99th-percentile 截断后显示；红框=冻结 Top-128。全宽图显示 selected bank 跨多个中后层分布，因此“Top-128”是宽 bank-level necessity，不表示 128 枚 heads 各自同等必要，也不表示一个单层 module。</figcaption></figure><figure><h3 class="figure-title">图 2e · Gemma Top-6 layer×head atlas（L0–20 / L21–41 分栏放大）</h3><div class="attention-atlas-frame">{gemma_head_map}</div><figcaption>左右 panel 分别是 L0–L20 与 L21–L41；每个 panel 横轴=H0–H7，纵轴=逐层 Lℓ，红框仅标当前冻结 Top-6，而非历史 Top-8。颜色在 Gemma 内独立按同一规则缩放。分栏只是版式变换，未改变 layer/head 坐标、分数、rank 或 bank membership；因此可读性提高但科研对象不变。</figcaption></figure></div>
<p><strong>Atlas 的证据边界。</strong> 高 attention mass 是 localization 指标，不等于该 head 因果必要；红框本身也可能包含冗余 heads。真正的 selection specificity 来自下一节把 selected bank 与同层、同 K random bank 做配对 ablation。反过来，random bank 也可能偶然包含有用 heads，所以 selected−random 是比 selected−clean 更严格、通常也更小的效应。</p>

<h3>4.4 关闭 frozen bank 后，retrieval failure 是否传播到最终计数</h3>
<div class="figure-primer"><div><strong>Panel A</strong>selected bank 相对同层、同 K random bank 增加多少 final-count failure。</div><div><strong>Panel B</strong>同一 trial 中同时出现 retrieval failure 与 count failure 的概率差；它检验干预是否传播到任务输出，但不是 formal natural-effects mediation。</div><div><strong>完全对齐</strong>两模型各使用完全相同的 10 个 confirmation seeds 与 semantic trial-key multiset；每个模型只做内部 selected−random 对比，不用横条绝对长度排序模型强弱。</div></div>
<figure><h3 class="figure-title">图 2f · Exact-aligned targeted-retrieval necessity 与输出传播</h3>{effect_small_multiples_svg('Experiment 4.4 · aligned targeted-retrieval confirmation', targeted_panels)}<figcaption>横轴是 selected−random 的 paired probability difference，点为 10 个 seed 等权的均值，线为 seed-cluster bootstrap 95% CI；纵直零线表示没有 selection specificity。两个 panel 独立缩放。Qwen/Gemma 的 bank size 分别为 Top-128/Top-6，故这里只读各模型内的 treatment−control effect；不能把 raw magnitude 当作同一 circuit 容量的跨模型比较。</figcaption></figure>
{targeted_table}
<p><strong>结果分析。</strong>两模型的 clean final-count accuracy 都是 100%。相对 layer-matched random bank，Qwen 的 final-count failure 增加 {probability_effect_ci(targeted_effects['Qwen3-8B']['targeted_bank_changes_final_count'])}，joint retrieval+count failure 增加 {probability_effect_ci(targeted_effects['Qwen3-8B']['retrieval_failure_propagates_to_count'])}；Gemma 分别为 {probability_effect_ci(targeted_effects['Gemma4-E4B']['targeted_bank_changes_final_count'])} 与 {probability_effect_ci(targeted_effects['Gemma4-E4B']['retrieval_failure_propagates_to_count'])}。这支持 bank-level necessity 及其向 final task 的传播；它不说明 bank 内每枚 head 同等必要，也不把 joint endpoint 当作完整中介估计。</p>
<div class="section-conclusion"><strong>Experiment 4 结论。</strong>在 exact-aligned confirmation support 上，关闭模型内冻结的 targeted-retrieval bank 会特异性增加最终计数失败，并在同一 trials 中增加 retrieval+count 的联合失败。因此 targeted retrieval 是两模型 counting pathway 的一条必要边；现有证据不要求它是唯一边或唯一 head set。</div></section>

<section id="write"><p class="eyebrow">05 · Write and recurrent propagation</p><h2>把 trace 内部的三条边分开检验</h2>
<p class="lead">这一节不是一次复杂 patch 得出整条链，而是依次问三个较小的问题。每个问题的 treatment、control 和读数都不同：</p>
<div class="edge-roadmap"><div><span>5.1 · READ → CARRIER</span><strong>关掉 targeted heads 后，紧随检索出现的 grammar state 会不会变？</strong></div><div><span>5.2 · CARRIER → COMMIT</span><strong>在 heads 仍被关闭时，只把 clean carrier 补回来，item-end commit 会不会恢复？</strong></div><div><span>5.3 · COMMIT → NEXT READ</span><strong>把“已完成 k”的 commit 换成“已完成 k+1”，下一次 query 会不会改读再下一条 record？</strong></div></div>

<h3>5.1–5.2 共用什么样本与实验底座</h3>
<p class="lead">5.1 与 5.2 不是先后另找两批“有效样本”。它们读取同一个预冻结的 <code>teacher_forced_targeted_counter_write</code> 七臂实验：5.1 看 retrieval query 之后的 carrier，5.2 看更晚的 item-end commit。这样才能在同一条 trace、同一次局部 head damage 下检验中介链。</p>
<div class="experiment-frame"><div><span class="experiment-label">独立样本</span><strong>一个 seed 对应一条完整 trace，才算一个独立样本。</strong>每模型使用 discovery seeds 1234–1253 的 20 条 trace，以及完全不重叠的 confirmation seeds 1254–1263 的 10 条 trace。每个 seed outcome-blind 地冻结一条 T10000 benchmark instance；该 trace 必须含恰好 <em>N</em> 个 parser-registered items，并有可定位的最后一次 <em>N−1</em>→<em>N</em> retrieval。</div><div><span class="experiment-label">不是样本</span>同一 seed 的 7 个 forward arms 是配对条件，不是 7 个独立样本；多个 carrier tokens、hidden coordinates 和 layers 也不是额外样本。统计时先在每个 seed 内算 treatment−control，再让 20 或 10 个 seeds 等权。</div><div><span class="experiment-label">实际规模</span>每模型 30 条独立 traces × 7 arms = 210 condition rows：discovery 20×7=140，confirmation 10×7=70。两模型合计 60 条独立 traces、420 condition rows。</div><div><span class="experiment-label">如何选 trace</span>先在每个 phase 内强制 grammar timing 平衡，再在可行候选中优先更高 <em>N</em>、固定 grammar preference 与 request id；不读取实验 outcome，也不使用 <code>selection_rank</code>。因此 count 分布不是 1–10 均匀抽样，而是一个 grammar-balanced、偏高 count 的因果 panel。</div></div>
{write_sample_table}
<p class="term-note"><strong>表中两个 grammar timing：</strong><em>city→rank</em> 表示 trace 先写 city，之后才写“这是第六条”一类 rank/count marker；<em>rank→city</em> 表示 trace 先写“第六条”，随后检索并写 city。每模型 discovery 严格 10/10，confirmation 严格 5/5。由于本实验需要 <em>N−1</em>→<em>N</em> transition，<em>N</em>=1 不可能进入样本。</p>
<h4>同一条 trace 的正式主分析使用哪 6 个 forward</h4>
{write_arm_table}
<p class="term-note">“Pre-O mask”指在注册 query 上，把入选 attention heads 的 <span class="formula">Σ<sub>t</sub>A(q,t)V(t)</span> slice 在进入该层 output projection <em>W</em><sub>O</sub> 前清零。它不删除整层，不改 prompt，也不直接 patch logits。Qwen 使用先前冻结的 Top-128；Gemma 使用 Top-6。三套 random banks 在每一层都匹配 selected bank 的 head 数量。原冻结协议额外跑过一个后来被判定为不可解释的 control，因此运行审计仍是 7 arms / trace；该 arm 不进入正文主分析，原因仅在 Appendix D 留档。</p>

<h3>5.1 关闭 targeted bank 后，检索结果有没有写入 grammar carrier</h3>
<div class="experiment-frame"><div><span class="experiment-label">要检验的箭头</span><strong>Targeted retrieval query → grammar carrier state。</strong>Experiment 4 已证明 bank mask 会使自由生成的 next city 检索失败；这里进一步问，即使把可见 trace 固定不让它分叉，关闭同一 bank 是否仍会改变检索之后的内部 state。</div><div><span class="experiment-label">Token 固定</span>把这条完整 clean trace 作为 teacher-forced token sequence，所有 arms 输入完全相同。Head mask 只发生在最后一次注册的 <em>N−1</em>→<em>N</em> query；所以不同 arm 不会因为生成了不同 city 或不同句子而落到不同 token positions。</div><div><span class="experiment-label">Carrier 怎么定</span>按 grammar outcome-blind 地注册。对于 <em>city→rank</em>，检索到 city 后才出现的新进度信息是 rank-marker core，因此读取该 marker 的全部 tokens；对于 <em>rank→city</em>，rank 已经先出现，检索后的新信息从 retrieved-city 第一个 token 延续到 item-end commit，因此读取整个 city-to-commit tail。Carrier 可以是多个 tokens，不要求固定长度。</div><div><span class="experiment-label">读取哪些层</span>Qwen 在 post-block L19–L35 读取 carrier，共 17 层；Gemma 在 L16–L41 读取，共 26 层。Source layers L19/L16 与 frozen targeted-counter protocol 一起预注册，没有按本实验 effect 重选。</div><div><span class="experiment-label">每 seed 的读数</span><span class="formula">D<sup>a</sup><sub>carrier</sub> = (1/|L|) Σ<sub>ℓ∈L</sub> ||H<sup>a,ℓ</sup><sub>carrier</sub> − H<sup>clean,ℓ</sup><sub>carrier</sub>||<sub>F</sub> / √(m·d)<br>Primary = D<sup>selected</sup><sub>carrier</sub><br>Selected-vs-random = D<sup>selected</sup><sub>carrier</sub> − (1/3)Σ<sub>r=1</sub><sup>3</sup>D<sup>random-r</sup><sub>carrier</sub></span><em>m</em> 是该 trace 的 carrier token 数，<em>d</em> 是 hidden width。先对 token×hidden dimensions 做 RMS，再对已注册 layers 平均，最后才跨 seeds 平均。</div><div><span class="experiment-label">具体例子 A</span><em>City→rank，N=6：</em>固定 trace 中出现“... Paris received a score ... That is the sixth record.”。5→6 query 位于检索 Paris 之前；我们只 mask 该 query 的 targeted heads，仍强制喂入完全相同的 Paris 和后续文字。读数取 “sixth” 这一 marker core 的 hidden states。如果它偏离 clean，说明 retrieval computation 会影响随后写出的进度 carrier。</div><div><span class="experiment-label">具体例子 B</span><em>Rank→city，N=6：</em>固定 trace 类似“Sixth, Paris ...”。Rank “Sixth” 在 query 前已可见，因此不能把它当作检索之后才写入的结果；这里 carrier 改取 Paris 第一个 token 到该 item commit 的 tail。这样两个 grammar 都遵守同一原则：只测 query 之后出现的状态。</div></div>
<div class="condition-list"><div class="condition-row"><strong>Primary：selected vs clean</strong><span>同一 seed、同一 token sequence、同一 carrier positions</span><span>回答 frozen bank 是否会改变 carrier；clean distance 按定义为 0</span></div><div class="condition-row"><strong>Secondary：selected vs 3×random</strong><span>random bank 逐层匹配 head 数，先在 seed 内取三者平均</span><span>回答变化是否特别归因于 selected identity，而不只是“关一批 heads 都会扰动”</span></div></div>
<div class="figure-primer"><div><strong>Panel A · 直接 damage</strong>selected-mask carrier 离 clean carrier 多远；回答“关 bank 有没有传到 carrier”。</div><div><strong>Panel B · head-identity control</strong>selected 的距离减去三套 layer-matched random 的平均距离；回答“是否特别是这组 heads”。</div><div><strong>每个 panel 独立缩放</strong>横轴都是 RMS distance，但 panel 自己定标；点=均值，线=95% CI。</div></div>
<figure><h3 class="figure-title">图 3a · 关掉 targeted heads 后 carrier hidden state 如何变化</h3>{effect_small_multiples_svg('Experiment 5.1 · targeted-head mask and carrier-state deformation', write_damage_panels)}<figcaption>两个 panel 都只属于 Experiment 5.1，不是 5.2 的 restoration。A 是 selected-mask 相对 clean 的总形变；B 是 selected 相对三套 random masks 的附加形变。在 exact-aligned 10-seed confirmation 中，两模型的 A/B 区间都完全高于 0。不同模型的 residual 尺度、bank width 与累计读取层数不同，不能用点的绝对距离排序机制强弱。</figcaption></figure>
<p><strong>Confirmation 结果。</strong> Selected mask 的 carrier deformation 为 Qwen {ci(write_effects['Qwen3-8B']['selected_carrier_deformation'])}、Gemma {ci(write_effects['Gemma4-E4B']['selected_carrier_deformation'])}。Selected 相对三套 layer-matched random 的附加 deformation 为 Qwen {ci(write_effects['Qwen3-8B']['selected_carrier_deformation_specificity'])}、Gemma {ci(write_effects['Gemma4-E4B']['selected_carrier_deformation_specificity'])}。</p>
<p><strong>如何读结果。</strong>两模型的 selected→clean 与 selected−random 都为正，所以冻结 targeted-bank intervention 会传到 query 之后的 carrier，而且该形变大于三套逐层 head-count matched random lesions。这个 identity specificity 支持“选中的 bank 与写入功能有关”，但不支持“只有这组 heads 能写入”或“其他 heads 完全冗余”。</p>
<h4>5.1b NCC：carrier 不只是“变了”，是否朝错误 count centroid 移动</h4>
<div class="experiment-frame"><div><span class="experiment-label">为什么重新分支运行</span>旧 NCC 虽然在拟合时区分 City→rank 与 Rank→city，却把两种 timing 放进同一个 20/10 panel、用二者平均 OOF 共同选层，并在最后汇总两个不可直接比较的 raw squared-distance margins。更重要的是，旧 City→rank carrier 直接读取 teacher-forced rank/count marker，本身已经包含要解码的显式 count token。新实验把 row panel、head bank、endpoint、layer selection 和 confirmation 统计全部按 timing 分开。</div><div><span class="experiment-label">City→rank 主 endpoint</span>使用 discovery atlas 中与 <code>adjacent_rank_after_city</code> 匹配的 P0 Top-K bank；读出点改为 <code>pre_marker_state</code>，即 rank marker 第一个 token 之前的精确 post-block state。注册的 secondary endpoint 是以该位置结尾的四-token suffix。全量 tokenizer audit 已验证两者均位于 retrieved city 之后、rank marker 之前，不含任何 marker/count token。</div><div><span class="experiment-label">Rank→city 主 endpoint</span>保留原冻结 rank-before bank 与 query anchor：Qwen 为 adjacent-rank-before / post-marker Top-128，Gemma 为 same-unit-rank-before / P0 Top-6；carrier 仍是 retrieved city 第一个 token 到 item commit 的 tail。</div><div><span class="experiment-label">Bank 与因果层</span>每个 timing 使用自己的 selected bank 与三套 selected-excluded、逐层 head 数完全匹配的 random banks。query head 在 L<em>h</em> 的输出最早只能影响更晚 token 的 L<em>h</em>+1 state，因此只在完整 bank 最高层之后读出；layer/head identity 是允许的架构特异参数。</div><div><span class="experiment-label">样本与冻结</span>本版先取两模型自然 trace 的 exact intersection，再运行同一 semantic multiset：City→rank 为 9 discovery / 4 confirmation，Rank→city 为 10 / 5；Qwen 与 Gemma 的 <code>(phase, seed, N, timing, condition)</code> 完全相同。缺少的 seed 不用 outcome 补齐。panel 与 bank 均不读取本实验 outcome；confirmation 不参与 PCA、centroid 或 layer selection。由于旧 NCC 变体已被看过，本轮仍标为 registered-split retrospective extension，而不是 pristine prospective confirmation。</div><div><span class="experiment-label">可比较效应</span>每个 endpoint 仍用 <span class="formula">loss=m(clean,c)−m(mask,c)</span>；raw margin 只在该模型×timing 坐标系内报告。跨分支汇报时再除以 discovery grouped-OOF correct-centroid margin 的标准差，得到 SD units。即便 intervention contrast 为正，若 clean confirmation 的平均 correct-centroid margin 不为正，则不能把它解释为“损伤了正确 count geometry”；这个 readout-validity 是看到初始 contrast 后增加的解释资格审计，不冒充预注册 gate。</div></div>
{stratified_ncc_bank_table}
{stratified_ncc_result_table}
<figure><h3 class="figure-title">图 3a-2 · Targeted-head mask 对 NCC count geometry 的影响</h3>{effect_small_multiples_svg('Experiment 5.1b · timing-stratified NCC carrier corruption', stratified_ncc_panels)}<figcaption>四行是模型×timing 的独立 primary estimand。Panel A 为 selected margin loss，Panel B 为 selected−三套 random mean；均以各自 discovery OOF margin SD 标准化。点=confirmation seed mean，线=95% seed-bootstrap CI。raw margins 不跨行合并。绿色需要同时满足有效 clean readout 与支持方向的完整区间；橙色表示区间触零、方向不支持或 readout validity 失败。</figcaption></figure>
<p><strong>City→rank：Qwen 得到小的 directional-specific geometry damage；Gemma clean readout 失效。</strong>Qwen 的 exact pre-marker endpoint 在 L{int(stratified_primary['Qwen3-8B']['rank_after_city']['selected_layer'])} 上有有效 clean confirmation readout（balanced accuracy={float(stratified_primary['Qwen3-8B']['rank_after_city']['readout_validity']['clean_confirmation_balanced_accuracy']):.3f}，mean correct-centroid margin={float(stratified_primary['Qwen3-8B']['rank_after_city']['readout_validity']['clean_confirmation_mean_correct_centroid_margin']):+.1f}）。Selected loss / specificity 为 {ci(stratified_primary['Qwen3-8B']['rank_after_city']['standardized_primary_estimand'], 4)} / {ci(stratified_primary['Qwen3-8B']['rank_after_city']['standardized_specificity_estimand'], 4)} SD：specificity interval 为正，但 selected-loss interval 跨 0，所以只能称 directional-specific，不能称两道 interval gate 均确认。Gemma 在 L{int(stratified_primary['Gemma4-E4B']['rank_after_city']['selected_layer'])} 的 clean balanced accuracy={float(stratified_primary['Gemma4-E4B']['rank_after_city']['readout_validity']['clean_confirmation_balanced_accuracy']):.3f}，但 mean correct-centroid margin={float(stratified_primary['Gemma4-E4B']['rank_after_city']['readout_validity']['clean_confirmation_mean_correct_centroid_margin']):+.1f}&lt;0；其 {ci(stratified_primary['Gemma4-E4B']['rank_after_city']['standardized_primary_estimand'], 4)} / {ci(stratified_primary['Gemma4-E4B']['rank_after_city']['standardized_specificity_estimand'], 4)} SD shift 不能解释成正确 count geometry 被损伤。四-token suffix secondary 没有提供比这更强的 two-gate 结论。</p>
<p><strong>Rank→city：两模型的 confirmation correct-centroid readout 都不具解释资格。</strong>Qwen L{int(stratified_primary['Qwen3-8B']['rank_before_city']['selected_layer'])} 的 standardized loss / specificity 为 {ci(stratified_primary['Qwen3-8B']['rank_before_city']['standardized_primary_estimand'], 4)} / {ci(stratified_primary['Qwen3-8B']['rank_before_city']['standardized_specificity_estimand'], 4)} SD，两个 interval 虽都为正，但 clean balanced accuracy={float(stratified_primary['Qwen3-8B']['rank_before_city']['readout_validity']['clean_confirmation_balanced_accuracy']):.3f}、mean margin={float(stratified_primary['Qwen3-8B']['rank_before_city']['readout_validity']['clean_confirmation_mean_correct_centroid_margin']):+.1f}，readout validity 失败。Gemma L{int(stratified_primary['Gemma4-E4B']['rank_before_city']['selected_layer'])} 的 shift 为 {ci(stratified_primary['Gemma4-E4B']['rank_before_city']['standardized_primary_estimand'], 4)} / {ci(stratified_primary['Gemma4-E4B']['rank_before_city']['standardized_specificity_estimand'], 4)} SD；clean balanced accuracy={float(stratified_primary['Gemma4-E4B']['rank_before_city']['readout_validity']['clean_confirmation_balanced_accuracy']):.3f} 但 mean margin={float(stratified_primary['Gemma4-E4B']['rank_before_city']['readout_validity']['clean_confirmation_mean_correct_centroid_margin']):+.1f}&lt;0，同样不能把正均值写成 confirmed geometry damage。</p>
<p><strong>Qwen/Gemma 对应性审计。</strong>两模型现在在设计上真正对应：各 timing 独立 exact-common panel、独立匹配 bank、独立 discovery 选层、完整 bank 均已因果可达、同一五臂 clean/selected/3×random factorial，并用同一 discovery-only SD 规则标准化。结果并不要求两模型或两种 grammar 同号：四支中只有 Qwen City→rank 同时通过 clean-readout validity 并出现 positive specificity interval；其余三支因 clean correct-centroid margin 为负而不可作 NCC damage 解释。这种异质性说明 NCC 是依赖 endpoint/层/坐标系的诊断，不是跨模型统一 counter code。两模型仍不是同剂量比较：Qwen/Gemma bank 分别为 128/6 heads，且没有 model×mask interaction；两个 timing bank 的历史重合度为 Qwen {int(stratified_ncc_inputs['Qwen3-8B']['selected_bank_overlap']['head_count'])}/128、Gemma {int(stratified_ncc_inputs['Gemma4-E4B']['selected_bank_overlap']['head_count'])}/6，该元数据只描述 bank overlap，不改变本轮 exact-common outcome。</p>
<h4>5.1c 直接 count-output margin：绕过 NCC centroid，检验最终答案分布</h4>
<div class="experiment-frame"><div><span class="experiment-label">为什么再加一个 margin</span>NCC 要先假设某层 hidden state 的欧氏 centroid geometry 是模型实际使用的 count code；clean decoder 若无效，或有效信息位于非线性/非欧氏方向，真实的输出变化也可能被 NCC 漏掉。直接 output margin 不拟合 PCA、decoder、centroid，也不再选择读出层，而是直接问局部 retrieval lesion 是否持续到最终 count distribution。</div><div><span class="experiment-label">Primary endpoint</span>在冻结 trace 的最终 <code>Total:</code> answer query 上，对候选答案字符串 1,…,10 计算完整 autoregressive sequence log probability：<span class="formula">M<sub>out</sub>=log p(y=c|prefix)−max<sub>j≠c</sub> log p(y=j|prefix)<br>Selected loss=M<sub>clean</sub>−M<sub>selected</sub><br>Specificity=Selected loss−mean<sub>r=1..3</sub>(Random-r loss)</span>正 margin 表示正确 count 的完整序列分数高于最强错误候选；正 loss 表示 intervention 削弱这一优势。</div><div><span class="experiment-label">干预仍然局部</span>五个 arms 与 timing-matched NCC 完全对应：clean、selected bank、三套逐层匹配 random banks。Mask 只在冻结的最后一次 <em>N−1</em>→<em>N</em> retrieval query 执行；随后 answer-candidate tokens 在没有 hook 的状态下计分。因此测到的是该精确 lesion 是否持久影响最终答案，不是直接在 answer token 上动手。</div><div><span class="experiment-label">三个判据</span>先要求 clean mean margin&gt;0 且 candidate accuracy&gt;chance；方向性证据再要求 selected loss&gt;0 且 specificity&gt;0；最强的 interval-confirmed 判据要求这两个 confirmation seed-bootstrap 95% CI 的下界都&gt;0。四个模型×timing 分支分别报告，不合并 raw margins。</div><div><span class="experiment-label">注册边界</span>endpoint、五臂 contrast 与 gate 都在查看本轮 logit-margin outcome 前冻结；但沿用的 confirmation split 已在 NCC 分析中看过，所以这是 <em>registered existing-split retrospective extension</em>，不是 pristine prospective confirmation。模型间也没有 model×mask interaction，不能把 Gemma 与 Qwen 的数值差直接当作已检验的模型差异。</div></div>
<p class="term-note"><strong>证据层级。</strong>5.1a 的 carrier RMS deformation 才是“局部 hidden state 是否被改变”的主终点；5.1c 是更下游的 secondary functional diagnostic，不是确认 state change 的必要条件。最终答案还会受后续层重算、冗余通路、answer-time retrieval、候选 tokenization 与输出校准影响，所以 output-margin null 不能否定 hidden-state change，output-margin positive 也不能把全部效应归因于一个局部 count code。</p>
{direct_margin_result_table}
<p class="term-note">† <em>Directional-specific</em> 表示 clean readout 有效，且 confirmation 的 selected loss 与 selected−random specificity 均值都为正；它不自动要求 discovery 同号或两条 confirmation CI 都排除 0。<em>Interval-confirmed</em> 才要求同一行的两条 confirmation CI 下界都大于 0。各 timing 的 discovery/confirmation 值在表内分开显示。</p>
<figure><h3 class="figure-title">图 3a-3 · Retrieval-query mask 对最终 count-output margin 的影响</h3>{effect_small_multiples_svg('Experiment 5.1c · direct final count-output margin', direct_margin_panels)}<figcaption>四行仍是模型×timing 的独立 primary estimands。Panel A 是 clean−selected 的最终答案 sequence-log-prob margin loss；Panel B 再减去三套 random-mask losses 的均值。点=confirmation seed mean，线=95% seed-bootstrap CI。绿色只表示该单项 CI 完全高于 0；只有 Gemma City→rank 同时通过 A/B 两条 interval gate。其余行必须按方向、区间与 discovery 对照分别解释。</figcaption></figure>
<p><strong>先看 readout 是否可信。</strong>四支 clean candidate accuracy 都是 1.000，confirmation clean mean margin 为 Qwen City→rank {float(direct_margin_primary['Qwen3-8B']['rank_after_city']['confirmation']['clean_mean_margin']):+.3f}、Rank→city {float(direct_margin_primary['Qwen3-8B']['rank_before_city']['confirmation']['clean_mean_margin']):+.3f}，Gemma 分别为 {float(direct_margin_primary['Gemma4-E4B']['rank_after_city']['confirmation']['clean_mean_margin']):+.3f} 与 {float(direct_margin_primary['Gemma4-E4B']['rank_before_city']['confirmation']['clean_mean_margin']):+.3f}。因此四个 primary 都通过 readout-validity gate；这里不存在 Qwen Rank→city NCC 那种 clean geometry 已经把正确 count 放在错误侧的问题。</p>
<p><strong>Qwen：City→rank 只有 confirmation 方向性，Rank→city 没有 specificity。</strong>City→rank 的 confirmation selected loss / specificity 为 {ci(direct_margin_primary['Qwen3-8B']['rank_after_city']['confirmation']['selected_margin_loss'])} / {ci(direct_margin_primary['Qwen3-8B']['rank_after_city']['confirmation']['selected_vs_random_specificity'])}；两者均值为正但区间都跨 0，而且 discovery specificity 为 {float(direct_margin_primary['Qwen3-8B']['rank_after_city']['development']['selected_vs_random_specificity']['mean_effect']):+.3f}，并未跨 split 同号。Rank→city 为 {ci(direct_margin_primary['Qwen3-8B']['rank_before_city']['confirmation']['selected_margin_loss'])} / {ci(direct_margin_primary['Qwen3-8B']['rank_before_city']['confirmation']['selected_vs_random_specificity'])}，confirmation specificity 为负。因此 Qwen 的这项 downstream diagnostic 不能承担稳定 selected-specific output effect 的主张。</p>
<p><strong>Gemma：City→rank 通过两条 interval gate；Rank→city 只有 directional-specific support。</strong>City→rank 的 selected loss / specificity 为 {ci(direct_margin_primary['Gemma4-E4B']['rank_after_city']['confirmation']['selected_margin_loss'])} / {ci(direct_margin_primary['Gemma4-E4B']['rank_after_city']['confirmation']['selected_vs_random_specificity'])}，两条 CI 下界都大于 0，是本轮唯一 interval-confirmed 分支。Rank→city 为 {ci(direct_margin_primary['Gemma4-E4B']['rank_before_city']['confirmation']['selected_margin_loss'])} / {ci(direct_margin_primary['Gemma4-E4B']['rank_before_city']['confirmation']['selected_vs_random_specificity'])}：specificity interval 为正，但 selected-loss CI 仍跨 0。按 clean margin 归一化，两支 selected loss 约为 {100*float(direct_margin_primary['Gemma4-E4B']['rank_after_city']['confirmation']['selected_margin_loss']['mean_effect'])/float(direct_margin_primary['Gemma4-E4B']['rank_after_city']['confirmation']['clean_mean_margin']):.1f}% 与 {100*float(direct_margin_primary['Gemma4-E4B']['rank_before_city']['confirmation']['selected_margin_loss']['mean_effect'])/float(direct_margin_primary['Gemma4-E4B']['rank_before_city']['confirmation']['clean_mean_margin']):.1f}%。这说明局部 retrieval lesion 可在 Gemma 的最终答案分布中留下小而可测的净效应；由于沿用已看过 NCC 的 split，它仍是 registered existing-split extension，不是 pristine prospective confirmation。</p>
<p><strong>City→rank 的局部 marker secondary 没有提供更强结论。</strong>Qwen 的 clean local sequence margin={float(direct_margin_local['Qwen3-8B']['confirmation']['clean_mean_margin']):+.3f}、accuracy={float(direct_margin_local['Qwen3-8B']['confirmation']['clean_accuracy']):.3f}，readout validity 失败；marker 候选的表面 tokenization/长度差使其正 specificity 不可解释。Gemma 的 local readout 有效（clean margin={float(direct_margin_local['Gemma4-E4B']['confirmation']['clean_mean_margin']):+.3f}、accuracy={float(direct_margin_local['Gemma4-E4B']['confirmation']['clean_accuracy']):.3f}），但 selected loss / specificity 为 {ci(direct_margin_local['Gemma4-E4B']['confirmation']['selected_margin_loss'])} / {ci(direct_margin_local['Gemma4-E4B']['confirmation']['selected_vs_random_specificity'])}，均不支持 damage。也就是说，Gemma 的阳性更明确地出现在最终答案 margin，而不是紧邻 City→rank marker 的这个特定表面读出；这不等于已经定位了中间的完整中介路径。</p>
<div class="qualification"><strong>NCC 与 output margin 并不矛盾。</strong>RMS norm 证明 hidden state 被 intervention 改变；NCC 问它是否沿一个 discovery-fitted centroid geometry 移动；output margin 则记录经过全部后续计算后的净答案效应。Gemma 的组合结果至少说明 state change 与弱 downstream signal 可以并存；但仅凭当前实验无法区分“NCC 没捕获到模型使用的方向”“count code 在后层被重新编码”与“其他通路补偿后只留下小净效应”，更不能据此宣称一个唯一 count code。</div>
<details class="paper-appendix"><summary>历史 frozen pooled NCC 与事后 layer×timing 诊断（仅用于复现旧 headline）</summary>
<div class="experiment-frame"><div><span class="experiment-label">为什么补 NCC</span>RMS distance 只回答 masked state 离 clean state 多远；它不知道移动方向是否与 running count 有关。Nearest-centroid classifier（NCC）把 discovery clean carrier 的 count geometry 冻结下来，再问 confirmation state 更靠近哪个 count centroid。</div><div><span class="experiment-label">训练样本</span>City→rank 使用两模型共同可用的 9 discovery seeds，Rank→city 使用共同可用的 10 seeds；每个已完成 occurrence 都取 grammar carrier vector。每个 stratum 内先 standardize，再 PCA-16，最后计算各 count centroid。交叉验证按 seed 分组，任何同一 seed 的 occurrence 都不会同时出现在 train/test。</div><div><span class="experiment-label">冻结的层选择</span>每个模型×timing×endpoint 只在对应 discovery common support 上计算 grouped-OOF balanced accuracy并冻结层；4/5 个 confirmation seeds 不参与 PCA、centroid 或 layer selection。旧 pooled NCC 与逐层诊断只留在 appendix，不替换本版 exact-common primary。</div><div><span class="experiment-label">Confirmation 条件</span>每条 confirmation trace 在同一 frozen carrier 上比较 clean、selected bank mask 与三套逐层匹配 head 数的 random masks。所有条件共享完全相同的 teacher-forced tokens 与 positions；区别只在注册 retrieval query 的 pre-O head output 是否被清零。</div><div><span class="experiment-label">Correct-centroid margin</span><span class="formula">m(z,c)=min<sub>j≠c</sub>||z−μ<sub>j</sub>||²−||z−μ<sub>c</sub>||²<br>Margin loss = m(clean,c)−m(mask,c)</span>正 margin 表示正确 centroid 比任何错误 centroid 更近；正 loss 表示 mask 把 state 推离正确 count geometry。Specificity 再减去三套 random-mask loss 的 seed 内平均。</div><div><span class="experiment-label">重要尺度限制</span>City→rank 与 Rank→city 使用各自独立的 standardization、PCA 与 centroids，因此 raw squared-distance margins 不具有天然相同尺度；不同模型的 bank size 也不同。正文只作模型内 treatment−control 推断，不按条长比较模型强弱。</div></div>
<h4>冻结主分析：保留结果，但收窄解释</h4>
{ncc_result_table}
<figure><h3 class="figure-title">历史图 H5.1 · 原 pooled NCC margin</h3>{effect_small_multiples_svg('Historical Experiment 5.1b · pooled NCC carrier corruption', ncc_panels)}<figcaption>这是原冻结层与原 pooled estimand，只用于复现旧 headline。Panel A 是 clean→mask correct-centroid margin loss；Panel B 再减去三套 layer-matched random masks 的平均 loss。它不提供 Qwen–Gemma interaction test，也不校准两个 grammar-specific NCC 坐标系的 margin 尺度。</figcaption></figure>
<p><strong>冻结 NCC 结果。</strong> Qwen selected margin loss={ci(ncc_primary['Qwen3-8B'])}、selected−random={ci(ncc_specificity['Qwen3-8B'])}，只能记为未确认的方向性信号；hard NCC exact 保持 {float(ncc_condition['Qwen3-8B']['clean']['exact_accuracy']):.2f}→{float(ncc_condition['Qwen3-8B']['selected_mask']['exact_accuracy']):.2f}。Gemma pooled selected loss={ci(ncc_primary['Gemma4-E4B'])}、specificity={ci(ncc_specificity['Gemma4-E4B'])}，hard exact 同样保持 {float(ncc_condition['Gemma4-E4B']['clean']['exact_accuracy']):.2f}→{float(ncc_condition['Gemma4-E4B']['selected_mask']['exact_accuracy']):.2f}。因此 frozen primary 既没有确认 Gemma 的 late full-Top-6 NCC damage，也没有检验出“Gemma 比 Qwen 更弱”的模型差异。</p>

<h4>Qwen/Gemma 对应性审计：数据可复现，但不是同剂量、同 cohort 的模型比较</h4>
{ncc_correspondence_table}
<p><strong>先确认结果文件本身。</strong>我们从两模型原始 NPZ 独立重跑 frozen analyzer：Qwen 仍选择 L23，Gemma 仍选择 L34；selected loss、specificity 与 hard exact 等冻结汇总的最大绝对差分别只有 {float(ncc_layerwise_diagnostic['Qwen3-8B']['frozen_result_reproduction']['maximum_absolute_difference']):.6f} 和 {float(ncc_layerwise_diagnostic['Gemma4-E4B']['frozen_result_reproduction']['maximum_absolute_difference']):.6f}，来自不同平台 SVD/浮点舍入。条件顺序、teacher-forced token contract、20/10 split、10/10 与 5/5 timing 配额均一致。因此 shard、聚合代码和报告数字之间没有发现错位。</p>
<p class="term-note"><strong>Hook audit 字段说明。</strong>旧 NPZ 的 condition metadata 中 <code>head_hook_applications</code> 为空，是 capture caller 读取了旧字段名；底层 intervention helper 仍会强制每个注册 layer 恰好执行一次、否则直接报错，且逐层 state difference 只从因果可达层开始出现。该问题不改变已有 vectors 或统计，但削弱了旧 shard 的持久化可读审计；后续 capture 已同时保存 application counts 与 post-zero maximum。</p>
<p><strong>再确认“对应”到什么程度。</strong>两者使用同一 NCC pipeline 和同构 selected/random 条件，可以做各自模型内的因果诊断；但 Qwen confirmation 主要是 N=10，而 Gemma 的 N 更分散，两个 bank 的大小、定位 grammar/anchor 和层分布也不同。更关键的是，query head 在 L 的输出要到后续 carrier 的 L+1 才可达：Qwen frozen L23 实际只有 24/128 selected heads 已能影响读数，Gemma L34 则为 6/6。因此 raw margin loss 不能作为 calibrated Qwen-vs-Gemma effect size，当前也没有 model×mask interaction test。</p>

<h4>同一 frozen layer 内拆开 timing 后，两模型呈现相同的符号结构</h4>
{ncc_frozen_timing_table}
<p><strong>核心对应关系。</strong>两个 frozen banks 都是在 rank-before-city 家族上定位的。与 bank 匹配的 Rank→city stratum 中，Qwen L23 selected loss={float(qwen_ncc_l23_before['selected_margin_loss_mean']):+.2f}、specificity={float(qwen_ncc_l23_before['selected_vs_random_specificity_mean']):+.2f}，5/5 seeds 为正；Gemma L34 也保持正方向（{float(gemma_ncc_l34_before['selected_margin_loss_mean']):+.2f}、{float(gemma_ncc_l34_before['selected_vs_random_specificity_mean']):+.2f}），只是较弱。相反，非 bank-matched 的 City→rank stratum 在两模型都是负方向：Qwen {float(qwen_ncc_l23_after['selected_margin_loss_mean']):+.2f}，Gemma {float(gemma_ncc_l34_after['selected_margin_loss_mean']):+.2f}。所以 Qwen pooled +29.39 与 Gemma pooled −16.58 不是相反机制；它们都是“Rank→city 正、City→rank 负”，区别在于两个独立坐标系的哪一项数值占主导。</p>
<p><strong>统计边界。</strong>Qwen Rank→city 的 bootstrap interval 为 {ci(qwen_ncc_l23_before['selected_margin_loss_summary'])}，但 n=5 的精确 sign-flip <em>p</em>={float(qwen_ncc_l23_before['selected_margin_loss_summary']['p_value']):.3f}；Gemma L34 Rank→city 区间跨 0。因此这次拆分更正了机制方向的解释，却没有把任一模型升级成新的确认性结果。</p>

<h4>Gemma layer×timing 事后诊断：负的 pooled result 从哪里来</h4>
<div class="experiment-frame"><div><span class="experiment-label">重算对象</span>直接读取原 20 discovery + 10 confirmation NPZ，在每个 L16–L41、每个 grammar timing 上重新拟合同构 discovery NCC，并计算同一批 confirmation mask effects；没有重新运行模型，也没有改变任何 token 或 intervention。</div><div><span class="experiment-label">因果层窗口</span>Gemma Top-6 由 L17 三头与 L29 三头构成。query token 在 L17 的输出最早只能影响后续 carrier token 的 L18 state；L29 heads 最早从 L30 才可达。因此 L16–L17 是零效应 sanity check，L18–L29 主要隔离 L17 三头，L30+ 才包含 full Top-6 的传播。</div><div><span class="experiment-label">推断边界</span>这次诊断是在看到原 confirmation 结果后进行。L27 是按 discovery-only、bank-matched rank→city OOF 在 L17-only causal window 内选出的诊断层；L28 是相邻层稳定性检查。二者都不能升级为新的 confirmation gate，必须用新 seeds 复验。</div></div>
{gemma_ncc_layerwise_table}
<p><strong>层选择错配。</strong>Rank→city 的全层 discovery OOF 最大值其实在 L16（0.532），但该层位于 L17 heads 上游，mask 在这里按因果结构必须为 0。进入 L17-only 可达窗口后，discovery-only 最佳层是 L27（OOF={float(gemma_ncc_l27_before['discovery_grouped_oof_balanced_accuracy']):.3f}）；full Top-6 可达窗口内最佳层才是 L34，但其 rank→city OOF 只剩 {float(gemma_ncc_l34_before['discovery_grouped_oof_balanced_accuracy']):.3f}。与此同时，city→rank OOF 在 L{int(gemma_ncc_layerwise_diagnostic['timing_crossover']['rank_after_first_oof_at_least_0_95'])} 首次超过 0.95，并从 L{int(gemma_ncc_layerwise_diagnostic['timing_crossover']['rank_after_first_perfect_oof'])} 起达到 1.00；原平均-OOF rule 因而被晚层 city→rank decoder 主导，选到 L34。</p>
<p><strong>中层存在探索性 rank→city damage。</strong>在 discovery-only 诊断层 L27，selected loss={ci(gemma_ncc_l27_before['selected_margin_loss_summary'])}，selected−random={ci(gemma_ncc_l27_before['selected_vs_random_specificity_summary'])}，4/5 seeds 为正。相邻 L28 上，selected loss={ci(gemma_ncc_l28_before['selected_margin_loss_summary'])}，specificity={ci(gemma_ncc_l28_before['selected_vs_random_specificity_summary'])}，5/5 seeds 为正；但精确 sign-flip <em>p</em>={float(gemma_ncc_l28_before['selected_margin_loss_summary']['p_value']):.3f}，受 n=5 的离散分辨率限制，仍不构成确认性证据。该窗口位于 L29 heads 可传播之前，因此更具体地指向 L17 三个 retrieval heads，而不是完整 Top-6。</p>
<p><strong>为什么最终变成 −16.58。</strong>在 frozen L34，rank→city selected loss 仍是 {float(gemma_ncc_l34_before['selected_margin_loss_mean']):+.2f}、specificity {float(gemma_ncc_l34_before['selected_vs_random_specificity_mean']):+.2f}；city→rank 却是 selected loss {float(gemma_ncc_l34_after['selected_margin_loss_mean']):+.2f}、specificity {float(gemma_ncc_l34_after['selected_vs_random_specificity_mean']):+.2f}。后者的 clean margin 均值约 {float(gemma_ncc_l34_after['clean_margin_mean']):.0f}，前者约 {float(gemma_ncc_l34_before['clean_margin_mean']):.0f}。两个独立 NCC 坐标系被等权汇总后，高尺度、晚层、可完美解码的 city→rank 项把 pooled mean 拉成负值。这更像显式 marker/grammar readout 与 layer pooling 的组合，而不是“Gemma 没有 count state”。</p>
<div class="plain-language"><strong>最简单的读法：</strong>Gemma 的 frozen L34 pooled NCC 没有确认 full Top-6 会损伤 late count geometry；但原始逐层数据也不支持“Gemma 完全没有 NCC effect”。更精确的定位是：L17 targeted heads 对 L18–L28 的 rank→city geometry 有探索性的选择性损伤，随后该 geometry 在 L29 附近被重写、修复或换成晚层 marker-dominated representation。</div>
</details>
<div class="section-conclusion"><strong>Experiment 5.1 结论。</strong>主结论来自 RMS hidden-state distance：retrieval lesion 会改变 query 之后的 carrier；这个结论不依赖最终 count 是否翻转。修复 marker 泄露、timing pooling 与因果层可达性后，完整 frozen bank→late correct-count NCC chain 仍受 readout validity/interval 稳定性限制。更下游的 final count-output margin 中，四支 clean readout 全部有效：Qwen 未得到跨 split 稳定的 selected-specific output effect；Gemma City→rank 通过两条 confirmation interval gate，Rank→city 只得到 directional-specific support。这个小净效应支持局部 lesion 可传播到最终分布，但不把它等同于局部 state 的全部功能或唯一通路。原 pooled headline 只保留为历史复现。</div>

<h3>5.2 在同一 query damage 下，恢复 clean carrier 能否救回 item-end commit</h3>
<div class="experiment-frame"><div><span class="experiment-label">要检验的箭头</span><strong>Grammar carrier state → later commit state。</strong>5.1 只说明 carrier 会随 retrieval damage 改变，仍可能只是旁观者。5.2 保持 selected head mask 不变，只操纵 carrier，观察更晚的 commit 是否被救回。</div><div><span class="experiment-label">三个必要条件</span>Clean forward 提供 clean carrier 与 clean commit；selected-mask forward 提供 damaged baseline；selected-mask + clean-carrier clamp 是主 restoration。后两臂使用同一 query mask、visible tokens 和 carrier positions，唯一改变是否把 clean carrier hidden vectors 写回。</div><div><span class="experiment-label">Clamp 的位置和层</span>Qwen 在 L19–L34、Gemma 在 L16–L40，把 clean carrier vectors 写回相同 carrier token positions；即从 source layer 一直 clamp 到倒数第二个 block。最后一层不再 clamp，而是在 Qwen L35 / Gemma L41 读取 boundary commit。逐层 cumulative clamp 回答“在 head damage 已存在时，维持正确 carrier 是否足以让下游 commit 恢复”，不回答“只 patch 哪一层就足够”。</div><div><span class="experiment-label">Commit 怎么读</span>Boundary commit 是 parser/causal compiler 注册的 <code>post_update_commit_state</code> tokens，严格位于 terminal item 内、answer query 之前。只在最后一个 post-block layer 计算它到 clean boundary 的 full-vector RMS distance。</div><div><span class="experiment-label">每 seed 的读数</span><span class="formula">D<sup>a</sup><sub>commit</sub> = ||H<sup>a,last</sup><sub>boundary</sub> − H<sup>clean,last</sup><sub>boundary</sub>||<sub>F</sub> / √(b·d)<br>Clean-carrier recovery = D<sup>selected</sup><sub>commit</sub> − D<sup>selected+clean-carrier</sup><sub>commit</sub></span><em>b</em> 是 boundary token 数。正的 recovery 表示：在同一 selected-head damage 下，补回 clean carrier 使 commit 更接近 clean reference。</div><div><span class="experiment-label">完整例子</span>仍用 <em>city→rank, N=6</em>。先运行 clean arm，保存 “sixth” carrier 在各层的 vectors 和句末 commit。第二个 arm 在 5→6 query 关闭 selected heads，得到偏离 clean 的 commit。第三个 arm 保持同一 mask，但在每一层把 clean “sixth” vectors 写回 “sixth” positions。若 selected commit distance=0.80，restored distance=0.30，则 recovery=0.50。</div></div>
<div class="plain-language"><strong>图 3b 只回答两件事。</strong>A 先确认 selected-mask damage 确实到了 later commit；B 再检验在同一 damage 下补回 clean carrier 能否 rescue commit。A 是 damage sanity check，B 是 5.2 的核心因果对比。</div>
<div class="figure-primer"><div><strong>Panel A · damage sanity check</strong>selected mask 下的 commit 离 clean commit 多远；若为 0，restoration 就没有可救对象。</div><div><strong>Panel B · main causal rescue</strong>同一 selected mask 下，clean-carrier clamp 减少了多少 commit distance；这是 5.2 的核心数字。</div><div><strong>每个 panel 独立缩放</strong>点是 10 confirmation seeds 的 paired mean，线是 95% CI；两 panel 量纲相同，但为了可读性独立定标。</div></div>
<figure><h3 class="figure-title">图 3b · 同一 head damage 下，clean carrier 能否救回 later commit</h3>{effect_small_multiples_svg('Experiment 5.2 · carrier-mediated commit restoration', write_mediation_panels)}<figcaption>A 是 selected-mask commit damage；B 是 clean-carrier 相对该 damaged baseline 减少的 commit distance。两个对比都在相同 seed、相同 teacher-forced trace 和相同 query mask 内配对计算；图中只保留主因果问题必需的 damage 与 rescue。</figcaption></figure>
{write_table}
<p><strong>Confirmation 结果。</strong> Clean-carrier recovery 为 Qwen {ci(write_effects['Qwen3-8B']['clean_carrier_restoration'])}、Gemma {ci(write_effects['Gemma4-E4B']['clean_carrier_restoration'])}。正值表示 clean-carrier clamp 使 later commit 更接近同一 trace 的 clean commit。</p>
<p><strong>换成更直观的恢复比例。</strong>Selected mask 对 final commit 造成的平均 damage 是 Qwen {float(write_effects['Qwen3-8B']['selected_boundary_deformation']['mean_effect']):.3f}、Gemma {float(write_effects['Gemma4-E4B']['selected_boundary_deformation']['mean_effect']):.3f}；clean-carrier recovery 分别是 {float(write_effects['Qwen3-8B']['clean_carrier_restoration']['mean_effect']):.3f} 与 {float(write_effects['Gemma4-E4B']['clean_carrier_restoration']['mean_effect']):.3f}。用两个 confirmation mean 做描述性比值，约恢复原 damage 的 {100*write_recovery_fraction['Qwen3-8B']:.1f}% 和 {100*write_recovery_fraction['Gemma4-E4B']:.1f}%。这个比例是帮助读数的 effect-size summary，不是另外注册的 seed-level estimand，所以不为它单独报 CI。</p>
<div class="plain-language"><strong>用一句话概括 5.1–5.2：</strong>同一条固定 trace 上，先只破坏最后一次 targeted retrieval，随后看到 carrier 与 commit 偏离 clean；保持这次 query damage 不变，只补回正确 carrier，commit 又朝 clean 回来。因为 damage 和 mediator restoration 在同一 seed 内配对，这比“carrier 与 commit 相关”更接近一条串联因果边。</div>
<div class="section-conclusion"><strong>Experiment 5.2 结论。</strong> 在 10 条独立 confirmation traces/model 上，保持 targeted-head damage 不变时，cumulative clean-carrier clamp 能部分恢复最终 item-end commit。因此 carrier 不只是随 retrieval 一起变化的旁观 state；它对 later commit 具有受控的因果作用。证据针对当前 grammar-balanced final-transition panel，不声称单层、单 token 或所有 counts 上都同样充分。</div>

<h3>5.3 Commit state 是否决定下一次 targeted query 读向哪里</h3>
<p class="lead">5.1–5.2 已经说明一次 retrieval 会写出 carrier，再提交成 item-end state；5.3 换一个问题：<strong>这个 item-end state 会不会成为下一轮 retrieval 的控制输入？</strong> 若会，trace 中就形成 READ→WRITE→COMMIT→NEXT READ 的 recurrent loop。</p>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>只改“已完成 item <em>k</em>”处的 commit hidden state，随后观察紧接着的 targeted query 是否改读另一个 prompt record。下一 query 的 token、prompt records 和 targeted head bank 均不改。</div><div><span class="experiment-label">什么是一个样本</span>一个 seed 的完整 trace 是一个独立样本。每个 seed 内 outcome-blind 地编译两个 same-trajectory 相邻 pairs：signed offset <em>d</em>=−1 与 +1；五个 conditions 都是 seed 内配对重复，不当作新的独立 seeds。Discovery 为 20 seeds×2 pairs，confirmation 为 10 seeds×2 pairs；Qwen/Gemma 的 <code>(phase, seed, gold_count, receiver, donor, d, condition)</code> 多重集完全相同。</div><div><span class="experiment-label">角色</span>Receiver 是要被改写的 completed-item commit；donor 是同一 trace 中相邻 occurrence 的 commit。Qwen 在 L19、Gemma 在 L16 读取 donor 的 single-token post-block state并写到 receiver 的同层位置。层是架构特异参数；semantic pairs 不是。</div><div><span class="experiment-label">五个条件</span>Clean 保留自然轨迹；self patch 控制 hook；full donor patch 移植完整 commit vector；count-subspace transplant 只移植 frozen linear count component；orthogonal patch 提供同范数的非该 subspace 扰动。下表把每个 arm 的因果问题逐项列出。</div><div><span class="experiment-label">主终点</span><span class="formula">Y = Σ<sub>h∈bank</sub>A<sub>h</sub>(donor-successor record) − Σ<sub>h∈bank</sub>A<sub>h</sub>(receiver-successor record)<br>Direct effect = Y<sub>full donor</sub> − Y<sub>self</sub><br>Local specificity = Y<sub>full donor</sub> − Y<sub>orthogonal</sub></span>Y&gt;0 表示下一 query 的 bank-level attention 更偏向 donor 所对应的下一条 record。</div><div><span class="experiment-label">简单例子</span>Receiver 表示“刚完成第 4 条”，自然下一 query 应读 record 5；donor 表示“刚完成第 5 条”，其 successor 是 record 6。若只把 donor 的 commit hidden state 写到 receiver，下一 query 对 record 6 相对 record 5 的 attention 增加，就说明 commit 在控制下一轮 routing。可见文本中没有额外写入数字 5，也没有 patch query 本身。</div></div>
{query_sample_table}
<p class="term-note"><strong>实验规模。</strong> Discovery 每模型 20×6=120 pairs，confirmation 每模型 10×6=60 pairs；每 pair 注册 5 个 conditions。因此协议对应每模型 600 个 discovery condition rows 与 300 个 confirmation condition rows。正式统计先在相同 seed、相同 pair 内求 treatment−control，再对 seeds 等权。</p>
{query_arm_table}
<div class="figure-primer"><div><strong>第一行</strong>full donor−self：回答“完整 commit 换过去，下一 query 是否真的转向 donor successor”。这是直接边。</div><div><strong>第二行</strong>full donor−orthogonal：回答“完整 donor state 是否优于一个同等 count-component 范数、但不在 frozen count subspace 内的一般扰动”。这是更严格的局部 specificity。</div><div><strong>读数单位</strong>两行都是 frozen-bank-summed raw attention mass。Qwen bank 有 128 heads，Gemma 只有 6 heads，因此<strong>不能用条长绝对值跨模型比较机制强弱</strong>；应在各模型内比较 treatment 与 control。</div></div>
<figure><h3 class="figure-title">图 4a · Commit patch 是否把下一次 query 转向 donor-successor record</h3>{grouped_bars_svg('Confirmation mean effects', query_groups)}<figcaption>每个横条是 10 confirmation seeds 的 paired mean effect。Qwen full-vs-self / full-vs-orthogonal 分别为 {float(query_effects['Qwen3-8B']['self']['mean_effect']):+.3f} / {float(query_effects['Qwen3-8B']['orthogonal']['mean_effect']):+.3f}；Gemma 为 {float(query_effects['Gemma4-E4B']['self']['mean_effect']):+.3f} / {float(query_effects['Gemma4-E4B']['orthogonal']['mean_effect']):+.3f}。正值表示 full donor commit 更能让下一 query 偏向 donor 的下一条 record。</figcaption></figure>
{query_table}
<figure><h3 class="figure-title">图 4b · Exact-common 相邻 pair 的 local specificity</h3>{grouped_bars_svg('Full commit − orthogonal targeted-attention effect', [('|donor−receiver| = 1', {model: float(query_effects[model]['orthogonal']['mean_effect']) for model in MODELS})])}<figcaption>唯一正式距离为 |donor−receiver|=1；Qwen/Gemma 使用完全相同的 20+10 seeds 与相邻 pair keys。横条是 full-donor patch 相对 orthogonal patch 的 targeted-attention effect。Orthogonal 扰动只与 count-projected delta 等范数，不与完整 donor−receiver delta 等范数，因此它是局部线性-subspace control，不是对完整 state 的等范数 placebo。</figcaption></figure>
{query_distance_table}
<div class="qualification"><strong>为什么 Gemma 的 commit→next-query 现在标 confirmed。</strong> 最直接的 full donor−self 对比在独立 confirmation 上是 {ci(query_effects['Gemma4-E4B']['self'])}；更严格的 |d|=1 full−orthogonal 对比是 {ci(query_effects['Gemma4-E4B']['orthogonal'])}，区间同样完全高于 0。因此 exact-common panel 同时确认“完整 commit 会改变下一 query”和“它优于当前冻结的 norm-matched orthogonal placebo”。限定仍然重要：该 placebo 只匹配 count-projected component 的范数，并不匹配完整 donor−receiver state difference；所以结果确认一条受控因果边，不确认唯一低维 counter 或排他回路。</div>
<div class="scope-compare"><div><h4>不能用跨模型条长排序机制强弱</h4><p>图的 outcome 对 bank heads 求和；Qwen 是 Top-128，Gemma 是 Top-6，所以 raw attention-mass effect 不在同一容量尺度。结论只读取每个模型内 treatment−control 的方向与区间。</p></div><div><h4>Full−orthogonal 回答的是较窄问题</h4><p>若该对比弱，可能因为 commit 的功能信息分布在 frozen count subspace 之外，也可能因为 bank 对一般 query-state 扰动敏感；当前实验不能区分。它不应被改写成“完整 commit state 没有作用”。</p></div></div>
<p><strong>结果分析。</strong> Qwen 的 direct effect 与 local specificity 都大，支持强 recurrent routing。Gemma 的完整 commit 同样会重定向下一 query，但“这条作用只来自一个狭窄线性 counter direction”没有得到同等强度的支持。因而合理 claim 是 shared recurrent computation、不同 state geometry，而不是两模型都实现同一条线性 counter axis。</p>
<div class="section-conclusion"><strong>Experiment 5.3 结论。</strong> 两模型的 commit state 都会因果改变下一次 targeted query：Qwen 是强且控制特异的边；Gemma 是已复现的 direct edge，但 local matched-control specificity 有 † 限制。至此，targeted retrieval→carrier→commit→next targeted retrieval 的 trace 内循环已经接上。</div></section>

<section id="answer"><p class="eyebrow">06 · Terminal readout</p><h2>Answer query 自然依赖 trace source；terminal state 在 fixed-suffix 中能控制 count margin</h2>
<h3>6.1 Answer 到底在读 trace，还是回 prompt 重数</h3>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>区分两种 answer-time 路径：（1）从已写好的 trace states 读 count；（2）在 answer query 再对 prompt records 做一次 broad retrieval。</div><div><span class="experiment-label">干预如何实现</span>“Blank”不是删除 token，也不是缩短序列：目标 token 的 embedding 与每一层 post-block residual state 都置零，绝对位置、序列长度和 query token 保持不变。左 panel 的 matched control 置零等数量的 ordinary prompt tokens。</div><div><span class="experiment-label">样本与加权</span>左侧 next-city localization 只报告两模型完全相同的 10 confirmation seeds、每 seed 一条注册 transition；<code>(seed,N,from,to,condition,repeat)</code> 多重集一致。右侧 answer source 为每模型同一 10 seeds、100 prompts per condition；输入审计确认两模型各 500 rows 的 <code>(seed,N,condition)</code> 多重集一致。所有汇总先在 seed 内平均，再让 seeds 等权。</div><div><span class="experiment-label">两套 answer banks</span>Discovery seeds 1234–1253 分别按 answer query 对 trace-item source 和 prompt-record source 的 attention 排名，冻结各自 Top-32；confirmation 不重新选 heads。图 5a 的 exact generation 不依赖选择哪套 bank 来“读数”，两套 instrumentation 得到相同生成结果。</div><div><span class="experiment-label">主终点</span><span class="formula">Exact = 1[greedy parsed answer = gold N]<br>Trace necessity = Exact<sub>clean</sub> − Exact<sub>trace blank</sub></span></div><div><span class="experiment-label">简单例子</span>如果保留 trace 而擦除 prompt records 后仍答对 6，但保留 prompt 而擦除 trace 后答错，则此时 answer 主要使用 trace 已写入的状态；这不排除 prompt 在其他样本中提供并行补充。</div></div>
<div class="figure-primer"><div><strong>左图/上图</strong>比较 clean、prompt-record blank 和 full-trace blank 的 greedy exact accuracy。</div><div><strong>Head-bank 图</strong>展示 answer query 在 prompt-record 与 trace-context 两类 source 上的 attention composition。</div><div><strong>不能过度解释</strong>Source blank 证明内容必要性，不证明某个 Top-32 bank 是唯一 readout。</div></div>
<figure><h3 class="figure-title">图 5a · Token-source ablation 对局部检索与最终 count 的影响</h3>{token_source_ablation_svg(token_evidence)}<figcaption>左 panel 纵轴是 teacher-forced next-city full-span exact rate：给定同一可见后续 token 序列，目标 city 的完整 token span 是否逐 token 预测正确；它不是 free-generation retrieval rate。柱是 trace-source treatment，黑色短横线是置零等数量 ordinary prompt tokens 的 matched control。横轴从左到右逐步区分远期历史、最近 transition 与全 trace 的作用；两模型均为完全相同的 10 confirmation transitions。右 panel 纵轴是 greedy exact-count accuracy，横轴依次比较自然输入、只擦原 prompt records、只擦完整 trace、以及擦完整 prompt+trace；两模型各 10 seeds/100 prompts per condition。右 panel 中 Qwen clean 0.97→trace blank 0.01，Gemma 0.70→0.12；prompt-record blank 的损伤远小于 trace blank。</figcaption></figure>
{token_condition_table}
<p class="term-note"><strong>横轴为何不是一条简单的“擦得越来越多”剂量轴。</strong> Early half、cumulative 与 recent 是三个有语义的 source partitions：前两者定位远期累计历史，recent 定位最后一次局部更新；full trace 才是它们的联合必要性。右侧 prompt-record blank 与 trace blank 则是两个互补信息源，不代表相同 token 数。因此应比较每个 condition 回答的机制问题，而不是只按 blank token 数解释柱高。</p>
<figure><h3 class="figure-title">图 5b · Answer-query head banks 的 source composition</h3><div class="figure-scroll">{answer_source_rerouting_svg(token_evidence)}</div><figcaption>每行是一个冻结 answer bank 在一种 token condition 下的 source composition。横向堆叠条在 prompt-record 与 trace-context 两个互斥 source groups 内归一；右侧 Σ 是两组 raw bank-summed mass 之和。颜色说明 attention 来源，不是 accuracy。</figcaption></figure>
<p><strong>结果分析。</strong> 两模型都可在 answer query 读 prompt，但当 trace 已存在时，最终 exact answer 对 trace content 的依赖更强。这与“count 随 trace stream 传递”一致，也允许 broad prompt retrieval 作为并行补充路径。</p>
<div class="section-conclusion"><strong>Experiment 6.1 结论。</strong> Trace 是 final answer 的主要自然信息源；prompt-broad retrieval 并未被排除，但“在 answer 时才从 prompt 重数”不足以解释 source-blank 对比。</div>

<h3>6.2 最后一条 trace 的 grammar state 能否推动正确答案</h3>
<p class="lead">6.1 证明“整段 trace content 对自然答案很重要”，但还没有定位最后是谁把 count 交给 answer query。6.2 因此构造一个受损 trace，再只恢复最后一个 item 中特定语义位置的 hidden states，检验 terminal grammar-state→answer-score 的局部因果边。</p>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>把 6.1 的 whole-trace necessity 缩小成一个可定位、可做 matched control 的 terminal bridge：最后一个 item 的哪类 state 能让正确 count candidate 的分数恢复？</div><div><span class="experiment-label">什么是一个样本</span>一条 outcome-blind 冻结的完整 trace/seed 是一个独立样本。本版先取两模型自然 grammar trace 的 exact intersection：19 discovery seeds、9 个不重叠 confirmation seeds；timing 为 discovery 9 city→rank / 10 rank→city、confirmation 4 / 5。12 个 arms、多个 tokens 和多层 clamp 都是 seed 内配对条件，不增加独立样本量。</div><div><span class="experiment-label">损坏基线</span>对每个 parsed trace item，用 prompt 中不属于 needle records 的 background tokens 作等长替换；不重分词、不 padding、不移动 answer query。这样保留语法长度与绝对位置，但移除原 trace item 内容，得到 uninformative arm。</div><div><span class="experiment-label">五种位点</span>在最后一个 item 内注册 full item、marker core、retrieved city、grammar-timed tail、boundary commit 五种位置。每种都从 clean run 捕获 same-position post-block states，再写入 uninformative run 的相同 positions。Discovery 只用于描述性选择 restoration 最大的非-full geometry；Qwen 选到 <code>{terminal_selected_geometry['Qwen3-8B']}</code>，Gemma 选到 <code>{terminal_selected_geometry['Gemma4-E4B']}</code>，随后在共同的 9-seed confirmation 上复验各自冻结 geometry。若两模型 geometry 不同，只能称功能对应的模型内 replication。</div><div><span class="experiment-label">层与时序</span>Qwen 从 L19、Gemma 从 L16 开始，在每个后续 decoder block 的同一 terminal positions 重施 clean state（cumulative clamp）。Answer query hidden state从未直接 patch；它只能通过正常 attention 读取这些 terminal positions。三个正式对比 arms（uninformative、semantic restore、matched-random restore）的 visible token prefix 完全相同；clean arm只作自然参考，不进入这两个差分。</div><div><span class="experiment-label">两个正式差分</span><span class="formula">Restoration = Margin<sub>semantic restore</sub> − Margin<sub>uninformative</sub><br>Specificity = Margin<sub>semantic restore</sub> − Margin<sub>matched ordinary state</sub></span>第一个问“补回语义 state 能否救回分数”；第二个问“相同 token 预算、相近深度的一般 clean state 是否也能做到”。</div><div><span class="experiment-label">计分</span>对候选答案字符串 1,…,10 做 teacher-forced sequence scoring，记 S(j) 为候选 <em>j</em> 的完整 sequence log score：<span class="formula">Margin(N)=S(N)−max<sub>j≠N</sub>S(j), &nbsp; j∈{{1,…,10}}</span>Margin&gt;0 才表示 gold N 在十个候选中排名第一；这里的单位是 log-score margin，不是“恢复了几个 counts”或 accuracy 百分点。</div><div><span class="experiment-label">具体数字例子</span>Gold N=6。若 uninformative 的 margin=−1.2，semantic restore 后=+0.8，则 restoration=+2.0；若 matched ordinary state 后仍为−0.9，则 specificity=+1.7。这个结果说明“正确 terminal semantic state”在固定 pre-answer context 中有功能，不等于单独一个 token 就能让自由生成必然输出 6。</div></div>
{terminal_sample_table}
<p class="term-note"><strong>实际规模。</strong> 每条 trace 有 clean + uninformative + 5 semantic restores + 5 matched-random restores = 12 arms。因而每模型 discovery 是 19×12=228 condition rows，confirmation 是 9×12=108 rows；两模型合计 672 rows。统计仍先在 seed 内求差，再让 seeds 等权。</p>
{terminal_geometry_table}
{terminal_arm_table}
<div class="figure-primer"><div><strong>第一行</strong>正确 terminal state 相对 uninformative trace，让 gold-number margin 上升多少。</div><div><strong>第二行</strong>正确 terminal state 相对 equal-token ordinary-state patch，多提供多少恢复。</div><div><strong>坐标</strong>横轴单位是 log-probability margin，不是正确率；绿=Qwen，紫=Gemma。</div></div>
<figure><h3 class="figure-title">图 5c · Terminal grammar-state 对 correct-count margin 的受控恢复</h3>{grouped_bars_svg('Fixed-suffix terminal-state confirmation effects', terminal_groups)}<figcaption>两行均为两模型完全相同的 9 confirmation seeds 的 seed-level paired mean effect。Qwen（<code>{terminal_selected_geometry['Qwen3-8B']}</code>）restoration / ordinary-state specificity 为 {float(terminal_effects['Qwen3-8B']['restoration']['mean_effect']):+.3f} / {float(terminal_effects['Qwen3-8B']['matched_random_specificity']['mean_effect']):+.3f}；Gemma（<code>{terminal_selected_geometry['Gemma4-E4B']}</code>）为 {float(terminal_effects['Gemma4-E4B']['restoration']['mean_effect']):+.3f} / {float(terminal_effects['Gemma4-E4B']['matched_random_specificity']['mean_effect']):+.3f}。正值表示正确 grammar state 比受损基线或普通 state 更有利于 gold count；跨模型条长不用于强弱排序。</figcaption></figure>
<p><strong>受控结果。</strong> Terminal-state restoration 为 Qwen {ci(terminal_effects['Qwen3-8B']['restoration'])}、Gemma {ci(terminal_effects['Gemma4-E4B']['restoration'])}；相对 ordinary-state control 的 specificity 分别为 {ci(terminal_effects['Qwen3-8B']['matched_random_specificity'])} 与 {ci(terminal_effects['Gemma4-E4B']['matched_random_specificity'])}。是否达到 interval-level support 只按各自 CI 是否完全高于 0 判断，不因另一模型的结果升级。</p>
<div class="scope-compare"><div><h4>本实验已经确认：controlled local effect</h4><p>三个正式差分 arms 使用相同 pre-answer visible tokens、相同 answer query 与相同 candidate scorer；唯一关键差别是 terminal semantic hidden state 是否写回。因此 paired margin 变化可归因于当前 state intervention。Clean arm只提供自然参照。</p><p><strong>对应标签：</strong>controlled only。</p></div><div><h4>本实验没有确认：free-running sufficiency</h4><p>更强命题是：只补这一处 state 后，让模型自由继续运行，它是否必然改变最终 greedy 数字。自由运行可同时读取其他 trace states、回 prompt、或在生成后续 token 时自我修正，所以局部 log-score 效应不自动推出全局输出改变。</p><p>Qwen 的 free-running extension 在 distribution、expected count、margin 和 exact answer 上没有稳定恢复，见 Appendix C。</p></div></div>
<div class="plain-language"><strong>为什么总览只写 controlled local bridge？</strong> “Confirmed”若不加限定，读者很容易理解成“一个 terminal token 单独足以决定最终自然答案”。本实验实际检验的是更窄的命题：<strong>在位置与后续输入都受控的 counterfactual context 中，恢复 discovery-frozen terminal semantic state 是否提高 gold-count candidate margin，并优于等预算 ordinary-state patch。</strong> 它没有覆盖自由生成的全部旁路与自我修正，因此即使两项对比都通过，也不升级成全局 sufficiency。</div>
<div class="section-conclusion"><strong>Experiment 6.2 结论。</strong>严格结论按上面的 restoration 与 specificity 数值/区间分别读取；它检验的是固定 answer prefix 下的 controlled local bridge。即使两个模型都为正，也不能据此声称该 terminal state 在 free-running 中单独决定最终答案，或两模型使用同一 token-level carrier。</div></section>

<section id="walkthrough"><p class="eyebrow">07 · Non-thinking-style case study</p><h2>一个 outcome-blind seed 从 item 1 走到 item 10</h2>
<div class="experiment-frame"><div><span class="experiment-label">实验目的</span>复制 Non-thinking 报告里最直观的“一条轨迹走到底”展示，检查 Native 的单个 item/carrier state 在丧失其余上下文后是否足以指定答案 <em>k</em>。</div><div><span class="experiment-label">设定</span>用 identity hash outcome-blind 冻结一个 N=10 confirmation case。Prompt records 与完整 trace context 都被等长 ordinary text 擦除；随后只恢复第 <em>k</em> 个 full item、grammar carrier，或等 token ordinary state。Answer query 始终不 patch。</div><div><span class="experiment-label">计算方法</span><span class="formula">E[count | condition,k] = Σ<sub>n=1</sub><sup>10</sup> n · P(answer=n | condition,k)</span>横轴为被恢复的 occurrence <em>k</em>，理想充分状态应产生 y=<em>k</em> 对角线。</div><div><span class="experiment-label">简单例子</span>当只恢复第 4 个 item 时，若该 state 是独立 counter，则 expected count 应靠近 4；若仍靠近 scrub baseline，则它需要其他 trace dynamics 才能被读出。</div></div>
<figure><h3 class="figure-title">图 6 · Full-context scrub 后的 expected-count path</h3>{walkthrough_svg(walkthrough)}<figcaption>橙色虚线是理想路径 y=k。Qwen 三条恢复路径大多停在高 count 区；Gemma 大多停在低 count 区。两者都没有形成 1→10 的对角路径。</figcaption></figure>
{walkthrough_table}
<div class="walkthrough-callout"><div><strong>控制成功。</strong><p>Clean case 均输出 10；uninformative baseline 分别退化为 Qwen candidate 9、Gemma candidate 1，且 P(10) 分别仅 {walkthrough['Qwen3-8B']['baselines']['uninformative']['gold_count_probability']:.4f} / {walkthrough['Gemma4-E4B']['baselines']['uninformative']['gold_count_probability']:.4f}。</p></div><div><strong>Restoration 失败。</strong><p>单 item hidden state 不能在被擦除的下游 trace 动力学中独立决定答案。V1 只擦 parsed items，遗漏 trace tail 的答案泄露，作为 failed-control audit 保留，不进入结果。</p></div></div>
<h3>7.2 无显式 running index 的 20/10-seed old-HTML restoration</h3>
<p class="lead">单 seed null 可能只是个例；同时，自然 trace 常直接写出“第六条”“count=6”等数字标签，使任何 state→count 结果都可能只是复制显式文本。这个补充实验因此复刻旧 HTML 的 full-span restore，但在一个没有 occurrence 序号的受控 trace grammar 上做正式 discovery/confirmation。</p>
<div class="experiment-frame"><div><span class="experiment-label">样本与 trace</span>每模型固定 30 个 N=10 benchmark prompts：seeds 1234–1253 用于 discovery，1254–1263 用于 confirmation。现有 300 条自然 archive 中没有一条通过严格格式审计；继续重试直到“碰巧合规”会按生成文本选样，因此我们 outcome-blind 地 teacher-force 完全相同的 grammar：每个 gold record 只写一行 <code>- City: score 61</code>。</div><div><span class="experiment-label">“无数字”具体指什么</span>这里严格排除的是<strong>计数数字</strong>：没有 item 1/2/3、first/second、running subtotal，也没有 answer 前的总数。城市的 score 数字仍保留，因为它是 needle 的证据内容而非 running count label。报告因此使用“no explicit running index”，不使用容易误解的“trace 完全没有 digit”。</div><div><span class="experiment-label">为什么叫受控反事实</span>Reasoning body 与最终 <code>Total: N</code> channel 被标准化并重新分词；prompt、native-thinking container、seed split 与 gold records 保持固定。它能检验模型在处理这种 trace 时是否形成可用 state，但不能估计模型自然生成这种 grammar 的概率，也不能冒充自然 generation confirmation。</div><div><span class="experiment-label">Uninformative receiver</span>把 prompt 中全部 10 个 needle records，以及 trace 中全部 10 个 bullet-item spans，都替换成 prompt 里同长度、非 needle 的 ordinary token windows。替换不 padding、不移动位置，sequence length 和 attention mask 完全不变；answer query 不 patch。这样 receiver 看不到任何 city/score needle content。</div><div><span class="experiment-label">Patch 几何</span>对每个 k=2,…,9，从 clean donor 捕获第 k 个完整 bullet span 的所有 token hidden states，再写到 receiver 的完全相同 positions；从 source layer 到最后一层逐层重施（cumulative clamp）。这与旧 HTML 的“clean needle-span hidden states → no-needle receiver same span”相同，不是单 token patch。</div><div><span class="experiment-label">Discovery 与 confirmation</span>Qwen 在 L18/22/26/30，Gemma 在 L16/20/24/28/32/36 扫描。按 seed-equal patched exact、再 target margin、再较晚层的冻结规则选择 Qwen L{int(unnumbered_counter['Qwen3-8B']['selected_layer'])} / Gemma L{int(unnumbered_counter['Gemma4-E4B']['selected_layer'])}；随后只在 10 个新 seeds 上跑这一层。</div><div><span class="experiment-label">实际规模</span>每个 seed 有 clean、fully-uninformative 两个 baseline；discovery 再有 8×4=32（Qwen）或 8×6=48（Gemma）个 patch arms，分别是 34/50 rows per seed。Confirmation 每 seed 只有 2+8=10 rows。两模型合计 1,880 condition rows，但独立样本仍只是每模型 20 discovery / 10 confirmation seeds。</div><div><span class="experiment-label">Early-stop 读数</span>在不自由生成后续解释的条件下，对答案 1,…,10 做完整 sequence log scoring。对 target k 定义 <span class="formula">Margin(k)=S(k)−max<sub>j≠k</sub>S(j)</span>；patch−scrub 是 target-margin gain。Hard exact 则要求 k 在十个候选中真正排名第一。</div><div><span class="experiment-label">简单例子</span>Gold trace 有 10 条，但只恢复第 4 个 item state。如果该 state 本身携带“目前完成 4 条”的计数信号，候选 4 的 margin 应相对 fully-uninformative receiver 上升；强 old-HTML sufficiency 还要求最后 argmax 直接变成 4。</div></div>
{counter_result_table}
<figure><h3 class="figure-title">图 6b · 无 running-index trace 的 full-item state 是否推动 early-stop k</h3>{internal_counter_restoration_svg(unnumbered_occurrence)}<figcaption>横轴是被恢复的 occurrence k=2…9。Panel A 是 confirmation 中 candidate k 的 patch−scrub mean margin gain；Panel B 是 hard exact accuracy gain。每个点先在对应 k 的 10 seeds 内平均。零线表示 patch 没有帮助；Qwen/Gemma 各自使用冻结 L18/L16。连续 margin 在大多数 k 上为正，但 exact gain 很小，因此曲线显示 count-aligned signal，而不是旧 HTML 式强 diagonal recovery。</figcaption></figure>
<p><strong>Confirmation 结果。</strong>Qwen 的 target-margin gain={ci(counter_margin['Qwen3-8B'])}，exact gain={ci(counter_exact['Qwen3-8B'])}，hard exact 从 {float(unnumbered_counter['Qwen3-8B']['selected_layer_metrics']['baseline_exact_accuracy']):.3f} 到 {float(unnumbered_counter['Qwen3-8B']['selected_layer_metrics']['patched_exact_accuracy']):.3f}；8/8 个 k 的 mean margin 都为正。Gemma margin gain={ci(counter_margin['Gemma4-E4B'])}，exact gain={ci(counter_exact['Gemma4-E4B'])}，hard exact 从 {float(unnumbered_counter['Gemma4-E4B']['selected_layer_metrics']['baseline_exact_accuracy']):.3f} 到 {float(unnumbered_counter['Gemma4-E4B']['selected_layer_metrics']['patched_exact_accuracy']):.3f}；7/8 个 k 为正。两模型都未达到预注册 old-HTML magnitude gate（patched exact≥0.50、gain≥0.25、mean margin&gt;0、至少 6/8 k 为正）。</p>
<div class="scope-compare"><div><h4>可以得到的结论</h4><p>在没有显式 running-index label 的受控 trace 中，第 k 个 full-item hidden state 含有跨 seed 可复现、与 k 对齐的可转移信号；把它写入无 needle receiver，会提高候选 k 的分数。这个结果排除了“只是在复制 item-number token”这一最直接混淆。</p></div><div><h4>仍不能得到的结论</h4><p>单个 item state 并不足以重建完整 counter：hard argmax 很少直接变成 k，而且该 trace 是 teacher-forced counterfactual，不是自然生成样本。更合理的机制解释是 count state 分布在 recurrent trace dynamics 中，单 span 携带其中一部分，但后续 propagation/readout 仍是必要条件。</p></div></div>
<div class="section-conclusion"><strong>Experiment 7 结论。</strong> 原单-seed natural-trace scrub 是 descriptive null；新的 20/10-seed、format-conditioned no-running-index panel 在两模型中给出 count-aligned margin gain，但强 old-HTML early-stop sufficiency 失败。它只说明受控 grammar 下存在可转移的 count-aligned signal，不构成 Gemma 的自然 no-index internal-counter 证据；本文的自然 no-index causal claim 仍仅来自 Qwen。</div></section>

<section id="comparison"><p class="eyebrow">08 · Mechanism synthesis</p><h2>最终机制图景</h2>
<div class="mechanism"><div class="stage"><span class="stage-no">01</span><h3>Retrieve</h3><p>Targeted heads 读取下一条 prompt city。Qwen 使用宽 Top-128；Gemma 使用窄 Top-6。</p><span class="evidence">causal necessity</span></div><div class="stage"><span class="stage-no">02</span><h3>Write & commit</h3><p>检索改变 grammar-specific carrier，carrier 将进度提交到 residual stream。</p><span class="evidence">deform + restore</span></div><div class="stage"><span class="stage-no">03</span><h3>Loop & read</h3><p>commit 改变下一次 targeted query；终端 state 再被 answer-time readout 使用。</p><span class="evidence">recurrent + conditional terminal</span></div></div>
{table(('Model','Strongest connected path','Remaining confound'),((
    'Qwen','Top-128 retrieval → carrier → commit → next query','terminal readout 只在固定 suffix 局部成立；free-running recovery 与 single-seed walk 不成立'),(
    'Gemma','Top-6 retrieval → carrier → commit → next query','orthogonal placebo 不匹配完整 state-delta 范数；terminal relay 为部分中介；single-seed walk 不成立')))}
<p><strong>整体分析。</strong> 两模型共享同一计算抽象，但实现宽度不同：Qwen 使用宽 retrieval bank 和大效应 full commit；Gemma 使用窄 retrieval bank，commit→query 效应较小且对 local control 更敏感。这不要求两模型共享同一组 heads 或同一线性 counter。</p>
<div class="section-conclusion"><strong>与 Non-thinking 的核心差异。</strong> Non-thinking 更接近 answer-time broad aggregation + late write；Native-thinking 的主链先在 trace 内部做 targeted retrieval 与 recurrent state update，到 terminal 才由 answer readout 使用。两种模式都可在 answer query 表征 count，但 count 的形成和传递路径不同。</div></section>

<section id="appendix"><p class="eyebrow">Appendix · Negative and secondary evidence</p><h2>不改变主链、但必须保留的结果</h2>
<details class="appendix-block" open><summary>Appendix A · Gemma next-query 边：历史 directional 标签如何被 exact-common confirmation 取代</summary><p>旧标签来自模型各自 cohort 与更宽 donor-distance panel，不能直接与 Qwen 比较。本版改用两模型完全相同的 semantic pair multiset，只报告预注册相邻 primary |d|=1。Full−self 与 full−orthogonal 分别回答“完整 state 是否有直接作用”和“作用是否优于 frozen count-subspace 的等投影范数扰动”；二者在 Gemma confirmation 中都取得区间正效应。Orthogonal control 仍不与完整 donor−receiver delta 等范数，因此不能把 positive contrast 解释成已经隔离出唯一 count component。</p><div class="section-conclusion"><strong>Appendix A 结论。</strong>历史 directional/confirmed† 标签已由 exact-common confirmed edge 取代；保留的限制属于 control 几何与排他性，而不是该局部边未复现。</div></details>

<details class="appendix-block" open><summary>Appendix B · Gemma query-mediation geometry ladder：full state 有效，但狭窄 Top-6 output mediation 未闭合</summary><p><strong>实验目的。</strong> 检验 commit state 的作用是否必须经由冻结 Top-6 targeted heads 的 pre-O outputs 传到 donor-successor city log-odds。按 outcome-blind 顺序检查 endpoint、suffix4、suffix8；每个 geometry 只跑 20 discovery seeds，只有 geometry_pass 才允许 confirmation。</p><p><strong>四个 estimands。</strong> A 是 full donor state 相对 self patch 的 intact 效应；B 是 selected-mask 对该 full-state 效应的削弱；C 在 selected-mask 条件下只恢复 selected-head pre-O outputs，看能否把 city log-odds救回；D 比较 selected mask 与同层同 K random mask。完整中介 gate 要求 A/B/C 的 95% CI 下界都大于 0；D 是 specificity diagnostic。</p>
<figure><h3 class="figure-title">图 B1 · Gemma endpoint→suffix4→suffix8 query-mediation ladder</h3>{effect_small_multiples_svg('Gemma query-mediation discovery ladder', mediation_panels)}<figcaption>每个 panel 横轴都是 donor−receiver query city log-odds effect，但四个 panel 为了显示数量级而使用独立对称尺度；点=20 discovery seeds 的 mean，线=95% CI，绿=该单项 gate 支持，橙=未满足。三行依次为 endpoint、suffix4、suffix8。Full-state effect 在三种 geometry 都大；selected-mask interaction 在较宽 suffix 上为正；关键的 selected-head pre-O restoration 仍触零，selected-vs-random specificity 也不稳定。</figcaption></figure>
<p><strong>结果与替代解释。</strong> suffix8 的 full-state effect 为 {ci(effect(gemma_query_mediation['suffix8']['estimands'], 'full_state_effect_intact'))}，selected-mask interaction 为 {ci(effect(gemma_query_mediation['suffix8']['estimands'], 'full_selected_mask_interaction'))}，但 pre-O restoration 只有 {ci(effect(gemma_query_mediation['suffix8']['estimands'], 'full_head_output_restore'))}；random mask interaction 也可为正。这与“Top-6 参与 routing”相容，却不足以证明所有可用 full-state 信息都串行穿过这 6 枚 heads。可能原因包括 head-output patch 的几何/时序不完整、selected bank 与旁路协同，或 layer-matched random 中也包含一般 query-support heads。</p><div class="section-conclusion"><strong>Appendix B 结论。</strong> 三个 geometry 均未达到 geometry_pass，confirmation 未开启。因此该 null 只否定当前狭窄 pre-O restoration 中介操作，不否定正文已经确认的 full commit→next-query 直接边或 Top-6 retrieval necessity。</div></details>

<details class="appendix-block" open><summary>Appendix C · Qwen free-running targeted-counter null：局部 margin 与自然答案充分性不同</summary><p><strong>实验目的。</strong> 将 fixed-suffix 的 terminal restoration 推广到更自由的 answer generation：要求 clean carrier restoration 同时把 count-candidate distribution、expected count、correct-count margin 与 greedy exact answer向 clean reference 拉回。20 discovery/10 confirmation 与 bank/position controls 均冻结。</p>
<figure><h3 class="figure-title">图 C1 · Qwen free-running clean-carrier restoration 的四个 answer endpoints</h3>{effect_small_multiples_svg('Qwen free-running targeted-counter confirmation null', q_free_panels)}<figcaption>四个 panel 分别显示 candidate-distribution TV recovery、expected-count recovery、correct-count margin recovery 与 greedy exact-count recovery。每个 panel 都有自己的对称横轴，因为量纲不同；点=10 confirmation seeds mean，线=95% CI。前三个 distribution/expectation quantities 接近数值零，margin 为 −0.075 [−0.500, +0.287]，greedy exact recovery 为 0。独立尺度只用于看每个 endpoint 是否离开 0，不能跨 panel 比条长。</figcaption></figure>
<p><strong>为何不是矛盾。</strong> Fixed-suffix patch 保留了后续 token trajectory，只问某个 terminal grammar state 能否局部提高 gold-number margin；free-running 实验允许后续状态、多个 trace sources、prompt retrieval 与自我校正共同决定答案。某个 ordinary-position control 可产生 margin 变化，也不等于 clean carrier 把完整 distribution 恢复到了 clean。</p><div class="section-conclusion"><strong>Appendix C 结论。</strong> Qwen 的 local terminal bridge 保留在正文，但自然 free-running count recovery 不成立；因此报告使用 controlled only，而不把 terminal grammar state 写成 answer 的独立充分统计量。</div></details>

<details class="appendix-block" open><summary>Appendix D · 失败 control 审计：ordinary-state clamp 与 single-seed scrub</summary>
<h3>D.1 为什么 5.2 的 ordinary-state clamp 不进入主结果</h3><p>这个 arm 原想控制“只要在同样的 positions 写入同样数量的 hidden vectors，commit 就会自然恢复”。但结果显示，将语义不相容的非-item states 主动写入 carrier positions 会产生很大的额外 damage；它不是一个接近“什么都不做”的 inert placebo。所以 clean carrier 相对该 arm 的巨大优势主要表明“普通 state 会主动破坏”，不能校准 carrier restoration 的因果大小。本版因此删去其数值、图和正文 claim，只保留这条 protocol audit。</p><div class="section-conclusion"><strong>D.1 结论。</strong>5.2 的主证据只是 selected-mask damaged baseline 与 selected-mask + clean-carrier restoration 的配对差；ordinary-state arm 不提供有效 specificity 证据。</div>
<h3>D.2 为什么第一版 single-seed scrub 不能解释，第二版如何修正</h3>
<p><strong>这个 appendix 在审计什么。</strong> 正文图 6 想检验：当 prompt 和其余 trace 都不再提供 count clue 时，只恢复第 <em>k</em> 个 item/carrier state，是否足以让答案接近 <em>k</em>。要解释这个实验，首先必须确认“什么都不恢复”的 baseline 真的失去 count 信息。</p>
<div class="scope-compare"><div><h4>V1：只擦 parser 找到的 item spans</h4><p>Prompt record spans 被替换，parser 明确认出的 numbered/bulleted items 也被替换；但 parser item 之后、answer query 之前的 trace tail 没有全部覆盖。</p><p><strong>问题：</strong>tail 可能含“therefore the total is 10”一类可见 clue，也可能已有能被 answer query 读取的 final-count hidden state。于是 baseline 仍可能知道答案。</p></div><div><h4>V2：擦完整信息源</h4><p>Prompt records 与整个 trace source——包括 items、连接文本和 terminal tail——都用等长 ordinary tokens 替换；answer query 不 patch。</p><p><strong>先验检查：</strong>只有 uninformative baseline 的答案表现确实明显退化后，单 item/carrier restoration 才开始解释。</p></div></div>
<div class="experiment-frame"><div><span class="experiment-label">具体例子</span>Gold N=10，我们想测试“只恢复第 4 个 item 是否让答案走向 4”。如果 V1 遗留的 tail 仍写着“所以一共有 10 条”，模型输出 10 既可能来自 tail，也可能来自被恢复 item；即便恢复 item 4 后仍输出 10，也不能判定 item 4 没有 count 信息。V2 删除这条旁路后，baseline 先退化，再恢复 item 4，结果才可解释。</div><div><span class="experiment-label">等长替换的作用</span>替换后的 token budget 与位置拓扑保持一致，避免把“信息被删掉”与“序列突然变短、answer query 移位”混在一起。</div><div><span class="experiment-label">V2 仍然是什么</span>它仍只是 outcome-blind 单 seed case study，不是 20/10 seed 群体 confirmation；其作用是给 sufficiency 一个直观而严格的 sanity check。</div></div>
<figure><h3 class="figure-title">图 D1 · V1 与 V2 的 scrub coverage</h3>{failed_control_svg()}<figcaption>从左到右表示 answer query 可能读取的信息源。上排 V1 只覆盖 prompt records 与 parser-observed items，橙框 trace tail 仍可泄露 final count，因此 V1 的 restoration 正负结果都不可归因。下排 V2 用等长 ordinary tokens 覆盖 prompt records 和完整 trace context，并要求 uninformative baseline 先退化。图只表示 control topology，不表示 effect size。</figcaption></figure>
<div class="plain-language"><strong>最简单的判断规则：</strong>如果“什么都不恢复”的 baseline 还答得很好，就不能用该实验判断单个 state 是否充分。V1 没满足这条规则；V2 满足后才进入正文。</div>
<div class="section-conclusion"><strong>Appendix D 结论。</strong>两个 failed controls 都不进入主 claim：5.2 的 non-inert state clamp 不用于 specificity；single-seed V1 暴露 trace-tail leakage，正文图 6 只使用 V2。V2 的 null 可以写成“在这两个冻结 case 中，单 item/carrier 不能在完整 source scrub 后独立指定 count”，不能外推成“模型中不存在 count state”。</div></details>

<details class="appendix-block" open><summary id="appendix-e">Appendix E · 其他 grammar 的 attention maps（8 张 SVG 已内嵌在本 HTML）</summary>
<p>正文图 2 使用冻结主线 <code>adjacent_rank_after_city</code>。这里补两种不同表面结构，检查 target-following attention 是否只在正文那一种写法中出现。展开本 appendix 后，E1–E8 会逐张显示；每张图是独立 SVG，已直接写进 HTML，不依赖本地图片路径或 Git 中的实验目录。</p>
<div class="appendix-e-index"><div><strong>E1–E4 · same-unit rank-before-city</strong>序号在 city 前；Qwen/Gemma 各有一张 layer×head atlas 和一张 ordinal×ranked-head map。</div><div><strong>E5–E8 · structural-invariant bullet</strong>没有显式数字 rank；Qwen/Gemma 同样各有两种 map，用来检查 heads 是否只追踪数字 marker。</div></div>
<p class="appendix-e-proof"><strong>读图规则：</strong>layer×head 图回答“哪些层、哪些 heads 对正确 target 较亮”；ordinal×head 图回答“同一批 ranked heads 是否随 target ordinal 1→10 移动”。每张图有自己的色标，颜色深浅不能跨模型或 grammar 比 raw magnitude。Ordinal 图保留横向滚动以保证 head 标签可读。</p>
<div class="attention-atlas-stack appendix-e-gallery"><figure class="appendix-e-figure" id="figure-e1"><h3 class="figure-title">图 E1 · Qwen same-unit rank-before-city · layer×head</h3><div class="attention-atlas-frame">{appendix_e_svgs['qwen_same_head']}</div><figcaption>横轴=Qwen head index H0–H31，纵轴=layer L0–L35；每格颜色是该 grammar 在 exact-P0 query 对正确 next-record span 的 discovery seed-equal raw attention mass。白边格是高排名 heads。它定位 heads，但不显示 target ordinal。</figcaption></figure>
<figure class="appendix-e-figure" id="figure-e2"><h3 class="figure-title">图 E2 · Gemma same-unit rank-before-city · layer×head</h3><div class="attention-atlas-frame">{appendix_e_svgs['gemma_same_head']}</div><figcaption>横轴=Gemma head index H0–H7，纵轴=layer L0–L41；其余定义与 E1 相同。Gemma 使用独立色标，不能用色深与 Qwen 比绝对 attention mass。</figcaption></figure>
<figure class="appendix-e-figure" id="figure-e3"><h3 class="figure-title">图 E3 · Qwen same-unit rank-before-city · target ordinal×ranked head</h3><div class="attention-atlas-frame">{appendix_e_svgs['qwen_same_ordinal']}</div><figcaption>横轴=按 discovery score 排序的 Qwen heads，纵轴=正确 next record 的 ordinal；颜色=20 discovery seeds 的 seed-equal raw target mass。纵向多个 ordinal 上反复出现亮列，表示同一 head 可跨 occurrence 重复参与 routing。</figcaption></figure>
<figure class="appendix-e-figure" id="figure-e4"><h3 class="figure-title">图 E4 · Gemma same-unit rank-before-city · target ordinal×ranked head</h3><div class="attention-atlas-frame">{appendix_e_svgs['gemma_same_ordinal']}</div><figcaption>轴与 E3 相同，但列是 Gemma 的 ranked heads。只看同一图内亮列是否跨 ordinal 延续；不要把图宽或色深与 Qwen 当作机制容量比较。</figcaption></figure>
<figure class="appendix-e-figure" id="figure-e5"><h3 class="figure-title">图 E5 · Qwen structural-invariant bullet · layer×head</h3><div class="attention-atlas-frame">{appendix_e_svgs['qwen_bullet_head']}</div><figcaption>Bullet grammar 没有显式数字 rank。横轴=head、纵轴=layer、颜色=正确 target mass。仍有局部亮 heads，说明 localization 不完全依赖“第 k 条”的数字 token。</figcaption></figure>
<figure class="appendix-e-figure" id="figure-e6"><h3 class="figure-title">图 E6 · Gemma structural-invariant bullet · layer×head</h3><div class="attention-atlas-frame">{appendix_e_svgs['gemma_bullet_head']}</div><figcaption>轴定义与 E5 相同，使用 Gemma 自己的色标。该图是 grammar-robustness visualization，不改变 frozen Top-6 的正式因果 bank。</figcaption></figure>
<figure class="appendix-e-figure" id="figure-e7"><h3 class="figure-title">图 E7 · Qwen structural-invariant bullet · target ordinal×ranked head</h3><div class="attention-atlas-frame">{appendix_e_svgs['qwen_bullet_ordinal']}</div><figcaption>横轴=Qwen discovery-ranked heads，纵轴=target ordinal，颜色=seed-equal raw target mass。它检查没有数字 marker 时，heads 是否仍随正确 record ordinal 路由。</figcaption></figure>
<figure class="appendix-e-figure" id="figure-e8"><h3 class="figure-title">图 E8 · Gemma structural-invariant bullet · target ordinal×ranked head</h3><div class="attention-atlas-frame">{appendix_e_svgs['gemma_bullet_ordinal']}</div><figcaption>轴定义与 E7 相同；该图只支持 descriptive target-following pattern。正式因果必要性仍来自正文 frozen Top-6 selected-vs-random ablation。</figcaption></figure></div>
<div class="section-conclusion"><strong>Appendix E 结论。</strong> 八张对应图显示 target-following attention 不只存在于正文的 rank-after-city grammar，也可在 rank-before-city 与无显式数字的 bullet grammar 中定位。它们扩展的是可视化稳健性，不把 descriptive attention 升格为新的因果 confirmation。</div></details>
<div class="section-conclusion"><strong>Appendix 结论。</strong> 这些 null 缩小了主张：我们确认一条可干预 recurrent pathway，但不确认单头排他中介、单 state 全局充分，或所有 grammar 与距离上完全同质的 circuit。</div></section>

<section id="audit"><p class="eyebrow">09 · Boundaries and reproducibility</p><h2>边界、复现与底层文件</h2>
<ul><li>本报告证明一条 pathway，不证明唯一性、排他性或所有 grammar 共用完全相同的 heads。</li><li>CI 与 p-value 保留用于审计；正文的“强/弱”判断同时考虑 effect size、控制组和跨 phase 复现。</li><li>单 seed walkthrough 不进入 discovery/confirmation gate；V2 是在 V1 暴露 trace-tail 泄露后修正的 exploratory control。</li><li>Qwen 与 Gemma 的状态几何、bank 宽度和最后一条边不同，不强行合并成完全同构 circuit。</li></ul>
<details class="paper-appendix"><summary>底层报告与外部证据包</summary><div class="source-list"><a href="NiaH_Native-Thinking_P0_Targeted_Retrieval_Atlas.html">P0 targeted-retrieval atlas</a><br><a href="NiaH_Native-Thinking_Parser_and_Token_Sites.html">Parser / token sites</a><br><a href="NiaH_Geometry_Comparison.html">Representation geometry</a><br><span>逐 seed、逐 arm、claim-gate 与运行审计文件保存在外部实验归档，不随 Git 仓库分发。报告中的聚合值与输入哈希已冻结；复算时通过构建器参数挂载对应 evidence bundle。</span></div></details>
<p class="audit">Generated UTC: {esc(generated)}<br>Schema: realistic_niah_v5_native_thinking_restructured_v14</p></section>

</main></article><script>
{point_cloud_script(geometry_3d)}
document.querySelectorAll('[data-attention-selector]').forEach(function(selector) {{
  var container = selector.closest('.attention-switcher').querySelector('[data-attention-container]');
  function showSelected() {{
    container.querySelectorAll('[data-attention-example]').forEach(function(panel) {{
      panel.style.display = panel.dataset.attentionExample === selector.value ? 'block' : 'none';
    }});
  }}
  selector.addEventListener('change', showSelected);
  showSelected();
}});
</script></body></html>"""

    # Recompose the report around the same inferential spine as the
    # Non-thinking report. The original sections remain useful evidence
    # containers, but their previous order mixed primary claims, extensions,
    # and negative controls.
    sections = extract_top_level_sections(html_text)

    baseline_section = f"""<section id="baseline"><p class="eyebrow">01 · Task and behavioral baseline</p>
<h2>1. 任务与行为基线：解释对象是 first-pass trace 中逐项推进，而不只是最后的数字</h2>
<div class="experiment-frame">
  <div><span class="experiment-label">实验目的</span>固定本文要解释的行为单位：模型既要从约 10k-token passage 中依次找出 N 条 needle records，也要在 reasoning trace 中决定“下一条读谁”，最后输出 total N。</div>
  <div><span class="experiment-label">任务设定</span>主因果 cohort 只使用 Qwen3-8B、N=10、自然生成的 first-pass no-index enumeration。<em>No-index</em> 指被分析的逐项枚举行不出现 <code>Count=k</code>、<code>Item k</code>、<code>Excerpt k</code>、ordinal 或 running subtotal；recap 与最终答案不进入 patch context。</div>
  <div><span class="experiment-label">记号与单位</span><em>N</em> 是真实 needle 总数；<em>k</em> 是 donor 已完成的 occurrence；<em>j</em> 是 receiver occurrence。Donor successor 是 <em>k</em>+1，receiver successor 是 <em>j</em>+1。一个独立样本是一条 seed-level trace；同一 seed 内的多个 k、方向、层与条件只是配对读数。</div>
  <div><span class="experiment-label">简单例子</span>若 N=10、receiver 当前完成 j=5，它自然应检索第 6 条；将 donor k=6 的状态写入 receiver 的同一绝对目标位置后，因果问题是下一次检索是否跳过第 6 条、改读 donor successor 第 7 条。Donor/receiver patch token 的字面文本不要求相同，<code>surface_token_matched</code> 逐 cell 记录。</div>
</div>
<div class="completion-note"><strong>计划内运行状态。</strong> Qwen 自然 no-index 主实验已经冻结 20 discovery + 10 disjoint confirmation seeds；服务器当前无遗留实验进程。Gemma 原 prompt 未形成足够的自然 no-index cohort，因此没有用改 prompt 的样本替换这一缺口。</div>
<div class="reading-contract">
  <div class="contract-row"><strong>Discovery</strong><span>{', '.join(map(str, qwen_noindex_discovery_seeds))}</span></div>
  <div class="contract-row"><strong>Confirmation</strong><span>{', '.join(map(str, qwen_noindex_confirmation_seeds))}</span></div>
  <div class="contract-row"><strong>完整样本内容</strong><span>30 条 prompt、first-pass enumeration、token sites 与逐 seed patch 结果保存在配套的 <a href="NiaH_Native-thinking_Internal-counter_report.html">Internal-counter seed browser</a>；本报告只保留机制所需的聚合与代表性例子。</span></div>
</div>
<p><strong>结果。</strong>这 30 条 trace 通过严格 no-index grammar 审计，并按 20/10 冻结；confirmation 不参与层、scope 或 head-bank 选择。</p>
<p><strong>分析。</strong>这一筛选使“复制可见 running index”不再解释 Qwen 主结果；但 event 中仍含 city、score 与语法，因此后续必须用 endpoint、tail 与 item-span scope controls 区分完整事件内容和更窄的 progress signal。</p>
<div class="section-conclusion"><strong>Experiment 1 结论。</strong>本文的 internal-counter 问题被限定为：在没有显式位置编号的 Qwen N=10 first-pass enumeration 中，内部 state 是否记录当前进度并因果控制从哪里继续。它不是对所有 Native-thinking 模型或所有 prompt 的普适性估计。</div></section>"""
    representation_section = sections["representation"].replace(
        '<section id="representation"><p class="eyebrow">03 · Representation</p><h2>Trace commit 与 answer query 都含有可读的 count geometry</h2>',
        '<section id="representation"><p class="eyebrow">02 · Measurement framework</p><h2>2. 通用测量框架：Representation 负责定位，Causal test 负责判定</h2>',
        1,
    )
    representation_section = (
        representation_section
        .replace("<h3>3.1 ", "<h3>2.1 ")
        .replace("<h3>3.2 ", "<h3>2.2 ")
        .replace("<h3>3.3 ", "<h3>2.3 ")
        .replace("图 1a", "图 2a")
        .replace("图 1b", "图 2b")
        .replace("图 1c", "图 2c")
        .replace("图 1d", "图 2c")
        .replace("Experiment 3 结论", "Experiment 2 结论")
    )
    representation_measurement = f"""
<div class="experiment-frame">
  <div><span class="experiment-label">因果量的目的</span>Probe 只问 state 中能否读出 k；state transplant 才问模型的后续计算是否使用 donor 所携带的信息。所有主因果量都先在同一 seed、同一 donor–receiver pair 内做 patch−self，再让 seeds 等权。</div>
  <div><span class="experiment-label">Routing score</span>令 S(c) 为候选 successor c 的 teacher-forced完整 sequence log score。<span class="formula">R = S(donor successor) − S(receiver successor)<br>Δroute = R<sub>patch</sub> − R<sub>self</sub></span>Δroute&gt;0 表示 patch 使模型相对更偏好 donor 的下一项；它不是“增加了几个 count”。</div>
  <div><span class="experiment-label">Attention score</span>对 discovery 冻结的 targeted head bank 与对应 record span 求和：<span class="formula">Q = log(A<sub>donor</sub>+ε) − log(A<sub>receiver</sub>+ε)<br>Δattention = Q<sub>patch</sub> − Q<sub>self</sub>, &nbsp; A=Σ<sub>h∈bank,t∈record</sub>α<sub>h</sub>(q,t)</span>bank-summed A 可以大于 1，且 Qwen Top-128 与 Gemma Top-6 的绝对量不能跨模型比较。</div>
  <div><span class="experiment-label">行为读数</span><em>Donor argmax</em> 表示 donor successor 在十个候选中得分最高；<em>first-city transfer</em> 表示自由 continuation 中首个匹配 gold city 的 ordinal 等于 donor successor。人工审计只接受 recap 之前的 first-pass 命中。</div>
  <div><span class="experiment-label">Representation common support</span>Running item-end geometry 只使用两模型 exact intersection：{int(sample_alignment_audit['evidence']['running_representation_common_support']['common_trajectory_cells'])} 条 trajectory cells、每模型 {int(sample_alignment_audit['evidence']['running_representation_common_support']['aligned_cells'])} 个 <code>(split, seed, N, k)</code> states；Qwen 与 Gemma 的 key-multiset hash 完全一致。层选择只看该 common support 的 discovery rows。</div>
  <div><span class="experiment-label">简单例子</span>Receiver 应读 N6，donor 应读 N7。若 self 时 S(N7)−S(N6)=−4，patch 后变为 +3，则 Δroute=+7；若首个生成 city 也从 N6 的 city 变成 N7 的 city，才得到行为层 transfer。</div>
</div>
<div class="subsection-conclusion"><strong>测量合同结论。</strong>下文始终把“可解码”“候选分数改变”“attention 改道”和“生成内容改变”分成四级证据；只有最后三者能贡献 causal routing claim。</div>
"""
    representation_section = representation_section.replace(
        "</div>\n\n<h3>2.1 可解码性如何随层变化</h3>",
        "</div>\n" + representation_measurement + "\n<h3>2.1 可解码性如何随层变化</h3>",
        1,
    )
    representation_section = representation_section.replace(
        "</figure>\n\n<h3>2.2 Count clouds 在低维中长什么样</h3>",
        "</figure>\n<div class=\"subsection-conclusion\"><strong>Experiment 2.1 结论。</strong>Running commit 与 answer query 都出现高于 chance 的 held-out count readout，但层峰和模型差异明显；这一步只定位候选 state，不建立自然使用。</div>\n\n<h3>2.2 Count clouds 在低维中长什么样</h3>",
        1,
    )
    representation_section = representation_section.replace(
        "\n<h3>2.3 冻结层的数值结果</h3>",
        "\n<div class=\"subsection-conclusion\"><strong>Experiment 2.2 结论。</strong>PCA3 提供 count clouds 的直观几何，但坐标轴是 discovery-fitted 方差方向而非预设 counter axes；正式数值仍来自 PCA16 held-out probe，因果解释留给后续 patching。</div>\n<h3>2.3 冻结层的数值结果</h3>",
        1,
    )
    retrieval_section = sections["retrieval"].replace(
        '<section id="retrieval"><p class="eyebrow">04 · Targeted retrieval</p><h2>下一条 city 由 model-specific targeted head banks 定向检索</h2>',
        '<section id="retrieval"><p class="eyebrow">04 · Logical chain B</p><h2>4. 逻辑链 B — Targeted retrieval：下一条 city 由 model-specific head banks 定向读取</h2>',
        1,
    )
    retrieval_section = (
        retrieval_section
        .replace("图 2a", "图 4a")
        .replace("图 2b", "图 4b")
        .replace("图 2c", "图 4c")
        .replace("图 2d", "图 4d")
        .replace("图 2e", "图 4e")
        .replace("图 2f", "图 4f")
    )
    retrieval_section = retrieval_section.replace(
        "\n<h3>4.2 单头、整 bank 与跨-seed attention pattern</h3>",
        "\n<div class=\"subsection-conclusion\"><strong>Experiment 4.1 结论。</strong>两模型都存在随目标 ordinal 移动的 attention 对角带；这是 head-bank localization，不是必要性证据。</div>\n<h3>4.2 单头、整 bank 与跨-seed attention pattern</h3>",
        1,
    ).replace(
        "\n<h3>4.3 Head bank 在 layer×head 空间中的位置</h3>",
        "\n<div class=\"subsection-conclusion\"><strong>Experiment 4.2 结论。</strong>单头、整 bank 与跨-seed聚合都能看到 target-following pattern，但单头存在冗余，不能据图挑一个唯一 counting head。</div>\n<h3>4.3 Head bank 在 layer×head 空间中的位置</h3>",
        1,
    ).replace(
        "\n<h3>4.4 关闭 bank 后，下一条 city 是否失败</h3>",
        "\n<div class=\"subsection-conclusion\"><strong>Experiment 4.3 结论。</strong>冻结 bank 跨多个层分布；atlas 只解释 bank 的空间组成，真正的 selection specificity 由下一节 selected-vs-random ablation 判定。</div>\n<h3>4.4 关闭 bank 后，下一条 city 是否失败</h3>",
        1,
    )
    answer_section = sections["answer"].replace(
        '<section id="answer"><p class="eyebrow">06 · Terminal readout</p><h2>Answer query 自然依赖 trace source；terminal state 在 fixed-suffix 中能控制 count margin</h2>',
        '<section id="answer"><p class="eyebrow">06 · Logical chain D</p><h2>6. 逻辑链 D — Terminal readout：trace state 被条件化地读成最终 count</h2>',
        1,
    )
    answer_section = (
        answer_section
        .replace("图 5a", "图 6a")
        .replace("图 5b", "图 6b")
        .replace("图 5c", "图 6c")
        .replace("<span class=\"experiment-label\">具体数字例子</span>", "<span class=\"experiment-label\">简单例子</span>")
        .replace("见 Appendix C", "见 Appendix O")
    )
    answer_query_execution_section = f"""
<h3>6.3 Answer-query residual 本身是否已经是可执行的 count state</h3>
<p class="lead">6.2 只 patch trace 末端，并让未干预的 answer query 自行读取；因此它检验 terminal trace→answer 的局部 bridge。这里补上与 Non-thinking <em>Consolidate</em> 完全对称的另一半：直接把 donor 的完整 answer-query residual 写入 receiver，观察实际 greedy 数字是否采用 donor count。</p>
<div class="experiment-frame">
  <div><span class="experiment-label">实验目的</span>区分“answer query 上可以线性解码 count”与“该 state 真能被后续层执行”。若完整 donor state 在某层写入后使 receiver 直接输出 donor 的数字，则该层之后的网络能把这份 state 当作可执行 readout 使用。</div>
  <div><span class="experiment-label">精确 token site</span><code>answer_query_v3</code> 定义为<strong>第一个数字答案 token 之前，trace 中最后一个字面 token</strong>。Donor 与 receiver 使用同一个注册语义/字面 site，但不强求绝对 token index 相等；因此这是 surface-site matched，不是 absolute-position matched。</div>
  <div><span class="experiment-label">Cohort 与配对</span>只使用 seeds 1254–1263 的 confirmation traces，并同时要求两模型都满足 strict one-to-one、baseline final answer exact。先对每个 seed 取两模型可用 count 的交集，再选靠近低端与高端的两条 common-grid edges，并各自双向运行；因此<strong>两模型完全共享同一 40 个有向 <code>(seed, receiver count, donor count)</code> keys</strong>，每 seed 恰好 4 pairs。Pair selection 不读取 patch outcome。</div>
  <div><span class="experiment-label">层网格</span>在查看本轮 patch outcome 前，按架构深度近似均匀固定 Qwen L0/5/10/15/20/25/30/35 与 Gemma L0/6/12/18/23/29/35/41。没有按 confirmation 峰值重新选层。由于这些 confirmation seeds 早已用于报告其他分析，本实验准确标签是 <em>registered existing-split extension</em>，不是 pristine 新 cohort。</div>
  <div><span class="experiment-label">两个 arms</span><strong>Self patch</strong> 把 receiver 自己的 residual 写回同一位置，必须逐 cell 重新生成 receiver gold；<strong>full donor patch</strong> 把 donor 的完整单-token residual 写到 receiver query position。每次只 patch 一个 post-block layer，之后自由 greedy 生成最多 16 tokens。</div>
  <div><span class="experiment-label">主终点与统计</span><span class="formula">Adopt<sub>i,l</sub> = 1[ŷ<sup>full donor</sup><sub>i,l</sub> = N<sup>donor</sup><sub>i</sub>]<br>Layer effect<sub>l</sub> = (1 / S) Σ<sub>s=1</sub><sup>S</sup> mean<sub>i∈seed s</sub> Adopt<sub>i,l</sub></span>输出必须是实际 deterministic greedy continuation 解析出的 1,…,10；不可解析或越界输出保留在分母中，按 adoption failure 计。95% CI 对 seed cluster bootstrap。</div>
  <div><span class="experiment-label">简单例子</span>Receiver 的 clean answer 是 3，same-seed donor 的 clean answer 是 4。若在 L25 把 donor answer-query residual 写到 receiver 后，实际 continuation 从“3”变成“4”，该 cell 记 donor adoption；只让某个 probe 更接近 4 不算。</div>
</div>
{answer_query_layer_table}
<div class="figure-primer"><div><strong>横轴</strong>zero-based post-block layer；两个模型各用预先固定的八层网格。</div><div><strong>纵轴</strong>pair 内 donor-count adoption，再先 seed 内平均、最后 10 seeds 等权。</div><div><strong>误差线</strong>seed-cluster bootstrap 95% CI；不是把每模型 40 pairs 当独立样本。</div></div>
<figure><h3 class="figure-title">图 6d · Answer-query full-state patch 的逐层 donor-count adoption</h3>{answer_query_layer_plot}<figcaption>每个点都来自同一冻结 confirmation protocol。Self patch 在每个 pair×layer cell 都必须重新生成 receiver gold，否则整个分析失败；曲线只画 full-donor adoption。Qwen 的描述性 onset 为 {('none' if answer_query_onsets['Qwen3-8B'] is None else f'L{int(answer_query_onsets["Qwen3-8B"])}')}，最终采样层 L35 为 {probability_ci(answer_query_final_rows['Qwen3-8B'], 'full_donor_adoption')}；Gemma onset 为 {('none' if answer_query_onsets['Gemma4-E4B'] is None else f'L{int(answer_query_onsets["Gemma4-E4B"])}')}，L41 为 {probability_ci(answer_query_final_rows['Gemma4-E4B'], 'full_donor_adoption')}。Onset 只按“mean≥0.5 且 CI 下界&gt;0”作描述，不用于选择被测层。</figcaption></figure>
<p><strong>与旧 selected-layer assay 的关系。</strong>旧的 discovery-selected 单层实验已在 56 个 directed pairs 上得到完整 donor-state adoption：Qwen {float(historical_answer_controls['Qwen3-8B']['full_donor_adoption']['effect']):.3f}、Gemma {float(historical_answer_controls['Gemma4-E4B']['full_donor_adoption']['effect']):.3f}。但其 rank-3 projected-state adoption 只有 Qwen {float(historical_answer_controls['Qwen3-8B']['projected_donor_adoption']['effect']):.3f}、Gemma {float(historical_answer_controls['Gemma4-E4B']['projected_donor_adoption']['effect']):.3f}。旧实验与本轮 pair registry 不同，因此不把数值拼成同一条曲线；它只提供一个边界：<strong>完整 answer state 可执行，不等于已经隔离出共享的低维线性 counter direction。</strong></p>
<div class="scope-compare"><div><h4>本实验已经确认：answer state → number execution</h4><p>在 clean-correct、surface-site matched 的 receiver/donor pairs 上，单层完整 answer residual replacement 可以直接改变实际 greedy 数字。它比 probe 强，因为 outcome 是模型真正生成的 count。</p></div><div><h4>本实验仍未隔离：纯 count component</h4><p>被移植的是完整 donor residual，可能同时携带 count、trace length、position 与其他上下文。因而本节证明 state 可执行，不证明一个低维、content-free scalar register；absolute-position matched control 仍列为进一步隔离实验。</p></div></div>
<div class="section-conclusion"><strong>Experiment 6.3 结论。</strong>两模型都形成了可由后续层直接执行的 answer-query count state；这一结果闭合的是 answer-state→数字输出，而不是 memoryless <code>+1</code> recurrence。它不单独回答 earlier trace state 是否经该 answer state 传递；同一 trial 的有序中介由下一节检验。</div>

<h3>6.4 Earlier terminal state 是否经 post-terminal suffix 与 answer query 传递</h3>
<p class="lead">Non-thinking 报告的整链证据依赖同一 forward 中的有序中介，而不能只把两个独立 patch 实验首尾相接。这里在<strong>完全相同的 86 个 model-common confirmation pairs</strong>上做 2×3 factorial relay assay，直接检验 terminal trace state→post-terminal suffix / answer-query state→count margin 的同-trial 部分中介。</p>
<div class="experiment-frame">
  <div><span class="experiment-label">完全对齐的分析单位</span>Qwen 与 Gemma 都只保留同一 multiset 的 <code>(seed, gold_count, donor_offset)</code>：10 个 confirmation seeds、86 个 directed pairs/model、19 个 count×offset cells。Common-support intersection 在看 outcome 前由 trial keys 决定；两个模型不合并估计效应。</div>
  <div><span class="experiment-label">2×3 factorial arms</span>Source condition 是 <strong>self patch</strong> 或 <strong>full donor terminal-state patch</strong>；relay condition 是 <strong>natural relay</strong>、<strong>clean post-terminal-suffix reset</strong>、或 <strong>clean answer-query-only reset</strong>。下游 reset 保留相同的 upstream donor patch，只把指定 relay state 恢复为 receiver clean state。</div>
  <div><span class="experiment-label">三个量</span><span class="formula">Damage<sub>natural</sub> = M(self,natural) − M(donor,natural)<br>Damage<sub>reset</sub> = M(self,reset) − M(donor,reset)<br>Specific mediation = Damage<sub>natural</sub> − Damage<sub>reset</sub></span><em>M</em> 是 gold count 相对最强错误 count 的完整候选 sequence log-score margin。正的 specific mediation 表示：恢复下游 relay state 后，upstream donor patch 造成的答案损伤变小。</div>
  <div><span class="experiment-label">简单例子</span>若 terminal donor patch 让正确答案 margin 相对 self 下降 4.0，而把 post-terminal suffix 恢复为 clean 后只下降 1.5，则 suffix-specific mediation=2.5、explained fraction=62.5%。剩余 1.5 说明还有旁路或未重置的 downstream state，不能声称完全中介。</div>
  <div><span class="experiment-label">推断边界</span>Post-terminal suffix 是 primary relay；answer-query-only reset 是 secondary localization。二者的中介量高度嵌套，<strong>不能相加</strong>。实验固定候选答案 suffix 并计分，因此证明有序的 causal pathway；自由生成可执行性另由 6.3 的 greedy donor adoption 支持。</div>
</div>
<div class="figure-primer"><div><strong>Panel A</strong>横轴为正确 count sequence-margin 的损伤；natural damage 是 upstream state patch 的总效应，clean-suffix reset 后的 residual 是未被该 relay 解释的部分。</div><div><strong>Panel B</strong>横轴为 patch×reset interaction，即 specific mediation；正值表示该 downstream state 承担了 upstream patch 的一部分作用。</div><div><strong>点与线</strong>点是先 seed 内平均、再让 10 seeds 等权的 mean；线是 seed-cluster bootstrap 95% CI。每 panel 独立缩放，不能用 Qwen/Gemma 绝对长度比较机制强弱。</div></div>
<figure><h3 class="figure-title">图 6e · Model-common terminal relay 的同一-trial 部分中介</h3>{effect_small_multiples_svg('Experiment 6.4 · terminal-state relay mediation on exact common support', terminal_relay_panels)}<figcaption>Panel A 同时画 natural patch damage 与 clean post-terminal-suffix reset 后的 residual；Panel B 画 suffix-specific 与 answer-query-only specific mediation。横轴单位都是完整候选 count 的 log-score margin 或其 interaction；竖直零线表示无效应。两模型都满足 natural damage 与 specific mediation 的 CI 下界大于 0，但 suffix reset 后仍保留正 residual，因此证据支持<strong>有序、部分、非唯一</strong>的 relay，而不是完全中介。</figcaption></figure>
{terminal_relay_table}
<p><strong>结果分析。</strong>Qwen 的 suffix-specific mediation 解释 natural patch effect 的 {100 * float(terminal_relay_summary['models']['Qwen3-8B']['effects']['post_terminal_suffix_explained_fraction']['estimate']):.1f}%（bootstrap ratio CI {100 * float(terminal_relay_summary['models']['Qwen3-8B']['effects']['post_terminal_suffix_explained_fraction']['ci_low']):.1f}%–{100 * float(terminal_relay_summary['models']['Qwen3-8B']['effects']['post_terminal_suffix_explained_fraction']['ci_high']):.1f}%）；Gemma 为 {100 * float(terminal_relay_summary['models']['Gemma4-E4B']['effects']['post_terminal_suffix_explained_fraction']['estimate']):.1f}%（{100 * float(terminal_relay_summary['models']['Gemma4-E4B']['effects']['post_terminal_suffix_explained_fraction']['ci_low']):.1f}%–{100 * float(terminal_relay_summary['models']['Gemma4-E4B']['effects']['post_terminal_suffix_explained_fraction']['ci_high']):.1f}%）。两者的 residual damage CI 也都在 0 以上，所以这些 relay states 承担 terminal perturbation 传向 answer margin 的一条通路，但不是唯一通路。Gemma 在这组 exact-common pairs 上的 suffix-reset 与 answer-query-only interaction 数值完全相同；它们不能当作两次独立证据，也不能在 Gemma 内区分 wider suffix 与 query-only mediation，只能把共同下游位点解释为已定位的部分 relay。Qwen 两个 estimand 不同，因此其 wider suffix 与最后 query 位置可作嵌套、但不可相加的定位。</p>
<div class="section-conclusion"><strong>Experiment 6.4 结论。</strong>在完全样本对齐的 confirmation common support 上，terminal trace-state patch 对 correct-count margin 的影响会被 clean post-terminal suffix / answer-query reset 显著削弱；结合 6.3 的 answer-state→greedy number execution，可以把 <code>trace state → answer-query state → count output</code> 严谨地写成一条<strong>有序的部分因果通路</strong>。剩余效应明确排除“单一、完全、排他中介”的说法。</div>
"""
    require(answer_section.endswith("</section>"), "Malformed terminal section")
    answer_section = (
        answer_section[: -len("</section>")]
        + answer_query_execution_section
        + "</section>"
    )

    original_write = sections["write"]
    old_53_marker = "<h3>5.3 Commit state 是否决定下一次 targeted query 读向哪里</h3>"
    require(old_53_marker in original_write, "Cannot isolate historical 5.3 section")
    write_prefix, historical_commit = original_write.split(old_53_marker, 1)
    historical_commit = old_53_marker + historical_commit.rsplit("</section>", 1)[0]
    historical_commit = historical_commit.replace(
        "若会，trace 中就形成 READ→WRITE→COMMIT→NEXT READ 的 recurrent loop。",
        "该受控比较只检验完整 commit state 是否影响下一次 query；它不等同于自然 no-index trace 中的 arithmetic recurrence。",
    ).replace(
        "至此，targeted retrieval→carrier→commit→next targeted retrieval 的 trace 内循环已经接上。",
        "该结果保留为历史的 indexed/controlled routing evidence；它不单独证明自然 no-index trace 中存在同样的 counter。",
    ).replace(
        "直接 recurrent edge 已确认",
        "受控 commit→query edge 已确认",
    ).replace(
        "支持强 recurrent routing",
        "支持强 controlled routing",
    ).replace(
        "shared recurrent computation、不同 state geometry",
        "shared state-dependent routing、不同 state geometry",
    )
    write_prefix = write_prefix.replace(
        '<section id="write"><p class="eyebrow">05 · Write and recurrent propagation</p><h2>把 trace 内部的三条边分开检验</h2>',
        '<section id="write"><p class="eyebrow">05 · Logical chain C</p><h2>5. 逻辑链 C — Counter update：从检索结果到下一项选择</h2>',
        1,
    ).replace(
        '<div><span>5.3 · COMMIT → NEXT READ</span><strong>把“已完成 k”的 commit 换成“已完成 k+1”，下一次 query 会不会改读再下一条 record？</strong></div>',
        '<div><span>5.3 · EVENT STATE → NEXT ITEM</span><strong>在自然 no-index trace 中移植 donor event state，后续 attention 与首个检索 city 会不会共同跟随 donor？</strong></div>',
        1,
    )
    shared_write_marker = "<h3>5.1–5.2 共用什么样本与实验底座</h3>"
    write_51_marker = "<h3>5.1 关闭 targeted bank 后，检索结果有没有写入 grammar carrier</h3>"
    write_52_marker = "<h3>5.2 在同一 query damage 下，恢复 clean carrier 能否救回 item-end commit</h3>"
    write_51_diagnostic_marker = "<h4>5.1b NCC：carrier 不只是“变了”，是否朝错误 count centroid 移动</h4>"
    require(
        all(
            marker in write_prefix
            for marker in (
                shared_write_marker,
                write_51_marker,
                write_52_marker,
                write_51_diagnostic_marker,
            )
        ),
        "Cannot split the write section into main and diagnostic evidence",
    )
    write_header, write_after_header = write_prefix.split(shared_write_marker, 1)
    write_shared_body, write_after_shared = write_after_header.split(write_51_marker, 1)
    write_51_full, write_52_main = write_after_shared.split(write_52_marker, 1)
    write_51_main, write_51_diagnostics = write_51_full.split(
        write_51_diagnostic_marker, 1
    )
    old_51_conclusion = '<div class="section-conclusion"><strong>Experiment 5.1 结论。'
    if old_51_conclusion in write_51_diagnostics:
        write_51_diagnostics = write_51_diagnostics.split(old_51_conclusion, 1)[0]
    write_header = write_header.replace(
        "这一节不是一次复杂 patch 得出整条链，而是依次问三个较小的问题。每个问题的 treatment、control 和读数都不同：",
        "这一节把同一个 counting loop 拆成三条可单独证伪的边。5.1–5.2 在两模型的受控 grammar family 中检验写入；5.3 只在 Qwen 的自然 no-index trace 中检验下一项 routing。三条边来自不同但相互衔接的干预，不能误读成单次 end-to-end mediation。",
        1,
    )
    write_shared = shared_write_marker + write_shared_body
    write_51_main = (
        write_51_marker
        + write_51_main.replace("图 3a", "图 5a")
        .replace('<span class="experiment-label">要检验的箭头</span>', '<span class="experiment-label">实验目的</span>', 1)
        .replace('<span class="experiment-label">具体例子 A</span>', '<span class="experiment-label">简单例子 A</span>', 1)
        .replace('<span class="experiment-label">具体例子 B</span>', '<span class="experiment-label">简单例子 B</span>', 1)
        + '<div class="section-conclusion"><strong>Experiment 5.1 结论。</strong>在完全样本对齐的固定 trace 上，关闭冻结 targeted bank 会使两模型 query 之后的 grammar carrier 离开 clean state，且 selected−random identity control 在两模型都为正。该结果确认 bank-related carrier damage，不宣称 bank 排他或 carrier 是唯一 count code。NCC 与最终答案 margin 的附加诊断移至 Appendix H。</div>'
    )
    write_52_main = (
        write_52_marker
        + write_52_main.replace("图 3b", "图 5b")
        .replace('<span class="experiment-label">要检验的箭头</span>', '<span class="experiment-label">实验目的</span>', 1)
        .replace('<span class="experiment-label">完整例子</span>', '<span class="experiment-label">简单例子</span>', 1)
    )
    write_51_diagnostics = (
        write_51_diagnostic_marker
        + write_51_diagnostics.replace("图 3a-2", "图 H1").replace("图 3a-3", "图 H2")
    )
    indexed_discovery_table = table(
        (
            "Model / frozen grammar",
            "Discovery sweep",
            "Automatic / confirmation layer",
            "Median paired Δ log-odds",
            "Positive cells",
            "Donor argmax",
            "Mean width / exact full item",
        ),
        (
            (
                f"{SHORT[model]} · {indexed_cohort_manifest['surface_template']}",
                f"k=6, two directions, {int(indexed_progress_active_discovery[model]['cell_count'])} cells",
                (
                    f"auto L{int(indexed_progress_selected[model]['selected_layer'])} / "
                    f"confirm L{indexed_confirmation_layers[model]}"
                ),
                f"{float(indexed_progress_active_discovery[model]['median_paired_logodds_shift']):+.2f}",
                f"{100*float(indexed_progress_active_discovery[model]['positive_shift_rate']):.1f}%",
                f"{100*float(indexed_progress_active_discovery[model]['patched_donor_argmax_rate']):.1f}%",
                (
                    f"{float(indexed_progress_active_discovery[model]['mean_patch_width']):.1f} / "
                    f"{100*float(indexed_progress_active_discovery[model]['equal_length_complete_item_rate']):.1f}%"
                ),
            )
            for model in MODELS
        ),
        class_name="compact-table",
    )
    indexed_confirmation_table = table(
        (
            "Model / frozen layer",
            "Median paired Δ log-odds",
            "Positive log-odds",
            "Mean attention Δ / positive",
            "Donor argmax",
            "First city: patch / receiver / gain",
            "Seeds with ≥1 transfer",
        ),
        (
            (
                f"{SHORT[model]} · L{indexed_confirmation_layers[model]}",
                f"{float(indexed_progress_summary[model]['median_paired_logodds_shift']):+.2f}",
                f"{100*float(indexed_progress_summary[model]['positive_logodds_shift_rate']):.1f}%",
                (
                    f"{float(indexed_progress_summary[model]['mean_paired_attention_shift']):+.2f} / "
                    f"{100*float(indexed_progress_summary[model]['positive_attention_shift_rate']):.1f}%"
                ),
                f"{int(round(60*float(indexed_progress_summary[model]['patched_donor_argmax_rate'])))}/60",
                (
                    f"{int(indexed_progress_summary[model]['patched_first_known_city_donor_adoption_count'])}/60 / "
                    f"{int(indexed_progress_summary[model]['receiver_first_known_city_donor_adoption_count'])}/60 / "
                    f"{100*float(indexed_progress_summary[model]['paired_first_known_city_donor_adoption_gain']):+.1f} pp"
                ),
                f"{int(indexed_progress_generation_audit['models'][model]['seed_with_any_incremental_adoption_count'])}/10",
            )
            for model in MODELS
        ),
        class_name="compact-table",
    )
    indexed_crossk_table = table(
        (
            "Model",
            "Donor k",
            "Direction",
            "Median paired Δ log-odds",
            "Donor argmax",
            "First city: patch / receiver",
            "Attention positive",
        ),
        (
            (
                SHORT[model],
                str(int(row["donor_occurrence_k"])),
                "forward skip" if row["direction"] == "forward_skip" else "backward rewind",
                f"{float(row['median_paired_logodds_shift']):+.2f}",
                f"{int(round(10*float(row['patched_donor_argmax_rate'])))}/10",
                (
                    f"{int(row['patched_first_known_city_donor_adoption_count'])}/10 / "
                    f"{int(row['receiver_first_known_city_donor_adoption_count'])}/10"
                ),
                f"{int(round(10*float(row['positive_attention_shift_rate'])))}/10",
            )
            for model in MODELS
            for row in indexed_progress_summary[model]["by_direction_k"]
        ),
        class_name="compact-table",
    )
    qwen_indexed_transfer = int(
        indexed_progress_summary["Qwen3-8B"][
            "patched_first_known_city_donor_adoption_count"
        ]
    )
    qwen_indexed_k6_rows = [
        row
        for row in indexed_progress_summary["Qwen3-8B"]["by_direction_k"]
        if int(row["donor_occurrence_k"]) == 6
    ]
    require(len(qwen_indexed_k6_rows) == 2, "Qwen indexed k=6 support changed")
    qwen_indexed_k6_transfer = sum(
        int(row["patched_first_known_city_donor_adoption_count"])
        for row in qwen_indexed_k6_rows
    )
    qwen_indexed_k6_receiver = sum(
        int(row["receiver_first_known_city_donor_adoption_count"])
        for row in qwen_indexed_k6_rows
    )
    indexed_qwen_comparison = (
        f"Qwen 的显式-index frozen-L{indexed_confirmation_layers['Qwen3-8B']} "
        f"k=6 first-city transfer 为 {qwen_indexed_k6_transfer}/20（receiver "
        f"baseline {qwen_indexed_k6_receiver}/20）。自然 no-index robustness 使用 "
        "L16 与不同 grammar/cohort，因此不把二者作同层增益比较。"
    )
    indexed_control_section = f"""
<h3>J.1 显式 index positive control：可见 progress label 会不会让同一 assay 更容易转移？</h3>
<p class="lead">这是对 5.3 的 assay calibration，不是扩大 internal-counter claim。Qwen 与 Gemma 使用<strong>完全相同的 N=10 内层枚举 surface</strong>与同一 20+10 seeds；item 中明确出现 <code>k</code>，所以任何成功效应都可能直接利用 visible position label。</p>
<div class="experiment-frame">
  <div><span class="experiment-label">实验目的</span>确认同一 donor→receiver transplant assay 在存在清楚 progress cue 时能够产生预期的 successor routing，并观察 Qwen/Gemma 的方向差异。</div>
  <div><span class="experiment-label">共享 grammar / cohort</span>两个模型的内层 trace 都是 <code>k. City - score</code>；shared keys 为 <code>(phase, seed, gold_count=10)</code>，20 discovery + 10 held-out confirmation，模型原生 chat/channel wrapper 可不同。</div>
  <div><span class="experiment-label">Intervention</span>与 no-index 主实验使用同一 endpoint-aligned maximal-common-span transplant 与 self-patch control。Patch 宽度和 tokenizer span 可以随模型/项目变化，但每个模型内 donor/receiver boundary、coverage 与 effective width 都逐 cell 审计。</div>
  <div><span class="experiment-label">计算方法</span>使用第 2 节定义的 paired Δroute、paired Δattention、10-way donor argmax 与 first-city transfer。Discovery layer sweep 只作 profile；active confirmation layer 不从 indexed outcome 选择。</div>
  <div><span class="experiment-label">Pre-confirmation amendment / freeze</span>在读取任何 indexed outcome 前，直接使用 exact cross-model common-support running representation 的 discovery-frozen层：Qwen L{indexed_confirmation_layers['Qwen3-8B']}、Gemma L{indexed_confirmation_layers['Gemma4-E4B']}。这个 amendment 只更正模型特异的外部 representation-frozen layer，并在 confirmation outcome 不可见时落盘；这样避免把 L0 最大 downstream recomputation depth 当成机制定位，也避免用 positive-control outcome 反选层。</div>
  <div><span class="experiment-label">简单例子</span>Receiver 的可见 label 是 5，donor 是 6；若 patch 后下一项从 <code>6. ...</code> 改为 <code>7. ...</code>，这说明 assay 能复制带显式位置线索的 progress state，但不能证明无 label 时也存在同一内部变量。</div>
</div>
<figure><h3 class="figure-title">图 J1 · 显式-index positive control 的 discovery-only layer profile</h3><div class="scope-layer-figure">{indexed_progress_layer_svg}</div><figcaption>横轴是 zero-based post-block layer；纵轴是 paired donor-successor Δroute。深色线是每个 seed 两方向 effect 的 median；绿/紫分别是 forward/backward cell median。两 panel 使用独立 y scale，不能据线高比较模型。空心圆是该 profile 自身的自动 peak-rule layer；橙色菱形是 outcome-blind external representation freeze（Qwen L{indexed_confirmation_layers['Qwen3-8B']}、Gemma L{indexed_confirmation_layers['Gemma4-E4B']}）。Confirmation 只运行后者。</figcaption></figure>
{indexed_discovery_table}
<p><strong>Held-out cross-k confirmation。</strong>冻结层后只在新 10 seeds 上测试 k={{4,6,8}}、双方向共 60 cells，并同时读取 transition likelihood、frozen targeted-head attention、10-way successor argmax 与自由 continuation 的第一个已知 city。</p>
{indexed_confirmation_table}
<p>{indexed_qwen_comparison} Gemma 没有同口径的自然 no-index cohort，因此这里只能报告 positive-control 结果，不能计算 Gemma 的 indexed-vs-no-index 增益。</p>
<p><strong>行为审计。</strong>Qwen 的 {int(indexed_progress_generation_audit['models']['Qwen3-8B']['patched_donor_adoption_count'])} 次、Gemma 的 {int(indexed_progress_generation_audit['models']['Gemma4-E4B']['patched_donor_adoption_count'])} 次 patched donor-first-city 命中都在 continuation 前 80 characters 内出现；没有一次依赖 reasoning close 后的 answer/recap。扣除同 cell receiver baseline 后，至少一次出现新增 transfer 的 seeds 为 Qwen {int(indexed_progress_generation_audit['models']['Qwen3-8B']['seed_with_any_incremental_adoption_count'])}/10、Gemma {int(indexed_progress_generation_audit['models']['Gemma4-E4B']['seed_with_any_incremental_adoption_count'])}/10。</p>
<p><strong>模型内结果，不作效应量强弱比较。</strong>Qwen 的 positive-log-odds / positive-attention 比例为 {int(round(60*float(indexed_progress_summary['Qwen3-8B']['positive_logodds_shift_rate'])))}/60 与 {int(round(60*float(indexed_progress_summary['Qwen3-8B']['positive_attention_shift_rate'])))}/60，first-city paired gain={100*float(indexed_progress_summary['Qwen3-8B']['paired_first_known_city_donor_adoption_gain']):+.1f} pp；Gemma 对应为 {int(round(60*float(indexed_progress_summary['Gemma4-E4B']['positive_logodds_shift_rate'])))}/60、{int(round(60*float(indexed_progress_summary['Gemma4-E4B']['positive_attention_shift_rate'])))}/60、{100*float(indexed_progress_summary['Gemma4-E4B']['paired_first_known_city_donor_adoption_gain']):+.1f} pp。Bank width、层与 tokenizer span 架构特定，因此这些数值只用于判断各模型中 assay 是否工作，不用于声称谁更强。</p>
<p class="main-note"><strong>解释边界。</strong>显式序号既可能进入被移植的 span，也会通过上下文写进其 hidden states；item span 还保留 city、score 与语法。因而成功只表明 assay 能在有强 progress cue 时转移 state-dependent routing，不能证明模型在没有 index 时维护同一种 counter，也不能把 Qwen 的自然 no-index 结论外推到 Gemma。</p>
<div class="section-conclusion"><strong>Appendix J 结论。</strong>共享 surface、共享 seed/count/k/direction/condition keys 的 explicit-index control 校准了 transplant assay；其结果只证明带可见 progress cue 时的 state-dependent routing 能力。本文的 natural no-index internal-counter/progress-controller claim 仍严格只来自 Qwen 5.3。</div>"""
    write_section = f"""{write_header}{write_shared}{write_51_main}{write_52_main}
<h3>5.3 自然 no-index counter state 是否改写下一项 retrieval</h3>
<p class="lead">这里不再把单个 endpoint 当作完整 counter。Qwen N=10 的 transplant 让 receiver patch target、绝对位置与后续 grammar 保持不变；donor alignment 通过调整非-record filler 完成，因此 donor/receiver 的字面 patch token 不必相同。只替换 frozen layer 上的 event state，然后同时读取 donor-successor 的相对 likelihood、targeted attention 与自由 continuation 中第一个已知 city。</p>
<div class="experiment-frame">
  <div><span class="experiment-label">实验目的</span>闭合 Native counting 通路中最关键的一条边：已经写入 trace 的 contextual event/progress state，是否真的决定下一轮应该检索哪条 record。</div>
  <div><span class="experiment-label">设定</span>Qwen 自然 no-index N=10 confirmation 的同 10 seeds；固定 L16，donor k=6，分别做 forward 5←6 与 backward 7←6，共 20 个 pair。Patch 同层、同绝对目标位置的 endpoint-aligned maximal-common span；self-patch 控制 hook。只有 donor/receiver item 等长时它才是双方的完整 item，不能把所有 cells 统称为 literal full-span patch。</div>
  <div><span class="experiment-label">计算方法</span>沿用第 2 节的 Δroute、Δattention、10-way donor argmax 与 first-city transfer。L16 没有重新扫 confirmation 层；它是对 L0 frozen scope confirmation 的预先指定中层 robustness comparator。</div>
  <div><span class="experiment-label">简单例子</span>在 forward pair 中，receiver 已完成 N5、自然将读 N6；donor state 表示已完成 N6。若 patch 后 attention 与首个生成 city 都改指 N7，说明被移植 state 携带了足以推进下一项检索的功能信息。</div>
</div>
<figure><h3 class="figure-title">图 5c · 同一 L16 state intervention 从内部 routing 传播到自由 continuation</h3>{natural_progress_bridge}<figcaption>横轴按计算顺序列出四个 readout：paired Δroute&gt;0、paired Δattention&gt;0、donor successor 在十个候选中 argmax、以及自由 continuation 的首个已知 city 跟随 donor。纵轴是 20 个配对 cells 中满足该判据的比例；柱顶给出命中数。四柱来自同一批 10 seeds、k=6 双方向 L16 item-span patches，因此展示的是从内部连续量到行为结果的证据梯度，而不是四组独立实验。</figcaption></figure>
<div class="evidence-ladder">
  <div><span>LIKELIHOOD</span><strong>L16: 20/20 正向</strong><p>median donor-successor Δ log-odds = {float(item_span_l16_summary['median_paired_logodds_shift']):+.2f}。</p></div>
  <div><span>ATTENTION</span><strong>L16: 20/20 正向</strong><p>mean targeted-attention shift = {float(item_span_l16_summary['mean_paired_attention_shift']):+.2f}。</p></div>
  <div><span>DECISION</span><strong>17/20 donor argmax</strong><p>候选下一项的 ranking 跟随 donor progress state。</p></div>
  <div><span>BEHAVIOR</span><strong>16/20 first-city transfer</strong><p>forward {l16_item_forward[0]}/{l16_item_forward[1]}，backward {l16_item_backward[0]}/{l16_item_backward[1]}；10/10 seeds 至少一次。</p></div>
</div>
<p class="main-note"><strong>证据等级。</strong> L16 是真正的中层 post-block state intervention，且所有 16 次命中都在 continuation 前 40 characters 内出现、0 次仅在 recap 中出现。但它使用与 L0 confirmation 相同的 10 seeds，并只复核 k=6，因此是 causal robustness comparator，而不是第二组独立 confirmation。</p>
<p>冻结的四-token event tail 给出更窄的 routing carrier：系统 scope assay 在 held-out 60 cells 中 median Δ log-odds={float(patch_scope_confirmation['event_tail_w4']['median_paired_logodds_shift']):+.2f}、attention 58/60 朝 donor 移动、首个 city 15/60 跟随 donor，并覆盖 8/10 seeds。它不含 city 名，但仍含 score numeral；所以可以称为 <em>counter-like routing information</em>，不能称为 content-free counter register。</p>
<div class="section-conclusion"><strong>Experiment 5.3 结论。</strong> 在 Qwen 的自然 no-index N=10 trace 中，中层分布式 counter state 对下一项 retrieval 具有行为层面的因果控制。结合 3.2 的 L0 多步 continuation，这条边支持“counter state 决定从哪里继续”；L16 comparator 本身仍只验证首步，不负责隔离纯算术更新。</div>
</section>"""

    mirror_table = table(
        ("共同逻辑阶段", "Native-thinking 对应对象", "本报告的主证据", "证据边界"),
        (
            (
                "State formation / localization",
                "trace item 的 endpoint、event tail 与 endpoint-aligned maximal-common span",
                "held-out probe + frozen scope transplant",
                "表征可读不等于因果使用；L0 full span 是 upper bound",
            ),
            (
                "Retrieval",
                "每一步 exact query 对下一条 prompt record 的 targeted read",
                "selected-vs-layer-matched-random head-bank ablation",
                "bank-level necessity，不是唯一单头 circuit",
            ),
            (
                "Write / progress control",
                "retrieved event → grammar carrier → commit；event state → next item",
                "carrier damage/rescue + Qwen L16 no-index transplant",
                "两类实验来自衔接的受控 cohort，不是一次完整 mediation",
            ),
            (
                "Terminal readout",
                "trace state 向 answer 传递；完整 answer state 被读成数字",
                "trace-source blank + terminal restore + answer-query layer sweep + 86-pair same-trial relay reset",
                "answer state 可执行且构成有序部分通路；仍非完全、唯一中介",
            ),
        ),
        class_name="mirror-table",
    )
    parser_grammar_table = table(
        ("Setting", "可见表面 / prompt 改动", "Parser contract", "在本文中的证据角色"),
        (
            (
                "Qwen natural no-index",
                "原始 prompt byte-identical；自然 first-pass 文本",
                "按 gold city+score 注册首次唯一 evidence mention；要求完整覆盖、无重复，并通过 strict no-index cue gate",
                "唯一进入自然 no-index 主 claim 的 cohort",
            ),
            (
                "Gemma prompt-conditioned no-visible-index",
                "passage 不变；只改 passage 后 task tail；assistant prefix 为 FOUND: ",
                "只接受 FOUND: <city> | score <score>；拒绝 ordinal、labeled index、running total、非 FOUND prose、重复或缺失 event；只允许一个 Total:",
                "simulatively confirmed†（controlled surrogate）；不是 natural confirmation",
            ),
            (
                "Explicit-index positive control",
                "Qwen: k. City - score；Gemma: * Record k: (City, score)",
                "恰好十个 spans；visible marker 必须为 1…10；gold city/score 一一对应且表面格式精确匹配",
                "校准 transplant assay；因 visible k 混淆，不能支持 no-index claim",
            ),
        ),
        class_name="parser-table",
    )
    parser_contract_section = f"""<div class="parser-contract" id="parser-design-contract">
<div class="parser-contract-head"><div><span class="parser-contract-kicker">Pre-experiment contract</span><h3>实验前置 · Parser 与因果设计合同</h3></div><p>以下四项是所有主实验之前冻结的解析与干预规则。它们决定一条 trace 能否进入 cohort、字符区间如何编译成 token sites、一个 donor→receiver cell 如何构造，以及什么才算行为转移。折叠内容默认收起；展开后可复核定义、公式和 false-positive gate。Parser 是测量合同，不是机制结果本身。</p></div>
<details class="parser-disclosure"><summary>A · Natural no-index cohort parser：first-pass、<code>t*</code> 与 global-clean</summary><div class="parser-disclosure-body">
<div class="parser-flow"><div><strong>1 · Raw archive</strong><br>原始 prompt 与自然 reasoning</div><span>→</span><div><strong>2 · Evidence registry</strong><br>首次、唯一、score-supported gold event</div><span>→</span><div><strong>3 · <code>t*</code> boundary</strong><br>第 N 个唯一 event 的字符末端</div><span>→</span><div><strong>4 · Strict gate</strong><br>coverage、uniqueness、no-index</div><span>→</span><div><strong>5 · Token context</strong><br>覆盖 <code>t*</code> 的最小完整 token prefix</div></div>
<div class="parser-grid">
  <div><strong>Evidence unit</strong>只有同时匹配 gold city 与对应 score 的 mention 才算一次 event；同一 city 后续再次出现不增加 progress，重复会进入审计。</div>
  <div><strong>First-pass boundary <code>t*</code></strong><code>t*</code> 是第 N 个首次唯一 evidence mention 的结束字符。它把最早完成 evidence enumeration 的前缀与之后 recap / rethink 分开。</div>
  <div><strong>Prefix eligibility</strong>在 <code>t*</code> 以前必须覆盖全部 N 个 gold records，且每条恰好出现一次；任何显式 progress cue 都使该 trace 不合格。</div>
  <div><strong>Explicit-cue families</strong>包括 <code>Count=k</code> / running progress、<code>Item k</code> / <code>Excerpt k</code>、ordinal、编号 evidence line、city 后括号 index、以及 gold-city 前导 index。</div>
  <div><strong>Primary frozen field</strong>主 cohort 使用 <code>strict_eligible_no_explicit_count_cue</code>，它等同于更严格的 <code>global_clean</code>：整段 reasoning 不得出现 per-record index；terminal aggregate total 仍允许。</div>
  <div><strong>Outcome blindness</strong>按固定 split 与 seed 升序选 20 discovery + 10 confirmation；筛选器不读取最终 answer、patch outcome、attention、generation 或 mechanism score。</div>
  <div><strong>Prompt integrity</strong>Qwen natural setting 的 system/user prompt 与 archive 逐 byte 比较；任何 prompt 改写都会被排除，而不是悄悄并入 natural cohort。</div>
  <div><strong>字符→token 编译</strong>机制 context 使用 archived prompt 加覆盖 <code>t*</code> 的最小 whole-token output prefix；跨过边界的末 token spill 单独记录，未来 recap 在因果上不可见。</div>
</div>
<div class="parser-warning"><strong>容易误读的地方。</strong><code>global_clean</code> 不是“完整输出中不能出现任何数字”，而是“不能出现逐项 count/index cue”。Needle 自带的 score numeral 和最终一次 aggregate total 属于任务内容，不被当作 running index。主结果中的 no-index 也只指被分析的 first-pass enumeration。</div>
</div></details>
<details class="parser-disclosure"><summary>B · 三套 trace grammar：natural、prompt-conditioned 与 indexed control</summary><div class="parser-disclosure-body">
<p>三套 parser 共享 gold-record 一一对应、span 不重叠和 outcome-blind selection，但它们回答的科学问题不同，不能互相替代。</p>
{parser_grammar_table}
<p><span class="parser-tag">PRIMARY</span>Qwen natural no-index　<span class="parser-tag">AUXILIARY</span>Gemma prompt-conditioned　<span class="parser-tag">CALIBRATION</span>explicit-index control</p>
<div class="parser-warning"><strong>Claim firewall。</strong><code>FOUND:</code> marker 本身不携带 count，因此 Gemma auxiliary 能说明在受控 no-visible-index grammar 下可诱发 successor routing；但 task tail 已改变。Explicit-index control 又直接暴露 k。两者共同只支持 <em>simulatively confirmed†</em>（controlled surrogate positive），都不把 Gemma 升格为 natural no-index confirmation。</div>
</div></details>
<details class="parser-disclosure"><summary>C · Span compiler 与 patch geometry：从字符 event 到同绝对位置 hidden states</summary><div class="parser-disclosure-body">
<p>Trace parser 先把每个 event 注册为 half-open token span <code>[start, end)</code>。随后只在注册 item 内选择 endpoint；不会跨入下一项之间的 lead-in、空白或 recap 文本。Donor 或 receiver prefix 会做等长位置对齐，使两个 commit sites 落在同一绝对 token index，且 prefix sequence length 相同。</p>
<div class="parser-grid">
  <div><strong>Endpoint w1</strong>只移植注册 item endpoint 的一个 post-block residual vector；这是最窄但也最弱的 scope。</div>
  <div><strong>Event tail w4</strong>以 endpoint 为右边界移植最后四个 tokens；固定宽度不得越过 donor 或 receiver 的 item boundary。</div>
  <div><strong>Item span</strong>令 donor/receiver endpoint 前可用宽度为 <code>a<sub>d</sub>, a<sub>r</sub></code>，有效宽度 <code>w=min(a<sub>d</sub>,a<sub>r</sub>)</code>；两边都取 endpoint-aligned suffix <code>[end−w,end)</code>。</div>
  <div><strong>Unequal tokenization</strong>等宽时移植完整 item；不等宽时移植较短 item 的全部 tokens 与较长 item 的等宽 suffix，不对 hidden states 插值或 resample。</div>
  <div><strong>Site matching</strong>候选点限于 item tail；优先相同 token，并审计相同 tail offset。固定 tail-offset robustness 允许 token 不同，但显式记录 <code>surface_token_matched</code>。</div>
  <div><strong>Coverage audit</strong>每个 cell 保存 donor/receiver coverage、有效宽度、endpoint alignment、完整 item 是否等长，以及 patch-span token ids/text。</div>
</div>
<span class="parser-code">receiver patch = [e<sub>r</sub>−w, e<sub>r</sub>)　←　donor state [e<sub>d</sub>−w, e<sub>d</sub>)<br>Lℓ = decoder block ℓ 之后的 post-block residual；L0 是第一层输出，不是 raw embedding。</span>
<div class="parser-warning"><strong>为什么这一步重要。</strong>“同一个 token”本身不保证同一个语义 site；parser 同时约束 item membership、tail offset、绝对位置和 prefix 长度。这样 patch 不会因为选到 recap 空白、跨 item span 或位置编码变化而制造假跳转。</div>
</div></details>
<details class="parser-disclosure"><summary>D · Causal cell 与 readout parser：什么叫“下一项跟随 donor”</summary><div class="parser-disclosure-body">
<p>一个 cell 由 receiver progress <em>j</em> 与 donor progress <em>k</em> 定义。Receiver 原本应继续到 <em>N</em><sub>j+1</sub>；patch 后的 donor hypothesis 是继续到 <em>N</em><sub>k+1</sub>。<em>j&lt;k</em> 是 forward skip，<em>j&gt;k</em> 是 backward rewind。例：receiver 已完成 N4、donor 表示完成 N6，则自然候选是 N5，donor 候选是 N7。</p>
<div class="parser-flow"><div><strong>1 · Likelihood</strong><br>paired donor-vs-receiver Δ log-odds</div><span>→</span><div><strong>2 · Attention</strong><br>target query → donor-successor record edge</div><span>→</span><div><strong>3 · Decision</strong><br>donor successor 是否为 10-way argmax</div><span>→</span><div><strong>4 · Generation</strong><br>首个已知 city 是否为 donor successor</div></div>
<span class="parser-code">Δroute = [S<sub>patch</sub>(k+1) − S<sub>patch</sub>(j+1)] − [S<sub>self</sub>(k+1) − S<sub>self</sub>(j+1)]</span>
<div class="parser-grid">
  <div><strong>Self-patch control</strong>把 receiver 自己的 state 写回同一位置，控制 hook、复制和写回操作；主差分不是 patch 对完全不干预的裸比较。</div>
  <div><strong>Exact attention edge</strong>只读取已冻结 targeted heads 上“下一次 query token → prompt 中 donor-successor needle span”的 attention mass；不汇总到任意相关文本。</div>
  <div><strong>Generation parser</strong><code>first_generated_known_city_ordinal</code> 在最早 <code>&lt;/think&gt;</code>、<code>&lt;|im_end|&gt;</code> 或 <code>&lt;end_of_turn&gt;</code> 前，用 word-boundary 搜索首个 gold city。</div>
  <div><strong>False-positive audit</strong>窄 bullet-line parser 只作审计，不是 primary endpoint；所有 donor-adoption completion 再人工核对，recap-only mention 不计成功。</div>
  <div><strong>Discovery freeze</strong>Qwen N=10 discovery 只用 k=6 双方向扫 36 层；scope/layer 只看 registered transition likelihood，attention 与 generation 对选择不可见。</div>
  <div><strong>Held-out test</strong>Confirmation 固定到 10 个新 seeds，测试 k∈{{4,6,8}} 与双方向；L16 k=6 是同一 confirmation seeds 的 post-hoc robustness，不伪装成第二次独立 confirmation。</div>
</div>
<div class="parser-warning"><strong>最小成功标准与 claim 层级。</strong>连续量改变说明 routing 被因果推动；candidate argmax 与首个生成 city 跟随 donor 才说明改变越过决策边界；无后续 patch 的多步 donor-aligned sequence 才说明 counter 重定向会持续。即使这些层级都成立，也仍不证明一个 content-free scalar 经固定算子实现 <code>c ← c + 1</code>。</div>
</div></details>
</div>"""
    summary_section = f"""<section id="summary"><p class="eyebrow">Conclusion first</p>
<h2>先说机制：Native-thinking 维护一个控制后续枚举的 internal counter</h2>
<div class="core-claim"><strong>本文主张（仅限 Qwen3-8B 的自然 no-index trace）。</strong> Qwen 维护一个分布式、与 event content 绑定的 internal counter：它记录当前枚举进度，并决定下一项从哪里继续。一次 counter-state transplant 不仅改变下一项，还会让后续枚举沿 donor 所指向的序列继续，无需再次干预。Gemma 尚无对应的自然 no-index 因果结果。</div>
<p class="lead">最强证据来自 Qwen3-8B 的自然 first-pass、无显式 running index、N=10 trace。Discovery 只用 k=6 双向 patch 扫 36 层并冻结 scope/layer；held-out confirmation 再测试 k∈{{4,6,8}} 的 forward skip 与 backward rewind。Item-span scope——即 endpoint-aligned maximal-common span，而非每个 cell 都是双方 literal full item——在 60/60 cells 中把 donor successor 推到候选第一，实际 continuation 的首个已知 city 在 43/60 cells 跟随 donor。对原始 completion 的多步重解析进一步显示：这 43 次首步转移全部继续到 donor 的下一 successor；在可观察四步的条件中为 27/27。固定到更深的 L16、只复核 k=6，仍有 16/20 首步行为转移。</p>
<div class="reading-contract">
  <div class="contract-row"><strong>Internal counter</strong><span>本文把“记录当前数到哪里，并因果决定从哪里继续”的内部状态称为 counter。它不要求某个神经元等于 k，也不预设一个纯标量算术寄存器；现有 counter state 同时包含 event content、语法与 progress。</span></div>
  <div class="contract-row"><strong>Event tail / item span</strong><span>Event tail 是 item 结束前冻结的四-token 窄窗口；item span 是 donor/receiver 的 endpoint-aligned maximal-common range：等长时覆盖双方完整 item，不等长时覆盖较短 item 的全部 tokens 与较长 item 的等宽 suffix。Tail 不含 city，但仍含 score numeral；span 通常含 city、score 与 syntax。</span></div>
  <div class="contract-row"><strong>Cross-model sample alignment</strong><span>凡正文把 Qwen 与 Gemma 并列展示，必须同时通过 input-panel 与 realized-output 两层 fail-closed audit：两模型使用完全相同 multiset 的 phase、seed、gold count、donor/receiver 或 transition keys，以及相同 condition/repeat multiplicity；不能只证明运行前计划相同。架构特定的层、head identity、bank width 与 tokenizer span 可以不同；绝对效应量不作模型间强弱比较。</span></div>
  <div class="contract-row"><strong>证据标签</strong><span><em>Natural confirmed</em> 表示原始 prompt 的自然 no-index cohort 在 discovery 冻结后于独立 confirmation 复现；<em>simulatively confirmed†</em> 是本文对 controlled surrogate positive 的简写：原始 prompt 下无法建立对应 cohort，但在 prompt-conditioned no-visible-index 与 explicit-index 两类受控 surrogate setting 中都观察到 state-guided successor transfer。它确认的是可诱发的机制能力，不是自然使用；<em>controlled only</em> 表示固定后续 visible suffix 时有局部因果效应，但尚未证明单 state 在自由运行中足以决定最终输出。</span></div>
</div>
<p class="main-note"><strong>跨模型校准已另做，但不并入 claim。</strong>显式-index clean grammar positive control 在冻结层的 patched first-city transfer 为 Qwen {qwen_indexed_transfer}/60、Gemma {int(indexed_progress_summary['Gemma4-E4B']['patched_first_known_city_donor_adoption_count'])}/60；相对各自 receiver baseline 的配对净增为 Qwen {100*float(indexed_progress_summary['Qwen3-8B']['paired_first_known_city_donor_adoption_gain']):+.1f} pp、Gemma {100*float(indexed_progress_summary['Gemma4-E4B']['paired_first_known_city_donor_adoption_gain']):+.1f} pp。Gemma 在 Appendix K 的 prompt-conditioned no-visible-index auxiliary 中另有 22/30 first-city transfer。两类受控设置共同支持图 S1 的 <em>simulatively confirmed†</em> 标记；但前者明示 k，后者修改了输出格式 prompt，因此都不能作为 Gemma 的自然 no-index internal-counter 证据。</p>
<h3>与 Non-thinking 报告如何对仗</h3>
<p>两份报告使用同一推理顺序，但不强迫两类模型共享同一实现。Non-thinking 的核心单位更接近 prompt-side aggregation；Native-thinking 的核心单位是 trace 内反复生成、被下一次 query 读取的 contextual event state。</p>
{mirror_table}
<div class="figure-primer"><div><strong>主文回答什么</strong>状态在哪里形成、能否控制 retrieval、以及最终如何被读取。</div><div><strong>模型范围</strong>自然 no-index causal result 目前只有 Qwen；Gemma 的旧结果含显式 progress grammar 或属于受控 carrier/readout，不能并入该 claim。</div><div><strong>不做什么外推</strong>不把 full-span copying、score-bearing tail、显式-index positive control 或单 endpoint 当成已隔离的算术寄存器。</div></div>
<figure><h3 class="figure-title">机制图 S1 · Native-thinking counting 通路与当前证据等级</h3><div class="chain-scroll">{chain_svg()}</div><figcaption>这是一张阶段图，没有数值坐标轴。横向箭头表示候选计算顺序：targeted retrieval 读取下一条 record，retrieved event 被写入 grammar carrier 与 commit/event state，该 state 再控制 next-item routing，最终 answer query 从 trace 读取 count。纵向两行分别是 Qwen 与 Gemma；每格文字是该阶段目前最高证据等级。Qwen 的 <em>natural confirmed</em> 来自原始 prompt 的自然 no-index confirmation；Gemma 的 <em>simulatively confirmed†</em> 是 controlled-surrogate 标签，只表示 prompt-conditioned no-visible-index 与 explicit-index 两类 auxiliary setting 均出现 successor transfer，不表示已经取得原始 prompt 的自然 no-index cohort或因果结果。Terminal 格写成 <em>partial relay + answer exec.</em>：86 个完全 model-common confirmation pairs 的同一-trial reset assay 支持 trace state 经 post-terminal suffix / answer query 的有序部分中介，逐层 answer-query full-state patch 又确认该 state 可执行成实际 greedy 数字。Reset 后仍有 residual effect，所以箭头表示一条通路，不表示唯一或完全中介。</figcaption></figure>
<div class="claim-tier-grid">
  <div><h3>Established</h3><p>Qwen 具有分布式 internal counter：counter-state transplant 可改变下一项 likelihood、attention、candidate argmax 与生成内容，并在成功启动后持续改变后续枚举序列。</p></div>
  <div><h3>Supported, not isolated</h3><p>event tail 比 endpoint 稳定且更 norm-efficient，含有 progress-correlated routing information；但 tail 仍含 score，item span 仍含 city/score/syntax。</p></div>
  <div><h3>Not established</h3><p>单一 counter cell、content-free scalar register、context-invariant arithmetic <code>+1</code> operator、唯一 circuit，或 Qwen 结果对 Gemma 的直接外推。</p></div>
</div>
{parser_contract_section}
<div class="section-conclusion"><strong>Summary 结论。</strong>在 Qwen 的自然 no-index trace 中，我们观察到一个分布式、与事件内容绑定的 internal counter：它记录枚举进度，控制下一项 retrieval，并在一次 state transplant 后持续改变后续枚举。这个结论不意味着已经找到独立的纯算术寄存器，也不能外推到 Gemma。计划内结果均已落盘；第 9 节列的是进一步隔离 counter component 所需的新实验。</div></section>"""

    multi_hop_continuation_table = table(
        (
            "Donor-implied hop",
            "Eligible cells",
            "Exact continuation",
            "Among runs that adopted the first successor",
        ),
        (
            ("<code>k+1</code>", "60", "43/60", "43/43"),
            ("<code>k+2</code>", "60", "43/60", "<strong>43/43</strong>"),
            ("<code>k+3</code>", "40", "27/40", "<strong>27/27</strong>"),
            ("<code>k+4</code>", "40", "27/40", "<strong>27/27</strong>"),
            ("<code>k+5</code>", "20", "13/20", "13/14"),
            ("<code>k+6</code>", "20", "12/20", "12/14"),
        ),
        class_name="compact-table",
    )
    formation_section = f"""<section id="formation"><p class="eyebrow">03 · Logical chain A</p>
<h2>3. 逻辑链 A — Counter state：单 endpoint 很弱，event tail 与 item span 逐级增强</h2>
<p class="lead">Representation 只能说明 count/progress 可读。这里用自然 donor→receiver transplant 问更强的问题：第 k 个 event 的 contextual state 是否足以把 receiver 的下一项选择改成 donor 所暗示的 successor？</p>
<div class="experiment-frame">
  <div><span class="experiment-label">实验目的</span>定位能承载 functional progress information 的最小语义范围：它集中在 item endpoint、较窄 event tail，还是必须依赖完整 event span。</div>
  <div><span class="experiment-label">Cohort</span>Qwen3-8B，N=10；20 discovery + 10 held-out confirmation。全部是 first-pass no-index enumeration，recap 不进入 patch context。</div>
  <div><span class="experiment-label">Discovery</span>只用 k=6 的 forward/backward 两个方向扫全部 36 层；三种 scope 预先固定为 endpoint w1、event tail w4、endpoint-aligned max-common item span。</div>
  <div><span class="experiment-label">Freeze rule</span>只看 paired donor-successor log-odds，要求双向为正，并选 seed-median robust peak 95% 范围内的最早层。Attention 与 generation 对层选择不可见。</div>
  <div><span class="experiment-label">Confirmation</span>冻结 endpoint L26、tail L0、item span L0；在 k={{4,6,8}}、双方向共 60 cells 上读取 likelihood、attention、10-way argmax 与自由 continuation。</div>
  <div><span class="experiment-label">计算方法</span>主量是第 2 节定义的 paired Δroute；同时记录每个 scope 的 patch L2 norm 与 Δroute/norm。Confirmation 的 attention、argmax 与 first-city 都不参与 discovery selection。</div>
  <div><span class="experiment-label">简单例子</span>把 donor k=6 写到 receiver j=5：endpoint 只替换 event 最后一个 token；tail 替换最后四个 tokens；item span 替换整条 event。若只有完整 span 让下一 city 从 N6 跳到 N7，说明单 endpoint 不足。</div>
</div>
<figure><h3 class="figure-title">图 3a · 三种 patch scope 的 36-layer discovery profile</h3><div class="scope-layer-figure">{patch_scope_layer_svg}</div><figcaption>横轴是 zero-based post-block layer；纵轴是 paired donor-successor Δroute（patch−self 的 log-score 差）。不同线/面板对应 endpoint w1、event tail w4 与 endpoint-aligned maximal-common item span，并分别显示 forward/backward 或其稳健聚合。层只在 discovery 上选择。L0 指第一个 decoder block 的输出，不是 raw token embedding；早层 span patch 有最多 downstream computation，因此 L0 peak 是 causal upper bound，不能解释为“机制定位在 L0”。</figcaption></figure>
<h3>3.1 Discovery：scope gradient 已经排除“单 endpoint 就是 counter”</h3>
{patch_scope_discovery_table}
<p>Endpoint 的 median shift 只有 +4.30，且 7/40 cells 方向错误；四-token tail 达 +23.61、40/40 正向，maximal-common item span 达 +58.03、40/40 正向。更窄的 tail 每单位 patch norm 最有效，但较宽 span 最能越过实际决策边界。</p>
<div class="subsection-conclusion"><strong>Experiment 3.1 结论。</strong>Discovery 中的 scope gradient 不支持“单 endpoint 是完整 counter”。Tail 含更高的单位范数 routing signal，完整 event state 则具有最大的总因果效应。</div>
<h3>3.2 Frozen confirmation：ranking、attention 与实际 continuation 是否一起改变</h3>
{patch_scope_confirmation_table}
<p>Maximal-common item-span L0 的实际 transfer 为 forward {l0_item_forward[0]}/{l0_item_forward[1]}、backward {l0_item_backward[0]}/{l0_item_backward[1]}；10/10 seeds 至少出现一次。人工审计显示 40 次 donor city 在前 40 characters 内出现，另 3 次在短 repair preamble 后出现，0 次仅在 recap 中出现。</p>
<h4>一次 transplant 会不会继续影响后面的项目？</h4>
{multi_hop_continuation_table}
<p class="completion-note"><strong>Post-hoc multi-hop reparse。</strong>这里按 completion 中出现的已知 city ordinal 顺序重解析同一批冻结 L0 trials，没有新增干预。43 个从 donor successor 启动的 runs 全部继续到 <code>k+2</code>；所有有资格观察四步的成功 runs 都连续到 <code>k+4</code>。两个 k=4 runs 分别在 <code>k+4</code> 与 <code>k+5</code> 后触及 96-token generation cap；在截断前没有观察到已启动序列转向错误 ordinal。由于该多步 endpoint 是事后重解析，正文应报告 effect size 与完整规则，并在 fresh cohort 中预注册复现。</p>
<div class="scope-compare"><div><h4>L0 item span 是 causal upper bound</h4><p>它保留 city、score、syntax 与 progress，且有 35 个后续 blocks 可传播。它证明完整 contextual event state 足以控制 continuation，不隔离 count component，也不能定位最小机制层。</p></div><div><h4>L16 是 mid-layer robustness</h4><p>在同一 held-out seeds 的 k=6 双向 20 cells 中，median Δ log-odds={float(item_span_l16_summary['median_paired_logodds_shift']):+.2f}，argmax 17/20，first city 16/20；说明中层 event state 仍有因果效力，但不是 fresh confirmation。</p></div></div>
<div class="section-conclusion"><strong>Experiment 3.2 / 逻辑链 A 结论。</strong>Causal sufficiency 随 patch scope 从 endpoint→tail→maximal-common span 增强。较宽 event/progress state 不仅改变下一项；一旦成功重定向，后续枚举会继续沿 donor trajectory 运行。最简洁的解释是：Qwen 在自然 trace 中维护一个分布式、content-bound internal counter，而不是把完整计数压在单 endpoint cell 上。</div></section>"""

    integrated_section = f"""<section id="integrated-chain"><p class="eyebrow">Key causal bridge</p>
<h2>关键闭环：改变 counter state，模型是否从新的位置继续？</h2>
<p class="lead">L16 item-span comparator 把机制读数与行为读数放在同一个干预里：donor counter state 写入 receiver 后，目标不是只让某个 probe 更像 k，而是让后续计算真的选择 donor 所暗示的 successor。L0 multi-hop reparse 再检查这次转向是否会在没有第二次 patch 的情况下延续。</p>
<div class="evidence-ladder">
  <div><span>STATE</span><strong>L16 donor item span</strong><p>同位置、同 grammar；patch width 6–11 tokens。</p></div>
  <div><span>ROUTING</span><strong>attention +{float(item_span_l16_summary['mean_paired_attention_shift']):.2f}</strong><p>20/20 cells 朝 donor successor 增强。</p></div>
  <div><span>RANKING</span><strong>17/20 donor argmax</strong><p>median Δ log-odds {float(item_span_l16_summary['median_paired_logodds_shift']):+.2f}。</p></div>
  <div><span>CONTINUATION</span><strong>16/20 first city</strong><p>全部为 early continuation，0 recap-only。</p></div>
</div>
<div class="section-conclusion"><strong>闭环成立到什么程度。</strong> 对 Qwen 的自然 no-index trace，一次 counter-state transplant 同时改变 attention、successor ranking、首个生成项目与后续枚举轨迹。因此可以说模型维护并使用一个 internal counter。现有 span 仍混合 city、score 与 syntax，所以这里不进一步声称它是独立、content-free 的 arithmetic register；Gemma 尚未完成同口径 no-index 闭环。</div></section>"""

    evidence_ledger_table = table(
        ("Evidence", "Design / held-out result", "What it establishes", "Hard boundary"),
        (
            (
                "Qwen L16 item-span causal bridge",
                "k=6 bidirectional; log-odds 20/20+, attention 20/20+, argmax 17/20, first city 16/20",
                "mid-layer distributed counter state can control next retrieval",
                "same 10 seeds as L0 confirmation; item content retained",
            ),
            (
                "Qwen frozen scope confirmation",
                "endpoint 10/60 vs tail 15/60 vs item span 43/60 first-city transfer; item-span adopters continue 43/43 to k+2 and 27/27 to k+4 where eligible",
                "a single distributed counter-state transplant can redirect the subsequent enumeration trajectory",
                "multi-hop endpoint is a post-hoc reparse; L0 span is an early-layer upper bound and retains item content",
            ),
            (
                "Qwen four-token event tail",
                "median Δ log-odds +21.98; attention 58/60; 8/10 seeds with transfer",
                "narrow event tail contains counter-related routing information",
                "tail retains score numeral; not content-free",
            ),
            (
                "Explicit-index positive control",
                (
                    f"Qwen L{indexed_confirmation_layers['Qwen3-8B']}: "
                    f"{qwen_indexed_transfer}/60 first city "
                    f"({100*float(indexed_progress_summary['Qwen3-8B']['paired_first_known_city_donor_adoption_gain']):+.1f} pp vs receiver); Gemma "
                    f"L{indexed_confirmation_layers['Gemma4-E4B']}: "
                    f"{int(indexed_progress_summary['Gemma4-E4B']['patched_first_known_city_donor_adoption_count'])}/60 "
                    f"({100*float(indexed_progress_summary['Gemma4-E4B']['paired_first_known_city_donor_adoption_gain']):+.1f} pp)"
                ),
                "the same transplant assay can be calibrated on clean indexed grammars",
                "visible k is a direct progress confound; not no-index counter evidence",
            ),
            (
                "Targeted retrieval banks",
                (
                    "exact-aligned confirmation: final-count / joint retrieval+count failure Δ, "
                    f"Qwen {probability_effect_ci(targeted_effects['Qwen3-8B']['targeted_bank_changes_final_count'])} / "
                    f"{probability_effect_ci(targeted_effects['Qwen3-8B']['retrieval_failure_propagates_to_count'])}; "
                    f"Gemma {probability_effect_ci(targeted_effects['Gemma4-E4B']['targeted_bank_changes_final_count'])} / "
                    f"{probability_effect_ci(targeted_effects['Gemma4-E4B']['retrieval_failure_propagates_to_count'])}"
                ),
                "model-specific head banks are necessary and the lesion propagates to final count",
                "joint failure is not a formal natural-effects mediation estimate; bank widths are not cross-model comparable",
            ),
            (
                "Carrier → commit",
                f"clean-carrier restoration: Qwen {ci(write_effects['Qwen3-8B']['clean_carrier_restoration'])}; Gemma {ci(write_effects['Gemma4-E4B']['clean_carrier_restoration'])}",
                "retrieved information is written into later trace state",
                "write evidence comes from controlled grammar families",
            ),
            (
                "Terminal state restoration",
                f"Qwen {ci(terminal_effects['Qwen3-8B']['restoration'])}; Gemma {ci(terminal_effects['Gemma4-E4B']['restoration'])}",
                "terminal trace state can affect correct-count margin",
                "fixed suffix only; free-running end-to-end sufficiency not shown",
            ),
            (
                "Answer-query full-state execution",
                (
                    f"8-layer registered existing-split sweep; final sampled layer: "
                    f"Qwen {probability_ci(answer_query_final_rows['Qwen3-8B'], 'full_donor_adoption')}; "
                    f"Gemma {probability_ci(answer_query_final_rows['Gemma4-E4B'], 'full_donor_adoption')}"
                ),
                "complete answer-query state can causally control the actual greedy count",
                "full residual carries mixed context; same-trial relay is only partial and does not isolate a low-dimensional counter",
            ),
            (
                "Same-trial terminal relay",
                "86 exact model-common directed pairs/model; terminal patch × suffix/answer reset factorial assay",
                "trace state → post-terminal suffix / answer-query → count margin is an ordered partial causal pathway",
                "fixed-suffix scoring; residual effect excludes complete, unique, or exclusive mediation",
            ),
        ),
        class_name="evidence-ledger",
    )
    ledger_section = f"""<section id="ledger"><p class="eyebrow">07 · Evidence synthesis</p>
<h2>7. Evidence synthesis：按 claim 贡献排序，而不是按实验时间排序</h2>
{evidence_ledger_table}
<div class="core-claim"><strong>综合结论。</strong> 主证据支持一个 distributed, content-bound internal counter：它可在中层控制下一项 retrieval，并在成功重定向后继续影响后续枚举。Event tail 提供更窄的 routing signal；single endpoint、continued counting 与 operator scan 都不支持把它描述为独立的 memoryless arithmetic register。</div>
<div class="section-conclusion"><strong>Section 7 结论。</strong>证据强度排序是：Qwen 自然 no-index 行为级 transplant ＞ 两模型 targeted-retrieval necessity 与 carrier/terminal controlled edges ＞显式-index或改 prompt 校准 ＞ standalone decoder 与失败的算子扫描。论文 claim 应按这一顺序书写。</div></section>"""

    extension_section = f"""<section id="extension-audit"><p class="eyebrow">08 · Native-thinking extension audit</p>
<h2>8. 扩展问题审计：哪些实验强化主张，哪些实验只负责限定边界？</h2>
<p class="lead">这些实验不再与主链平铺。它们的作用是区分“可读 geometry”“局部 steering”“完整 event-state sufficiency”和“真正的 memoryless recurrence”。完整定义与失败模式按同一顺序放在 Appendix B–G。</p>
{extension_audit_table}
<div class="section-conclusion"><strong>Extension audit 结论。</strong> CountScope 与 steering 说明 counter information 可读、可局部操纵；separator 与 K/V 说明其历史载体是分布式 event memory。Cross-context continued counting、maximum 与 operator scan 没有给出 context-invariant +1 算子，但这不否定自然 item-span transplant 已观察到的持续枚举。</div></section>"""

    limitations_section = """<section id="limitations"><p class="eyebrow">09 · What remains</p>
<h2>9. What remains：若要把 counter 再隔离成纯算术 component，还缺哪些实验？</h2>
<ol>
  <li><strong>Answer-state 的 absolute-position matched control。</strong>本轮 donor/receiver 固定了同一语义/字面 query site，但没有要求绝对 token index 相等。可用 length-balanced prompt filler 让不同 count 的 answer query 对齐，并加入 same-count/different-position control，区分可执行 count state 与 trace-length/position signal。</li>
  <li><strong>Same-progress / different-score tail control。</strong>固定 k 与 grammar，只改变 score surface，或投影掉 score direction；检验四-token tail 的 routing 是否保留。</li>
  <li><strong>Fixed-L16 matched item controls + fresh cohort。</strong>加入 same-k cross-seed、shuffled item、within-item permutation 与完整 donor−receiver delta 等范数 span；预注册 L16 后在新 10 seeds 上复现，区分 progress 与可复制 event content。</li>
  <li><strong>Same-trial terminal relay 的 fresh replication。</strong>当前 86-pair exact-common factorial assay 已经接实一条有序部分通路；若需要把它升级成最强 headline，可在新 seeds 上原样复现冻结的 terminal geometry、suffix reset 与 answer-query reset，而不是再发明一套独立 patch。</li>
  <li><strong>若要主张 memoryless +1，必须检验 donor-dependent transition。</strong>在同一新 item 与相同 surface context 下，只改变进入 transition 前的 state/history；必要时联合 residual + exact K/V splice，并观察 offset 是否跨 hop 1、hop 2 保留。</li>
  <li><strong>跨模型自然使用。</strong>在 Gemma 上重新建立原 prompt 下的自然 no-index cohort 与同一 scope protocol；Appendix K 的 prompt-conditioned positive control 证明能力存在，但不能代替自然生成实验。</li>
</ol>
<div class="section-conclusion"><strong>与 Non-thinking 对照后的判断。</strong>当前 6.4 已经提供同一 trial 的 ordered partial mediation，6.3 又单独确认 answer state 可执行，因此若论文只 claim“这是一条 counting pathway”，没有缺失的必补桥接实验。下一优先级是 absolute-position 与 event-content controls，它们负责把 counter 从位置和内容中进一步隔离；fresh relay replication 负责提高证据等级。Donor-dependent transition 只在声称固定 arithmetic <code>+1</code> operator 时必要；Gemma natural cohort 只在声称跨模型自然普适性时必要。</div></section>"""

    legacy_appendices = section_body(sections["appendix"])
    legacy_appendices = legacy_appendices[legacy_appendices.index("<details") :]
    for old_label, new_label in (
        ("Appendix A", "Appendix M"),
        ("Appendix B", "Appendix N"),
        ("Appendix C", "Appendix O"),
        ("Appendix D", "Appendix P"),
        ("Appendix E", "Appendix Q"),
    ):
        legacy_appendices = legacy_appendices.replace(old_label, new_label)
    legacy_appendices = legacy_appendices.replace(
        "我们确认一条可干预 recurrent pathway",
        "我们确认若干可干预的 state-to-routing edges",
    )
    walkthrough_appendix = section_body(sections["walkthrough"]).replace(
        "count state 分布在 recurrent trace dynamics 中",
        "count-aligned information 分布在 trace dynamics 中",
    )
    audit_appendix = section_body(sections["audit"]).replace(
        "本报告证明一条 pathway",
        "本报告连接若干 causal edges",
    )
    audit_appendix = re.sub(
        r"Generated UTC: [^<]+<br>Schema: realistic_niah_v5_native_thinking_restructured_v\d+",
        (
            f"Generated UTC: {esc(generated)}<br>Schema: "
            "realistic_niah_v5_native_thinking_restructured_v14"
        ),
        audit_appendix,
        count=1,
    )

    appendix_section = f"""<section id="appendix"><p class="eyebrow">Appendix · Ordered auxiliary evidence</p>
<h2>Appendix：辅助实验、失败实验与复现材料</h2>
<p class="lead">附录按“定义 → standalone readout → recurrence → steering → structural alternatives → historical controls”的逻辑顺序排列。它们不改变正文 claim 层级。</p>
<div class="appendix-sequence">
<details><summary>Appendix A · 定义、实验合同与判据</summary><div>{section_body(sections['definitions'])}{section_body(sections['design'])}</div></details>
<details><summary>Appendix B · CountScope：为什么高 NCC 不保证大 k transplant 成功</summary><div>
<div class="appendix-method"><p><strong>实验目的。</strong>检验一个可跨 seed 解码的 count geometry，是否也是脱离原历史后可直接“翻译成数字”的 context-invariant state。</p><p><strong>设定与计算。</strong>把完整序列第 k 个 item hidden state 移植到仅有一个 matched placeholder item 的最小 receiver；对候选 1,…,N 做 sequence scoring，记录 candidate argmax 是否等于 k。NCC 只用 discovery centroids 判最近类别，衡量方向一致性而非 transplant compatibility。</p><p><strong>简单例子。</strong>Donor 在完整 trace 中已完成第 7 项；若它真是独立寄存器，把该 state 放到单 placeholder receiver 后仍应输出 7，而不依赖原 prefix/KV history。</p></div>
<p><strong>结果。</strong>N=3 的 k=1/2/3 candidate accuracy 为 0.90/0.70/1.00；N=10 只在 k=1–4 尚可，k≥5 基本失效。</p><p><strong>分析。</strong>高 NCC 只说明多个 donor states 在某个表示空间中按 k 有序；充分性还要求 donor state 与 receiver 的 prefix history、绝对位置、KV memory 与 downstream readout 兼容，两者不矛盾。</p><div class="section-conclusion"><strong>Appendix B 结论。</strong>Count information 可读，但单 hidden state 不是跨上下文通用的数字寄存器。</div></div></details>
<details><summary>Appendix C · Continued counting：没有观察到稳定 donor-dependent +1</summary><div>
<div class="appendix-method"><p><strong>实验目的。</strong>直接检验 source 的最新 latent count 能否在 target suffix 上继续递推，而无需重新读取全部 source history。</p><p><strong>设定与计算。</strong>Source 含 N<sub>s</sub> 个 items，target 含 N<sub>t</sub> 个；把 source 最后 m 个 item states 写到 target 开头 m 个位置。若存在 memoryless continuation，紧接的 item 应编码 N<sub>s</sub>+1，最终应趋向 N<sub>s</sub>+N<sub>t</sub>−m。分别读取 hop 1、hop 2 与 final candidate。</p><p><strong>简单例子。</strong>Source=5、target=4、m=2；理想 counter 应让 target 的下一项从 6 开始，最终输出 7，而不是仍按 target 自己的局部位置继续。</p></div>
<p><strong>结果。</strong>N=3 的 hop 1 仅有短暂 0.3–0.7 candidate；N=10 hop 1≤0.20，hop 2 与 final 约为 0。</p><p><strong>分析。</strong>Residual-only transplant 与 target-history mismatch 可能使真机制失配，因此 null 不是“模型绝无任何递推”；但它没有提供 donor-dependent +1 的正证据。</p><div class="section-conclusion"><strong>Appendix C 结论。</strong>当前 cross-context assay 不支持一个脱离 event history、按固定公式运行的 content-free arithmetic <code>+1</code> operator。</div></div></details>
<details><summary>Appendix D · Geometry steering：局部 count-like direction，不是全局算术轴</summary><div>
<div class="appendix-method"><p><strong>实验目的。</strong>检验 count geometry 的局部 +1 方向是否能因果推动同层 state，而不只被 probe 被动读出。</p><p><strong>设定与计算。</strong>在每层用 discovery 拟合 <span class="formula">v<sub>ℓ</sub><sup>+1</sup>=mean<sub>k,seed</sub>(h<sub>ℓ,k+1</sub>−h<sub>ℓ,k</sub>)</span>，对单 endpoint 注入正向 steering，并减去 opposite / orthogonal control 的 decoded-count displacement。各层独立运行；“全层扫描”是寻找 effect 随深度的 profile，不是同时向所有层注入。</p><p><strong>简单例子。</strong>若原 state 被 decoder 读作 5，加入 L19 的 +1 direction 后期望更靠近 6；加入等范数 orthogonal direction不应产生同样移动。</p></div>
<p><strong>结果。</strong>共同 peak 在 L19：N=3 的对照校正位移 +0.622 [0.471, 0.767]，N=10 为 +0.215 [0.107, 0.334]。</p><p><strong>分析。</strong>方向有局部因果相关性，但 effect 小于一个完整 count step，N=10 更弱，且没有行为级 successor adoption。</p><div class="section-conclusion"><strong>Appendix D 结论。</strong>存在局部 count-like geometry；尚未定位跨层稳定、可执行整数加法的全局算术轴。</div></div></details>
<details><summary>Appendix E · Separator dose：full event 明显强于 marker</summary><div>
<div class="appendix-method"><p><strong>实验目的。</strong>检验历史进度主要存于重复 separator/marker，还是分布在完整 event content 中。</p><p><strong>设定与计算。</strong>把 later events 的 marker、closing 或 full-event states 依次 collapse 到第一个可用 event state；以被 collapse 的 event 数为剂量，拟合 outcome 对剂量的 per-event slope β。更负的 β 表示删除每个历史 event 造成更大损伤。</p><p><strong>简单例子。</strong>若 trace 已有 8 个 events，只把后 4 个逗号状态改成第一个逗号；若逗号就是完整 counter，这应接近替换后 4 个完整 events 的损伤。</p></div>
<p><strong>结果。</strong>N=10 的 per-event slopes 为 marker −0.125、closing −0.219、full-event −0.690。</p><p><strong>分析。</strong>Marker/closing 有贡献，但远弱于完整 event；这与 event-local state 和历史 KV memory 共同承载进度相容。</p><div class="section-conclusion"><strong>Appendix E 结论。</strong>Separator 不是充分的单点 counter，完整 event 是更主要的功能单位。</div></div></details>
<details><summary>Appendix F · Maximum-count：没有一般化的 max operator</summary><div>
<div class="appendix-method"><p><strong>实验目的。</strong>排查 transplant 后的输出是否只是取 donor 与 target history 中较大的 latent count，而非从 donor 继续。</p><p><strong>设定与计算。</strong>把 source last-m states 写到 target last-m，比较候选 <span class="formula">max(N<sub>s</sub>, N<sub>t</sub>−m)</span> 与 donor-continuation、target-retention 等读数。</p><p><strong>简单例子。</strong>N<sub>s</sub>=7、N<sub>t</sub>=10、m=2 时，max rule 预测 8；若 N<sub>s</sub>=9 则预测 9。两个分支都必须超出未干预 target baseline 才能说明存在 max operator。</p></div>
<p><strong>结果。</strong>只有 donor-dominant cells 出现 0.13–0.30 candidate；target−m branch 与保留的 target history 完全重合。</p><p><strong>分析。</strong>后一个分支无法与普通 target retention 区分，因此不构成 max computation 的独立证据。</p><div class="section-conclusion"><strong>Appendix F 结论。</strong>没有观察到可一般化的 maximum-count operator；该实验只排除了一个简单替代解释。</div></div></details>
<details><summary>Appendix G · Marker K/V 与递推算子扫描</summary><div>
<div class="appendix-method"><p><strong>实验目的。</strong>区分 marker 的 key、value 与联合 KV 对历史读取的贡献，并穷举若干候选递推算子。</p><p><strong>设定与计算。</strong>在固定 marker attention edge 上分别 splice K-only、V-only、K/V，并扫描 layer bands；成功率是干预后目标 routing/候选满足预注册判据的 cell fraction。Operator scan 将结果分类为 reset、target +1 与其他。</p><p><strong>简单例子。</strong>若 marker key 只负责“去哪里找”、value 负责“读出什么”，V-only 应强于 K-only，而联合 K/V 可能最佳；若 state 真执行 +1，成功 local arm 的下一 hop 应持续落在 target+1 类。</p></div>
<p><strong>结果。</strong>Marker K/V=0.835、V-only=0.500、K-only=0.276，L20–23 band=0.540；operator scan 中 reset 占 97.08%，target +1 仅 0.625%，成功 first-stage local arms 仍无 next +1。</p><p><strong>分析。</strong>K/V 结果说明 event-indexed memory 是合理 substrate；operator 分布却与稳定加法更新不相容。</p><div class="section-conclusion"><strong>Appendix G 结论。</strong>证据更支持 distributed event memory / late aggregation，而不是 residual endpoint 上的 arithmetic recurrence。</div></div></details>
<details><summary>Appendix H · Carrier readout diagnostics：NCC 与最终 count margin</summary><div><p class="main-note">正文 5.1 只使用预注册 carrier RMS deformation。以下 NCC 与 final output margin 回答更下游、更依赖 readout 选择的问题，因此保留为诊断而不阻塞主边。</p>{write_51_diagnostics}<div class="section-conclusion"><strong>Appendix H 结论。</strong>Carrier hidden state 的 direct damage 清楚；但经 centroid geometry 或最终答案边际读取时，模型/grammar 的特异性不一致，不能据此升级为统一线性 counter code。</div></div></details>
<details><summary>Appendix I · 单 endpoint、grammar stratification 与历史四-token tail</summary><div><div class="appendix-method"><p><strong>实验目的。</strong>检查窄位点效应是否跨 grammar、layer 与 seed 稳定，并区分 endpoint register 与 event-tail routing carrier。</p><p><strong>设定与计算。</strong>系统 scope confirmation 固定 endpoint L26；历史 assay 在 discovery 定位 L16 width-4 tail。两者都比较 donor-successor likelihood/attention 与自由 continuation。</p><p><strong>简单例子。</strong>只替换 item 最后一个句号若能稳定令 N6→N7，才支持 endpoint register；替换句号前四 tokens 成功则只说明更宽 tail 含 routing information。</p></div><p><strong>结果。</strong>Endpoint L26 只有 8/60 donor argmax、10/60 first-city transfer，并集中在 2/10 seeds。旧 L31 plain-period panel 的 11/16 与其他 grammar 的低命中说明 endpoint effect 很脆弱。历史 L16 width-4 tail 在 confirmation 中 60/60 likelihood 正向、59/60 attention 正向，但生成只有 {int(round(60*float(historical_event_tail_summary['patched_greedy_donor_adoption_rate'])))}/60。</p><p><strong>分析。</strong>Tail 是窄 routing evidence；它仍不是 content-free endpoint register。</p><div class="section-conclusion"><strong>Appendix I 结论。</strong>稳定作用需要超过单 endpoint 的 event-local范围，并受 grammar 与 seed 调节。</div></div></details>
<details><summary>Appendix J · 显式-index与历史 controlled routing 校准</summary><div>{indexed_control_section}<h3>J.2 逐 k、逐方向 confirmation cells</h3><p>所有行都来自同一组 10 confirmation seeds，并使用 representation-frozen 的模型特异层（Qwen L{indexed_confirmation_layers['Qwen3-8B']}、Gemma L{indexed_confirmation_layers['Gemma4-E4B']}）；没有使用 indexed outcomes 反选层。First-city transfer 是自由 continuation 中第一个匹配 gold city 的 ordinal 是否等于 donor k+1。</p>{indexed_crossk_table}<h3>J.3 历史 controlled commit→query patch</h3><p class="main-note">以下实验保留其 direct causal routing 价值，但 trace grammar/控制条件不允许它承担 natural no-index recurrent counter 的主 claim。</p>{historical_commit}<div class="section-conclusion"><strong>Appendix J 总结。</strong>这些结果校准 state-transplant assay，并证明显式/受控条件下的 state-dependent routing；可见 index 与 grammar confounds 使它们不能替代 Qwen 自然 no-index 主实验。</div></div></details>
<details><summary>Appendix K · Gemma prompt-conditioned no-visible-index forward transplant</summary><div>
<div class="appendix-method"><p><strong>实验目的。</strong>在原 prompt 无法筛出足够 Gemma 自然 no-index traces 后，单独检验 Gemma 是否具备形成可 transplant 的 no-visible-index event state 的能力；它不是自然采用率实验。</p><p><strong>设定。</strong>只改 passage 后的任务尾：每条 event 使用同一个 <code>FOUND:</code> marker，禁止编号、ordinal、running subtotal、prose 与 recap；另加固定、无 count 信息的 assistant prefill <code>FOUND: </code>。Passage tokens 不变。</p><p><strong>冻结与计算。</strong>从 seed 1234 连续扫描 52 个 prompts，30 个通过逐行 gold city/score 与无显式 progress cue 审计；最早 20 个作 discovery、随后 10 个作 confirmation。筛选不读取 hidden states、patch outcome 或 terminal-total correctness。固定 L16 完整 event span、forward groups 3←4、5←6、7←8，并使用第 2 节四级读数。</p><p><strong>简单例子。</strong>Receiver 的第四条与 donor 的第五条都只以 <code>FOUND:</code> 开头，没有“4/5”表面 label；若把 donor event 写入 receiver 后首个 city 跟随 donor successor，说明 Gemma 能在这种受控 grammar 中形成有效 event state。</p></div>
<p><strong>Prompt intervention 的逐字内容。</strong>完整输入仍包含每个样本各自的原 passage；下列内容只展示静态且真正被替换的 post-passage user-task tail，以及在 chat template 之后另行追加的 assistant prefill。没有增加额外 system message。</p>
<p><strong>原始 task tail（被逐字替换的部分）</strong></p>
<pre class="parser-code">How many city-score audit records are in the passage?
Reason concisely without repeating or restarting.
Stop as soon as you determine the count, then output exactly one line:
Total: &lt;integer&gt;</pre>
<p><strong>Gemma controlled task tail（替换后的逐字内容）</strong></p>
<pre class="parser-code">How many city-score audit records are in the passage?
During reasoning, write each matching record on one line using exactly:
FOUND: &lt;city&gt; | score &lt;score&gt;
Always use the same literal marker FOUND:. Do not number records, use ordinal
words, item/record/excerpt indices, mention how many have been found, or write a
running subtotal. Do not write a preamble, recap, verification, or explanation.
Immediately after the final FOUND: line, output exactly one line:
Total: &lt;integer&gt;
Your assistant response has already begun with the literal text 'FOUND: ';
continue that first line directly with the first city and score.</pre>
<p><strong>Assistant prefill（逐字内容）</strong></p>
<pre class="parser-code">FOUND: </pre>
<p class="main-note"><strong>解释边界。</strong>这不是解码器层面的 grammar constraint；格式由 task-tail instruction 与 assistant prefill 诱发，再由 exact-format parser 决定 cohort eligibility。因此该条件检验的是定制 prompt 下可诱发的机制能力，而不是原始 prompt 下的自然机制采用。</p>
{gemma_prompt_k_table}
<p><strong>与 Qwen 最可比的 forward 子集。</strong>Qwen 自然 no-index L0 item-span 在 k∈{{4,6,8}} 的 30 个 forward cells 中为 donor argmax {qwen_l0_forward_argmax}/30、greedy donor {l0_item_forward[0]}/30、self donor {qwen_l0_forward_self_donor}/30、attention Δ&gt;0 {qwen_l0_forward_attention}/30、log-odds Δ&gt;0 {qwen_l0_forward_logodds}/30。Gemma 对应为 donor argmax {int(gemma_prompt_noindex_confirm_pooled['donor_argmax_patch']['hits'])}/30、greedy donor {int(gemma_prompt_noindex_confirm_pooled['greedy_donor_adoption_patch']['hits'])}/30、self donor {int(gemma_prompt_noindex_confirm_pooled['greedy_donor_adoption_self']['hits'])}/30、attention Δ&gt;0 {int(gemma_prompt_noindex_confirm_pooled['positive_attention_gain']['hits'])}/30、log-odds Δ&gt;0 {int(gemma_prompt_noindex_confirm_pooled['positive_logodds_gain']['hits'])}/30。</p>
<p><strong>分析。</strong>两模型的 candidate ranking 与自由 continuation 效应同量级：Qwen forward 为 25/30，Gemma 为 22/30；Gemma 的冻结 attention bank 则明显更不一致（12/30 vs Qwen 29/30）。由于 prompt、layer、span geometry 与模型不同，这不是正式 model×condition 比较，也不证明共享同一 head circuit。</p>
<div class="section-conclusion"><strong>Appendix K 结论。</strong>Gemma 在 prompt-conditioned、无可见 index 的固定 grammar 中具备 event-state routing 能力；自然 no-index 主张仍严格限于 Qwen，不能用该能力参照外推。</div>
<p class="audit-list">本地归档：<a href="../work/gemma_prompt_conditioned_noindex_20260827/README.md">实验说明</a> · <a href="../work/gemma_prompt_conditioned_noindex_20260827/cohort_full_20_10/manifest.json">cohort manifest</a> · <a href="../work/gemma_prompt_conditioned_noindex_20260827/forward_l16_item_span/analysis.json">causal analysis</a></p>
</div></details>
<details><summary>Appendix L · 单-seed walkthrough 与旧 no-running-index early-stop restoration</summary><div>{walkthrough_appendix}</div></details>
{legacy_appendices}
<details><summary>Appendix R · Reproducibility ledger</summary><div>{audit_appendix}<p><span class="audit-badge">Qwen no-index primary</span><span class="audit-badge">20 discovery / 10 confirmation</span><span class="audit-badge">Gemma prompt-conditioned auxiliary</span><span class="audit-badge">explicit-index positive control</span><span class="audit-badge">manual generation audit</span><span class="audit-badge">p-values secondary</span></p><p>Systematic scope selection never used attention or generation. L16 Qwen item-span is explicitly labeled post-hoc robustness on the same confirmation seeds. All no-index Qwen item-span generation hits were manually checked against recap-only false positives. The indexed Qwen/Gemma layers were independently frozen on discovery likelihood only; visible progress labels make those panels positive controls. Gemma prompt-conditioned cohort selection was patch-outcome blind and used a count-free marker/prefill, but the prompt intervention prevents treating it as natural-generation confirmation.</p><p>Companion seed browser and full internal-counter appendix: <a href="NiaH_Native-thinking_Internal-counter_report.html">NiaH_Native-thinking_Internal-counter_report.html</a>.</p></div></details>
</div>
<div class="section-conclusion"><strong>Appendix 结论。</strong> 对 Qwen 的自然 no-index 证据，主实验支持一个 distributed, content-bound internal counter：event span、tail、KV history 与下游 routing 共同承担计算，单 endpoint 或 standalone CountScope 都不足以概括它。Cross-context arithmetic assays 的失败限制的是“独立固定 +1 operator”这一更强解释，而不是自然 trace 中 counter 的存在。Gemma 的 prompt-conditioned no-visible-index 结果复现了同量级的行为路由能力，但仍须标成受控参照，不能升级为自然 no-index claim。</div></section>"""

    main_start = html_text.index("<main>") + len("<main>")
    main_end = html_text.index("</main>", main_start)
    new_main = "\n".join(
        (
            summary_section,
            baseline_section,
            representation_section,
            formation_section,
            retrieval_section,
            write_section,
            answer_section,
            integrated_section,
            ledger_section,
            extension_section,
            limitations_section,
            appendix_section,
        )
    )
    html_text = html_text[:main_start] + "\n" + new_main + "\n" + html_text[main_end:]

    # Keep secondary and null-result material available without expanding the
    # initial reading path. Readers can open each appendix independently.
    html_text = html_text.replace(
        '<details class="appendix-block" open>',
        '<details class="appendix-block">',
    )

    input_paths = [
        args.reference_report,
        args.sample_alignment_audit,
        args.output_alignment_audit,
        args.aligned_grammar_panel_root / "shared_alignment_keys.jsonl",
        args.aligned_routed_panel_root / "shared_alignment_keys.jsonl",
        *(
            args.aligned_run_root / model / "suite_complete.json"
            for model in MODELS
        ),
        *(
            args.aligned_run_root
            / model
            / "targeted_retrieval"
            / f"analysis_{phase}"
            / "claim_gates.json"
            for model in MODELS
            for phase in ("discovery", "confirmation")
        ),
        *(
            args.aligned_run_root
            / model
            / "targeted_counter_write"
            / f"analysis_{phase}"
            / "claim_gates.json"
            for model in MODELS
            for phase in ("discovery", "confirmation")
        ),
        *(
            args.aligned_run_root
            / model
            / "grammar_span"
            / f"analysis_{phase}"
            / "claim_gates.json"
            for model in MODELS
            for phase in ("discovery", "confirmation")
        ),
        *(
            args.aligned_run_root
            / model
            / "commit_state_to_targeted_query"
            / f"analysis_{phase}"
            / "claim_gates.json"
            for model in MODELS
            for phase in ("discovery", "confirmation")
        ),
        *(
            args.aligned_run_root
            / model
            / "stratified_ncc"
            / timing
            / "analysis"
            / "claim_gates.json"
            for model in MODELS
            for timing in ("rank_after_city", "rank_before_city")
        ),
        *(
            args.aligned_run_root
            / model
            / "direct_count_logit_margin"
            / timing
            / "analysis"
            / "claim_gates.json"
            for model in MODELS
            for timing in ("rank_after_city", "rank_before_city")
        ),
        args.patch_scope_layer_sweep,
        args.patch_scope_layer_plot,
        args.patch_scope_frozen_confirmation,
        args.patch_scope_generation_audit,
        args.item_span_l16,
        args.item_span_l16_generation_audit,
        args.historical_event_tail_confirmation,
        args.indexed_progress_cohort_manifest,
        args.indexed_progress_freeze_manifest,
        args.indexed_progress_generation_audit,
        args.gemma_prompt_conditioned_noindex_cohort_manifest,
        args.gemma_prompt_conditioned_noindex_analysis,
        args.historical_causal_chain_statistics,
        args.terminal_relay_common_support / "summary.json",
        args.terminal_relay_common_support / "common_support_pairs.csv",
        args.terminal_relay_common_support / "common_support_cell_counts.csv",
        args.answer_query_plan_root / "alignment_manifest.json",
        *(
            args.answer_query_plan_root / f"{model}_pairs.jsonl"
            for model in MODELS
        ),
        *(
            args.answer_query_layer_root
            / model
            / "answer_query_layer_sweep"
            / "trials.jsonl"
            for model in MODELS
        ),
        *(
            args.answer_query_layer_root
            / model
            / "answer_query_layer_sweep"
            / "analysis"
            / filename
            for model in MODELS
            for filename in (
                "audit.json",
                "layer_effects.csv",
                "seed_effects.csv",
                "pair_effects.csv",
            )
        ),
        *(
            args.indexed_progress_discovery_root
            / model
            / "indexed_progress_control"
            / "discovery_layer_sweep"
            / "layer_sweep_analysis.json"
            for model in MODELS
        ),
        *(
            args.indexed_progress_confirmation_root
            / model
            / "indexed_progress_control"
            / "confirmation_analysis.json"
            for model in MODELS
        ),
        args.qwen_targeted_analysis,
        args.gemma_targeted_analysis,
        args.running_pca3_coordinates,
        args.representation_root / "site_selected.csv",
        args.representation_root / "site_layer_candidates.csv",
        args.dual_endpoint_root / "Qwen3-8B" / "pca16_whiten" / "final_count_selected.csv",
        args.dual_endpoint_root / "Gemma4-E4B" / "pca16_whiten" / "final_count_selected.csv",
        args.dual_endpoint_root / "Qwen3-8B" / "pca16_whiten" / "final_count_candidate_metrics.csv",
        args.dual_endpoint_root / "Gemma4-E4B" / "pca16_whiten" / "final_count_candidate_metrics.csv",
        *(args.band_diagnostic_root / model / "band_diagnostic.json" for model in MODELS),
        args.geometry_comparison_report,
        args.atlas_root / "p0_head_atlas_manifest.json",
        args.atlas_root / "p0_targeted_retrieval_head_scores.csv",
        args.atlas_root / "Qwen3-8B_adjacent_rank_after_city_Top128_p0_attention_sum.svg",
        args.atlas_root / "Gemma4-E4B_adjacent_rank_after_city_Top6_p0_attention_sum.svg",
        *(Path(example["path"]) for example in attention_examples),
        *(args.atlas_root / filename for filename in (
            "Qwen3-8B_same_unit_rank_before_city_p0_head_map.svg",
            "Gemma4-E4B_same_unit_rank_before_city_p0_head_map.svg",
            "Qwen3-8B_structural_invariant_bullet_p0_head_map.svg",
            "Gemma4-E4B_structural_invariant_bullet_p0_head_map.svg",
        )),
        *(args.snapshot_root / "targeted_counter_write_20260822" / model / "complete.json" for model in MODELS),
        *grammar_anchor_paths.values(),
        *(args.snapshot_root / "commit_state_query_20260822" / model / "commit_to_query_complete.json" for model in MODELS),
        args.snapshot_root / "qwen_grammar_span_decomposition_complete.json",
        args.snapshot_root / "gemma_grammar_span_decomposition_complete.json",
        args.snapshot_root / "targeted_counter_20260822" / "Qwen3-8B" / "targeted_counter_complete.json",
        *(args.snapshot_root / "single_seed_walkthrough_20260822_v2" / model / "analysis" / "walkthrough_complete.json" for model in MODELS),
        *(
            args.ncc_supplement_root / model / "ncc_analysis" / "claim_gates.json"
            for model in MODELS
        ),
        *(
            args.ncc_supplement_root
            / model
            / "ncc_analysis"
            / "layerwise_timing_diagnostic.json"
            for model in MODELS
        ),
        args.stratified_ncc_root
        / "synthesis"
        / "stratified_ncc_synthesis.json",
        *(
            args.stratified_ncc_root
            / model
            / timing
            / "analysis"
            / "claim_gates.json"
            for model in MODELS
            for timing in ("rank_after_city", "rank_before_city")
        ),
        *(
            args.stratified_ncc_root
            / model
            / "stratified_ncc_input_manifest.json"
            for model in MODELS
        ),
        *(
            args.stratified_ncc_root
            / model
            / f"{timing}_site_audit.json"
            for model in MODELS
            for timing in ("rank_after_city", "rank_before_city")
        ),
        args.logit_margin_root
        / "synthesis"
        / "targeted_logit_margin_synthesis.json",
        *(
            args.logit_margin_root
            / model
            / timing
            / "analysis"
            / "claim_gates.json"
            for model in MODELS
            for timing in ("rank_after_city", "rank_before_city")
        ),
        *(
            args.logit_margin_root
            / model
            / "targeted_logit_margin_complete.json"
            for model in MODELS
        ),
        *(
            args.ncc_supplement_root
            / model
            / "unnumbered_analysis_confirmation"
            / "claim_gates.json"
            for model in MODELS
        ),
        *(
            args.ncc_supplement_root
            / model
            / "unnumbered_analysis_confirmation"
            / "occurrence_metrics.csv"
            for model in MODELS
        ),
        *(args.ncc_supplement_root / model / "generation_manifest.json" for model in MODELS),
        *(
            args.snapshot_root / f"gemma_query_mediation_{geometry}_discovery_claim_gates.json"
            for geometry in ("endpoint", "suffix4", "suffix8")
        ),
        args.snapshot_root / "gemma_query_mediation_complete.json",
        *(args.atlas_root / filename for filename in (
            "Qwen3-8B_same_unit_rank_before_city_p0_needle_ordinal_by_head.svg",
            "Gemma4-E4B_same_unit_rank_before_city_p0_needle_ordinal_by_head.svg",
            "Qwen3-8B_structural_invariant_bullet_p0_needle_ordinal_by_head.svg",
            "Gemma4-E4B_structural_invariant_bullet_p0_needle_ordinal_by_head.svg",
        )),
        *(Path(path) for evidence in token_evidence.values() for path in evidence["input_files"]),
    ]
    manifest = {
        "schema_version": "realistic_niah_v5_native_thinking_restructured_v14",
        "status": "PASS",
        "generated_at": generated,
        "output": str(args.output),
        "scientific_contract": {
            "discovery_seed_count": 20,
            "confirmation_seed_count": 10,
            "outcome_blind": True,
            "cross_model_sample_alignment_audit_status": (
                sample_alignment_audit["status"]
            ),
            "cross_model_sample_alignment_evidence_families": len(
                sample_alignment_audit["evidence"]
            ),
            "cross_model_output_alignment_audit_status": (
                output_alignment_audit["status"]
            ),
            "cross_model_output_alignment_evidence_families": len(
                output_alignment_audit["evidence"]
            ),
            "parser_design_contract_in_main_text": True,
            "parser_design_disclosure_count": 4,
            "qwen_natural_noindex_eligibility_field": "strict_eligible_no_explicit_count_cue",
            "qwen_natural_noindex_primary_gate": "global_clean",
            "natural_context_excludes_future_recap": True,
            "generation_primary_endpoint": "first_generated_known_city_ordinal",
            "generation_bullet_parser_is_audit_only": True,
            "token_source_primary_retrieval_endpoint": {
                model: token_evidence[model]["target_exact_endpoint"]
                for model in MODELS
            },
            "token_source_endpoint_is_teacher_forced_not_free_generation": True,
            "selection_rank_used": False,
            "qwen_targeted_bank": 128,
            "gemma_targeted_bank": 6,
            "qwen_no_index_primary_n": 10,
            "qwen_scope_discovery_k": [6],
            "qwen_scope_confirmation_k": [4, 6, 8],
            "qwen_scope_layer_selection_hid_attention_and_generation": True,
            "l16_item_span_is_posthoc_robustness_not_fresh_confirmation": True,
            "indexed_positive_control_models": list(MODELS),
            "indexed_positive_control_n": 10,
            "indexed_positive_control_discovery_k": [6],
            "indexed_positive_control_confirmation_k": [4, 6, 8],
            "indexed_positive_control_seed_split": [20, 10],
            "indexed_positive_control_selection_hid_attention_generation_and_confirmation": True,
            "indexed_positive_control_visible_progress_confound": True,
            "indexed_positive_control_automatic_discovery_layer": {
                model: int(indexed_progress_selected[model]["selected_layer"])
                for model in MODELS
            },
            "indexed_positive_control_active_confirmation_layer": indexed_confirmation_layers,
            "indexed_positive_control_preconfirmation_protocol_amendment": True,
            "indexed_positive_control_midlayer_anchor_is_external": True,
            "gemma_prompt_conditioned_noindex_n": 10,
            "gemma_prompt_conditioned_noindex_seed_split": [20, 10],
            "gemma_prompt_conditioned_noindex_layer": 16,
            "gemma_prompt_conditioned_noindex_k": [4, 6, 8],
            "gemma_prompt_conditioned_noindex_direction": "forward_only",
            "gemma_prompt_conditioned_noindex_marker_contains_count": False,
            "gemma_prompt_conditioned_noindex_selection_patch_outcome_blind": True,
            "answer_query_layer_sweep_models": list(MODELS),
            "answer_query_layer_sweep_seed_count": 10,
            "answer_query_layer_sweep_seed_range": [1254, 1263],
            "answer_query_layer_sweep_pair_counts": ANSWER_QUERY_PAIR_COUNT,
            "answer_query_layer_sweep_layer_grids": ANSWER_QUERY_LAYER_GRID,
            "answer_query_layer_sweep_conditions": [
                "self_patch",
                "full_donor_patch",
            ],
            "answer_query_layer_sweep_pair_selection_uses_patch_outcome": False,
            "answer_query_layer_sweep_is_pristine_new_confirmation": False,
            "answer_query_layer_sweep_status": (
                "registered_existing_split_extension"
            ),
            "answer_query_site_absolute_position_match_required": False,
            "terminal_relay_common_support_pairs_per_model": 86,
            "terminal_relay_common_support_seed_count": 10,
            "terminal_relay_matching_keys": [
                "seed",
                "gold_count",
                "donor_offset",
            ],
            "terminal_relay_selection_uses_outcomes": False,
            "terminal_relay_gemma_suffix_and_query_estimands_distinguishable": False,
        },
        "claim_scope": {
            "distributed_content_bound_event_progress_state_supported": True,
            "midlayer_event_state_controls_next_retrieval_supported": True,
            "narrow_event_tail_counter_like_routing_supported": True,
            "single_endpoint_sufficient": False,
            "memoryless_arithmetic_plus_one_recurrence_supported": False,
            "content_free_counter_component_isolated": False,
            "qwen_no_index_scope_result_extrapolated_to_gemma": False,
            "functional_progress_controller_supported": True,
            "exclusive_circuit_claimed": False,
            "natural_end_to_end_single_state_sufficiency": False,
            "single_seed_walkthrough_inferential": False,
            "gemma_commit_to_query_direct_effect_confirmed": True,
            "gemma_commit_to_query_local_specificity_qualified": True,
            "gemma_commit_to_query_orthogonal_interval_positive": True,
            "commit_query_orthogonal_control_matches_count_projection_norm_not_full_delta_norm": True,
            "gemma_narrow_pre_o_query_mediation_confirmed": False,
            "qwen_free_running_terminal_restoration_confirmed": False,
            "ncc_frozen_results_independently_reproduced": True,
            "ncc_timing_stratified_recapture_complete": True,
            "ncc_city_to_rank_marker_tokens_excluded": True,
            "ncc_bank_matched_timing_raw_direction_corresponds_across_models": True,
            "ncc_cross_model_effect_size_comparison_allowed": False,
            "ncc_model_by_mask_interaction_confirmed": False,
            "qwen_historical_ncc_pooled_directionally_supported": True,
            "qwen_stratified_rank_to_city_readout_validity_pass": False,
            "gemma_ncc_pooled_directionally_supported": False,
            "gemma_stratified_rank_to_city_directional_specific_only": True,
            "ncc_marker_free_city_to_rank_damage_supported": False,
            "ncc_readout_validity_gate_post_analysis": True,
            "qwen_full_bank_to_late_ncc_confirmed": False,
            "gemma_full_bank_to_late_ncc_confirmed": False,
            "gemma_l17_rank_before_midlayer_damage_exploratory": True,
            "direct_count_output_margin_complete": True,
            "direct_margin_all_clean_readouts_valid": True,
            "qwen_direct_margin_directional_specific_supported": False,
            "gemma_direct_margin_both_timings_directional_specific": True,
            "gemma_direct_margin_interval_confirmed": True,
            "direct_margin_confirmation_pristine_prospective": False,
            "direct_margin_model_by_mask_interaction_tested": False,
            "no_running_index_count_signal_confirmed": True,
            "legacy_no_running_index_early_stop_single_span_strong_sufficiency": False,
            "qwen_natural_no_index_item_span_candidate_argmax": "60/60",
            "qwen_natural_no_index_item_span_first_city_transfer": "43/60",
            "qwen_natural_no_index_l16_first_city_transfer": "16/20",
            "qwen_natural_no_index_item_span_generation_audited": True,
            "indexed_positive_control_complete": True,
            "indexed_positive_control_supports_no_index_internal_counter": False,
            "qwen_indexed_positive_control_strong": True,
            "gemma_indexed_positive_control_directional_partial": True,
            "gemma_indexed_positive_control_bidirectional_replication": False,
            "explicit_index_uniformly_stronger_across_models": False,
            "qwen_indexed_positive_control_first_city_transfer": f"{qwen_indexed_transfer}/60",
            "qwen_indexed_positive_control_first_city_paired_gain": float(
                indexed_progress_summary["Qwen3-8B"][
                    "paired_first_known_city_donor_adoption_gain"
                ]
            ),
            "gemma_indexed_positive_control_first_city_transfer": (
                f"{int(indexed_progress_summary['Gemma4-E4B']['patched_first_known_city_donor_adoption_count'])}/60"
            ),
            "gemma_indexed_positive_control_first_city_paired_gain": float(
                indexed_progress_summary["Gemma4-E4B"][
                    "paired_first_known_city_donor_adoption_gain"
                ]
            ),
            "qwen_indexed_positive_control_seed_any_incremental_transfer": (
                f"{int(indexed_progress_generation_audit['models']['Qwen3-8B']['seed_with_any_incremental_adoption_count'])}/10"
            ),
            "gemma_indexed_positive_control_seed_any_incremental_transfer": (
                f"{int(indexed_progress_generation_audit['models']['Gemma4-E4B']['seed_with_any_incremental_adoption_count'])}/10"
            ),
            "indexed_positive_control_all_adoptions_within_first_80_chars": True,
            "gemma_natural_no_index_causal_result_available": False,
            "gemma_next_item_routing_status": (
                "simulatively confirmed under auxiliary settings"
            ),
            "gemma_simulative_support_sources": [
                "prompt_conditioned_no_visible_index",
                "explicit_index",
            ],
            "gemma_prompt_conditioned_no_index_auxiliary_complete": True,
            "gemma_prompt_conditioned_no_index_candidate_argmax": "30/30",
            "gemma_prompt_conditioned_no_index_first_city_transfer": "22/30",
            "gemma_prompt_conditioned_no_index_attention_positive": "12/30",
            "gemma_prompt_conditioned_supports_natural_no_index_claim": False,
            "gemma_prompt_conditioned_supports_shared_head_circuit_claim": False,
            "all_layer_pca3_is_descriptive": True,
            "qwen_answer_query_full_state_executable": True,
            "gemma_answer_query_full_state_executable": True,
            "answer_query_full_state_implies_low_dimensional_counter": False,
            "trace_to_answer_ordered_partial_path_supported": True,
            "trace_to_answer_same_trial_mediation_common_support": True,
            "trace_to_answer_complete_or_exclusive_mediation_supported": False,
            "same_trial_trace_to_answer_serial_mediation_complete": False,
        },
        "derived_display_data_sha256": {
            "geometry_3d": hashlib.sha256(
                json.dumps(geometry_3d, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "patch_scope_confirmation": hashlib.sha256(
                json.dumps(
                    patch_scope_confirmation, sort_keys=True
                ).encode("utf-8")
            ).hexdigest(),
            "indexed_progress_discovery": hashlib.sha256(
                json.dumps(indexed_progress_discovery, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "indexed_progress_confirmation": hashlib.sha256(
                json.dumps(indexed_progress_confirmation, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "indexed_progress_generation_audit": hashlib.sha256(
                json.dumps(
                    indexed_progress_generation_audit, sort_keys=True
                ).encode("utf-8")
            ).hexdigest(),
            "gemma_prompt_conditioned_noindex": hashlib.sha256(
                json.dumps(gemma_prompt_noindex, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "answer_query_layer_effects": hashlib.sha256(
                json.dumps(answer_query_layer_rows, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        },
        "inputs_sha256": {str(path): sha256(path) for path in input_paths},
    }
    return html_text, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, default=Path("reports/NiaH_Non-thinking_report.html"))
    parser.add_argument(
        "--sample-alignment-audit",
        type=Path,
        default=Path(
            "work/v5_native_sample_aligned_20260829/alignment_audit.json"
        ),
    )
    parser.add_argument(
        "--output-alignment-audit",
        type=Path,
        default=Path(
            "work/v5_native_sample_aligned_20260829/output_alignment_audit.json"
        ),
    )
    parser.add_argument(
        "--aligned-run-root",
        type=Path,
        default=Path("work/v5_native_sample_aligned_20260829/runs"),
    )
    parser.add_argument(
        "--aligned-grammar-panel-root",
        type=Path,
        default=Path(
            "work/v5_native_sample_aligned_20260829/shared_grammar_panel_v2"
        ),
    )
    parser.add_argument(
        "--aligned-routed-panel-root",
        type=Path,
        default=Path(
            "work/v5_native_sample_aligned_20260829/shared_routed_transition_panel"
        ),
    )
    parser.add_argument(
        "--patch-scope-layer-sweep",
        type=Path,
        default=Path(
            "work/same_site_progress_transplant_20260827/"
            "n10_patch_scope_layer_sweep_v2/layer_sweep_analysis.json"
        ),
    )
    parser.add_argument(
        "--patch-scope-layer-plot",
        type=Path,
        default=Path(
            "work/same_site_progress_transplant_20260827/"
            "n10_patch_scope_layer_sweep_v2/layer_sweep_effect_sizes.svg"
        ),
    )
    parser.add_argument(
        "--patch-scope-frozen-confirmation",
        type=Path,
        default=Path(
            "work/same_site_progress_transplant_20260827/"
            "n10_patch_scope_frozen_v2/frozen_scope_analysis.json"
        ),
    )
    parser.add_argument(
        "--patch-scope-generation-audit",
        type=Path,
        default=Path(
            "work/same_site_progress_transplant_20260827/"
            "n10_patch_scope_frozen_v2/item_span_generation_manual_audit.json"
        ),
    )
    parser.add_argument(
        "--item-span-l16",
        type=Path,
        default=Path(
            "work/same_site_progress_transplant_20260827/"
            "n10_item_span_contextual_l16_v1/frozen_scope_analysis.json"
        ),
    )
    parser.add_argument(
        "--item-span-l16-generation-audit",
        type=Path,
        default=Path(
            "work/same_site_progress_transplant_20260827/"
            "n10_item_span_contextual_l16_v1/item_span_generation_manual_audit.json"
        ),
    )
    parser.add_argument(
        "--historical-event-tail-confirmation",
        type=Path,
        default=Path(
            "work/same_site_progress_transplant_20260827/"
            "n10_natural_crossk_attention_generation_v1/"
            "confirmation10_frozen_crossk_analysis.json"
        ),
    )
    parser.add_argument(
        "--indexed-progress-cohort-manifest",
        type=Path,
        default=Path(
            "work/v5_native_sample_aligned_20260829/"
            "indexed_progress_control_panel/alignment_manifest.json"
        ),
    )
    parser.add_argument(
        "--indexed-progress-freeze-manifest",
        type=Path,
        default=Path(
            "work/v5_native_sample_aligned_20260829/"
            "indexed_progress_control_freeze.json"
        ),
    )
    parser.add_argument(
        "--indexed-progress-discovery-root",
        type=Path,
        default=Path(
            "work/v5_native_sample_aligned_20260829/runs"
        ),
    )
    parser.add_argument(
        "--indexed-progress-confirmation-root",
        type=Path,
        default=Path(
            "work/v5_native_sample_aligned_20260829/runs"
        ),
    )
    parser.add_argument(
        "--indexed-progress-generation-audit",
        type=Path,
        default=Path(
            "work/v5_native_sample_aligned_20260829/"
            "indexed_progress_generation_audit.json"
        ),
    )
    parser.add_argument(
        "--gemma-prompt-conditioned-noindex-cohort-manifest",
        type=Path,
        default=Path(
            "work/gemma_prompt_conditioned_noindex_20260827/"
            "cohort_full_20_10/manifest.json"
        ),
    )
    parser.add_argument(
        "--gemma-prompt-conditioned-noindex-analysis",
        type=Path,
        default=Path(
            "work/gemma_prompt_conditioned_noindex_20260827/"
            "forward_l16_item_span/analysis.json"
        ),
    )
    parser.add_argument("--qwen-targeted-analysis", type=Path, default=Path("reports/v5_native_final_localizers/analysis/qwen_final_merged_dose_grid.json"))
    parser.add_argument("--gemma-targeted-analysis", type=Path, default=Path("reports/v5_native_hybrid_supplement/Gemma4-E4B/analysis_hybrid_supplement_registered_v1/hybrid_dose_grid_complete.json"))
    parser.add_argument(
        "--representation-root",
        type=Path,
        default=Path("reports/v5_native_cross_model_aligned_geometry"),
    )
    parser.add_argument(
        "--running-pca3-coordinates",
        type=Path,
        default=Path(
            "reports/v5_native_cross_model_aligned_geometry/"
            "pca3_confirmation_coordinates.json"
        ),
    )
    parser.add_argument("--dual-endpoint-root", type=Path, default=Path("reports/v5_dual_endpoint_geometry_full300"))
    parser.add_argument("--band-diagnostic-root", type=Path, default=Path("reports/native_geometry_band_diagnostic_full300"))
    parser.add_argument("--geometry-comparison-report", type=Path, default=Path("reports/NiaH_Geometry_Comparison.html"))
    parser.add_argument("--atlas-root", type=Path, default=Path("reports/v5_native_p0_head_atlas"))
    parser.add_argument("--token-ablation-root", type=Path, default=Path("reports/v5_native_token_level_ablation"))
    parser.add_argument("--snapshot-root", type=Path, default=Path("work_remote_snapshots"))
    parser.add_argument(
        "--ncc-supplement-root",
        type=Path,
        default=Path("work_remote_snapshots/ncc_unnumbered_supplement_20260823"),
    )
    parser.add_argument(
        "--stratified-ncc-root",
        type=Path,
        default=Path("work_remote_snapshots/stratified_ncc_20260823"),
    )
    parser.add_argument(
        "--logit-margin-root",
        type=Path,
        default=Path("work_remote_snapshots/targeted_logit_margin_20260823_v2"),
    )
    parser.add_argument(
        "--answer-query-layer-root",
        type=Path,
        default=Path("work/v5_native_sample_aligned_20260829/runs"),
    )
    parser.add_argument(
        "--answer-query-plan-root",
        type=Path,
        default=Path(
            "work/v5_native_sample_aligned_20260829/"
            "answer_query_layer_sweep_plan"
        ),
    )
    parser.add_argument(
        "--terminal-relay-common-support",
        type=Path,
        default=Path(
            "work/relay_bridge_export_20260829/"
            "common_support_confirmation_rerun"
        ),
    )
    parser.add_argument(
        "--historical-causal-chain-statistics",
        type=Path,
        default=Path(
            "reports/v5_native_thinking_causal_chain_v2/"
            "combined_statistics_cross_model_holm.csv"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/NiaH_Native-Thinking_report.html"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/v5_native_final_localizers/report_manifest_restructured.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report, manifest = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    manifest["output_sha256"] = sha256(args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "sha256": manifest["output_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
