from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_realistic_niah_v4_representation_report as full


FOCUS_VARIANT = "v4.4"
MODELS = full.MODELS

FIRST_LOCATOR_RELATIVE_ROOT = Path(
    "reports/v4_non-thinking_causal/v4_4_extension/first_span_locator"
)
FIRST_LOCATOR_MODELS = ("Qwen3-8B", "Gemma4-E4B")
FIRST_LOCATOR_COLORS = {"Qwen3-8B": "#6750E8", "Gemma4-E4B": "#00A88F"}
FIRST_LOCATOR_K = (1, 2, 4, 8, 16, 32)
FIRST_LOCATOR_ENDPOINTS = (
    "all_absolute_error_increase",
    "correct_to_wrong_rate",
)


def _pct(value: Any, digits: int = 1) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{100 * float(value):.{digits}f}%"


def _num(value: Any, digits: int = 3, *, signed: bool = False) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    spec = f"+.{digits}f" if signed else f".{digits}f"
    return format(float(value), spec)


def _ci(row: dict[str, Any], center: str, low: str, high: str, digits: int = 3) -> str:
    return (
        f"{_num(row[center], digits, signed=True)} "
        f"[{_num(row[low], digits, signed=True)}, "
        f"{_num(row[high], digits, signed=True)}]"
    )


def _details_table(
    *,
    title: str,
    columns: list[str],
    rows: list[list[str]],
    open_by_default: bool = False,
) -> str:
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    opened = " open" if open_by_default else ""
    return (
        f'<details class="data-table"{opened}>'
        f"<summary>{html.escape(title)} · {len(rows)} rows</summary>"
        '<div class="table-scroll"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div></details>"
    )


def _inventory_table() -> str:
    rows = [
        [
            "Head ablation",
            "同一个 V4.4 high-count prompt",
            "在最终 Total: query 处，将 discovery-ranked span-end top-4 / top-8 heads 的 pre-o_proj slice 置零",
            "置零相同数量、逐层数量匹配的 deterministic random heads",
            "(ranked−clean) − (random−clean) 的生成 count shift；并报告 prediction-changed 与 MAE",
        ],
        [
            "Needle-end residual patch",
            "同 model、seed、V4.4 的 nested count pair；receiver 与 donor 只在已知 slot 的 active/inactive 内容上不同",
            "把 donor toggled-slot 最后 token 的 post-block residual 搬到 receiver 对应 token，并从 start layer 一直 clamp 到末层",
            "clean receiver；本 screen 没有 matched-random residual control",
            "direction-aligned greedy count shift 与是否更接近 donor gold",
        ],
        [
            "Answer-query residual patch",
            "同 model、seed、V4.4 的 5↔6、7↔8、9↔10、5↔10 directed pair",
            "在某一层只替换 receiver 的最终 Total: query residual：h_receiver(q,l) ← h_donor(q,l)",
            "clean receiver；跨层 onset 以 L0 为 reference",
            "当 clean receiver prediction ≠ donor prediction 时，patched output 是否采用 donor prediction",
        ],
        [
            "Steering v1",
            "V4.4 confirmation receiver 与指定 target count",
            "单层 full-dimensional residual-preserving delta：h′=h+α(μ_target−μ_receiver)，α=1",
            "同一 prompt / layer 上与 delta 等范数且正交的随机方向",
            "geometric−random 的 direction-aligned greedy count shift 与 moved/target-hit effect",
        ],
        [
            "Steering v2",
            "held-out V4.4 confirmation receiver；single 和 multi plans 先由 disjoint discovery screen 锁定",
            "锁定 single layer 或多层 layer set 上施加 full-dimensional centroid delta",
            "每层等范数正交随机方向；invalid-as-failure",
            "V4.4 panel 内 geometric−random 的 aligned shift、moved 与 exact-target effects",
        ],
    ]
    return _details_table(
        title="完成的 intervention：source / receiver / control / estimand",
        columns=["实验", "receiver / 配对单位", "实验条件", "对照条件", "主要比较量"],
        rows=rows,
        open_by_default=True,
    )


def _variant_only_frames(
    frames: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, dict[str, pd.DataFrame]]:
    filtered: dict[str, dict[str, pd.DataFrame]] = {}
    for model, stages in frames.items():
        filtered[model] = {}
        for stage, frame in stages.items():
            if "design_variant" in frame.columns:
                selected = frame[
                    frame["design_variant"].astype(str) == FOCUS_VARIANT
                ].copy()
                if selected.empty:
                    raise RuntimeError(f"{model}/{stage}: no {FOCUS_VARIANT} rows")
                filtered[model][stage] = selected
            else:
                filtered[model][stage] = frame.copy()
    return filtered


def _filter_projection_rows(
    projections: dict[str, dict[str, Any]], *, row_variant_index: int = 0
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, value in projections.items():
        selected = dict(value)
        selected["rows"] = [
            row for row in value["rows"] if str(row[row_variant_index]) == FOCUS_VARIANT
        ]
        if not selected["rows"]:
            raise RuntimeError(f"{key}: no {FOCUS_VARIANT} projection rows")
        result[key] = selected
    return result


def _prompt_payload(projections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in projections.items():
        if value["pooling"] != "span_end":
            continue
        item = dict(value)
        item["rows"] = [
            [row[1], row[2], row[3], row[4], row[5], row[6], *row[7:]]
            for row in value["rows"]
        ]
        payload[f"{value['model']}|{value['layer']}"] = item
    return payload


def _answer_payload(projections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in projections.items():
        item = dict(value)
        item["rows"] = [
            [row[1], "saved", row[3], row[4], row[5], row[2], *row[6:]]
            for row in value["rows"]
        ]
        payload[key] = item
    return payload


def _joint_payload(projections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in projections.items():
        if value["pooling"] != "span_end" or value["mode"] != "role_centered":
            continue
        selected = dict(value)
        selected["rows"] = [row for row in value["rows"] if row[0] == FOCUS_VARIANT]
        if selected["rows"]:
            result[f"{value['model']}|{value['layer']}"] = selected
    return result


def _prompt_selection_rows(layer_rows: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for model in MODELS:
        selected = [
            row
            for row in layer_rows
            if row["model"] == model
            and row["pooling"] == "span_end"
            and (row["probe_optimal"] or row["manifold_display"])
        ]
        for row in sorted(selected, key=lambda item: int(item["layer"])):
            role = "/".join(
                label
                for label, flag in (
                    ("probe-optimal", row["probe_optimal"]),
                    ("manifold-display", row["manifold_display"]),
                )
                if flag
            )
            rows.append(
                [
                    html.escape(model),
                    f"L{int(row['layer'])}",
                    role,
                    _num(row["full_space_discovery_cv_r2"]),
                    _num(row["pca_evr_pc1_3"]),
                    _num(row["count_signal_capture_pc1_3"]),
                    _num(row["discovery_compactness"]),
                ]
            )
    return rows


def _answer_selection_rows(projections: dict[str, dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for model in MODELS:
        model_items = [
            item
            for item in projections.values()
            if item["model"] == model and item["fit_cohort"] == "all"
        ]
        for item in sorted(model_items, key=lambda entry: int(entry["layer"])):
            if not (item.get("probe_optimal") or item.get("manifold_display")):
                continue
            role = "/".join(
                label
                for label, flag in (
                    ("probe-optimal", item.get("probe_optimal")),
                    ("manifold-display", item.get("manifold_display")),
                )
                if flag
            )
            rows.append(
                [
                    html.escape(model),
                    f"L{int(item['layer'])}",
                    role,
                    _num(sum(item["explained_variance_ratio"][:3])),
                    _num(item["pca3_discovery_cv_r2"]),
                    _num(item["count_signal_capture_pc1_3"]),
                    _num(item["discovery_compactness"]),
                ]
            )
    return rows


def _behavior_rows(labels: dict[str, pd.DataFrame]) -> list[list[str]]:
    rows: list[list[str]] = []
    for model in MODELS:
        selected = labels[model][
            (labels[model]["split"].astype(str) == "confirmation")
            & (labels[model]["design_variant"].astype(str) == FOCUS_VARIANT)
        ].copy()
        selected["is_correct_bool"] = selected["is_correct"].map(full._bool)
        for count, frame in selected.groupby("gold_count", sort=True):
            prediction = pd.to_numeric(frame["parsed_count"], errors="coerce")
            error = prediction - int(count)
            rows.append(
                [
                    html.escape(model),
                    str(int(count)),
                    f"{int(frame['is_correct_bool'].sum())}/{len(frame)}",
                    _pct(frame["is_correct_bool"].mean()),
                    _num(prediction.mean(), 2),
                    _pct((error < 0).mean()),
                ]
            )
    return rows


def _phenotype_rows(phenotypes: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for model in MODELS:
        selected = [row for row in phenotypes if row["model"] == model]
        counts = pd.Series([row["phenotype"] for row in selected]).value_counts()
        for phenotype, count in counts.sort_index().items():
            candidates = sorted(
                [row for row in selected if row["phenotype"] == phenotype],
                key=lambda item: (
                    10_000
                    if item["candidate_rank"] is None
                    or pd.isna(item["candidate_rank"])
                    else int(item["candidate_rank"]),
                    int(item["layer"]),
                    int(item["head"]),
                ),
            )
            example = candidates[0]
            rows.append(
                [
                    html.escape(model),
                    html.escape(str(phenotype)),
                    str(int(count)),
                    f"L{int(example['layer'])}H{int(example['head'])}",
                    _num(example["effective_number_mean"], 2),
                    _num(example["dominant_occurrence_mean_share"], 3),
                ]
            )
    return rows


def _first_locator_paths(repo_root: Path) -> dict[str, Path]:
    root = repo_root / FIRST_LOCATOR_RELATIVE_ROOT
    return {
        "qwen_scores": root / "Qwen3-8B.all_head_first_span_scores.csv",
        "gemma_scores": root / "Gemma4-E4B.all_head_first_span_scores.csv",
        "layer_matched": root / "first_span_ablation_statistics.csv",
        "m10_matched": root / "first_span_M10_matched_statistics.csv",
        "matching": root / "first_span_M10_matching_summary.csv",
    }


def _load_first_locator_data(
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = _first_locator_paths(repo_root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing first-locator appendix inputs: {missing}")
    scores = pd.concat(
        [pd.read_csv(paths["qwen_scores"]), pd.read_csv(paths["gemma_scores"])],
        ignore_index=True,
    )
    layer_matched = pd.read_csv(paths["layer_matched"])
    m10_matched = pd.read_csv(paths["m10_matched"])
    matching = pd.read_csv(paths["matching"])

    if set(scores["model"].astype(str)) != set(FIRST_LOCATOR_MODELS):
        raise RuntimeError("First-locator score table has unexpected models")
    for model in FIRST_LOCATOR_MODELS:
        model_scores = scores[scores["model"].astype(str) == model]
        if int((model_scores["first_span_absolute_mass_rank"] == 1).sum()) != 1:
            raise RuntimeError(f"Expected one rank-1 first-span head for {model}")
        top = model_scores.loc[
            model_scores["first_span_absolute_mass_rank"] == 1
        ].iloc[0]
        profile = np.array(
            [float(top[f"span_{index}_mass"]) for index in range(1, 11)]
        )
        if not np.isclose(profile.sum(), float(top["ten_span_total_mass_mean"]), atol=1e-10):
            raise RuntimeError(f"Ten-span mass profile does not sum to M10 for {model}")
        if not np.isclose(profile[0], float(top["first_span_mass_mean"]), atol=1e-12):
            raise RuntimeError(f"First-span mass disagrees with profile for {model}")

    expected = {
        (model, k, endpoint)
        for model in FIRST_LOCATOR_MODELS
        for k in FIRST_LOCATOR_K
        for endpoint in FIRST_LOCATOR_ENDPOINTS
    }
    for label, frame in (("layer-matched", layer_matched), ("M10-matched", m10_matched)):
        observed = {
            (str(row.model), int(row.top_k), str(row.endpoint))
            for row in frame.itertuples(index=False)
        }
        if observed != expected:
            raise RuntimeError(f"Incomplete {label} first-locator statistics")
    return scores, layer_matched, m10_matched, matching


def _mix_hex(low: str, high: str, fraction: float) -> str:
    fraction = min(1.0, max(0.0, float(fraction)))
    low_rgb = tuple(int(low[index : index + 2], 16) for index in (1, 3, 5))
    high_rgb = tuple(int(high[index : index + 2], 16) for index in (1, 3, 5))
    mixed = tuple(
        round(start + fraction * (end - start))
        for start, end in zip(low_rgb, high_rgb)
    )
    return "#" + "".join(f"{value:02X}" for value in mixed)


def _first_locator_heatmap_svg(
    scores: pd.DataFrame,
    *,
    metric: str,
    title: str,
    description: str,
    relative: bool,
) -> str:
    width, plot_width = 1080, 900
    left, top = 105.0, 78.0
    panel_specs: list[tuple[str, float, float]] = []
    cursor = top
    for model in FIRST_LOCATOR_MODELS:
        head_count = int(scores.loc[scores["model"] == model, "head"].max()) + 1
        cell_height = 8.5 if head_count > 16 else 14.0
        panel_height = head_count * cell_height
        panel_specs.append((model, cursor, cell_height))
        cursor += panel_height + 82
    height = int(cursor + 52)
    maximum = float(scores[metric].max())
    svg_id = "first-locator-share-map" if relative else "first-locator-mass-map"
    parts = [
        f'<svg class="stat-svg first-locator-map" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{svg_id}-title {svg_id}-desc">',
        f'<title id="{svg_id}-title">{html.escape(title)}</title>',
        f'<desc id="{svg_id}-desc">{html.escape(description)}</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFDF8"/>',
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#20242D">',
        f'<text x="{width/2:.1f}" y="27" text-anchor="middle" font-size="17" font-weight="700">{html.escape(title)}</text>',
    ]
    low, high = ("#F3EEE4", "#00A88F") if relative else ("#F3EEE4", "#6750E8")
    legend_x, legend_y, legend_w = 630.0, 43.0, 300.0
    for index in range(100):
        parts.append(
            f'<rect x="{legend_x + index * legend_w / 100:.2f}" y="{legend_y}" '
            f'width="{legend_w / 100 + 0.2:.2f}" height="12" fill="{_mix_hex(low, high, index / 99)}"/>'
        )
    legend_low = "0%" if relative else "0"
    legend_high = f"{100 * maximum:.1f}%" if relative else f"{maximum:.3f}"
    scale_note = "linear" if relative else "log1p color scale"
    parts.extend(
        [
            f'<text x="{legend_x - 8}" y="{legend_y + 11}" text-anchor="end" font-size="10" fill="#5E6672">{legend_low}</text>',
            f'<text x="{legend_x + legend_w + 8}" y="{legend_y + 11}" font-size="10" fill="#5E6672">{legend_high}</text>',
            f'<text x="{legend_x - 28}" y="{legend_y + 11}" text-anchor="end" font-size="10" fill="#5E6672">{scale_note}</text>',
        ]
    )
    for model, y0, cell_height in panel_specs:
        subset = scores[scores["model"] == model].copy()
        layer_count = int(subset["layer"].max()) + 1
        head_count = int(subset["head"].max()) + 1
        cell_width = plot_width / layer_count
        top_row = subset.loc[subset["first_span_absolute_mass_rank"] == 1].iloc[0]
        top_label = f'L{int(top_row["layer"])}H{int(top_row["head"])}'
        parts.append(
            f'<text x="{left}" y="{y0 - 14:.1f}" font-size="14" font-weight="700">'
            f'{html.escape(model)} · rank-1 {top_label}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{y0:.1f}" width="{plot_width}" height="{head_count * cell_height:.1f}" '
            'fill="#F3EEE4" stroke="#C9C2B6"/>'
        )
        for row in subset.itertuples(index=False):
            value = float(getattr(row, metric))
            if relative:
                color_fraction = value
            else:
                color_fraction = np.log1p(value / 1e-6) / np.log1p(maximum / 1e-6)
            x = left + int(row.layer) * cell_width
            y = y0 + int(row.head) * cell_height
            is_top = int(row.first_span_absolute_mass_rank) == 1
            stroke = "#D6B52C" if is_top else "#FFFDF8"
            stroke_width = 2.5 if is_top else 0.25
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_width + 0.15:.2f}" '
                f'height="{cell_height + 0.15:.2f}" fill="{_mix_hex(low, high, color_fraction)}" '
                f'stroke="{stroke}" stroke-width="{stroke_width}">'
                f'<title>{html.escape(model)} L{int(row.layer)}H{int(row.head)}; '
                f'm1={float(row.first_span_mass_mean):.9f}; M10={float(row.ten_span_total_mass_mean):.9f}; '
                f'm1/M10={100 * float(row.first_share_ratio_of_means):.3f}%</title></rect>'
            )
        layer_ticks = sorted(set([0, layer_count - 1, *range(5, layer_count, 5)]))
        for layer in layer_ticks:
            x = left + (layer + 0.5) * cell_width
            parts.append(
                f'<text x="{x:.1f}" y="{y0 + head_count * cell_height + 18:.1f}" '
                f'text-anchor="middle" font-size="9" fill="#5E6672">L{layer}</text>'
            )
        head_ticks = sorted(set([0, head_count - 1, *range(4, head_count, 4)]))
        for head in head_ticks:
            y = y0 + (head + 0.5) * cell_height + 3
            parts.append(
                f'<text x="{left - 9}" y="{y:.1f}" text-anchor="end" font-size="9" fill="#5E6672">H{head}</text>'
            )
        parts.append(
            f'<text x="{left + plot_width / 2:.1f}" y="{y0 + head_count * cell_height + 38:.1f}" '
            'text-anchor="middle" font-size="11" fill="#5E6672">decoder layer</text>'
        )
    parts.append('</g></svg>')
    return "".join(parts)


def _first_span_profile_svg(scores: pd.DataFrame) -> str:
    width, height = 1040, 640
    left, right, panel_height = 105.0, 35.0, 235.0
    plot_width = width - left - right
    parts = [
        f'<svg class="stat-svg first-span-profile-svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="first-span-profile-title first-span-profile-desc">',
        '<title id="first-span-profile-title">Absolute attention mass across the ten complete needle spans</title>',
        '<desc id="first-span-profile-desc">Separate model panels show the rank-one first-span head. Bar heights are absolute attention mass; exact absolute and relative values are provided in the following table.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFDF8"/>',
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#20242D">',
        f'<text x="{width/2:.1f}" y="27" text-anchor="middle" font-size="17" font-weight="700">Rank-1 complete-first-span heads · absolute mass</text>',
    ]
    for panel_index, model in enumerate(FIRST_LOCATOR_MODELS):
        row = scores[
            (scores["model"] == model)
            & (scores["first_span_absolute_mass_rank"] == 1)
        ].iloc[0]
        y0 = 72.0 + panel_index * 280.0
        plot_bottom = y0 + panel_height
        masses = [float(row[f"span_{index}_mass"]) for index in range(1, 11)]
        m10 = float(row["ten_span_total_mass_mean"])
        ymax = max(masses) * 1.16
        head = f'L{int(row["layer"])}H{int(row["head"])}'
        color = FIRST_LOCATOR_COLORS[model]
        parts.append(
            f'<text x="{left}" y="{y0 - 17:.1f}" font-size="14" font-weight="700">'
            f'{html.escape(model)} · {head} · M10={m10:.6f} · share1={100 * masses[0] / m10:.2f}%</text>'
        )
        for tick in np.linspace(0.0, ymax, 5):
            y = plot_bottom - tick / ymax * panel_height
            parts.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#DED8CE"/>'
            )
            parts.append(
                f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="10" fill="#5E6672">{tick:.3f}</text>'
            )
        step = plot_width / 10
        for index, mass in enumerate(masses, start=1):
            x = left + (index - 1) * step + step * 0.18
            bar_width = step * 0.64
            bar_height = mass / ymax * panel_height
            y = plot_bottom - bar_height
            share = mass / m10
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
                f'fill="{color}" opacity="{1.0 if index == 1 else 0.78}"><title>span {index}: '
                f'mass={mass:.9f}; share of M10={100 * share:.3f}%</title></rect>'
            )
            label_y = max(y - 7, y0 + 9)
            parts.append(
                f'<text x="{x+bar_width/2:.1f}" y="{label_y:.1f}" text-anchor="middle" font-size="9" fill="#38414D">'
                f'{mass:.4f}</text>'
            )
            parts.append(
                f'<text x="{x+bar_width/2:.1f}" y="{plot_bottom+17:.1f}" text-anchor="middle" font-size="10" fill="#5E6672">S{index}</text>'
            )
        parts.append(
            f'<text transform="translate(20 {y0 + panel_height/2:.1f}) rotate(-90)" text-anchor="middle" font-size="11" fill="#5E6672">absolute attention mass</text>'
        )
    parts.append('</g></svg>')
    return "".join(parts)


def _first_locator_ablation_svg(
    frame: pd.DataFrame,
    *,
    control_label: str,
    svg_id: str,
) -> str:
    width, height = 1120, 690
    left, top, panel_w, panel_h = 88.0, 72.0, 430.0, 225.0
    col_gap, row_gap = 115.0, 90.0
    endpoint_titles = {
        "all_absolute_error_increase": "All-sample absolute-error increase",
        "correct_to_wrong_rate": "Clean-correct → wrong rate",
    }
    ranges: dict[str, tuple[float, float]] = {}
    for endpoint in FIRST_LOCATOR_ENDPOINTS:
        subset = frame[frame["endpoint"] == endpoint]
        low = min(0.0, float(subset["ci95_low"].min()))
        high = max(0.0, float(subset["ci95_high"].max()))
        padding = max((high - low) * 0.12, 0.02)
        ranges[endpoint] = (low - padding, high + padding)
    parts = [
        f'<svg class="stat-svg first-locator-ablation-svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{svg_id}-title {svg_id}-desc">',
        f'<title id="{svg_id}-title">First-span top-k ablation versus {html.escape(control_label)}</title>',
        f'<desc id="{svg_id}-desc">Four panels show ranked-minus-control effects and seed-bootstrap 95 percent confidence intervals over K equals 1, 2, 4, 8, 16 and 32. Positive values mean the first-span-ranked bank is more damaging.</desc>',
        f'<rect width="{width}" height="{height}" fill="#FFFDF8"/>',
        '<g font-family="Segoe UI,Arial,sans-serif" fill="#20242D">',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-size="17" font-weight="700">Top-k specificity · ranked minus {html.escape(control_label)}</text>',
    ]
    for row_index, model in enumerate(FIRST_LOCATOR_MODELS):
        for col_index, endpoint in enumerate(FIRST_LOCATOR_ENDPOINTS):
            x0 = left + col_index * (panel_w + col_gap)
            y0 = top + row_index * (panel_h + row_gap)
            ymin, ymax = ranges[endpoint]
            x = lambda index: x0 + index / (len(FIRST_LOCATOR_K) - 1) * panel_w
            y = lambda value: y0 + (ymax - float(value)) / (ymax - ymin) * panel_h
            subset = frame[
                (frame["model"] == model) & (frame["endpoint"] == endpoint)
            ].sort_values("top_k")
            parts.append(
                f'<text x="{x0 + panel_w/2:.1f}" y="{y0 - 15:.1f}" text-anchor="middle" font-size="13" font-weight="700">'
                f'{html.escape(model)} · {html.escape(endpoint_titles[endpoint])}</text>'
            )
            for tick in np.linspace(ymin, ymax, 5):
                yy = y(tick)
                parts.append(
                    f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+panel_w}" y2="{yy:.1f}" stroke="#DED8CE"/>'
                )
                parts.append(
                    f'<text x="{x0-9}" y="{yy+4:.1f}" text-anchor="end" font-size="9" fill="#5E6672">{tick:.2f}</text>'
                )
            zero_y = y(0.0)
            parts.append(
                f'<line x1="{x0}" y1="{zero_y:.1f}" x2="{x0+panel_w}" y2="{zero_y:.1f}" stroke="#20242D" stroke-width="1.4" stroke-dasharray="5 4"/>'
            )
            points: list[tuple[float, float]] = []
            color = FIRST_LOCATOR_COLORS[model]
            for index, row in enumerate(subset.itertuples(index=False)):
                xx, yy = x(index), y(row.ranked_minus_random)
                low_y, high_y = y(row.ci95_low), y(row.ci95_high)
                points.append((xx, yy))
                parts.extend(
                    [
                        f'<line x1="{xx:.1f}" y1="{low_y:.1f}" x2="{xx:.1f}" y2="{high_y:.1f}" stroke="{color}" stroke-width="2"/>',
                        f'<line x1="{xx-5:.1f}" y1="{low_y:.1f}" x2="{xx+5:.1f}" y2="{low_y:.1f}" stroke="{color}" stroke-width="2"/>',
                        f'<line x1="{xx-5:.1f}" y1="{high_y:.1f}" x2="{xx+5:.1f}" y2="{high_y:.1f}" stroke="{color}" stroke-width="2"/>',
                        f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5" fill="{color}"><title>K={int(row.top_k)}; ranked={float(row.ranked_mean):.6f}; control={float(row.random_mean):.6f}; contrast={float(row.ranked_minus_random):.6f}; 95% CI [{float(row.ci95_low):.6f}, {float(row.ci95_high):.6f}]; Holm p={float(row.holm_p_within_model_endpoint):.6g}</title></circle>',
                        f'<text x="{xx:.1f}" y="{y0+panel_h+18:.1f}" text-anchor="middle" font-size="10" fill="#5E6672">{int(row.top_k)}</text>',
                    ]
                )
            parts.append(
                '<polyline points="'
                + " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in points)
                + f'" fill="none" stroke="{color}" stroke-width="3"/>'
            )
            parts.append(
                f'<text x="{x0 + panel_w/2:.1f}" y="{y0+panel_h+39:.1f}" text-anchor="middle" font-size="10" fill="#5E6672">top-k heads</text>'
            )
    parts.append(
        '<text x="18" y="350" transform="rotate(-90 18 350)" text-anchor="middle" font-size="11" fill="#5E6672">ranked ablation damage − control damage</text>'
    )
    parts.append('</g></svg>')
    return "".join(parts)


def _first_locator_appendix(repo_root: Path) -> str:
    scores, layer_matched, m10_matched, matching = _load_first_locator_data(
        repo_root
    )
    span_rows: list[list[str]] = []
    top_summaries: list[str] = []
    for model in FIRST_LOCATOR_MODELS:
        top = scores[
            (scores["model"] == model)
            & (scores["first_span_absolute_mass_rank"] == 1)
        ].iloc[0]
        head = f'L{int(top["layer"])}H{int(top["head"])}'
        m10 = float(top["ten_span_total_mass_mean"])
        top_summaries.append(
            f'<strong>{html.escape(model)} {head}</strong>: '
            f'<code>mean(m1)={float(top["first_span_mass_mean"]):.6f}</code>, '
            f'<code>mean(M10)={m10:.6f}</code>, '
            f'<code>mean(m1)/mean(M10)={100 * float(top["first_share_ratio_of_means"]):.2f}%</code>'
        )
        for index in range(1, 11):
            mass = float(top[f"span_{index}_mass"])
            span_rows.append(
                [model, head, str(index), f"{mass:.9f}", f"{100 * mass / m10:.3f}%"]
            )

    ablation_rows: list[list[str]] = []
    for control, frame in (
        ("layer-matched", layer_matched),
        ("layer + M10-nearest", m10_matched),
    ):
        for row in frame.sort_values(["model", "endpoint", "top_k"]).itertuples(index=False):
            ablation_rows.append(
                [
                    control,
                    str(row.model),
                    str(row.endpoint),
                    str(int(row.top_k)),
                    f"{float(row.ranked_mean):.6f}",
                    f"{float(row.random_mean):.6f}",
                    f"{float(row.ranked_minus_random):+.6f} "
                    f"[{float(row.ci95_low):+.6f}, {float(row.ci95_high):+.6f}]",
                    f"{float(row.holm_p_within_model_endpoint):.6g}",
                ]
            )
    matching_rows = [
        [
            str(row.model),
            str(int(row.top_k)),
            f"{100 * float(row.mean_overlap_fraction):.1f}%",
            f"{float(row.ranked_M10_mean):.6f}",
            f"{float(row.control_M10_mean):.6f}",
            f"{float(row.mean_absolute_M10_gap):.6f}",
        ]
        for row in matching.sort_values(["model", "top_k"]).itertuples(index=False)
    ]
    return f"""
<section id="appendix-first-locator" class="appendix first-locator-appendix">
<h2>Appendix A · Complete-first-span first-locator</h2>
<div class="callout warning"><strong>与正文的关系。</strong>本附录独立记录 first-locator 的定义、观测 attention phenotype 与负面 causal specificity 结果；它不进入正文的主机制链，也不改变正文对 distributed broad retrieval 的结论。</div>

<h3>A.1 定义、样本与排序</h3>
<p>对一个 N=10 discovery prompt，令 <code>q</code> 为最终 <code>Total:</code> answer query，<code>Sᵢ</code> 为第 <em>i</em> 条完整 needle 的全部 literal-token positions。head <code>h</code> 在第 <em>i</em> 条 needle 上的绝对 attention mass 定义为：</p>
<div class="equation">mᵢ,ₕ(x) = Σⱼ∈Sᵢ(x) αₕ(q,j)；　M₁₀,ₕ(x) = Σᵢ₌₁¹⁰ mᵢ,ₕ(x)</div>
<p>排序分数是 20 个 discovery seeds（1234–1253）上的 <code>meanₓ[m₁,ₕ(x)]</code>。下文的 span 相对占比定义为 <code>mean(mᵢ)/mean(M₁₀)</code>，所以十个 span 的占比精确加总为 100%；它与“先逐样本取比例再平均”是不同 estimand。因果 confirmation 使用 fresh seeds 1336–1355、gold counts 1–5，共 100 prompts/model。</p>
<p>{'<br>'.join(top_summaries)}</p>
<div class="conclusion"><strong>A.1 结论。</strong>first-locator 的候选集只由 discovery <code>mean(m₁)</code> 冻结；<code>m₁</code>、<code>M₁₀</code> 与 <code>mean(m₁)/mean(M₁₀)</code> 分别度量第一条绝对质量、十条总质量与相对集中度，不能互相替代，也都不直接等于因果重要性。</div>

<h3>A.2 All-head attention maps</h3>
<figure id="appendix-first-locator-map-absolute">{_first_locator_heatmap_svg(scores, metric="first_span_mass_mean", title="All-head first-span absolute mass", description="Every Qwen and Gemma head is placed by layer and head index. Color encodes mean absolute attention mass on the complete first needle span; the rank-one head has a gold outline.", relative=False)}<figcaption><strong>图 A1 · 第一条完整 needle 的绝对 mass。</strong>横轴为 decoder layer，纵轴为 head index，每格是一个真实 layer/head；颜色对应 <code>mean(m₁)</code>，采用 log1p 色阶以同时显示极小值与高-mass outlier。金色边框标出各模型按该量冻结的 rank-1 head。悬停可查看精确 <code>m₁</code>、<code>M₁₀</code> 和相对占比。</figcaption></figure>
<figure id="appendix-first-locator-map-relative">{_first_locator_heatmap_svg(scores, metric="first_share_ratio_of_means", title="All-head first-span share of ten-span mass", description="Every Qwen and Gemma head is placed by layer and head index. Color encodes mean first-span mass divided by mean total ten-span mass; the rank-one absolute-mass head has a gold outline.", relative=True)}<figcaption><strong>图 A2 · 第一条 span 在十条 needle mass 中的相对占比。</strong>横轴为 decoder layer，纵轴为 head index；颜色是 <code>mean(m₁)/mean(M₁₀)</code> 的线性0–100%标度。该图回答“已分配给十条 needles 的 mass 中有多少落在第一条”，而图 A1回答绝对分配了多少；高占比不必然意味着高绝对 mass。</figcaption></figure>
<div class="conclusion"><strong>A.2 结论。</strong>两个模型都存在第一条 span 集中的 heads，但绝对 mass 与相对 share 的全头排序并不相同；因此后续因果检验必须保留总 retrieval mass 的 matched sensitivity，不能把高 share 直接解释为专门的 first-locator circuit。</div>

<h3>A.3 Rank-1 heads 的十条 span profile</h3>
<figure id="appendix-first-locator-profile">{_first_span_profile_svg(scores)}<figcaption><strong>图 A3 · 真实 absolute mass profile。</strong>横轴为 needle occurrence index <code>i=1,…,10</code>，纵轴为 discovery 20 seeds 上的真实 <code>mean(mᵢ)</code>；两模型面板使用各自独立纵轴，柱高不是归一化分数。精确到9位小数的 mass 与其在 <code>M₁₀</code> 中的占比见下表。</figcaption></figure>
{_details_table(title="Exact rank-1 ten-span masses", columns=["model", "head", "span i", "mean absolute mass mᵢ", "mean(mᵢ) / mean(M₁₀)"], rows=span_rows, open_by_default=True)}
<div class="conclusion"><strong>A.3 结论。</strong>Qwen L9H19 与 Gemma L11H6 的真实十维 profile 都在第一条 needle 处达到峰值；该结果确认 attention phenotype 的形状，尚不说明删除这些 heads 会比匹配 heads 造成更大行为损伤。</div>

<h3>A.4 First-span-ranked top-k ablation curves</h3>
<p>按 discovery <code>mean(m₁)</code> 冻结 K∈{{1,2,4,8,16,32}} 的 nested head banks，只在最终 <code>Total:</code> query 将其 pre-O slices 置零。令 <code>gᵢ</code> 为 gold count，<code>y⁽⁰⁾ᵢ</code> 为 clean completion 在最终 <code>Total:</code> 后解析出的数字，<code>y⁽ᶜ⁾ₖᵢ</code> 为 condition <code>c</code> 干预后解析出的数字。左列的单样本 damage 是：</p>
<div class="equation">dᵃᵇˢᵢ,ᶜ = |y⁽ᶜ⁾ₖᵢ − gᵢ| − |y⁽⁰⁾ᵢ − gᵢ|</div>
<p>曲线再画 seed-mean 的 <code>dᵃᵇˢ(ranked) − dᵃᵇˢ(control)</code>。因此它来自最终输出数字，但比较的是<strong>相对 gold 的绝对误差增加量之差</strong>，不是原始的 <code>y(ranked) − y(control)</code>。右列先限制到 <code>y⁽⁰⁾ᵢ=gᵢ</code> 的 clean-correct 样本，再计算：</p>
<div class="equation">dᶠᵃⁱˡᵢ,ᶜ = 1[y⁽ᶜ⁾ₖᵢ ≠ gᵢ]；curve = Eₛₑₑd[dᶠᵃⁱˡ(ranked) − dᶠᵃⁱˡ(control)]</div>
<p>两列的正值才表示 first-span ranking 比 matched alternative 造成更大的行为损害；误差线为 20 seed clusters 的 bootstrap 95% CI。</p>
<figure id="appendix-first-locator-ablation-layer">{_first_locator_ablation_svg(layer_matched, control_label="layer-matched random", svg_id="first-locator-layer-matched")}<figcaption><strong>图 A4 · Layer-matched control。</strong>横轴为冻结的 nested <code>K=1,2,4,8,16,32</code>，纵轴为 ranked ablation damage 减去三个 layer-matched controls 的平均 damage；上、下 panel 分别是 absolute-error increase 与 clean-correct→wrong excess，0 表示无特异损伤。Qwen 所有K均无正向显著 specificity；Gemma K=32 的显著结果在负方向，表示 ranked bank 反而比 control 更少破坏。</figcaption></figure>
<figure id="appendix-first-locator-ablation-m10">{_first_locator_ablation_svg(m10_matched, control_label="layer + M10-nearest", svg_id="first-locator-m10-matched")}<figcaption><strong>图 A5 · Layer + M₁₀-nearest control。</strong>横轴、纵轴和两个 panel 的定义与图 A4相同，但 controls 还按总 retrieval mass <code>M₁₀</code> 做 nearest-neighbor matching。全部K仍无正向显著结果。Qwen rank-1 <code>M₁₀</code> 是极端 outlier，因此该模型的 matching 不能完全隔离 first-span concentration 与总 retrieval strength。</figcaption></figure>
{_details_table(title="Exact first-locator top-k ablation statistics", columns=["control", "model", "endpoint", "K", "ranked mean", "control mean", "ranked − control [95% CI]", "Holm p"], rows=ablation_rows)}
{_details_table(title="M10 matching and ranked/control overlap audit", columns=["model", "K", "mean overlap", "ranked M10 mean", "control M10 mean", "mean |M10 gap|"], rows=matching_rows)}

<div class="conclusion"><strong>Appendix A 结论</strong>表示结果为正：Qwen L9H19 与 Gemma L11H6 都呈现完整第一条 needle span 的 attention concentration。因果特异性结果为负：两套 control、两个 endpoint 和全部 K 都没有显示 first-span-ranked heads 造成显著更大的正向损害。因此当前证据支持 first-span attention phenotype，不支持“first-span locator 是独特必要 circuit”。</div>
</section>"""


def _outcome_rows(outcomes: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in sorted(
        outcomes,
        key=lambda item: (
            item["model"],
            item["pooling"],
            int(item["head_rank"]),
            item["metric"],
        ),
    ):
        rows.append(
            [
                html.escape(str(row["model"])),
                html.escape(str(row["pooling"])),
                f"#{int(row['head_rank'])} · L{int(row['layer'])}H{int(row['head'])}",
                html.escape(str(row["metric"])),
                _ci(
                    row,
                    "wrong_minus_correct_count_adjusted",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                ),
                "yes" if float(row["bootstrap_ci_low"]) > 0 or float(row["bootstrap_ci_high"]) < 0 else "no",
            ]
        )
    return rows


def _nested_v44_rows(run_root: Path) -> list[dict[str, Any]]:
    """Paired N→N+1 endpoint diagnostic, restricted before aggregation to V4.4."""
    results: list[dict[str, Any]] = []
    for model in MODELS:
        path = (
            full._attention_analysis_root(run_root, model)
            / "tables"
            / "nested_increment_diagnostics.csv"
        )
        frame = pd.read_csv(path)
        frame = frame[
            (frame["split"] == "confirmation")
            & (frame["design_variant"] == FOCUS_VARIANT)
            & (frame["pooling"] == "span_end")
            & (pd.to_numeric(frame["count"], errors="coerce") >= 2)
            & (pd.to_numeric(frame["omission_count"], errors="coerce") > 0)
            & frame["increment_status"].isin(
                ["failed_to_increment", "registered_plus_one"]
            )
        ].copy()
        frame["omission_count"] = pd.to_numeric(
            frame["omission_count"], errors="raise"
        ).astype(int)
        frame["new_needle_low_attention_rank"] = pd.to_numeric(
            frame["new_needle_low_attention_rank"], errors="raise"
        ).astype(int)
        frame["new_needle_normalized_share"] = pd.to_numeric(
            frame["new_needle_normalized_share"], errors="raise"
        )
        frame["new_in_bottom_k"] = (
            frame["new_needle_low_attention_rank"] <= frame["omission_count"]
        ).astype(float)
        frame["status"] = frame["increment_status"].map(
            {
                "failed_to_increment": "failed",
                "registered_plus_one": "registered",
            }
        )

        block_status = frame.groupby(["seed", "status"], sort=True)[
            ["new_in_bottom_k", "new_needle_normalized_share"]
        ].mean()
        wide_bottom = block_status["new_in_bottom_k"].unstack("status").dropna()
        wide_share = (
            block_status["new_needle_normalized_share"].unstack("status").dropna()
        )
        common_seeds = wide_bottom.index.intersection(wide_share.index)
        wide_bottom = wide_bottom.loc[common_seeds]
        wide_share = wide_share.loc[common_seeds]
        if wide_bottom.empty or not {"failed", "registered"}.issubset(
            wide_bottom.columns
        ):
            raise RuntimeError(f"No paired V4.4 nested blocks in {path}")
        results.append(
            {
                "model": model,
                "paired_seeds": int(len(common_seeds)),
                "failed_bottom": float(wide_bottom["failed"].mean()),
                "registered_bottom": float(wide_bottom["registered"].mean()),
                "failed_share": float(wide_share["failed"].mean()),
                "registered_share": float(wide_share["registered"].mean()),
            }
        )
    return results


def _omission_rows(
    alignment: list[dict[str, Any]], nested: list[dict[str, Any]]
) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in sorted(alignment, key=lambda item: (item["model"], item["variant"])):
        rows.append(
            [
                html.escape(str(row["model"])),
                "undercount tail",
                _num(row["tail_prefix_ratio"], 3),
                _num(row["overlap"], 3),
                "A = tail/prefix mass ratio; B = bottom-k overlap. A<1 means the inferred omitted tail receives less attention than the prefix.",
            ]
        )
    for row in sorted(nested, key=lambda item: item["model"]):
        rows.append(
            [
                html.escape(str(row["model"])),
                "nested N→N+1",
                f"bottom-k failed/registered = {_num(row['failed_bottom'], 3)}/{_num(row['registered_bottom'], 3)}",
                f"new-needle share failed/registered = {_num(row['failed_share'], 3)}/{_num(row['registered_share'], 3)}",
                f"Paired within {int(row['paired_seeds'])} V4.4 seeds; failed means the model did not increment after the added needle.",
            ]
        )
    return rows


def _ablation_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [
            html.escape(row["model"]),
            f"top-{int(row['top_n'])}",
            str(int(row["prompts"])),
            f"{_pct(row['ranked_changed'])} / {_pct(row['random_changed'])}",
            _ci(
                row,
                "count_shift_difference",
                "count_shift_difference_low",
                "count_shift_difference_high",
            ),
            _ci(row, "error_difference", "error_difference_low", "error_difference_high"),
            _num(row["count_shift_p_holm"], 4),
        ]
        for row in rows
    ]


def _endpoint_patch_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [
            html.escape(row["model"]),
            f"L{int(row['layer'])}→final",
            str(int(row["rows"])),
            _pct(row["changed_rate"]),
            f"{_pct(row['moved_rate'])} [{_pct(row['moved_rate_low'])}, {_pct(row['moved_rate_high'])}]",
            _ci(row, "aligned_shift", "aligned_shift_low", "aligned_shift_high"),
            _num(row["aligned_shift_p_holm"], 4),
        ]
        for row in rows
    ]


def _answer_patch_rows(frame: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in frame.sort_values(["model", "layer"]).to_dict("records"):
        rows.append(
            [
                html.escape(str(row["model"])),
                f"L{int(row['layer'])}",
                str(int(row["rows"])),
                str(int(row["eligible_donor_prediction_rows"])),
                f"{_pct(row['eligible_donor_adoption_rate'])} "
                f"[{_pct(row['eligible_donor_adoption_rate_ci95_low'])}, "
                f"{_pct(row['eligible_donor_adoption_rate_ci95_high'])}]",
                f"{_num(row['mean_direction_aligned_shift'], 3, signed=True)} "
                f"[{_num(row['mean_direction_aligned_shift_ci95_low'], 3, signed=True)}, "
                f"{_num(row['mean_direction_aligned_shift_ci95_high'], 3, signed=True)}]",
                _pct(row["patched_valid_rate"]),
            ]
        )
    return rows


def _steering_v1_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [
            html.escape(row["model"]),
            f"L{int(row['layer'])}",
            f"{_pct(row['geometric_changed'])} / {_pct(row['random_changed'])}",
            f"{_pct(row['geometric_moved'])} / {_pct(row['random_moved'])}",
            _ci(row, "aligned_difference", "aligned_difference_low", "aligned_difference_high"),
            f"{_pct(row['geometric_target_hit'])} / {_pct(row['random_target_hit'])}",
            _num(row["aligned_difference_p_holm"], 4),
        ]
        for row in rows
    ]


def _steering_v2_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        screen_root = full._completed_steering_v2_root(
            run_root, model=model, phase="screen"
        )
        confirmation_root = full._completed_steering_v2_root(
            run_root, model=model, phase="confirmation"
        )
        selection = full._read_json(screen_root / "selection.json")
        detail = pd.read_csv(confirmation_root / "detail.csv.gz", compression="gzip")
        paired = full._steering_v2_paired_effects(detail)
        model_rows: list[dict[str, Any]] = []
        for protocol in ("single_layer", "multi_layer"):
            locked = selection["selected"][protocol]
            layer_set = str(locked["layer_set"])
            alpha = float(locked["alpha"])
            selected = paired[
                (paired["design_variant"].astype(str) == FOCUS_VARIANT)
                & (paired["steering_protocol"].astype(str) == protocol)
                & (paired["layer_set"].astype(str) == layer_set)
                & np.isclose(pd.to_numeric(paired["alpha"]), alpha)
            ].copy()
            if selected.empty:
                raise RuntimeError(f"{model}/{protocol}: no {FOCUS_VARIANT} steering rows")

            def estimate(metric: str) -> tuple[float, float, float, float]:
                seed_values = (
                    selected.groupby("seed", sort=True)[metric].mean().to_numpy(float)
                )
                if len(seed_values) != 10:
                    raise RuntimeError(f"{model}/{protocol}: expected ten seeds")
                center, low, high = full._seed_bootstrap(
                    seed_values,
                    label=f"v44-steering-v2:{model}:{protocol}:{metric}",
                )
                return center, low, high, full._exact_sign_flip_p(seed_values)

            aligned = estimate("strict_aligned_shift_effect")
            moved = estimate("strict_moved_effect")
            target = estimate("strict_target_hit_effect")
            valid = estimate("strict_valid_effect")
            model_rows.append(
                {
                    "model": model,
                    "protocol": protocol,
                    "layer_set": layer_set,
                    "alpha": alpha,
                    "paired_rows": int(len(selected)),
                    "aligned_effect": aligned[0],
                    "aligned_effect_low": aligned[1],
                    "aligned_effect_high": aligned[2],
                    "p_raw": aligned[3],
                    "moved_effect": moved[0],
                    "moved_effect_low": moved[1],
                    "moved_effect_high": moved[2],
                    "target_hit_effect": target[0],
                    "target_hit_effect_low": target[1],
                    "target_hit_effect_high": target[2],
                    "valid_effect": valid[0],
                }
            )
        adjusted = full._holm_adjust([row["p_raw"] for row in model_rows])
        for row, p_holm in zip(model_rows, adjusted):
            row["p_holm"] = p_holm
        rows.extend(model_rows)
    return rows


def _steering_v2_table(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [
            html.escape(row["model"]),
            html.escape(row["protocol"].replace("_", "-")),
            html.escape(row["layer_set"]),
            _num(row["alpha"], 2),
            _ci(row, "aligned_effect", "aligned_effect_low", "aligned_effect_high"),
            f"{_pct(row['moved_effect'])} [{_pct(row['moved_effect_low'])}, {_pct(row['moved_effect_high'])}]",
            f"{_pct(row['target_hit_effect'])} [{_pct(row['target_hit_effect_low'])}, {_pct(row['target_hit_effect_high'])}]",
            _num(row["p_holm"], 4),
        ]
        for row in rows
    ]


def _causal_summary_html(
    ablation: list[dict[str, Any]],
    endpoint: list[dict[str, Any]],
    answer: pd.DataFrame,
    steering_v2: list[dict[str, Any]],
) -> str:
    statements: list[str] = []
    for model in MODELS:
        top8 = next(
            row for row in ablation if row["model"] == model and row["top_n"] == 8
        )
        best_endpoint = max(
            (row for row in endpoint if row["model"] == model),
            key=lambda row: abs(float(row["aligned_shift"])),
        )
        final = answer[answer["model"] == model].sort_values("layer").iloc[-1]
        single = next(
            row
            for row in steering_v2
            if row["model"] == model and row["protocol"] == "single_layer"
        )
        statements.append(
            f"<strong>{html.escape(model)}</strong>：top-8 ranked-vs-random ablation 的 count-shift contrast "
            f"{_num(top8['count_shift_difference'], 3, signed=True)}；needle-end patch 最大绝对 aligned shift "
            f"{_num(best_endpoint['aligned_shift'], 3, signed=True)}；final-layer answer-query donor adoption "
            f"{_pct(final['eligible_donor_adoption_rate'])}；V4.4 held-out single-layer steering effect "
            f"{_num(single['aligned_effect'], 3, signed=True)}。"
        )
    return "<p>" + " ".join(statements) + "</p>"


def _ablation_result_html(rows: list[dict[str, Any]]) -> str:
    by_key = {(row["model"], int(row["top_n"])): row for row in rows}
    q4 = by_key[("Qwen3-8B", 4)]
    q8 = by_key[("Qwen3-8B", 8)]
    g4 = by_key[("Gemma4-E4B", 4)]
    g8 = by_key[("Gemma4-E4B", 8)]
    return (
        '<div class="conclusion"><strong>4.1 结果与边界</strong>'
        f"Qwen top-4 ranked bank 相对随机 bank 造成额外 count shift "
        f"{_ci(q4, 'count_shift_difference', 'count_shift_difference_low', 'count_shift_difference_high')}，"
        f"而 top-8 contrast 为 {_ci(q8, 'count_shift_difference', 'count_shift_difference_low', 'count_shift_difference_high')}；"
        "因此 Qwen 不是简单的“k 越大越必要”。"
        f"Gemma top-4 / top-8 contrasts 分别为 {_num(g4['count_shift_difference'], 3, signed=True)} / "
        f"{_num(g8['count_shift_difference'], 3, signed=True)}，但这 40 个 V4.4 high-count clean baselines 的正确率为 "
        f"{_pct(g4['baseline_correct'])}，所以它证明 ranked bank 会改变生成数值，不足以证明该 bank 在正确计数中必要。"
        "当前证据是 bank-level、mixed-phenotype necessity screen，不是单 head 必要性证明。</div>"
    )


def _endpoint_patch_result_html(rows: list[dict[str, Any]]) -> str:
    pieces = []
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        peak = max(model_rows, key=lambda row: abs(float(row["aligned_shift"])))
        pieces.append(
            f"{html.escape(model)} 最大绝对 aligned shift 出现在 L{int(peak['layer'])}→final："
            f"{_ci(peak, 'aligned_shift', 'aligned_shift_low', 'aligned_shift_high')}"
        )
    return (
        '<div class="conclusion"><strong>4.2 结果与边界</strong>'
        + "；".join(pieces)
        + "。所有 Holm-adjusted p=1，且 effect 接近 0。故单个 toggled needle-end token 的 exact residual，"
        "即使从中层持续 clamp 到末层，在本设计下仍不足以运输一次 count increment。"
        "这不否定 full-span、多 token 或 coordinated multi-layer state 的充分性。</div>"
    )


def _answer_patch_result_html(frame: pd.DataFrame) -> str:
    pieces = []
    for model in MODELS:
        model_rows = frame[frame["model"] == model].sort_values("layer")
        onset = model_rows[model_rows["eligible_donor_adoption_rate"] >= 0.5].iloc[0]
        final = model_rows.iloc[-1]
        pieces.append(
            f"{html.escape(model)} 在 L{int(onset['layer'])} 首次达到至少 50% donor adoption "
            f"({_pct(onset['eligible_donor_adoption_rate'])})，末层为 {_pct(final['eligible_donor_adoption_rate'])}"
        )
    return (
        '<div class="conclusion"><strong>4.3 结果与边界</strong>'
        + "；".join(pieces)
        + "。这给出清楚的 late-layer onset：完整 answer-query residual 在后层几乎可以复制 donor 已算出的预测。"
        "Primary outcome 是 donor <em>prediction</em> adoption；若 donor 本身答错，它不等同于变成 donor gold。</div>"
    )


def _steering_v1_result_html(rows: list[dict[str, Any]]) -> str:
    pieces = []
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        best = max(model_rows, key=lambda row: float(row["aligned_difference"]))
        pieces.append(
            f"{html.escape(model)} 的最大 geometric−random effect 在 L{int(best['layer'])}："
            f"{_ci(best, 'aligned_difference', 'aligned_difference_low', 'aligned_difference_high')}，"
            f"Holm p={_num(best['aligned_difference_p_holm'], 4)}"
        )
    return (
        '<div class="conclusion"><strong>4.4 结果与边界</strong>'
        + "；".join(pieces)
        + "。早/中层效应接近 0，而后层 count-centroid direction 显著优于等范数正交随机方向。"
        "这证明 late answer state 可沿 count direction 被操纵，但不证明精确设定到目标数字。</div>"
    )


def _steering_v2_result_html(rows: list[dict[str, Any]]) -> str:
    pieces = []
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        single = next(row for row in model_rows if row["protocol"] == "single_layer")
        multi = next(row for row in model_rows if row["protocol"] == "multi_layer")
        pieces.append(
            f"{html.escape(model)} single / multi aligned effects 为 "
            f"{_num(single['aligned_effect'], 3, signed=True)} / {_num(multi['aligned_effect'], 3, signed=True)}"
        )
    return (
        '<div class="conclusion"><strong>4.5 结果与边界</strong>'
        + "；".join(pieces)
        + "，四个 95% CI 均高于 0。Discovery-locked plan 在 held-out V4.4 复现，但 multi-layer 没有稳定超过 single-layer；"
        "exact-target gain 仅 3.3%–6.7%。因此最强结论是稳定的方向性控制，而不是精确 set-to-count 或 multi-layer 优势。</div>"
    )


def _figure(title: str, intro: str, body: str, caption: str) -> str:
    return (
        '<div class="figure-block">'
        f"<h3>{title}</h3>"
        f'<p class="figure-intro">{intro}</p>'
        f"<figure>{body}<figcaption>{caption}</figcaption></figure>"
        "</div>"
    )


REPORT_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Realistic NIAH V4.4 · non-thinking mechanism report</title>
<style>
:root{--paper:#F3EEE4;--surface:#FFFDF8;--ink:#20242D;--muted:#5E6672;--line:#C9C2B6;--indigo:#23165C;--violet:#6750E8;--cyan:#00A9D8;--teal:#00A88F;--green:#2DBE77;--pink:#D94B86;--yellow:#D6B52C;--gray:#718096}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Segoe UI",Arial,sans-serif;line-height:1.62}nav{position:sticky;top:0;z-index:4;background:rgba(243,238,228,.96);border-bottom:1px solid var(--line);padding:10px 20px;display:flex;gap:18px;flex-wrap:wrap}nav a{color:var(--indigo);text-decoration:none;font-size:14px;font-weight:650}main{max-width:1180px;margin:0 auto;padding:36px 28px 80px}header{max-width:900px;padding:24px 0 28px;border-bottom:2px solid var(--ink)}.eyebrow{font:700 12px/1.2 Consolas,monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--teal)}h1{font-size:40px;line-height:1.08;letter-spacing:-.035em;margin:10px 0 18px}h2{font-size:28px;line-height:1.22;letter-spacing:-.02em;margin:0 0 14px}h3{font-size:20px;line-height:1.3;margin:30px 0 8px}p{max-width:88ch;margin:10px 0 16px}.lead{font-size:18px;color:#3E4651}.meta{font:12px/1.6 Consolas,monospace;color:var(--muted)}section{padding:48px 0;border-bottom:1px solid var(--line)}.callout,.conclusion{background:var(--surface);border-left:4px solid var(--teal);padding:16px 20px;margin:20px 0}.warning{border-left-color:var(--yellow)}.conclusion strong:first-child{display:block;color:var(--indigo);margin-bottom:5px}.equation{font-family:Consolas,monospace;background:#ECE6DA;border:1px solid var(--line);padding:12px 14px;overflow:auto}.figure-block{margin:30px 0}.figure-intro{color:#38414D;margin-bottom:12px}figure{margin:0;background:var(--surface);border:1px solid var(--line);padding:16px}figcaption{font-size:14px;color:var(--muted);margin-top:12px}.controls{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 12px}.controls label{font-size:12px;color:var(--muted);display:grid;gap:3px}.controls select,.controls button,.switcher button{font:13px "Segoe UI",sans-serif;color:var(--ink);background:#FAF7F0;border:1px solid #AAA195;border-radius:4px;padding:6px 9px}.controls button:active,.switcher button:active{transform:translateY(1px)}.plot-shell{position:relative;background:#15112B;border-radius:4px;overflow:hidden}.plot-shell canvas{display:block;width:100%;height:580px;touch-action:none}.plot-tooltip{position:absolute;display:none;pointer-events:none;background:rgba(255,253,248,.96);color:var(--ink);border:1px solid var(--line);padding:8px 10px;font-size:12px;max-width:260px}.plot-stats{font:12px/1.55 Consolas,monospace;color:var(--muted);margin-top:8px}.legend{display:flex;gap:10px;flex-wrap:wrap;font-size:12px;margin:10px 0}.legend span{display:inline-flex;align-items:center;gap:4px}.legend i{width:9px;height:9px;border-radius:50%;display:inline-block}.stat-svg{width:100%;height:auto;display:block}.switcher{display:flex;gap:8px;margin-bottom:10px}.switcher button[aria-pressed="true"]{background:var(--indigo);color:#fff}.atlas-panel[hidden]{display:none}details.data-table{background:var(--surface);border:1px solid var(--line);margin:16px 0}details.data-table summary{cursor:pointer;padding:12px 14px;font-weight:650;color:var(--indigo)}.table-scroll{overflow:auto;border-top:1px solid var(--line)}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 11px;border-bottom:1px solid #DED8CE;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#ECE6DA;color:#303744}tbody tr:hover{background:#FAF6EE}code{font-family:Consolas,monospace;background:#EAE4D8;padding:1px 4px}.small{font-size:13px;color:var(--muted)}@media(max-width:760px){main{padding:24px 14px 60px}h1{font-size:31px}.plot-shell canvas{height:480px}nav{gap:10px}figure{padding:10px}}
</style>
</head>
<body>
<nav><a href="#scope">范围</a><a href="#prompt">Prompt hidden state</a><a href="#answer">Answer hidden state</a><a href="#attention">Attention heads</a><a href="#causal">Causal tests</a><a href="#limits">结论与缺口</a><a href="#appendix-first-locator">Appendix A</a></nav>
<main>
<header>
<div class="eyebrow">Realistic NIAH · V4.4 only · non-thinking</div>
<h1>从 counter representation 到分布式 retrieval 与 late readout</h1>
<p class="lead">本报告把已完成 V4 实验中的 V4.4 panel 单独取出。章节顺序固定为 prompt-reading hidden state、answer-query hidden state、attention-head map，再把每类表征与对应的 ablation、patching 和 steering 对齐。</p>
<p class="meta">run @@RUN@@ · generated @@GENERATED@@ · commit @@COMMIT@@</p>
</header>

<section id="scope">
<h2>范围与比较对象</h2>
<p>跨 V4.1–V4.4 的完整报告已经显示：以 needle-end 读取 prompt hidden state 时，count geometry 在逐步放宽 position、city-score 顺序和内容后仍保留。因此本报告把 <strong>V4.4 + needle-end</strong> 作为后续主设置。为了避免重新选择最有利坐标，prompt/answer PCA layer 与 basis 仍由 V4.1 discovery 冻结，再投影 V4.4；这使 V4.4 图是跨 panel generalization，而不是在同一 panel 重新拟合后的漂亮图。</p>
<div class="callout warning"><strong>推断范围。</strong>当前 causal 数据最初覆盖四个 panel；这里重新计算 V4.4-only seed-cluster estimates。它们是 panel-restricted reanalysis，不会把原先 pooled primary estimand 改写成预注册的 V4.4-only primary test。未来新实验可以直接在 V4.4 内划分 discovery / confirmation。</div>
@@INVENTORY@@
@@BEHAVIOR_TABLE@@
</section>

<section id="prompt">
<h2>1 · Prompt-reading counter representation</h2>
<p>对第 n 条 needle，读取其最后一个 token 经过 decoder block 后的 residual state。每个 model/layer 的 PCA 只用 V4.1 discovery 的全部 n=1…10 states 拟合；下面仅显示 V4.4。颜色表示 occurrence index n，而不是最终预测。整条 N=10 trajectory 的点边框使用该 seed 的真实 greedy N=10 outcome，因此这是 trajectory-level outcome audit。</p>
@@PROMPT_LAYER_TABLE@@
@@PROMPT_FIGURE@@
<div class="conclusion"><strong>本节结论</strong>V4.4 的 needle-end states 仍沿冻结坐标形成有序 count trajectory。这个结果证明 count/index information 在 prompt 读取期间可解码，但不等价于单个 endpoint state 足以改变最终答案；后面的 exact endpoint patch 直接检验充分性。</div>
</section>

<section id="answer">
<h2>2 · Answer-query counter representation</h2>
<p>Answer-query state 是 prompt 末尾 <code>Total:</code> query token 的 post-block residual。它与 needle-end state 不是同一 token role，也不共享 PCA basis。主 basis 使用所有 V4.1 discovery rows；correct-only basis 只作敏感性分析，并将同一批 V4.4 states 投影进去。点的 correct/wrong/invalid 标签来自该 prompt 的完整 greedy 数字输出。</p>
@@ANSWER_LAYER_TABLE@@
@@ANSWER_FIGURE@@
@@JOINT_FIGURE@@
<div class="conclusion"><strong>本节结论</strong>Answer-query 在后层形成与 count 有关的聚合状态；prompt 与 answer trajectory 具有相关的全局几何，但不是逐步复制的同一条局部路径。Exact query-state patch 与 centroid steering 分别检验“完整 donor state 是否充分”和“count direction 是否被 readout 使用”。</div>
</section>

<section id="attention">
<h2>3 · V4.4 attention-head representation</h2>
<p>Atlas 的每一个格子是一个实际保存 full-attention row 的 layer/head。Query 固定为最终 <code>Total:</code> token；key pooling 可切换为 needle endpoint 或完整 needle span 的 literal sum。颜色是 discovery primary score 的对数尺度，只用于同一视图内排序。它不是单个 needle 的概率，也不能跨两种 pooling 直接比较颜色深浅。</p>
@@ATTENTION_ATLAS@@
@@OUTCOME_TABLE@@
@@OMISSION_TABLE@@
<div class="conclusion"><strong>本节结论</strong>V4.4 中 broad retrieval 由多个 heads 分担。Correct/wrong 的 aggregate mass 差异稀疏，attention association 本身不能替代因果检验。</div>
</section>

<section id="causal">
<h2>4 · Causal tests：分别回答 necessity、state sufficiency 与 geometric manipulability</h2>

<h3>4.1 Head ablation · mixed ranked bank 是否比 layer-matched random 更重要？</h3>
<p class="figure-intro">对 V4.4 count 7–10 的同一批 receiver，分别 zero discovery-ranked span-end top-k heads 与逐层数量匹配的随机 heads。每个条件的 generated-count shift 都先减 clean baseline，再取 ranked−random；负值表示 ranked bank 被移除后出现额外 undercount。</p>
<figure>@@ABLATION_SVG@@<figcaption><strong>Head-ablation forest.</strong> 横轴是 paired mean count-shift contrast：ranked ablation 减 layer-matched random ablation；0 表示两种 ablation 影响相同，负值表示 ranked bank 对维持输出 count magnitude 更必要。点为十个 confirmation seed 聚类后的均值，线为 seed bootstrap 95% CI。纵轴列出 model 与 top-k。</figcaption></figure>
@@ABLATION_TABLE@@
@@ABLATION_RESULT@@

<h3>4.2 Needle-end patching · 单个 toggled endpoint state 是否足以运输 count increment？</h3>
<p class="figure-intro">同 seed 的 nested pair 确定唯一 toggled slot。Insertion 从 active donor 拷贝该 slot 的最后 token residual；removal 使用 matched inactive donor state。状态从指定 start layer 起在后续每层持续 clamp。主要量把 insertion 的正向变化和 removal 的负向变化统一成 direction-aligned shift。这里没有 random residual control，因此结论重点是接近零的充分性检验。</p>
<figure>@@ENDPOINT_PATCH_SVG@@<figcaption><strong>Exact needle-end residual transport.</strong> 横轴是 patch 后相对 clean receiver 的 direction-aligned greedy count shift；正值表示向 donor count 移动。点为 seed-cluster mean，线为 95% CI；纵轴给出 model 与开始 clamp 的 layer。该图只检验一个 endpoint token 的 exact state transport，不检验 full-span 或多-token coordinated patch。</figcaption></figure>
@@ENDPOINT_PATCH_TABLE@@
@@ENDPOINT_PATCH_RESULT@@

<h3>4.3 Answer-query patching · late query state 是否足以运输 donor 已算出的 prediction？</h3>
<p class="figure-intro">在一个 layer 只把 donor 的 <code>Total:</code> query residual 完整替换到 receiver，然后继续完整 greedy generation。Primary denominator 只含 clean receiver prediction 与 donor prediction 不同的 rows；成功指 patched prediction 等于 donor baseline prediction。它比较的是 computed prediction transport，而不是 donor gold accuracy。</p>
<figure>@@ANSWER_PATCH_SVG@@<figcaption><strong>Exact answer-query donor transport.</strong> 横轴为 eligible rows 中采用 donor baseline prediction 的比例，0 表示不运输，1 表示全部运输；线为十个 seed 的 bootstrap 95% CI。纵轴是单层 patch site。V4.4 没有 Gemma 的多-token 11 invalid 特例；该特例仅发生在 V4.1。</figcaption></figure>
@@ANSWER_PATCH_TABLE@@
@@ANSWER_PATCH_RESULT@@

<h3>4.4 Steering v1 · 单层 centroid delta 是否优于等范数随机方向？</h3>
<p class="figure-intro">对同一个 V4.4 receiver，在 answer-query residual 上加入 discovery centroid difference μ_target−μ_receiver；随机 arm 使用与该向量等范数且正交的方向。横轴是 geometric−random 的 direction-aligned greedy count shift，因此排除了“只因注入一个大向量而改变输出”的一阶解释。</p>
<figure>@@STEERING_V1_SVG@@<figcaption><strong>Single-layer centroid-delta steering.</strong> 横轴为 geometric arm 减 norm-matched orthogonal random arm 的 direction-aligned count shift；正值表示 count geometry 在该层被生成 readout 使用。点为 paired seed mean，线为 95% CI；纵轴列出 model/layer。</figcaption></figure>
@@STEERING_V1_TABLE@@
@@STEERING_V1_RESULT@@

<h3>4.5 Steering v2 · discovery-locked single 与 multi 是否在 held-out V4.4 复现？</h3>
<p class="figure-intro">四个 discovery seeds 在全部 panel 上用 worst-panel score 从 15 个 plan 中各锁定一个 single 和 multi plan；随后十个 disjoint seeds 不再调参。这里单独报告 held-out V4.4 panel。Random arm 仍是逐层等范数正交方向；invalid 被记为零效果/失败。</p>
<figure>@@STEERING_V2_SVG@@<figcaption><strong>Discovery-locked V4.4 steering.</strong> 横轴是 held-out V4.4 中 geometric−random 的 strict direction-aligned count shift；点为 seed mean，线为 95% CI。纵轴给出 model、single/multi protocol、锁定 layer set 与 α。该图检验方向性操纵，不代表精确 set-to-target。</figcaption></figure>
@@STEERING_V2_TABLE@@
@@STEERING_V2_RESULT@@

<div class="conclusion"><strong>本节联合结论</strong>@@CAUSAL_SUMMARY@@ Late answer-query state 是强充分 carrier，late centroid direction 也可操纵输出；单个 prompt-side needle endpoint state 则不具备相同运输能力。Head-bank ablation 支持分布式 attention bank 的必要贡献，而不是单 head arithmetic counter。</div>
</section>

<section id="limits">
<h2>5 · 当前机制与下一步</h2>
<p>目前最小机制是：多个 attention heads 对 needles 做分布式 retrieval；prompt-side endpoint 留下可解码但不可单点运输的 count/index information；后层 <code>Total:</code> query 形成可执行的预测状态，LM readout 从该状态产生数字。V4.4 high-count undercount 更像系统性漏增量/饱和，而不是同一 count 下的 seed scatter 随 n 增大。</p>
<div class="callout warning"><strong>尚未完成。</strong>本轮没有完成 full-needle-span residual patch、逐 token coordinated multi-layer restoration 与 head-output patching。代码中曾注册但 screen 未运行的条件不属于本报告证据。下一步若固定 V4.4，应在 disjoint V4.4 discovery/confirmation seeds 上做 full-span active↔inactive patch 加 matched random/content controls，以及 nested N→N+1 的 query-state increment tracing。</div>
</section>
@@FIRST_LOCATOR_APPENDIX@@
</main>

<script>
const PROMPT_DATA=@@PROMPT_DATA@@;
const ANSWER_DATA=@@ANSWER_DATA@@;
const JOINT_DATA=@@JOINT_DATA@@;
const COUNT_COLORS=['#23165C','#4430A8','#6750E8','#0077B6','#00A9D8','#00A88F','#2DBE77','#A6C84A','#D6B52C','#D94B86'];

function makeProjector(prefix,data,mode){
 const canvas=document.getElementById(prefix+'-canvas'),ctx=canvas.getContext('2d'),tooltip=document.getElementById(prefix+'-tooltip');
 const controls={model:document.getElementById(prefix+'-model'),layer:document.getElementById(prefix+'-layer'),fit:document.getElementById(prefix+'-fit'),split:document.getElementById(prefix+'-split'),outcome:document.getElementById(prefix+'-outcome'),points:document.getElementById(prefix+'-points'),x:document.getElementById(prefix+'-x'),y:document.getElementById(prefix+'-y'),z:document.getElementById(prefix+'-z')};
 let yaw=-.72,pitch=.44,zoom=1,drag=false,lastX=0,lastY=0,screenPoints=[];
 const keyFor=()=>mode==='answer'?`${controls.model.value}|${controls.layer.value}|${controls.fit.value}`:`${controls.model.value}|${controls.layer.value}`;
 function layers(){const model=controls.model.value;return [...new Set(Object.keys(data).filter(k=>k.startsWith(model+'|')).map(k=>+k.split('|')[1]))].sort((a,b)=>a-b)}
 function refreshLayers(){const values=layers();controls.layer.innerHTML='';for(const layer of values){const item=data[mode==='answer'?`${controls.model.value}|${layer}|all`:`${controls.model.value}|${layer}`];const option=document.createElement('option');option.value=layer;option.textContent=`L${layer}${item?.manifold_display?' · manifold-display':(item?.probe_optimal?' · probe-optimal':'')}`;controls.layer.appendChild(option)}const preferred=values.find(layer=>data[mode==='answer'?`${controls.model.value}|${layer}|all`:`${controls.model.value}|${layer}`]?.manifold_display);controls.layer.value=String(preferred??values[values.length-1])}
 function active(){return data[keyFor()]}
 function rows(){const item=active();if(!item)return[];return item.rows.filter(r=>(!controls.split||controls.split.value==='all'||r[1]===controls.split.value)&&(controls.outcome.value==='all'||r[2]===controls.outcome.value))}
 function stats(rs,axes){if(!rs.length)return null;const values=axes.map(a=>rs.map(r=>r[6+a])),mins=values.map(v=>Math.min(...v)),maxs=values.map(v=>Math.max(...v));return{mins,maxs,center:mins.map((v,i)=>(v+maxs[i])/2),range:mins.map((v,i)=>Math.max(maxs[i]-v,1e-8))}}
 function transformFor(rs,axes,w,h){const s=stats(rs,axes);if(!s)return null;const common=Math.max(...s.range),radius=Math.min(w,h)*.36*zoom;return p=>{let x=(p[0]-s.center[0])*2/common,y=(p[1]-s.center[1])*2/common,z=(p[2]-s.center[2])*2/common;const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),x1=cy*x+sy*z,z1=-sy*x+cy*z,y1=cp*y-sp*z1,z2=sp*y+cp*z1;return{x:w/2+x1*radius,y:h/2-y1*radius,z:z2}}}
 function centroids(rs){const out=[];for(let count=1;count<=10;count++){const g=rs.filter(r=>r[5]===count);if(!g.length)continue;const p=[];for(let pc=0;pc<6;pc++)p.push(g.reduce((s,r)=>s+r[6+pc],0)/g.length);out.push({count,p,n:g.length})}return out}
 function draw(){const rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height;ctx.clearRect(0,0,w,h);ctx.fillStyle='#15112B';ctx.fillRect(0,0,w,h);const rs=rows(),axes=[+controls.x.value,+controls.y.value,+controls.z.value],tf=transformFor(rs,axes,w,h);screenPoints=[];if(!tf||!rs.length){ctx.fillStyle='#D6B52C';ctx.font='16px Segoe UI';ctx.fillText('No V4.4 states match this filter.',20,30);return}const pts=rs.map(r=>({r,q:tf(axes.map(a=>r[6+a]))})).sort((a,b)=>a.q.z-b.q.z);if(controls.points.value!=='centroids'){for(const item of pts){const r=item.r,q=item.q;ctx.globalAlpha=r[2]==='correct'?.62:.30;ctx.fillStyle=COUNT_COLORS[r[5]-1];ctx.strokeStyle=r[2]==='correct'?'#FFFDF8':(r[2]==='invalid'?'#D94B86':'#14111D');ctx.lineWidth=r[2]==='correct'?1.7:.8;ctx.beginPath();ctx.arc(q.x,q.y,r[1]==='confirmation'?3.1:2.6,0,Math.PI*2);ctx.fill();ctx.stroke();screenPoints.push({x:q.x,y:q.y,r})}ctx.globalAlpha=1}const path=centroids(rs).map(x=>({...x,q:tf(axes.map(a=>x.p[a]))}));ctx.strokeStyle='#FFFDF8';ctx.lineWidth=2.5;ctx.beginPath();path.forEach((p,i)=>i?ctx.lineTo(p.q.x,p.q.y):ctx.moveTo(p.q.x,p.q.y));ctx.stroke();for(const p of path){ctx.fillStyle=COUNT_COLORS[p.count-1];ctx.strokeStyle='#14111D';ctx.lineWidth=1;ctx.beginPath();ctx.arc(p.q.x,p.q.y,5.8,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#FFFDF8';ctx.font='11px Consolas';ctx.fillText(String(p.count),p.q.x+7,p.q.y-7)}const item=active();document.getElementById(prefix+'-stats').innerHTML=`<strong>${item.model} · L${item.layer} · V4.4</strong> · PCA fit ${item.fit_cohort??'all'} V4.1 discovery · EVR PC1–3 ${(100*item.explained_variance_ratio.slice(0,3).reduce((a,b)=>a+b,0)).toFixed(1)}% · ${rs.length} states`;}
 function resize(){const rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));ctx.setTransform(dpr,0,0,dpr,0,0);draw()}
 controls.model.addEventListener('change',()=>{refreshLayers();draw()});[controls.layer,controls.fit,controls.split,controls.outcome,controls.points,controls.x,controls.y,controls.z].filter(Boolean).forEach(x=>x.addEventListener('change',draw));canvas.addEventListener('pointerdown',e=>{drag=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(drag){yaw+=(e.clientX-lastX)*.008;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-lastY)*.008));lastX=e.clientX;lastY=e.clientY;draw();return}const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;let best=null,d=Infinity;for(const p of screenPoints){const v=(p.x-x)**2+(p.y-y)**2;if(v<d){d=v;best=p}}if(best&&d<90){tooltip.style.display='block';tooltip.style.left=Math.min(rect.width-260,x+14)+'px';tooltip.style.top=Math.max(8,y-12)+'px';tooltip.innerHTML=`seed ${best.r[0]} · count ${best.r[5]}<br>${best.r[1]} · ${best.r[2]} · predicted ${best.r[3]??'invalid'} · error ${best.r[4]??'NA'}`}else tooltip.style.display='none'});canvas.addEventListener('pointerup',()=>drag=false);canvas.addEventListener('pointercancel',()=>drag=false);canvas.addEventListener('mouseleave',()=>tooltip.style.display='none');canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.5,Math.min(2.5,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});refreshLayers();new ResizeObserver(resize).observe(canvas);resize();
}

function makeJoint(){const canvas=document.getElementById('joint-canvas'),ctx=canvas.getContext('2d'),model=document.getElementById('joint-model'),layer=document.getElementById('joint-layer'),xSel=document.getElementById('joint-x'),ySel=document.getElementById('joint-y'),zSel=document.getElementById('joint-z');let yaw=-.72,pitch=.44,zoom=1,drag=false,lx=0,ly=0;function refresh(){const ls=[...new Set(Object.keys(JOINT_DATA).filter(k=>k.startsWith(model.value+'|')).map(k=>+k.split('|')[1]))].sort((a,b)=>a-b);layer.innerHTML='';for(const l of ls){const o=document.createElement('option');o.value=l;o.textContent=`L${l}${JOINT_DATA[`${model.value}|${l}`]?.manifold_display?' · answer manifold-display':''}`;layer.appendChild(o)}const p=ls.find(l=>JOINT_DATA[`${model.value}|${l}`]?.manifold_display);layer.value=String(p??ls[ls.length-1])}function draw(){const rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height;ctx.clearRect(0,0,w,h);ctx.fillStyle='#15112B';ctx.fillRect(0,0,w,h);const d=JOINT_DATA[`${model.value}|${layer.value}`],axes=[+xSel.value,+ySel.value,+zSel.value];if(!d)return;const rows=d.rows,values=axes.map(a=>rows.map(r=>r[4+a])),mins=values.map(v=>Math.min(...v)),maxs=values.map(v=>Math.max(...v)),center=mins.map((v,i)=>(v+maxs[i])/2),range=Math.max(...mins.map((v,i)=>Math.max(maxs[i]-v,1e-8))),radius=Math.min(w,h)*.36*zoom;const tf=p=>{let x=(p[0]-center[0])*2/range,y=(p[1]-center[1])*2/range,z=(p[2]-center[2])*2/range;const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),x1=cy*x+sy*z,z1=-sy*x+cy*z,y1=cp*y-sp*z1,z2=sp*y+cp*z1;return{x:w/2+x1*radius,y:h/2-y1*radius,z:z2}};const roles={prompt_occurrence:[],answer_query:[]};for(const r of rows)roles[r[1]].push({count:r[2],q:tf(axes.map(a=>r[4+a]))});for(const k of Object.keys(roles))roles[k].sort((a,b)=>a.count-b.count);const aq=new Map(roles.answer_query.map(a=>[a.count,a]));ctx.strokeStyle='rgba(214,181,44,.35)';ctx.lineWidth=1;for(const p of roles.prompt_occurrence){const a=aq.get(p.count);if(a){ctx.beginPath();ctx.moveTo(p.q.x,p.q.y);ctx.lineTo(a.q.x,a.q.y);ctx.stroke()}}for(const [role,items] of Object.entries(roles)){ctx.strokeStyle=role==='prompt_occurrence'?'#FFFDF8':'#D6B52C';ctx.lineWidth=2.4;ctx.setLineDash(role==='prompt_occurrence'?[]:[7,5]);ctx.beginPath();items.forEach((p,i)=>i?ctx.lineTo(p.q.x,p.q.y):ctx.moveTo(p.q.x,p.q.y));ctx.stroke();ctx.setLineDash([]);for(const p of items){ctx.fillStyle=COUNT_COLORS[p.count-1];ctx.strokeStyle=role==='prompt_occurrence'?'#FFFDF8':'#D6B52C';if(role==='prompt_occurrence'){ctx.beginPath();ctx.arc(p.q.x,p.q.y,5.5,0,Math.PI*2);ctx.fill();ctx.stroke()}else{ctx.fillRect(p.q.x-4.5,p.q.y-4.5,9,9);ctx.strokeRect(p.q.x-4.5,p.q.y-4.5,9,9)}}}document.getElementById('joint-stats').textContent=`linear CKA ${d.trajectory_linear_cka.toFixed(3)} · centroid-distance r ${d.trajectory_distance_correlation.toFixed(3)} · adjacent-step cosine ${d.successive_step_cosine.toFixed(3)}`}
 function resize(){const rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);draw()}model.addEventListener('change',()=>{refresh();draw()});[layer,xSel,ySel,zSel].forEach(e=>e.addEventListener('change',draw));canvas.addEventListener('pointerdown',e=>{drag=true;lx=e.clientX;ly=e.clientY;canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-lx)*.008;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-ly)*.008));lx=e.clientX;ly=e.clientY;draw()});canvas.addEventListener('pointerup',()=>drag=false);canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.5,Math.min(2.5,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});refresh();new ResizeObserver(resize).observe(canvas);resize();}

makeProjector('prompt',PROMPT_DATA,'prompt');makeProjector('answer',ANSWER_DATA,'answer');makeJoint();
document.querySelectorAll('[data-atlas]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('[data-atlas]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));document.querySelectorAll('.atlas-panel').forEach(panel=>panel.hidden=panel.dataset.atlasPanel!==button.dataset.atlas)}));
</script>
</body></html>"""


def _projector_controls(prefix: str, *, answer: bool) -> str:
    extra = (
        f'<label>PCA fit<select id="{prefix}-fit"><option value="all">all rows</option><option value="correct_only">correct-only sensitivity</option></select></label>'
        if answer
        else f'<label>split<select id="{prefix}-split"><option value="all">all</option><option value="discovery">discovery</option><option value="confirmation">confirmation</option></select></label>'
    )
    return f"""
<div class="controls">
<label>model<select id="{prefix}-model"><option>Qwen3-8B</option><option>Gemma4-E4B</option></select></label>
<label>layer<select id="{prefix}-layer"></select></label>{extra}
<label>outcome<select id="{prefix}-outcome"><option value="all">all</option><option value="correct">correct</option><option value="wrong">wrong</option><option value="invalid">invalid</option></select></label>
<label>points<select id="{prefix}-points"><option value="all">points + centroids</option><option value="centroids">centroids only</option></select></label>
<label>x<select id="{prefix}-x"><option value="0">PC1</option><option value="1">PC2</option><option value="2">PC3</option><option value="3">PC4</option><option value="4">PC5</option><option value="5">PC6</option></select></label>
<label>y<select id="{prefix}-y"><option value="1">PC2</option><option value="0">PC1</option><option value="2">PC3</option><option value="3">PC4</option><option value="4">PC5</option><option value="5">PC6</option></select></label>
<label>z<select id="{prefix}-z"><option value="2">PC3</option><option value="0">PC1</option><option value="1">PC2</option><option value="3">PC4</option><option value="4">PC5</option><option value="5">PC6</option></select></label>
</div><div class="plot-shell"><canvas id="{prefix}-canvas"></canvas><div class="plot-tooltip" id="{prefix}-tooltip"></div></div><div class="plot-stats" id="{prefix}-stats"></div>
"""


def _joint_controls() -> str:
    return """
<div class="controls"><label>model<select id="joint-model"><option>Qwen3-8B</option><option>Gemma4-E4B</option></select></label><label>layer<select id="joint-layer"></select></label><label>x<select id="joint-x"><option value="0">PC1</option><option value="1">PC2</option><option value="2">PC3</option></select></label><label>y<select id="joint-y"><option value="1">PC2</option><option value="0">PC1</option><option value="2">PC3</option></select></label><label>z<select id="joint-z"><option value="2">PC3</option><option value="0">PC1</option><option value="1">PC2</option></select></label></div><div class="plot-shell"><canvas id="joint-canvas"></canvas></div><div class="plot-stats" id="joint-stats"></div>
"""


def build_report(run_root: Path, output: Path, repo_root: Path) -> None:
    run_root = run_root.resolve()
    probe_layers: dict[str, dict[str, int]] = {}
    labels_lookup: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    labels_frames: dict[str, pd.DataFrame] = {}
    all_labels: dict[str, pd.DataFrame] = {}
    layer_rows: list[dict[str, Any]] = []
    prompt_projections: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        model_root = run_root / model / "numeric"
        probe_layers[model] = full._primary_layers(model_root)
        labels_lookup[model], labels_frames[model] = full._n10_labels(model_root)
        all_labels[model] = full._all_generation_labels(model_root)
        sweep, _display, pca_models = full._layer_sweep(
            model_root, model=model, probe_layers=probe_layers[model]
        )
        layer_rows.extend(sweep)
        prompt_projections.update(
            full._load_prompt_projection_layers(
                model_root,
                model=model,
                labels=labels_lookup[model],
                pca_models=pca_models,
                layer_rows=sweep,
            )
        )

    prompt_projections = _filter_projection_rows(prompt_projections)
    answer_projection_data = full._answer_query_projection_data(run_root)
    answer_projections = _filter_projection_rows(answer_projection_data)
    joint_projections = full._joint_counter_projection_data(
        run_root, answer_projection_data
    )

    atlas = [
        row
        for row in full._attention_head_atlas_rows(run_root)
        if row["variant"] == FOCUS_VARIANT
    ]
    phenotypes = [
        row
        for row in full._attention_head_phenotypes(run_root)
        if row["variant"] == FOCUS_VARIANT
    ]
    outcomes = [
        row
        for row in full._attention_outcome_effect_rows(run_root)
        if row["design_variant"] == FOCUS_VARIANT
    ]
    alignments = [
        row
        for row in full._span_end_alignment_rows(run_root)
        if row["variant"] == FOCUS_VARIANT
    ]
    nested = _nested_v44_rows(run_root)

    causal_frames, _paths = full._causal_frames(run_root)
    v44_causal = _variant_only_frames(causal_frames)
    ablation = full._causal_ablation_rows(v44_causal)
    endpoint_patch = full._causal_patching_rows(v44_causal)
    steering_v1 = full._causal_steering_rows(v44_causal)
    steering_v2 = _steering_v2_rows(run_root)
    answer_frames, _audit = full._answer_query_frames(run_root)
    answer_patch = answer_frames["variant_summary"]
    answer_patch = answer_patch[
        answer_patch["design_variant"].astype(str) == FOCUS_VARIANT
    ].copy()

    atlas_html = (
        '<div class="switcher"><button type="button" data-atlas="span_end" aria-pressed="false">endpoint-key mass</button>'
        '<button type="button" data-atlas="span_sum" aria-pressed="true">full-span literal mass</button></div>'
        '<div class="atlas-panel" data-atlas-panel="span_end" hidden>'
        + full._attention_head_atlas_svg(
            atlas, phenotypes, variant=FOCUS_VARIANT, pooling="span_end"
        )
        + '</div><div class="atlas-panel" data-atlas-panel="span_sum">'
        + full._attention_head_atlas_svg(
            atlas, phenotypes, variant=FOCUS_VARIANT, pooling="span_sum"
        )
        + "</div>"
    )

    answer_forest_rows = answer_patch.to_dict("records")
    replacements = {
        "@@RUN@@": html.escape(run_root.name),
        "@@GENERATED@@": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "@@COMMIT@@": html.escape(full._git_commit(repo_root)[:12]),
        "@@INVENTORY@@": _inventory_table(),
        "@@BEHAVIOR_TABLE@@": _details_table(
            title="V4.4 confirmation behavior context",
            columns=["model", "gold N", "correct", "accuracy", "mean prediction", "undercount rate"],
            rows=_behavior_rows(all_labels),
        ),
        "@@PROMPT_LAYER_TABLE@@": _details_table(
            title="Frozen prompt span-end layer selection",
            columns=["model", "layer", "role", "full-space CV R²", "EVR PC1–3", "count-signal capture", "seed compactness"],
            rows=_prompt_selection_rows(layer_rows),
        ),
        "@@ANSWER_LAYER_TABLE@@": _details_table(
            title="Frozen answer-query layer selection",
            columns=["model", "layer", "role", "EVR PC1–3", "PCA3 CV R²", "count-signal capture", "seed compactness"],
            rows=_answer_selection_rows(answer_projections),
        ),
        "@@PROMPT_FIGURE@@": _figure(
            "1.1 Interactive V4.4 prompt counter",
            "图中每个小点是一个 seed 在第 n 条 needle 末尾的 residual；彩色大点和白线是同一 n 的 centroid path。拖拽旋转、滚轮缩放，并可切换所有保存 layer。",
            _projector_controls("prompt", answer=False),
            "Prompt-reading needle-end counter. 三个坐标轴是冻结 V4.1-discovery PCA 的可切换 PC score；颜色从 indigo 到 pink 对应 n=1…10。白边点的整条 N=10 trajectory 最终回答正确，黑边为错误，粉边为 invalid。PCA 只决定显示坐标，不参与行为标签。",
        ),
        "@@ANSWER_FIGURE@@": _figure(
            "2.1 Interactive V4.4 answer-query counter",
            "每个点是一个独立 V4.4 prompt 在 Total: query 的 residual。Layer selector 包含 Qwen L0–L35 与 Gemma L0–L41；all/correct-only 只改变 PCA fit cohort，不改变被投影的 V4.4 states。",
            _projector_controls("answer", answer=True),
            "Answer-query counter. 坐标轴是对应 layer 与 fit cohort 的 PC score；颜色是 gold count 1…10，边框是该 prompt 的完整 greedy outcome。Centroid path 对所有当前筛选点取均值，因此切换 correct-only outcome 时缺失 count 会直接不画，而不会伪造 centroid。",
        ),
        "@@JOINT_FIGURE@@": _figure(
            "2.2 Prompt 与 answer counter 的共同坐标",
            "为避免 token-role 的固定均值差制造虚假一致性，先分别减去 prompt/answer role mean，再在配对 V4.1 discovery states 上拟合 shared PCA；这里只显示 V4.4 centroids。",
            _joint_controls(),
            "Role-centered joint geometry. 圆点/实线是 prompt needle-end centroid trajectory，方块/虚线是 answer-query trajectory；同颜色细线只连接相同 count。三个轴是 shared PC scores。图下的 linear CKA、centroid-distance correlation 和 adjacent-step cosine 均在 full residual space 计算，不依赖 PCA 旋转。",
        ),
        "@@ATTENTION_ATLAS@@": _figure(
            "3.1 All-head V4.4 atlas",
            "用按钮切换 endpoint-key 与 full-span-key pooling。横轴是 post-block decoder layer，纵轴是 head index；只有 full-attention layers 才存在完整行。",
            atlas_html,
            "V4.4 answer-query attention atlas. 每格对应一个 layer/head，hover title 给出 primary score。颜色按当前 pooling 的 discovery log primary score 独立缩放；因此 endpoint 与 span-sum 的颜色不可当成相同绝对 mass。候选/phenotype symbol 只帮助定位，不能替代 causal test。",
        ),
        "@@FIRST_LOCATOR_APPENDIX@@": _first_locator_appendix(repo_root),
        "@@OUTCOME_TABLE@@": _details_table(
            title="V4.4 count-adjusted wrong−correct attention effects",
            columns=["model", "pooling", "rank/head", "metric", "wrong−correct [95% CI]", "CI excludes 0"],
            rows=_outcome_rows(outcomes),
        ),
        "@@OMISSION_TABLE@@": _details_table(
            title="V4.4 undercount-tail and nested-increment diagnostics",
            columns=["model", "diagnostic", "value A", "value B", "interpretation"],
            rows=_omission_rows(alignments, nested),
        ),
        "@@ABLATION_SVG@@": full._forest_svg(
            ablation,
            estimate_key="count_shift_difference",
            low_key="count_shift_difference_low",
            high_key="count_shift_difference_high",
            title="V4.4 ranked head-bank ablation versus layer-matched random",
            axis_label="paired count shift: ranked minus random",
            label=lambda row: f"{row['model']} top-{row['top_n']}",
        ),
        "@@ABLATION_TABLE@@": _details_table(
            title="V4.4 head-ablation estimates",
            columns=["model", "set", "prompts", "changed ranked/random", "Δ count shift [CI]", "Δ MAE [CI]", "Holm p"],
            rows=_ablation_rows(ablation),
        ),
        "@@ABLATION_RESULT@@": _ablation_result_html(ablation),
        "@@ENDPOINT_PATCH_SVG@@": full._forest_svg(
            endpoint_patch,
            estimate_key="aligned_shift",
            low_key="aligned_shift_low",
            high_key="aligned_shift_high",
            title="V4.4 exact toggled-needle-end transport",
            axis_label="direction-aligned generated-count shift",
            label=lambda row: f"{row['model']} L{row['layer']}→final",
        ),
        "@@ENDPOINT_PATCH_TABLE@@": _details_table(
            title="V4.4 exact needle-end patch estimates",
            columns=["model", "clamp", "rows", "changed", "moved toward donor [CI]", "aligned shift [CI]", "Holm p"],
            rows=_endpoint_patch_rows(endpoint_patch),
        ),
        "@@ENDPOINT_PATCH_RESULT@@": _endpoint_patch_result_html(endpoint_patch),
        "@@ANSWER_PATCH_SVG@@": full._forest_svg(
            answer_forest_rows,
            estimate_key="eligible_donor_adoption_rate",
            low_key="eligible_donor_adoption_rate_ci95_low",
            high_key="eligible_donor_adoption_rate_ci95_high",
            title="V4.4 exact answer-query donor-prediction transport",
            axis_label="eligible donor-prediction adoption rate",
            label=lambda row: f"{row['model']} L{int(row['layer'])}",
        ),
        "@@ANSWER_PATCH_TABLE@@": _details_table(
            title="V4.4 answer-query patch layer sweep",
            columns=["model", "layer", "rows", "eligible", "donor adoption [CI]", "aligned shift [CI]", "valid"],
            rows=_answer_patch_rows(answer_patch),
        ),
        "@@ANSWER_PATCH_RESULT@@": _answer_patch_result_html(answer_patch),
        "@@STEERING_V1_SVG@@": full._forest_svg(
            steering_v1,
            estimate_key="aligned_difference",
            low_key="aligned_difference_low",
            high_key="aligned_difference_high",
            title="V4.4 centroid-delta steering versus norm-matched random",
            axis_label="paired aligned count shift: geometric minus random",
            label=lambda row: f"{row['model']} L{row['layer']}",
        ),
        "@@STEERING_V1_TABLE@@": _details_table(
            title="V4.4 steering-v1 estimates",
            columns=["model", "layer", "changed geo/random", "moved geo/random", "Δ aligned [CI]", "target-hit geo/random", "Holm p"],
            rows=_steering_v1_rows(steering_v1),
        ),
        "@@STEERING_V1_RESULT@@": _steering_v1_result_html(steering_v1),
        "@@STEERING_V2_SVG@@": full._forest_svg(
            steering_v2,
            estimate_key="aligned_effect",
            low_key="aligned_effect_low",
            high_key="aligned_effect_high",
            title="Discovery-locked steering on held-out V4.4",
            axis_label="strict aligned count shift: geometric minus random",
            label=lambda row: f"{row['model']} {row['protocol'].replace('_','-')} {row['layer_set']}",
        ),
        "@@STEERING_V2_TABLE@@": _details_table(
            title="Held-out V4.4 steering-v2 estimates",
            columns=["model", "protocol", "layers", "α", "Δ aligned [CI]", "Δ moved [CI]", "Δ exact target [CI]", "Holm p"],
            rows=_steering_v2_table(steering_v2),
        ),
        "@@STEERING_V2_RESULT@@": _steering_v2_result_html(steering_v2),
        "@@CAUSAL_SUMMARY@@": _causal_summary_html(
            ablation, endpoint_patch, answer_patch, steering_v2
        ),
        "@@PROMPT_DATA@@": json.dumps(
            _prompt_payload(prompt_projections), ensure_ascii=False, separators=(",", ":")
        ),
        "@@ANSWER_DATA@@": json.dumps(
            _answer_payload(answer_projections), ensure_ascii=False, separators=(",", ":")
        ),
        "@@JOINT_DATA@@": json.dumps(
            _joint_payload(joint_projections), ensure_ascii=False, separators=(",", ":")
        ),
    }
    rendered = REPORT_TEMPLATE
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    leftovers = sorted(set(re.findall(r"@@[A-Z0-9_]+@@", rendered)))
    if leftovers:
        raise RuntimeError(f"Unresolved report placeholders: {leftovers}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")

    table_root = output.parent / "v4_non-thinking_causal" / "v4_4"
    table_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ablation).to_csv(table_root / "v4_4_head_ablation.csv", index=False)
    pd.DataFrame(endpoint_patch).to_csv(
        table_root / "v4_4_needle_end_patching.csv", index=False
    )
    answer_patch.to_csv(table_root / "v4_4_answer_query_patching.csv", index=False)
    pd.DataFrame(steering_v1).to_csv(
        table_root / "v4_4_steering_v1.csv", index=False
    )
    pd.DataFrame(steering_v2).to_csv(
        table_root / "v4_4_steering_v2.csv", index=False
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "bytes": output.stat().st_size,
                "variant": FOCUS_VARIANT,
                "prompt_projection_panels": len(_prompt_payload(prompt_projections)),
                "answer_projection_panels": len(_answer_payload(answer_projections)),
                "attention_atlas_rows": len(atlas),
                "causal_rows": {
                    "ablation": len(ablation),
                    "needle_end_patching": len(endpoint_patch),
                    "answer_query_patching": len(answer_patch),
                    "steering_v1": len(steering_v1),
                    "steering_v2": len(steering_v2),
                },
                "tables": str(table_root.resolve()),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a self-contained V4.4-only representation-to-causality report."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    build_report(args.run_root, args.output, args.repo_root)


if __name__ == "__main__":
    main()
