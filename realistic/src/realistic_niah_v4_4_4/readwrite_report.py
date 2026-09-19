from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from realistic_niah_v4_4_3.io import atomic_text, stage_root

from .readwrite_spec import V444ReadWriteConfig


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}g}"


def _polyline_svg(
    rows: pd.DataFrame,
    *,
    metrics: Sequence[tuple[str, str, str]],
    width: int = 880,
    height: int = 330,
) -> str:
    selected = rows[rows["metric"].isin([item[0] for item in metrics])].copy()
    selected = selected[selected["stratum"].eq("all") & selected["layer"].notna()]
    if selected.empty:
        return "<p>No downstream trace rows.</p>"
    layers = sorted(int(value) for value in selected["layer"].unique())
    values = selected["mean"].to_numpy(float)
    low = float(min(0.0, np.nanmin(values)))
    high = float(max(0.0, np.nanmax(values)))
    if high - low < 1e-9:
        high = low + 1.0
    left, right, top, bottom = 64, width - 24, 24, height - 48

    def x(layer: int) -> float:
        if len(layers) == 1:
            return (left + right) / 2
        return left + (layer - layers[0]) / (layers[-1] - layers[0]) * (right - left)

    def y(value: float) -> float:
        return bottom - (value - low) / (high - low) * (bottom - top)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Downstream write propagation by layer">',
        f'<rect width="{width}" height="{height}" fill="#fbfaf6" rx="12"/>',
        f'<line x1="{left}" y1="{y(0):.2f}" x2="{right}" y2="{y(0):.2f}" '
        'stroke="#b8b2a7" stroke-dasharray="4 4"/>',
    ]
    for layer in layers:
        parts.append(
            f'<text x="{x(layer):.2f}" y="{bottom + 24}" text-anchor="middle" '
            f'font-size="12" fill="#524b43">L{layer}</text>'
        )
    for index in range(5):
        value = low + index / 4 * (high - low)
        parts.append(
            f'<text x="{left - 10}" y="{y(value) + 4:.2f}" text-anchor="end" '
            f'font-size="11" fill="#756d64">{value:.3g}</text>'
        )
    for metric, label, color in metrics:
        series = selected[selected["metric"].eq(metric)].set_index("layer")
        points = [
            (x(layer), y(float(series.loc[layer, "mean"])))
            for layer in layers
            if layer in series.index
        ]
        if not points:
            continue
        path = " ".join(
            ("M" if offset == 0 else "L") + f" {px:.2f} {py:.2f}"
            for offset, (px, py) in enumerate(points)
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        for px, py in points:
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="{color}"/>')
        parts.append(
            f'<text x="{right - 4}" y="{top + 16 + 18 * metrics.index((metric, label, color))}" '
            f'text-anchor="end" font-size="12" fill="{color}">{html.escape(label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _table(frame: pd.DataFrame, columns: Sequence[tuple[str, str]]) -> str:
    header = "".join(f"<th>{html.escape(label)}</th>" for _name, label in columns)
    rows = []
    for record in frame.to_dict("records"):
        cells = "".join(
            f"<td>{_fmt(record.get(name))}</td>" for name, _label in columns
        )
        rows.append(f"<tr>{cells}</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def _partition_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, seed), values in frame.groupby(
        ["position_group", "seed"], sort=True
    ):
        x = values["gold_count"].to_numpy(float)
        y = values["global_axis_contribution_coefficient"].to_numpy(float)
        centered = x - x.mean()
        slope = float(np.dot(centered, y - y.mean()) / np.dot(centered, centered))
        rows.append({"position_group": group, "seed": int(seed), "slope": slope})
    result = pd.DataFrame(rows)
    return (
        result.groupby("position_group", sort=True)["slope"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_count_slope", "std": "seed_sd", "count": "seeds"})
    )


def _mechanical_position_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in sorted(frame["position_group"].unique()):
        selected = frame[frame["position_group"].eq(group)]
        for component in ("full", "value", "routing"):
            column = f"{component}_mechanical_transport"
            seed_values = selected.groupby("seed", sort=True)[column].mean()
            rows.append(
                {
                    "position_group": group,
                    "component": component,
                    "mean_normalized_transport": seed_values.mean(),
                    "seed_sd": seed_values.std(),
                    "positive_seed_fraction": (seed_values > 0).mean(),
                    "seeds": len(seed_values),
                }
            )
    return pd.DataFrame(rows)


def build_html_report(
    *, run_root: str | Path, config: V444ReadWriteConfig
) -> Path:
    root = Path(run_root)
    analysis_root = stage_root(root, config.model_label, "read_write_analysis")
    payload = json.loads(
        (
            analysis_root / "realistic_niah_v4_4_4_read_write_analysis.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    summary = pd.read_csv(analysis_root / "metric_summary.csv.gz")
    discovery_root = stage_root(root, config.model_label, "read_write_discovery")
    partition = pd.read_csv(discovery_root / "natural_partition_detail.csv.gz")
    partition_summary = _partition_slopes(partition)
    evaluation_root = stage_root(root, config.model_label, "read_write_evaluation")
    mechanical = pd.read_csv(
        evaluation_root / "read_mechanical_detail.csv.gz"
    )
    mechanical_position = _mechanical_position_summary(mechanical)
    anchored = mechanical[mechanical["position_group"].eq("all_positions")]
    numerical_audit = pd.DataFrame(
        [
            {
                "metric": "receiver eager endpoint relative L2",
                "maximum": anchored[
                    "receiver_endpoint_reconstruction_relative_l2"
                ].max(),
                "gate": config.edge_reconstruction_relative_tolerance,
            },
            {
                "metric": "donor eager endpoint relative L2",
                "maximum": anchored[
                    "donor_endpoint_reconstruction_relative_l2"
                ].max(),
                "gate": config.edge_reconstruction_relative_tolerance,
            },
            {
                "metric": "anchored Shapley closure relative L2",
                "maximum": anchored["closure_relative_l2"].max(),
                "gate": config.closure_relative_tolerance,
            },
        ]
    )
    decision = payload["primary_decision"]
    read_decision = decision["read_mode"]
    write_decision = decision["write_propagation"]
    read_metrics = summary[
        summary["metric"].isin(
            [
                "read_full_behavior_transport",
                "read_value_behavior_transport",
                "read_routing_behavior_transport",
                "read_value_ov_mediation_specificity",
                "read_routing_ov_mediation_specificity",
                "read_value_minus_routing_transport",
            ]
        )
        & summary["stratum"].eq("all")
    ][
        ["metric", "mean", "ci95_low", "ci95_high", "exact_sign_flip_p", "positive_seed_fraction"]
    ]
    outcome = summary[
        summary["family"].eq("read_outcome_sensitivity")
        & summary["metric"].isin(
            ["read_full_behavior_transport", "read_value_behavior_transport", "read_routing_behavior_transport"]
        )
    ][["metric", "stratum", "seed_count", "mean", "ci95_low", "ci95_high"]]
    write_svg = _polyline_svg(
        summary,
        metrics=(
            ("write_natural_residual_slope", "natural OV", "#d1495b"),
            ("write_orthogonal_residual_slope", "in-span orthogonal", "#287271"),
            ("write_residual_specificity", "specificity", "#6c4ab6"),
        ),
    )
    style = """
    :root{--ink:#211d19;--muted:#6d655d;--paper:#f2efe7;--card:#fffdf8;--line:#d8d1c5;--accent:#6c4ab6}
    *{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,Segoe UI,system-ui,sans-serif;line-height:1.55}
    main{max-width:1180px;margin:auto;padding:36px 28px 80px}h1{font-family:Georgia,serif;font-size:2.4rem;margin:.1em 0}h2{margin-top:2.1em;border-bottom:1px solid var(--line);padding-bottom:.3em}
    .hero,.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:24px;margin:18px 0;box-shadow:0 8px 28px rgba(54,43,31,.06)}
    .kpis{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:20px}.kpi{background:#f6f2ff;border-radius:12px;padding:16px}.kpi span{display:block;color:var(--muted);font-size:.85rem}.kpi strong{font-size:1.25rem}
    code{background:#eee9df;padding:.12em .35em;border-radius:4px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:.9rem}th{text-align:left;background:#eee9df}th,td{padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
    .formula{font-family:Cambria Math,Georgia,serif;background:#f7f4ed;border-left:4px solid var(--accent);padding:14px 18px}.warning{border-left:4px solid #d18b28;padding:12px 16px;background:#fff7e8}
    @media(max-width:760px){.kpis{grid-template-columns:1fr}main{padding:20px 14px}}
    """
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V4.4.4 Supplement · Read / OV Write</title><style>{style}</style></head><body><main>
    <section class="hero"><p>Realistic NIAH · Qwen3-8B · L28 H16/H19</p><h1>V4.4.4 补充：模型怎样读取 state，并通过 OV 写回</h1>
    <p>本报告把 donor→receiver 的 attention 变化精确拆成 V-content 与 alpha-routing，并追踪自然 OV step 从 L28 到最终层的 residual 传播。</p>
    <div class="kpis"><div class="kpi"><span>read mode</span><strong>{html.escape(str(read_decision['classification']))}</strong></div><div class="kpi"><span>write propagation</span><strong>{write_decision['supported']}</strong></div><div class="kpi"><span>serial read→write</span><strong>{decision['serial_read_write_supported']}</strong></div></div></section>
    <p class="warning"><strong>证据边界：</strong>这是使用父 V4.4.4 evaluation seeds 的机制补充，不是全新 seed 的独立复现。结果冻结后仍需 1294–1313 replication。</p>
    <h2>1. 读取分解</h2><div class="card"><p class="formula">Δz<sub>full</sub> = Δz<sub>value</sub> + Δz<sub>route</sub></p><p>value 是在 receiver/donor routing 上平均后的 V-state 改变；route 是在 receiver/donor V 上平均后的 alpha 改变。主分析以真实 fused pre-O receiver/donor state 为两个端点，因此二者严格重构实际 donor-Z patch；crossed 端点由显式 alpha-V 计算。闭合是逐样本数值门，而非相关性。</p>
    {_table(read_metrics, (("metric","metric"),("mean","mean"),("ci95_low","CI low"),("ci95_high","CI high"),("exact_sign_flip_p","p"),("positive_seed_fraction","positive seeds")))}<h3>数值桥审计</h3><p>前两行衡量 eager alpha-V 端点与模型实际 fused pre-O state 的差异；第三行检验锚定后的 value+route 是否闭合到真实 donor-Z。</p>{_table(numerical_audit, (("metric","metric"),("maximum","max observed"),("gate","required ≤")))}</div>
    <h2>2. state 位于哪些位置</h2><div class="card"><p>四个位置组互斥并覆盖完整 key axis。第一张表的 slope 是自然 forward 中，该组经 H16/H19 写到 frozen OV axis 的 count slope。第二张表把 directed donor movement 除以 donor-receiver count gap：正值表示该组把 read state 朝 donor count 推动。二者都是 attribution；在没有组级 removal 前，不单独当作 necessity 证据。</p>{_table(partition_summary, (("position_group","position group"),("mean_count_slope","natural count slope"),("seed_sd","seed SD"),("seeds","seeds")))}{_table(mechanical_position, (("position_group","position group"),("component","read component"),("mean_normalized_transport","mean normalized transport"),("seed_sd","seed SD"),("positive_seed_fraction","positive seeds"),("seeds","seeds")))}</div>
    <h2>3. OV 写入后的层间传播</h2><div class="card">{write_svg}<p>横轴为 post-block layer；纵轴为 ±β 中央差分在该层自然 answer-query count axis 上的 coefficient。natural 与同 H16/H19 W<sub>O</sub> span、等 post-O 范数正交 control 比较。</p></div>
    <h2>4. 正确与错误 baseline 的敏感性</h2><div class="card"><p>所有主检验仍使用完整 paired trials；此表仅在固定轴上分层，不重新拟合 PCA/count axis。</p>{_table(outcome, (("metric","metric"),("stratum","stratum"),("seed_count","seed n"),("mean","mean"),("ci95_low","CI low"),("ci95_high","CI high")))}</div>
    <h2>5. 当前能得到的结论</h2><div class="card"><p>只有当 read component 本身推动 count、其 effect 被 frozen natural OV axis block 特异性削弱、并且自然 OV write 在下游 count state 与答案分布中持续存在时，才支持完整的 terminal read→write 路径。</p><p>即便通过，也不定位构造 source value-state 的上游 heads/MLP；那是下一轮 layerwise residual path patch 的问题。</p><p>Audit: <strong>{payload['audit']['all_checks_pass']}</strong> ({payload['audit']['check_count']} checks).</p></div>
    </main></body></html>"""
    output = analysis_root / "realistic_niah_v4_4_4_read_write_report.html"
    atomic_text(output, document)
    return output
