#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_COLORS = {
    "Qwen3-8B": "#6750E8",
    "Gemma4-E4B": "#00A88F",
}
ROLE_COLORS = {"candidate": "#D94B86", "matched_control": "#718096"}


def _figure_uri(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight", facecolor="#FFFDF8")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_by_seed(
    frame: pd.DataFrame,
    value: str,
    *,
    repetitions: int = 4000,
    seed: int = 443,
) -> tuple[float, float, float]:
    values = frame.groupby("seed")[value].mean().to_numpy(float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean, mean
    generator = np.random.default_rng(seed)
    boot = generator.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return mean, float(low), float(high)


def _mapping_figure(model_root: Path, model: str) -> str:
    frame = pd.read_csv(model_root / "mapping" / "head_mapping_scores.csv.gz")
    selection = _read_json(model_root / "mapping" / "head_selection.json")
    candidates = {(int(row["layer"]), int(row["head"])) for row in selection["candidate_heads"]}
    controls = {
        (int(row["layer"]), int(row["head"]))
        for row in selection["matched_control_heads"].values()
    }
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.scatter(
        frame["fit_mapping_cosine"],
        frame["heldout_count_mapping_cosine"],
        s=24,
        color="#B7B0A6",
        alpha=0.55,
        label="all registered heads",
    )
    for row in frame.itertuples():
        key = (int(row.layer), int(row.head))
        if key not in candidates and key not in controls:
            continue
        is_candidate = key in candidates
        color = ROLE_COLORS["candidate" if is_candidate else "matched_control"]
        marker = "o" if is_candidate else "X"
        ax.scatter(
            [row.fit_mapping_cosine],
            [row.heldout_count_mapping_cosine],
            s=110,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=1.1,
            zorder=3,
        )
        ax.annotate(
            f"L{int(row.layer)}H{int(row.head)}",
            (row.fit_mapping_cosine, row.heldout_count_mapping_cosine),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(0, color="#6E675E", linewidth=0.8)
    ax.axvline(0, color="#6E675E", linewidth=0.8)
    ax.set_xlabel("fit-count OV mapping cosine (used for ranking)")
    ax.set_ylabel("held-out-count OV mapping cosine")
    ax.set_title(f"{model}: prompt direction mapped through head OV")
    ax.grid(alpha=0.2)
    ax.text(
        0.01,
        0.99,
        "upper-right = same signed mapping on fit and held-out counts",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color="#5E6672",
    )
    return _figure_uri(fig)


def _patch_figure(model_root: Path, model: str) -> str:
    frame = pd.read_csv(model_root / "analysis" / "patch_contrasts.csv.gz")
    order = []
    for parent in sorted(frame["parent_candidate"].unique()):
        for role in ("candidate", "matched_control"):
            if ((frame["parent_candidate"] == parent) & (frame["head_role"] == role)).any():
                order.append((parent, role))
    metrics = (
        ("z_minus_norm_control_transport", "Z donor - equal-norm control"),
        ("alpha_minus_scramble_transport", "alpha donor - position scramble"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.2), sharex=True)
    x = np.arange(len(order), dtype=float)
    for ax, (metric, title) in zip(axes, metrics):
        means, lows, highs, colors = [], [], [], []
        for parent, role in order:
            row = frame[
                frame["parent_candidate"].eq(parent)
                & frame["head_role"].eq(role)
                & frame["contrast_name"].eq(metric)
            ].iloc[0]
            means.append(float(row["mean_contrast"]))
            lows.append(float(row["contrast_ci95_low"]))
            highs.append(float(row["contrast_ci95_high"]))
            colors.append(ROLE_COLORS[role])
        means_array = np.asarray(means)
        errors = np.vstack((means_array - np.asarray(lows), np.asarray(highs) - means_array))
        ax.bar(x, means_array, color=colors, width=0.72)
        ax.errorbar(x, means_array, yerr=errors, fmt="none", ecolor="#20242D", capsize=3)
        ax.axhline(0, color="#20242D", linewidth=0.9)
        ax.set_ylabel("normalized count transport contrast")
        ax.set_title(title + " (positive supports the registered prediction)")
        ax.grid(axis="y", alpha=0.2)
    labels = [f"{parent}\n{'selected' if role == 'candidate' else 'control'}" for parent, role in order]
    axes[-1].set_xticks(x, labels, rotation=0)
    fig.suptitle(f"{model}: staged causal decomposition", y=1.01, fontsize=14)
    fig.tight_layout()
    return _figure_uri(fig)


def _injection_figure(model_root: Path, model: str) -> str:
    frame = pd.read_csv(model_root / "directed" / "detail.csv.gz")
    frame = frame[frame["intervention"].astype(str).str.startswith("signed_answer_direction_injection")]
    parents = sorted(frame["parent_candidate"].unique())
    fig, axes = plt.subplots(1, len(parents), figsize=(4.6 * len(parents), 4.2), sharey=True)
    if len(parents) == 1:
        axes = [axes]
    for panel_index, (ax, parent) in enumerate(zip(axes, parents)):
        for role in ("candidate", "matched_control"):
            group = frame[frame["parent_candidate"].eq(parent) & frame["head_role"].eq(role)]
            if group.empty:
                continue
            points = []
            for beta, beta_group in group.groupby("beta", sort=True):
                mean, low, high = _bootstrap_by_seed(
                    beta_group,
                    "delta_expected_count",
                    # This intervention is applied at the layer post-O residual,
                    # not inside a head.  Candidate/matched rows are registered
                    # duplicates, so use the same resample as an explicit audit.
                    seed=443500 + panel_index * 31,
                )
                points.append((float(beta), mean, low, high))
            values = np.asarray(points, dtype=float)
            color = ROLE_COLORS[role]
            label = "selected registration" if role == "candidate" else "matched registration"
            linestyle = "-" if role == "candidate" else "--"
            ax.plot(values[:, 0], values[:, 1], marker="o", color=color, linestyle=linestyle, label=label)
            ax.fill_between(values[:, 0], values[:, 2], values[:, 3], color=color, alpha=0.14)
        ax.axhline(0, color="#20242D", linewidth=0.8)
        ax.axvline(0, color="#20242D", linewidth=0.8)
        ax.set_title(parent)
        ax.set_xlabel("signed beta along answer-count direction")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("change in expected count")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"{model}: layer-level answer-subspace injection (non-localizing)",
        y=1.02,
        fontsize=14,
    )
    fig.tight_layout()
    return _figure_uri(fig)


def _removal_figure(model_root: Path, model: str) -> str:
    frame = pd.read_csv(model_root / "analysis" / "removal_summary.csv.gz")
    order = []
    for parent in sorted(frame["parent_candidate"].unique()):
        for role in ("candidate", "matched_control"):
            if ((frame["parent_candidate"] == parent) & (frame["head_role"] == role)).any():
                order.append((parent, role))
    definitions = (
        (
            "mean_delta_expected_count_absolute_error_answer_minus_orthogonal",
            "delta_expected_count_absolute_error_answer_minus_orthogonal_ci95_low",
            "delta_expected_count_absolute_error_answer_minus_orthogonal_ci95_high",
            "absolute-error contrast (positive supports)",
        ),
        (
            "mean_delta_correct_margin_answer_minus_orthogonal",
            "delta_correct_margin_answer_minus_orthogonal_ci95_low",
            "delta_correct_margin_answer_minus_orthogonal_ci95_high",
            "correct-margin contrast (negative supports)",
        ),
    )
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.0), sharex=True)
    x = np.arange(len(order), dtype=float)
    for ax, (mean_name, low_name, high_name, title) in zip(axes, definitions):
        means, lows, highs, colors = [], [], [], []
        for parent, role in order:
            row = frame[
                frame["parent_candidate"].eq(parent) & frame["head_role"].eq(role)
            ].iloc[0]
            means.append(float(row[mean_name]))
            lows.append(float(row[low_name]))
            highs.append(float(row[high_name]))
            colors.append(ROLE_COLORS[role])
        values = np.asarray(means)
        errors = np.vstack((values - np.asarray(lows), np.asarray(highs) - values))
        ax.bar(x, values, color=colors, width=0.72)
        ax.errorbar(x, values, yerr=errors, fmt="none", ecolor="#20242D", capsize=3)
        ax.axhline(0, color="#20242D", linewidth=0.9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(
        x,
        [f"{parent}\n{'selected' if role == 'candidate' else 'control'}" for parent, role in order],
    )
    fig.suptitle(f"{model}: answer-direction removal vs equal-norm orthogonal removal", y=1.01, fontsize=14)
    fig.tight_layout()
    return _figure_uri(fig)


def _decision_table(payload: dict[str, Any]) -> str:
    fields = (
        ("mapping_pass", "OV map"),
        ("z_transport_vs_norm_control_pass", "Z > norm ctrl"),
        ("alpha_vs_scramble_pass", "alpha > scramble"),
        ("signed_injection_pass", "layer inject"),
        ("direction_removal_vs_orthogonal_pass", "direction remove"),
        ("joint_ov_head_support", "frozen joint rule"),
        ("head_specific_directional_support", "selected > matched direction"),
        ("head_specific_exact_p_le_0_05", "all head-specific p≤.05"),
    )
    lines = ["<table><thead><tr><th>model</th><th>head</th>"]
    lines.extend(f"<th>{html.escape(label)}</th>" for _, label in fields)
    lines.append("</tr></thead><tbody>")
    for model, model_payload in payload["model_analyses"].items():
        for row in model_payload["candidate_decisions"]:
            lines.append(f"<tr><td>{html.escape(model)}</td><td>{html.escape(row['parent_candidate'])}</td>")
            for key, _label in fields:
                passed = bool(row[key])
                lines.append(
                    f"<td class={'pass' if passed else 'fail'}>{'PASS' if passed else 'FAIL'}</td>"
                )
            lines.append("</tr>")
    lines.append("</tbody></table>")
    return "".join(lines)


def _mapping_selection_table(run_root: Path, payload: dict[str, Any]) -> str:
    lines = [
        "<table><thead><tr><th>model</th><th>selected head</th><th>V source</th>",
        "<th>fit cosine</th><th>held-out cosine</th><th>within-layer rank</th>",
        "<th>same-layer empirical p</th><th>norm-matched empirical p</th>",
        "</tr></thead><tbody>",
    ]
    for model, model_payload in payload["model_analyses"].items():
        model_root = run_root / "models" / model
        frame = pd.read_csv(model_root / "mapping" / "head_mapping_scores.csv.gz")
        selection = _read_json(model_root / "mapping" / "head_selection.json")
        source_by_layer = {
            int(row["layer"]): int(row["value_source_layer"])
            for row in model_payload["mapping_estimands"]
        }
        for candidate in selection["candidate_heads"]:
            layer, head = int(candidate["layer"]), int(candidate["head"])
            row = frame[
                frame["layer"].astype(int).eq(layer)
                & frame["head"].astype(int).eq(head)
            ].iloc[0]
            lines.append(
                "<tr><td>{model}</td><td>L{layer}H{head}{sentinel}</td><td>L{source}</td>"
                "<td>{fit:+.5f}</td><td>{held:+.5f}</td><td>{rank}</td>"
                "<td>{same:.5f}</td><td>{norm:.5f}</td></tr>".format(
                    model=html.escape(model),
                    layer=layer,
                    head=head,
                    sentinel=" (sentinel)" if bool(candidate.get("sentinel")) else "",
                    source=source_by_layer[layer],
                    fit=float(row["fit_mapping_cosine"]),
                    held=float(row["heldout_count_mapping_cosine"]),
                    rank=int(row["fit_rank_within_layer"]),
                    same=float(row["same_layer_empirical_p"]),
                    norm=float(row["norm_matched_empirical_p"]),
                )
            )
    lines.append("</tbody></table>")
    return "".join(lines)


def _specificity_table(payload: dict[str, Any]) -> str:
    labels = {
        "z_transport": "Z transport",
        "alpha_localization": "alpha localization",
        "removal_absolute_error": "removal: absolute error",
        "removal_correct_margin": "removal: correct margin",
        "layer_injection_nonlocalization_audit": "layer injection duplicate audit",
    }
    lines = [
        "<table><thead><tr><th>model</th><th>registered head</th><th>contrast</th>",
        "<th>selected − matched</th><th>95% seed-bootstrap CI</th>",
        "<th>one-sided exact p</th><th>predicted sign</th></tr></thead><tbody>",
    ]
    for model, model_payload in payload["model_analyses"].items():
        for row in model_payload["head_specificity_contrasts"]:
            raw_p = row["one_sided_sign_flip_p"]
            p_value = np.nan if raw_p is None else float(raw_p)
            p_text = "—" if not np.isfinite(p_value) else f"{p_value:.5f}"
            lines.append(
                "<tr><td>{model}</td><td>{head}</td><td>{contrast}</td>"
                "<td>{mean:+.5g}</td><td>[{low:+.5g}, {high:+.5g}]</td>"
                "<td>{p}</td><td>{sign}</td></tr>".format(
                    model=html.escape(model),
                    head=html.escape(str(row["parent_candidate"])),
                    contrast=html.escape(labels.get(str(row["contrast_name"]), str(row["contrast_name"]))),
                    mean=float(row["mean_candidate_minus_matched"]),
                    low=float(row["ci95_low"]),
                    high=float(row["ci95_high"]),
                    p=p_text,
                    sign=html.escape(str(row["support_direction"])),
                )
            )
    lines.append("</tbody></table>")
    return "".join(lines)


def _joint_candidate_summary(payload: dict[str, Any]) -> str:
    metric_order = (
        ("z_transport", "Z"),
        ("alpha_localization", "alpha"),
        ("removal_absolute_error", "removal-error"),
        ("removal_correct_margin", "removal-margin"),
    )
    lines = ["<ul>"]
    found = False
    for model, model_payload in payload["model_analyses"].items():
        rows = model_payload["head_specificity_contrasts"]
        for decision in model_payload["candidate_decisions"]:
            if not bool(decision["joint_ov_head_support"]):
                continue
            found = True
            parent = str(decision["parent_candidate"])
            by_name = {
                str(row["contrast_name"]): row
                for row in rows
                if str(row["parent_candidate"]) == parent
            }
            stats = []
            for metric, label in metric_order:
                row = by_name[metric]
                p_value = row["one_sided_sign_flip_p"]
                p_text = "—" if p_value is None else f"{float(p_value):.5f}"
                stats.append(
                    f"{label} Δ={float(row['mean_candidate_minus_matched']):+.4g}, p={p_text}"
                )
            strict = "PASS" if decision["head_specific_exact_p_le_0_05"] else "FAIL"
            lines.append(
                "<li><strong>{model} {head}</strong>：冻结联合方向规则 PASS；"
                "selected−matched 为 {stats}；严格 head-specific 联合判定 <strong>{strict}</strong>。</li>".format(
                    model=html.escape(model),
                    head=html.escape(parent),
                    stats=html.escape("；".join(stats)),
                    strict=strict,
                )
            )
    if not found:
        lines.append("<li>没有候选通过冻结联合方向规则。</li>")
    lines.append("</ul>")
    return "".join(lines)


def _implementation_audit_table(payload: dict[str, Any]) -> str:
    lines = [
        "<table><thead><tr><th>model</th><th>output layer</th><th>V source</th>",
        "<th>mapping estimand</th><th>cache raw max</th><th>cache centered max</th>",
        "<th>centered tolerance</th></tr></thead><tbody>",
    ]
    for model, model_payload in payload["model_analyses"].items():
        cache = model_payload["attention_cache_audit"]
        centered = cache["centered_candidate_logit_max_abs_delta"]
        centered_text = (
            "not recorded"
            if centered is None or not np.isfinite(centered)
            else f"{centered:.4g}"
        )
        for row in model_payload["mapping_estimands"]:
            lines.append(
                "<tr><td>{model}</td><td>L{layer}</td><td>L{source}</td>"
                "<td><code>{kind}</code></td><td>{raw:.4g}</td><td>{centered}</td>"
                "<td>{tolerance:.4g}</td></tr>".format(
                    model=html.escape(model),
                    layer=int(row["layer"]),
                    source=int(row["value_source_layer"]),
                    kind=html.escape(str(row["value_path_estimand"])),
                    raw=float(cache["raw_candidate_logit_max_abs_delta"]),
                    centered=centered_text,
                    tolerance=float(cache["centered_tolerance"]),
                )
            )
    lines.append("</tbody></table>")
    return "".join(lines)


def build_report(run_root: Path, output: Path) -> None:
    campaign = _read_json(run_root / "analysis" / "analysis.json")
    figures: dict[str, dict[str, str]] = {}
    for model in campaign["model_analyses"]:
        model_root = run_root / "models" / model
        figures[model] = {
            "mapping": _mapping_figure(model_root, model),
            "patch": _patch_figure(model_root, model),
            "injection": _injection_figure(model_root, model),
            "removal": _removal_figure(model_root, model),
        }
    sections = []
    for model, images in figures.items():
        sections.append(
            f"""
<section id="{html.escape(model)}"><h2>{html.escape(model)}</h2>
<figure><img src="{images['mapping']}" alt="OV mapping"><figcaption>只有右上象限表示 fit-count 与 held-out-count 方向一致；红色为预先选择 head，灰蓝叉为同层 mapped-norm 匹配对照。</figcaption></figure>
<figure><img src="{images['patch']}" alt="staged patch"><figcaption>上图检验 value/output transport 是否超过等范数输出控制；下图检验 donor attention pattern 是否超过位置打乱。</figcaption></figure>
<figure><img src="{images['injection']}" alt="signed injection"><figcaption>这是 layer post-O residual 上的干预，不在某个 head 内实施。selected/matched 只是同一层干预的重复登记，曲线按设计应完全重合；它验证层级 answer direction 可控性，不能定位单头。</figcaption></figure>
<figure><img src="{images['removal']}" alt="direction removal"><figcaption>answer-direction removal 与等范数正交 removal 的配对差；误差增大为正、正确 margin 降低为负才符合方向性预测。</figcaption></figure>
</section>"""
        )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Realistic NIAH V4.4.3 · OV causal report</title>
<style>
:root{{--paper:#F3EEE4;--surface:#FFFDF8;--ink:#20242D;--muted:#5E6672;--line:#C9C2B6;--indigo:#23165C;--teal:#00A88F;--pink:#D94B86;--green:#247A4A;--red:#A23C3C}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:"Segoe UI",Arial,sans-serif;line-height:1.62}}main{{max-width:1120px;margin:auto;padding:40px 26px 80px}}h1{{font-size:40px;line-height:1.08}}h2{{font-size:28px;margin-top:0}}header,section{{padding:28px 0 44px;border-bottom:1px solid var(--line)}}.lead{{font-size:18px;max-width:86ch}}.callout{{background:var(--surface);border-left:4px solid var(--teal);padding:16px 20px;margin:20px 0}}figure{{margin:24px 0;background:var(--surface);border:1px solid var(--line);padding:15px}}figure img{{display:block;width:100%;height:auto}}figcaption{{font-size:13px;color:var(--muted);margin-top:10px}}.table-scroll{{overflow:auto;background:var(--surface);border:1px solid var(--line)}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px 10px;border-bottom:1px solid #DED8CE;text-align:left}}th{{background:#ECE6DA}}td.pass{{color:var(--green);font-weight:700}}td.fail{{color:var(--red);font-weight:700}}code{{background:#EAE4D8;padding:1px 4px}}.small{{font-size:13px;color:var(--muted)}}
</style></head><body><main>
<header><div class="small">Realistic NIAH · V4.4.3 · frozen discovery/screen/confirmation split</div><h1>OV vertical geometry 的因果检验</h1><p class="lead">本报告检验：prompt running-index 方向是否被一个候选 attention head 的 V/O 路径读出，并写入 answer-count 方向。几何映射只用于冻结候选；head 定位主要由 staged patch、direction removal 与 matched-head 配对约束，layer-level signed injection 只检验 answer subspace 的可控性。</p>
<div class="callout"><strong>自动判定：</strong>{html.escape(str(campaign['conclusion']))}</div>
<p class="small">run root: <code>{html.escape(str(run_root))}</code></p></header>
<section><h2>关键候选</h2>{_joint_candidate_summary(campaign)}
<p class="small">这里的 exact p 是以 seed 为单位的单侧 sign-flip p，未做多重比较校正；严格联合判定在未校正层面已经失败，因此校正不会改变“尚不能确认单头 circuit”的结论。</p></section>
<section><h2>证据矩阵</h2><div class="table-scroll">{_decision_table(campaign)}</div>
<p class="small">“frozen joint rule”严格保留预注册的方向规则：mapping、Z transport、layer injection、direction removal 同时通过；alpha-vs-scramble 是 QK 定位证据，单独报告。“all head-specific p≤.05”只使用 selected-vs-matched 的 Z 与两项 removal 配对 exact sign-flip test。PASS/FAIL 方向规则本身不等价于显著性。</p></section>
<section><h2>冻结的 mapping 候选</h2><div class="table-scroll">{_mapping_selection_table(run_root, campaign)}</div>
<p class="small">候选只按 discovery seeds 的 fit-count cosine 排序，held-out-count cosine 不参与选择。empirical p 是有限 head 排名的加一置换分辨率；本实验所有候选虽为层内 rank 1 且 held-out 方向为正，但没有一个 mapping empirical p≤.05。</p></section>
<section><h2>单头特异性统计</h2><div class="table-scroll">{_specificity_table(campaign)}</div>
<p class="small">统计单位是 seed；screen 与 confirmation 各只有 n=5，因此单侧 exact sign-flip 的最小可达 p 值为 1/32=0.03125。alpha 行是 QK 辅助证据；layer injection duplicate audit 的目标值是 0，故不计算方向性 p 值。</p></section>
<section><h2>实现与数值审计</h2><div class="table-scroll">{_implementation_audit_table(campaign)}</div>
<p class="small">本 checkpoint 的 Gemma L36–L38 使用 L22 的 shared sliding-KV（运行时 graph resolver 记录在表中）；centered cache 差去除了所有 count token 的共同平移，只有它与 expected-count/count-margin 的相对结构直接相关。</p></section>
{''.join(sections)}
<section><h2>解释边界</h2><div class="callout"><strong>若严格单头证据不足：</strong>这只说明“当前预注册 late-layer 单头尚不足以解释 geometry”，不能推出 OV 不参与。下一步应检验 small head set、MLP-mediated re-encoding 或跨层分布式路径。</div>
<p>候选分数使用完整答案加 chat termination 的联合 log probability，因为 count 10 在两个 tokenizer 中均为两 token。non-count KL 只是 answer-query 的局部 specificity control，不是通用语言能力测试。<code>z_h</code> 与 <code>o_h=W_Oz_h</code> 的数值一致性是实现审计，不被重复计作两份因果证据。</p></section>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_root / "analysis" / "realistic_niah_v4_4_3_ov_causal_report.html"
    build_report(args.run_root.resolve(), output.resolve())
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
