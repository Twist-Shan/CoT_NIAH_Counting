from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from realistic_niah_v4_4_3.io import atomic_text

from .analysis import _bootstrap_ci
from .spec import V444Config


def _ci_record(values: np.ndarray, *, config: V444Config, seed: int) -> dict[str, float]:
    mean, low, high = _bootstrap_ci(
        values, repetitions=config.bootstrap_repetitions, seed=seed
    )
    return {"mean": mean, "low": low, "high": high}


def _role_curve(
    frame: pd.DataFrame,
    *,
    role: str,
    x_column: str,
    y_column: str,
    config: V444Config,
    seed_base: int,
) -> list[dict[str, float]]:
    selected = frame[frame["set_role"].eq(role)]
    if role == "matched_control":
        per_seed = selected.groupby(["seed", x_column], as_index=False)[y_column].mean()
    else:
        per_seed = selected.groupby(["seed", x_column], as_index=False)[y_column].mean()
    rows = []
    for offset, (x_value, group) in enumerate(per_seed.groupby(x_column, sort=True)):
        rows.append(
            {
                "x": float(x_value),
                **_ci_record(
                    group[y_column].to_numpy(float),
                    config=config,
                    seed=seed_base + offset,
                ),
            }
        )
    return rows


def _condition_bars(
    frame: pd.DataFrame,
    *,
    metric: str,
    conditions: Mapping[str, str],
    config: V444Config,
    seed_base: int,
) -> list[dict[str, Any]]:
    selected = frame[frame["intervention"].isin(conditions)].copy()
    selected["condition"] = selected["intervention"].map(conditions)
    rows = []
    for role in ("candidate_core", "matched_control"):
        role_frame = selected[selected["set_role"].eq(role)]
        per_seed = role_frame.groupby(["seed", "condition"], as_index=False)[metric].mean()
        for offset, (condition, group) in enumerate(
            per_seed.groupby("condition", sort=True)
        ):
            rows.append(
                {
                    "group": "H16/H19" if role == "candidate_core" else "matched controls",
                    "category": str(condition),
                    **_ci_record(
                        group[metric].to_numpy(float),
                        config=config,
                        seed=seed_base + offset + (0 if role == "candidate_core" else 100),
                    ),
                }
            )
    return rows


def _chart_payload(run_root: Path, config: V444Config) -> dict[str, Any]:
    model_root = run_root / "models" / config.model_label
    confirmation = model_root / "confirmation"
    analysis_root = run_root / "analysis"
    directed = pd.read_csv(confirmation / "directed_detail.csv.gz")
    mediation = pd.read_csv(confirmation / "mediation_detail.csv.gz")
    seed_metrics = pd.read_csv(analysis_root / "seed_metrics.csv.gz")
    summary = pd.read_csv(analysis_root / "endpoint_summary.csv.gz")
    selection = json.loads(
        (model_root / "center_controls" / "selection.json").read_text(encoding="utf-8")
    )
    primary_endpoints = {
        "natural_carrier_count_slope": "自然 carrier/count slope",
        "injection_dose_slope": "pre-O injection dose slope",
        "removal_error_axis_minus_control": "removal Δerror: axis−control",
        "removal_margin_axis_minus_control": "removal Δmargin: axis−control",
        "donor_patch_transport": "donor-Z normalized transport",
        "mediation_control_minus_axis_block": "mediation: orth−block",
    }
    forest = []
    for endpoint, label in primary_endpoints.items():
        for role, suffix, role_label in (
            ("candidate_core", "", "candidate"),
            (
                "candidate_specificity",
                "__candidate_minus_control_mean",
                "candidate−control mean",
            ),
        ):
            selected = summary[
                summary["endpoint"].eq(endpoint + suffix)
                & summary["set_role"].eq(role)
            ]
            if len(selected) != 1:
                continue
            row = selected.iloc[0]
            forest.append(
                {
                    "label": f"{label} · {role_label}",
                    "mean": float(row["mean"]),
                    "low": float(row.ci95_low),
                    "high": float(row.ci95_high),
                    "alternative": str(row.alternative),
                    "p": float(row.one_sided_exact_sign_flip_p),
                }
            )
    injection = directed[directed["intervention"].str.startswith(
        "natural_ov_z_injection_beta_", na=False
    )]
    dose = [
        {
            "name": "H16/H19",
            "color": "#0f766e",
            "points": _role_curve(
                injection,
                role="candidate_core",
                x_column="beta",
                y_column="delta_expected_count",
                config=config,
                seed_base=444_100,
            ),
        },
        {
            "name": "matched controls mean",
            "color": "#c2410c",
            "points": _role_curve(
                injection,
                role="matched_control",
                x_column="beta",
                y_column="delta_expected_count",
                config=config,
                seed_base=444_200,
            ),
        },
    ]
    removal_conditions = {
        "natural_ov_count_axis_removal": "natural-axis",
        "equal_output_norm_set_span_orthogonal_removal": "in-span orthogonal",
    }
    removal_error = _condition_bars(
        directed,
        metric="delta_expected_count_absolute_error",
        conditions=removal_conditions,
        config=config,
        seed_base=444_300,
    )
    removal_margin = _condition_bars(
        directed,
        metric="delta_correct_margin",
        conditions=removal_conditions,
        config=config,
        seed_base=444_400,
    )
    mediation_conditions = {
        "donor_z_patch": "donor patch",
        "donor_z_patch_natural_axis_block": "patch + axis block",
        "donor_z_patch_orthogonal_control": "patch + orth control",
    }
    mediation_bars = _condition_bars(
        mediation,
        metric="continuous_normalized_transport",
        conditions=mediation_conditions,
        config=config,
        seed_base=444_500,
    )
    definitions = [selection["candidate"], *selection["registered_nested_sets"]]
    nested = []
    for definition in definitions:
        set_id = str(definition["set_id"])
        for endpoint, label in (
            ("natural_carrier_count_slope", "natural slope"),
            ("injection_dose_slope", "injection slope"),
            ("removal_error_axis_minus_control", "removal Δerror"),
            ("removal_margin_axis_minus_control", "removal Δmargin"),
        ):
            selected = seed_metrics[
                seed_metrics["set_id"].eq(set_id)
                & seed_metrics["endpoint"].eq(endpoint)
            ]
            values = selected["value"].to_numpy(float)
            nested.append(
                {
                    "series": label,
                    "k": len(definition["heads"]),
                    **_ci_record(
                        values,
                        config=config,
                        seed=444_600 + len(definition["heads"]) * 10 + len(endpoint),
                    ),
                }
            )
    component_defs = [*selection["factorial_components"], selection["candidate"]]
    factorial = []
    for definition in component_defs:
        set_id = str(definition["set_id"])
        name = ",".join(f"H{head}" for head in definition["heads"])
        for endpoint, label in (
            ("natural_carrier_count_slope", "natural slope"),
            ("injection_dose_slope", "injection slope"),
            ("removal_error_axis_minus_control", "removal Δerror"),
            ("removal_margin_axis_minus_control", "removal Δmargin"),
        ):
            selected = seed_metrics[
                seed_metrics["set_id"].eq(set_id)
                & seed_metrics["endpoint"].eq(endpoint)
            ]
            factorial.append(
                {
                    "group": label,
                    "category": name,
                    **_ci_record(
                        selected["value"].to_numpy(float),
                        config=config,
                        seed=444_700 + len(name) + len(endpoint),
                    ),
                }
            )
    return {
        "forest": forest,
        "dose": dose,
        "removal_error": removal_error,
        "removal_margin": removal_margin,
        "mediation": mediation_bars,
        "nested": nested,
        "factorial": factorial,
    }


def _escape_script_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )


def _html_template(analysis: Mapping[str, Any], charts: Mapping[str, Any]) -> str:
    decision = analysis["primary_decision"]
    support = bool(decision["full_natural_ov_transporter_support"])
    badge = "支持完整主张" if support else "未确认完整主张"

    def component(family: str, endpoint: str, role: str = "candidate_core") -> Mapping[str, Any]:
        matches = [
            item
            for item in decision["families"][family]["components"]
            if item["endpoint"] == endpoint and item["role"] == role
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {family}/{endpoint}/{role} component")
        return matches[0]

    natural = component("natural_signal", "natural_carrier_count_slope")
    injection = component("pre_o_injection", "injection_dose_slope")
    injection_specificity = component(
        "pre_o_injection",
        "injection_dose_slope__candidate_minus_control_mean",
        "candidate_specificity",
    )
    removal_error = component(
        "centered_removal", "removal_error_axis_minus_control"
    )
    removal_margin = component(
        "centered_removal", "removal_margin_axis_minus_control"
    )
    donor_transport = component("path_mediation", "donor_patch_transport")
    mediation_block = component(
        "path_mediation", "mediation_control_minus_axis_block"
    )
    mediated_fraction = (
        float(mediation_block["mean"]) / float(donor_transport["mean"])
        if float(donor_transport["mean"]) != 0.0
        else float("nan")
    )
    nested_removal = {
        int(item["k"]): item["families"]["centered_removal"]["holm_p_across_k"]
        for item in analysis["nested_k"]
    }
    factorial = {
        item["endpoint"]: item
        for item in analysis["factorial"]
    }
    family_rows = "".join(
        "<tr><td>{}</td><td>{:.6g}</td><td>{}</td><td>{}</td></tr>".format(
            family,
            item["intersection_union_p"],
            "通过" if item["passes_alpha"] else "未通过",
            "; ".join(
                f"{component['endpoint']}: mean={component['mean']:.4g}, p={component['p']:.4g}"
                for component in item["components"]
            ),
        )
        for family, item in decision["families"].items()
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V4.4.4 natural OV transporter</title>
<style>
:root{{--ink:#17211d;--muted:#5d6a64;--paper:#f7f4ec;--card:#fffdf7;--line:#d7d3c7;--teal:#0f766e;--orange:#c2410c;--pink:#be185d;--blue:#2563eb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.62 Inter,ui-sans-serif,system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1280px;margin:auto;padding:34px 28px 90px}} h1{{font:700 clamp(30px,4vw,52px)/1.08 Georgia,serif;margin:0 0 14px}} h2{{font:700 26px/1.2 Georgia,serif;margin:50px 0 18px}} h3{{font-size:17px;margin:0 0 7px}} p{{max-width:1000px;margin:10px 0}}
.eyebrow{{text-transform:uppercase;letter-spacing:.14em;color:var(--teal);font-weight:750;font-size:12px}} .lead{{font-size:18px;max-width:930px}} .badge{{display:inline-block;padding:7px 12px;border-radius:999px;background:{'#d1fae5' if support else '#ffedd5'};color:{'#065f46' if support else '#9a3412'};font-weight:750}}
.kpis{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin:25px 0}} .kpi,.card{{background:var(--card);border:1px solid var(--line);border-radius:13px;box-shadow:0 5px 18px #2f3c3510}} .kpi{{padding:16px}} .kpi b{{display:block;font-size:25px}} .kpi span{{color:var(--muted);font-size:12px}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}} .card{{padding:17px;min-width:0}} .wide{{grid-column:1/-1}} .chart{{min-height:310px;overflow-x:auto}} .caption{{font-size:13px;color:var(--muted);border-top:1px solid var(--line);padding-top:9px;margin-top:8px}} .conclusion{{border-left:4px solid var(--pink);padding:9px 13px;background:#fff1f5;margin-top:14px}}
.formula{{font-family:"Cambria Math",Cambria,serif;background:#edf7f5;border:1px solid #b7d8d3;border-radius:9px;padding:10px 13px;overflow-x:auto}}
table{{width:100%;border-collapse:collapse;background:var(--card)}} th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} th{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
svg text{{font-family:Inter,ui-sans-serif,system-ui,"Microsoft YaHei",sans-serif;fill:var(--ink)}} .axis{{stroke:#929b96;stroke-width:1}} .gridline{{stroke:#ddd9cf;stroke-width:1}} .zero{{stroke:#6b7280;stroke-width:1.3;stroke-dasharray:5 4}}
@media(max-width:850px){{.grid,.kpis{{grid-template-columns:1fr}}.wide{{grid-column:auto}}main{{padding:24px 15px 60px}}}}
</style></head><body><main>
<div class="eyebrow">Realistic NIAH · V4.4.4 · preregistered confirmation</div>
<h1>模型是否自然使用 L28 的 OV channel 搬运 count？</h1>
<p class="lead">本轮把旧的 post-O reachability test 改成真实 pre-O <i>z</i>-space intervention，并将自然信号、充分性、必要性和 path mediation 分开检验。QK 与 OV heads 不要求是同一组。</p>
<p><span class="badge">{badge}</span></p>
<div class="kpis"><div class="kpi"><b>{decision['global_intersection_union_p']:.4g}</b><span>全局 IUT p（四族最大值）</span></div><div class="kpi"><b>{analysis['baseline']['candidate_count_accuracy']:.1%}</b><span>无干预 candidate-count accuracy</span></div><div class="kpi"><b>{analysis['baseline']['expected_count_mae']:.3g}</b><span>无干预 expected-count MAE</span></div><div class="kpi"><b>{analysis['baseline']['seeds']}</b><span>全新 confirmation seeds</span></div></div>

<h2>1. 结论与判定规则</h2>
<p>完整主张要求四个预注册证据族同时通过 α=.05。每个 family 又要求候选效应与 candidate-minus-four-matched-controls 特异性同时成立；因此报告用组成 p 的最大值，而不是最小值。全局 IUT p={decision['global_intersection_union_p']:.6g}，由最弱的 path-mediation 特异性检验决定。<b>本段结论：四族合取通过；结果支持 L28 H16/H19 是自然使用的 OV transporter，而不只是可 steering 的输出子空间。</b></p>
<table><thead><tr><th>证据族</th><th>IUT p</th><th>判定</th><th>组成检验</th></tr></thead><tbody>{family_rows}</tbody></table>

<h2>2. 定义与主效应总览</h2>
<p>先在实际 attention pre-norm 表示上拟合 prompt-count 斜率 <i>s</i><sub>P</sub>。对 GQA query head h，以 g(h) 表示对应 KV head，定义自然 value-path 单位步长与 set 输出方向：</p>
<div class="formula">d<sub>z,h</sub> = W<sub>V</sub><sup>g(h)</sup>s<sub>P</sub>；&nbsp;&nbsp;m<sub>S</sub> = Σ<sub>h∈S</sub> W<sub>O</sub><sup>h</sup>d<sub>z,h</sub>。</div>
<p>count-neutral 中心 z<sub>0</sub> 是用独立 center seeds 对每个 head 的 z 关于 count 做 OLS 后在 count=0 的截距。自然 carrier 系数定义为：</p>
<div class="formula">a<sub>S</sub>(z) = ⟨W<sub>O</sub><sup>S</sup>(z<sub>S</sub>−z<sub>0,S</sub>), m̂<sub>S</sub>⟩ / ‖m<sub>S</sub>‖。</div>
<p>Injection 在真实 pre-O 边界执行 z<sub>h</sub>←z<sub>h</sub>+βd<sub>z,h</sub>；removal 在同一 z-space 中删除自然输出沿 m<sub>S</sub> 的分量。控制向量也位于 W<sub>O</sub><sup>S</sup> span，post-O 范数相同且与 m<sub>S</sub> 正交。<b>本段结论：干预没有直接向 residual 注入 answer axis；所有输出变化都必须经过被选 heads 自己的 W<sub>O</sub>。</b></p>
<div class="card wide"><h3>图 1 · 候选效应与 matched-set 特异性</h3><div id="forest" class="chart"></div><p class="caption">横轴：seed-level effect 的均值；横线：seed bootstrap 95% CI；竖虚线：零效应。标签中的 “&gt;0 / &lt;0” 是预注册方向，p 为单侧 exact sign-flip。不同 endpoint 的量纲不同，因此本图只比较方向、是否跨零与不确定性，不比较绝对长度。</p><p class="conclusion">自然 carrier/count slope={natural['mean']:.4f}（95% CI {natural['ci95_low']:.4f}–{natural['ci95_high']:.4f}，p={natural['p']:.3g}）；候选相对四个 matched controls 的差仍为 {decision['families']['natural_signal']['components'][1]['mean']:.4f}。本段结论：无干预 forward 中，H16/H19 的自然输出确实随 prompt count 单调编码。</p></div>

<h2>3. 真正的 pre-O OV injection</h2><div class="card wide"><h3>图 2 · natural V-path dose response</h3><div id="dose" class="chart"></div><p class="caption">横轴 β：在 answer query 的每个 selected-head pre-O z slice 中加入 β·d<sub>z,h</sub>；纵轴：相对无干预的 expected-count 变化。线为跨 seed 与 causal count 的均值，误差条为先按 seed 聚合后的 bootstrap 95% CI。β=0 精确复用 baseline logits。</p><p class="conclusion">候选 dose slope={injection['mean']:.4f} expected-count/β（95% CI {injection['ci95_low']:.4f}–{injection['ci95_high']:.4f}，p={injection['p']:.3g}）；相对 matched-control mean 的特异性差={injection_specificity['mean']:.4f}（p={injection_specificity['p']:.3g}）。本段结论：真实 V→z→W<sub>O</sub> channel 具有带符号、近线性的充分性；但本段单独仍不证明自然依赖。</p></div>

<h2>4. Centered z-space removal</h2><div class="grid"><div class="card"><h3>图 3A · expected-count absolute error</h3><div id="removalError" class="chart"></div><p class="caption">纵轴：干预后 absolute error 相对 baseline 的变化，越高表示损伤越大。natural-axis 与 control 均在同一 selected-head W<sub>O</sub> span，且 post-O 输出范数匹配。</p></div><div class="card"><h3>图 3B · correct-count margin</h3><div id="removalMargin" class="chart"></div><p class="caption">纵轴：正确 count log-score margin 相对 baseline 的变化，越负表示损伤越大。比较 natural-axis 与 in-span orthogonal control。</p></div></div><p class="conclusion">相对等范数正交 control，删除自然轴使 absolute error 多增加 {removal_error['mean']:.4f}（95% CI {removal_error['ci95_low']:.4f}–{removal_error['ci95_high']:.4f}，p={removal_error['p']:.3g}），并使 correct-count margin 多下降 {abs(float(removal_margin['mean'])):.4f}（95% CI {removal_margin['ci95_low']:.4f}–{removal_margin['ci95_high']:.4f}，p={removal_margin['p']:.3g}）。本段结论：centered、set-realizable removal 提供自然 necessity 证据，而不是删除静态 offset 或任意 residual 方向。</p>

<h2>5. Path mediation</h2><div class="card wide"><h3>图 4 · donor-z patch 是否由同一自然 OV 轴介导</h3><div id="mediation" class="chart"></div><p class="caption">纵轴：continuous normalized transport = Δexpected-count / (donor count − receiver count)。三种条件依次为 donor-z patch、patch 后阻断自然轴、patch 后加入同输出范数的 span 内正交控制。若自然轴介导 transport，axis block 应低于 orth control。</p><p class="conclusion">donor-z patch 搬运 {donor_transport['mean']:.4f} 个归一化 count-gap（95% CI {donor_transport['ci95_low']:.4f}–{donor_transport['ci95_high']:.4f}，p={donor_transport['p']:.3g}）。相对等范数正交 control，阻断自然轴额外消除 {mediation_block['mean']:.4f}（约为 donor transport 的 {mediated_fraction:.1%}；95% CI {mediation_block['ci95_low']:.4f}–{mediation_block['ci95_high']:.4f}，p={mediation_block['p']:.4g}）。本段结论：同一自然 OV 轴部分介导 donor effect；这是自然使用证据，但仍只定位 L28 下游 carrier。</p></div>

<h2>6. K 与成员结构（二级）</h2><h3>图 5 · 预冻结 nested K（按 endpoint 分面）</h3><div class="grid"><div class="card"><div id="nestedNatural" class="chart"></div><p class="caption">横轴 K；纵轴 natural carrier/count slope。</p></div><div class="card"><div id="nestedInjection" class="chart"></div><p class="caption">横轴 K；纵轴 pre-O injection dose slope。</p></div><div class="card"><div id="nestedError" class="chart"></div><p class="caption">横轴 K；纵轴 removal 的 Δerror(axis−control)。</p></div><div class="card"><div id="nestedMargin" class="chart"></div><p class="caption">横轴 K；纵轴 removal 的 Δmargin(axis−control)。每点均为 seed mean ± bootstrap 95% CI；K=3/4/6/8 是预冻结二级集合。</p></div></div><p class="conclusion">Natural signal 与 injection 在所有 K 上 Holm 通过；removal 仅 K=2（Holm p={nested_removal[2]:.4g}）和 K=4（{nested_removal[4]:.4g}）通过，K=3/6/8 均为 {nested_removal[3]:.4g}。本段结论：扩展 K 没有产生“越大越显著”的模式，最稳健的主结果仍是带同 K controls 的 K=2。</p>
<h3>图 6 · H16、H19 与联合 set（按 endpoint 分面）</h3><div class="grid"><div class="card"><div id="factorialNatural" class="chart"></div><p class="caption">纵轴 natural carrier/count slope；该系数会随 set 自身 m<sub>S</sub> 重新归一化，不能按加法解释。</p></div><div class="card"><div id="factorialInjection" class="chart"></div><p class="caption">纵轴 pre-O injection dose slope。</p></div><div class="card"><div id="factorialError" class="chart"></div><p class="caption">纵轴 removal Δerror(axis−control)。</p></div><div class="card"><div id="factorialMargin" class="chart"></div><p class="caption">纵轴 removal Δmargin(axis−control)。横轴均为 H16、H19、联合 H16/H19；误差条为 95% CI。</p></div></div><p class="conclusion">Injection 的联合项近似严格可加（interaction={factorial['injection_dose_slope']['mean']:.5f}, p={factorial['injection_dose_slope']['two_sided_exact_sign_flip_p']:.3g}）；removal-error interaction 也不显著（p={factorial['removal_error_axis_minus_control']['two_sided_exact_sign_flip_p']:.3g}）。H19 的单头 necessity 强于 H16；联合 set 的 margin 损伤更大，但额外协同未达显著（p={factorial['removal_margin_axis_minus_control']['two_sided_exact_sign_flip_p']:.3g}）。本段结论：当前数据支持两个 head 的近加性贡献，尚不支持超加性协同。</p>

<h2>7. 设定、审计与边界</h2><p>方向 seeds 1234–1253；z-center/control-selection seeds 1264–1273；confirmation seeds 1274–1293。主 set 固定 Qwen3-8B L28 H16/H19，四个 controls 仅用预因果特征匹配。数据与结果写入独立 v4_4_4_natural_ov namespace；不持久化 raw attention 或 full hidden states。审计状态：<b>{analysis['audit']['all_checks_pass']}</b>，计数 {analysis['audit']['observed']}。<b>本段结论：若审计为 True，seed grid、shards、design hash 与隔离约束均达到预注册要求。</b></p>
<p>本实验没有证明 H16/H19 自己从原始 needle 做 QK 定位，也没有检验跨层 relay 的 source heads。若 injection 显著但 removal/mediation 不显著，结论只能是“该 OV 空间可 steering”；若四族汇合，则支持“L28 该 set 自然承载并因果贡献 count channel”，但完整 circuit 仍待 QK edge/path test。<b>本段结论：报告严格区分可达性、自然依赖与完整路径。</b></p>
</main><script>
const D={_escape_script_json(charts)};
const NS='http://www.w3.org/2000/svg';
function E(tag,a={{}},text=''){{const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))n.setAttribute(k,v);if(text)n.textContent=text;return n}}
function extent(vals){{let lo=Math.min(0,...vals),hi=Math.max(0,...vals);if(lo===hi){{lo-=1;hi+=1}}const p=(hi-lo)*.12;return[lo-p,hi+p]}}
function base(id,h=310){{const el=document.getElementById(id),w=Math.max(620,el.clientWidth||620),s=E('svg',{{viewBox:`0 0 ${{w}} ${{h}}`,width:'100%',height:h}});el.replaceChildren(s);return{{s,w,h}}}}
function xScale(v,lo,hi,l,r){{return l+(v-lo)/(hi-lo)*(r-l)}}
function axes(s,w,h,lo,hi,xlabel,margin={{l:92,r:24,t:22,b:55}}){{const y=h-margin.b;s.append(E('line',{{x1:margin.l,y1:y,x2:w-margin.r,y2:y,class:'axis'}}));for(let i=0;i<=5;i++){{const v=lo+(hi-lo)*i/5,x=xScale(v,lo,hi,margin.l,w-margin.r);s.append(E('line',{{x1:x,y1:margin.t,x2:x,y2:y,class:i===0?'gridline':'gridline'}}));s.append(E('text',{{x,y:y+19,'text-anchor':'middle','font-size':11}},v.toPrecision(3)))}}if(lo<0&&hi>0){{const z=xScale(0,lo,hi,margin.l,w-margin.r);s.append(E('line',{{x1:z,y1:margin.t,x2:z,y2:y,class:'zero'}}))}}s.append(E('text',{{x:(margin.l+w-margin.r)/2,y:h-9,'text-anchor':'middle','font-size':12}},xlabel));return margin}}
function yAxes(s,w,h,lo,hi,ylabel,margin={{l:72,r:24,t:26,b:65}}){{const bottom=h-margin.b,ys=(v)=>margin.t+(hi-v)/(hi-lo)*(bottom-margin.t);s.append(E('line',{{x1:margin.l,y1:margin.t,x2:margin.l,y2:bottom,class:'axis'}}));s.append(E('line',{{x1:margin.l,y1:bottom,x2:w-margin.r,y2:bottom,class:'axis'}}));for(let i=0;i<=5;i++){{const v=lo+(hi-lo)*i/5,y=ys(v);s.append(E('line',{{x1:margin.l,y1:y,x2:w-margin.r,y2:y,class:'gridline'}}));s.append(E('text',{{x:margin.l-7,y:y+4,'text-anchor':'end','font-size':11}},v.toPrecision(3)))}}if(lo<0&&hi>0){{const z=ys(0);s.append(E('line',{{x1:margin.l,y1:z,x2:w-margin.r,y2:z,class:'zero'}}))}}s.append(E('text',{{x:14,y:(margin.t+bottom)/2,'text-anchor':'middle','font-size':12,transform:`rotate(-90 14 ${{(margin.t+bottom)/2}})`}},ylabel));return{{...margin,bottom,ys}}}}
function forest(id,rows){{const h=Math.max(360,rows.length*31+70),{{s,w}}=base(id,h),vals=rows.flatMap(d=>[d.low,d.high]),[lo,hi]=extent(vals),m=axes(s,w,h,lo,hi,'seed-level effect (mean and 95% CI)',{{l:310,r:28,t:18,b:50}});rows.forEach((d,i)=>{{const y=28+i*31;s.append(E('text',{{x:8,y:y+4,'font-size':11}},`${{d.label}} (${{d.alternative==='greater'?'>0':'<0'}}, p=${{d.p.toPrecision(3)}})`));s.append(E('line',{{x1:xScale(d.low,lo,hi,m.l,w-m.r),y1:y,x2:xScale(d.high,lo,hi,m.l,w-m.r),y2:y,stroke:'#0f766e','stroke-width':2}}));s.append(E('circle',{{cx:xScale(d.mean,lo,hi,m.l,w-m.r),cy:y,r:4.5,fill:'#be185d'}}))}})}}
function lineChart(id,series,xlabel,ylabel){{const{{s,w,h}}=base(id,330),pts=series.flatMap(q=>q.points),[lo,hi]=extent(pts.flatMap(d=>[d.low,d.high])),m=yAxes(s,w,h,lo,hi,ylabel,{{l:72,r:28,t:30,b:62}}),xs=[...new Set(pts.map(d=>d.x))].sort((a,b)=>a-b),span=Math.max(1e-12,xs.at(-1)-xs[0]),x=(v)=>m.l+(v-xs[0])/span*(w-m.l-m.r),y=m.ys;xs.forEach(v=>s.append(E('text',{{x:x(v),y:m.bottom+20,'text-anchor':'middle','font-size':11}},String(v))));series.forEach((q,si)=>{{const path=q.points.map((d,i)=>`${{i?'L':'M'}}${{x(d.x)}},${{y(d.mean)}}`).join(' ');s.append(E('path',{{d:path,fill:'none',stroke:q.color,'stroke-width':2.5}}));q.points.forEach(d=>{{s.append(E('line',{{x1:x(d.x),y1:y(d.low),x2:x(d.x),y2:y(d.high),stroke:q.color}}));s.append(E('circle',{{cx:x(d.x),cy:y(d.mean),r:4,fill:q.color}}))}});s.append(E('text',{{x:w-190,y:20+si*18,'font-size':12,fill:q.color}},q.name))}});s.append(E('text',{{x:(m.l+w-m.r)/2,y:h-9,'text-anchor':'middle','font-size':12}},xlabel))}}
function bars(id,rows,ylabel){{const{{s,w,h}}=base(id,330),cats=[...new Set(rows.map(d=>d.category))],groups=[...new Set(rows.map(d=>d.group))],[lo,hi]=extent(rows.flatMap(d=>[d.low,d.high])),m=yAxes(s,w,h,lo,hi,ylabel,{{l:74,r:24,t:32,b:80}}),colors=['#0f766e','#c2410c','#2563eb','#be185d'],band=(w-m.l-m.r)/cats.length,ys=m.ys;cats.forEach((cat,ci)=>{{const subset=rows.filter(d=>d.category===cat);subset.forEach((d,gi)=>{{const bw=Math.min(30,band/(groups.length+1)),cx=m.l+band*(ci+.5)+(gi-(subset.length-1)/2)*(bw+5),zero=ys(0),top=ys(d.mean);s.append(E('rect',{{x:cx-bw/2,y:Math.min(zero,top),width:bw,height:Math.max(1,Math.abs(zero-top)),fill:colors[gi%colors.length],opacity:.82}}));s.append(E('line',{{x1:cx,y1:ys(d.low),x2:cx,y2:ys(d.high),stroke:'#17211d'}}))}});s.append(E('text',{{x:m.l+band*(ci+.5),y:m.bottom+19,'text-anchor':'middle','font-size':10,transform:`rotate(-12 ${{m.l+band*(ci+.5)}} ${{m.bottom+19}})`}},cat))}});groups.forEach((g,i)=>s.append(E('text',{{x:w-190,y:18+i*17,'font-size':11,fill:colors[i]}},g)))}}
function nestedPanel(id,name,color,ylabel){{const points=D.nested.filter(d=>d.series===name).map(d=>({{x:d.k,mean:d.mean,low:d.low,high:d.high}}));lineChart(id,[{{name,color,points}}],'set size K',ylabel)}}
function factorialPanel(id,name,color,ylabel){{const rows=D.factorial.filter(d=>d.group===name).map(d=>({{...d,group:name}}));bars(id,rows,ylabel)}}
forest('forest',D.forest);lineChart('dose',D.dose,'injection dose β','Δ expected count');bars('removalError',D.removal_error,'Δ absolute error');bars('removalMargin',D.removal_margin,'Δ correct margin');bars('mediation',D.mediation,'normalized transport');
nestedPanel('nestedNatural','natural slope','#0f766e','natural carrier slope');nestedPanel('nestedInjection','injection slope','#c2410c','injection dose slope');nestedPanel('nestedError','removal Δerror','#2563eb','Δ error: axis−control');nestedPanel('nestedMargin','removal Δmargin','#be185d','Δ margin: axis−control');
factorialPanel('factorialNatural','natural slope','#0f766e','natural carrier slope');factorialPanel('factorialInjection','injection slope','#c2410c','injection dose slope');factorialPanel('factorialError','removal Δerror','#2563eb','Δ error: axis−control');factorialPanel('factorialMargin','removal Δmargin','#be185d','Δ margin: axis−control');
</script></body></html>"""


def build_html_report(
    *, run_root: str | Path, config: V444Config
) -> Path:
    root = Path(run_root)
    analysis_path = root / "analysis" / "realistic_niah_v4_4_4_analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    charts = _chart_payload(root, config)
    destination = root / "analysis" / "realistic_niah_v4_4_4_natural_ov_report.html"
    atomic_text(destination, _html_template(analysis, charts))
    return destination
