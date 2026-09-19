from __future__ import annotations

"""Build the self-contained V4.4.0 native-thinking counter-noise side report."""

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work" / "v440_native_thinking_noise_results"
OUTPUT = ROOT / "reports" / "realistic_niah_v4_4_0_native_thinking_noise_report.html"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


PROMPT_ALL = read_rows(DATA / "prompt_factors" / "prompt_noise_factor_models.csv")
PROMPT_CORRECT = read_rows(
    DATA / "prompt_factors_correct" / "prompt_noise_factor_models.csv"
)
TRACE_ALL = read_rows(DATA / "trace_factors" / "trace_noise_factor_models.csv")
TRACE_CORRECT = read_rows(
    DATA / "trace_factors_correct" / "trace_noise_factor_models.csv"
)
GEOMETRY = read_rows(DATA / "decomposition" / "geometric_noise_decomposition.csv")


MODELS = ["Qwen3-8B", "Gemma4-E4B"]
MODEL_SHORT = {"Qwen3-8B": "Qwen", "Gemma4-E4B": "Gemma"}
TARGET_SHORT = {
    "noise_total_rms": "总 RMS 偏差",
    "noise_orthogonal_rms": "正交 RMS 偏差",
    "count_axis_deviation_abs": "第一计数轴绝对偏差",
}
ALGO_SHORT = {
    "elastic_net": "ElasticNet",
    "random_forest": "Random forest",
    "extra_trees": "ExtraTrees",
    "hist_gradient_boosting": "HGB",
}
PROMPT_FEATURES = [
    "count_only",
    "plus_endpoint_position",
    "plus_content_context",
    "plus_outcome_diagnostic",
]
TRACE_FEATURES = [
    "count_only",
    "plus_trace_position",
    "plus_trace_form",
    "plus_outcome_diagnostic",
]
FEATURE_SHORT = {
    "count_only": "计数",
    "plus_endpoint_position": "+位置",
    "plus_content_context": "+内容/上下文",
    "plus_trace_position": "+位置",
    "plus_trace_form": "+trace 形式",
    "plus_outcome_diagnostic": "+结果诊断",
}


def value(
    rows: list[dict[str, str]],
    *,
    model: str,
    target: str,
    algorithm: str,
    feature_set: str,
) -> float:
    for row in rows:
        label = row.get("model_label") or row.get("scope", "").split("/")[0]
        if (
            label == model
            and row["target"] == target
            and row["model"] == algorithm
            and row["feature_set"] == feature_set
        ):
            return float(row["heldout_r2_log1p"])
    raise KeyError((model, target, algorithm, feature_set))


def f3(number: float) -> str:
    return f"{number:.3f}"


def table(headers: list[str], rows: list[list[str]], classes: str = "") -> str:
    head = "".join(f"<th>{html.escape(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{item}</td>" for item in row) + "</tr>"
        for row in rows
    )
    return (
        f'<div class="table-wrap"><table class="{classes}">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def algorithm_table(role: str) -> str:
    if role == "prompt":
        all_rows, correct_rows = PROMPT_ALL, PROMPT_CORRECT
        feature = "plus_content_context"
        targets = [
            "noise_total_rms",
            "noise_orthogonal_rms",
            "count_axis_deviation_abs",
        ]
        algorithms = [
            "elastic_net",
            "random_forest",
            "extra_trees",
            "hist_gradient_boosting",
        ]
    else:
        all_rows, correct_rows = TRACE_ALL, TRACE_CORRECT
        feature = "plus_trace_form"
        targets = ["noise_total_rms", "count_axis_deviation_abs"]
        algorithms = ["elastic_net", "hist_gradient_boosting"]
    output: list[list[str]] = []
    for model in MODELS:
        for target in targets:
            cells = [MODEL_SHORT[model], TARGET_SHORT[target]]
            for algorithm in algorithms:
                all_value = value(
                    all_rows,
                    model=model,
                    target=target,
                    algorithm=algorithm,
                    feature_set=feature,
                )
                correct_value = value(
                    correct_rows,
                    model=model,
                    target=target,
                    algorithm=algorithm,
                    feature_set=feature,
                )
                cells.append(f"{f3(all_value)} / {f3(correct_value)}")
            output.append(cells)
    return table(
        ["模型", "因变量", *[ALGO_SHORT[name] + "（all / correct）" for name in algorithms]],
        output,
        "metrics",
    )


def line_chart(role: str, target: str) -> str:
    if role == "prompt":
        all_rows, correct_rows, features = PROMPT_ALL, PROMPT_CORRECT, PROMPT_FEATURES
        role_title = "Prompt running counter"
    else:
        all_rows, correct_rows, features = TRACE_ALL, TRACE_CORRECT, TRACE_FEATURES
        role_title = "Trace running counter"
    width, height = 760, 300
    left, right, top, bottom = 62, 24, 26, 54
    inner_w, inner_h = width - left - right, height - top - bottom
    y_min, y_max = -0.15, 1.0

    def x_at(index: int) -> float:
        return left + index * inner_w / max(1, len(features) - 1)

    def y_at(number: float) -> float:
        return top + (y_max - number) * inner_h / (y_max - y_min)

    series = [
        ("Qwen · all", "Qwen3-8B", all_rows, "#2563eb", ""),
        ("Qwen · correct", "Qwen3-8B", correct_rows, "#2563eb", "7 5"),
        ("Gemma · all", "Gemma4-E4B", all_rows, "#dc2626", ""),
        ("Gemma · correct", "Gemma4-E4B", correct_rows, "#dc2626", "7 5"),
    ]
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{role_title} {TARGET_SHORT[target]} held-out R squared">',
        f"<title>{role_title}：{TARGET_SHORT[target]} 的 held-out R²</title>",
        f'<rect x="{left}" y="{top}" width="{inner_w}" height="{inner_h}" class="plot-frame"/>',
    ]
    for tick in [-0.1, 0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_at(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" class="tick">{tick:.2f}</text>')
    zero = y_at(0.0)
    parts.append(f'<line x1="{left}" y1="{zero:.2f}" x2="{width-right}" y2="{zero:.2f}" class="zero"/>')
    for index, feature in enumerate(features):
        x = x_at(index)
        parts.append(f'<text x="{x:.2f}" y="{height-25}" text-anchor="middle" class="tick">{FEATURE_SHORT[feature]}</text>')
    parts.append(f'<text x="{left+inner_w/2:.2f}" y="{height-5}" text-anchor="middle" class="axis-title">累积 feature block</text>')
    parts.append(f'<text transform="translate(16 {top+inner_h/2:.2f}) rotate(-90)" text-anchor="middle" class="axis-title">held-out R²（log1p noise）</text>')
    for label, model, rows, color, dash in series:
        values = [
            value(
                rows,
                model=model,
                target=target,
                algorithm="hist_gradient_boosting",
                feature_set=feature,
            )
            for feature in features
        ]
        points = " ".join(f"{x_at(index):.2f},{y_at(number):.2f}" for index, number in enumerate(values))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"{dash_attr}/>')
        for index, number in enumerate(values):
            parts.append(f'<circle cx="{x_at(index):.2f}" cy="{y_at(number):.2f}" r="4" fill="{color}"><title>{label} · {FEATURE_SHORT[features[index]]}: R²={number:.3f}</title></circle>')
    legend_y = 13
    legend_x = 70
    for label, _model, _rows, color, dash in series:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+26}" y2="{legend_y}" stroke="{color}" stroke-width="2.5"{dash_attr}/>')
        parts.append(f'<text x="{legend_x+32}" y="{legend_y+4}" class="legend">{label}</text>')
        legend_x += 160
    parts.append("</svg>")
    return "".join(parts)


def geometry_table() -> str:
    selected = [
        row
        for row in GEOMETRY
        if row["site"] in {"needle_span_end", "all_sites"}
        and row["population"] in {"all_confirmation", "correct_only_confirmation"}
    ]
    rows: list[list[str]] = []
    for model in MODELS:
        for role, site in [("prompt_running", "needle_span_end"), ("trace_running", "all_sites")]:
            all_row = next(
                row
                for row in selected
                if row["model_label"] == model
                and row["role"] == role
                and row["site"] == site
                and row["population"] == "all_confirmation"
            )
            correct_row = next(
                row
                for row in selected
                if row["model_label"] == model
                and row["role"] == role
                and row["site"] == site
                and row["population"] == "correct_only_confirmation"
            )
            rows.append(
                [
                    MODEL_SHORT[model],
                    "prompt" if role == "prompt_running" else "trace（all sites）",
                    f"{float(all_row['rms_noise_total']):.3f} / {float(correct_row['rms_noise_total']):.3f}",
                    f"{100*float(all_row['parallel_energy_share']):.1f}%",
                    f"{100*float(all_row['orthogonal_energy_share']):.1f}%",
                    f"{100*float(all_row['first_count_axis_energy_share']):.1f}%",
                ]
            )
    return table(
        ["模型", "counter", "总 RMS（all / correct）", "rank-3 内能量", "正交能量", "第一轴能量"],
        rows,
        "metrics",
    )


prompt_feature_rows = [
    ["P0 · count_only", "running_index", "当前第几个 needle（1–10）"],
    [
        "P1 · + endpoint position",
        "P0 + count_progress, token_start, token_end, token_span, prompt_progress, previous_endpoint_gap",
        "计数进度 running_index/10；当前 needle span 的绝对位置、长度、相对 prompt 位置；与前一个 endpoint 的 token 间距",
    ],
    [
        "P2 · + content/context",
        "P1 + input_tokens, score, realization_seed, content_seed; city, haystack_source_mode, haystack_source_files",
        "prompt 总长度、该 needle 的 score、实现/内容种子，以及城市和 haystack 构造来源。种子 ID 只作构造上下文控制，不作可解释因果变量",
    ],
    [
        "P3 · + outcome diagnostic",
        "P2 + absolute_deviation; baseline_correct, predicted_count",
        "加入最终答案误差、是否答对和预测 count；它们发生在表征之后，只能用于诊断，不能作为前因解释",
    ],
]

trace_feature_rows = [
    ["T0 · count_only", "running_index, gold_count; site_scope", "当前 running index、目标总 count；合并分析时控制 trace site"],
    [
        "T1 · + trace position",
        "T0 + count_progress, trace_position, trace_progress, previous_same_site_gap, token_span",
        "running_index/gold_count；相对生成起点的绝对/归一化位置；同类 site 的前后间距；当前 span 长度",
    ],
    [
        "T2 · + trace form",
        "T1 + input_tokens, output_tokens, sequence_length, item_count, duplicate_gold_city_items, city_occurrences_in_item; state_kind, marker_kind, termination_kind, trace_order_class, trace_one_to_one",
        "输入/输出规模、结构化 item 数、重复和城市出现次数，以及 trace 状态/marker/停止/顺序/一一对应等形式变量",
    ],
    [
        "T3 · + outcome diagnostic",
        "T2 + baseline_correct, cutoff_correct",
        "加入完整生成与 cutoff 生成是否正确；同样只作后验诊断",
    ],
]


prompt_table = algorithm_table("prompt")
trace_table = algorithm_table("trace")
geometry = geometry_table()


document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V4.4.0 · Native-thinking counter noise 的定义、预测因素与结果</title>
<style>
:root {{ --ink:#172033; --muted:#607086; --paper:#fbfaf6; --panel:#ffffff; --line:#d9d5ca; --blue:#2563eb; --red:#dc2626; --green:#16865c; --amber:#b7791f; --soft:#f1efe8; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:var(--paper); font-family:Inter,"Noto Sans SC","Microsoft YaHei",system-ui,sans-serif; line-height:1.72; }}
main {{ max-width:1180px; margin:0 auto; padding:48px 42px 96px; }}
h1 {{ font-size:38px; line-height:1.2; margin:0 0 12px; letter-spacing:-.02em; }}
h2 {{ margin:58px 0 18px; font-size:27px; border-top:1px solid var(--line); padding-top:28px; }}
h3 {{ margin:30px 0 10px; font-size:20px; }}
p {{ margin:10px 0; }}
.lede {{ font-size:18px; color:#334155; max-width:950px; }}
.meta {{ color:var(--muted); font-size:14px; }}
.answer {{ margin:26px 0; padding:20px 24px; border-left:5px solid var(--blue); background:#eef4ff; }}
.conclusion {{ margin:16px 0 24px; padding:14px 18px; border-left:4px solid var(--green); background:#edf8f2; }}
.warning {{ margin:16px 0; padding:14px 18px; border-left:4px solid var(--amber); background:#fff8e8; }}
.formula {{ overflow:auto; padding:18px 20px; background:#111827; color:#f8fafc; font-family:"Cascadia Mono",Consolas,monospace; white-space:pre; border-radius:4px; }}
.steps {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:22px 0; }}
.step {{ background:var(--panel); border:1px solid var(--line); padding:16px; }}
.step b {{ display:block; color:var(--blue); margin-bottom:5px; }}
.table-wrap {{ overflow-x:auto; margin:16px 0 24px; }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); font-size:14px; }}
th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ background:var(--soft); font-weight:650; white-space:nowrap; }}
.metrics td:nth-child(n+3), .metrics th:nth-child(n+3) {{ text-align:right; font-variant-numeric:tabular-nums; }}
.figure-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; margin:20px 0; }}
figure {{ margin:0; min-width:0; }}
figure svg {{ display:block; width:100%; height:auto; background:var(--panel); border:1px solid var(--line); }}
figcaption {{ color:var(--muted); font-size:13px; margin-top:8px; }}
.plot-frame {{ fill:#fff; stroke:#cbd5e1; }} .grid {{ stroke:#e5e7eb; stroke-width:1; }} .zero {{ stroke:#64748b; stroke-width:1.4; }}
.tick,.legend,.axis-title {{ fill:#334155; font-family:inherit; font-size:12px; }} .axis-title {{ font-weight:600; }}
code {{ background:#eef1f5; padding:1px 5px; border-radius:3px; font-family:"Cascadia Mono",Consolas,monospace; }}
ul {{ padding-left:22px; }}
.small {{ font-size:13px; color:var(--muted); }}
@media(max-width:820px) {{ main {{ padding:28px 18px 72px; }} h1 {{ font-size:30px; }} .steps,.figure-grid {{ grid-template-columns:1fr; }} }}
@media print {{ body {{ background:white; }} main {{ max-width:none; padding:24px; }} }}
</style>
</head>
<body><main>
<h1>V4.4.0 · Native-thinking counter noise</h1>
<p class="lede">定义、特征回归、correct-only 对照与可解释边界。本文专门回答：所谓 “noise” 到底是什么、哪些变量被用来预测它、不同回归器得到了什么结果，以及为何 trace counter 比 prompt counter 更 noisy。</p>
<div class="warning"><b>版本定位：</b>本报告使用原始保存的 native-thinking 9000-sample archive，是独立的 <b>V4.4.0 side analysis</b>，不属于 V4.4 non-thinking 主机制线，也不应与 V4.4 主报告中的样本直接合并。原始样本来自 <code>v26_v44_native_9000</code>，本次隔离分析根目录为 <code>v28_v44_native_noise</code>。</div>
<p class="meta">数据：native-thinking confirmation split；Qwen3-8B 与 Gemma4-E4B；counter subspace 固定为 discovery split 拟合的 rank-3 basis。报告由保存的审计表自动生成。</p>

<div class="answer"><b>直接回答。</b>可以把本实验理解为“用多个 feature 多元回归 hidden-state 相对其 count 中心的偏差大小”，但要加两点限定：第一，中心是同一模型、同一 site、同一 running count 的 discovery centroid，而不是全局中心；第二，我们没有把完整偏差向量作为多输出因变量，而是对三个非负标量分别做单输出回归：总 RMS、counter subspace 正交 RMS、第一计数轴绝对偏差。所有回归的实际目标都是 <code>log(1 + noise)</code>。</div>

<h2>1. “Noise” 的严格定义</h2>
<p>对 confirmation 样本 <i>i</i>，设 hidden state 为 h<sub>i</sub>，其 running count 为 c<sub>i</sub>。先只用 discovery split 估计 count-conditioned centroid μ<sub>c</sub>，并从十个 count centroids 的变化中冻结 rank-3 正交基 U=[u₁,u₂,u₃]。样本的中心偏差为 ε<sub>i</sub>=h<sub>i</sub>−μ<sub>cᵢ</sub>。</p>
<div class="formula">total RMS       = sqrt( ||εᵢ||² / d )
orthogonal RMS  = sqrt( ||(I − UUᵀ) εᵢ||² / (d − 3) )
axis-1 abs dev  = | u₁ᵀ εᵢ |
regression y    = log(1 + target)</div>
<p>总 RMS 衡量完整 hidden space 中离 count 中心多远；正交 RMS 排除 rank-3 counter subspace 后的偏差；第一轴绝对偏差衡量沿主要计数方向离中心多远。最后一个量丢弃了正负号，问的是“偏离程度”而不是偏向较大或较小 count。</p>
<h3>实际读取的 layer 与 site</h3>
{table(
    ["模型/role", "hidden state 位置", "层的选择", "样本设定"],
    [
        ["Qwen · prompt", "每个 needle span 的末 token", "L8；在 discovery 上按 NSR 选择；模型共 36 层，hidden=4096", "600 discovery + 300 confirmation"],
        ["Gemma · prompt", "每个 needle span 的末 token", "L5；在 discovery 上按 NSR 选择；模型共 42 层，hidden=2560", "600 discovery + 300 confirmation"],
        ["Qwen · trace", "city/item/marker 的 start, middle, span mean, end（12 sites）", "各 site 在 discovery 上独立选层：city L15/16/22/21；item L19/18/20/23；marker L19/13/12/15", "4,520 discovery + 2,278 confirmation samples"],
        ["Gemma · trace", "同上 12 sites", "city L16/24/27/26；item L4/7/10/21；marker L4/4/4/10", "2,365 discovery + 1,199 confirmation samples"],
    ],
)}
<p class="small">每组四个 layer 按 start / middle / span mean / end 的顺序列出。选层和 basis 均在 discovery 完成，confirmation 不参与选择。</p>
{geometry}
<p class="conclusion"><b>本节结论：</b>“noise”是对 discovery-frozen、count-conditioned 中心的离散程度，不等同于模型答错，也不等同于随机生成噪声。correct-only 的 RMS 几乎没有下降，说明即使答案正确，counter 表征仍存在大量 trial-to-trial/position-to-position variation。</p>

<h2>2. 回归设计：如何避免把样本内相关性当作泛化</h2>
<div class="steps">
  <div class="step"><b>① 冻结几何</b>centroid 与 rank-3 basis 仅用 discovery split；confirmation 只评估。</div>
  <div class="step"><b>② 构造一行</b>每个 prompt needle endpoint 或 trace counter site 的 hidden state 是一条回归观测。</div>
  <div class="step"><b>③ 按 seed 分折</b>5-fold GroupKFold；同一 <code>model/seed</code> 的所有位置与 realization 不跨 train/test。</div>
  <div class="step"><b>④ 只报 held-out</b>数值变量在每折内中位数填补并标准化；类别变量众数填补并 one-hot；报告 held-out R²/MAE。</div>
</div>
<p>Prompt all-sample 回归含 6,000 行、20 个 model/seed groups；correct-only 含 4,200 行。Trace 因原始行数更大，在预先固定的随机子样本上分析：all-sample 41,087 行（20 groups，sample fraction 0.20），correct-only 50,833 行（20 groups，sample fraction 0.25）。R²=0 表示不优于测试折均值；R²&lt;0 表示比测试折均值基线更差。</p>
<p>回归器不是逐层或逐 target 挑最优。固定模型族为：ElasticNet（α=0.01, l1_ratio=0.25）、Random forest（400 trees, min leaf=4, max_features=sqrt）、ExtraTrees（同样 400/min leaf=4/sqrt）和 HistGradientBoosting（learning rate=0.06, 300 iterations, 31 leaves）。Trace 只运行 ElasticNet 与 HGB。为避免 winner’s curse，正文以同一个 HGB 作为主要非线性读数，并完整报告其他模型作为敏感性分析。</p>
<p class="conclusion"><b>本节结论：</b>这是按 seed 外推的预测实验，而不是在同一 seed 内随机切行。结果回答“这些已测因素能否预测新 seed 上的偏差幅度”，不能单凭回归系数推出因果关系。</p>

<h2>3. Prompt counter：具体用了哪些自变量</h2>
<p>四个 feature block 是累积的；P1 包含 P0，P2 包含 P1。P3 发生在输出之后，仅用于检查“最终结果是否能额外解释 geometry noise”，不纳入机制性解释。</p>
{table(["Block", "实际自变量", "定义/作用"], prompt_feature_rows)}
<div class="figure-grid">
<figure>{line_chart('prompt','noise_total_rms')}<figcaption>图 1a｜Prompt counter 的总 RMS。横轴为累积 feature block；纵轴为按 seed held-out 的 log1p-target R²。实线=all samples，虚线=correct-only；蓝=Qwen，红=Gemma。P3 为后验诊断。</figcaption></figure>
<figure>{line_chart('prompt','count_axis_deviation_abs')}<figcaption>图 1b｜Prompt counter 的第一计数轴绝对偏差。坐标轴与线型同图 1a。负 R² 表示模型在新 seed 上不如预测训练折均值。</figcaption></figure>
</div>
<h3>固定的最终 pre-outcome block（P2）下，各回归器结果</h3>
<p>每格为 <b>all / correct-only</b> held-out R²；没有逐模型、逐 target 选择“最好的一格”。</p>
{prompt_table}
<p>Gemma 的 prompt noise 有明显可预测结构：HGB 在 P2 上解释总 RMS 的 68.7%（correct-only 67.7%）、正交 RMS 的 63.7%（62.7%）和第一轴绝对偏差的 56.7%（53.2%）。Qwen 的同一固定 HGB 在 P2 上为负 R²；树袋装模型只能得到约 6%–13% 的正 R²，说明宏观位置/内容变量对 Qwen prompt noise 的跨-seed解释弱且对模型族不稳定。</p>
<p>加入结果诊断 P3 的增量接近零：Gemma HGB 总 RMS 0.687→0.694，axis 0.567→0.573；correct-only 中没有增量。Qwen 也没有稳定改善。因此，最终是否答对并不是已有 geometry noise 的主要解释。</p>
<p class="conclusion"><b>Prompt 结论：</b>Gemma prompt counter 的偏差与 endpoint 位置和内容构造高度相关；Qwen prompt counter 的同类宏观因素没有稳定的 held-out 解释力。这个模型差异在 correct-only 子集仍保留。</p>

<h2>4. Trace counter：具体用了哪些自变量</h2>
<p>Trace 分析合并多个 state site，因此最基础 block 还包含 <code>site_scope</code>。T1 描述 trace 中“走到哪里”，T2 再描述生成形式和样本结构；T3 仍是后验结果诊断。</p>
{table(["Block", "实际自变量", "定义/作用"], trace_feature_rows)}
<div class="figure-grid">
<figure>{line_chart('trace','noise_total_rms')}<figcaption>图 2a｜Trace counter 的总 RMS。横轴为累积 feature block；纵轴为 seed-held-out R²。HGB 对两模型均有很高解释力，且 all/correct-only 曲线重合。</figcaption></figure>
<figure>{line_chart('trace','count_axis_deviation_abs')}<figcaption>图 2b｜Trace counter 的第一计数轴绝对偏差。坐标轴与线型同图 2a；T3 的结果变量几乎不带来增量。</figcaption></figure>
</div>
<h3>固定的最终 pre-outcome block（T2）下，各回归器结果</h3>
{trace_table}
<p>在固定 HGB 下，Trace form 对总 RMS 的 held-out R² 为 Qwen 0.914（correct-only 0.909）和 Gemma 0.852（0.850）；对第一计数轴绝对偏差为 Qwen 0.669（0.673）和 Gemma 0.553（0.566）。ElasticNet 已能解释一部分总 RMS，但 HGB 显著更强，说明关系包含非线性和变量交互。</p>
<p>T0→T1→T2 的递增也有含义。Qwen 总 RMS 为 0.808→0.899→0.914，Gemma 为 0.599→0.745→0.852：running index/gold count 已解释主干，trace 位置再解释一部分，marker/termination/order、长度和重复结构继续解释剩余变化。T3 基本不再增加 R²。</p>
<p class="conclusion"><b>Trace 结论：</b>Trace counter 虽然几何上更分散，但这种分散大多不是不可预测的白噪声；它与计数阶段、生成位置和 trace 形式高度耦合，而且在只保留答对样本后几乎不变。</p>

<h2>5. 为什么 trace counter 比 prompt counter 更 noisy</h2>
<p>Prompt endpoint 是输入中的固定语义锚点；模型只需编码“目前见过第几个 needle”。Trace state 则位于自回归生成过程中，同时叠加至少四类变化：当前 count、已生成长度和绝对位置、marker/item/termination 等局部句法状态，以及此前生成 token 造成的历史依赖。因此相同 running index 的 trace hidden state 并不处在相同计算上下文。</p>
<ul>
  <li><b>几何幅度：</b>Qwen trace 总 RMS 0.970，而 prompt 为 0.119；Gemma trace 为 0.667，而 prompt 为 0.345（all samples）。</li>
  <li><b>正确样本仍 noisy：</b>correct-only 几乎得到同样 RMS，排除了“主要由答错样本拉散”的解释。</li>
  <li><b>偏差主要在 counter subspace 外：</b>Qwen trace 的 rank-3 正交能量占 92.8%，Gemma 占 85.6%；它更像生成上下文叠加在 count signal 上，而不是 count 本身完全丢失。</li>
  <li><b>noise 可被结构因素预测：</b>T2-HGB 对总 RMS 的 R² 达 0.85–0.91；若是与所有测量变量无关的独立随机扰动，不会得到这种跨-seed解释力。</li>
</ul>
<p class="conclusion"><b>综合结论：</b>更准确的表述不是“trace 的计数器更差”，而是“trace counter 的 hidden state 同时承载 count 与生成过程状态，因而围绕 count centroid 的条件方差更大；其中大部分可由 trace 阶段和形式解释”。</p>

<h2>6. 统计边界与论文写法</h2>
<div class="warning"><b>不能把 residual 直接称为随机噪声。</b>当前设计没有对完全相同 prompt/trace 条件做重复随机生成，因此回归后未解释部分混合了遗漏变量、特征测量误差、模型非线性和可能的随机性。本文统一称其为“within-count geometric deviation”或“unexplained deviation”。</div>
<ul>
  <li>这些 feature block 高度相关，增量 R² 是按既定加入顺序计算的，不是唯一的方差分解；不能把不同因素的边际 η² 相加。</li>
  <li><code>realization_seed</code> 与 <code>content_seed</code> 是构造控制变量，不应被解释成机制；其用途是检验新 parent seed 上仍可预测的构造依赖。</li>
  <li>结果变量 P3/T3 是 post-outcome diagnostic。它只回答“答对/答错是否与已有偏差额外关联”，不能解释偏差如何产生。</li>
  <li>R² 使用 log1p target，因此“解释 90%”指 log-noise 的 held-out variance，而非原始 RMS 的 90%。</li>
</ul>
<p class="conclusion"><b>可用于论文的最小主张：</b>Native trace counter 的 within-count geometry 显著比 prompt running counter 更分散；这种额外离散在 correct-only 样本中持续存在，并可由 count stage、trace position 和 trace-form covariates 在新 seed 上高度预测。由此，trace noise 更符合“结构化、上下文依赖的表征变异”，而非错误样本或不可预测随机噪声。</p>

<p class="small">复现输入：<code>work/v440_native_thinking_noise_results</code>；生成器：<code>scripts/build_realistic_niah_v4_4_0_native_thinking_noise_report.py</code>。所有数值均直接读取保存的 CSV/audit，不由 HTML 手工重算。</p>
</main></body></html>
"""

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(document, encoding="utf-8")
print(OUTPUT)
