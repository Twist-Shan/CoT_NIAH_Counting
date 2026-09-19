"""Render the frozen, fresh-cohort Native-aligned continuation experiment."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_v58_commit_query_section import ORTH, SELF, ci, pick

EXPERIMENT = "v58_native_continuation_20260905"
SCOPES = {"item_end_w1": "单 marker", "item_span_w2": "完整两-token item"}
METRICS = ["donor_marker_adoption", "donor_continuation_adoption", "donor_prefix_h2", "donor_prefix_h3", "donor_prefix_h4"]
LABELS = ["Next marker", "Distinguishing\nprefix (q<=4)", "Prefix h=2", "Prefix h=3", "Prefix h=4"]
NAMES = ["下一 marker", "最短可区分前缀（q≤4）", "完整两步前缀", "完整三步前缀", "完整四步前缀"]


def embedded(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def audited_row(frame, condition, metric):
    rows = frame.loc[frame.condition.eq(condition) & frame.metric.eq(metric)]
    if len(rows) != 1:
        raise ValueError((condition, metric, len(rows)))
    return rows.iloc[0]


def fraction(row):
    return f"{int(row['sum'])}/{int(row.pairs)}"


def build_section(analysis: Path, layer_figure: Path, behavior_figure: Path):
    root = analysis / EXPERIMENT
    manifest = json.loads((root / "manifest.json").read_text())
    selection = json.loads((root / "selected_layers.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    assert manifest["status"] == "complete"
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    assert manifest["selection_sha256"] == manifest["files"]["selected_layers.json"]
    assert selection["plan_sha256"] == manifest["files"]["plan.json"]
    assert not selection["confirmation_inference_started"]
    assert selection["selection_completed_unix"] < manifest["completed_unix"]
    data = {}
    for scope in SCOPES:
        assert selection["scopes"][scope]["selected_layer"] == 2
        data[scope] = {name: pd.read_csv(root / scope / f"{name}.csv") for name in
                       ["rollout_contrasts", "local_contrasts", "continuation_audit", "rollout_trials"]}
    full_rows = [d["rollout_trials"].loc[d["rollout_trials"].condition.eq("full_donor_patch")]
                 .set_index("pair_id").sort_index() for d in data.values()]
    same_full_continuation = int(full_rows[0].continuation_tokens.eq(full_rows[1].continuation_tokens).sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained", sharey=True)
    for ax, (scope, label) in zip(axes, SCOPES.items()):
        sel = selection["scopes"][scope]
        frame = pd.DataFrame(sel["layer_summaries"])
        for metric, color, style, title in [
            ("median_prompt_mean", "#2563a6", "o-", "Median prompt-mean shift"),
            ("forward_median", "#23856d", "s--", "Forward pair median"),
            ("backward_median", "#d97706", "^--", "Backward pair median")]:
            ax.plot(frame.layer, frame[metric], style, color=color, label=title)
        ax.axhline(sel["threshold"], color="#7b8794", ls=":", label="95% of eligible peak")
        ax.axvline(sel["selected_layer"], color="#2563a6", alpha=.25, lw=7)
        ax.axvspan(3.65, 4.2, color="#dddddd", alpha=.6)
        ax.set(xlabel="Patched post-block depth", xticks=[1, 2, 3, 4],
               xticklabels=["L1", "L2\nselected", "L3", "L4\nexcluded"],
               title="Single marker" if scope == "item_end_w1" else "Two-token item")
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Discovery donor/receiver log-odds shift")
    axes[0].legend(fontsize=8)
    fig.suptitle("Discovery-only layer selection | 15 identifiable pairs / 10 prompts", fontsize=13)
    layer_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(layer_figure, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    for i, scope in enumerate(SCOPES):
        layer = selection["scopes"][scope]["selected_layer"]
        frame = data[scope]["rollout_contrasts"]
        clean = [pick(frame, m, SELF, layer).control_mean * 100 for m in METRICS]
        full = [pick(frame, m, SELF, layer).treatment_mean * 100 for m in METRICS]
        orth = [pick(frame, m, ORTH, layer).control_mean * 100 for m in METRICS]
        for values, color, title in [(clean, "#7b8794", "Clean / self"), (orth, "#d97706", "Full-norm orth. mean"), (full, "#2563a6", "Full donor")]:
            axes[i, 0].plot(range(5), values, "o-", color=color, label=title)
        for c, (contrast, color, title) in enumerate([(SELF, "#2563a6", "Full - self"), (ORTH, "#d97706", "Full - full-norm orth.")]):
            rows = [pick(frame, m, contrast, layer) for m in METRICS]
            v = np.array([r.effect for r in rows]) * 100
            lo = np.array([r.ci_low for r in rows]) * 100
            hi = np.array([r.ci_high for r in rows]) * 100
            axes[i, 1].errorbar(np.arange(5) + (c-.5)*.13, v, yerr=[np.maximum(0, v-lo), np.maximum(0, hi-v)],
                               fmt="o", capsize=3, color=color, label=title)
        axes[i, 0].set_ylim(0, 65)
        axes[i, 0].set_ylabel("Donor-prefix adoption (%)")
        axes[i, 1].set_ylabel("Paired adoption change (percentage points)")
        axes[i, 1].axhline(0, color="#777", lw=.8)
        for j in range(2):
            axes[i, j].set_xticks(range(5), LABELS, fontsize=8)
            axes[i, j].set_xlabel("Continuation endpoint (input-defined eligible subsets)")
            axes[i, j].set_title(f"{'ABCD'[i*2+j]} | {'Single marker' if i == 0 else 'Two-token item'} / L{layer}")
            axes[i, j].grid(axis="y", alpha=.2)
            axes[i, j].spines[["top", "right"]].set_visible(False)
            if i == 0:
                axes[i, j].legend(fontsize=8)
    fig.suptitle("Native-aligned free continuation | fresh 60-prompt confirmation", fontsize=14)
    fig.savefig(behavior_figure, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    rows, conditional_rows, scope_paragraphs = [], [], []
    for scope, label in SCOPES.items():
        layer = selection["scopes"][scope]["selected_layer"]
        frame, audit = data[scope]["rollout_contrasts"], data[scope]["continuation_audit"]
        for m, title in zip(METRICS, NAMES):
            s, o = pick(frame, m, SELF, layer), pick(frame, m, ORTH, layer)
            a = audited_row(audit, "full_donor_patch", m)
            rows.append(f"<tr><td>{label}</td><td>{title}</td><td>{int(s.pairs)} / {int(s.prompts)}</td>"
                        f"<td>{100*s.control_mean:.1f}%</td><td>{100*o.control_mean:.1f}%</td>"
                        f"<td>{100*s.treatment_mean:.1f}%（{fraction(a)}）</td><td>{ci(o,100,2)}</td></tr>")
        for h in [2, 3, 4]:
            a = audited_row(audit, "full_donor_patch", f"conditional_h{h}_given_first_transfer")
            conditional_rows.append(f"<tr><td>{label}</td><td>{h}</td><td>{fraction(a)}</td><td>{100*a.pair_mean:.1f}%</td></tr>")
        forward = pick(frame, METRICS[0], ORTH, layer, "forward")
        backward = pick(frame, METRICS[0], ORTH, layer, "backward")
        same = pick(frame, METRICS[0], ORTH, layer, "same_commit_marker")
        final = pick(frame, "final_correct", SELF, layer)
        route = pick(frame, "donor_route_adoption", ORTH, layer)
        low = pick(frame, METRICS[0], "count_subspace_transplant - norm_matched_orthogonal_patch", layer)
        qk = pick(data[scope]["local_contrasts"], "qk_margin_mean", ORTH, layer)
        scope_paragraphs.append(f"<p><b>{label}的次要检验。</b>下一marker full−orth效应：forward {ci(forward,100,2)} pp，backward {ci(backward,100,2)} pp；"
                                f"same-commit-marker子组 {ci(same,100,2)} pp（{int(same.pairs)} pairs/{int(same.prompts)} prompts）。"
                                f"Rank-3投影相对投影等范数对照为{ci(low,100,2)} pp。"
                                f"固定下一query的bank平均QK-relative margin full−orth为{ci(qk)}；"
                                f"Targeted-bank occurrence argmax full−orth为{ci(route,100,2)} pp；"
                                f"final-count accuracy clean/self {100*final.control_mean:.1f}%→full {100*final.treatment_mean:.1f}%，差{ci(final,100,2)} pp。"
                                "这些结果分别描述内容依赖、低维投影、检索位置与整题输出，不作为continuation充分性的附加通过条件。</p>")

    span = data["item_span_w2"]["rollout_contrasts"]
    first = pick(span, METRICS[0], SELF, 2)
    strict = pick(span, METRICS[0], ORTH, 2)
    multi = pick(span, "donor_prefix_h4", ORTH, 2)
    assert strict.ci_low > 0, "Revisit the written local sufficiency conclusion."
    fragment = f"""
<h3 id="native-continuation">Experiment D · Discovery-selected donor-directed continuation</h3>
<div class="purpose"><span class="label">实验目的。</span>按Native-thinking §5.3的选层逻辑，检验完成item的contextual state是否足以使后续自由生成转向donor的后续内容。
成功标准为相对self及完整位移等范数对照的donor-directed continuation；整题count变化、attention-bank argmax翻转及抽象±1运算分别属于其他待检验主张。</div>
<p><b>数据与模型。</b>保持v58 Thinking step10,000 checkpoint、原始无index trace和256-char receiver；没有重训。
从既有外部count-10行为集排除实验C全部80条prompt、历史progress样本及head-selection/reporting的canonical keys；余下120条按hash排序，取20条discovery和60条confirmation。
新机制cohort与旧实验无prompt/key重合，但来自已评过行为的数据，不能称为新的未见行为测试集；仍只有一个训练seed。
Discovery固定donor k=6、receiver k=5/7；confirmation为donor k∈{{4,6,8}}、receiver k−1/k+1，共360 pairs。
不按clean正确率或干预效果筛选。两种scope均报告：单marker和完整&lt;Sep&gt; marker item。</p>
<p><b>位置、干预与控制。</b>在donor正文删除/增加2个非needle filler，使正文长度为254/258，将donor item的绝对位置与receiver对齐；十个needle身份/顺序及trace文本不变。
Receiver维持256字符。给定gold trace prefix到receiver第r个item末端，patch post-block residual，随后greedy自由生成最多28 tokens，不强制下一&lt;Sep&gt;或marker。
替换仅固定在原item位置，不干预未来query或item。八个条件为clean、self、full donor、rank-3 projected donor、投影位移等范数orthogonal及3个完整位移等范数orthogonal；basis仅在20条discovery拟合。</p>
<div class="formula"><b>选层规则。</b>对每个discovery pair计算 Δℓ=[log p(d<sub>next</sub>)−log p(r<sub>next</sub>)]<sub>full</sub>−[log p(d<sub>next</sub>)−log p(r<sub>next</sub>)]<sub>clean</sub>，query固定为item之后原本的&lt;Sep&gt;。
要求forward/backward的pair中位数均&gt;0；将pair效应在prompt内平均，再取跨prompt中位数Sℓ，选择达到eligible peak的95%的最早层。
末层L4没有下游block，预先排除。Synthetic L1表示第一个block之后，对应Native的零基L0。
Attention、NCC、free generation及confirmation均不参与选层；选层JSON在读取confirmation states前写入并冻结。</div>
<p><b>选层结果与覆盖率。</b>两种scope均选L2。重复marker使部分pair的donor/receiver下一字符相同：40个discovery pairs中仅15个（10/20 prompts；forward8、backward7）能用于身份区分的log-odds统计。
这些不可区分pairs保留在数据导出，不能为选层提供身份判别信息。相较Native多token city候选，这是明确的测量适配与覆盖限制。</p>
<figure><img src="{embedded(layer_figure)}" alt="Discovery-only layer selection for two patch scopes">
<figcaption>图7c｜两个patch范围独立选层。横轴为synthetic post-block depth；纵轴为donor/receiver log-odds的paired变化。
蓝线为跨prompt中位数Sℓ，绿/橙虚线分别为forward/backward pair中位数，灰色水平点线为95%阈值，蓝色竖带为选中L2，灰区为预先排除的末层L4。
仅含15个身份可区分的discovery pairs，无confirmation信息；曲线没有置信区间。当前结论：按预定规则可在confirmation前冻结L2，选层稳健性仍需更多discovery prompts或训练seeds检验。</figcaption></figure>
<div class="formula"><b>Continuation指标。</b>①首marker adoption=1[generated marker=d<sub>next</sub>]，仅当d<sub>next</sub>≠r<sub>next</sub>可识别。
②令D、R为donor/receiver的gold未来marker序列；q=min{{h≤4:D<sub>1:h</sub>≠R<sub>1:h</sub>且双方前缀存在}}。
最短可区分前缀adoption=1[G<sub>1:q</sub>=D<sub>1:q</sub>]；无q则记为不可识别。
③固定h=2/3/4，要求双方前缀存在且不同，完整匹配全部h个marker才计成功。缺失输出或生成错误均计0；不按生成是否正确筛样本。
每个条件先在prompt内平均，再对prompt等权平均；10,000次paired prompt bootstrap给出95%区间，未作多重比较校正。</div>
<div class="example"><span class="label">说明性示例。</span>Receiver未来为[a,b,c]，donor未来为[a,c,b]。首个a无法区分两个进度；最短可区分前缀的q=2。
生成[a,c]满足donor前缀；生成[a,b]不满足。若patch后先完成donor的第一个marker，再沿donor后续序列继续，提供多步continuation证据；该指标仍不能唯一定位正文中重复字符的具体occurrence。</div>
<figure><img src="{embedded(behavior_figure)}" alt="Donor-directed free continuation and paired control contrasts">
<figcaption>图7d｜独立60-prompt confirmation的continuation结果。上/下排为单marker/完整两-token item，均在discovery-selected L2。
横轴为五种continuation指标；从左到右eligible pairs为162、257、189、175、145，均覆盖53个prompts，另外7个prompt在这些前缀内不能区分donor/receiver。
左列纵轴为prompt等权adoption，灰色clean/self、蓝色full donor、橙色三个full-norm orthogonal的均值；连线只辅助阅读，五个点不构成同一分母的生存曲线。
右列纵轴为paired百分点差，蓝色full−self，橙色full−orth，误差棒为prompt-clustered 95% CI；0表示无差异。</figcaption></figure>
<table><thead><tr><th>Scope</th><th>Endpoint</th><th>Pairs / prompts</th><th>Clean/self</th><th>Full-norm orth.</th><th>Full donor（raw successes/pairs）</th><th>Full−orth pp [95% CI]</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p><b>主结果。</b>完整item patch将首marker donor adoption从{100*first.control_mean:.1f}%提高到{100*first.treatment_mean:.1f}%；相对严格等范数对照的增量为{ci(strict,100,2)} pp。
因此v58已得到contextual item state对donor-directed自由continuation的局部因果充分性证据。
四步前缀相对严格对照的变化为{ci(multi,100,2)} pp；多步转移强度低于首步，不能将首步成功直接当作持续完整递推。</p>
<p><b>两个scope的比较。</b>单marker与完整item的full-donor自由输出逐token一致（{same_full_continuation}/360 pairs）。
两种scope相对orthogonal对照的增量不同，来源于各自干预空间内的随机对照结果；不能据此声称增加&lt;Sep&gt; patch提高了full-donor转移率。
单marker四步前缀相对严格对照的95%区间跨0，应保留这一范围差异。当前结论：局部continuation效应在两种scope均出现，完整item的额外独立贡献尚未建立。</p>
<h4>与Native-thinking多步continuation审计的对应</h4>
<p>下面按Native审计口径，另在“首marker已跟随donor”的pair中统计完整h步前缀；每一行还要求双方未来长度足够，故分母随h变化。
这是按结果条件化的描述性比例，不能与上面的无条件因果增量混用，也不表示后续每一步都经独立干预验证。</p>
<table><thead><tr><th>Scope</th><th>完整前缀长度h</th><th>首步转移后仍匹配h步 / 可评估pairs</th><th>Pair等权比例</th></tr></thead><tbody>{''.join(conditional_rows)}</tbody></table>
<p>Native最新报告的discovery-selected item-span L0有43/60首city转移；其中43/43继续donor第二个successor，可观察四步的子集为27/27。
L16是同一10个confirmation seeds、k=6的固定中层复核，首city转移16/20，不能当作第二次独立confirmation。
Synthetic与Native在“contextual state transplant → donor-directed continuation”的证据类型上对齐；synthetic多步持续性更弱。
两边的token单位、marker重复、patch span、深度和样本结构不同，raw比例不能作为严格matched benchmark；Native结果本身也不单独隔离抽象±1算术算子。</p>
<p><b>Prefix来源的差异。</b>本次synthetic干预前使用gold trace prefix，干预后的后缀完全自由生成；Native主要证据来自自然first-pass trace。
因此已确认的充分性以给定正确prefix为条件。对synthetic从prompt开始自由生成、在实际到达的item state上在线干预，尚未由本次实验覆盖。</p>
{''.join(scope_paragraphs)}
<p><b>解释范围与未查明原因。</b>完整state同时含内容和进度，rank-3投影未显示同等首步效应；因此当前结果支持contextual-state continuation，纯进度方向的独立可执行性仍为结论待定。
Patch后多步效应衰减的原因未查明。原prefix其余state、位置线索和内容信息的独立作用，需在固定样本下分别干预验证。
本次同时采用新cohort、Native donor-index约定和discovery-selected L2，不能用与实验C的数值差直接证明“换层导致提升”。</p>
<div class="conclusion"><span class="label">Experiment D结论。</span>可与Native-thinking对齐的主张为：无显式index trace中，完成item的contextual state对后续donor-directed生成具有局部因果充分性，并可延续多个marker。
无需额外要求final count改变或证明universal aggregator。稳定的长程进度递推、content-independent arithmetic counter、targeted-to-count完整mediation仍未建立。
本次5,760条rollout无token-cap截断；clean与self逐条一致，末层logit null为0；完整输出哈希已核验。</div>
"""
    return fragment, {"experiment": EXPERIMENT, "manifest": manifest, "selection": selection,
                      "full_donor_identical_continuations_across_scopes": same_full_continuation,
                      "plan": plan, "figures": [str(layer_figure), str(behavior_figure)]}
