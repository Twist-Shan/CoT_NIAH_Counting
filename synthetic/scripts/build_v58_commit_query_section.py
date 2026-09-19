"""Render the frozen commit-query assay and its explicitly secondary aligned scope."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCOPES = ["v58_commit_query_20260905", "v58_aligned_item_query_20260905"]
SELF = "full_donor_patch - self_patch"
ORTH = "full_donor_patch - full_norm_orthogonal_mean"


def pick(frame, metric, contrast=ORTH, layer=1, subset="all"):
    rows=frame.loc[frame.split.eq("confirmation") & frame.layer.eq(layer) & frame.subset.eq(subset)
                   & frame.metric.eq(metric) & frame.contrast.eq(contrast)]
    if len(rows)!=1:
        raise ValueError((metric,contrast,layer,subset,len(rows)))
    return rows.iloc[0]


def ci(row, multiplier=1, digits=3):
    return f"{row.effect*multiplier:+.{digits}f} [{row.ci_low*multiplier:+.{digits}f}, {row.ci_high*multiplier:+.{digits}f}]"


def build_section(analysis: Path, figure_path: Path):
    datasets=[]
    for scope in SCOPES:
        root=analysis/scope
        manifest=json.loads((root/"manifest.json").read_text())
        assert manifest["status"]=="complete"
        for name,expected in manifest["files"].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected, (scope,name)
        datasets.append((pd.read_csv(root/"local_contrasts.csv"),pd.read_csv(root/"rollout_contrasts.csv"),manifest))
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout="constrained")
    colors=["#2563a6","#d97706"]
    scope_labels=["Single marker / cross-position (primary)","Two-token item / position-aligned (sensitivity)"]
    for i,(local,roll,_) in enumerate(datasets):
        for j,(metric,ylabel) in enumerate([("routing_y","Paired change in bank-summed attention"),("qk_margin_mean","Paired change in mean log A(d)/A(r)")]):
            ax=axes[i,j]
            for c,(contrast,label) in enumerate([(SELF,"Full donor - self"),(ORTH,"Full donor - full-norm orth.")]):
                rows=[pick(local,metric,contrast,layer) for layer in range(1,5)]
                values=np.array([r.effect for r in rows]); lows=np.array([r.ci_low for r in rows]); highs=np.array([r.ci_high for r in rows])
                ax.errorbar(np.arange(1,5)+(c-.5)*.14,values,yerr=[np.maximum(0,values-lows),np.maximum(0,highs-values)],fmt="o-",capsize=3,color=colors[c],label=label)
            ax.axhline(0,color="#777",linewidth=.8)
            ax.set_xticks(range(1,5),["L1\nprimary","L2","L3\nQK null","L4\nnull"])
            ax.set_xlabel("Patched post-block depth")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{'ABCDEF'[i*3+j]} | {['Attention mass','QK-relative routing'][j]}")
            if i==0 and j==0:
                ax.legend(fontsize=8)
        ax=axes[i,2]
        metrics=["donor_marker_adoption","donor_route_adoption","donor_first_three_routes_exact","final_correct"]
        for c,contrast in enumerate([SELF,ORTH]):
            rows=[pick(roll,m,contrast) for m in metrics]
            v=np.array([r.effect for r in rows])*100
            lo=np.array([r.ci_low for r in rows])*100; hi=np.array([r.ci_high for r in rows])*100
            ax.errorbar(np.arange(4)+(c-.5)*.16,v,yerr=[np.maximum(0,v-lo),np.maximum(0,hi-v)],fmt="o",capsize=3,color=colors[c])
        ax.axhline(0,color="#777",linewidth=.8)
        ax.set_xticks(range(4),["Next\nmarker*","Next needle\n(bank argmax)","Donor route\nprefix**","Final count\ncorrect"])
        ax.set_ylabel("Paired change (percentage points)")
        ax.set_title(f"{'CF'[i]} | L1 free continuation")
        for a in axes[i]:
            a.grid(axis="y",alpha=.22)
            a.spines[["top","right"]].set_visible(False)
        axes[i,0].text(0,1.18,scope_labels[i],transform=axes[i,0].transAxes,fontweight="bold",fontsize=11)
    fig.suptitle("Commit state changes retrieval preference more reliably than the retrieved occurrence",fontsize=14)
    figure_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(figure_path,dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    encoded=base64.b64encode(figure_path.read_bytes()).decode()
    rows=[]
    for label,(local,roll,_) in zip(["单 marker；跨位置","两-token item；位置对齐（补充）"],datasets):
        for metric,title,factor in [("routing_y","Attention mass full−self",1),("qk_margin_mean","QK margin full−orth",1)]:
            row=pick(local,metric,SELF if metric=="routing_y" else ORTH)
            rows.append(f"<tr><td>{label}</td><td>{title}</td><td>{ci(row,factor)}</td></tr>")
        for metric,title in [("donor_marker_adoption","下一 marker adoption full−orth（pp）"),("donor_route_adoption","下一 needle adoption full−orth（pp）"),("final_correct","Final accuracy full−self（pp）")]:
            rows.append(f"<tr><td>{label}</td><td>{title}</td><td>{ci(pick(roll,metric,SELF if metric=='final_correct' else ORTH),100,2)}</td></tr>")
    primary,roll,_=datasets[0]
    aligned,aroll,_=datasets[1]
    marker=pick(roll,"donor_marker_adoption",SELF)
    needle=pick(roll,"donor_route_adoption",SELF)
    final=pick(roll,"final_correct",SELF)
    aligned_marker=pick(aroll,"donor_marker_adoption",SELF)
    aligned_needle=pick(aroll,"donor_route_adoption",SELF)
    aligned_final=pick(aroll,"final_correct",SELF)
    primary_trials=pd.read_csv(analysis/SCOPES[0]/"local_trials.csv")
    selected=primary_trials.loc[primary_trials.split.eq("confirmation") & primary_trials.layer.eq(1) & primary_trials.condition.eq("full_donor_patch")]
    norms=selected.groupby("same_commit_marker").full_delta_norm.median()
    aligned_trials=pd.read_csv(analysis/SCOPES[1]/"rollout_trials.csv")
    matched=aligned_trials.loc[aligned_trials.condition.eq("full_donor_patch")].merge(
        aligned_trials.loc[aligned_trials.condition.eq("clean")],on="pair_id",suffixes=("_full","_clean"))
    diagnostics={"L1_same_marker_median_full_delta_norm":float(norms[True]),
                 "L1_different_marker_median_full_delta_norm":float(norms[False]),
                 "aligned_changed_continuations":int(matched.continuation_tokens_full.ne(matched.continuation_tokens_clean).sum()),
                 "aligned_changed_marker_counts":int(matched.generated_marker_count_full.ne(matched.generated_marker_count_clean).sum()),
                 "aligned_changed_final_answers":int(matched.generated_final_count_full.ne(matched.generated_final_count_clean).sum()),
                 "diagnostic_status":"post-outcome descriptive analysis, not a newly selected primary endpoint"}
    fragment=f"""
<h3 id="commit-query">Experiment C · Commit state → next retrieval：与 Native-thinking §5.3 对齐</h3>
<p><b>历史实验说明。</b>本节保留预定L1实验及同cohort补充结果。后续按Native选层规则完成的新cohort自由continuation检验见Experiment D；本节occurrence-level attention结果不作为否定局部continuation充分性的条件。</p>
<div class="purpose"><span class="label">实验目的。</span>检验已完成 item 的状态是否因果影响下一轮应该检索哪条 needle。
局部 routing 改变与最终 count 改变是不同主张；最终答案不变并不能否定局部因果边，但下一字符变化也不能单独证明检索位置改变。</div>
<p><b>模型与样本。</b>不重训；固定 v58 Thinking step 10,000。Count 固定为10，与Qwen §5.3一致。
从既有、已测过行为的外部样本中，排除历史head-selection/reporting keys及历史progress prompts后，按hash排序取20条拟合basis、60条做mechanistic confirmation。
每条取k=4/6/8及donor k±1，共360 confirmation pairs；不按正确率、attention或marker是否重复筛选。
这60条是新的机制评估样本，不是未见过的行为测试集。Patch层L1沿用历史discovery进度干预选层，L2为预定敏感性分析，L3/L4为结构对照；没有按本轮confirmation结果重选主层。</p>
<p><b>干预与对照。</b>主实验只替换完成第k项的marker post-block state，不替换下一query。
Receiver gold prefix到此结束，局部测量固定追加原本的&lt;Sep&gt;；自由生成从item末端开始，后续不强制任何token。
条件为clean、self、完整donor、rank-3子空间donor投影、投影位移等范数orthogonal，以及3个完整donor位移等范数orthogonal。
所有方向在读取confirmation结果前冻结。随机对照匹配干预范数，不是把低attention head当作null。</p>
<div class="formula"><b>计算定义。</b>Discovery k=1…9类均值中心化后SVD取前三个正交方向U；P=UUᵀ。
子空间干预h′=h<sub>r</sub>+P(h<sub>d</sub>−h<sub>r</sub>)；orthogonal方向为(I−P)z并归一化到对应位移范数。
Targeted bank固定为L4H5/H0/H1/H4。Y=Σ<sub>h∈bank</sub>[A<sub>h</sub>(donor-successor)−A<sub>h</sub>(receiver-successor)]；
QK-relative margin=mean<sub>h</sub>log[A<sub>h</sub>(d)/A<sub>h</sub>(r)]，可消去每个head的softmax分母。
Next-needle adoption要求Σ<sub>h</sub>A<sub>h</sub>在正文十个needle位置中的argmax等于donor-successor。
Marker adoption仅在两个successor字符不同时统计（157 pairs、50 prompts），不能把重复字符当作特定needle的命中。
配对差异先在prompt内平均，再按prompt进行10,000次bootstrap；表中为95%区间，多个层/子组区间未做多重比较校正。</div>
<div class="example"><span class="label">说明性示例。</span>Receiver已生成第4项，下一项应检索正文needle 5；donor已完成第5项。
若patch后query更偏向needle 6，支持局部progress→routing；若随后依次检索6、7、8，才进一步支持持续的进度转移。
若needle 5与6都叫字符a，仅看到输出a无法判断具体occurrence；source-position attention可补充定位，多个未来字符组成的可区分前缀也可检验donor-directed continuation（见Experiment D）。</div>
<p><b>按最新版Native设置补充的检验。</b>Qwen §5.3用绝对位置对齐的maximal-common item span（不总等于完整item），并报告L16中层结果。
主实验结束后，我们在同一批样本上补做两-token完整item（&lt;Sep&gt;+marker）patch：donor k+1删去最后两个非needle filler，donor k−1追加两个非needle filler，使正文长度为254/258，item两位置与256长receiver精确一致。
Receiver正文、trace及下一query不变；donor的十个needle身份/顺序和trace文本也不变。
此补充同时改变位置对齐和patch范围，不能单独归因哪一项；它不是独立confirmation，也没有升级为主结果。
两-token子空间在拼接后的1024维state上拟合，范数对照也在同一空间计算。</p>
<figure><img src="data:image/png;base64,{encoded}" alt="Commit-query primary and position-aligned sensitivity with paired bootstrap confidence intervals">
<figcaption>图7b｜上排为单marker跨位置主实验，下排为位置对齐两-token item补充实验。
A/D横轴为patch depth，纵轴为bank-summed attention差异的配对变化；B/E为相同横轴和消除softmax分母的QK-relative变化。
C/F固定主层L1，横轴依次为下一marker adoption、下一needle attention argmax adoption、donor后续routing前缀、final-count accuracy，纵轴为百分点变化。
蓝色full−self，橙色full−3个full-norm orthogonal均值；误差棒为prompt-clustered 95% CI。
*Marker仅限successor字符不同的子集。**Routing前缀要求前min(3,10−donor k)次检索均与donor后续一致，包含最后仅剩1次检索的pairs，不能称为所有pairs均检验了完整三步。</figcaption></figure>
<table><thead><tr><th>Scope</th><th>Endpoint / contrast</th><th>Effect [95% CI]</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p><b>主实验结果。</b>L1 full−self attention变化为{ci(pick(primary,'routing_y',SELF))}；full−full-norm orthogonal为{ci(pick(primary,'routing_y'))}，QK变化为{ci(pick(primary,'qk_margin_mean'))}。
QK结果支持检索相对偏好改变；full patch后平均Y仍为{pick(primary,'routing_y',SELF).treatment_mean:.3f}&lt;0，平均仍偏receiver。
自由生成donor-marker adoption为{100*marker.control_mean:.1f}%→{100*marker.treatment_mean:.1f}%（prompt等权），
而donor-needle argmax仅为{100*needle.control_mean:.1f}%→{100*needle.treatment_mean:.1f}%；与完整位移等范数对照的差仅{ci(pick(roll,'donor_route_adoption'),100,2)} pp。
Final accuracy为{100*final.control_mean:.1f}%→{100*final.treatment_mean:.1f}%，其相对self差异的区间跨0。
结论：局部检索偏好和下一marker生成有因果变化；指定bank的occurrence级转移较弱。Continuation充分性与该bank的occurrence argmax属于不同测量。</p>
<p><b>方向与内容限制。</b>主实验相对full-norm orthogonal的attention变化，forward为{ci(pick(primary,'routing_y',subset='forward'))}，backward为{ci(pick(primary,'routing_y',subset='backward'))}；不是对称的±1更新。
相同commit marker子组QK效应仅{ci(pick(primary,'qk_margin_mean',subset='same_commit_marker'))}，该子组可区分successor的66 pairs/47 prompts里marker adoption相对对照的变化为0。
这提示完整patch的大部分行为变化与item内容相连，但尚不能把剩余效应归因于纯进度寄存器。
L3 patch可微弱改变raw attention，却不改变逐head QK-relative margin；L4为零。这由四层架构决定，不是有序表示不存在。</p>
<div class="example"><span class="label">实际反例（不是总体代表性样本）。</span>单marker实验pair <code>047d6841…</code>，receiver完成第6项，donor完成第5项：receiver下一marker应为B，donor successor为g。
Patch后确实先生成g，但targeted-bank attention argmax仍是receiver应检索的needle 7，而非donor successor needle 6；随后继续B、f、f，最终仍输出10。
这个样本说明“下一marker跟随donor”不等于“下一needle检索已跟随donor”。</div>
<p><b>位置对齐补充结果。</b>L1 full−self attention变化为{ci(pick(aligned,'routing_y',SELF))}；相对完整范数orthogonal的QK变化为{ci(pick(aligned,'qk_margin_mean'))}。
自由生成下一marker adoption为{100*aligned_marker.control_mean:.1f}%→{100*aligned_marker.treatment_mean:.1f}%，下一needle argmax为{100*aligned_needle.control_mean:.1f}%→{100*aligned_needle.treatment_mean:.1f}%；
next-needle相对严格对照的差为{ci(pick(aroll,'donor_route_adoption'),100,2)} pp，donor routing前缀差为{ci(pick(aroll,'donor_first_three_routes_exact'),100,2)} pp。
最终accuracy为{100*aligned_final.control_mean:.1f}%→{100*aligned_final.treatment_mean:.1f}%。此处结论以原始效应和区间为准，不能用单字符命中替代source-occurrence转移。</p>
<p><b>低维投影是否足够？</b>Rank-3 transplant相对其投影范数orthogonal对照，主实验QK变化为{ci(pick(primary,'qk_margin_mean','count_subspace_transplant - norm_matched_orthogonal_patch'))}，
但next-needle adoption仅{ci(pick(roll,'donor_route_adoption','count_subspace_transplant - norm_matched_orthogonal_patch'),100,2)} pp。
位置对齐补充的对应值为{ci(pick(aligned,'qk_margin_mean','count_subspace_transplant - norm_matched_orthogonal_patch'))}和
{ci(pick(aroll,'donor_route_adoption','count_subspace_transplant - norm_matched_orthogonal_patch'),100,2)} pp。
结论：低维方向有小幅连续routing效应，尚不足以稳定改写离散检索决策；不能据此声称已找到可执行的纯counter寄存器。</p>
<p><b>与大模型的结果对齐。</b>Qwen最新§5.3的L16 item-span comparator报告20/20 attention朝donor移动、17/20候选donor argmax、16/20首个city跟随donor；该20 pairs来自10 seeds，并非独立于其L0实验的新confirmation。
Synthetic对应的是可重复字符而非多token city、两-token item而非多token span、固定L1主层而非Qwen L16；raw bank mass也不能跨bank大小比较。
我们对齐了干预问题、位置/范围敏感性、连续routing和free-continuation测量，不能把这些比例当作严格matched benchmark，更不能声称已复制其强行为转移。</p>
<h4>为什么局部效应没有变成稳定的progress transfer？</h4>
<p><b>内容混合是当前最直接的线索。</b>事后按commit字符身份分层，主实验L1 donor−receiver的欧氏位移范数中位数：相同字符{norms[True]:.2f}，不同字符{norms[False]:.2f}，相差约{norms[False]/norms[True]:.1f}倍。
完整state差异因此很大程度随字符身份变化。可区分successor字符的子集中，不同commit字符的next-marker donor adoption为7/91→65/91；相同commit字符为6/66→6/66。
这是pair等权的描述性分层，与上文prompt等权主结果的估计量不同；字符分层不是随机化实验，不能单凭该差异证明唯一机制。</p>
<p><b>内容变化与计数变化分离。</b>位置对齐full patch改变了{diagnostics['aligned_changed_continuations']}/360条continuation，但只有{diagnostics['aligned_changed_marker_counts']}/360条的marker总数改变、{diagnostics['aligned_changed_final_answers']}/360条的最终数字改变。
这与此前targeted-content与successor-cardinality ablation分工一致：更符合内容/顺序路径受扰动，而cardinality/termination路径大多保持完整。
固定两-token grammar和未被替换的其他历史state仍提供原progress相关信息，可能使后续重新回到receiver轨迹；冗余state或位置线索是否负责恢复，当前没有直接隔离。</p>
<p><b>站点选择的限制。</b>L1是历史marker-logit/progress选层，不等于全模型最可执行的counter站点；位置对齐L2的QK-relative效应为{ci(pick(aligned,'qk_margin_mean',layer=2))}，大于L1。实验C未测L2自由生成；新cohort的discovery选层与确认实验已补充在Experiment D。
因此不能把L1的弱行为转移推广为“v58没有counter”，也不能归因于RoPE或loss权重；这些训练因素没有在本实验中被操纵。
在相同marker、相同目标位置条件下分离progress方向，仍是进一步区分内容与进度的待检验路径。</p>
<div class="conclusion"><span class="label">Experiment C结论。</span>支持“完成item的contextual state可以改变下一query的检索偏好”。
本节对特定bank的occurrence级控制证据较弱，且效应有方向不对称与内容混合；较强的纯counter持续递推主张仍未建立。
局部continuation充分性不要求抽象±1算子或final answer改变。Experiment D采用预先冻结的选层规则和新cohort进一步检验该主张；内容/进度分离及同一损伤baseline上的完整mediation仍待验证。</div>
"""
    return fragment,{"scopes":SCOPES,"manifests":[x[2] for x in datasets],"primary_layer":1,"figure":str(figure_path),"posthoc_diagnostics":diagnostics}
