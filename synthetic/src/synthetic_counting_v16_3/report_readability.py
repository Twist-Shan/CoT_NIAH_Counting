"""Make the single v16.3 report readable before each visualization is encountered."""

from __future__ import annotations

import re
from pathlib import Path


REPORT_NAME = "v16_3_full_causal_report.html"

LEGACY_REPORT_NAMES = (
    "syn_v16_3_report.html",
    "v16_3_v10_full_causal_report.html",
    "v16_3_rope_complete_report.html",
    "v16_3_rope_complete_report_en.html",
    "v16_3_rope_complete_report_zh.html",
    "v16_3_rope_core_report_zh.html",
    "v16_3_rope_reference_report.html",
)


COMMON_PRIMER = r"""
      <aside class="report-reading-primer" id="reading-primer">
        <h3>阅读前置定义：后文所有图先按这套记号理解</h3>
        <p>本报告先给定义、再给图、最后解释结果。若某幅图使用更特殊的量，图前的蓝色“读图前先定义”框会补充该量的计算、坐标轴、基线和聚合方式。</p>
        <div class="table-wrap"><table class="compact">
          <thead><tr><th>术语</th><th>在本报告中的严格含义</th></tr></thead>
          <tbody>
            <tr><td>Training step / checkpoint</td><td><strong>step</strong> 是 optimizer update 次数；每 500 步保存一个 checkpoint。所有动态曲线的一个点都对应一个保存的 checkpoint。竖虚线 step=1,500 表示 loss 从全序列预测切换为只优化任务输出。</td></tr>
            <tr><td>Nonthinking / thinking</td><td>两者使用相同 RoPE transformer、数据与训练预算。Nonthinking 直接输出 count；thinking 先输出按顺序对齐的 index–marker trace，再输出 count。</td></tr>
            <tr><td>AR exact-count accuracy</td><td>Autoregressive 自由生成时，最终 count 完全正确的样本比例，即 <code>mean[generated_count = gold_count]</code>。0.8 表示 80%，不是 token-level accuracy。</td></tr>
            <tr><td>Attention mass</td><td>某个 query 对指定 token 集合的 softmax 权重之和，范围 [0,1]。例如 trace-readout mass 是答案 query 对全部 trace index 与 marker token 的权重总和。</td></tr>
            <tr><td>Needle enrichment</td><td><code>(目标位置 attention / 全 prompt attention) ÷ (目标位置数 / prompt 长度)</code>。1× 等于在 prompt 内均匀注意；20× 表示目标位置获得的相对份额是均匀基线的 20 倍。</td></tr>
            <tr><td>Correct-k top-1 minus chance</td><td>第 k 个 trace query 在真实目标位置内部把第 k 个 occurrence 排第一的样本比例，减去随机基线 <code>1/count</code>。0 表示不高于随机；0.4 表示高 40 个百分点。</td></tr>
            <tr><td>Nearest-centroid accuracy / ridge R²</td><td>前者用训练区每个 count/progress 的 hidden-state 均值作为 centroid，把 held-out state 分到最近 centroid；后者是在训练区拟合 ridge，再在 held-out state 上报告决定系数。R²=1 是完美线性预测，R²&lt;0 表示比恒定均值预测更差。</td></tr>
            <tr><td>Margin / normalized recovery</td><td>Margin 是两个候选 logit 的差。Normalized recovery = <code>(patched−corrupt)/(clean−corrupt)</code>：0 表示未恢复，1 表示恢复到 clean；它可能小于 0 或大于 1。</td></tr>
            <tr><td>PCA / effective dimension / CKA</td><td>PCA 只对语义标签 centroids 拟合。Effective dimension 为 <code>1/Σ r_j²</code>；越大表示方差分布在更多 centroid-PC 上。Linear CKA 衡量两个 checkpoint 表示几何的相似度，1 表示在该指标下相同。</td></tr>
            <tr><td>线、带与误差条</td><td>除非图前另有说明，点/线是固定 held-out suite 的样本均值；灰线是 matched-random 均值，灰带是固定随机路径的 min–max，<strong>不是置信区间</strong>；只有明确标为 SEM 的误差条才是跨样本标准误。</td></tr>
          </tbody>
        </table></div>
      </aside>
"""


FIGURE_SPECS = (
    (
        "fig-01",
        '<span class="figure-tag">图 1。</span>',
        "四个 panel 的横轴都是 training step。左上纵轴是 AR exact-count accuracy；右上是各 checkpoint 在所有层中取最高的 held-out nearest-centroid accuracy；左下是最终 checkpoint 选定的固定 head 在 heldout-reporting prompts 上的 needle enrichment（1×=prompt 内均匀注意）；右下是答案 query 对全部 trace token 的 attention mass。点是 held-out 均值，竖虚线是 step 1,500。",
    ),
    (
        "hypothesis-h2",
        "v16.3 的两条候选机制与干预位置",
        "这是机制流程图，没有数值坐标。矩形是信息所在的 token/site 或 residual state；实线箭头是本报告直接做过 ablation、patching 或 steering 的方向；灰色虚线只表示待检验的跨路线共享表示。右侧 necessity / sufficiency / geometry 分别指删除是否造成损伤、clean donor 是否能恢复结果、hidden-state 方向是否可执行。",
    ),
    (
        "fig-02",
        '<span class="figure-tag">图 2。</span>',
        "横轴是一个三字符 needle set 的三个字符在训练语料中的频率之和；纵轴是落入该频率 bin 的 needle-set 数量，不是 prompt 数或 occurrence 数。每根柱统计一个频率区间中的集合数；20 个 bin 各 5 个集合意味着 pool 在频率轴上按设计覆盖。",
    ),
    (
        "hypothesis-h1",
        '<span class="figure-tag">假设图 H1。</span>',
        "三个 panel 共用横轴 training step。Thinking panel 同时画 correct-k top-1-minus-chance 与 AR accuracy；nonthinking panel 同时画 top-n target recall 与 AR accuracy；熵 panel 的纵轴是目标 occurrence 内归一化 entropy（0=集中在单一目标，1=均匀覆盖全部目标）。不同量只比较各自随训练的出现时间，不把它们的绝对高度当成同一单位。",
    ),
    (
        "fig-13",
        '<span class="figure-tag">图 13。</span>',
        "所有 panel 横轴都是 training step。Cross-entropy 的纵轴单位是每序列平均负对数似然（越低越好）；AR accuracy 的纵轴是最终 count 完全正确的比例（越高越好）。Raw / task / mixture 指不同 token 范围的固定评估 loss；step 1,500 后只有任务输出 token 继续被优化。",
    ),
    (
        "attention-a1",
        "注意力图 A1：",
        "横轴是所选 checkpoint，纵轴按 Layer 1–4 × Head 0–3 排列；颜色是 broad score <code>B=M_N·H̄_N</code>。其中 <code>M_N</code> 是答案 query 给全部目标 occurrences 的 attention mass，<code>H̄_N</code> 是这些目标内部的归一化 entropy。颜色越高表示既读取目标、又广泛覆盖多个目标。",
    ),
    (
        "attention-a2",
        "注意力图 A2：",
        "横轴是 checkpoint，纵轴是 Layer×Head；颜色是第 k 个 trace-index query 直接给第 k 个 prompt occurrence 的原始 attention mass <code>a(q_k,p_k)</code>，范围 [0,1]。它是对整个 causal prefix 的绝对 softmax 份额，因此不能与目标集合内的条件 top-1 概率混为一谈。",
    ),
    (
        "attention-a3",
        "注意力图 A3：",
        "横轴是 checkpoint，纵轴是 Layer×Head；颜色是 <code>P[p_k 在所有真实目标中 attention 最大]−1/count</code>。0 表示随机水平，正值表示正确的第 k 个 occurrence 被相对优先选择；该指标不说明有多少绝对 attention 真正到达 prompt。",
    ),
    (
        "attention-a4",
        "注意力图 A4：",
        "横轴是 checkpoint，纵轴是 Layer×Head；颜色是最终答案 query 对全部 trace index 和 marker token 的 attention mass 之和，范围 [0,1]。0.9 表示约 90% 的该 head 注意力落在 trace token 上；它表示“从哪里读”，不等于 trace 内容本身正确。",
    ),
    (
        "fig-03",
        '<span class="figure-tag">图 3。</span>',
        "横轴是 training step；纵轴是固定 final-checkpoint head 的 needle enrichment，单位为相对 prompt 内均匀注意的倍数，1× 是无富集。每条线在所有 checkpoint 都追踪同一个 head，而不是每一步重新挑最好 head，因此曲线表示同一电路的形成过程。",
    ),
    (
        "fig-04",
        '<span class="figure-tag">图 4。</span>',
        "两幅图横轴都是 training step，并固定追踪 nonthinking L4H2。左图纵轴是 needle enrichment（1×=均匀）；右图纵轴是 top-n recall：在 prompt 内 attention 最大的 n 个位置中，真实 n 个目标 occurrences 被覆盖的比例。橙/蓝线分别对最终 AR 错误/正确样本取均值。",
    ),
    (
        "fig-05",
        '<span class="figure-tag">图 5。</span>',
        "横轴是 training step；纵轴是答案 query 的 attention mass。Trace mass 是落在全部 trace index+marker 上的权重和；prompt mass 是直接落在原 prompt token 上的权重和。两者都是 [0,1] 的 softmax 份额，但剩余质量还可能落到 BOS、task prefix 等位置，所以两条线不必相加为 1。",
    ),
    (
        "fig-06",
        '<span class="figure-tag">图 6。</span>',
        "横轴是 training step，不是 k。每条线的线型对应一个 query index k=1…10；纵轴是该 k 的 correct-k top-1-minus-chance，0 为随机基线。图中的带汇总不同 k/样本的离散；它用于比较不同 occurrence 的检索在训练中何时出现。",
    ),
    (
        "fig-07",
        '<span class="figure-tag">图 7。</span>',
        "横轴/列是 training step；纵轴/行是 model variant 与 hidden-state depth（0=embedding output，1–4=各 Transformer layer 输出）。颜色是 final-answer state 的 held-out nearest-centroid count accuracy，0–1；每个格子都用训练区 centroids 评估固定 held-out states。",
    ),
    (
        "fig-08",
        '<span class="figure-tag">图 8。</span>',
        "横轴/列是 training step；纵轴/行是 thinking 的 trace site（index 或 marker）与 depth。颜色是 total count 固定为 10 后，用 hidden state 线性预测 progress k 的 held-out ridge R²；1 是完美，0 等于只报均值，负值比均值基线更差。",
    ),
    (
        "fig-16",
        '<span class="figure-tag">图 16。</span>',
        "A–C 的坐标轴是对语义 centroids 拟合的 PC1/PC2，括号是各轴解释的 centroid 方差比例；点标签是 count 或 progress k。D 的横轴是相邻更新起点 k，纵轴是连续两段 centroid 位移的 cosine：1=同方向直线前进，0=正交转弯，−1=反向。",
    ),
    (
        "fig-17",
        '<span class="figure-tag">图 17。</span>',
        "每个 panel 都独立对十个 256 维语义 centroids 拟合 PCA；横轴 PC1、纵轴 PC2，括号是各自解释方差比例。深色编号点是 centroid，淡点是 held-out 单样本投影，灰线按标签顺序连接。不同 panel 的轴方向、正负号和单位不能直接比较。",
    ),
    (
        "fig-18",
        '<span class="figure-tag">图 18。</span>',
        "每个 3D panel 的三个轴是该 panel 独立拟合的 centroid PC1/PC2/PC3；轴标签后的百分比是单轴解释方差。数字是 count/progress，连线只表示标签顺序。Joint panel 才让 index 与 marker 共用一个 PCA 基底；其他 panel 的屏幕方向不可跨图比较。",
    ),
    (
        "fig-18a",
        'id="v162-hs3d-root"',
        "控件先选择 model/site、checkpoint、depth、三条 PC 轴和标签范围。画布三轴就是所选 PC；点是标签 centroid，线按标签连接。上方 variance 是所选 PC 的解释比例；effective dimension、adjacent cosine 与 path straightness 的公式在图后给出。每次切换 checkpoint/site/depth 都会重新拟合 PCA。",
    ),
    (
        "fig-19",
        '<span class="figure-tag">图 19。</span>',
        "四个 panel 横轴都是 layer output 1–4。纵轴依次为：PC1–PC3 累计解释方差（0–1）、effective dimension、相邻<strong>标签位移</strong>的平均 cosine（不是相邻训练 step）、以及首尾 chord 长度/相邻路径总长（1=直线，越小越弯）。",
    ),
    (
        "fig-20",
        '<span class="figure-tag">图 20。</span>',
        "四个 panel 横轴都是 training step。纵轴依次是 held-out ridge R²、PC1 对 centroid 总方差的解释比例、使用全部 centroid PCs 计算的 effective dimension、以及相邻标签位移与 PC1 同向程度的一致性。虚线 step=1,500 是 loss-scope 切换。",
    ),
    (
        "fig-09",
        '<span class="figure-tag">图 9。</span>',
        "横轴是 training step；纵轴是跨 site 线性预测的 held-out R²。每条线固定一个方向（trace→answer 或 answer→trace）和 layer。R²=1 表示源 site 拟合的线性 count/progress 方向可无损迁移；R²&lt;0 表示迁移后比目标 site 的均值基线更差。",
    ),
    (
        "fig-10",
        '<span class="figure-tag">图 10。</span>',
        "横轴是 training step；纵轴是仅用 centroid PC1 坐标线性预测 count 标签得到的 R²，范围通常 0–1；不同线型是 layer。高值表示 count 顺序主要沿一个主轴排列，但不表示整套 representation 只有一维。",
    ),
    (
        "fig-11",
        '<span class="figure-tag">图 11。</span>',
        "横轴是 training step；纵轴是该 checkpoint 与最终 step=10,000 表示矩阵之间的 linear CKA，0 表示在此度量下不相似，1 表示相同。不同线型是 layer；该指标比较样本间几何，不要求神经元逐维对齐。",
    ),
    (
        "fig-21",
        '<span class="figure-tag">图 21。</span>',
        "横轴是按独立 selection split 排序后累计删除的 head 数；纵轴是干预后的<strong>绝对</strong>准确率，不是相对下降。三行对应三条机制，左列在所有 query 删除，右列只在机制定义的 query 删除；彩线是机制排序，灰线/灰带是 matched-random 均值/min–max。",
    ),
    (
        "fig-22",
        '<span class="figure-tag">图 22。</span>',
        "横轴是累计写回 corrupt receiver 的 clean head-output slices 数。左图纵轴是 marker logit margin 的 normalized recovery；右图纵轴是 patch 后正确 marker margin&gt;0 的样本比例。1 表示完全恢复 clean margin/100% 正确；灰带是随机 head 顺序的 min–max。",
    ),
    (
        "fig-23",
        '<span class="figure-tag">图 23。</span>',
        "横轴是累计 patch 的 head slices 数；纵轴是 continue-vs-close 二元 logit margin 的 normalized recovery。左图把 continue donor 写入 close receiver，右图反向；0=仍像 corrupt receiver，1=达到 clean donor 对应 margin。",
    ),
    (
        "fig-24",
        '<span class="figure-tag">图 24。</span>',
        "左图横轴按每层的 pre-residual、加 attention 后、加 MLP 后排列；纵轴是 clean−shortened 条件的 continue logit-lens margin。右图横轴是 layer，纵轴是 attention/MLP additive component 在目标 unembedding 方向上的 clean−short 投影；正值表示该 component 增加 continue 证据。",
    ),
    (
        "fig-25",
        '<span class="figure-tag">图 25。</span>',
        "左侧横轴是按 target-aligned evidence 排名后累计纳入的 MLP feature 数（log₂ 刻度），纵轴是累计正证据或绝对证据占全部证据的比例。右侧横轴是 patch 的 feature 数，纵轴是 normalized recovery 或决策翻转比例；实线为 ranked features，淡虚线为 matched random。",
    ),
    (
        "fig-26",
        '<span class="figure-tag">图 26。</span>',
        "横轴是累计从 donor count=m 写入 receiver count=n 的 final-query head slices 数；纵轴是 expected-count transport slope，即 patched 期望 count 对 donor offset (m−n) 的过原点回归系数。0=不随 donor 变化，1=一比一搬运 donor offset。",
    ),
    (
        "fig-27",
        '<span class="figure-tag">图 27。</span>',
        "横轴是六种 trace 内容/长度干预。左图纵轴是最终 argmax 跟随原 count n 或 n−1 的样本比例；右图纵轴是 <code>z(n)−z(n−1)</code> 相对 clean 的变化。前五类保持总长度与答案位置，最后一类删除 final pair 并让答案位置左移两格。",
    ),
    (
        "fig-28",
        '<span class="figure-tag">图 28。</span>',
        "横轴是被 patch 的 layer；纵轴是 clean n-vs-(n−1) margin 的 normalized recovery。三种柱分别写回完整 attention output、MLP output 或 post-layer residual；1=恢复 clean margin。误差条是跨样本 SEM，而不是 checkpoint 波动。",
    ),
    (
        "fig-29",
        '<span class="figure-tag">图 29。</span>',
        "横轴是 receiver residual 在哪一层后被替换/移动；纵轴是 expected-count transport slope，0=无 donor-count 搬运，1=一比一。Natural donor 写入完整 donor state；centroid delta 写入 <code>α(μ_m−μ_n)</code>，橙/绿分别 α=0.5/1。",
    ),
    (
        "fig-30",
        '<span class="figure-tag">图 30。</span>',
        "横轴是 donor final-marker residual 在 Layer 1–4 哪层后写入位置匹配的 receiver interior marker。左图纵轴是 close margin <code>z(&lt;/Think&gt;)−z(&lt;k+1&gt;)</code> 的增量；右图纵轴是 patched receiver 选择 close 的样本比例。",
    ),
    (
        "fig-31",
        '<span class="figure-tag">图 31。</span>',
        "左图横轴是 query-local 累计删除的 role-head 数，纵轴同时报告下游 L4 nearest-centroid state accuracy 与最终输出 accuracy。右图横轴是被测 top L3 head，纵轴是 progress-state transplant 后，对 donor occurrence 相对 receiver occurrence 的 attention-mass shift；正值表示 routing 被 donor progress 拉动。",
    ),
    (
        "fig-15",
        '<span class="figure-tag">图 15。</span>',
        "A：横轴 true union count，纵轴训练/均衡评估中的概率。B：横轴 count band，纵轴最终 AR exact accuracy。C：横轴是在控制 count 后的 needle-set 语料频率四分位，纵轴 accuracy。D：横轴是控制 count 后的最长非目标字符 run 四分位，纵轴 accuracy。C–D 是每个 representation 各 100 个 prompts 的观察均值，不是随机干预。",
    ),
    (
        "fig-14",
        '<span class="figure-tag">图 14。</span>',
        "这是横向条形图。横轴是同一 instrumented scope/block 的成功 timing events 累计 wall time，单位为秒；纵轴列出 pipeline block，条越长表示该 block 累计占时越多。它不是每步平均耗时，也不是 GPU kernel time；并行或 I/O 等待都计入 wall time。",
    ),
)


CAPTION_FIXES = {
    "fig-01": "21 个 checkpoint 的四项总览。四图横轴均为 training step；纵轴依次是 AR exact-count accuracy、最佳层 nearest-centroid accuracy、固定头 needle enrichment（× uniform）和答案 query 的 trace attention mass。竖虚线标记 step 1,500 的 loss-scope 切换。",
    "fig-06": "有序 trace-to-prompt 检索的训练动态。横轴为 training step，纵轴为 correct-k top-1-minus-chance；不同线型对应 query k=1–10。0 是各 count 自己的随机基线，正值表示第 k 个 occurrence 被相对优先检索。",
    "fig-07": "最终答案 count 的 nearest-centroid decoding heatmap。列为 training step，行为 model×depth；颜色为 held-out accuracy（0–1）。Thinking 很早在中间层形成可读 count state；nonthinking 后期才在最终层明显形成。",
    "fig-08": "Trace progress 的去混淆 ridge-decoding heatmap。列为 training step，行为 thinking site×depth；颜色为 held-out ridge R²。所有 progress k 都来自 total count=10 的固定套件，因此 progress 与 total count 不共线。",
    "fig-09": "跨位置线性迁移。横轴为 training step，纵轴为源 site 拟合、目标 site 评估的 held-out R²；线型区分 layer 与迁移方向。负 R² 表示迁移比目标均值基线更差，因此大幅负值不是百分比。",
    "fig-10": "Nonthinking final-answer centroid geometry 的涌现。横轴为 training step，纵轴为仅用 centroid PC1 预测 count 标签的 R²；不同线型为 layer。高值表示主要沿一个主轴排序，不等于完整状态只有一维。",
    "fig-19": "层间 centroid geometry 摘要。横轴均为 Layer output 1–4；纵轴依次为 PC1–PC3 累计解释率、effective dimension、相邻标签位移 cosine 与 chord/path straightness。相邻 cosine 中的“相邻”指 count/progress 标签，不是训练 checkpoint。",
    "fig-14": "运行时间分解。横轴为各 instrumented block 的累计 wall time（秒），纵轴为 scope:block 名称；条形按累计耗时排序。全部 523 个 timing events 成功完成。",
}


READABILITY_CSS = r"""
    /* report-readability-v1 */
    .report-reading-primer{border:1px solid #9bb7d3;border-left:5px solid #2f6f9f;background:#f4f8fc;padding:16px 18px;margin:18px 0 24px;border-radius:0 8px 8px 0}
    .report-reading-primer h3{margin-top:0}.report-reading-primer table{margin-bottom:0}
    .figure-reading-guide{border-left:4px solid #2f6f9f;background:#f4f8fc;padding:11px 14px;margin:22px 0 8px;color:#25364d;line-height:1.62}
    .figure-reading-guide strong{color:#174f78}.figure-reading-guide p{margin:4px 0 0}
"""


NAVIGATION = r"""
    <nav class="toc"><strong>按 v10 证据链组织的阅读顺序</strong><ol>
      <li><a href="#questions">研究问题与两条机制假设</a></li>
      <li><a href="#setup">实验设定、数据与序列</a></li>
      <li><a href="#definitions">术语、指标与公式</a></li>
      <li><a href="#learning">行为与学习动态</a></li>
      <li><a href="#attention-representation">描述性 attention representation</a></li>
      <li><a href="#residual-representation">描述性 hidden-state representation</a></li>
      <li><a href="#causal-heads">Head ablation：必要性</a></li>
      <li><a href="#causal-retrieval-conversion">Head/MLP patching：局部充分性</a></li>
      <li><a href="#causal-final-readout">最终 count 的来源与桥接</a></li>
      <li><a href="#causal-state">Hidden-state steering 与 early stop</a></li>
      <li><a href="#causal-bidirectional">Head ↔ state 双向因果关系</a></li>
      <li><a href="#data-noise">训练结构与 noise</a></li>
      <li><a href="#limits">综合结论与证据边界</a></li>
      <li><a href="#runtime-repro">附录：运行、产物与复现</a></li>
    </ol></nav>
"""


SECTION_ORDER = (
    "questions",
    "setup",
    "definitions",
    "learning",
    "attention-representation",
    "residual-representation",
    "causal-heads",
    "causal-retrieval-conversion",
    "causal-final-readout",
    "causal-state",
    "causal-bidirectional",
    "data-noise",
    "limits",
    "runtime-repro",
)


def _section(document: str, section_id: str) -> tuple[int, int, str]:
    match = re.search(
        rf'<section id="{re.escape(section_id)}">.*?</section>', document, flags=re.S
    )
    if not match:
        raise ValueError(f"missing report section: {section_id}")
    return match.start(), match.end(), match.group(0)


def _split_bidirectional_section(document: str) -> str:
    if '<section id="causal-bidirectional">' in document:
        return document
    start, end, block = _section(document, "causal-state")
    marker = '<h3>10.3 Head→state 与 state→head 的双向关系</h3>'
    if marker not in block:
        raise ValueError("cannot locate the bidirectional subsection")
    prefix, suffix = block.split(marker, 1)
    suffix = suffix.rsplit("</section>", 1)[0]
    prefix = prefix.replace(
        "<h2>10. Hidden-state causality：centroid steering、early stop 与 head↔state</h2>",
        "<h2>10. Hidden-state interventions：centroid steering 与 position-matched early stop</h2>",
    )
    state_block = prefix + "</section>"
    bidirectional = f"""
    <section id="causal-bidirectional">
      <p class="section-kicker">v10 §11 · Bidirectional causal link</p>
      <h2>11. Attention head 与 hidden state 的双向因果联系</h2>
      <h3>11.1 Head→state 与 state→head 的双向关系</h3>{suffix}
    </section>
    """
    return document[:start] + state_block + bidirectional + document[end:]


def _renumber_tail_sections(document: str) -> str:
    _, _, data = _section(document, "data-noise")
    data = data.replace("<h2>11.", "<h2>12.").replace("<h3>11.", "<h3>12.")
    start, end, _ = _section(document, "data-noise")
    document = document[:start] + data + document[end:]

    _, _, runtime = _section(document, "runtime-repro")
    runtime = runtime.replace(
        "<h2>12. 运行成本、产物与复现审计</h2>",
        "<h2>附录 A. 运行成本、产物与复现审计</h2>",
    )
    runtime = runtime.replace("<h3>12.", "<h3>A.")
    start, end, _ = _section(document, "runtime-repro")
    return document[:start] + runtime + document[end:]


def _reorder_sections(document: str) -> str:
    matches = [_section(document, section_id) for section_id in SECTION_ORDER]
    first = min(item[0] for item in matches)
    last = max(item[1] for item in matches)
    blocks = {section_id: _section(document, section_id)[2] for section_id in SECTION_ORDER}
    ordered = "\n\n".join(blocks[section_id] for section_id in SECTION_ORDER)
    return document[:first] + ordered + document[last:]


def _guide(spec_key: str, text: str) -> str:
    return (
        f'<div class="figure-reading-guide" data-figure-key="{spec_key}">'
        '<strong>读图前先定义</strong>'
        f'<p>{text}</p></div>\n'
    )


def polish_report_html(document: str) -> str:
    """Insert definitions before figures and restore the v10-style evidence order."""

    document = re.sub(
        r'\s*<aside class="report-reading-primer".*?</aside>\s*', "\n", document, flags=re.S
    )
    document = re.sub(
        r'\s*<div class="figure-reading-guide".*?</div>\s*', "\n", document, flags=re.S
    )
    document = _split_bidirectional_section(document)
    document = _renumber_tail_sections(document)

    spec_by_key = {key: (needle, guide) for key, needle, guide in FIGURE_SPECS}
    seen: set[str] = set()
    overview_wrapped: str | None = None

    def replace_figure(match: re.Match[str]) -> str:
        nonlocal overview_wrapped
        block = match.group(0)
        matches = [key for key, (needle, _) in spec_by_key.items() if needle in block]
        if len(matches) != 1:
            raise ValueError(f"figure matched {matches or 'no'} readability definitions")
        key = matches[0]
        if key in seen:
            raise ValueError(f"duplicate report figure key: {key}")
        seen.add(key)
        if key in CAPTION_FIXES:
            caption = re.search(r'<figcaption>(.*?)</figcaption>', block, flags=re.S)
            if not caption:
                raise ValueError(f"figure {key} has no caption")
            tag = re.search(r'<span class="figure-tag">.*?</span>', caption.group(1), flags=re.S)
            prefix = tag.group(0) if tag else ""
            replacement = f"<figcaption>{prefix}{CAPTION_FIXES[key]}</figcaption>"
            block = block[: caption.start()] + replacement + block[caption.end() :]
        wrapped = _guide(key, spec_by_key[key][1]) + block
        if key == "fig-01":
            overview_wrapped = wrapped
            return ""
        return wrapped

    document = re.sub(r'<figure\b.*?</figure>', replace_figure, document, flags=re.S)
    expected = set(spec_by_key)
    if seen != expected or overview_wrapped is None:
        raise ValueError(f"report figure coverage mismatch: missing={sorted(expected - seen)}")

    learning_start, learning_end, learning = _section(document, "learning")
    learning = re.sub(
        r'(<h2>4\. 行为表现与 learning dynamics</h2>)',
        rf'\1\n{overview_wrapped}',
        learning,
        count=1,
    )
    document = document[:learning_start] + learning + document[learning_end:]

    questions_start, questions_end, questions = _section(document, "questions")
    questions = re.sub(
        r'(<h2>1\. 研究问题与核心结论</h2>)',
        rf'\1\n{COMMON_PRIMER}',
        questions,
        count=1,
    )
    document = document[:questions_start] + questions + document[questions_end:]

    document = _reorder_sections(document)
    document = re.sub(r'<nav class="toc">.*?</nav>', NAVIGATION.strip(), document, count=1, flags=re.S)
    document = re.sub(
        r'<div class="callout success preserved">.*?</div>',
        '<div class="callout success preserved"><strong>单一报告入口。</strong> 本目录只保留这一份完整中文 HTML；正文按 v10 的“设定与定义 → 学习动态 → 描述性 representation → 必要性 → 充分性 → hidden-state 因果 → 综合边界”证据链组织。</div>',
        document,
        count=1,
        flags=re.S,
    )
    if "report-readability-v1" not in document:
        document = document.replace("</style>", READABILITY_CSS + "\n</style>", 1)
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", document)


def cleanup_legacy_reports(run_dir: Path, keep: Path) -> list[Path]:
    """Remove only the explicitly known legacy report aliases."""

    removed: list[Path] = []
    keep = keep.resolve()
    for name in LEGACY_REPORT_NAMES:
        candidate = (run_dir / name).resolve()
        if candidate == keep or not candidate.is_file():
            continue
        candidate.unlink()
        removed.append(candidate)
    return removed
