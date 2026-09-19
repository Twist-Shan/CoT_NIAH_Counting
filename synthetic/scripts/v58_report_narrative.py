"""Reader-facing v58 narrative. Numerical inputs come from the frozen-panel builder.

This module changes exposition and figures only; it never runs interventions or
selects experimental conditions. Full result tables remain available in details.
"""
from __future__ import annotations

import html
import json
import re

import matplotlib.pyplot as plt
import pandas as pd


ENDPOINTS = {
    'nonthinking_prompt_occurrence': 'Non-thinking · 正文 running index',
    'thinking_item_end': 'Thinking · trace running index',
    'nonthinking_answer_query': 'Non-thinking · final count',
    'thinking_answer_query': 'Thinking · final count',
}


def render_report(c):
    """Render the report from the builder's already-audited data and figures."""
    table, fig, conclusion = c['table_fn'], c['fig_fn'], c['conclusion_fn']

    def tbl(rows, headers):
        return table(pd.DataFrame(rows, columns=headers))

    def ledger(title, frame):
        return '<details class="ledger"><summary>'+html.escape(title)+'</summary><p class="caption">复核表保留原始列名；比例为0–1，pp为百分点，NaN/—表示不适用或未定义。重复对照编号不代表独立训练seed。</p>'+table(frame)+'</details>'

    def protocol(purpose, setup, example):
        return '<div class="protocol"><p><b>实验目的。</b>'+purpose+'</p><p><b>实验设置。</b>'+setup+'</p><p><b>说明性示例。</b>'+example+'</p></div>'

    sections = [
        ('overview', '阅读导览与结论摘要'),
        ('setup', '1. 任务、模型与训练方法'),
        ('measurement', '2. 测量位置、指标与证据标准'),
        ('behavior', '3. 行为：Thinking 是否更准确？'),
        ('geometry', '4. Geometry：进度与最终答案如何表征？'),
        ('retrieval', '5. Retrieval：检索方式与 head-bank 分工'),
        ('transport', '6. Formation 与 transport：信息能否被恢复？'),
        ('progress', '7. Progress update：状态能否改变后续检索？'),
        ('readout', '8. Final readout：答案从哪里读出？'),
        ('paths', '9. Path mediation：哪些通路尚未闭合？'),
        ('dynamics', '10. Training dynamics：角色如何形成？'),
        ('alignment', '11. 与大模型对齐的结果与限制'),
        ('reproduce', '附录：完整规模、原始表与复现入口'),
    ]
    toc = '<nav aria-label="报告目录"><h2>目录</h2><ol>'+''.join('<li><a href="#'+a+'">'+t+'</a></li>' for a,t in sections)+'</ol></nav>'
    def heading(key):
        return '<h2 id="'+key+'">'+dict(sections)[key]+'</h2>'

    ablation = c['ablation']
    clean = ablation.loc[ablation.arm.eq('clean')]
    behavior_rows = []
    for mode in ['nonthinking','thinking']:
        f=clean.loc[clean['mode'].eq(mode)]
        behavior_rows.append([mode, len(f), f'{f.ar_accuracy.mean():.0%}',
            '—' if mode=='nonthinking' else f'{f.trace_exact.mean():.0%}',
            '—' if mode=='nonthinking' else f'{f.trace_marker_count_accuracy.mean():.0%}'])
    bf, ba = plt.subplots(figsize=(9,3.8),layout='constrained')
    for mode,color in [('nonthinking','#ce7241'),('thinking','#267cb0')]:
        f=clean.loc[clean['mode'].eq(mode)].groupby('count').ar_accuracy.mean()
        ba.plot(f.index,100*f,'o-',label=mode,color=color)
    ba.set(xlabel='True count N',ylabel='Free-running answer accuracy (%)',xticks=range(1,11),ylim=(-3,103))
    ba.legend(); ba.grid(alpha=.2)
    behavior_image=fig(bf,'behavior_by_count_readable.png','图｜统一输入上的自由生成答案准确率。横轴是真实count N，纵轴为答案准确率（%）；蓝色Thinking、橙色Non-thinking。每个数字每mode为同样10题，每点一题对应10个百分点；单训练seed，无误差条。未输出可解析答案计错，未按clean正确与否筛题。')

    geometry_rows=[]
    for r in c['selected_geometry'].itertuples():
        geometry_rows.append([ENDPOINTS[r.endpoint],f'L{r.selected_layer}',f'{r.discovery_value:.2%}',f'{r.confirmation_value:.2%}',f'L{r.common_decoder_selected_layer}'])
    selected_all=c['selections']
    metrics=[]
    for r in selected_all.itertuples():
        metrics.append([ENDPOINTS[r.endpoint],r.selector,f'L{r.selected_layer}',r.confirmation_value])
    # The comparison figure uses all depths, avoiding selection-dependent visual contrasts.
    gf, ga = plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    for ep,g in c['clean_geometry'].loc[c['clean_geometry'].layer.gt(0)].groupby('endpoint'):
        is_t=ep.startswith('thinking'); ax=ga[1 if 'answer_query' in ep else 0]
        ax.plot(g.layer,100*g.confirmation_ncc_balanced_accuracy,'o-',color='#267cb0' if is_t else '#ce7241',label='Thinking' if is_t else 'Non-thinking')
    for ax,title in zip(ga,['Running index: clean NCC','Final count: clean NCC']):
        ax.set(title=title,xlabel='Post-block layer',ylabel='Balanced NCC accuracy (%)',xticks=[1,2,3,4],ylim=(0,103))
        ax.axhline(10,color='gray',ls=':',label='10-class chance');ax.legend(fontsize=8);ax.grid(alpha=.2)
    geometry_image=fig(gf,'geometry_ncc_layers_readable.png','图｜四个endpoint的全层clean NCC。横轴为post-block层L1–L4，纵轴为十类等权NCC准确率；左为running index，右为final count，颜色区分mode，灰线为10%参考。每层分别仅用200题discovery拟合变换及类中心，评相同100题confirmation；running含550条状态、final含100条状态。全层曲线是描述性展示，正文选层仅依据discovery交叉验证。')

    heads=[]
    for mode,role in [('nonthinking','broad'),('thinking','targeted'),('thinking','broad')]:
        sites=json.loads((c['ALIGN']/mode/'frozen_sites.json').read_text())
        heads.append([mode,role,', '.join(f'L{l}H{h}' for l,h,_ in sites['ranking'][role][:4])])
    factorial=c['factorial'].groupby('arm')[['ar_accuracy','trace_exact','trace_marker_count_accuracy']].mean().reset_index()
    role_names={'clean':'无干预','targeted_top2':'Targeted Top-2 · trace query','broad_top2':'Broad Top-2 · answer query','targeted_plus_broad':'Targeted + broad','successor_top1':'Successor Top-1 · marker','targeted_plus_successor':'Targeted + successor'}
    role_rows=[[role_names.get(r.arm,r.arm),f'{r.ar_accuracy:.0%}',f'{r.trace_exact:.0%}',f'{r.trace_marker_count_accuracy:.0%}'] for r in factorial.itertuples()]

    body='''<header><p class="eyebrow">CONTROLLED COUNTING · SYNTHETIC V58</p>
<h1>无显式编号的 Thinking 如何完成计数？</h1>
<p class="subtitle">任务与训练 · Geometry Comparison · Causal Experiments · Training Dynamics</p>
<p class="caption">阅读版更新：2026-09-06。主结果取自同一套已完成的200题discovery／100题confirmation面板，模型为v58最终step 10,000。此轮只重写报告和整理图表，未重训、未增加干预样本。</p></header>'''+toc+heading('overview')
    body+='''<p>本研究用两份独立训练的小型Transformer比较直接回答与无编号trace计数。两者接收相同正文与目标字符集合；Thinking先逐项输出检索到的字符，再给出答案。我们关注三个问题：Thinking的行为优势是否存在？逐项检索和计数进度是否对应不同内部作用？这些作用何时在训练中形成？</p>
<div class="summary"><b>主结论。</b>在当前100题上，Thinking自由生成答案准确率为95%，Non-thinking为21%。Thinking的末层targeted heads参与marker内容检索；successor head干预更明显地影响trace长度与答案。已有自身trace后，答案依赖trace信息；进度状态patch支持短程donor-directed continuation。当前还不能把这些局部证据合并为完整、唯一的内部计数通路。</div>'''
    body+=tbl([
        ['行为','Thinking 95%，Non-thinking 21%','同一100题，单训练seed；每count只有10题'],
        ['表征','Thinking running NCC 31.37%，final NCC 99%','discovery分别选层；可读出不等于实际使用，固定trace位置与count混杂'],
        ['检索内容','Thinking Top-4消融：trace exact 87%→51%，答案95%→86%','内容损伤大于count损伤；未闭合检索内容→count的串行中介'],
        ['进度与读出','双token item patch有短程continuation效应；trace删除损伤答案','长程循环、位置不变的纯数值状态均未建立'],
        ['Non-thinking通路','完成broad、source恢复与联合干预','低行为基线及对照输出故障限制特异性判断'],
    ],['证据类型','当前结果','解释边界'])
    body+='''<p><b>阅读方式。</b>先读第1–2节了解任务和指标，再依次看“表征→检索→进度→答案”的实验链。每个实验先交代目的、操作和示例，再给结果与结论。大网格与原始列名放在可展开的复核表中；零效应和不适用实验仍保留。本文中的Thinking是受监督的separator trace，不能直接等同于大模型自然产生的Native-thinking。</p>'''

    body+=heading('setup')+'''<h3>1.1 输入、输出及一个完整例子</h3>
<p>目标集合包含3个字符。给定256字符正文，统计集合内字符出现的总次数N（重复出现分别计数），N限制为1–10。Thinking按正文出现顺序复述目标字符；trace里没有“1、2、3”等显式进度编号。</p>
<div class="example"><b>说明性短例子（真实正文长度为256）。</b><br>目标集合：{a,b,c}；正文：<code>x a y b z a</code>；因此N=3。<br>
Non-thinking输出：<code>&lt;Ans&gt; &lt;3&gt; &lt;EOS&gt;</code><br>
Thinking输出：<code>&lt;Think&gt; &lt;Sep&gt; a &lt;Sep&gt; b &lt;Sep&gt; a &lt;/Think&gt; &lt;Ans&gt; &lt;3&gt; &lt;EOS&gt;</code><br>
这里&lt;3&gt;代表一个atomic答案token，显示文本用于说明；实际marker字符由字符词表编码。两个a是两个不同occurrences。</div>
<p>序列模板为<code>&lt;BOS&gt; query[5] data[256] suffix</code>。两个mode的正文前缀内容相同，但模型参数独立，因此prompt hidden states不自动相同。Thinking每个item恰好为两个token：separator与marker。固定N决定trace长度，也决定answer query的绝对位置，这是后续几何与readout解释的重要限制。</p>
<h3>1.2 数据与采样</h3>
<p>字符来源为Shakespeare语料，按连续语料区段作80%／10%／10%的train／validation／test划分。目标字符集合池大小100，每集合含3字符。训练使用max-entropy set×count联合采样，within-cell保留全部合法起点；目标是在可行集合与count之间减小采样偏差。集合顺序打乱，训练时对抽到的窗口字符重新随机排列，保留该窗口的字符计数。机制面板来自保存的test-region输入；实验均复用其精确token序列，不重新排列或重新采样。</p>'''
    body+=f'<p>已记录训练count占比范围为{c["sampling"].fraction.min():.3%}–{c["sampling"].fraction.max():.3%}。这是训练采样审计，不能代替确认集的逐count正确率。各计数样本可以包含重复marker，donor/receiver未来的第一个字符也可能相同；progress实验因此单列可识别前缀。</p>'
    body+='<h3>1.3 模型与优化器</h3>'+tbl([
        ['参数与结构','每mode 12,658,176参数；4层、每层8 heads；hidden512，MLP2048；最大384 positions'],
        ['位置编码','RoPE，base=10,000；当前v58使用RoPE'],
        ['词表与readout','count1–10各为一个atomic token；这些count输出行独立，其余输入embedding与输出权重共享（tied）'],
        ['训练预算','两份独立模型；每mode 10,000 optimizer steps × batch128 = 1,280,000次样本呈现；训练seed1234'],
        ['优化器','AdamW；β=(0.9,0.999)，weight decay0.01，梯度范数clip1；BF16'],
        ['学习率','前500步线性warmup到3×10⁻⁴，此后cosine衰减，到10,000步为0'],
        ['未使用的方法','无联合mode训练、无测试时训练（TTT）、无scheduled sampling、无额外对比损失'],
        ['保存与分析','每100步FP16科学snapshot，每500步optimizer恢复状态；主报告使用step10,000'],
    ],['项目','实际设置'])
    body+='''<p><b>Tied的含义。</b>某token的输入向量同时用作输出分类器中该token的权重行；count1–10例外，使用独立输出行。两mode采用相同参数化，训练期间正常优化全部可训练参数，并非只更新十个数字行。</p>
<h3>1.4 训练loss：两个阶段需要分开解释</h3>
<p>Step1–1500：对全部非padding位置计算teacher-forced next-token交叉熵。Step1501–10,000：只监督任务输出，从NT的&lt;Ans&gt;或Thinking的&lt;Think&gt;到&lt;EOS&gt;。先在每题的各分区内对token loss取均值，再在batch的有效题目间平均，得到L<sub>count</sub>、L<sub>marker</sub>、L<sub>structure</sub>。</p>
<div class="equation">L<sub>Thinking</sub> = 8 L<sub>count</sub> + 8 L<sub>marker</sub> + 16 L<sub>structure</sub><br>
L<sub>Non-thinking</sub> = 8 L<sub>count</sub> + 16 L<sub>structure</sub></div>
<p>Count分区为最终答案token；marker分区为trace中的目标字符身份；structure包含边界、separator、&lt;Ans&gt;和&lt;EOS&gt;。代码没有再除以系数总和。25%／25%／50%与33.3%／66.7%仅为系数比例，实际loss值和梯度贡献还取决于各分区误差。<code>final_count_loss_weight=1</code>与task-output分区系数8属于不同配置项，不能混同。</p>
<p><b>说明性例子。</b>Thinking输出10个marker时，先平均这10个marker的CE再乘8；不会因为10个marker而把整个marker分区乘10。分区归一化减小trace长短对分区总权重的直接影响；两mode的监督内容与总loss尺度仍有差别。</p>'''+c['loss_image']
    body+=conclusion('当前比较固定了架构、输入、训练步数和trace格式，保留了两mode不同的监督任务。行为差异成立于这一训练设置，尚未证明在所有公平预算定义或训练seed下都稳定。')

    body+=heading('measurement')+'''<h3>2.1 五个位置：避免把不同query混为一谈</h3>'''
    body+=tbl([
        ['正文needle位置 pₖ','正文第k个目标字符','NT running状态；正文source干预'],
        ['Trace retrieval query qₖ','第k个marker之前的&lt;Sep&gt;','预测第k个marker；targeted attention与query-local消融'],
        ['Trace item-end rₖ','第k个marker自身位置','Thinking running状态；progress patch；successor attention'],
        ['Terminal item','最后一个&lt;Sep&gt; marker','bridge／relay恢复位置；与answer query不同'],
        ['Answer query q_ans','&lt;Ans&gt;的输入位置','该位置的logits预测count；answer-side broad及答案状态patch'],
    ],['名称','具体位置','测量或干预用途'])
    body+='''<p>Head标号层为L1–L4，head为H0–H7；L0仅表示embedding输出。Residual通常指block之后的hidden state。Pre-O head output是合并heads前、输出投影W<sub>O</sub>之前的单head输出；post-O write是该head经W<sub>O</sub>写入residual的向量。</p>
<h3>2.2 Attention score与归一化</h3>
<p>设A<sub>h</sub>(q,p)为某head从query q到key p的attention权重。模型softmax已使每个q对全部可见keys的权重和为1。</p>
<div class="equation">Targeted：T<sub>h</sub> = mean<sub>题目i</sub> [ (1/Nᵢ) Σₖ A<sub>h</sub>(qᵢ,ₖ, pᵢ,ₖ) ]</div>
<p>k-to-k指第k次trace检索对应正文第k个occurrence。Absolute k-to-k保留这个原始attention mass，不再跨head归一化；接近1表示正确needle获得几乎全部attention。先题内平均、再题间等权，避免N大的题自动获得更大权重。</p>
<div class="equation">Broad：M = Σₖ A<sub>h</sub>(q_ans,pₖ)，p̃ₖ=A<sub>h</sub>(q_ans,pₖ)/M<br>
C = exp(−Σₖ p̃ₖ log p̃ₖ)/N，B<sub>h</sub> = mean<sub>题目</sub>(M·C)</div>
<p>M为正文needles总mass；C为有效覆盖度，均匀覆盖N个needles时为1，只集中于一个needle时为1/N。M=0时B定义为0。高coverage与很小mass可以同时存在，所以两者应一起读。本文Thinking broad始终指answer位置对正文needles的覆盖；尚未补算同一trace query上的broad熵／覆盖度。单token needle使span与endpoint相同。当前Synthetic选头依据B排序，未把大模型报告的全部span-enrichment条件纳入筛选，因此协议并非完全等价。</p>
<div class="equation">Successor：S<sub>h</sub> = mean<sub>题目</sub> meanₖ A<sub>h</sub>(rₖ,qₖ)<br>
相对份额：R<sub>h</sub> = score<sub>h</sub> / Σ<sub>32 heads</sub> score</div>
<p>Successor描述marker位置对前一个separator的attention，不自动表示执行数值加一。相对份额图在每个checkpoint内对32 heads归一化；若一个head的score=0.06，其余heads合计0.08，它的份额为42.9%。份额高仅说明相对集中，原始score仍为0.06。</p>
<h3>2.3 统一样本与两种前向条件</h3>
<p>Discovery为200题、每count20题；confirmation为100题、每count10题。相同输入在两个mode及所有主实验条件间复用；每个实验的派生状态数、pair数单独报告。选头、选层、PCA、类中心和干预方向仅由discovery确定。输入按hash选择并排除旧选择集，未按当前准确率或干预效果筛选。该来源池的行为此前已被评估，因此本面板是固定的机制扩展，不能称为全新未见的行为测试。</p>'''
    body+=tbl([
        ['Gold-prefix / teacher-forced','给模型正确trace或正确prefix，再读状态或局部logits','Geometry、attention动态、局部value patch等；不等于模型自己生成了正确trace'],
        ['Free-running','模型逐token生成自己的输出','答案／trace accuracy、动态head消融与progress continuation'],
        ['Own-generated-prefix readout','先自由生成到&lt;Ans&gt;，再干预来源并预测count','保留真实生成prefix，但不重跑被干预前的整段trace'],
    ],['条件','含义','对应证据'])
    body+='''<p>Confirmation分为10个balanced blocks，每个block含各count一题。它们是统计分组，不是10个独立训练seed。配对效应先在输入／pair内相减，再按block聚合；10,000次bootstrap给出点态95%区间，未作多重比较校正。Progress在count10子集按prompt聚类，多个pairs不会增加独立prompt数。</p>
<h3>2.4 证据标准</h3>'''+tbl([
        ['Geometry／NCC','标签信息可读出或类结构存在','模型实际使用该方向'],
        ['Attention','特定query对哪些source关注','读取的信息如何写入，或对答案的必要性'],
        ['Ablation','在给定scope内的必要性证据','唯一通路、纯count效应；需匹配对照与格式审计'],
        ['Patch／restoration','在指定损坏baseline上的局部因果充分性','完整rollout恢复或串行中介'],
        ['Free continuation','状态干预可改变后续生成','纯±1数值运算或稳定长程循环'],
    ],['证据','直接回答','不能单独推出'])+conclusion('本文所有数值都绑定token位置、prefix条件、选择规则和统计单位；跨层、跨mode或跨实验的比较不省略这些条件。')

    body+=heading('behavior')+protocol('确认当前设置的行为优势，并检查是否只有少数count学会。','同一100题，每count10题；提供共同正文及规定的mode起始token（NT的&lt;Ans&gt;、Thinking的&lt;Think&gt;），其后独立greedy生成直到EOS；NT最多4个新token，Thinking最多26个新token。未给出可解析答案计错，未提供gold marker或gold答案。','真实N=3，生成三个marker但字符顺序错误、最终答3：答案正确，trace exact错误，marker数量正确。')
    body+=tbl(behavior_rows,['Mode','输入数','答案准确率','Trace exact','Trace marker数量正确率'])+behavior_image
    body+='''<p><b>结果与分析。</b>Thinking在N=1–5为10/10正确，N=6–10为9/10；Non-thinking在N=5–7为0/10，整体21/100。当前样本未出现Thinking某个数字系统性失败，但每count只有10题，尚不能精确证明各数字的总体正确率相等。Thinking trace exact为87%，低于答案95%，说明内容完整性与最终数量需要分开分析。</p>'''+conclusion('当前固定面板存在较大Thinking答案优势；这一行为差异本身不能决定模型采用哪条内部计数通路。')+ledger('复核：每count原始行为结果',c['bycount'])

    body+=heading('geometry')+protocol('检查running index与final count是否形成可读出的类结构，并比较两mode。','NT取正文needle末端，Thinking取trace marker末端作为running状态；两mode取&lt;Ans&gt;作为final状态。Running为1100个discovery／550个confirmation状态，final为200／100；均来自相同200／100题，Thinking使用gold trace。','正文共7个needle时，第3个trace marker的running标签为k=3，answer query的final标签为N=7；二者回答不同问题。')
    body+='''<h3>4.1 NCC如何计算，层如何选？</h3>
<p>在discovery上拟合StandardScaler，再拟合最多16维whitened PCA。对每个标签c计算类中心μ<sub>c</sub>；confirmation点z的预测为argmin<sub>c</sub>‖z−μ<sub>c</sub>‖²。NCC balanced accuracy为标签1–10各自正确率的等权平均，机会参考为10%。Running中低k出现更多；按标签等权避免这种数量差异直接主导准确率。</p>
<p>主表用5-fold grouped discovery CV的NCC指标选层；另一个共同decoder选层按NCC与logistic discovery分数均值选层，用于post-Top-K与geometry dynamics。两套规则预先区分，不能把不同层的结果当作同一endpoint的数值矛盾。完整全层结果保留供复核。</p>'''+tbl(geometry_rows,['Endpoint','NCC选层','Discovery CV NCC','Confirmation NCC','共同decoder选层'])+geometry_image
    body+='''<p><b>结果。</b>Thinking的running NCC为31.37%（L4），高于NT的17.60%（L2）；final NCC为99%（L2），NT为24%（L4）。这里每个endpoint都按同一NCC规则独立选层；全层曲线展示层依赖。Final的高可读性仍可能利用固定trace长度／answer位置。</p>
<h3>4.2 2D／3D状态几何：可选层的四端点对比</h3>
<p>下面只作可视化：discovery拟合非whitened PCA3，再投影confirmation。左右为Non-thinking／Thinking，上下为running／final；2D与3D分别选L1–L4，3D可拖拽旋转。每个endpoint和层单独拟合坐标系，不能比较跨panel绝对坐标距离。</p>'''+c['widget']+'''<p class="caption">图｜横纵轴为PC1／PC2，3D另有PC3，轴注明discovery解释方差。颜色为k或N；散点为confirmation状态，红线连接类均值辅助观察。Running每panel550点、final100点；这些点来自100个输入，不能当作550个独立样本。</p>'''
    body+=conclusion('Thinking在本设置中有更高的标签可读性；二维聚团与高NCC尚不能证明位置无关的representation compression，更不能证明模型使用一个纯数值counter。')
    body+='''<h3>4.3 其他几何指标与post-Top-K NCC</h3>
<p>几何指标从不同角度检验类结构。SNR为10log₁₀(trΣ<sub>B</sub>/trΣ<sub>W</sub>)，区分类间差异与类内变化；Fisher trace使用discovery类内协方差加ridge的逆矩阵度量confirmation类间散布；Mahalanobis silhouette在discovery类内白化空间内衡量同类与异类距离；ordinal RSA为类中心距离与|k−k′|的Spearman相关。Logistic balanced accuracy是另一种可读性指标。各指标单独discovery选层，不能只挑最高confirmation值。</p>'''+ledger('复核：全部几何指标的独立discovery选层结果',pd.DataFrame(metrics,columns=['Endpoint','Metric','Discovery-selected layer','Confirmation value']))
    body+=protocol('检验检索头关闭后，clean可读出的count信息是否仍保留。','冻结clean discovery的变换与类中心，在相同confirmation状态上评query-local Top-K消融；损坏后不重新拟合probe。正文使用共同decoder选层，all-layer结果保留。','关闭末层L4的trace-query head后去读L2状态，计算图上L2不会被回改；NCC不变在这种情况下没有提供额外机制证据。')
    body+=ledger('复核：冻结clean probe的post-Top-K NCC',pd.concat(c['probe_selected']))+ledger('复核：全层clean几何指标及状态支持数',c['clean_geometry'])+conclusion('Post-Top-K NCC实验已完成。须先确认干预在读出状态的上游；时间顺序导致的零变化不能解释为表征不参与任务。')

    body+=heading('retrieval')+protocol('比较answer-side broad与trace-side targeted，并检验候选head bank对内容和count的分别贡献。','200题discovery在最终checkpoint排序冻结Top-K。NT仅在answer query清零所选pre-O head输出；Thinking在每个trace separator query清零。之后自由生成。K=1、2各3个同层等头数不重叠对照，K=4仅1个可行补集。','生成第3个marker时移除targeted heads，可能使该marker身份错误，但模型仍输出3个items并答3。')
    body+=tbl(heads,['Mode','选头指标','Discovery Top-4（按排名）'])
    body+='''<p>位置限定只覆盖trace里的separator，不包括prompt query分隔符。旧探索代码的范围问题已经修正，旧面板数值留在历史报告。本报告没有按消融效果重新选头或替换失败的对照。</p>'''+c['image1']
    body+='''<p><b>结果。</b>Thinking Top-4消融使trace exact由87%降至51%，答案由95%降至86%；同层补集对应为74%与91%。NT Top-4消融答案21%→7%，100题仍全部给出数值；其唯一补集对照答案为0%，且100题全部未给出数值答案，属于明显的通用输出故障。NT的这一对照不能提供干净的count-specific特异性比较。</p>'''+ledger('复核：Top-K效应、输出率与配对95%区间',c['abl_effects'])+conclusion('Thinking targeted bank对trace内容有因果影响，对最终count的损伤相对较小。NT broad-bank特异性必要性仍未建立，必须保留低基线和对照格式失败的限制。')
    body+='<h3>5.2 Targeted、successor与Thinking answer-side broad的分工</h3>'+tbl(role_rows,['同一100题的干预条件','答案准确率','Trace exact','Marker数量正确率'])
    body+='''<p>Successor按marker→preceding-separator attention在discovery选择，Top-1为L2H3。关闭该head后答案95%→45%；三个同层对照答案为79%、94%、83%。Targeted Top-2的trace exact降至62%，答案仍91%；Thinking answer-query broad Top-2消融后答案仍95%，与targeted联合消融后91%。这些是特定head、位置和预算下的效应，不能外推为全部broad heads无效。</p>'''+conclusion('实验支持内容检索与trace数量／停止相关作用的分工；targeted消融对count较弱与这种分工一致。Successor究竟实现了何种状态变换，原因未查明；不能直接称为“加一头”。')

    body+=heading('transport')+'<h3>6.1 正文source formation与状态恢复</h3>'+protocol('检验正文source状态是否向答案传递了可恢复的count信息。','两mode每条件100题；把全部正文needles替换为等长度普通字符，在同一损坏输入上共同恢复这些位置的clean embedding或L1–L4状态；普通位置恢复作为等token预算对照。Thinking这里固定gold trace。','一题有三个目标字符a、b、a，把三个位置都换成普通字符；再共同恢复这三个位置的clean L2状态，观察原题答案3能否恢复。')
    body+='''<p>状态恢复还记录到discovery正确running类中心的Euclidean距离，除以该类discovery RMS半径，先题内平均；这是raw-space距离，与PCA-NCC不同。单token needle使whole-span与endpoint恢复完全重合；embedding恢复相当于恢复输入身份，是上界对照。</p>
<p><b>结果。</b>NT clean答案21%，needle损坏后13%；embedding恢复为21%，L1–L4恢复仅13%–14%。Thinking在固定gold trace下对正文source损坏不敏感。</p>'''+ledger('复核：source恢复的准确率、margin与类中心距离',c['source'])+conclusion('当前未复现强的post-block source-state恢复。Thinking在gold trace条件下可以利用已经给定的trace读答案，不能据此推断生成trace时不需要正文。')
    body+='<h3>6.2 Targeted value transport：是否传输正确marker内容？</h3>'+protocol('把attention定位与实际传输marker身份连接起来。','每题取固定k=max(1,floor(N/2))，共100题；将第k个needle换为同目标集合另一字符。在同一损坏baseline上恢复选中head在该source的clean value，或恢复query的clean residual；包含同层对照。','正确字符a被换成b；比较patch前后模型在当前query预测a相对b的logit差。')
    body+='''<p>Marker margin=logit(正确marker)−logit(替换marker)，restoration=patched margin−damaged margin。主结果用原始差值；clean−damaged很小时，归一化恢复比例不稳定，不能用来夸大效果。</p>
<p><b>结果。</b>Thinking Top-2 value patch将margin从−3.783恢复至1.190，三个同层对照平均约−2.769。选中heads传输对应marker内容的局部证据成立。</p>'''+ledger('复核：value与residual恢复全表',c['transport_summary'])+conclusion('对应source的value恢复支持targeted heads参与内容传输。此处是一次检索的局部logit实验，尚不能替代自由生成整条trace或最终count的充分性。')

    body+=heading('progress')+'<h3>7.1 下一次检索依赖正文，还是已有trace？</h3>'+protocol('区分原始内容来源与近期进度状态的作用。','相同100题，每题固定k=max(1,floor(N/2))；预测第k个marker前累计清零正文needles、全部前序items、最近item、更早items或前半items；普通token数量匹配。空历史输入仍保留，并报告nonempty数。','N=6时取k=3：正文保留六个目标字符，分别删除前两个trace items或仅最近item2，观察下一次是否仍正确生成marker3。')
    body+='''<p><b>结果。</b>下一marker正确率clean为96%；删除正文needles后21%，对应普通token对照95%；删除最近item或全部历史后79%，对照96%。</p>'''+ledger('复核：source-next的实际支持数与marker指标',c['source_next_summary'])+conclusion('下一次检索同时依赖正文与近期trace。该实验没有隔离纯数值进度方向，也未定位历史处理必须发生在哪一层。')
    body+='<h3>7.2 Native-style progress patch与自由continuation</h3>'+protocol('检验item状态改变后，模型是否继续生成donor对应的后续marker序列。','统一面板的N=10子集：20题discovery、10题confirmation。保持needle顺序，通过调整donor普通filler使patch绝对位置与receiver对齐；receiver仍256字符。分别测试marker末端单token与完整separator+marker双tokenitem。','Receiver未来[a,b,c]，donor未来[a,c,b]；第一个a无法区分进度，生成[a,c]才支持最短可区分donor前缀。')
    body+='''<p><b>选层。</b>Discovery donor k=6、receiver k=5/7，共40pairs/scope。根据不同successor身份的donor／receiver log-odds变化选层：forward与backward中位数均为正，选prompt平均效应中位数达到有效峰值95%的最早层，预先排除末层。两个scope在当前共同面板都选L1。</p>
<p><b>Confirmation。</b>Donor k=4/6/8、receiver k±1，共60pairs/scope。运行clean、self、完整donor、rank-3投影donor、投影等范数正交对照与3个完整等范数正交对照；所有方向仅由discovery拟合。Patch后greedy自由生成，不强制下一marker。首marker、最短可识别前缀及2／3／4-marker前缀分别使用自己的eligible分母。</p>'''+c['progress_image']
    body+='''<p><b>结果。</b>首marker可识别为26pairs／8prompts；完整双tokenitem patch相对等范数对照提高16.04个百分点，95%区间[4.17,28.96]。两marker前缀提高26.04个百分点，[4.17,52.08]；三步区间跨零，四步区间接触零。不能把60pairs当成60个独立输入，亦不能忽略重复字符造成的识别限制。</p>'''+ledger('复核：各scope、各continuation终点与实际分母',c['progress'])+ledger('复核：完整discovery选层记录',c['progress_selection'])+conclusion('当前支持contextual item状态改变短程后续生成，尚未确认稳定长程循环或纯±1计数更新。旧cohort的L2选层保留在历史记录，当前统一面板使用L1，不混合两批结果。')

    body+=heading('readout')+'<h3>8.1 生成自身trace后，答案依赖哪类source？</h3>'+protocol('检验最终答案主要依赖正文needles，还是已生成trace。','每个模型先自由生成到&lt;Ans&gt;，本批全部到达；再在embedding及全部post-block层累计清零指定source，保持序列长度与answer query位置不变，随后预测答案token。包括正文records、trace和等token预算普通位置。','Thinking已经生成三个items；保留&lt;Ans&gt;位置不变，清零trace位置状态，观察答案3是否仍被预测。')
    body+='''<p><b>结果。</b>Thinking clean答案95%，正文needles清零后95%，trace清零后59%，trace预算普通token对照95%。NT无trace，trace条件为identity对照。当前数据说明已有trace的答案读出依赖；没有重新执行被干预前的生成过程。</p>'''+ledger('复核：own-generated-prefix answer-source干预',c['answer_source'])+conclusion('已有自身trace后，Thinking的答案依赖trace信息。Trace-to-answer readout足以作为当前主张，不需要声称存在一个必要的universal final broad aggregator。')
    body+='<h3>8.2 Answer residual是否携带可执行的count？</h3>'+protocol('区分信息可读出与状态能够直接改变答案。','两mode均使用180个相邻count有向pairs／层，receiver与donor处于各自&lt;Ans&gt;语义位置；比较self、相邻count donor及same-count不同context donor。另列与Native式1↔2、5↔6对应的40-pair子集，不按clean正确率筛选。','Receiver真实N=5，donor N=6，把donor answer residual写入receiver，输出6记为donor adoption。')
    body+='''<p><b>结果。</b>Thinking L3／L4全状态patch后180/180 pairs输出donor count；NT对应比例较低。它证明该位置的完整状态可执行地控制输出，但状态同时包含内容、位置与trace长度；全状态patch不能隔离纯计数操作。</p>'''+ledger('复核：全部180-pair、全层answer-state干预',c['answer_summary'])+ledger('复核：Native式40-pair子集',c['native_summary'])
    body+='<h3>8.3 Count-aligned方向的必要性与充分性</h3>'+protocol('进一步检验类中心定义的count方向是否具有因果作用。','Discovery类中心提取rank-3正交基U。必要性删除h的计数分量：h′=h−UUᵀ(h−μ)；充分性注入相邻类中心差的rank-3投影。对照方向与count子空间正交，并匹配实际删除／注入范数；充分性有3个正交对照。','从count5类中心指向count6类中心的投影方向加入receiver，观察输出6的比例，与同样向量长度的正交扰动比较。')
    body+='''<p>Directed expected shift为预测count分布期望沿donor方向的变化；donor adoption为输出等于donor count的比例。Thinking L4方向注入的donor adoption为45.6%，三个正交对照均为0%。方向删除结果见上方answer-state复核表，方向注入全层结果如下。</p>'''+ledger('复核：rank-3方向注入与等范数对照',c['vector_summary'])+conclusion('Answer-state中的count相关信息具有可执行性，并有方向干预支持；位置／长度混杂仍未排除，不能据此声称纯整数寄存器或纯±1操作。')

    body+=heading('paths')+'''<p>本节检查前面的局部发现能否连成同一因果路径。它保留弱恢复、零效应和架构不适用结果，避免把若干独立正结果直接写成完整机制。</p>
<h3>9.1 Non-thinking：source → retrieval write → late answer</h3>'''+protocol('在同一个损坏baseline中检验三个阶段是否构成串行中介。','2个source深度×是否恢复source×3种retrieval处理×3种late处理，共36组合，每组合100题。Retrieval write取选中answer-side broad bank经W_O的输出；count方向由discovery类中心rank3定义，类内正交控制匹配实际扰动范数。','先损坏正文并恢复source；若再删除retrieval write的count分量使恢复消失，且晚层状态同步变化，才进一步支持中介链。')
    body+='''<p>当前retrieval write在L1；source-L1恢复发生在本层write之后。只有embedding→L1 write→L4满足这组操作的所列顺序，不能把source-L1条件解释为L1 write的上游中介。若晚层干预不能改变更早的读数，这属于计算图约束。</p>'''+ledger('复核：NT自然retrieval write的方向删除',c['trials'].loc[c['trials'].family.eq('retrieval_natural')].groupby(['mode','arm'])[['accuracy','margin','expected_abs_error']].mean().reset_index())+ledger('复核：同一forward的36组合',c['serial'])+conclusion('联合干预已完成；结合弱post-block恢复、低行为基线和对照问题，当前未建立特异的完整NT串行中介。')
    body+='<h3>9.2 Thinking：terminal bridge与下游relay</h3>'+protocol('检验最后一个trace item是否通过特定中间状态影响答案。','Bridge：100题，把所有trace items替换成等长度普通字符，从L2起累计恢复最后item／marker／separator的clean状态，比较相同位置的ordinary状态恢复。Relay：180个相邻count pairs，交叉L2 terminal donor/self与L3后缀或answer reset，保留L4下游计算。','五个items的trace被等长替换，只恢复item5；随后检查正确答案margin。Relay另测试中间状态reset是否消除terminal donor的自然效应。')
    body+='''<p><b>结果。</b>Bridge受损答案68%，按semantic恢复条件平均为69%，ordinary恢复平均68%；具体scope全表保留。Terminal donor的自然答案效应也小，因此source×reset未建立清晰relay。先要求自然source效应可测，再解释中介交互。</p>'''+ledger('复核：terminal bridge各scope',c['bridge'])+ledger('复核：terminal relay全组合',c['relay_summary'])+conclusion('本批terminal局部恢复较弱，未达到大模型相应实验的证据强度；已有trace-source必要性与terminal局部通路成立是两个不同结论。')
    body+='<h3>9.3 Read → carrier → commit的结构限制</h3>'+protocol('检验检索损伤能否经独立carrier传播到后续commit状态。','90个N≥2输入，在最后retrieval query消融Top-2／4及同层对照，读取item-end各层RMS变化。当前item仅有separator和marker两个token。','要先在一个位置检索，再在其后独立carrier写入，最后在更晚commit测状态；双tokenitem无法为这些角色分别提供原方案中的独立位置。')+conclusion('在保持当前trace不变的要求下，多token Native carrier闭环无法原样复制。末层query-local干预对同层其他固定token状态的零变化是结构性结果，该实验不计为成功复现。')

    body+=heading('dynamics')+protocol('检查原始检索强度、head-bank份额、行为和表征是否同步形成。','同一100题在每100步checkpoint上计算attention（含step0，共101点）；预先固定11个里程碑评行为、干预和几何：0/100/200/400/800/1500/2500/4000/6000/8000/10000。每个图固定物理head，不重新排序。','如果其他heads从几乎零增长，L1H6原始score即使变化不大，其全模型份额也会下降；原始mass与归一化份额必须同时保留。')
    body+='''<p><b>阅读顺序。</b>先看absolute broad／targeted主图，再看逐层定位曲线。相对份额放在折叠辅助图；绝对targeted固定0–1色标，broad在两mode与全部steps间共用固定色标。Log横轴只重新分配横向间距，不改变数据，不能单独证明突变。虚线step1500为训练目标切换。</p>'''+c['images_dynamics']
    body+='''<h3>10.1 L1H6为何变暗，深层增强意味着什么？</h3>
<p>从step1800到10000，L1H6的原始k-to-k mass为0.0634→0.0305，份额为45.17%→0.73%；全模型targeted score总和同期约增加29.8倍。其trace-query对全部needles的mass由12.18%降到5.06%，正确needle Top-1由29.11%到28.17%。均匀选择needle的题均参考为(1/10)Σₙ₌₁¹⁰1/n=29.29%，包含N=1；它不是统计显著性阈值。早期相对份额高尚未建立成熟的正确定位。</p>
<p>最终L1／L2／L3／L4平均原始targeted mass为0.0087／0.0020／0.0979／0.4133；平均正确needle Top-1为29.24%／28.04%／32.97%／67.18%。深层在原始定位强度与准确率上都更强。L1H6的answer-query broad score虽由0.00257到0.00625，最终正文needle总mass仅0.75%；当前不能将其称为明确的targeted→broad角色转换。</p>
<p><b>原因未查明。</b>浅层历史状态是否帮助深层构造下一次targeted query，需要逐层历史状态干预并同时测L4定位与下一marker行为，再做对应状态恢复。Successor先出现、targeted后来增强只提供时间顺序证据；不同query上的broad与targeted不共享同一attention行的预算，不能从热图推导竞争。1500步目标切换及单seed也限制形成时间的解释。</p>'''+conclusion('当前支持深层targeted定位增强；尚未建立浅层功能迁移、successor→targeted串行机制或broad竞争导致深层集中的因果解释。')
    body+='<h3>10.2 Geometry dynamics与行为是否同步？</h3>'+c['geometry_dynamic_image']+'''<p>几何动态固定共同decoder所选的物理层，每个checkpoint分别只在相同discovery上拟合变换与类中心，再评相同confirmation。Thinking L2 final NCC在step400已100%，当时自由生成答案准确率12%；L2 running NCC在step200为72.6%，最终34.2%。这与前节NCC独立选层L4的31.37%属于不同规则，不互相替换。</p>'''+conclusion('Gold trace下的标签可读性与自由生成能力不同步；当前数据不支持representation compression随训练单调增强。早期NCC可能受trace位置／长度影响，具体来源尚需控制实验。')

    body+=heading('alignment')+'''<p>结构参照归档plan中的“任务与证据标准→行为→表征→检索／写入→进度更新→最终读出→路径中介→学习动态”，以及大模型Non-thinking的Form／Retrieve／Consolidate与Native-thinking的Counter／Retrieve／Update／Readout。以下比较计算角色和干预设计，不把Synthetic完成实验等同于复现了大模型结果。</p>'''
    body+=tbl([
        ['Running／final geometry','4 endpoints、discovery拟合、confirmation读出、可选层2D／3D','Thinking可读性较高','固定trace位置混杂；Synthetic是独立模型，Native是预训练模型模式'],
        ['Non-thinking broad retrieval','answer-query head bank、Top-K及同层对照','完成，特异性结论待定','低基线；部分对照无答案；单token span；enrichment筛选未完全对齐'],
        ['Targeted内容检索','trace query的k-to-k排序、定点消融、对应source value恢复','局部内容作用已确认','trace损伤较大、count损伤较小；未解释全部准确率优势'],
        ['Progress与continuation','discovery选层、等范数对照、可识别前缀、自由rollout','短程效果得到支持','仅10个confirmation prompts；长程与纯±1未建立'],
        ['Answer state／trace readout','own-trace来源删除、全层donor patch、方向干预','来源依赖与可执行性已确认','位置／长度不变的纯语义读出尚未隔离'],
        ['Terminal bridge／relay','相同损坏baseline、ordinary恢复、下游reset','完成，恢复较弱','没有复现同等强度的terminal局部通路'],
        ['Read→carrier→commit','query损伤与下游状态测量','架构不适用','双tokenitem缺少独立carrier／later commit'],
        ['训练动态','固定checkpoint／输入／物理head，attention+行为+causal+geometry','深层targeted形成可描述','单seed、目标切换；未证明竞争或稳定相变'],
    ],['计算角色','对齐的实验','Synthetic当前结论','仍有差异'])
    body+='''<p><b>如何解释Thinking优势。</b>行为、targeted内容检索、successor数量作用、contextual continuation与trace-to-answer读出共同支持功能分工。当前不能声称“targeted retrieval通过完整内部counter链解释了95%准确率”，因为targeted消融对count损伤较小、部分桥接与中介实验较弱，且位置／长度信息未隔离。</p>'''+conclusion('可报告受控无编号trace模型中的角色分工与局部因果证据，同时明确Non-thinking通路、完整progress recurrence、terminal中介及位置不变压缩尚未全部复现。正结果、弱结果和架构限制共同构成当前完整结论。')

    body+=heading('reproduce')+'<h3>A. 统一实验量与结果入口</h3>'+table(c['coverage'])
    body+='''<p>200／100始终指独立输入；running的1100／550条状态、answer的180个相邻有向pairs、progress的60pairs均为派生单位。Count10 progress是同一面板的20／10输入子集。所有旧主体实验已在共同面板重算；历史不同规模结果仅作审计，不参与本报告汇总。</p>
<p>报告中的公开PNG和精简CSV位于<code>reports/assets/v58_unified_20260905/</code>。完整逐输入结果在本地<code>work/v58_final/analysis/</code>下三个目录：<code>v58_alignment_supplement_20260905</code>、<code>v58_unified_legacy_20260905</code>、<code>v58_unified_additional_20260905</code>。模型权重、完整hidden states及逐prompt attention未打包进HTML／GitHub；完整复跑需要相应checkpoint与实验数据。HTML中的图片与Plotly几何数据嵌入，阅读不依赖远程CDN。</p>
<h3>B. 常用列名与统计单位</h3>'''+tbl([
        ['accuracy / ar_accuracy','正确答案比例；具体是局部或自由生成由该节prefix条件确定'],
        ['trace_exact','trace区间完整token列表与gold一致，含separator、marker顺序和长度；trace关闭另有trace_closed指标'],
        ['trace_marker_count_accuracy','生成marker数与真实N相同，不要求身份顺序正确'],
        ['donor_adoption','预测等于donor目标的比例；progress各前缀指标仅在eligible子集计算'],
        ['margin','答案实验为正确count相对其余count最大logit差；transport为正确marker相对替换marker差'],
        ['restoration','patched margin减damaged margin；不默认表示归一化恢复率'],
        ['expected_abs_error','|E[count]−真实N|：count期望的绝对误差；不同于E[|count−N|]'],
        ['layer / top_k / repeat','层／消融头数／匹配对照编号，均非新的独立输入'],
    ],['原始列名','解释'])
    body+='''<h3>C. 复现与版本</h3>
<pre><code># 仓库根目录，先准备完整 work/v58_final 数据
.venv/Scripts/python.exe scripts/audit_v58_unified_outputs.py --analysis work/v58_final/analysis
.venv/Scripts/python.exe scripts/build_v58_unified_report.py --output reports/NiaH_Synthetic_report.html
node scripts/check_v58_report_static.cjs reports/NiaH_Synthetic_report.html</code></pre>
<p>训练实际配置：<code>work/v58_final/config.json</code>；loss实现：<code>src/synthetic_counting_v20/training.py</code>；几何实现：<code>src/synthetic_counting_v20/aligned_geometry.py</code>。报告生成器为<code>scripts/build_v58_unified_report.py</code>，阅读版叙述为<code>scripts/v58_report_narrative.py</code>。实验runner、协议与实际状态见仓库scripts及docs目录。</p>
<p>参考文档：<a href="../../Paper_Draft/archive/plan.tex">项目归档plan.tex</a> · <a href="../../Realistic_CoT_NiaH_Count/reports/NiaH_Non-thinking_report.html">Non-thinking大模型报告</a> · <a href="../../Realistic_CoT_NiaH_Count/reports/NiaH_Native-Thinking_report.html">Native-thinking大模型报告</a> · <a href="../../Realistic_CoT_NiaH_Count/reports/NiaH_Geometry_Comparison.html">Geometry Comparison</a>。这些相对链接需要同级项目目录，单独取得本HTML时可能不可用；本文核心实验说明已独立写全。</p>
<p>历史结果：<a href="NiaH_Synthetic_report_pre_alignment_20260905.html">统一面板前报告</a>。输入registry的SHA256为<code>64a276f75b5997ddb7f616a0740a0b5a021e7ff9fabd1ef871f5329d39f9dc15</code>；当前报告及实验manifest哈希见同目录<code>NiaH_Synthetic_report_manifest.json</code>。</p>
<p class="caption">校验范围：构建时检查三份实验完成manifest、实际样本量与输入对齐；静态校验检查JavaScript语法、5200个几何投影点及16个endpoint×layer面板。保留2D／3D选层与3D旋转配置；此前浏览器安全策略阻止本地HTML预览，不能将静态检查描述为现场交互验证。</p>'''
    return finalize_structure(body)


def finalize_structure(body):
    """Make figure numbers follow narrative order without touching embedded JS."""
    number=0
    def caption(match):
        nonlocal number
        number+=1
        text=re.sub(r'^图\s*[0-9]*(?:[a-z])?\s*[｜|：:]\s*','',match.group(1))
        return f'<figcaption>图 {number}｜{text}</figcaption>'
    return re.sub(r'<figcaption>(.*?)</figcaption>',caption,body,flags=re.S)
