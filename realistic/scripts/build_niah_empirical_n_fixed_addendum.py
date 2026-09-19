"""Render the V3.2-only fixed-effect comparison into the shared report."""
import json
import re
import base64
from pathlib import Path
import pandas as pd
from analyze_realistic_niah_v3_2_n_fixed import OUT, ROOT

START = '<!-- V3_2_N_FIXED_START -->'
END = '<!-- V3_2_N_FIXED_END -->'
FRONT_START = '<!-- LENGTH_FINDINGS_FRONT_START -->'
FRONT_END = '<!-- LENGTH_FINDINGS_FRONT_END -->'


def all_n_linear_section():
    if (ROOT / 'reports/assets/niah_empirical_all_n_length/comparison_manifest.json').exists():
        from build_niah_all_n_length_comparison import build_section
        return build_section()
    folder = ROOT / 'outputs/anvil_realistic_niah_v3_3_long_context_20260906_holdout/analysis/v3_3_all_n_linear_length_20260909'
    if not (folder / 'manifest.json').exists():
        return ''
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['fits'] == 4 and manifest['archived_cv_reproduced']
    metrics = pd.read_csv(folder / 'metrics.csv')
    per_n = pd.read_csv(folder / 'per_N_metrics.csv')
    coefficients = pd.read_csv(folder / 'coefficients.csv')
    rows = []
    for _, row in metrics.iterrows():
        rows.append({'模型':row.model, '模式':'Non-thinking' if row['mode']=='direct' else 'Native-thinking',
            '共同斜率 β':f'{row.beta_L_k:.4f}', '整体 R²':f'{row.cell_R2:.3f}',
            '留出条件 R²':f'{row.cell_CV_R2:.3f}', '长度项带来的损失下降':f'{100*row.cv_d2_vs_N_fixed:.1f}%'})
    summary = pd.DataFrame(rows).to_html(index=False,classes='data-table',border=0)
    figures = ''
    for index,model in enumerate(('Gemma4-31B','Qwen3-32B'),3):
        path = ROOT / f'reports/assets/niah_empirical_all_n_length/{model}_all_N.png'
        src = 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode('ascii')
        figures += f'<figure><img src="{src}" alt="{model}：全部 N 共用长度斜率的拟合"><figcaption><strong>图 P{index}</strong><span>{model}。每个小图固定一个 N；横轴为 1k–100k 原文长度，使用线性刻度，纵轴为正确回答概率。橙色实线/圆点是 Non-thinking，紫色虚线/三角是 Native-thinking。点为每个条件 30 次请求的正确率，线为全部 N 共同拟合的结果；竖直点线标出 20k。两种模式分别估计 β。</span></figcaption></figure>'
    per_n_html = per_n.to_html(index=False,classes='data-table',border=0,float_format=lambda x:f'{x:.4f}',na_rep='未定义')
    coefs_html = coefficients[['model','mode','term','estimate','ci95_low','ci95_high']].to_html(index=False,classes='data-table',border=0,float_format=lambda x:f'{x:.5f}',na_rep='不可用')
    return r'''<div id="all-n-linear-length"><h3>让不同 N 共用一个长度斜率</h3>
<p>这次保留全部 14 个 N，每个 N 有自己的起点，但共用一个线性长度系数 β。两个模型、两种模式分别拟合，共四组；每组只有 14 个截距和 1 个长度系数。所有 N 都保留，包括 1、2、3。</p>
<p>数据仍是两模型的 1k–100k，共 28,560 次请求。这次统一使用线性 L，回答的是“一个共同斜率能否解释不同 N”；之前固定 N=10 的线性/对数比较仍单独保留。</p>
<div class="table-wrap">@@TABLE@@</div>
<p>四组共同斜率都为负，加入长度项后都改善了预测。“损失下降”是相对只有各个 N 起点、没有长度项的模型，按留出条件上的 log loss 计算。整体 R² 同时包含 N 和 L 的解释能力，不能全部算作长度的贡献。</p>
@@FIGURES@@
<p><strong>主要发现：</strong>Gemma 的整体拟合较好；Qwen，尤其 Non-thinking，存在共同斜率难以解释的 N。例如 Qwen Non-thinking 在 N=10 上，拟合 R² 为 −1.116：这一组观测在长端回升，共同负斜率无法跟随。汇总后的负斜率不会消除这个局部例外。</p>
<p>因此，α_N+βL 是一个有预测价值的简洁近似，但不能认为所有 N 都已经被同一种长度规律充分解释。每个 N 的检查结果放在下面。</p>
<details><summary>各 N 的拟合检查、系数与计算方式</summary>
<div class="math-block">\[\operatorname{logit}p(N,L)=\alpha_N+\beta L_k,\qquad L_k=L/1000.\]</div>
<p>共享的是 logit 概率上的斜率，概率图中的曲线不必平行。采用 Bernoulli GLM；14 个 N 指示变量提供截距，不额外添加全局截距。每组训练数据为 14×17×30=7,140 次请求。预测检查沿用五折留条件规则，截距基线由训练折内相同 N 的回答估计；不代表 100k 之外的外推能力。</p>
<p>整体 R² 使用全部 238 个条件的观测概率；各 N 的 R² 使用该 N 的 17 个长度。负值表示比该组观测概率均值更差；观测概率无变化时 R² 未定义。系数区间采用原方法的 HC3 95% 区间，未校正多重比较。</p>
<div class="table-wrap">@@PER_N@@</div><div class="table-wrap">@@COEFS@@</div>
<p>复现：python -s scripts/analyze_niah_all_n_linear_length.py；随后 python -s scripts/build_niah_empirical_n_fixed_addendum.py。输入哈希、逐条件预测与各 N 指标保存在 v3_3_all_n_linear_length_20260909 目录。</p></details></div>'''.replace('@@TABLE@@',summary).replace('@@FIGURES@@',figures).replace('@@PER_N@@',per_n_html).replace('@@COEFS@@',coefs_html)


def length_audit(metrics, coefs):
    """Paired comparisons use primary loss, separate from squared-error gains."""
    rows = []
    for (mode, outcome), block in metrics.groupby(['prompt_mode', 'outcome_family']):
        losses = block.pivot(index='comparison_slot', columns='length_term', values='primary_loss')
        for term in ('L_k', 'logL'):
            slopes = coefs.loc[coefs.prompt_mode.eq(mode) & coefs.outcome_family.eq(outcome) & coefs.term.eq('beta_' + term)]
            rows.append(dict(prompt_mode=mode, outcome_family=outcome, length_term=term,
                models=len(losses), positive_slopes=int(slopes.estimate.gt(0).sum()),
                negative_slopes=int(slopes.estimate.lt(0).sum()), median_beta=slopes.estimate.median(),
                improved_primary_loss=int(losses[term].lt(losses['none']).sum()),
                log_beats_linear=int(losses.logL.lt(losses.L_k).sum()),
                median_log_vs_linear_loss_reduction=((losses.L_k-losses.logL)/losses.L_k).median()))
    return pd.DataFrame(rows)


def inject(report):
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    assert not manifest['failures'] and manifest['metric_rows'] == 432
    metrics = pd.read_csv(OUT / 'tables/metrics.csv')
    coefs = pd.read_csv(OUT / 'tables/coefficients.csv')
    metrics['length_term'] = metrics.length_term.fillna('none')
    coefs['length_term'] = coefs.length_term.fillna('none')
    audit = length_audit(metrics, coefs)
    audit.to_csv(OUT / 'tables/length_findings_audit.csv', index=False)
    metrics['length_gain'] = metrics.cv_d2_vs_N_fixed.fillna(metrics.cv_relative_sse_reduction_vs_N_fixed)
    summary = metrics.groupby(['outcome_family', 'prompt_mode', 'length_term'], sort=False).agg(
        models=('comparison_slot', 'size'), median_CV_score=('primary_score', 'median'),
        valid_CV_scores=('primary_score', 'count'), median_length_gain=('length_gain', 'median'),
        improved_models=('length_gain', lambda x: int((x > 1e-10).sum()))).reset_index()
    summary.to_csv(OUT / 'tables/report_summary.csv', index=False)
    def table(frame):
        return frame.to_html(index=False, classes='data-table', border=0, float_format=lambda v: f'{v:.4g}', na_rep='undefined')
    main = summary.loc[summary.prompt_mode.isin(['direct', 'native_thinking'])]
    section = r'''<style>#v3-2-n-fixed-effects{order:8} #v3-2-n-fixed-effects .table-wrap{overflow-x:auto} #v3-2-n-fixed-effects .data-table{font-size:11px}</style><section id="v3-2-n-fixed-effects" class="appendix">
<div class="section-head"><div class="section-no">V3.2 supplement</div><div><h2>N 固定效应与共享长度斜率</h2><p class="lede">在 V3.2 原有 12 个模型比较槽、四种提示模式、1k–20k 范围内，检验控制 N 后的长度预测信息。</p></div></div>
<div class="math-block">\[g\{\mu(N,L)\}=\alpha_N+\beta L_k,\qquad L_k=L/1000.\]</div>
<p>每个 N 水平都有独立截距，β 在同一模型、同一提示模式的各个 N 之间共享；不同模型及提示模式分别拟合。使用完整 14 个 N 指示变量，因此不再添加全局截距，但模型仍含 N 特异截距。没有 N×L 交叉项。比较 α_N、α_N+βL_k 和 α_N+βln(L_k) 三种形式；本补充保留 N=1、2、3，不使用分段筛选。</p>
<p>Accuracy 使用 request-level Bernoulli logit；MAE 和 signed bias 使用原有 10% 双侧截尾 cell estimand、identity link 和 cell 等权 OLS。沿用原网格的五折 held-condition CV，同一 N 的其他长度仍出现在训练集中。此验证衡量已见 N 下的未见条件预测，不支持未见 N 或 20k 以外的外推。MAE identity 回归不裁剪负预测。</p>
<p>表中 CV score 对 Accuracy 为相对全局截距 OOF 基线的 D²，对 MAE/Bias 为 R²。length gain 对 Accuracy 为 1−logloss/logloss(α_N)，对 MAE/Bias 为 1−SSE/SSE(α_N)，正值表示长度项改善预测。中位数跨 12 个比较槽计算；improved_models 是 gain 为正的槽数。零方差导致的 undefined 不作为零分参与中位数，valid_CV_scores 给出有效槽数。</p>
<div class="table-wrap">@@MAIN@@</div>
<h3>长度回归的主要发现</h3>
<p><strong>Accuracy：Native-thinking 的对数长度形式呈现一致的跨槽优势。</strong>控制 N 后，两种长度形式的 Accuracy slope 在 Non-thinking 和 Native-thinking 的全部 12 个比较槽中均为负。Native-thinking 中，α_N+βln(L_k) 在 12/12 个槽上降低 α_N 基线的 CV log loss，也在 12/12 个槽上优于线性长度形式；相对线性形式的 log loss 降幅中位数为 2.71%，范围 1.10%–7.19%。Non-thinking 中对数形式优于线性形式的槽数为 8/12，降幅中位数为 0.82%，函数形状的一致性较弱。</p>
<div class="math-block">\[\operatorname{logit}p_N(L)=\alpha_N+\beta\ln L_k\quad\Longrightarrow\quad\frac{p_N(L)}{1-p_N(L)}=e^{\alpha_N}L_k^{\beta}.\]</div>
<p>因此候选规律是“正确回答的 odds 随长度呈幂律变化”。Native-thinking 的 β 中位数为 −1.741；各槽长度翻倍的拟合 odds ratio 中位数为 0.299（四分位范围 0.249–0.345）。这不是准确率本身下降 70%，也不表示所有槽共享同一个 β。该解释限于 V3.2 的 1k–20k 范围；CV 优势是探索性证据，12 个槽也不是相互独立的随机样本。</p>
<p><strong>MAE/Bias：平方误差改善与绝对误差改善需要分别报告。</strong>加入线性长度项后，Non-thinking 的 MAE/Bias CV R² 中位数为 0.807/0.750，Native-thinking 为 0.268/0.213。但相对 α_N 基线，Native-thinking 的 MAE/Bias 分别只有 4/12、3/12 个槽降低 CV 绝对误差，尽管各有 11/12 个槽降低 CV 平方误差。长度项可能主要帮助预测较大的条件误差；该解释仍需检查逐 cell 残差。仅凭跨槽 R² 中位数，不足以决定线性长度或对数长度哪种普遍更好。</p>
<details><summary>逐模型配对比较的计数与斜率汇总</summary><p>improved_primary_loss：Accuracy 使用 CV log loss；MAE/Bias 使用对 cell estimand 的 CV 绝对预测误差。log_beats_linear 为相同槽内 logL 优于 L_k 的数量，两个 length_term 行重复展示同一配对计数。</p><div class="table-wrap">@@AUDIT@@</div></details>
<details><summary>四种提示模式的完整汇总</summary><div class="table-wrap">@@SUMMARY@@</div></details>
<details><summary>432 个独立拟合的 CV 指标</summary><div class="table-wrap">@@METRICS@@</div></details>
<details><summary>共享长度斜率及 HC3 95% 区间</summary><p>区间来自完整样本拟合，未校正多重比较；数值协方差非有限时不作显著性判断。所有 α_N 系数保存在 coefficients.csv。</p><div class="table-wrap">@@COEFS@@</div></details>
<p>版本范围：本组固定效应分析归属 V3.2。V3.3 用于 Gemma4-31B 与 Qwen3-32B 各自的长上下文拟合和冻结公式外推检验。</p>
<div class="methods">Source: @@INPUT@@
SHA256: @@SHA@@
Requests: 161280; cells: 5376; eligible continuous cells: @@CELLS@@
Fits: 432; failures: 0; bootstrap: 0
Rebuild: python -s scripts/analyze_realistic_niah_v3_2_n_fixed.py
         python -s scripts/build_niah_empirical_n_fixed_addendum.py
Output: @@OUT@@</div></section>'''
    slopes = coefs.loc[coefs.term.str.startswith('beta_'), ['comparison_slot', 'prompt_mode', 'outcome_family', 'term', 'estimate', 'ci95_low', 'ci95_high']]
    shown = metrics[['comparison_slot','prompt_mode','outcome_family','length_term','n_rows','n_parameters','primary_loss','primary_score','length_gain']]
    for key, value in {'MAIN':table(main),'SUMMARY':table(summary),'METRICS':table(shown),'COEFS':table(slopes),'AUDIT':table(audit),
        'INPUT':manifest['input'],'SHA':manifest['input_sha256'],'CELLS':str(manifest['eligible_cells']), 'OUT':str(OUT.relative_to(ROOT))}.items():
        section = section.replace('@@'+key+'@@', value)
    findings_start = section.index('<h3>长度回归的主要发现</h3>')
    findings_end = section.index('<details>', findings_start)
    section = section[:findings_start] + '<p>主要发现见报告开头的<a href="#length-findings">长度回归发现</a>；以下保留完整配对比较与回归表。</p>' + section[findings_end:]
    assets = ROOT / 'reports/assets/niah_empirical_front'
    def front_image(name):
        return 'data:image/png;base64,' + base64.b64encode((assets/name).read_bytes()).decode('ascii')
    front = r'''<style>#length-findings{order:0}</style><section id="length-findings">
<div class="section-head"><div class="section-no">主要发现</div><div><h2>文本变长后，两种模式怎么变化？</h2><p class="lede">Native-thinking 的长度变化规律更一致，更接近对数形式。Non-thinking 的表现更依赖具体模型，目前还没有统一的长度规律。</p></div></div>
<h3>先看 12 个比较槽的整体表现</h3>
<p>这里的 Probability 就是“回答正确的概率”。每个小图比较同一槽中的 Non-thinking 和 Native-thinking。N 是需要数的目标数，L 是原文长度。</p>
<figure><img src="@@PROBABILITY@@" alt="12 个比较槽的正确回答概率，两种模式对应展示"><figcaption><strong>图 P1</strong><span>横轴：目标数 N；纵轴：正确回答概率。颜色表示原文长度。实线和圆点为 Non-thinking，虚线和三角为 Native-thinking。点是每个条件下 30 次请求的正确率，线是拟合结果。数据范围为 1k–20k，共 14 种 N、8 种 L；N 轴使用对数刻度。</span></figcaption></figure>
<p>为了单独看长度的影响，我们允许每个 N 有自己的起点，再比较两种长度曲线。<strong>Native-thinking 的 12 个槽都更适合对数长度；Non-thinking 有 8 个槽更适合对数长度，另外 4 个更适合线性长度。</strong>这里的“更适合”根据未参与拟合的长度条件上的预测表现判断。</p>
<h3>再看两个模型在 1k–100k 上的表现</h3>
<p>下面固定 N=10，只改变原文长度。左右两列是 Gemma4-31B 和 Qwen3-32B，上下两行是两种模式。每组都比较线性 L 和对数 ln L。</p>
<figure><img src="@@LENGTH@@" alt="固定 N=10，两个模型和两种模式的长度回归"><figcaption><strong>图 P2</strong><span>横轴：原文长度，使用对数刻度；纵轴：正确回答概率。空心点是观测结果。实线为线性 L，虚线为对数 ln L；彩色曲线是选中的形式，灰色是另一种形式。竖直点线标出 20k。曲线使用 1k–100k 数据重新拟合，不能当作短程公式的长程预测结果。</span></figcaption></figure>
<div class="table-wrap"><table class="data-table"><thead><tr><th>模型与模式</th><th>选中的形式</th><th>曲线贴合程度 R²</th><th>留出预测分数 D²</th></tr></thead><tbody><tr><td>Gemma / Non-thinking</td><td>线性 L</td><td>0.265</td><td>0.138</td></tr><tr><td>Gemma / Native-thinking</td><td>对数 ln L</td><td>0.953</td><td>0.297</td></tr><tr><td>Qwen / Non-thinking</td><td>线性 L</td><td>0.500</td><td>0.064</td></tr><tr><td>Qwen / Native-thinking</td><td>对数 ln L</td><td>0.624</td><td>0.228</td></tr></tbody></table></div>
<p><strong>Native-thinking：</strong>两个模型都更适合对数形式，Gemma 的曲线尤其贴近观测点。<strong>Non-thinking：</strong>两个模型都选中线性形式，但方向不同：Gemma 随长度下降，Qwen 略有上升。Qwen 的留出预测分数较低，不能把它当成稳定的“越长越好”。</p>
<p>R² 看曲线是否贴近图中的点；D² 看留出一些长度后，模型能否预测这些条件。两者采用不同的误差算法，数值不能直接比较。</p>
@@ALL_N_LINEAR@@
<h3>线性和对数，分别在说什么？</h3>
<p><strong>线性形式关注“又增加了多少文本”。</strong>例如，增加 1k tokens，在短文本和长文本上有相同的模型化影响。它适合描述随新增长度不断累积的作用。</p>
<p><strong>对数形式关注“文本变成了原来的几倍”。</strong>例如，2k→4k 和 20k→40k 都是翻倍，在这个模型里有相同的模型化影响。</p>
<p>这里的“影响”指正确与错误的相对倾向，经统计变换后的变化。它不表示准确率每次下降相同的百分点。公式放在下方，读懂主要发现不需要先读公式。</p>
@@MECHANISM@@
<details><summary>统计细节：公式、拟合分数与完整比较</summary>
<p>12 槽分析使用 V3.2 的 1k–20k 数据。每个 N 独立截距，不同模型、模式分别拟合，没有 N×L 交叉项。两模型长程图另用 V3.3 扩展数据，固定 N=10，在 17 个长度上重新拟合。GLM 与 Ministral 的比较槽包含配对的不同模型版本，模式差异不能全部归因于思考开关。</p>
<div class="math-block">\[\text{线性：}\quad\operatorname{logit}p=\alpha_N+\beta x;\qquad \text{对数：}\quad\operatorname{logit}p=\alpha_N+\beta\ln x,\quad x=L/1000.\]</div>
<p>odds=p/(1−p)。线性形式中，每增加 Δx，odds 乘以 exp(βΔx)；对数形式中，长度乘以 c，odds 乘以 c^β。对数形式的幂律描述的是 odds。V3.2 Native-thinking 的 β 中位数为 −1.741，长度翻倍的拟合 odds ratio 中位数为 0.299。</p>
<p>R²=1−Σ(p_obs−p_fit)²/Σ(p_obs−mean(p_obs))²，使用完整拟合对 17 个观测概率评分。CV D²=1−留出 log loss/训练折截距基线的留出 log loss，使用每次回答对错评分。Gemma Native-thinking 的对数拟合，样本内 D²=0.302，CV D²=0.297；与 R²=0.953 的差距主要来自误差算法。候选选择沿用原来的五折验证规则。</p>
<div class="table-wrap"><table class="data-table"><thead><tr><th>V3.2 比较</th><th>Non-thinking</th><th>Native-thinking</th></tr></thead><tbody><tr><td>对数长度斜率为负</td><td>12/12</td><td>12/12</td></tr><tr><td>加入对数长度后，优于仅按 N 拟合</td><td>11/12</td><td>12/12</td></tr><tr><td>对数优于线性的槽数</td><td>8/12</td><td>12/12</td></tr><tr><td>相对线性的 log loss 降幅中位数</td><td>0.82%</td><td>2.71%</td></tr></tbody></table></div>
<p>MAE 和 bias 的长度规律较弱。Native-thinking 加入线性长度项后，各有 11/12 个槽改善平方误差，但分别只有 4/12、3/12 个槽改善绝对误差。因此本节主要突出正确率的结果。Probability 将解析失败计为回答错误。</p>
<p>定义参考：<a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.d2_log_loss_score.html">D²</a>；<a href="https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.links.Logit.html">logit</a>。</p></details>
<p><a href="#v3-2-n-fixed-effects">完整固定效应结果</a> · <a href="#v3-3-long-context">两模型长程分析</a></p></section>'''
    mechanism = r'''<div id="length-mechanism-hypothesis"><h3>可能的原因：两种模式处理原文的方式不同</h3>
<p><strong>Non-thinking：回答时直接汇总原文中的多个目标。</strong>我们把这种广泛读取称为 broad retrieval。文本变长后，目标可能更难同时被读到，也可能更容易混入无关信息。因此，最终结果可能随目标位置和文本内容而波动。</p>
<p>如果每多一段文本，就让有效信息减少一些，长度影响可能近似线性。这是对线性项的一种解释。现有实验还没有证明 broad retrieval 确实按这个速度变差。</p>
<p><strong>Native-thinking：先逐项找出目标，写成中间记录，再据此回答。</strong>这些中间记录就是 trace。记录已经正确写好时，最终回答可能不必再次从长原文中汇总所有目标，因此更容易保持稳定。已有擦除实验支持 trace 对最终回答的重要性。</p>
<p>不过，写出记录之前，模型仍要从原文里找到目标。如果目标本身仍容易辨认，主要困难只是候选越来越多，那么长度影响可能更接近对数形式：重要的是文本增加了几倍，而不是又增加了多少。这仍是假设，不能只凭曲线形状确认。</p>
<div class="conclusion"><strong>目前最简洁的解释：</strong>Non-thinking 可能更受“直接从长原文汇总信息”的影响；Native-thinking 可能用逐项检索和中间记录减轻这一负担。这个差异能帮助理解两种长度曲线，但还不能证明线性和对数分别由这两条机制产生。</div>
<h3>还缺哪一步证据？</h3>
<p>最直接的检验是：给 Native-thinking 同一份正确的中间记录，再改变原文长度。如果最终答案明显更稳定，就支持“主要困难在找目标，记录写好后较稳健”。Non-thinking 则需要直接检查：原文变长时，读到的目标是否更少，以及修复这些读取后能否改善答案。</p>
<p>还要在同一批模型上验证。现有主要机制实验使用 Qwen3-8B 和 Gemma4-E4B，长程图使用 Qwen3-32B 和 Gemma4-31B。Qwen Non-thinking 的弱正斜率也是一个待解释的例外。</p>
<details><summary>机制推导与来源（可跳过）</summary>
<p><strong>条件推导。</strong>令 x=L/1000；在一个注册检索位置，将目标 token 的未归一化 attention 权重之和记为 T(x)，其他 token 的权重之和记为 D(x)。由 softmax，目标 attention mass 为 q=T/(T+D)。这是 <a href="https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1">attention 归一化定义</a>的直接结果；q 是 attention mass，与整题正确率 p 需分别测量。</p>
<div class="math-block">\[T(x)=T_0e^{-\kappa x},\qquad D(x)=D_0x^\gamma\quad\Longrightarrow\quad \operatorname{logit}q=\log\frac{T_0}{D_0}-\kappa x-\gamma\ln x.\]</div>
<p>这里 T 的指数衰减等价于 log target weight 在测试区间近似线性减弱，是待测假设；D 的幂次增长同样待测。若有效干扰数量正比于长度、每个干扰的平均未归一化权重稳定，则 γ≈1。若每个干扰平均权重也变动，γ 可以不同。不能用“有更多干扰”直接认定特定 γ。</p>

<p>这个推导描述 attention 分配到目标上的比例。最终答对还取决于多次检索、重复或遗漏，以及如何生成答案；两者之间的关系需要另外验证。</p>
<p>本地证据：<a href="NiaH_Non-thinking_report.html">Non-thinking 机制报告</a>；<a href="NiaH_Native-Thinking_report.html">Native-thinking 机制报告</a>。位置偏差也可能影响长度结果，相关研究见 <a href="https://arxiv.org/abs/2406.16008">Found in the Middle</a>。</p></details></div>'''
    front = front.replace('@@MECHANISM@@', mechanism)
    front = front.replace('@@ALL_N_LINEAR@@', all_n_linear_section())
    front = front.replace('@@PROBABILITY@@', front_image('probability_12_slots.png')).replace('@@LENGTH@@', front_image('length_two_models.png'))
    report = re.sub(re.escape(FRONT_START)+r'.*?'+re.escape(FRONT_END), '', report, flags=re.S)
    main_anchor = '<main class="page">'
    if report.count(main_anchor) != 1:
        raise ValueError('Expected one main report container')
    report = report.replace(main_anchor, main_anchor + FRONT_START + front + FRONT_END, 1)
    nav_link = '<a href="#length-findings">长度回归发现</a>'
    if nav_link not in report:
        report = report.replace('<a href="#design">', nav_link + '<a href="#design">', 1)
    report = re.sub(re.escape(START)+r'.*?'+re.escape(END), '', report, flags=re.S)
    anchor = report.find('<!-- V3_3_LONG_CONTEXT')
    if anchor < 0:
        anchor = report.find('<section id="v3-3-long-context"')
    if anchor < 0:
        anchor = report.find('<section id="repro')
    if anchor < 0:
        raise ValueError('Missing report insertion anchor')
    return report[:anchor] + START + section + END + report[anchor:]


if __name__ == '__main__':
    path = ROOT / 'reports/NiaH_Empirical-law_report.html'
    path.write_text(inject(path.read_text(encoding='utf-8')), encoding='utf-8')
    print(path)
