"""Additional report section for the separately audited category-mask variant."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

B=Path(__file__).resolve().parents[1]
VERSION='category_target_broad_20260909_v1'


def section(report, table, para, ci, percent):
    run=B/'runs'/VERSION;root=run/'downloaded'
    if not (run/'package/protocol.json').exists(): return '',None,{}
    title='<h3 id="category-target-broad">后续实验：Non-thinking category 按所问类别选头</h3>'
    if not (root/'analysis/audit.json').exists():
        return title+para('新实验已准备：问 city 时只用 city 记录，问 flower 时只用 flower 记录计算 Broad 的 M、H 和 J；双模型分别重新选头与消融。完整核验通过后，本节展示结果；下方原v3结果作为全部记录选头的参考。'),dict(version=VERSION,complete=False),{}
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    def rows(p):
        with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    cfg=read(root/'protocol.json');audit=read(root/'analysis/audit.json')
    assert cfg['version']==VERSION and cfg['record_scope']=='question_target_category'
    assert audit['status']=='PASS' and audit['checks']['points']==1300 and audit['checks']['discovery_rows_rescored']==400
    assert audit['protocol_sha256']==sha(root/'protocol.json')==sha(run/'package/protocol.json')
    summary=rows(root/'analysis/summary.csv');tests=rows(root/'analysis/hypothesis_tests.csv')
    names={'all_examples':'合并问题','city_questions':'只看 city 问题','flower_questions':'只看 flower 问题'}
    populations=list(names);models=cfg['models']
    def get(model,k,pop,metric):
        matches=[r for r in summary if r['model']==model and int(r['k'])==k and r['population']==pop and r['metric']==metric]
        assert len(matches)==1
        return matches[0]
    best={m:min(cfg['sizes'][m],key=lambda k:(-float(get(m,k,'all_examples','delta')['mean']),k)) for m in models}
    part=title+para('本次仅改变 Non-thinking category 的选头记录范围。问 city 时，M、H、J 只由 city 记录计算；问 flower 时只用 flower。每题 J 为所问类别的记录数（1/3/5/7/9）。两类问题在同一 category discovery 集合内先 seed 内、再 seed 等权汇总，得到每模型独立的一套全局排序。', '原始题目、评分、最终回答干预位置、K 网格、随机对照及64-token预算均保持原样。')
    part+=para('双模型400条 discovery attention 重新计算并冻结排序，1300个 case×K、6500个条件输出记录重评分通过；主分析每模型100输入（city / flower 各50）。表中 K 按合并问题的 Random−Selected 最大值事后选取，city / flower 分组沿用同一 K。若并列，展示最小 K；完整扫描如下。', '峰值属于探索性汇总，区间为逐点95%区间，未校正选 K；city / flower 分组只作描述。')
    data=[]
    for model in models:
        k=best[model]
        for pop in populations:
            rr={metric:get(model,k,pop,metric) for metric in ['clean','selected','random','delta','delta_gain_vs_all_records']}
            p=next(t for t in tests if t['model']==model and int(t['k'])==k and t['population']=='all_examples')
            data.append([model,names[pop],k,rr['delta']['n'],percent(rr['clean']),percent(rr['selected']),percent(rr['random']),ci(rr['delta']),ci(rr['delta_gain_vs_all_records']),f"{float(p['p_holm']):.4f}" if pop=='all_examples' else '描述性'])
    part+='<div id="target-category-best">'+table(['模型','问题范围','K','n','Clean %','Selected %','Random %','Δ pp [95% CI]','相对全部记录选头的 Δ 增量 pp [95% CI]','Holm p'],data)+'</div>'
    for effect in [False,True]:
        fig,axs=plt.subplots(3,2,figsize=(11.5,10.6))
        for i,pop in enumerate(populations):
            for j,model in enumerate(models):
                ax=axs[i,j];ks=cfg['sizes'][model]
                for metric,color in ([('delta','#c66a33')] if effect else [('clean','#688070'),('selected','#c66a33'),('random','#2563a6')]):
                    rr=[get(model,k,pop,metric) for k in ks]
                    y,lo,hi=[[100*float(r[field]) for r in rr] for field in ['mean','lower','upper']]
                    ax.plot(ks,y,'-o',lw=1.7,ms=3.5,color=color,label='Target-category selection' if effect else metric.title())
                    ax.fill_between(ks,lo,hi,color=color,alpha=.12)
                if effect:
                    old=[100*(float(get(model,k,pop,'delta')['mean'])-float(get(model,k,pop,'delta_gain_vs_all_records')['mean'])) for k in ks]
                    ax.plot(ks,old,'--s',lw=1.4,ms=3,color='#2563a6',label='All-record selection (v3)')
                    ax.axhline(0,color='#777',ls=':',lw=.8)
                else:ax.set_ylim(-2,102)
                if model.startswith('Qwen'):ax.set_xscale('log',base=2)
                ax.set_xticks(ks,[str(k) for k in ks]);ax.set_xlabel('Number of ablated heads K')
                if j==0:ax.set_ylabel('Random - Selected (pp)' if effect else 'Final-answer accuracy (%)')
                ax.set_title(model+' / '+{'all_examples':'All questions','city_questions':'City questions','flower_questions':'Flower questions'}[pop])
                ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout()
        caption='按所问类别选头的 Non-thinking category 完整K扫描。行分别为合并问题（每模型100输入）、city问题（50输入）、flower问题（50输入）；列分别为Qwen和Gemma。每组10个seed，所有K共用相同输入。横轴为消融头数，Qwen使用log₂刻度。'
        caption+=('纵轴为Random−Selected（百分点，正值表示Selected损伤更大）；橙色为新目标类别选头，蓝色虚线为原v3全部记录选头点估计，零线为灰色。阴影仅为新实验逐点95%区间；新旧Δ增量的配对区间见表。' if effect else '纵轴为最终答案准确率（%）；绿为Clean、橙为Selected、蓝为三个Random的均值。阴影为seed配对bootstrap逐点95%区间，错误与解析失败均计入。')
        part+=report.figure(fig,'category_target_broad_'+('effect' if effect else 'accuracy'),caption)
    claims=[]
    for model in models:
        k=best[model];r=get(model,k,'all_examples','delta');gain=get(model,k,'all_examples','delta_gain_vs_all_records')
        p=next(t for t in tests if t['model']==model and int(t['k'])==k and t['population']=='all_examples')
        claims.append(f"{model}，观察峰值 K={k}：Δ={ci(r)} pp，相比原全部记录选头的配对Δ增量={ci(gain)} pp，Holm p={float(p['p_holm']):.4f}。")
    part+=para('<br>'.join(claims),'只有Δ增量的方向及其配对区间能直接说明当前数据上选头范围改变后的效应变化；不能仅凭新峰值大于旧的某个取点来判断改进。')
    significant=sum(float(t['p_holm'])<.05 for t in tests if t['population']=='all_examples')
    part+=para(f'本变体主分析13个比较中，{significant}个达到Holm p<0.05。Gemma的效应增强主要体现在city问题，但合并问题的新旧Δ增量区间跨零；Qwen的观察峰值未改善。', '当前数据不支持目标类别选头对两模型均有稳定改进。')
    part+=para('本变体合并问题主分析的Holm族包含双模型全部13个K比较，Clean-correct另成同样范围的比较族；city / flower分组和新旧Δ增量仅作描述。原v3的Broad族有52个比较，两者的校正p不是同一比较族。样本划分曾反复用于分析，本变体是在观察原结果后提出。', '自然准确率、kth、Native-thinking和Targeted仍使用原v3结果，后续变体不替换其数据。')
    numeric=audit.get('numeric_comparison')
    if numeric and numeric['mode']=='ulp8':
        part+='<details><summary>本次本地数值复核与恢复说明</summary>'+para(f"本地重算的{numeric['scores_compared']}个discovery分数中，{numeric['nonidentical_scores']}个存在浮点末位差异，最大绝对差{numeric['max_absolute_difference']:.3g}，最大{numeric['max_ulp_distance']}个binary64可表示数间隔。显式ulp8复核最多允许8个间隔，零值须精确一致；这是事后的跨平台数值验证调整。完整头顺序、全部Top-K及输出评分一致；冻结协议、分数和GPU产物未改写。", '完整头排序和实验结果通过核验；分数不能表述为跨平台逐位一致。')+'</details>'
    sources=[root/'protocol.json',root/'analysis/audit.json',root/'analysis/summary.csv',root/'analysis/hypothesis_tests.csv',root/'analysis/per_case.csv']
    metadata=dict(version=VERSION,complete=True,record_scope=cfg['record_scope'],summary=summary,tests=tests,best_k=best,points=1300)
    return part,metadata,{str(p.relative_to(B)):sha(p) for p in sources}
