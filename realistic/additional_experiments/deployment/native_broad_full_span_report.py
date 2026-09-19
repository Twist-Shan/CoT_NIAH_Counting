"""Display separately audited full-span Native Broad with unchanged old results."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt

B=Path(__file__).resolve().parents[1];VERSION='native_broad_full_span_20260909_v1'

def section(report,table,para,ci,percent):
    run=B/'runs'/VERSION;root=run/'downloaded'
    if not (run/'package/protocol.json').exists():return '',None,{}
    title='<h3 id="native-full-span-broad">Native Broad 对齐重跑：完整生成记录 span</h3>'
    audit_path=root/'analysis/audit.json'
    candidate=json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else None
    exception_path=run/'numeric_recovery/accepted_verification.json'
    exception=json.loads(exception_path.read_text(encoding='utf-8')) if exception_path.exists() else None
    accepted_exception=False
    if exception is not None:
        assert exception['status']=='ACCEPTED_NUMERIC_EXCEPTION' and exception['version']==VERSION
        assert exception['authorization']['approved'] is True
        assert candidate and candidate['status']=='PASS' and candidate['numeric_comparison']['mode']=='exact'
        for relative,expected in exception['evidence_hashes'].items():
            assert hashlib.sha256((run/relative).read_bytes()).hexdigest()==expected,relative
        accepted_exception=True
    locally_verified=bool(candidate and candidate.get('status')=='PASS' and not candidate.get('diagnostic_only')
        and candidate.get('numeric_comparison',{}).get('mode')=='ulp8'
        and candidate['numeric_comparison']['max_ulp_distance']<=8)
    locally_verified=locally_verified or accepted_exception
    if not locally_verified and (run/'delivery_failure.json').exists():
        failure=json.loads((run/'delivery_failure.json').read_text(encoding='utf-8'))
        progress=json.loads((run/'delivery_progress.json').read_text(encoding='utf-8'))
        if progress.get('status')=='COMPLETE':
            explanation='四组完整span实验已完成，GPU端精确审计通过；本地审计尚未通过，最终验收暂停。'
            diagnostic=run/'numeric_recovery/local_analysis/audit.json'
            if diagnostic.exists():
                detail=json.loads(diagnostic.read_text(encoding='utf-8'))
                if detail.get('status')=='NUMERIC_HOLD':
                    n=detail['numeric_comparison'];count=len(detail['numeric_violations'])
                    explanation+=f"本地已重评分{detail['checks']['arms_rescored']}条输出并复现4套完整头排序；{n['scores_compared']}个选头分数中{count}个相差{n['max_ulp_distance']}个浮点间隔，超过原定8个间隔的上限。阈值及原始分数保持不变。"
            return title+para(explanation,'完整span结果尚未作为最终验收结果发布；下方v3 Native Broad仍为末尾token历史参考。'),dict(version=VERSION,complete=False,status='LOCAL_AUDIT_HOLD',full_points=progress['full_points']),{}
    if not locally_verified:
        return title+para('完整记录span对齐实验正在进行：两个任务、两个模型分别重新选头、排序和消融。直接使用主实验的完整生成记录span定义，最终回答query、评分、K网格和随机对照保持现状。下方原v3 Native Broad仍为末尾token选头的历史参考，不能当作本次对齐后的结果。','新结果将在完整审计通过后更新。'),dict(version=VERSION,complete=False),{}
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    def rows(p):
        with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    cfg=read(root/'protocol.json');audit=read(root/'analysis/audit.json')
    assert cfg['version']==VERSION and cfg['record_scope']=='registered_generated_full_record_spans'
    assert audit['status']=='PASS' and audit['checks']['points']==audit['expected_full_points']
    assert audit['checks']['complete_rankings_reproduced']==4 and audit['checks']['frozen_doses']==26
    assert audit['protocol_sha256']==sha(root/'protocol.json')==sha(run/'package/protocol.json')
    summary=rows(root/'analysis/summary.csv');tests=rows(root/'analysis/hypothesis_tests.csv');coverage=rows(root/'analysis/coverage.csv')
    names={'kth':'第 k 条记录','category':'指定类别计数'}
    def get(m,t,k,metric,pop='all_examples'):
        found=[r for r in summary if r['model']==m and r['task']==t and int(r['k'])==k and r['metric']==metric and r['population']==pop]
        assert len(found)==1;return found[0]
    best=[];data=[]
    for task in cfg['tasks']:
        for model in cfg['models']:
            k=min(cfg['sizes'][model],key=lambda k:(-float(get(model,task,k,'delta')['mean']),k))
            best.append(dict(task=task,model=model,k=k))
            p=next(r for r in tests if r['model']==model and r['task']==task and int(r['k'])==k and r['population']=='all_examples')
            data.append([names[task]+' / '+model,k,get(model,task,k,'delta')['n'],percent(get(model,task,k,'clean')),
                percent(get(model,task,k,'selected')),percent(get(model,task,k,'random')),ci(get(model,task,k,'delta')),
                ci(get(model,task,k,'delta_gain_vs_end_token')),f"{float(p['p_holm']):.4f}"])
    part=title+para('本次直接调用主实验的记录解析、literal token边界、完整可见记录span及Broad分数函数（M×exp(H)/J，epsilon=1e-12）。Native Broad的key由单个记录末尾token改为整个已注册生成记录的原始token span；category不额外按所问类别筛选自然轨迹。', '已对齐Native Broad的完整记录span定义；任务自己的最终回答query、准确率评分、K展示和随机对照方案保留。')
    part+=para(f"1200条Native plans的完整记录边界已核验，4套任务自己的discovery排序和全部26个剂量已冻结并重新执行；{audit['checks']['points']}个case×K、{audit['checks']['arms_rescored']}条条件输出重评分通过。表中K取合并主分析Random−Selected的观察峰值，并列取最小K。",'K仍为confirmation上的事后探索性选择，逐点区间未校正选K。')
    part+=table(['任务 / 模型','K','有效 n','Clean %','Selected %','Random %','Δ pp [95% CI]','相对末尾token版的Δ增量 pp [95% CI]','Holm p'],data).replace('<div class="tablewrap">','<div id="native-full-span-best" class="tablewrap">',1)
    if all(float(r['mean'])==0 for r in summary if r['population']=='all_examples' and r['metric']=='delta'):
        part+=para('四组在全部扫描K下，Clean、Selected和Random的最终答案准确率相同，Δ均为0；本次未观察到Native-thinking Broad的准确率效应。所有K并列，表中按既定规则显示最小K=1，不能据此认定K=1更优。','结论限于本次完整span选头、最终回答query和有效样本；零效应的原因尚未验证。')
    for effect in [False,True]:
        fig,axs=plt.subplots(2,2,figsize=(11.5,7.2))
        for i,task in enumerate(cfg['tasks']):
            for j,model in enumerate(cfg['models']):
                ax=axs[i,j];ks=cfg['sizes'][model]
                for metric,color in ([('delta','#c66a33')] if effect else [('clean','#688070'),('selected','#c66a33'),('random','#2563a6')]):
                    rr=[get(model,task,k,metric) for k in ks]
                    y,lo,hi=[[100*float(r[f]) for r in rr] for f in ['mean','lower','upper']]
                    ax.plot(ks,y,'-o',lw=1.7,ms=3.5,color=color,label='Full record spans' if effect else metric.title())
                    ax.fill_between(ks,lo,hi,color=color,alpha=.12)
                if effect:
                    ax.plot(ks,[100*float(get(model,task,k,'old_end_token_delta')['mean']) for k in ks],'--s',lw=1.4,ms=3,color='#2563a6',label='End tokens (matched cases)')
                    ax.axhline(0,color='#777',ls=':',lw=.8)
                else:ax.set_ylim(-2,102)
                if model.startswith('Qwen'):ax.set_xscale('log',base=2)
                ax.set_xticks(ks,[str(k) for k in ks]);ax.set_xlabel('Number of ablated heads K')
                if j==0:ax.set_ylabel('Random - Selected (pp)' if effect else 'Final-answer accuracy (%)')
                ax.set_title(model+' / '+task+f" / n={get(model,task,ks[0],'delta')['n']}")
                ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout()
        caption='完整记录span Native Broad扫描。行分别为kth与category，列分别为Qwen和Gemma；各面板标出有效输入数，按10个confirmation seed等权。横轴为消融头数K（Qwen为log₂刻度），全部K使用相同有效端点，包含Clean错误样本。'
        caption+=('纵轴为Random−Selected（百分点，正值表示Selected损伤更大）；橙线为完整span新结果，蓝色虚线为末尾token旧结果在同一批有效输入上的重算值。阴影为新结果seed配对bootstrap逐点95%区间，新旧Δ增量区间见表。' if effect else '纵轴为最终答案准确率（%），绿为Clean、橙为Selected、蓝为三个Random均值；阴影为seed配对bootstrap逐点95%区间，解析失败计错。')
        part+=report.figure(fig,'native_full_span_'+('effect' if effect else 'accuracy'),caption)
    support=[]
    for task in cfg['tasks']:
        for model in cfg['models']:
            rs=[r for r in coverage if r['task']==task and r['model']==model]
            support.append([names[task],model,len(rs),sum(r['available']=='True' for r in rs),'; '.join(sorted({r['reason'] for r in rs if r['reason']})) or '无'])
    part+='<details><summary>完整span覆盖率与不可用原因</summary>'+table(['任务','模型','候选','有效','不可用原因'],support)+para('缺失原始最终答案query的输入沿用旧版不可用状态；无可精确映射的完整span时单独记录。有效样本由几何阶段确定，不根据消融结果选择；新旧比较仅使用相同有效输入和K。')+'</details>'
    sig=sum(float(r['p_holm'])<.05 for r in tests if r['population']=='all_examples')
    part+=para(f'本次Native Broad主分析跨任务、模型及全部26个K做Holm校正，{sig}个比较达到p<0.05；Clean-correct独立成族。原v3 Broad族含52项，目标类别Non-thinking变体含13项，校正p属于不同的比较族。','依据逐任务效应、配对增量区间及校正p解释结果；样本和确认集划分曾用于旧分析。')
    numeric=audit['numeric_comparison']
    if accepted_exception:
        numeric=exception['local_numeric_comparison']
        part+='<details><summary>复核信息与已确认的数值验收例外</summary>'+para(f"GPU原环境精确复算通过；本地4套完整头排序、所有K与随机对照、8340条输出重评分、Clean基线和六份分析表逐字节一致。本地{numeric['scores_compared']}个分数中2个相差9 ULP，超过原定8 ULP；原失败记录和阈值保留。只读对照已定位到AVX512数学执行路径，两个差异均未改变头排序。",'用户已确认以GPU精确复算及本地完整排序、输出和六表一致接受本次数值例外；本地原始审计仍标记NUMERIC_HOLD，不伪记为原阈值通过。')+'</details>'
    else:
        part+='<details><summary>复核信息</summary>'+para(f"完整头顺序、所有K和随机对照、原始输出及Clean基线一致性均通过。本地分数复核模式{numeric['mode']}，最大ULP距离{numeric['max_ulp_distance']}，最大绝对差{numeric['max_absolute_difference']:.3g}；允许最多8个binary64间隔，零值及完整头顺序仍须精确一致。")+'</details>'
    files=[root/'protocol.json']+[root/'analysis'/f for f in ['audit.json','summary.csv','hypothesis_tests.csv','per_case.csv','coverage.csv','geometry.csv']]
    if accepted_exception:files += [exception_path,run/'numeric_recovery/local_analysis/audit.json']
    metadata=dict(version=VERSION,complete=True,record_scope=cfg['record_scope'],summary=summary,tests=tests,coverage=coverage,best_k=best,points=audit['checks']['points'])
    if accepted_exception:metadata['accepted_numeric_exception']=exception
    return part,metadata,{str(p.relative_to(B)):sha(p) for p in files}
