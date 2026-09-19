"""Render audited measured dose curves; never interpolate missing experiments."""
import csv,json
import numpy as np
import matplotlib.pyplot as plt

def build_section(base,para,figure,svg,table):
    folder=base/'runs/topk_completion_20260907_v1/analysis'
    if not (folder/'audit.json').exists():return None
    assert json.loads((folder/'audit.json').read_text())['status']=='PASS'
    with (folder/'summary.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    models=['Qwen3-8B','Gemma4-E4B'];assays=[('nonthinking','broad','Non-thinking Broad'),('native_thinking','broad','Native-thinking Broad'),('native_thinking','targeted','Native-thinking Targeted')]
    section='<section id="topk"><h2>6. Top-K ablation：两任务的完整剂量扫描</h2>'
    section+=para('目的：检验消融更多高排名检索头时，正确率下降是否超过同数量随机头。保持自然trace、端点、排名、评分和confirmation样本不变，仅改变头数K。说明性例子：K=2使用排名前两头，K=4包含这两头及之后两头；二者使用相同输入，各配三组同层同数量且不含selected头的随机对照。','嵌套集合与配对样本使曲线反映固定选头规则下的剂量效应。')
    section+=para('Broad排名使用200个discovery输入，按种子等权汇总注意力指标并限制每层不超过半数头。Targeted迁移正文shared冻结排名；Gemma原六头扩展为正文八头集合。曲线仅评价10个confirmation seeds，每个seed10条原始输入，不按正确与否筛选；不可用端点记入覆盖表。每个剂量使用相同clean输出。相同端点与头列表的旧干预经哈希和评分核验后复用。','正文标准中的嵌套剂量、discovery选头与confirmation评价、三个随机对照均用于本次扫描；未按新任务表现调整Targeted排名。')
    section+=para('本次保留additional原协议：Broad只消融最终答案query，最多续写64 tokens；Targeted消融当前query及后续decode，最多128 tokens。Qwen的rank-before端点保留post-marker，其他情况及Gemma使用P0。正文部分Targeted路线统一从P0起、Gemma shared配置为256 tokens；本次未改变这些执行细节。','本次直接检验现有additional结果对头数的敏感性；不能把差异归因于与正文完全相同的时间窗口。')
    section+=para('正确率先在每个seed内对有效样本取平均，再对10个seed等权平均。Random先对三个随机条件取平均。Δ(K)=Random正确率−Selected正确率，以百分点表示；正值表示选定头损伤更大。95%区间由种子配对bootstrap重采样20000次计算，所有K使用相同重采样索引。','Δ区间完全高于0支持该剂量下选定头的特异性损伤；跨剂量寻找最大值属于追加敏感性分析，未作多重比较校正。')
    for ti,task in enumerate(['kth','category']):
        title='寻找第k条记录' if task=='kth' else '混合city/flower计数'
        section+=f'<h3>6.{ti+1} {title}</h3>'
        def get(model,mode,assay,metric):return sorted([x for x in rows if (x['task'],x['model'],x['mode'],x['assay'],x['metric'])==(task,model,mode,assay,metric)],key=lambda x:int(x['k']))
        for effect in [False,True]:
            fig,axs=plt.subplots(3,2,figsize=(12,11),squeeze=False)
            for i,(mode,assay,label) in enumerate(assays):
                for j,model in enumerate(models):
                    ax=axs[i,j]
                    metrics=[('delta','#7d4caa','Random − Selected')] if effect else [('clean','#677787','Clean'),('selected','#2563a6','Selected'),('random','#c66a33','Random')]
                    for metric,col,legend in metrics:
                        ss=get(model,mode,assay,metric);assert ss
                        xx=[int(s['k']) for s in ss];yy=[100*float(s['mean']) for s in ss]
                        ax.plot(xx,yy,'--' if metric=='clean' else 'o-',color=col,label=legend,lw=1.7,ms=4)
                        if metric!='clean':ax.fill_between(xx,[100*float(s['lower']) for s in ss],[100*float(s['upper']) for s in ss],color=col,alpha=.15)
                    ax.set(title=f'{model} · {label}',xlabel='Number of ablated heads K',ylabel='Extra failure (percentage points)' if effect else 'Correctness (%)',xticks=xx)
                    if model=='Qwen3-8B' and assay=='broad':
                        ax.set_xscale('log',base=2)
                        ax.set_xticks(xx,labels=[str(k) for k in xx])
                        ax.set_xlabel('Number of ablated heads K (log2 scale)')
                    if not effect:ax.set_ylim(-3,103)
                    else:
                        ax.axhline(0,color='#777',lw=.8,ls='--')
                        ax.set_ylim([(-15,20),(-2,10),(-10,100)][i])
                    ax.tick_params(axis='x',labelsize=8,rotation=45);ax.grid(axis='y',alpha=.15);ax.legend(fontsize=8,loc='best')
            fig.tight_layout()
            figid=5+2*ti+int(effect)
            caption=f'图{figid}｜{title}：'+('配对额外失败率。纵轴Δ=Random−Selected（百分点），越高表示选定头损伤越大；紫色阴影为95%种子bootstrap区间，虚线0表示两种干预无平均差异。' if effect else '消融后正确率。蓝色Selected、橙色三个Random平均、灰色虚线Clean；阴影为各曲线95%种子bootstrap区间。纵轴越低表示损伤越大。')+'横轴是消融头数K；各行依次为Non-thinking Broad、Native-thinking Broad、Native-thinking Targeted，各列为Qwen/Gemma。Broad评分最终答案，Targeted评分短续写中的下一条语义记录。连线仅连接实际运行或严格核验复用的点，各K使用同一批有效confirmation端点。'
            caption+='Qwen Broad横轴使用以2为底的对数刻度，使1、2、4等小剂量清晰可读；其他面板使用线性刻度。'
            if effect:caption+='两任务的同类干预使用相同纵轴范围；不同行范围不同，请按刻度比较效应大小。'
            section+=figure(svg(fig),caption)
            section+=para('图中Broad与Targeted使用不同输出终点，需分别读取曲线；不能把Targeted下一条记录的正确率当作整题答案准确率。','上述图的因果解释限于各自已注册的干预位置与评分终点。')
        result=[];conclusions=[]
        for mode,assay,label in assays:
            for model in models:
                ss=get(model,mode,assay,'delta');last=ss[-1]
                supported=[int(x['k']) for x in ss if float(x['lower'])>1e-10]
                result.append([model,label,last['n'],', '.join(str(int(x['k'])) for x in ss),f"{100*float(last['mean']):.2f} [{100*float(last['lower']):.2f}, {100*float(last['upper']):.2f}]",', '.join(map(str,supported)) or '无'])
                conclusions.append(f"{model} {label}在最大K={last['k']}时Δ={100*float(last['mean']):.2f} pp，95% CI [{100*float(last['lower']):.2f}, {100*float(last['upper']):.2f}]")
        section+=table(['模型','干预','有效n/100','已测K','最大K的Δ pp [95% CI]','Δ区间下界>0的K'],result)
        section+=para('；'.join(conclusions)+'。','上表报告各最大剂量与逐点区间；只有下界高于0的剂量获得对应的随机对照差异证据，未达到这一条件不等于证明没有作用。')
        if task=='kth':
            section+=para('Qwen Targeted的Δ从K32的0.88 pp上升到K112的80.07 pp，K128为72.99 pp；K32区间包含0，K64及更大剂量的区间高于0。Gemma Targeted在K4、K6、K8的Δ分别为66.37、26.67、46.07 pp；对应Selected正确率分别为23.33%、59.44%、36.00%，所以非单调性同时出现在Selected曲线上，不能只归因于随机对照均值变化。','两模型在多个剂量支持下一记录检索的局部依赖；非单调变化的原因未查明，本实验没有识别头间补偿或交互机制。')
            section+=para('Non-thinking Broad在最大剂量下，Qwen K128的Δ为5.33 [2.00, 9.33] pp，Gemma K8为5.00 [1.67, 8.33] pp；Gemma K6的10.67 pp高于K8。Native Broad在两模型所有已测K上的平均Δ均为0。','直接回答时Broad作用有跨模型支持，但不呈一致单调剂量关系；最终答案query上的Native Broad扫描未发现相对随机对照的平均额外损伤。')
        else:
            section+=para('Qwen Targeted的Δ随K32、64、80、96、112、128依次为8.51、23.30、36.40、45.37、74.27、76.70 pp，逐点区间均高于0。Gemma K8的Δ为44.06 [32.96, 56.19] pp，Selected正确率11.11%，Random55.17%；K1、4、6、8的区间高于0，K2跨0。','混合类别任务在两模型多个剂量上支持Targeted局部依赖；Qwen曲线在已测剂量上单调，Gemma曲线不单调。')
            section+=para('Qwen Native Broad在K128时Selected90.17%、Random94.39%，Δ=4.22 [1.00, 7.44] pp；K32仅1.25 [0.00, 3.75] pp。Gemma Native Broad所有已测K的Δ均为0。Gemma Non-thinking Broad仅K1的逐点区间明确高于0；K8为2.00 [−4.00, 8.00] pp。','Qwen Native Broad的结论需从旧固定K32的弱证据更新为大剂量下存在小幅效应；混合任务Broad结果仍依赖模型和剂量。')
    with (folder/'coverage.csv').open(encoding='utf-8') as f:coverage=list(csv.DictReader(f))
    section+='<h3>6.3 覆盖与审计</h3>'
    coverage_rows=[]
    for task in ['kth','category']:
        for model in models:
            for mode,assay,label in assays:
                group=[x for x in coverage if (x['task'],x['model'],x['mode'],x['assay'])==(task,model,mode,assay)]
                available=sum(x['available']=='True' for x in group)
                coverage_rows.append([task,model,label,len(group),available,len(group)-available])
    section+=table(['任务','模型','干预','候选端点','有效端点（每个K相同）','不可用端点'],coverage_rows)
    section+=para('审计覆盖6739个样本×剂量点和33695条条件记录，全部重新评分一致；100个候选端点不可用。7839条条件记录复用严格匹配的旧结果（包括共享clean的重复引用）；25856条为新生成的干预记录。另有24个canary剂量点。8086条条件记录达到生成长度上限，该数量按剂量计，包含同一clean的重复引用；它不等于8086个独立样本或评分失败。','运行、覆盖、选头、随机对照、评分与源文件哈希均通过核验；报告保留生成上限与局部评分范围。')
    section+=para('每个输入的有效性、每个剂量的生成文本、干预头和随机头列表、评分及复用出处均保留。统计源为runs/topk_completion_20260907_v1/analysis/{summary.csv,per_case.csv,coverage.csv,audit.json}，协议为downloaded/protocol.json。历史Qwen K256保留在原扫描文件中，本次按用户要求最多128头。','两任务、两模型的三类干预现均有真实多剂量曲线；当前结论以配对正确率及其区间为依据。')
    return section+'</section>'
