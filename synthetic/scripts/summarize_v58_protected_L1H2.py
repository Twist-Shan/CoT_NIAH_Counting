"""Reproduce protected-control summaries and the diagnostic report."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    args=p.parse_args()
    root=args.input
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for name,h in manifest['files'].items():
        assert digest(root/name)==h,name
    records=[]
    metrics=['ar_accuracy','ar_answered','eos_reached','answer_and_eos_correct',
             'normalized_count_shift','absolute_count_shift']
    for path in sorted((root/'nonthinking/arms').glob('*.csv')):
        d=pd.read_csv(path)
        assert len(d)==100 and d.key.nunique()==100
        r={c:d[c].iloc[0] for c in ['arm','top_k','repeat','heads','overlap']}
        assert 'L1H2' not in r['heads'].split(';')
        r.update({m:d[m].mean() for m in metrics})
        r.update({m+'_n':int(d[m].notna().sum()) for m in metrics})
        records.append(r)
    d=pd.DataFrame(records)
    assert len(d)==352
    assert d.ar_answered.eq(1).all()
    d.to_csv(root/'condition_summary.csv',index=False)
    rows=[]
    for k,g in d.groupby('top_k'):
        s=g[g.arm=='selected'].iloc[0]
        c=g[g.arm=='control']
        for m in metrics:
            rows.append(dict(top_k=k,metric=m,selected=s[m],control_mean=c[m].mean(),
                             control_min=c[m].min(),control_max=c[m].max(),
                             controls=len(c),overlap=int(c.overlap.iloc[0]),inputs=100))
    curves=pd.DataFrame(rows)
    curves.to_csv(root/'selected_vs_controls.csv',index=False)
    accuracy=curves[curves.metric=='ar_accuracy']
    shift=curves[curves.metric=='normalized_count_shift']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelsize':14,
                         'axes.titlesize':16,'xtick.labelsize':12,'ytick.labelsize':12,
                         'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
                         'axes.spines.right':False,'axes.edgecolor':'#161923',
                         'text.color':'#161923','axes.labelcolor':'#161923'})
    fig,axes=plt.subplots(1,2,figsize=(9.6,4.15))
    for ax,g,title,label,limits,baseline in [
        (axes[0],accuracy,'A. Count accuracy','Accuracy (%)',(-1,31),21),
        (axes[1],shift,'B. Count displacement','Normalized count shift (%)',(-4,114),0)]:
        ax.fill_between(g.top_k,100*g.control_min,100*g.control_max,color='#8190A5',alpha=.18,linewidth=0)
        ax.plot(g.top_k,100*g.selected,'o-',color='#B52F6B',linewidth=2.2,markersize=5,label='Selected')
        ax.plot(g.top_k,100*g.control_mean,'s--',color='#8190A5',linewidth=2,markersize=4,label='Control mean')
        ax.axhline(baseline,color='#161923',linestyle=':',linewidth=1.6,label='Clean')
        ax.set(title=title,xlabel='Top-k',ylabel=label,ylim=limits,xticks=range(1,9))
        ax.set_axisbelow(True)
        ax.grid(axis='y',color='#E3E4EA',linewidth=.8)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.015))
    fig.subplots_adjust(left=.09,right=.985,bottom=.23,top=.85,wspace=.35)
    for ext in ('pdf','svg','png'):
        fig.savefig(root/f'protected_L1H2_ablation.{ext}',dpi=190,facecolor='white')
    plt.close(fig)
    table='\n'.join(f'| {int(r.top_k)} | {r.selected*100:.2f} | {r.control_mean*100:.2f} | {r.control_min*100:.0f}–{r.control_max*100:.0f} | {int(r.controls)} | {int(r.overlap)} |' for r in accuracy.itertuples())
    shift_table='\n'.join(f'| {int(r.top_k)} | {r.selected*100:.2f} | {r.control_mean*100:.2f} |' for r in shift.itertuples())
    eos=curves[curves.metric=='eos_reached']
    eos_table='\n'.join(f'| {int(r.top_k)} | {r.selected*100:.2f} | {r.control_mean*100:.2f} |' for r in eos.itertuples())
    report=f'''# Non-thinking：L1H2 功能与保护该头的 control

L1H2 在原始答案查询处读取背景字符 `o`，提供近似固定的向量。移除该向量会严重干扰数字答案读出。按照用户要求，新 control 全部保留 L1H2；原始 control 和所有失败输出均保留。新结果中每个条件的数字可解析率均为 100%，Top-3/4 selected 的计数损伤更大，但全 Top-k 范围并不一致。

## L1H2 的作用：直接证据

功能探针位于相邻目录 `v58_L1H2_function_20260908_v2`；完整零消融/平均替换诊断位于 `v58_nonthinking_readout_diagnosis_20260908`。

1. 200 discovery + 100 confirmation 输入全部检查。答案查询时，L1H2 对背景字符 `o` 的平均 attention mass 为 99.9995%；对实际 target positions 的平均 mass 约为 2.6e-8（confirmation）。
2. 第一层的 value 来自 token embedding 和 LayerNorm，RoPE 仅用于 Q/K。因此同一个 `o` 在不同位置具有相同 value。无论 attention 分配到哪个 `o`，只要总质量接近 1，头的加权 value 就接近同一个固定向量。
3. discovery 上 L1H2 的 pre-O 向量 RMS 范数为 14.5688，去均值后的 RMS 仅约 0.000034。近似不变的输出不提供可观的逐输入计数变化。
4. 将 L1H2 置零：100/100 输入的首 token 变成 `<Ans>`，数字 token 的概率总质量从 99.9543% 降至 0.0415%。保留 full-vocabulary greedy，不跳过或过滤这些失败。
5. 用 discovery 均值替换 L1H2：100/100 首 token 与 clean 一致，准确率仍为 21%。更直接地，用 checkpoint 单独算出的固定 `V(o)` 替换它，也复现全部 clean 首 token；十个数字的 logit 最大绝对差仅 0.000103。
6. 原 Top-4 control 为 L1H1/H2/H5/H7。只恢复 H2、其余三个继续消融时，数字可解析率由 0% 恢复为 100%，准确率由 0% 恢复为 19%。分别恢复另外三个头均没有这个效果。这四个条件来自原全量 sweep，未挑选新的成功样本。

这些证据支持 **答案查询处的近似常量读出分量** 这一解释。它的作用也影响后续数字 logits：置零后即使只在 1–10 中取 argmax，准确率仍由 21% 降至 12%。因此不能说它“只影响格式”，也不应把它的损伤直接解释为丢失了该头携带的 count 信息。结论限定于这里测试的答案查询，未证明它在所有 token 位置上的功能相同。

## 新 control 规则与完整结果

- 保留 frozen discovery 的全 32 头 broad ranking，selected Top-1 至 Top-8 不重排。
- control 中排除 L1H2；每层头数仍与 selected 相同。优先完全不重叠，只有该层可用头不足时才使用最少重叠；枚举全部合法组合，不根据结果进一步删头或删组合。
- 从原始 `<Ans>` 查询起持续消融 pre-O head slices，直至 EOS 或 64 个新 token。使用 full-vocabulary greedy 和原始严格数字 parser。全部 100 confirmation 输入保留。
- 这是检查原 control 损伤后新增的诊断性分析；保护头规则是事后修订，不能写成预先注册的独立确认。固定 L1H2 后，比较的问题也相应限定为“保留该读出分量时，其他头的计数作用”。
- 共 344 个 control + 8 个 selected，连同 clean 为 35,300 条轨迹。复用完全相同 head set 的 28 个旧条件，新跑 32,400 条轨迹。

Clean accuracy = 21%。以下为计数准确率，越低表示对正确计数损伤越大。

| K | Selected (%) | Control mean (%) | Control range (%) | 全部 control 数 | 必要重叠数 |
|---|---:|---:|---:|---:|---:|
{table}

**Top-3 最易解释**：selected 8% vs controls 16.75%，4 个 control 都完全不重叠。Top-4 为 7% vs 15%，每个 control 必须与 selected 重叠 1 个头。较大的 K 接近低准确率区间，且重叠更高；Top-1/2/8 并不支持 selected 总是更重要的说法。以上均为描述性结果，不进行显著性检验。

与主实验相同的 normalized count shift 为 `abs(ablated_count - clean_count) / true_count`。它衡量对 clean 输出的改变，不等同于对正确答案的损伤。这次所有数字都可解析，所以每个条件均使用完整的 100 个输入。

| K | Selected shift (%) | Control mean shift (%) |
|---|---:|---:|
{shift_table}

Top-3/4 的 count shift 同样高于 controls。Top-5 至 Top-8 的 selected shift 则没有超过 control 均值，不能将 Top-3/4 的结果外推到全部 K。

## 数字合法与终止行为分开

所有 selected/control 条件的数字可解析率均为 100%。这说明重复 `<Ans>` 导致的原始数字解析崩溃已消除；完整生成仍有停止异常，不能将“数字可解析”写成“全部输出格式正确”。EOS 到达率如下。

| K | Selected EOS (%) | Control mean EOS (%) |
|---|---:|---:|
{eos_table}

准确率沿用原 protocol，只判断 `<Ans>` 后的原子数字，不要求 EOS。停止失败继续保存于原始 CSV，不通过事后过滤修饰数字准确率。

## 其他诊断与验证

原始 496 个 Top-k 条件还进行了原查询 logits 复验和 discovery mean ablation，外加所有 32 个单头与 L1H2 缩放曲线。共 104,900 个 readout 记录。原 zero 条件首 token 与原始持续消融逐条一致；平均替换使全部 Top-k 条件恢复 100% 数字首 token，但 selected/control 的差异依 K 而变，不能只保留其中较有利的条件。Mean replacement 的这轮结果只覆盖原查询，尚未评估随后整个生成过程。

新 protected controls 在 Top-4/8 上通过完整前缀与 KV-cache 解码的逐 token 一致性检查，并检查了实际 pre-O 消融切片和未选切片。控制集合的所有 K 都通过匹配、唯一性、最小重叠和保护 H2 的检查。下载后的所有原始文件 SHA-256 与服务器 manifest 一致。

`protected_L1H2_ablation.pdf/png/svg` 只呈现本次持续消融的两个计数指标。灰色阴影为全部 control 的最小到最大值，不是置信区间。准确率与 displacement 使用不同纵轴。

Paper_Draft/appendix/synthetic.tex 只新增一个 `\\GPT{{...}}`，说明 L1H2 机制及 control 修订。全文编译完成，新增段落第 17 页已检查；原文已有一个空引用警告，本次未改动。
'''
    (root/'RESULTS.md').write_text(report,encoding='utf-8')
    derived=['condition_summary.csv','selected_vs_controls.csv','RESULTS.md']+[f'protected_L1H2_ablation.{e}' for e in ['pdf','png','svg']]
    (root/'summary_manifest.json').write_text(json.dumps({'script_sha256':digest(Path(__file__)),
        'raw_manifest_sha256':digest(root/'manifest.json'),
        'files':{f:digest(root/f) for f in derived}},indent=2),encoding='utf-8')
    print('PROTECTED CONTROL SUMMARY COMPLETE')
    print(accuracy.to_string(index=False))


if __name__=='__main__':
    main()
