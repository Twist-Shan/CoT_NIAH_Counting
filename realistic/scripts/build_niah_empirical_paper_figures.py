"""Paper figure drafts from existing empirical-law fits; no refitting or filtering."""
from pathlib import Path
import base64
import hashlib
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import build_niah_empirical_law_v3_2_report as v32
from build_niah_all_n_length_comparison import load_results, MODE_RULE, MODELS, MODES

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/assets/niah_empirical_paper'
LONG=ROOT/'reports/assets/niah_empirical_all_n_length'
SHORT=ROOT/'outputs/anvil_realistic_niah_v3_1_20260819_formal/analysis/v3_2_inverse_n_candidate_extension/tables'


def save(fig,stem):
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'{stem}.{ext}',dpi=240,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def axis_style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.14,linewidth=.5)
    ax.set_ylim(-.025,1.025)
    ax.set_yticks([0,.25,.5,.75,1])


def count_figure():
    cells=pd.read_csv(SHORT/'cell_outcomes.csv.gz')
    coefficients=pd.read_csv(SHORT/'selected_model_coefficients.csv')
    selected=pd.read_csv(SHORT/'selected_mode_laws.csv')
    assert cells.comparison_slot.nunique()==12
    levels=sorted(cells.N.unique());lengths=sorted(cells.L.unique())
    assert len(levels)==14 and len(lengths)==8
    colors=plt.colormaps['plasma_r'](np.linspace(.10,.95,len(lengths)))
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.35),sharex=True,sharey=True)
    records=[]
    for i,(ax,mode) in enumerate(zip(axes,MODES)):
        fitted=v32.predictions_by_slot(coefficients,mode,'accuracy_bernoulli_logit',levels,lengths)
        assert fitted.shape==(12,14,8) and np.isfinite(fitted).all()
        for j,(length,color) in enumerate(zip(lengths,colors)):
            observed=cells.loc[cells.prompt_mode.eq(mode)&cells.L.eq(length)].pivot(index='comparison_slot',columns='N',values='parsed_exact_accuracy').reindex(index=v32.SLOT_ORDER,columns=levels).to_numpy()
            assert observed.shape==(12,14) and np.isfinite(observed).all()
            oq=np.quantile(observed,[.25,.5,.75],axis=0);fq=np.quantile(fitted[:,:,j],[.25,.5,.75],axis=0)
            ax.fill_between(levels,fq[0],fq[2],color=color,alpha=.055,linewidth=0)
            ax.plot(levels,fq[1],color=color,lw=1.15)
            ax.errorbar(levels,oq[1],yerr=[oq[1]-oq[0],oq[2]-oq[1]],fmt='o',ms=2.0,mfc='white',mec=color,mew=.5,
                        ecolor=color,elinewidth=.4,capsize=.7,alpha=.65,zorder=3)
            records.extend(dict(mode=mode,N=int(n),L=int(length),observed_q25=float(oq[0,k]),observed_median=float(oq[1,k]),
                observed_q75=float(oq[2,k]),fitted_q25=float(fq[0,k]),fitted_median=float(fq[1,k]),fitted_q75=float(fq[2,k])) for k,n in enumerate(levels))
        ax.set_title(f"({chr(97+i)}) {'Non-thinking' if mode=='direct' else 'Native-thinking'}",loc='left',fontsize=9.5,fontweight='bold')
        ax.set_xscale('log',base=2);ax.set_xlim(.94,21)
        ticks=[1,2,3,4,5,7,10,15,20]
        ax.set_xticks(ticks);ax.set_xticklabels([str(x) for x in ticks]);ax.minorticks_off()
        ax.set_xlabel('Target count N (log scale)');axis_style(ax)
    axes[0].set_ylabel('Probability of a correct count')
    handles=[Line2D([],[],color=color,lw=1.6,label=f'{length//1000}k') for length,color in zip(lengths,colors)]
    fig.legend(handles=handles,title='Passage length (tokens)',loc='upper center',bbox_to_anchor=(.51,1.00),ncol=8,
               frameon=False,fontsize=7,title_fontsize=7.5,handlelength=1.7,columnspacing=1.0)
    fig.subplots_adjust(left=.085,right=.995,top=.78,bottom=.15,wspace=.12)
    save(fig,'figure1_count_across_slots')
    pd.DataFrame(records).to_csv(OUT/'figure1_plotted_quantiles.csv',index=False)
    selected.loc[selected.outcome_family.eq('accuracy_bernoulli_logit')&selected.prompt_mode.isin(MODES)].to_csv(OUT/'figure1_selected_laws.csv',index=False)
    return records


def length_panel(ax,model,data,curves,colors,levels,modes=MODES):
    cells=data['cell_predictions']
    for n,color in zip(levels,colors):
        for mode in modes:
            term=MODE_RULE[mode]
            g=curves.loc[curves.model.eq(model)&curves['mode'].eq(mode)&curves.N.eq(n)].sort_values('L')
            assert len(g)==250 and g.length_term.eq(term).all()
            observed=cells.loc[cells.model.eq(model)&cells['mode'].eq(mode)&cells.length_term.eq(term)&cells.N.eq(n)].sort_values('L')
            assert len(observed)==17
            ax.plot(g.L/1000,g.prediction,color=color,lw=1.10,ls='-' if mode=='direct' else (0,(3,1.8)),zorder=2)
            ax.scatter(observed.L/1000,observed.observed,s=6,marker='o' if mode=='direct' else '^',facecolor='none',edgecolor=color,lw=.42,alpha=.38,zorder=3)
    ax.set_xscale('log');ax.set_xlim(.94,106)
    ax.set_xticks([1,2,5,10,20,50,100]);ax.set_xticklabels(['1','2','5','10','20','50','100']);ax.minorticks_off()
    ax.axvline(20,color='#8E99AA',ls=':',lw=.65)
    ax.set_xlabel('Passage length (k tokens; log scale)');axis_style(ax)


def length_figure(data,curves,models=MODELS,stem='figure2_length_two_models'):
    levels=sorted(data['cell_predictions'].N.unique())
    colors=plt.colormaps['plasma_r'](np.linspace(.12,.95,len(levels)))
    fig,axes=plt.subplots(1,len(models),figsize=(7.4 if len(models)==2 else 4.35,3.55),sharex=True,sharey=True,squeeze=False)
    for i,(ax,model) in enumerate(zip(axes.flat,models)):
        length_panel(ax,model,data,curves,colors,levels)
        ax.set_title(f'({chr(97+i)}) {model}',loc='left',fontsize=9.5,fontweight='bold')
    axes[0,0].set_ylabel('Probability of a correct count')
    handles=[Line2D([],[],color='#313847',ls='-',lw=1.3,marker='o',mfc='none',ms=3,label='Non-thinking: linear L'),
             Line2D([],[],color='#313847',ls=(0,(3,1.8)),lw=1.3,marker='^',mfc='none',ms=3,label='Native-thinking: log L')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.48,1.00),ncol=2,frameon=False,fontsize=7.2,handlelength=2.7,columnspacing=1.5)
    cax=fig.add_axes([.928,.23,.016,.52]);cmap=ListedColormap(colors);norm=BoundaryNorm(np.arange(len(levels)+1)-.5,len(levels))
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,ticks=np.arange(len(levels)))
    bar.ax.set_yticklabels([str(n) for n in levels]);bar.ax.tick_params(labelsize=6.2,length=2)
    bar.ax.set_title('N',fontsize=8,pad=7)
    note='Common-form illustration; Qwen Non-thinking favors log L in the paired comparison.' if 'Qwen3-32B' in models else 'Common-form illustration; these forms also minimize CV log loss for Gemma.'
    fig.text(.085,.015,note,fontsize=6.7,color='#565E6B')
    fig.subplots_adjust(left=.085 if len(models)==2 else .145,right=.90,top=.82,bottom=.18,wspace=.13)
    save(fig,stem)


def write_preview(data):
    table=pd.read_csv(LONG/'comparison_summary.csv')
    summary=[]
    for _,r in table.iterrows():
        summary.append({'Model':r.model,'Mode':'Non-thinking' if r['mode']=='direct' else 'Native-thinking',
            'Linear: R²':f'{r.linear_cell_R2:.3f}','Log: R²':f'{r.log_cell_R2:.3f}',
            'Linear: CV log loss':f'{r.linear_CV_log_loss:.4f}','Log: CV log loss':f'{r.log_CV_log_loss:.4f}'})
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'paper_length_comparison.csv',index=False)
    fig1='Counting performance across 12 comparison slots. Each panel shows one prompting mode over 14 target counts and eight passage lengths (1k–20k tokens). Open markers and error bars show the median and interquartile range of observed accuracy across slots; curves and ribbons show the corresponding summaries of independently fitted slot-specific predictions. Ribbons describe variation across slots. Parsing failures count as incorrect responses. The selected predictor bases are ln N + L/1000 for Non-thinking and N + ln(L/1000) for Native-thinking; coefficients are estimated separately for each slot.'
    fig2='Length dependence in Gemma4-31B and Qwen3-32B. Each panel overlays both prompting modes; color denotes target count N, solid lines and circles denote Non-thinking, and dashed lines and triangles denote Native-thinking. Curves use logit p = alpha_N + beta f(L), with f(L) = L/1000 for Non-thinking and ln(L/1000) for Native-thinking. Each model and mode has 14 N-specific intercepts and one shared length coefficient. All 14 N values and 17 lengths (1k–100k tokens) are included, with 30 requests per condition. These curves are full-range refits. The vertical dotted line marks 20k tokens. The displayed forms provide a common descriptive approximation; Qwen Non-thinking achieves 2.15% lower cross-validated log loss with log length. Overall R2 includes variation attributable to both N and L.'
    notes=f'''建议正文保留两张图和一张简短比较表。

图 1 回答：两种模式的正确率怎样随 N 和 L 变化？保留跨 12 槽汇总，把逐槽图放入附录。误差条和阴影都表示槽间四分位范围，不是 95% 置信区间。

图 2 回答：允许各 N 有不同起点后，两种模式的长度曲线是什么形状？每个模型一个面板，同一面板叠加两种模式。保留全部 N。该图使用完整 1k–100k 数据重新拟合。

正文比较表保留每组线性和对数的 CV log loss；可以同时报告样本内 cell R2。函数形式的比较使用同组的 CV log loss。

建议正文结论：Native-thinking 的对数长度形式在 V3.2 的 12 个槽中均优于线性形式，并在两个长程模型中保持优势。Non-thinking 更依赖模型；线性形式可作为两模型长程结果的统一近似，Qwen 的最佳形式仍为对数。

图 1 使用 N 的指定函数形式；图 2 允许每个 N 有独立截距。这是两种不同约束下的分析，图 2 不验证图 1 的完整公式，也不检验短程到长程的外推。

附录保留：12 槽逐模型曲线、线性/对数全部比较、逐 N 残差与曲线、冻结短程系数的长程外推、MAE/bias、分段、无截距及交叉项敏感性。Non-thinking broad retrieval 与 Native-thinking trace 的机制解释放入 Discussion，明确其与长度函数形状的对应关系仍待验证。

Figure 1 caption:
{fig1}

Figure 2 caption:
{fig2}
'''
    (OUT/'paper_selection_notes.md').write_text(notes,encoding='utf-8')
    def image(stem):
        return 'data:image/png;base64,'+base64.b64encode((OUT/f'{stem}.png').read_bytes()).decode('ascii')
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Empirical law：论文图稿建议</title>
<style>body{{max-width:1080px;margin:36px auto;padding:0 24px;font:16px/1.7 system-ui;color:#202735}}h1{{font-size:26px}}h2{{font-size:20px;margin-top:34px}}img{{width:100%}}figcaption{{font-size:14px;color:#505c6d}}figure{{margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}details{{margin:24px 0}}</style>
<h1>建议正文保留两张图和一张比较表</h1><p>图 1 展示跨模型的整体趋势，图 2 展示两个模型在长文本上的具体变化。MAE、bias 和完整敏感性分析放入附录。</p>
<h2>图 1：目标数与文本长度的共同影响</h2><p>这是 12 个比较槽的中位数汇总。误差条和阴影表示槽间四分位范围。</p><figure><img src="{image('figure1_count_across_slots')}" alt="12 槽数量回归"><figcaption>{fig1}</figcaption></figure>
<h2>图 2：每个模型叠加两种模式</h2><p>左图 Gemma，右图 Qwen；颜色表示 N，实线/圆点为 Non-thinking，虚线/三角为 Native-thinking。保留全部 N 与长度。</p><figure><img src="{image('figure2_length_two_models')}" alt="两模型叠加两种模式的长度回归"><figcaption>{fig2}</figcaption></figure>
<h2>正文比较表</h2><p>R² 衡量完整拟合与观测条件的贴合程度；CV log loss 衡量留出条件预测，越低越好。Qwen Non-thinking 的例外在正文保留。</p>{summary.to_html(index=False,border=0)}
<p><strong>适用范围：</strong>图 1 使用 N 的指定函数形式；图 2 为每个 N 单独估计截距。图 2 使用整个长度范围重新拟合，不能当作图 1 原公式的长程验证。</p>
<h2>建议放入附录</h2><p>12 槽逐模型图、逐 N 拟合诊断、线性与对数的完整比较、冻结短程系数的外推结果、MAE/bias、分段、无截距和交叉项敏感性。</p>
<p>关于 broad retrieval 和 trace 的解释放入 Discussion；目前的曲线比较尚不能确认对应的因果机制。</p></html>'''
    (ROOT/'reports/NiaH_Empirical-law_paper_preview.html').write_text(page,encoding='utf-8')


def main():
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6})
    data,_=load_results();curves=pd.read_csv(LONG/'mode_rule_plotted_predictions.csv.gz')
    source_manifest=json.loads((LONG/'comparison_manifest.json').read_text(encoding='utf-8'))
    for name,digest in source_manifest['input_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    assert len(curves)==14000
    count_records=count_figure()
    length_figure(data,curves)
    for model in MODELS:length_figure(data,curves,models=(model,),stem=f'{model}_length_overlay')
    curves.to_csv(OUT/'figure2_plotted_predictions.csv.gz',index=False,compression='gzip')
    data['cell_predictions'].loc[data['cell_predictions'].apply(lambda r:r.length_term==MODE_RULE[r['mode']],axis=1)].to_csv(OUT/'figure2_observed_cells.csv',index=False)
    write_preview(data)
    inputs=[SHORT/'cell_outcomes.csv.gz',SHORT/'selected_model_coefficients.csv',SHORT/'selected_mode_laws.csv',
            LONG/'mode_rule_plotted_predictions.csv.gz',LONG/'comparison_summary.csv',LONG/'comparison_manifest.json',Path(__file__)]
    manifest=dict(figure1_slots=12,figure1_conditions=len(count_records),figure2_models=2,figure2_modes=2,
        figure2_N_levels=14,figure2_lengths=17,figure2_conditions=952,figure2_curve_points=len(curves),refits=0,
        figure1_spread='interquartile range across model comparison slots',figure2_form=MODE_RULE,
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name not in {'build_manifest.json','validation_manifest.json'}},
        elapsed_seconds=time.perf_counter()-start)
    (OUT/'build_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(OUT)


if __name__=='__main__':main()
