"""Paper figure drafts from existing empirical-law fits; no refitting or filtering."""
from pathlib import Path
import argparse
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
import build_niah_all_n_length_comparison as length_comparison
from build_niah_all_n_length_comparison import load_results, MODE_RULE, MODELS, MODES

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/assets/niah_empirical_paper'
LONG=ROOT/'reports/assets/niah_empirical_all_n_length'
SHORT=ROOT/'outputs/realistic_niah_v3_1_20260819_formal/analysis/v3_2_inverse_n_candidate_extension/tables'


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




def main():
    global OUT, LONG, SHORT, PREVIEW
    parser=argparse.ArgumentParser(description=__doc__)
    length_comparison.add_input_arguments(parser)
    parser.add_argument('--short-tables',type=Path,help='Short-context cell outcomes and selected-law coefficient tables')
    parser.add_argument('--length-comparison-dir',type=Path,help='Outputs of build_niah_all_n_length_comparison.py')
    parser.add_argument('--output-dir',type=Path,help='Directory for figure files and build manifest')
    args=parser.parse_args()
    if args.short_tables is not None:SHORT=args.short_tables.resolve()
    if args.length_comparison_dir is not None:LONG=args.length_comparison_dir.resolve()
    if args.output_dir is not None:OUT=args.output_dir.resolve()
    length_comparison.configure_inputs(linear_dir=args.linear_dir,log_dir=args.log_dir,
        holdout_metrics=args.holdout_metrics,short_range_metrics=args.short_range_metrics)
    start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6})
    data,_=load_results();curves=pd.read_csv(LONG/'mode_rule_plotted_predictions.csv.gz')
    source_manifest=json.loads((LONG/'comparison_manifest.json').read_text(encoding='utf-8'))
    for name,digest in source_manifest['input_sha256'].items():
        source_path=(length_comparison.input_paths()[name]
                     if source_manifest.get('input_path_format')=='configured-inputs-v1' else ROOT/name)
        assert hashlib.sha256(source_path.read_bytes()).hexdigest()==digest,name
    assert len(curves)==14000
    count_records=count_figure()
    length_figure(data,curves)
    for model in MODELS:length_figure(data,curves,models=(model,),stem=f'{model}_length_overlay')
    curves.to_csv(OUT/'figure2_plotted_predictions.csv.gz',index=False,compression='gzip')
    data['cell_predictions'].loc[data['cell_predictions'].apply(lambda r:r.length_term==MODE_RULE[r['mode']],axis=1)].to_csv(OUT/'figure2_observed_cells.csv',index=False)
    inputs={f'short/{name}':SHORT/name for name in ('cell_outcomes.csv.gz','selected_model_coefficients.csv','selected_mode_laws.csv')}
    inputs.update({f'length-comparison/{name}':LONG/name for name in
                   ('mode_rule_plotted_predictions.csv.gz','comparison_summary.csv','comparison_manifest.json')})
    inputs['source/build_niah_empirical_paper_figures.py']=Path(__file__)
    manifest=dict(figure1_slots=12,figure1_conditions=len(count_records),figure2_models=2,figure2_modes=2,
        figure2_N_levels=14,figure2_lengths=17,figure2_conditions=952,figure2_curve_points=len(curves),refits=0,
        figure1_spread='interquartile range across model comparison slots',figure2_form=MODE_RULE,
        input_sha256={label:hashlib.sha256(p.read_bytes()).hexdigest() for label,p in inputs.items()},
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.name not in {'build_manifest.json','validation_manifest.json'}},
        elapsed_seconds=time.perf_counter()-start)
    (OUT/'build_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(OUT)


if __name__=='__main__':main()
