"""Reproducible front-page probability and paired-mode length figures."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
import build_niah_empirical_law_v3_2_report as v32
from analyze_realistic_niah_v3_3_regression_scan import normalize_requests, add_predictors, coefficient_prediction

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'reports/assets/niah_empirical_front'
ANALYSIS = ROOT / 'outputs/realistic_niah_v3_1_20260819_formal/analysis'
LONG = ROOT / 'outputs/realistic_niah_v3_3_long_context_20260906_holdout/analysis/v3_3_regression_scan'


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    cell_path = ANALYSIS / 'v3_2_trimmed_count_error_extension/tables/count_error_cells.csv.gz'
    coef_path = ANALYSIS / 'v3_2_inverse_n_candidate_extension/tables/selected_model_coefficients.csv'
    cells, coefficients = pd.read_csv(cell_path), pd.read_csv(coef_path)
    assert cells.comparison_slot.nunique() == 12
    v32.set_plot_style()
    v32.plot_model_law_panels(cells, coefficients, family='accuracy_bernoulli_logit',
        column='parsed_exact_accuracy', modes=('direct','native_thinking'), x_axis='N',
        path=ASSETS/'probability_12_slots.png',
        title='Probability of a correct count across 12 comparison slots', ylabel='Probability of a correct count')
    manifest = json.loads((LONG/'analysis_manifest.json').read_text(encoding='utf-8'))
    paths = [LONG/'input/v3_1_two_model_request_level.csv.gz', LONG/'input/v3_3_two_model_request_level.csv.gz']
    for key, path in zip(('old','holdout'), paths):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['inputs'][key]['sha256']
    requests = pd.concat([normalize_requests(p, key) for p,key in zip(paths, ('old','holdout'))],ignore_index=True)
    assert not requests.request_id.duplicated().any()
    block = requests.loc[requests.N.eq(10)]
    obs = block.groupby(['model_label','prompt_mode','L']).exact_count.agg(['mean','size']).reset_index()
    selections = pd.read_csv(LONG/'tables/exploratory_selected_laws.csv')
    coefs = pd.read_csv(LONG/'tables/exploratory_candidate_coefficients.csv')
    candidate_metrics = pd.read_csv(LONG/'tables/exploratory_candidate_metrics.csv')
    selected = selections.loc[selections.dataset.eq('combined_1k_100k_fixed_N10') & selections.outcome_family.eq('accuracy_bernoulli_logit')]
    fig, axes = plt.subplots(2,2,figsize=(12.5,7.9),sharex=True,sharey=True)
    records, fit_scores = [], []
    for i,mode in enumerate(('direct','native_thinking')):
        color = '#D96B16' if i == 0 else '#7655C9'
        label = 'Non-thinking' if i == 0 else 'Native-thinking'
        for j,model in enumerate(('Gemma4-31B','Qwen3-32B')):
            ax=axes[i,j]
            points=obs.loc[obs.model_label.eq(model)&obs.prompt_mode.eq(mode)].sort_values('L')
            assert len(points)==17 and points['size'].eq(30).all()
            sel=selected.loc[selected.model_label.eq(model)&selected.prompt_mode.eq(mode)].iloc[0]
            grid=add_predictors(pd.DataFrame({'N':10,'L':np.geomspace(1000,100000,300)}))
            selected_r2 = None
            for term in ('L_k','logL'):
                c=coefs.loc[coefs.dataset.eq('combined_1k_100k_fixed_N10') & coefs.model_label.eq(model) & coefs.prompt_mode.eq(mode) & coefs.outcome_family.eq('accuracy_bernoulli_logit') & coefs.candidate.eq(term)]
                assert len(c)==2
                pred=expit(coefficient_prediction(grid,c))
                chosen=term==sel.selected_candidate
                at_observed = expit(coefficient_prediction(add_predictors(points.assign(N=10)), c))
                y = points['mean'].to_numpy(float)
                sst = float(np.sum((y-y.mean())**2))
                r2 = 1-float(np.sum((y-at_observed)**2))/sst if sst > 0 else np.nan
                metric = candidate_metrics.loc[candidate_metrics.dataset.eq('combined_1k_100k_fixed_N10') & candidate_metrics.model_label.eq(model) & candidate_metrics.prompt_mode.eq(mode) & candidate_metrics.outcome_family.eq('accuracy_bernoulli_logit') & candidate_metrics.candidate.eq(term)].iloc[0]
                fit_scores.append(dict(model=model,mode=mode,form=term,cell_R2=r2,cv_D2=metric.cv_d2,in_sample_D2=metric.in_sample_d2))
                if chosen: selected_r2 = r2
                ax.plot(grid.L/1000,pred,color=color if chosen else '#8F99A8',linewidth=2.5 if chosen else 1.4,
                    linestyle='-' if term=='L_k' else '--',label=('Linear L' if term=='L_k' else 'Log length ln L')+(' (selected)' if chosen else ''))
                for l,p in zip(grid.L,pred): records.append(dict(model=model,mode=mode,candidate=term,L=l,prediction=p))
            ax.scatter(points.L/1000,points['mean'],s=33,marker='o' if i==0 else '^',facecolor='white',edgecolor=color,linewidth=1.2,zorder=5,label='Observed (30 trials/cell)')
            ax.axvline(20,color='#657487',linewidth=1,linestyle=':',label='V3.2 upper length')
            ax.set_xscale('log');ax.set_ylim(-.025,1.025)
            ax.set_xticks([1,2,5,10,20,50,100]);ax.set_xticklabels(['1','2','5','10','20','50','100'])
            ax.set_title(f'{model} | {label}',loc='left',fontsize=12,fontweight='bold')
            ax.text(.03,.84 if i==0 else .08,f"Selected: {'L' if sel.selected_candidate=='L_k' else 'ln L'} | Cell R² = {selected_r2:.3f}\nCV D² = {sel.selected_cv_score:.3f}",transform=ax.transAxes,color=color,fontsize=9,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
            ax.grid(alpha=.22);ax.spines[['top','right']].set_visible(False)
            if j==0: ax.set_ylabel('Probability of a correct count')
            if i==1: ax.set_xlabel('Passage length (thousand tokens; log scale)')
            ax.legend(loc='upper right' if i==0 else 'lower left',bbox_to_anchor=None if i==0 else (0,.21),fontsize=7.5,framealpha=.9)
    fig.suptitle('Length regressions at fixed N = 10',x=.07,ha='left',fontsize=18,fontweight='bold')
    fig.text(.07,.924,'Two models × two modes | 1k–100k exploratory refits | logit(p) = a + b f(L)',fontsize=10,color='#536174')
    fig.subplots_adjust(left=.075,right=.985,top=.86,bottom=.09,wspace=.12,hspace=.25)
    for ext in ('png','pdf'): fig.savefig(ASSETS/f'length_two_models.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(records).to_csv(ASSETS/'length_plotted_predictions.csv',index=False)
    pd.DataFrame(fit_scores).to_csv(ASSETS/'length_fit_metric_comparison.csv',index=False)
    obs.to_csv(ASSETS/'length_observed_cells.csv',index=False)
    selected.to_csv(ASSETS/'length_selected_results.csv',index=False)
    inputs=[cell_path,coef_path,*paths,LONG/'tables/exploratory_selected_laws.csv',LONG/'tables/exploratory_candidate_coefficients.csv',LONG/'tables/exploratory_candidate_metrics.csv']
    (ASSETS/'build_manifest.json').write_text(json.dumps({'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'probability_slots':12,'length_cells':len(obs),'fixed_N':10,'length_range':[1000,100000],'refit':True},indent=2),encoding='utf-8')
    print(ASSETS)


if __name__=='__main__': main()
