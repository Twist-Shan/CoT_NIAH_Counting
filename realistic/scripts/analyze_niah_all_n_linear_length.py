"""Two-model 1k--100k probability fits: N intercepts and a shared length slope.

Reuses the existing Bernoulli GLM and held-condition folds. Refits all N levels,
exports per-N diagnostics, and verifies against the archived numeric fits.
"""
from pathlib import Path
import sys
import json
import hashlib
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_realistic_niah_v3_3_regression_scan import (
    normalize_requests, n_fixed_design, fit_n_fixed_accuracy_cv, N_LEVELS,
)
from scripts.analyze_realistic_niah_v3_2_empirical_laws import (
    fit_glm, condition_fold, HEADLINE_ACCURACY, clip_probability,
)

LONG = ROOT/'outputs/anvil_realistic_niah_v3_3_long_context_20260906_holdout/analysis/v3_3_regression_scan'
OUT = LONG.parent/'v3_3_all_n_linear_length_20260909'
ASSETS = ROOT/'reports/assets/niah_empirical_all_n_length'
MODELS = ('Gemma4-31B','Qwen3-32B')
MODES = ('direct','native_thinking')
COLORS = {'direct':'#D86F19','native_thinking':'#7552BE'}


def loss(y,p):
    p=clip_probability(p)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))


def r2(y,p):
    sst=float(np.sum((y-np.mean(y))**2))
    return 1-float(np.sum((y-p)**2))/sst if sst>0 else np.nan


def plot_n_grid(model,cells,coefficients,length_term='L_k'):
    fig,axes=plt.subplots(3,5,figsize=(15.8,8.6),sharex=True,sharey=True)
    for ax,n in zip(axes.flat,N_LEVELS):
        for mode in MODES:
            block=cells.loc[cells.model.eq(model)&cells['mode'].eq(mode)&cells.N.eq(n)].sort_values('L')
            c=coefficients.loc[coefficients.model.eq(model)&coefficients['mode'].eq(mode)].set_index('term').estimate
            grid=np.linspace(1000,100000,300)
            feature=grid/1000 if length_term=='L_k' else np.log(grid/1000)
            prediction=expit(c[f'alpha_N={n}']+c[f'beta_{length_term}']*feature)
            ax.plot(grid/1000,prediction,color=COLORS[mode],linestyle='-' if mode=='direct' else '--',linewidth=1.8)
            ax.scatter(block.L/1000,block.observed,s=16,marker='o' if mode=='direct' else '^',facecolor='white',edgecolor=COLORS[mode],linewidth=.8,zorder=3)
        ax.set_title(f'N = {n}',loc='left',fontweight='bold',fontsize=11)
        ax.set_xlim(0,102);ax.set_ylim(-.03,1.03);ax.set_xticks([1,50,100]);ax.set_yticks([0,.5,1])
        ax.axvline(20,color='#A9B0BC',linestyle=':',linewidth=.7)
        ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False)
    legend_ax=axes.flat[-1];legend_ax.axis('off')
    legend_ax.legend(handles=[Line2D([0],[0],color=COLORS['direct'],marker='o',markerfacecolor='white',label='Non-thinking'),Line2D([0],[0],color=COLORS['native_thinking'],marker='^',markerfacecolor='white',linestyle='--',label='Native-thinking')],loc='upper left',frameon=False,fontsize=11)
    legend_ax.text(.04,.51,'Open markers: observed\nLines: full-data GLM fit\nDotted vertical: 20k\n\nOne slope per mode;\n14 N-specific intercepts.',transform=legend_ax.transAxes,fontsize=10,color='#536174',va='top')
    form='linear' if length_term=='L_k' else 'log'
    formula='(L / 1000)' if length_term=='L_k' else 'ln(L / 1000)'
    fig.suptitle(f'{model}: shared {form} length slope across all N',x=.055,ha='left',fontsize=18,fontweight='bold')
    fig.text(.055,.927,f'logit p(N, L) = alpha_N + beta {formula} | 1k–100k | no interaction',fontsize=11,color='#536174')
    fig.supxlabel('Passage length (thousand tokens; linear scale)',y=.02)
    fig.supylabel('Probability of a correct count',x=.015)
    fig.subplots_adjust(left=.06,right=.99,top=.87,bottom=.09,hspace=.30,wspace=.12)
    stem=f'{model}_all_N' + ('' if length_term=='L_k' else '_logL')
    for suffix in ('png','pdf'):fig.savefig(ASSETS/f'{stem}.{suffix}',dpi=170,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    start=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length-term',choices=('L_k','logL'),default='L_k')
    length_term=parser.parse_args().length_term
    out=OUT if length_term=='L_k' else LONG.parent/'v3_3_all_n_log_length_20260909'
    out.mkdir(parents=True,exist_ok=True);ASSETS.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((LONG/'analysis_manifest.json').read_text(encoding='utf-8'))
    paths=[LONG/'input/v3_1_two_model_request_level.csv.gz',LONG/'input/v3_3_two_model_request_level.csv.gz']
    frames=[];hashes={}
    for key,path in zip(('old','holdout'),paths):
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==manifest['inputs'][key]['sha256']
        hashes[str(path.relative_to(ROOT))]=digest
        frames.append(normalize_requests(path,key))
    requests=pd.concat(frames,ignore_index=True)
    assert len(requests)==28560 and not requests.request_id.duplicated().any()
    lengths=tuple(sorted(requests.L.unique()))
    assert len(lengths)==17
    archived=pd.read_csv(LONG/'tables/n_fixed_shared_length_metrics.csv')
    rows,coef_rows,cell_tables,diagnostics=[],[],[],[]
    for model in MODELS:
        for mode in MODES:
            frame=requests.loc[requests.model_label.eq(model)&requests.prompt_mode.eq(mode)].copy()
            tags=dict(model=model,mode=mode)
            metric,coefs=fit_n_fixed_accuracy_cv(frame,length_term,N_LEVELS,lengths)
            beta=pd.Series({c['term']:c['estimate'] for c in coefs})
            x,names=n_fixed_design(frame,N_LEVELS,length_term)
            folds=condition_fold(frame,N_LEVELS,lengths);y=frame.exact_count.to_numpy(float)
            fitted=expit(x@beta.loc[list(names)].to_numpy())
            oof=np.full(len(frame),np.nan);baseline=oof.copy()
            for fold in range(5):
                test=folds==fold;train=~test
                assert set(frame.loc[train,'N'])==set(N_LEVELS)
                assert np.linalg.matrix_rank(x[train])==15
                fit=fit_glm(y[train],x[train],HEADLINE_ACCURACY,robust=False)
                oof[test]=fit.predict(x[test])
                rates=frame.loc[train].groupby('N').exact_count.mean()
                baseline[test]=frame.loc[test,'N'].map(rates).to_numpy(float)
            assert np.isfinite(oof).all() and np.isfinite(baseline).all()
            assert np.isclose(loss(y,oof),metric['cv_log_loss'],atol=1e-10)
            old=archived.loc[archived.model_label.eq(model)&archived.prompt_mode.eq(mode)&archived.outcome_family.eq(HEADLINE_ACCURACY)&archived.length_term.eq(length_term)&archived.evaluation_scheme.eq('combined_5fold_held_condition_cv')].iloc[0]
            assert np.isclose(metric['primary_loss'],old.primary_loss,rtol=1e-8)
            frame=frame.assign(fitted=fitted,oof=oof,n_baseline_oof=baseline)
            cells=frame.groupby(['N','L']).agg(observed=('exact_count','mean'),n_requests=('request_id','size'),fitted=('fitted','mean'),oof=('oof','mean'),n_baseline_oof=('n_baseline_oof','mean')).reset_index()
            assert len(cells)==238 and cells.n_requests.eq(30).all()
            coef_rows.extend(dict(c,**tags) for c in coefs)
            cell_tables.append(cells.assign(**tags))
            slope=next(c for c in coefs if c['term']==f'beta_{length_term}')
            rows.append(dict(metric,**tags,cell_R2=r2(cells.observed,cells.fitted),cell_CV_R2=r2(cells.observed,cells.oof),**{f'beta_{length_term}':slope['estimate']},beta_ci95_low=slope['ci95_low'],beta_ci95_high=slope['ci95_high']))
            for n,block in cells.groupby('N'):
                diagnostics.append(dict(tags,N=int(n),cell_R2=r2(block.observed,block.fitted),cell_CV_R2=r2(block.observed,block.oof),cv_log_loss=loss(block.observed,block.oof),alpha_N=beta[f'alpha_N={n}']))
            print(model,mode,'verified',flush=True)
    cells=pd.concat(cell_tables,ignore_index=True);coefs=pd.DataFrame(coef_rows);metrics=pd.DataFrame(rows)
    metrics.to_csv(out/'metrics.csv',index=False);coefs.to_csv(out/'coefficients.csv',index=False)
    cells.to_csv(out/'cell_predictions.csv',index=False);pd.DataFrame(diagnostics).to_csv(out/'per_N_metrics.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    for model in MODELS:plot_n_grid(model,cells,coefs,length_term)
    (out/'manifest.json').write_text(json.dumps(dict(input_sha256=hashes,requests=len(requests),conditions=len(cells),fits=4,parameters_per_fit=15,length_term=length_term,N_levels=list(N_LEVELS),L_levels=[int(l) for l in lengths],archived_cv_reproduced=True,elapsed_seconds=time.perf_counter()-start,estimand='Bernoulli exact-count probability; all N retained; each model/mode fitted separately',R2_note='Per-N zero variance returns undefined; no replacement by zero'),indent=2),encoding='utf-8')
    print(metrics[['model','mode',f'beta_{length_term}','cell_R2','cell_CV_R2','cv_d2_vs_global_intercept','cv_d2_vs_N_fixed']].to_string(index=False))


if __name__=='__main__':main()
