"""Complete the Section 3 appendix figures using saved observations and fits."""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import runpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MaxNLocator
import numpy as np
import pandas as pd

from analyze_length import FORMS, effective_phi

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
BASE=ROOT/'figures/empirical_section3_refit'
API=runpy.run_path(str(BASE/'analyze.py'))
TARGET=ROOT/'runs/paper_figures/figures/empirical_appendix'
MODES=[('direct','Non-thinking','#B52F6B','-','o'),
       ('native_thinking','Thinking','#007EAB','--','^')]
STYLE={
    'phi_linear':('#D55E00','-',r'Linear $\varphi$'),
    'phi_log':('#7A329B','-',r'Log $\varphi$'),
    'hazard_linear':('#D55E00','--',r'Linear $\lambda$'),
    'hazard_log':('#7A329B','--',r'Log $\lambda$'),
    'reciprocal':('#46735E',':',r'Reciprocal $q$')}


def style():
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
        'mathtext.fontset':'stix','font.size':8.5,'axes.labelsize':9,
        'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.linewidth':.65,'text.color':'#202020','axes.labelcolor':'#202020',
        'xtick.color':'#202020','ytick.color':'#202020',
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})


def axes_style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#8D99A5')
    ax.grid(axis='y',color='#E7E8EE',linewidth=.55)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5,width=.6,pad=2)
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))


def save(fig,name,coverage):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    items=[{'text':a.get_text(),'bbox':a.get_window_extent(renderer).extents.tolist()}
           for a in fig.findobj(matplotlib.text.Text) if a.get_visible() and a.get_text().strip()]
    outside=[a for a in items if a['bbox'][0]<-.5 or a['bbox'][1]<-.5
             or a['bbox'][2]>fig.bbox.width+.5 or a['bbox'][3]>fig.bbox.height+.5]
    overlaps=[]
    for a,b in combinations(items,2):
        x,y=a['bbox'],b['bbox']
        if min(x[2],y[2])-max(x[0],y[0])>1 and min(x[3],y[3])-max(x[1],y[1])>1:
            overlaps.append([a['text'],b['text']])
    (OUT/f'{name}_layout.json').write_text(json.dumps({'outside':outside,'overlaps':overlaps},indent=2),encoding='utf-8')
    fig.savefig(OUT/f'{name}.png',dpi=240,facecolor='white')
    assert not outside and not overlaps, (name,outside,overlaps)
    for extension in ['pdf','svg']:
        fig.savefig(OUT/f'{name}.{extension}',facecolor='white')
    plt.close(fig)
    TARGET.mkdir(exist_ok=True)
    (TARGET/f'{name}.pdf').write_bytes((OUT/f'{name}.pdf').read_bytes())
    coverage['pdf_sha256']=hashlib.sha256((OUT/f'{name}.pdf').read_bytes()).hexdigest()
    coverage['layout_no_overlaps_or_clipping']=True
    return coverage


def median_figure(primary,medians):
    fig=plt.figure(figsize=(6.5,2.15))
    lengths=sorted(medians.L.unique())
    colors=plt.colormaps['plasma_r'](np.linspace(.12,.94,len(lengths)))
    records=[]
    for i,(mode,title,_,ls,marker) in enumerate(MODES):
        ax=fig.add_axes([.085+i*.50,.265,.41,.615])
        fitted=next(p for p in primary if p['scope']=='median_short' and p['mode']==mode
                    and p['form']==('gaussian_linear' if mode=='direct' else 'interaction_free_length'))
        for length,color in zip(lengths,colors):
            rows=medians.loc[medians['mode'].eq(mode)&medians.L.eq(length)].sort_values('N')
            dense=pd.DataFrame({'N':np.geomspace(1,20,300),'L':length})
            ax.plot(dense.N,API['predict'](dense,fitted),color=color,ls=ls,lw=1.05)
            ax.plot(rows.N,rows.observed,ls='none',marker=marker,ms=2.5,mfc='white',mec=color,mew=.55)
            records+=rows[['N','L','observed']].assign(mode=mode).to_dict('records')
        ax.set_title(f'{chr(65+i)}  {title}',loc='left',pad=5)
        ax.set_xscale('log',base=2)
        ax.set_xlim(.94,21)
        ax.set_xticks([1,2,3,5,10,20],labels=['1','2','3','5','10','20'])
        ax.set_ylim(-.025,1.035)
        ax.set_yticks([0,.25,.5,.75,1])
        ax.set_xlabel('Target count $N$',labelpad=3)
        axes_style(ax)
    fig.text(.008,.565,'Median exact accuracy',rotation=90,va='center',fontsize=9)
    handles=[Line2D([],[],color=c,lw=1.3,label=f'{l//1000}k') for l,c in zip(lengths,colors)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.57,.018),ncol=8,
               frameon=False,handlelength=1.25,handletextpad=.35,columnspacing=1.05,borderaxespad=0)
    fig.text(.13,.052,'$L$ (tokens):',va='center',fontsize=8)
    pd.DataFrame(records).to_csv(OUT/'median_figure_observations.csv',index=False)
    return save(fig,'empirical_median_fits',{'cells':224,'counts':14,'lengths':8,'modes':2,
        'observations':'unsmoothed cross-model medians','curves':'primary full fits'})


def wilson(p,n=30):
    z=1.959963984540054
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half,center+half


def full_grid(current,model,name):
    whole=current.loc[current.model.eq(model)].copy()
    counts=sorted(whole.N.unique())
    assert len(counts)==14 and len(whole)==476 and whole.L.nunique()==17
    assert whole.n_requests.eq(30).all() and whole.n_seeds.eq(30).all()
    fig=plt.figure(figsize=(6.5,5.45))
    grid=fig.add_gridspec(4,4,left=.08,right=.980,bottom=.12,top=.965,hspace=.57,wspace=.24)
    for i,n in enumerate(counts):
        row,col=divmod(i,4)
        ax=fig.add_subplot(grid[row,col])
        if model=='Gemma4-31B':
            ax.axvspan(20,25,color='#ECEEF2',alpha=.85,lw=0,zorder=0)
        for mode,title,color,ls,marker in MODES:
            data=whole.loc[whole['mode'].eq(mode)&whole.N.eq(n)].sort_values('L')
            assert len(data)==17
            segments=[data] if model=='Qwen3-32B' else [data.loc[data.L.le(20000)],data.loc[data.L.ge(25000)]]
            for segment in segments:
                x=segment.L.to_numpy()/1000
                y=segment.observed.to_numpy()
                lo,hi=wilson(y)
                ax.fill_between(x,lo,hi,color=color,alpha=.105,lw=0)
                ax.plot(x,y,color=color,ls=ls,lw=.9,marker=marker,ms=1.9,mfc='white',mew=.45)
        ax.set_title(f'$N={n}$',loc='left',fontsize=9,pad=3)
        ax.set_xlim(0,102)
        ax.set_ylim(-.035,1.035)
        ax.set_xticks([0,50,100])
        ax.set_yticks([0,.5,1])
        axes_style(ax)
        ax.tick_params(labelleft=(col==0))
    legend_ax=fig.add_subplot(grid[3,2:])
    legend_ax.axis('off')
    handles=[Line2D([],[],color=color,lw=1.15,ls=ls,marker=marker,ms=3,
                    mfc='white',mew=.55,label=title) for _,title,color,ls,marker in MODES]
    legend_ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(.015,1.10),
        frameon=False,handlelength=2.4,labelspacing=.55,borderaxespad=0)
    legend_ax.text(.02,.32,'Shading: 95% Wilson intervals\n30 paired seeds per condition',fontsize=8,va='top',linespacing=1.7)
    fig.text(.009,.565,'Exact accuracy',rotation=90,va='center',fontsize=9)
    fig.text(.39,.025,'Length $L$ (k tokens)',ha='center',fontsize=9)
    whole.to_csv(OUT/f'{name}_observations.csv',index=False)
    return save(fig,name,{'model':model,'cells':len(whole),'counts':[int(x) for x in counts],
        'lengths':[int(x) for x in sorted(whole.L.unique())],'modes':2,'smoothing':False,
        'pointwise_wilson_intervals':True,'gemma_batches_separated':model=='Gemma4-31B'})


def length_figure(primary,candidates):
    fig=plt.figure(figsize=(6.5,2.65))
    specs=[('median_short','Median across 12 groups','A  Across-model median',20000),
           ('current_full','Qwen3-32B','B  Qwen3-32B',100000),
           ('current_full','Gemma4-31B','C  Gemma-4-31B',100000)]
    records=[]
    for i,(scope,model,title,last) in enumerate(specs):
        ax=fig.add_axes([.075+i*.33,.31,.255,.57])
        fitted=next(f for f in primary if f['scope']==scope and f['model']==model
                    and f['mode']=='native_thinking' and f['form']=='interaction_free_length')
        lengths=np.array(fitted['lengths'])
        phi=-np.expm1(-np.array(fitted['parameters'][1:])/20)
        ax.plot(lengths/1000,phi,ls='none',marker='o',ms=3,color='#333333',zorder=5)
        largest=float(phi.max())
        for form in FORMS:
            fit=next(f for f in candidates if f['scope']==scope and f['model']==model and f['form']==form)
            dense=np.linspace(1000,last,300)
            curve=effective_phi(dense,fit)
            color,ls,label=STYLE[form]
            ax.plot(dense/1000,curve,color=color,ls=ls,lw=1.05,label=label)
            largest=max(largest,float(curve.max()))
            records += [dict(scope=scope,model=model,form=form,L=float(l),phi=float(p)) for l,p in zip(dense,curve)]
        for length,p in zip(lengths,phi):
            records.append(dict(scope=scope,model=model,form='free_length',L=float(length),phi=float(p)))
        ax.set_title(title,loc='left',fontsize=9.2,pad=5)
        ax.set_xlim(0,last/1000*1.02)
        ax.set_xticks([0,10,20] if last==20000 else [0,50,100])
        ticks=MaxNLocator(nbins=4,steps=[1,2,5,10]).tick_values(0,max(.01,largest*1.10))
        ax.set_ylim(0,ticks[-1])
        ax.set_yticks(ticks)
        axes_style(ax)
    fig.text(.008,.605,r'Fitted per-step failure parameter $\widehat{\varphi}_L$',rotation=90,va='center',fontsize=9)
    fig.text(.51,.15,'Length $L$ (k tokens)',ha='center',fontsize=9)
    handles=[Line2D([],[],color='#333333',ls='none',marker='o',ms=3,label='Free by length')]
    handles += [Line2D([],[],color=STYLE[f][0],ls=STYLE[f][1],lw=1.1,label=STYLE[f][2]) for f in FORMS]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.015),ncol=6,frameon=False,
               handlelength=1.7,handletextpad=.4,columnspacing=.9,borderaxespad=0)
    pd.DataFrame(records).to_csv(OUT/'length_figure_values.csv',index=False)
    return save(fig,'empirical_length_candidates',{'panels':3,'candidate_families':5,
        'dots':'joint count-model fits with free parameter at each length',
        'curves':'all candidate fits to unsmoothed accuracies; not fitted to the dots',
        'panel_y_scales_differ':True})


def main():
    style()
    _,current,medians=API['load_data']()
    primary=json.loads((BASE/'full_fit_parameters.json').read_text(encoding='utf-8'))
    candidates=json.loads((OUT/'length_parameters.json').read_text(encoding='utf-8'))
    manifest={}
    manifest['median']=median_figure(primary,medians)
    manifest['qwen']=full_grid(current,'Qwen3-32B','empirical_qwen_all_counts')
    manifest['gemma']=full_grid(current,'Gemma4-31B','empirical_gemma_all_counts')
    manifest['length']=length_figure(primary,candidates)
    manifest['existing_12group_figure']='figures/empirical_law_12models/empirical_accuracy_12models.pdf'
    manifest['primary_parameter_sha256']=hashlib.sha256((BASE/'full_fit_parameters.json').read_bytes()).hexdigest()
    manifest['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k in ['median','qwen','gemma','length']},indent=2))


if __name__=='__main__':
    main()
