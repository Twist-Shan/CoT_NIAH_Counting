"""Align Thinking and Bullet display conventions using archived statistics only."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter,FixedLocator,FixedFormatter
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];REPO=ROOT/'realistic'
PREVIOUS=ROOT/'figures/enumeration_bullet_only_20260917'
MODELS=[('Qwen3-8B','Qwen','#168DCA',36),('Gemma4-E4B','Gemma','#E87824',42)]
SOURCES,FIGURES={},{}
def read(path):
    raw=path.read_bytes();SOURCES[path.relative_to(ROOT).as_posix()]=hashlib.sha256(raw).hexdigest();return raw.decode('utf-8-sig')
def js(path):return json.loads(read(path))
def module(name,path):
    read(path);spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
def save(fig,name,rows,extra=None):
    style.save(fig,name,rows,extra);FIGURES[name]=style.FIGURES[name]
    folder='cot_appendix' if name.startswith('cot_') else 'enumeration_bullet'
    shutil.copy2(OUT/f'{name}.pdf',ROOT/'runs/paper_figures/figures'/folder/f'{name}.pdf')
    (OUT/'data'/f'{name}.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
def two_panels(height=2.65):
    fig=plt.figure(figsize=(6.5,height));return fig,[fig.add_axes([x,.29,.365,.47]) for x in [.10,.61]]
def model_legend(fig):
    fig.legend([Line2D([],[],color=m[2]) for m in MODELS],[m[1] for m in MODELS],loc='center',bbox_to_anchor=(.5,.97),ncol=2)
def four_panels(height=4.25):
    fig=plt.figure(figsize=(6.5,height));return fig,[[fig.add_axes([x,y,.365,.275]) for x in [.10,.61]] for y in [.60,.13]]
def cell(report,model):
    rows=[r for r in report['cells'] if r['mode']=='enumeration_bullet' and r['model']==model];assert len(rows)==1;return rows[0]

def head_scores():
    native=list(csv.DictReader(read(ROOT/'figures/cot-reasoning/attention/data/head_scores.csv').splitlines()))
    bullet=js(PREVIOUS/'data/enumeration_head_scores.json')
    for mode,source,name in [('thinking',native,'cot_full_head_scores'),('enumeration_bullet',bullet,'enumeration_head_scores')]:
        fig=plt.figure(figsize=(6.5,3.5));rows=[];banks={}
        for j,(model,short,color,_) in enumerate(MODELS):
            if mode=='thinking':
                group=[r for r in source if r['model']==model and r['displayed_global_attention']=='True']
                rr=[dict(model=model,mode=mode,layer=int(r['layer']),head=int(r['head']),score=float(r['mean_targeted_score']),selected=r['highlighted_ablation_bank']=='True',rank=int(r['discovery_rank'])) for r in group]
            else:
                group=[r for r in source if r['model']==model]
                rr=[dict(model=model,mode=mode,layer=int(r['layer'])+1,head=int(r['head'])+1,score=float(r['score']),selected=i<(128 if j==0 else 6),rank=i+1) for i,r in enumerate(group)]
            layers=sorted({r['layer'] for r in rr});heads=sorted({r['head'] for r in rr})
            assert len(rr)==len(layers)*len(heads)==(1152 if j==0 else 56)
            bank=[r for r in rr if r['selected']];assert len(bank)==(128 if j==0 else 6)
            matrix=np.array([[next(r['score'] for r in rr if r['layer']==l and r['head']==h) for h in heads] for l in layers])
            assert np.isfinite(matrix).all() and 0<=matrix.min()<=matrix.max()<=1
            ax=fig.add_axes([[.095,.59][j],.18,.335,.66])
            mesh=ax.pcolormesh(np.arange(len(heads)+1)+.5,np.arange(len(layers)+1)+.5,matrix,
                cmap=LinearSegmentedColormap.from_list(short,['#F7FAFC',color]),norm=Normalize(0,1),edgecolors='white',linewidth=.18,rasterized=False)
            ax.set(xlim=(.5,len(heads)+.5),ylim=(len(layers)+.5,.5))
            xt=[1,9,17,25,32] if j==0 else list(range(1,9));yi=[0,5,11,17,23,29,35] if j==0 else list(range(7))
            ax.set_xticks(xt);ax.set_yticks([i+1 for i in yi],[layers[i] for i in yi])
            ax.set_xlabel('Head index',labelpad=4);ax.set_ylabel('Layer' if j==0 else 'Global layer',labelpad=4)
            ax.set_title(f'{"AB"[j]}. {short}',loc='left',fontsize=11,fontweight='normal',pad=8)
            ax.tick_params(length=0,pad=4);ax.spines[:].set_visible(False)
            for r in bank:ax.add_patch(Rectangle((r['head']-.5,layers.index(r['layer'])+.5),1,1,fill=False,edgecolor='#273743',linewidth=.5))
            cb=fig.colorbar(mesh,cax=fig.add_axes([[.443,.938][j],.18,.015,.66]),ticks=[0,.5,1]);cb.ax.tick_params(labelsize=9,length=2,width=.5,pad=2);cb.outline.set_linewidth(.5)
            banks[model]=[dict(layer=r['layer'],head=r['head'],rank=r['rank']) for r in bank];rows.extend(rr)
        fig.text(.5,.975,r'Targeted retrieval score $T_h$ (shared 0--1 scale)',ha='center',va='top',fontsize=9)
        save(fig,name,rows,dict(frozen_banks=banks,axis_order='x=head; y=layer, shallow to deep downward',color_range=[0,1],in_cell_rank_labels=False))

def readouts():
    rows=js(PREVIOUS/'data/enumeration_representations.json')
    selected=js(PREVIOUS/'manifest.json')['figures']['enumeration_representations']['selected_layers']
    fig=plt.figure(figsize=(6.5,2.55));axes=[fig.add_axes([x,.215,.400,.535]) for x in [.095,.580]]
    for j,endpoint in enumerate(['running_index','final_count']):
        ax=axes[j];style.panel(ax,['A. Running index','B. Final count'][j],'Balanced accuracy' if j==0 else None,'Layer')
        for model,_,color,_ in MODELS:
            for method,ls in [('ncc','-'),('logistic','--')]:
                rr=sorted([r for r in rows if r['model']==model and r['endpoint']==endpoint and r['method']==method],key=lambda r:r['layer_display_one_based'])
                ax.plot([r['layer_display_one_based'] for r in rr],[r['balanced_accuracy'] for r in rr],color=color,ls=ls)
            chosen,=[r for r in selected if r['model_label']==model and r['endpoint']==endpoint]
            layer,value=chosen['layer_display_one_based'],chosen['discovery_oof_ncc_balanced_accuracy']
            ax.scatter(layer,value,s=27,color=color,edgecolor='white',linewidth=.6,zorder=4)
            ax.annotate(f'L{layer}',(layer,value),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8,color=color)
        ax.set(xlim=(.5,42.5),xticks=[1,10,20,30,42],ylim=(0,1.12),yticks=[0,.2,.4,.6,.8,1]);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.axhline(.1,color=style.GRAY,ls=':',lw=.9)
    fig.legend([Line2D([],[],color=m[2]) for m in MODELS]+[Line2D([],[],color=style.INK,ls=s) for s in ['-','--']],['Qwen','Gemma','Nearest centroid','Logistic'],loc='upper center',bbox_to_anchor=(.535,.995),ncol=4,handlelength=1.7,handletextpad=.45,columnspacing=1.4)
    save(fig,'enumeration_representations',rows,dict(selected_layers=selected,selected_layer_annotation_pt=8,reference='cot_count_readouts'))

def dose():
    native=list(csv.DictReader(read(ROOT/'figures/cot_completion_20260912/cot_current_bank_dose.csv').splitlines()))
    bullet=js(PREVIOUS/'data/enumeration_retrieve.json')
    normalized={
      'cot_current_bank_dose':[dict(model=r['model'],mode='thinking',metric={'next_item':'next_city_failure','final_count':'final_exact_count_failure'}[r['outcome']],condition='selected_minus_clean' if r['condition']=='selected_bank' else 'random_minus_clean',k=int(r['k']),estimate=float(r['mean']),ci95=[float(r['ci95_low']),float(r['ci95_high'])],random_control='layer_matched') for r in native],
      'enumeration_retrieve':bullet}
    lower=min(-.04,min(r['ci95'][0] for rows in normalized.values() for r in rows)-.04)
    for name,rows in normalized.items():
        fig,axes=four_panels()
        for i,(model,short,color,_) in enumerate(MODELS):
            for j,metric in enumerate(['next_city_failure','final_exact_count_failure']):
                ax=axes[i][j];style.panel(ax,f'{"ABCD"[2*i+j]}. {short}: '+['next item','final count'][j],'Failure increase',r'Ablated heads, $K$')
                for condition,ls,marker in [('selected_minus_clean','-','o'),('random_minus_clean','--','D')]:
                    rr=sorted([r for r in rows if r['model']==model and r['metric']==metric and r['condition']==condition],key=lambda r:r['k'])
                    ks=[r['k'] for r in rr];ci=np.asarray([r['ci95'] for r in rr]);ax.plot(ks,[r['estimate'] for r in rr],color=color,ls=ls,marker=marker,ms=3.5)
                    ax.fill_between(ks,ci[:,0],ci[:,1],color=color,alpha=.12,lw=0)
                ax.set_xscale('log',base=2);ax.set_xlim(ks[0]/1.12,ks[-1]*1.12);ax.xaxis.set_major_locator(FixedLocator(ks));ax.xaxis.set_major_formatter(FixedFormatter([str(k) for k in ks]));ax.minorticks_off()
                ax.set(ylim=(lower,1.04),yticks=[0,.5,1]);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.axhline(0,color=style.GRAY,lw=.7)
        fig.legend([Line2D([],[],color=style.INK,marker='o'),Line2D([],[],color=style.INK,ls='--',marker='D')],['Frozen Top-$K$ prefix','Random control'],loc='upper center',bbox_to_anchor=(.535,.998),ncol=2)
        save(fig,name,rows,dict(model_rows=['Qwen','Gemma'],outcome_columns=['next item','final count'],statistics_recomputed=False))

def update():
    report=js(REPO/'outputs/enumeration_replay_midlayer_20260917/midlayer_audit/combined_update_audit.json')
    fig,axes=style.four_panels(height=4.65);rows=[]
    for j,(model,short,color,_) in enumerate(MODELS):
        c=cell(report,model);groups=c['groups'];layer=c['layer_one_based']
        ax=axes[0][j]
        for di,direction in enumerate(['forward','backward']):
            marker,ls=[('o','-'),('D','--')][di]
            for x,scope in enumerate(['endpoint','four_token_tail','item_span']):
                g=next(r for r in groups if r['scope']==scope and r['direction']==direction and r['donor_k']=='all')
                for condition,offset,filled in [('donor_to_receiver',-.042,True),('receiver_self',.042,False)]:
                    h=g['conditions'][condition]['continuation']['1'];s=h['conditional'];mean=s['estimate'];lo,hi=s['ci95']
                    assert h['conditional_eligible']==30
                    ax.errorbar(x+[-.14,.14][di]+offset,mean,yerr=[[mean-lo],[hi-mean]],fmt=marker,color=color,mfc=color if filled else 'white',mec=color,ms=4.2,capsize=2.5,lw=1,zorder=4)
                    rows.append(dict(model=model,mode='enumeration_bullet',layer=layer,direction=direction,scope=scope,condition=condition,metric='target_successor_adoption',numerator=h['successes'],denominator=30,estimate=mean,ci95=s['ci95']))
            g=next(r for r in groups if r['scope']=='item_span' and r['direction']==direction and r['donor_k']=='all')
            hops=[g['conditions']['donor_to_receiver']['continuation'][str(h)] for h in range(1,5)]
            helper.curve(axes[1][j],list(range(1,5)),[h['conditional'] for h in hops],color,linestyle=ls,marker=marker)
            rows.extend(dict(model=model,mode='enumeration_bullet',layer_one_based=layer,direction=direction,scope='item_span',metric='conditional_next_step',hop=h['hop'],numerator=h['successes'],denominator=h['conditional_eligible'],horizon_eligible=h['horizon_eligible'],**h['conditional']) for h in hops)
        ax.set(xticks=[0,1,2],xticklabels=['Endpoint','Four-token\ntail','Item span'],xlim=(-.45,2.45),ylim=(-.04,1.19),yticks=[0,.5,1]);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        style.panel(ax,f'{"AB"[j]}. {short} L{layer}: patch scope','Target-successor adoption')
        ax=axes[1][j];ax.set(xticks=[1,2,3,4],xlim=(.85,4.15),ylim=(-.04,1.08),yticks=[0,.5,1]);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        style.panel(ax,f'{"CD"[j]}. {short}: continued prefix','Conditional success','Continuation step')
    fig.legend([Line2D([],[],marker='o',color=style.INK,ls='None'),Line2D([],[],marker='o',color=style.INK,mfc='white',ls='None')],['Target patch','Self patch'],loc='upper center',bbox_to_anchor=(.535,1),ncol=2,columnspacing=2)
    fig.legend([Line2D([],[],color=style.INK,ls=ls,marker=m,ms=4) for ls,m in [('-','o'),('--','D')]],['Forward','Backward'],loc='center',bbox_to_anchor=(.5,.025),ncol=2)
    save(fig,'enumeration_update',rows,dict(scope_estimand='Target and self adoption displayed separately, as in Thinking',statistics_recomputed=False))

def readout():
    rows=js(PREVIOUS/'data/enumeration_readout.json');fig,axes=two_panels()
    for mi,(model,_,color,_) in enumerate(MODELS):
        for condition,ls in [('full_donor_patch','-'),('self_patch','--')]:
            rr=[r for r in rows if r['model']==model and r['metric']=='answer_donor_adoption' and r['condition']==condition]
            helper.curve(axes[0],[r['layer'] for r in rr],rr,color,linestyle=ls)
        for x,condition in enumerate(['clean','prompt_records_blank','trace_all_blank']):
            r,=[r for r in rows if r['model']==model and r['metric']=='blanking_accuracy' and r['condition']==condition]
            helper.point(axes[1],x+(mi-.5)*.17,r,color)
    axes[0].set(xlim=(1,42),xticks=[1,10,20,30,40],ylim=(-.04,1.06),yticks=[0,.5,1])
    axes[1].set(xticks=[0,1,2],xticklabels=['Original','Prompt\nblank','Trace\nblank'],xlim=(-.4,2.4),ylim=(-.04,1.06),yticks=[0,.5,1])
    for ax in axes:ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    style.panel(axes[0],'A. Answer-state transfer','Target-count adoption','Layer');style.panel(axes[1],'B. Trace blanking','Exact-count accuracy')
    model_legend(fig);fig.legend([Line2D([],[],color=style.INK,ls=ls) for ls in ['-','--']],['Target patch','Self patch'],loc='center',bbox_to_anchor=(.5,.025),ncol=2)
    save(fig,'enumeration_readout',rows,dict(statistics_recomputed=False))

def pca():
    name='enumeration_pca_bullet';rec=js(PREVIOUS/'manifest.json')['figures'][name]
    for ext in ['pdf','png','svg']:
        path=PREVIOUS/f'{name}.{ext}';raw=path.read_bytes();h=hashlib.sha256(raw).hexdigest();assert h==rec['artifacts_sha256'][path.name]
        SOURCES[path.relative_to(ROOT).as_posix()]=h;shutil.copy2(path,OUT/path.name)
    rows=js(PREVIOUS/'data'/f'{name}.json');(OUT/'data'/f'{name}.json').write_text(json.dumps(rows,indent=2)+'\n')
    FIGURES[name]=dict(rec,rendering='Unchanged; already matches Thinking PCA renderer and cameras')

if __name__=='__main__':
    style=module('thinking_style',ROOT/'figures/thinking_appendix_style_20260913/build_figures.py');style.OUT=OUT;style.setup()
    helper=module('curve_helpers',ROOT/'figures/enumeration_midlayer_20260917/build_causal.py')
    head_scores();readouts();dose();update();readout();pca();read(Path(__file__))
    (OUT/'manifest.json').write_text(json.dumps(dict(status='PASS',figures=FIGURES,source_sha256=SOURCES,statistics_recomputed=False,inference=False,font_profile=dict(family='Times New Roman',title=11,axis=10,ticks_legend=9,selected_layer_annotation=8,width_inches=6.5)),indent=2)+'\n')
    print('PASS: eight aligned figures; no model inference, refitting or resampling.')
