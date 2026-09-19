"""Paper-layout revision. Read archived results; never run models or alter sources."""
from pathlib import Path
import hashlib
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from camera_study import draw as draw_pca, CMAP

OUT=Path(__file__).resolve().parent
WORK=OUT.parents[1]
OLD=WORK/'figures/synthetic_appendix_aurora'
PREV=WORK/'figures/synthetic_appendix_review_20260908'
DATA=WORK/'synthetic/work'
PAPER=WORK/'runs/paper_figures/figures/synthetic_appendix/paper_20260908'
NT,T,GRAY,INK,VIOLET,ORANGE='#B52F6B','#007EAB','#8190A5','#161923','#6750E8','#C77939'
SOURCES,FIGURES,CHECKS={ },[],[]
CAMERA=(15,135)  # User-selected rightmost view from the camera contact sheet.

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def source(p):
    SOURCES[str(p.relative_to(WORK))]=sha(p)
    return p

def read(p):
    return pd.read_csv(source(p))

def export(d,name):
    d.to_csv(OUT/'plot_data'/name,index=False)

def register(stem,title):
    FIGURES.append(dict(stem=stem,title=title))
    shutil.copy2(OUT/f'{stem}.pdf',PAPER/f'{stem}.pdf')

def save(fig,stem,title):
    fig.canvas.draw()
    for ext in ['pdf','png','svg']:
        fig.savefig(OUT/f'{stem}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    register(stem,title)

def copy(folder,old,new,title):
    for ext in ['pdf','png','svg']:
        shutil.copy2(source(folder/f'{old}.{ext}'),OUT/f'{new}.{ext}')
    register(new,title)

def panel(ax,title,ylabel=None):
    ax.set_title(title,loc='left',fontsize=11,pad=8)
    if ylabel: ax.set_ylabel(ylabel)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#E7E8EE',lw=.6)
    ax.set_axisbelow(True)

def box(ax,xy,w,h,text,color=INK,fill='#F5F6F9',size=10):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(xy,w,h,boxstyle='round,pad=0.014,rounding_size=0.022',
                 linewidth=.7,edgecolor=color,facecolor=fill,transform=ax.transAxes))
    ax.text(xy[0]+w/2,xy[1]+h/2,text,ha='center',va='center',fontsize=size,
            color=color,transform=ax.transAxes,linespacing=1.3)

def arrow(ax,start,end,color=GRAY):
    ax.annotate('',xy=end,xytext=start,xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>',lw=1,color=color,shrinkA=3,shrinkB=3))

def retrieval():
    nt=read(DATA/'v58_protected_L1H2_controls_20260908_v2/selected_vs_controls.csv')
    nt=nt[nt.top_k.le(4)&nt.metric.eq('ar_accuracy')].sort_values('top_k')
    t=read(DATA/'v58_top1to8_final_query_20260908/selected_vs_all_controls.csv')
    t=t[t['mode'].eq('thinking')&t.scope.eq('sustained')&t.support.eq('primary')&t.metric.eq('next_marker_correct')].sort_values('top_k')
    assert list(nt.top_k)==[1,2,3,4] and list(t.top_k)==list(range(1,9))
    assert nt.inputs.eq(100).all() and t.inputs.eq(88).all()
    export(nt,'02_nonthinking.csv'); export(t,'02_thinking.csv')
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.48))
    fig.subplots_adjust(left=.085,right=.985,bottom=.30,top=.85,wspace=.30)
    for ax,d,color,title,base,limits in [
        (axes[0],nt,NT,'A. Non-thinking: final count',21,(0,40)),
        (axes[1],t,T,'B. Thinking: next marker',100*86/88,(60,100))]:
        ax.fill_between(d.top_k,100*d.control_min,100*d.control_max,color=GRAY,alpha=.2,lw=0)
        ax.plot(d.top_k,100*d.selected,'o-',color=color,ms=4)
        ax.plot(d.top_k,100*d.control_mean,'s--',color=GRAY,ms=3.3)
        ax.axhline(base,color=INK,ls=':',lw=1)
        panel(ax,title,'Accuracy (%)')
        ax.set(xlabel='Ablated heads',ylim=limits,yticks=np.arange(limits[0],limits[1]+1,10),
               xlim=(d.top_k.min()-.2,d.top_k.max()+.2),xticks=d.top_k)
    fig.legend([Line2D([],[],color=NT,marker='o'),Line2D([],[],color=T,marker='o'),
                Line2D([],[],color=GRAY,ls='--',marker='s'),Line2D([],[],color=INK,ls=':')],
               ['Broad heads','Targeted heads','Control mean (range)','Clean'],
               ncol=4,loc='lower center',bbox_to_anchor=(.52,.02),fontsize=9,
               handlelength=1.5,columnspacing=1.2,handletextpad=.4)
    CHECKS.append(dict(figure='02_retrieval_ablation',y_limits=[[0,40],[60,100]],equal_span_pp=40))
    save(fig,'02_retrieval_ablation','Retrieval-head ablation')

def transport():
    raw=read(DATA/'v58_final/analysis/v58_unified_legacy_20260905/thinking/transport_trials.csv')
    d=raw.groupby(['condition','top_k','repeat'])[['clean_margin','corrupt_margin','margin','restoration']].mean().reset_index()
    export(d,'03_value_transport.csv')
    fig=plt.figure(figsize=(6.5,2.7))
    a=fig.add_axes([.01,.14,.47,.72]); a.axis('off')
    a.set_title('A. Change the source, restore its value',loc='left',fontsize=11,pad=8)
    box(a,(.05,.76),.38,.18,'Clean source: a',T,'#EDF7FA')
    box(a,(.56,.76),.38,.18,'Changed to: b',ORANGE,'#FFF5ED')
    arrow(a,(.44,.85),(.55,.85))
    box(a,(.16,.33),.68,.24,'Keep b in the input\nRestore only clean V(a)',T,'#EDF7FA')
    arrow(a,(.75,.74),(.62,.58),ORANGE)
    arrow(a,(.24,.74),(.37,.58),T)
    a.text(.5,.14,'Same trace query: prefer a or b?',ha='center',va='center',fontsize=10)
    a.text(.5,-.02,'Illustrative a / b; total count unchanged',ha='center',fontsize=9,color=GRAY)
    b=fig.add_axes([.62,.37,.36,.49])
    for cond,color,style,label in [('value_selected',T,'-','Selected'),('value_control',GRAY,'--','Controls')]:
        g=d[d.condition.eq(cond)].groupby('top_k').margin.mean()
        b.plot(g.index,g.values,'o'+style,color=color,label=label,ms=4)
    for cond,color,label in [('clean',VIOLET,'Clean'),('damaged',ORANGE,'Changed source')]:
        b.axhline(d.loc[d.condition.eq(cond),'margin'].iloc[0],color=color,ls=':',label=label,lw=1.4)
    b.axhline(0,color=INK,lw=.65)
    panel(b,'B. Original-marker preference',r'Logit(a) $-$ logit(b)')
    b.set(xlabel='Restored heads',xticks=[1,2,4],xlim=(.85,4.15),ylim=(-5,11),yticks=[-4,0,4,8])
    fig.legend(*b.get_legend_handles_labels(),loc='lower right',bbox_to_anchor=(.99,.01),ncol=2,
               fontsize=9,handlelength=1.5,columnspacing=1.,labelspacing=.35)
    save(fig,'03_source_value_restoration','Does the head carry the source character?')

def readability():
    d=read(OLD/'plot_data/04_ncc_by_layer.csv')
    selected=read(OLD/'plot_data/04_ncc_discovery_selection.csv')
    export(d,'04_ncc_by_layer.csv'); export(selected,'04_ncc_layer_selection.csv')
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.05))
    fig.subplots_adjust(left=.09,right=.98,bottom=.32,top=.80,wspace=.32)
    for j,ax in enumerate(axes):
        for mode,color,label in [('nonthinking',NT,'Non-thinking'),('thinking',T,'Thinking')]:
            f=d[d['mode'].eq(mode)&d.layer.ge(1)&d.endpoint.str.contains('answer_query').eq(j==1)].sort_values('layer')
            ax.plot(f.layer,100*f.confirmation_ncc_balanced_accuracy,'o-',color=color,label=label,ms=3.5)
            s=selected[selected['mode'].eq(mode)&selected.endpoint.str.contains('answer_query').eq(j==1)].iloc[0]
            ax.scatter(s.selected_layer,100*s.confirmation_value,s=66,marker='s',facecolors='none',edgecolors=color,lw=1.3)
        ax.axhline(10,color=GRAY,ls=':',label='Chance (10%)',lw=1)
        panel(ax,['A. Running index','B. Final count'][j],'NCC accuracy (%)')
        ax.set(xlabel='Post-block layer',xticks=[1,2,3,4],ylim=(0,105),yticks=[0,25,50,75,100])
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.54,-.005),ncol=3,
               fontsize=9,handlelength=1.4,handletextpad=.35,columnspacing=1.)
    save(fig,'04_count_readability','Count readability across layers')

def geometry():
    d=read(OLD/'plot_data/11_pca_common_l2_points.csv')
    export(d,'05_pca_l2_points.csv')
    fig=plt.figure(figsize=(6.5,4.65))
    specs=[('nonthinking',False,'A. Non-thinking: running index'),
           ('thinking',False,'B. Thinking: running index'),
           ('nonthinking',True,'C. Non-thinking: final count'),
           ('thinking',True,'D. Thinking: final count')]
    variances=[]
    for i,(mode,answer,title) in enumerate(specs):
        f=d[d['mode'].eq(mode)&d.endpoint.str.contains('answer_query').eq(answer)]
        assert len(f)==(100 if answer else 550) and f.layer.eq(2).all()
        assert f.prompt_sha256.nunique()==100
        x=.01 if i%2==0 else .515; y=.555 if i<2 else .12
        ax=fig.add_axes([x,y,.42,.36],projection='3d',computed_zorder=False)
        draw_pca(ax,f,CAMERA)
        ax.tick_params(labelsize=8.8,pad=0)
        ax.set_xlabel('PC1',labelpad=0,fontsize=9)
        ax.set_ylabel('PC2',labelpad=0,fontsize=9)
        ax.set_zlabel('')
        ax.text2D(1.15,.54,'PC3',transform=ax.transAxes,fontsize=9)
        # Keep headings clear of all 3D projections and labels.
        fig.text(x+.01,y+.395,title,fontsize=10.5,color=NT if mode=='nonthinking' else T)
        pct=100*sum(f[f'pc{k}_variance_ratio'].iloc[0] for k in [1,2,3])
        variances.append(dict(mode=mode,endpoint='final_count' if answer else 'running_index',
                              points=len(f),explained_variance_percent=pct,elevation=CAMERA[0],azimuth=CAMERA[1]))
    cax=fig.add_axes([.31,.065,.38,.017])
    cb=fig.colorbar(ScalarMappable(norm=Normalize(1,10),cmap=CMAP),cax=cax,orientation='horizontal')
    cb.set_ticks([1,2,4,6,8,10]); cb.ax.tick_params(labelsize=9,length=2,pad=2)
    fig.text(.5,.012,'Running index / final count',ha='center',fontsize=9.5)
    export(pd.DataFrame(variances),'05_pca_variance_and_camera.csv')
    CHECKS.append(dict(figure='05_geometry_l2_3d',all_states_retained=True,
                       camera=CAMERA,projection='orthographic',equal_coordinate_units=True))
    save(fig,'05_geometry_l2_3d','Count geometry at common layer L2')

def sources():
    d=read(OLD/'plot_data/06_next_marker_sources.csv')
    export(d,'06_next_marker_sources_all_conditions.csv')
    fig=plt.figure(figsize=(6.5,2.44))
    a=fig.add_axes([.015,.14,.37,.70]); a.axis('off')
    a.set_title('A. Which states are zeroed?',loc='left',fontsize=11,pad=8)
    box(a,(.03,.69),.92,.23,'Prompt:  ... a ... b ... a ...',T,'#EDF7FA')
    box(a,(.03,.34),.92,.23,'Trace:  <Sep> a   <Sep> b',VIOLET,'#F3F1FD')
    a.text(.97,.02,'Next marker: ?',ha='right',fontsize=10)
    a.text(.48,-.075,'Token positions stay fixed',ha='center',fontsize=9,color=GRAY)
    ax=fig.add_axes([.52,.29,.46,.55])
    arms=['records','recent','history']; x=np.arange(3); w=.32
    for ctrl,off,color,label in [(False,-w/2,T,'Source states'),(True,w/2,GRAY,'Ordinary states')]:
        vals=[100*float(d.loc[d.arm.eq(arm)&d.ordinary_control.eq(ctrl),'accuracy'].iloc[0]) for arm in arms]
        bars=ax.bar(x+off,vals,w,color=color,label=label)
        ax.bar_label(bars,fmt='%.0f',padding=2,fontsize=9)
    ax.axhline(96,color=INK,ls=':',lw=1,label='Clean (96%)')
    panel(ax,'B. Next-marker prediction','Accuracy (%)')
    ax.set(xticks=x,xticklabels=['Prompt\ntargets','Recent\ntrace item','Full trace\nhistory'],ylim=(0,115),yticks=[0,25,50,75,100])
    fig.legend(*ax.get_legend_handles_labels(),loc='lower right',bbox_to_anchor=(.99,.005),
               ncol=3,fontsize=9,handlelength=1.1,columnspacing=1.,handletextpad=.4)
    save(fig,'06_progress_sources','Which information supports the next marker?')

def continuation():
    d=read(OLD/'plot_data/06_continuation_effects.csv')
    d=d[d.scope.eq('item_span_w2')].copy()
    assert d.prompts.eq(8).all() and list(d.pairs)==[26,30,26,20]
    export(d,'07_donor_continuation.csv')
    fig=plt.figure(figsize=(6.5,2.6))
    a=fig.add_axes([.01,.15,.38,.70]); a.axis('off')
    a.set_title('A. Transfer a completed trace item',loc='left',fontsize=11,pad=8)
    box(a,(.03,.72),.92,.23,'Donor at another progress step',VIOLET,'#F3F1FD')
    box(a,(.03,.21),.92,.23,'Receiver: same visible prefix',T,'#EDF7FA')
    arrow(a,(.48,.70),(.48,.46),VIOLET)
    a.text(.52,.57,'copy L1 states',fontsize=9.5,ha='left',va='center',color=VIOLET)
    a.text(.48,.04,'Two positions: <Sep> + marker',fontsize=9.5,ha='center')
    a.text(.48,-.08,'Then generate the continuation',fontsize=9,color=GRAY,ha='center')
    ax=fig.add_axes([.53,.31,.45,.54]); x=np.arange(4); w=.34
    for col,off,color,label in [('treatment_mean',-w/2,T,'Donor state'),('control_mean',w/2,GRAY,'Orthogonal control')]:
        bars=ax.bar(x+off,100*d[col],w,color=color,label=label)
        ax.bar_label(bars,fmt='%.1f',fontsize=9,padding=2)
    panel(ax,'B. Does the output follow the donor?','Donor-prefix match (%)')
    ax.set(xticks=x,xticklabels=['1','2','3','4'],xlabel='Number of subsequent markers',ylim=(0,60),yticks=[0,20,40,60])
    fig.legend(*ax.get_legend_handles_labels(),loc='lower right',bbox_to_anchor=(.99,.015),ncol=2,
               fontsize=9,handlelength=1.3,columnspacing=1.)
    save(fig,'07_progress_continuation','Does a transplanted item change the continuation?')

def answer():
    a=read(OLD/'plot_data/07_answer_sources.csv'); b=read(OLD/'plot_data/07_answer_state_donor.csv')
    export(a,'08_answer_sources_all_modes.csv'); export(b,'08_answer_state_donor.csv')
    fig,axs=plt.subplots(1,2,figsize=(6.5,2.85))
    fig.subplots_adjust(left=.085,right=.985,bottom=.33,top=.84,wspace=.35)
    ax=axs[0]; x=np.arange(2); w=.31
    f=a[a['mode'].eq('thinking')]
    for arms,off,color,label in [(['records','trace'],-w/2,T,'Source states'),
                                 (['ordinary_records_budget','ordinary_trace_budget'],w/2,GRAY,'Ordinary states')]:
        vals=[100*f.loc[f.arm.eq(arm),'accuracy'].iloc[0] for arm in arms]
        bars=ax.bar(x+off,vals,w,color=color,label=label)
        ax.bar_label(bars,fmt='%.0f',padding=2,fontsize=9)
    ax.axhline(95,color=INK,ls=':',lw=1,label='Clean (95%)')
    panel(ax,'A. Thinking: answer sources','Answer accuracy (%)')
    ax.set(xticks=x,xticklabels=['Prompt targets','Generated trace'],ylim=(0,115),yticks=[0,25,50,75,100])
    ax.text(.5,-.27,'States zeroed at these positions',ha='center',transform=ax.transAxes,fontsize=9)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.05,.005),ncol=3,
               fontsize=8.5,columnspacing=.7,handlelength=1.,handletextpad=.35)
    ax=axs[1]
    for mode,color,label in [('nonthinking',NT,'Non-thinking'),('thinking',T,'Thinking')]:
        f=b[b['mode'].eq(mode)].sort_values('layer')
        assert f.pairs.eq(180).all()
        ax.plot(f.layer,100*f.adoption,'o-',color=color,label=label)
    panel(ax,'B. Transfer the answer state','Outputs donor count (%)')
    ax.set(xticks=[1,2,3,4],xlabel='Layer of state transplant',ylim=(0,105),yticks=[0,25,50,75,100])
    ax.legend(loc='center left',fontsize=9,handlelength=1.4)
    save(fig,'08_answer_readout','What supports the final answer?')

def attention_dynamics():
    d=read(OLD/'plot_data/09_attention_roles.csv')
    export(d,'09_attention_roles.csv')
    cmap=LinearSegmentedColormap.from_list('aurora_warm',
        ['#161923','#40204F','#963878','#E65D91','#F7A35C','#F9EDAD'])
    heads=[(l,h) for l in range(1,5) for h in range(8)]
    broad_max=float(d.broad.max())
    fig=plt.figure(figsize=(6.5,2.80))
    images=[]
    for i,(mode,role,title) in enumerate([
        ('nonthinking','broad','A. Non-thinking\nBroad'),
        ('thinking','broad','B. Thinking\nBroad'),
        ('thinking','targeted','C. Thinking\nTargeted'),
        ('thinking','successor','D. Thinking\nSuccessor')]):
        ax=fig.add_axes([.075+.235*i,.34,.195,.49])
        m=d[d['mode'].eq(mode)].pivot(index=['layer','head'],columns='step',values=role).reindex(heads)
        assert m.shape==(32,101) and np.isfinite(m.to_numpy()).all()
        im=ax.pcolormesh(m.columns.to_numpy(),np.arange(32),m.to_numpy(),cmap=cmap,
                         vmin=0,vmax=broad_max if i<2 else 1,shading='nearest',rasterized=True)
        images.append(im)
        ax.set_title(title,loc='left',fontsize=10.3,pad=7,linespacing=1.2)
        ax.set(ylim=(31.5,-.5),xlim=(0,10000),xticks=[0,5000,10000],xticklabels=['0','5k','10k'])
        ax.set_yticks(range(0,32,4),[f'L{l}H{h}' for l,h in heads[::4]] if i==0 else [])
        ax.tick_params(axis='y',length=0,pad=3,labelsize=8.5)
        ax.tick_params(axis='x',labelsize=9,length=3,pad=3)
        for y in [7.5,15.5,23.5]: ax.axhline(y,color='white',lw=.5,alpha=.7)
        ax.axvline(1500,color='white',lw=.6,ls=':',alpha=.8)
    fig.text(.525,.235,'Training step',ha='center',fontsize=10)
    for image,x,label,ticks in [(images[0],.155,'Broad score',[0,.025,.05]),
                                (images[2],.635,'Targeted / successor score',[0,.5,1])]:
        cax=fig.add_axes([x,.12,.235,.026])
        cb=fig.colorbar(image,cax=cax,orientation='horizontal',ticks=ticks)
        cb.ax.tick_params(labelsize=9,length=2,pad=2)
        cb.set_label(label,fontsize=9,labelpad=2)
    CHECKS.append(dict(figure='09_attention_dynamics',matrix_shape=[32,101],
                       shared_broad_scale=[0,broad_max],targeted_successor_scale=[0,1]))
    save(fig,'09_attention_dynamics','Attention roles during training')

def metric_dynamics():
    gs=read(PREV/'plot_data/08_ncc_dynamics.csv')
    ts=read(PREV/'plot_data/08_value_dynamics.csv')
    export(gs,'10_ncc_dynamics.csv'); export(ts,'10_value_dynamics.csv')
    fig=plt.figure(figsize=(6.5,2.28))
    for i,title in enumerate(['A. Running index','B. Final count','C. Value patching']):
        ax=fig.add_axes([(.49+i*2.15)/6.5,.29,1.65/6.5,.52])
        if i<2:
            for mode,color,name in [('nonthinking',NT,'NT'),('thinking',T,'T')]:
                d=gs[gs['mode'].eq(mode)&gs.endpoint.str.contains('answer_query').eq(i==1)].sort_values('step')
                assert d.layer.nunique()==1
                ax.plot(d.step,100*d.ncc_correct,'o-',color=color,ms=3,label=f'{name} (L{int(d.layer.iloc[0])})')
            ax.axhline(10,color=GRAY,ls=':',lw=.8)
            ax.set(ylim=(-3,105),yticks=[0,25,50,75,100])
            ylabel='NCC accuracy (%)'
        else:
            for cond,color,ls,label in [('value_selected',T,'-','Selected Top-2'),('value_control',GRAY,'--','Controls')]:
                d=ts[ts.condition.eq(cond)&ts.top_k.eq(2)].groupby('step').restoration.mean()
                ax.plot(d.index,d.values,'o'+ls,color=color,ms=3,label=label)
            ax.axhline(0,color=GRAY,lw=.6)
            ylabel='Margin gain'
        ax.set_xscale('symlog',linthresh=200,linscale=.65)
        ax.axvline(1500,color=GRAY,ls=':',lw=.9)
        ax.set(xlim=(-20,11500),xlabel='Training step',xticks=[0,200,1000,10000],xticklabels=['0','200','1k','10k'])
        panel(ax,title,ylabel)
        ax.tick_params(labelsize=8.5)
        ax.legend(loc=['upper right','center right','upper left'][i],fontsize=8.2,handlelength=1.,
                  handletextpad=.3,labelspacing=.2,frameon=True,facecolor='white',edgecolor='none',framealpha=1,borderpad=.2)
    save(fig,'10_readability_transport_dynamics','Readability and value patching during training')

def main():
    (OUT/'plot_data').mkdir(exist_ok=True); PAPER.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'mathtext.fontset':'stix',
        'axes.labelsize':10,'xtick.labelsize':9.5,'ytick.labelsize':9.5,'legend.fontsize':9,
        'legend.frameon':False,'axes.linewidth':.7,'lines.linewidth':1.5,'lines.markersize':4,
        'pdf.fonttype':42,'svg.fonttype':'none','text.color':INK,'axes.labelcolor':INK,'axes.titlecolor':INK})
    for p in [WORK/'runs/paper_figures/main.tex',WORK/'runs/paper_figures/appendix/synthetic.tex',
              PREV/'manifest.json',PREV/'appendix_figures_review.pdf']:
        source(p)
    copy(OLD,'01_training_and_behavior','01_training_and_behavior','Training and counting behavior')
    retrieval(); transport()
    readability()
    geometry(); sources(); continuation(); answer()
    attention_dynamics()
    metric_dynamics()
    for p,h in SOURCES.items(): assert sha(WORK/p)==h,p
    manifest=dict(figures=FIGURES,source_sha256=SOURCES,checks=CHECKS,model_runs=False,
                  manuscript_modified=False,old_files_preserved=True,camera=CAMERA,
                  omitted_from_preview=['count-shift panel','L4 geometry','2D geometry','count-direction injection','successor figure'],
                  successor='One concluding sentence, original protocol explicitly identified.',
                  rendered_sha256={f'{f["stem"]}.{ext}':sha(OUT/f'{f["stem"]}.{ext}') for f in FIGURES for ext in ['pdf','png','svg']})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Built',len(FIGURES),'paper figures; sources unchanged.')

if __name__=='__main__': main()
