"""Reproduce the paper's three-panel Synthetic v58 figure without retraining.

Run from the counting workspace:
python figures/synthetic_training_dynamics_three-panel/plot_synthetic_section.py
Outputs are placed beside this script. All plotted summaries are also exported.
"""
from pathlib import Path
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent/'synthetic'
DATA = ROOT/'work/v58_final/analysis/v58_unified_legacy_20260905'
REGISTRY = ROOT/'work/v58_final/analysis/v58_alignment_supplement_20260905/input_registry.csv'
MODES = ['nonthinking', 'thinking']
HEADS = [(l,h) for l in range(1,5) for h in range(8)]
# Warm heatmap: dark purple through pink and orange to pale yellow.
AURORA_STOPS = ['#161923', '#40204F', '#963878', '#E65D91', '#F7A35C', '#F9EDAD']
AURORA_CMAP = LinearSegmentedColormap.from_list('synthetic_warm', AURORA_STOPS, N=256)
AURORA_LINES = {'nonthinking': '#B52F6B', 'thinking': '#007EAB'}


def load():
    registry = pd.read_csv(REGISTRY)
    confirmation = registry.loc[registry.split.eq('confirmation')]
    assert confirmation.groupby('count').size().to_dict() == {n:10 for n in range(1,11)}
    keys=set(confirmation.key)
    attention, behavior, paths = [], [], [REGISTRY]
    for mode in MODES:
        ap=DATA/mode/'dynamics_attention_summary.csv'
        bp=DATA/mode/'dynamics_behavior_trials.csv'
        paths.extend([ap,bp])
        a=pd.read_csv(ap)
        b=pd.read_csv(bp, usecols=['mode','step','count','prompt_sha256','ar_accuracy'])
        assert set(a['mode']) == {mode} and set(b['mode']) == {mode}
        assert len(a)==101*32 and not a.duplicated(['step','layer','head']).any()
        assert set(a.step)==set(range(0,10001,100))
        for _,f in b.groupby('step'):
            assert len(f)==100 and set(f.prompt_sha256)==keys
            assert f.groupby('count').size().to_dict()=={n:10 for n in range(1,11)}
        assert len(b.step.unique())==11
        attention.append(a)
        behavior.append(b.groupby(['mode','step']).ar_accuracy.mean().reset_index())
    return pd.concat(attention),pd.concat(behavior),paths


def matrix(data, mode, role):
    f=data.loc[data['mode'].eq(mode)]
    m=f.pivot(index=['layer','head'],columns='step',values=role).reindex(HEADS)
    assert np.isfinite(m.to_numpy()).all()
    assert ((m.to_numpy()>=0)&(m.to_numpy()<=1)).all()
    return m


def draw(attention, behavior, full_heads=False):
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,
        'mathtext.fontset':'stix',
        'axes.labelsize':11,'axes.titlesize':11,'xtick.labelsize':10,
        'ytick.labelsize':10,'legend.fontsize':10,'axes.linewidth':.6,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(12.5,5.2) if full_heads else (9.0,2.8),layout='constrained')
    fig.get_layout_engine().set(h_pad=.012, w_pad=.035)
    gs=fig.add_gridspec(1,3,width_ratios=[1,1.13,1.13],wspace=.045)
    axes=[fig.add_subplot(gs[i]) for i in range(3)]
    for mode,color,name in [('nonthinking',AURORA_LINES['nonthinking'],'Non-thinking'),('thinking',AURORA_LINES['thinking'],'Thinking')]:
        f=behavior.loc[behavior['mode'].eq(mode)]
        axes[0].plot(f.step,100*f.ar_accuracy,'o-',color=color,label=name,lw=1.4,ms=3.2)
    axes[0].set(title='(a) Answer accuracy',ylabel='Free-running accuracy (%)',ylim=(0,103),yticks=range(0,101,20))
    axes[0].grid(color='#e5e7eb',lw=.5)
    axes[0].legend(loc='upper left',frameon=True,edgecolor='#ddd',framealpha=.95,handlelength=1.4,borderpad=.35)
    broad=matrix(attention,'nonthinking','broad')
    targeted=matrix(attention,'thinking','targeted')
    broad_max=math.ceil(float(broad.to_numpy().max())*100)/100
    for ax,m,title,vmax,label in [
        (axes[1],broad,'(b) Non-thinking: broad retrieval',broad_max,r'Broad score $B_h$'),
        (axes[2],targeted,'(c) Thinking: targeted retrieval',1.,r'Targeted needle mass $T_h$')]:
        # Fixed physical order and raw score; no normalization across heads/steps.
        im=ax.pcolormesh(m.columns.to_numpy(),np.arange(32),m.to_numpy(),
            shading='nearest',cmap=AURORA_CMAP,vmin=0,vmax=vmax,rasterized=True,
            edgecolors='none',linewidth=0,antialiased=False)
        ax.set(title=title,ylim=(31.5,-.5))
        for y in [7.5,15.5,23.5]: ax.axhline(y,color='white',alpha=.5,lw=.45)
        if full_heads:
            ax.set_yticks(range(32),[f'L{l}H{h}' for l,h in HEADS])
            ax.tick_params(axis='y',labelsize=8.5,length=0,pad=3)
        else:
            ax.set_yticks([3.5,11.5,19.5,27.5],['L1','L2','L3','L4'])
            ax.tick_params(axis='y',length=0,pad=3)
        bar=fig.colorbar(im,ax=ax,pad=.025,fraction=.046,aspect=24)
        bar.set_ticks([0,vmax/2,vmax])
        bar.ax.tick_params(labelsize=10,length=2,pad=2)
        bar.set_label(label,fontsize=10,labelpad=3)
    for i,ax in enumerate(axes):
        ax.set_xlim(0,10000)
        ax.set_xticks([0,2500,5000,7500,10000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_: '0' if x==0 else f'{x/1000:g}k'))
        ax.axvline(1500,color='#666' if i==0 else '#e0f2fe',ls=(0,(3,2)),lw=.7,alpha=.9)
    fig.supxlabel('Training steps',fontsize=11,fontfamily='Times New Roman')
    name='synthetic_section_full_heads' if full_heads else 'synthetic_section'
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=400)
    plt.close(fig)
    return broad_max


def main():
    attention, behavior, sources=load()
    vmax=draw(attention,behavior)
    draw(attention,behavior,full_heads=True)
    behavior.to_csv(OUT/'synthetic_section_accuracy.csv',index=False)
    wanted=attention.loc[attention['mode'].eq('thinking'),['mode','step','layer','head','targeted']].rename(columns={'targeted':'score'})
    wanted['role']='targeted'
    broad=attention.loc[attention['mode'].eq('nonthinking'),['mode','step','layer','head','broad']].rename(columns={'broad':'score'})
    broad['role']='broad'
    pd.concat([broad,wanted]).to_csv(OUT/'synthetic_section_head_scores.csv',index=False)
    manifest={'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'confirmation_prompts':100,'prompts_per_count':10,'training_seeds':1,
        'attention_checkpoints':101,'behavior_checkpoints':11,
        'raw_scores':True,'normalization_across_heads':False,
        'broad_color_limits':[0,vmax],'targeted_color_limits':[0,1],
        'layer_order':'L1 to L4, top to bottom','head_order':'H0 to H7 within each layer, top to bottom',
        'main_size_inches':[9.0,2.8],'full_head_size_inches':[12.5,5.2],
        'title_label_fontsize_pt':11,'tick_legend_fontsize_pt':10,
        'full_head_label_fontsize_pt':8.5,'vertical_layout_padding_inches':.012,
        'font_family':'Times New Roman','math_font_family':'STIX','shared_x_label':'Training steps',
        'palette':'Warm pink-orange-yellow','heatmap_color_stops':AURORA_STOPS,'line_colors':AURORA_LINES,
        'png_dpi':400,'error_bars':'none; descriptive fixed panel, single training seed'}
    (OUT/'synthetic_section_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'status':'passed','outputs':str(OUT),'broad_max':vmax,
        'final_accuracy':behavior.loc[behavior.step.eq(10000)].to_dict('records')},indent=2))


if __name__=='__main__':
    main()
