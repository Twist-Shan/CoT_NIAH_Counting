"""Add a reviewed appendix figure set; preserve all older figures and data."""
from pathlib import Path
import sys as _font_sys
_font_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_style import preview_font
import hashlib
import html
import json
import shutil
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parent
WORK=OUT.parents[1]
OLD=WORK/'figures/synthetic_appendix_aurora'
DATA=WORK/'synthetic/work'
LEGACY=DATA/'v58_final/analysis/v58_unified_legacy_20260905'
EXTRA=DATA/'v58_final/analysis/v58_unified_additional_20260905'
PAPER=WORK/'runs/paper_figures/figures/synthetic_appendix/review_20260908'
NT,T,GRAY,INK,VIOLET,ORANGE='#B52F6B','#007EAB','#8190A5','#161923','#6750E8','#C77939'
FIGURES,SOURCES,CHECKS=[],{},[]
OLD_META={f['stem']:f for f in json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))['figures']}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source(p):
    SOURCES[str(p.relative_to(WORK))]=digest(p)
    return p


def read(p):
    return pd.read_csv(source(p))


def register(stem,title,caption,note,group='main',width=1.):
    FIGURES.append(dict(stem=stem,title=title,caption=caption,note=note,group=group,width_fraction=width))
    shutil.copy2(OUT/f'{stem}.pdf',PAPER/f'{stem}.pdf')


def copied(old_stem,new_stem,title,caption=None,note='',group='main'):
    for ext in ['pdf','png','svg']:
        shutil.copy2(source(OLD/f'{old_stem}.{ext}'),OUT/f'{new_stem}.{ext}')
    if caption is None:
        caption=OLD_META[old_stem]['caption']
    register(new_stem,title,caption,note,group)


def panel(ax,title,ylabel=None):
    ax.set_title(title,loc='left',fontsize=9.5,pad=6)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis='y',color='#E8E8ED',linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)


def save(fig,stem,title,caption,note,group='main',width=1.):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    from matplotlib.text import Text
    clipped=[]
    for obj in fig.findobj(Text):
        if not obj.get_visible() or not obj.get_text():
            continue
        if obj.axes is not None and obj not in obj.axes.texts and obj not in [obj.axes.xaxis.label,obj.axes.yaxis.label,obj.axes._left_title]:
            continue
        b=obj.get_window_extent(renderer)
        if b.width>0 and (b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.x1+1 or b.y1 > fig.bbox.y1+1):
            clipped.append(obj.get_text())
    CHECKS.append({'figure':stem,'outside_canvas_text':clipped})
    assert not clipped,(stem,clipped)
    for ext in ['pdf','png','svg']:
        fig.savefig(OUT/f'{stem}.{ext}',dpi=300,facecolor='white',bbox_inches='tight',pad_inches=.055)
    plt.close(fig)
    register(stem,title,caption,note,group,width)


def retrieval():
    nt=read(DATA/'v58_protected_L1H2_controls_20260908_v2/selected_vs_controls.csv')
    nt=nt[nt.top_k.le(4)].copy()
    t=read(DATA/'v58_top1to8_final_query_20260908/selected_vs_all_controls.csv')
    t=t[(t['mode']=='thinking')&(t.scope=='sustained')&(t.support=='primary')&(t.metric=='next_marker_correct')]
    assert set(t.top_k)==set(range(1,9)) and set(nt.top_k)==set(range(1,5))
    nt.to_csv(OUT/'plot_data/02_broad_top1to4.csv',index=False)
    t.to_csv(OUT/'plot_data/02_targeted_top1to8.csv',index=False)
    fig=plt.figure(figsize=(5.5,2.45))
    specs=[(nt[nt.metric=='ar_accuracy'],NT,'A. Count accuracy','Accuracy (%)',(0,30),21),
           (nt[nt.metric=='normalized_count_shift'],NT,'B. Count shift','Normalized shift (%)',(0,80),0),
           (t,T,'C. Next marker','Accuracy (%)',(0,102),100*86/88)]
    for i,(d,color,title,ylabel,limits,baseline) in enumerate(specs):
        ax=fig.add_axes([(0.46+i*1.81)/5.5,.61/2.45,1.33/5.5,1.45/2.45])
        d=d.sort_values('top_k')
        ax.fill_between(d.top_k,100*d.control_min,100*d.control_max,color=GRAY,alpha=.2,linewidth=0)
        ax.plot(d.top_k,100*d.selected,'o-',color=color,markersize=3.7)
        ax.plot(d.top_k,100*d.control_mean,'s--',color=GRAY,markersize=3.1)
        ax.axhline(baseline,color=INK,linestyle=':',linewidth=1.2)
        panel(ax,title,ylabel)
        ax.set(xlabel='Ablated heads',ylim=limits,xlim=(.8,4.2) if i<2 else (.6,8.4),
               xticks=[1,2,3,4] if i<2 else [1,2,4,6,8])
        ax.set_yticks([0,10,20,30] if i==0 else [0,20,40,60,80] if i==1 else [0,25,50,75,100])
    fig.legend([Line2D([],[],color=NT,marker='o'),Line2D([],[],color=T,marker='o'),
                Line2D([],[],color=GRAY,ls='--',marker='s'),Line2D([],[],color=INK,ls=':')],
               ['Broad selected','Targeted selected','Control mean','Clean'],loc='lower center',
               bbox_to_anchor=(.51,.012),ncol=4,fontsize=8.1,handlelength=1.25,columnspacing=.8,handletextpad=.35)
    caption=('Sustained pre-O head ablation from the original answer query (A/B, non-thinking) or the final retrieval query (C, thinking), until EOS or 64 new tokens. '
             'A: final-count accuracy on all 100 inputs. B: absolute change from the clean count divided by the true count, on the same 100 inputs. '
             'C: next-marker accuracy on the fixed 88 inputs with count at least two and an available clean final query; final-count errors do not affect this metric. '
             'The globally ranked selected banks are unchanged. Non-thinking controls retain L1H2 to preserve its approximately constant answer-readout contribution; this is a diagnostic revision after the original control audit. '
             'All controls match per-layer head counts and use the minimum necessary overlap; every feasible combination is included. '
             'Non-thinking control counts for K=1/2/3/4 are 6/10/4/4, with one shared head at K=4. Thinking control counts are 7/15/10/1/10/15/7/49, with overlap 0/0/0/0/2/4/6/6. '
             'Shading spans all control sets, not confidence intervals. All non-thinking numeric answers are parseable; EOS failures are retained separately. '
             'Both full K=1-8 sweeps remain archived; the displayed ranges are non-thinking K=1-4 and thinking K=1-8. The three panels use different vertical scales.')
    save(fig,'02_retrieval_ablation','Retrieval-head ablation',caption,
         '新版持续消融。Broad 显示 Top-1～4，Targeted 只显示 next-marker 的 Top-1～8；NT control 保留 L1H2。')


def value_transport():
    raw=read(LEGACY/'thinking/transport_trials.csv')
    ts=raw.groupby(['condition','top_k','repeat'])[['clean_margin','corrupt_margin','margin','restoration']].mean().reset_index()
    ts.to_csv(OUT/'plot_data/03_value_transport.csv',index=False)
    fig,ax=plt.subplots(figsize=(3.8,2.70))
    fig.subplots_adjust(left=.17,right=.98,top=.85,bottom=.28)
    for cond,name,color,style in [('value_selected','Selected',T,'-'),('value_control','Control',GRAY,'--')]:
        f=ts[ts.condition==cond].groupby('top_k').margin.mean()
        ax.plot(f.index,f.values,'o'+style,color=color,label=name)
    for cond,name,color in [('clean','Clean',VIOLET),('damaged','Damaged',ORANGE)]:
        ax.axhline(float(ts.loc[ts.condition==cond,'margin'].iloc[0]),color=color,ls=':',label=name)
    ax.axhline(0,color=INK,lw=.6)
    panel(ax,'A. Source-value restoration','Logit margin')
    ax.set(xlabel='Patched heads',xticks=[1,2,4],xlim=(.85,4.15))
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=4,fontsize=8.2,
               handlelength=1.2,columnspacing=.9,handletextpad=.35,bbox_to_anchor=(.52,.015))
    save(fig,'03_targeted_value_transport','Targeted source-value transport',
         'A single source character is replaced by another target character. Restoring the original value at the corresponding source in selected heads increases the correct-versus-replacement logit margin. '
         'Each condition uses the same 100 confirmation inputs and a fixed gold-prefix retrieval query. Control means average three head sets for K=1/2 and one for K=4. Dotted lines show clean and damaged local margins. '
         'This is a local prediction assay; it does not measure complete-trace or final-answer recovery.',
         '从原复杂组合图拆出；只展示 source value 是否传输目标字符信息。',width=.72)


def dynamics():
    gd=pd.concat([read(EXTRA/m/'geometry_dynamics_trials.csv') for m in ['nonthinking','thinking']])
    gs=gd.groupby(['mode','endpoint','step','layer','occurrence']).ncc_correct.mean().groupby(['mode','endpoint','step','layer']).mean().reset_index()
    gs.to_csv(OUT/'plot_data/08_ncc_dynamics.csv',index=False)
    transport=pd.concat([read(f/'transport_trials.csv').assign(step=int(f.name.rsplit('_',1)[1]))
                         for f in sorted((LEGACY/'thinking').glob('transport_step_*'))])
    ts=transport.groupby(['step','condition','top_k','repeat']).restoration.mean().reset_index()
    ts.to_csv(OUT/'plot_data/08_value_dynamics.csv',index=False)
    fig=plt.figure(figsize=(5.5,2.65))
    for i,title in enumerate(['A. Running index','B. Final count','C. Value transport']):
        ax=fig.add_axes([(0.46+i*1.81)/5.5,.66/2.65,1.33/5.5,1.55/2.65])
        if i<2:
            for mode,color,name in [('nonthinking',NT,'NT'),('thinking',T,'T')]:
                d=gs[(gs['mode']==mode)&gs.endpoint.str.contains('answer_query').eq(i==1)].sort_values('step')
                ax.plot(d.step,100*d.ncc_correct,'o-',color=color,markersize=3,label=f'{name} (L{int(d.layer.iloc[0])})')
            ax.axhline(10,color=GRAY,ls=':',lw=1)
            ax.set(ylim=(-3,103),yticks=[0,25,50,75,100])
            ylabel='NCC accuracy (%)'
        else:
            for cond,color,style,name in [('value_selected',T,'-','Selected Top-2'),('value_control',GRAY,'--','Controls')]:
                d=ts[(ts.condition==cond)&ts.top_k.eq(2)].groupby('step').restoration.mean()
                ax.plot(d.index,d.values,'o'+style,color=color,markersize=3,label=name)
            ax.axhline(0,color=GRAY,lw=.7)
            ylabel='Margin gain'
        ax.set_xscale('symlog',linthresh=200,linscale=.65)
        ax.axvline(1500,color=GRAY,ls=':',lw=1)
        ax.set(xlim=(-20,11500),xlabel='Training step',xticks=[0,200,1000,10000],xticklabels=['0','200','1k','10k'])
        panel(ax,title,ylabel)
        ax.legend(loc='upper right' if i==0 else 'center right' if i==1 else 'upper left',fontsize=8,
                  handlelength=1.1,handletextpad=.3,labelspacing=.25,frameon=True,
                  facecolor='white',edgecolor='none',framealpha=1,borderpad=.3)
    save(fig,'08_learning_dynamics','Readability and transport during training',
         'Eleven fixed checkpoints reuse the same 100 confirmation inputs. A/B: discovery-fitted nearest-centroid classifiers; physical layers stay fixed, with NT/T at L4/L2 for running index and L2/L2 for final count. '
         'Running-index accuracy averages labels equally over 550 occurrence states from those inputs. C: correct-versus-replacement logit-margin gain after selected Top-2 source-value restoration, compared with the archived matched head sets. '
         'The horizontal axis is linear through step 200 and logarithmic afterwards; step zero is retained. The vertical dotted line marks the scheduled objective change at step 1500. '
         'The older query-local Top-2 ablation dynamics are retained in the previous figure archive and are not substituted for the revised sustained-intervention experiment.',
         '保留可读性与 value transport 动态；旧 control 口径的 ablation dynamics 留在历史版本。')


def successor():
    raw=read(LEGACY/'thinking/factorial_trials.csv')
    arms=['clean','successor_top1','successor_control_0','successor_control_1','successor_control_2']
    f=raw.groupby('arm')[['ar_accuracy','trace_marker_count_accuracy','trace_exact']].mean().loc[arms]
    f.to_csv(OUT/'plot_data/S4_successor.csv')
    fig,axs=plt.subplots(1,2,figsize=(5.5,2.6))
    fig.subplots_adjust(left=.19,right=.96,bottom=.23,top=.84,wspace=.35)
    for j,(metric,title) in enumerate([('ar_accuracy','A. Final answer'),('trace_marker_count_accuracy','B. Marker count')]):
        ax=axs[j]
        for i,arm in enumerate(arms):
            color=GRAY if 'control' in arm else VIOLET if 'successor' in arm else T
            val=float(f.loc[arm,metric])*100
            ax.plot([0,val],[i,i],lw=1.6,color=color,alpha=.3)
            ax.scatter(val,i,s=20,color=color,zorder=3)
            ax.text(val+3,i,f'{val:.0f}',va='center',fontsize=8.5)
        panel(ax,title)
        ax.set(yticks=range(5),yticklabels=['Clean','L2H3 ablated','Control 1','Control 2','Control 3'] if j==0 else [],
               xlim=(0,113),xticks=[0,50,100],xlabel='Accuracy (%)',ylim=(4.5,-.5))
    save(fig,'S4_successor_original_protocol','Successor-head ablation: original protocol',
         'Separate presentation of the original successor-head assay on all 100 confirmation inputs. L2H3 is selected by discovery marker-to-preceding-separator attention; the three same-layer control heads are shown individually. '
         'Panels report final-count accuracy and generated marker-count accuracy. This archived experiment applies the successor intervention at its original marker query sites. It has not been rerun under the sustained intervention used in Figure 02, so it remains a candidate awaiting protocol alignment. '
         'The full original factorial table, including joint interventions, remains in the previous archive.',
         '待更新候选：原 successor 查询位置干预；尚未按持续干预重新运行。单独展示，未混入新版检索消融。',group='candidate')


def write_gallery():
    sections=[]
    tex=['% Standalone figure candidates; not automatically included in main.tex.\n']
    md=['# Synthetic appendix：当前图集\n',
        '已更新 Broad Top-1～4 / Targeted Top-1～8。旧数据、完整扫描、旧图与正文内容均保留。\n',
        '主候选 9 张；PCA 对照 3 张；原 successor 局部干预候选 1 张。SNR、silhouette 和 p-value 不在图集中。\n',
        'Non-thinking 保留 L1H2 的依据及事后修订性质已在 appendix 中说明；完整 Top-1～8 扫描仍可查阅。\n']
    for i,f in enumerate(FIGURES,1):
        tag='主候选' if f['group']=='main' else 'PCA 对照' if f['group']=='alternative' else '待更新口径'
        sections.append(f'<section id="{f["stem"]}"><div class="tag">{tag}</div><h2>{i:02d} · {html.escape(f["title"])}</h2><p class="note">{html.escape(f["note"])}</p><a href="{f["stem"]}.png"><img src="{f["stem"]}.png" alt="{html.escape(f["title"])}"></a><p class="links"><a href="{f["stem"]}.pdf">PDF</a> · <a href="{f["stem"]}.svg">SVG</a> · <a href="{f["stem"]}.png">PNG</a></p><p class="caption">{html.escape(f["caption"])}</p></section>')
        md.append(f'## {i:02d} {f["title"]}（{tag}）\n\n{f["note"]}\n\n[PDF]({f["stem"]}.pdf) · [PNG]({f["stem"]}.png)\n\n{f["caption"]}\n')
        if f['group']=='main':
            cap=f['caption'].replace('%',r'\%').replace('_',r'\_')
            tex.append('\n\\GPT{\n\\begin{figure}[p]\n\\centering\n\\includegraphics[width='+str(f['width_fraction'])+'\\linewidth]{figures/synthetic_appendix/review_20260908/'+f['stem']+'.pdf}\n\\caption{'+cap+'}\n\\label{fig:synthetic-review-'+f['stem'].replace('_','-')+'}\n\\end{figure}\n}\n')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Synthetic appendix review</title><style>body{max-width:1120px;margin:32px auto;padding:0 22px;color:#161923;font:16px/1.6 system-ui;background:#fff}h1{font-size:28px}h2{font-size:21px;margin:8px 0}.tag{font-size:13px;color:#007EAB}.note{font-weight:600}.caption{font-family:Georgia,serif;font-size:15px}section{border-top:1px solid #dde1e6;padding-top:20px;margin:38px 0}img{width:100%;height:auto;display:block;margin:14px 0;cursor:zoom-in}a{color:#007EAB}.links{font-size:14px}</style><h1>Synthetic appendix · 当前候选图</h1><p>Broad: Top-1–4 · Targeted next marker: Top-1–8 · Aurora colors</p><p><a href="appendix_figures_review.pdf">全部图与图注 PDF</a> · <a href="appendix_main_figures.pdf">9 张主候选 PDF</a></p><p>点击图像查看原尺寸。末尾单独列出 PCA 对照和待更新口径的 successor 图。</p>'''+''.join(sections)+'</html>'
    (OUT/'index.html').write_text(page,encoding='utf-8')
    (OUT/'README.md').write_text('\n'.join(md),encoding='utf-8')
    (OUT/'captions.tex').write_text(''.join(tex),encoding='utf-8')
    shutil.copy2(OUT/'captions.tex',PAPER/'captions.tex')
    # Contact sheet is only an index; full-resolution images and PDFs are separate.
    main=[f for f in FIGURES if f['group']=='main']
    sheet=Image.new('RGB',(1800,1650),'white')
    draw=ImageDraw.Draw(sheet)
    font=preview_font(21)
    for i,f in enumerate(main):
        im=Image.open(OUT/f'{f["stem"]}.png').convert('RGB')
        im.thumbnail((570,470))
        x,y=(i%3)*600,(i//3)*550
        draw.text((x+15,y+10),textwrap.fill(f'{i+1:02d}  {f["title"]}',38),font=font,fill=INK)
        sheet.paste(im,(x+(600-im.width)//2,y+65+(470-im.height)//2))
    sheet.save(OUT/'overview.png')


def main():
    (OUT/'plot_data').mkdir(exist_ok=True)
    PAPER.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'mathtext.fontset':'stix',
        'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,
        'legend.frameon':False,'axes.linewidth':.65,'lines.linewidth':1.5,'lines.markersize':4,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','text.color':INK,
        'axes.labelcolor':INK,'axes.titlecolor':INK,'figure.facecolor':'white'})
    copied('01_training_and_behavior','01_training_and_behavior','Training and counting behavior',note='训练/验证区域的固定 token loss 与按 count 划分的准确率。')
    retrieval()
    value_transport()
    copied('04_count_readability','04_count_readability','Count readability across layers',note='Running index 与 final count 分开；全层 NCC 曲线。')
    common_caption=('Confirmation states projected using separately discovery-fitted, non-whitened PCA bases. All four panels use the same post-block layer L{layer}. '
        'Running-index panels show 550 states; final-count panels show 100 states from the same 100 confirmation inputs. All states are retained. '
        'Outlined points are confirmation class means joined in label order. NT and T denote non-thinking and thinking. '
        'Equal coordinate units are used within each panel; independent PCA bases prevent comparing absolute coordinates across panels. Gold prefixes are used, so length and position remain possible contributors to count readability.')
    copied('11_pca_common_l2','05_count_geometry_common_l2','Count geometry at common layer L2',caption=common_caption.format(layer=2),note='两种模式、两类端点统一使用 L2；L4 与 3D 对照放在末尾。')
    copied('06_progress_state','06_progress_state','Progress-state interventions',note='下一 marker 的信息来源，以及 donor item state 对短程 continuation 的影响。')
    copied('07_answer_readout','07_answer_readout','Final-answer readout',note='答案信息来源、完整状态 transplant、count direction 三项同属最终答案读出。')
    dynamics()
    copied('09_retrieval_roles','09_retrieval_roles','Complete attention-role dynamics',note='全 32 heads、101 checkpoints；绝对 attention score，不做跨头归一化。')
    copied('12_pca_common_l4','S1_count_geometry_common_l4','Count geometry at common layer L4',caption=common_caption.format(layer=4),note='与主候选 L2 图配套的固定深度对照。',group='alternative')
    for layer,old_stem,new_stem in [(2,'13_pca_common_l2_3d','S2_count_geometry_common_l2_3d'),(4,'14_pca_common_l4_3d','S3_count_geometry_common_l4_3d')]:
        cap=common_caption.format(layer=layer)+' PC3 uses the same archived basis as PC1/2. The camera, orthographic projection and coordinate unit scaling are consistent across panels; these are 3D views of the corresponding 2D data.'
        copied(old_stem,new_stem,f'3D count geometry at common layer L{layer}',caption=cap,note=f'L{layer} 同一批点、同一 PCA 基底的 3D 对照；供判断 3D 是否更清晰。',group='alternative')
    successor()
    write_gallery()
    assert len(FIGURES)==13 and sum(f['group']=='main' for f in FIGURES)==9
    for path,h in SOURCES.items():
        assert digest(WORK/path)==h
    manifest={'figures':FIGURES,'source_sha256':SOURCES,'checks':CHECKS,'mode_colors':{'nonthinking':NT,'thinking':T},
              'displayed_k':{'nonthinking':[1,2,3,4],'targeted':list(range(1,9))},
              'new_model_runs':False,'new_significance_tests':False,'old_figures_preserved':True,
              'source_builder_sha256':digest(Path(__file__)),
              'rendered_sha256':{f'{f["stem"]}.{e}':digest(OUT/f'{f["stem"]}.{e}') for f in FIGURES for e in ['pdf','png','svg']}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print('REVIEW FIGURES READY',len(FIGURES),flush=True)


if __name__=='__main__':
    main()
