"""Build a compact four-panel CoT main figure and three-panel supplement."""
import hashlib
import json
import time
from pathlib import Path
import sys as _font_sys
_font_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_style import paper_font
FONT_FAMILY = paper_font()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import LOAD_NO_HINTING
from matplotlib.lines import Line2D
from matplotlib.mathtext import MathTextParser
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.ticker import PercentFormatter
from matplotlib.transforms import Bbox, ScaledTranslation
import numpy as np
import prepare_data as d
import prepare_extra as e

OUT, FINAL = d.OUT, d.FINAL
Q, G = d.COLORS
INK, AXIS, GRID, BROWN, WARM, EDGE = d.INK, d.AXIS, d.GRID, d.BROWN, d.WARM, d.EDGE
FONT_PROFILE = json.loads((OUT/'font_profile.json').read_text(encoding='utf-8'))

def main_font(role):
    """Match the reference's printed size despite the different canvas widths."""
    return FONT_PROFILE['paper_pt'][role]*11.7/FONT_PROFILE['paper_width_inches']

plt.rcParams.update({'font.family':FONT_FAMILY,'font.size':13.5,'mathtext.fontset':'stix',
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.labelcolor':INK,'text.color':INK,
    'axes.edgecolor':AXIS,'xtick.color':INK,'ytick.color':INK,'axes.linewidth':.8,
    'lines.linewidth':1.8,'legend.frameon':False})
fig = plt.figure(figsize=(11.7,6.6),facecolor='white')
art = fig.add_axes([0,0,1,1]);art.set(xlim=(0,1),ylim=(0,1));art.axis('off')
boxed_labels = []
box_math_parser = MathTextParser('path')

def text(x,y,s,size=13.5,bold=False,**kw):
    return art.text(x,y,s,fontsize=size,fontweight='bold' if bold else 'normal',
        ha=kw.pop('ha','center'),va=kw.pop('va','center'),**kw)

def panel_title(x,y,letter,title):
    # The reference uses distinct sizes for the panel letter and title text.
    prefix='a_' if letter=='A' else ''
    return [text(x,y,letter,size=main_font(prefix+'letter'),bold=True,ha='left'),
            text(x+.026,y,title,size=main_font(prefix+'title'),bold=True,ha='left')]

def visible_text_bounds(label):
    s,ismath=label._preprocess_math(label.get_text())
    prop=label.get_fontproperties()
    if not ismath:
        return TextPath((0,0),s,size=prop.get_size_in_points(),prop=prop).get_extents()
    # Parse at the actual PDF font size. TextPath normalizes math to 100 pt,
    # which changes rounded math baselines when scaled back to small labels.
    _,_,_,glyphs,rects=box_math_parser.parse(s,72,prop)
    bounds=[]
    for font,size,code,x,y in glyphs:
        font.clear();font.set_size(size,72);font.load_char(code,flags=LOAD_NO_HINTING)
        vertices,codes=font.get_path()
        if len(vertices):
            bounds.append(MplPath(vertices+np.array([x,y]),codes).get_extents())
    bounds.extend(Bbox.from_bounds(x,y,w,h) for x,y,w,h in rects)
    return Bbox.union(bounds)

def box(x,y,w,h,s,fill='#FAF9F6',size=14,edge=EDGE,lw=1):
    patch=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.002,rounding_size=0.003',
        facecolor=fill,edgecolor=edge,linewidth=lw)
    art.add_patch(patch)
    label=text(x,y,s,size=size,ha='left',va='baseline')
    # Center visible glyphs, including math subscripts, rather than the font's
    # ascent/descent box. Keep the label as selectable text in PDF and SVG.
    ink=visible_text_bounds(label)
    dx,dy=-(ink.x0+ink.x1)/2,-(ink.y0+ink.y1)/2
    label.set_transform(art.transData+ScaledTranslation(dx/72,dy/72,fig.dpi_scale_trans))
    boxed_labels.append((patch,label,ink))

def arrow(a,b,color=BROWN,lw=1.25):
    art.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,color=color,lw=lw,shrinkA=0,shrinkB=0))

def axis(rect,labelsize=15.2,ticksize=14):
    ax=fig.add_axes(rect);ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color=GRID,lw=.65,zorder=0);ax.set_axisbelow(True)
    ax.tick_params(length=3,width=.7,labelsize=ticksize)
    ax.xaxis.label.set_size(labelsize);ax.yaxis.label.set_size(labelsize)
    return ax

def errors(ax,x,r,color,marker='o',size=5.7):
    y,lo,hi=[r[k] for k in ['mean','ci95_low','ci95_high']]
    ax.errorbar(x,y,yerr=[[y-lo],[hi-y]],fmt=marker,color=color,ms=size,capsize=3,elinewidth=1.15,zorder=3)

handles=[Line2D([],[],color=c,marker='o',ms=4.5,lw=1.7) for c in [Q,G]]

def check_and_save(pdf,stem,title):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    # Annotation.get_window_extent also includes its leader line. Use the base
    # text extent when checking text collisions, and audit data clearance below.
    labels=[(t.get_text(),matplotlib.text.Text.get_window_extent(t,renderer)) for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
    outside=[s for s,b in labels if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1]
    overlaps=[]
    for i,(s,bb) in enumerate(labels):
        for t,cc in labels[i+1:]:
            if min(bb.x1,cc.x1)-max(bb.x0,cc.x0)>1.5 and min(bb.y1,cc.y1)-max(bb.y0,cc.y0)>1.5:overlaps.append([s,t])
    assert not outside,outside
    assert not overlaps,overlaps
    spacing={}
    if stem=='cot_reasoning_main':
        box_audit=[]
        for patch,label,ink in boxed_labels:
            if label.figure is not fig:
                continue
            frame=patch.get_window_extent(renderer)
            baseline=label.get_transform().transform(label.get_position())
            glyph=ink.transformed(matplotlib.transforms.Affine2D().scale(fig.dpi/72).translate(*baseline))
            delta=np.array([(glyph.x0+glyph.x1-frame.x0-frame.x1)/2,
                            (glyph.y0+glyph.y1-frame.y0-frame.y1)/2])*72/fig.dpi
            assert np.max(np.abs(delta))<1e-6,('Box label is not centered',label.get_text(),delta)
            assert frame.contains(glyph.x0,glyph.y0) and frame.contains(glyph.x1,glyph.y1),label.get_text()
            box_audit.append({'text':label.get_text(),'center_error_pt':delta.tolist(),
                'frame_figure_fraction':(frame.extents/np.array([fig.bbox.width,fig.bbox.height]*2)).tolist()})
        bounds=[Bbox.union([a.get_tightbbox(renderer) for a in b_axes]+
                          [a.get_window_extent(renderer) for a in b_labels]+[b_shapes.get_window_extent(renderer)]),
                Bbox.union([a.get_window_extent(renderer) for a in c_items]),
                Bbox.union([af.get_tightbbox(renderer),d_models.get_window_extent(renderer)]+
                           [a.get_window_extent(renderer) for a in d_titles])]
        paper_width_pt=6.5*72
        gaps=[(bounds[i+1].x0-bounds[i].x1)/fig.bbox.width*paper_width_pt for i in range(2)]
        assert min(gaps)>=8,('Inter-panel gutters at manuscript width',gaps)
        # All B intervals and markers occupy the two model columns. Ensure each
        # value label clears both columns horizontally, including their caps.
        ablation_clearance=[]
        for ax,t,_ in b_callouts:
            bb=matplotlib.text.Text.get_window_extent(t,renderer)
            for model_x in [0,1]:
                px=ax.transData.transform((model_x,0))[0]
                radius=4*fig.dpi/72
                ablation_clearance.append(max(px-radius-bb.x1,bb.x0-px-radius))
        assert min(ablation_clearance)>2,('B value labels overlap data/interval columns',ablation_clearance)
        spacing={'manuscript_width_inches':6.5,'panel_gutters_pt':dict(zip(['B_to_C','C_to_D'],gaps)),
                 'box_label_alignment':{'method':'visible glyph bounds; selectable text retained','labels':box_audit},
                 'smallest_label_pt':min(t.get_fontsize() for t in fig.findobj(matplotlib.text.Text)
                     if t.get_visible() and t.get_text())*6.5/fig.get_figwidth(),
                 'C_column_pitch_pt':float((table_x[1]-table_x[0])*paper_width_pt),
                 'B_value_label_clearance_pt':min(ablation_clearance)/fig.bbox.width*paper_width_pt,
                 'D_style':{'curve':'solid without point markers','legend':'line samples above axes',
                            'endpoint_callouts':False,'y_tick_step_percentage_points':20}}
    fig.savefig(pdf,metadata={'Title':title,'Author':'Anonymous Authors','CreationDate':None})
    fig.savefig(OUT/f'{stem}.svg');fig.savefig(OUT/f'{stem}.png',dpi=220)
    plt.close(fig)
    return {'outside_text':outside,'text_overlaps':overlaps,**spacing}

# A. Token/site-aligned state diagram, matching the Non-thinking visual grammar.
# Horizontal placement is generation order, not a universal layer assignment.
# Each functional edge is tied to a recorded intervention in mechanism_evidence.md.
art.set_position([0,.607,1,.393]);art.set_ylim(.59,1)
panel_title(.016,.976,'A','Thinking mechanism')
for y,s in [(.808,'States'),(.696,'Tokens'),(.633,'Positions')]:
    text(.025,y,s,size=16,bold=True,ha='left')
columns=[.18,.31,.445,.590,.723,.843,.947]
states=[r'$h_{\tau_k}^{(\ell)}$',r'$H_{k-1}$',r'$h_{q_k}$',r'$H_k$',r'$h_{q_{k+1}}$',r'$H_N$',r'$a_T$']
tokens=[r'$N_k$',r'$e_{k-1}$','query','item end','query',r'$e_N$',r'$\langle\mathrm{Ans}\rangle$']
positions=[r'$\tau_k$',r'$t_{k-1}$',r'$q_k$',r'$t_k$',r'$q_{k+1}$',r'$t_N$',r'$T$']
for x,state,token,pos in zip(columns,states,tokens,positions):
    text(x,.808,state,size=18)
    box(x,.696,.079,.048,token,fill=WARM if token=='item end' or x==columns[-1] else '#FAF9F6',size=15)
    arrow((x,.725),(x,.774))
    text(x,.633,pos,size=15)
text(.31,.867,'Counter state',size=15)
text(.843,.902,'Terminal',size=15)
# A recorded prompt span supplies the query's targeted read; prior event state
# controls which successor is selected. The upper elbow separates these inputs.
art.plot([.18,.18,.445],[.84,.924,.924],color=BROWN,lw=1.25)
arrow((.445,.924),(.445,.841))
text(.37,.94,'Retrieve',size=15.5,va='bottom')
arrow((.335,.808),(.422,.808));text(.3785,.758,'Guide',size=15)
# The controlled carrier interventions support a functional update pathway.
# Its internal decomposition stays in the supporting evidence, not as two
# independently established stages of free-running reasoning in the main panel.
arrow((.468,.808),(.569,.808));text(.5185,.867,'State update',size=15)
# Unroll the next query at its own token position. The forward arc carries
# H_k to h_(q_(k+1)); it never points back to the already computed h_(q_k).
next_step=MplPath([(.590,.84),(.590,.967),(.723,.967),(.723,.84)],
    [MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4])
art.add_patch(FancyArrowPatch(path=next_step,arrowstyle='-|>',mutation_scale=11,color=BROWN,lw=1.25))
text(.6565,.956,'Repeat',size=15)
# Ellipses indicate further iterations; there is no asserted scalar +1 edge.
for y in [.808,.696]:text(.7835,y,r'$\cdots$',size=15,color='#64665F')
# Same-trial suffix resets support a partial terminal-to-answer pathway.
arrow((.861,.808),(.926,.808));text(.897,.855,'Read',size=15)
box(.947,.936,.084,.047,'Total: '+r'$N$',size=16)
arrow((.947,.842),(.947,.909))
art=fig.add_axes([0,0,1,1]);art.set(xlim=(0,1),ylim=(0,1));art.axis('off')

# B. Match Non-thinking: model on x, K below, one condition legend.
# Separate outcomes vertically so each model has one clear intervention column.
b_titles=panel_title(.016,.562,'B','Head ablation')
b_labels=[*b_titles,text(.015,.284,'Ablation effect',size=main_font('axis'),rotation=90)]
shapes=[Line2D([],[],marker='o',color=INK,ls='',mfc='white',ms=4),
    Line2D([],[],marker='D',color=INK,ls='',ms=4),Line2D([],[],marker='o',color=INK,ls='',ms=4)]
b_shapes=fig.legend(shapes,['Original','Random',r'Top-$K$'],loc='center',bbox_to_anchor=(.139,.518),
    ncol=3,fontsize=main_font('legend'),handlelength=.4,handletextpad=.34,columnspacing=.42)
b_axes=[];b_callouts=[]
for yi,(outcome,label) in enumerate([('next_city_failure','Next item'),('final_count_failure','Final count')]):
    b=axis([.080,.315 if yi==0 else .109,.1755,.143],
           labelsize=main_font('axis'),ticksize=main_font('tick'))
    b_axes.append(b)
    b_labels.append(text(.16775,.480 if yi==0 else .274,label,size=main_font('axis')))
    for x,(m,c) in enumerate(zip(d.MODELS,[Q,G])):
        rr=[r for r in e.ablation if r['model']==m and r['outcome']==outcome]
        b.vlines(x,0,next(r for r in rr if r['condition']=='selected_bank')['mean'],color=c,lw=.8,alpha=.55)
        for cond,mark in [('clean','o'),('layer_matched_random','D'),('selected_bank','o')]:
            r=next(r for r in rr if r['condition']==cond)
            if cond=='clean':b.plot(x,0,'o',ms=4.8,mfc='white',mec=c,mew=1,zorder=5)
            else:
                errors(b,x,r,c,mark,5.0 if mark=='D' else 6.0)
                callout=b.annotate(f"{100*r['mean']:.0f}%",(x,r['mean']),xytext=(-9 if x==0 else 9,3),
                    textcoords='offset points',ha='right' if x==0 else 'left',va='bottom',fontsize=main_font('annotation'),color=c,
                    bbox=dict(facecolor='white',edgecolor='none',pad=.2))
                b_callouts.append((b,callout,x))
    b.axhline(0,color=AXIS,lw=.8)
    b.set(xlim=(-.65,1.65),ylim=(-.12,1.22),yticks=[0,.5,1],xticks=[0,1])
    b.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    if yi==0:
        b.tick_params(axis='x',bottom=False,labelbottom=False)
    else:
        b.set_xticklabels(['Qwen','Gemma'],fontsize=main_font('tick'))
        for x,k in enumerate([128,6]):
            b_labels.append(b.text(x,-.39,rf'$K={k}$',transform=b.get_xaxis_transform(),
                                  ha='center',va='top',fontsize=main_font('axis')))

# C. A concrete Target -> Receiver transplant, with both model outcomes.
# Translate the whole panel to equalize the measured B-C and C-D gutters.
art=fig.add_axes([-.0055,0,1,1]);art.set(xlim=(0,1),ylim=(0,1));art.axis('off')
c_before=set(art.get_children())
panel_title(.289,.562,'C','Counter-state patching')
# One reference line replaces the six repeated city boxes.
text(.501,.490,'Mumbai '+r'$\rightarrow$'+' Islamabad '+r'$\rightarrow$'+' Osaka '+r'$\rightarrow$'+' Lima '+r'$\rightarrow\cdots$',size=main_font('axis'))
text(.289,.414,'Original',size=main_font('axis'),bold=True,ha='left')
box(.44,.414,.145,.082,'... Mumbai',fill='#F7F6F3',size=main_font('axis'))
arrow((.518,.414),(.542,.414),color='#64665F')
box(.63,.414,.163,.082,'Islamabad',size=main_font('axis'))
box(.44,.324,.263,.050,'Target state after Islamabad',fill=WARM,size=main_font('axis'))
arrow((.44,.295),(.44,.279))
text(.289,.233,'Patched',size=main_font('axis'),bold=True,ha='left')
box(.44,.233,.145,.082,'... Mumbai',fill='#F7F6F3',size=main_font('axis'))
arrow((.518,.233),(.542,.233))
box(.63,.233,.163,.082,'Osaka '+r'$\rightarrow$'+' Lima '+r'$\rightarrow\cdots$',fill='#EFF7FB',edge=Q,size=main_font('axis'))
# A separate, aligned statistics block has a clear gap from the example.
art.plot([.289,.716],[.172,.172],color=GRID,lw=.7)
table_x=np.linspace(.32,.69,5)
text(float(np.mean(table_x[1:])),.146,'Stepwise success',size=main_font('tick'))
for x,label in zip(table_x,['Model',r'$k+1$',r'$k+2$',r'$k+3$',r'$k+4$']):
    text(x,.107,label,size=main_font('axis'),bold=True)
for y,r,c in zip([.067,.027],e.progress,[Q,G]):
    text(table_x[0],y,'Qwen' if r['model']==d.MODELS[0] else 'Gemma',size=main_font('axis'),color=c)
    for x,hop in zip(table_x[1:],[1,2,3,4]):
        v=next(h for h in e.progress_hops if h['model']==r['model'] and h['hop']==hop)
        text(x,y,(f"{v['successes']}/{v['eligible']}" if v['eligible'] else "—"),size=main_font('axis'),color=c,bold=True)
c_items=[a for a in art.get_children() if a not in c_before]

# D. Actual greedy adoption of the Target's count after answer-state patching.
art=fig.add_axes([0,0,1,1]);art.set(xlim=(0,1),ylim=(0,1));art.axis('off')
d_titles=panel_title(.744,.562,'D','Answer-state patching')
af=axis([.807,.145,.18,.30],labelsize=main_font('axis'),ticksize=main_font('tick'))
# Match Non-thinking D: plain solid lines, a light confidence band, and line
# samples in the legend. Exact endpoints are stated in the caption.
d_handles=[Line2D([],[],color=c,lw=2.4) for c in [Q,G]]
d_models=fig.legend(d_handles,['Qwen','Gemma'],loc='center',bbox_to_anchor=(.897,.496),ncol=2,
    fontsize=main_font('legend'),handlelength=1.6,handletextpad=.45,columnspacing=1.0)
for m,c in zip(d.MODELS,[Q,G]):
    rr=sorted([r for r in d.answer if r['model']==m],key=lambda r:r['layer'])
    x,y,lo,hi=[np.array([r[k] for r in rr]) for k in ['layer','mean','ci95_low','ci95_high']]
    af.plot(x+1,y,color=c,lw=2.4,zorder=3)
    af.fill_between(x+1,lo,hi,color=c,alpha=.16,lw=0)
af.axhline(0,color=AXIS,lw=.6)
af.set(xlim=(.5,42.5),ylim=(-.04,1.12),yticks=[0,.2,.4,.6,.8,1],xticks=[1,11,21,31,41],xlabel='Intervention layer',ylabel='Target-count adoption')
af.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
main_audit=check_and_save(FINAL,'cot_reasoning_main','CoT reasoning: targeted retrieval and counter-state routing')

# Supplement: supporting write, source-blank, and mediation experiments.
CONTROL=d.ROOT/'output/pdf/cot_reasoning_controls.pdf'
fig=plt.figure(figsize=(11.7,3.65),facecolor='white')
art=fig.add_axes([0,0,1,1]);art.set(xlim=(0,1),ylim=(0,1));art.axis('off')
for x,s in [(.015,'A  Counter-state update'),(.355,'B  Trace dependence'),(.693,'C  Terminal mediation')]:
    text(x,.942,s,size=17,bold=True,ha='left')
ad=axis([.084,.218,.204,.548],13.5,12);ae=axis([.425,.218,.204,.548],13.5,12);ag=axis([.769,.218,.209,.548],13.5,12)
for i,(r,c) in enumerate(zip(e.write,[Q,G])):
    ad.vlines(i,0,r['mean'],color=c,lw=.8,alpha=.6);errors(ad,i,r,c)
    ad.annotate(f"{r['mean']:.3f}",(i,r['ci95_high']),xytext=(0,5),textcoords='offset points',ha='center',va='bottom',color=c,fontsize=12.5)
ad.set(xlim=(-.55,1.55),ylim=(0,.41),yticks=[0,.1,.2,.3,.4],xticks=[0,1],xticklabels=['Qwen','Gemma'],ylabel='Commit rescue (RMS)')
text(.186,.065,'Clean carrier restored under mask',size=12)
for mi,(m,c) in enumerate(zip(d.MODELS,[Q,G])):
    rr=[next(r for r in e.blank if r['model']==m and r['condition']==cond) for cond in ['clean','prompt_records_blank','trace_all_blank']]
    for x,r in zip(np.arange(3)+(-.12 if mi==0 else .12),rr):
        ae.bar(x,r['mean'],width=.21,color=c,alpha=.85,zorder=2)
        ae.errorbar(x,r['mean'],yerr=[[r['mean']-r['ci95_low']],[r['ci95_high']-r['mean']]],color=c,capsize=2,elinewidth=.9,fmt='none',zorder=3)
ae.set(xlim=(-.5,2.5),ylim=(0,1.11),yticks=[0,.5,1],xticks=[0,1,2],xticklabels=['Original','Records\nblank','Trace\nblank'],ylabel='Count accuracy')
ae.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
ae.legend(handles,['Qwen','Gemma'],ncol=2,loc='lower center',bbox_to_anchor=(.5,1.025),fontsize=12,handlelength=.8,handletextpad=.3,columnspacing=.7,borderaxespad=0)
for i,(r,c) in enumerate(zip(e.relay,[Q,G])):
    ag.vlines(i,0,r['mean'],color=c,lw=.8,alpha=.6);errors(ag,i,r,c)
    ag.annotate(f"{100*r['mean']:.1f}%",(i,r['ci95_high']),xytext=(0,5),textcoords='offset points',ha='center',va='bottom',color=c,fontsize=12.5)
ag.set(xlim=(-.55,1.55),ylim=(0,1.1),yticks=[0,.5,1],xticks=[0,1],xticklabels=['Qwen','Gemma'],ylabel='Effect mediated by suffix')
ag.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
text(.874,.065,'86 directed pairs per model',size=12)
control_audit=check_and_save(CONTROL,'cot_reasoning_controls','Supporting evidence for counter-state update and the Read stage')
manifest={'status':'PASS','main_figure':str(FINAL.relative_to(d.ROOT)),'main_panels':list('ABCD'),
    'supplement_figure':str(CONTROL.relative_to(d.ROOT)),'supplement_panels':list('ABC'),
    'reference_thread_id':'01a05071-c0ac-71c0-9efe-14c3ea1963cd','source_sha256':d.SOURCES,
    'pdf_sha256':hashlib.sha256(FINAL.read_bytes()).hexdigest(),'supplement_sha256':hashlib.sha256(CONTROL.read_bytes()).hexdigest(),
    'terminology':{'Target':'state source for patching','Receiver':'run receiving the transplanted state'},
    'style':{'font':'Times New Roman','model_colors':dict(zip(d.MODELS,[Q,G])),'mechanism_arrow':BROWN,'main_size_inches':[11.7,6.6]},
    'font_alignment':FONT_PROFILE,
    'mechanism_evidence':'figures/cot-reasoning/mechanism_evidence.md',
    'mechanism_state_update':{'display':'State update','carrier_column':False,
        'evidence_scope':'Functional summary of retrieval-head ablation and NCC-selected item-state transfer; layer ranges and no-index cohorts are specified per model',
        'decomposition':'Full item spans jointly carry record content and progress; an arithmetic update operation is not isolated'},
    'mechanism_read':{'display':'Read','source':'H_N','target':'a_T','scope':'Partial terminal-to-answer pathway'},
    'mechanism_state_to_query':{'left_label':'Guide','right_label':'Repeat',
        'edges':[['H_{k-1}','h_{q_k}'],['H_k','h_{q_{k+1}}']],
        'scope':'Guide denotes prior-state influence on the current query; Repeat indicates the next enumeration iteration'},
    'mechanism_next_step':{'source':'H_k','target':'h_{q_{k+1}}',
        'source_position':'t_k','target_position':'q_{k+1}',
        'layout':'Explicit next-query column in generation order; forward arc'},
    'statistics':{'B':'Failure(condition)-Failure(Original), paired percentile bootstrap over 10 seeds',
        'C':'Forward k=4,6,8; 10 seeds,30 cells/model; Qwen L19 and Gemma L21 from no-index discovery NCC; Gemma candidates L1--L22 retain later K/V writes in both attention types; all columns show stepwise success conditional on all preceding steps being correct and enough remaining items',
        'D':d.ANSWER_SCOPE,
        'supplement_A':'Registered clean-carrier restoration RMS effect','supplement_B':'100 prompts/condition, count average within seed',
        'supplement_C':'Registered 86-pair,10-seed primary suffix-mediated fraction'},
    'limits':['Gemma C is prompt-conditioned, not a natural no-index replication','Qwen L19 and Gemma L21 (one-based) with distinct seed cohorts; existing confirmation seeds reused',
        'B persistent masking; baseline Gemma next-item failure is 20%','No exclusive circuit or scalar-register claim',
        'Pointwise seed-cluster intervals; subsequent-step denominators condition on earlier success'],
    'layout_audit':{'main':main_audit,'supplement':control_audit},'build_seconds':time.perf_counter()-d.START}
(OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','main_pdf':str(FINAL),'supplement_pdf':str(CONTROL),'sources':len(d.SOURCES),'seconds':manifest['build_seconds']},indent=2))
