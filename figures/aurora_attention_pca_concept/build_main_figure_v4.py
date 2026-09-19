"""Native draw.io v4: readable needle-attention rows and preserved v3 PCA."""
import copy
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import build_main_figure_v3 as base

HERE=Path(__file__).resolve().parent
C_Y=560
C_PLOT_SCALE=.94
C_HEIGHT=464
HEIGHT=C_Y+C_HEIGHT
QWEN='#168DCA'
ATTENTION_LOW='#FFFFFF'
B_MARKERS=[2,3,4]
QUERY='#B87652'
STRIP_X,STRIP_W,STRIP_H=812,438,30

def heat(v):
    return base.mix(ATTENTION_LOW,QWEN,max(0.,min(1.,float(v))))

def needle(d,k,x,y,w=82):
    return d.box('N'+str(k).translate(str.maketrans('0123456789','₀₁₂₃₄₅₆₇₈₉')),x,y,w,40,
                 color=base.SPAN_BORDER,fill=base.PROMPT_FILL,size=25)

def query(d,text,x,y,w,size=24):
    return d.vertex(text,x,y,w,38,
        f'rounded=0;html=0;whiteSpace=wrap;fillColor=#FFFFFF;strokeColor={QUERY};strokeWidth=1.8;'
        f'align=center;verticalAlign=middle;fontFamily={base.FONT};fontSize={size};fontColor={base.INK};spacing=0;')

def outcome(d,x,y,correct):
    color='#24845C' if correct else '#C94F55'
    d.vertex('',x,y,34,34,f'ellipse;fillColor={color};strokeColor=none;')
    if correct:
        d.edge([(x+8,y+17),(x+14,y+23),(x+26,y+10)],color='#FFFFFF',width=3.4)
    else:
        d.edge([(x+10,y+10),(x+24,y+24)],color='#FFFFFF',width=3.4)
        d.edge([(x+10,y+24),(x+24,y+10)],color='#FFFFFF',width=3.4)

def main():
    source=ET.parse(HERE/'main_figure_v2.drawio')
    v2={c.get('id'):c for c in source.getroot().iter('mxCell')}
    frozen=HERE/'assets/v4_source_v3.drawio'
    state=json.loads((HERE/'assets/v4_source_v3_settings.json').read_text(encoding='utf-8'))
    d=base.Diagram()
    d.model.set('pageHeight',str(HEIGHT))
    d.panel('A',0,228,'Non-thinking: broad retrieval')
    d.text('Prompt\nspan',30,80,95,56,size=23,bold=True)
    for k,x in [(1,164),(2,329),(3,494),(10,659)]:
        needle(d,k,x,88)
        d.edge([(353,177),(x+41,149),(x+41,128)],color=QWEN,width=2,curve=True,arrow=True)
    for x in [129,278,443,608,747]:d.text('···',x,88,20,40,size=23,color=base.MUTED,align='center')
    d.text('Direct\nanswer',30,168,100,56,size=23,bold=True)
    query(d,'<Ans>',300,177,106)
    d.text('Total: 8',434,176,115,40,size=26)
    outcome(d,568,179,False)
    d.edge([(784,53),(784,217)],color=base.RULE,width=1)
    d.text('Attention from <ans> token to needle span',803,17,454,30,size=23,bold=True,align='center')

    # Same 170 prompt bins and q98.5 square-root display transform as v2.
    raw=HERE.parents[1]/'realistic/exports/run_20260731_v4_numeric_presentation_v3/Qwen3-8B/numeric/attention/capture/raw_shards/v4.4/V4_4_T10000_N10_seed1255.npz'
    with np.load(raw) as payload: attention=np.asarray(payload['layer_028'][19],dtype=float)
    edges=np.linspace(0,len(attention),171,dtype=int)
    bins=np.array([attention[edges[i]:edges[i+1]].sum() for i in range(170)])
    high=float(np.quantile(bins,.985)) or 1.
    xx,yy,ww,hh=STRIP_X,107,STRIP_W,STRIP_H
    for i,v in enumerate(bins):
        level=.035+.87*math.sqrt(min(1.,float(v)/high))
        d.rect(xx+i*ww/170,yy,ww/170+.1,hh,heat(level),'none',0,
               f'Prompt bin {i}; raw attention mass={v:.12g}')
    for k in range(1,11):
        g=v2[str(204+2*(k-1))].find('mxGeometry/mxPoint')
        px=xx+(float(g.get('x'))-54)/832*ww
        d.edge([(px,yy-6),(px,yy+hh+5)],color=base.MUTED,width=.8)
        label_y=yy-47 if k in (4,8,10) else yy-27
        d.text('N'+str(k),px-18,label_y,36,24,size=19,bold=True,align='center')
    d.text('0',xx-5,148,25,25,size=19,color=base.MUTED)
    d.text('Prompt position',xx+85,148,250,25,size=20,color=base.MUTED,align='center')
    d.text('10k tokens',xx+ww-94,148,99,25,size=19,color=base.MUTED,align='right')
    # Same nonlinear map as the strip, with ticks in raw bin-mass units.
    bar_x,bar_y,bar_w=944,193,206
    d.text('Attention mass',788,189,128,26,size=19,color=base.MUTED,align='right')
    for i in range(100):
        d.rect(bar_x+i*bar_w/100,bar_y,bar_w/100+.1,9,heat(.035+.87*i/99),'none',0)
    for t,label in [(0,'0'),(.5,f'{high/4:.3f}'),(1,f'≥{high:.3f}')]:
        d.text(label,bar_x+t*bar_w-32,205,64,23,size=19,color=base.MUTED,align='center')

    d.panel('B',236,316,'Thinking: targeted retrieval')
    d.text('Prompt\nspan',30,304,95,56,size=23,bold=True)
    centers=[264,450,636]
    for k,cx in zip([3,4,5],centers):
        needle(d,k,cx-41,312)
        d.edge([(cx,394),(cx,352)],color=QWEN,width=2,arrow=True)
    for x in [186,347,533,706]:d.text('···',x,312,20,40,size=23,color=base.MUTED,align='center')
    d.text('CoT\ntrace',30,385,95,56,size=23,bold=True)
    d.text('···',115,394,18,38,size=23,color=base.MUTED,align='center')
    for k,cx in zip([2,3,4],centers):
        d.text(f'{k}. City',cx-124,394,62,38,size=20.5)
        query(d,f'<marker {k}>',cx-56,394,112,size=20.5)
    d.text('···',714,394,20,38,size=23,color=base.MUTED,align='center')
    d.text('Final\nanswer',30,467,100,56,size=23,bold=True)
    query(d,'<Ans>',300,476,106)
    d.text('Total: 10',434,475,115,40,size=26)
    outcome(d,568,478,True)
    d.edge([(784,283),(784,545)],color=base.RULE,width=1)
    d.text('Attention from CoT markers to needle span',803,246,454,30,size=23,bold=True,align='center')
    matrix=state['attention']['B']['within_record_share_matrix']
    # Validate frozen needle shares against the captured token vectors. Raw
    # token bins are retained as provenance, not used for the displayed rows.
    token_source=HERE/'seed1255_native_token_attention.npz'
    with np.load(token_source) as payload:
        token_rows=np.asarray(payload['prompt_attention'],dtype=float)
        marker_ids=payload['from_occurrences'].tolist()
        prompt_spans=np.asarray(payload['record_spans'],dtype=int)
    assert marker_ids==B_MARKERS
    prompt_length=token_rows.shape[1]
    b_edges=np.linspace(0,prompt_length,171,dtype=int)
    b_bins=np.asarray([[row[b_edges[i]:b_edges[i+1]].sum() for i in range(170)]
                       for row in token_rows])
    for row,j in zip(token_rows,B_MARKERS):
        span_mass=np.asarray([row[s:e].sum() for s,e in prompt_spans])
        if not np.allclose(span_mass/span_mass.sum(),matrix[j],rtol=0,atol=5e-5):
            raise ValueError(f'Marker {j}: token attention and archived span shares disagree')
    # Larger needle cells and a horizontal legend keep the three lookups legible
    # while matching A's typography, colorbar alignment, and right-hand boundary.
    mx,my,cell_width=892,334,35.8
    row_height,row_pitch=30,62
    matrix_width=10*cell_width
    d.text('Needle',mx,275,matrix_width,26,size=23,align='center')
    for k in range(10):
        d.text(str(k+1),mx+k*cell_width,303,cell_width,25,size=21.5,align='center')
    d.text('⋮',826,307,28,24,size=22,color=base.MUTED,align='center')
    for row,j in enumerate(B_MARKERS):
        row_y=my+row*row_pitch
        d.text(f'Marker {j}',796,row_y,86,row_height,size=22,align='right')
        for k,v in enumerate(matrix[j]):
            d.rect(mx+k*cell_width,row_y,cell_width,row_height,heat(v),'#DCE3E7',.65,
                   f'Marker {j}; needle {k+1}; within-needle share={v:.12g}; row sums to 1')
        d.rect(mx+j*cell_width,row_y,cell_width,row_height,'none',QWEN,1.3)
        d.text(f'{matrix[j][j]:.0%}',mx+(j+.5)*cell_width-32,row_y+32,64,26,
               size=23,bold=True,align='center')
    d.text('⋮',826,496,28,24,size=22,color=base.MUTED,align='center')
    d.text('Relative mass',788,514,128,26,size=19,color=base.MUTED,align='right')
    for i in range(100):
        d.rect(bar_x+i*bar_w/100,518,bar_w/100+.1,9,heat(i/99),'none',0)
    for t,label in [(0,'0%'),(.5,'50%'),(1,'100%')]:
        d.text(label,bar_x+t*bar_w-32,530,64,23,size=19,color=base.MUTED,align='center')

    # Keep C headers aligned; uniformly scale each projected plot about its top center.
    c_count=0
    for original in ET.parse(frozen).getroot().find('diagram/mxGraphModel/root'):
        cell=original if original.tag=='mxCell' else original.find('mxCell')
        if cell is None or cell.get('parent')!='layer_C':continue
        obj=copy.deepcopy(original); copied=obj if obj.tag=='mxCell' else obj.find('mxCell')
        if copied.get('value'):
            copied.set('value', copied.get('value').replace('CoT-Reasoning', 'Thinking'))
        ident=copied.get('id') or obj.get('id')
        if copied.get('id'):copied.set('id','v4C_'+ident)
        else:obj.set('id','v4C_'+ident)
        g=copied.find('mxGeometry')
        if g is not None:
            if ident.startswith('v2_pca_'):
                x0=float(g.get('x',next((pt.get('x') for pt in g.iter('mxPoint') if pt.get('as')!='offset'),'320')))
                center=320 if x0<640 else 960
                if g.get('x') is not None:g.set('x',str(center+C_PLOT_SCALE*(float(g.get('x'))-center)))
                if g.get('y') is not None:g.set('y',str(C_Y+46+C_PLOT_SCALE*(float(g.get('y'))-544)))
                for key in ('width','height'):
                    if g.get(key) is not None:g.set(key,str(float(g.get(key))*C_PLOT_SCALE))
                for pt in g.iter('mxPoint'):
                    offset=pt.get('as')=='offset'
                    if pt.get('x') is not None:pt.set('x',str(float(pt.get('x'))*C_PLOT_SCALE if offset else center+C_PLOT_SCALE*(float(pt.get('x'))-center)))
                    if pt.get('y') is not None:pt.set('y',str(float(pt.get('y'))*C_PLOT_SCALE+(1-C_PLOT_SCALE) if offset else C_Y+46+C_PLOT_SCALE*(float(pt.get('y'))-544)))
                copied.set('style',re.sub(r'(fontSize|strokeWidth|endSize)=([0-9.]+)',lambda m:f'{m.group(1)}={float(m.group(2))*C_PLOT_SCALE:g}',copied.get('style','')))
            else:
                if g.get('y') is not None:g.set('y',str(float(g.get('y'))+C_Y-498))
                if float(g.get('width','0'))>1200:g.set('height',str(C_HEIGHT-1.5))
                for pt in g.iter('mxPoint'):
                    if pt.get('y') is not None and pt.get('as')!='offset':pt.set('y',str(C_Y+46+C_PLOT_SCALE*(float(pt.get('y'))-544)))
        d.root.append(obj);c_count+=1
    for layer,y in [('A',232),('B',556)]:
        d.layer='layer_'+layer;d.edge([(20,y),(1260,y)],color=base.RULE,width=1.1)
    d.write(HERE/'main_figure_v4.drawio')
    settings={'canvas':[1280,HEIGHT],'version':'v4','source_v3':state,'C':{'copied_cells':c_count,'header_translation':[0,62],'plot_scale':C_PLOT_SCALE,'plot_transform':'x about each plot center 320/960; y mapped from old plot top 544 to new plot top C_Y+46','height':C_HEIGHT,'data_camera_colors_unchanged':True},
              'A':{'source':str(raw),'head':[28,19],'bins':bins.tolist(),'clip_quantile':.985,'clip_value':high,'display':'0.035+0.87*sqrt(min(bin_mass/q98.5,1))','colorbar_raw_mass_ticks':[0,high/4,high],'colorbar_top_label':'greater than or equal to clip threshold','span_marker_positions':'v2 source geometry'},
              'B':{'head':[24,29],'source':str(token_source),'matrix_queries_by_needles':matrix,'displayed_markers':B_MARKERS,'displayed_matrix':[matrix[j] for j in B_MARKERS],'prompt_token_count':prompt_length,'prompt_spans':prompt_spans.tolist(),'bin_edges':b_edges.tolist(),'raw_prompt_bins':b_bins.tolist(),'prompt_attention_mass':token_rows.sum(axis=1).tolist(),'target_shares':[matrix[j][j] for j in B_MARKERS],'display_orientation':'three separated rows=Markers 2,3,4; ten equal-width needle-span cells per row','normalization':'each displayed row sums to 1 over the ten complete needle spans','display':'linear within-needle attention share; independent 0%-100% scale','target_annotation':'blue outlined next-needle cell with 23-unit bold percentage below','colorbar_orientation':'horizontal, aligned with A','raw_token_data_role':'retained for validation and provenance; not used as displayed bins'},
              'single_hue_attention':[ATTENTION_LOW,QWEN],
              'colorbar_labels':{'A':'Attention mass','B':'Relative mass'},
              'attention_arrow_color':QWEN,
              'query_border_color':QUERY,
              'palette_source':'figures/cot-reasoning/attention/build_figure.py: Qwen3-8B white-to-#168DCA',
              'spacing_revision':{'B_prompt_y':312,'B_trace_y':394,'B_answer_y':476,'B_row_pitch':82,'B_attention_gap':42,
                                  'B_matrix_width':matrix_width,'B_matrix_row_height':row_height,'B_matrix_row_pitch':row_pitch,'B_axis_font':21.5,'height_reduction':54,
                                  'B_row_labels':'Marker 2, Marker 3, Marker 4; vertical ellipses above and below',
                                  'B_column_labels':'Needle; indices 1-10','B_cell_width':cell_width,'B_percent_font':23},
              'schematic_note':'Numbered City items 2/3/4 abstract full item content. Blue outlined <marker 2/3/4> placeholders locate captured item-end attention readouts, attending to needles 3/4/5; labels are annotations, not generated tokens or needle markers. Marker 0 denotes initialization. No fixed newline or bullet token is asserted. <Ans> denotes the final answer query, not a literal special token.'}
    (HERE/'main_figure_v4_settings.json').write_text(json.dumps(settings,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'C_cells':c_count,'canvas':[1280,HEIGHT],'A_bins':len(bins),'B_displayed_matrix':[3,10],'B_retained_matrix':[10,10]}))

if __name__=='__main__':main()
