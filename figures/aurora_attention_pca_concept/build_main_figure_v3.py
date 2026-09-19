#!/usr/bin/env python3
"""Build v3 as native, individually editable draw.io shapes.

A/B aggregate the existing seed-1255 captures over the same record spans.
C preserves the v2 projection with a uniform display zoom and cropped outer grid.
Export the resulting .drawio using export_main_figure_v3.ps1.
"""
from __future__ import annotations

import copy
import base64
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_style import stix_font_path
import re
from urllib.parse import quote
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
# Canonical layout: A, B, C stacked vertically in that order.
W, H = 1280, 988
A_HEIGHT, B_Y, B_HEIGHT, C_Y, C_HEIGHT = 232, 240, 250, 498, 490
PCA_ZOOM = 1.25
PCA_OLD_ORIGINS = ((314.,1007.),(959.,1007.))
PCA_NEW_ORIGINS = ((320.,C_Y+274.),(960.,C_Y+274.))
PCA_VIEWPORTS = ((32.,C_Y+46.,624.,C_Y+C_HEIGHT-12.),
                 (656.,C_Y+46.,1264.,C_Y+C_HEIGHT-12.))
PALETTE = {
    'Midnight Indigo':'#23165C', 'Polar Violet':'#6750E8',
    'Ice Cyan':'#00C2FF', 'Aurora Yellow':'#F6E36A',
    'Aurora Teal':'#00D4B4', 'Aurora Green':'#39E58C',
    'Polar Magenta':'#C04DFF', 'Sunset Pink':'#FF5FA2',
    'Night Black':'#161923', 'Snow White':'#F8FBFF',
    'Frost Gray':'#8190A5', 'Warm Brown':'#765347',
}
STYLE_PALETTE = {
    'Ink':'#30312E', 'Muted':'#64665F', 'Rule':'#DDDCD5',
    'Prompt border':'#B1B0A7', 'Prompt fill':'#FAFAF7',
    'Mechanism border':'#969182', 'Retrieval fill':'#F2F0E8',
    'Answer fill':'#E7E3D7', 'Attention':'#F16913',
    'Attention arrow':'#B87652', 'Wrong':'#AD6155', 'Correct':'#4F7A66',
}
PAPER = '#FFFFFF'
INK = STYLE_PALETTE['Ink']
BLUE = PALETTE['Ice Cyan']
TRACE = PALETTE['Aurora Teal']
PURPLE = PALETTE['Polar Violet']
DISPLAY_ROWS = (2,3,4)
FONT = 'Times New Roman'
MATH_FONT = 'STIXGeneral'
MATH_FONT_PATH = stix_font_path()
MATH_FONT_DATA = 'data:font/ttf;base64,'+base64.b64encode(MATH_FONT_PATH.read_bytes()).decode('ascii')
MATH_FONT_SOURCE = quote(MATH_FONT_DATA, safe='')
MAP_X, CELL = 884, 31
ANSWER_STAGE_LABEL = 'Answer\nconsolidation'
ATTENTION_TITLE = 'Attention to needle span'
AB_CARD_SCALE = .90
AB_FONT_SCALE = 1.0
AB_SMALL_FONT = 20.5
PRINT_WIDTH_MM = 165.1


def mix(a: str, b: str, t: float) -> str:
    av = [int(a[i:i+2], 16) for i in (1, 3, 5)]
    bv = [int(b[i:i+2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(x+(y-x)*t):02X}' for x, y in zip(av, bv))


def tint(color, strength):
    return mix(PAPER,color,strength)


MUTED = STYLE_PALETTE['Muted']
RULE = STYLE_PALETTE['Rule']
SPAN_BORDER = STYLE_PALETTE['Prompt border']
PROMPT_FILL = STYLE_PALETTE['Prompt fill']
# Keep the approved warm-neutral mechanisms; restore vivid orange charts.
# Standard Oranges anchors, with the previous lighter endpoint interval.
ATTENTION_STOPS = (
    '#FFF5EB','#FEE6CE','#FDD0A2','#FDAE6B','#FD8D3C',
    '#F16913','#D94801','#A63603','#7F2704',
)
ATTENTION_FILL = ATTENTION_STOPS[5]
ATTENTION_STROKE = STYLE_PALETTE['Attention arrow']
HEATMAP_MAX_POSITION = .625
TRACE_BASE = STATE_BASE = '#827D6D'
TRACE_BORDER = STATE_BORDER = STYLE_PALETTE['Mechanism border']
TRACE_FILL = STYLE_PALETTE['Retrieval fill']
STATE_FILL = STYLE_PALETTE['Answer fill']
WRONG_COLOR = STYLE_PALETTE['Wrong']
CORRECT_COLOR = STYLE_PALETTE['Correct']

# Immutable v2 source colors identify the copied count marks. The displayed
# gradient is redesigned independently, identically for both modes.
OLD_COUNT_COLORS = {
    1:'#6750E8',2:'#00A9D8',3:'#00A88F',4:'#2DBE77',5:'#A7C957',
    6:'#D6B52C',7:'#F29E4C',8:'#E76F51',9:'#D94B86',10:'#8E5DB7',
}
COUNT_STOPS = tuple(OLD_COUNT_COLORS.values())
COUNT_COLORS = {k: mix(c, '#000000', .08) for k,c in OLD_COUNT_COLORS.items()}
SCATTER_OPACITY = 88
SCATTER_SIZE_SCALE = 1.20
CENTROID_DIAMETER = 24.5
COUNT_FONT_SIZE = 16.4
TWO_DIGIT_FONT_SIZE = 16.4
# Ink-bounds corrections measured at the final 20.5-unit Times New Roman Bold
# size and 100x resolution, then mapped back through the 1.25x display zoom.
# mxGraph's SVG baseline is center+size/2-1.
# Offsets affect text only. The paired circle and the label box stay centered.
COUNT_LABEL_OFFSETS = {
    1: (.088,-1.656), 2: (.200,-1.656), 3: (.340,-1.776),
    4: (.128,-1.656), 5: (.032,-1.884), 6: (-.056,-1.772),
    7: (-.080,-1.884), 8: (0.,-1.792), 9: (.040,-1.772),
    10: (-.240,-1.772),
}


def luminance(color):
    values=[int(color[i:i+2],16)/255 for i in (1,3,5)]
    values=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in values]
    return sum(a*b for a,b in zip(values,(.2126,.7152,.0722)))


def white_label_background(color):
    # Darken along the same RGB hue, only as far as needed for white text.
    for amount in range(101):
        shade=mix(color,'#000000',amount/100)
        if 1.05/(luminance(shade)+.05)>=4.5:
            return shade
    raise AssertionError('No white-label background found')


CENTROID_COLORS={k:white_label_background(c) for k,c in COUNT_COLORS.items()}


def mass_color(v: float) -> str:
    # Preserve the measured 0..1 share, with a lighter orange color endpoint.
    position=min(1.,max(0.,v))*HEATMAP_MAX_POSITION*(len(ATTENTION_STOPS)-1)
    index=min(int(position),len(ATTENTION_STOPS)-2)
    return mix(ATTENTION_STOPS[index],ATTENTION_STOPS[index+1],position-index)


class Diagram:
    def __init__(self):
        self.file = ET.Element('mxfile', host='Electron', agent='Codex', version='31.3.2')
        page = ET.SubElement(self.file, 'diagram', id='main-figure-v3', name='Main figure v3')
        self.model = ET.SubElement(page, 'mxGraphModel', {
            'dx':str(W), 'dy':str(H), 'grid':'0', 'gridSize':'10', 'guides':'1',
            'tooltips':'1', 'connect':'1', 'arrows':'1', 'fold':'0', 'page':'1',
            'pageScale':'1', 'pageWidth':str(W), 'pageHeight':str(H),
            'background':'#FFFFFF', 'math':'0', 'shadow':'0',
            'extFonts':MATH_FONT+'^'+MATH_FONT_DATA,
        })
        self.root = ET.SubElement(self.model, 'root')
        ET.SubElement(self.root, 'mxCell', id='0')
        for name in ('A', 'B', 'C'):
            ET.SubElement(self.root, 'mxCell', id='layer_'+name, value='Panel '+name, parent='0')
        self.layer = 'layer_A'
        self.index = 0

    def ident(self, kind='shape'):
        self.index += 1
        return f'{kind}_{self.index}'

    def vertex(self, value, x, y, w, h, style, *, tooltip=None):
        # Register the embedded font in both the editor and export renderer.
        if f'fontFamily={MATH_FONT};' in style:
            style += f'fontSource={MATH_FONT_SOURCE};'
        id_ = self.ident()
        parent = self.root
        if tooltip:
            parent = ET.SubElement(parent, 'object', id=id_, label=value, tooltip=tooltip)
        attrs = {'value':value, 'style':style, 'vertex':'1', 'parent':self.layer}
        if not tooltip:
            attrs['id'] = id_
        cell = ET.SubElement(parent, 'mxCell', attrs)
        ET.SubElement(cell, 'mxGeometry', x=f'{x:.3f}', y=f'{y:.3f}',
                      width=f'{w:.3f}', height=f'{h:.3f}', **{'as':'geometry'})
        return cell

    def text(self, value, x, y, w, h=30, *, size=23, color=INK, bold=False, align='left'):
        font=FONT
        if value in ('···','⋯','⋮'):
            value='⋯' if value=='···' else value
            font=MATH_FONT
        return self.vertex(value, x,y,w,h,
            f'text;html=0;whiteSpace=wrap;overflow=hidden;strokeColor=none;fillColor=none;'
            f'align={align};verticalAlign=middle;fontFamily={font};fontSize={size};'
            f'fontColor={color};fontStyle={int(bold)};spacing=0;')

    def box(self, value, x,y,w,h, *, color=RULE, fill='#FFFFFF', size=23, bold=False, tooltip=None):
        font,html=FONT,0
        if value.startswith('N') and all(c in '₀₁₂₃₄₅₆₇₈₉ₖ₊' for c in value[1:]) and len(value)>1:
            index=value[1:].translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉ₖ₊','0123456789k+'))
            index=index.replace('k','𝑘')
            # STIXGeneral contains the mathematical italic alphabet, but does
            # not contain Unicode subscript digits. Real subscript typography
            # keeps the numeral upright and avoids a fallback to Cambria Math.
            value='𝑁<sub style="font-size:70%;vertical-align:-0.25em;line-height:0;">'+index+'</sub>'
            font,html=MATH_FONT,1
        return self.vertex(value,x,y,w,h,
            f'rounded=1;arcSize=9;absoluteArcSize=1;html={html};whiteSpace=wrap;'
            f'fillColor={fill};strokeColor={color};strokeWidth=1.5;'
            f'fontFamily={font};fontSize={size};fontColor={INK};fontStyle={int(bold)};'
            'align=center;verticalAlign=middle;spacing=0;', tooltip=tooltip)

    def rect(self,x,y,w,h,fill,stroke='none',sw=1,tooltip=None):
        return self.vertex('',x,y,w,h,
            f'rounded=0;html=0;fillColor={fill};strokeColor={stroke};strokeWidth={sw};',
            tooltip=tooltip)

    def edge(self, points, *, color=INK, width=1.8, arrow=False, dashed=False, curve=False):
        cell=ET.SubElement(self.root,'mxCell',{
            'id':self.ident('edge'), 'value':'', 'parent':self.layer,'edge':'1',
            'style':f'edgeStyle=none;html=0;rounded=1;curved={int(curve)};'
            f'strokeColor={color};strokeWidth={width};startArrow=none;'
            f'endArrow={"classic" if arrow else "none"};endFill=1;endSize=8;'
            f'dashed={int(dashed)};dashPattern=5 4;',
        })
        g=ET.SubElement(cell,'mxGeometry',relative='1',**{'as':'geometry'})
        for point,role in ((points[0],'sourcePoint'),(points[-1],'targetPoint')):
            ET.SubElement(g,'mxPoint',x=str(point[0]),y=str(point[1]),**{'as':role})
        if len(points)>2:
            arr=ET.SubElement(g,'Array',**{'as':'points'})
            for x,y in points[1:-1]:
                ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
        return cell

    def panel(self, name,y,h,title):
        self.layer='layer_'+name
        # White backgrounds preserve the export bounds without an outer frame.
        self.rect(.75,y+.75,W-1.5,h-1.5,'#FFFFFF','none',0)
        self.text(name,20,y+13,28,34,size=28,bold=True)
        self.text(title,65,y+11,W-90,40,size=29,bold=True)

    def write(self, path):
        ET.indent(self.file,space='  ')
        ET.ElementTree(self.file).write(path,encoding='utf-8',xml_declaration=True)


def record_axis(d,y):
    d.text('Prompt record',MAP_X,y-34,10*CELL,26,size=22,align='center')
    for j in range(10):
        d.text(str(j+1),MAP_X+j*CELL,y,CELL,25,size=20,align='center')


def sources(d,y,*,targeted=False):
    d.text('Prompt span',48,y-37,650,29,size=23,bold=True)
    entries = [('N₁',58),('N₂',180),('N₃',302),('N₁₀',556)]
    if targeted:
        entries=[('N₁',58),('Nₖ',180),('Nₖ₊₁',302),('N₁₀',556)]
    for idx,(label,x) in enumerate(entries):
        active=targeted and idx==2
        d.box(label,x,y,82,43,color=ATTENTION_STROKE if active else SPAN_BORDER,
              fill=tint(ATTENTION_FILL,.08) if active else PROMPT_FILL,size=23,bold=active)
    d.text('···',447,y,52,43,size=29,align='center',color=MUTED)
    d.text('···',668,y,42,43,size=29,align='center',color=MUTED)


def panel_a(d,shares):
    d.panel('A',0,A_HEIGHT,'Non-thinking')
    sources(d,80)
    # Shift the retrieval / consolidation / output group together.
    dx=-56
    # Arrowheads ONLY denote attention: source query -> attended record keys.
    for x,tx in zip((99,221,343,597),(260,290,325,355)):
        d.edge([(tx+dx,159),(x,142),(x,123)],color=ATTENTION_STROKE,width=2.1,curve=True,arrow=True)
    d.box('Broad retrieval',210+dx,159,195,64,color=TRACE_BORDER,fill=TRACE_FILL,size=24)
    d.edge([(405+dx,191),(456+dx,191)],color=INK)
    d.box(ANSWER_STAGE_LABEL,456+dx,155,158,72,color=STATE_BORDER,fill=STATE_FILL,size=24)
    d.edge([(614+dx,191),(652+dx,191)],color=INK)
    d.box('Total: 8',652+dx,172,116,38,color=SPAN_BORDER,fill=PAPER,size=24,bold=True)
    d.text('wrong',652+dx,208,116,24,size=AB_SMALL_FONT,bold=True,color=WRONG_COLOR,align='center')
    d.edge([(784,51),(784,224)],color=RULE,width=1,arrow=False)
    # Bar height exposes A's distributed attention without a faint heatmap or
    # a nonlinear color transform. Units and the 0..20% scale are explicit.
    d.text(ATTENTION_TITLE,819,15,415,29,size=23,bold=True,align='center')
    bottom, plot_h, maximum = 171, 95, .20
    for val in (0,.10,.20):
        yy=bottom-plot_h*val/maximum
        d.edge([(MAP_X,yy),(MAP_X+10*CELL,yy)],color=RULE,width=.8)
        d.text(f'{val*100:.0f}%',801,yy-12,72,24,size=19,color=MUTED,align='right')
    for j,v in enumerate(shares):
        height=plot_h*v/maximum
        d.rect(MAP_X+j*CELL+5,bottom-height,CELL-10,height,ATTENTION_FILL,'none',0,
               f'Final query; record N{j+1}; within-record share={v:.10f}')
        d.text(str(j+1),MAP_X+j*CELL,bottom+5,CELL,25,size=20,align='center')
    d.text('Needle index',MAP_X,200,10*CELL,28,size=23,color=MUTED,align='center')


def colorbar(d,gx,gy,gw):
    for i in range(100):
        d.rect(gx+i*gw/100,gy,gw/100+.02,9,mass_color(i/99))
    for value in (0,.5,1):
        px=gx+value*gw
        d.text(f'{value*100:.0f}%',px-25,gy+11,50,23,size=18,align='center',color=MUTED)


def vertical_colorbar(d,gx,gy,gh):
    for i in range(100):
        d.rect(gx,gy+i*gh/100,10,gh/100+.02,mass_color(1-i/99))
    for value in (0,.5,1):
        py=gy+(1-value)*gh
        d.text(f'{value*100:.0f}%',gx+16,py-13.5,51,27,
               size=AB_SMALL_FONT,align='left',color=MUTED)


def give_ab_more_space(d):
    """Keep compact cards but size every A/B label for the paper width."""
    cells=[c for c in d.root.iter('mxCell') if c.get('parent') in ('layer_A','layer_B')]
    resized={name:[] for name in ('layer_A','layer_B')}
    for cell in cells:
        style=cell.get('style','');g=cell.find('mxGeometry')
        if g is None:continue
        if cell.get('vertex')=='1' and style.startswith('rounded=1;'):
            x,y,w,h=[float(g.get(k,0)) for k in ('x','y','width','height')]
            cx,cy=x+w/2,y+h/2
            resized[cell.get('parent')].append((x,y,w,h,cx,cy))
            for name,value in (('x',cx-w*AB_CARD_SCALE/2),('y',cy-h*AB_CARD_SCALE/2),
                               ('width',w*AB_CARD_SCALE),('height',h*AB_CARD_SCALE)):
                g.set(name,f'{value:.4f}')
            style=re.sub(r'strokeWidth=([0-9.]+)',
                         lambda m:f'strokeWidth={float(m.group(1))*AB_CARD_SCALE:g}',style)
        if cell.get('value') not in ('A','B'):
            def print_font(match):
                previous=float(match.group(1))
                size=max(AB_SMALL_FONT,round(previous*AB_FONT_SCALE*2)/2)
                return f'fontSize={size:g}'
            style=re.sub(r'fontSize=([0-9.]+)',print_font,style)
        # A uses bar height as its magnitude encoding; slimmer bars add space.
        if cell.get('parent')=='layer_A' and cell.get('vertex')=='1' and f'fillColor={ATTENTION_FILL};' in style and float(g.get('width',0))==CELL-10:
            x,w=float(g.get('x')),float(g.get('width'))
            g.set('x',f'{x+w*(1-AB_CARD_SCALE)/2:.4f}')
            g.set('width',f'{w*AB_CARD_SCALE:.4f}')
        cell.set('style',style)
    for cell in cells:
        if cell.get('edge')!='1':continue
        g=cell.find('mxGeometry')
        endpoints=[p for p in g.iter('mxPoint') if p.get('as') in ('sourcePoint','targetPoint')]
        for point in endpoints:
            px,py=float(point.get('x')),float(point.get('y'))
            for x,y,w,h,cx,cy in resized[cell.get('parent')]:
                inside=x-1e-6<=px<=x+w+1e-6 and y-1e-6<=py<=y+h+1e-6
                boundary=min(abs(px-x),abs(px-x-w),abs(py-y),abs(py-y-h))<1e-6
                if inside and boundary:
                    point.set('x',f'{cx+(px-cx)*AB_CARD_SCALE:.4f}')
                    point.set('y',f'{cy+(py-cy)*AB_CARD_SCALE:.4f}')
                    break
        if endpoints and max(float(p.get('x')) for p in endpoints)<780:
            style=re.sub(r'(strokeWidth|endSize)=([0-9.]+)',
                         lambda m:f'{m.group(1)}={float(m.group(2))*AB_CARD_SCALE:g}',cell.get('style',''))
            cell.set('style',style)


def panel_b(d,matrix):
    y=B_Y
    d.panel('B',y,B_HEIGHT,'Thinking')
    # Two mechanism rows: prompt spans above, trace and answer formation below.
    d.text('Prompt\nspan',30,y+61,110,56,size=23,bold=True,align='left')
    centers=(215,322,429)
    subscripts=str.maketrans('0123456789','₀₁₂₃₄₅₆₇₈₉')
    for k,cx in zip(DISPLAY_ROWS,centers):
        d.box('N'+str(k+1).translate(subscripts),cx-36,y+69,72,40,
              color=SPAN_BORDER,fill=PROMPT_FILL,size=24)
        d.edge([(cx,y+156),(cx,y+109)],color=ATTENTION_STROKE,width=2.6,arrow=True)
    # A shared label beside the three attention arrows, clear of both rows.
    d.text('Targeted Retrieval',462,y+119,245,30,size=23,color=ATTENTION_STROKE)
    # Interior ellipses omit surrounding prompt text, not needle indices.
    for x in (143,257.5,364.5,483):
        d.text('···',x,y+69,22,40,size=24,color=MUTED,align='center')
    d.text('Thinking\ntrace',30,y+162,110,56,size=23,bold=True,align='left')
    for i,(k,cx) in enumerate(zip(DISPLAY_ROWS,centers)):
        d.box(f'After\nitem {k}',cx-45,y+156,90,68,color=TRACE_BORDER,
              fill=TRACE_FILL,size=23)
        if i<2:
            d.edge([(cx+45,y+190),(centers[i+1]-45,y+190)],color=MUTED,width=1.5)
    for x in (143,483):
        d.text('···',x,y+167,22,46,size=24,color=MUTED,align='center')
    d.edge([(165,y+190),(170,y+190)],color=MUTED,width=1.5)
    d.edge([(474,y+190),(483,y+190)],color=MUTED,width=1.5)
    # Omitted later trace steps remain explicit before terminal consolidation.
    d.edge([(505,y+190),(510,y+190)],color=MUTED,width=1.5)
    d.box(ANSWER_STAGE_LABEL,510,y+154,158,72,color=STATE_BORDER,fill=STATE_FILL,size=24)
    d.edge([(668,y+190),(664,y+190)],color=INK)
    d.box('Total: 10',664,y+171,116,38,color=SPAN_BORDER,fill=PAPER,size=24,bold=True)
    d.text('correct',664,y+207,116,24,size=AB_SMALL_FONT,bold=True,color=CORRECT_COLOR,align='center')
    d.edge([(784,y+53),(784,y+B_HEIGHT-14)],color=RULE,width=1,arrow=False)
    d.text(ATTENTION_TITLE,801,y+15,449,29,size=23,bold=True,align='center')
    map_x,cell=904,29
    for j in range(10):
        d.text(str(j+1),map_x+j*cell,y+48,cell,25,size=20,align='center')
    d.text('⋮',831,y+48,22,23,size=22,color=MUTED,align='center')
    for display_index,k in enumerate(DISPLAY_ROWS):
        row=matrix[k]
        my=y+79+display_index*54
        d.text(f'After item {k}',790,my,104,27,size=AB_SMALL_FONT,align='right')
        for j,v in enumerate(row):
            d.rect(map_x+j*cell,my,cell,26,mass_color(v),RULE,.7,
                   f'Completed items k={k}; record N{j+1}; within-record share={v:.10f}')
        # One direct target label per row, below its actual column. This is
        # measured mass, not an idealized one-hot target or a row-max summary.
        target=map_x+k*cell
        d.rect(target,my,cell,26,'none',ATTENTION_FILL,1.8)
        d.text(f'{row[k]*100:.0f}%',target-12,my+28,cell+24,26,
               size=AB_SMALL_FONT,bold=True,color=INK,align='center')
    d.text('⋮',831,y+222,22,23,size=22,color=MUTED,align='center')
    vertical_colorbar(d,1211,y+79,150)


def attention_details(matrix,shares):
    """A second editable page preserves all queries without crowding page 1."""
    detail=Diagram()
    page=detail.file.find('diagram')
    page.set('id','complete-attention-evidence')
    page.set('name','Attention - all 10 lookups')
    detail.model.set('pageWidth','860')
    detail.model.set('pageHeight','790')
    detail.rect(.5,.5,859,789,'#FFFFFF',RULE,1)
    detail.text('Complete attention evidence',30,20,800,40,size=29,bold=True)
    detail.text('Qwen3-8B · seed 1255 · ten prompt records',30,66,800,29,size=23,color=MUTED)
    mx,cc=292,43
    detail.text('Needle index',mx,104,10*cc,28,size=23,align='center')
    for j in range(10):
        detail.text(str(j+1),mx+j*cc,139,cc,29,size=21,align='center')
    detail.text('Non-thinking\nfinal query',44,175,224,51,size=23,bold=True,align='right')
    for j,v in enumerate(shares):
        detail.rect(mx+j*cc,179,cc,43,mass_color(v),RULE,.8,
                    f'Final query; record N{j+1}; within-record share={v:.10f}')
    detail.text('Thinking',44,245,224,29,size=24,bold=True,align='right')
    for k,row in enumerate(matrix):
        yy=283+k*cc
        label='Start' if k==0 else f'After item {k}'
        detail.text(label,44,yy,224,cc,size=22,align='right')
        for j,v in enumerate(row):
            detail.rect(mx+j*cc,yy,cc,cc,mass_color(v),RULE,.8,
                        f'Completed items k={k}; record N{j+1}; within-record share={v:.10f}')
    detail.text('Within-record attention share',31,741,340,27,size=21,color=MUTED)
    colorbar(detail,415,732,300)
    return page


def clip_grid_segment(p0,p1,viewport):
    """Clip only the background grid, leaving data and coordinate axes intact."""
    xmin,ymin,xmax,ymax=viewport
    x,y=p0; dx,dy=p1[0]-x,p1[1]-y
    lo,hi=0.,1.
    for direction,distance in ((-dx,x-xmin),(dx,xmax-x),(-dy,y-ymin),(dy,ymax-y)):
        if abs(direction)<1e-12:
            if distance<0:return None
        elif direction<0:lo=max(lo,distance/direction)
        else:hi=min(hi,distance/direction)
        if lo>hi:return None
    return ((x+lo*dx,y+lo*dy),(x+hi*dx,y+hi*dy))


def panel_c(d,source):
    # Put the two mode names and accuracies on one header line. The caption
    # identifies C as the representation comparison without a repeated title.
    d.panel('C',C_Y,C_HEIGHT,'')
    d.text('Non-thinking',65,C_Y+11,260,34,size=26,bold=True)
    d.text('Thinking',698,C_Y+11,280,34,size=26,bold=True)
    d.text('NCC Accuracy: 46%',381,C_Y+13,237,30,size=23,bold=True,align='right')
    d.text('NCC Accuracy: 98%',1021,C_Y+13,237,30,size=23,bold=True,align='right')
    d.edge([(640,C_Y+50),(640,C_Y+C_HEIGHT-12)],color=RULE,width=1,arrow=False)
    copied=[]
    omitted_grid=[]
    count_labels=[]
    source_cells={c.get('id'):c for c in source.findall('.//mxCell')}
    for original in source.findall('.//mxCell'):
        try: id_=int(original.get('id',''))
        except ValueError: continue
        if not (236<=id_<=549): continue
        cell=copy.deepcopy(original)
        cell.set('id','v2_pca_'+str(id_))
        cell.set('parent','layer_C')
        g=cell.find('mxGeometry')
        if g is None: continue
        is_scatter=cell.get('style','').startswith('ellipse;') and float(g.get('width',0))==5.6
        is_centroid=cell.get('style','').startswith('ellipse;') and float(g.get('width',0))==15.5
        marker_colors=CENTROID_COLORS if is_centroid else COUNT_COLORS
        color_map={OLD_COUNT_COLORS[k]:marker_colors[k] for k in COUNT_COLORS}
        color_map.update({'#4F6176':MUTED,'#D4DBE4':'#E9E8E2',
                          '#D7DDE6':RULE,'#8190A5':'#929187',
                          '#55575E':'#707168','#161923':INK})
        style=cell.get('style','')
        style=re.sub(r'#[0-9A-Fa-f]{6}',lambda m:color_map.get(m.group(0).upper(),m.group(0)),style)
        value=cell.get('value','')
        if is_scatter:
            style=re.sub(r'opacity=[^;]+','opacity='+str(SCATTER_OPACITY),style)
        if value.isdigit() and int(value) in COUNT_COLORS:
            style=re.sub(r'fontColor=#[0-9A-Fa-f]{6}','fontColor='+PAPER,style)
            size=TWO_DIGIT_FONT_SIZE if len(value)==2 else COUNT_FONT_SIZE
            style=re.sub(r'fontSize=[^;]+','fontSize='+str(size),style)
            # Native SVG text keeps the same baseline in draw.io, PNG and PDF,
            # avoiding the HTML line-box rounding seen on small number labels.
            style+='convertToSvg=1;'
            # Derive the label box from its paired centroid, not independent
            # rounded coordinates. Equal margins keep the number at the center.
            cg=source_cells[str(id_-1)].find('mxGeometry')
            for axis,extent in (('x','width'),('y','height')):
                center=float(cg.get(axis))+float(cg.get(extent))/2
                g.set(axis,f'{center-float(g.get(extent))/2:.3f}')
            ox,oy=COUNT_LABEL_OFFSETS[int(value)]
            ET.SubElement(g,'mxPoint',x=str(ox),y=str(oy),**{'as':'offset'})
        cell.set('style',style)
        dx=83 if id_<=392 else 266
        dy=90
        if cell.get('vertex')=='1':
            g.set('x',f'{float(g.get("x",0))+dx:.3f}')
            g.set('y',f'{float(g.get("y",0))+dy:.3f}')
        for p in g.iter('mxPoint'):
            if p.get('as')=='offset': continue
            if p.get('x') is not None: p.set('x',f'{float(p.get("x"))+dx:.3f}')
            if p.get('y') is not None: p.set('y',f'{float(p.get("y"))+dy:.3f}')
        mode=0 if id_<=392 else 1
        ox,oy=PCA_OLD_ORIGINS[mode]
        nx,ny=PCA_NEW_ORIGINS[mode]
        def projected(x,y):
            return nx+PCA_ZOOM*(x-ox),ny+PCA_ZOOM*(y-oy)
        if cell.get('vertex')=='1':
            x,y=projected(float(g.get('x',0)),float(g.get('y',0)))
            g.set('x',f'{x:.4f}'); g.set('y',f'{y:.4f}')
            for extent in ('width','height'):
                g.set(extent,f'{float(g.get(extent,0))*PCA_ZOOM:.4f}')
        for point in g.iter('mxPoint'):
            if point.get('as')=='offset':
                point.set('x',f'{float(point.get("x",0))*PCA_ZOOM:.4f}')
                # Native SVG baseline includes a fixed -1 canvas-unit term.
                point.set('y',f'{float(point.get("y",0))*PCA_ZOOM+1-PCA_ZOOM:.4f}')
            elif point.get('x') is not None and point.get('y') is not None:
                x,y=projected(float(point.get('x')),float(point.get('y')))
                point.set('x',f'{x:.4f}');point.set('y',f'{y:.4f}')
        style=re.sub(r'(strokeWidth|endSize)=([0-9.]+)',
                     lambda m:f'{m.group(1)}={float(m.group(2))*PCA_ZOOM:g}',style)
        if value.isdigit() and int(value) in COUNT_COLORS:
            style=re.sub(r'fontSize=([0-9.]+)',
                         lambda m:f'fontSize={float(m.group(1))*PCA_ZOOM:g}',style)
        cell.set('style',style)
        if is_scatter or is_centroid:
            # Enlarge marks around their measured positions, without changing
            # the projected data geometry or the camera.
            cx=float(g.get('x'))+float(g.get('width'))/2
            cy=float(g.get('y'))+float(g.get('height'))/2
            diameter=(float(g.get('width'))*SCATTER_SIZE_SCALE
                      if is_scatter else CENTROID_DIAMETER)
            g.set('x',f'{cx-diameter/2:.4f}')
            g.set('y',f'{cy-diameter/2:.4f}')
            g.set('width',f'{diameter:.4f}')
            g.set('height',f'{diameter:.4f}')
        if '#D4DBE4' in original.get('style','').upper() and cell.get('edge')=='1':
            start=g.find("mxPoint[@as='sourcePoint']")
            end=g.find("mxPoint[@as='targetPoint']")
            segment=clip_grid_segment(
                (float(start.get('x')),float(start.get('y'))),
                (float(end.get('x')),float(end.get('y'))),PCA_VIEWPORTS[mode])
            if segment is None:
                omitted_grid.append(id_)
                continue
            for point,(x,y) in zip((start,end),segment):
                point.set('x',f'{x:.4f}');point.set('y',f'{y:.4f}')
        if value.isdigit() and int(value) in COUNT_COLORS:
            # Nearby centroid circles already overlap in the fixed projection.
            # Keep every centered white digit above all marker fills.
            count_labels.append(cell)
        else:
            d.root.append(cell)
        copied.append(id_)
    d.root.extend(count_labels)
    return copied,omitted_grid


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    settings_path=HERE/'main_figure_v2_settings.json'
    transitions_path=HERE/'seed1255_native_transitions.json'
    source_path=HERE/'main_figure_v2.drawio'
    old=json.loads(settings_path.read_text(encoding='utf-8'))
    capture=json.loads(transitions_path.read_text(encoding='utf-8'))
    assert (capture['seed'],capture['layer'],capture['head'])==(1255,24,29)
    masses=old['nonthinking']['record_masses']
    total=sum(masses)
    assert math.isclose(total,old['nonthinking']['needle_total_mass'],abs_tol=1e-7)
    shares=[v/total for v in masses]
    events={int(e['from_occurrence']):e for e in capture['events']}
    matrix=[]
    for k in range(10):
        event=events[k]
        assert event['to_occurrence']==k+1
        values=[float(r['mass']) for r in event['records']]
        assert len(values)==10 and all(v>=0 and math.isfinite(v) for v in values)
        row=[v/sum(values) for v in values]
        assert math.isclose(row[k],old['native_thinking']['target_shares'][k],abs_tol=1e-8)
        matrix.append(row)
    d=Diagram()
    panel_a(d,shares)
    panel_b(d,matrix)
    give_ab_more_space(d)
    copied,omitted_grid=panel_c(d,ET.parse(source_path))
    assert len(copied)+len(omitted_grid)==314
    # A single rule in each inter-panel gap replaces three rectangular frames.
    for name,yy in (('A',(A_HEIGHT+B_Y)/2),('B',(B_Y+B_HEIGHT+C_Y)/2)):
        d.layer='layer_'+name
        rule=d.edge([(20,yy),(W-20,yy)],color=RULE,width=1.1)
        rule.set('id','panel_separator_after_'+name)
    visible=[c.get('value','') for c in d.root.iter('mxCell')]
    assert not any('Native-thinking' in x or 'native thinking' in x for x in visible)
    d.file.append(attention_details(matrix,shares))
    d.write(HERE/'main_figure_v3.drawio')
    settings={
        'schema_version':'mechanism_attention_main_figure_v3',
        'canvas':[W,H], 'font_family':FONT,
        'typography':{'text_font':FONT,'math_font':MATH_FONT,
                      'math_scope':['needle variables and their subscripts','horizontal and vertical mathematical ellipses'],
                      'needle_typography':'STIX mathematical italic N with upright typeset subscript digits',
                      'math_font_file':'Matplotlib mpl-data/fonts/ttf/STIXGeneral.ttf',
                      'math_font_sha256':sha(MATH_FONT_PATH),
                      'font_embedding':'data-URI fontSource on mathematical cells, plus model extFonts; SVG export includes the font',
                      'license_file':'assets/fonts/LICENSE_STIX'},
        'layout':{'panel_A':[0,A_HEIGHT],'panel_B':[B_Y,B_HEIGHT],
                  'panel_C':[C_Y,C_HEIGHT],'B_mechanism_rows':2,
                  'panel_separation':'two single horizontal rules; no outer frames',
                  'AB_card_scale':AB_CARD_SCALE,'AB_font_scale':AB_FONT_SCALE,
                  'AB_font_minimum':AB_SMALL_FONT,'AB_font_rule':'20.5-unit minimum for small labels; 23-24 units for mechanism text',
                  'AB_spacing':'compact cards; taller multiline boxes and wider B answer-state box; connectors retargeted; bar heights and heatmap geometry unchanged',
                  'previous_canvas':[1280,1264],'area_reduction_at_fixed_width':1-H/1264,
                  'C_header':'one row for both mode names and NCC accuracies'},
        'display_names':['Non-thinking','Thinking'],
        'terminology':{
            'mode_names':['Non-thinking','Thinking'],
            'generated_sequence':'Thinking trace',
            'shared_answer_stage':'Answer consolidation',
            'consolidation_definition':'functional formation of an executable final answer-query count state; applies to the terminal stage of Thinking',
            'consolidation_scope':'supported by ordered partial trace-to-answer mediation and answer-state execution; does not establish identical operations or a shared unique circuit across modes',
            'readout_stage':'Answer readout',
            'readout_layout':'integrated into the thinking-trace row; no separate row label',
            'outcome_labels':{'A':'wrong','B':'correct'},
            'shared_attention_title':ATTENTION_TITLE,
            'B_retrieval_label':'Targeted Retrieval',
        },
        'palette':{'name':'Warm-neutral mechanisms with vivid Aurora data colors','anchors':STYLE_PALETTE,'original_aurora_reference':PALETTE,'paper':PAPER,
                   'tints':'mechanisms: warm gray and parchment; attention charts: sequential Oranges',
                   'shared_attention':{'colormap':'sequential Oranges',
                                       'stops':ATTENTION_STOPS,'bar_color':ATTENTION_FILL,
                                       'heatmap_endpoint':mass_color(1.),
                                       'mechanism_arrow_color':ATTENTION_STROKE,
                                       'heatmap_palette_interval':[0.,HEATMAP_MAX_POSITION],
                                       'mapping':'normalized share 0..1 maps linearly to Oranges palette positions 0..0.625; fixed bar color #F16913'},
                   'A_attention':'fixed vivid orange bar fill; magnitude encoded by height only',
                   'B_attention':'light-to-vivid orange heatmap; magnitude encoded by within-record share',
                   'shared_mechanism_roles':{
                       'prompt':{'border':SPAN_BORDER,'fill':PROMPT_FILL},
                       'retrieval_query':{'border':TRACE_BORDER,'fill':TRACE_FILL},
                       'answer_state':{'border':STATE_BORDER,'fill':STATE_FILL},
                       'output':{'border':SPAN_BORDER,'fill':PAPER},
                       'attention_arrow':ATTENTION_STROKE},
                   'outcome_labels':{'wrong':WRONG_COLOR,'correct':CORRECT_COLOR},
                   'thinking_trace':{'base':TRACE_BASE,'border':TRACE_BORDER,'fill':TRACE_FILL},
                   'answer_state':{'base':STATE_BASE,'border':STATE_BORDER,'fill':STATE_FILL},
                   'count_gradient_source':'restored continuous Aurora-derived count gradient; same colors in both modes',
                   'count_stops':COUNT_STOPS,'count_colors':COUNT_COLORS,
                   'count_color_adjustment':'source count colors with 8% black; centroid fills darkened further only for white-label contrast',
                   'centroid_colors':CENTROID_COLORS},
        'model':'Qwen3-8B','seed':1255,'gold_count':10,
        'attention':{
            'aggregation':'sum over each full registered prompt-record span',
            'normalization':'each query row divided by its sum across the ten record spans',
            'display':'shared sequential Oranges palette; A: fixed-color bars on explicit 0-20% axis; B: light-to-vivid orange palette on a linear 0-100% share scale',
            'A':{'layer':28,'head':19,'query':'final answer-prefix token',
                 'raw_record_masses':masses,'within_record_shares':shares,'needle_total_mass':total},
            'B':{'layer':24,'head':29,'main_display_rows_completed_items':list(DISPLAY_ROWS),
                 'main_colorbar':{'orientation':'vertical on the right','bottom_share':0.,'top_share':1.},
                 'main_display_row_selection':'user-requested item-end queries after items 2, 3, 4; earlier and later queries retained on page 2',
                 'rows_completed_items':list(range(10)),
                 'columns_prompt_records':list(range(1,11)),
                 'within_record_share_matrix':matrix,
                 'orientation':'rows are trace queries; columns are prompt records (transpose of v2 D)',
                 'complete_evidence_location':'draw.io page 2: Attention - all 10 lookups'},
        },
        'mechanism':{
            'arrow_semantics':'attention ONLY: source query to attended record keys; widths schematic',
            'other_mechanism_notation':'A: unnumbered retrieval/consolidation/output group shifted left by 56 units; an ellipsis after N10 denotes remaining prompt context. B: two mechanism rows; Prompt span on the left, centered on the needle row; trace queries continue horizontally through an ellipsis for later steps into answer consolidation and output; all continuation links are unheaded',
            'confirmed_stages_only':True,
            'scope':'connected functional chain; complete unique mediation is unproven',
            'progress_state':'distributed, content-bound event/progress state; no isolated scalar register claimed',
            'sources':['NiaH_Non-thinking_report.html','NiaH_Native-Thinking_report.html'],
        },
        'representation':{
            'source':'main_figure_v2.drawio', 'source_cell_ids':copied,
            'transforms':{'initial_left_translation':[83,90],'initial_right_translation':[266,90],
                          'uniform_display_zoom':PCA_ZOOM,'previous_origins':PCA_OLD_ORIGINS,
                          'new_origins':PCA_NEW_ORIGINS,'grid_clip_viewports':PCA_VIEWPORTS},
            'omitted_outer_grid_cell_ids':omitted_grid,
            'changes':'same 3D projection and PCA data with retained 1.25x display zoom; point centers, axes and paths unchanged; scatter diameters increased by 20%; 24.5-unit centroid circles and 20.5-unit centered white digits; all data points retained',
            'scatter_opacity':SCATTER_OPACITY/100,
            'scatter_size_scale':SCATTER_SIZE_SCALE,
            'scatter_diameter':5.6*PCA_ZOOM*SCATTER_SIZE_SCALE,
            'centroid_diameter':CENTROID_DIAMETER,
            'centroid_labels':{'color':PAPER,'font_size':COUNT_FONT_SIZE*PCA_ZOOM,
                               'two_digit_font_size':TWO_DIGIT_FONT_SIZE*PCA_ZOOM,
                               'alignment':'center/middle; label bounds centered on paired centroid; native SVG text with Times New Roman Bold ink-bounds offsets',
                               'glyph_offsets':{k:[x*PCA_ZOOM,y*PCA_ZOOM+1-PCA_ZOOM]
                                                for k,(x,y) in COUNT_LABEL_OFFSETS.items()},
                               'minimum_white_fill_contrast':4.5},
            'pca_basis':'separate discovery-fitted basis per mode; common camera only',
            'cohort':old['pca_display_cohort'], 'layer_selection':old['pca_layer_selection'],
            'ncc':old['pca_ncc'], 'camera':old['pca_shared_view'],
        },
        'input_sha256':{p.name:sha(p) for p in (settings_path,transitions_path,source_path)},
        'style_reference':{'sources':['https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-tracing','https://transformer-circuits.pub/2025/linebreaks/index.html'], 'scope':'visual adaptation for mechanism nodes only; charts retain vivid orange and Aurora-derived count colors; no adoption of source arrow semantics or mechanism claims'},
        'print_readability':{'reference_width_mm':PRINT_WIDTH_MM,
                             'reference_width_pt':468,
                             'mechanism_font_units':[23,24],
                             'small_label_font_units':AB_SMALL_FONT,
                             'count_label_font_units':COUNT_FONT_SIZE*PCA_ZOOM,
                             'layout':'vertical A-B-C, 1280x988; original point centers and camera retained'},
    }
    (HERE/'main_figure_v3_settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'drawio':str(HERE/'main_figure_v3.drawio'),'native_cells':len(list(d.root.iter('mxCell'))),
                      'pca_cells_preserved':len(copied),'A_share_sum':sum(shares),
                      'B_row_sums':[sum(row) for row in matrix]},indent=2))


if __name__=='__main__':
    main()
