"""Measure physical text sizes and text bounding boxes before figure export."""
from itertools import combinations
import numpy as np

def inspect_figure(fig,stem):
    renderer=fig.canvas.get_renderer()
    texts=list(fig.texts)
    for ax in fig.axes:
        texts.extend(ax.texts)
        texts.extend([ax._left_title,ax.title,ax._right_title])
        if ax.axison or hasattr(ax,'zaxis'):
            for axis in [ax.xaxis,ax.yaxis]+([ax.zaxis] if hasattr(ax,'zaxis') else []):
                texts.append(axis.label)
                lo,hi=sorted(axis.get_view_interval())
                for tick in axis.get_major_ticks():
                    if lo-1e-7<=tick.get_loc()<=hi+1e-7:
                        texts.extend([tick.label1,tick.label2])
        if ax.get_legend() is not None: texts.extend(ax.get_legend().get_texts())
    for leg in fig.legends: texts.extend(leg.get_texts())
    texts=list({id(t):t for t in texts if t.get_visible() and t.get_text().strip()}.values())
    entries=[]; clipped=[]
    for t in texts:
        b=t.get_window_extent(renderer)
        if not np.isfinite(b.extents).all(): continue
        row={'text':t.get_text(),'font_pt':t.get_fontsize(),'bbox_px':b.extents.tolist()}
        entries.append(row)
        if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5:
            clipped.append(row)
    overlaps=[]
    for a,b in combinations(entries,2):
        x1,y1,x2,y2=a['bbox_px']; x3,y3,x4,y4=b['bbox_px']
        width=min(x2,x4)-max(x1,x3); height=min(y2,y4)-max(y1,y3)
        if width>1.2 and height>1.2:
            overlaps.append({'text_a':a['text'],'text_b':b['text'],'intersection_px':[width,height]})
    minimum=min(t['font_pt'] for t in entries)
    assert minimum>=9,(stem,minimum)
    result={'figure':stem,'physical_size_in':fig.get_size_inches().tolist(),'minimum_font_pt':minimum,
            'text_outside_canvas':clipped,'text_overlap_candidates':overlaps,'text_count':len(entries)}
    if clipped or overlaps: print('LAYOUT REVIEW',stem,result,flush=True)
    return result
