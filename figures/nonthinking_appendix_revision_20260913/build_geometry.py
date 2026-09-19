"""Render frozen PCA coordinates in the main figure's visual format.

No model inference, fitting, layer selection, or condition-specific alignment.
Run from the project root with python -s <this file>.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
import numpy as np

START = time.perf_counter()
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / 'figures/nonthinking_ncc_selection_20260913'
MAIN = ROOT / 'figures/aurora_attention_pca_concept'
DATA = SOURCE / 'pca_plot_data.csv'
DOMAIN_DATA = SOURCE / 'domain_report_payload.json'
META = SOURCE / 'geometry_manifest.json'
STYLE_FILE = MAIN / 'main_figure_v4_settings.json'
GRID_FILE = MAIN / 'assets/v4_source_v3.drawio'
style = json.loads(STYLE_FILE.read_text(encoding='utf-8'))['source_v3']
metadata = json.loads(META.read_text(encoding='utf-8'))
INK, GRAY, GRID = '#30312E', '#64665F', '#E9E8E2'
COLORS = {int(k): v for k, v in style['palette']['count_colors'].items()}
MEAN_COLORS = {int(k): v for k, v in style['palette']['centroid_colors'].items()}
# One camera per comparison row. The shared half-turn places PC1/PC3 on
# the left/right as in the main figure, while rotating data and axes together.
CAMERAS = {'cue': {'yaw_deg': 180 + math.degrees(-.72), 'pitch_deg': -52.},
           'domain': {'yaw_deg': 180 + math.degrees(-.72), 'pitch_deg': math.degrees(-.45)}}
# Condition, legend label, marker, centroid-path style, centroid size.
CONDITIONS = {
    'cue': [('cue_present', 'Cue present', 'o', '-', 25),
            ('cue_absent', 'Cue absent', '^', '--', 34)],
    'domain': [('city', 'City', 'o', '-', 25),
               ('flower', 'Flower', '^', '--', 34),
               ('animal', 'Animal', 's', ':', 24)],
}


def load_rows():
    """Supplement the archived two-domain display with frozen Animal points."""
    with DATA.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 800
    payload = json.loads(DOMAIN_DATA.read_text(encoding='utf-8'))
    for model, letter in [('Qwen3-8B', 'C'), ('Gemma4-E4B', 'D')]:
        v = payload['models'][model]['non_thinking']['visualization']
        settings = next(p for p in metadata['panels'] if p['panel'] == letter)
        assert v['basis'] == 'StandardScaler + PCA3 fitted on city discovery rows only'
        assert v['fit_rows'] == settings['fit_rows'] == 200
        np.testing.assert_array_equal(v['explained_variance_ratio'], settings['evr'])
        points = v['points']
        indexed = {(p['entity_domain'], int(p['seed']), int(p['gold_count'])): p
                   for p in points}
        expected = {(c, seed, n) for c, *_ in CONDITIONS['domain']
                    for seed in range(1254, 1264) for n in range(1, 11)}
        assert len(points) == len(indexed) == 300 and set(indexed) == expected
        assert all(p['analysis_split'] == 'confirmation' for p in points)
        existing = [r for r in rows if r['analysis'] == 'domain' and r['model'] == model]
        assert len(existing) == 200
        for r in existing:
            assert int(r['layer_source_zero_based']) == v['layer']
            assert int(r['layer_display_one_based']) == v['layer'] + 1
            p = indexed[(r['condition'], int(r['seed']), int(r['count_or_index']))]
            np.testing.assert_array_equal([float(r[k]) for k in ('pc1', 'pc2', 'pc3')],
                                          [p[k] for k in ('x', 'y', 'z')])
        for p in points:
            if p['entity_domain'] == 'animal':
                rows.append({'analysis': 'domain', 'model': model, 'site': 'answer_query',
                             'layer_source_zero_based': v['layer'],
                             'layer_display_one_based': v['layer'] + 1,
                             'condition': 'animal', 'seed': p['seed'],
                             'count_or_index': p['gold_count'],
                             **dict(zip(('pc1', 'pc2', 'pc3'), (p[k] for k in ('x', 'y', 'z'))))})
    assert len(rows) == 1000
    return rows


def rotation(xyz, camera):
    yaw, pitch = [math.radians(camera[k]) for k in ('yaw_deg', 'pitch_deg')]
    x, y, z = np.asarray(xyz).T
    x1 = math.cos(yaw)*x + math.sin(yaw)*z
    z1 = -math.sin(yaw)*x + math.cos(yaw)*z
    return np.column_stack([x1, math.cos(pitch)*y-math.sin(pitch)*z1,
                           math.sin(pitch)*y+math.cos(pitch)*z1])


def grid_styles():
    cells = {c.get('id'): c for c in ET.parse(GRID_FILE).getroot().iter('mxCell')}
    result = []
    for idx in range(236, 243):
        s = dict(v.split('=', 1) for v in cells[f'v2_pca_{idx}'].get('style').split(';') if '=' in v)
        assert s['strokeColor'] == GRID
        result.append((float(s['opacity'])/100, float(s['strokeWidth'])*6.5*72/1280))
    return result


def panel(ax, rows, settings, title, analysis):
    width, height = ax.get_position().width*6.5*72, ax.get_position().height*5.25*72
    ax.set(xlim=(0, width), ylim=(0, height), aspect='equal')
    ax.set_axis_off()
    ax.text(0, 1.025, title, transform=ax.transAxes, ha='left', va='bottom', fontsize=11)
    ax.text(1, 1.025, f'PC1-3: {100*sum(settings["evr"]):.1f}%',
            transform=ax.transAxes, ha='right', va='bottom', color=GRAY, fontsize=9)
    xyz = np.asarray([[float(r[k]) for k in ('pc1', 'pc2', 'pc3')] for r in rows])
    specs = CONDITIONS[analysis]
    assert xyz.shape == (100*len(specs), 3) and np.isfinite(xyz).all()
    labels = np.array([int(r['count_or_index']) for r in rows])
    conditions = [c for c, *_ in specs]
    masks = [np.array([r['condition'] == c for r in rows]) for c in conditions]
    assert all(m.sum() == 100 for m in masks)
    assert all((m & (labels == k)).sum() == 10 for m in masks for k in range(1, 11))
    means = np.array([xyz[m & (labels == k)].mean(0) for m in masks for k in range(1, 11)])
    # Exactly one centering, RMS scale, depth transform, and screen fit across
    # all conditions, following project_cloud in build_main_figure.py.
    center = xyz.mean(0)
    rms = max(float(np.sqrt(np.mean(np.sum((xyz-center)**2, axis=1)))), 1e-8)
    rotated = rotation((xyz-center)/rms, CAMERAS[analysis])
    mean_rotated = rotation((means-center)/rms, CAMERAS[analysis])
    depth_scale = max(float(np.std(rotated[:, 2])), 1e-8)
    def perspective(r):
        return r[:, :2] / np.clip(1 + .09*r[:, 2:3]/depth_scale, .78, 1.22)
    point_xy, mean_xy = perspective(rotated), perspective(mean_rotated)
    combined = np.vstack([point_xy, mean_xy])
    low, high = combined.min(0), combined.max(0)
    span = np.maximum(high-low, 1e-8)
    fit = min((width-24)/span[0], (height-22)/span[1])
    shift = np.array([width, height])/2 - (low+high)*fit/2
    point_xy, mean_xy = point_xy*fit+shift, mean_xy*fit+shift
    assert np.all(point_xy > 0) and np.all(point_xy < [width, height])

    directions = rotation(np.eye(3), CAMERAS[analysis])[:, :2]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    origin = np.array([width*.50, height*.45])
    for d in directions:
        normal = np.array([-d[1], d[0]])
        for k, (alpha, lw) in zip(range(-3, 4), grid_styles()):
            anchor = origin + k*min(width, height)*.125*normal
            segment = np.vstack([anchor-400*d, anchor+400*d])
            ax.plot(segment[:, 0], segment[:, 1], color=GRID, alpha=alpha,
                    lw=lw, zorder=-2, clip_on=True)
    for j, d in enumerate(directions):
        end = origin + min(width*.36, height*.43)*d
        ax.annotate('', xy=end, xytext=origin,
                    arrowprops={'arrowstyle': '-|>', 'color': GRAY, 'lw': .7,
                                'mutation_scale': 8, 'alpha': .8, 'shrinkA': 0, 'shrinkB': 0},
                    zorder=-1)
        label = ax.text(*end, f'PC{j+1}', color=GRAY, fontsize=10,
                        ha='center', va='center', zorder=5)
        label._axis_end = end
        label._axis_direction = d
    ax.scatter(*origin, s=9, color=GRAY, alpha=.8, zorder=-1)

    nearest = 1-(rotated[:, 2]-rotated[:, 2].min())/max(np.ptp(rotated[:, 2]), 1e-8)
    ax._point_boxes = []
    for n, (_, _, marker, line, _) in enumerate(specs):
        mxy = mean_xy[n*10:(n+1)*10]
        ax.plot(mxy[:, 0], mxy[:, 1], color=GRAY, lw=.9, ls=line, alpha=.78, zorder=1)
    markers = {c: marker for c, _, marker, _, _ in specs}
    for idx in np.argsort(nearest):
        marker = markers[rows[idx]['condition']]
        size = 7+4*nearest[idx]
        ax.scatter(*point_xy[idx], s=size, color=COLORS[int(labels[idx])],
                   marker=marker, alpha=.44+.40*nearest[idx], edgecolor='white', lw=.3, zorder=2)
        ax._point_boxes.append((point_xy[idx], size))
    for n in reversed(range(len(specs))):
        _, _, marker, _, size = specs[n]
        mxy = mean_xy[n*10:(n+1)*10]
        ax.scatter(mxy[:, 0], mxy[:, 1], s=size, marker=marker,
                   facecolors=[MEAN_COLORS[k] for k in range(1, 11)],
                   edgecolors=INK, linewidths=.55, zorder=3+n*.01)
        ax._point_boxes.extend((p, size) for p in mxy)
    return {'title': title, 'rows': len(rows),
            'condition_counts': {c: int(m.sum()) for c, m in zip(conditions, masks)},
            'condition_markers': markers, 'camera': CAMERAS[analysis],
            'shared_center': center.tolist(), 'shared_rms_scale': rms,
            'shared_screen_scale': float(fit), 'axis_directions': directions.tolist(),
            'source_coordinate_range': [xyz.min(0).tolist(), xyz.max(0).tolist()]}


def place_labels(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        if not hasattr(ax, '_point_boxes'):
            continue
        for label in ax.texts:
            if not hasattr(label, '_axis_end'):
                continue
            end = ax.transData.transform(label._axis_end)
            options = []
            for distance in [5, 9, 14, 20, 27]:
                for d in [(1,-1), (-1,-1), (1,1), (-1,1), (1,0), (-1,0), (0,1), (0,-1)]:
                    label.set_position(ax.transData.inverted().transform(end+np.array(d)*distance*fig.dpi/72))
                    label.set_ha('left' if d[0]>0 else 'right' if d[0]<0 else 'center')
                    label.set_va('bottom' if d[1]>0 else 'top' if d[1]<0 else 'center')
                    b = label.get_window_extent(renderer)
                    outside = b.x0 < ax.bbox.x0+1 or b.x1 > ax.bbox.x1-1 or b.y0 < ax.bbox.y0+1 or b.y1 > ax.bbox.y1-1
                    collisions = 0
                    for p, size in ax._point_boxes:
                        x,y = ax.transData.transform(p);r = math.sqrt(size)*fig.dpi/144+1
                        collisions += b.x0-r<x<b.x1+r and b.y0-r<y<b.y1+r
                    for other in ax.texts:
                        if other is not label:
                            collisions += b.overlaps(other.get_window_extent(renderer))
                    options.append((10000*outside+1000*collisions+distance, label.get_position(), label.get_ha(), label.get_va()))
            _, pos, ha, va = min(options, key=lambda r:r[0])
            label.set_position(pos);label.set_ha(ha);label.set_va(va)


def verify(fig):
    fig.canvas.draw();renderer = fig.canvas.get_renderer()
    texts = [(t,t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text()]
    outside = [t.get_text() for t,b in texts if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5]
    overlaps = [[a.get_text(),b.get_text()] for i,(a,ab) in enumerate(texts)
                for b,bb in texts[i+1:] if min(ab.x1,bb.x1)-max(ab.x0,bb.x0)>.8
                and min(ab.y1,bb.y1)-max(ab.y0,bb.y0)>.8]
    collisions = []
    for ax in fig.axes:
        if not hasattr(ax,'_point_boxes'):continue
        for label in ax.texts:
            if not hasattr(label,'_axis_end'):continue
            box=label.get_window_extent(renderer)
            for p,size in ax._point_boxes:
                x,y=ax.transData.transform(p);r=math.sqrt(size)*fig.dpi/144
                if box.x0-r<x<box.x1+r and box.y0-r<y<box.y1+r:
                    collisions.append(label.get_text());break
    assert not outside and not overlaps and not collisions,(outside,overlaps,collisions)
    return {'text_outside':outside,'text_overlaps':overlaps,'axis_label_marker_overlaps':collisions}


def main():
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix',
                         'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                         'text.color':INK,'figure.facecolor':'white','savefig.facecolor':'white'})
    rows=load_rows()
    with (OUT/'pca_plot_data.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    fig=plt.figure(figsize=(6.5,5.25))
    panels=[]
    for analysis,y in [('cue',.562),('domain',.106)]:
        for col,(model,short) in enumerate([('Qwen3-8B','Qwen'),('Gemma4-E4B','Gemma')]):
            group=[r for r in rows if r['analysis']==analysis and r['model']==model]
            layers={int(r['layer_display_one_based']) for r in group};assert len(layers)==1
            letter=('AB' if analysis=='cue' else 'CD')[col]
            settings=next(p for p in metadata['panels'] if p['panel']==letter)
            ax=fig.add_axes([[.03,.53][col],y,.44,.320])
            panels.append(panel(ax,group,settings,f'{letter}. {short} L{layers.pop()}',analysis))
    for title,analysis,y in [('Opening cue: needle-end states','cue',.993),
                             ('Needle domain: answer-query states','domain',.537)]:
        fig.text(.5,y,title,fontsize=11,ha='center',va='top')
        handles=[Line2D([],[],color=GRAY,lw=.9,ls=line,marker=marker,ms=4)
                 for _, _, marker, line, _ in CONDITIONS[analysis]]
        labels=[label for _, label, *_ in CONDITIONS[analysis]]
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,y-.035),
                   ncol=len(labels),frameon=False,fontsize=9,handlelength=1.6,handletextpad=.4,
                   borderaxespad=0,borderpad=.05,columnspacing=2)
    cmap=ListedColormap([COLORS[k] for k in range(1,11)]);bounds=np.arange(.5,11,1)
    norm=BoundaryNorm(bounds,cmap.N)
    cax=fig.add_axes([.31,.035,.38,.017])
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',
                    boundaries=bounds,ticks=[1,10],drawedges=True)
    cb.dividers.set_color('white');cb.dividers.set_linewidth(.65)
    cb.outline.set_edgecolor(INK);cb.outline.set_linewidth(.6)
    cb.ax.tick_params(labelsize=9,length=2,pad=2,width=.6)
    fig.text(.5,.001,'Running index / final count',fontsize=9,ha='center',va='bottom')
    place_labels(fig);qa=verify(fig)
    for ext in ['pdf','png','svg']:fig.savefig(OUT/f'nonthinking_cue_domain_pca.{ext}',dpi=300)
    plt.close(fig)
    manifest={'status':'PASS','elapsed_seconds':time.perf_counter()-START,'pca_refitted':False,
              'inference':False,'individual_states':len(rows),'panel_data':panels,'layout_audit':qa,
              'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [DATA,DOMAIN_DATA,META,STYLE_FILE,GRID_FILE,MAIN/'build_main_figure.py']},
              'data_audit':{'original_city_flower_coordinates_identical':True,
                            'domain_basis':'Frozen city-discovery StandardScaler + unwhitened PCA3',
                            'domain_confirmation_seeds':list(range(1254,1264)),
                            'domain_counts':list(range(1,11))},
              'style':'Main-figure count colors, neutral PC triads, seven-line fading grids and weak perspective; appendix typography.',
              'projection':'Joint centering, joint RMS scale, common weak-perspective depth transform and isotropic screen scale for all conditions in each panel.',
              'font_points':{'title':11,'axis':10,'legend':9},'canvas_inches':[6.5,5.25],
              'versions':{'numpy':np.__version__,'matplotlib':matplotlib.__version__}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','seconds':manifest['elapsed_seconds'],'qa':qa}))


if __name__=='__main__':main()
