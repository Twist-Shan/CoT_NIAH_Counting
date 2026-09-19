"""Render cue/domain controls with frozen NCC layer choices using prior styling."""
from pathlib import Path
import json
import math
import numpy as np
from reselect import module

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
selection = json.loads((OUT / 'selection.json').read_text())['selected']
plot = module('prior_cue_plot', ROOT / 'figures/nonthinking_cue_domain_pca_20260911/build_figure.py')
plot.OUT = OUT
plot.DOMAIN_SOURCE = OUT / 'domain_report_payload.json'
plot.MODELS = [(model, short, color, selection[model]['running_index']['layer'],
                selection[model]['domain_transfer']['layer'])
               for model, short, color, _, _ in plot.MODELS]
original_place_labels = plot.place_axis_labels

def place_labels(fig):
    """Keep the source layout, widening only the PC-label collision search."""
    original_place_labels(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        if not hasattr(ax, '_pca_markers'):
            continue
        for label in ax.texts:
            if not hasattr(label, '_axis_endpoint'):
                continue
            end = ax.transData.transform(label._axis_endpoint)
            candidates = []
            for distance in [5, 9, 14, 20, 27, 35, 45]:
                for dx, dy in [(1,1), (-1,-1), (1,-1), (-1,1), (0,1), (0,-1), (1,0), (-1,0)]:
                    label.set_position(ax.transData.inverted().transform(end + np.array([dx,dy])*distance*fig.dpi/72))
                    label.set_ha('left' if dx>0 else 'right' if dx<0 else 'center')
                    label.set_va('bottom' if dy>0 else 'top' if dy<0 else 'center')
                    b = label.get_window_extent(renderer)
                    outside = b.x0 < ax.bbox.x0+2 or b.x1 > ax.bbox.x1-2 or b.y0 < ax.bbox.y0+2 or b.y1 > ax.bbox.y1-2
                    collisions = 0
                    for point, size in ax._pca_markers:
                        x,y = ax.transData.transform(point)
                        r = math.sqrt(size)*fig.dpi/144 + 1
                        collisions += b.x0-r < x < b.x1+r and b.y0-r < y < b.y1+r
                    for other in ax.texts:
                        if other is not label:
                            q = other.get_window_extent(renderer)
                            collisions += b.overlaps(q)
                    candidates.append((10000*outside + 1000*collisions + distance,
                                       label.get_position(), label.get_ha(), label.get_va()))
            _, pos, ha, va = min(candidates, key=lambda v:v[0])
            label.set_position(pos)
            label.set_ha(ha)
            label.set_va(va)

plot.place_axis_labels = place_labels
plot.main()
path = OUT / 'manifest.json'
manifest = json.loads(path.read_text())
manifest['layer_selection'] = selection
manifest['domain_refit_audit'] = 'audit.json'
manifest['pca_refitted'] = {'cue': False, 'domain': 'Qwen city-discovery PCA3 refitted at NCC-selected L24; Gemma retained at L39'}
(OUT / 'geometry_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
