"""Render the PCA16 steering scan in the existing appendix style."""
from pathlib import Path
import importlib.util,json
HERE=Path(__file__).resolve().parent
STYLE=HERE.parent/'nonthinking_appendix_style_20260911/build_figures.py'
spec=importlib.util.spec_from_file_location('style',STYLE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
b.OUT=HERE/'rendered';b.OUT.mkdir(exist_ok=True)
b.plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":10,
    "axes.titlesize":11,"axes.titleweight":"normal","axes.labelsize":10,"xtick.labelsize":9,
    "ytick.labelsize":9,"legend.fontsize":9,"legend.frameon":False,"axes.linewidth":.7,
    "lines.linewidth":1.5,"lines.markersize":4,"pdf.fonttype":42,"ps.fonttype":42,
    "svg.fonttype":"none","text.color":b.INK,"axes.labelcolor":b.INK,"axes.titlecolor":b.INK,
    "xtick.color":b.INK,"ytick.color":b.INK,"figure.facecolor":"white","savefig.facecolor":"white"})
b.steering(HERE/'layer_summary.csv')
assert not any(a['outside'] or a['text_overlaps'] for a in b.AUDITS.values())
(b.OUT/'audit.json').write_text(json.dumps({'inputs':b.INPUTS,'figures':b.AUDITS,'display_layers':'one-based'},indent=2))
