"""Shared typography measured from the manuscript's appendix figures.

All role sizes are FINAL PRINT sizes at a 6.5 inch manuscript linewidth.
For a larger construction canvas, scale by canvas_width / 6.5. Do not shrink
font sizes to make a crowded panel fit; enlarge/rearrange the panel instead.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_style import paper_font

PAPER_WIDTH_INCHES = 6.5
PAPER_PT = {"title": 11.0, "letter": 11.0, "axis": 10.0, "tick": 9.0,
            "legend": 9.0, "annotation": 9.0, "dense": 7.5}
FONT_FAMILY = paper_font()
TITLE_WEIGHT = "normal"
MODELS = ["Qwen3-8B", "Gemma4-E4B"]
COLORS = ["#168DCA", "#E87824"]
INK, AXIS, GRID = "#161923", "#8190A5", "#E7E8EE"
REFERENCE = "figures/nonthinking_appendix_style_20260911/build_figures.py"


def scaled_sizes(canvas_width=6.5):
    """Return Matplotlib role sizes that reproduce PAPER_PT at print width."""
    return {role: value * float(canvas_width) / PAPER_WIDTH_INCHES
            for role, value in PAPER_PT.items()}


def apply_style(canvas_width=6.5):
    """Register reference fonts, configure Matplotlib, and return role sizes."""
    import matplotlib.pyplot as plt
    sizes = scaled_sizes(canvas_width)
    scale = float(canvas_width) / PAPER_WIDTH_INCHES
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": sizes["tick"],
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "axes.labelcolor": INK, "text.color": INK,
        "axes.edgecolor": AXIS, "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": .7 * scale, "lines.linewidth": 1.5 * scale,
        "legend.frameon": False, "axes.labelsize": sizes["axis"],
        "axes.titlesize": sizes["title"], "axes.titleweight": TITLE_WEIGHT,
        "xtick.labelsize": sizes["tick"], "ytick.labelsize": sizes["tick"],
        "legend.fontsize": sizes["legend"],
    })
    return sizes


def font_profile():
    return {"reference": REFERENCE, "paper_width_inches": PAPER_WIDTH_INCHES,
            "paper_pt": dict(PAPER_PT), "family": FONT_FAMILY,
            "title_weight": TITLE_WEIGHT,
            "policy": "Final printed sizes; use appendix references, not the main-figure font profile."}
