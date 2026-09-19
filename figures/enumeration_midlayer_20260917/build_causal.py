"""Build compact Enumeration causal figures from the completed, verified audits."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter, ScalarFormatter
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REPO = ROOT / "realistic"
sys.path.insert(0, str(REPO / "src"))
from realistic_niah_v6.aligned_reporting import continuation_summary
PAPER = ROOT / "runs/paper_figures/figures/enumeration_appendix"
MODELS = [("Qwen3-8B", "Qwen", "#168DCA"), ("Gemma4-E4B", "Gemma", "#E87824")]
MODES = ["enumeration_index", "enumeration_bullet"]
SOURCES, FIGURES = {}, {}


def read(path):
    raw = path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def cell(report, model, mode):
    rows = [c for c in report["cells"] if c["model"] == model and c["mode"] == mode]
    assert len(rows) == 1
    return rows[0]


def value(stat):
    return stat.get("estimate", stat.get("value"))


def curve(ax, x, stats, color, *, linestyle="-", marker=None, alpha=.13):
    y = [value(s) for s in stats]
    assert all(v is None or np.isfinite(v) for v in y)
    y = [np.nan if v is None else v for v in y]
    ci = np.asarray([s["ci95"] if s["ci95"] is not None else [np.nan, np.nan] for s in stats], dtype=float)
    ax.fill_between(x, ci[:, 0], ci[:, 1], color=color, alpha=alpha, linewidth=0)
    ax.plot(x, y, color=color, ls=linestyle, marker=marker, ms=3.5, lw=1.35)


def point(ax, x, stat, color, marker="o"):
    y, ci = value(stat), stat["ci95"]
    assert np.isfinite(y) and ci[0] <= y + 1e-10 <= ci[1] + 1e-10
    ax.errorbar(x, y, yerr=[[y-ci[0]], [ci[1]-y]], color=color, marker=marker,
                ms=4, lw=1, capsize=2, linestyle="none")


def models_legend(fig, y=.975):
    fig.legend([Line2D([], [], color=c, lw=1.6) for _, _, c in MODELS],
               [name for _, name, _ in MODELS], loc="center", bbox_to_anchor=(.5, y), ncol=2)


def save(fig, name, records):
    style.save(fig, name, records)
    raw = (OUT / (name + ".pdf")).read_bytes()
    (PAPER / (name + ".pdf")).write_bytes(raw)
    FIGURES[name] = style.FIGURES[name]
    plt.close(fig)


def retrieve(report):
    fig, axes = style.four_panels(height=4.0)
    records = []
    for row, mode in enumerate(MODES):
        for col, metric in enumerate(("next_city_failure", "final_exact_count_failure")):
            ax = axes[row][col]
            for model, _, color in MODELS:
                doses = cell(report, model, mode)["dose_response"]
                for key, ls in (("selected_minus_clean", "-"), ("random_minus_clean", "--")):
                    stats = [d["metrics"][metric][key] for d in doses]
                    curve(ax, [d["dose_k"] for d in doses], stats, color, linestyle=ls,
                          marker="o" if key.startswith("selected") else None)
                    records.extend(dict(model=model, mode=mode, metric=metric, condition=key,
                                        k=d["dose_k"], estimate=value(s), ci95=s["ci95"], random_control=d["random_control"])
                                   for d, s in zip(doses, stats))
            ax.set_xscale("log", base=2)
            ax.set(xticks=[1, 2, 4, 8, 32, 128], xlim=(.85, 153), ylim=(-.05, 1.08), yticks=[0, .5, 1])
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            title = f'{"ABCD"[2*row+col]}. {mode.split("_")[-1].title()}: ' + ("next city" if col == 0 else "final count")
            style.panel(ax, title, "Failure increase" if col == 0 else None, "Ablated heads")
    models_legend(fig)
    fig.legend([Line2D([], [], color=".25", ls=ls) for ls in ("-", "--")], ["Selected", "Random"],
               loc="center", bbox_to_anchor=(.5, .025), ncol=2)
    save(fig, "enumeration_retrieve_aligned", records)


def head_scores(report):
    fig, axes = style.four_panels(height=4.0)
    archive = REPO / "work/et2/src"
    inventory_path = archive / "backup_manifest.json"
    assert hashlib.sha256(inventory_path.read_bytes()).hexdigest() == report["source_inventory_sha256"]
    inventory = read(inventory_path)
    records = []
    for row, mode in enumerate(MODES):
        for col, (model, short, _) in enumerate(MODELS):
            relative = f"fresh_retrieve_gpu_v2/jobs/localize/{model}/{mode}/ranking.json"
            path = archive / relative
            assert hashlib.sha256(path.read_bytes()).hexdigest() == inventory["files"][relative]["sha256"]
            ranking = read(path)
            layers = sorted({r["layer"] for r in ranking})
            heads = sorted({r["head"] for r in ranking})
            assert len(ranking) == len(layers) * len(heads) == (1152 if col == 0 else 56)
            matrix = np.full((len(heads), len(layers)), np.nan)
            for r in ranking:
                matrix[heads.index(r["head"]), layers.index(r["layer"])] = r["score"]
            assert np.isfinite(matrix).all() and matrix.min() >= 0 and matrix.max() <= 1
            ax = axes[row][col]
            if row == 0:
                ax.set_position([.10 if col == 0 else .61, .56, .365, .28])
            im = ax.imshow(matrix, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
            for r in ranking[:128 if col == 0 else 6]:
                ax.add_patch(Rectangle((layers.index(r["layer"])-.5, heads.index(r["head"])-.5), 1, 1, fill=False,
                                       edgecolor="black", linewidth=.35 if col == 0 else .8))
            ix = [0, 11, 23, 35] if col == 0 else list(range(7))
            iy = [0, 7, 15, 23, 31] if col == 0 else [0, 3, 7]
            ax.set(xticks=ix, xticklabels=[layers[x]+1 for x in ix], yticks=iy, yticklabels=[heads[y]+1 for y in iy])
            style.panel(ax, f'{"ABCD"[2*row+col]}. {mode.split("_")[-1].title()}: {short}', "Head" if col == 0 else None,
                        "Layer" if row == 1 else None)
            ax.grid(False)
            records.extend(dict(model=model, mode=mode, **r) for r in ranking)
    cax = fig.add_axes([.53, .962, .37, .014])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, .5, 1])
    cb.ax.tick_params(labelsize=9, pad=2, length=2)
    fig.text(.10, .968, "Target-record attention", fontsize=9, va="center")
    save(fig, "enumeration_head_scores_aligned", records)


def update(report):
    fig, axes = style.four_panels(height=4.0)
    records = []
    for col, mode in enumerate(MODES):
        for mi, (model, _, color) in enumerate(MODELS):
            groups = cell(report, model, mode)["groups"]
            for di, direction in enumerate(("forward", "backward")):
                marker, ls = ("o", "-") if di == 0 else ("D", "--")
                for j, scope in enumerate(("endpoint", "four_token_tail", "item_span")):
                    g = next(g for g in groups if g["scope"] == scope and g["direction"] == direction and g["donor_k"] == "all")
                    stat = g["paired_target_minus_self_adoption"]
                    point(axes[0][col], j + [-.18, -.06, .06, .18][2*mi+di], stat, color, marker)
                    records.append(dict(model=model, mode=mode, layer_one_based=cell(report, model, mode)["layer_one_based"], direction=direction, scope=scope,
                                        metric="paired_adoption", estimate=value(stat), ci95=stat["ci95"]))
                g = next(g for g in groups if g["scope"] == "item_span" and g["direction"] == direction and g["donor_k"] == "all")
                hops = [g["conditions"]["donor_to_receiver"]["continuation"][str(h)] for h in range(1, 5)]
                assert all(h["total_trials"] == 30 for h in hops)
                curve(axes[1][col], list(range(1, 5)), [h["conditional"] for h in hops], color, linestyle=ls, marker=marker)
                records.extend(dict(model=model, mode=mode, layer_one_based=cell(report, model, mode)["layer_one_based"], direction=direction, scope="item_span", metric="conditional_next_step",
                                    hop=h["hop"], numerator=h["successes"], denominator=h["conditional_eligible"],
                                    horizon_eligible=h["horizon_eligible"], **h["conditional"]) for h in hops)
        ax = axes[0][col]
        ax.set(xticks=[0, 1, 2], xticklabels=["Endpoint", "4 tokens", "Item span"], xlim=(-.45, 2.45), ylim=(-.10, 1.08), yticks=[0, .5, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.axhline(0, color=".55", lw=.6)
        style.panel(ax, f'{"AB"[col]}. {mode.split("_")[-1].title()}: patch scope', "Target minus self" if col == 0 else None)
        ax = axes[1][col]
        ax.set(xticks=[1, 2, 3, 4], xlim=(.85, 4.15), ylim=(-.05, 1.08), yticks=[0, .5, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        style.panel(ax, f'{"CD"[col]}. {mode.split("_")[-1].title()}: continued prefix', "Conditional success" if col == 0 else None, "Continuation step")
    models_legend(fig)
    fig.legend([Line2D([], [], color=".25", ls=ls, marker=m, ms=4) for ls, m in (("-", "o"), ("--", "D"))],
               ["Forward", "Backward"], loc="center", bbox_to_anchor=(.5, .025), ncol=2)
    save(fig, "enumeration_update_aligned", records)


def readout(answer, blanking):
    fig, axes = style.four_panels(height=3.55)
    records = []
    for col, mode in enumerate(MODES):
        ax = axes[0][col]
        for model, _, color in MODELS:
            layers = cell(answer, model, mode)["layerwise"]
            for condition, ls in (("full_donor_patch", "-"), ("self_patch", "--")):
                stats = [l["conditions"][condition]["donor_adoption"] for l in layers]
                curve(ax, [l["layer_one_based"] for l in layers], stats, color, linestyle=ls)
                records.extend(dict(model=model, mode=mode, layer=l["layer_one_based"], condition=condition,
                                    metric="answer_donor_adoption", estimate=value(s), ci95=s["ci95"]) for l, s in zip(layers, stats))
        ax.set(xlim=(1, 42), xticks=[1, 10, 20, 30, 40], ylim=(-.05, 1.08), yticks=[0, .5, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        style.panel(ax, f'{"AB"[col]}. {mode.split("_")[-1].title()}: answer state', "Target-count adoption" if col == 0 else None, "Layer")
        ax = axes[1][col]
        for mi, (model, _, color) in enumerate(MODELS):
            for ci, condition in enumerate(("clean", "prompt_records_blank", "trace_all_blank")):
                stat = cell(blanking, model, mode)["conditions"][condition]["accuracy"]
                point(ax, ci + (mi-.5)*.17, stat, color)
                records.append(dict(model=model, mode=mode, condition=condition, metric="blanking_accuracy",
                                    estimate=value(stat), ci95=stat["ci95"], numerator=stat["numerator"], denominator=stat["denominator"]))
        ax.set(xticks=[0, 1, 2], xticklabels=["Clean", "Record blank", "Trace blank"], xlim=(-.4, 2.4), ylim=(-.05, 1.08), yticks=[0, .5, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        style.panel(ax, f'{"CD"[col]}. {mode.split("_")[-1].title()}: trace blanking', "Exact-count accuracy" if col == 0 else None)
    models_legend(fig)
    fig.legend([Line2D([], [], color=".25", ls=ls) for ls in ("-", "--")],
               ["Target state", "Self patch"], loc="center", bbox_to_anchor=(.5, .025), ncol=2)
    save(fig, "enumeration_readout_aligned", records)


if __name__ == "__main__":
    start = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--update-audit", type=Path, required=True)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location("thinking_appendix_style", ROOT / "figures/thinking_appendix_style_20260913/build_figures.py")
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    style.OUT = OUT
    style.setup()
    (OUT / "data").mkdir(exist_ok=True)
    base = REPO / "outputs/enumeration_alignment_20260916"
    reports = {name: read(base / directory / "audit.json") for name, directory in (
        ("retrieve", "retrieve_audit_1550"), ("read", "read_audit_v1"))}
    reports["answer"] = read(REPO / "outputs/enumeration_followup_20260917/audit_v1/answer_audit.json")
    reports["update"] = read(args.update_audit.resolve())
    assert all(r["status"] == "PASS" for r in reports.values())
    assert reports["update"]["schema"] == "enumeration_midlayer_combined_audit_v1"
    retrieve(reports["retrieve"])
    head_scores(reports["retrieve"])
    update(reports["update"])
    readout(reports["answer"], reports["read"])
    SOURCES.update({str(Path(__file__).relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "figures/thinking_appendix_style_20260913/build_figures.py": hashlib.sha256(Path(style.__file__).read_bytes()).hexdigest()})
    (OUT / "causal_manifest.json").write_text(json.dumps({"status": "PASS", "figures": FIGURES, "source_sha256": SOURCES,
        "font_profile": {"family": "Times New Roman", "titles_pt": 11, "axes_pt": 10, "ticks_legend_pt": 9, "width_inches": 6.5},
        "inference_or_refitting": False, "seconds": time.monotonic()-start}, indent=2) + "\n", encoding="utf-8")
    print("PASS: complete head scores and three causal figures built from verified audits.")
