"""Restyle the seven Non-thinking appendix figures from frozen result tables.

Run from the workspace root with python -s <this file>.
Typography follows the current synthetic/additional-task appendix figures:
Times New Roman, STIX mathematics, 11 pt regular titles, 10 pt axis labels,
9 pt ticks/legends. Dense heatmap ranks use 7.5 pt. No model runs or refits.
"""
from pathlib import Path
import csv
import gzip
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath
from matplotlib.ticker import PercentFormatter
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REPO = ROOT / "realistic"
REPORT = REPO / "reports/v4_non-thinking_causal"
OLD = ROOT / "figures/nonthinking_appendix_20260911"
MODELS = ["Qwen3-8B", "Gemma4-E4B"]
NAMES = ["Qwen", "Gemma"]
COLORS = ["#168DCA", "#E87824"]
INK, GRAY, GRID = "#161923", "#8190A5", "#E7E8EE"
INPUTS, AUDITS = {}, {}


def data_bytes(path):
    data = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data


def read_csv(path):
    data = data_bytes(path)
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    return list(csv.DictReader(data.decode("utf-8-sig").splitlines()))


def read_json(path):
    return json.loads(data_bytes(path).decode("utf-8-sig"))


def match(rows, **filters):
    return [r for r in rows if all(str(r[k]) == str(v) for k, v in filters.items())]


def panel(ax, title, ylabel=None, xlabel=None, percent=False):
    ax.set_title(title, loc="left", fontsize=11, fontweight="normal", pad=7)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=4)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.7, pad=3)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


def four_panels(height=4.65):
    fig = plt.figure(figsize=(6.5, height))
    axes = np.array([[fig.add_axes([x, y, .365, .27]) for x in [.10, .61]]
                     for y in [.60, .145]])
    return fig, axes


def model_handles():
    return [Line2D([], [], color=c, lw=1.5) for c in COLORS]


def legend(ax, **kwargs):
    return ax.legend(frameon=False, fontsize=9, borderaxespad=.2,
                     handlelength=1.6, handletextpad=.4, labelspacing=.25,
                     **kwargs)


def curve(ax, rows, color, ls="-", label=None):
    rows = sorted(rows, key=lambda r: int(float(r["layer"])))
    x = np.array([int(float(r["layer"])) for r in rows])
    y = np.array([float(r["mean"]) for r in rows])
    lo = np.array([float(r["ci95_low"]) for r in rows])
    hi = np.array([float(r["ci95_high"]) for r in rows])
    assert np.isfinite([x, y, lo, hi]).all() and np.all(lo <= hi)
    ax.plot(x, y, color=color, ls=ls, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=.12, linewidth=0)


def save(fig, name, ranks=None):
    # Source tables use zero-based blocks/heads; publication labels start at one.
    for ax in fig.axes:
        if ax.get_xlabel() in {'Layer', 'Intervention layer', 'Readout layer', 'Head index'}:
            ticks = ax.get_xticks()
            ax.set_xticks(ticks, [str(int(tick) + 1) for tick in ticks])
        if ax.get_ylabel() in {'Layer', 'Global layer'}:
            ticks = ax.get_yticks()
            labels = [str(int(t.get_text()) + 1) for t in ax.get_yticklabels()]
            ax.set_yticks(ticks, labels)
    # Center the actual digit outlines. Matplotlib's default vertical alignment
    # centers the font line box (including unused descender space), which makes
    # numeral-only labels appear above the center of small heatmap cells.
    rank_geometry = []
    for t, ax, head, row in ranks or []:
        t.set_horizontalalignment("left")
        t.set_verticalalignment("baseline")
        t.set_fontstyle("normal")
        ink = TextPath((0, 0), t.get_text(), size=t.get_fontsize(),
                       prop=t.get_fontproperties()).get_extents()
        center = ax.transData.transform([head, row])
        ink_center = np.array([(ink.x0+ink.x1)/2, (ink.y0+ink.y1)/2])
        baseline = center - ink_center * fig.dpi / 72
        t.set_position(ax.transData.inverted().transform(baseline))
        cell = ax.transData.transform([[head-.5, row-.5], [head+.5, row+.5]])
        rank_geometry.append({"rank": t.get_text(), "panel": ax.get_title(loc="left"),
            "cell_pdf_points": [float(cell[:,0].min()*72/fig.dpi),
                float((fig.bbox.height-cell[:,1].max())*72/fig.dpi),
                float(cell[:,0].max()*72/fig.dpi),
                float((fig.bbox.height-cell[:,1].min())*72/fig.dpi)],
            "ink_center_error_points": float(np.linalg.norm(
                ax.transData.transform(t.get_position()) + ink_center*fig.dpi/72-center)*72/fig.dpi)})
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text()]
    outside = [t.get_text() for t, b in texts if b.x0 < -.5 or b.y0 < -.5
               or b.x1 > fig.bbox.width + .5 or b.y1 > fig.bbox.height + .5]
    overlaps = []
    for i, (a, ab) in enumerate(texts):
        for b, bb in texts[i + 1:]:
            if (min(ab.x1, bb.x1) - max(ab.x0, bb.x0) > .8
                    and min(ab.y1, bb.y1) - max(ab.y0, bb.y0) > .8):
                overlaps.append([a.get_text(), b.get_text()])
    rank_fit = True
    for t, ax, head, row in ranks or []:
        box = t.get_window_extent(renderer)
        corners = ax.transData.transform([[head - .5, row - .5], [head + .5, row + .5]])
        rank_fit &= bool(box.x0 > corners[:, 0].min() and box.x1 < corners[:, 0].max()
                         and box.y0 > corners[:, 1].min() and box.y1 < corners[:, 1].max())
    AUDITS[name] = {"inches": list(fig.get_size_inches()), "outside": outside,
                    "text_overlaps": overlaps, "rank_labels_inside_cells": rank_fit,
                    "text_sizes": sorted({t.get_fontsize() for t, _ in texts}),
                    "text_families": sorted({t.get_fontfamily()[0] for t, _ in texts})}
    if ranks:
        AUDITS[name]["rank_centering"] = rank_geometry
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def full_head_scores():
    scores = read_csv(OLD / "nonthinking_head_scores.csv")
    members = read_csv(REPORT / "v4_4_causal_v2/full_span_topk/full_span_topk_membership.csv")
    fig = plt.figure(figsize=(6.5, 6.8))
    rank_labels = []
    specs = [(MODELS[0], COLORS[0], list(range(36)), 32, 32, [.095, .355, .775, .59],
              "A. Qwen: all heads"),
             (MODELS[1], COLORS[1], [5, 11, 17, 23, 29, 35, 41], 8, 6,
              [.095, .078, .775, .175], "B. Gemma: global-attention heads")]
    for model, color, layers, nheads, k, rect, title in specs:
        data = {(int(r["layer"]), int(r["head"])): float(r["broad_retrieval_score"])
                for r in scores if r["model"] == model}
        assert set(data) == {(l, h) for l in layers for h in range(nheads)}
        matrix = np.array([[data[l, h] for h in range(nheads)] for l in layers])
        ax = fig.add_axes(rect)
        cmap = LinearSegmentedColormap.from_list(model, ["#F7FAFC", color], N=256)
        mesh = ax.pcolormesh(np.arange(nheads+1)-.5, np.arange(len(layers)+1)-.5,
                            matrix, cmap=cmap, norm=Normalize(0, .6), shading="flat",
                            edgecolors="white", linewidth=.18, rasterized=False)
        ax.set(xlim=(-.5, nheads-.5), ylim=(len(layers)-.5, -.5))
        xt = list(range(0, nheads, 4)) + [31] if nheads == 32 else list(range(nheads))
        yt = list(range(0, 36, 3)) + [35] if nheads == 32 else list(range(len(layers)))
        ax.set_xticks(xt, xt)
        ax.set_yticks(yt, [layers[i] for i in yt])
        ax.set_xlabel("Head index", labelpad=4)
        ax.set_ylabel("Layer" if nheads == 32 else "Global layer", labelpad=5)
        ax.set_title(title, loc="left", fontsize=11, fontweight="normal", pad=8)
        ax.tick_params(length=0, pad=4)
        ax.spines[:].set_visible(False)
        selected = [r for r in members if r["model_label"] == model
                    and int(r["top_n"]) == 32 and int(r["rank"]) <= k]
        assert {int(r["rank"]) for r in selected} == set(range(1, k+1))
        for r in selected:
            layer, head, rank = int(r["layer"]), int(r["head"]), int(r["rank"])
            row = layers.index(layer)
            ax.add_patch(Rectangle((head-.5, row-.5), 1, 1, fill=False,
                                   edgecolor="#273743", linewidth=.65))
            t = ax.text(head, row, str(rank), ha="center", va="center", fontsize=7.5,
                        fontweight="bold", color=INK)
            rank_labels.append((t, ax, head, row))
        cax = fig.add_axes([.902, rect[1], .022, rect[3]])
        cb = fig.colorbar(mesh, cax=cax, ticks=[0, .2, .4, .6])
        cb.set_label(r"Broad retrieval score $B_h$", fontsize=10, labelpad=4)
        cb.ax.tick_params(labelsize=9, length=2, width=.5, pad=2)
        cb.outline.set_linewidth(.5)
        cb.outline.set_edgecolor(GRAY)
    save(fig, "nonthinking_full_head_scores", rank_labels)


def representation():
    ridge = read_csv(REPORT / "v4_4_extension/geometry/count_regression_summary.csv")
    ranks = read_csv(REPORT / "v4_4_extension/geometry/rank_and_compression_by_layer.csv")
    classifiers = {m: read_csv(REPORT / f"v4_4_extension/classification/classification_all_{s}/answer_classifier_metrics.csv")
                   for m, s in zip(MODELS, ["qwen", "gemma"])}
    fig = plt.figure(figsize=(6.5, 2.75))
    axes = [fig.add_axes([x, .30, .235, .55]) for x in [.085, .418, .75]]
    for ax, title, ylabel in zip(axes,
            ["A. Running-index readout", "B. Needle-end geometry", "C. Final-count readout"],
            [r"Held-out $R^2$", "Variance in top 3 PCs", "Classification accuracy"]):
        panel(ax, title, ylabel, "Layer")
        ax.set(xlim=(-1, 42), xticks=[0, 10, 20, 30, 40], ylim=(0, 1.05))
    for model, color, n in zip(MODELS, COLORS, [36, 42]):
        rr = sorted(match(ridge, model_label=model, role="prompt_running", algorithm="ridge"), key=lambda r:int(r["layer"]))
        gg = sorted(match(ranks, model_label=model, role="prompt_running"), key=lambda r:int(r["layer"]))
        assert len(rr) == len(gg) == n
        axes[0].plot([int(r["layer"]) for r in rr], [float(r["r2_mean"]) for r in rr], color=color)
        for metric, ls in [("total_variance_capture_k3", "-"), ("centroid_curve_capture_k3", "--")]:
            axes[1].plot([int(r["layer"]) for r in gg], [float(r[metric]) for r in gg], color=color, ls=ls)
        for algorithm, ls in [("nearest_centroid", "-"), ("logistic_l2", "--")]:
            cc = sorted(match(classifiers[model], algorithm=algorithm), key=lambda r:int(r["layer"]))
            assert len(cc) == n
            axes[2].plot([int(r["layer"]) for r in cc], [float(r["accuracy"]) for r in cc], color=color, ls=ls)
    for ax in axes[1:]:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[2].set(ylim=(0, .88), yticks=[0, .2, .4, .6, .8])
    axes[2].axhline(.1, color=GRAY, ls=":", lw=.8)
    for x, handles, labels in [(.2025, model_handles(), NAMES),
              (.5355, [Line2D([],[],color=INK,ls=s) for s in ["-","--"]], ["Individual states", "Centroids"]),
              (.8675, [Line2D([],[],color=INK,ls=s) for s in ["-","--"]], ["Nearest centroid", "Logistic"])]:
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(x, .006),
                   ncol=1, fontsize=9, frameon=False, handlelength=1.5,
                   handletextpad=.4, labelspacing=.2)
    save(fig, "nonthinking_representation_diagnostics")


def steering(source=None):
    rows = read_csv(Path(source) if source is not None else Path(__file__).resolve().parent.parent / "nonthinking_steering_pca16_20260921/layer_summary.csv")
    assert len(rows) == 312
    fig = plt.figure(figsize=(6.5, 2.65))
    axes = [fig.add_axes([x, .30, .365, .54]) for x in [.10, .61]]
    for ax, model, name, color, n, letter in zip(axes, MODELS, NAMES, COLORS, [36,42], "AB"):
        for beta, ls in [(-1, "--"), (1, "-")]:
            for condition, alpha in [("ridge", 1), ("random", .4)]:
                rr = [r for r in rows if r["model"] == model and float(r["beta"]) == beta and r["condition"] == condition]
                assert [int(r["layer"]) for r in rr] == list(range(n))
                ax.plot([int(r["layer"]) for r in rr], [float(r["mean_expected_count_shift"]) for r in rr],
                        color=color, ls=ls, alpha=alpha)
        panel(ax, f"{letter}. {name}", "Expected-count change", "Intervention layer")
        ax.set_xticks([0, 10, 20, 30] if n==36 else [0, 10, 20, 30, 40])
        ax.axhline(0, color=GRAY, lw=.7, zorder=0)
        low, high = ax.get_ylim()
        ax.set_yticks([tick for tick in ax.get_yticks() if low <= tick <= high])
    handles = [Line2D([], [], color=INK, alpha=a, ls=ls, label=f"{c}, $\\beta={b:+d}$")
               for c,a in [("Ridge",1),("Random",.4)] for b,ls in [(-1,"--"),(1,"-")]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.535,.015),
               ncol=4, frameon=False, fontsize=9, columnspacing=1., handlelength=1.8, handletextpad=.45)
    save(fig, "nonthinking_prompt_steering_10k")


def cue_domain():
    html = data_bytes(REPORT / "v4_4_2/realistic_niah_v4_4_2_mode_geometry_attention_report.html").decode("utf-8")
    marker = "const PROMPT_GEOM="
    cue,_ = json.JSONDecoder().raw_decode(html[html.index(marker)+len(marker):])
    domain = read_json(REPO / "work/domain_transfer_geometry/analysis/report_payload.json")
    assert domain["design"]["confirmation_seeds"] == list(range(1254,1264))
    fig, axs = four_panels(4.5)
    fig.legend(model_handles(), NAMES, loc="upper center", bbox_to_anchor=(.54,1.003), ncol=2, frameon=False)
    panel(axs[0,0], "A. Opening-cue removal", "Centroid CKA", "Layer")
    panel(axs[0,1], "B. Needle-end readout", r"Leave-one-seed-out $R^2$")
    panel(axs[1,0], "C. Answer-query domain transfer", "Nearest-centroid accuracy", "Evaluation domain", True)
    panel(axs[1,1], "D. Answer-query domain transfer", "Logistic accuracy", "Evaluation domain", True)
    for j,(model,color,L) in enumerate(zip(MODELS,COLORS,[8,9])):
        assert cue["coverage"][model]["paired_endpoint_states"] == 100
        rr = sorted([r for r in cue["statistics"].values() if r["model"] == model], key=lambda r:r["layer"])
        axs[0,0].plot([int(r["layer"]) for r in rr], [float(r["centroid_cka"]) for r in rr], color=color)
        r = cue["statistics"][f"{model}|prompt_counter|{L}"]
        xs, ys = [j-.14,j+.14], [r["r2_present"],r["r2_absent"]]
        axs[0,1].plot(xs,ys,color=color,lw=.8)
        for x,y,face in zip(xs,ys,[color,"white"]):
            axs[0,1].plot(x,y,"o",ms=5,mec=color,mfc=face,mew=1.)
            axs[0,1].text(x,y+.06,f"{y:.3f}",ha="center",fontsize=9)
        metrics = domain["models"][model]["non_thinking"]["metrics"]
        assert domain["models"][model]["non_thinking"]["audit"]["rows_by_domain"] == {"city":100,"flower":100,"animal":100}
        for ax, metric in [(axs[1,0],"ncc_balanced_accuracy"),(axs[1,1],"logistic_balanced_accuracy")]:
            ys = [metrics["count_by_evaluation_domain"][d][metric] for d in ["city","flower","animal"]]
            xs = np.arange(3)+(j-.5)*.065
            ax.plot(xs,ys,color=color,lw=1.2)
            for x,y,symbol in zip(xs,ys,["o","s","^"]):
                ax.plot(x,y,marker=symbol,color=color,ms=5)
                ax.annotate(f"{100*y:.0f}%",(x,y),xytext=(0,9 if j==0 else -14),textcoords="offset points",
                            ha="center",fontsize=9,color=color)
    axs[0,0].set(xlim=(-1,42),xticks=[0,10,20,30,40],ylim=(.94,1.003),yticks=[.94,.96,.98,1.])
    axs[0,0].axhline(1,color=GRAY,ls=":",lw=.8)
    axs[0,1].set(xticks=[0,1],xticklabels=["Qwen L9","Gemma L10"],xlim=(-.45,1.45),ylim=(0,1.08))
    handles = [Line2D([],[],marker="o",color=INK,ls="",mfc=f) for f in [INK,"white"]]
    axs[0,1].legend(handles,["Cue present","Cue absent"],loc="lower left",ncol=2,frameon=False,
                     fontsize=9,handletextpad=.35,columnspacing=.65,borderaxespad=.1,handlelength=1.)
    for ax in axs[1]:
        ax.set(xticks=[0,1,2],xticklabels=["City","Flower","Animal"],xlim=(-.25,2.25),ylim=(0,.75),yticks=[0,.2,.4,.6])
        ax.axhline(.1,color=GRAY,ls=":",lw=.8)
    save(fig,"nonthinking_cue_domain_controls")


def retrieval():
    ablation = read_csv(ROOT / "figures/nonthinking_appendix_20260910/retrieval/ablation_summary.csv")
    rw = read_csv(REPORT / "v4_4_4/read_write/metric_summary.csv.gz")
    fig,axs = four_panels(4.75)
    for j,(model,color,name) in enumerate(zip(MODELS,COLORS,NAMES)):
        ax=axs[0,j]
        panel(ax,f"{'AB'[j]}. {name}: ablation dose","Ablation effect","Number of ablated heads",True)
        for condition,ls,symbol,label in [("ranked","-","o","Ranked"),("layer_matched_random","--","D","Random")]:
            rr=sorted([r for r in ablation if r["model"]==model and r["metric"]=="relative_count_shift"
                       and r["condition"]==condition and int(r["k"])!=6],key=lambda r:int(r["k"]))
            x,y,lo,hi=[np.array([float(r[key]) for r in rr]) for key in ["k","mean","ci95_low","ci95_high"]]
            ax.plot(x,y,ls=ls,color=color,marker=symbol,ms=3,label=label)
            ax.fill_between(x,lo,hi,color=color,alpha=.12 if condition=="ranked" else .06,lw=0)
        if j==1:
            for condition in ["ranked","layer_matched_random"]:
                r=next(r for r in ablation if r["model"]==model and int(r["k"])==6
                       and r["metric"]=="relative_count_shift" and r["condition"]==condition)
                y,lo,hi=[float(r[k]) for k in ["mean","ci95_low","ci95_high"]]
                ax.errorbar(6,y,yerr=[[y-lo],[hi-y]],marker="*",ms=7,color=color,mfc="white",mew=.9,lw=.8,capsize=2)
        ax.set_xscale("log",base=2)
        ax.set_xticks([1,2,4,8,16,32],[1,2,4,8,16,32])
        ax.set(xlim=(.8,38),ylim=(-.015,.76))
        legend(ax,loc="upper left",ncol=2,columnspacing=.7)
        if j==1:
            ax.text(.035,.68,r"$\star$  $k=6$ extension",transform=ax.transAxes,fontsize=9)
    ax=axs[1,0]
    panel(ax,"C. Qwen: reading components","Source-directed transfer",percent=True)
    for i,metric in enumerate(["read_full_behavior_transport","read_routing_behavior_transport","read_value_behavior_transport"]):
        r=next(r for r in rw if r["metric"]==metric and r["stratum"]=="all")
        y,lo,hi=[float(r[k]) for k in ["mean","ci95_low","ci95_high"]]
        ax.errorbar(i,y,yerr=[[y-lo],[hi-y]],marker="o",ms=4,color=COLORS[0],capsize=3,lw=1.25)
    ax.set(xlim=(-.5,2.5),ylim=(0,.15),xticks=[0,1,2],xticklabels=["Full state","Routing","Value"],yticks=[0,.05,.10,.15])
    ax=axs[1,1]
    panel(ax,"D. Qwen: downstream OV write","Count-axis response","Readout layer")
    for metric,color,ls,label in [("write_natural_residual_slope",COLORS[0],"-","Natural"),
                                  ("write_orthogonal_residual_slope",GRAY,"--","Orthogonal")]:
        curve(ax,match(rw,metric=metric,stratum="all"),color,ls,label)
    ax.set(xlim=(27.7,35.3),ylim=(0,.068),xticks=[28,30,32,35])
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.793,.012),ncol=2,frameon=False,
               fontsize=9,columnspacing=.7,handlelength=1.5,handletextpad=.4)
    save(fig,"nonthinking_retrieval_interventions")


def patching():
    restore=read_csv(OLD / "causal/restore_control_plot.csv")
    import importlib.util
    analysis = Path(__file__).resolve().parents[1] / 'answer_state_source_count_20260921/analyze.py'
    spec = importlib.util.spec_from_file_location('answer_source_count', analysis)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    curves, _, audit = module.compute(REPO / 'exports')
    names = {'donor_transport': 'Different count', 'same_count_seed': 'Same count', 'self_patch': 'Self patch'}
    answer = [dict(r, layer=r['layer']-1, arm=names[r['condition']]) for r in curves]
    for relative, digest in audit['input_sha256'].items():
        INPUTS[str(REPO / 'exports' / relative)] = digest
    assert len(restore)==len(answer)==234
    fig,axs=four_panels(4.65)
    for i,(model,name,color,n) in enumerate(zip(MODELS,NAMES,COLORS,[36,42])):
        ax=axs[0,i]
        panel(ax,f"{'AB'[i]}. {name}: restoration sites","Restoration effect","Intervention layer",True)
        for arm,ls,c in [("Full span","-",color),("Endpoint","--",color),("Ordinary",":",GRAY)]:
            curve(ax,match(restore,model=model,arm=arm),c,ls,arm)
        ax.set(xlim=(-.5,n-.5),xticks=list(range(0,n,10)),ylim=(-.055,.7),yticks=[0,.2,.4,.6])
        legend(ax,loc="upper right")
        ax=axs[1,i]
        panel(ax,f"{'CD'[i]}. {name}: answer state controls","Source-count match rate","Intervention layer",True)
        for arm,ls,c in [("Different count","-",color),("Self patch",":",GRAY),("Same count","--",color)]:
            curve(ax,match(answer,model=model,arm=arm),c,ls,arm)
        ax.set(xlim=(-.5,n-.5),xticks=list(range(0,n,10)),ylim=(-.04,1.08),yticks=[0,.5,1])
        legend(ax,loc="upper left")
    save(fig,"nonthinking_patching_controls")


def answer_mediation():
    correct=match(read_csv(REPORT / "v4_4_causal_v2/correct_patching_aggregate.csv"),family="answer_patching")
    removal=match(read_csv(REPORT / "v4_4_extension/layerwise_subspace/answer_query_removal/layerwise_answer_query_removal_statistics.csv"),population="all",endpoint="absolute_error_specificity")
    serial=read_csv(OLD / "causal/serial_plot.csv")
    assert len(correct)==12 and len(removal)==23 and len(serial)==10
    fig,axs=four_panels(4.65)
    fig.legend(model_handles(),NAMES,loc="upper center",bbox_to_anchor=(.54,1.003),ncol=2,frameon=False)
    ax=axs[0,0]
    panel(ax,"A. Transfer between correct inputs","Source-count match",r"Source count $-$ target count",True)
    for i,(model,color) in enumerate(zip(MODELS,COLORS)):
        rr=sorted(match(correct,model_label=model),key=lambda r:int(r["k"])*(1 if r["target_direction"]=="increase" else -1))
        y,lo,hi=[np.array([float(r[k]) for r in rr]) for k in ["average_patching_acc","ci95_low","ci95_high"]]
        ax.errorbar(np.arange(6)+(-.1 if i==0 else .1),y,yerr=[y-lo,hi-y],color=color,fmt="o-",ms=3,lw=1.1,capsize=2)
    ax.set(xticks=range(6),xticklabels=["$-5$","$-3$","$-1$","+1","+3","+5"],ylim=(.80,1.03),yticks=[.8,.9,1.])
    ax=axs[0,1]
    panel(ax,"B. Answer-subspace removal","Excess error (counts)","Intervention layer")
    for model,color in zip(MODELS,COLORS):
        rr=[dict(layer=int(r["layer"]),mean=float(r["mean_effect"]),ci95_low=float(r["bootstrap_95ci_low"]),ci95_high=float(r["bootstrap_95ci_high"]))
            for r in removal if r["model_label"]==model]
        curve(ax,rr,color)
        ax.plot([r["layer"] for r in rr],[r["mean"] for r in rr],"o",ms=2.5,color=color)
    ax.set(xlim=(-.5,41.5),xticks=[0,10,20,30,40],ylim=(-.2,1.62),yticks=[0,.5,1,1.5])
    for ax,metrics,labels,title in [
            (axs[1,0],["source_repair","retrieval_mediation","late_mediation"],["Span repair","Retrieval","Answer"],"C. Sequential interventions"),
            (axs[1,1],["joint_interaction","remaining_repair"],["Interaction","Remaining repair"],"D. Interaction and residual effect")]:
        panel(ax,title,"Effect (counts)")
        for i,(model,color) in enumerate(zip(MODELS,COLORS)):
            rr=[next(r for r in serial if r["model"]==model and r["metric"]==metric) for metric in metrics]
            y,lo,hi=[np.array([float(r[k]) for r in rr]) for k in ["mean","ci95_low","ci95_high"]]
            ax.errorbar(np.arange(len(metrics))+(-.10 if i==0 else .10),y,yerr=[y-lo,hi-y],color=color,fmt="o",ms=3.5,lw=1.2,capsize=2.5)
        ax.set(xticks=range(len(metrics)),xticklabels=labels,xlim=(-.55,len(metrics)-.45))
        ax.axhline(0,color=GRAY,lw=.7)
        ax.set(ylim=(-.12,3.4),yticks=[0,1,2,3]) if len(metrics)==3 else ax.set(ylim=(-.72,1.98),yticks=[-.5,0,.5,1,1.5])
    save(fig,"nonthinking_answer_and_mediation")


def main():
    plt.rcParams.update({"font.family":"Times New Roman","mathtext.fontset":"stix","font.size":10,
        "axes.titlesize":11,"axes.titleweight":"normal","axes.labelsize":10,"xtick.labelsize":9,
        "ytick.labelsize":9,"legend.fontsize":9,"legend.frameon":False,"axes.linewidth":.7,
        "lines.linewidth":1.5,"lines.markersize":4,"pdf.fonttype":42,"ps.fonttype":42,
        "svg.fonttype":"none","text.color":INK,"axes.labelcolor":INK,"axes.titlecolor":INK,
        "xtick.color":INK,"ytick.color":INK,"figure.facecolor":"white","savefig.facecolor":"white"})
    for build in [full_head_scores,representation,steering,cue_domain,retrieval,patching,answer_mediation]:
        build()
    result={"style_reference":["figures/synthetic_appendix_evidence_revision_20260908/build_figures.py",
                "figures/synthetic_appendix_label_revision_20260908/build_retrieval_head_scores.py",
                "figures/additional_tasks_aurora/build_figures.py"],
            "font_points":{"title":11,"axis":10,"tick":9,"legend":9,"dense_heatmap_rank":7.5},
            "palette":dict(zip(MODELS,COLORS)),"input_sha256":INPUTS,"figures":AUDITS,
            "indexing":{"source":"zero-based","display":"one-based layers and heads","ranks":"unchanged"}}
    (OUT/"manifest.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(AUDITS,indent=2))


if __name__=="__main__":
    main()
