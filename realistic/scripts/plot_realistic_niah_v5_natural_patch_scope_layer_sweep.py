#!/usr/bin/env python3
"""Plot raw and norm-adjusted natural patch-scope layer effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "item_end_w1": "item-end · 1 token",
    "event_tail_w4": "event tail · 4 tokens",
    "item_span": "endpoint-aligned item span",
}
COLORS = {
    "item_end_w1": "#a95d2b",
    "event_tail_w4": "#1f6f5f",
    "item_span": "#272b28",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    scopes = {row["scope"]: row for row in payload["scopes"]}
    expected = set(LABELS)
    if set(scopes) != expected:
        raise ValueError(f"Unexpected patch scopes: {sorted(scopes)}")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 1, figsize=(10.4, 7.2), sharex=True)
    figure.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.10, hspace=0.28)

    for scope in ("item_end_w1", "event_tail_w4", "item_span"):
        row = scopes[scope]
        layers = [int(item["layer"]) for item in row["layer_summaries"]]
        raw = [
            float(item["median_seed_mean_effect"])
            for item in row["layer_summaries"]
        ]
        adjusted = [
            float(item["median_logodds_shift_per_patch_norm"])
            for item in row["layer_summaries"]
        ]
        selected = int(row["selected_layer"])
        color = COLORS[scope]
        axes[0].plot(layers, raw, color=color, linewidth=2.0, label=LABELS[scope])
        axes[1].plot(
            layers,
            adjusted,
            color=color,
            linewidth=2.0,
            label=LABELS[scope],
        )
        selected_index = layers.index(selected)
        for axis, values in zip(axes, (raw, adjusted)):
            axis.scatter(
                [selected],
                [values[selected_index]],
                s=42,
                facecolor=color,
                edgecolor="white",
                linewidth=1.2,
                zorder=5,
            )
            axis.annotate(
                f"L{selected}",
                (selected, values[selected_index]),
                xytext=(4, 6),
                textcoords="offset points",
                color=color,
                fontsize=8,
            )

    axes[0].set_title("A · donor-directed complete-transition effect")
    axes[0].set_ylabel("median seed-mean Δ log-odds")
    axes[1].set_title("B · effect relative to realized patch magnitude")
    axes[1].set_ylabel("median Δ log-odds / patch L2 norm")
    axes[1].set_xlabel("decoder block output patched")
    axes[1].set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 32, 35])
    for axis in axes:
        axis.axhline(0.0, color="#9ba19c", linewidth=0.9)
        axis.grid(axis="y", color="#dedfd8", linewidth=0.7, alpha=0.75)
        axis.set_xlim(0, 35)
    axes[0].legend(loc="upper right", frameon=False, ncol=1)
    figure.suptitle(
        "N=10 no-index progress-state transplant · discovery20 · k=6 bidirectional",
        fontsize=12,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.025,
        "Markers: earliest L0–L34 layer within 95% of the robust discovery peak, with both directions positive.",
        ha="center",
        fontsize=8.5,
        color="#5c645e",
    )

    args.png.parent.mkdir(parents=True, exist_ok=True)
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.png, dpi=190, facecolor="white")
    figure.savefig(args.svg, facecolor="white")
    plt.close(figure)
    print(json.dumps({"png": str(args.png), "svg": str(args.svg)}, indent=2))


if __name__ == "__main__":
    main()
