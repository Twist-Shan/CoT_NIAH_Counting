#!/usr/bin/env python3
"""Build a compact one-row Qwen3 accuracy-by-target-count figure.

This is the target-count-axis companion to build_qwen_accuracy_l_compact.py.
It does not refit any model: observed points and selected-law curves are read
from the same frozen V3.2 tables used by the full report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from build_niah_empirical_law_v3_2_report import (
    AURORA,
    MODE_SHORT,
    SLOT_ORDER,
    apply_all_n_ticks,
    length_colors,
    predictions_by_slot,
    set_plot_style,
)


FAMILY = "accuracy_bernoulli_logit"
MODES = ("direct", "native_thinking")
QWEN_SLOTS = tuple(SLOT_ORDER[:4])


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    tables = (
        root
        / "outputs"
        / "anvil_realistic_niah_v3_1_20260819_formal"
        / "analysis"
        / "v3_2_inverse_n_candidate_extension"
        / "tables"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, default=tables / "cell_outcomes.csv.gz")
    parser.add_argument(
        "--coefficients",
        type=Path,
        default=tables / "selected_model_coefficients.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "NiaH_accuracy_Qwen3_target_count_compact.png",
    )
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=root / "reports" / "NiaH_accuracy_Qwen3_target_count_compact.pdf",
    )
    return parser.parse_args()


def build_figure(
    cells: pd.DataFrame,
    coefficients: pd.DataFrame,
    output: Path,
    pdf_output: Path,
) -> None:
    n_levels = sorted(int(value) for value in cells["N"].unique())
    l_levels = sorted(int(value) for value in cells["L"].unique())
    colors = length_colors(l_levels)
    fitted = {
        mode: predictions_by_slot(coefficients, mode, FAMILY, n_levels, l_levels)
        for mode in MODES
    }
    for mode in MODES:
        if fitted[mode].shape != (len(SLOT_ORDER), len(n_levels), len(l_levels)):
            raise ValueError(f"Unexpected prediction grid for {mode}: {fitted[mode].shape}")

    set_plot_style()
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.78), sharex=True, sharey=True)
    fig.patch.set_facecolor("#FFFFFF")

    for slot_index, (slot, ax) in enumerate(zip(QWEN_SLOTS, axes)):
        ax.set_facecolor("#FFFFFF")
        for mode_index, mode in enumerate(MODES):
            line_style = "-" if mode_index == 0 else "--"
            marker = "o" if mode_index == 0 else "^"
            for length_index, (length, color) in enumerate(zip(l_levels, colors)):
                observed = (
                    cells[
                        cells["comparison_slot"].eq(slot)
                        & cells["prompt_mode"].eq(mode)
                        & cells["L"].eq(length)
                    ]
                    .set_index("N")
                    .reindex(n_levels)["parsed_exact_accuracy"]
                    .to_numpy(float)
                )
                predicted = fitted[mode][slot_index, :, length_index]
                ax.plot(
                    n_levels,
                    predicted,
                    color=color,
                    linewidth=1.25,
                    linestyle=line_style,
                    alpha=0.94,
                    zorder=2,
                )
                ax.scatter(
                    n_levels,
                    observed,
                    s=12.5 if mode_index == 0 else 14.5,
                    marker=marker,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.7,
                    alpha=0.9,
                    zorder=3,
                )

        apply_all_n_ticks(ax, n_levels, fontsize=7.3)
        ax.set_xlim(0.9, 21.1)
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(color=AURORA["gray"], alpha=0.22, linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(slot, loc="left", fontsize=10.2, fontweight="bold", pad=6)
        ax.set_xlabel("Target count N (log2 positions)", fontsize=8.2, labelpad=7)
        ax.tick_params(axis="y", labelsize=7.7)

    axes[0].set_ylabel("Parsed exact accuracy", fontsize=8.8, labelpad=6)

    mode_handles = [
        plt.Line2D(
            [], [], color=AURORA["indigo"], linestyle="-", marker="o",
            markerfacecolor="white", linewidth=1.7, markersize=4.2,
            label=MODE_SHORT[MODES[0]],
        ),
        plt.Line2D(
            [], [], color=AURORA["indigo"], linestyle="--", marker="^",
            markerfacecolor="white", linewidth=1.7, markersize=4.5,
            label=MODE_SHORT[MODES[1]],
        ),
    ]
    length_handles = [
        plt.Line2D([], [], color=color, linewidth=2.0, label=f"L={length // 1000}k")
        for length, color in zip(l_levels, colors)
    ]

    mode_legend = fig.legend(
        handles=mode_handles,
        title="Mode",
        loc="upper right",
        bbox_to_anchor=(0.972, 0.82),
        frameon=False,
        fontsize=7.8,
        title_fontsize=8.3,
        handlelength=2.6,
        borderaxespad=0,
    )
    fig.add_artist(mode_legend)
    fig.legend(
        handles=length_handles,
        title="Passage length",
        loc="center right",
        bbox_to_anchor=(0.976, 0.39),
        ncol=2,
        frameon=False,
        fontsize=7.4,
        title_fontsize=8.3,
        handlelength=2.25,
        columnspacing=0.95,
        labelspacing=0.46,
        borderaxespad=0,
    )

    fig.suptitle(
        "Exact-count accuracy by target count · Qwen3 family",
        x=0.052,
        y=0.975,
        ha="left",
        fontsize=13.2,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.052, right=0.86, top=0.85, bottom=0.225, wspace=0.13)

    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {
        "bbox_inches": "tight",
        "pad_inches": 0.01,
        "facecolor": "#FFFFFF",
    }
    fig.savefig(output, dpi=240, **save_kwargs)
    fig.savefig(pdf_output, **save_kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.cells.exists() or not args.coefficients.exists():
        raise FileNotFoundError((args.cells, args.coefficients))
    cells = pd.read_csv(args.cells)
    coefficients = pd.read_csv(args.coefficients)
    build_figure(cells, coefficients, args.output, args.pdf_output)
    print(args.output.resolve())
    print(args.pdf_output.resolve())


if __name__ == "__main__":
    main()
