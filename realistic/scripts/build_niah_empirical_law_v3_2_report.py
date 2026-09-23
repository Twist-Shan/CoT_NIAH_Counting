"""Shared empirical-law calculations and plot helpers used by paper figure builders."""

from __future__ import annotations

from pathlib import Path

import matplotlib

import matplotlib.colors as mcolors

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

AURORA = {
    "indigo": "#23165C",
    "violet": "#6750E8",
    "cyan": "#00C2FF",
    "yellow": "#F6E36A",
    "teal": "#00D4B4",
    "green": "#39E58C",
    "magenta": "#C04DFF",
    "pink": "#FF5FA2",
    "black": "#161923",
    "white": "#F8FBFF",
    "gray": "#8190A5",
    "brown": "#765347",
}

MODE_SHORT = {
    "direct": "Non-thinking",
    "native_thinking": "Native-thinking",
    "enumeration_index": "Index",
    "enumeration_bullet": "Bullet",
}

MAE_FAMILY = "trimmed_conditional_mae_10"

def length_colors(levels: list[int]) -> list[str]:
    """Sample the scaling-law Plasma ramp for every registered L level."""
    colors = matplotlib.colormaps["plasma_r"](
        np.linspace(0.0, 1.0, len(levels))
    )
    return [mcolors.to_hex(color).upper() for color in colors]

def apply_all_n_ticks(ax: plt.Axes, n_levels: list[int], *, fontsize: float) -> None:
    """Show every registered N value on the log2 axis."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(n_levels)
    ax.set_xticklabels(
        [str(value) for value in n_levels],
        rotation=52,
        ha="right",
        rotation_mode="anchor",
        fontsize=fontsize,
    )
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

def apply_all_l_ticks(ax: plt.Axes, l_levels: list[int], *, fontsize: float) -> None:
    """Show all registered passage lengths on a log-positioned L axis."""
    ax.set_xscale("log")
    ax.set_xticks(l_levels)
    ax.set_xticklabels(
        [f"{value // 1000}k" for value in l_levels],
        rotation=36,
        ha="right",
        rotation_mode="anchor",
        fontsize=fontsize,
    )
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

SLOT_ORDER = [
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
    "Gemma4-E4B",
    "Gemma4-12B",
    "Gemma4-26B-A4B",
    "Gemma4-31B",
    "Nemotron-3-Nano-4B",
    "Nemotron-Nano-v2-9B",
    "GLM-4/Z1-9B",
    "Ministral-3-8B pair",
]

MATCHED_PAIR_SLOTS = {"GLM-4/Z1-9B", "Ministral-3-8B pair"}

def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": AURORA["white"],
            "axes.facecolor": AURORA["white"],
            "axes.edgecolor": "#D5DCE6",
            "axes.labelcolor": AURORA["black"],
            "xtick.color": "#536176",
            "ytick.color": "#536176",
            "text.color": AURORA["black"],
            "grid.color": "#DCE3ED",
            "grid.linewidth": 0.65,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
        }
    )

def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor=AURORA["white"])
    plt.close(fig)

def design_values(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    n = frame["N"].to_numpy(float)
    lk = frame["L"].to_numpy(float) / 1000.0
    logn = np.log(n)
    logl = np.log(lk)
    invn = 1.0 / n
    return {
        "N": n,
        "L_k": lk,
        "logN": logn,
        "logL": logl,
        "N_x_L_k": n * lk,
        "logN_x_logL": logn * logl,
        "N_x_logL": n * logl,
        "logN_x_L_k": logn * lk,
        "invN": invn,
        "invN_x_L_k": invn * lk,
        "invN_x_logL": invn * logl,
    }

def predictions_by_slot(
    coefficients: pd.DataFrame,
    mode: str,
    family: str,
    n_levels: list[int],
    l_levels: list[int],
) -> np.ndarray:
    """Return selected-law predictions as slot × N × L.

    The V3.2 law shares an item structure across comparison slots but never a
    pooled coefficient vector.  Keeping the slot dimension here lets the plot
    show model heterogeneity rather than inventing an "average model".
    """
    grid = pd.DataFrame(
        [(n, length) for n in n_levels for length in l_levels],
        columns=["N", "L"],
    )
    values = design_values(grid)
    predictions = []
    for slot in SLOT_ORDER:
        block = coefficients[
            coefficients["comparison_slot"].eq(slot)
            & coefficients["prompt_mode"].eq(mode)
            & coefficients["outcome_family"].eq(family)
        ]
        if block.empty:
            continue
        eta = np.zeros(len(grid), dtype=float)
        for row in block.itertuples(index=False):
            if row.term == "intercept":
                eta += float(row.estimate)
            else:
                eta += float(row.estimate) * values[row.term]
        if family.startswith("accuracy_"):
            eta = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        elif family == MAE_FAMILY:
            # V3.2 reports the requested identity-scale OLS fit.  Do not clip
            # negative fitted values: they are scientifically useful diagnostics
            # of where a local linear approximation violates MAE's support.
            eta = eta
        predictions.append(eta)
    if not predictions:
        raise ValueError(f"No selected coefficients for {mode} / {family}")
    return np.vstack(predictions).reshape(-1, len(n_levels), len(l_levels))

def plot_model_law_panels(
    cells: pd.DataFrame,
    coefficients: pd.DataFrame,
    *,
    family: str,
    column: str,
    modes: tuple[str, str],
    x_axis: str,
    path: Path,
    title: str,
    ylabel: str,
) -> None:
    """Render 12 model-resolved panels with every registered N and L.

    One axis is horizontal and every level of the other axis becomes a curve.
    Both prompt modes are overlaid, so no median model is invented.  MAE uses a
    symmetric-log display only to retain negative identity-link predictions;
    the regression itself remains in raw count units.
    """
    if x_axis not in {"N", "L"}:
        raise ValueError(f"Unsupported x_axis={x_axis}")
    n_levels = sorted(int(x) for x in cells["N"].unique())
    l_levels = sorted(int(x) for x in cells["L"].unique())
    curve_levels = l_levels if x_axis == "N" else n_levels
    curve_colors = length_colors(curve_levels)
    x_levels = n_levels if x_axis == "N" else l_levels
    fitted_by_mode = {
        mode: predictions_by_slot(coefficients, mode, family, n_levels, l_levels)
        for mode in modes
    }
    for mode, values in fitted_by_mode.items():
        if values.shape[0] != len(SLOT_ORDER):
            raise ValueError(
                f"Expected {len(SLOT_ORDER)} slots for {mode}/{family}, got {values.shape[0]}"
            )

    fig, axes = plt.subplots(3, 4, figsize=(16.4, 11.1), sharex=True, sharey=True)
    for slot_index, slot in enumerate(SLOT_ORDER):
        row_index, column_index = divmod(slot_index, 4)
        ax = axes[row_index, column_index]
        if family in {"trimmed_signed_bias_10", MAE_FAMILY}:
            ax.axhline(0, color=AURORA["gray"], linewidth=0.7, alpha=0.58, zorder=0)
        for mode_index, mode in enumerate(modes):
            line_style = "-" if mode_index == 0 else "--"
            marker = "o" if mode_index == 0 else "^"
            for curve_index, (curve_level, color) in enumerate(zip(curve_levels, curve_colors)):
                if x_axis == "N":
                    observed = (
                        cells[
                            cells["comparison_slot"].eq(slot)
                            & cells["prompt_mode"].eq(mode)
                            & cells["L"].eq(curve_level)
                        ]
                        .set_index("N")
                        .reindex(n_levels)[column]
                        .to_numpy(float)
                    )
                    fitted = fitted_by_mode[mode][slot_index, :, curve_index]
                else:
                    observed = (
                        cells[
                            cells["comparison_slot"].eq(slot)
                            & cells["prompt_mode"].eq(mode)
                            & cells["N"].eq(curve_level)
                        ]
                        .set_index("L")
                        .reindex(l_levels)[column]
                        .to_numpy(float)
                    )
                    fitted = fitted_by_mode[mode][slot_index, curve_index, :]
                ax.plot(
                    x_levels,
                    fitted,
                    color=color,
                    linewidth=1.25,
                    linestyle=line_style,
                    alpha=0.92,
                )
                ax.scatter(
                    x_levels,
                    observed,
                    s=11 if mode_index == 0 else 13,
                    marker=marker,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.62,
                    alpha=0.88,
                    zorder=3,
                )
        if x_axis == "N":
            apply_all_n_ticks(ax, n_levels, fontsize=5.7)
            ax.set_xlim(0.9, 22)
        else:
            apply_all_l_ticks(ax, l_levels, fontsize=6.0)
        if family.startswith("accuracy_"):
            ax.set_ylim(-0.02, 1.02)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        elif family == MAE_FAMILY:
            ax.set_yscale("symlog", linthresh=0.25, linscale=0.75)
        ax.grid(color=AURORA["gray"], alpha=0.20, linewidth=0.52)
        ax.spines[["top", "right"]].set_visible(False)
        dagger = "†" if slot in MATCHED_PAIR_SLOTS else ""
        ax.set_title(f"{slot}{dagger}", loc="left", fontsize=9.0, fontweight="bold", pad=5)
        if row_index == 2:
            ax.set_xlabel(
                "Target count N (log2 positions)" if x_axis == "N" else "Passage length L (tokens; log positions)",
                fontsize=8.0,
            )
        if column_index == 0:
            ax.set_ylabel(ylabel, fontsize=8.2)
        ax.tick_params(labelsize=6.8)

    mode_handles = [
        plt.Line2D([], [], color=AURORA["indigo"], linestyle="-", marker="o", markerfacecolor="white", linewidth=1.6, markersize=4, label=MODE_SHORT[modes[0]]),
        plt.Line2D([], [], color=AURORA["indigo"], linestyle="--", marker="^", markerfacecolor="white", linewidth=1.6, markersize=4.2, label=MODE_SHORT[modes[1]]),
    ]
    curve_handles = [
        plt.Line2D(
            [], [], color=color, linewidth=1.8,
            label=(f"L={level // 1000}k" if x_axis == "N" else f"N={level}"),
        )
        for level, color in zip(curve_levels, curve_colors)
    ]
    legend_columns = 5 if x_axis == "N" else 8
    fig.legend(
        handles=mode_handles + curve_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=legend_columns,
        frameon=False,
        fontsize=6.7,
        handlelength=2.1,
        columnspacing=0.9,
    )
    fig.suptitle(title, x=0.055, y=0.997, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.895), h_pad=1.25, w_pad=0.95)
    savefig(fig, path)
