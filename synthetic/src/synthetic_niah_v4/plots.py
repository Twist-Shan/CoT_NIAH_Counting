from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _placeholder(path: Path, title: str) -> None:
    plt.figure(figsize=(6, 3.5))
    plt.text(0.5, 0.5, "No data", ha="center", va="center")
    plt.axis("off")
    plt.title(title)
    _save(path)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()


def _has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return not df.empty and all(col in df.columns for col in columns)


def _clean_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _has_finite(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(values).any())


def _select_probe_accuracy_rows(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    logistic = raw[raw["probe_type"].eq("multiclass_logistic")].copy()
    if _has_columns(logistic, ["accuracy"]) and _has_finite(logistic["accuracy"]):
        logistic["display_probe_type"] = "logistic"
        return logistic
    ridge = raw[raw["probe_type"].eq("ridge_scalar")].copy()
    if _has_columns(ridge, ["accuracy"]) and _has_finite(ridge["accuracy"]):
        ridge["display_probe_type"] = "ridge_rounded"
        return ridge
    return pd.DataFrame()


def _plot_bar_or_placeholder(
    df: pd.DataFrame,
    path: Path,
    title: str,
    x: str,
    y: str,
    hue: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    needed = [x, y] + ([hue] if hue else [])
    if not _has_columns(df, needed):
        _placeholder(path, title)
        return
    clean = _clean_numeric(df, [y]).dropna(subset=[x, y])
    if clean.empty or not _has_finite(clean[y]):
        _placeholder(path, title)
        return
    plt.figure(figsize=(10, 5))
    sns.barplot(data=clean, x=x, y=y, hue=hue, errorbar=None)
    plt.xticks(rotation=35, ha="right")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title(title)
    _save(path)


def _plot_heatmap_or_placeholder(
    mat: pd.DataFrame,
    path: Path,
    title: str,
    fmt: str = ".2f",
    center: float | None = None,
) -> None:
    if mat.empty:
        _placeholder(path, title)
        return
    mat = mat.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(mat.to_numpy(dtype=float)).any():
        _placeholder(path, title)
        return
    plt.figure(figsize=(8, 5))
    sns.heatmap(mat, annot=True, fmt=fmt, cmap="vlag", center=center)
    plt.title(title)
    _save(path)


def make_probe_plots(run_dir: Path) -> None:
    figures = run_dir / "figures"
    probe = _read(run_dir / "tables" / "probe_results.csv")
    if probe.empty:
        _placeholder(figures / "probe_acc_by_layer_anchor.png", "Probe accuracy by layer/anchor")
        _placeholder(figures / "probe_r2_by_layer_anchor.png", "Probe R2 by layer/anchor")
        _placeholder(figures / "probe_minus_baseline_heatmap.png", "Probe minus baseline")
        return
    raw = probe[probe["raw_or_residualized"].eq("raw")]
    acc = _select_probe_accuracy_rows(raw)
    _plot_bar_or_placeholder(
        acc,
        figures / "probe_acc_by_layer_anchor.png",
        "v4 count-probe accuracy by layer and anchor",
        x="anchor_name",
        y="accuracy",
        hue="layer",
        ylim=(0, 1),
    )

    ridge = raw[raw["probe_type"].eq("ridge_scalar")]
    if ridge.empty:
        ridge = raw
    _plot_bar_or_placeholder(
        ridge,
        figures / "probe_r2_by_layer_anchor.png",
        "v4 ridge R2 by layer and anchor",
        x="anchor_name",
        y="r2",
        hue="layer",
    )

    if _has_columns(acc, ["anchor_name", "layer", "accuracy", "position_baseline_acc"]):
        acc = _clean_numeric(acc, ["accuracy", "position_baseline_acc"])
        acc["minus_position"] = acc["accuracy"] - acc["position_baseline_acc"]
        mat = acc.pivot_table(index="anchor_name", columns="layer", values="minus_position", aggfunc="mean")
    else:
        mat = pd.DataFrame()
    _plot_heatmap_or_placeholder(
        mat,
        figures / "probe_minus_baseline_heatmap.png",
        "Probe accuracy minus position baseline",
        center=0.0,
    )


def make_direction_plots(run_dir: Path) -> None:
    figures = run_dir / "figures"
    directions = _read(run_dir / "tables" / "direction_metrics.csv")
    if directions.empty:
        _placeholder(figures / "direction_cosine_heatmap.png", "Direction cosines")
        _placeholder(figures / "projection_by_count.png", "Projection by count")
        _placeholder(figures / "input_geometry_projection_trajectories.png", "Input geometry")
        return
    if _has_columns(directions, ["direction_type", "anchor_name", "cosine_with_unembedding"]):
        mat = directions.pivot_table(index="direction_type", columns="anchor_name", values="cosine_with_unembedding", aggfunc="mean")
    else:
        mat = pd.DataFrame()
    _plot_heatmap_or_placeholder(
        mat,
        figures / "direction_cosine_heatmap.png",
        "Cosine with unembedding adjacent direction",
        center=0.0,
    )

    if _has_columns(directions, ["projection_r2", "anchor_name", "projection_slope", "direction_type"]):
        top = _clean_numeric(directions, ["projection_r2", "projection_slope"]).dropna(subset=["projection_r2"]).sort_values("projection_r2", ascending=False).head(24)
    else:
        top = pd.DataFrame()
    _plot_bar_or_placeholder(
        top,
        figures / "projection_by_count.png",
        "Projection slope by anchor/direction",
        x="anchor_name",
        y="projection_slope",
        hue="direction_type",
    )

    geom = _read(run_dir / "tables" / "input_geometry_results.csv")
    _plot_bar_or_placeholder(
        geom,
        figures / "input_geometry_projection_trajectories.png",
        "Input geometry projection summary",
        x="anchor_name",
        y="projection_slope",
        hue="direction_type",
    )


def make_steering_plots(run_dir: Path) -> None:
    figures = run_dir / "figures"
    steering = _read(run_dir / "tables" / "steering_results.csv")
    if steering.empty:
        _placeholder(figures / "steering_heatmap_anchor_layer.png", "Steering heatmap")
        _placeholder(figures / "steering_dose_response_top_configs.png", "Steering dose response")
        _placeholder(figures / "steering_controls.png", "Steering controls")
        return
    main = steering[steering["control_type"].eq("none")]
    if _has_columns(main, ["anchor_name", "layer", "mean_count_shift"]):
        mat = main.pivot_table(index="anchor_name", columns="layer", values="mean_count_shift", aggfunc="mean")
    else:
        mat = pd.DataFrame()
    _plot_heatmap_or_placeholder(
        mat,
        figures / "steering_heatmap_anchor_layer.png",
        "Mean count shift by anchor/layer",
        center=0.0,
    )

    if _has_columns(main, ["model_type", "anchor_name", "layer", "direction_type", "mean_count_shift", "alpha", "mean_pred_steered"]):
        main_clean = _clean_numeric(main, ["mean_count_shift", "alpha", "mean_pred_steered"])
        top_keys = (
            main_clean.groupby(["model_type", "anchor_name", "layer", "direction_type"], as_index=False)["mean_count_shift"]
            .apply(lambda s: s.abs().max())
            .rename(columns={"mean_count_shift": "abs_shift"})
            .dropna(subset=["abs_shift"])
            .sort_values("abs_shift", ascending=False)
            .head(6)
        )
        merged = main_clean.merge(top_keys[["model_type", "anchor_name", "layer", "direction_type"]]) if not top_keys.empty else pd.DataFrame()
    else:
        merged = pd.DataFrame()
    if merged.empty or not _has_finite(merged["mean_pred_steered"]):
        _placeholder(figures / "steering_dose_response_top_configs.png", "Top steering dose response")
    else:
        plt.figure(figsize=(9, 5))
        sns.lineplot(data=merged, x="alpha", y="mean_pred_steered", hue="anchor_name", style="direction_type", marker="o", errorbar=None)
        plt.title("Top steering dose response")
        _save(figures / "steering_dose_response_top_configs.png")

    _plot_bar_or_placeholder(
        steering,
        figures / "steering_controls.png",
        "Steering controls",
        x="control_type",
        y="mean_count_shift",
        hue="direction_type",
    )


def make_patching_plots(run_dir: Path) -> None:
    figures = run_dir / "figures"
    patch = _read(run_dir / "tables" / "interchange_patching_results.csv")
    if patch.empty:
        _placeholder(figures / "interchange_patch_matrix.png", "Interchange patch matrix")
        return
    if _has_columns(patch, ["receiver_count", "donor_count", "causal_effect_size"]):
        mat = patch.pivot_table(index="receiver_count", columns="donor_count", values="causal_effect_size", aggfunc="mean")
    else:
        mat = pd.DataFrame()
    _plot_heatmap_or_placeholder(
        mat,
        figures / "interchange_patch_matrix.png",
        "Patched pred shift by receiver/donor count",
        fmt=".1f",
        center=0.0,
    )


def make_all_plots(run_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    make_probe_plots(run_dir)
    make_direction_plots(run_dir)
    make_steering_plots(run_dir)
    make_patching_plots(run_dir)
