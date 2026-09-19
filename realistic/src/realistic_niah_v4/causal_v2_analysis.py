from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

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
    "white": "#F6F1E8",
    "gray": "#8190A5",
    "brown": "#765347",
}


def _stable_seed(label: str) -> int:
    digest = hashlib.sha256(str(label).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_647


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_csv_gzip_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def paired_control_adjusted_transport(
    detail: pd.DataFrame,
    *,
    family: str,
) -> pd.DataFrame:
    """Pair each causal treatment with its within-example control mean.

    Patching treatments are ``donor_transport`` and their controls are the
    available self/same-count transports.  Steering treatments are
    ``geometric`` and their controls are norm-matched orthogonal directions.
    All invalid generations already contribute zero through
    ``strict_normalized_transport``; invalid rate is retained separately.
    """

    if family in {"prompt_patching", "answer_patching"}:
        condition_columns = [
            "model_label",
            "site",
            "patch_protocol",
            "start_layer",
            "k",
        ]
        target_column = "donor_count"
        treatment_name = "donor_transport"
    elif family == "steering":
        condition_columns = ["model_label", "steering_protocol", "layer_set", "k"]
        target_column = "target_count"
        treatment_name = "geometric"
    else:
        raise KeyError(f"Unknown causal-v2 family: {family}")
    required = {
        *condition_columns,
        "condition",
        "seed",
        "receiver_count",
        target_column,
        "target_direction",
        "strict_normalized_transport",
        "patched_format_valid",
        "transport_numeric_valid",
    }
    missing = sorted(required - set(detail.columns))
    if missing:
        raise ValueError(f"{family} detail is missing columns: {missing}")
    work = detail.copy()
    if "status" in work.columns:
        failed = work[work["status"].astype(str).ne("ok")]
        if not failed.empty:
            raise ValueError(f"{family} detail contains skipped interventions")
        work = work[work["status"].astype(str).eq("ok")].copy()
    identity = [
        *condition_columns,
        "seed",
        "receiver_count",
        target_column,
        "target_direction",
    ]
    treatment = work[work["condition"].astype(str).eq(treatment_name)].copy()
    controls = work[~work["condition"].astype(str).eq(treatment_name)].copy()
    if treatment.empty or controls.empty:
        raise ValueError(f"{family} needs treatment and control rows")
    control = controls.groupby(identity, as_index=False, dropna=False).agg(
        control_transport=("strict_normalized_transport", "mean"),
        control_valid_rate=("transport_numeric_valid", "mean"),
        control_patched_valid_rate=("patched_format_valid", "mean"),
        control_conditions=("condition", "nunique"),
    )
    paired = treatment.merge(control, on=identity, how="inner", validate="many_to_one")
    if len(paired) != len(treatment):
        raise ValueError(f"{family} has incomplete paired controls")
    paired["control_adjusted_transport"] = pd.to_numeric(
        paired["strict_normalized_transport"], errors="raise"
    ) - pd.to_numeric(paired["control_transport"], errors="raise")
    paired["treatment_transport_valid"] = paired["transport_numeric_valid"].astype(bool)
    paired["treatment_patched_valid"] = paired["patched_format_valid"].astype(bool)
    return paired


def _cluster_bootstrap_mean(
    seed_values: np.ndarray,
    *,
    seed: int,
    repetitions: int,
) -> tuple[float, float, float]:
    values = np.asarray(seed_values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(repetitions), len(values)))
    distribution = values[indices].mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def summarize_layer_k_transport(
    paired: pd.DataFrame,
    *,
    family: str,
    bootstrap_repetitions: int = 10_000,
) -> pd.DataFrame:
    if family in {"prompt_patching", "answer_patching"}:
        groups = [
            "model_label",
            "site",
            "patch_protocol",
            "start_layer",
            "k",
            "target_direction",
        ]
    elif family == "steering":
        groups = [
            "model_label",
            "steering_protocol",
            "layer_set",
            "k",
            "target_direction",
        ]
    else:
        raise KeyError(f"Unknown causal-v2 family: {family}")
    rows: list[dict[str, Any]] = []
    for keys, frame in paired.groupby(groups, sort=True, dropna=False):
        seed_means = (
            frame.groupby("seed")["control_adjusted_transport"]
            .mean()
            .to_numpy(dtype=float)
        )
        mean, low, high = _cluster_bootstrap_mean(
            seed_means,
            seed=_stable_seed(
                f"layer-k:{family}:" + ":".join(str(value) for value in keys)
            ),
            repetitions=bootstrap_repetitions,
        )
        rows.append(
            {
                **dict(zip(groups, keys)),
                "examples": int(len(frame)),
                "seeds": int(frame["seed"].nunique()),
                "mean_control_adjusted_transport": mean,
                "ci95_low": low,
                "ci95_high": high,
                "treatment_transport_valid_rate": float(
                    frame["treatment_transport_valid"].mean()
                ),
                "treatment_patched_valid_rate": float(
                    frame["treatment_patched_valid"].mean()
                ),
                "control_valid_rate": float(frame["control_valid_rate"].mean()),
                "control_patched_valid_rate": float(
                    frame["control_patched_valid_rate"].mean()
                ),
                "positive_seed_fraction": float((seed_means > 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def summarize_ablation_sweep(detail: pd.DataFrame) -> pd.DataFrame:
    """Return ranked-minus-random answer-query ablation effects by top-k."""

    identifiers = ["stimulus_id", "seed", "head_bank", "top_n"]
    metrics = ["accuracy_delta", "absolute_error_delta", "prediction_changed"]
    ranked = detail[detail["condition"].astype(str).eq("ranked")].copy()
    random = detail[detail["condition"].astype(str).eq("layer_matched_random")]
    random_mean = random.groupby(identifiers, as_index=False).agg(
        **{f"{metric}_random": (metric, "mean") for metric in metrics},
        random_overlap_mean=("ranked_random_head_overlap", "mean"),
    )
    paired = ranked.merge(
        random_mean, on=identifiers, how="inner", validate="one_to_one"
    )
    if len(paired) != len(ranked):
        raise ValueError("Ablation sweep has incomplete random controls")
    for metric in metrics:
        paired[f"{metric}_ranked_minus_random"] = pd.to_numeric(
            paired[metric]
        ) - pd.to_numeric(paired[f"{metric}_random"])
    return (
        paired.groupby(["model_label", "head_bank", "top_n"], as_index=False)
        .agg(
            examples=("stimulus_id", "size"),
            seeds=("seed", "nunique"),
            accuracy_effect=("accuracy_delta_ranked_minus_random", "mean"),
            absolute_error_effect=("absolute_error_delta_ranked_minus_random", "mean"),
            prediction_change_effect=("prediction_changed_ranked_minus_random", "mean"),
            ranked_valid_rate=("patched_format_valid", "mean"),
            random_overlap_mean=("random_overlap_mean", "mean"),
        )
        .sort_values(["model_label", "head_bank", "top_n"])
        .reset_index(drop=True)
    )


def _plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": AURORA["white"],
            "axes.facecolor": AURORA["white"],
            "savefig.facecolor": AURORA["white"],
            "text.color": AURORA["black"],
            "axes.labelcolor": AURORA["black"],
            "axes.edgecolor": AURORA["gray"],
            "xtick.color": AURORA["black"],
            "ytick.color": AURORA["black"],
            "grid.color": "#D7D0C5",
            "font.size": 10,
        }
    )


def plot_ablation_sweep(summary: pd.DataFrame, path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _plot_style()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
    colors = {"broad_aggregation": AURORA["cyan"], "first_locator": AURORA["pink"]}
    labels = {
        "broad_aggregation": "broad aggregation ranked heads",
        "first_locator": "first-needle locator ranked heads",
    }
    for bank, frame in summary.groupby("head_bank"):
        frame = frame.sort_values("top_n")
        axes[0].plot(
            frame["top_n"],
            frame["accuracy_effect"],
            color=colors[str(bank)],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=labels[str(bank)],
        )
        axes[1].plot(
            frame["top_n"],
            frame["absolute_error_effect"],
            color=colors[str(bank)],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=labels[str(bank)],
        )
    axes[0].set_ylabel("ranked − random accuracy change")
    axes[1].set_ylabel("ranked − random absolute-error change")
    for axis in axes:
        axis.axhline(0, color=AURORA["indigo"], linewidth=1, linestyle="--")
        axis.set_xlabel("ablated top-k heads")
        axis.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
        axis.grid(alpha=0.55)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def _transport_matrix(
    frame: pd.DataFrame,
    *,
    layer_column: str,
) -> tuple[np.ndarray, list[int], list[int]]:
    layers = sorted(int(value) for value in frame[layer_column].unique())
    ks = sorted(int(value) for value in frame["k"].unique())
    pivot = frame.pivot_table(
        index="k",
        columns=layer_column,
        values="mean_control_adjusted_transport",
        aggfunc="mean",
    ).reindex(index=ks, columns=layers)
    return pivot.to_numpy(dtype=float), layers, ks


def plot_layer_k_heatmaps(
    summary: pd.DataFrame,
    *,
    family: str,
    output_dir: str | Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    _plot_style()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cmap = LinearSegmentedColormap.from_list(
        "aurora_diverging", [AURORA["pink"], AURORA["white"], AURORA["teal"]]
    )
    if family in {"prompt_patching", "answer_patching"}:
        facet_columns = ["site", "patch_protocol"]
        layer_column = "start_layer"
    elif family == "steering":
        singles = summary[summary["steering_protocol"].astype(str).eq("single_layer")]
        summary = singles.copy()
        summary["layer"] = summary["layer_set"].astype(int)
        facet_columns = ["steering_protocol"]
        layer_column = "layer"
    else:
        raise KeyError(f"Unknown causal-v2 family: {family}")
    paths: list[Path] = []
    for facet, facet_frame in summary.groupby(facet_columns, sort=True, dropna=False):
        facet_tuple = facet if isinstance(facet, tuple) else (facet,)
        maximum = float(
            np.nanmax(np.abs(facet_frame["mean_control_adjusted_transport"]))
        )
        maximum = max(maximum, 0.05) if np.isfinite(maximum) else 0.05
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
        image = None
        for axis, direction in zip(axes, ("increase", "decrease")):
            selected = facet_frame[
                facet_frame["target_direction"].astype(str).eq(direction)
            ]
            matrix, layers, ks = _transport_matrix(selected, layer_column=layer_column)
            image = axis.imshow(
                matrix,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                norm=TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
            )
            tick_step = max(1, len(layers) // 8)
            tick_positions = list(range(0, len(layers), tick_step))
            if tick_positions[-1] != len(layers) - 1:
                tick_positions.append(len(layers) - 1)
            axis.set_xticks(tick_positions)
            axis.set_xticklabels([layers[index] for index in tick_positions])
            axis.set_yticks(range(len(ks)))
            axis.set_yticklabels(ks)
            axis.set_xlabel("decoder layer")
            axis.set_title(f"{direction}: receiver → target")
        axes[0].set_ylabel("k = |target count − receiver count|")
        title = " · ".join(str(value) for value in facet_tuple)
        figure.suptitle(f"{family}: {title}")
        if image is not None:
            colorbar = figure.colorbar(image, ax=axes, shrink=0.88, pad=0.02)
            colorbar.set_label("treatment − matched-control normalized transport")
        figure.subplots_adjust(
            left=0.08, right=0.90, bottom=0.14, top=0.84, wspace=0.14
        )
        slug = "_".join(str(value).replace(" ", "_") for value in facet_tuple)
        path = output / f"{family}_{slug}_layer_k_heatmap.png"
        figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return paths


def write_causal_v2_analysis(
    *,
    output_dir: str | Path,
    ablation_detail: pd.DataFrame,
    prompt_screen_detail: pd.DataFrame,
    answer_screen_detail: pd.DataFrame,
    steering_screen_detail: pd.DataFrame,
    confirmation_tables: Iterable[tuple[str, pd.DataFrame]] = (),
    bootstrap_repetitions: int = 10_000,
) -> dict[str, Path]:
    output = Path(output_dir)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    ablation = summarize_ablation_sweep(ablation_detail)
    paths["ablation_sweep"] = tables / "ablation_top_k_sweep.csv"
    _write_csv_atomic(ablation, paths["ablation_sweep"])
    paths["ablation_figure"] = plot_ablation_sweep(
        ablation, figures / "ablation_top_k_sweep.png"
    )

    for family, detail in (
        ("prompt_patching", prompt_screen_detail),
        ("answer_patching", answer_screen_detail),
        ("steering", steering_screen_detail),
    ):
        paired = paired_control_adjusted_transport(detail, family=family)
        summary = summarize_layer_k_transport(
            paired,
            family=family,
            bootstrap_repetitions=bootstrap_repetitions,
        )
        paired_path = tables / f"{family}_paired_effects.csv.gz"
        summary_path = tables / f"{family}_layer_k_summary.csv"
        _write_csv_gzip_atomic(paired, paired_path)
        _write_csv_atomic(summary, summary_path)
        paths[f"{family}_paired"] = paired_path
        paths[f"{family}_summary"] = summary_path
        for index, figure_path in enumerate(
            plot_layer_k_heatmaps(summary, family=family, output_dir=figures),
            start=1,
        ):
            paths[f"{family}_figure_{index}"] = figure_path

    for name, frame in confirmation_tables:
        path = tables / f"{name}_confirmation_statistics.csv"
        _write_csv_atomic(frame, path)
        paths[f"{name}_confirmation"] = path
    manifest_path = output / "analysis_manifest.json"
    _write_json_atomic(
        manifest_path,
        {
            "schema_version": "realistic_niah_v4_causal_v2_analysis_v1",
            "metric": "treatment_minus_matched_control_strict_normalized_transport",
            "invalid_policy": "invalid numeric generations contribute zero effect and are reported separately",
            "paths": {name: str(path) for name, path in sorted(paths.items())},
        },
    )
    paths["manifest"] = manifest_path
    return paths
