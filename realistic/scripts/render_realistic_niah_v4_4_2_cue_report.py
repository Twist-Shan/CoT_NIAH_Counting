from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


MODELS = ("Qwen3-8B", "Gemma4-E4B")
MODES = ("nonthinking", "native_thinking")
PROMPTS = ("cue_present", "cue_absent")
REGIONS = (
    "cue",
    "needle_span",
    "needle_end",
    "passage",
    "question",
    "other_prompt",
    "trace",
    "final",
)
COLORS = {"nonthinking": "#4C78A8", "native_thinking": "#E45756"}


def bootstrap_ci(values: pd.Series, rng: np.random.Generator) -> list[float]:
    array = values.dropna().to_numpy(float)
    draws = rng.choice(array, size=(5000, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def json_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def paired_behavior(behavior: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        subset = behavior[behavior.model_label == model]
        indexed = {
            (mode, prompt): frame.set_index("stimulus_id")["correct"]
            for (mode, prompt), frame in subset.groupby(["mode", "prompt_variant"])
        }
        pairs = []
        for mode in MODES:
            pairs.append(
                (
                    "cue_effect",
                    mode,
                    "cue_absent - cue_present",
                    indexed[(mode, "cue_present")],
                    indexed[(mode, "cue_absent")],
                )
            )
        for prompt in PROMPTS:
            pairs.append(
                (
                    "mode_effect",
                    prompt,
                    "native_thinking - nonthinking",
                    indexed[("nonthinking", prompt)],
                    indexed[("native_thinking", prompt)],
                )
            )
        for effect_type, condition, contrast, left, right in pairs:
            pair = pd.concat([left, right], axis=1, join="inner")
            pair.columns = ["left", "right"]
            delta = pair.right.astype(int) - pair.left.astype(int)
            interval = bootstrap_ci(delta, rng)
            left_only = int((pair.left & ~pair.right).sum())
            right_only = int((~pair.left & pair.right).sum())
            discordant = left_only + right_only
            rows.append(
                {
                    "effect_type": effect_type,
                    "model_label": model,
                    "condition": condition,
                    "contrast": contrast,
                    "n": len(pair),
                    "left_accuracy": float(pair.left.mean()),
                    "right_accuracy": float(pair.right.mean()),
                    "accuracy_delta": float(delta.mean()),
                    "ci95_low": interval[0],
                    "ci95_high": interval[1],
                    "left_only_correct": left_only,
                    "right_only_correct": right_only,
                    "mcnemar_exact_p": float(
                        binomtest(min(left_only, right_only), discordant, 0.5).pvalue
                    )
                    if discordant
                    else 1.0,
                }
            )
    return pd.DataFrame(rows)


def stimulus_summary(
    frame: pd.DataFrame,
    value: str,
    group_columns: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    per_stimulus = (
        frame.groupby(group_columns + ["stimulus_id"], as_index=False)[value].mean()
    )
    rows = []
    for keys, part in per_stimulus.groupby(group_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        interval = bootstrap_ci(part[value], rng)
        row.update(
            {
                "stimuli": int(part.stimulus_id.nunique()),
                "mean": float(part[value].mean()),
                "ci95_low": interval[0],
                "ci95_high": interval[1],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    xlabels: list[str],
    ylabels: list[str] | None = None,
    vmax: float | None = None,
    xlabel: str = "Key region",
    ylabel: str = "Layer",
) -> Any:
    finite = np.abs(matrix[np.isfinite(matrix)])
    if vmax is None:
        vmax = float(np.quantile(finite, 0.98)) if finite.size else 1.0
    vmax = max(vmax, 1e-8)
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right", fontsize=8)
    if ylabels is not None:
        step = max(1, len(ylabels) // 8)
        positions = np.arange(0, len(ylabels), step)
        ax.set_yticks(positions, [ylabels[index] for index in positions], fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    return image


def load_npz_layers(path: Path) -> dict[str, dict[int, np.ndarray]]:
    result: dict[str, dict[int, np.ndarray]] = {model: {} for model in MODELS}
    pattern = re.compile(r"(.+)__layer_(\d+)__mean")
    with np.load(path) as saved:
        for key in saved.files:
            match = pattern.fullmatch(key)
            if match:
                result[match.group(1)][int(match.group(2))] = saved[key]
    return result


def chunked_mode_attention(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    layer_parts = []
    stimulus_parts = []
    for chunk in pd.read_csv(path, chunksize=400_000):
        chunk = chunk[chunk.comparison.isin(["cue_present_mode_effect", "cue_absent_mode_effect"])]
        layer_parts.append(
            chunk.groupby(
                ["comparison", "model_label", "layer", "region"], as_index=False
            ).agg(total=("mass_delta", "sum"), n=("mass_delta", "size"))
        )
        stimulus_parts.append(
            chunk.groupby(
                ["comparison", "model_label", "stimulus_id", "region"],
                as_index=False,
            ).agg(total=("mass_delta", "sum"), n=("mass_delta", "size"))
        )
    layers = pd.concat(layer_parts).groupby(
        ["comparison", "model_label", "layer", "region"], as_index=False
    ).sum()
    layers["mass_delta"] = layers.total / layers.n
    stimuli = pd.concat(stimulus_parts).groupby(
        ["comparison", "model_label", "stimulus_id", "region"], as_index=False
    ).sum()
    stimuli["mass_delta"] = stimuli.total / stimuli.n
    return layers, stimuli


def render(tables: Path, cue: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(442)
    behavior = pd.read_csv(tables / "behavior.csv.gz")
    behavior["correct"] = behavior.exact_count.astype(str).str.lower().isin(
        ["1", "true", "yes"]
    )
    paired = paired_behavior(behavior, rng)

    # Figure 1: paired behavioral effects.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, effect_type, title in zip(
        axes,
        ["cue_effect", "mode_effect"],
        ["Removing the opening cue", "Enabling native thinking"],
    ):
        part = paired[paired.effect_type == effect_type].reset_index(drop=True)
        labels = [f"{row.model_label}\n{row.condition}" for row in part.itertuples()]
        values = part.accuracy_delta.to_numpy()
        errors = np.vstack(
            [values - part.ci95_low.to_numpy(), part.ci95_high.to_numpy() - values]
        )
        colors = [
            COLORS.get(row.condition, "#59A14F") for row in part.itertuples()
        ]
        ax.bar(range(len(part)), values, color=colors, alpha=0.85)
        ax.errorbar(range(len(part)), values, yerr=errors, fmt="none", color="black", capsize=4)
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_xticks(range(len(part)), labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("Paired accuracy delta")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "behavior_effects.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    hidden_cue = pd.read_csv(cue / "hidden_cue_effect.csv.gz")
    hidden_native = pd.read_csv(tables / "hidden_paired_effects.csv.gz")
    hidden_native = hidden_native[hidden_native.comparison == "native_cue_effect"]
    hidden_cue_layer = hidden_cue.groupby(
        ["model_label", "mode", "layer"], as_index=False
    ).relative_l2_delta.mean()
    hidden_native_layer = hidden_native.groupby(
        ["model_label", "site", "layer"], as_index=False
    ).relative_l2_delta.mean()

    # Figure 2: layerwise hidden cue effects.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, model in enumerate(MODELS):
        ax = axes[row, 0]
        for mode in MODES:
            part = hidden_cue_layer[
                (hidden_cue_layer.model_label == model)
                & (hidden_cue_layer["mode"] == mode)
            ]
            ax.plot(
                part.layer,
                part.relative_l2_delta,
                label=mode.replace("_", " "),
                color=COLORS[mode],
                linewidth=1.8,
            )
        ax.set_title(f"{model}: answer-query geometry")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Relative L2, cue absent − present")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
        ax = axes[row, 1]
        for site, color in zip(
            ["answer_query", "trace_mean", "trace"],
            ["#4C78A8", "#59A14F", "#E45756"],
        ):
            part = hidden_native_layer[
                (hidden_native_layer.model_label == model)
                & (hidden_native_layer.site == site)
            ]
            ax.plot(part.layer, part.relative_l2_delta, label=site, color=color, linewidth=1.8)
        ax.set_title(f"{model}: native trace sites")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Relative L2, cue absent − present")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(output / "hidden_cue_effect_by_layer.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    answer_cue = pd.read_csv(cue / "answer_attention_cue_effect.csv.gz")
    answer_layer = answer_cue.groupby(
        ["model_label", "mode", "layer", "region"], as_index=False
    ).mean_head_mass_delta.mean()

    # Figure 3: answer-query attention cue effects by layer and region.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    images = []
    matrices = []
    for model in MODELS:
        for mode in MODES:
            part = answer_layer[
                (answer_layer.model_label == model) & (answer_layer["mode"] == mode)
            ]
            pivot = part.pivot(index="layer", columns="region", values="mean_head_mass_delta").reindex(columns=REGIONS)
            matrices.append(pivot.to_numpy())
    vmax = float(np.quantile(np.abs(np.concatenate([x.ravel() for x in matrices])), 0.98))
    for ax, model, mode, matrix in zip(axes.flat, np.repeat(MODELS, 2), MODES * 2, matrices):
        image = heatmap(
            ax,
            matrix,
            f"{model} · {mode.replace('_', ' ')}",
            list(REGIONS),
            [str(value) for value in range(matrix.shape[0])],
            vmax=vmax,
        )
        images.append(image)
    fig.colorbar(images[0], ax=axes, shrink=0.75, label="Attention mass delta (absent − present)")
    fig.savefig(output / "answer_attention_cue_effect_by_layer.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    trace_cue = pd.read_csv(cue / "native_trace_attention_cue_effect.csv.gz")
    trace_layer = trace_cue.groupby(
        ["model_label", "layer", "region"], as_index=False
    ).mean_head_mass_delta.mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    matrices = []
    pivots = []
    for model in MODELS:
        part = trace_layer[trace_layer.model_label == model]
        pivot = part.pivot(index="layer", columns="region", values="mean_head_mass_delta").reindex(columns=REGIONS)
        pivots.append(pivot)
        matrices.append(pivot.to_numpy())
    vmax = float(np.quantile(np.abs(np.concatenate([x.ravel() for x in matrices])), 0.98))
    for ax, model, pivot in zip(axes, MODELS, pivots):
        image = heatmap(
            ax,
            pivot.to_numpy(),
            model,
            list(REGIONS),
            [str(value) for value in pivot.index],
            vmax=vmax,
        )
    fig.colorbar(image, ax=axes, shrink=0.8, label="Trace attention mass delta")
    fig.savefig(output / "native_trace_attention_cue_effect_by_layer.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 5: trace-time effects for selected regions.
    trace_maps = load_npz_layers(cue / "native_trace_region_map_cue_effect.npz")
    selected_regions = ["cue", "needle_span", "question", "trace"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for row, model in enumerate(MODELS):
        layers = sorted(trace_maps[model])
        cube = np.stack([trace_maps[model][layer] for layer in layers])
        vmax = float(np.nanquantile(np.abs(cube[:, :, [REGIONS.index(r) for r in selected_regions]]), 0.99))
        for col, region in enumerate(selected_regions):
            matrix = cube[:, :, REGIONS.index(region)]
            ax = axes[row, col]
            image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{model} · {region}", fontsize=9)
            ax.set_xlabel("Normalized trace query time")
            ax.set_ylabel("Layer")
            ax.set_xticks([0, 32, 64, 96, 127], ["0", ".25", ".5", ".75", "1"])
            step = max(1, len(layers) // 6)
            pos = np.arange(0, len(layers), step)
            ax.set_yticks(pos, [layers[i] for i in pos])
        fig.colorbar(image, ax=axes[row, :], shrink=0.7, label="Attention mass delta")
    fig.savefig(output / "native_trace_time_cue_effect.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 6: trace-to-trace maps at early/middle/late layers.
    ttt_maps = load_npz_layers(cue / "native_trace_to_trace_cue_effect.npz")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for row, model in enumerate(MODELS):
        layers = sorted(ttt_maps[model])
        selected = [layers[0], layers[len(layers) // 2], layers[-1]]
        vmax = float(np.nanquantile(np.abs(np.stack([ttt_maps[model][x] for x in selected])), 0.995))
        for col, layer in enumerate(selected):
            matrix = ttt_maps[model][layer]
            ax = axes[row, col]
            image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
            ax.set_title(f"{model} · layer {layer}", fontsize=9)
            ax.set_xlabel("Trace key time")
            ax.set_ylabel("Trace query time")
            ax.set_xticks([0, 64, 127], ["0", ".5", "1"])
            ax.set_yticks([0, 64, 127], ["0", ".5", "1"])
        fig.colorbar(image, ax=axes[row, :], shrink=0.7, label="Trace-to-trace mass delta")
    fig.savefig(output / "native_trace_to_trace_cue_effect.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Mode-effect layer plots for the report appendix.
    mode_hidden = pd.read_csv(tables / "hidden_paired_effects.csv.gz")
    mode_hidden = mode_hidden[mode_hidden.comparison.isin(["cue_present_mode_effect", "cue_absent_mode_effect"])]
    mode_hidden_layer = mode_hidden.groupby(
        ["comparison", "model_label", "layer"], as_index=False
    ).relative_l2_delta.mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        for comparison, label, color in [
            ("cue_present_mode_effect", "cue present", "#4C78A8"),
            ("cue_absent_mode_effect", "cue absent", "#E45756"),
        ]:
            part = mode_hidden_layer[
                (mode_hidden_layer.model_label == model)
                & (mode_hidden_layer.comparison == comparison)
            ]
            ax.plot(part.layer, part.relative_l2_delta, label=label, color=color, linewidth=1.8)
        ax.set_title(model)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Relative L2, native − non-thinking")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.savefig(output / "hidden_mode_effect_by_layer.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    mode_attn_layer, mode_attn_stim = chunked_mode_attention(
        tables / "attention_paired_effects.csv.gz"
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    matrices = []
    labels = []
    for model in MODELS:
        for comparison, prompt in [
            ("cue_present_mode_effect", "cue present"),
            ("cue_absent_mode_effect", "cue absent"),
        ]:
            part = mode_attn_layer[
                (mode_attn_layer.model_label == model)
                & (mode_attn_layer.comparison == comparison)
            ]
            pivot = part.pivot(index="layer", columns="region", values="mass_delta").reindex(columns=REGIONS)
            matrices.append(pivot.to_numpy())
            labels.append((model, prompt, [str(value) for value in pivot.index]))
    vmax = float(np.quantile(np.abs(np.concatenate([x.ravel() for x in matrices])), 0.98))
    for ax, matrix, (model, prompt, layers) in zip(axes.flat, matrices, labels):
        image = heatmap(ax, matrix, f"{model} · {prompt}", list(REGIONS), layers, vmax=vmax)
    fig.colorbar(image, ax=axes, shrink=0.75, label="Attention mass delta (native − non-thinking)")
    fig.savefig(output / "answer_attention_mode_effect_by_layer.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    hidden_cue_summary = stimulus_summary(
        hidden_cue,
        "relative_l2_delta",
        ["model_label", "mode", "site"],
        rng,
    )
    answer_cue_summary = stimulus_summary(
        answer_cue,
        "mean_head_mass_delta",
        ["model_label", "mode", "region"],
        rng,
    )
    trace_cue_summary = stimulus_summary(
        trace_cue,
        "mean_head_mass_delta",
        ["model_label", "region"],
        rng,
    )
    mode_hidden_summary = stimulus_summary(
        mode_hidden,
        "relative_l2_delta",
        ["comparison", "model_label", "site"],
        rng,
    )
    mode_attention_summary = stimulus_summary(
        mode_attn_stim,
        "mass_delta",
        ["comparison", "model_label", "region"],
        rng,
    )
    top_hidden_layers = (
        hidden_cue_layer.sort_values(
            ["model_label", "mode", "relative_l2_delta"],
            ascending=[True, True, False],
        )
        .groupby(["model_label", "mode"], as_index=False)
        .head(5)
    )
    head_summary = pd.read_csv(cue / "answer_attention_head_cue_effect.csv.gz")
    top_heads = (
        head_summary[head_summary.region.isin(["cue", "needle_span", "question", "trace"])]
        .assign(abs_delta=lambda x: x.mean_mass_delta.abs())
        .sort_values(
            ["model_label", "mode", "region", "abs_delta"],
            ascending=[True, True, True, False],
        )
        .groupby(["model_label", "mode", "region"], as_index=False)
        .head(3)
        .drop(columns="abs_delta")
    )
    result = {
        "paired_behavior": json_records(paired),
        "hidden_cue_summary": json_records(hidden_cue_summary),
        "answer_attention_cue_summary": json_records(answer_cue_summary),
        "trace_attention_cue_summary": json_records(trace_cue_summary),
        "mode_hidden_summary": json_records(mode_hidden_summary),
        "mode_attention_summary": json_records(mode_attention_summary),
        "top_hidden_layers": json_records(top_hidden_layers),
        "top_answer_attention_heads": json_records(top_heads),
    }
    (output / "report_statistics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", required=True)
    parser.add_argument("--cue-effects", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = render(Path(args.tables), Path(args.cue_effects), Path(args.output_dir))
    print(json.dumps({key: len(value) for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
