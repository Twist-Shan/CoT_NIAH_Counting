from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _figure_dir_for_values_csv(run_dir: Path, values_csv: Path) -> Path:
    rel = values_csv.relative_to(run_dir / "tables")
    return run_dir / "figures" / rel.parent


def _plot_group(
    *,
    rows: list[dict[str, str]],
    out_path: Path,
    linthresh_percentile: float,
) -> None:
    first = rows[0]
    mode = str(first.get("mode", "unknown_mode"))
    layer = int(float(first.get("layer", 0)))
    row_id = str(first.get("row_id", first.get("local_example_index", "unknown")))
    local_example_index = int(float(first.get("local_example_index", 0)))
    gold_count = int(float(first.get("gold_count", 0)))
    model_exact_match = first.get("model_exact_match")

    rows = sorted(rows, key=lambda row: int(float(row["token_position"])))
    positions = np.asarray(
        [int(float(row["token_position"])) for row in rows], dtype=np.int64
    )
    normalized = np.asarray(
        [float(row["normalized_position"]) for row in rows], dtype=np.float32
    )
    projection = np.asarray([float(row["projection"]) for row in rows], dtype=np.float32)
    any_mask = np.asarray(
        [_as_bool(row.get("is_any_needle_token", False)) for row in rows], dtype=bool
    )
    matching_mask = np.asarray(
        [_as_bool(row.get("is_matching_needle_token", False)) for row in rows],
        dtype=bool,
    )
    non_matching_mask = any_mask & ~matching_mask

    finite_projection = projection[np.isfinite(projection)]
    linthresh = 1.0
    if finite_projection.size:
        linthresh = max(
            1.0,
            float(np.nanpercentile(np.abs(finite_projection), linthresh_percentile)),
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(positions, projection, color="#555555", linewidth=0.7, alpha=0.35)
    base = ax.scatter(
        positions[~any_mask],
        projection[~any_mask],
        c=normalized[~any_mask],
        cmap="Blues",
        s=11,
        alpha=0.78,
        linewidths=0,
        label="non-needle tokens",
    )
    if np.any(non_matching_mask):
        ax.scatter(
            positions[non_matching_mask],
            projection[non_matching_mask],
            color="#f39c12",
            marker="x",
            s=28,
            linewidths=1.2,
            label="non-matching needle tokens",
        )
    if np.any(matching_mask):
        ax.scatter(
            positions[matching_mask],
            projection[matching_mask],
            color="#d62728",
            marker="D",
            s=24,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.35,
            label="matching needle tokens",
        )
    ax.set_yscale("symlog", linthresh=linthresh)
    cbar = fig.colorbar(base, ax=ax)
    cbar.set_label("normalized token position")
    exact_label = (
        "unknown"
        if model_exact_match in {None, ""}
        else ("success" if _as_bool(model_exact_match) else "failed")
    )
    ax.set_xlabel("token position")
    ax.set_ylabel("ridge prediction (symlog)")
    ax.set_title(
        f"{mode} L{layer} row={row_id} gold={gold_count} {exact_label} "
        f"(linthresh={linthresh:.2g})"
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def regenerate_sequence_projection_plots(
    *,
    run_dir: Path,
    overwrite: bool,
    suffix: str,
    linthresh_percentile: float,
) -> list[Path]:
    if not (run_dir / "tables").exists():
        raise FileNotFoundError(f"Run directory has no tables/ folder: {run_dir}")
    values_files = sorted(
        (run_dir / "tables" / "counting_features" / "probe_diagnostics").glob(
            "*/sequence_projection/sequence_projection_values_layer_*.csv"
        )
    )
    if not values_files:
        raise FileNotFoundError(
            "No sequence projection value CSVs found under "
            f"{run_dir / 'tables' / 'counting_features' / 'probe_diagnostics'}"
        )

    written: list[Path] = []
    for values_csv in values_files:
        rows = _load_rows(values_csv)
        if not rows:
            continue
        by_example: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_example[str(row["local_example_index"])].append(row)
        figure_dir = _figure_dir_for_values_csv(run_dir, values_csv)
        layer = int(float(rows[0].get("layer", 0)))
        for local_example_index, example_rows in sorted(
            by_example.items(), key=lambda item: int(float(item[0]))
        ):
            stem = f"sequence_projection_layer_{layer}_example_{local_example_index}"
            filename = f"{stem}.png" if overwrite else f"{stem}{suffix}.png"
            out_path = figure_dir / filename
            _plot_group(
                rows=example_rows,
                out_path=out_path,
                linthresh_percentile=linthresh_percentile,
            )
            written.append(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate saved sequence-projection diagnostic plots from CSV data "
            "using a symlog y-axis, without rerunning model inference or probing."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run folder containing tables/ and figures/ subfolders.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Write new PNGs with a suffix instead of replacing existing plots.",
    )
    parser.add_argument(
        "--suffix",
        default="_symlog",
        help="Filename suffix used with --no-overwrite.",
    )
    parser.add_argument(
        "--linthresh-percentile",
        type=float,
        default=99.0,
        help=(
            "Percentile of absolute projection values used for symlog linthresh. "
            "Default: 99."
        ),
    )
    args = parser.parse_args()
    if not math.isfinite(args.linthresh_percentile) or not (
        0 < args.linthresh_percentile <= 100
    ):
        raise ValueError("--linthresh-percentile must be in (0, 100].")
    written = regenerate_sequence_projection_plots(
        run_dir=args.run_dir,
        overwrite=not args.no_overwrite,
        suffix=str(args.suffix),
        linthresh_percentile=float(args.linthresh_percentile),
    )
    print(f"Wrote {len(written)} sequence projection plot(s).")
    for path in written[:10]:
        print(path)
    if len(written) > 10:
        print(f"... {len(written) - 10} more")


if __name__ == "__main__":
    main()
