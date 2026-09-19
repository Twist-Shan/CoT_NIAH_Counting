from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows: list[dict[str, object]] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        path = args.campaign_root / "inputs" / f"{model}.M10_matched_controls.json"
        plan = json.loads(path.read_text(encoding="utf-8"))
        for top_k, sets in sorted(plan["sets"].items(), key=lambda item: int(item[0])):
            for frozen in sorted(sets, key=lambda row: int(row["replicate"])):
                ranked = [float(match["ranked_M10"]) for match in frozen["matches"]]
                control = [float(match["control_M10"]) for match in frozen["matches"]]
                gaps = [abs(a - b) for a, b in zip(ranked, control)]
                rows.append(
                    {
                        "model": model,
                        "top_k": int(top_k),
                        "replicate": int(frozen["replicate"]),
                        "ranked_control_overlap": int(frozen["ranked_control_overlap"]),
                        "overlap_fraction": int(frozen["ranked_control_overlap"]) / int(top_k),
                        "ranked_M10_mean": sum(ranked) / len(ranked),
                        "control_M10_mean": sum(control) / len(control),
                        "mean_absolute_M10_gap": sum(gaps) / len(gaps),
                        "max_absolute_M10_gap": max(gaps),
                        "heads": ",".join(
                            f"L{int(head['layer'])}H{int(head['head'])}"
                            for head in frozen["heads"]
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    summary = (
        frame.groupby(["model", "top_k"], as_index=False)
        .agg(
            mean_overlap_fraction=("overlap_fraction", "mean"),
            max_overlap_fraction=("overlap_fraction", "max"),
            ranked_M10_mean=("ranked_M10_mean", "mean"),
            control_M10_mean=("control_M10_mean", "mean"),
            mean_absolute_M10_gap=("mean_absolute_M10_gap", "mean"),
            max_absolute_M10_gap=("max_absolute_M10_gap", "max"),
        )
    )
    summary.to_csv(args.output.with_name("first_span_M10_matching_summary.csv"), index=False)


if __name__ == "__main__":
    main()
