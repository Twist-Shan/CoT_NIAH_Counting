#!/usr/bin/env python3
"""Build the self-contained native-thinking internal-counter research report."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "NiaH_Native-thinking_Internal-counter_report.html"

COHORT_FILES = {
    3: ROOT
    / "work"
    / "audit_n3_first_pass_noindex_20260827"
    / "selected_rows_first_pass_noindex_v5.jsonl",
    10: ROOT
    / "work"
    / "audit_n10_first_pass_noindex_20260827"
    / "selected_rows_first_pass_noindex_v5.jsonl",
}

CONTEXT_FILES = {
    3: ROOT
    / "work"
    / "audit_n3_first_pass_noindex_20260827"
    / "tstar_first_pass_v2"
    / "tstar_first_pass_contexts_v2.jsonl",
    10: ROOT
    / "work"
    / "audit_n10_first_pass_noindex_20260827"
    / "tstar_first_pass_v2"
    / "tstar_first_pass_contexts_v2.jsonl",
}

MANIFEST_FILES = {
    3: ROOT
    / "work"
    / "audit_n3_first_pass_noindex_20260827"
    / "cohort_manifest_first_pass_noindex_v5.json",
    10: ROOT
    / "work"
    / "audit_n10_first_pass_noindex_20260827"
    / "cohort_manifest_first_pass_noindex_v5.json",
}

GEOMETRY_FILES = {
    3: ROOT
    / "work"
    / "counting_mechanism_diagnostics_20260827"
    / "all_layer_plus1_n3_v3"
    / "analysis"
    / "summary.json",
    10: ROOT
    / "work"
    / "counting_mechanism_diagnostics_20260827"
    / "all_layer_plus1_n10_v3"
    / "analysis"
    / "summary.json",
}

KGRID_FILES = {
    "uniform5_overall": ROOT
    / "work"
    / "same_site_progress_transplant_20260827"
    / "n10_kgrid5_v1"
    / "analysis"
    / "overall.json",
    "uniform5_cells": ROOT
    / "work"
    / "same_site_progress_transplant_20260827"
    / "n10_kgrid5_v1"
    / "analysis"
    / "cells.jsonl",
    "grammar20_manifest": ROOT
    / "work"
    / "same_site_progress_transplant_20260827"
    / "n10_kgrid20_geometry_v1"
    / "manifest.json",
    "seed1791_overall": ROOT
    / "work"
    / "same_site_progress_transplant_20260827"
    / "n10_kgrid_seed1791_v1"
    / "analysis"
    / "overall.json",
    "seed1791_cells": ROOT
    / "work"
    / "same_site_progress_transplant_20260827"
    / "n10_kgrid_seed1791_v1"
    / "analysis"
    / "cells.jsonl",
    "combined6_overall": ROOT
    / "work"
    / "same_site_progress_transplant_20260827"
    / "n10_kgrid_combined6_v1"
    / "overall.json",
    "combined6_by_grammar": ROOT
    / "work"
    / "same_site_progress_transplant_20260827"
    / "n10_kgrid_combined6_v1"
    / "per_commit_grammar.json",
    "confirmation_grammar": ROOT
    / "work"
    / "same_site_progress_transplant_20260827"
    / "n10_confirmation10_l31_v1"
    / "grammar_stratified_v1"
    / "by_commit_grammar.json",
}

EVENT_TAIL_FILES = {
    "discovery20": ROOT
    / "work"
    / "same_site_progress_transplant_20260827"
    / "n10_natural_crossk_attention_generation_v1"
    / "discovery20_frozen_crossk_analysis.json",
    "confirmation10": ROOT
    / "work"
    / "same_site_progress_transplant_20260827"
    / "n10_natural_crossk_attention_generation_v1"
    / "confirmation10_frozen_crossk_analysis.json",
    "site_candidates": ROOT
    / "work"
    / "same_site_progress_transplant_20260827"
    / "n10_kgrid20_site_candidates_w12_v1"
    / "site_candidate_analysis.json",
    "period_span": ROOT
    / "work"
    / "same_site_progress_transplant_20260827"
    / "n10_natural_span_direction_localization_v1"
    / "period_like_l16_k6"
    / "paired_span_direction_analysis.json",
    "whitespace_span": ROOT
    / "work"
    / "same_site_progress_transplant_20260827"
    / "n10_natural_span_direction_localization_v1"
    / "whitespace_tail0_l16_k6"
    / "paired_span_direction_analysis.json",
}

PATCH_SCOPE_FILES = {
    "layer_sweep": ROOT
    / "work"
    / "same_site_progress_transplant_20260827"
    / "n10_patch_scope_layer_sweep_v2"
    / "layer_sweep_analysis.json",
    "layer_plot": ROOT
    / "work"
    / "same_site_progress_transplant_20260827"
    / "n10_patch_scope_layer_sweep_v2"
    / "layer_sweep_effect_sizes.png",
    "frozen_confirmation": ROOT
    / "work"
    / "same_site_progress_transplant_20260827"
    / "n10_patch_scope_frozen_v2"
    / "frozen_scope_analysis.json",
    "item_span_generation_audit": ROOT
    / "work"
    / "same_site_progress_transplant_20260827"
    / "n10_patch_scope_frozen_v2"
    / "item_span_generation_manual_audit.json",
    "item_span_l16": ROOT
    / "work"
    / "same_site_progress_transplant_20260827"
    / "n10_item_span_contextual_l16_v1"
    / "frozen_scope_analysis.json",
    "item_span_l16_generation_audit": ROOT
    / "work"
    / "same_site_progress_transplant_20260827"
    / "n10_item_span_contextual_l16_v1"
    / "item_span_generation_manual_audit.json",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    # Do not use splitlines(): model text may contain Unicode paragraph separators.
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def build_samples() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    cohort_meta: dict[str, Any] = {}
    for count in (3, 10):
        rows = read_jsonl(COHORT_FILES[count])
        contexts = read_jsonl(CONTEXT_FILES[count])
        manifest = json.loads(MANIFEST_FILES[count].read_text(encoding="utf-8"))
        by_seed = {int(row["seed"]): row for row in rows}
        context_by_seed = {int(row["seed"]): row for row in contexts}
        expected = set(manifest["discovery_seeds"] + manifest["confirmation_seeds"])
        if set(by_seed) != expected or set(context_by_seed) != expected:
            raise ValueError(f"N={count}: cohort/context seeds do not match manifest")
        if len(rows) != 30 or len(contexts) != 30:
            raise ValueError(f"N={count}: expected exactly 30 rows")

        cohort_meta[str(count)] = {
            "attemptedSeedCount": manifest["attempted_seed_count"],
            "lastScannedSeed": manifest["last_scanned_seed"],
            "discoverySeeds": manifest["discovery_seeds"],
            "confirmationSeeds": manifest["confirmation_seeds"],
            "globalCleanSensitivity": manifest["strict_sensitivity"],
            "sourceSha256": sha256(COHORT_FILES[count]),
            "contextSha256": sha256(CONTEXT_FILES[count]),
            "manifestSha256": sha256(MANIFEST_FILES[count]),
        }

        ordered_seeds = manifest["discovery_seeds"] + manifest["confirmation_seeds"]
        for seed in ordered_seeds:
            row = by_seed[int(seed)]
            context = context_by_seed[int(seed)]
            audit = row[f"noindex_n{count}_format_audit"]
            cohort = row[f"noindex_n{count}_cohort"]
            if cohort["split"] != context["split"]:
                raise ValueError(f"N={count} seed={seed}: formal split mismatch")
            if not audit["primary_eligible_prefix_clean"]:
                raise ValueError(f"N={count} seed={seed}: prefix-clean gate failed")
            if context["future_recap_available_to_context"]:
                raise ValueError(f"N={count} seed={seed}: future recap leaked")

            trace = row.get("trace_parse") or {}
            reasoning = trace.get("reasoning_text") or row["clean_output_text"]
            full_output = row["clean_output_text"]
            samples.append(
                {
                    "n": count,
                    "seed": int(seed),
                    "split": context["split"],
                    "rank": int(cohort["rank_within_split"]),
                    "requestId": row["request_id"],
                    "stimulusId": row["stimulus_id"],
                    "model": row["model_label"],
                    "promptTokens": int(row["prompt_token_count"]),
                    "fullOutputTokens": int(row["output_tokens"]),
                    "mechanismPrefixTokens": int(context["output_prefix_token_count"]),
                    "removedOutputTokens": int(context["removed_output_token_count"]),
                    "goldCount": int(row["gold_count"]),
                    "goldRecords": row["gold_records"],
                    "firstOccurrences": context["first_occurrences"],
                    "mechanismPrefix": context["raw_prefix_text"],
                    "fullOutput": full_output,
                    "fullReasoning": reasoning,
                    "userPrompt": row["user_text"],
                    "finalText": trace.get("final_text", ""),
                    "parsedCount": trace.get("parsed_count"),
                    "exactCount": bool(trace.get("exact_count", False)),
                    "tStarChar": int(context["t_star_char"]),
                    "stopCharEnd": int(context["stop_char_end"]),
                    "rightSpillChars": int(context["token_boundary_right_spill_chars"]),
                    "rightSpillText": context["token_boundary_right_spill_text"],
                    "stoppingRule": context["stopping_rule"],
                    "prefixClean": bool(audit["primary_eligible_prefix_clean"]),
                    "globalClean": bool(audit["global_clean_eligible"]),
                    "strictNoExplicitCue": bool(
                        audit["strict_eligible_no_explicit_count_cue"]
                    ),
                    "firstPassComplete": bool(audit["first_pass_complete"]),
                    "preTStarEventCount": int(
                        audit["pre_tstar_score_supported_event_count"]
                    ),
                    "preTStarDuplicateEvidence": len(
                        audit["pre_tstar_repeated_gold_evidence"]
                    ),
                    "prefixCueCount": len(audit["prefix_cues"]),
                    "futureRecapExcluded": not bool(
                        context["future_recap_available_to_context"]
                    ),
                    "sourceRowSha256": context["source_row_sha256"],
                    "sourceOutputSha256": context["source_raw_output_sha256"],
                    "contextStatus": context["status"],
                }
            )
    return samples, cohort_meta


def build_geometry() -> dict[str, list[dict[str, float]]]:
    output: dict[str, list[dict[str, float]]] = {}
    condition_names = {
        "position_difference": "real",
        "opposite_position_difference": "opposite",
        "norm_matched_orthogonal": "orthogonal",
    }
    for count, path in GEOMETRY_FILES.items():
        summary = json.loads(path.read_text(encoding="utf-8"))
        groups: dict[tuple[str, str], list[float]] = {}
        for row in summary["linear_additivity"]:
            key = (row["steering_band"], condition_names[row["condition"]])
            groups.setdefault(key, []).append(
                float(row["metrics"]["donor_aligned_expected_shift"]["mean"])
            )
        series: list[dict[str, float]] = []
        for layer in range(36):
            band = f"layer_{layer:02d}"
            means = {
                name: sum(groups[(band, name)]) / len(groups[(band, name)])
                for name in ("real", "opposite", "orthogonal")
            }
            series.append(
                {
                    "layer": layer,
                    **means,
                    "contrast": means["real"]
                    - 0.5 * (means["opposite"] + means["orthogonal"]),
                }
            )
        output[str(count)] = series
    return output


def validate_kgrid() -> dict[str, str]:
    base = json.loads(KGRID_FILES["uniform5_overall"].read_text(encoding="utf-8"))
    replication = json.loads(
        KGRID_FILES["seed1791_overall"].read_text(encoding="utf-8")
    )
    grammar = json.loads(
        KGRID_FILES["grammar20_manifest"].read_text(encoding="utf-8")
    )
    combined = json.loads(
        KGRID_FILES["combined6_overall"].read_text(encoding="utf-8")
    )
    confirmation_grammar = {
        row["commit_grammar"]: row
        for row in json.loads(
            KGRID_FILES["confirmation_grammar"].read_text(encoding="utf-8")
        )
    }
    expected = {
        "base_cells": 40,
        "base_skips": 5,
        "replication_cells": 8,
        "replication_skips": 6,
        "grammar_cells": 160,
        "combined_cells": 48,
        "combined_skips": 11,
        "confirmation_plain_cells": 2,
        "confirmation_plain_skips": 2,
        "confirmation_other_cells": 8,
        "confirmation_other_skips": 0,
    }
    observed = {
        "base_cells": int(base["eligible_cell_count"]),
        "base_skips": int(base["first_successor_skip_count"]),
        "replication_cells": int(replication["eligible_cell_count"]),
        "replication_skips": int(replication["first_successor_skip_count"]),
        "grammar_cells": int(grammar["passed_cell_count"]),
        "combined_cells": int(combined["eligible_cell_count"]),
        "combined_skips": int(combined["first_successor_skip_count"]),
        "confirmation_plain_cells": int(
            confirmation_grammar["plain_period"]["eligible_cell_count"]
        ),
        "confirmation_plain_skips": int(
            confirmation_grammar["plain_period"]["first_successor_skip_count"]
        ),
        "confirmation_other_cells": sum(
            int(row["eligible_cell_count"])
            for grammar_name, row in confirmation_grammar.items()
            if grammar_name != "plain_period"
        ),
        "confirmation_other_skips": sum(
            int(row["first_successor_skip_count"])
            for grammar_name, row in confirmation_grammar.items()
            if grammar_name != "plain_period"
        ),
    }
    if observed != expected:
        raise ValueError(f"Natural k-grid summary changed: {observed} != {expected}")
    return {name: sha256(path) for name, path in KGRID_FILES.items()}


def validate_event_tail() -> dict[str, str]:
    discovery = json.loads(EVENT_TAIL_FILES["discovery20"].read_text(encoding="utf-8"))[
        "pooled_summary"
    ]
    confirmation = json.loads(
        EVENT_TAIL_FILES["confirmation10"].read_text(encoding="utf-8")
    )["pooled_summary"]
    expected = {
        "discovery_cells": 120,
        "discovery_seeds": 20,
        "discovery_positive_logodds": 1.0,
        "discovery_positive_attention": 116 / 120,
        "confirmation_cells": 60,
        "confirmation_seeds": 10,
        "confirmation_positive_logodds": 1.0,
        "confirmation_positive_attention": 59 / 60,
        "confirmation_greedy": 7 / 60,
    }
    observed = {
        "discovery_cells": int(discovery["cell_count"]),
        "discovery_seeds": int(discovery["seed_count"]),
        "discovery_positive_logodds": float(discovery["positive_logodds_shift_rate"]),
        "discovery_positive_attention": float(discovery["positive_attention_shift_rate"]),
        "confirmation_cells": int(confirmation["cell_count"]),
        "confirmation_seeds": int(confirmation["seed_count"]),
        "confirmation_positive_logodds": float(
            confirmation["positive_logodds_shift_rate"]
        ),
        "confirmation_positive_attention": float(
            confirmation["positive_attention_shift_rate"]
        ),
        "confirmation_greedy": float(
            confirmation["patched_greedy_donor_adoption_rate"]
        ),
    }
    if observed != expected:
        raise ValueError(f"Event-tail summary changed: {observed} != {expected}")
    if not (
        discovery["receiver_baseline_argmax_rate"] == 1.0
        and discovery["native_donor_argmax_rate"] == 1.0
        and confirmation["receiver_baseline_argmax_rate"] == 1.0
        and confirmation["native_donor_argmax_rate"] == 1.0
    ):
        raise ValueError("Event-tail receiver/native gates are no longer perfect")
    return {name: sha256(path) for name, path in EVENT_TAIL_FILES.items()}


def validate_patch_scope() -> dict[str, str]:
    layer = json.loads(PATCH_SCOPE_FILES["layer_sweep"].read_text(encoding="utf-8"))
    selected = {row["scope"]: row for row in layer["scopes"]}
    observed_layers = {
        scope: int(row["selected_layer"]) for scope, row in selected.items()
    }
    expected_layers = {"event_tail_w4": 0, "item_end_w1": 26, "item_span": 0}
    if observed_layers != expected_layers:
        raise ValueError(
            f"Patch-scope selected layers changed: {observed_layers} != {expected_layers}"
        )
    discovery_expected = {
        "event_tail_w4": (40, 23.610791863873658, 1.0),
        "item_end_w1": (40, 4.296321213245335, 0.825),
        "item_span": (40, 58.03137669800344, 1.0),
    }
    for scope, (cells, median_shift, positive_rate) in discovery_expected.items():
        summary = selected[scope]["selected_layer_summary"]
        observed = (
            int(summary["cell_count"]),
            float(summary["median_paired_logodds_shift"]),
            float(summary["positive_shift_rate"]),
        )
        expected = (cells, median_shift, positive_rate)
        if observed[0] != expected[0] or any(
            abs(observed[index] - expected[index]) > 1e-9 for index in (1, 2)
        ):
            raise ValueError(f"Patch-scope discovery changed for {scope}: {observed}")

    frozen = json.loads(
        PATCH_SCOPE_FILES["frozen_confirmation"].read_text(encoding="utf-8")
    )
    summaries = {row["scope"]: row for row in frozen["summaries"]}
    confirmation_expected = {
        "event_tail_w4": (60, 21.979133581281303, 0.10, 0.25),
        "item_end_w1": (60, 4.437459770590067, 8 / 60, 10 / 60),
        "item_span": (60, 63.85209316574037, 1.0, 43 / 60),
    }
    for scope, expected in confirmation_expected.items():
        summary = summaries[scope]
        observed = (
            int(summary["cell_count"]),
            float(summary["median_paired_logodds_shift"]),
            float(summary["patched_donor_argmax_rate"]),
            float(summary["patched_first_known_city_donor_adoption_rate"]),
        )
        if observed[0] != expected[0] or any(
            abs(observed[index] - expected[index]) > 1e-9
            for index in (1, 2, 3)
        ):
            raise ValueError(f"Patch-scope confirmation changed for {scope}: {observed}")

    audit = json.loads(
        PATCH_SCOPE_FILES["item_span_generation_audit"].read_text(encoding="utf-8")
    )
    if audit["summary"] != {
        "adoption_after_first_40_chars_count": 3,
        "adoption_within_first_40_chars_count": 40,
        "cell_count": 60,
        "donor_adoption_count": 43,
        "donor_adoption_rate": 43 / 60,
        "strict_bullet_adoption_count": 5,
    }:
        raise ValueError("Item-span generation audit changed")
    if audit["manual_review"]["recap_only_false_positive_count"] != 0:
        raise ValueError("Item-span manual audit now contains a recap-only hit")

    l16 = json.loads(PATCH_SCOPE_FILES["item_span_l16"].read_text(encoding="utf-8"))[
        "summaries"
    ][0]
    if not (
        int(l16["cell_count"]) == 20
        and abs(float(l16["median_paired_logodds_shift"]) - 51.1471609615255)
        < 1e-9
        and abs(float(l16["patched_donor_argmax_rate"]) - 0.85) < 1e-9
        and abs(
            float(l16["patched_first_known_city_donor_adoption_rate"]) - 0.80
        )
        < 1e-9
    ):
        raise ValueError("L16 contextual item-span comparator changed")
    l16_audit = json.loads(
        PATCH_SCOPE_FILES["item_span_l16_generation_audit"].read_text(
            encoding="utf-8"
        )
    )
    if not (
        l16_audit["summary"]["cell_count"] == 20
        and l16_audit["summary"]["donor_adoption_count"] == 16
        and l16_audit["manual_review"]["recap_only_false_positive_count"] == 0
    ):
        raise ValueError("L16 item-span generation audit changed")
    return {name: sha256(path) for name, path in PATCH_SCOPE_FILES.items()}


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>NiaH Native-thinking Internal Counter Report</title>
  <link rel="icon" href="data:,">
  <style>
    :root {
      --paper: #f5f4ef;
      --surface: #fbfaf6;
      --ink: #171b18;
      --muted: #68706a;
      --faint: #8b938d;
      --line: #d8d8d0;
      --line-strong: #bfc3bb;
      --accent: #1f6f5f;
      --accent-soft: #e1eee9;
      --warning: #a95d2b;
      --warning-soft: #f3e7dc;
      --danger: #98413a;
      --danger-soft: #f1e1df;
      --code: #202722;
      --mono: "Cascadia Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      --sans: "Segoe UI Variable", "Segoe UI", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
      --shadow: 0 20px 52px rgba(37, 43, 38, .08);
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: var(--sans);
      line-height: 1.62;
      text-rendering: optimizeLegibility;
    }
    button, input, select { font: inherit; }
    button { color: inherit; }
    a { color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 3px; }
    code, .mono { font-family: var(--mono); }
    ::selection { background: #cce2da; }

    .shell { width: min(1500px, calc(100% - 36px)); margin: 20px auto 70px; }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(320px, .65fr);
      gap: 52px;
      padding: 58px 62px 46px;
      border: 1px solid var(--line);
      background: var(--surface);
      box-shadow: var(--shadow);
    }
    .eyebrow, .section-kicker, .metric-label, .panel-label {
      margin: 0 0 9px;
      color: var(--accent);
      font: 700 11px/1.3 var(--mono);
      letter-spacing: .12em;
      text-transform: uppercase;
    }
    h1 { max-width: 920px; margin: 0; font-size: clamp(36px, 5vw, 67px); line-height: 1.02; letter-spacing: -.045em; font-weight: 650; }
    .hero-dek { max-width: 820px; margin: 22px 0 0; color: #3f4842; font-size: 18px; }
    .hero-aside { align-self: end; border-top: 3px solid var(--accent); }
    .hero-stat { display: grid; grid-template-columns: 90px 1fr; gap: 16px; padding: 15px 0; border-bottom: 1px solid var(--line); }
    .hero-stat strong { font: 650 22px/1.2 var(--mono); }
    .hero-stat span { color: var(--muted); font-size: 13px; }

    .nav {
      position: sticky;
      top: 0;
      z-index: 20;
      display: flex;
      gap: 22px;
      padding: 12px 26px;
      overflow-x: auto;
      border: 1px solid var(--line);
      border-top: 0;
      background: rgba(245, 244, 239, .93);
      backdrop-filter: blur(14px);
      white-space: nowrap;
    }
    .nav a { color: #465049; font-size: 13px; text-decoration: none; }
    .nav a:hover { color: var(--accent); }

    main { padding: 0 62px 62px; border: 1px solid var(--line); border-top: 0; background: var(--surface); }
    section { padding: 62px 0 34px; border-bottom: 1px solid var(--line); }
    section:last-child { border-bottom: 0; }
    h2 { max-width: 1050px; margin: 0 0 18px; font-size: clamp(28px, 3.2vw, 44px); line-height: 1.12; letter-spacing: -.03em; font-weight: 640; }
    h3 { margin: 28px 0 10px; font-size: 20px; line-height: 1.3; }
    h4 { margin: 0 0 7px; font-size: 15px; }
    p { margin: 10px 0; }
    .lead { max-width: 940px; color: #434c46; font-size: 17px; }
    .small { color: var(--muted); font-size: 12px; }

    .callout {
      max-width: 1120px;
      margin: 24px 0;
      padding: 18px 20px;
      border-left: 4px solid var(--accent);
      background: var(--accent-soft);
    }
    .callout.warning { border-left-color: var(--warning); background: var(--warning-soft); }
    .callout.danger { border-left-color: var(--danger); background: var(--danger-soft); }
    .callout strong { font-weight: 700; }

    .seed-lab { margin-top: 28px; border-top: 3px solid var(--ink); }
    .lab-controls {
      display: grid;
      grid-template-columns: auto auto minmax(180px, 1fr) auto;
      gap: 14px;
      align-items: end;
      padding: 18px 0;
      border-bottom: 1px solid var(--line);
    }
    .control-label { display: block; margin-bottom: 6px; color: var(--muted); font: 650 11px/1.2 var(--mono); letter-spacing: .07em; text-transform: uppercase; }
    .segmented { display: inline-flex; padding: 3px; border: 1px solid var(--line-strong); background: #efeee8; }
    .segmented button, .icon-button, .seed-button, .tab-button {
      border: 0;
      background: transparent;
      cursor: pointer;
      transition: transform .18s ease, background-color .18s ease, color .18s ease;
    }
    .segmented button { min-width: 58px; padding: 7px 11px; color: var(--muted); font-size: 12px; }
    .segmented button.active { color: #fff; background: var(--ink); }
    .segmented button:active, .icon-button:active, .seed-button:active, .tab-button:active { transform: translateY(1px); }
    .search-wrap input {
      width: 100%;
      height: 38px;
      padding: 0 12px;
      border: 1px solid var(--line-strong);
      border-radius: 0;
      outline: none;
      background: #fff;
    }
    .search-wrap input:focus { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
    .stepper { display: flex; gap: 6px; }
    .icon-button { min-width: 40px; height: 38px; padding: 0 11px; border: 1px solid var(--line-strong); background: #fff; }
    .icon-button:hover { background: var(--accent-soft); }
    .icon-button:disabled { cursor: not-allowed; opacity: .35; }

    .lab-body { display: grid; grid-template-columns: 246px minmax(0, 1fr); min-height: 720px; }
    .seed-rail { padding: 16px 16px 16px 0; border-right: 1px solid var(--line); }
    .seed-count { margin: 0 0 10px; color: var(--muted); font-size: 12px; }
    .seed-list { display: grid; gap: 3px; max-height: 820px; overflow-y: auto; padding-right: 8px; }
    .seed-button {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      width: 100%;
      padding: 9px 10px;
      text-align: left;
      border-left: 2px solid transparent;
      color: #475049;
      font: 600 12px/1.2 var(--mono);
    }
    .seed-button small { color: var(--faint); font: 10px/1.2 var(--mono); }
    .seed-button:hover { background: #efeee8; }
    .seed-button.active { border-left-color: var(--accent); color: var(--ink); background: var(--accent-soft); }
    .sample-panel { min-width: 0; padding: 24px 0 10px 28px; }
    .sample-head { display: grid; grid-template-columns: 1fr auto; gap: 20px; align-items: start; padding-bottom: 18px; border-bottom: 1px solid var(--line); }
    .sample-title { margin: 0; font: 650 clamp(24px, 3vw, 34px)/1.1 var(--mono); letter-spacing: -.03em; }
    .badge-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
    .badge { padding: 4px 7px; border: 1px solid var(--line-strong); color: var(--muted); background: #fff; font: 650 10px/1.2 var(--mono); text-transform: uppercase; }
    .badge.pass { border-color: #94b8ab; color: #15584b; background: var(--accent-soft); }
    .badge.warn { border-color: #d1aa8d; color: #7d431f; background: var(--warning-soft); }
    .sample-meta { display: grid; grid-template-columns: repeat(4, minmax(110px, 1fr)); gap: 0; margin: 18px 0 0; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
    .sample-meta > div { padding: 12px 14px; border-right: 1px solid var(--line); }
    .sample-meta > div:last-child { border-right: 0; }
    .sample-meta strong { display: block; font: 650 17px/1.2 var(--mono); }
    .sample-meta span { color: var(--muted); font-size: 11px; }

    .gold-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px; margin: 18px 0; background: var(--line); border: 1px solid var(--line); }
    .gold-item { padding: 10px 12px; background: #fff; }
    .gold-item strong { display: block; font-size: 13px; }
    .gold-item span { color: var(--muted); font: 12px/1.4 var(--mono); }

    .trace-tabs { display: flex; gap: 0; margin-top: 24px; overflow-x: auto; border-bottom: 1px solid var(--line-strong); }
    .tab-button { flex: 0 0 auto; padding: 10px 14px; border-bottom: 3px solid transparent; color: var(--muted); font-size: 12px; }
    .tab-button.active { color: var(--ink); border-bottom-color: var(--accent); }
    .trace-toolbar { display: flex; justify-content: space-between; gap: 16px; align-items: center; padding: 10px 0; }
    .trace-toolbar p { margin: 0; color: var(--muted); font-size: 12px; }
    .trace-box {
      max-height: 610px;
      min-height: 330px;
      overflow: auto;
      padding: 22px 24px;
      border: 1px solid var(--line);
      background: var(--code);
      color: #edf1ed;
      font: 12px/1.72 var(--mono);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      tab-size: 2;
    }
    .trace-box mark { padding: 1px 2px; color: #10261f; background: #9ed2c2; box-decoration-break: clone; -webkit-box-decoration-break: clone; }
    .audit-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1px; background: var(--line); border: 1px solid var(--line); }
    .audit-row { display: grid; grid-template-columns: minmax(150px, .8fr) 1fr; gap: 14px; padding: 11px 12px; background: #fff; font-size: 12px; }
    .audit-row span:first-child { color: var(--muted); }
    .empty-state { padding: 80px 24px; color: var(--muted); text-align: center; border: 1px dashed var(--line-strong); }

    .metric-strip { display: grid; grid-template-columns: 1fr 1.4fr 1fr; margin: 30px 0; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
    .metric { padding: 20px 22px; border-right: 1px solid var(--line); }
    .metric:last-child { border-right: 0; }
    .metric strong { display: block; margin-bottom: 5px; font: 650 28px/1.1 var(--mono); }
    .metric p { margin: 0; color: var(--muted); font-size: 12px; }

    .chart-shell { margin: 28px 0; border-top: 1px solid var(--line-strong); border-bottom: 1px solid var(--line); }
    .scope-figure { margin: 24px 0 30px; padding: 16px; border: 1px solid var(--line); background: #fff; }
    .scope-figure img { display: block; width: 100%; height: auto; }
    .scope-figure figcaption { margin-top: 12px; color: var(--muted); font-size: 12px; }
    .chart-head { display: flex; justify-content: space-between; gap: 20px; align-items: end; padding: 16px 0; }
    .chart-head h3 { margin: 0; }
    .chart-legend { display: flex; gap: 14px; flex-wrap: wrap; color: var(--muted); font-size: 11px; }
    .legend-dot { display: inline-block; width: 18px; height: 3px; margin-right: 6px; vertical-align: middle; }
    #geometryChart { width: 100%; min-height: 390px; overflow-x: auto; }
    #geometryChart svg { display: block; width: 100%; min-width: 780px; height: auto; }
    .chart-note { padding: 14px 0 18px; color: var(--muted); font-size: 12px; }

    .flow { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 0; margin: 28px 0; border-top: 1px solid var(--line-strong); border-bottom: 1px solid var(--line); }
    .flow-step { position: relative; min-height: 190px; padding: 18px 18px 20px; border-right: 1px solid var(--line); }
    .flow-step:last-child { border-right: 0; }
    .flow-step::after { content: "→"; position: absolute; top: 22px; right: -10px; z-index: 2; width: 20px; color: var(--accent); background: var(--surface); text-align: center; }
    .flow-step:last-child::after { display: none; }
    .flow-step code { display: block; margin: 10px 0; color: var(--accent); font-size: 12px; }
    .flow-step p { color: var(--muted); font-size: 12px; }

    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 34px; }
    .evidence-table { display: block; width: 100%; max-width: 100%; margin: 20px 0; overflow-x: auto; border-collapse: collapse; font-size: 12px; }
    .evidence-table th, .evidence-table td { padding: 10px 11px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    .evidence-table th { color: var(--muted); background: #efeee8; font-weight: 650; white-space: nowrap; }
    .evidence-table td:not(:first-child), .evidence-table th:not(:first-child) { font-variant-numeric: tabular-nums; }
    .status { font: 700 10px/1.2 var(--mono); letter-spacing: .05em; text-transform: uppercase; }
    .status.positive { color: var(--accent); }
    .status.mixed { color: var(--warning); }
    .status.negative { color: var(--danger); }

    .claim-grid { display: grid; grid-template-columns: 1.2fr .8fr; gap: 36px; margin-top: 26px; }
    .claim-column { border-top: 3px solid var(--accent); }
    .claim-column.no { border-top-color: var(--danger); }
    .claim-item { padding: 16px 0; border-bottom: 1px solid var(--line); }
    .claim-item p { margin: 5px 0 0; color: var(--muted); font-size: 13px; }

    .next-list { counter-reset: next; max-width: 1120px; margin: 26px 0 0; padding: 0; list-style: none; }
    .next-list li { counter-increment: next; display: grid; grid-template-columns: 52px 1fr; gap: 18px; padding: 18px 0; border-top: 1px solid var(--line); }
    .next-list li::before { content: counter(next, decimal-leading-zero); color: var(--accent); font: 650 18px/1.3 var(--mono); }
    .next-list strong { display: block; margin-bottom: 5px; }
    .next-list span { color: var(--muted); font-size: 13px; }

    details { margin: 16px 0; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
    summary { padding: 12px 0; cursor: pointer; font-weight: 650; }
    .provenance { padding: 0 0 18px; color: var(--muted); font: 11px/1.65 var(--mono); overflow-wrap: anywhere; }

    .toast { position: fixed; right: 22px; bottom: 22px; z-index: 40; padding: 10px 13px; color: #fff; background: var(--ink); font-size: 12px; opacity: 0; pointer-events: none; transform: translateY(8px); transition: opacity .2s ease, transform .2s ease; }
    .toast.show { opacity: 1; transform: translateY(0); }
    :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

    @media (max-width: 980px) {
      .shell { width: min(100% - 18px, 1500px); margin-top: 9px; }
      .hero { grid-template-columns: 1fr; gap: 34px; padding: 38px 28px; }
      main { padding: 0 28px 46px; }
      .nav { padding-inline: 18px; }
      .lab-controls { grid-template-columns: 1fr 1fr; }
      .search-wrap { grid-column: 1 / -1; grid-row: 2; }
      .stepper { justify-self: end; }
      .lab-body { grid-template-columns: 1fr; }
      .seed-rail { padding: 14px 0; border-right: 0; border-bottom: 1px solid var(--line); }
      .seed-list { grid-template-columns: repeat(5, minmax(0, 1fr)); max-height: 210px; }
      .sample-panel { padding: 24px 0 8px; }
      .sample-meta { grid-template-columns: repeat(2, 1fr); }
      .sample-meta > div:nth-child(2) { border-right: 0; }
      .sample-meta > div:nth-child(-n+2) { border-bottom: 1px solid var(--line); }
      .flow { grid-template-columns: 1fr; }
      .flow-step { min-height: 0; border-right: 0; border-bottom: 1px solid var(--line); }
      .flow-step::after { display: none; }
      .two-col, .claim-grid { grid-template-columns: 1fr; }
    }

    @media (max-width: 640px) {
      .hero { padding: 32px 18px; }
      main { padding: 0 18px 36px; }
      section { padding-top: 48px; }
      .lab-controls { grid-template-columns: 1fr; align-items: stretch; }
      .search-wrap { grid-column: auto; grid-row: auto; }
      .stepper { justify-self: stretch; }
      .stepper button { flex: 1; }
      .seed-list { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .sample-head { grid-template-columns: 1fr; }
      .sample-head .icon-button { justify-self: start; }
      .sample-meta, .audit-grid, .metric-strip { grid-template-columns: 1fr; }
      .sample-meta > div, .sample-meta > div:nth-child(2), .metric { border-right: 0; border-bottom: 1px solid var(--line); }
      .sample-meta > div:last-child, .metric:last-child { border-bottom: 0; }
      .audit-row { grid-template-columns: 1fr; gap: 3px; }
      .trace-box { padding: 16px; font-size: 11px; }
      .chart-head { align-items: start; flex-direction: column; }
      .next-list li { grid-template-columns: 38px 1fr; }
    }

    @media (prefers-reduced-motion: reduce) {
      html { scroll-behavior: auto; }
      *, *::before, *::after { transition-duration: .01ms !important; animation-duration: .01ms !important; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="hero">
      <div>
        <p class="eyebrow">Mechanism audit · Qwen3-8B · 2026-08-27</p>
        <h1>Native-thinking<br>Internal Counter</h1>
        <p class="hero-dek">先看冻结样本，再看机制。报告收录 N=3 与 N=10 各 30 个 first-pass、无显式 index 的 seed；随后区分“可读、可 steering 的 count-like state”和“尚未成立的 memoryless +1 recurrence”。</p>
      </div>
      <aside class="hero-aside" aria-label="report summary">
        <div class="hero-stat"><strong>60</strong><span>完整 cohort records<br>每个 N：20 discovery + 10 confirmation</span></div>
        <div class="hero-stat"><strong>L19</strong><span>逐层单点 steering 的共同峰值<br>不是全层同时注入</span></div>
        <div class="hero-stat"><strong>0.215</strong><span>N=10 L19 对照校正后的期望 count 位移<br>单位是 count，不是 accuracy</span></div>
        <div class="hero-stat"><strong>43/60</strong><span>held-out item-span patch 后首个已知 city 跟随 donor successor<br>L16 k=6 复核为 16/20</span></div>
      </aside>
    </header>

    <nav class="nav" aria-label="报告目录">
      <a href="#samples">01 样本展示板</a>
      <a href="#geometry">02 Geometry steering</a>
      <a href="#continued">03 Continued counting</a>
      <a href="#natural-transplant">04 Natural transplant</a>
      <a href="#results">05 实验结果</a>
      <a href="#claims">06 Claim 边界</a>
      <a href="#next">07 下一步</a>
      <a href="#audit">08 审计</a>
    </nav>

    <main>
      <section id="samples">
        <p class="section-kicker">01 · Frozen cohort browser</p>
        <h2>先把 30 个 seed 的具体内容摊开</h2>
        <p class="lead">“30 个 seed”按正式设计是每个 N 各 30 个，因此展示板共含 60 条记录。机制实验只使用已冻结的 whole-token <code>t*</code> 前缀；完整输出里 <code>t*</code> 之后的 recap/rethink 仅供审计，未进入机制 context。</p>
        <div class="callout warning"><strong>split 口径：</strong>使用 <code>noindex_n*_cohort.split</code> / t* context split，不使用复用旧 generation 时遗留的顶层 <code>split</code>。N=10 seed 1359 是已知例子：正式为 discovery。</div>

        <div class="seed-lab" id="seedLab">
          <div class="lab-controls">
            <div>
              <span class="control-label">Cohort</span>
              <div class="segmented" id="nToggle" aria-label="选择 N">
                <button type="button" data-n="3" class="active">N = 3</button>
                <button type="button" data-n="10">N = 10</button>
              </div>
            </div>
            <div>
              <span class="control-label">Split</span>
              <div class="segmented" id="splitToggle" aria-label="选择 split">
                <button type="button" data-split="all" class="active">全部</button>
                <button type="button" data-split="discovery">Discovery</button>
                <button type="button" data-split="confirmation">Confirmation</button>
              </div>
            </div>
            <label class="search-wrap">
              <span class="control-label">Search seed / city</span>
              <input id="seedSearch" type="search" inputmode="numeric" placeholder="例如 1307 或 Geneva">
            </label>
            <div>
              <span class="control-label">Navigate</span>
              <div class="stepper">
                <button class="icon-button" id="prevSeed" type="button" aria-label="上一个 seed">←</button>
                <button class="icon-button" id="nextSeed" type="button" aria-label="下一个 seed">→</button>
              </div>
            </div>
          </div>
          <div class="lab-body">
            <aside class="seed-rail">
              <p class="seed-count" id="seedCount"></p>
              <div class="seed-list" id="seedList"></div>
            </aside>
            <article class="sample-panel" id="samplePanel"></article>
          </div>
        </div>
      </section>

      <section id="geometry">
        <p class="section-kicker">02 · Geometry steering</p>
        <h2>为什么“全局 steering”只有 0.2：因为它其实不是全局注入</h2>
        <p class="lead">目录名 <code>all_layer_plus1</code> 表示“把 36 层都逐层扫描一遍”。每个 trial 只在一个 layer、一个 item-closing endpoint 注入一次 discovery centroid delta；没有把 L0–L35 同时加上，也没有在整个 item span 或 K/V cache 上持续施加。</p>

        <div class="metric-strip">
          <div class="metric"><p class="metric-label">N=3 peak</p><strong>+0.622</strong><p>L19 contrast，95% CI [0.471, 0.767]</p></div>
          <div class="metric"><p class="metric-label">What 0.215 means</p><strong>0.207 − ½(−0.024 + 0.008)</strong><p>N=10 L19：real expected-count shift 减 opposite / orthogonal controls 的均值。</p></div>
          <div class="metric"><p class="metric-label">N=10 peak</p><strong>+0.215</strong><p>L19 contrast，95% CI [0.107, 0.334]</p></div>
        </div>

        <div class="callout"><strong>它不是 20% accuracy。</strong>单位是 18-way candidate distribution 的期望 count 位移。理想的 hard +1 会移动 1，但 <code>alpha=1</code> 只是“一条自然 centroid delta 的激活尺度”，没有被校准为一个完整行为 count。</div>

        <div class="chart-shell">
          <div class="chart-head">
            <div>
              <p class="panel-label">Single-layer endpoint intervention</p>
              <h3>36 层 steering profile</h3>
            </div>
            <div>
              <div class="segmented" id="geometryToggle" aria-label="Geometry chart cohort">
                <button type="button" data-geometry-n="3">N = 3</button>
                <button type="button" data-geometry-n="10" class="active">N = 10</button>
              </div>
              <div class="chart-legend" aria-label="图例">
                <span><i class="legend-dot" style="background:#1f6f5f"></i>real +1</span>
                <span><i class="legend-dot" style="background:#171b18"></i>control-adjusted</span>
                <span><i class="legend-dot" style="background:#a95d2b"></i>opposite</span>
                <span><i class="legend-dot" style="background:#8b938d"></i>orthogonal</span>
              </div>
            </div>
          </div>
          <div id="geometryChart" role="img" aria-label="每层 Geometry steering 的期望计数位移"></div>
          <p class="chart-note">每个 layer 值先在 receiver occurrences 上取均值：N=3 使用 k=1,2；N=10 使用 k=2,5,8。峰值层来自同一 post-hoc scan，因此还不能当作 fresh held-out layer confirmation。</p>
        </div>

        <div class="two-col">
          <div>
            <h3>为什么 N=10 明显弱于 N=3</h3>
            <p>不是 NCC 失效，而是 <strong>可分离不等于局部 state 对行为充分</strong>。当前其实已对每个 receiver 使用各自的 <code>μ(k+1)−μ(k)</code>，不是同一条全局向量；但 L19 contrast 随 k 明显衰减：k=2 为 0.449，k=5 为 0.162，k=8 仅 0.035，三者汇总才是 0.215。N=10 的 curved / hairpin manifold 与更强的高-k history dependence 都会让 late state 更难被一次 endpoint edit 推过决策边界。</p>
            <p>此外，count information 同时依赖 item span 与 marker-indexed K/V history。一次 residual endpoint 注入会被后续层重新 contextualize；晚期可读 state 并不是已证明的独立 register。</p>
          </div>
          <div>
            <h3>怎样才算真正的“global”复核</h3>
            <p>应在 discovery 上联合选择 L17–23 band、每层 dose normalization 与 alpha grid，再固定到 fresh confirmation。方向仍保持按 receiver k 定义的 local tangent <code>μ(k+1)−μ(k)</code>，并用 leave-one-seed-out centroid 检查方向是否被少数 seed 主导。</p>
            <p>如果多层联合 steering 明显放大 adoption，说明当前 0.2 主要是单点剂量不足；若仍弱，则更支持“count readout 依赖 distributed K/V context，而不是 residual geometry 自身充分”。</p>
          </div>
        </div>
      </section>

      <section id="continued">
        <p class="section-kicker">03 · Continued counting</p>
        <h2>Continued counting 到底怎么做</h2>
        <p class="lead">这是 paper-style “last-k source → first-k target” transplant，并补了 immediate early-stop readout。它检验 donor-dependent continuation，而不是只问 source state 能不能被 probe 解码。</p>

        <div class="flow" aria-label="continued counting flow">
          <div class="flow-step"><p class="panel-label">Step 1</p><h4>选 donor 结尾</h4><code>Nₛ, k</code><p>取 source occurrence <code>Nₛ−k+1 … Nₛ</code> 的完整 block-input states。</p></div>
          <div class="flow-step"><p class="panel-label">Step 2</p><h4>对齐到 target 开头</h4><code>1 … k</code><p>按 marker / closing / full region 的 token alignment，把 source last-k 写到 target first-k。</p></div>
          <div class="flow-step"><p class="panel-label">Step 3</p><h4>全层替换</h4><code>all 36 block inputs</code><p>与 Geometry 不同：Continued 对选中的 span 在所有 decoder-block input layers 做 replacement。</p></div>
          <div class="flow-step"><p class="panel-label">Step 4</p><h4>保留 target 后缀</h4><code>k+1 … Nₜ</code><p>后续 target items 与其 K/V history 完整存在；它们可能重建 target 的自然 progress。</p></div>
          <div class="flow-step"><p class="panel-label">Step 5</p><h4>读取预期 count</h4><code>r̃ = Nₛ + Nₜ − k</code><p>同时在 hop 1 / hop 2 立即查询，预期分别为 <code>Nₛ+1</code>、<code>Nₛ+2</code>。</p></div>
        </div>

        <div class="callout warning"><strong>例子：</strong>source end <code>Nₛ=6</code>、target <code>Nₜ=10</code>、<code>k=1</code>。把 source 的第 6 项 state 写入 target 第 1 项。如果 counter 从 donor 继续，target 第 2 项后的立即 count 应为 7，完整 target 末端应为 15。</div>

        <table class="evidence-table">
          <thead><tr><th>Cohort</th><th>source end</th><th>readout</th><th>candidate</th><th>greedy</th><th>结论</th></tr></thead>
          <tbody>
            <tr><td>N=3 full, k=1</td><td>2</td><td>hop 1</td><td>0.70</td><td>0.50</td><td class="status mixed">transient</td></tr>
            <tr><td>N=3 full, k=1</td><td>3</td><td>hop 1</td><td>0.30</td><td>0.60</td><td class="status mixed">transient</td></tr>
            <tr><td>N=10 full, k=1</td><td>3</td><td>hop 1</td><td>0.20</td><td>0.20</td><td class="status negative">weak</td></tr>
            <tr><td>N=10 full, k=1</td><td>6</td><td>hop 1</td><td>0.10</td><td>0.20</td><td class="status negative">weak</td></tr>
            <tr><td>N=10 full, k=1</td><td>9</td><td>hop 1</td><td>0.10</td><td>0.10</td><td class="status negative">weak</td></tr>
            <tr><td>N=10</td><td>3 / 6 / 9</td><td>hop 2 / final</td><td>≈0</td><td>≈0</td><td class="status negative">no maintained recurrence</td></tr>
          </tbody>
        </table>

        <div class="two-col">
          <div><h3>为什么做不出来</h3><p>截断位置确实解释了一部分：原来的 final readout 会把短暂 hop-1 effect 淹没。但改成 hop-1 immediate query 后，N=10 仍只有 0.1–0.2 adoption，hop 2 又归零。</p><p>更核心的是，replacement 只改 residual spans；target 后续语义 event 与 marker-indexed K/V entries 仍在，会把 progress state 重建为 target 自己的轨迹。去掉 norm rescaling、改到 post-item timing 都没有救回来。</p></div>
          <div><h3>它目前证明什么</h3><p>full-item patch 比 closing / post-item 更强，说明 causal information 分布在 event 内容而非一个 boundary cell。N=3 存在短暂 one-hop influence，但没有证据表明 donor offset 被下一 item 稳定保留。</p><p>所以结果支持 <strong>context-dependent reconstruction / reset</strong>，不支持 memoryless <code>S(k)+new item→S(k+1)</code>。</p></div>
        </div>
      </section>

      <section id="natural-transplant">
        <p class="section-kicker">04 · Natural same-site transplant</p>
        <h2>只看下一项：能否跳过 N<sub>k</sub>，直接从 N<sub>k+1</sub> 开始</h2>
        <p class="lead">这里不看后续 recap、最终总数或停止时间。对每个相邻 pair 取自然 step <code>j=k−1</code> 与自然 step <code>k</code>，在同一 trace 内选择完全相同的 item-closing token；再从 donor 的早期 prompt filler 删除精确 token 数，使 donor 和 receiver commit 落在相同绝对位置。最后只把 L31 的单点 post-block state 从 donor 写入 receiver。</p>

        <div class="flow" aria-label="natural same-site transplant flow">
          <div class="flow-step"><p class="panel-label">Receiver</p><h4>自然 step j</h4><code>next = N<sub>k</sub></code><p>self control 必须首先生成 receiver successor。</p></div>
          <div class="flow-step"><p class="panel-label">Donor</p><h4>自然 step k</h4><code>next = N<sub>k+1</sub></code><p>native donor 必须在完整 transition likelihood 上选择 donor successor。</p></div>
          <div class="flow-step"><p class="panel-label">Match</p><h4>token + position</h4><code>p<sub>j</sub> = p<sub>k</sub></code><p>删除只发生在 prompt filler，不碰 records、special tokens 或 reasoning。</p></div>
          <div class="flow-step"><p class="panel-label">Patch</p><h4>single L31 state</h4><code>h<sub>k</sub> → h<sub>j</sub></code><p>receiver 的其余 KV/history 全部保留。</p></div>
          <div class="flow-step"><p class="panel-label">Primary</p><h4>first known city</h4><code>N<sub>k+1</sub>?</code><p>只认 completion 的第一个已知 city；晚期回顾不计。</p></div>
        </div>

        <table class="evidence-table">
          <thead><tr><th>donor k</th><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th><th>pooled</th></tr></thead>
          <tbody>
            <tr><td><strong>统一 5-seed panel</strong></td><td>0/5</td><td>0/5</td><td>1/5</td><td>1/5</td><td>1/5</td><td>1/5</td><td>1/5</td><td>0/5</td><td><strong>5/40</strong></td></tr>
            <tr><td><strong>plain-period seed1267</strong></td><td>0</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td><strong>5/8</strong></td></tr>
            <tr><td><strong>grammar-matched seed1791</strong></td><td>0</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td><strong>6/8</strong></td></tr>
          </tbody>
        </table>

        <div class="metric-strip">
          <div class="metric"><p class="metric-label">Geometry gate</p><strong>40/40</strong><p>初始 panel 全部 matched，self/native donor gate 全过。</p></div>
          <div class="metric"><p class="metric-label">Plain-period panel</p><strong>11/16</strong><p>k=4–8 为 10/10；k=2–3 为 0/4；k=9 为 1/2。</p></div>
          <div class="metric"><p class="metric-label">Other initial grammars</p><strong>0/32</strong><p>强烈提示 single-site sufficiency 受 commit grammar / trace context 调制。</p></div>
        </div>

        <div class="callout"><strong>此前 held-out k=6 panel 的 post-hoc 交叉检查：</strong>plain-period seeds 是 2/2 first-successor skip、2/2 candidate argmax；其余 6 个 whitespace 与 2 个 quote-closing+newline seeds 合计 0/8。连同 discovery cells，现有 k=6 是 period 4/4、其他表面 0/12。该分层并非预注册，但说明关联不只来自1267/1791两条 discovery trace。</div>

        <div class="two-col">
          <div><h3>为什么这是更直接的 causal result</h3><p>在 seed1267 和 seed1791 的所有 11 个成功 cell 中，receiver control 先输出 <code>N<sub>k</sub></code>，transplant 则第一句话直接从 <code>N<sub>k+1</sub></code> 开始；同时 donor transition 都成为 candidate argmax。人工检查排除了 recap-only match。</p><p>例如 seed1791 的 k=6：receiver 从 Warsaw (N6) 开始，写入 step-6 donor state 后从 Shanghai (N7) 开始。它检验的是 commit state 对下一次 retrieval/emission 的局部充分性。</p></div>
          <div><h3>为什么仍不能叫普适 arithmetic counter</h3><p>初始 5-seed panel 的五次成功全部来自 seed1267。之后先用 outcome-blind 20-seed tokenizer audit 定义 all-period grammar，再测试唯一另一个同型 seed1791；它确实复现，但仍只有两条完整 k-grid trace。grammar 假设源于观察1267，因此不赋 confirmatory p-value。</p><p>更具体的机制解释是：普通句号可能是真正稳定的 semantic event-commit/write site；space 或 quote-closing token 的 null 也可能是 site localization 失败。最稳妥的说法仍是 grammar/context-dependent successor controller，而不是所有 trace 都有一个可独立移植的 counter cell。</p></div>
        </div>

        <div class="callout warning"><strong>为什么原来的 non-period null 不再成立：</strong>tokenizer audit 发现许多 whitespace step 以 score 数字结束；旧 same-token 规则只能选到数字前的空格，尚未到 event completion。把 site 移到 item-final 后，whitespace 的 L16 单-token paired shift 在 5/5 seeds 上为正，mean +16.15。</div>

        <h3>系统 scope × layer sweep：7/60 不是最终结果</h3>
        <p class="lead">在 discovery20 上只用 k=6 双向 cells 扫 L0–L35；layer 选择完全不看 attention 或 generation，而以 paired donor-directed transition log-odds 为准：要求两个方向的 median 都为正，并选择达到 peak seed-median 95% 的最早层。随后冻结 layer，在 confirmation10 上测 k∈{4,6,8} 双向，共 60 cells / scope。主报告只给 effect size、分位数与行为翻转，不把 p-value 当主证据。</p>

        <div class="flow" aria-label="patch scope comparison">
          <div class="flow-step"><p class="panel-label">Item end · w1</p><h4>真实注册 endpoint</h4><p>只 patch 每个 item 的实际终止 token，而不是固定去猜句号或空格。它是最窄、最接近旧 single-token 实验的版本。</p></div>
          <div class="flow-step"><p class="panel-label">Event tail · w4</p><h4>终点前四个 token</h4><p>不含 city 名，但通常包含 score numeral 与 closing punctuation；检验分布式 commit tail。</p></div>
          <div class="flow-step"><p class="panel-label">Item span</p><h4>最大共同末端 span</h4><p>两项等长时 patch 完整 item；不等长时取较短 item 的全部 token，并取较长 item 的等宽末端，不做 hidden-state 插值。</p></div>
        </div>

        <figure class="scope-figure">
          <img src="data:image/png;base64,__PATCH_SCOPE_LAYER_PLOT__" alt="三种 patch scope 的逐层 donor-directed transition effect；左图为原始 paired log-odds shift，右图按 patch delta norm 归一化">
          <figcaption>Discovery20 layer sweep。左：median seed-mean paired donor-directed transition log-odds shift；右：median 每单位 realized patch-delta norm。圆点为冻结层；每条线先在 seed 内合并双向 repeated cells，避免把 cell 当作独立样本。</figcaption>
        </figure>

        <table class="evidence-table">
          <thead><tr><th>Discovery scope</th><th>冻结层</th><th>Median Δ log-odds</th><th>Q10–Q90</th><th>正向 cells</th><th>Donor argmax</th><th>Mean ‖Δh‖</th><th>Mean Δ/‖Δh‖</th></tr></thead>
          <tbody>
            <tr><td>item end, w1</td><td>L26</td><td>+4.30</td><td>−0.02 – +19.85</td><td>33/40</td><td>3/40</td><td>112.98</td><td>+0.078</td></tr>
            <tr><td>event tail, w4</td><td>L0</td><td>+23.61</td><td>+11.24 – +33.87</td><td>40/40</td><td>1/40</td><td>6.01</td><td><strong>+4.019</strong></td></tr>
            <tr><td>max-common item span</td><td>L0</td><td><strong>+58.03</strong></td><td>+45.72 – +72.93</td><td>40/40</td><td>37/40</td><td>21.07</td><td>+2.911</td></tr>
          </tbody>
        </table>

        <div class="callout"><strong>layer sweep 的核心：</strong>full item-span 在 raw behavior-relevant effect 上远强于单 endpoint；四-token event tail 则在单位 patch norm 上最有效。item-end L26 虽有少数极大值，但 median 只有 +4.30、Q10 略低于零，且扰动范数反而最大，说明“一个 closing vector 就是 counter”不是稳定解释。</div>

        <table class="evidence-table">
          <thead><tr><th>Held-out scope</th><th>Cells</th><th>Median Δ log-odds</th><th>Q10–Q90</th><th>Mean attention Δ</th><th>Donor argmax</th><th>首个已知 city = donor successor</th><th>覆盖 seeds</th></tr></thead>
          <tbody>
            <tr><td>item end · L26</td><td>60</td><td>+4.44</td><td>+0.24 – +57.00</td><td>+2.71</td><td>8/60</td><td>10/60</td><td>2/10</td></tr>
            <tr><td>event tail · L0</td><td>60</td><td>+21.98</td><td>+10.30 – +46.28</td><td>+1.29</td><td>6/60</td><td>15/60</td><td>8/10</td></tr>
            <tr><td><strong>item span · L0</strong></td><td>60</td><td><strong>+63.85</strong></td><td>+47.28 – +89.67</td><td>+2.31</td><td><strong>60/60</strong></td><td><strong>43/60</strong></td><td><strong>10/10</strong></td></tr>
            <tr><td>item span · L16 · k=6 only</td><td>20</td><td>+51.15</td><td>+40.61 – +67.46</td><td>+2.69</td><td>17/20</td><td>16/20</td><td>10/10</td></tr>
          </tbody>
        </table>

        <div class="metric-strip">
          <div class="metric"><p class="metric-label">Item span behavior</p><strong>43/60</strong><p>L0：forward 25/30，backward 18/30；10/10 seeds 至少一次翻转。</p></div>
          <div class="metric"><p class="metric-label">Not just layer 0</p><strong>16/20</strong><p>L16 k=6：forward 6/10，backward 10/10；20/20 likelihood 与 attention shifts 为正。</p></div>
          <div class="metric"><p class="metric-label">Efficient narrow carrier</p><strong>15/60</strong><p>event-tail L0 首个 city 翻转；比历史 L16 tail 的 7/60 高，但仍远弱于 item span。</p></div>
        </div>

        <div class="two-col">
          <div><h3>生成核验</h3><p>Primary endpoint 是 patch 后出现的<strong>第一个 gold city</strong>，不是要求模型另起一个 Markdown bullet。43 个 item-span L0 命中已逐条检查：40 个 donor city 在 completion 前 40 个字符内出现，另 3 个在短暂 repair preamble 后出现；43/43 都先进入 donor successor continuation，0 个只是后续 recap 命中。</p><p>严格 bullet-line parser 只有 5/60，因为多数 native trace 继续原来的 quoted / comma-separated enumeration；这反映表面 grammar，不应把 43/60 错降成 5/60。</p></div>
          <div><h3>它证明什么、仍混入什么</h3><p>item-span 结果证明：在无显式 index 的原生轨迹里，前一 item 的 contextual event span 足以把下一次 continuation 从 receiver successor 改到 donor successor，而且 L16 仍可复现，不是纯 L0 假象。</p><p>但 span 内含 donor city、score 和 punctuation，因此这是<strong>完整 progress/content state 的因果充分性上界</strong>，不能单独归因于抽象 count。更窄的 event tail 不含 city 名、跨 k/方向仍系统地移动 likelihood / attention，并有 15/60 行为翻转；它是更 counter-like、但仍含 score numeral 的证据。</p></div>
        </div>

        <div class="callout warning"><strong>如何理解之前的 end-token patch：</strong>旧 L31 same-token 结果是统一五-seed panel 5/40；其阳性集中在 plain-period 两条 trace（11/16），held-out k=6 的 plain-period 为 2/2、其他 grammar 为 0/8。现在对真实 item endpoint 做全层 sweep 后，冻结 L26 在完整 held-out panel 中仍只有 10/60 first-city adoption，而且命中集中于 2/10 seeds。结论不是“end token 完全没用”，而是它高度 grammar/seed dependent；系统效应主要分布在更宽的 item span。</div>
      </section>

      <section id="results">
        <p class="section-kicker">05 · Evidence map</p>
        <h2>目前实验设计与结果</h2>
        <p class="lead">下面把当前 first-pass no-index 移植实验与此前 marker-ledger 证据放在同一张图里。表中的“阳性”只表示对应 estimand 成立，不自动升级为 arithmetic counter。</p>

        <table class="evidence-table">
          <thead><tr><th>实验</th><th>设计</th><th>N=3</th><th>N=10 / 主要结果</th><th>判定</th></tr></thead>
          <tbody>
            <tr><td><strong>PCA / NCC</strong></td><td>discovery20 fit，confirmation10 eval；closing / centered states</td><td>L3 起多层 NCC 1.00</td><td>L19 NCC 0.82；PC1/count 0.33；hairpin</td><td class="status positive">count-structured geometry</td></tr>
            <tr><td><strong>CountScope</strong></td><td>full-item all-layer transplant 到 matched one-item receiver</td><td>k=1/2/3 candidate 0.90/0.70/1.00</td><td>k=1–4 尚可；k≥5 基本失效</td><td class="status mixed">readable, not standalone</td></tr>
            <tr><td><strong>Continued</strong></td><td>source last-k → target first-k；final + hop1/2</td><td>hop1 短暂 0.3–0.7 candidate</td><td>hop1 ≤0.20；hop2/final ≈0</td><td class="status negative">no stable +1</td></tr>
            <tr><td><strong>Geometry</strong></td><td>每层单 endpoint +1 delta；opposite / orthogonal controls</td><td>L19 +0.622 [0.471,0.767]</td><td>L19 +0.215 [0.107,0.334]</td><td class="status positive">local causal direction</td></tr>
            <tr><td><strong>Natural item endpoint</strong></td><td>单 token；36-layer discovery sweep 后冻结 L26；k=4/6/8 双向</td><td>—</td><td>held-out median Δlog-odds +4.44；argmax 8/60；first city 10/60，集中于 2/10 seeds。旧 L31 plain-period panel 为 11/16</td><td class="status mixed">brittle / grammar-dependent</td></tr>
            <tr><td><strong>Natural event tail</strong></td><td>width 4；36-layer discovery sweep 后冻结 L0；k=4/6/8 双向</td><td>—</td><td>held-out median Δlog-odds +21.98；attention 58/60 donor-directed；first city 15/60，覆盖 8/10 seeds</td><td class="status positive">efficient narrow routing carrier</td></tr>
            <tr><td><strong>Natural item span</strong></td><td>endpoint-aligned max-common span；冻结 L0；另做 L16 k=6 对照</td><td>—</td><td>L0：argmax 60/60，first city 43/60；L16：17/20、16/20；均覆盖 10/10 seeds</td><td class="status positive">strong distributed event-state sufficiency</td></tr>
            <tr><td><strong>Separator dose</strong></td><td>把 later events 逐个 collapse 到第一可用 event state</td><td>marker-only 近零；full 强</td><td>per-event slope：marker −0.125，closing −0.219，full −0.690</td><td class="status mixed">distributed event carrier</td></tr>
            <tr><td><strong>Maximum-count</strong></td><td>source last-k → target last-k；检验 max(Nₛ,Nₜ−k)</td><td>短程强</td><td>donor-dominant candidate 0.13–0.30；target−k branch confounded</td><td class="status negative">no general max operator</td></tr>
            <tr><td><strong>Marker K/V</strong></td><td>exact cache splice；K-only / V-only / layer band</td><td colspan="2">Qwen N=10 prior evidence：marker K/V 0.835；V 0.500；K 0.276；L20–23 0.540</td><td class="status positive">event-memory substrate</td></tr>
            <tr><td><strong>Recurrence operator scan</strong></td><td>wide donor/receiver operator family</td><td colspan="2">reset 97.08%；target +1 0.625%；successful first-stage local arms 仍 0 next +1</td><td class="status negative">reset dominates</td></tr>
          </tbody>
        </table>

        <div class="callout"><strong>整合解释：</strong>native states 在原生 history 中可被分开、可被局部 steering；单 endpoint 的行为充分性脆弱，而 endpoint-aligned item span 在全 grammar held-out cohort 上稳定地改写 candidate ranking 与实际 continuation。四-token tail 以更小范数产生一致的 likelihood / attention routing，但行为翻转较少。与此同时，continued recurrence 仍不稳定。最小共同机制是 marker-indexed / distributed event memory 被 contextual aggregation 成一个 causal progress/content controller，而不是独立 arithmetic register。</div>
      </section>

      <section id="claims">
        <p class="section-kicker">06 · Claim boundary</p>
        <h2>论文里现在可以怎么说</h2>
        <div class="claim-grid">
          <div class="claim-column">
            <h3>可以说</h3>
            <div class="claim-item"><strong>存在 count-structured contextual state。</strong><p>first-pass no-index enumeration 的 native item states 在 held-out seeds 上呈有序、非线性的 count/progress geometry。</p></div>
            <div class="claim-item"><strong>该 state 有局部因果效力。</strong><p>CountScope 在早期 k 可翻译 donor count；L17–23 的 +1 direction 对 opposite / orthogonal controls 有 specificity。</p></div>
            <div class="claim-item"><strong>自然 commit state 可直接控制 successor。</strong><p>在两个 all-period traces 中，L31 单点 transplant 于 k=4–8 达到 10/10 first-successor skip，并在 k=9 达到 1/2；每次都同时翻转 candidate argmax。此前 held-out k=6 panel 的 period cells 也为 2/2，而其他表面为 0/8（post-hoc stratification）。</p></div>
            <div class="claim-item"><strong>自然 item-span state 对 successor 有强因果控制。</strong><p>layer 只在 discovery 上选择后，held-out L0 item-span 使 donor successor 成为 candidate argmax 60/60，并让实际 continuation 的首个 gold city 在 43/60 cells 跟随 donor；10/10 seeds 均出现。固定到更深的 L16、只复核 k=6，仍为 17/20 argmax、16/20 continuation adoption。</p></div>
            <div class="claim-item"><strong>更窄的 event tail 可跨 grammar 控制 routing。</strong><p>冻结 L0 width-4 后，held-out median likelihood shift 为 +21.98，attention 在 58/60 cells donor-directed，首个 city 在 15/60 cells 改为 donor successor，覆盖 8/10 seeds。其单位 patch-norm effect 高于完整 item span。</p></div>
            <div class="claim-item"><strong>历史更像分布式 event memory。</strong><p>marker K/V、closing 和 full-event dose 共同表明 count information 不只在 hidden-state endpoint，也存在可被晚期读取的 event-indexed cache history。</p></div>
            <div class="claim-item"><strong>更合适的机制名。</strong><p><em>marker-indexed distributed event memory with a late count-like progress readout</em>，或 <em>context-dependent counting controller</em>。</p></div>
          </div>
          <div class="claim-column no">
            <h3>现在不能说</h3>
            <div class="claim-item"><strong>显式 arithmetic register。</strong><p>没有证据支持隐藏状态中逐 item 执行 <code>counter ← counter + 1</code>。</p></div>
            <div class="claim-item"><strong>一个 context-invariant counter cell。</strong><p>N=10 CountScope 在 k≥5 失去 sufficiency，高 NCC 不能替代 transplant 成功。</p></div>
            <div class="claim-item"><strong>content-free event-tail register。</strong><p>width-4 span 不含 city 名，但包含不同 score numeral；目前尚未用 same-progress/different-score control 排除 event-content contribution。</p></div>
            <div class="claim-item"><strong>item-span 效应等于抽象 count。</strong><p>完整或 max-common item span 同时携带 city、score、syntax 与 progress；43/60 证明 distributed event-state sufficiency，不隔离 count component。</p></div>
            <div class="claim-item"><strong>稳定 memoryless recurrence。</strong><p>Continued counting 的 donor effect 在 hop 1 已弱，hop 2 / final 消失。</p></div>
            <div class="claim-item"><strong>普适 maximum 或 separator shortcut。</strong><p>maximum 的高分支与 target-history confound 重合；marker residual 远弱于 full event。</p></div>
          </div>
        </div>

        <div class="callout"><strong>推荐英文：</strong>During unindexed first-pass enumeration, Qwen3-8B forms a distributed causal event state that controls which item is continued next. After selecting the intervention layer on discovery data, transplanting an endpoint-aligned donor item span made the donor-implied successor the top-scoring candidate in 60/60 held-out cells and changed the first generated gold city in 43/60 cells; a deeper L16 replication at k=6 retained 16/20 behavioral transfers. A four-token event tail produced smaller but more norm-efficient donor-directed likelihood and attention shifts, with 15/60 generated transfers. These results support a context-dependent progress controller, but item content, score-bearing tails, and failed continued-counting interventions do not establish a content-free memoryless <code>+1</code> register.</div>
      </section>

      <section id="next">
        <p class="section-kicker">07 · Decision-focused next experiments</p>
        <h2>还需要做什么</h2>
        <ol class="next-list">
          <li><div><strong>真正的 multi-layer Geometry steering</strong><span>Discovery 扫 L17–23 的 joint band、per-layer dose normalization、alpha ∈ {0.25,0.5,1,2,4}；继续使用 per-k local tangent，并分别报告 k=2/5/8，避免 pooled mean 掩盖高-k 衰减。冻结 band/dose 后用 fresh confirmation，primary endpoint 用 candidate argmax 与 greedy adoption。</span></div></li>
          <li><div><strong>Continued 的 state+history 版本</strong><span>不仅换 residual span，还做 donor event 的 exact K/V splice 或 residual+K/V 联合 transplant；保持同一语义 carrier，分别在 hop1、hop2 立即生成。若这仍不保留 donor offset，memoryless counter 基本可排除。</span></div></li>
          <li><div><strong>把 receiver-history mismatch 变成受控变量</strong><span>CountScope 设计 1-item、matched-k-history、same-seed blank-history 三类 receiver。若大 k 只在 matched history 恢复，能直接证明 count readout 依赖 distributed context。</span></div></li>
          <li><div><strong>Fresh cohort 复现 L19 / L17–23</strong><span>当前逐层峰值来自 post-hoc scan。预注册固定层带、固定 metric、固定 per-k panel，再收一组全新 seeds；避免把同一 confirmation cohort 同时用于定位和确认。</span></div></li>
          <li><div><strong>same-progress / different-score control</strong><span>event-tail width-4 span 包含 score numeral。构造相同 k、相同 grammar、不同 score surface 的 donor，或把 score-value direction 投影掉；若 donor-directed routing 保留，才能把 causal carrier 更明确地归因于 progress 而非 event content。</span></div></li>
          <li><div><strong>item-span advantage 的 matched controls</strong><span>当前已系统比较 endpoint、四-token tail 与 max-common item span。下一步固定 L16，加入等宽 shuffled item、same-k cross-seed item、within-item token permutation 和等范数随机 span；这样才能把“完整 event 内容使 continuation 可复制”与“ordinal progress state”分开。</span></div></li>
          <li><div><strong>Separator / maximum 只做机制排歧</strong><span>Separator 优先复现 full-event dose 与 marker+closing K/V synergy；maximum 必须只报告 donor-dominant cells，并加入“删除最后 k 个 target events”的显式 confound control。</span></div></li>
        </ol>
      </section>

      <section id="audit">
        <p class="section-kicker">08 · Reproducibility</p>
        <h2>数据与口径审计</h2>
        <p class="lead">本报告是离线、自包含 HTML；60 条 prompt / trace 内容与交互逻辑均内嵌，不依赖 CDN。展示内容来自正式 selected rows 与冻结的 t* contexts。</p>
        <details open>
          <summary>输入文件与 SHA-256</summary>
          <div class="provenance" id="provenance"></div>
          <div class="provenance">__KGRID_PROVENANCE__<br>__EVENT_TAIL_PROVENANCE__<br>__PATCH_SCOPE_PROVENANCE__</div>
        </details>
        <details>
          <summary>解释等级</summary>
          <div class="provenance">Cohort selection: outcome-blind, 20 discovery + 10 confirmation per N.<br>All-layer steering profile: post-hoc descriptive scan on the reserved cohort; layer peak is not fresh confirmation.<br>PCA/NCC: discovery fit, confirmation projection; within-seed centering is a geometry diagnostic, not a deployable single-state decoder.<br>Natural k-grid: the initial five-seed panel used the same frozen L31 for all k. The all-period grammar hypothesis was generated after inspecting seed1267; seed1791 was selected outcome-blind from the tokenizer-only 20-seed audit, but the resulting two-trace panel is descriptive rather than a fresh population confirmation. The 2/2 versus 0/8 held-out k=6 grammar split is also post-hoc because the confirmation outcomes already existed when the grammar hypothesis was formed.<br>Historical event-tail assay: site/layer/width were localized on discovery only, then frozen to L16, item-final, width 4 before confirmation; it yielded 7/60 generated transfers.<br>Systematic scope assay: discovery20 used only k=6 in both directions to sweep all 36 layers for three prespecified scopes. Selection used paired transition log-odds only and chose the earliest layer within 95% of the seed-median peak while requiring both directions positive; attention and generation were hidden from selection. Layers were frozen at endpoint L26, tail L0 and item-span L0 before confirmation10 tested k={4,6,8} in both directions. The L16 item-span k=6 run is a labeled robustness comparator, not a second layer-selection step. Item spans retain lexical event content; event tails retain score numerals.<br>Generation endpoint: first gold city after the patch boundary. All 43 L0 item-span hits and all 16 L16 hits were manually reviewed; none was a recap-only false positive. The narrow Markdown-bullet parser is secondary because most native traces continue inline quoted/comma-separated lists.<br>Continued, CountScope, separator and maximum summaries: seed-level evaluation; candidate probability shifts do not equal greedy sufficiency.</div>
        </details>
        <p class="small">Generated: __GENERATED__ · Report schema: niah_native_thinking_internal_counter_report_v1</p>
      </section>
    </main>
  </div>

  <div class="toast" id="toast" role="status" aria-live="polite"></div>
  <script id="sampleData" type="application/json">__SAMPLES__</script>
  <script id="cohortMeta" type="application/json">__COHORT_META__</script>
  <script id="geometryData" type="application/json">__GEOMETRY__</script>
  <script>
    (() => {
      const samples = JSON.parse(document.getElementById('sampleData').textContent);
      const meta = JSON.parse(document.getElementById('cohortMeta').textContent);
      const geometry = JSON.parse(document.getElementById('geometryData').textContent);
      const state = { n: 3, split: 'all', query: '', seed: samples.find(s => s.n === 3).seed, tab: 'prefix', geometryN: 10 };

      const els = {
        nToggle: document.getElementById('nToggle'), splitToggle: document.getElementById('splitToggle'),
        search: document.getElementById('seedSearch'), list: document.getElementById('seedList'),
        count: document.getElementById('seedCount'), panel: document.getElementById('samplePanel'),
        prev: document.getElementById('prevSeed'), next: document.getElementById('nextSeed'),
        toast: document.getElementById('toast'), geometryToggle: document.getElementById('geometryToggle'),
        geometryChart: document.getElementById('geometryChart'), provenance: document.getElementById('provenance')
      };

      const escapeHtml = (value) => String(value ?? '').replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
      const filtered = () => samples.filter(sample => {
        if (sample.n !== state.n) return false;
        if (state.split !== 'all' && sample.split !== state.split) return false;
        const haystack = [sample.seed, sample.requestId, ...sample.goldRecords.map(r => r.city)].join(' ').toLowerCase();
        return haystack.includes(state.query.trim().toLowerCase());
      });

      function showToast(message) {
        els.toast.textContent = message;
        els.toast.classList.add('show');
        window.clearTimeout(showToast.timer);
        showToast.timer = window.setTimeout(() => els.toast.classList.remove('show'), 1600);
      }

      function highlightedPrefix(sample) {
        const text = sample.mechanismPrefix;
        const spans = [...sample.firstOccurrences]
          .map(item => ({ start: item.offset_char_start, end: item.offset_char_end }))
          .sort((a, b) => a.start - b.start);
        let cursor = 0;
        let output = '';
        for (const span of spans) {
          const start = Math.max(cursor, Math.min(text.length, span.start));
          const end = Math.max(start, Math.min(text.length, span.end));
          output += escapeHtml(text.slice(cursor, start));
          output += '<mark>' + escapeHtml(text.slice(start, end)) + '</mark>';
          cursor = end;
        }
        output += escapeHtml(text.slice(cursor));
        return output;
      }

      function auditMarkup(sample) {
        const values = [
          ['Formal split', sample.split + ' #' + sample.rank],
          ['Prefix-clean gate', sample.prefixClean ? 'PASS' : 'FAIL'],
          ['Global-clean sensitivity', sample.globalClean ? 'PASS' : 'NO — post-t* recap exists'],
          ['Strict full-output no-cue', sample.strictNoExplicitCue ? 'PASS' : 'NO'],
          ['First-pass complete', sample.firstPassComplete ? 'PASS' : 'FAIL'],
          ['Events before t*', sample.preTStarEventCount + ' / expected ' + sample.n],
          ['Repeated evidence before t*', String(sample.preTStarDuplicateEvidence)],
          ['Prefix cue count', String(sample.prefixCueCount)],
          ['Future recap in mechanism context', sample.futureRecapExcluded ? 'NO' : 'YES'],
          ['Whole-token spill', sample.rightSpillChars + ' chars ' + JSON.stringify(sample.rightSpillText)],
          ['t* / stop char', sample.tStarChar + ' / ' + sample.stopCharEnd],
          ['Parsed final count', String(sample.parsedCount) + (sample.exactCount ? ' (exact)' : ' (not exact)')],
          ['Context status', sample.contextStatus],
          ['Source row SHA-256', sample.sourceRowSha256],
          ['Source output SHA-256', sample.sourceOutputSha256],
          ['Stopping rule', sample.stoppingRule]
        ];
        return '<div class="audit-grid">' + values.map(([key, value]) => `<div class="audit-row"><span>${escapeHtml(key)}</span><strong>${escapeHtml(value)}</strong></div>`).join('') + '</div>';
      }

      function tabContent(sample) {
        if (state.tab === 'prefix') return `<div class="trace-toolbar"><p>机制实际输入：smallest whole-token prefix covering the K-th first occurrence；绿色高亮为 K 个首次 evidence events。</p><button class="icon-button" id="copyTrace" type="button">复制</button></div><div class="trace-box" id="activeTrace">${highlightedPrefix(sample)}</div>`;
        if (state.tab === 'full') return `<div class="trace-toolbar"><p>完整原生 generation，仅用于审计。t* 之后文本未进入机制实验。</p><button class="icon-button" id="copyTrace" type="button">复制</button></div><div class="trace-box" id="activeTrace">${escapeHtml(sample.fullOutput)}</div>`;
        if (state.tab === 'prompt') return `<div class="trace-toolbar"><p>完整 user prompt，含长 passage 与注入的 city-score records。</p><button class="icon-button" id="copyTrace" type="button">复制</button></div><div class="trace-box" id="activeTrace">${escapeHtml(sample.userPrompt)}</div>`;
        return `<div class="trace-toolbar"><p>资格、截断与 provenance 字段。</p></div>${auditMarkup(sample)}`;
      }

      function renderPanel(sample) {
        if (!sample) {
          els.panel.innerHTML = '<div class="empty-state"><strong>没有匹配的 seed</strong><br>清空搜索词或切换 split。</div>';
          return;
        }
        const globalBadge = sample.globalClean
          ? '<span class="badge pass">global-clean</span>'
          : '<span class="badge warn">post-t* recap excluded</span>';
        const gold = sample.goldRecords.map(record => `<div class="gold-item"><strong>${escapeHtml(record.city)}</strong><span>score ${escapeHtml(record.score)} · slot ${escapeHtml(record.slot_index)}</span></div>`).join('');
        const tabs = [
          ['prefix', '机制 t* 前缀'], ['full', '完整模型输出'], ['prompt', '完整 prompt'], ['audit', '审计字段']
        ].map(([id, label]) => `<button type="button" class="tab-button ${state.tab === id ? 'active' : ''}" data-tab="${id}" role="tab" aria-selected="${state.tab === id}">${label}</button>`).join('');
        els.panel.innerHTML = `
          <div class="sample-head">
            <div><p class="panel-label">N=${sample.n} · ${escapeHtml(sample.split)} #${sample.rank}</p><h3 class="sample-title">seed ${sample.seed}</h3>
              <div class="badge-row"><span class="badge pass">prefix-clean</span>${globalBadge}<span class="badge">${escapeHtml(sample.model)}</span><span class="badge">${sample.exactCount ? 'final exact' : 'final mismatch'}</span></div>
            </div>
            <button class="icon-button" id="copyId" type="button">复制 ID</button>
          </div>
          <div class="sample-meta">
            <div><strong>${sample.promptTokens.toLocaleString()}</strong><span>prompt tokens</span></div>
            <div><strong>${sample.mechanismPrefixTokens}</strong><span>t* prefix tokens</span></div>
            <div><strong>${sample.removedOutputTokens}</strong><span>tokens removed after t*</span></div>
            <div><strong>${sample.goldCount}</strong><span>gold records</span></div>
          </div>
          <div class="gold-grid">${gold}</div>
          <div class="trace-tabs" role="tablist">${tabs}</div>
          <div id="tabContent">${tabContent(sample)}</div>`;
        els.panel.querySelectorAll('[data-tab]').forEach(button => button.addEventListener('click', () => {
          state.tab = button.dataset.tab;
          renderPanel(sample);
        }));
        document.getElementById('copyId')?.addEventListener('click', () => navigator.clipboard.writeText(sample.requestId).then(() => showToast('Request ID 已复制')));
        document.getElementById('copyTrace')?.addEventListener('click', () => {
          const value = state.tab === 'prefix' ? sample.mechanismPrefix : state.tab === 'full' ? sample.fullOutput : sample.userPrompt;
          navigator.clipboard.writeText(value).then(() => showToast('当前文本已复制'));
        });
      }

      function renderSeeds() {
        const rows = filtered();
        if (!rows.some(row => row.seed === state.seed)) state.seed = rows[0]?.seed ?? null;
        els.count.textContent = `${rows.length} / ${samples.filter(s => s.n === state.n).length} seeds`;
        els.list.innerHTML = rows.map(row => `<button type="button" class="seed-button ${row.seed === state.seed ? 'active' : ''}" data-seed="${row.seed}"><span>${row.seed}</span><small>${row.split === 'discovery' ? 'D' : 'C'}${row.rank}</small></button>`).join('');
        els.list.querySelectorAll('[data-seed]').forEach(button => button.addEventListener('click', () => {
          state.seed = Number(button.dataset.seed);
          renderSeeds();
        }));
        const current = rows.find(row => row.seed === state.seed);
        renderPanel(current);
        const index = rows.findIndex(row => row.seed === state.seed);
        els.prev.disabled = index <= 0;
        els.next.disabled = index < 0 || index >= rows.length - 1;
      }

      function stepSeed(delta) {
        const rows = filtered();
        const index = rows.findIndex(row => row.seed === state.seed);
        const target = rows[index + delta];
        if (target) { state.seed = target.seed; renderSeeds(); }
      }

      els.nToggle.addEventListener('click', event => {
        const button = event.target.closest('[data-n]'); if (!button) return;
        state.n = Number(button.dataset.n); state.seed = filtered()[0]?.seed ?? samples.find(s => s.n === state.n)?.seed; state.tab = 'prefix';
        els.nToggle.querySelectorAll('button').forEach(b => b.classList.toggle('active', b === button)); renderSeeds();
      });
      els.splitToggle.addEventListener('click', event => {
        const button = event.target.closest('[data-split]'); if (!button) return;
        state.split = button.dataset.split; state.seed = null;
        els.splitToggle.querySelectorAll('button').forEach(b => b.classList.toggle('active', b === button)); renderSeeds();
      });
      els.search.addEventListener('input', () => { state.query = els.search.value; state.seed = null; renderSeeds(); });
      els.prev.addEventListener('click', () => stepSeed(-1));
      els.next.addEventListener('click', () => stepSeed(1));

      function renderGeometry() {
        const rows = geometry[String(state.geometryN)];
        const width = 1040, height = 410, left = 62, right = 30, top = 28, bottom = 54;
        const yMin = state.geometryN === 3 ? -0.12 : -0.08;
        const yMax = state.geometryN === 3 ? 0.72 : 0.26;
        const x = layer => left + layer / 35 * (width - left - right);
        const y = value => top + (yMax - value) / (yMax - yMin) * (height - top - bottom);
        const path = key => rows.map((row, i) => `${i ? 'L' : 'M'}${x(row.layer).toFixed(1)},${y(row[key]).toFixed(1)}`).join(' ');
        const ticks = state.geometryN === 3 ? [-.1, 0, .2, .4, .6] : [-.05, 0, .05, .1, .15, .2, .25];
        const grid = ticks.map(tick => `<g><line x1="${left}" x2="${width-right}" y1="${y(tick)}" y2="${y(tick)}" stroke="${tick === 0 ? '#9ba19c' : '#dedfd8'}" stroke-width="${tick === 0 ? 1.3 : 1}"/><text x="${left-10}" y="${y(tick)+4}" text-anchor="end" fill="#68706a" font-size="11" font-family="var(--mono)">${tick.toFixed(2)}</text></g>`).join('');
        const xTicks = [0,5,10,15,19,25,30,35].map(layer => `<g><line x1="${x(layer)}" x2="${x(layer)}" y1="${top}" y2="${height-bottom}" stroke="#ecece6"/><text x="${x(layer)}" y="${height-bottom+24}" text-anchor="middle" fill="#68706a" font-size="11" font-family="var(--mono)">L${layer}</text></g>`).join('');
        const peak = rows.reduce((best, row) => row.contrast > best.contrast ? row : best, rows[0]);
        els.geometryChart.innerHTML = `<svg viewBox="0 0 ${width} ${height}" aria-hidden="true">
          <rect x="${x(17)}" y="${top}" width="${x(23)-x(17)}" height="${height-top-bottom}" fill="#e1eee9" opacity=".72"/>
          ${grid}${xTicks}
          <path d="${path('orthogonal')}" fill="none" stroke="#8b938d" stroke-width="1.6" stroke-dasharray="5 5"/>
          <path d="${path('opposite')}" fill="none" stroke="#a95d2b" stroke-width="1.7" stroke-dasharray="3 5"/>
          <path d="${path('real')}" fill="none" stroke="#1f6f5f" stroke-width="2.5"/>
          <path d="${path('contrast')}" fill="none" stroke="#171b18" stroke-width="2.2"/>
          <circle cx="${x(peak.layer)}" cy="${y(peak.contrast)}" r="5" fill="#171b18" stroke="#fbfaf6" stroke-width="2"/>
          <text x="${x(peak.layer)+10}" y="${y(peak.contrast)-10}" fill="#171b18" font-size="12" font-weight="700" font-family="var(--mono)">L${peak.layer}  ${peak.contrast.toFixed(3)}</text>
          <text x="${(left+width-right)/2}" y="${height-10}" text-anchor="middle" fill="#465049" font-size="12">decoder block-input layer</text>
          <text x="16" y="${(top+height-bottom)/2}" transform="rotate(-90 16 ${(top+height-bottom)/2})" text-anchor="middle" fill="#465049" font-size="12">expected-count shift</text>
        </svg>`;
      }

      els.geometryToggle.addEventListener('click', event => {
        const button = event.target.closest('[data-geometry-n]'); if (!button) return;
        state.geometryN = Number(button.dataset.geometryN);
        els.geometryToggle.querySelectorAll('button').forEach(b => b.classList.toggle('active', b === button)); renderGeometry();
      });

      els.provenance.innerHTML = [3,10].map(n => {
        const item = meta[String(n)];
        return `N=${n} selected rows: ${item.sourceSha256}<br>N=${n} t* contexts: ${item.contextSha256}<br>N=${n} cohort manifest: ${item.manifestSha256}`;
      }).join('<br>');

      renderSeeds();
      renderGeometry();
    })();
  </script>
</body>
</html>
'''


def main() -> None:
    samples, cohort_meta = build_samples()
    if len(samples) != 60:
        raise ValueError(f"Expected 60 samples, found {len(samples)}")
    geometry = build_geometry()
    kgrid_hashes = validate_kgrid()
    event_tail_hashes = validate_event_tail()
    patch_scope_hashes = validate_patch_scope()
    kgrid_provenance = "<br>".join(
        f"{name}: {digest}" for name, digest in kgrid_hashes.items()
    )
    event_tail_provenance = "<br>".join(
        f"event_tail_{name}: {digest}" for name, digest in event_tail_hashes.items()
    )
    patch_scope_provenance = "<br>".join(
        f"patch_scope_{name}: {digest}"
        for name, digest in patch_scope_hashes.items()
    )
    patch_scope_layer_plot = base64.b64encode(
        PATCH_SCOPE_FILES["layer_plot"].read_bytes()
    ).decode("ascii")
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    document = (
        TEMPLATE.replace("__SAMPLES__", safe_json(samples))
        .replace("__COHORT_META__", safe_json(cohort_meta))
        .replace("__GEOMETRY__", safe_json(geometry))
        .replace("__KGRID_PROVENANCE__", kgrid_provenance)
        .replace("__EVENT_TAIL_PROVENANCE__", event_tail_provenance)
        .replace("__PATCH_SCOPE_PROVENANCE__", patch_scope_provenance)
        .replace("__PATCH_SCOPE_LAYER_PLOT__", patch_scope_layer_plot)
        .replace("__GENERATED__", generated)
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(document, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "bytes": OUTPUT.stat().st_size,
                "samples": len(samples),
                "n3": sum(sample["n"] == 3 for sample in samples),
                "n10": sum(sample["n"] == 10 for sample in samples),
                "sha256": sha256(OUTPUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
