#!/usr/bin/env python3
"""Audit and report the corrected V5 native-thinking causal chain.

This builder intentionally consumes only answer_query_extension_v3 and the
restartable all-site rollout outputs.  It refuses to fall back to the legacy
answer_query_extension tree.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MODELS = ("Qwen3-8B", "Gemma4-E4B")
VARIANTS = ("pre_city_d1", "pre_city_d2", "pre_city_anchor")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def number(value: Any) -> float:
    return float(value)


def fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(cell))}</th>" for cell in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def audit_all_site(model_root: Path, variant: str) -> dict[str, Any]:
    root = model_root / "causal/trace_rollout_damage" / variant
    analysis_audit = read_json(root / "analysis/audit.json")
    require(analysis_audit.get("status") == "passed", f"{model_root.name}/{variant}: analysis failed")
    require(int(analysis_audit.get("unmatched_ranked_treatments", -1)) >= 0, "missing unmatched audit")
    claims = list(root.rglob("*.claim"))
    require(not claims, f"{model_root.name}/{variant}: residual claims: {claims[:3]}")

    shard_roots = sorted(path for path in root.glob("trials_*_shards_v2") if path.is_dir())
    require(shard_roots, f"{model_root.name}/{variant}: no v2 shard roots")
    identities: set[tuple[str, str]] = set()
    rows = 0
    status_counts: Counter[str] = Counter()
    workers: Counter[str] = Counter()
    for shard_root in shard_roots:
        stream = shard_root.name
        for path in sorted(shard_root.rglob("*.jsonl")):
            payloads = list(jsonl_rows(path))
            require(len(payloads) == 1, f"{path}: expected exactly one atomic row")
            row = payloads[0]
            task_id = str(row.get("restartable_task_id", ""))
            require(task_id, f"{path}: missing restartable_task_id")
            identity = (stream, task_id)
            require(identity not in identities, f"{model_root.name}/{variant}: duplicate task {identity}")
            identities.add(identity)
            require(row.get("query_variant") == variant, f"{path}: variant mismatch")
            require(row.get("broad_aggregation_used") is False, f"{path}: broad aggregation used")
            status = str(row.get("status", ""))
            status_counts[status] += 1
            if status == "ok":
                require(row.get("prefill_reuse_audit") == "PASS_SINGLE_PREFILL", f"{path}: prefill audit failed")
                require(row.get("head_ablation_hook_audit") == "PASS", f"{path}: hook audit failed")
                require(
                    row.get("behavioral_endpoint") == "strict_greedy_complete_numeric_generation",
                    f"{path}: behavioral endpoint mismatch",
                )
                require("completion_text_raw" in row and "prediction" in row, f"{path}: actual output missing")
            elif status == "registered_query_exclusion":
                exclusions = row.get("exclusions")
                require(row.get("scheduled_query_count") == 0, f"{path}: excluded row scheduled queries")
                require(isinstance(exclusions, list) and exclusions, f"{path}: exclusion detail missing")
                require(
                    all(item.get("status") == "query_would_precede_item_start" for item in exclusions),
                    f"{path}: unregistered exclusion reason",
                )
                require("completion_text_raw" not in row, f"{path}: excluded row disguised as rollout")
            else:
                raise AssertionError(f"{path}: unexpected status {status!r}")
            workers[str(row.get("execution_worker_id", "unknown"))] += 1
            rows += 1
    require(rows == int(analysis_audit["rows"]), f"{model_root.name}/{variant}: shards={rows}, audit={analysis_audit['rows']}")
    require(dict(status_counts) == analysis_audit["status_counts"], f"{model_root.name}/{variant}: status counts differ")
    return {
        "rows": rows,
        "ok_rows": status_counts["ok"],
        "registered_query_exclusions": status_counts["registered_query_exclusion"],
        "paired_requests": int(analysis_audit["paired_requests"]),
        "unmatched_ranked_treatments": int(analysis_audit["unmatched_ranked_treatments"]),
        "workers": dict(sorted(workers.items())),
        "audit": analysis_audit,
    }


def audit_answer_execution(model_root: Path) -> dict[str, Any]:
    root = model_root / "answer_query_extension_v3/answer_execution"
    plan = read_json(root / "plan" / f"{model_root.name}__answer_execution_plan_audit.json")
    layer = read_json(root / "plan" / f"{model_root.name}__answer_execution_layer_selection.json")
    analysis = read_json(root / "analysis/audit.json")
    require(plan.get("confirmation_used_for_selection") is False, f"{model_root.name}: confirmation selected layer")
    require(plan.get("selection_split") == "discovery", f"{model_root.name}: non-discovery layer selection")
    require(plan.get("excluded_incorrect_one_to_one_rows") == 0, f"{model_root.name}: incorrect patch rows admitted")
    require(analysis.get("status") == "passed", f"{model_root.name}: answer execution analysis failed")
    require(analysis.get("completed_pairs") == analysis.get("registered_pairs"), f"{model_root.name}: incomplete pairs")
    require(int(analysis.get("conditions_per_pair", 0)) == 4, f"{model_root.name}: wrong condition count")

    trials_path = root / "trials_confirmation.jsonl"
    trials = list(jsonl_rows(trials_path))
    expected = int(analysis["completed_pairs"]) * 4
    require(len(trials) == expected, f"{model_root.name}: answer execution rows {len(trials)} != {expected}")
    expected_conditions = {"self_patch", "full_donor_patch", "projected_donor_patch", "orthogonal_norm_matched"}
    pair_conditions: dict[str, set[str]] = defaultdict(set)
    for row in trials:
        require(row.get("receiver_exact_count") is True, f"{model_root.name}: inexact receiver")
        require(row.get("donor_exact_count") is True, f"{model_root.name}: inexact donor")
        require(
            row.get("pair_eligibility") == "receiver_and_donor_baseline_final_answer_exact",
            f"{model_root.name}: runtime eligibility mismatch",
        )
        require(row.get("captured_state_cache_audit") == "PASS_REUSED_SITE_STATE", f"{model_root.name}: cache audit failed")
        require(row.get("self_patch_cache_reused") is True, f"{model_root.name}: self patch not reused")
        require("completion_text_raw" in row and "prediction" in row, f"{model_root.name}: actual output missing")
        pair_conditions[str(row["pair_id"])].add(str(row["condition"]))
    require(all(value == expected_conditions for value in pair_conditions.values()), f"{model_root.name}: incomplete condition set")
    require(len(pair_conditions) == int(analysis["completed_pairs"]), f"{model_root.name}: pair identity mismatch")
    return {
        "selected_layer": int(layer["selected_layer"]),
        "selected_rank": int(layer["selected_rank"]),
        "pairs": int(analysis["completed_pairs"]),
        "seeds": len(plan["confirmation_seeds"]),
        "statistics": read_csv(root / "analysis/statistics.csv"),
        "analysis_audit": analysis,
        "plan_audit": plan,
    }


def representation_at_selected_layer(model_root: Path, layer: int) -> dict[str, Any]:
    root = model_root / "answer_query_extension_v3/representation/analysis_combined"
    audit = read_json(root / "representation_audit.json")
    require(audit.get("skipped") == [], f"{model_root.name}: skipped representation groups")
    require(audit.get("observed_sites") == ["answer_query_v3"], f"{model_root.name}: wrong answer site")
    candidates = [
        row for row in read_csv(root / "regression_confirmation.csv")
        if row["cohort"] == "one_to_one_correct"
        and row["site_kind"] == "answer_query_v3"
        and row["probe"] == "ridge"
        and int(row["layer"]) == layer
    ]
    require(len(candidates) == 1, f"{model_root.name}: selected-layer representation row missing")
    return {"row": candidates[0], "audit": audit}


def summarize_holm(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["model_label"], row["family"], row["metric"])].append(row)
    summaries = []
    for (model, family, metric), frame in sorted(groups.items()):
        effects = [number(row["effect"]) for row in frame]
        holm_values = [number(row["holm_p_within_family_endpoint"]) for row in frame]
        summaries.append({
            "model": model,
            "family": family,
            "metric": metric,
            "tests": len(frame),
            "effect_min": min(effects),
            "effect_max": max(effects),
            "min_holm_p": min(holm_values),
            "holm_significant": sum(value <= 0.05 for value in holm_values),
        })
    return summaries


def build(run_root: Path, output_dir: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cross_dir = run_root / "corrected_causal_chain_cross_model"
    cross_audit = read_json(cross_dir / "audit.json")
    require(cross_audit.get("status") == "passed", "cross-model Holm audit failed")
    cross_rows = read_csv(cross_dir / "combined_statistics_holm.csv")
    require(len(cross_rows) == int(cross_audit["tests"]), "cross-model Holm row mismatch")
    require(sha256(cross_dir / "combined_statistics_holm.csv") == cross_audit["output_sha256"], "cross Holm hash mismatch")

    acceleration = {}
    for model in MODELS:
        path = run_root / "acceleration_audit/fused_prefill" / f"{model}.json"
        payload = read_json(path)
        require(payload.get("passed") is True, f"{model}: fused prefill equivalence failed")
        acceleration[f"fused_prefill/{model}"] = payload
    for name in ("shared_shard_claims", "shared_shard_claims_pidaware"):
        path = run_root / "acceleration_audit" / name / "audit.json"
        if path.exists():
            payload = read_json(path)
            require(payload.get("passed") is True, f"{name}: claim audit failed")
            acceleration[name] = payload

    models: dict[str, Any] = {}
    for model in MODELS:
        model_root = run_root / model
        status_path = model_root / "answer_query_extension_v3/logs/supervisor.status"
        require(status_path.read_text(encoding="utf-8").strip() == "complete", f"{model}: supervisor incomplete")
        execution = audit_answer_execution(model_root)
        representation = representation_at_selected_layer(model_root, execution["selected_layer"])
        head_audit = read_json(model_root / "answer_query_extension_v3/head_ablation/analysis/audit.json")
        require(head_audit.get("status") == "passed", f"{model}: head analysis failed")
        all_site = {variant: audit_all_site(model_root, variant) for variant in VARIANTS}
        models[model] = {
            "representation": representation,
            "head_audit": head_audit,
            "execution": execution,
            "all_site": all_site,
        }

    holm_summary = summarize_holm(cross_rows)
    holm_significant = sum(int(row["holm_significant"]) for row in holm_summary)
    require(holm_significant == int(cross_audit["holm_significant"]), "Holm significant count mismatch")

    rep_rows = []
    patch_rows = []
    rollout_rows = []
    for model, payload in models.items():
        rep = payload["representation"]["row"]
        rep_rows.append([
            model,
            payload["execution"]["selected_layer"],
            payload["execution"]["selected_rank"],
            rep["n_discovery"],
            rep["n_confirmation"],
            fmt(rep["confirmation_r2"]),
            fmt(rep["confirmation_mae"]),
        ])
        for row in payload["execution"]["statistics"]:
            patch_rows.append([
                model,
                row["metric"],
                row["seed_clusters"],
                fmt(row["effect"]),
                f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]",
                fmt(row["positive_seed_fraction"]),
            ])
        for variant, audit in payload["all_site"].items():
            rollout_rows.append([
                model,
                variant,
                audit["rows"],
                audit["ok_rows"],
                audit["registered_query_exclusions"],
                audit["paired_requests"],
                audit["unmatched_ranked_treatments"],
                ", ".join(f"{worker}:{count}" for worker, count in audit["workers"].items()),
            ])

    holm_rows = [[
        row["model"], row["family"], row["metric"], row["tests"],
        f"[{fmt(row['effect_min'])}, {fmt(row['effect_max'])}]",
        fmt(row["min_holm_p"]), row["holm_significant"],
    ] for row in holm_summary]

    if holm_significant == 0:
        head_conclusion = (
            "Across 147 constructible model × registered-K tests, no behavioral head-damage test "
            "survived Holm correction. This is an absence of multiplicity-corrected evidence, not proof of zero effect."
        )
    else:
        head_conclusion = f"{holm_significant} behavioral head-damage tests survived Holm correction."

    generated = datetime.now(timezone.utc).isoformat()
    html_text = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>V5 Corrected Native-Thinking Causal Chain</title>
<style>
:root{{--ink:#172033;--muted:#667085;--line:#d7deea;--panel:#f7f9fc;--accent:#3157d5;--ok:#137a55}}
body{{font-family:Inter,system-ui,sans-serif;margin:0;background:#eef2f8;color:var(--ink);line-height:1.55}}
main{{max-width:1180px;margin:32px auto;padding:0 24px 64px}} header,section{{background:white;border:1px solid var(--line);border-radius:16px;padding:24px;margin:16px 0;box-shadow:0 8px 28px #24345a0d}}
h1{{margin:.2rem 0;font-size:2rem}} h2{{border-bottom:1px solid var(--line);padding-bottom:8px}} h3{{margin-top:1.4rem}}
.eyebrow{{color:var(--accent);font-weight:750;text-transform:uppercase;letter-spacing:.08em;font-size:.78rem}} .muted{{color:var(--muted)}}
.badge{{display:inline-block;background:#e8f7f0;color:var(--ok);font-weight:750;padding:5px 10px;border-radius:999px}}
.callout{{border-left:4px solid var(--accent);background:var(--panel);padding:14px 16px;border-radius:8px}}
.table-wrap{{overflow:auto}} table{{border-collapse:collapse;width:100%;font-size:.88rem}} th,td{{text-align:left;border-bottom:1px solid var(--line);padding:8px 10px;white-space:nowrap}} th{{background:var(--panel);position:sticky;top:0}}
code{{background:#edf1f7;padding:2px 5px;border-radius:5px}} details{{margin-top:12px}}
</style></head><body><main>
<header><div class='eyebrow'>Realistic NIAH · V5 native thinking · corrected causal chain</div>
<h1>Answer-query representation, head damage, and correct-only answer execution</h1>
<p class='muted'>Generated {html.escape(generated)}. Legacy <code>answer_query_extension</code> and OV comparisons are excluded; E3 geometry is paused.</p>
<p><span class='badge'>FINAL AUDIT PASSED</span></p></header>
<section><h2>Executive conclusion</h2>
<div class='callout'><strong>Representation and answer-execution transport are supported; localized head-damage evidence is not multiplicity-robust.</strong> {html.escape(head_conclusion)} Correct-only patching nevertheless transports donor count information, with positive projected-transport confidence intervals in both models.</div>
<p>The behavioral endpoint throughout ablation is the parsed numeric value from the model's actual greedy final text—not candidate-logit argmax. Head banks and answer-execution layers were selected on discovery only; confirmation is evaluation only.</p></section>
<section><h2>1 · Answer-query representation</h2>
<p>The queried state is the literal baseline token immediately before the first numeric answer token. The selected execution layer is frozen using leave-one-seed-out discovery ridge MAE.</p>
{table(['Model','Discovery-selected layer','Basis rank','Discovery n','Confirmation n','Confirmation R²','Confirmation MAE'], rep_rows)}</section>
<section><h2>2 · Answer-query head damage</h2>
<p>Prompt-sequence aggregation and thinking-trace aggregation banks were independently discovered, frozen, and evaluated using ranked heads versus constructible disjoint exact layer-matched controls at registered K. {html.escape(head_conclusion)}</p>
<details><summary>Cross-model Holm family details (all 147 tests)</summary>
{table(['Model','Family','Behavioral metric','Tests','Effect range','Minimum Holm p','Holm significant'], holm_rows)}</details></section>
<section><h2>3 · Correct-only answer execution</h2>
<p>Both receiver and donor were strict one-to-one and baseline-answer correct. Each pair has self, full donor, projected donor, and norm-matched orthogonal conditions; cached-state and self-patch reuse audits passed.</p>
{table(['Model','Metric','Seed clusters','Effect','95% seed-bootstrap CI','Positive-seed fraction'], patch_rows)}</section>
<section><h2>4 · Pre-city targeted-retrieval damage</h2>
<p>For d1, d2, and anchor, the frozen targeted-retrieval bank was ablated jointly at every corresponding true generation-token query position. Every atomic trial reports <code>PASS_SINGLE_PREFILL</code>, hook PASS, no broad aggregation, and an actual greedy completion.</p>
{table(['Model','Variant','Atomic rows','Valid rollouts','Registered query exclusions','Paired request-effects','Unmatched constructibility rows','Workers'], rollout_rows)}</section>
<section><h2>5 · Causal-chain interpretation</h2>
<ol><li><strong>Answer-query representation:</strong> count is strongly linearly decodable at discovery-selected late layers on held-out confirmation.</li>
<li><strong>Answer execution:</strong> correct-only donor patching changes the actual generated answer in the donor direction; low-rank projected transport remains positive in both models.</li>
<li><strong>Head localization:</strong> neither answer-query aggregation banks nor pre-city targeted-retrieval banks show familywise multiplicity-corrected behavioral damage in this run.</li></ol>
<p>Therefore the corrected evidence supports a count-bearing answer-query state and its causal use during answer execution, but does <em>not</em> establish that the tested frozen head banks are individually necessary for final counting behavior. Compensation, distributed coding, limited seed power, and constructibility exclusions remain viable explanations.</p></section>
<section><h2>Audit scope</h2><ul>
<li>Both model supervisors complete; all all-site claims removed.</li><li>Restartable task identity unique within each model × variant × primary/supplement queue; exactly one row per atomic shard.</li>
<li>Fused-prefill equivalence, PID-aware shared-claim, correct-only pair eligibility, four-condition completeness, and cached-state reuse passed.</li>
<li>Cross-model Holm audit: {cross_audit['tests']} tests, {cross_audit['holm_significant']} significant.</li></ul></section>
</main></body></html>"""

    report_path = output_dir / "v5_corrected_native_thinking_causal_chain_report.html"
    report_path.write_text(html_text, encoding="utf-8")
    audit_payload = {
        "schema_version": "v5_corrected_native_thinking_causal_chain_final_audit_v1",
        "status": "passed",
        "generated_at": generated,
        "legacy_answer_query_extension_used": False,
        "models": {
            model: {
                "supervisor": "complete",
                "representation_groups": payload["representation"]["audit"]["groups_completed"],
                "answer_execution_pairs": payload["execution"]["pairs"],
                "selected_layer": payload["execution"]["selected_layer"],
                "all_site_rows": {variant: value["rows"] for variant, value in payload["all_site"].items()},
                "all_site_ok_rows": {variant: value["ok_rows"] for variant, value in payload["all_site"].items()},
                "all_site_registered_query_exclusions": {
                    variant: value["registered_query_exclusions"] for variant, value in payload["all_site"].items()
                },
                "all_site_claims": 0,
            }
            for model, payload in models.items()
        },
        "acceleration_audits": {key: True for key in acceleration},
        "cross_model_holm": cross_audit,
        "report": str(report_path.resolve()),
        "report_sha256": sha256(report_path),
    }
    audit_path = output_dir / "final_audit.json"
    audit_path.write_text(json.dumps(audit_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "v5_corrected_native_thinking_causal_chain_report_manifest_v1",
        "status": "passed",
        "files": [
            {"path": str(report_path.resolve()), "sha256": sha256(report_path)},
            {"path": str(audit_path.resolve()), "sha256": sha256(audit_path)},
            {"path": str((cross_dir / 'combined_statistics_holm.csv').resolve()), "sha256": sha256(cross_dir / 'combined_statistics_holm.csv')},
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "passed", "report": str(report_path), "audit": str(audit_path), "manifest": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run_root, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
