#!/usr/bin/env python3
"""Audit and report the V5 response-aware causal-chain v3 experiments.

The report treats the parser-defined trace ``pre_reference_d1`` experiment as
the new primary targeted-retrieval result, retains the response-type/position
run as supplementary evidence, incorporates the non-thinking-compatible broad
answer aggregation experiment, and carries forward the already-audited v2
chain.  All behavioral endpoints are parsed actual greedy generations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MODELS = ("Qwen3-8B", "Gemma4-E4B")
KS = {1, 2, 4, 8, 16, 32}
RESPONSE_TYPES = {"bare_or_list", "record_template", "semantic_cue"}
START_KINDS = {"exact_record_prefix", "exact_city_fallback"}
SCHEMA = "realistic_niah_v5_response_aware_causal_v3_report_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing CSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, f"empty CSV: {path}")
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL {path}:{number}: {error}") from error
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(value: Any) -> float:
    result = float(value)
    require(math.isfinite(result), f"non-finite value: {value!r}")
    return result


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def holm(values: list[float]) -> list[float]:
    require(values and all(0 <= value <= 1 for value in values), "invalid p-values")
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [float("nan")] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, values[index] * (len(values) - rank)))
        adjusted[index] = running
    return adjusted


def condition_kind(condition: str) -> str:
    if condition == "clean":
        return "clean"
    if condition.endswith("_ranked"):
        return "ranked"
    if "layer_matched_random" in condition:
        return "random"
    raise AssertionError(f"unknown condition: {condition}")


def audit_attention_registry(path: Path, model: str) -> dict[str, Any]:
    unique: dict[tuple[str, int], tuple[str, str, str, int, str]] = {}
    rows = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            require(row["model_label"] == model, f"{model}: attention model mismatch")
            require(row["position_variant"] == "pre_reference_d1", f"{model}: wrong position")
            require(row["query_variant"] == "pre_reference_d1", f"{model}: wrong query")
            require(row["token_distance_before_citation"] == "1", f"{model}: query distance")
            require(
                row["k_to_k_registry_audit"]
                == "PASS_RESPONSE_OCCURRENCE_TO_EXACT_CITY_PROMPT_SPAN",
                f"{model}: failed k-to-k registry",
            )
            raw = as_float(row["target_needle_raw_mass"])
            total = as_float(row["all_active_needles_raw_mass"])
            require(raw >= 0 and total >= 0 and raw <= total + 1e-8, f"{model}: invalid mass")
            if total > 0 and row["target_needle_relative_mass"]:
                relative = as_float(row["target_needle_relative_mass"])
                require(abs(relative - raw / total) <= 1e-7, f"{model}: relative mass mismatch")
            key = (row["request_id"], int(row["occurrence"]))
            payload = (
                row["response_type"], row["citation_start_kind"], row["target_city"],
                int(row["citation_first_token"]), row["response_reference_parser"],
            )
            if key in unique:
                require(unique[key] == payload, f"{model}: inconsistent registry occurrence {key}")
            else:
                unique[key] = payload
    require(rows > 0 and unique, f"{model}: empty attention registry")
    types = Counter(value[0] for value in unique.values())
    starts = Counter(value[1] for value in unique.values())
    parsers = Counter(value[4] for value in unique.values())
    require(set(types) == RESPONSE_TYPES, f"{model}: response-type coverage {types}")
    require(set(starts) <= START_KINDS and starts, f"{model}: citation starts {starts}")
    require(len(parsers) == 1, f"{model}: multiple parser versions")
    return {
        "csv_rows": rows,
        "registered_occurrences": len(unique),
        "response_types": dict(sorted(types.items())),
        "citation_start_kinds": dict(sorted(starts.items())),
        "parser": next(iter(parsers)),
        "raw_mass_reported": True,
        "relative_mass_audited": True,
        "sha256": sha256(path),
    }


def audit_target_trials(root: Path, model: str) -> dict[str, Any]:
    files = (
        root / "trials_primary_confirmation.jsonl",
        root / "trials_supplement_n10_confirmation.jsonl",
    )
    rows = [row for path in files for row in read_jsonl(path)]
    status = Counter(str(row.get("status", "ok")) for row in rows)
    ok = [row for row in rows if row.get("status", "ok") == "ok"]
    require(ok, f"{model}: no valid target trials")
    for row in ok:
        require(row["model_label"] == model, f"{model}: trial model mismatch")
        require(row["position_variant"] == "pre_reference_d1", f"{model}: trial position")
        require(row["token_distance_before_citation"] == 1, f"{model}: trial query distance")
        require(
            row["behavioral_endpoint"] == "actual_greedy_next_needle_token_sequence",
            f"{model}: non-greedy endpoint",
        )
        require(row["final_count_evaluated"] is False, f"{model}: final count used")
        require(
            row["k_to_k_registry_audit"]
            == "PASS_RESPONSE_OCCURRENCE_TO_EXACT_CITY_PROMPT_SPAN",
            f"{model}: trial registry failed",
        )
        require(row["citation_start_kind"] in START_KINDS, f"{model}: trial citation start")
    clean: set[tuple[Any, ...]] = set()
    interventions: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    for row in ok:
        base = (
            row["request_id"], row["response_type"], int(row["occurrence"]),
        )
        kind = condition_kind(str(row["condition"]))
        if kind == "clean":
            clean.add(base)
        else:
            interventions[(*base, int(row["bank_size"]))][kind] += 1
    require({key[-1] for key in interventions} == KS, f"{model}: K mismatch")
    for key, counts in interventions.items():
        require(key[:3] in clean, f"{model}: missing clean {key}")
        require(counts == Counter({"ranked": 1, "random": 3}), f"{model}: controls {key}: {counts}")
    types = Counter(str(row["response_type"]) for row in ok)
    starts = Counter(str(row["citation_start_kind"]) for row in ok)
    return {
        "rows": len(rows),
        "ok_rows": len(ok),
        "status_counts": dict(sorted(status.items())),
        "clean_occurrences": len(clean),
        "paired_bank_tests": len(interventions),
        "bank_sizes": sorted(KS),
        "response_types": dict(sorted(types.items())),
        "citation_start_kinds": dict(sorted(starts.items())),
        "three_exact_layer_matched_controls_per_test": True,
        "actual_greedy_endpoint_only": True,
        "input_sha256": {path.name: sha256(path) for path in files},
    }


def normalized_stat(
    row: dict[str, str], *, stage: str, mechanism: str, scope: str,
    response_type: str, source: Path, primary: bool,
) -> dict[str, Any]:
    return {
        "model_label": row["model_label"],
        "stage": stage,
        "mechanism": mechanism,
        "scope": scope,
        "response_type": response_type,
        "bank_size": row.get("bank_size", ""),
        "signed_count_gap": row.get("signed_count_gap", ""),
        "metric": row["metric"],
        "primary_endpoint": primary,
        "seed_clusters": int(float(row["seed_clusters"])),
        "effect": as_float(row["effect"]),
        "ci95_low": as_float(row["ci95_low"]),
        "ci95_high": as_float(row["ci95_high"]),
        "sign_flip_p": as_float(row["sign_flip_p"]),
        "sign_flip_method": row["sign_flip_method"],
        "sign_flip_assignments": int(float(row["sign_flip_assignments"])),
        "source_statistics": str(source.resolve()),
        "source_sha256": sha256(source),
    }


def load_statistics(run_root: Path, inputs: set[Path]) -> list[dict[str, Any]]:
    combined = run_root / "native_causal_chain_v2_report/combined_statistics_cross_model_holm.csv"
    rows = []
    for row in read_csv(combined):
        rows.append(normalized_stat(
            row, stage=row["stage"], mechanism=row["mechanism"], scope=row["scope"],
            response_type="", source=combined, primary=truthy(row["primary_endpoint"]),
        ))
    inputs.add(combined)
    for model in MODELS:
        v3 = run_root / model / "corrected_causal_chain_v3_response_aware"
        broad = v3 / "answer_broad_aggregation/analysis/statistics.csv"
        for row in read_csv(broad):
            rows.append(normalized_stat(
                row, stage="answer_broad_aggregation_v3", mechanism=row["family"],
                scope="actual_greedy_final_count", response_type="", source=broad,
                primary=truthy(row["is_primary_endpoint"]),
            ))
        inputs.add(broad)
        target = v3 / "targeted_retrieval_trace_pre_needle_d1/analysis/statistics.csv"
        for row in read_csv(target):
            response_type = row["response_type"]
            rows.append(normalized_stat(
                row, stage="trace_pre_reference_targeted_retrieval_v3",
                mechanism="pre_reference_d1_position_consensus",
                scope="actual_greedy_next_citation", response_type=response_type,
                source=target,
                primary=(response_type == "pooled_cross_response_types" and truthy(row["primary_endpoint"])),
            ))
        inputs.add(target)
        supplementary = v3 / "targeted_retrieval_by_response_type/analysis/statistics.csv"
        for row in read_csv(supplementary):
            rows.append(normalized_stat(
                row, stage="targeted_retrieval_response_type_supplement_v3",
                mechanism=row["query_variant"], scope="archived_multi_position_robustness",
                response_type=row["response_type"], source=supplementary, primary=False,
            ))
        inputs.add(supplementary)
    identities = set()
    families: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        identity = tuple(row[key] for key in (
            "model_label", "stage", "mechanism", "scope", "response_type",
            "bank_size", "signed_count_gap", "metric",
        ))
        require(identity not in identities, f"duplicate test {identity}")
        identities.add(identity)
        family = "::".join(str(row[key]) for key in (
            "stage", "mechanism", "scope", "response_type", "metric",
        ))
        row["holm_family"] = family
        families[family].append(index)
    for family, indices in families.items():
        adjusted = holm([rows[index]["sign_flip_p"] for index in indices])
        for index, value in zip(indices, adjusted):
            rows[index]["holm_family_size"] = len(indices)
            rows[index]["holm_p_cross_model"] = value
            rows[index]["holm_significant_0_05"] = value <= 0.05
    rows.sort(key=lambda row: tuple(str(row[key]) for key in (
        "stage", "mechanism", "scope", "response_type", "metric",
        "model_label", "bank_size", "signed_count_gap",
    )))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model_label", "stage", "mechanism", "scope", "response_type",
        "bank_size", "signed_count_gap", "metric", "primary_endpoint",
        "seed_clusters", "effect", "ci95_low", "ci95_high", "sign_flip_p",
        "sign_flip_method", "sign_flip_assignments", "holm_family",
        "holm_family_size", "holm_p_cross_model", "holm_significant_0_05",
        "source_statistics", "source_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def table(headers: list[str], body: Iterable[Iterable[Any]]) -> str:
    content = "".join(
        "<tr>" + "".join(f"<td>{html.escape(fmt(cell))}</td>" for cell in row) + "</tr>"
        for row in body
    )
    return "<div class='scroll'><table><thead><tr>" + "".join(
        f"<th>{html.escape(header)}</th>" for header in headers
    ) + f"</tr></thead><tbody>{content}</tbody></table></div>"


def render_report(audit: dict[str, Any], stats: list[dict[str, Any]]) -> str:
    primary = [row for row in stats if row["primary_endpoint"]]
    stage_rows = []
    for stage in sorted({row["stage"] for row in primary}):
        selected = [row for row in primary if row["stage"] == stage]
        stage_rows.append((
            stage, len(selected), sum(row["holm_significant_0_05"] for row in selected),
            min(row["holm_p_cross_model"] for row in selected),
            "显著" if any(row["holm_significant_0_05"] for row in selected) else "未达 Holm 0.05",
        ))
    v3_primary = [row for row in primary if row["stage"].endswith("_v3")]
    detail = [(
        row["model_label"], row["stage"], row["mechanism"], row["response_type"],
        row["bank_size"], row["metric"], row["effect"],
        f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]",
        row["sign_flip_p"], row["holm_p_cross_model"], row["seed_clusters"],
    ) for row in v3_primary]
    model_rows = []
    for model in MODELS:
        item = audit["models"][model]
        model_rows.append((
            model, item["supervisor"], item["attention_registry"]["registered_occurrences"],
            item["attention_registry"]["citation_start_kinds"],
            item["target_trials"]["paired_bank_tests"],
            item["broad_audit"]["paired_requests"],
        ))
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>V5 response-aware causal chain v3</title><style>
body{{margin:0;background:#f4f6fa;color:#18202b;font:15px/1.65 'Segoe UI',sans-serif}}main{{max-width:1200px;margin:auto;padding:38px 28px 80px}}h1,h2{{font-family:Georgia,'Noto Serif SC',serif}}h1{{font-size:38px}}section{{background:white;border:1px solid #dce2eb;margin:18px 0;padding:25px}}.ok{{border-left:4px solid #078b73;background:#effaf7;padding:14px}}.note{{border-left:4px solid #7357c7;background:#f4f0ff;padding:14px}}.chain{{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}}.chain div{{background:#edf1f7;padding:14px;border-top:3px solid #7357c7}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border-bottom:1px solid #e1e5ed;padding:8px;text-align:left;vertical-align:top}}th{{background:#edf1f7;position:sticky;top:0}}code{{background:#eceff4;padding:1px 4px}}@media(max-width:800px){{.chain{{grid-template-columns:1fr}}}}
</style></head><body><main><header><div>Realistic NIAH · V5 native thinking</div><h1>Response-aware causal chain v3</h1><p>生成时间：{html.escape(audit['generated_at'])}<br>代码提交：<code>{html.escape(audit['code_commit'])}</code></p></header>
<section><h2>1. 最终审计</h2><div class='ok'><b>PASS。</b> 双模型 parser-defined pre-reference targeted retrieval、broad answer aggregation、旧 v2 因果链、registry、三组 exact layer-matched controls、实际 greedy endpoint 与跨模型 Holm 汇总全部通过。</div>{table(['模型','状态','attention occurrences','citation start kinds','target paired K tests','broad paired requests'], model_rows)}</section>
<section><h2>2. 冻结定义</h2><div class='chain'><div><b>定位</b><br>parser 注册每次引用的完整起点；canonical record prefix 优先，否则 exact city fallback。</div><div><b>Query</b><br>唯一使用 citation 首 token 前一个真实 baseline token，即 <code>pre_reference_d1</code>。</div><div><b>k-to-k 排名</b><br>第 k 次引用对 prompt 第 k 个 exact needle span；raw mass 为主，active-needle relative mass 为审计量；三回答类型等权。</div><div><b>干预</b><br>K=1/2/4/8/16/32；ranked 对三套不重叠、exact layer-matched controls；endpoint 为实际 greedy citation。</div></div></section>
<section><h2>3. 哪些机制显著</h2><p>“显著”严格指预注册 primary endpoint 的跨模型/K family Holm p≤0.05；“未达显著”不是零效应证明。</p>{table(['阶段','primary tests','Holm significant','最小 Holm p','判定'], stage_rows)}</section>
<section><h2>4. 新 v3 primary 统计</h2>{table(['模型','阶段','机制','回答类型','K','指标','效应','95% CI','raw p','cross-Holm p','seed clusters'], detail)}</section>
<section><h2>5. 证据链解释</h2><ol><li>旧 v2 保留 marker hidden-state transport、旧 trace-local heads、answer-state transport/adoption 证据，不被 v3 覆盖。</li><li>新 targeted retrieval 把 query 精确移动到 parser 注册的引用起点之前，因此更直接测试“即将引用哪个 needle”的局部检索必要性。</li><li>broad answer aggregation 分别按 prompt spans 与 thinking-trace item spans 的 non-thinking 同款 broad score 排名，并比较 clean/prompt/trace/joint 的实际最终 count。</li><li>targeted retrieval 只支持局部 citation 生成的必要性；answer aggregation 才面向最终 count。两类 endpoint 不互换。</li></ol></section>
<section><h2>6. 审计边界</h2><p class='note'>Primary 与 supplement 分离；旧 response-type/multi-position run 仅作为 supplementary robustness。所有 superseded/unconstructible-control 目录均未进入统计。候选 logit argmax 未作为行为主指标。跨回答类型只冻结一套 position-consensus bank，per-type 仅作异质性审计。</p></section>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs: set[Path] = set()
    old_audit_path = run_root / "native_causal_chain_v2_report/final_audit.json"
    old_audit = read_json(old_audit_path)
    require(old_audit.get("status") == "passed", "v2 final audit failed")
    inputs.add(old_audit_path)
    models = {}
    for model in MODELS:
        v3 = run_root / model / "corrected_causal_chain_v3_response_aware"
        target = v3 / "targeted_retrieval_trace_pre_needle_d1"
        require((target / "logs/supervisor.status").read_text().strip() == "complete", f"{model}: target incomplete")
        target_audit = read_json(target / "analysis/audit.json")
        plan_audit = read_json(target / "plan/causal_plan_audit.json")
        require(target_audit.get("status") == "passed" and target_audit.get("unmatched") == 0, f"{model}: target audit")
        require(plan_audit.get("selection_split") == "discovery", f"{model}: selection split")
        require(plan_audit.get("confirmation_used_for_selection") is False, f"{model}: leakage")
        require(set(plan_audit.get("registered_bank_sizes", [])) == KS, f"{model}: plan K")
        require(plan_audit.get("position_variants") == ["pre_reference_d1"], f"{model}: plan position")
        broad_audit = read_json(v3 / "answer_broad_aggregation/analysis/audit.json")
        broad_plan = read_json(v3 / "answer_broad_aggregation/plan/answer_query_causal_plan_audit.json")
        require(broad_audit.get("status") == "passed" and broad_audit.get("unmatched_ranked_treatments") == 0, f"{model}: broad audit")
        require(broad_plan.get("selection_split") == "discovery" and broad_plan.get("confirmation_used_for_selection") is False, f"{model}: broad leakage")
        require(set(broad_plan.get("registered_bank_sizes", [])) == KS, f"{model}: broad K")
        supplementary_audit = read_json(v3 / "targeted_retrieval_by_response_type/analysis/audit.json")
        require(supplementary_audit.get("status") == "passed", f"{model}: supplementary audit")
        key_files = {
            target / "logs/supervisor.status", target / "analysis/audit.json",
            target / "plan/causal_plan_audit.json", v3 / "answer_broad_aggregation/analysis/audit.json",
            v3 / "answer_broad_aggregation/plan/answer_query_causal_plan_audit.json",
            v3 / "targeted_retrieval_by_response_type/analysis/audit.json",
        }
        inputs.update(key_files)
        models[model] = {
            "supervisor": "complete",
            "attention_registry": audit_attention_registry(target / "attention_primary.csv", model),
            "target_trials": audit_target_trials(target, model),
            "target_analysis": target_audit,
            "target_plan": plan_audit,
            "broad_audit": broad_audit,
            "broad_plan": broad_plan,
            "response_type_supplement_audit": supplementary_audit,
        }
    statistics = load_statistics(run_root, inputs)
    stats_path = output / "combined_statistics_cross_model_holm.csv"
    write_csv(stats_path, statistics)
    generated_at = datetime.now(timezone.utc).isoformat()
    audit = {
        "schema_version": SCHEMA,
        "status": "passed",
        "generated_at": generated_at,
        "code_commit": args.code_commit,
        "behavioral_endpoint_policy": "parsed actual greedy generation only; no candidate-logit argmax",
        "selection_policy": "discovery-only ranking; frozen confirmation evaluation",
        "primary_targeted_retrieval": "parser-defined pre_reference_d1 position-consensus bank",
        "supplement_policy": "primary and supplement retained separately; archived multi-position run is supplementary only",
        "superseded_outputs_consumed": False,
        "v2_final_audit_status": old_audit["status"],
        "models": models,
        "cross_model_holm": {
            "family_definition": "stage x mechanism x scope x response-type stratum x endpoint; correction spans model and K without endpoint mixing",
            "tests": len(statistics),
            "families": len({row["holm_family"] for row in statistics}),
            "primary_tests": sum(row["primary_endpoint"] for row in statistics),
            "primary_significant_0_05": sum(row["primary_endpoint"] and row["holm_significant_0_05"] for row in statistics),
            "all_significant_0_05": sum(row["holm_significant_0_05"] for row in statistics),
            "output": str(stats_path),
            "sha256": sha256(stats_path),
        },
        "input_manifest": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in sorted(inputs)
        ],
    }
    report_path = output / "v5_response_aware_causal_chain_v3_report.html"
    report_path.write_text(render_report(audit, statistics), encoding="utf-8")
    audit["report"] = str(report_path)
    audit["report_sha256"] = sha256(report_path)
    audit_path = output / "final_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in (stats_path, report_path, audit_path)
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "passed", "output": str(output), "tests": len(statistics),
        "primary_significant": audit["cross_model_holm"]["primary_significant_0_05"],
    }, indent=2))


if __name__ == "__main__":
    main()
