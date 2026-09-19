#!/usr/bin/env python3
"""Build and audit the aligned V5 native-thinking causal-chain report.

This entry point consumes only ``corrected_causal_chain_v2`` outputs.  It
rechecks the aligned marker/answer registries, applies Holm correction across
both models without mixing distinct endpoints, and writes a hash-addressed
HTML report plus machine-readable audit files.
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
VARIANTS = ("pre_city_d1", "pre_city_d2", "pre_city_anchor")
ANSWER_CONDITIONS = (
    "clean",
    "prompt_aggregation_ablation",
    "trace_aggregation_ablation",
    "joint_prompt_and_trace_aggregation_ablation",
)
EXPECTED_MARKER_STRATA = {
    f"{split}:N{left}_to_N{right}"
    for split in ("discovery", "confirmation")
    for left, right in ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10))
}
EXPECTED_GAPS = {-3: 1, -2: 3, -1: 24, 1: 24, 2: 3, 3: 1}
SCHEMA = "realistic_niah_v5_native_causal_chain_v2_report_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL {path}:{line_number}: {error}") from error
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing CSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, f"empty CSV: {path}")
    return rows


def as_float(value: Any) -> float:
    parsed = float(value)
    require(math.isfinite(parsed), f"non-finite statistic: {value!r}")
    return parsed


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def holm(values: list[float]) -> list[float]:
    require(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values), "invalid p-value")
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [float("nan")] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, values[index] * (len(values) - rank)))
        adjusted[index] = running
    return adjusted


def audit_registry(run_root: Path, inputs: set[Path]) -> dict[str, Any]:
    align_root = run_root / "patch_alignment_v1"
    audit_path = align_root / "alignment_audit.json"
    audit = read_json(audit_path)
    inputs.add(audit_path)
    require(audit.get("status") == "passed", "alignment audit did not pass")
    require(audit["marker"].get("query_constructability_filter") == "pre_city_d1/pre_city_d2/pre_city_anchor all required", "wrong marker query filter")

    result: dict[str, Any] = {"status": "passed", "models": {}}
    for model in MODELS:
        marker_path = align_root / "marker" / f"{model}__marker_aligned_pairs.jsonl"
        answer_path = align_root / "answer" / f"{model}__answer_aligned_pairs.jsonl"
        query_audit_path = align_root / "query_audit" / f"{model}.jsonl.audit.json"
        marker_rows = read_jsonl(marker_path)
        answer_rows = read_jsonl(answer_path)
        query_audit = read_json(query_audit_path)
        inputs.update({marker_path, answer_path, query_audit_path})

        require(query_audit.get("status") == "passed", f"{model}: query constructability audit failed")
        require(len(marker_rows) == 10, f"{model}: expected 10 marker pairs")
        require(len({row["pair_id"] for row in marker_rows}) == 10, f"{model}: duplicate marker pair_id")
        strata = Counter(str(row["alignment_stratum"]) for row in marker_rows)
        require(set(strata) == EXPECTED_MARKER_STRATA and set(strata.values()) == {1}, f"{model}: marker strata mismatch")
        require(all(row.get("model_label") == model for row in marker_rows), f"{model}: marker model mismatch")

        require(len(answer_rows) == 56, f"{model}: expected 56 answer pairs")
        require(len({row["pair_id"] for row in answer_rows}) == 56, f"{model}: duplicate answer pair_id")
        gaps = Counter(int(row["signed_count_gap"]) for row in answer_rows)
        require(dict(sorted(gaps.items())) == EXPECTED_GAPS, f"{model}: signed-gap distribution mismatch: {gaps}")
        require(all(row.get("model_label") == model for row in answer_rows), f"{model}: answer model mismatch")
        require(all(row.get("pair_eligibility") == "receiver_and_donor_baseline_final_answer_exact" for row in answer_rows), f"{model}: non-correct answer pair")
        require(all(row.get("selection_split") == "discovery" and not bool(row.get("confirmation_used_for_layer_selection")) for row in answer_rows), f"{model}: answer selection leakage")

        for family, path, rows in (
            ("marker", marker_path, marker_rows),
            ("answer", answer_path, answer_rows),
        ):
            expected = audit["outputs"][family][model]
            require(len(rows) == int(expected["rows"]), f"{model}/{family}: audit row mismatch")
            require(sha256(path) == expected["sha256"], f"{model}/{family}: registry hash mismatch")
        result["models"][model] = {
            "marker_pairs": len(marker_rows),
            "marker_strata": dict(sorted(strata.items())),
            "answer_pairs": len(answer_rows),
            "answer_signed_gaps": {str(key): value for key, value in sorted(gaps.items())},
            "query_constructability_status": query_audit["status"],
        }
    return result


def normalized_row(
    row: dict[str, str], *, stage: str, mechanism: str, scope: str,
    source: Path, primary_key: str, bank_size: str = "", signed_gap: str = "",
) -> dict[str, Any]:
    return {
        "model_label": row["model_label"],
        "stage": stage,
        "mechanism": mechanism,
        "scope": scope,
        "bank_size": bank_size,
        "signed_count_gap": signed_gap,
        "metric": row["metric"],
        "primary_endpoint": primary_key == "__all__" or truthy(row.get(primary_key, "false")),
        "seed_clusters": int(float(row["seed_clusters"])),
        "effect": as_float(row["effect"]),
        "ci95_low": as_float(row["ci95_low"]),
        "ci95_high": as_float(row["ci95_high"]),
        "sign_flip_p": as_float(row["sign_flip_p"]),
        "sign_flip_method": row["sign_flip_method"],
        "sign_flip_assignments": int(float(row["sign_flip_assignments"])),
        "holm_p_within_model": as_float(row["holm_p_within_model"]),
        "source_statistics": str(source.resolve()),
        "source_sha256": sha256(source),
    }


def audit_model(run_root: Path, model: str, inputs: set[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = run_root / model / "corrected_causal_chain_v2"
    status_path = root / "logs" / "supervisor.status"
    require(status_path.read_text(encoding="utf-8").strip() == "complete", f"{model}: supervisor incomplete")
    inputs.add(status_path)

    audits = {
        "marker_patch": root / "marker_adjacent_patch/analysis/audit.json",
        "answer_aggregation": root / "answer_aggregation_factorial/analysis/audit.json",
        "answer_execution": root / "answer_execution_reanalysis/analysis/audit.json",
        **{
            f"next_needle_{variant}": root / f"next_needle_ablation/{variant}/analysis/audit.json"
            for variant in VARIANTS
        },
    }
    payload = {name: read_json(path) for name, path in audits.items()}
    inputs.update(audits.values())
    require(all(audit.get("status") == "passed" for audit in payload.values()), f"{model}: failed component audit")
    require(payload["marker_patch"].get("selection_split") == "discovery", f"{model}: marker selection split")
    require(payload["marker_patch"].get("confirmation_used_for_selection") is False, f"{model}: marker confirmation leakage")
    require(tuple(payload["answer_aggregation"].get("answer_aggregation_conditions", [])) == ANSWER_CONDITIONS, f"{model}: wrong answer four-condition report")
    require(
        payload["answer_execution"].get("pair_eligibility")
        == "receiver and donor original final answers exact; self-patch must regenerate receiver gold",
        f"{model}: wrong answer pair eligibility",
    )
    require(all(payload[f"next_needle_{variant}"].get("final_count_evaluated") is False for variant in VARIANTS), f"{model}: trace-local test evaluated final count")

    stats: list[dict[str, Any]] = []
    marker_path = root / "marker_adjacent_patch/analysis/confirmation_statistics.csv"
    inputs.add(marker_path)
    for row in read_csv(marker_path):
        stats.append(normalized_row(
            row, stage="marker_hidden_state_patch", mechanism=row["query_variant"],
            scope="confirmation_selected_layer", source=marker_path,
            primary_key="primary_endpoint",
        ))
    for variant in VARIANTS:
        path = root / f"next_needle_ablation/{variant}/analysis/statistics.csv"
        inputs.add(path)
        for row in read_csv(path):
            stats.append(normalized_row(
                row, stage="trace_local_targeted_retrieval_ablation", mechanism=variant,
                scope="confirmation_occurrence", source=path, primary_key="primary_endpoint",
                bank_size=row["bank_size"],
            ))
    aggregation_path = root / "answer_aggregation_factorial/analysis/statistics.csv"
    inputs.add(aggregation_path)
    for row in read_csv(aggregation_path):
        stats.append(normalized_row(
            row, stage="answer_aggregation_ablation", mechanism=row["family"],
            scope="confirmation_actual_greedy_count", source=aggregation_path,
            primary_key="is_primary_endpoint", bank_size=row["bank_size"],
        ))
    execution_path = root / "answer_execution_reanalysis/analysis/statistics.csv"
    inputs.add(execution_path)
    for row in read_csv(execution_path):
        stats.append(normalized_row(
            row, stage="answer_hidden_state_patch", mechanism="answer_state_transport",
            scope=row["scope"], source=execution_path, primary_key="__all__",
            signed_gap=row.get("signed_count_gap", ""),
        ))

    summary = {
        "status": "passed",
        "supervisor": "complete",
        "component_audits": {name: audit["status"] for name, audit in payload.items()},
        "answer_conditions": list(ANSWER_CONDITIONS),
        "answer_four_condition_rows": int(payload["answer_aggregation"]["answer_aggregation_four_condition_rows"]),
        "answer_execution_pairs": int(payload["answer_execution"]["completed_pairs"]),
        "statistics_rows": len(stats),
    }
    return summary, stats


def apply_cross_model_holm(rows: list[dict[str, Any]]) -> None:
    identities: set[tuple[Any, ...]] = set()
    families: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        identity = (
            row["model_label"], row["stage"], row["mechanism"], row["scope"],
            row["bank_size"], row["signed_count_gap"], row["metric"],
        )
        require(identity not in identities, f"duplicate statistical test: {identity}")
        identities.add(identity)
        family = "::".join((row["stage"], row["mechanism"], row["scope"], row["metric"]))
        row["holm_family"] = family
        families[family].append(index)
    for family, indices in families.items():
        adjusted = holm([rows[index]["sign_flip_p"] for index in indices])
        for index, value in zip(indices, adjusted):
            rows[index]["holm_family_size"] = len(indices)
            rows[index]["holm_p_cross_model"] = value
            rows[index]["holm_significant_0_05"] = value <= 0.05
    rows.sort(key=lambda row: (
        row["stage"], row["mechanism"], row["scope"], row["metric"],
        row["model_label"], str(row["bank_size"]), str(row["signed_count_gap"]),
    ))


def write_statistics(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model_label", "stage", "mechanism", "scope", "bank_size",
        "signed_count_gap", "metric", "primary_endpoint", "seed_clusters",
        "effect", "ci95_low", "ci95_high", "sign_flip_p", "sign_flip_method",
        "sign_flip_assignments", "holm_p_within_model", "holm_family",
        "holm_family_size", "holm_p_cross_model", "holm_significant_0_05",
        "source_statistics", "source_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def html_table(headers: list[str], body: Iterable[Iterable[Any]]) -> str:
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(fmt(cell))}</td>" for cell in row) + "</tr>"
        for row in body
    )
    return "<div class='scroll'><table><thead><tr>" + "".join(
        f"<th>{html.escape(header)}</th>" for header in headers
    ) + f"</tr></thead><tbody>{rows}</tbody></table></div>"


def render_report(
    run_root: Path, model_audits: dict[str, Any], registry: dict[str, Any],
    rows: list[dict[str, Any]], generated_at: str,
) -> str:
    primary = [row for row in rows if row["primary_endpoint"]]
    if not primary:
        primary = rows
    stage_summary = []
    for model in MODELS:
        for stage in sorted({row["stage"] for row in rows}):
            active = [row for row in rows if row["model_label"] == model and row["stage"] == stage]
            stage_summary.append((model, stage, len(active), sum(bool(row["holm_significant_0_05"]) for row in active)))
    stat_rows = [
        (
            row["model_label"], row["stage"], row["mechanism"], row["scope"],
            row["bank_size"], row["signed_count_gap"], row["metric"],
            row["effect"], f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]",
            row["sign_flip_p"], row["holm_p_cross_model"], row["seed_clusters"],
        )
        for row in primary
    ]
    model_rows = [
        (
            model, model_audits[model]["supervisor"],
            registry["models"][model]["marker_pairs"],
            registry["models"][model]["answer_pairs"],
            model_audits[model]["answer_four_condition_rows"],
            model_audits[model]["answer_execution_pairs"],
        )
        for model in MODELS
    ]
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>V5 native-thinking aligned causal chain v2</title>
<style>
body{{margin:0;background:#f5f6fa;color:#171a24;font:15px/1.6 'Segoe UI',sans-serif}}main{{max-width:1200px;margin:auto;padding:38px 28px 80px}}h1,h2{{font-family:Georgia,serif}}h1{{font-size:40px}}section{{background:#fff;border:1px solid #dce1eb;margin:18px 0;padding:25px}}.lead{{font-size:18px;color:#3f4958}}.chain{{display:grid;grid-template-columns:repeat(4,1fr);gap:9px}}.chain div{{background:#eef0fb;padding:14px;border-top:3px solid #5746c7}}.ok{{border-left:4px solid #078b73;background:#effaf7;padding:13px}}.warn{{border-left:4px solid #c58a1d;background:#fff8e9;padding:13px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border-bottom:1px solid #e1e5ed;padding:8px;text-align:left;vertical-align:top}}th{{background:#eef1f6;position:sticky;top:0}}code{{background:#eceef3;padding:1px 4px}}@media(max-width:800px){{.chain{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><div>Realistic NIAH · V5 native thinking · aligned causal audit</div><h1>CoT 内部因果链：检索、聚合与答案执行</h1><p class='lead'>报告只消费 <code>corrected_causal_chain_v2</code>。Head discovery 与 layer selection 只使用 discovery；confirmation 只检验。行为主指标均来自模型实际 greedy 输出，不使用候选 logit argmax。</p><p>生成时间：{html.escape(generated_at)}<br>Run root：{html.escape(str(run_root.resolve()))}</p></header>
<section><h2>1. 最终审计结论</h2><div class='ok'><b>PASS。</b> 两模型 supervisor、所有组件 audit、10-pair marker registry、56-pair answer registry、query constructability、四条件 answer ablation、correct-only answer patch 与跨模型 Holm 均通过。</div>{html_table(['Model','Supervisor','Marker pairs','Answer pairs','Four-condition rows','Execution pairs'], model_rows)}</section>
<section><h2>2. 注册因果链</h2><div class='chain'><div><b>Marker state transport</b><br>N=k 与删去一个 needle 的同源 prompt/trace，在 pre-city query 交换 hidden state；endpoint 是下一 needle。</div><div><b>Trace-local necessity</b><br>在每个 pre-city 位置消融 targeted-retrieval heads；只评价后续 trace 的 needle 准确率，不评价 final count。</div><div><b>Answer aggregation</b><br>答案 token 前分别报告 clean、prompt heads、trace heads、joint heads；endpoint 是实际 greedy count。</div><div><b>Answer execution</b><br>仅在 receiver/donor baseline 均正确的样本间 patch；按 signed gap 报告 count transport/adoption，并对照 self 与 equal-norm orthogonal。</div></div><p class='warn'><b>未越界声明：</b>OV 未比较；non-thinking↔CoT E3 geometry 按用户要求暂停。本报告不将它们伪装为已完成证据。</p></section>
<section><h2>3. 跨模型 multiplicity</h2><p>Holm family 不混合不同 endpoint：marker 按 variant×metric；trace-local 按 variant×metric 并跨模型/K；answer aggregation 按 prompt/trace/joint family×metric 并跨模型/K；answer execution 按 scope×metric，signed-gap scope 内跨模型/gap。</p>{html_table(['Model','Stage','Tests','Cross-Holm significant'], stage_summary)}</section>
<section><h2>4. 注册 primary endpoints</h2>{html_table(['Model','Stage','Mechanism','Scope','K','Gap','Metric','Effect','95% CI','Raw p','Cross-Holm p','Seed clusters'], stat_rows)}</section>
<section><h2>5. 解释边界</h2><ul><li>显著的 targeted-retrieval ablation 支持这些 heads 对局部下一-needle 生成具有必要性；它不单独证明最终 count 由同一 head bank直接执行。</li><li>Answer 四条件区分 prompt-sequence aggregation、thinking-trace aggregation及其 joint union；主要行为指标是解析后的实际输出 count。</li><li>Marker patch 检验 hidden state 是否携带可交换的 needle 信息；answer patch 检验 count state 的 transport/adoption。两者 endpoint 不可互换。</li><li>每一统计结论应以 effect、CI 与跨模型 Holm p 联合解释；“流水线完成”不等于“所有机制效应显著”。</li></ul></section>
</main></body></html>"""


def build(run_root: Path, output_dir: Path) -> dict[str, Any]:
    inputs: set[Path] = set()
    registry = audit_registry(run_root, inputs)
    model_audits: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        model_audits[model], model_rows = audit_model(run_root, model, inputs)
        rows.extend(model_rows)
    apply_cross_model_holm(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = output_dir / "combined_statistics_cross_model_holm.csv"
    report_path = output_dir / "v5_native_thinking_aligned_causal_chain_v2_report.html"
    audit_path = output_dir / "final_audit.json"
    manifest_path = output_dir / "manifest.json"
    write_statistics(statistics_path, rows)
    generated_at = datetime.now(timezone.utc).isoformat()
    report_path.write_text(
        render_report(run_root, model_audits, registry, rows, generated_at),
        encoding="utf-8",
    )
    audit = {
        "schema_version": SCHEMA,
        "status": "passed",
        "generated_at": generated_at,
        "models": model_audits,
        "registry": registry,
        "behavioral_endpoint_policy": "actual greedy generated text parsed as count; never candidate-logit argmax",
        "selection_policy": "discovery-only selection; confirmation evaluation only",
        "cross_model_holm": {
            "tests": len(rows),
            "families": len({row["holm_family"] for row in rows}),
            "significant_0_05": sum(bool(row["holm_significant_0_05"]) for row in rows),
            "family_definition": "stage x mechanism/query-variant x scope x endpoint; no endpoint mixing",
            "output": str(statistics_path.resolve()),
            "sha256": sha256(statistics_path),
        },
        "ov_comparison_status": "not_run_by_user_request",
        "cross_mode_e3_status": "paused_by_user_request",
        "report": str(report_path.resolve()),
        "report_sha256": sha256(report_path),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": f"{SCHEMA}_manifest",
        "generated_at": generated_at,
        "status": "passed",
        "run_root": str(run_root.resolve()),
        "outputs": [
            {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256(path)}
            for path in (statistics_path, report_path, audit_path)
        ],
        "inputs": [
            {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(inputs)
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**audit, "manifest": str(manifest_path.resolve()), "manifest_sha256": sha256(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.run_root, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
