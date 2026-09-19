#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "realistic_niah_v5_native_thinking_causal_chain_report_v1"
MODELS = ("Qwen3-8B", "Gemma4-E4B")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL {path}:{line_number}: {error}") from error
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def fmt(value: Any, digits: int = 3) -> str:
    parsed = number(value)
    if not finite(parsed):
        return "NA"
    if parsed != 0 and abs(parsed) < 0.001:
        return f"{parsed:.2e}"
    return f"{parsed:.{digits}f}"


def esc(value: Any) -> str:
    return html.escape(str(value))


def table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>")
    return (
        "<div class='table-scroll'><table><thead><tr>"
        + "".join(f"<th>{esc(header)}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def request_count(rows: list[dict[str, Any]]) -> int:
    return len(
        {
            str(row.get("request_id", row.get("stimulus_id", row.get("pair_id", ""))))
            for row in rows
        }
    )


def summarize_representation(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    summaries = []
    for site in sorted({row["site_kind"] for row in rows}):
        active = [row for row in rows if row["site_kind"] == site and row.get("probe") == "ridge"]
        if not active:
            continue
        best = max(active, key=lambda row: number(row.get("confirmation_r2")))
        summaries.append(
            {
                "site_kind": site,
                "best_layer": int(float(best["layer"])),
                "confirmation_r2": number(best["confirmation_r2"]),
                "confirmation_mae": number(best["confirmation_mae"]),
                "n_discovery": int(float(best["n_discovery"])),
                "n_confirmation": int(float(best["n_confirmation"])),
            }
        )
    return summaries


def representation_at_layer(
    path: Path, site_kind: str, layer: int, *, cohort: str = "one_to_one"
) -> dict[str, Any]:
    rows = [
        row for row in read_csv(path)
        if row["site_kind"] == site_kind
        and row.get("probe") == "ridge"
        and int(float(row["layer"])) == int(layer)
    ]
    cohort_rows = [row for row in rows if row.get("cohort") == cohort]
    if cohort_rows:
        rows = cohort_rows
    if len(rows) != 1:
        raise ValueError(
            f"Expected one {site_kind}/L{layer}/{cohort} ridge row in {path}; "
            f"found {len(rows)}"
        )
    row = rows[0]
    return {
        "site_kind": site_kind,
        "cohort": row.get("cohort"),
        "layer": int(float(row["layer"])),
        "confirmation_r2": number(row["confirmation_r2"]),
        "confirmation_mae": number(row["confirmation_mae"]),
        "n_discovery": int(float(row["n_discovery"])),
        "n_confirmation": int(float(row["n_confirmation"])),
    }


def summarize_geometry(
    path: Path, site_kind: str, layer: int, *, cohort: str = "one_to_one"
) -> dict[str, Any]:
    rows = [
        row for row in read_csv(path)
        if row["site_kind"] == site_kind and int(float(row["layer"])) == int(layer)
    ]
    cohort_rows = [row for row in rows if row.get("cohort") == cohort]
    if cohort_rows:
        rows = cohort_rows
    if not rows:
        return {}
    if len(rows) != 1:
        raise ValueError(
            f"Expected one {site_kind}/L{layer}/{cohort} geometry row in {path}; "
            f"found {len(rows)}"
        )
    best = rows[0]
    return {
        "layer": int(float(best["layer"])),
        "centroid_rank_3_fraction": number(best["centroid_rank_3_fraction"]),
        "label_distance_spearman": number(best["label_distance_spearman"]),
        "within_label_fraction": number(best["within_label_fraction"]),
    }


def summarize_head_statistics(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    return sorted(
        rows,
        key=lambda row: (
            row.get("analysis_population", ""),
            row.get("endpoint", ""),
            int(float(row.get("bank_size", 0))),
        ),
    )


def summarize_execution_statistics(path: Path) -> list[dict[str, Any]]:
    return sorted(
        read_csv(path),
        key=lambda row: (
            row.get("analysis_population", ""),
            row.get("endpoint", ""),
            row.get("treatment", ""),
        ),
    )


def status_word(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else "missing"


def model_payload(run_root: Path, model: str) -> dict[str, Any]:
    root = run_root / model
    ext = root / "answer_query_extension"
    required = {
        "capture_primary": ext / "representation/capture_primary/capture_index.jsonl",
        "capture_primary_exclusions": ext / "representation/capture_primary/capture_exclusions.jsonl",
        "capture_supplement": ext / "representation/capture_supplement_n10/capture_index.jsonl",
        "capture_supplement_exclusions": ext / "representation/capture_supplement_n10/capture_exclusions.jsonl",
        "representation_regression": ext / "representation/analysis_primary/regression_confirmation.csv",
        "representation_geometry": ext / "representation/analysis_primary/geometry_summary.csv",
        "attention_primary": ext / "head_ablation/attention_primary.csv",
        "attention_supplement": ext / "head_ablation/attention_supplement_n10.csv",
        "head_plan": ext / "head_ablation/plan/answer_query_causal_plan.csv",
        "head_trials_primary": ext / "head_ablation/trials_primary_confirmation.jsonl",
        "head_trials_supplement": ext / "head_ablation/trials_supplement_n10_confirmation.jsonl",
        "head_primary_statistics": ext / "analysis/head_ablation_primary_confirmation_statistics.csv",
        "head_supplement_statistics": ext / "analysis/head_ablation_supplement_n10_confirmation_statistics.csv",
        "execution_layer": ext / f"answer_execution/plan/{model}__answer_execution_layer_selection.json",
        "execution_pairs": ext / f"answer_execution/plan/{model}__answer_execution_pairs.jsonl",
        "execution_trials": ext / "answer_execution/trials_confirmation.jsonl",
        "execution_statistics": ext / "analysis/answer_execution_statistics.csv",
        "analysis_audit": ext / "analysis/answer_query_extension_analysis_audit.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{model} missing required report inputs: {missing}")

    primary_capture = read_jsonl(required["capture_primary"])
    primary_exclusions = read_jsonl(required["capture_primary_exclusions"])
    supplement_capture = read_jsonl(required["capture_supplement"])
    supplement_exclusions = read_jsonl(required["capture_supplement_exclusions"])
    attention_primary = read_csv(required["attention_primary"])
    attention_supplement = read_csv(required["attention_supplement"])
    head_plan = read_csv(required["head_plan"])
    head_primary_trials = read_jsonl(required["head_trials_primary"])
    head_supplement_trials = read_jsonl(required["head_trials_supplement"])
    execution_pairs = read_jsonl(required["execution_pairs"])
    execution_trials = read_jsonl(required["execution_trials"])
    original_regression = root / "representation/regression_confirmation.csv"
    original_geometry = root / "representation/geometry_summary.csv"
    first_later = root / "position_head_analysis/first_vs_later_confirmation_summary.csv"
    first_locator = root / "position_head_analysis/targeted_vs_first_locator_confirmation_summary.csv"
    prior_inputs = [
        root / "representation/representation_audit.json",
        root / "position_head_analysis/position_head_analysis_audit.json",
        root / "causal/head_tests/head_causal_audit.json",
        root / "causal/head_tests_item_end_fallback_v2/head_causal_audit.json",
        root / "attention/pre_city_token/attention.audit.json",
        root / "causal/pre_city_token/head_tests/pre_city_causal_audit.json",
    ]
    sources = list(required.values()) + [
        path for path in [original_regression, original_geometry, first_later, first_locator, *prior_inputs]
        if path.exists()
    ]
    aliases = {str(row.get("alignment_strategy", "")) for row in attention_primary}
    request_alias_rows = {
        str(row["request_id"]): row for row in attention_primary
    }
    legacy_present = sum(
        str(row.get("legacy_answer_query_present", "")).lower() == "true"
        for row in request_alias_rows.values()
    )
    legacy_same = sum(
        str(row.get("legacy_answer_query_same_endpoint", "")).lower() == "true"
        for row in request_alias_rows.values()
    )
    layer_payload = read_json(required["execution_layer"])
    selected_layer = int(layer_payload["selected_layer"])
    answer_rep = representation_at_layer(
        required["representation_regression"],
        "answer_query_v2",
        selected_layer,
        cohort="one_to_one",
    )
    answer_geometry = summarize_geometry(
        required["representation_geometry"],
        "answer_query_v2",
        selected_layer,
        cohort="one_to_one",
    )
    return {
        "model": model,
        "root": root,
        "ext": ext,
        "supervisor": status_word(ext / "logs/supervisor.status"),
        "primary_capture_requests": request_count(primary_capture),
        "primary_exclusions": len(primary_exclusions),
        "supplement_capture_requests": request_count(supplement_capture),
        "supplement_exclusions": len(supplement_exclusions),
        "aliases": sorted(aliases),
        "legacy_present": legacy_present,
        "legacy_same": legacy_same,
        "attention_primary_requests": len({row["request_id"] for row in attention_primary}),
        "attention_primary_rows": len(attention_primary),
        "attention_supplement_requests": len({row["request_id"] for row in attention_supplement}),
        "attention_supplement_rows": len(attention_supplement),
        "head_plan_rows": len(head_plan),
        "head_primary_trials": len(head_primary_trials),
        "head_primary_requests": request_count(head_primary_trials),
        "head_supplement_trials": len(head_supplement_trials),
        "head_supplement_requests": request_count(head_supplement_trials),
        "head_primary_stats": summarize_head_statistics(required["head_primary_statistics"]),
        "head_supplement_stats": summarize_head_statistics(required["head_supplement_statistics"]),
        "execution_selected_layer": selected_layer,
        "execution_selected_rank": int(layer_payload["selected_rank"]),
        "execution_pairs": len(execution_pairs),
        "execution_seeds": len({int(row["seed"]) for row in execution_pairs}),
        "execution_trials": len(execution_trials),
        "execution_stats": summarize_execution_statistics(required["execution_statistics"]),
        "answer_rep": answer_rep,
        "answer_geometry": answer_geometry,
        "original_rep": summarize_representation(original_regression) if original_regression.exists() else [],
        "first_later": read_csv(first_later) if first_later.exists() else [],
        "first_locator": read_csv(first_locator) if first_locator.exists() else [],
        "prior_audits": {str(path.relative_to(run_root)): read_json(path) for path in prior_inputs if path.exists()},
        "sources": sources,
    }


def significant_count(rows: list[dict[str, Any]], endpoint: str, population: str) -> tuple[int, int]:
    active = [
        row for row in rows
        if row.get("endpoint") == endpoint and row.get("analysis_population") == population
    ]
    positive_significant = sum(
        number(row.get("effect")) > 0 and number(row.get("holm_p")) <= 0.05 for row in active
    )
    return positive_significant, len(active)


def render_model(payload: dict[str, Any]) -> str:
    answer_rep = payload["answer_rep"]
    geometry = payload["answer_geometry"]
    head_sig, head_total = significant_count(
        payload["head_primary_stats"],
        "ranked_minus_random_logp_damage",
        "all_one_to_one",
    )
    execution_sig = [
        row for row in payload["execution_stats"]
        if row.get("analysis_population") == "all_one_to_one"
        and row.get("endpoint") == "treatment_minus_control_transport_gain"
        and number(row.get("effect")) > 0
        and number(row.get("holm_p")) <= 0.05
    ]
    def head_table_rows(statistics: list[dict[str, Any]]) -> list[list[Any]]:
        rows = []
        for row in statistics:
            if row.get("endpoint") != "ranked_minus_random_logp_damage":
                continue
            rows.append(
                [
                    row.get("analysis_population"),
                    row.get("bank_size"),
                    fmt(row.get("effect")),
                    f"[{fmt(row.get('ci95_low'))}, {fmt(row.get('ci95_high'))}]",
                    fmt(row.get("holm_p")),
                    fmt(row.get("ranked_target_needle_raw_mass")),
                    fmt(row.get("ranked_target_needle_relative_mass")),
                    fmt(row.get("random_target_needle_raw_mass")),
                    fmt(row.get("random_target_needle_relative_mass")),
                    row.get("seed_clusters"),
                ]
            )
        return rows

    head_rows = head_table_rows(payload["head_primary_stats"])
    supplement_head_rows = head_table_rows(payload["head_supplement_stats"])
    head_headers = [
        "Population", "K", "Ranked−random log-p damage",
        "95% seed bootstrap CI", "Holm p", "Ranked raw mass",
        "Ranked relative mass", "Random raw mass", "Random relative mass",
        "Seed clusters",
    ]
    execution_rows = []
    for row in payload["execution_stats"]:
        execution_rows.append(
            [
                row.get("analysis_population"),
                row.get("treatment"),
                row.get("endpoint"),
                fmt(row.get("effect")),
                f"[{fmt(row.get('ci95_low'))}, {fmt(row.get('ci95_high'))}]",
                fmt(row.get("holm_p")),
                row.get("seed_clusters"),
            ]
        )
    model_conclusion = (
        f"在 discovery 冻结的 execution layer 上，answer-query state 可解码"
        f"（held-out ridge R²={fmt(answer_rep.get('confirmation_r2'))}）；"
        f"注册 K 中 {head_sig}/{head_total} 个在 ranked-vs-random log-probability damage 上为正且通过 Holm；"
        f"两种 answer-state patch 中 {len(execution_sig)}/2 个在 transport specificity 上为正且通过 Holm。"
    )
    section_number = "3" if payload["model"] == "Qwen3-8B" else "4"
    return f"""
    <section id='{esc(payload['model'])}'>
      <h2>{section_number} · {esc(payload['model'])}</h2>
      <div class='conclusion'><strong>模型级因果链判定。</strong>{esc(model_conclusion)}</div>
      <h3>Answer-query 位点有效性与样本覆盖</h3>
      {table(
          ["Supervisor", "主集 captured", "主集 exclusions", "补样 captured", "补样 exclusions", "Alignment", "旧位点存在 / 同 endpoint"],
          [[payload['supervisor'], payload['primary_capture_requests'], payload['primary_exclusions'],
            payload['supplement_capture_requests'], payload['supplement_exclusions'], ', '.join(payload['aliases']),
            f"{payload['legacy_present']} / {payload['legacy_same']}"]],
      )}
      <p class='small'>注册的 <code>answer_query_v2</code> 是第一个数字答案 token 之前最后一个 literal <code>Total:</code> token。旧位点只保留作 alias 审计；旧位点缺失时不会静默换成另一个语义 endpoint。</p>
      <h3>Representation：答案查询处的 consolidation</h3>
      {table(
          ["Site", "Discovery 冻结层", "Cohort", "Discovery n", "Confirmation n", "Held-out ridge R²", "Held-out MAE", "Centroid rank-3 fraction", "Spearman"],
          [[answer_rep.get('site_kind', 'answer_query_v2'), answer_rep.get('layer', 'NA'), answer_rep.get('cohort', 'NA'),
            answer_rep.get('n_discovery', 'NA'), answer_rep.get('n_confirmation', 'NA'),
            fmt(answer_rep.get('confirmation_r2')), fmt(answer_rep.get('confirmation_mae')),
            fmt(geometry.get('centroid_rank_3_fraction')), fmt(geometry.get('label_distance_spearman'))]],
      )}
      <h3>Head necessity：full prompt-needle-span ranked bank vs exact layer-matched random bank</h3>
      <p>只用 discovery 按 answer-query 对 prompt 中 exact needle spans 并集的 attention 排序。Confirmation 测量 clean-minus-ablated teacher-forced answer log-probability damage；正值表示 discovery-ranked bank 比 matched random bank 造成更大损伤。</p>
      {table(head_headers, head_rows)}
      <details><summary>N10 strict one-to-one 补样的独立 confirmation replication</summary>{table(head_headers, supplement_head_rows)}</details>
      <p class='small'>主集 attention：{payload['attention_primary_requests']} requests / {payload['attention_primary_rows']} rows；补样：{payload['attention_supplement_requests']} requests / {payload['attention_supplement_rows']} rows。Head trials：主集 {payload['head_primary_requests']} requests / {payload['head_primary_trials']} rows；补样 {payload['head_supplement_requests']} requests / {payload['head_supplement_trials']} rows。结构上不可构造的 disjoint controls 保持不可估计，不伪造替代 control。</p>
      <h3>Answer execution：discovery 冻结层上的 donor-state transport</h3>
      <p>Layer {payload['execution_selected_layer']} 由 discovery leave-one-seed-out ridge MAE 选择，并冻结 rank-{payload['execution_selected_rank']} discovery centroid basis。Confirmation 只用同 seed 最近的 lower/higher donor pairs。完整 residual 与 projected donor patch 分别对照 self-patch 和 equal-norm orthogonal control。</p>
      {table(["Population", "Treatment", "Endpoint", "Treatment−control", "95% seed bootstrap CI", "Holm p", "Seed clusters"], execution_rows)}
      <p class='small'>Execution 覆盖：{payload['execution_pairs']} pairs，来自 {payload['execution_seeds']} 个 confirmation seeds；共 {payload['execution_trials']} 条原始 condition rows。补充的 N10 样本用于 answer-query 位点与 head-bank replication；因其不能构造同 seed 跨 count donor，故不插入 execution pairs。</p>
    </section>
    """


def build_report(run_root: Path, output_dir: Path) -> dict[str, Any]:
    payloads = [model_payload(run_root, model) for model in MODELS]
    generated = datetime.now(timezone.utc).isoformat()

    def audit_by_suffix(payload: dict[str, Any], suffix: str) -> dict[str, Any]:
        for path, audit in payload["prior_audits"].items():
            if path.replace("\\", "/").endswith(suffix):
                return audit
        return {}

    coverage_rows = [
        [
            payload["model"], payload["primary_capture_requests"], payload["primary_exclusions"],
            payload["supplement_capture_requests"], payload["attention_primary_requests"],
            payload["head_primary_requests"], payload["execution_pairs"], payload["supervisor"],
        ]
        for payload in payloads
    ]
    upstream_rows = []
    for payload in payloads:
        representation = audit_by_suffix(
            payload, "/representation/representation_audit.json"
        )
        position = audit_by_suffix(
            payload, "/position_head_analysis/position_head_analysis_audit.json"
        )
        causal = audit_by_suffix(
            payload, "/causal/head_tests/head_causal_audit.json"
        )
        fallback = audit_by_suffix(
            payload,
            "/causal/head_tests_item_end_fallback_v2/head_causal_audit.json",
        )
        e4 = audit_by_suffix(
            payload, "/causal/pre_city_token/head_tests/pre_city_causal_audit.json"
        )
        upstream_rows.extend(
            [
                [payload["model"], "running representation", representation.get("rows_loaded", "NA"), representation.get("groups_completed", "NA"), "site/cohort frozen before confirmation"],
                [payload["model"], "E1/E2 position heads", "confirmation", "/".join(map(str, position.get("bank_sizes", []))), "discovery-ranked; V4.4 registry frozen"],
                [payload["model"], "main causal", causal.get("ok_rows", "NA"), causal.get("excluded_rows", "NA"), "targeted_retrieval and progress_transition separate"],
                [payload["model"], "progress fallback", fallback.get("ok_rows", "not applicable"), fallback.get("excluded_rows", "not applicable"), "Qwen item_end-only transparent fallback"],
                [payload["model"], "E4 pre-city causal", e4.get("ok_rows", "NA"), e4.get("excluded_rows", "NA"), "variant-specific banks; no broad aggregation"],
            ]
        )
    report_body = f"""
    <header>
      <div class='eyebrow'>Realistic NIAH · V5 native thinking · integrated causal audit</div>
      <h1>从 running-state formation 到 answer execution</h1>
      <p class='lead'>本报告按因果角色组织 CoT 内部证据：prompt-side state formation、retrieval/progress routing、answer-query consolidation、head necessity，以及可执行的 answer-state transport。可解码性不等同于因果性；non-thinking 的 OV 结论也不会被直接移植到 CoT。</p>
      <p class='meta'>生成时间 {esc(generated)} · root {esc(run_root)}</p>
    </header>
    <section id='verdict'>
      <h2>1 · 范围与判定纪律</h2>
      <div class='chain'>
        <div><b>1. Formation</b><span>item-end running / enumeration representation</span></div>
        <div><b>2. Routing</b><span>targeted retrieval、progress transition、position sensitivity</span></div>
        <div><b>3. Consolidation</b><span>最终 <code>Total:</code> query 的 count decodability</span></div>
        <div><b>4. Necessity</b><span>discovery-ranked answer-query heads vs layer controls</span></div>
        <div><b>5. Execution</b><span>donor/projected answer-state transport vs orthogonal controls</span></div>
      </div>
      <div class='warning'><strong>Claim boundary。</strong>两模型的 CoT answer-query representation、head ablation 与 answer execution 均在本报告内检验。按用户要求，本轮不重跑 OV；因此“CoT 与 non-thinking 因为都在 answer token 所以 OV 相同”仍是待检假设，而非实验结论。暂停的 non-thinking↔CoT E3 geometry 也不属于本轮完成声明。</div>
      {table(["Model", "主集 captured", "Exclusions", "N10 supplement", "Attention requests", "Head-test requests", "Execution pairs", "Supervisor"], coverage_rows)}
    </section>
    <section id='upstream'>
      <h2>2 · 本次扩展前已经冻结的上游 CoT 链条</h2>
      <p><b>Formation。</b>主 representation 在注册的 item-end 位点测量 running/enumeration state；probe 只在 discovery 拟合，在 confirmation 评估。</p>
      <p><b>Routing。</b>主 attention/causal 将 <code>targeted_retrieval</code> 与 <code>progress_transition</code> 分开，不使用 broad aggregate bank。E1/E2 在 discovery 冻结 targeted-first 排名，在 confirmation 检验 first-vs-later 及 V4.4 first-locator contrast。E4 对 <code>pre_city_d1</code>、<code>pre_city_d2</code>、<code>pre_city_anchor</code> 分别独立选 head 和测试。</p>
      <p><b>Boundary handling。</b>Qwen 的 strict marker boundary trial 完整保留；item-end fallback 只透明补测 progress-transition，无法构造的 targeted marker-end rows 继续明确排除。</p>
      {table(["Model", "Stage", "完成 / 评估量", "排除 / groups", "冻结规则"], upstream_rows)}
      <div class='conclusion'><strong>解释。</strong>这些阶段定位 count-related state 的可读位置，并在注册 controls 下检验 prompt-side routing。统计支持必须逐 endpoint 判读；表跑完本身不代表 effect 非零。它们也不能单独证明最终 answer query 已包含或执行 count；后续模型小节补上 consolidation、necessity 与 execution。</div>
    </section>
    {''.join(render_model(payload) for payload in payloads)}
    <section id='synthesis'>
      <h2>5 · 按因果链整合两模型</h2>
      <ol class='synthesis'>
        <li><b>Prompt formation：</b>注册 item-end states 在 held-out confirmation seeds 上编码 running/enumeration progress。</li>
        <li><b>Selective routing：</b>targeted retrieval 与 progress transition 分别选择和测试；E1/E2 及 pre-city sensitivity 做角色定位，不跨角色平均。</li>
        <li><b>Answer consolidation：</b>两种 tokenizer 都用同一个审计过的 endpoint 定义抓取 literal pre-answer <code>Total:</code> state，并检验 held-out count decodability。</li>
        <li><b>Head necessity：</b>discovery-ranked heads 只在 answer query 消融，并在可构造处与 disjoint exact layer-matched banks 比较。</li>
        <li><b>Execution/sufficiency：</b>discovery-frozen answer states 在同 seed 不同 count 间搬运；greedy answer generation 下与 self 和 equal-norm orthogonal patches 比较。</li>
      </ol>
      <div class='warning'><strong>更强的 circuit identity claim 还缺什么。</strong>若要断言两种 mode 使用同一个 writer circuit，还需要 CoT-specific pre-O/OV decomposition、downstream mediation，以及恢复 paired non-thinking↔CoT analysis。本报告有意不作这些越界声明。</div>
    </section>
    <section id='audit'>
      <h2>6 · 可复现性与审计</h2>
      <p>所有 heads 与 execution layer/basis 只在 discovery 选择，confirmation 仅评估。推断单位为 seed-cluster mean；使用 20,000 次 seed bootstrap，cluster≤20 时 exact sign flip，超过时使用 deterministic 1,000,000-draw sign flip。Head tests 的 Holm family 跨 K，execution 的 family 跨 treatment。</p>
      <p><b>数值限制。</b>高维 Ridge 的 seed-CV 日志出现 <code>LinAlgWarning: ill-conditioned matrix</code>。流水线不会吞掉该警告；最终审计要求统计表为有限值，但靠近奇异的拟合仍应被视为 conditioning sensitivity，而不是额外的机制证据。</p>
      <p>原始 V5 结果、Qwen 旧 supplement 与新的 strict one-to-one supplements 继续目录隔离；相邻 manifest 对全部报告输入逐文件哈希。</p>
    </section>
    """
    css = """
    :root{--ink:#171923;--muted:#5f6b7a;--paper:#f7f8fb;--surface:#fff;--line:#d8deea;--violet:#5946d2;--teal:#008f7a;--amber:#8a5a00}
    *{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 Aptos,"Segoe UI",sans-serif}main{max-width:1180px;margin:auto;padding:42px 30px 90px}header{border-bottom:4px solid #24175c;padding:28px 0 34px}.eyebrow{font:700 12px Consolas,monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--violet)}h1,h2,h3{font-family:Georgia,serif;line-height:1.2}h1{font-size:43px;max-width:850px;margin:10px 0 16px}h2{font-size:29px;margin-top:0}h3{font-size:20px;margin-top:30px}.lead{font-size:18px;max-width:90ch;color:#3f4a5a}.meta,.small{font-size:12px;color:var(--muted)}section{padding:48px 0;border-bottom:1px solid var(--line)}.chain{display:grid;grid-template-columns:repeat(5,1fr);border:1px solid var(--line);background:var(--surface);margin:22px 0}.chain div{padding:16px;border-right:1px solid var(--line)}.chain div:last-child{border:0}.chain b,.chain span{display:block}.chain b{color:#24175c;margin-bottom:6px}.chain span{font-size:13px;color:var(--muted)}.conclusion,.warning{background:var(--surface);border-left:4px solid var(--teal);padding:16px 19px;margin:20px 0;box-shadow:0 7px 20px rgba(30,40,70,.04)}.conclusion strong,.warning strong{display:block;margin-bottom:5px}.warning{border-left-color:#d29b22}.table-scroll{overflow:auto;border:1px solid var(--line);background:var(--surface);margin:16px 0}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px 11px;border-bottom:1px solid #e4e8f0;text-align:left;vertical-align:top}th{background:#edf0f6;color:#303744;position:sticky;top:0}details{background:var(--surface);border:1px solid var(--line);margin:17px 0}summary{cursor:pointer;padding:12px 14px;color:#24175c;font-weight:700}details .table-scroll{border-width:1px 0 0;margin:0}code{font-family:Consolas,monospace;background:#eceef3;padding:1px 4px}.synthesis li{margin:10px 0;max-width:95ch}@media(max-width:850px){main{padding:25px 15px 60px}.chain{grid-template-columns:1fr}.chain div{border-right:0;border-bottom:1px solid var(--line)}h1{font-size:34px}}
    """
    document = (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>V5 native-thinking integrated causal chain</title>"
        f"<style>{css}</style></head><body><main>{report_body}</main></body></html>"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "v5_native_thinking_integrated_causal_chain_report.html"
    html_path.write_text(document, encoding="utf-8")
    source_paths = sorted({path.resolve() for payload in payloads for path in payload["sources"]})
    manifest = {
        "schema_version": SCHEMA,
        "generated_at": generated,
        "run_root": str(run_root.resolve()),
        "report": str(html_path.resolve()),
        "models": list(MODELS),
        "causal_order": ["formation", "routing", "consolidation", "necessity", "execution"],
        "selection_split": "discovery",
        "confirmation_used_for_selection": False,
        "ov_comparison_status": "not_run_by_user_request",
        "cross_mode_e3_status": "paused_by_user_request",
        "report_sha256": sha256(html_path),
        "inputs": [
            {"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)}
            for path in source_paths
        ],
    }
    manifest_path = output_dir / "v5_native_thinking_integrated_causal_chain_report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args.run_root, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
