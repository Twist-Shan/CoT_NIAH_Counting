#!/usr/bin/env python3
"""Build a self-contained report for the four V6 answer/trace extension cells."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from realistic_niah_v6.answer_trace_extension import (  # noqa: E402
    load_relay_geometry_amendment,
)


PROMPT_MODES = ("enumeration_index", "enumeration_bullet")
MODE_LABELS = {
    "enumeration_index": "Index enumeration",
    "enumeration_bullet": "Bullet enumeration",
}
MODELS = ("Qwen3-8B", "Gemma4-E4B")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def _gate(gates: dict[str, Any], name: str) -> dict[str, Any]:
    value = gates["gates"][name]
    estimate = value.get("estimate", value.get("mean", value.get("value")))
    low = value.get("ci95_low", value.get("ci_low"))
    high = value.get("ci95_high", value.get("ci_high"))
    result = {
        "pass": bool(value["pass"]),
        "estimate": estimate,
        "low": low,
        "high": high,
    }
    for field in ("rule", "seed_count", "relative_equivalence_bound", "role"):
        if field in value:
            result[field] = value[field]
    return result


def _relay_support(audit: dict[str, Any], *, geometry: str) -> dict[str, Any]:
    eligible = int(audit.get("eligible_seed_count", -1))
    return {
        "geometry": geometry,
        "estimable": bool(audit.get("relay_estimable", eligible > 0)),
        "planned_seed_count": int(audit.get("planned_seed_count", -1)),
        "eligible_seed_count": eligible,
        "geometry_not_applicable_full_seed_count": int(
            audit.get("geometry_not_applicable_full_seed_count", -1)
        ),
        "geometry_not_applicable_full_seeds": [
            int(seed)
            for seed in audit.get("geometry_not_applicable_full_seeds", [])
        ],
        "not_estimable_reason": audit.get("not_estimable_reason"),
        "scientific_result": audit.get("scientific_result"),
        "partial_mediation_primary_pass": bool(
            audit.get("partial_mediation_primary_pass", False)
        ),
    }


def build(
    run_root: Path,
    output_root: Path,
    *,
    relay_geometry_amendment_path: Path | None = None,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    contract_hashes: set[str] = set()
    extension_contract_path = (
        ROOT / "configs" / "realistic_niah_v6_answer_trace_extension_v1.json"
    )
    relay_geometry_amendment = (
        load_relay_geometry_amendment(
            relay_geometry_amendment_path,
            extension_contract_path=extension_contract_path,
        )
        if relay_geometry_amendment_path is not None
        else None
    )
    relay_geometry_amendment_sha256 = (
        _sha(relay_geometry_amendment_path)
        if relay_geometry_amendment_path is not None
        else None
    )
    for prompt_mode in PROMPT_MODES:
        for model in MODELS:
            root = (
                run_root
                / prompt_mode
                / model
                / "causal"
                / "answer_trace_extension_v1"
            )
            complete_path = root / "extension_complete.json"
            answer_audit_path = (
                root
                / "answer_query_layer_sweep"
                / "analysis"
                / "v6_extension_audit.json"
            )
            layer_path = (
                root
                / "answer_query_layer_sweep"
                / "analysis"
                / "layer_effects.csv"
            )
            original_relay_audit_path = (
                root
                / "terminal_relay_partial_confirmation"
                / "relay_analysis_confirmation"
                / "v6_extension_audit.json"
            )
            original_gates_path = (
                root
                / "terminal_relay_partial_confirmation"
                / "relay_analysis_confirmation"
                / "claim_gates.json"
            )
            use_task_adapted_relay = (
                relay_geometry_amendment is not None
                and prompt_mode == "enumeration_bullet"
            )
            relay_directory = (
                str(relay_geometry_amendment["adaptation"]["output_directory_name"])
                if use_task_adapted_relay
                else "terminal_relay_partial_confirmation"
            )
            relay_audit_path = (
                root
                / relay_directory
                / "relay_analysis_confirmation"
                / "v6_extension_audit.json"
            )
            gates_path = (
                root
                / relay_directory
                / "relay_analysis_confirmation"
                / "claim_gates.json"
            )
            for path in (
                complete_path,
                answer_audit_path,
                layer_path,
                original_relay_audit_path,
                original_gates_path,
                relay_audit_path,
                gates_path,
            ):
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"Missing V6 answer/trace artifact: {path}")
            complete = _json(complete_path)
            answer_audit = _json(answer_audit_path)
            original_relay_audit = _json(original_relay_audit_path)
            relay_audit = _json(relay_audit_path)
            gates = _json(gates_path)
            if complete.get("status") != "PASS_EXECUTION_COMPLETE":
                raise ValueError(f"Extension cell is incomplete: {prompt_mode}/{model}")
            contract_hashes.add(str(complete["extension_contract_sha256"]))
            if str(relay_audit.get("extension_contract_sha256")) != str(
                complete["extension_contract_sha256"]
            ):
                raise ValueError(
                    "Terminal relay and answer extension contract hashes disagree: "
                    f"{prompt_mode}/{model}"
                )
            relay_geometry = str(
                relay_audit.get(
                    "relay_geometry", "suffix4" if use_task_adapted_relay else "suffix8"
                )
            )
            expected_geometry = "suffix4" if use_task_adapted_relay else "suffix8"
            if relay_geometry != expected_geometry:
                raise ValueError(
                    "Terminal relay report selected the wrong geometry: "
                    f"{prompt_mode}/{model}"
                )
            if use_task_adapted_relay and relay_audit.get(
                "relay_geometry_amendment_sha256"
            ) != relay_geometry_amendment_sha256:
                raise ValueError(
                    "Task-adapted Bullet relay amendment hash changed: "
                    f"{prompt_mode}/{model}"
                )
            layer_rows = _rows(layer_path)
            if not layer_rows:
                raise ValueError(f"No answer layer rows: {prompt_mode}/{model}")
            terminal = max(layer_rows, key=lambda row: int(row["layer"]))
            onset = answer_audit["native_analysis_audit"][
                "descriptive_onset_layer"
            ].get(model)
            relay_planned_seed_count = int(
                relay_audit.get("planned_seed_count", -1)
            )
            relay_eligible_seed_count = int(
                relay_audit.get("eligible_seed_count", -1)
            )
            relay_estimable = bool(
                relay_audit.get(
                    "relay_estimable", relay_eligible_seed_count > 0
                )
            )
            relay_full_na_seeds = [
                int(seed)
                for seed in relay_audit.get(
                    "geometry_not_applicable_full_seeds", []
                )
            ]
            relay_full_na_count = int(
                relay_audit.get(
                    "geometry_not_applicable_full_seed_count", -1
                )
            )
            if relay_planned_seed_count != 10:
                raise ValueError(
                    "Terminal relay must retain all 10 preregistered seeds: "
                    f"{prompt_mode}/{model}"
                )
            if relay_estimable and not (
                0 < relay_eligible_seed_count <= relay_planned_seed_count
            ):
                raise ValueError(
                    "Terminal relay has an invalid geometry-eligible seed count: "
                    f"{prompt_mode}/{model}"
                )
            if not relay_estimable and relay_eligible_seed_count != 0:
                raise ValueError(
                    "Non-estimable terminal relay has nonzero support: "
                    f"{prompt_mode}/{model}"
                )
            if (
                relay_full_na_count != len(relay_full_na_seeds)
                or len(set(relay_full_na_seeds)) != len(relay_full_na_seeds)
                or relay_eligible_seed_count + relay_full_na_count
                != relay_planned_seed_count
            ):
                raise ValueError(
                    "Terminal relay planned/eligible/full-NA seed accounting is "
                    f"inconsistent: {prompt_mode}/{model}"
                )
            answer_layer_effects = [
                {
                    "layer": int(row["layer"]),
                    "seed_clusters": int(row["seed_clusters"]),
                    "pairs": int(row["pairs"]),
                    "full_donor_adoption": float(row["full_donor_adoption"]),
                    "full_donor_adoption_ci95_low": float(
                        row["full_donor_adoption_ci95_low"]
                    ),
                    "full_donor_adoption_ci95_high": float(
                        row["full_donor_adoption_ci95_high"]
                    ),
                    "adoption_specificity": float(row["adoption_specificity"]),
                    "adoption_specificity_ci95_low": float(
                        row["adoption_specificity_ci95_low"]
                    ),
                    "adoption_specificity_ci95_high": float(
                        row["adoption_specificity_ci95_high"]
                    ),
                    "registered_numeric_valid": float(
                        row["registered_numeric_valid"]
                    ),
                }
                for row in sorted(layer_rows, key=lambda row: int(row["layer"]))
            ]
            relay_gate_ids = (
                "terminal_state_patch_effect",
                "post_terminal_suffix_specific_mediation",
                "post_terminal_suffix_residual_equivalence",
                "self_reset_is_nondamaging",
                "answer_query_only_mediation",
            )
            relay_gates = {
                gate_id: _gate(gates, gate_id) for gate_id in relay_gate_ids
            }
            original_suffix8_support = _relay_support(
                original_relay_audit, geometry="suffix8"
            )
            cells.append(
                {
                    "prompt_mode": prompt_mode,
                    "mode_label": MODE_LABELS[prompt_mode],
                    "model_label": model,
                    "answer_registered_pairs": int(
                        answer_audit["pair_registry_audit"]["registered_pairs"]
                    ),
                    "answer_seed_clusters": int(terminal["seed_clusters"]),
                    "answer_terminal_layer": int(terminal["layer"]),
                    "answer_terminal_adoption": float(
                        terminal["full_donor_adoption"]
                    ),
                    "answer_terminal_ci_low": float(
                        terminal["full_donor_adoption_ci95_low"]
                    ),
                    "answer_terminal_ci_high": float(
                        terminal["full_donor_adoption_ci95_high"]
                    ),
                    "answer_descriptive_onset": onset,
                    "answer_layer_effects": answer_layer_effects,
                    "relay_gates": relay_gates,
                    "relay_geometry": relay_geometry,
                    "relay_original_geometry": "suffix8",
                    "relay_evidence_label": (
                        "post_hoc_task_adapted_bullet_relay_replication"
                        if use_task_adapted_relay
                        else "original_registered_suffix8"
                    ),
                    "relay_geometry_amendment_sha256": (
                        relay_geometry_amendment_sha256
                        if use_task_adapted_relay
                        else None
                    ),
                    "original_suffix8_relay": original_suffix8_support,
                    "terminal_patch": relay_gates["terminal_state_patch_effect"],
                    "suffix_mediation": relay_gates[
                        "post_terminal_suffix_specific_mediation"
                    ],
                    "suffix_residual_ratio": relay_gates[
                        "post_terminal_suffix_residual_equivalence"
                    ],
                    "self_reset_ratio": relay_gates["self_reset_is_nondamaging"],
                    "query_mediation": relay_gates["answer_query_only_mediation"],
                    "relay_planned_seed_count": relay_planned_seed_count,
                    "relay_eligible_seed_count": relay_eligible_seed_count,
                    "relay_estimable": relay_estimable,
                    "relay_not_estimable_reason": relay_audit.get(
                        "not_estimable_reason"
                    ),
                    "relay_geometry_not_applicable_full_seed_count": (
                        relay_full_na_count
                    ),
                    "relay_geometry_not_applicable_full_seeds": (
                        relay_full_na_seeds
                    ),
                    "partial_mediation_pass": bool(
                        relay_audit["partial_mediation_primary_pass"]
                    ),
                    "complete_mediation_not_claimed": True,
                    "seed_aliasing": False,
                    "artifact_hashes": {
                        "completion": _sha(complete_path),
                        "answer_audit": _sha(answer_audit_path),
                        "layer_effects": _sha(layer_path),
                        "relay_audit": _sha(relay_audit_path),
                        "claim_gates": _sha(gates_path),
                    },
                    "original_suffix8_artifact_hashes": {
                        "relay_audit": _sha(original_relay_audit_path),
                        "claim_gates": _sha(original_gates_path),
                    },
                }
            )
    if len(contract_hashes) != 1:
        raise ValueError("Four V6 extension cells do not share one frozen contract")

    table_rows = []
    for cell in cells:
        terminal = cell["terminal_patch"]
        suffix = cell["suffix_mediation"]
        query = cell["query_mediation"]
        onset = (
            "NA"
            if cell["answer_descriptive_onset"] is None
            else f"L{int(cell['answer_descriptive_onset'])}"
        )
        full_na_seeds = cell[
            "relay_geometry_not_applicable_full_seeds"
        ]
        relay_seed_accounting = (
            f"{cell['relay_eligible_seed_count']}/"
            f"{cell['relay_planned_seed_count']}"
        )
        if full_na_seeds:
            relay_seed_accounting += "; full-NA=" + ", ".join(
                str(seed) for seed in full_na_seeds
            )
        else:
            relay_seed_accounting += "; full-NA=none"
        if cell["relay_estimable"]:
            terminal_result = (
                f"{_fmt(terminal['estimate'])} "
                f"[{_fmt(terminal['low'])}, {_fmt(terminal['high'])}] "
                f"({'pass' if terminal['pass'] else 'fail'})"
            )
            suffix_result = (
                f"{_fmt(suffix['estimate'])} "
                f"[{_fmt(suffix['low'])}, {_fmt(suffix['high'])}] "
                f"({'pass' if suffix['pass'] else 'fail'})"
            )
            query_result = (
                f"{_fmt(query['estimate'])} "
                f"[{_fmt(query['low'])}, {_fmt(query['high'])}] "
                f"({'pass' if query['pass'] else 'fail'})"
            )
            relay_conclusion = (
                "partial mediation supported"
                if cell["partial_mediation_pass"]
                else "primary gates not both passed"
            )
        else:
            reason = str(
                cell.get("relay_not_estimable_reason")
                or f"{cell['relay_geometry']} N/A"
            )
            terminal_result = suffix_result = query_result = "NA (0 eligible seeds)"
            relay_conclusion = f"not estimable: {reason}"
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(cell['mode_label'])}</td>"
            f"<td>{html.escape(cell['model_label'])}</td>"
            f"<td><code>{html.escape(cell['relay_geometry'])}</code><br>"
            f"{html.escape(cell['relay_evidence_label'])}</td>"
            f"<td>{cell['answer_registered_pairs']}</td>"
            f"<td>{onset}</td>"
            f"<td>L{cell['answer_terminal_layer']}: "
            f"{_fmt(cell['answer_terminal_adoption'])} "
            f"[{_fmt(cell['answer_terminal_ci_low'])}, "
            f"{_fmt(cell['answer_terminal_ci_high'])}]</td>"
            f"<td>{html.escape(terminal_result)}</td>"
            f"<td>{html.escape(suffix_result)}</td>"
            f"<td>{html.escape(query_result)}</td>"
            f"<td>{html.escape(relay_seed_accounting)}</td>"
            f"<td>{html.escape(relay_conclusion)}</td>"
            "</tr>"
        )
    generated = dt.datetime.now(dt.timezone.utc).isoformat()
    task_adapted_enabled = relay_geometry_amendment is not None
    relay_design_text = (
        "Index 保留原注册 suffix8；Bullet-Qwen/Gemma 统一使用 suffix4 的 "
        "post-hoc task-adapted replication。原 Bullet suffix8 结果不覆盖，仍在逐格审计中保留。"
        if task_adapted_enabled
        else "四格均使用原注册 suffix8。"
    )
    amendment_text = (
        f" · Relay amendment SHA256: {relay_geometry_amendment_sha256}"
        if task_adapted_enabled
        else ""
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V6 Answer-token Patching and Trace-to-answer Extension</title>
<style>
body{{font:15px/1.55 system-ui,sans-serif;color:#182230;background:#f6f7f9;margin:0}}
main{{max-width:1320px;margin:32px auto;background:#fff;padding:34px;box-shadow:0 2px 18px #0001}}
h1,h2{{line-height:1.2}} .note{{border-left:4px solid #476a8a;padding:10px 14px;background:#eef4f8}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border:1px solid #d8dee6;padding:8px;vertical-align:top}}
th{{background:#eef1f5;text-align:left}} code{{background:#f0f2f5;padding:1px 4px}}
.small{{color:#52606d;font-size:13px}}
</style></head><body><main>
<h1>V6 answer-token patching 与 trace→answer 扩展</h1>
  <p class="small">Generated UTC: {html.escape(generated)} · Contract SHA256: {html.escape(next(iter(contract_hashes)))}{html.escape(amendment_text)}</p>
<div class="note"><strong>证据标签。</strong>这是用户追加的 registered existing-split extension，
在读取 V6 intervention outcomes 前冻结；它不改原 20-frame 主套件、K、seed split 或主结论。
四个 cell 都使用冻结的 coherent confirmation cohort，统计 identity 是真实 source seed。</div>
<h2>与 Native-thinking 对齐的方法</h2>
<p><code>answer_query_v3</code> full donor residual patch 使用完全相同的 Qwen/Gemma 8 层网格、
self/full 两臂、greedy 16 tokens、donor-count adoption 与 10,000 次 seed-cluster bootstrap。
  trace→answer 保持 natural relay、answer-query clean reset、post-terminal suffix clean reset 的同一 2×3 factorial
  与同一 sequence-margin estimator。{html.escape(relay_design_text)}这里只允许 partial-mediation claim；不声称 complete mediation。</p>
<h2>四个模型×格式结果</h2>
  <table><thead><tr><th>格式</th><th>模型</th><th>relay geometry / evidence</th><th>有向 pairs</th><th>answer onset</th>
<th>最末层 donor adoption [95% CI]</th><th>terminal patch</th><th>suffix mediation</th>
<th>answer-query mediation</th><th>relay 可估计/预注册 seeds</th><th>结论</th></tr></thead><tbody>
{''.join(table_rows)}
</tbody></table>
<p class="small">Answer onset 是描述性规则，不用于选层。Relay 分母始终是预注册的 10 个 true source seeds；
  若某个 seed 的所有 registered pairs 都短于该 cell 的注册 geometry，它保留在 planned audit 中并列为 full-NA，
  数值估计只在 geometry-eligible seeds 上进行。Unparsable/out-of-range greedy outputs 保留为 adoption failure。
  Scientific gate failure 不等于流水线失败；完整执行与正/负结果严格分开。Task-adapted suffix4 不替换原 suffix8 assay，
  也不改 cohort、pair rule、layers、estimands、bootstrap 或 gates。</p>
</main></body></html>"""
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "NiaH_V6_Answer_Trace_Extension_report.html"
    _atomic_text(report_path, document)
    summary = {
        "schema_version": "realistic_niah_v6_answer_trace_extension_report_v2",
        "status": "PASS_COMPLETE",
        "generated_utc": generated,
        "extension_contract_sha256": next(iter(contract_hashes)),
        "relay_geometry_amendment_sha256": relay_geometry_amendment_sha256,
        "relay_geometry_policy": (
            "index_suffix8_bullet_suffix4_task_adapted"
            if task_adapted_enabled
            else "all_cells_original_suffix8"
        ),
        "original_suffix8_artifacts_preserved": True,
        "cells": cells,
        "report": str(report_path.resolve()),
        "report_sha256": _sha(report_path),
    }
    _atomic_text(
        output_root / "report_summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    _atomic_text(output_root / "report.COMPLETE", "PASS\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--relay-geometry-amendment", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.run_root,
                args.output_root,
                relay_geometry_amendment_path=args.relay_geometry_amendment,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
