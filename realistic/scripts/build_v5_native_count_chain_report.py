#!/usr/bin/env python3
"""Build the final native-thinking targeted-to-count causal-chain report."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
from pathlib import Path
from typing import Any


MODELS = ("Qwen3-8B", "Gemma4-E4B")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(value: dict[str, Any], *, digits: int = 3) -> str:
    estimate = float(value["estimate"])
    low = float(value["ci_low"])
    high = float(value["ci_high"])
    return f"{estimate:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _pct_metric(value: dict[str, Any]) -> str:
    estimate = 100 * float(value["estimate"])
    low = 100 * float(value["ci_low"])
    high = 100 * float(value["ci_high"])
    return f"{estimate:.1f}% [{low:.1f}%, {high:.1f}%]"


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _read_evidence(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    values: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    claim_contract = root / "claim_contract.json"
    if claim_contract.exists():
        values["claim_contract"] = _load(claim_contract)
        hashes[str(claim_contract.relative_to(root))] = _sha(claim_contract)
    prospective_manifest = root / "prospective_evidence_manifest.json"
    if prospective_manifest.exists():
        values["prospective_evidence_manifest"] = _load(prospective_manifest)
        hashes[str(prospective_manifest.relative_to(root))] = _sha(
            prospective_manifest
        )
    metadata_fix = root / "metadata_fix_ledger.json"
    if metadata_fix.exists():
        values["metadata_fix_ledger"] = _load(metadata_fix)
        hashes[str(metadata_fix.relative_to(root))] = _sha(metadata_fix)
    for model in MODELS:
        for kind in ("targeted", "readout", "integrated"):
            path = root / model / f"{kind}_complete.json"
            if not path.exists():
                raise FileNotFoundError(path)
            values[f"{model}:{kind}"] = _load(path)
            hashes[str(path.relative_to(root))] = _sha(path)
        plan_meta = root / model / "targeted_plan_meta.json"
        if not plan_meta.exists():
            raise FileNotFoundError(plan_meta)
        values[f"{model}:targeted_plan_meta"] = _load(plan_meta)
        hashes[str(plan_meta.relative_to(root))] = _sha(plan_meta)
        extension = root / model / "prospective_extension_complete.json"
        if extension.exists():
            values[f"{model}:prospective_extension"] = _load(extension)
            hashes[str(extension.relative_to(root))] = _sha(extension)
        extension_protocol = root / model / "prospective_extension_protocol.json"
        if extension_protocol.exists():
            hashes[str(extension_protocol.relative_to(root))] = _sha(
                extension_protocol
            )
        for phase in ("discovery", "confirmation"):
            path = root / model / f"integrated_{phase}_audit.json"
            if path.exists():
                values[f"{model}:integrated_{phase}_audit"] = _load(path)
                hashes[str(path.relative_to(root))] = _sha(path)
    return values, hashes


def _assert_contract(evidence: dict[str, Any]) -> None:
    claim_contract = evidence.get("claim_contract")
    if claim_contract is not None:
        if not str(claim_contract.get("status", "")).startswith("FROZEN_"):
            raise ValueError("Prospective claim contract is not frozen")
        shared = claim_contract["shared_protocol"]
        if list(shared.get("discovery_seeds", [])) != list(range(1234, 1254)):
            raise ValueError("Claim-contract discovery seeds changed")
        if list(shared.get("confirmation_seeds", [])) != list(range(1254, 1264)):
            raise ValueError("Claim-contract confirmation seeds changed")
        if shared.get("outcome_blind") is not True:
            raise ValueError("Claim contract is not outcome-blind")
        if shared.get("selection_rank_used") is not False:
            raise ValueError("Claim contract used selection_rank")
        metadata_fix = evidence.get("metadata_fix_ledger")
        prospective_manifest = evidence.get("prospective_evidence_manifest")
        if metadata_fix is None or prospective_manifest is None:
            raise ValueError("Prospective evidence lacks metadata-fix provenance")
        expected_status = (
            "RESULT_INDEPENDENT_METADATA_ONLY_FIX_BEFORE_ANY_PROSPECTIVE_BRIDGE_RUN"
        )
        if metadata_fix.get("status") != expected_status:
            raise ValueError("Metadata-fix provenance is not frozen")
        if metadata_fix.get("fix") != {
            "Qwen3-8B": "post_marker",
            "Gemma4-E4B": "p0_item_end",
        }:
            raise ValueError("Metadata-fix anchor-role mapping changed")
        source_hashes = prospective_manifest.get("source_sha256", {})
        source_path = str(metadata_fix.get("file"))
        if source_hashes.get(source_path) != metadata_fix.get("new_sha256"):
            raise ValueError("Metadata-fix source hash is absent or stale")
        ledger_path = (
            "configs/realistic_niah_v5_integrated_bridge_metadata_fix_v1.json"
        )
        if ledger_path not in source_hashes:
            raise ValueError("Metadata-fix ledger source hash is absent")
    for model in MODELS:
        targeted = evidence[f"{model}:targeted"]
        targeted_plan = evidence[f"{model}:targeted_plan_meta"]
        readout = evidence[f"{model}:readout"]
        integrated = evidence[f"{model}:integrated"]
        if targeted.get("status") != "PASS":
            raise ValueError(f"{model} targeted endpoint is not PASS")
        if str(targeted_plan.get("model_label")) != model:
            raise ValueError(f"{model} targeted plan metadata has a model mismatch")
        if int(targeted_plan.get("bank_size", 0)) <= 0:
            raise ValueError(f"{model} targeted plan metadata lacks a valid bank size")
        if targeted_plan.get("selection_rank_used") is not False:
            raise ValueError(f"{model} targeted plan metadata used selection_rank")
        if readout.get("status") != "PASS":
            raise ValueError(f"{model} readout is not PASS")
        extension = evidence.get(f"{model}:prospective_extension")
        if extension is not None:
            if claim_contract is None:
                raise ValueError(f"{model} prospective extension lacks claim contract")
            contract_model = claim_contract["models"][model]
            contract_signature = (
                int(contract_model["prospective_bank_size"]),
                str(contract_model["selected_bank_sha256"]),
            )
            extension_signature = (
                int(extension.get("bank_size", -1)),
                str(extension.get("selected_bank_sha256")),
            )
            if contract_signature != extension_signature:
                raise ValueError(f"{model} prospective extension violates claim contract")
            if str(extension.get("model_label")) != model:
                raise ValueError(f"{model} prospective extension model mismatch")
            if extension.get("selection_rank_used") is not False:
                raise ValueError(f"{model} prospective extension used selection_rank")
            extension_status = str(extension.get("status"))
            if extension_status not in {"PASS", "PROTOCOL_EXHAUSTED"}:
                raise ValueError(
                    f"{model} prospective extension is not terminal: {extension_status}"
                )
            if extension_status == "PASS":
                if extension.get("endpoint_status") != "PASS":
                    raise ValueError(f"{model} PASS extension endpoint is not PASS")
                if extension.get("bridge_status") != "PASS":
                    raise ValueError(f"{model} PASS extension bridge is not PASS")
                if int(extension.get("bank_size", -1)) != int(
                    targeted_plan["bank_size"]
                ):
                    raise ValueError(f"{model} prospective bank size is not primary")
                expected_sha = extension.get("selected_bank_sha256")
                observed_sha = targeted_plan.get("selected_bank_sha256")
                if observed_sha != expected_sha:
                    raise ValueError(f"{model} prospective bank hash is not primary")
        bridge_flags = (
            bool(integrated.get("integrated_serial_bridge_pass")),
            bool(integrated.get("integrated_mediator_restoration_pass")),
        )
        status = str(integrated.get("status"))
        if status == "PASS" and sum(bridge_flags) != 1:
            raise ValueError(
                f"{model} must pass exactly one registered integrated bridge"
            )
        if status == "PRE_REGISTERED_BRANCHES_EXHAUSTED":
            if sum(bridge_flags) != 0:
                raise ValueError(f"{model} exhausted ledger cannot contain a pass")
            if integrated.get("pre_registered_branches_exhausted") is not True:
                raise ValueError(f"{model} lacks an exhausted-branch audit")
            outcomes = integrated.get("branch_outcomes")
            expected_names = [
                "exact_query_transfer",
                "persistent_transfer",
                "suffix8_restoration",
                "fullspan_restoration",
            ]
            if not isinstance(outcomes, list) or [
                str(value.get("name")) for value in outcomes
            ] != expected_names:
                raise ValueError(f"{model} branch ledger is incomplete")
            if any(str(value.get("status")) == "PASS" for value in outcomes):
                raise ValueError(f"{model} exhausted ledger contains a PASS branch")
        elif status != "PASS":
            raise ValueError(f"{model} integrated outcome is not terminal: {status}")
        required_phases = [("discovery", 20)]
        if status == "PASS":
            required_phases.append(("confirmation", 10))
        for phase, expected in required_phases:
            key = f"{model}:integrated_{phase}_audit"
            if key not in evidence:
                raise ValueError(f"{model} lacks integrated {phase} audit")
            audit = evidence[key]
            if int(audit["seed_count"]) != expected:
                raise ValueError(f"{model} {phase} seed contract changed")
            if int(audit.get("applicable_seed_count", -1)) != expected:
                raise ValueError(f"{model} {phase} effective seed contract changed")
            if audit.get("selection_rank_used") is not False:
                raise ValueError(f"{model} {phase} used selection_rank")


def _mediator_label(model: str, integrated: dict[str, Any]) -> str:
    geometry = str(integrated.get("mediator_geometry", "suffix8"))
    if geometry == "suffix8":
        span = "terminal suffix8 full state"
    elif geometry == "full_span":
        span = "terminal full-trace-item hidden state"
    else:
        raise ValueError(f"Unknown mediator geometry for {model}: {geometry}")
    layers = "L19–25" if model == "Qwen3-8B" else "L16–41"
    return f"{span} · {layers}"


def _chain(
    model: str, integrated: dict[str, Any], targeted_plan: dict[str, Any]
) -> str:
    bank_size = int(targeted_plan["bank_size"])
    if model == "Qwen3-8B":
        middle = _mediator_label(model, integrated)
        readout = "L26 residual relay + direct trace reread"
    else:
        middle = _mediator_label(model, integrated)
        readout = "distributed all-layer/all-head trace-source readout"
    bank = f"frozen K{bank_size} targeted bank"
    boxes = [
        ("Targeted retrieval", bank),
        ("Count state", middle),
        ("Answer readout", readout),
        ("Behavior", "parsed discrete final count"),
    ]
    nodes = [
        f'<div class="node"><b>{_esc(title)}</b><small>{_esc(body)}</small></div>'
        for title, body in boxes
    ]
    if integrated.get("status") == "PASS":
        return '<div class="chain">' + '<span class="arrow">→</span>'.join(nodes) + "</div>"
    nodes[1] = (
        f'<div class="node"><b>{_esc(boxes[1][0])}</b>'
        '<small>independently confirmed terminal count state</small></div>'
    )
    return (
        '<div class="chain">'
        + nodes[0]
        + '<span class="break">⇢? unsupported bridge</span>'
        + '<span class="arrow">→</span>'.join(nodes[1:])
        + "</div>"
    )


def _bridge_summary(
    integrated: dict[str, Any], audit: dict[str, Any]
) -> tuple[str, str, str, str, str]:
    if integrated.get("integrated_serial_bridge_pass"):
        gates = integrated["confirmation_claim_gates"]["gates"]
        return (
            "persistent state transfer",
            _metric(gates["targeted_bank_changes_terminal_state_readout"]),
            _metric(gates["matched_readout_control_preserves_damage"]),
            _metric(gates["readout_cut_occludes_targeted_state_effect"]),
            _metric(gates["cut_residual_equivalence"]),
        )
    if integrated.get("integrated_mediator_restoration_pass"):
        gates = integrated["confirmation_claim_gates"]["gates"]
        geometry = str(integrated.get("mediator_geometry", "suffix8"))
        geometry_label = "suffix8" if geometry == "suffix8" else "full-span"
        return (
            f"within-example state restoration ({geometry_label})",
            _metric(gates["targeted_receiver_damage"]),
            _metric(gates["clean_state_restores_selected_receiver"]),
            _metric(gates["readout_cut_occludes_restoration"]),
            _metric(gates["cut_restoration_residual_equivalence"]),
        )
    if integrated.get("status") == "PRE_REGISTERED_BRANCHES_EXHAUSTED":
        gates = integrated["final_branch_claim_gates"]["gates"]
        return (
            "all pre-registered bridges exhausted",
            "FAIL " + _metric(gates["targeted_receiver_damage"]),
            "FAIL " + _metric(gates["clean_state_restores_selected_receiver"]),
            "FAIL " + _metric(gates["readout_cut_occludes_restoration"]),
            "FAIL " + _metric(gates["cut_restoration_residual_equivalence"]),
        )
    raise ValueError("Unknown integrated bridge result")


def build(evidence: dict[str, Any], hashes: dict[str, str]) -> str:
    q_target = evidence["Qwen3-8B:targeted"]["confirmation"]["gates"]
    g_target = evidence["Gemma4-E4B:targeted"]["confirmation"]["gates"]
    q_read = evidence["Qwen3-8B:readout"]["confirmation_claim_gates"]["gates"]
    g_read = evidence["Gemma4-E4B:readout"]["confirmation_claim_gates"]["gates"]
    q_audit = evidence.get(
        "Qwen3-8B:integrated_confirmation_audit",
        evidence["Qwen3-8B:integrated_discovery_audit"],
    )
    g_audit = evidence.get(
        "Gemma4-E4B:integrated_confirmation_audit",
        evidence["Gemma4-E4B:integrated_discovery_audit"],
    )

    targeted_rows = [
        (
            "Qwen3-8B",
            _pct_metric(q_target["clean_endpoint_adequacy"]),
            _pct_metric(q_target["targeted_bank_changes_final_count"]),
            _pct_metric(q_target["retrieval_failure_propagates_to_count"]),
        ),
        (
            "Gemma4-E4B",
            _pct_metric(g_target["clean_endpoint_adequacy"]),
            _pct_metric(g_target["targeted_bank_changes_final_count"]),
            _pct_metric(g_target["retrieval_failure_propagates_to_count"]),
        ),
    ]
    readout_rows = [
        (
            "Qwen3-8B",
            _metric(q_read["storage_main_effect"]),
            (
                "relay " + _metric(q_read["residual_relay_contribution"])
                + "; reread " + _metric(q_read["direct_reread_contribution_after_relay"])
            ),
            _metric(q_read["joint_cut_residual_equivalence"]),
        ),
        (
            "Gemma4-E4B",
            _metric(g_read["storage_main_effect"]),
            _metric(g_read["trace_source_specific_occlusion"]),
            _metric(g_read["trace_mask_residual_equivalence"]),
        ),
    ]
    bridge_rows = []
    for model, integrated, audit in (
        ("Qwen3-8B", evidence["Qwen3-8B:integrated"], q_audit),
        ("Gemma4-E4B", evidence["Gemma4-E4B:integrated"], g_audit),
    ):
        bridge_kind, write_effect, state_link, occlusion, residual = _bridge_summary(
            integrated, audit
        )
        bridge_rows.append(
            (
                model,
                bridge_kind,
                write_effect,
                state_link,
                occlusion,
                residual,
                f"{audit['applicable_sample_count']} / {audit['planned_sample_count']}",
            )
        )

    extension_rows = []
    for model in MODELS:
        extension = evidence.get(f"{model}:prospective_extension")
        if extension is None:
            continue
        extension_rows.append(
            (
                model,
                f"K{int(extension['bank_size'])}",
                str(extension["endpoint_status"]),
                str(extension["bridge_status"]),
                str(extension["status"]),
            )
        )
    metadata_fix = evidence.get("metadata_fix_ledger")
    metadata_note = ""
    if metadata_fix is not None:
        metadata_note = (
            '<p class="note"><b>Result-independent provenance fix:</b> '
            '在任何 prospective bridge 启动、任何 endpoint 聚合效应或 claim gate '
            '被读取前，仅修正输出元数据中的模型特异 anchor-role 标签：'
            'Qwen=<code>post_marker</code>，Gemma=<code>p0_item_end</code>。'
            'head、layer、样本、patch、readout cut 与统计 gate 均未改变；修复记录及源码 '
            'SHA-256 已纳入 evidence ledger。</p>'
        )

    def table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
        head = "".join(f"<th>{_esc(value)}</th>" for value in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{_esc(value)}</td>" for value in row) + "</tr>"
            for row in rows
        )
        return f"<div class=table-wrap><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"

    hash_rows = "".join(
        f"<li><code>{_esc(path)}</code><br><span>{_esc(digest)}</span></li>"
        for path, digest in sorted(hashes.items())
    )
    generated = dt.datetime.now(dt.timezone.utc).isoformat()
    passed_models = [
        model for model in MODELS if evidence[f"{model}:integrated"].get("status") == "PASS"
    ]
    if len(passed_models) == len(MODELS):
        verdict = (
            "Qwen 与 Gemma 均支持 targeted retrieval → terminal count state → "
            "model-specific readout → final count。"
        )
    elif passed_models:
        failed = [model for model in MODELS if model not in passed_models]
        verdict = (
            f"{', '.join(passed_models)} 获得完整 confirmation 链；"
            f"{', '.join(failed)} 只确认了链条两端，所有预注册的 "
            "targeted-retrieval→terminal-state bridge 分支均未通过。"
        )
    else:
        verdict = (
            "Qwen 与 Gemma 的 targeted-retrieval→final-count endpoint 以及 "
            "terminal-state→answer-readout 均分别通过，但所有预注册的中间 "
            "bridge 均未获得 confirmation；因此不支持完整串行中介链。"
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Native-thinking count mechanism · confirmed causal chains</title>
<style>
:root{{--ink:#16201d;--muted:#5d6864;--paper:#f4f1e9;--card:#fffdf8;--line:#d8d2c4;--q:#275d73;--g:#167064;--accent:#a64b2a}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.62 system-ui,-apple-system,"Noto Sans SC",sans-serif}}
main{{max-width:1120px;margin:auto;padding:48px 28px 90px}} header{{padding:36px;border:1px solid var(--line);background:var(--card);border-radius:22px}}
.eyebrow{{font-size:12px;letter-spacing:.16em;color:var(--accent);font-weight:750}} h1{{font-family:Georgia,"Noto Serif SC",serif;font-size:clamp(34px,5vw,58px);line-height:1.08;margin:.25em 0}} h2{{margin-top:54px;font-family:Georgia,"Noto Serif SC",serif;font-size:30px}} h3{{font-size:21px;margin-top:34px}}
.lead{{font-size:19px;color:var(--muted);max-width:900px}} .verdict{{margin-top:22px;padding:18px 20px;border-left:5px solid var(--accent);background:#fff8ef}}
.model{{margin:24px 0;padding:24px;border:1px solid var(--line);border-radius:18px;background:var(--card)}} .model.q{{border-top:5px solid var(--q)}} .model.g{{border-top:5px solid var(--g)}}
.chain{{display:flex;align-items:stretch;gap:9px;overflow-x:auto;padding:8px 0}} .node{{min-width:180px;flex:1;padding:16px;border:1px solid var(--line);border-radius:13px;background:white}} .node b,.node small{{display:block}} .node small{{margin-top:7px;color:var(--muted)}} .arrow{{align-self:center;font-size:25px;color:var(--accent)}} .break{{align-self:center;min-width:145px;padding:7px 9px;border:1px dashed var(--accent);border-radius:9px;color:var(--accent);font-size:12px;text-align:center}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:14px;background:var(--card)}} table{{border-collapse:collapse;width:100%;min-width:760px}} th,td{{padding:12px 14px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}} th{{font-size:13px;background:#ece8dd}} code{{font-size:13px}} .note{{padding:16px 18px;border:1px solid var(--line);background:#fbfaf5;border-radius:12px;color:var(--muted)}} details{{margin-top:28px}} li span{{font:12px ui-monospace,monospace;color:var(--muted);word-break:break-all}}
</style></head><body><main>
<header><div class="eyebrow">REALISTIC NIAH · NATIVE THINKING · 20D/10C CONFIRMATION</div><h1>从 targeted retrieval 到最终 count：模型特异的因果证据链</h1><p class="lead">本报告把 free-generation endpoint、terminal hidden-state patching、answer readout cut 与 integrated serial bridge 分开估计，再按时间顺序组合。所有正式实验固定 20 discovery seeds、10 confirmation seeds、outcome-blind sample plan，且禁用 selection_rank。Prospective 结论按结果出现前冻结的 claim contract 判定。</p><div class="verdict"><b>结论：</b>{_esc(verdict)} 这不等同于自然效应中介、最小 head circuit 或单轴标量计数器。</div></header>
<section><h2>1 · 两条模型特异链</h2><div class="model q"><h3>Qwen3-8B</h3>{_chain('Qwen3-8B', evidence['Qwen3-8B:integrated'], evidence['Qwen3-8B:targeted_plan_meta'])}<p>Qwen 的 answer readout 是并行而非单一路径：post-terminal residual relay 与 direct trace reread 各自留下显著残余，联合切断才把 state-patch effect 压到 equivalence bound 内。</p></div><div class="model g"><h3>Gemma4-E4B</h3>{_chain('Gemma4-E4B', evidence['Gemma4-E4B:integrated'], evidence['Gemma4-E4B:targeted_plan_meta'])}<p>Gemma 的 terminal state 通过分布式 trace-source attention 被读取；全层全头 source mask 是功能通路定位，不是稀疏最小 head 集。</p></div></section>
<section><h2>2 · Targeted retrieval 传播到离散最终 count</h2>{table(('Model','Clean count accuracy','Selected−random count failure','Joint retrieval+count failure'), targeted_rows)}<p class="note">Qwen 的 undercount 与 exact N−1 次级方向门未通过；其主要失败形态包含无法解析、截断或错误 city。因此只声称 targeted bank 改变最终 count 正确性及联合失败，不声称它稳定地产生 N−1。</p></section>
<section><h2>3 · Terminal state 与 answer readout</h2>{table(('Model','State-patch effect','Readout-specific effect','Residual ratio after cut'), readout_rows)}</section>
<section><h2>4 · Integrated causal bridge</h2><p>在同一 teacher-forced trace 上，对冻结的最终 N−1→N routed targeted query 施加 selected 或三组 layer-matched random bank ablation。Persistent transfer 直接比较 ablation 诱发的 terminal state；若该预注册分支 discovery 失败，则使用预先冻结的 within-example mediator restoration，检验 clean terminal state 是否特异地救回 selected receiver，且该 rescue 是否被模型特异 readout cut 消除。</p>{table(('Model','Bridge design','Targeted write effect','State link / restoration','Cut occlusion','Cut residual ratio','Applicable / planned'), bridge_rows)}</section>
{('<section><h2>5 · Prospective targeted-bank extensions</h2>' + table(('Model','Frozen bank','Endpoint','Same-bank bridge','Protocol result'), extension_rows) + '<p class="note">PASS 仅在同一个预先冻结 bank 的 targeted→final-count endpoint 与 full-span bridge 都通过 confirmation 后成立；若 endpoint discovery 失败，bridge 保持封存。</p>' + metadata_note + '</section>') if extension_rows else ''}
<section><h2>{'6' if extension_rows else '5'} · 推断边界</h2><ul><li>三段 assay 是有向、时间有序且由 integrated bridge 连接的因果证据；不是单一 trial 的自然直接/间接效应分解。</li><li>Qwen 的两条 readout route 是功能类别；Gemma 的 all-head mask 是分布式 route；均不证明唯一最小 circuit。</li><li>Full hidden-state patch 证明状态充分性与传递，不证明 count 只编码在一个标量或一个低维子空间。</li><li>Greedy exact-count 的部分 patch-only supplementary gates 未通过；离散终点由独立 free-generation endpoint confirmation承担。</li></ul></section>
<details><summary>Evidence ledger · SHA-256</summary><p>Generated UTC: {_esc(generated)}</p><ul>{hash_rows}</ul></details>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence, hashes = _read_evidence(args.evidence_root)
    _assert_contract(evidence)
    document = build(evidence, hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(args.output)
    manifest = {
        "schema_version": "realistic_niah_v5_native_count_chain_report_v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "output": str(args.output),
        "output_sha256": _sha(args.output),
        "evidence_sha256": hashes,
        "status": "PASS",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
