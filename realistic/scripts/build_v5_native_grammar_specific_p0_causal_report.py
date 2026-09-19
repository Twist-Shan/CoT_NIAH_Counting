#!/usr/bin/env python3
"""Build the final Native-thinking grammar-specific P0 causal report."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports" / "v5_native_grammar_specific_p0"
DEFAULT_OUTPUT = ROOT / "reports" / "NiaH_Native-Thinking_Causal_Ablation_report.html"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _random_counts(row: dict[str, Any]) -> tuple[int, int]:
    n = int(
        row.get(
            "random",
            row.get("global_random", row.get("layer_matched_random", 0)),
        )
    )
    failures = int(
        row.get(
            "random_failures",
            row.get(
                "global_random_failures",
                row.get("layer_matched_random_failures", 0),
            ),
        )
    )
    return n, failures


def _split_summary(full: dict[str, Any], split: str) -> dict[str, Any]:
    selected_n = selected_failures = random_n = random_failures = 0
    for grammar in full["grammars"]:
        row = grammar.get("by_split", {}).get(split, {})
        selected_n += int(row.get("selected", 0))
        selected_failures += int(row.get("selected_failures", 0))
        n, failures = _random_counts(row)
        random_n += n
        random_failures += failures
    selected_rate = _rate(selected_failures, selected_n)
    random_rate = _rate(random_failures, random_n)
    return {
        "split": split,
        "anchors": selected_n,
        "selected_rate": selected_rate,
        "random_rate": random_rate,
        "delta": (
            selected_rate - random_rate
            if selected_rate is not None and random_rate is not None
            else None
        ),
    }


def _dose_rows(dose: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [row for row in dose["overall"] if row["scope"] == "all_registered_grammars"],
        key=lambda row: int(row["bank_size"]),
    )


def _grammar_rows(full: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for grammar in full["grammars"]:
        confirmation = grammar["by_split"]["confirmation"]
        selected_n = int(confirmation.get("selected", 0))
        selected_failures = int(confirmation.get("selected_failures", 0))
        random_n, random_failures = _random_counts(confirmation)
        selected_rate = _rate(selected_failures, selected_n)
        random_rate = _rate(random_failures, random_n)
        rows.append(
            {
                "grammar": grammar["grammar"],
                "full_anchors": int(grammar["anchors"]),
                "confirmation_anchors": selected_n,
                "selected_rate": selected_rate,
                "random_rate": random_rate,
                "delta": (
                    selected_rate - random_rate
                    if selected_rate is not None and random_rate is not None
                    else None
                ),
                "status": "exploratory" if selected_n < 10 else "claim-grade",
                "control": confirmation.get(
                    "random_condition",
                    full.get("random_control_matching", "registered_random"),
                ),
            }
        )
    return rows


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(value)}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _model_section(
    model: str,
    full: dict[str, Any],
    dose: dict[str, Any],
    *,
    image_path: str,
) -> str:
    doses = _dose_rows(dose)
    primary = doses[-1]
    split_rows = [_split_summary(full, split) for split in ("discovery", "confirmation")]
    dose_table = _table(
        ["K", "Confirmation anchors", "Selected failure", "Random failure", "Selected − random"],
        [
            [
                str(row["bank_size"]),
                str(row["confirmation_anchors"]),
                _pct(row["selected_failure_rate"]),
                _pct(row["random_failure_rate"]),
                f"<strong>{_pct(row['selected_minus_random_failure_rate'])}</strong>",
            ]
            for row in doses
        ],
    )
    split_table = _table(
        ["Split", "Anchors", "Selected failure", "Random failure", "Selected − random"],
        [
            [
                row["split"],
                str(row["anchors"]),
                _pct(row["selected_rate"]),
                _pct(row["random_rate"]),
                _pct(row["delta"]),
            ]
            for row in split_rows
        ],
    )
    grammar_table = _table(
        ["Grammar", "Full anchors", "Confirmation anchors", "Selected", "Random", "Δ", "Control", "Status"],
        [
            [
                f"<code>{html.escape(row['grammar'])}</code>",
                str(row["full_anchors"]),
                str(row["confirmation_anchors"]),
                _pct(row["selected_rate"]),
                _pct(row["random_rate"]),
                _pct(row["delta"]),
                f"<code>{html.escape(str(row['control']))}</code>",
                f'<span class="status {row["status"]}">{row["status"]}</span>',
            ]
            for row in _grammar_rows(full)
        ],
    )
    return f"""
    <section id="{html.escape(model)}">
      <div class="kicker">{html.escape(model)} · exact P0</div>
      <h2>{html.escape(model)} dose response</h2>
      <div class="summary-grid">
        <div><span>Primary K</span><strong>{int(primary['bank_size'])}</strong></div>
        <div><span>Confirmation anchors</span><strong>{int(primary['confirmation_anchors'])}</strong></div>
        <div><span>Selected failure</span><strong>{_pct(primary['selected_failure_rate'])}</strong></div>
        <div><span>Selected − random</span><strong>{_pct(primary['selected_minus_random_failure_rate'])}</strong></div>
      </div>
      <figure><img src="{html.escape(image_path)}" alt="{html.escape(model)} P0 dose response"><figcaption>图：confirmation-only、registered-anchor weighted 的整体 dose curve。三条线分别是 selected-bank failure、三组 random control 的 pooled failure，以及两者之差。</figcaption></figure>
      {dose_table}
      <div class="conclusion"><strong>本模型结论。</strong> Primary K={int(primary['bank_size'])} 时，selected failure={_pct(primary['selected_failure_rate'])}，random failure={_pct(primary['random_failure_rate'])}，差值={_pct(primary['selected_minus_random_failure_rate'])}。曲线只使用 confirmation；discovery 不与其混合。</div>
      <h3>Primary-K discovery / confirmation 分离</h3>
      {split_table}
      <h3>Grammar-specific confirmation</h3>
      {grammar_table}
      <p class="small">confirmation anchors &lt;10 的 grammar 仅作 exploratory 描述。若 exact layer-matched control 容量不足，表中明确记录冻结后的 global same-K fallback；这不会改变 selected bank。</p>
    </section>
    """


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = {
        "qwen_full": args.root / "Qwen3-8B" / "qwen_grammar_specific_p0_k128_full_panel_complete.json",
        "qwen_dose": args.root / "Qwen3-8B" / "qwen_grammar_specific_p0_dose_grid_complete.json",
        "gemma_full": args.root / "Gemma4-E4B" / "gemma_grammar_specific_p0_k8_full_panel_complete.json",
        "gemma_dose": args.root / "Gemma4-E4B" / "gemma_grammar_specific_p0_dose_grid_complete.json",
    }
    data = {name: _load(path) for name, path in paths.items()}
    for name, payload in data.items():
        if payload.get("status") != "PASS":
            raise ValueError(f"{name} is not PASS")
        if not payload.get("persistent_ablation", True):
            raise ValueError(f"{name} does not use persistent ablation")
    generated = datetime.now(timezone.utc).isoformat()
    qwen_section = _model_section(
        "Qwen3-8B",
        data["qwen_full"],
        data["qwen_dose"],
        image_path="v5_native_grammar_specific_p0/Qwen3-8B/overall_confirmation_dose_response.svg",
    )
    gemma_section = _model_section(
        "Gemma4-E4B",
        data["gemma_full"],
        data["gemma_dose"],
        image_path="v5_native_grammar_specific_p0/Gemma4-E4B/overall_confirmation_dose_response.svg",
    )
    ledger = "<br>".join(
        f"{html.escape(str(path))}: {_sha(path)}" for path in paths.values()
    )
    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Native-thinking P0 Causal Ablation</title>
<style>
:root{{--paper:#f4efe6;--card:#fffdf8;--ink:#172333;--muted:#657286;--line:#d5cec2;--navy:#12243a;--teal:#0a8f87;--amber:#c47617}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.67 "Segoe UI","Noto Sans SC",sans-serif}}header{{background:linear-gradient(135deg,#102035,#183d4a,#0b7c76);color:white;padding:58px max(24px,calc((100vw - 1120px)/2))}}header h1{{font-size:clamp(35px,5vw,58px);line-height:1.04;margin:10px 0 16px;max-width:900px}}header p{{max-width:900px;font-size:18px;color:#d6e5ea}}.kicker{{text-transform:uppercase;letter-spacing:.11em;font-size:12px;font-weight:850;color:#55c8bd}}nav{{position:sticky;top:0;background:#fffdf8eF;border-bottom:1px solid var(--line);padding:10px 20px;display:flex;gap:18px;z-index:3}}nav a{{color:#164f62;text-decoration:none;font-weight:750}}main{{max-width:1120px;margin:auto;padding:28px 22px 70px}}section{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:28px;margin:22px 0;box-shadow:0 9px 28px #2434470b}}h2{{font-size:30px;margin:5px 0 16px}}h3{{font-size:20px;margin-top:30px}}code{{background:#edf1ef;border:1px solid #dce4e0;padding:1px 5px;border-radius:4px}}.summary-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:20px 0}}.summary-grid div{{border:1px solid var(--line);padding:13px;background:#f7f3ec}}.summary-grid span{{display:block;color:var(--muted);font-size:12px}}.summary-grid strong{{display:block;font-size:23px;color:#0b756e}}figure{{margin:24px 0}}figure img{{display:block;width:100%;border:1px solid var(--line)}}figcaption{{font-size:13px;color:var(--muted);margin-top:9px}}.table-wrap{{overflow:auto;border:1px solid var(--line);margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px 10px;border-bottom:1px solid #e1dbd2;text-align:left;vertical-align:top}}th{{background:#eee9df;white-space:nowrap}}.status{{display:inline-flex;padding:2px 7px;border-radius:99px;font-size:10px;font-weight:850;text-transform:uppercase}}.claim-grade{{background:#dcf1e6;color:#17603a}}.exploratory{{background:#fff0d8;color:#96510e}}.definition,.conclusion{{border-left:5px solid var(--teal);background:#e8f3f0;padding:15px 18px;margin:18px 0}}.conclusion{{border-left-color:var(--amber);background:#fff0d8}}.small{{font-size:12px;color:var(--muted)}}.links{{display:flex;flex-wrap:wrap;gap:9px}}.links a{{background:var(--navy);color:white;padding:8px 11px;border-radius:7px;text-decoration:none}}.audit{{font:11px/1.65 Consolas,monospace;color:var(--muted);word-break:break-all}}@media(max-width:760px){{.summary-grid{{grid-template-columns:1fr 1fr}}section{{padding:18px}}}}@media(max-width:430px){{.summary-grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><div class="kicker">Realistic NIAH · Native-thinking V5 · final P0 intervention</div><h1>Grammar-specific P0<br>Targeted-Retrieval Ablation</h1><p>每个 grammar 在自己的 exact <code>p0_item_end</code> events 内独立 ranking；attention ranking 与 ablation 使用同一 token；从 P0 起把 selected pre-O head slices 持续关闭到 decode 结束。</p></header>
<nav><a href="#design">Design</a><a href="#Qwen3-8B">Qwen</a><a href="#Gemma4-E4B">Gemma</a><a href="#audit">Audit</a></nav><main>
<section id="design"><div class="kicker">Frozen design</div><h2>实验目的与可识别量</h2><p>目的不是询问某个 head 是否“看见 needle”，而是检验：在模型即将从 item k 过渡到 k+1 的 P0 token 上，关闭按正确 next-needle attention mass 排出的 head bank，是否比关闭同层数量匹配的随机 heads 更容易使模型首先生成错误 city。</p><div class="definition"><strong>Ranking。</strong> 每个 seed 内先对该 grammar 的 P0 events 平均，再对 discovery seeds 等权；confirmation 从不参与排 head。<br><strong>Endpoint。</strong> 自由生成的第一个 semantic city 是否等于 registry 中的 next city。<br><strong>Primary contrast。</strong> selected failure − mean(three registered random-control failures)。</div><p>Qwen 使用 K={{32,64,80,96,112,128}}；Gemma 使用 K={{1,2,4,6,8}}。所有 K 都是各 grammar 排名的嵌套前缀。Qwen K32–112 优先 exact layer-matched random，K128 使用 global same-K random；Gemma 优先 exact layer-matched，容量不足时冻结为 global same-K fallback，并在 grammar 表逐项记录。</p></section>
{qwen_section}{gemma_section}
<section id="audit"><div class="kicker">Reproducibility</div><h2>审计与相关页面</h2><div class="links"><a href="NiaH_Native-Thinking_P0_Targeted_Retrieval_Atlas.html">P0 head map / attention atlas</a><a href="NiaH_Native-Thinking_report.html">Native-thinking synthesis</a><a href="NiaH_Native-Thinking_Parser_and_Token_Sites.html">Parser / token sites</a></div><p class="audit">Generated UTC: {html.escape(generated)}<br>{ledger}<br>Schema: realistic_niah_v5_grammar_specific_p0_causal_report_v1</p><div class="conclusion"><strong>总论。</strong> 两模型在 exact P0 都表现出 selection-specific targeted-retrieval necessity，但回路宽度显著不同：Gemma 从 K1 已出现强效应并在 K8 达到高破坏；Qwen 的效应随 bank 变宽而增加，K112–128 才进入明显破坏区。低样本 grammar 只作 exploratory，不用于单类泛化结论。</div></section>
</main></body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
