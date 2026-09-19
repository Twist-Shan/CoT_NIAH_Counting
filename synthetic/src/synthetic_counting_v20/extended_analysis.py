"""Supplemental high-power behavior and interactive attention diagnostics for v20."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .attention_metrics import broad_profile_metrics as _broad_profile_metrics
from .config import V20Config, config_from_dict
from .data import V20Example, V20Rendered, V20Vocab, collate_v20, load_corpus_split, load_corpus_text, load_suite_manifests, render_v20
from .model import build_model
from .needle_pool import load_needle_pool
from .training import atomic_csv, checkpoint_steps


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _balanced_split(
    examples: Sequence[V20Example], count_max: int, per_count: int, *, offset: int = 0
) -> list[V20Example]:
    result: list[V20Example] = []
    for count in range(1, count_max + 1):
        bucket = [item for item in examples if int(item.count or 0) == count]
        selected = bucket[offset : offset + per_count]
        if len(selected) != per_count:
            raise ValueError(f"count={count}: need {offset + per_count} held-out examples")
        result.extend(selected)
    return result


@torch.inference_mode()
def _broad_metric_matrices(
    model,
    cfg: V20Config,
    vocab: V20Vocab,
    items: Sequence[V20Rendered],
) -> tuple[dict[str, np.ndarray], int]:
    totals = {
        name: np.zeros((cfg.n_layer, cfg.n_head), dtype=np.float64)
        for name in (
            "total_target_mass",
            "effective_number",
            "effective_coverage",
            "normalized_entropy",
            "broad_score",
            "entropy_broad_score",
            "legacy_entropy_broad_score",
        )
    }
    observations = 0
    batch_size = min(8, cfg.analysis_batch_size)
    for start in range(0, len(items), batch_size):
        batch = list(items[start : start + batch_size])
        ids, _, mask = collate_v20(batch, vocab, cfg.device)
        output = model(input_ids=ids, attention_mask=mask, output_attentions=True)
        assert output.attentions is not None
        for row, item in enumerate(batch):
            assert item.spans is not None
            needles = list(item.prompt_needle_positions)
            for layer, weights in enumerate(output.attentions):
                values = (
                    weights[row, :, item.spans.ans_pos, needles]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                metrics = _broad_profile_metrics(values)
                for name, metric_values in metrics.items():
                    totals[name][layer] += metric_values
            observations += 1
    denominator = max(observations, 1)
    return {name: values / denominator for name, values in totals.items()}, observations


@torch.inference_mode()
def _broad_score_matrix(
    model,
    cfg: V20Config,
    vocab: V20Vocab,
    items: Sequence[V20Rendered],
) -> tuple[np.ndarray, int]:
    """Backward-compatible wrapper returning the effective-coverage score."""

    metrics, observations = _broad_metric_matrices(model, cfg, vocab, items)
    return metrics["broad_score"], observations


def _best_head(matrix: np.ndarray) -> tuple[int, int]:
    layer, head = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
    return int(layer + 1), int(head)


def collect_dense_attention_roles(run_dir: str | Path, *, device: str | None = None) -> Path:
    """Aggregate four head-role maps over all saved scientific checkpoints."""

    run_dir = Path(run_dir).resolve()
    cfg = config_from_dict(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    from dataclasses import replace

    cfg = replace(cfg, device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = V20Vocab.load(run_dir / "vocab.json")
    corpus = load_corpus_text()
    split = load_corpus_split(run_dir / "data/corpus_split.json", cfg, corpus)
    pool = load_needle_pool(
        run_dir / "data/needle_pool.json",
        cfg,
        split_fingerprint=split.split_fingerprint,
        vocab_fingerprint=vocab.fingerprint,
    )
    curves, _ = load_suite_manifests(
        run_dir / "data/loss_suite_manifests.json",
        split_fingerprint=split.split_fingerprint,
        pool_fingerprint=pool.pool_fingerprint,
    )
    heldout = list(curves["heldout"]["task"])
    selection = _balanced_split(
        heldout, cfg.count_max_threshold, cfg.phase_head_selection_examples_per_count
    )
    reporting = _balanced_split(
        heldout,
        cfg.count_max_threshold,
        cfg.phase_examples_per_count,
        offset=cfg.phase_head_selection_examples_per_count,
    )

    fixed_roles = json.loads(
        (run_dir / "analysis/phase_transition/fixed_head_roles.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    broad_fixed: dict[str, dict[str, int]] = {}
    for mode in ("nonthinking", "thinking"):
        entries = checkpoint_steps(run_dir, "rope", mode)
        if not entries:
            raise FileNotFoundError(f"no dense snapshots for rope/{mode}")
        by_shard: dict[Path, list[int]] = {}
        for step, shard in entries:
            by_shard.setdefault(shard, []).append(step)
        model = build_model(cfg, vocab, "rope", cfg.device).eval()
        final_shard = entries[-1][1]
        payload = torch.load(final_shard, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dicts"][str(cfg.train_steps)])
        selection_matrix, _ = _broad_score_matrix(
            model,
            cfg,
            vocab,
            [render_v20(example, vocab, mode) for example in selection],
        )
        fixed_layer, fixed_head = _best_head(selection_matrix)
        role = f"{mode}_broad"
        broad_fixed[role] = {"layer": fixed_layer, "head": fixed_head}
        del payload

        reporting_items = [render_v20(example, vocab, mode) for example in reporting]
        for shard, steps in by_shard.items():
            payload = torch.load(shard, map_location="cpu", weights_only=False)
            for step in sorted(steps):
                model.load_state_dict(payload["model_state_dicts"][str(step)])
                metric_matrices, observations = _broad_metric_matrices(
                    model, cfg, vocab, reporting_items
                )
                matrix = metric_matrices["broad_score"]
                for layer in range(cfg.n_layer):
                    for head in range(cfg.n_head):
                        rows.append(
                            {
                                "step": int(step),
                                "role": role,
                                "mode": mode,
                                "layer": layer + 1,
                                "head": head,
                                "score": float(matrix[layer, head]),
                                "broad_score": float(matrix[layer, head]),
                                "total_target_mass": float(
                                    metric_matrices["total_target_mass"][layer, head]
                                ),
                                "effective_number": float(
                                    metric_matrices["effective_number"][layer, head]
                                ),
                                "effective_coverage": float(
                                    metric_matrices["effective_coverage"][layer, head]
                                ),
                                "normalized_entropy": float(
                                    metric_matrices["normalized_entropy"][layer, head]
                                ),
                                "entropy_broad_score": float(
                                    metric_matrices["entropy_broad_score"][layer, head]
                                ),
                                "legacy_entropy_broad_score": float(
                                    metric_matrices["legacy_entropy_broad_score"][
                                        layer, head
                                    ]
                                ),
                                "is_fixed_role_head": float(
                                    (layer + 1, head) == (fixed_layer, fixed_head)
                                ),
                                "observations": observations,
                                "selection_split": "disjoint_final_checkpoint",
                                "reporting_split": "heldout_reporting",
                                "score_definition": (
                                    "mean(total_target_mass*exp(shannon_entropy)"
                                    "/num_target_occurrences)"
                                ),
                            }
                        )
            del payload
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del model

    existing = pd.read_csv(
        run_dir / "analysis/phase_transition/tables/dense_fixed_head_dynamics.csv"
    ).copy()
    existing["mode"] = "thinking"
    existing["selection_split"] = "disjoint_final_checkpoint"
    existing["reporting_split"] = "heldout_reporting"
    combined = pd.concat((pd.DataFrame(rows), existing), ignore_index=True, sort=False)
    role_order = {
        "nonthinking_broad": 0,
        "thinking_broad": 1,
        "targeted_retrieval": 2,
        "marker_successor": 3,
    }
    combined["_order"] = combined["role"].map(role_order)
    combined = combined.sort_values(["step", "_order", "layer", "head"]).drop(columns="_order")
    output_dir = run_dir / "analysis/extended"
    table_path = output_dir / "tables/attention_role_dynamics.csv"
    atomic_csv(combined, table_path)
    roles = {
        **broad_fixed,
        "targeted_retrieval": fixed_roles["targeted_retrieval"],
        "marker_successor": fixed_roles["marker_successor"],
    }
    _atomic_text(output_dir / "fixed_attention_roles.json", json.dumps(roles, indent=2))
    write_attention_dynamics_html(
        combined, roles, output_dir / "interactive_attention_dynamics.html"
    )
    return output_dir


def write_attention_dynamics_html(
    frame: pd.DataFrame,
    fixed_roles: dict[str, dict[str, int]],
    output: str | Path,
) -> None:
    output = Path(output)
    roles = [
        ("nonthinking_broad", "Nonthinking broad (M × effective coverage)"),
        ("thinking_broad", "Thinking broad (M × effective coverage)"),
        ("targeted_retrieval", "Thinking targeted retrieval"),
        ("marker_successor", "Thinking successor-like"),
    ]
    n_layer = int(frame["layer"].max())
    n_head = int(frame["head"].max()) + 1
    payload: dict[str, Any] = {
        "steps": sorted(int(value) for value in frame.step.unique()),
        "n_layer": n_layer,
        "n_head": n_head,
        "roles": {},
    }
    for role, label in roles:
        selected = frame[frame.role == role]
        values: dict[str, list[float]] = {}
        for step, group in selected.groupby("step"):
            matrix = np.full((n_layer, n_head), np.nan)
            for row in group.itertuples(index=False):
                matrix[int(row.layer) - 1, int(row.head)] = float(row.score)
            values[str(int(step))] = [round(float(value), 6) for value in matrix.reshape(-1)]
        payload["roles"][role] = {
            "label": label,
            "fixed": fixed_roles[role],
            "max": float(np.nanmax(selected.score.to_numpy(dtype=float))),
            "values": values,
        }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>v20 attention role dynamics</title>
<style>
:root{{--bg:#f8fafc;--panel:#fff;--text:#13233a;--muted:#5c6b7d;--border:#d9e2ec;--accent:#145c91;--hot:#e66b3d}}*{{box-sizing:border-box}}body{{margin:0;padding:18px;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}}main{{max-width:1500px;margin:auto}}h1{{font-size:22px;margin:0 0 6px}}p{{color:var(--muted);margin:4px 0 14px}}.controls{{display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center;margin:14px 0 18px}}input[type=range]{{width:100%}}#stepValue{{font-variant-numeric:tabular-nums;font-weight:650}}.panels{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}section{{background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:14px;overflow-x:auto}}h2{{font-size:16px;margin:0 0 4px}}.sub{{font-size:12px;color:var(--muted);margin-bottom:10px}}.grid{{display:grid;grid-template-columns:42px repeat({n_head},minmax(46px,1fr));gap:5px;align-items:stretch;min-width:{42 + 51 * n_head}px}}.axis{{display:grid;place-items:center;color:var(--muted);font-size:12px}}.cell{{min-height:58px;border-radius:8px;display:grid;place-items:center;border:2px solid transparent;font-variant-numeric:tabular-nums;color:#102033}}.cell.fixed{{border-color:#121826;box-shadow:inset 0 0 0 1px #fff}}.legend{{height:8px;border-radius:5px;background:linear-gradient(90deg,#edf2f7,#f4b183,#cf3f2d);margin-top:10px}}.scale{{display:flex;justify-content:space-between;color:var(--muted);font-size:11px}}.curve{{margin-top:18px;background:var(--panel);border:1px solid var(--border);border-radius:12px;padding:12px}}svg{{width:100%;height:230px;display:block}}.gridline{{stroke:#d9e2ec;stroke-width:1}}.cursor{{stroke:#111827;stroke-width:1.5;stroke-dasharray:4 3}}.series{{fill:none;stroke-width:2}}.point{{stroke:#fff;stroke-width:1.5}}@media(max-width:760px){{.panels{{grid-template-columns:1fr}}.controls{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Attention head 角色随训练的变化</h1><p>拖动 training step。每个 panel 是 {n_layer} layers × {n_head} heads；broad panel 使用 B=M×exp(H)/N（effective coverage），targeted/successor panel 使用各自原始 role score。黑框是由独立 selection split 在最终 checkpoint 选出的固定 head。每个 panel 使用自己的固定色标，跨 panel 请比较数字，不比较颜色深浅。</p>
<div class="controls"><label for="step">Training step</label><input id="step" type="range" min="0" max="{len(payload['steps'])-1}" value="{len(payload['steps'])-1}" step="1"><output id="stepValue"></output></div><div class="panels" id="panels"></div><div class="curve"><svg id="curve" role="img" aria-label="四个固定 attention head 角色分数随 training step 的折线图"></svg></div></main>
<script>const D={data};const roles=['nonthinking_broad','thinking_broad','targeted_retrieval','marker_successor'];const colors=['#315f9f','#d97745','#6f4aa8','#16877d'];const panels=document.getElementById('panels');function color(v,max){{const t=Math.max(0,Math.min(1,v/Math.max(max,1e-9)));const a=[237,242,247],b=t<.55?[244,177,131]:[207,63,45],u=t<.55?t/.55:(t-.55)/.45,s=t<.55?a:[244,177,131];return `rgb(${{Math.round(s[0]+(b[0]-s[0])*u)}},${{Math.round(s[1]+(b[1]-s[1])*u)}},${{Math.round(s[2]+(b[2]-s[2])*u)}})`}}function renderPanels(step){{panels.innerHTML='';roles.forEach(role=>{{const r=D.roles[role],vals=r.values[String(step)],sec=document.createElement('section');const fixed=`L${{r.fixed.layer}}H${{r.fixed.head}}`;sec.innerHTML=`<h2>${{r.label}}</h2><div class="sub">score range 0–${{r.max.toFixed(3)}} · fixed ${{fixed}}</div>`;const g=document.createElement('div');g.className='grid';const headIds=Array.from({{length:D.n_head}},(_,i)=>i);g.innerHTML='<div></div>'+headIds.map(h=>`<div class="axis">H${{h}}</div>`).join('');for(let l=1;l<=D.n_layer;l++){{g.insertAdjacentHTML('beforeend',`<div class="axis">L${{l}}</div>`);for(let h=0;h<D.n_head;h++){{const v=vals[(l-1)*D.n_head+h],c=document.createElement('div');c.className='cell'+(l===r.fixed.layer&&h===r.fixed.head?' fixed':'');c.style.background=color(v,r.max);c.textContent=v.toFixed(3);c.setAttribute('aria-label',`${{r.label}} layer ${{l}} head ${{h}} score ${{v.toFixed(4)}}`);g.appendChild(c)}}}}sec.appendChild(g);sec.insertAdjacentHTML('beforeend',`<div class="legend"></div><div class="scale"><span>0</span><span>${{r.max.toFixed(3)}}</span></div>`);panels.appendChild(sec)}})}}function drawCurve(step){{const svg=document.getElementById('curve'),W=1100,H=230,m={{l:58,r:20,t:44,b:36}},x=s=>m.l+(s/D.steps[D.steps.length-1])*(W-m.l-m.r),y=v=>m.t+(1-v)*(H-m.t-m.b);svg.setAttribute('viewBox',`0 0 ${{W}} ${{H}}`);let z=`<line class="gridline" x1="${{m.l}}" y1="${{y(0)}}" x2="${{W-m.r}}" y2="${{y(0)}}"/><line class="gridline" x1="${{m.l}}" y1="${{y(.5)}}" x2="${{W-m.r}}" y2="${{y(.5)}}"/><line class="gridline" x1="${{m.l}}" y1="${{y(1)}}" x2="${{W-m.r}}" y2="${{y(1)}}"/><text x="8" y="${{y(.5)+4}}" fill="#5c6b7d">role score</text>`;roles.forEach((role,i)=>{{const r=D.roles[role],idx=(r.fixed.layer-1)*D.n_head+r.fixed.head,pts=D.steps.map(s=>[x(s),y(r.values[String(s)][idx])]);z+=`<polyline class="series" stroke="${{colors[i]}}" points="${{pts.map(p=>p.join(',')).join(' ')}}"/>`;const v=r.values[String(step)][idx],legendX=m.l+(i%2)*520,legendY=14+Math.floor(i/2)*17;z+=`<circle class="point" fill="${{colors[i]}}" cx="${{x(step)}}" cy="${{y(v)}}" r="4"/><text x="${{legendX}}" y="${{legendY}}" fill="${{colors[i]}}">${{r.label}} ${{v.toFixed(3)}}</text>`}});z+=`<line class="cursor" x1="${{x(step)}}" y1="${{m.t}}" x2="${{x(step)}}" y2="${{H-m.b}}"/><text x="${{m.l}}" y="${{H-8}}">0</text><text x="${{W-m.r-36}}" y="${{H-8}}">10000</text><text x="${{W/2-40}}" y="${{H-8}}">training step</text>`;svg.innerHTML=z}}function update(){{const step=D.steps[+document.getElementById('step').value];document.getElementById('stepValue').textContent=step.toLocaleString();renderPanels(step);drawCurve(step)}}document.getElementById('step').addEventListener('input',update);update();</script></body></html>"""
    _atomic_text(output, document)


__all__ = [
    "_broad_profile_metrics",
    "collect_dense_attention_roles",
    "write_attention_dynamics_html",
]
