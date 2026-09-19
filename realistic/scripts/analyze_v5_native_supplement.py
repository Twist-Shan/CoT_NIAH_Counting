#!/usr/bin/env python3
"""Audit downloaded Native supplement results and build a reproducible report.

No GPU inference is performed. Incomplete or mismatched runs are rejected.
Historical data and the frozen execution bundle are never modified.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from realistic_niah_v5.native_supplement import (
    ANSWER_SEEDS, LAYERS, MODELS, PROGRESS_SEEDS, compare_replay, digest,
    read_rows, summarize_backward, validate_answer, validate_pairs,
    validate_progress, verify_bundle, write_json,
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_out(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_csv_values(expected: Path, actual: Path) -> None:
    left, right = pd.read_csv(expected), pd.read_csv(actual)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, atol=1e-12, rtol=1e-12)


def analyze(bundle: Path, runs: Path, out: Path) -> dict:
    plan = verify_bundle(bundle)
    out.mkdir(parents=True, exist_ok=True)
    sources = {}

    def record(path: Path):
        sources[str(path.resolve())] = digest(path)
        return path

    record(bundle / "plan.json")
    record(Path(__file__))
    record(ROOT / "src/realistic_niah_v5/native_supplement.py")
    spec = importlib.util.spec_from_file_location(
        "frozen_native_answer_analysis", bundle / "code/scripts/analyze_v5_answer_query_layer_sweep.py"
    )
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    dense_summary, layer_export, audits, common = {}, [], {}, []
    for model, name in zip(MODELS, ("qwen_dense", "gemma_dense")):
        run, inputs = runs / name, bundle / "inputs" / model
        completion = load_json(record(run / "completion_audit.json"))
        manifest = load_json(record(run / "run_manifest.json"))
        assert completion["status"] == "PASS" and completion["model"] == model
        assert completion["task"] == "answer_dense" and completion["fresh_confirmation"] is False
        assert completion["plan_sha256"] == manifest["plan_sha256"] == plan["plan_sha256"]
        pairs = read_rows(record(inputs / "pairs.jsonl"))
        common.append(sorted(validate_pairs(pairs, model)))
        trials = run / "dense/trials.jsonl"
        record(trials)
        assert completion["trials_sha256"] == digest(trials)
        rows = read_rows(trials)
        grid = validate_answer(rows, pairs, list(range(LAYERS[model])))
        assert grid["trials"] == completion["trials"] == 80 * LAYERS[model]
        replay = compare_replay(
            read_rows(record(inputs / "historical_answer_trials.jsonl")),
            read_rows(record(run / "replay/trials.jsonl")),
        )
        assert replay == load_json(record(run / "historical_replay_audit.json"))
        assert replay["status"] == "PASS" and replay["compared_trials"] == 640
        for phase in ("replay", "missing"):
            loaded = load_json(record(run / phase / "loaded_model.json"))
            assert loaded["layers"] == LAYERS[model] and loaded["training"] is False
            assert loaded["effective_attention_backend"] == plan["attention_backend"]
            assert loaded["dtype"] == "torch.bfloat16"
            assert loaded["model_spec"] == plan["models"][model]["model_spec"]
            assert loaded["observed_revision"] in (None, loaded["model_spec"]["revision"])
            assert load_json(record(run / phase / "process_status.json"))["exit_code"] == 0
        recomputed = out / "recomputed" / name
        result = frozen.analyze(trials, inputs / "pairs.jsonl", recomputed,
                                expected_layers=list(range(LAYERS[model])))
        remote = load_json(record(run / "dense/analysis/audit.json"))
        assert remote["status"] == "passed" and remote["layers"] == list(range(LAYERS[model]))
        assert remote["trials_sha256"] == digest(trials)
        assert remote["pairs_sha256"] == digest(inputs / "pairs.jsonl")
        assert remote["bootstrap_repetitions"] == 10000
        for filename in ("layer_effects.csv", "seed_effects.csv", "pair_effects.csv", "detail.csv"):
            check_csv_values(record(run / "dense/analysis" / filename), recomputed / filename)
        layers = pd.read_csv(recomputed / "layer_effects.csv").to_dict("records")
        for layer in layers:
            measured = [r for r in rows if r["layer"] == layer["layer"] and r["condition"] == "full_donor_patch"]
            successes = sum(r["prediction"] == r["donor_count"] for r in measured)
            assert len(measured) == 40 and abs(successes / 40 - layer["full_donor_adoption"]) < 1e-12
            layer_export.append({
                "model": model, "layer_index_zero_based": int(layer["layer"]),
                "layer_one_based": int(layer["layer"]) + 1,
                "target_adoptions": successes, "pairs": 40, "seed_clusters": 10,
                "adoption_rate": layer["full_donor_adoption"],
                "ci95_low": layer["full_donor_adoption_ci95_low"],
                "ci95_high": layer["full_donor_adoption_ci95_high"],
                "invalid_outputs": sum(r["prediction"] not in range(1, 11) for r in measured),
                "receiver_retained": sum(r["prediction"] == r["gold_count"] for r in measured),
            })
        endpoint = [r for r in layer_export if r["model"] == model][-1]
        onset = result["descriptive_onset_layer"][model]
        dense_summary[model] = {
            "layers": LAYERS[model], "records": len(rows), "replay_records": 640,
            "new_layer_records": len(rows) - 640, "self_patch_correct": len(rows) // 2,
            "endpoint": endpoint, "descriptive_first_half_adoption_layer_one_based": None if onset is None else onset + 1,
            "elapsed_seconds": completion["elapsed_seconds"],
        }
        audits[name] = {"grid": grid, "replay": replay, "all_four_csv_exports_recomputed": True}
    assert common[0] == common[1]
    csv_out(out / "answer_all_layers.csv", layer_export)

    run, inputs = runs / "gemma_backward", bundle / "inputs/Gemma4-E4B"
    completion = load_json(record(run / "completion_audit.json"))
    manifest = load_json(record(run / "run_manifest.json"))
    assert completion["status"] == "PASS" and completion["trials"] == 90
    assert completion["task"] == "gemma_backward" and completion["model"] == "Gemma4-E4B"
    assert completion["plan_sha256"] == manifest["plan_sha256"] == plan["plan_sha256"]
    trials = run / "backward/trials.jsonl"
    assert completion["trials_sha256"] == digest(record(trials))
    rows = read_rows(trials)
    validate_progress(rows, [4, 6, 8], direction="backward")
    replay_rows = read_rows(record(run / "replay_forward_k6/trials.jsonl"))
    validate_progress(replay_rows, [6], direction="forward")
    replay = compare_replay(read_rows(record(inputs / "historical_forward_k6.jsonl")), replay_rows, progress=True)
    assert replay == load_json(record(run / "historical_replay_audit.json"))
    assert replay["status"] == "PASS" and replay["compared_trials"] == 30
    assert len({r["targeted_bank_sha256"] for r in replay_rows}) == 1
    assert {r["targeted_bank_sha256"] for r in rows} == {r["targeted_bank_sha256"] for r in replay_rows}
    for phase in ("replay_forward_k6", "backward_k4", "backward_k6", "backward_k8"):
        geometry = read_rows(record(run / phase / "geometry_audit.jsonl"))
        assert sorted(r["seed"] for r in geometry) == PROGRESS_SEEDS
        for row in geometry:
            assert row["endpoint_aligned"] and row["aligned_absolute_site"] == row["aligned_donor_site"]
            assert row["deletion_avoids_prompt_records"] and row["deletion_avoids_special_tokens"]
            assert not row["hidden_state_resampling"]
        loaded = load_json(record(run / phase / "loaded_model.json"))
        assert loaded["effective_attention_backend"] == "sdpa" and loaded["dtype"] == "torch.bfloat16"
        assert loaded["layers"] == 42 and loaded["training"] is False
        assert loaded["model_spec"] == plan["models"]["Gemma4-E4B"]["model_spec"]
        assert loaded["observed_revision"] in (None, loaded["model_spec"]["revision"])
        assert load_json(record(run / phase / "process_status.json"))["exit_code"] == 0
    backward = summarize_backward(rows)
    remote_backward = load_json(record(run / "backward/analysis.json"))
    numeric_keys = {"adoption_gain_patch_minus_self", "adoption_gain_ci95", "per_seed"}
    assert {k: v for k, v in backward.items() if k not in numeric_keys} == {
        k: v for k, v in remote_backward.items() if k not in numeric_keys}
    for key in ("adoption_gain_patch_minus_self", "adoption_gain_ci95"):
        np.testing.assert_allclose(backward[key], remote_backward[key], atol=1e-12, rtol=1e-12)
    assert [r["seed"] for r in backward["per_seed"]] == [r["seed"] for r in remote_backward["per_seed"]]
    np.testing.assert_allclose([r["adoption_gain"] for r in backward["per_seed"]],
                               [r["adoption_gain"] for r in remote_backward["per_seed"]], atol=1e-12, rtol=1e-12)
    csv_out(out / "gemma_backward_stepwise_trials.csv", backward["hop_rows"])
    csv_out(out / "gemma_backward_stepwise.csv", backward["stepwise"])
    csv_out(out / "gemma_backward_seed_effects.csv", backward["per_seed"])
    conditions = []
    for k in (4, 6, 8, "all"):
        for condition, label in (("receiver_self", "Receiver self-patch"), ("donor_to_receiver", "Target-to-Receiver patch")):
            selected = [r for r in rows if r["condition"] == condition and (k == "all" or r["donor_occurrence_k"] == k)]
            for row in selected:
                cities = row["generated_known_city_ordinals_any_surface"]
                assert row["first_generated_known_city_ordinal"] == (cities[0] if cities else None)
                assert row["greedy_receiver_successor_retention"] == (row["first_generated_known_city_ordinal"] == row["receiver_successor"])
            conditions.append({
                "target_k": k, "condition": label, "trials": len(selected),
                "target_successor_adoptions": sum(r["greedy_donor_successor_adoption"] for r in selected),
                "receiver_successor_retained": sum(r["greedy_receiver_successor_retention"] for r in selected),
                "truncated_generations": sum(r["generation_truncated"] for r in selected),
            })
    csv_out(out / "gemma_backward_conditions.csv", conditions)
    backward["conditions"] = conditions
    backward["elapsed_seconds"] = completion["elapsed_seconds"]
    audits["gemma_backward"] = {"replay": replay, "geometry_all_phases_passed": True,
                                 "backward_summary_recomputed": True, "records": 90}
    summary = {
        "status": "PASS", "plan_sha256": plan["plan_sha256"],
        "fresh_confirmation": False, "dense": dense_summary, "gemma_backward": backward,
        "audits": audits, "source_sha256": sources,
    }
    write_json(out / "summary.json", summary)
    render_report(out, summary, layer_export)
    return summary


def render_report(out: Path, summary: dict, layers: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7.0, 3.8), layout="constrained")
    for model, label, color in zip(MODELS, ("Qwen", "Gemma"), ("#168DCA", "#E87824")):
        rows = [r for r in layers if r["model"] == model]
        x = [r["layer_one_based"] for r in rows]
        ax.plot(x, [r["adoption_rate"] for r in rows], color=color, label=label, linewidth=1.8)
        ax.fill_between(x, [r["ci95_low"] for r in rows], [r["ci95_high"] for r in rows], color=color, alpha=.16, linewidth=0)
    ax.set(xlabel="Intervention layer (one-based)", ylabel="Target-count adoption",
           xlim=(.5, 42.5), ylim=(-.025, 1.04), xticks=[1, 11, 21, 31, 41])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / f"answer_all_layers.{extension}", dpi=180, metadata={"Creator": "Native supplement audited analysis"} if extension == "pdf" else None)
    plt.close(fig)

    b = summary["gemma_backward"]
    self_row, patch = [r for r in b["conditions"] if r["target_k"] == "all"]
    pct = lambda v: f"{100*v:.1f}%"
    lines = ["# Native-thinking 补充实验（2026-09-11）", "",
        "两模型全层 answer-state 扫描与 Gemma 提示条件下的 backward 干预已完成。历史复测、完整性审计及本地重新统计均通过。", "",
        "本轮在已观察过的 cohort 上补齐测量范围；所有结果属于既有样本扩展。新增层和 backward 设置在运行前冻结。", "",
        "## 1. 全层 answer-state 干预", "",
        "目的：测量完整 answer-query 残差状态在不同层使 Receiver 采用 Target 计数的比例。Target 为状态来源，Receiver 接收该状态。", "",
        "每模型使用同一组 40 对有向样本：seeds 1254–1263，每 seed 四对，两个原始答案均正确。每层分别进行 self-patch 和 Target-state patch，greedy 生成上限为 16 tokens。统计单位为 seed；先在 seed 内平均，再等权平均 10 个 seed，计算 10,000 次 seed-cluster bootstrap 的逐点 95% 区间。", "",
        "说明性示例：某 seed 的四个 patch 中两个采用 Target 计数，该 seed 的采用率为 2/4；不会把同一 seed 的四对样本当作四个独立 seed。无法解析或不在 1–10 内的答案保留为失败。", "",
        "| 模型 | 全部层数 | 新运行记录 | 历史复测差异 | 末层 Target adoption（95% CI） |", "| --- | ---: | ---: | ---: | --- |"]
    for model in MODELS:
        d = summary["dense"][model]
        e = d["endpoint"]
        lines.append(f"| {model} | {d['layers']} | {d['records']} | 0/640 | {e['target_adoptions']}/40 = {pct(e['adoption_rate'])} [{pct(e['ci95_low'])}, {pct(e['ci95_high'])}] |")
    lines += ["", "![全层计数采用率](answer_all_layers.png)", "",
        "横轴为从 1 开始的 decoder 层号，纵轴为 Target 计数采用率；蓝色为 Qwen，橙色为 Gemma，阴影为逐点 95% seed-cluster 区间。每个整数层均有实测值。图中样本均要求原始 Target 与 Receiver 答案正确。", "",
        "全部 3,120 条 self-patch 记录均重新生成 Receiver 的正确计数。6,240 条全层记录全部来自本次运行，其中 1,280 条为历史层复测、4,960 条为此前未测层的记录。", "",
        "结果支持完整残差状态在所测位置对答案输出的局部因果作用。当前实验未匹配绝对 query position，且状态包含内容信息；纯计数分量、必要性和唯一性尚未验证。", "",
        "## 2. Gemma 提示条件下的 backward 干预", "",
        "目的：检验已记录的内容相关进度状态能否将后续输出引向较早的 Target 后继项。使用 N=10、seeds 1276–1285、L17（代码索引 16）的 FOUND no-index 语法；将 Target k=4/6/8 的 item-span 状态移植到 Receiver k+1，共 30 个比较、90 条条件记录。native Target 为 teacher-forced 参照；self 和 patch 最多生成 96 tokens。原 forward k=6 的 30 条记录先行复测，逐条比较通过后执行 backward。", "",
        "说明性示例：Target 已列出第 4 项，Receiver 已列出第 5 项；干预成功的首个新 city 为第 5 项，自然 Receiver 的后继项为第 6 项。该例只说明指标定义。", "",
        f"首 city 的 Target 后继项采用：patch 为 **{patch['target_successor_adoptions']}/30**，self 为 **{self_row['target_successor_adoptions']}/30**。按 seed 等权的 patch−self 差值为 **{100*b['adoption_gain_patch_minus_self']:.1f} 个百分点**，95% CI 为 **[{100*b['adoption_gain_ci95'][0]:.1f}, {100*b['adoption_gain_ci95'][1]:.1f}]**。", "",
        "| 续接步数 | 成功 / 符合条件的样本 | 结构上仍有该项的样本 |", "| --- | ---: | ---: |"]
    for row in b["stepwise"]:
        lines.append(f"| k+{row['hop']} | {row['successes']}/{row['eligible']} | {row['structurally_available']} |")
    lines += ["", f"patch 中 {patch['truncated_generations']}/30 条生成触及上限；self 中 {self_row['truncated_generations']}/30 条触及上限。未输出预期 city 的样本计为失败。只有 k+h>N 属于结构上无后继项；其他样本须此前各步全部正确，才能进入下一步分母。逐条 eligibility、失败和截断记录均保留。", "",
        "Gemma 的结果适用于提示条件下的 no-index 语法及既定 cohort。自然 no-index 机制、纯算术状态与完整自由生成闭环的充分性仍待独立检验。多步成功率的分母经过前序成功筛选，不能解释为全部样本的无条件成功率。", "",
        "## 复现与审计", "",
        f"- 冻结 plan SHA-256：`{summary['plan_sha256']}`。",
        "- Qwen、Gemma D 各 640 条历史复测及 Gemma forward 30 条复测均为零差异。",
        "- 已核对模型 revision、SDPA、bfloat16、层数、输入/输出 hash、完整 pair-layer-condition 网格，以及 backward 的对齐位置和未改动 prompt records 条件。",
        "- 本地重新计算两模型四份 CSV 和 backward 汇总，与远端输出一致；summary.json 记录全部来源文件 SHA-256。",
        "- answer_all_layers.csv 保存 78 个层的采用率、区间及失败数；gemma_backward_conditions.csv 保存分 k/condition 的结果；gemma_backward_stepwise_trials.csv 保存逐条多步分母。", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.bundle.resolve(), args.runs.resolve(), args.output.resolve())
    print(json.dumps({"status": result["status"], "dense_records": sum(d["records"] for d in result["dense"].values()),
                      "backward_records": result["gemma_backward"]["trials"], "output": str(args.output)}, indent=2))
