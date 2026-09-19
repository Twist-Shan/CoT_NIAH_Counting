#!/usr/bin/env python3
"""Summarize only completed, audited CoT supplement results."""
from pathlib import Path
import argparse
import hashlib
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    audit = json.loads((root/'analysis/artifact_audit.json').read_text())
    assert audit['status'] == 'PASS_COMPLETE_ARTIFACT_AUDIT'
    for relative, expected in audit['artifact_sha256'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == expected, relative
    models = ['Qwen3-8B', 'Gemma4-E4B']
    reports = {(m, s):json.loads((root/f'analysis/{m}_{s}.json').read_text())
               for m in models for s in ['dose', 'recovery']}
    assert all(d['status'] == 'COMPLETE_AUDIT' for d in reports.values())
    lines = ['# CoT-reasoning 补实验结果（2026-09-12）', '',
             '冻结方案中的实验已完成。24 个作业、3,040 条条件记录通过完整性审计；其中包括旧/新后端成对复算和小样本验证，不能视为独立样本。所有新增剂量与恢复实验仍使用原有十个 seed。', '',
             'Gemma head 图与 Non-thinking 统一，只显示第 6、12、18、24、30、36、42 层的 56 个 head，保留原 Top-6 标色。局部注意力层按结构省略，没有按得分过滤。', '',
             '## 后端复算', '',
             '- 两个 blanking bank 各完成 500 条成对记录。旧实现逐条复现历史生成；修正后 trace blanking 为 15/100（旧值 12/100），其他条件准确率不变。',
             '- Gemma 前向无编号移植：21/30 对 2/30 self；后续条件成功数为 20/21、12/12、12/12。后向仍为 20/30 对 0/30 self。',
             '- Gemma 10×10 attention 图在两张 H100 上独立计算，十行结果完全一致；与历史八行缓存仍有最大 2.34 个百分点的归一化差异，已披露。', '',
             '## 当前 head bank 的剂量扫描', '',
             '| 模型 | K | 选定 heads：下一项正确率 | 随机：下一项正确率 | 选定 heads：最终计数正确率 | 随机：最终计数正确率 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for model in models:
        d = reports[model, 'dose']
        assert d['full_k_historical_replay']['compared'] == 50 and d['full_k_historical_replay']['changed_trials'] == 0
        by_key = {(r['k'],r['outcome']):r for r in d['summaries']}
        for k in sorted({r['k'] for r in d['summaries']}):
            a,b = by_key[k,'correct_next_needle'],by_key[k,'final_count_correct']
            lines.append(f"| {model} | {k} | {a['selected_mean']:.1%} | {a['random_mean']:.1%} | {b['selected_mean']:.1%} | {b['random_mean']:.1%} |")
    lines += ['', '两模型的 50 条全 bank、随机控制及 clean 记录均逐 token 复现历史结果。Qwen Top-64 下下一项仍有 9/10 正确，但最终计数为 0/10。Gemma Top-1 下两项正确率均为 1/10。不同模型的 K 不对应统一的回路容量。', '',
              '## 自由生成的 carrier 恢复', '',
              '| 模型 | 条件 | 最终计数正确 | 下一项正确 | 截断 | carrier 位置未全部处理 |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for model in models:
        for r in reports[model,'recovery']['summaries']:
            scheduled = str(r['incomplete_carrier_schedule']) if r['arm'].endswith(('restore','matched')) else '不适用'
            lines.append(f"| {model} | {r['arm']} | {r['final_count_correct']}/10 | {r['next_city_correct']}/10 | {r['truncated']}/10 | {scheduled} |")
    lines += ['', '持续 lesion 下，恢复 carrier 未使任一模型恢复最终计数。Qwen 的 local lesion 没有计数损失，因此该条件不支持恢复效应。Gemma 的 local lesion 从 5/10 变为 6/10；恢复相对 lesion 或 matched control 的差值均为 10 个百分点，95% seed-bootstrap 区间 [0,30]，不能判定为可靠提升。', '',
              '恢复使用完整 clean trace 中的未来状态，在事先冻结的绝对位置进行；这是 oracle 干预。控制匹配插入状态向量的 L2 范数，不保证实际扰动范数相同。Gemma 的提前终止造成部分 carrier 位置未访问；全部保留在分母中。阴性结果不等于证明所有恢复方法均无效。', '',
              '## 终态中介区间与复现', '',
              '原始 86 对共同样本保持不变。用 10,000 次 seed-cluster bootstrap、随机种子 20260912 重算：Qwen suffix-mediated fraction 为 47.1% [36.1%,55.1%]，Gemma 为 76.9% [71.6%,88.4%]。点估计复现，残余损伤仍为正，只支持部分中介。', '',
              '冻结代码、输入、运行命令、原始 JSONL、GPU/软件/模型版本及 SHA-256 位于本目录；`all_results_and_source.tar.gz` 保留完整远端快照。`analysis/artifact_audit.json` 记录 1,212 个文件哈希和每次 clamp 的实际覆盖。本文档生成时已再次校验全部原始文件哈希。', '',
              '这些结果补齐了已冻结的实现复算、剂量和行为恢复检验；没有建立内容无关的标量 counter、纯 +1 运算或完整串行回路。']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'status':'PASS_LOCAL_HASH_VERIFICATION', 'files':len(audit['artifact_sha256']), 'report':str(root/'RESULTS.md')}))


if __name__ == '__main__':
    main()
