"""Build the Additional report exclusively from verified task-local results."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import re
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

B = Path(__file__).resolve().parents[1]
VERSION = 'task_local_disjoint_first_20260908_v3'
NAT = B/'runs/report_refresh_20260908_v1'
MODELS = ['Qwen3-8B', 'Gemma4-E4B']
MODES = ['nonthinking', 'native_thinking']
TASKS = ['kth', 'category']
NAMES = {'nonthinking': 'Non-thinking', 'native_thinking': 'Native-thinking',
         'kth': '第 k 条记录', 'category': '指定类别计数'}
COLORS = {'nonthinking': '#2563a6', 'native_thinking': '#c66a33',
          'clean': '#688070', 'selected': '#c66a33', 'random': '#2563a6', 'delta': '#236294'}


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with p.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def esc(x): return html.escape(str(x))
def table(headers, data):
    return '<div class="tablewrap"><table><thead><tr>'+''.join(f'<th>{esc(x)}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{esc(x)}</td>' for x in r)+'</tr>' for r in data)+'</tbody></table></div>'
def para(s, conclusion=None):
    return '<p>'+s+(f'<span class="conclusion">当前结论：{conclusion}</span>' if conclusion else '')+'</p>'
def ci(r): return f"{100*float(r['mean']):.2f} [{100*float(r['lower']):.2f}, {100*float(r['upper']):.2f}]"
def percent(r): return f"{100*float(r['mean']):.2f}"


class Report:
    def __init__(self, root, allow_pending=False, status='新消融结果尚未通过完整审计'):
        self.root = root
        self.complete = (root/'analysis/audit.json').is_file()
        if not self.complete and not allow_pending:
            raise RuntimeError('Verified new results are required; --allow-pending explicitly creates a progress report')
        self.cfg = read(root/'protocol.json') if (root/'protocol.json').exists() else read(B/'runs'/VERSION/'package/protocol.json')
        assert self.cfg['version'] == VERSION and self.cfg['old_results_reused'] is False
        assert self.cfg['layer_cap'] is False and self.cfg['targeted_max_tokens'] == 256
        assert self.cfg['control_policy']=='disjoint_first_minimum_overlap'
        self.audit = read(NAT/'alignment_audit.json')
        assert self.audit['status'] == 'PASS' and self.audit['checks']['natural_outputs_rescored'] == 2400
        self.natural, self.percase = rows(NAT/'natural_summary.csv'), rows(NAT/'natural_per_case.csv')
        assert len(self.percase) == 2400
        self.figures, self.status, self.summary, self.secondary, self.tests, self.coverage = [], status, [], [], [], []
        if self.complete:
            self.newaudit = read(root/'analysis/audit.json')
            assert self.newaudit['status'] == 'PASS'
            assert self.newaudit['protocol_sha256'] == sha(root/'protocol.json')
            assert self.newaudit['checks']['points'] == self.cfg['expected_full_points'] == 6739
            self.status = '完整结果审计通过'
            self.summary = rows(root/'analysis/summary.csv')
            self.secondary = rows(root/'analysis/clean_correct_summary.csv')
            self.tests = rows(root/'analysis/hypothesis_tests.csv')
            self.coverage = rows(root/'analysis/coverage.csv')
            assert len(self.summary) == 370 and len(self.coverage) == 1200
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
            'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})

    def figure(self, fig, ident, caption):
        with matplotlib.rc_context({'svg.hashsalt': VERSION+'_'+ident}):
            out = io.StringIO(); fig.savefig(out, format='svg', bbox_inches='tight', metadata={'Date': None})
        plt.close(fig)
        svg = out.getvalue(); svg = svg[svg.index('<svg'):]
        svg = re.sub(r'id="([^"]+)"', lambda m: 'id="'+ident+'_'+m[1]+'"', svg)
        svg = re.sub(r'((?:xlink:)?href)="#([^"]+)"', lambda m: m[1]+'="#'+ident+'_'+m[2]+'"', svg)
        svg = re.sub(r'url\(#([^)]+)\)', lambda m: 'url(#'+ident+'_'+m[1]+')', svg)
        self.figures.append(ident)
        return f'<figure id="{ident}">{svg}<figcaption>图{len(self.figures)}｜{caption}</figcaption></figure>'

    def natural_section(self, task):
        data = [r for r in self.natural if r['task'] == task and r['split'] == 'all']
        part = '<h3>'+NAMES[task]+'</h3>'+table(['模型', '模式', '正确 / 输入', '准确率 % [95% CI]', '解析失败', '达到长度上限'],
            [[r['model'], NAMES[r['mode']], f"{r['correct']}/{r['n']}", ci(r), r['unparsed'], r['truncated']] for r in data])
        fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.8), sharey=True)
        levels = list(range(1, 11)) if task == 'kth' else [1, 3, 5, 7, 9]
        for ax, model in zip(axs, MODELS):
            for mode in MODES:
                values = []
                for level in levels:
                    rr = [r for r in self.percase if (r['task'], r['model'], r['mode'], int(r['level'])) == (task, model, mode, level)]
                    assert len(rr) == (30 if task == 'kth' else 60)
                    values.append(100*np.mean([int(r['correct']) for r in rr]))
                ax.plot(levels, values, '-o', color=COLORS[mode], label=NAMES[mode], lw=2, ms=4)
            ax.set(title=model, xticks=levels, ylim=(-2, 102), xlabel='Requested position k' if task == 'kth' else 'Target-category record count')
            ax.grid(axis='y', alpha=.2); ax.legend(fontsize=9)
        axs[0].set_ylabel('Final-answer accuracy (%)'); fig.tight_layout()
        part += self.figure(fig, 'natural_'+task, '完整自然输出的最终答案准确率。左右为两模型，蓝色 Non-thinking、橙色 Native-thinking。横轴为'+('请求序号 k，每点30输入。' if task == 'kth' else '目标类别记录数，每点60输入，city / flower 合并。')+'纵轴为正确率（%，越高越好）。包含错误、解析失败与截断；曲线未画区间，表中为30个 seed 的配对聚类 bootstrap 区间。')
        comparisons = []
        for model in MODELS:
            nt = next(r for r in data if r['model'] == model and r['mode'] == MODES[0])
            native = next(r for r in data if r['model'] == model and r['mode'] == MODES[1])
            comparisons.append(f"{model}：{percent(nt)}% → {percent(native)}%")
        part += para('从 Non-thinking 到 Native-thinking，'+ '；'.join(comparisons)+'。', '两模型的 Native-thinking 自然准确率均较高；该模式比较不能单独证明具体 attention heads 的因果作用。')
        confirmation = [r for r in self.natural if r['task'] == task and r['split'] == 'confirmation']
        part += '<details><summary>固定 confirmation 子集的自然准确率（每组100输入）</summary>'+table(['模型', '模式', '正确 / 输入', '准确率 % [95% CI]'], [[r['model'], NAMES[r['mode']], f"{r['correct']}/{r['n']}", ci(r)] for r in confirmation])+'</details>'
        return part

    def get(self, task, model, mode, assay, k, metric, secondary=False):
        data = self.secondary if secondary else self.summary
        matches = [r for r in data if (r['task'], r['model'], r['mode'], r['assay'], int(r['k']), r['metric']) == (task, model, mode, assay, k, metric)]
        assert len(matches) == 1, (task, model, mode, assay, k, metric)
        return matches[0]

    def sizes(self, model, mode, assay):
        return sorted({int(r['k']) for r in self.summary if (r['model'], r['mode'], r['assay']) == (model, mode, assay)})

    def best_ks(self, task, model, mode, assay):
        candidates = [self.get(task, model, mode, assay, k, 'delta')
                      for k in self.sizes(model, mode, assay)]
        peak = max(float(r['mean']) for r in candidates)
        return sorted(int(r['k']) for r in candidates if float(r['mean']) == peak)

    def effect_table(self, mode, assay, secondary=False, maximum=False):
        data = []
        for task in TASKS:
            for model in MODELS:
                ks = [max(self.sizes(model, mode, assay))] if maximum else self.best_ks(task, model, mode, assay)
                k = ks[0]
                label = str(k) if len(ks) == 1 else str(k)+'（全部 K 并列）' if ks == self.sizes(model, mode, assay) else str(k)+'（并列：'+', '.join(map(str, ks))+'）'
                if secondary and not any(r['task']==task and r['model']==model and r['mode']==mode and r['assay']==assay and int(r['k'])==k for r in self.secondary):
                    data.append([NAMES[task]+' / '+model, k, '0 / 0', '—', '—', '—', '—', '—', '—']); continue
                rr = {metric:self.get(task, model, mode, assay, k, metric, secondary) for metric in ['clean', 'selected', 'random', 'delta', 'clean_drop']}
                test = next(r for r in self.tests if (r['task'], r['model'], r['mode'], r['assay'], int(r['k']), r['population']) == (task, model, mode, assay, k, 'clean_correct' if secondary else 'all_examples'))
                p = f"{float(test['p_holm']):.4f}" if test['p_holm'] else '—'
                data.append([NAMES[task]+' / '+model, label, f"{rr['delta']['n']} / {rr['delta']['seeds']}", percent(rr['clean']), percent(rr['selected']), percent(rr['random']), ci(rr['delta']), ci(rr['clean_drop']), p])
        ident = ('secondary' if secondary else 'maximum' if maximum else 'best')+'_'+assay+'_'+mode
        return '<div id="'+ident+'">'+table(['任务 / 模型', 'K', 'n / seeds', 'Clean %', 'Selected %', 'Random %', 'Δ pp [95% CI]', 'Clean−Selected pp [95% CI]', 'Holm p'], data)+'</div>'

    def ablation_plot(self, mode, assay, effect=False):
        fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.4))
        subset = [r for r in self.summary if r['mode']==mode and r['assay']==assay and r['metric']=='delta']
        ymin = min(-5., min(float(r['lower'])*100 for r in subset)-5)
        ymax = max(5., max(float(r['upper'])*100 for r in subset)+5)
        for i, task in enumerate(TASKS):
            for j, model in enumerate(MODELS):
                ax = axs[i,j]; ks = self.sizes(model, mode, assay)
                for metric in (['delta'] if effect else ['clean', 'selected', 'random']):
                    rr = [self.get(task, model, mode, assay, k, metric) for k in ks]
                    y, lo, hi = [[100*float(r[key]) for r in rr] for key in ['mean', 'lower', 'upper']]
                    ax.plot(ks, y, '-o', lw=1.8, ms=3.5, color=COLORS[metric], label='Random − Selected' if effect else metric.title())
                    ax.fill_between(ks, lo, hi, color=COLORS[metric], alpha=.12)
                if model == MODELS[0] and assay == 'broad': ax.set_xscale('log', base=2)
                ax.set_xticks(ks, [str(k) for k in ks]); ax.set_ylim((ymin,ymax) if effect else (-2,102))
                ax.set_title(('Kth record' if task=='kth' else 'Category count')+' · '+model+f" · n={rr[0]['n']}")
                ax.set_xlabel('Number of ablated heads K'); ax.grid(axis='y', alpha=.2)
                if j == 0: ax.set_ylabel('Random − Selected (pp)' if effect else ('Next-record accuracy (%)' if assay=='targeted' else 'Final-answer accuracy (%)'))
                if effect: ax.axhline(0, color='#777', lw=.8, ls='--')
                ax.legend(fontsize=8, loc='best')
        fig.tight_layout()
        caption = f'{NAMES[mode]} / {assay.title()} 完整剂量扫描。行分别为第 k 条记录与指定类别计数，列分别为 Qwen 与 Gemma；各面板 n 为固定有效输入数，10个 confirmation seeds，所有 K 使用同一批端点。'
        caption += '横轴为消融头数 K（Qwen Broad 为 log₂，其余为线性刻度）。'
        caption += ('纵轴为 Random−Selected（百分点，正值表示选定头损伤更大），虚线为零。' if effect else '纵轴为'+('第一条有效语义记录的实体正确率' if assay=='targeted' else '最终答案准确率')+'（%，越高越好）；绿为 Clean、橙为 Selected、蓝为三组 Random 的均值。')
        caption += '阴影为 seed 配对 bootstrap 的逐点95%区间；包括 clean 错误样本。'+('未识别到记录且未命中精确前缀兜底时计错。' if assay=='targeted' else '解析失败计错。')+'区间未经多重比较校正，检验的 Holm p 另见表。'
        return self.figure(fig, assay+'_'+mode+('_effect' if effect else '_accuracy'), caption)

    def ablation_section(self, mode, assay):
        title = NAMES[mode]+' / '+assay.title()
        part = '<h3>'+title+'</h3>'
        if not self.complete:
            return part+para(esc(self.status)+'。全量结果通过逐样本重评分、头集合、端点覆盖与聚合核验后，本节填入新结果。', '当前不对本版本的消融效应方向或大小下结论。')
        part += para('下表逐任务、模型和模式展示预设网格内 Δ=Random−Selected 最大的 K，即相对匹配随机头的观察效应最强点。并列时列明并列情况，以最小 K 的数值展示。K 依据全部有效 confirmation 端点的结果事后选择，属于探索性峰值汇总；Clean-correct 补充分析沿用该 K。', '这些效应点估计可能因选择而偏高，逐点95%区间未校正选 K；Holm p 仍按原完整网格计算。若要确认所选 K 的可重复效应，需要独立数据验证。')
        part += self.effect_table(mode, assay)
        part += '<details><summary>补充：预设网格最大 K 的结果</summary>'+self.effect_table(mode, assay, maximum=True)+'</details>'
        part += self.ablation_plot(mode, assay)+self.ablation_plot(mode, assay, effect=True)
        claims = []
        for task in TASKS:
            for model in MODELS:
                ks = self.best_ks(task, model, mode, assay); k = ks[0]
                r = self.get(task, model, mode, assay, k, 'delta')
                test = next(t for t in self.tests if (t['task'],t['model'],t['mode'],t['assay'],int(t['k']),t['population'])==(task,model,mode,assay,k,'all_examples'))
                verdict = ('点估计为正，Holm 校正后检验通过' if float(test['p_holm'])<.05 else '点估计为正，Holm 校正后证据不足') if float(r['mean'])>0 else '未观察到选定头造成更大损伤的正向点估计'
                label = f'K={k}' if len(ks) == 1 else 'K='+', '.join(map(str, ks))+' 并列，以下数值取最小 K'
                claims.append(f"{NAMES[task]} / {model}，观察峰值 {label}：Δ={ci(r)} pp；{verdict}。")
        part += para('<br>'.join(claims), '效应应按任务、模型、模式和剂量分别解释；有效端点和头数不同，不能直接据效应数值比较模型机制强弱。')
        part += '<details><summary>补充：同一输出中的 Clean-correct 子集</summary>'+self.effect_table(mode, assay, secondary=True)+para('该子集按同一端点的 Clean 是否正确筛选，未重新生成输出。少于10个有支持的 seed 时仅作描述。主分析仍为全部有效端点。')+'</details>'
        return part

    def prompts(self):
        content = '<details><summary>冻结的实际提示词与模型输入（完整样例）</summary>'
        for task in TASKS:
            for model in MODELS:
                plans = read(B/'runs'/VERSION/'package/plans'/f'{task}_{model}.json')
                cid = next(p['case_id'] for p in plans if p['seed']==1234)
                for mode in MODES:
                    p = next(p for p in plans if p['case_id']==cid and p['mode']==mode)
                    remote = p['source'].split('/additional_experiments/',1)[1]
                    version, rel = remote.split('/',1)
                    source = B/'runs'/version/'downloaded'/rel/'prompt.json'
                    assert sha(source)==p['source_hashes']['prompt.json']
                    prompt = read(source)
                    content += f'<details><summary>{NAMES[task]} · {model} · {NAMES[mode]} · {esc(cid)}</summary><pre>'+esc(prompt['rendered_prompt'])+'</pre></details>'
        return content+'</details>'

    def build(self):
        style = (B/'deployment/additional_report_style.css').read_text(encoding='utf-8')
        style += '.status{background:#fff8e8;border-left:4px solid #ba842c;padding:14px 18px}.tablewrap table{min-width:700px}code{overflow-wrap:anywhere}h2,h3{scroll-margin-top:16px}'
        s = [f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NiaH Additional Tasks — 新任务独立选头实验</title><style>{style}</style></head><body><main>',
            '<header><div class="eyebrow">NIAH · ADDITIONAL TASKS</div><h1>新任务独立选头：准确率与两类消融</h1><p class="meta">两个任务 × 两个模型 · 随机对照优先不重叠 · '+VERSION+'</p></header>',
            '<nav><a href="#tasks">任务与样本</a><a href="#natural">自然准确率</a><a href="#broad">Broad</a><a href="#targeted">Targeted</a><a href="#audit">统计与核验</a></nav>']
        if self.complete:
            s += [para('本报告使用新任务各自的 discovery 数据选头、排序与冻结，并使用本版本重新执行的全部消融结果。', '样本、头集合、评分和结果聚合均通过审计；效应结论分项列于 Broad 与 Targeted。')]
        else:
            s += ['<div class="status"><strong>进度版：新消融尚未完成。</strong> '+esc(self.status)+'。自然准确率已核验；旧协议的消融数值不作为本版本结果。</div>']
        s += ['<section id="tasks"><h2>1. 任务、样本与实验对齐</h2>', table(['项目', '第 k 条记录', '指定类别计数'], [
            ['输入', '长背景中固定10条 city-score 记录；k=1…10', '固定10条记录；city数为1/3/5/7/9，其余为flower；分别问两类别'],
            ['最终答案', '第 k 条的城市和分数均正确', '指定类别记录数完全正确'],
            ['样本', '30 seeds × 10个k = 300输入', '30 seeds × 5种组成 × 2目标 = 300输入'],
            ['选头 / 确认', '20 discovery seeds / 10 confirmation seeds', '20 discovery seeds / 10 confirmation seeds']]),
            para('两个模型使用同一任务实例和 gold，两个任务共享30个 seed 的背景、原始记录顺序与分数。类别任务只按固定规则改变记录类型和实体；每个任务内两种模式保留对应模型自己的 chat template 与输出指令。', '任务输入对齐；不同 tokenizer 与自然思考轨迹产生的 token 位置及有效端点无需数值相同。'),
            '<div class="example">说明性例子：记录顺序为 Paris|71、Tokyo|84、London|63，k=2 的答案为 <code>Needle:Tokyo|84</code>。类别例子包含两条 city 和一条 flower，问 city 时答案为 <code>Total:2</code>。正式输入固定10条记录。</div>',
            self.prompts(), para('两任务分别执行全套实验。Broad 对模型 × 任务 × 模式独立选头；Targeted 对模型 × 任务独立选头。所有选择只使用 discovery seeds 1234–1253，冻结后评价 confirmation seeds 1254–1263。', '确认集不参与本版本选头，但这个 seed 划分已用于旧分析，因此不称为全新的独立确认队列。'), '</section>',
            '<section id="natural"><h2>2. 自然生成的最终答案准确率</h2>', para('自然结果来自已核验的2400条保存输出；每任务、模型和模式各300条。每条均从原始输出重新评分，未删除错误样本。Gemma 的第 k 条任务使用统一 SDPA 后重新采集的自然输出。')]
        for task in TASKS: s.append(self.natural_section(task))
        s += ['</section><section id="broad"><h2>3. Broad：最终回答处的消融</h2>',
            para('原v3在每个新任务最终回答 query 上，用本任务 discovery attention 计算 B=M×exp(H)/J。Non-thinking 的 key 范围为原文全部10条记录的完整 token spans；Native-thinking 的 key 范围为最终答案前已注册生成记录的末尾 token，每个位置取一个 token。M 为对这些位置的总 attention mass，H 为位置间归一化质量分布的熵，J 为参与计算的记录位置数。先输入内计算、再 seed 内平均、最后 seed 等权，按全模型分数排序。', '采用真正的全局 Top-K，不设每层半数上限；两种模式的 key 范围不同。后续目标类别变体单列于下文。'),
            para('2026-09-09 方法说明更正：此前报告及冻结 PROTOCOL.md 将两种模式均描述为完整原文记录 span。与实际运行 contract 匹配的源代码确认：Native-thinking 使用注册轨迹记录的末尾 token。冻结文件保留原样，本报告按执行事实更正。', '这是执行与冻结文字说明之间的差异；结果内部一致性审计通过不能替代这一方法核对。'),
            para('原v3 category 的选头数据和准确率汇总均合并“数 city”与“数 flower”两类问题。Non-thinking 的 Broad score 包含两类别共10条原文记录，未按 is_target 过滤；Native-thinking 包含自然轨迹中的全部已注册记录末尾位置，也未额外按目标类别过滤，实际覆盖哪些类别取决于已注册轨迹。最终答案仍只与题目要求的目标类别数量比较。', '原v3选头未按目标类别筛选；下述Non-thinking后续实验已改为按每题所问类别筛选记录。'),
            para('第 k 条记录任务要求定位给定序号的记录，任务定义不要求在最终回答位置汇总全部记录。Targeted 直接检验注册的下一记录生成；kth 的 Broad 可作为最终回答阶段依赖性的补充检验。', 'kth 的弱 Broad 效应不直接证明模型在整个解题过程中无需广泛读取记录。'),
            para('干预每个输入自己最终答案之前的 query，置零选定头的 pre-O 输出切片；只作用于该次 prefill 的回答 query，最多继续生成64 tokens。Broad 在两任务均以最终任务答案是否正确评分。'),
            para('三组 Random 逐层匹配 Selected 头数。未选头够用时，只在未选头中均匀无放回抽样；不够时，包含该层全部未选头，仅从 Selected 中随机补足缺口。每层不可避免的重叠数为 max(0, 2×所选头数−该层总头数)。各随机组之间允许重复。Broad 随机种子为7000–7002，Targeted 为6000–6002。', '能不重叠的层完全不重叠；只有容量不足的层使用最低必要重叠，不削减 Selected 的全局 Top-K。')]
        bank_audit_path=B/'runs'/VERSION/'runtime_bank_audit.json'
        if bank_audit_path.exists():
            bank_audit=read(bank_audit_path);assert bank_audit['status']=='PASS'
            assert bank_audit['counts']['banks']==12 and bank_audit['counts']['doses']==74
            required=[]
            for r in bank_audit['maximum_dose_overlap']:
                for layer in r['layers']:
                    required.append([NAMES[r['task']]+' / '+NAMES[r['mode']]+' / '+r['assay'].title(),r['model'],r['k'],layer['layer'],f"{layer['selected_count']} / {layer['width']}",layer['actual_overlap']])
            s.append('<details><summary>实际冻结头集合：哪些层需要允许重叠？</summary>'+table(['任务 / 模式 / 干预','模型','K','层（0起）','Selected / 层总头数','每组必要重叠'],required)+para('12组曲线、74个剂量设置均通过逐层核验。全部 Targeted 与 Native-thinking Broad 均可使用不重叠对照；上表两项 Non-thinking Broad 在最大K时需要最低数量的重叠。')+'</details>')
        from category_target_broad_report import section as category_target_section
        extra_section, extra_metadata, extra_hashes = category_target_section(self, table, para, ci, percent)
        s.append(extra_section)
        from native_broad_full_span_report import section as native_full_span_section
        native_section, native_metadata, native_hashes = native_full_span_section(self, table, para, ci, percent)
        s.append(native_section)
        if extra_section:s.append('<h3>原v3参考：各模式既有Broad设置</h3>')
        for mode in MODES: s.append(self.ablation_section(mode,'broad'))
        s += ['</section><section id="targeted"><h2>4. Targeted：下一条记录的局部检索</h2>',
            para('每个新任务在自己的 Native-thinking 自然轨迹上定位已注册的下一记录事件。使用该事件 query 对目标原文记录完整 token span 的 raw attention mass 排名；先 seed 内平均，再 seed 等权。所有头名单由本任务 discovery 数据产生并冻结。', '端点坐标与目标实体来自各自新任务，旧任务头名单不替代新任务选头。'),
            para('在冻结查询及后续 decode 持续置零所选 pre-O head slices，最多继续生成256 tokens，与已核实的论文主实验预算一致。取生成中的第一条有效语义记录，判断它的实体是否等于注册的下一记录；两任务、所有条件采用同一解析和评分口径。', 'Targeted 的主终点是下一条有效记录的实体正确率。最终任务答案会受后续步骤影响，不替代这个局部指标。'),
            '<div class="example">注册顺序为 Paris → Tokyo → London，在 Tokyo 后干预：第一条有效记录是 London 则记正确，即使最后整题答错；第一条是 Paris 则记错误，即使后来改成 London。没有有效记录且未满足下述精确前缀兜底时记错误。</div>',
            para('评分边界沿用主实验：仅当未识别到语义记录、且从注册目标位置（offset=0）开始的原始 target token IDs 完全匹配时，允许精确前缀兜底。若已识别到错误的首条记录，兜底不能把它改判正确。city / flower 均采用同一规则。'),
            self.ablation_section('native_thinking','targeted'), '</section>',
            '<section id="audit"><h2>5. 统计口径、核验与适用范围</h2><div class="formula">Aₛᶜ = seed s 内有效输入的正确率；Aᶜ = 各 seed 的等权平均<br>Aᴿ = 三组随机头的平均正确率<br>Δ = Aᴿ − Aˢᵉˡᵉᶜᵗᵉᵈ；另报告 Aᶜˡᵉᵃⁿ − Aˢᵉˡᵉᶜᵗᵉᵈ，单位均为百分点</div>',
            para('每组100个 confirmation 候选输入。端点不可定位时保留候选及原因，效应估计只纳入有效端点；不将缺失端点擅自计成正确或错误。相同端点的 Clean / Selected / Random 配对，所有 K 共用同一 Clean 与样本集合。主分析包括 Clean 答错的端点；Clean-correct 是补充分析。'),
            para('对10个 confirmation seeds 配对 bootstrap 20,000次，随机种子20260907，取2.5%和97.5%分位数。曲线区间为逐点区间；精确双侧 seed sign-flip 检验另在每种 assay × population 内跨任务、模型、模式与全部 K 做 Holm 校正。', '逐点区间未跨零不等于通过全扫描的多重比较校正；报告同时提供 Holm p。'),
            para('自然输入与评分核验：4800项源文件哈希、2400条自然输出重评分、600个任务实例及1200组跨模型提示词匹配均通过。旧协议的消融审计只能说明旧记录内部一致；本版本的消融结论必须等待本版本完整结果审计。' if not self.complete else '自然输入与评分核验通过；本版本6739个输入 × K 数据点及33695个条件输出逐一重评分，并核验冻结头集合、随机重叠、SDPA、端点前缀和汇总结果。')]
        if self.complete:
            grouped = defaultdict(list)
            for r in self.coverage: grouped[r['task'],r['model'],r['mode'],r['assay']].append(r)
            s.append(table(['任务','模型','模式','干预','有效 / 候选','不可用原因'], [[NAMES[t],m,NAMES[mode],a,sum(r['available']=='True' for r in rr),'; '.join(sorted({r['reason'] for r in rr if r['reason']})) or '无'] for (t,m,mode,a),rr in sorted(grouped.items())]).replace('<th>有效 / 候选</th>','<th>有效数（候选100）</th>'))
        s += [para('原始自然输出与划分沿用已核验数据；两类消融按新协议重新运行。端点来自自然轨迹，有效样本可能随模型和任务不同。局部 Targeted 结果说明下一记录生成的依赖关系，不能单独证明完整推理链、类别判断或最终答案的唯一机制。'),
            '<details><summary>数据来源与复现</summary>'+table(['内容','路径（相对 additional_experiments）'],[
                ['冻结协议、源码与输入 plans',f'runs/{VERSION}/package/protocol.json'],
                ['自然输出重新评分与输入核验','runs/report_refresh_20260908_v1/alignment_audit.json'],
                ['自然准确率','runs/report_refresh_20260908_v1/natural_summary.csv'],
                ['新消融主表',f'runs/{VERSION}/downloaded/analysis/summary.csv'],
                ['补充子集与多重比较',f'runs/{VERSION}/downloaded/analysis/clean_correct_summary.csv; hypothesis_tests.csv'],
                ['逐样本新结果审计',f'runs/{VERSION}/downloaded/analysis/audit.json'],
                ['逐层随机对照与必要重叠审计',f'runs/{VERSION}/downloaded/analysis/random_control_geometry.csv'],
                ['本报告生成器','deployment/build_task_local_html.py']])+para('图形由上述数值生成并内嵌为 SVG，HTML 无外部字体或绘图库依赖。未完成时，新消融文件路径表示预定产物。')+'</details></section>']
        metadata = dict(version=VERSION, complete=self.complete, generated_utc=datetime.now(timezone.utc).isoformat(),
            figures=self.figures, natural=self.natural, ablation=self.summary, status=self.status,
            source_hashes={str(p.relative_to(B)):sha(p) for p in [NAT/'natural_summary.csv', NAT/'alignment_audit.json']})
        if self.complete:
            metadata['k_selection'] = dict(
                rule='argmax_random_minus_selected', population='all_examples',
                data_split='confirmation', interpretation='post_hoc_exploratory',
                tie_break='smallest_k', secondary_uses_main_k=True,
                intervals='pointwise_not_selection_adjusted', holm_family='unchanged_full_grid',
                panels=[dict(task=task, model=model, mode=mode, assay=assay,
                             k=self.best_ks(task, model, mode, assay)[0],
                             tied_ks=self.best_ks(task, model, mode, assay))
                        for assay, mode in [('broad', m) for m in MODES]+[('targeted', 'native_thinking')]
                        for task in TASKS for model in MODELS])
            for name in ['summary.csv','clean_correct_summary.csv','hypothesis_tests.csv','coverage.csv','audit.json']:
                source = self.root/'analysis'/name
                metadata['source_hashes'][str(source.relative_to(B))] = sha(source)
        if bank_audit_path.exists():metadata['source_hashes'][str(bank_audit_path.relative_to(B))]=sha(bank_audit_path)
        if extra_metadata is not None:
            metadata['target_category_broad'] = extra_metadata
            metadata['source_hashes'].update(extra_hashes)
        if native_metadata is not None:
            metadata['native_full_span_broad'] = native_metadata
            metadata['source_hashes'].update(native_hashes)
        scope_audit_path = B/'runs'/VERSION/'category_broad_scope_audit_20260909.json'
        if scope_audit_path.exists():
            scope_audit = read(scope_audit_path)
            assert scope_audit['status'] == 'PASS'
            metadata['broad_scope_audit'] = scope_audit
            metadata['source_hashes'][str(scope_audit_path.relative_to(B))] = sha(scope_audit_path)
        payload = json.dumps(metadata,ensure_ascii=False).replace('<','\\u003c')
        s += [f'<script type="application/json" id="report-data">{payload}</script>', '<footer class="foot">NiaH Additional Tasks · 2026-09-09 · '+('新消融结果已审计' if self.complete else '进度版：新消融待完成')+'</footer></main></body></html>']
        return ''.join(s)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=B/'runs'/VERSION/'downloaded'); ap.add_argument('--allow-pending',action='store_true'); ap.add_argument('--status',default='新消融结果尚未通过完整审计'); args=ap.parse_args()
    started=time.perf_counter(); report=Report(args.root.resolve(),args.allow_pending,args.status); text=report.build()
    destinations=[B.parents[1]/'NiaH_Additional-tasks_report.html',B.parent/'reports/NiaH_Additional-tasks_report.html']
    for path in destinations:
        path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.html.tmp'); tmp.write_text(text,encoding='utf-8'); tmp.replace(path)
    result=dict(complete=report.complete,figures=len(report.figures),paths=[str(p) for p in destinations],sha256=sha(destinations[0]),elapsed_seconds=time.perf_counter()-started)
    payload=json.loads(re.search(r'<script type="application/json" id="report-data">(.*?)</script>',text,re.S)[1])
    result['component_completion']={VERSION:report.complete,**{payload[key]['version']:payload[key]['complete'] for key in ['target_category_broad','native_full_span_broad'] if key in payload}}
    result['all_requested_components_complete']=all(result['component_completion'].values())
    assert sha(destinations[1])==result['sha256']
    (B/'runs'/VERSION/'report_build.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (B/'runs/current_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__': main()
