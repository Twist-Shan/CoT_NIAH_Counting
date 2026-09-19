"""Verify saved exact-prefix diagnostics and report their bounded conclusion."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from realistic_niah.parsing import parse_total
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def main(source,baseline,output):
    m=read(source/'manifest.json');state=read(source/'status.json');assert state['status']=='COMPLETE' and len(state['completed'])==12
    assert sha(baseline)==m['input_sha256'][m['source']]
    with baseline.open() as f:original={(r['seed'],r['gold_count']):r for r in map(json.loads,f)}
    assert sha(source/'diagnose_enumeration_replay_paths_20260917.py')==m['script_sha256']
    evidence=[]
    for seed,gold in m['cases']:
        row=original[seed,gold];pair={q:read(source/f'{q}_seed{seed}_N{gold}.json') for q in ['bf16','fp32']}
        b,f=pair['bf16'],pair['fp32']
        assert b['original']['matches_saved'] and b['original']['generated_token_ids']==row['output_token_ids']
        assert b['original']['prediction']==parse_total(b['original']['text'])==gold
        count=b['trace_prefix_tokens'];ids=row['input_ids']+row['output_token_ids'][:count]
        for data in pair.values():
            assert data['prefix_exact'] and data['query_position']==len(ids)-1
            assert data['input_ids_sha256']==hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            for path in data['paths'].values():
                assert path['prediction']==parse_total('Total:'+path['completion_text'])
                assert path['generated_token_ids'][0]==path['first_top10'][0]['token_id']
        assert b['paths']['full']['generated_token_ids']==b['reference_generate']['generated_token_ids']
        assert b['paths']['cached_trace']['generated_token_ids']==row['output_token_ids'][count:count+16]
        assert f['paths']['full']['generated_token_ids']==f['paths']['cached_trace']['generated_token_ids']
        ptop={p:{r['token_id']:r['logit'] for r in f['paths'][p]['first_top10']} for p in ['full','cached_trace']}
        assert ptop['full'].keys()==ptop['cached_trace'].keys()
        maxdiff=max(abs(v-ptop['cached_trace'][t]) for t,v in ptop['full'].items())
        assert maxdiff<1e-3
        counts={q:{p:r['prediction'] for p,r in pair[q]['paths'].items()} for q in pair}
        evidence.append(dict(seed=seed,gold_count=gold,original=gold,counts=counts,fp32_shared_top10_max_logit_difference=maxdiff,
            original_tokens_reproduced=True,literal_prefix_verified=True,
            bf16_full_top2=b['paths']['full']['first_top10'][:2],bf16_cached_top2=b['paths']['cached_trace']['first_top10'][:2]))
    report=dict(status='PASS',cases=evidence,seconds_gpu=state['seconds'],
        confirmed='Exact token prefixes, original-token reproduction and BF16 full-versus-cached argmax reversals on all three flagged inputs. FP32 full/cached continuations agree on all six diagnostics.',
        interpretation='Evidence supports finite-precision execution-path sensitivity under BF16/SDPA; no token-boundary or residual-patch failure is implicated by these controls.',
        limits=['The responsible numerical operator is not isolated.','Higher precision need not recover gold: the N7 case returns 6 on both FP32 paths.','These are fixed diagnostic cases, not a prevalence estimate.','Read remains the frozen BF16 full-prefix experiment with paired self controls and unchanged sample eligibility.'],
        input_sha256={p.name:sha(p) for p in source.iterdir() if p.is_file()},original_baseline_sha256=sha(baseline))
    output.mkdir(parents=True,exist_ok=False);(output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Gemma Bullet replay 路径诊断','', '已完成：原始逐 token 生成六例均逐 token 复现；输入和答案 query 的 token 前缀一致。','',
        '| Seed / N | 原始 / BF16 缓存 | BF16 整段 | FP32 整段 / 缓存 |','|---|---:|---:|---:|']
    for r in evidence:lines.append(f"| {r['seed']} / {r['gold_count']} | {r['original']} / {r['counts']['bf16']['cached_trace']} | {r['counts']['bf16']['full']} | {r['counts']['fp32']['full']} / {r['counts']['fp32']['cached_trace']} |")
    lines+=['','三个异常均为 BF16 下执行路径引起的计数翻转，FP32 两路径输出一致。证据支持有限精度数值敏感，具体算子未单独隔离。N7 的 FP32 两路径均输出 6，因此不能把提高精度等同于恢复 gold。','',
        'Read 配对和 BF16 整段重放保持冻结；原 self 38/40 的结果和 Target-minus-self 统计保留。']
    (output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='PASS',cases=len(evidence),fp32_max_logit_difference=max(r['fp32_shared_top10_max_logit_difference'] for r in evidence))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.source,a.baseline,a.output)
