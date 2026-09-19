"""Offline coverage/hash audit and seed-equal descriptive retrieval summaries."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_kth_retrieval import estimate

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    root = args.root
    with (root/'frozen/cases.jsonl').open(encoding='utf-8') as f:
        cases = {c['case_id']: c for c in map(json.loads, f)}
    assert len(cases) == 300
    assert Counter(c['level'] for c in cases.values()) == Counter({k:30 for k in range(1,11)})
    confirmation = {k for k,c in cases.items() if c['split']=='confirmation'}
    metrics = defaultdict(lambda: defaultdict(list))
    by_k = defaultdict(list)
    counts = Counter()
    missing = Counter()
    for model in ('Qwen3-8B','Gemma4-E4B'):
        base = root/'downloaded/full'/model
        contract = read(base/'contract.json')
        assert contract['frozen_manifest_sha256'] == sha(root/'frozen/manifest.json')
        assert read(base/'complete.json')['status']=='PASS'
        banks = read(base/'frozen_selection.json')
        for mode in ('nonthinking','native_thinking'):
            cap = base/'captures'/mode
            assert {p.name for p in cap.iterdir() if p.is_dir()} == set(cases)
            causal = base/'causal'/mode
            assert {p.stem for p in causal.glob('*.json')} == confirmation
            for assay in ('broad','targeted'):
                bank = banks['broad'][mode]['heads'] if assay=='broad' else banks['targeted']
                controls = banks['broad'][mode]['random_heads'] if assay=='broad' else banks['targeted_random']
                assert len(controls)==3
                for ctrl in controls:
                    assert len(ctrl)==len(bank) and len(set(map(tuple,ctrl)))==len(ctrl)
                    assert not set(map(tuple,bank)) & set(map(tuple,ctrl))
                    if assay=='broad' or model=='Gemma4-E4B':
                        assert Counter(h[0] for h in bank)==Counter(h[0] for h in ctrl)
            for cid,c in cases.items():
                dest=cap/cid
                done=read(dest/'complete.json')
                assert done['status']=='PASS'
                for name,h in done['files'].items():
                    assert sha(dest/name)==h, str(dest/name)
                    counts['capture_file_hashes']+=1
                g=read(dest/'generation.json'); r=read(dest/'retrieval.json'); p=read(dest/'prompt.json')
                for field in ('user_text','rendered_prompt'):
                    assert hashlib.sha256(p[field].encode()).hexdigest()==p[field+'_sha256']
                counts['captures']+=1
                counts[f'{model}/{mode}/correct']+=g['correct']
                counts[f'{model}/{mode}/truncated']+=g['generation_truncated']
                counts[f'{model}/{mode}/parse_failed']+=not g['parse_ok']
                counts[f'{model}/{mode}/ids_reproduced']+=r['original_ids_reproduced']
                original=p['input_ids']+g['generated_token_ids']
                for key in ('final_prefix_ids','target_prefix_ids'):
                    if r[key] is not None:
                        assert r[key]==original[:len(r[key])]
                    else:
                        missing[f'{model}/{mode}/{key}']+=1
                counts[f'{model}/{mode}/trace_available']+=r['trace_record_sites']>0
                by_k[(model,mode,c['split'],c['level'])].append(int(g['correct']))
                for assay,field in [('broad','final_attention'),('targeted','target_attention')]:
                    rows={(x['layer'],x['head']):x for x in r[field]}
                    if not rows: continue
                    bank=banks['broad'][mode]['heads'] if assay=='broad' else banks['targeted']
                    keys=['target_mass','target_relative_mass'] if assay=='targeted' else ['prompt_broad','trace_broad']
                    for key in keys:
                        if key=='trace_broad' and r['trace_record_sites']==0: continue
                        vals=[rows[tuple(h)][key]['score'] if key.endswith('broad') else rows[tuple(h)][key] for h in bank]
                        metrics[(model,mode,c['split'],assay,key)][c['seed']].append(sum(vals)/len(vals))
                if cid not in confirmation: continue
                a=read(causal/(cid+'.json'))
                assert (a['case_id'],a['seed'],a['k'],a['mode'])==(cid,c['seed'],c['level'],mode)
                counts['causal_cases']+=1
                for assay,result in a['assays'].items():
                    if result['status']!='PASS':
                        missing[f'{model}/{mode}/{assay}/causal']+=1
                        continue
                    assert [x['name'] for x in result['arms']]==['clean','selected','random_0','random_1','random_2']
                    bank=banks['broad'][mode]['heads'] if assay=='broad' else banks['targeted']
                    controls=banks['broad'][mode]['random_heads'] if assay=='broad' else banks['targeted_random']
                    assert [x['heads'] for x in result['arms']]==[[],bank]+controls
                    counts['causal_arms']+=5
                    for arm in result['arms']:
                        counts[f'{model}/{mode}/{assay}/short_truncated']+=arm['generation']['generation_truncated']
                    if assay=='broad' or mode=='nonthinking':
                        counts[f'{model}/{mode}/{assay}/clean_behavior_disagreement']+=result['arms'][0]['score']['correct']!=g['correct']
    assert counts['captures']==1200 and counts['causal_cases']==400
    output=root/'audit'; output.mkdir(exist_ok=True)
    rows=[dict(zip(('model','mode','split','assay','metric'),key),**estimate([sum(v)/len(v) for v in seeds.values()]),cases=sum(map(len,seeds.values()))) for key,seeds in sorted(metrics.items())]
    for name,data in [('retrieval_summary.csv',rows),('accuracy_by_k.csv',[dict(zip(('model','mode','split','k'),k),correct=sum(v),cases=len(v),accuracy=sum(v)/len(v)) for k,v in sorted(by_k.items())])]:
        with (output/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    report={'status':'PASS','counts':dict(counts),'missing':dict(missing),'elapsed_seconds':time.perf_counter()-start,'script_sha256':sha(Path(__file__))}
    (output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
