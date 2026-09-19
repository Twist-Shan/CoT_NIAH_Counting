#!/usr/bin/env python3
"""Freeze current-bank dose inputs from audited historical ten-row shards."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    path = path.resolve()
    if __import__('os').name == 'nt':
        path = Path('\\\\?\\' + str(path))
    return [json.loads(s) for s in path.read_text(encoding='utf-8').split('\n') if s.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for model, tag, ks in [('Qwen3-8B', 'qwen_shared_k128', [1,2,4,8,16,32,64,128]),
                           ('Gemma4-E4B', 'gemma_shared_k6', [1,2,4,6])]:
        folder = out / model
        folder.mkdir()
        old = ROOT / 'work/v5_native_sample_aligned_20260829/runs' / model / 'targeted_retrieval/confirmation'
        rows = [r for f in sorted((old/'shards').glob('*.jsonl')) for r in read(f)]
        assert len(rows) == 50
        groups = defaultdict(list)
        for r in rows:
            groups[int(r['seed'])].append(r)
        assert sorted(groups) == list(range(1254,1264))
        selection = json.loads((ROOT / f'configs/realistic_niah_v5_{tag}_targeted_selection_frozen.json').read_text())
        selected = selection['development_selection']['primary_bank_heads']
        assert len(selected) == ks[-1]
        plans = []
        for seed, cells in sorted(groups.items()):
            clean = next(r for r in cells if r['condition'] == 'clean')
            original_selected = next(r for r in cells if r['condition'] == 'selected_bank')
            assert {tuple(h) for h in original_selected['heads']} == {tuple(h) for h in selected}
            assert len({r['request_id'] for r in cells}) == 1
            assert all(r['intervention_anchor_equivalence_ids'] == clean['intervention_anchor_equivalence_ids'] for r in cells)
            random_rows = sorted([r for r in cells if r['condition']=='layer_matched_random'],key=lambda r:int(r['repeat']))
            assert len(random_rows) == 3
            baseline = {'seed':seed,'request_id':clean['request_id'],'gold_count':clean['gold_count'],
                        'anchors':clean['intervention_anchor_equivalence_ids']}
            plans.append({**baseline,'k':0,'condition':'clean','repeat':0,'heads':[]})
            for k in ks:
                heads = selected[:k]
                profile = Counter(h[0] for h in heads)
                plans.append({**baseline,'k':k,'condition':'selected_bank','repeat':0,'heads':heads})
                for old_random in random_rows:
                    by_layer = defaultdict(list)
                    for layer, head in old_random['heads']:
                        by_layer[layer].append(head)
                    random_heads = []
                    for layer, count in sorted(profile.items()):
                        pool = sorted(by_layer[layer])
                        key = f'20260912/{model}/{old_random["repeat"]}/{layer}'
                        random.Random(int(hashlib.sha256(key.encode()).hexdigest(),16)).shuffle(pool)
                        random_heads.extend([[layer,h] for h in pool[:count]])
                    assert Counter(h[0] for h in random_heads) == profile
                    assert not {tuple(h) for h in random_heads} & {tuple(h) for h in selected}
                    if k == ks[-1]:
                        assert {tuple(h) for h in random_heads} == {tuple(h) for h in old_random['heads']}
                    plans.append({**baseline,'k':k,'condition':'layer_matched_random','repeat':old_random['repeat'],'heads':random_heads})
        ids={r['request_id'] for r in plans}
        generations=[r for r in read(ROOT/f'work/v5_trace_parser_v2/{model}_generations_reparsed.jsonl') if r['request_id'] in ids]
        assert len(generations)==10
        assert [groups[s][0]['gold_count'] for s in sorted(groups)]==[10,10,8,4,10,7,10,7,6,7]
        (folder/'generations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in generations),encoding='utf-8')
        (folder/'historical.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        routed=read(ROOT/f'work/v5_native_sample_aligned_20260829/shared_routed_transition_panel/{model}_anchor_panel.jsonl')
        routed=[r for r in routed if r['request_id'] in ids]
        assert len(routed)==10
        (folder/'routed.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in routed),encoding='utf-8')
        write(folder/'plan.json',{'model':model,'ks':ks,'max_new_tokens':512,'decode_head_ablation_steps':-1,
                                'random_rule':'Per layer deterministic shuffle of each historical random bank, then prefix matching the selected-bank layer profile. Full K exactly preserves each historical membership.',
                                'clean_reuse':'One identical clean generation per seed, shared by all K.',
                                'status':'FROZEN_BEFORE_NEW_OUTCOMES','trials':plans})
    hashes={str(p.relative_to(out)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    write(out/'manifest.json',hashes)
    with tarfile.open(out.with_suffix('.tar.gz'),'w:gz') as t:
        for p in out.rglob('*'):
            if p.is_file():t.add(p,arcname='dose_inputs/'+p.relative_to(out).as_posix())
    print(json.dumps({'files':len(hashes),'archive_bytes':out.with_suffix('.tar.gz').stat().st_size}))


if __name__=='__main__':
    main()
