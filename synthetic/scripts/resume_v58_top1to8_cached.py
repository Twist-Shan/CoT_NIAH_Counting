"""Resume the frozen sweep after verifying cached vs full-prefix generation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import pandas as pd
import run_v58_top1to8_aligned as original
import v58_cached_decode as cached


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=100)
    p.add_argument('--budget',type=int,default=64)
    args=p.parse_args()
    protocol=json.loads((args.output/'protocol.json').read_text())
    assert args.batch_size==protocol['batch_size'] and args.budget==protocol['budget']
    audit=original.audit
    execution={'created_unix':time.time(),'estimator_changed':False,
               'description':'KV cache stores the intervened states. Full-vocabulary greedy and frozen control/input policies unchanged.',
               'code_sha256':{str(Path(p).resolve()):audit.digest(p) for p in [__file__,cached.__file__]},
               'required_validation':'Exact generated-token agreement with every saved full-prefix condition, plus fresh clean/selected/control cases on ten count-balanced inputs for every scope.'}
    target=args.output/'cache_execution.json'
    if target.exists():
        previous=json.loads(target.read_text())
        assert previous['code_sha256']==execution['code_sha256']
    else:
        audit.write_json(target,execution)
    full_generate=original.generate
    validated=set()
    checks={}

    def dispatch(model,cfg,vocab,records,examples,heads,scope,batch_size,budget,verify=False):
        mode=records[0]['mode']
        if mode not in validated:
            small=[next(r for r in records if r['count']==n) for n in range(1,11)]
            cases=[[],protocol['plans'][mode]['1']['selected'],protocol['plans'][mode]['4']['selected'],
                   protocol['plans'][mode]['8']['selected'],protocol['plans'][mode]['4']['controls'][0]['heads'],
                   protocol['plans'][mode]['8']['controls'][0]['heads']]
            counts={'fresh_condition_checks':0,'fresh_input_trajectories':0,
                    'saved_condition_checks':0,'saved_input_trajectories':0}
            for s in protocol['scopes'][mode]:
                for hs in cases:
                    a,_=full_generate(model,cfg,vocab,small,examples,hs,s,len(small),budget)
                    b,_=cached.generate(model,cfg,vocab,small,examples,hs,s,len(small),budget)
                    if a.generated_tokens.tolist()!=b.generated_tokens.tolist():
                        a.to_csv(args.output/f'cache_mismatch_{mode}_full.csv',index=False)
                        b.to_csv(args.output/f'cache_mismatch_{mode}_cached.csv',index=False)
                        raise AssertionError(f'Cached/full trajectory mismatch: {mode} {s} {hs}')
                    counts['fresh_condition_checks']+=1
                    counts['fresh_input_trajectories']+=len(small)
            paths=sorted((args.output/mode/'arms').glob('*.csv'))
            for i,path in enumerate(paths):
                a=pd.read_csv(path)
                hs=[(int(h.split('H')[0][1:]),int(h.split('H')[1])) for h in str(a.heads.iloc[0]).split(';')]
                b,_=cached.generate(model,cfg,vocab,records,examples,hs,a.scope.iloc[0],batch_size,budget)
                assert a.set_index('key').generated_tokens.eq(b.set_index('key').generated_tokens.reindex(a.key)).all(), ('Saved/full mismatch',path)
                counts['saved_condition_checks']+=1
                counts['saved_input_trajectories']+=len(a)
                if (i+1)%20==0:
                    print('CACHE VERIFY',mode,i+1,'/',len(paths),flush=True)
            validated.add(mode)
            checks[mode]={'status':'PASS','exact_token_match':True,**counts}
            audit.write_json(args.output/'cached_equivalence.json',checks)
            print('CACHE VERIFIED',mode,counts,flush=True)
        if verify:
            return full_generate(model,cfg,vocab,records,examples,heads,scope,batch_size,budget,verify=True)
        return cached.generate(model,cfg,vocab,records,examples,heads,scope,batch_size,budget)

    original.generate=dispatch
    original.run(args,protocol)


if __name__=='__main__':
    main()
