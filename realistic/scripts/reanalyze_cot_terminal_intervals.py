#!/usr/bin/env python3
"""Recompute terminal intervals with an explicit seed, preserving old exports."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from analyze_realistic_niah_v5_terminal_relay_common_support import KEYS, OUTCOMES, analyze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    started=time.monotonic()
    pairs_path=args.source/'common_support_pairs.csv'
    old_summary=args.source/'summary.json'
    pairs=pd.read_csv(pairs_path)
    assert len(pairs)==86 and pairs['seed'].nunique()==10 and not pairs.duplicated(KEYS).any()
    frames=[]
    for suffix in ['qwen','gemma']:
        columns={f'{o}__{suffix}':o for o in OUTCOMES}
        frames.append(pairs[KEYS+list(columns)].rename(columns=columns))
    merged,cells,summary=analyze(*frames,phase='confirmation',bootstrap_samples=10000,random_seed=20260912)
    previous=json.loads(old_summary.read_text())
    differences=[]
    for model,value in summary['models'].items():
        for metric,effect in value['effects'].items():
            old=previous['models'][model]['effects'][metric]
            assert abs(effect['estimate']-old['estimate'])<1e-12
            differences.append({'model':model,'metric':metric,'point':effect['estimate'],
                                'archived_ci':[old['ci_low'],old['ci_high']],
                                'recomputed_ci':[effect['ci_low'],effect['ci_high']]})
    summary.update(bootstrap_draws=10000,bootstrap_random_seed=20260912,
                   bootstrap_unit='Equal-weight seed means after averaging common pairs within each seed.',
                   ratio_estimand='Ratio of seed-mean mediated damage to seed-mean natural damage, recomputed inside each resample.',
                   status='PASS_RECOMPUTED_INTERVALS',gpu_rerun=False)
    files=[pairs_path,old_summary,Path(__file__),Path(__file__).with_name('analyze_realistic_niah_v5_terminal_relay_common_support.py')]
    audit={'source_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
           'command':sys.argv,'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
           'seconds':time.monotonic()-started,'all_point_estimates_reproduced':True,'interval_comparison':differences}
    args.output.mkdir(parents=True,exist_ok=False)
    merged.to_csv(args.output/'common_support_pairs.csv',index=False)
    cells.to_csv(args.output/'common_support_cell_counts.csv',index=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'status':summary['status'],'suffix_fractions':{m:v['effects']['post_terminal_suffix_explained_fraction'] for m,v in summary['models'].items()}},indent=2))


if __name__=='__main__':main()
