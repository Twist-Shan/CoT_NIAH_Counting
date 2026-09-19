"""Continue an interrupted sweep using its already completed cache validation."""
import argparse
import json
from pathlib import Path
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
    execution=json.loads((args.output/'cache_execution.json').read_text())
    assert all(original.audit.digest(Path(p))==h for p,h in execution['code_sha256'].items())
    checks=json.loads((args.output/'cached_equivalence.json').read_text())
    assert set(checks)=={'thinking','nonthinking'}
    assert all(c['status']=='PASS' and c['exact_token_match'] for c in checks.values())
    audit={'resume_reason':'SSH connection reset after saving completed conditions',
           'cached_equivalence_sha256':original.audit.digest(args.output/'cached_equivalence.json'),
           'cached_code_sha256':original.audit.digest(Path(cached.__file__)),
           'continuation_code_sha256':original.audit.digest(Path(__file__)),
           'full_prefix_and_cache_code_unchanged':True}
    original.audit.write_json(args.output/'connection_resume.json',audit)
    full_generate=original.generate
    def dispatch(*args,**kwargs):
        if kwargs.get('verify'):
            return full_generate(*args,**kwargs)
        return cached.generate(*args,**kwargs)
    original.generate=dispatch
    original.run(args,protocol)


if __name__=='__main__':main()
