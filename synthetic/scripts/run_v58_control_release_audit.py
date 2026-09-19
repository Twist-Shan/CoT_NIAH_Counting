"""Release one head at a time from the frozen Top-4 control ablation.

Other three heads remain ablated throughout generation. A released head uses
its normal computation on the current trajectory; no clean activation patch.
"""
from pathlib import Path
import argparse
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / 'src', ROOT / 'scripts'):
    sys.path.insert(0, str(p))
import pandas as pd
import torch
import audit_v58_retrieval_generation as audit
from run_v58_control_stage_audit import generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    align = args.run_dir / 'analysis/v58_alignment_supplement_20260905'
    source = args.run_dir / 'analysis/behavior_confirmation_v58/examples.jsonl'
    registry = pd.read_csv(align / 'input_registry.csv')
    support = registry.loc[registry.split.eq('confirmation')]
    assert support.groupby('count').size().to_dict() == {k: 10 for k in range(1, 11)}
    lookup = {e.prompt_sha256: e for e in [audit.base.example_from_dict(json.loads(s)) for s in source.read_text().splitlines() if s.strip()]}
    examples = [lookup[k] for k in support.key]
    plans = {m: json.loads((align / m / 'frozen_sites.json').read_text())['controls']['4'][0] for m in ['nonthinking', 'thinking']}
    audit.write_json(args.output / 'protocol.json', {'created_unix': time.time(),
        'script_sha256': audit.digest(Path(__file__)),
        'generator_sha256': audit.digest(Path(__file__).with_name('run_v58_control_stage_audit.py')),
        'source_sha256': audit.digest(source), 'registry_sha256': audit.digest(align / 'input_registry.csv'),
        'top4_control_heads': plans, 'keys': [e.prompt_sha256 for e in examples],
        'scope': 'same sustained onset through EOS or 64 tokens',
        'intervention': 'For each of four heads, release it from the frozen Top-4 control bank; the remaining three heads stay ablated.',
        'selection': 'all four members tested; no selection by rescue magnitude',
        'interpretation': 'Conditional contribution of one head in the presence of three ablated control heads; effects need not be additive.',
    })
    frames, checks = [], []
    for mode in ['nonthinking', 'thinking']:
        cp = audit.checkpoint_path(args.run_dir, mode, 10000)
        assert audit.digest(cp) == json.loads((align / mode / 'manifest.json').read_text())['checkpoint_sha256']
        cfg, vocab, _, _, model = audit.base.load_v20_checkpoint_model(args.run_dir, 'rope', mode, step=10000, device='cuda')
        prior = pd.read_csv(args.run_dir / 'analysis/v58_retrieval_sustained_cap64_20260908' / mode / 'generation_trials.csv').fillna({'heads': ''})
        clean = prior.loc[prior.arm.eq('clean')].set_index('prompt_sha256').to_dict('index')
        bank = [tuple(h) for h in plans[mode]]
        for released in bank:
            masked = [h for h in bank if h != released]
            detail, check = generate(model, cfg, vocab, examples, masked, mode, 'all', 32, 64, clean=clean, verify_hook=True)
            detail['released_head'] = audit.canonical([released])
            detail['ablated_heads'] = audit.canonical(masked)
            frames.append(detail)
            checks.append(dict(mode=mode, released_head=audit.canonical([released]), **check))
            print(mode, 'released', audit.canonical([released]), 'answer', detail.ar_accuracy.mean(),
                  'trace', detail.trace_exact.mean(), 'EOS', detail.eos_reached.mean(), flush=True)
        del model
        torch.cuda.empty_cache()
    full = pd.concat(frames, ignore_index=True)
    assert len(full) == 800 and not full.duplicated(['mode', 'released_head', 'key']).any()
    full.to_csv(args.output / 'generation_trials.csv', index=False)
    metrics = ['ar_accuracy', 'ar_answered', 'trace_exact', 'trace_marker_count_accuracy', 'trace_format_valid',
               'eos_reached', 'answer_and_eos_correct', 'duplicate_ans', 'after_duplicate_accuracy']
    full.groupby(['mode', 'released_head', 'ablated_heads'])[metrics].mean().reset_index().to_csv(args.output / 'generation_summary.csv', index=False)
    audit.write_json(args.output / 'manifest.json', {'status': 'complete', 'rows': len(full), 'hook_checks': checks,
                     'files': {str(s.relative_to(args.output)): audit.digest(s) for s in args.output.rglob('*') if s.is_file()}})
    print('CONTROL RELEASE AUDIT COMPLETE', flush=True)


if __name__ == '__main__':
    main()
