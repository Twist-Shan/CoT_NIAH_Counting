#!/usr/bin/env python3
"""Audit frozen commands, completed grids, and actual free-generation patches."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import math


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').split('\n') if s.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    commands = sorted(root.glob('backend_replay/*/*/command.json'))
    commands += sorted(root.glob('generation/runs/*/*/command.json'))
    assert len(commands) == 24, len(commands)
    inputs, artifacts, jobs, coverage = {}, {}, [], []
    revisions = {}
    for command in commands:
        folder = command.parent
        job = json.loads(command.read_text())
        runtime = json.loads((folder/'runtime.json').read_text())
        status = json.loads((folder/'process_status.json').read_text())
        assert status['status'] == 'COMPLETE', folder
        assert runtime['job_sha256'] == sha(command)
        code = root / ('generation/code' if 'generation/runs' in folder.as_posix() else 'code')
        for relative, expected in job['code_sha256'].items():
            assert sha(code/relative) == expected, (folder, relative)
        for path, expected in job['input_sha256'].items():
            source = Path(path)
            assert sha(source) == expected, source
            inputs[path] = expected
        events = jsonl(folder/'backend_events.jsonl')
        loads = [e for e in events if e['event'] == 'model_loaded']
        assert len(loads) == 1 and not loads[0]['training']
        assert loads[0]['dtype'] == 'torch.bfloat16'
        assert loads[0]['backend']['text'] == 'sdpa'
        revision = loads[0]['revision']
        assert revision
        revisions.setdefault(loads[0]['model_class'], set()).add(revision)
        if job['backend_variant'] == 'fixed':
            observed = [e for e in events if e['event'] != 'model_loaded']
            assert all(e['before'] == e['after'] and e['after']['text'] == 'sdpa' for e in observed)
        task = folder.name
        all_result_files = sorted((folder/'results').rglob('*.jsonl'))
        files = (sorted((folder/'results/shards').glob('*.jsonl')) if task.startswith('blank_')
                 else [folder/'results/trials.jsonl'] if task.startswith('progress_')
                 else sorted((folder/'results').glob('*.jsonl')))
        rows = [r for f in files for r in jsonl(f)]
        expected = (500 if task.startswith('blank_') else 30 if task.startswith('progress_')
                    else 2 if task == 'dose_smoke' else 8 if task == 'recovery_smoke'
                    else 80 if task == 'recovery_full' else 330 if folder.parent.name == 'Qwen3-8B' else 170)
        assert len(rows) == expected, (folder, len(rows), expected)
        if task.startswith(('dose_', 'recovery_')):
            assert {r['status'] for r in rows} == {'ok'}
            assert all(r['generated_token_count'] == len(r['generated_token_ids']) <= 512 for r in rows)
        if task.startswith('recovery_'):
            for row in rows:
                assert row['forced_trace_tokens_after_query'] is False and row['oracle_clean_state_donor'] is True
                carrier = row['carrier_positions']
                assert len(set(carrier)) == len(carrier) == len(row['matched_positions'])
                assert not set(carrier) & set(row['matched_positions'])
                audit = row['carrier_audit']
                if audit:
                    assert audit['positions'] == carrier
                    assert set(map(int, audit['applications'])) == set(row['patch_layers'])
                    missing = []
                    for layer, applied in audit['applications'].items():
                        visited = audit['visited_through_position'][layer]
                        expected_applied = [pos for pos in carrier if pos <= visited]
                        assert applied == expected_applied, (row['seed'], row['condition'], layer)
                        expected_missing = [pos for pos in carrier if pos > visited]
                        assert audit['missing_positions'][layer] == expected_missing
                        rms = audit['realized_rms'][layer]
                        assert len(rms) == len(applied) and all(math.isfinite(v) and v >= 0 for v in rms)
                        missing += expected_missing
                    coverage.append(dict(model=folder.parent.name, task=task, seed=row['seed'],
                        condition=row['condition'], scheduled_positions=len(carrier),
                        applied_per_layer=len(next(iter(audit['applications'].values()))),
                        missing_layer_positions=len(missing),
                        generated_carrier_tokens_matching_clean=sum(c['same_token'] for c in row['carrier_generated_token_comparison']),
                        generated_token_count=row['generated_token_count'], truncated=row['generation_truncated']))
        jobs.append(dict(path=str(folder.relative_to(root)), records=len(rows),
                         generation_truncated=sum(bool(r.get('generation_truncated')) for r in rows),
                         seconds=status['seconds'], code_files=len(job['code_sha256']),
                         runtime=runtime['packages'], revision=revision))
        for f in [command, folder/'runtime.json', folder/'process_status.json', folder/'backend_events.jsonl', *all_result_files]:
            artifacts[str(f.relative_to(root))] = sha(f)
    assert all(len(values) == 1 for values in revisions.values())
    assert sum(j['records'] for j in jobs) == 3040
    report = dict(status='PASS_COMPLETE_ARTIFACT_AUDIT', jobs=jobs, total_records=3040,
                  input_sha256=inputs, artifact_sha256=artifacts, carrier_coverage=coverage,
                  analysis_source_sha256=sha(Path(__file__)),
                  scope='Rows include paired backend runs and smoke checks; they are not independent samples.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status':report['status'], 'jobs':len(jobs), 'records':3040,
                      'audited_clamped_trials':len(coverage), 'raw_files':len(artifacts)}))


if __name__ == '__main__':
    main()
