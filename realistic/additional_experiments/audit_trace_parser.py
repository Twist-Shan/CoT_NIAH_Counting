"""Run the offline parser on immutable captures and emit JSONL/CSV/HTML audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys
import time
from collections import defaultdict

from trace_parser import CONTRACT, parse_trace


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    with args.cases.open(encoding='utf-8') as f:
        cases = {r['case_id']: r for r in map(json.loads, f)}
    paths = sorted(args.captures.glob('*/captures/native_thinking/*/generation.json'))
    if not paths:
        raise ValueError('No native captures found')
    outputs, summary, source_hashes = [], defaultdict(lambda: defaultdict(int)), {}
    page = ['<!doctype html><meta charset="utf-8"><title>Trace parser audit</title>',
            '<style>body{max-width:1200px;margin:2rem auto;font:16px system-ui}pre{white-space:pre-wrap}td,th{border:1px solid #ccc;padding:6px}table{border-collapse:collapse}summary{cursor:pointer}</style>',
            '<h1>Record parser audit</h1><p>Character offsets only. Semantic rules are conservative; unknown is not false. Full trace is available below each case.</p>']
    for path in paths:
        generated = json.loads(path.read_text(encoding='utf-8'))
        case = cases[path.parent.name]
        model = path.parents[3].name
        raw = generated['completion_text_raw']
        parsed = parse_trace(case, raw)
        parsed.update(model=model, case_id=case['case_id'], task=case['task'])
        outputs.append(parsed)
        source_hashes[str(path.resolve())] = digest(path)
        stats = summary[(model, case['task'])]
        stats['cases'] += 1
        stats['structured_cases'] += parsed['status'] == 'structured'
        for item in parsed['items']:
            for name in ('block', 'identity_end', 'record_end', 'decision_end', 'count_end'):
                span = item[name]
                if span and raw[span['start']:span['end']] != span['text']:
                    raise AssertionError('Invalid span')
            stats['items'] += 1
            stats['record_sites'] += item['record_end'] is not None
            stats['decision_sites'] += item['decision_end'] is not None
            stats['count_sites'] += item['count_end'] is not None
        page += [f'<h2>{html.escape(model)} / {html.escape(case["case_id"])}</h2>',
                 '<p>' + html.escape(json.dumps({k: parsed.get(k) for k in
                     ('status', 'unavailable_reason', 'coverage', 'phase_description')}, ensure_ascii=False)) + '</p>',
                 '<table><tr><th>Step / source</th><th>Record block</th><th>Observed decision</th><th>Explicit count</th></tr>']
        for item in parsed['items']:
            decision = item['decision_end']
            count = item['count_end']
            page.append(f'<tr><td>{item["step"]} / {item["source_ordinal"]}</td><td><pre>{html.escape(item["block"]["text"])}</pre></td>'
                        f'<td>{html.escape(str(decision))}</td><td>{html.escape(str(count))}</td></tr>')
        page += ['</table><details><summary>Full original completion</summary><pre>' + html.escape(raw) + '</pre></details>']
    with (args.output / 'parsed.jsonl').open('w', encoding='utf-8') as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    fields = ['model', 'task', 'cases', 'structured_cases', 'items', 'record_sites', 'decision_sites', 'count_sites']
    with (args.output / 'coverage.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (model, task), stats in sorted(summary.items()):
            writer.writerow({'model': model, 'task': task, **{k: stats[k] for k in fields[2:]}})
    (args.output / 'audit.html').write_text('\n'.join(page), encoding='utf-8')
    manifest = {'contract': CONTRACT, 'command': sys.argv, 'cases_sha256': digest(args.cases),
                'source_hashes': source_hashes, 'parser_hashes': {name: digest(Path(__file__).with_name(name))
                    for name in ('trace_parser.py', 'list_early_stop.py', 'audit_trace_parser.py')},
                'elapsed_seconds': time.perf_counter() - started, 'cases': len(outputs),
                'span_checks': 'PASS', 'token_alignment_checked': False,
                'output_hashes': {name: digest(args.output / name) for name in ('parsed.jsonl', 'coverage.csv', 'audit.html')}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'cases': len(outputs), 'elapsed_seconds': manifest['elapsed_seconds'],
                      'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
