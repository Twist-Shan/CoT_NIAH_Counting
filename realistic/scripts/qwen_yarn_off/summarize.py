"""Read checkpoint status, or validate and merge the finished experiment."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import time
from full_run import expected_keys, json_sha, save, sha, summary, utc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--final', action='store_true')
    p.add_argument('--final-if-ready', action='store_true')
    args = p.parse_args()
    root = args.root
    started = time.perf_counter()
    config = json.loads((root/'config.json').read_text())
    rows, workers = {}, []
    for worker in range(config['workers']):
        wd = root/f'worker-{worker}'
        mp = wd/'run_manifest.json'
        if mp.exists():
            manifest = json.loads(mp.read_text())
            workers.append({k: manifest.get(k) for k in ['worker','status','completed_requests','expected_requests','truncations','parse_failures','generation_seconds','last_batch_seconds','last_updated_at_utc','last_batch_keys']})
        for path in sorted((wd/'parts').glob('*.json')):
            for row in json.loads(path.read_text()):
                key = row['request_key']
                assert key not in rows and row['config_sha256'] == json_sha(config)
                assert not row['evaluation']['truncated'] or not row['evaluation']['exact_count']
                rows[key] = row
    wanted = expected_keys(config)
    assert set(rows) <= wanted
    grouped = defaultdict(list)
    runtime = defaultdict(dict)
    for row in rows.values():
        grouped[(row['target_passage_tokens'],row['num_needles'],row['prompt_mode'])].append(row)
        runtime[(row['target_passage_tokens'],row['prompt_mode'])][row['batch_id']] = row
    cells = [dict(L=L,N=N,mode=mode,**summary(group)) for (L,N,mode),group in sorted(grouped.items())]
    timings=[]
    for (L, mode),batches in sorted(runtime.items()):
        seconds=sum(r['batch_wall_time_seconds'] for r in batches.values())
        count=sum(r['batch_size'] for r in batches.values())
        timings.append(dict(L=L,mode=mode,batches=len(batches),requests=count,batch_seconds=seconds,
                            seconds_per_request=seconds/count))
    report=dict(updated_at_utc=utc(),expected_requests=len(wanted),**summary(rows.values()),workers=workers,
                timing_by_length_mode=timings,validation_elapsed_seconds=time.perf_counter()-started)
    save(root/'status.json',report)
    save(root/'cell_summary.json',cells)
    print(json.dumps(report),flush=True)
    ready = set(rows)==wanted and len(workers)==config['workers'] and all(w['status']=='complete' for w in workers)
    if args.final or (args.final_if_ready and ready):
        assert set(rows)==wanted
        assert all(w['status']=='complete' for w in workers) and len(workers)==config['workers']
        assert len(cells)==17*14*2 and all(c['completed_requests']==30 for c in cells)
        path=root/'requests.jsonl'
        tmp=path.with_suffix('.jsonl.tmp')
        with tmp.open('w',encoding='utf-8') as f:
            for key in sorted(rows):f.write(json.dumps(rows[key],ensure_ascii=False,sort_keys=True)+'\n')
        tmp.replace(path)
        with (root/'cell_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(cells[0]));writer.writeheader();writer.writerows(cells)
        save(root/'completion.json',dict(passed=True,completed_at_utc=utc(),requests=len(rows),cells=len(cells),
             requests_sha256=sha(path.read_bytes()),all_cells_have_30_requests=True,all_truncations_scored_wrong=True))


if __name__=='__main__':main()
