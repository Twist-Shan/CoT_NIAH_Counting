"""Measure ten registered Gemma queries with the frozen model and saved tokens.

Only one query's attention is materialized at a time, using the project's
existing KV-cache capture. No generation, ranking, or sample selection occurs.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device-map', default='auto')
    parser.add_argument('--threads', type=int, help='Explicit PyTorch CPU thread count.')
    parser.add_argument('--markers', nargs='+', type=int,
                        help='Optional subset for a timing/numerical pilot; incomplete subsets are flagged.')
    args = parser.parse_args()
    start = time.perf_counter()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    assert data['schema'] == 'cot_marker_capture_input_v1'
    assert data['existing_eight_queries_match'] and data['gold_count'] == 10
    assert [q['marker'] for q in data['queries']] == list(range(10))
    requested_markers = list(range(10)) if args.markers is None else args.markers
    if not requested_markers or len(set(requested_markers)) != len(requested_markers):
        raise ValueError('Markers must be nonempty and unique.')
    if any(marker not in range(10) for marker in requested_markers):
        raise ValueError('Markers must lie in 0..9.')
    queries = [q for q in data['queries'] if q['marker'] in requested_markers]
    if args.output.exists():
        raise FileExistsError(f'Preserve the existing capture: {args.output}')
    sys.path.insert(0, str(args.repo_root / 'src'))
    import torch
    from realistic_niah_v4.modeling import load_registered_model, position_attention_outputs
    from realistic_niah_v4.spec import resolve_model_spec
    spec = resolve_model_spec(data['model_label'])
    assert spec.revision == data['model_revision']
    if args.threads is not None:
        if args.threads < 1:
            raise ValueError('The thread count must be positive.')
        torch.set_num_threads(args.threads)
    if args.device_map != 'cpu':
        assert torch.cuda.is_available(), 'Use a usable CUDA device, or explicitly request --device-map cpu.'
    print(json.dumps({'stage': 'loading_model', 'device_map': args.device_map,
                      'markers': requested_markers, 'torch_threads': torch.get_num_threads()}), flush=True)
    load_start = time.perf_counter()
    model, tokenizer, adapter = load_registered_model(spec, cache_dir=args.cache_dir,
        device_map=args.device_map, torch_dtype='bfloat16', attention_backend='sdpa')
    model.eval()
    model_device_types = sorted({p.device.type for p in model.parameters()})
    if args.device_map == 'cpu':
        assert model_device_types == ['cpu'], model_device_types
    load_seconds = time.perf_counter() - load_start
    print(json.dumps({'stage': 'model_loaded', 'seconds': round(load_seconds, 2),
                      'device_types': model_device_types}), flush=True)
    events, differences, backend_history = [], [], []
    reference = {e['from_occurrence']: e for e in data['reference_events']}
    for query in queries:
        query_start = time.perf_counter()
        configs = [model.config, model.config.get_text_config()]
        before = [config._attn_implementation for config in configs]
        assert before == ['sdpa', 'sdpa'], ('prefix_backend', query['marker'], before)
        prefix_end = query['query_output_token_index'] + 1
        ids = data['input_ids'] + data['output_token_ids'][:prefix_end]
        mask = data['attention_mask'] + [1] * prefix_end
        position = query['query_full_sequence_token']
        assert len(ids) == len(mask) == position + 1
        encoding = SimpleNamespace(input_ids=ids, attention_mask=mask, query_position=position)
        with torch.inference_mode():
            rows, starts, _ = position_attention_outputs(model, adapter, encoding, position)
        after = [config._attn_implementation for config in configs]
        assert after == before, ('backend_not_restored', query['marker'], before, after)
        backend_history.append({'marker': query['marker'], 'before': before, 'after': after})
        attention = rows[data['layer']][data['head']].float()
        key_start = int(starts[data['layer']])
        key_end = key_start + attention.numel()
        records = []
        for span in sorted(data['prompt_record_spans'], key=lambda p: p['slot_index']):
            lo, hi = max(span['start'], key_start), min(span['end'], key_end)
            mass = float(attention[lo-key_start:hi-key_start].sum()) if hi > lo else 0.0
            records.append({'source_index': span['slot_index'], 'city': span['city'],
                'token_start': span['start'], 'token_end': span['end'],
                'visible_token_count': max(0, hi-lo), 'mass': mass,
                'is_target': span['slot_index'] == query['needle']})
        total = float(attention.sum())
        needle_total = sum(r['mass'] for r in records)
        assert abs(total-1) < .01 and 0 < needle_total <= total + 1e-6
        event = {'from_occurrence': query['marker'], 'to_occurrence': query['needle'],
            'site_id': query['site_id'],
            'query_output_token_index': query['query_output_token_index'],
            'query_full_sequence_token': position, 'query_token_text': query['query_token_text'],
            'target_city': records[query['needle']-1]['city'], 'records': records,
            'non_needle_context_mass': total-needle_total, 'attention_total_mass': total}
        events.append(event)
        if query['marker'] in reference:
            old = reference[query['marker']]
            assert len(records) == len(old['records']) == 10
            assert [r['source_index'] for r in records] == [r['source_index'] for r in old['records']]
            error = max(abs(a['mass']-b['mass']) for a, b in zip(records, old['records']))
            old_total = sum(r['mass'] for r in old['records'])
            assert old_total > 0
            share_error = max(abs(a['mass']/needle_total-b['mass']/old_total)
                              for a, b in zip(records, old['records']))
            differences.append({'marker': query['marker'], 'maximum_record_mass_difference': error,
                                'maximum_needle_share_difference': share_error})
        print(json.dumps({'marker': query['marker'], 'complete': len(events),
                          'query_seconds': round(time.perf_counter()-query_start, 2),
                          'seconds': round(time.perf_counter()-start, 2)}), flush=True)
    maximum_error = max((r['maximum_record_mass_difference'] for r in differences), default=None)
    maximum_share_error = max((r['maximum_needle_share_difference'] for r in differences), default=None)
    queries_complete = [e['from_occurrence'] for e in events] == list(range(10))
    if queries_complete:
        assert len(differences) == 8
    output = {k: data[k] for k in ['model_label','request_id','model_revision','seed','gold_count','layer','head','grammar']}
    output.update({'schema': 'cot_marker_attention_capture_v1', 'query_site': 'main_figure_marker',
        'events': events, 'queries_complete': queries_complete,
        'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'replay_differences': differences, 'maximum_replay_mass_difference': maximum_error,
        'maximum_replay_share_difference': maximum_share_error,
        'replay_tolerances': {'raw_mass': .005, 'needle_share': .01},
        'replay_requires_review': maximum_error is None or maximum_error > .005 or maximum_share_error > .01,
        'execution': {'device_map': args.device_map, 'device_types': model_device_types,
            'dtype': 'bfloat16', 'torch_threads': torch.get_num_threads(),
            'hostname': platform.node(), 'affinity_cpus': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'requested_markers': requested_markers, 'model_load_seconds': load_seconds,
            'backend_history': backend_history,
            'modeling_sha256': hashlib.sha256(Path(inspect.getfile(load_registered_model)).read_bytes()).hexdigest(),
            'capture_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'packages': {p: importlib.metadata.version(p) for p in ['torch','transformers']},
        'seconds': time.perf_counter()-start})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'CAPTURED' if queries_complete else 'PILOT_CAPTURED', 'queries': len(events),
        'maximum_replay_mass_difference': maximum_error,
        'maximum_replay_share_difference': maximum_share_error,
        'replay_requires_review': output['replay_requires_review']}), flush=True)


if __name__ == '__main__':
    main()
