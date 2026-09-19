"""Register all ten Gemma marker queries from the saved trace, without inference."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE / 'dependencies'))
sys.path.insert(0, str(ROOT / 'realistic/src'))
from tokenizers import Tokenizer
from realistic_niah_v5.causal_sites import compile_causal_site_plan
from realistic_niah_v5.encoding import build_native_trace_encoding, build_native_causal_encoding


class LocalTokenizer:
    """Expose the saved Rust tokenizer through the parser's small HF interface."""
    def __init__(self, path):
        self.backend = Tokenizer.from_file(str(path))

    def encode(self, text, add_special_tokens=False):
        return self.backend.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        return self.backend.decode(ids, skip_special_tokens=skip_special_tokens)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        encoded = self.backend.encode(text, add_special_tokens=add_special_tokens)
        result = {'input_ids': encoded.ids, 'attention_mask': encoded.attention_mask}
        if return_offsets_mapping:
            result['offset_mapping'] = encoded.offsets
        return result


row_path = HERE / 'gemma_seed1240_generation.json'
row = json.loads(row_path.read_text(encoding='utf-8'))
tokenizer_path = ROOT / ('realistic/work/hf_tokenizers/'
    'models--google--gemma-4-E4B-it/snapshots/'
    'ee0ef6023621cff504d758262d4e04895a5af4a2/tokenizer.json')
tokenizer = LocalTokenizer(tokenizer_path)
assert tokenizer.encode(row['raw_output_text']) == row['output_token_ids']
plan = compile_causal_site_plan(row, tokenizer)
queries = []
for marker in range(10):
    if marker == 0:
        site = plan['block_pre_d1']
        assert site['status'] == 'ok'
        encoding = build_native_causal_encoding(row, tokenizer,
            query_output_token_index=int(site['output_token_index']),
            sequence_output_token_end=int(site['output_prefix_token_count']), selected_site=site)
        site_id = 'initial_block_pre_d1'
    else:
        site_id = f'item_end:{marker}'
        encoding = build_native_trace_encoding(row, tokenizer, site_id=site_id)
    output_index = encoding.query_position - encoding.prompt_token_count
    expected_ids = row['input_ids'] + row['output_token_ids'][:output_index + 1]
    assert list(encoding.input_ids) == expected_ids
    queries.append({'marker': marker, 'needle': marker + 1, 'site_id': site_id,
        'query_output_token_index': output_index,
        'query_full_sequence_token': encoding.query_position,
        'query_token_text': tokenizer.decode([encoding.input_ids[encoding.query_position]])})

atlas_path = ROOT / ('realistic/work/v5_native_p0_head_atlas_20260820/'
                     'p0_head_atlas_gemma.json')
atlas = json.loads(atlas_path.read_text(encoding='utf-8'))
reference = next(e for e in atlas['examples'] if e['grammar'] == 'adjacent_rank_after_city'
                 and (e['layer'], e['head']) == (29, 4))
assert reference['request_id'] == row['request_id']
for event in reference['events']:
    query = queries[event['from_occurrence']]
    assert query['query_full_sequence_token'] == event['query_full_sequence_token']
    assert query['query_output_token_index'] == event['query_output_token_index']

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
bundle = {'schema': 'cot_marker_capture_input_v1', 'status': 'PREPARED',
    'model_label': row['model_label'], 'model_revision': 'ee0ef6023621cff504d758262d4e04895a5af4a2',
    'request_id': row['request_id'], 'seed': row['seed'], 'gold_count': 10,
    'layer': 29, 'head': 4, 'grammar': 'adjacent_rank_after_city',
    'query_rule': 'Main-figure Marker 0 is block_pre_d1; Marker k is item_end:k for k=1..9.',
    'queries': queries, 'input_ids': row['input_ids'], 'attention_mask': row['attention_mask'],
    'output_token_ids': row['output_token_ids'], 'prompt_record_spans': row['prompt_record_spans'],
    'reference_events': reference['events'], 'existing_eight_queries_match': True,
    'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in
        [row_path, tokenizer_path, atlas_path,
         ROOT / 'realistic/src/realistic_niah_v5/causal_sites.py',
         ROOT / 'realistic/src/realistic_niah_v5/encoding.py']}}
(HERE / 'input.json').write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({'status': 'PREPARED', 'queries': queries, 'existing_eight_queries_match': True},
                 indent=2, ensure_ascii=True))
