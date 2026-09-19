import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / 'deployment'), str(HERE), str(HERE.parent / 'src')]
from task_local_inputs import make_plan, broad_score, broad_spans, expected_points, validate_cases
from prepare_fresh_task_local import verified_capture
from prepare_task_local import launcher, sha


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return dict(input_ids=list(map(ord, text)), offset_mapping=[(i, i+1) for i in range(len(text))])

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


def example():
    records, pieces, cursor = [], [], 0
    for i, name in enumerate(['Athens', 'Berlin', 'Cairo', 'Dublin', 'Essen', 'Florence', 'Geneva', 'Hanoi', 'Irvine', 'Jakarta']):
        text = f'In the 2024 city score audit, {name} received a score of {71+i}.'
        records.append(dict(city=name, score=71+i, ordinal=i+1, text=text, char_start=cursor, char_end=cursor+len(text)))
        pieces.append(text+'\n')
        cursor += len(text)+1
    case = dict(case_id='example', task='kth_needle', seed=1234, split='discovery',
                level=2, records=records, passage=''.join(pieces), gold='Berlin|72', answer_prefix='Needle:')
    text = case['passage']+'\nQuestion'
    prompt = dict(user_text=text, rendered_prompt=text, input_ids=list(map(ord, text)))
    raw = '\n'.join(f'{i+1}. {r["city"]} - {r["score"]}' for i, r in enumerate(records))+'\n</think>\nNeedle:Berlin|72'
    generation = dict(completion_text_raw=raw, generated_token_ids=list(map(ord, raw)),
                      generated_token_count=len(raw), generation_truncated=False, correct=True)
    return case, prompt, generation


class FreshTaskTests(unittest.TestCase):
    def test_original_tokens_and_anchor_do_not_depend_on_correctness(self):
        c, p, g = example()
        plan, _ = make_plan(c, p, g, CharacterTokenizer(), 'Qwen3-8B', 'native_thinking')
        self.assertIsNotNone(plan['target'])
        changed = {**g, 'correct': False}
        self.assertEqual(plan, make_plan(c, p, changed, CharacterTokenizer(), 'Qwen3-8B', 'native_thinking')[0])
        self.assertLessEqual(plan['target']['query_char_end'], plan['target']['target_city_start'])

    def test_corrupt_original_ids_rejected(self):
        c, p, g = example()
        g['generated_token_ids'][0] += 1
        with self.assertRaisesRegex(ValueError, 'token IDs'):
            make_plan(c, p, g, CharacterTokenizer(), 'Qwen3-8B', 'native_thinking')

    def test_missing_sites_are_explicit_and_confirmation_is_separate(self):
        c, p, g = example()
        c['seed'] = 1254
        g.update(completion_text_raw='No trace', generated_token_ids=list(map(ord, 'No trace')))
        plan, _ = make_plan(c, p, g, CharacterTokenizer(), 'Gemma4-E4B', 'native_thinking')
        self.assertEqual(plan['split'], 'confirmation')
        self.assertIsNone(plan['target'])
        self.assertIsNone(plan['broad_prefix'])
        self.assertIn('targeted', plan['unavailable'])

    def test_nonthinking_uses_all_source_records(self):
        c, p, g = example()
        plan, _ = make_plan(c, p, g, CharacterTokenizer(), 'Qwen3-8B', 'nonthinking')
        spans = broad_spans(plan, p, CharacterTokenizer())
        self.assertEqual(len(spans), 10)
        self.assertEqual(plan['broad_prefix'], len(p['input_ids']))
        self.assertEqual(spans[0], (0, len(c['records'][0]['text'])))

    def test_broad_score_uniform_concentrated_zero(self):
        from kth_retrieval import broad
        for masses in ([.1]*10, [1.]+[0.]*9, [0.]*10, [1e-14]*10):
            self.assertEqual(broad_score(masses), broad(masses)['score'])
        self.assertAlmostEqual(broad_score([.1]*10), 1.)
        self.assertAlmostEqual(broad_score([1.]+[0.]*9), .1)
        self.assertEqual(broad_score([0.]*10), 0.)
        with self.assertRaises(ValueError):
            broad_score([float('nan')])

    def test_expected_points_follow_eligibility(self):
        c, p, g = example()
        c['seed'] = 1254
        plan, _ = make_plan(c, p, g, CharacterTokenizer(), 'Qwen3-8B', 'native_thinking')
        self.assertEqual(expected_points([plan], 'Qwen3-8B'), 14)
        plan['target'] = None
        self.assertEqual(expected_points([plan], 'Qwen3-8B'), 8)
        plan['split'] = 'discovery'
        self.assertEqual(expected_points([plan], 'Qwen3-8B'), 0)

    def test_design_grid_and_hash_drift_rejected(self):
        c, p, g = example()
        cases = [{**copy.deepcopy(c), 'seed': seed, 'level': k, 'case_id': f'{seed}_{k}'}
                 for seed in range(1234, 1264) for k in range(1, 11)]
        for case in cases:
            target = case['records'][case['level']-1]
            case['gold'] = f"{target['city']}|{target['score']}"
        validate_cases(cases, 'kth')
        cases[0]['level'] = 2
        with self.assertRaisesRegex(ValueError, 'design'):
            validate_cases(cases, 'kth')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, value in [('prompt.json', p), ('generation.json', g)]:
                (root/name).write_text(json.dumps(value), encoding='utf-8')
            (root/'complete.json').write_text(json.dumps(dict(status='PASS',
                files={n:sha(root/n) for n in ('prompt.json', 'generation.json')})))
            self.assertEqual(verified_capture(root), (p, g))
            (root/'generation.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verified_capture(root)

    def test_launcher_uses_runtime_python(self):
        self.assertIn('PYTHON=${PYTHON:-python}', launcher())
        self.assertIn("PYTHON=${PYTHON:-'/a path/python'}", launcher('/a path/python'))


if __name__ == '__main__':
    unittest.main()
