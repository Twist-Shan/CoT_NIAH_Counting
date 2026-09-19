import copy
import unittest

from trace_parser import parse_trace, align_sites

CASE = {'task': 'topic_count', 'target_topic': 'astronomy', 'records': [
    {'city': 'Harbin', 'ordinal': 1, 'score': 76, 'is_target': True},
    {'city': 'Vilnius', 'ordinal': 2, 'score': 67, 'is_target': False},
    {'city': 'Seoul', 'ordinal': 3, 'score': 83, 'is_target': True}]}


class TraceParserTests(unittest.TestCase):
    def test_separate_record_decision_count(self):
        raw = '1. Harbin with 76.\n   Topic: comet (Astronomy). Count: 1\n2. Vilnius with 67. Not astronomy.\n\nDone.</think>\nTotal:1'
        parsed = parse_trace(CASE, raw)
        a, b = parsed['items']
        self.assertEqual(a['record_end']['text'], 'Harbin with 76')
        self.assertTrue(a['model_is_target'])
        self.assertEqual(a['model_explicit_count'], 1)
        self.assertFalse(b['model_is_target'])
        self.assertIsNone(b['model_explicit_count'])
        self.assertEqual(parsed['coverage']['unvisited_source_ordinals'], [3])

    def test_gold_is_not_a_decision_or_stop_rule(self):
        case = copy.deepcopy(CASE)
        case['gold'] = 999
        for r in case['records']:
            r['is_target'] = not r['is_target']
        raw = '1. Harbin with 76. Project tracks a comet.\n\nDone.'
        a, b = parse_trace(CASE, raw), parse_trace(case, raw)
        self.assertEqual(a['episode_span'], b['episode_span'])
        self.assertIsNone(a['items'][0]['model_is_target'])
        self.assertIsNone(b['items'][0]['model_is_target'])

    def test_repeat_error_and_wrong_score_preserved(self):
        parsed = parse_trace(CASE, '1. Seoul with 10.\n2. Harbin with 76.\n3. Harbin with 76.\n\nDone.')
        self.assertFalse(parsed['items'][0]['score_matches_source'])
        self.assertFalse(parsed['coverage']['passage_order_monotone'])
        self.assertEqual(parsed['coverage']['repeated_visits'], 1)
        self.assertEqual(parsed['items'][-1]['gold_target_visits_so_far'], 3)
        self.assertEqual(parsed['items'][-1]['gold_unique_targets_visited'], 2)

    def test_later_reclassification_not_merged(self):
        parsed = parse_trace(CASE, '1. Harbin with 76.\n\nCheck again: Harbin counts.\n</think>Harbin')
        self.assertIsNone(parsed['items'][0]['model_is_target'])
        self.assertEqual(len(parsed['mention_candidates']), 2)

    def test_conflicting_decisions_and_counts(self):
        a = parse_trace(CASE, '1. Harbin counts. Count: 1. Wait, not astronomy. Count: 0.\n\nDone.')['items'][0]
        self.assertEqual(a['decision_status'], 'conflict')
        self.assertIsNone(a['decision_end'])
        self.assertEqual(a['count_status'], 'multiple')
        self.assertIsNone(a['count_end'])

    def test_prose_is_inventory_only(self):
        parsed = parse_trace(CASE, 'First Harbin with 76, then Vilnius with 67.')
        self.assertEqual(parsed['status'], 'unavailable')
        self.assertEqual(len(parsed['mention_candidates']), 2)
        self.assertEqual(parsed['items'], [])

    def test_final_answer_excluded(self):
        parsed = parse_trace(CASE, 'No list.<channel|>1. Harbin with 76.\n2. Vilnius with 67.')
        self.assertEqual(parsed['mention_candidates'], [])

    def test_non_unique_identity_rejected(self):
        case = copy.deepcopy(CASE)
        case['records'].append(case['records'][0])
        with self.assertRaises(ValueError):
            parse_trace(case, '1. Harbin')

    def test_exact_tokens_only(self):
        parsed = parse_trace(CASE, '1. Harbin with 76.\n\nDone.')
        end = parsed['items'][0]['identity_end']['end']
        site = align_sites(parsed, [(0, end + 1)])[0]
        self.assertIsNone(site['position'])
        self.assertEqual(align_sites(parsed, [(0, end)])[0]['position'], 0)

    def test_city_dash_score(self):
        item = parse_trace(CASE, '1. Harbin - 76\n\nDone.')['items'][0]
        self.assertEqual(item['observed_score'], 76)
        self.assertTrue(item['score_matches_source'])

    def test_non_topic_counting_not_membership(self):
        case = copy.deepcopy(CASE)
        case['task'] = 'count_all'
        item = parse_trace(case, '1. Harbin - 76. This counts.\n\nDone.')['items'][0]
        self.assertIsNone(item['model_is_target'])


if __name__ == '__main__':
    unittest.main()
