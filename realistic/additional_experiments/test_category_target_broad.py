import copy
import math
import unittest
from category_target_broad import target_indices, record_geometry, score_record_masses, broad_score, rank_rows


def case(target='city'):
    records = []
    for i in range(10):
        category = 'city' if i < 3 else 'flower'
        records.append(dict(category=category, is_target=category == target,
                            char_start=i*2, char_end=i*2+2, text='x '))
    return dict(target_category=target, records=records, gold='3' if target == 'city' else '7', passage='x '*10)


class TargetBroadTests(unittest.TestCase):
    def test_question_selects_matching_records(self):
        self.assertEqual(target_indices(case()), [0, 1, 2])
        self.assertEqual(target_indices(case('flower')), list(range(3, 10)))

    def test_other_category_cannot_enter_score(self):
        self.assertEqual(score_record_masses(case(), [.02]*3+[.9]*7),
                         score_record_masses(case(), [.02]*3+[0]*7))
        self.assertEqual(score_record_masses(case('flower'), [.9]*3+[.02]*7),
                         score_record_masses(case('flower'), [0]*3+[.02]*7))

    def test_denominator_is_target_count(self):
        self.assertAlmostEqual(score_record_masses(case(), [.02]*10), .06)
        self.assertAlmostEqual(score_record_masses(case('flower'), [.02]*10), .14)
        self.assertAlmostEqual(broad_score([.3, 0, 0]), .1)
        self.assertEqual(broad_score([.3]), .3)
        self.assertEqual(broad_score([0, 0]), 0)

    def test_invalid_records_fail(self):
        bad = case(); bad['gold'] = '5'
        with self.assertRaises(ValueError): target_indices(bad)
        bad = case(); bad['records'][0]['is_target'] = False
        with self.assertRaises(ValueError): target_indices(bad)
        for bad in ([], [-1], [math.nan]):
            with self.assertRaises(ValueError): broad_score(bad)

    def test_geometry_uses_original_record_spans(self):
        c = case('flower'); rendered = 'HEADER|'+c['passage']+'end'
        offsets = [(i, i+1) for i in range(len(rendered))]
        geo = record_geometry(c, rendered, offsets)
        self.assertEqual(geo['all_record_token_spans'], [[7+i*2, 9+i*2] for i in range(10)])
        self.assertEqual(geo['target_record_count'], 7)

    def test_discovery_only_seed_equal_ranking(self):
        rows = [dict(case_id='a', seed=1, split='discovery', heads=[[0, 0, .9], [0, 1, .5]]),
                dict(case_id='b', seed=1, split='discovery', heads=[[0, 0, .9], [0, 1, .5]]),
                dict(case_id='c', seed=2, split='discovery', heads=[[0, 0, 0.], [0, 1, .5]])]
        self.assertEqual(rank_rows(rows), [[0, 1, .5], [0, 0, .45]])
        self.assertEqual(rank_rows(rows), rank_rows(list(reversed(rows))))
        bad = copy.deepcopy(rows); bad[0]['split'] = 'confirmation'
        with self.assertRaises(ValueError): rank_rows(bad)


if __name__ == '__main__': unittest.main()
