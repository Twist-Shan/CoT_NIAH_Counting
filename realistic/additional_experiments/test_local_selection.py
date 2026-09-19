import unittest
from local_selection import rank_discovery, random_control, select_heads, target_span, control_audit


class LocalSelectionTests(unittest.TestCase):
    def test_discovery_seed_equal_not_case_equal(self):
        rows = [dict(seed=1, case_id=str(i), split="discovery", heads=[[0, 0, 1], [0, 1, .6]]) for i in range(4)]
        rows.append(dict(seed=2, case_id="z", split="discovery", heads=[[0, 0, 0], [0, 1, .6]]))
        self.assertEqual(rank_discovery(rows)[0][:2], [0, 1])

    def test_no_confirmation_leakage(self):
        with self.assertRaises(ValueError):
            rank_discovery([dict(seed=1254, case_id="x", split="confirmation", heads=[[0, 0, 1]])])

    def test_global_topk_has_no_half_layer_cap(self):
        ranking = [[0, 0, .9], [0, 1, .8], [0, 2, .7], [1, 0, .6]]
        self.assertEqual(select_heads(ranking, 3), [[0, 0], [0, 1], [0, 2]])

    def test_full_layer_sampling_forces_overlap(self):
        selected = [[0, h] for h in range(4)] + [[1, 1]]
        sampled = random_control(selected, [4, 3], 7000)
        self.assertEqual(set(map(tuple, sampled[:4])), set(map(tuple, selected[:4])))
        self.assertEqual(len(sampled), len(set(map(tuple, sampled))))
        self.assertEqual(sampled, random_control(selected, [4, 3], 7000))

    def test_random_layer_counts(self):
        from collections import Counter
        selected = [[0, 0], [0, 2], [3, 1]]
        for seed in [6000, 6001, 6002, 7000, 7001, 7002]:
            result = random_control(selected, [8, 8, 8, 8], seed)
            self.assertEqual(Counter(l for l, h in selected), Counter(l for l, h in result))
            self.assertFalse(set(map(tuple, selected)) & set(map(tuple, result)))

    def test_only_saturated_layers_use_minimum_overlap(self):
        selected = [[0,h] for h in range(5)] + [[1,0], [1,1]]
        for seed in range(25):
            result = random_control(selected, [8,8], seed)
            overlap = set(map(tuple, selected)) & set(map(tuple, result))
            self.assertEqual(len(overlap), 2)
            self.assertTrue(all(layer == 0 for layer, _ in overlap))
            self.assertTrue({(0,5),(0,6),(0,7)} <= set(map(tuple,result)))
            self.assertEqual(control_audit(selected,[8,8],[result])[0][0]['minimum_overlap'],2)

    def test_half_layer_is_disjoint_and_extra_overlap_is_rejected(self):
        selected = [[0,h] for h in range(4)]
        result = random_control(selected,[8],6000)
        self.assertEqual(set(map(tuple,result)),{(0,h) for h in range(4,8)})
        with self.assertRaises(AssertionError):control_audit(selected,[8],[selected])

    def test_target_span_uses_unique_record(self):
        case = dict(passage="abcXYZdef", records=[dict(city="rose", char_start=3, char_end=6)])
        rendered = "!!" + case["passage"]
        offsets = [(i, i + 1) for i in range(len(rendered))]
        self.assertEqual(target_span(case, rendered, offsets, "Rose"), (5, 8))
        with self.assertRaises(ValueError):
            target_span(case, rendered, offsets, "tulip")


if __name__ == "__main__":
    unittest.main()
