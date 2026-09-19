import unittest
from collections import Counter
from kth_retrieval import broad, random_bank, select_broad, exact_pre_city, city_correct, K_LEVELS


class KthRetrievalTests(unittest.TestCase):
    def test_uniform_full_grid(self):
        self.assertEqual(K_LEVELS, tuple(range(1, 11)))

    def test_broad_mass_and_coverage(self):
        uniform=broad([.1]*10)
        focused=broad([1.]+[0.]*9)
        self.assertAlmostEqual(uniform['score'],1.,places=8)
        self.assertAlmostEqual(focused['score'],.1,places=8)
        self.assertEqual(broad([])['score'],0)

    def test_disjoint_controls(self):
        bank=[(0,0),(0,1),(1,2)]
        control=random_bank(bank,[4,4],123)
        self.assertFalse(set(bank)&set(control))
        self.assertEqual(Counter(l for l,h in bank),Counter(l for l,h in control))
        with self.assertRaises(ValueError):
            random_bank([(0,0),(0,1),(0,2)],[4],123)
        self.assertFalse(set(bank)&set(random_bank(bank,[4,4],123,global_control=True)))

    def test_rank_cap(self):
        bank=select_broad([(l,h) for l in range(2) for h in range(4)],[4,4],3)
        self.assertEqual(bank,[(0,0),(0,1),(1,0)])

    def test_first_city_not_later_list(self):
        text=' A city. A again.'
        offsets=[(0,1),(1,2),(2,7),(7,8),(8,9),(9,10)]
        anchor,reason=exact_pre_city(text,'A',offsets,0,len(text))
        self.assertIsNone(reason)
        self.assertEqual(anchor['prefix_length'],1)
        self.assertIsNone(exact_pre_city(text,'B',offsets,0,len(text))[0])

    def test_city_token_prior_text_rejected(self):
        self.assertIsNone(exact_pre_city('x Harbin','Harbin',[(0,1),(1,2),(0,8)],0,8)[0])

    def test_city_scoring_is_prefix_only(self):
        self.assertTrue(city_correct(' Harbin - 76','Harbin'))
        self.assertFalse(city_correct('Vilnius; actually Harbin','Harbin'))
        self.assertFalse(city_correct('Harbinville','Harbin'))


if __name__=='__main__':
    unittest.main()
