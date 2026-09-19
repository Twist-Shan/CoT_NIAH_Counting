import unittest
from category_trial import make_pair, prompt


class CategoryTests(unittest.TestCase):
    def test_template_background_and_paired_queries(self):
        passage, records = '', []
        for i in range(10):
            passage += f'Background {i}.\n'
            text = f'\u2029Excerpt:\nIn the 2024 city score audit, City{i} received a score of {70+i}.\nEnd excerpt.\u2029'
            start = len(passage)
            passage += text
            records.append({'city': f'City{i}', 'score': 70+i, 'text': text,
                            'char_start': start, 'char_end': len(passage)})
        source = {'seed': 1234, 'gold_count': 10, 'passage': passage, 'active_needle_spans': records}
        city, flower = list(make_pair(source, 2))
        self.assertEqual(city['passage'], flower['passage'])
        self.assertEqual((city['gold'], flower['gold']), ('2', '8'))
        for original, new in zip(records, city['records']):
            self.assertEqual(original['score'], new['score'])
            self.assertEqual(city['passage'][new['char_start']:new['char_end']], new['text'])
            if new['category'] == 'city':
                self.assertEqual(original['text'], new['text'])
        def background(text, spans):
            last, pieces = 0, []
            for r in spans:
                pieces.append(text[last:r['char_start']])
                last = r['char_end']
            return pieces + [text[last:]]
        self.assertEqual(background(passage, records), background(city['passage'], city['records']))
        self.assertIn('count only city-score', prompt(city, 'native_thinking'))
        self.assertIn('count only flower-score', prompt(flower, 'native_thinking'))


if __name__ == '__main__':
    unittest.main()
