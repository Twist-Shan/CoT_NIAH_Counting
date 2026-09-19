import unittest
from list_early_stop import first_record_list

RECORDS = [{'city': c, 'ordinal': i + 1, 'is_target': i != 1}
           for i, c in enumerate(('Harbin', 'Vilnius', 'Seoul'))]


class ListBoundaryTests(unittest.TestCase):
    def test_ignore_prose_and_post_list_repetition(self):
        text = 'Harbin 76 is first.\n\n1. Harbin 76\n2. Vilnius 67\n\nVilnius 67 is the answer.'
        boundary, reason = first_record_list(text, RECORDS)
        self.assertIsNone(reason)
        self.assertEqual([i['city'] for i in boundary['items']], ['Harbin', 'Vilnius'])
        self.assertEqual(text[:boundary['end']].splitlines()[-1], '2. Vilnius 67')

    def test_multiline_items_and_blank_lines(self):
        text = ('1. **Excerpt 1:**\n   Harbin score 76.\n   City: Harbin; astronomy. Count=1\n\n'
                '2. **Excerpt 2:**\n   Vilnius score 67.\n   Botany.\n\nThere are two records.')
        boundary, _ = first_record_list(text, RECORDS)
        self.assertEqual(len(boundary['items']), 2)
        self.assertTrue(boundary['items'][0]['line'].endswith('Count=1'))
        self.assertTrue(boundary['items'][1]['line'].endswith('Botany.'))

    def test_partial_next_number_does_not_stop(self):
        for tail in ('', '\n', '\n2', '\n2.', '\n**2.'):
            boundary, _ = first_record_list('1. Harbin\n' + tail, RECORDS)
            self.assertIsNone(boundary)
        boundary, _ = first_record_list('1. Harbin\n\nThe', RECORDS)
        self.assertIsNotNone(boundary)

    def test_skip_definition_list_without_record_cities(self):
        text = '1. Name a city.\n2. Give a score.\n\nScan:\n1. Harbin: astronomy\n2. Vilnius: botany\n\nDone.'
        boundary, _ = first_record_list(text, RECORDS)
        self.assertEqual(len(boundary['items']), 2)

    def test_first_episode_is_fixed_even_when_later_list_differs(self):
        text = '1. Harbin\n2. Vilnius\n1. Harbin\n2. Seoul\n\nDone.'
        boundary, _ = first_record_list(text, RECORDS)
        self.assertEqual([i['city'] for i in boundary['items']], ['Harbin', 'Vilnius'])

    def test_ambiguous_first_episode_not_replaced_by_later_valid_one(self):
        text = '1. Harbin and Vilnius\n\nAgain:\n1. Harbin\n2. Seoul\n\nDone.'
        boundary, reason = first_record_list(text, RECORDS)
        self.assertIsNone(boundary)
        self.assertEqual(reason, 'first_record_list_ambiguous_item')

    def test_record_order_errors_are_preserved(self):
        boundary, _ = first_record_list('1. Seoul\n2. Harbin\n\nDone.', RECORDS)
        self.assertEqual([i['ordinal'] for i in boundary['items']], [3, 1])
        self.assertFalse(boundary['passage_order_monotone'])

    def test_final_eos_without_following_paragraph(self):
        self.assertIsNone(first_record_list('1. Harbin', RECORDS)[0])
        self.assertIsNotNone(first_record_list('1. Harbin', RECORDS, final=True)[0])

    def test_nested_record_list_inside_workflow_step(self):
        text = ('1. **Scan:**\n    * Record 1: Harbin 76\n    * Record 2: Vilnius 67\n\n'
                '2. **Select:**\n    Vilnius is the answer.\n\n3. Format output.')
        boundary, reason = first_record_list(text, RECORDS)
        self.assertIsNone(reason)
        self.assertTrue(boundary['nested'])
        self.assertEqual(len(boundary['items']), 2)
        self.assertEqual(text[boundary['items'][0]['start']:boundary['items'][0]['end']].strip(),
                         '* Record 1: Harbin 76')

    def test_repeated_visits_inside_one_list_are_preserved(self):
        boundary, reason = first_record_list('1. Harbin\n2. Vilnius\n3. Vilnius\n\nDone.', RECORDS)
        self.assertIsNone(reason)
        self.assertEqual(len(boundary['items']), 3)
        self.assertEqual(boundary['repeated_record_visits'], 1)


if __name__ == '__main__':
    unittest.main()
