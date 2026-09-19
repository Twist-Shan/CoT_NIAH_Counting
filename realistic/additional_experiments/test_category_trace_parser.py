import unittest
from category_trace_parser import parse_category_trace

def case():
    return dict(task='category_count',target_category='flower',records=[
        dict(city='Taipei',ordinal=1,score=88,category='city',is_target=False),
        dict(city='iris',ordinal=2,score=63,category='flower',is_target=True)])

class Tests(unittest.TestCase):
    def test_model_error_preserved(self):
        raw='In the 2024 flower score audit, Taipei received a score of 88. (Flower score - Count = 1)'
        e=parse_category_trace(case(),raw)['events'][0]
        self.assertEqual(e['model_category'],'flower'); self.assertTrue(e['model_is_target'])
        self.assertEqual(e['gold_category'],'city')
        for k in ['record_end','identity_end']:
            s=e[k]; self.assertEqual(raw[s['start']:s['end']],s['text'])
    def test_quote_not_judgement(self):
        e=parse_category_trace(case(),'In the 2024 flower score audit, iris received a score of 63.')['events'][0]
        self.assertEqual(e['quoted_category']['value'],'flower'); self.assertIsNone(e['model_is_target'])
    def test_late_repeat_and_channel(self):
        raw='1. Taipei - 88\n\nLater iris (63). Taipei received 88.\n</think>iris - 63'
        p=parse_category_trace(case(),raw)
        self.assertEqual(len(p['events']),3); self.assertEqual(p['unique_source_records'],2)
        self.assertEqual(p['events'][-1]['visit_number'],2)
    def test_conflict(self):
        e=parse_category_trace(case(),'In the 2024 flower score audit, iris received a score of 63. (Flower score - Ignore)')['events'][0]
        self.assertEqual(e['decision_status'],'conflict'); self.assertIsNone(e['model_is_target'])
    def test_multiline(self):
        raw='Excerpt: In the 2024 flower score audit, iris received a score of 63.\n * Flower: iris\n * Score: 63\n * Type: Flower-score audit record. (Count = 1)\n\nFinal Count: 9'
        e=parse_category_trace(case(),raw)['events'][0]
        self.assertEqual(e['model_explicit_count'],1)
    def test_gold_independence(self):
        c=case(); raw='Taipei - 88'
        a=parse_category_trace(c,raw)['events'][0]
        c['records'][0]['is_target']=True; c['records'][0]['category']='flower'
        b=parse_category_trace(c,raw)['events'][0]
        for k in ['model_is_target','quoted_category','classification_evidence','record_end']:
            self.assertEqual(a[k],b[k])

if __name__=='__main__': unittest.main()
