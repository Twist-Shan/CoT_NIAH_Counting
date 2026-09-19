import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from realistic_niah_v5.parsing import TraceCharSite
from realistic_niah_v5.capture import _broad_span_metrics
from native_broad_full_span import geometry_from_sites, score_masses


class CharTokenizer:
    def encode(self,text,**kwargs):return [ord(c) for c in text]
    def decode(self,ids,**kwargs):return ''.join(chr(i) for i in ids)


class NativeFullSpanTests(unittest.TestCase):
    def site(self,start,end,i=1):
        return TraceCharSite(site_id=f'item_end:{i}',site_kind='item_end',occurrence=i,city='Paris',
            marker=i,boundary_kind='test',char_start=start,char_end=end,primary=True)
    def test_entire_record_and_original_absolute_coordinates(self):
        raw='1. Paris 71\n2. Tokyo 84\nTotal:'
        g=geometry_from_sites(CharTokenizer(),raw,list(map(ord,raw)),10,10+len(raw),[self.site(0,11),self.site(12,23,2)])
        self.assertEqual([(r['token_start'],r['token_end']) for r in g['records']],[(10,21),(22,33)])
        self.assertEqual(g['records'][0]['text'],'1. Paris 71')
        self.assertGreater(len(g['records'][0]['token_ids']),1)
    def test_future_record_cannot_enter_answer_query(self):
        raw='Paris\nTokyo'
        g=geometry_from_sites(CharTokenizer(),raw,list(map(ord,raw)),10,15,[self.site(0,5),self.site(6,11,2)])
        self.assertEqual(g['record_count'],1)
        self.assertEqual(len(g['excluded_records']),1)
    def test_nonliteral_boundary_is_excluded_not_rounded(self):
        class DifferentEncoding(CharTokenizer):
            def encode(self,text,**kwargs):return [999] if text=='Pa' else super().encode(text,**kwargs)
        raw='Paris'
        g=geometry_from_sites(DifferentEncoding(),raw,list(map(ord,raw)),10,15,[self.site(2,5)])
        self.assertFalse(g['available'])
        self.assertEqual(g['record_count'],0)
    def test_score_exactly_uses_main_metric_including_epsilon(self):
        for masses in [[0,0],[.3],[.02]*5,[.3,0,0],[1e-15,1e-15]]:
            self.assertEqual(score_masses(masses),_broad_span_metrics(masses)['score'])
        for bad in [[],[-1],[float('nan')],[float('inf')]]:
            with self.assertRaises(ValueError):score_masses(bad)


if __name__=='__main__':unittest.main()
