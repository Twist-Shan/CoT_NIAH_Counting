import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from task_scoring import score_generation, target_token_geometry


class TaskScoringTests(unittest.TestCase):
    def setUp(self):
        self.case = dict(task="kth_needle", gold="Paris|71", answer_prefix="Needle:", level=1,
                         records=[dict(city="Paris", score=71, ordinal=1),dict(city="Tokyo", score=84, ordinal=2)])

    def score(self, text):
        return score_generation(dict(completion_text_raw=text), self.case, mode="native_thinking", assay="targeted", target="Paris")

    def test_wrong_first_record_cannot_be_repaired_by_later_correct_record(self):
        text='In the 2024 city score audit, Tokyo received a score of 84.\nIn the 2024 city score audit, Paris received a score of 71.'
        self.assertFalse(self.score(text)["correct"])

    def test_correct_first_record_is_not_invalidated_by_later_wrong_final(self):
        text='In the 2024 city score audit, Paris received a score of 71.\n</think>\nNeedle:Tokyo|84'
        self.assertTrue(self.score(text)["correct"])

    def test_unknown_flower_before_target_is_an_error(self):
        text='In the 2024 flower score audit, sunflower received a score of 30.\nIn the 2024 city score audit, Paris received a score of 71.'
        result=self.score(text)
        self.assertEqual(result['prediction'],'sunflower')
        self.assertFalse(result['correct'])

    def test_no_semantic_record_is_failure(self):
        self.assertFalse(self.score('I cannot identify the next record.')['correct'])

    def test_main_token_fallback_only_without_identifiable_record(self):
        from realistic_niah_v5.causal import _retrieval_behavior_score
        for text,offset in [('unparsed continuation',0),('unparsed continuation',1),('Tokyo received a score of 84.',0)]:
            result=score_generation(dict(completion_text_raw=text,generated_token_ids=[7,8]),self.case,mode='native_thinking',assay='targeted',target='Paris',target_token_ids=[7],target_token_offset=offset)
            main=_retrieval_behavior_score(text,expected_city='Paris',gold_cities=['Paris','Tokyo'],exact_target_prefix=offset==0)
            self.assertEqual(result['correct'],main['correct_next_needle'])

    def test_token_geometry_retains_original_ids_when_reencoding_differs(self):
        class Tokenizer:
            pieces={1:'1.',2:' Paris',3:' received'}
            def decode(self,ids,**kwargs):return ''.join(self.pieces[i] for i in ids)
            def __call__(self,text,**kwargs):return dict(input_ids=[99],offset_mapping=[(0,len(text))])
        self.assertEqual(target_token_geometry(Tokenizer(),'1. Paris received',[1,2,3],3,'Paris',1),dict(target_token_ids=[2],target_token_offset=0))

    def test_broad_requires_city_and_score(self):
        g=dict(completion_text_raw='Paris|72',generation_truncated=False,generated_token_count=4)
        self.assertFalse(score_generation(g,self.case,mode='nonthinking',assay='broad')['correct'])
        g['completion_text_raw']='Paris|71'
        self.assertTrue(score_generation(g,self.case,mode='nonthinking',assay='broad')['correct'])


if __name__=='__main__':unittest.main()
