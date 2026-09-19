import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent/'deployment'))
from compile_transfer_registry import token_boundaries,compile_one

class Tokenizer:
    def __call__(self,text,**kwargs):
        return {'input_ids':[ord(c) for c in text], 'offset_mapping':[(i,i+1) for i in range(len(text))]}
    def decode(self,ids,**kwargs): return ''.join('ab' if i==1000 else chr(i) for i in ids)

class RegistryTests(unittest.TestCase):
    def test_original_segmentation_recovery(self):
        b,exact,method=token_boundaries(Tokenizer(),'abC',[1000,67])
        self.assertFalse(exact); self.assertEqual(b,{0:0,2:1,3:2})
        self.assertNotIn(1,b)  # Must not invent a boundary inside a fused token.
    def test_mismatch_unavailable(self):
        b,_,method=token_boundaries(Tokenizer(),'wrong',[65])
        self.assertEqual(b,{}); self.assertEqual(method,'original_decode_mismatch')
    def test_fulltrace_and_answer_scope(self):
        raw='A - 71\n\nLater B - 72\n</think>\n**Needle:** B|72'
        c={'task':'kth_needle','case_id':'x','seed':1,'split':'discovery','level':2,'records':[{'city':'A','score':71,'ordinal':1},{'city':'B','score':72,'ordinal':2}]}
        g={'completion_text_raw':raw,'generated_token_ids':[ord(x) for x in raw]}
        p={'rendered_prompt':'P','input_ids':[80]}
        r=compile_one(c,p,g,Tokenizer(),'Qwen3-8B','native_thinking')
        self.assertEqual(len(r['events']),2)
        self.assertEqual(r['target_query']['prefix_length'],1+raw.index('B'))
        self.assertEqual(r['answer_query']['prefix_length'],1+raw.index('Needle:')+7)
    def test_gemma_channel_excludes_final(self):
        raw='A - 71<channel|>Needle:A|71<turn|>'
        c={'task':'kth_needle','case_id':'x','seed':1,'split':'discovery','level':1,'records':[{'city':'A','score':71,'ordinal':1}]}
        r=compile_one(c,{'rendered_prompt':'P','input_ids':[80]}, {'completion_text_raw':raw,'generated_token_ids':list(map(ord,raw))},Tokenizer(),'Gemma4-E4B','native_thinking')
        self.assertTrue(r['reasoning_closed']); self.assertEqual(len(r['identities']),1)
        self.assertIn('query_position',r['answer_query'])

if __name__=='__main__': unittest.main()
