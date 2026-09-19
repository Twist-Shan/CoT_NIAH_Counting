from types import SimpleNamespace
import pytest
import torch
from test_v58_commit_query import fixture_model
from synthetic_counting_v20.data import render_v20
from synthetic_counting_v20.model import _apply_rope
from run_v58_top1to8_aligned import generate as full_generate
from v58_cached_decode import generate as cached_generate, rope_at


def test_rope_position_matches_original_full_sequence_rotation():
    torch.manual_seed(4)
    x=torch.randn(2,8,29,8)
    full=_apply_rope(x,base=10000.)
    for p in [0,1,17,28]:
        assert torch.equal(full[:,:,p:p+1],rope_at(x[:,:,p:p+1],p,10000.))


@pytest.mark.parametrize('scope',['sustained','answer_query_only'])
@pytest.mark.parametrize('heads',[[],[(1,0)],[(1,0),(2,3),(4,7)]])
def test_cached_trajectory_matches_full_recomputation(scope,heads):
    torch.set_num_threads(1)
    torch.manual_seed(13)
    model,vocab,e,_=fixture_model()
    item=render_v20(e,vocab,'nonthinking')
    r={'mode':'nonthinking','key':e.prompt_sha256,'count':e.count,'primary_eligible':True,
       'prefix_tokens':item.tokens[:item.spans.ans_pos+1]}
    cfg=SimpleNamespace(device='cpu',n_embd=64,n_head=8,n_positions=model.config.n_positions)
    a,_=full_generate(model,cfg,vocab,[r],{r['key']:e},heads,scope,1,12)
    b,_=cached_generate(model,cfg,vocab,[r],{r['key']:e},heads,scope,1,12)
    assert a.generated_tokens.tolist()==b.generated_tokens.tolist()


def test_cached_variable_length_thinking_queries_match_full_prefix_recomputation():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    model,vocab,e,item=fixture_model()
    records=[]
    for n in [1,4,10]:
        q=item.spans.trace_marker_positions[n-1]-1
        records.append({'mode':'thinking','key':str(n),'count':10,'primary_eligible':True,
                        'prefix_tokens':item.tokens[:q+1],'expected_marker':e.needle_markers[n-1]})
    lookup={r['key']:e for r in records}
    cfg=SimpleNamespace(device='cpu',n_embd=64,n_head=8,n_positions=model.config.n_positions)
    a,_=full_generate(model,cfg,vocab,records,lookup,[(1,3),(4,5)],'sustained',3,12)
    b,_=cached_generate(model,cfg,vocab,records,lookup,[(1,3),(4,5)],'sustained',3,12)
    assert a.generated_tokens.tolist()==b.generated_tokens.tolist()
