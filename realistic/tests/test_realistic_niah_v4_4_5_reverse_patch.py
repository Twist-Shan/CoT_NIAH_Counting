from dataclasses import replace
import pytest
import torch
from torch import nn
from types import SimpleNamespace
from realistic_niah_v4.prompts import PromptEncoding, TokenSpan
from realistic_niah_v4_4_5.restoration import CorruptionPlan, corrupt_encoding, residual_patch_hook
from realistic_niah_v4_4_5.reverse_patch import audit_alignment, damage_metrics


def pair():
    span=TokenSpan(0,2,4,True,'needle',2,2)
    enc=PromptEncoding('unit','v4.4',1234,'discovery',1,'toy','numeric','','',
        tuple(range(12)),(1,)*12,11,(span,),(span,),(),(),(),())
    plan=CorruptionPlan(((2,4),),((4,6),),((6,8),),((8,10),))
    corrupt,_=corrupt_encoding(enc,plan,condition='needle_corrupt')
    return enc,corrupt,plan


def test_alignment_checks_exact_replacement_and_coordinates():
    clean,corrupt,plan=pair()
    result=audit_alignment(clean,corrupt,plan,'needle')
    assert result['positions']==[2,3] and result['endpoints']==[3]
    assert result['changed_tokens']==2 and result['status']=='PASS'
    ordinary,_=corrupt_encoding(clean,plan,condition='ordinary_corrupt')
    assert audit_alignment(clean,ordinary,plan,'ordinary')['token_budget']==2
    for broken in (replace(corrupt,query_position=10),
                   replace(corrupt,attention_mask=(0,)+(1,)*11),
                   replace(corrupt,input_ids=corrupt.input_ids+(12,)),
                   replace(corrupt,input_ids=(99,)+corrupt.input_ids[1:]),
                   replace(corrupt,input_ids=clean.input_ids)):
        with pytest.raises(ValueError): audit_alignment(clean,broken,plan,'needle')


def test_damage_sign_overshoot_and_undefined_denominator():
    assert damage_metrics(8,3,6,8)['normalized_damage']==pytest.approx(.4)
    assert damage_metrics(8,3,6,8)['expected_error_increase']==2
    assert damage_metrics(8,3,1,8)['normalized_damage']>1
    assert damage_metrics(3,3,4,8)['normalized_damage'] is None
    assert damage_metrics(6,3,7,8)['expected_error_increase']==-1
    with pytest.raises(ValueError): damage_metrics(float('nan'),3,4,8)


def test_patch_changes_only_selected_positions_and_skips_decode():
    clean,_,_=pair()
    layer=nn.Identity()
    adapter=SimpleNamespace(layers=[layer],num_layers=1)
    source=torch.zeros(1,12,3)
    with residual_patch_hook(adapter,clean,layer=0,positions=[2,3],replacement=torch.ones(2,3)) as audit:
        result=layer(source)
        assert torch.equal(result[0,2:4],torch.ones(2,3))
        assert result.sum()==6 and source.sum()==0
        assert layer(torch.zeros(1,1,3)).sum()==0
        assert audit['count']==1
    assert layer(source).sum()==0
