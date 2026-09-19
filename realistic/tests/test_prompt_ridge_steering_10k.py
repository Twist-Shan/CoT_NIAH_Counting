from types import SimpleNamespace
import numpy as np
import pytest
import torch

from realistic_niah_v4.prompt_ridge_steering import fit_direction, additive_span_hook


def test_backprojected_probe_and_random_direction():
    rng=np.random.default_rng(1)
    x=rng.normal(size=(100,40)).astype(np.float32)
    y=3*x[:,0]-2*x[:,1]+5
    probe,audit=fit_direction(x,y,x)
    assert audit['projection_max_error']<1e-3
    assert abs(probe['unit']@probe['random_unit'])<1e-6
    assert np.isclose(np.linalg.norm(probe['unit']),1.)
    assert np.isfinite(x@probe['weight']+probe['intercept']).all()


@pytest.mark.parametrize('condition,beta',[('noop',0),('ridge',1),('random',-1)])
def test_hook_changes_only_span_once_and_matches_bf16_norm(condition,beta):
    block=torch.nn.Identity()
    adapter=SimpleNamespace(num_layers=1,layers=[block])
    encoding=SimpleNamespace(query_position=8,sequence_length=9)
    rng=np.random.default_rng(2)
    x=rng.normal(size=(100,40)).astype(np.float32)
    probe,_=fit_direction(x,x[:,0],x)
    hidden=torch.tensor(rng.normal(size=(1,9,40)),dtype=torch.bfloat16)
    with additive_span_hook(adapter,encoding,layer=0,positions=[2,3],probe=probe,
            beta=beta,condition=condition) as audit:
        value=block(hidden)
        decode=block(hidden[:,:1])
    assert torch.equal(decode,hidden[:,:1])
    assert torch.equal(value[:,[0,1,4,5,6,7,8]],hidden[:,[0,1,4,5,6,7,8]])
    assert audit['applications']==1
    if condition=='noop': assert torch.equal(value,hidden)
    else:
        assert audit['realized_norm']>0
        assert abs(audit['realized_norm_ratio']-1)<.05
    assert not block._forward_hooks


def test_missing_hook_is_error():
    adapter=SimpleNamespace(num_layers=1,layers=[torch.nn.Identity()])
    enc=SimpleNamespace(query_position=8,sequence_length=9)
    with pytest.raises(RuntimeError,match='Expected one'):
        with additive_span_hook(adapter,enc,layer=0,positions=[2],probe={},beta=1,condition='ridge'):
            pass
    assert not adapter.layers[0]._forward_hooks
