from types import SimpleNamespace
import pytest
torch = pytest.importorskip('torch')
from realistic_niah_v5.free_carrier_recovery import match_vector_norms, scheduled_carrier_clamp


def test_cached_positions_are_absolute_and_hooks_removed():
    layer = torch.nn.Identity()
    adapter = SimpleNamespace(layers=[layer])
    source = torch.tensor([[3.,4.],[6.,8.]])
    with scheduled_carrier_clamp(adapter,prefix_length=3,positions=[3,4],replacements={0:source}) as audit:
        x=torch.zeros(1,3,2)
        assert torch.equal(layer(x),x)
        assert torch.equal(layer(torch.zeros(1,1,2))[0,0],source[0])
        assert torch.equal(layer(torch.zeros(1,1,2))[0,0],source[1])
        assert torch.equal(layer(torch.zeros(1,1,2)),torch.zeros(1,1,2))
        assert not x.any()
    assert audit['applications']=={0:[3,4]}
    assert audit['missing_positions']=={0:[]}
    assert not layer._forward_hooks


def test_early_stop_records_unvisited_positions():
    adapter=SimpleNamespace(layers=[torch.nn.Identity()])
    with scheduled_carrier_clamp(adapter,prefix_length=3,positions=[3],replacements={0:torch.ones(1,2)}) as audit:
        adapter.layers[0](torch.zeros(1,3,2))
    assert audit['missing_positions']=={0:[3]}


def test_contract_error_cleans_hooks():
    layer=torch.nn.Identity()
    with pytest.raises(RuntimeError,match='contract'):
        with scheduled_carrier_clamp(SimpleNamespace(layers=[layer]),prefix_length=3,positions=[3],replacements={0:torch.ones(1,2)}):
            layer(torch.zeros(1,2,2))
    assert not layer._forward_hooks


def test_matched_control_preserves_direction_and_matches_norm():
    control=torch.tensor([[30.,40.],[4.,3.]])
    reference=torch.tensor([[0.,2.],[0.,3.]])
    matched=match_vector_norms(control,reference)
    assert torch.allclose(matched.norm(dim=-1),reference.norm(dim=-1))
    assert torch.allclose(torch.nn.functional.normalize(matched,dim=-1),torch.nn.functional.normalize(control,dim=-1))
    with pytest.raises(ValueError,match='zero'):
        match_vector_norms(torch.zeros_like(control),reference)
