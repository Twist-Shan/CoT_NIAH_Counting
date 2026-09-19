import numpy as np
import pandas as pd
import pytest
from realistic_niah_v5.cross_mode_geometry import ModeDataset
from realistic_niah_v6.own_state_representation import population, select_ncc
from realistic_niah_v6.spec import DISCOVERY_SEEDS, CONFIRMATION_SEEDS


def test_selection_ignores_confirmation_and_breaks_ties():
    rows=[dict(layer=layer,discovery_oof_ncc_balanced_accuracy=ncc,
        discovery_oof_logistic_balanced_accuracy=log,confirmation_ncc_balanced_accuracy=conf)
        for layer,ncc,log,conf in [(0,.8,.7,1),(1,.9,.8,0),(2,.9,.8,1),(3,.9,.7,1)]]
    assert select_ncc(rows)['layer']==1


def test_own_population_keeps_unique_available_states():
    rows=[dict(split=split,seed=seed,gold_count=10,occurrence=k) for split,seeds in
        [('discovery',DISCOVERY_SEEDS),('confirmation',CONFIRMATION_SEEDS)] for seed in seeds for k in range(1,11)]
    rows.append(dict(split='confirmation',seed=CONFIRMATION_SEEDS[0],gold_count=9,occurrence=1))
    d=ModeDataset('enumeration','test',pd.DataFrame(rows),{0:np.ones((len(rows),3))})
    assert population(d)['confirmation']['rows']==101
    d.metadata=pd.concat([d.metadata,d.metadata.tail(1)],ignore_index=True)
    d.states_by_layer={0:np.ones((len(d.metadata),3))}
    with pytest.raises(ValueError,match='duplicate'):
        population(d)
# Confirmation values must not change the fitted display transform.
def test_display_pca_discovery_only():
    import numpy as np
    from realistic_niah_v6.own_state_representation import fit_display_pca
    rng=np.random.default_rng(11);x=rng.normal(size=(35,12)).astype(np.float32);d=np.arange(35)<25
    scaler,pca,z=fit_display_pca(x,d)
    other=x.copy();other[~d]*=100
    scaler2,pca2,z2=fit_display_pca(other,d)
    np.testing.assert_array_equal(scaler.mean_,scaler2.mean_)
    np.testing.assert_array_equal(pca.components_,pca2.components_)
    np.testing.assert_array_equal(z[d],z2[d])
    assert not np.allclose(z[~d],z2[~d])
