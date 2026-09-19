"""Scientific invariants for the exploratory plateau/tail split."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import analyze_realistic_niah_v3_2_piecewise_n3 as analysis


def fixture():
    levels = (tuple(range(1,11)), (1000,2000,3000,4000,5000))
    frame = pd.DataFrame([(n,l) for n in levels[0] for l in levels[1]],columns=['N','L'])
    frame['y'] = np.where(frame.N.le(3), 0.25, 2*frame.N-5)
    return frame,levels,analysis.core.Candidate(id='N',terms=('N',))


def test_low_n_cannot_change_tail_coefficients():
    frame,levels,candidate = fixture()
    oof, fitted = analysis.predict(frame,'y',candidate,levels,tail_only=True,plateau='constant')
    np.testing.assert_allclose(oof,frame.y,atol=1e-12)
    frame.loc[frame.N.le(3),'y'] = 1e8
    changed,_ = analysis.predict(frame,'y',candidate,levels,tail_only=True,plateau='constant')
    np.testing.assert_allclose(changed[frame.N.gt(3)],oof[frame.N.gt(3)],atol=1e-12)


def test_held_out_low_n_values_do_not_leak_into_plateau():
    frame,levels,candidate = fixture()
    oof,_ = analysis.predict(frame,'y',candidate,levels,tail_only=True,plateau='constant')
    folds = analysis.core.condition_fold(frame,*levels)
    testlow = (folds==0) & frame.N.le(3).to_numpy()
    frame.loc[testlow,'y'] = 9999
    changed,_ = analysis.predict(frame,'y',candidate,levels,tail_only=True,plateau='constant')
    np.testing.assert_allclose(changed[testlow],oof[testlow],atol=1e-12)


def test_zero_plateau_and_negative_bias_are_preserved():
    frame,levels,candidate = fixture()
    frame['y'] *= -1
    oof,_ = analysis.predict(frame,'y',candidate,levels,tail_only=True,plateau='zero')
    np.testing.assert_array_equal(oof[frame.N.le(3)],0)
    np.testing.assert_allclose(oof[frame.N.gt(3)],frame.loc[frame.N.gt(3),'y'],atol=1e-12)
