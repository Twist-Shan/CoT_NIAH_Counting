import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.analyze_realistic_niah_v3_3_regression_scan import n_fixed_design

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/realistic_niah_v3_1_20260819_formal/analysis/v3_2_n_fixed_shared_length_20260909'


def test_shared_slope_recovers_arbitrary_n_intercepts():
    frame = pd.DataFrame([(n, l) for n in (1, 2, 3) for l in (1., 3., 8.)], columns=['N', 'L_k'])
    x, names = n_fixed_design(frame, (1, 2, 3), 'L_k')
    y = frame.N.map({1: 7., 2: -3., 3: 11.}) + 2.5 * frame.L_k
    assert x.shape == (9, 4) and np.linalg.matrix_rank(x) == 4
    assert np.allclose(np.linalg.lstsq(x, y, rcond=None)[0], [7, -3, 11, 2.5])


def test_saved_results_use_complete_v3_2_scope():
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['requests'] == 161280 and max(manifest['L_levels']) == 20000
    assert not manifest['failures'] and manifest['metric_rows'] == 432
    metrics = pd.read_csv(OUT / 'tables/metrics.csv', keep_default_na=False)
    assert metrics.comparison_slot.nunique() == 12 and metrics.prompt_mode.nunique() == 4
    assert set(metrics.evaluation_scheme) == {'v3_2_5fold_held_condition_cv'}
    assert set(metrics.n_parameters) == {14, 15}
