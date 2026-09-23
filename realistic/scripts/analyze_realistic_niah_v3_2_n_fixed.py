"""V3.2-only N fixed effects, with optional shared linear/log length slope."""
from pathlib import Path
import sys
import json
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from scripts import analyze_realistic_niah_v3_2_empirical_laws as core
from scripts import analyze_realistic_niah_v3_2_count_error_extension as ext
from scripts.analyze_realistic_niah_v3_3_regression_scan import (
    fit_n_fixed_accuracy_cv, fit_n_fixed_continuous_cv,
)

ANALYSIS = ROOT / 'outputs/realistic_niah_v3_1_20260819_formal/analysis'
OUT = ANALYSIS / 'v3_2_n_fixed_shared_length_20260909'
SCHEME = 'v3_2_5fold_held_condition_cv'


def main():
    start = time.perf_counter()
    config = ext.load_parent_frozen_config(core.DEFAULT_CONFIG, core.DEFAULT_FREEZE)
    source = Path(core.DEFAULT_INPUT)
    digest = core.file_sha256(source)
    if digest != 'c72db941c112b3bacb3295b304931e0dbbe2c16af87e8fa4af3bfd0ab172c819':
        raise ValueError('V3.2 frozen request input changed')
    requests = pd.read_csv(source)
    core.validate_requests(requests, config)
    cells = ext.build_count_error_cells(requests)
    levels = (tuple(config['immutable_input']['N_levels']), tuple(config['immutable_input']['L_levels']))
    assert len(requests) == 161280 and len(cells) == 5376
    assert requests.comparison_slot.nunique() == 12
    eligible = cells.loc[cells.bias_law_eligible.astype(bool)].copy()
    rows, coefficients, failures = [], [], []
    (OUT / 'tables').mkdir(parents=True, exist_ok=True)
    for (slot, mode), frame in requests.groupby(['comparison_slot', 'prompt_mode']):
        block = eligible.loc[eligible.comparison_slot.eq(slot) & eligible.prompt_mode.eq(mode)]
        for outcome in (core.HEADLINE_ACCURACY, ext.MAE_FAMILY, core.BIAS_FAMILY):
            for term in (None, 'L_k', 'logL'):
                tags = dict(comparison_slot=slot, prompt_mode=mode, evaluation_scheme=SCHEME)
                try:
                    if outcome == core.HEADLINE_ACCURACY:
                        row, coefs = fit_n_fixed_accuracy_cv(frame, term, *levels)
                    else:
                        row, coefs = fit_n_fixed_continuous_cv(block, outcome, term, *levels)
                    rows.append(dict(row, **tags))
                    coefficients.extend(dict(c, **tags) for c in coefs)
                except Exception as exc:
                    failures.append(dict(tags, outcome_family=outcome, length_term=term, error=repr(exc)))
        print(f'{slot} / {mode}: {len(rows)} fits, {len(failures)} failures', flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'tables/metrics.csv', index=False)
    pd.DataFrame(coefficients).to_csv(OUT / 'tables/coefficients.csv', index=False)
    manifest = dict(schema='v3_2_n_fixed_shared_length_v1', input=str(source.relative_to(ROOT)),
        input_sha256=digest, requests=len(requests), cells=len(cells), eligible_cells=len(eligible),
        N_levels=levels[0], L_levels=levels[1], metric_rows=len(rows), failures=failures,
        evaluation_scheme=SCHEME, elapsed_seconds=time.perf_counter()-start,
        note='Full N indicators contain N-specific intercepts; no additional global intercept. No N-by-L interaction. No V3.3 observations.')
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    if failures or len(rows) != 432:
        raise RuntimeError(f'Incomplete analysis: {len(rows)} / 432 fits; inspect manifest')


if __name__ == '__main__':
    main()
