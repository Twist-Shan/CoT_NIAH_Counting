"""Read-only local diagnosis; uses the unchanged frozen Broad score function."""
import argparse
import ast
from collections import Counter
import json
import math
import os
from pathlib import Path
import struct
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = Path(__file__).resolve().parents[1] / 'runs/native_broad_full_span_20260909_v1'
    plain = (run / 'downloaded').resolve()
    root = Path('\\\\?\\' + str(plain)) if os.name == 'nt' else plain
    source = (root / 'src/realistic_niah_v5/capture.py').read_text(encoding='utf-8')
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_broad_span_metrics')
    namespace = {'np': np, 'math': math, 'Sequence': list}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'frozen_capture_metrics', 'exec'), namespace)
    score = namespace['_broad_span_metrics']
    bits = lambda x: struct.unpack('>Q', struct.pack('>d', x))[0]
    counts = Counter()
    violations = []
    max_absolute = 0.0
    panels = {}
    sys.path.insert(0, str(root))
    from category_target_broad import rank_rows
    for path in sorted((root / 'discovery').glob('*/*/*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        if not row['available']:
            continue
        recalculated = []
        for saved, (layer, head, masses) in zip(row['heads'], row['record_masses']):
            actual = score(masses)['score']
            assert math.isfinite(actual) and actual >= 0 and ((actual == 0) == (saved[2] == 0))
            distance = abs(bits(saved[2]) - bits(actual))
            counts[distance] += 1
            max_absolute = max(max_absolute, abs(saved[2] - actual))
            recalculated.append([layer, head, actual])
            if distance > 8:
                violations.append(dict(model=row['model'], task=row['task'], case_id=row['case_id'], layer=layer,
                                       head=head, ulp=distance, saved=saved[2], local=actual, J=len(masses)))
        panels.setdefault((row['model'], row['task']), []).append(dict(row, heads=recalculated))
    rankings = []
    for (model, task), rows in panels.items():
        bank = json.loads((root / 'banks' / f'{task}_{model}.json').read_text(encoding='utf-8'))
        ranks = rank_rows(rows)
        rankings.append(dict(model=model, task=task, head_order_exact=[r[:2] for r in ranks] == [r[:2] for r in bank['ranking']]))
    result = dict(python=sys.version, numpy=np.__version__, numpy_path=np.__file__,
                  disabled_cpu_features=os.environ.get('NPY_DISABLE_CPU_FEATURES', ''),
                  scores=sum(counts.values()), counts=dict(sorted(counts.items())), exceeds_8=len(violations),
                  max_ulp=max(counts), max_absolute_difference=max_absolute, rankings=rankings, violations=violations)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'violations'}, indent=2))


if __name__ == '__main__':
    main()
