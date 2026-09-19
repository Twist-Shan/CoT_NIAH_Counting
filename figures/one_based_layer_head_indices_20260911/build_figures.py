"""Stage one-based publication labels; retain all frozen measurement indices."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
LABELS = WORK / 'figures/synthetic_appendix_label_revision_20260908'
BUILDERS = {
    'head_scores': LABELS / 'build_retrieval_head_scores.py',
    'dynamics': LABELS / 'build_joint_dynamics.py',
    'additional': WORK / 'figures/additional_tasks_aurora/build_figures.py',
    'nonthinking_appendix': WORK / 'figures/nonthinking_appendix_style_20260911/build_figures.py',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', choices=BUILDERS)
    target = parser.parse_args().target
    path = BUILDERS[target]
    out = HERE / target
    out.mkdir(exist_ok=True)
    (out / 'plot_data').mkdir(exist_ok=True)
    (out / 'paper').mkdir(exist_ok=True)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('one_based_figure', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = out
    if hasattr(module, 'PAPER'):
        module.PAPER = (out / 'paper' / module.PAPER.name
                        if module.PAPER.suffix == '.pdf' else out / 'paper')
    if hasattr(module, 'previous'):
        module.previous.OUT = out
        module.previous.PAPER = out / 'paper'
    if target == 'additional':
        scales = module.score_figures()
        (out / 'manifest.json').write_text(json.dumps({
            'source_hashes': module.SOURCES, 'figures': module.FIGURES,
            'checks': module.CHECKS, 'score_scales': scales,
        }, indent=2), encoding='utf-8')
    else:
        module.main()
    inputs = {}
    for obj in [module, getattr(module, 'original', None), getattr(module, 'previous', None)]:
        if obj is not None:
            for name in ['SOURCES', 'source_sha256', 'INPUTS']:
                values = getattr(obj, name, None)
                if isinstance(values, dict):
                    inputs.update(values)
    if target == 'head_scores':
        inputs[str(module.SOURCE.relative_to(WORK))] = hashlib.sha256(module.SOURCE.read_bytes()).hexdigest()
    for relative, expected in inputs.items():
        source = Path(relative)
        source = source if source.is_absolute() else WORK / source
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, source
    (out / 'one_based_manifest.json').write_text(json.dumps({
        'target': target, 'builder': str(path.relative_to(WORK)),
        'display_convention': 'One-based Transformer blocks and heads.',
        'source_conventions': {'realistic': 'zero-based blocks and heads',
                               'synthetic': 'one-based blocks, zero-based heads'},
        'input_hashes_verified': inputs, 'raw_data_modified': False,
        'installed_in_paper': False,
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
