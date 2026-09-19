"""Check figure source/package availability without executing artifact reads.

This static inventory catches absent imported helpers and declared editable
assets. It cannot certify dynamic sys.path behavior, data inputs or GPU stages.
Use the rendering checks in docs/VALIDATION.md for stronger, scoped evidence.
"""
import ast
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'figures'
SOURCE_ASSETS = [
    'synthetic_appendix_evidence_revision_20260908/camera_study.py',
    'synthetic_appendix_paper_20260908/camera_study.py',
    'synthetic_appendix_paper_checked_20260908/camera_study.py',
    'aurora_attention_pca_concept/main_figure_v2.drawio',
    'aurora_attention_pca_concept/main_figure_v2_settings.json',
    'aurora_attention_pca_concept/assets/v4_source_v3.drawio',
    'aurora_attention_pca_concept/assets/v4_source_v3_settings.json',
    'aurora_attention_pca_concept/main_figure_v4.drawio',
    'aurora_attention_pca_concept/main_figure_v4_settings.json',
    'aurora_attention_pca_concept/assets/fonts/LICENSE_STIX',
    'cot-reasoning/font_profile.json',
    'non-thinking/transformer_mechanism_polished.drawio',
]


def main():
    assets = json.loads((FIGURES/'paper_assets.json').read_text(encoding='utf-8'))['assets']
    problems = []
    for row in assets:
        for field in ('builder', 'editable_source'):
            if field in row and not (ROOT/row[field]).is_file():
                problems.append({'asset': row['manuscript_asset'], 'missing': row[field]})
    for relative in SOURCE_ASSETS:
        if not (FIGURES/relative).is_file():
            problems.append({'missing_source_asset': relative})
    local = {p.stem for p in FIGURES.rglob('*.py')}
    for component in ('realistic', 'synthetic'):
        local.update(p.stem for p in (ROOT/component/'scripts').rglob('*.py'))
        local.update(p.name for p in (ROOT/component/'src').iterdir() if p.is_dir())
    packages, scanned = set(), 0
    for path in FIGURES.rglob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        scanned += 1
        for node in ast.walk(tree):
            names = [n.name for n in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and not node.level and node.module else [])
            for name in names:
                top = name.split('.')[0]
                if top in sys.stdlib_module_names or top in local:
                    continue
                packages.add(top)
                if importlib.util.find_spec(top) is None:
                    problems.append({'file': path.relative_to(ROOT).as_posix(), 'missing_package': top})
    print(json.dumps(dict(status='FAIL' if problems else 'PASS', mapped_assets=len(assets),
        scanned_figure_modules=scanned, checked_source_assets=len(SOURCE_ASSETS),
        third_party_packages=sorted(packages), problems=problems,
        scope='static source/package inventory; not execution or a complete artifact dependency graph'), indent=2))
    return bool(problems)


if __name__ == '__main__':
    raise SystemExit(main())
