"""Freeze discovery NCC choices and reuse archived all-layer enumeration data."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import shutil
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
ORIGINAL = ROOT/'figures/enumeration'
DEST = OUT/'enumeration'
DEST.mkdir(exist_ok=True)
(DEST/'data').mkdir(exist_ok=True)
METRICS = ROOT/'realistic/work/v6_report_remote/native_aligned_representation'
sys.path.insert(0, str(ORIGINAL))
selected = {}
hashes = {}
for endpoint in ['running_index', 'final_count']:
    path = METRICS/f'{endpoint}_candidate_metrics.csv'
    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = list(csv.DictReader(path.open(encoding='utf-8')))
    chosen = []
    for mode in ['enumeration_index', 'enumeration_bullet']:
        for model in ['Qwen3-8B', 'Gemma4-E4B']:
            pool = [r for r in rows if r['prompt_mode'] == mode and r['model_label'] == model]
            row = max(pool, key=lambda r: (round(float(r['discovery_oof_ncc_balanced_accuracy']),12),
                                          round(float(r['discovery_oof_logistic_balanced_accuracy']),12), -int(r['layer'])))
            chosen.append(row)
    selected[endpoint] = chosen
# Freeze the choices before looking at confirmation projections.
(DEST/'selection.json').write_text(json.dumps({'rule': 'discovery NCC; ties logistic then earlier layer', 'source_sha256':hashes, 'selected':selected},indent=2))
for name in ['enumeration_representations','enumeration_pca_index','enumeration_pca_bullet']:
    backup = OUT/'before'/f'{name}.pdf'
    if not backup.exists():
        shutil.copy2(ROOT/'runs/paper_figures/figures'/f'{name}.pdf', backup)

# Reuse only the representation panel from the original renderer, preserving
# its causal figures and immutable source reports.
source = (ORIGINAL/'build_figures.py').read_text(encoding='utf-8')
namespace = {'__file__':str(ORIGINAL/'build_figures.py')}
exec(compile(source.split('# Figure 1:')[0], str(ORIGINAL/'build_figures.py'), 'exec'), namespace)
namespace.update(OUT=DEST, DATA=DEST/'data', ncc_selected=selected)
section = source[source.index('# Marker layers'):source.index('# Preserve exact plotted values')]
section = section.replace('    for column_index, mode in enumerate(MODES):','    selected_rows = ncc_selected[endpoint]\n    for column_index, mode in enumerate(MODES):')
section = section.replace('metric == "logistic"', 'metric == "ncc"')
section = section.replace('[("logistic", "-"), ("ncc", (0, (3.5, 2.2)))]','[("ncc", "-"), ("logistic", (0, (3.5, 2.2)))]')
section = section.replace('["Logistic", "NCC", "Discovery-selected layer", "Chance (10%)"]','["NCC", "Logistic", "NCC-selected layer", "Chance (10%)"]')
exec(compile(section, 'enumeration_ncc_readouts', 'exec'), namespace)
(DEST/'readout_manifest.json').write_text(json.dumps({'selection':selected,'layout_checks':namespace['CHECKS'],'plotted_rows':namespace['ROWS']},indent=2))

spec = importlib.util.spec_from_file_location('enum_pca', ORIGINAL/'build_pca.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
archived_source_data = module.source_data
def ncc_source_data():
    payload, old_selected, manifest, alignment = archived_source_data()
    for endpoint, short in [('running_index','running'),('final_count','final')]:
        for row in selected[endpoint]:
            key = row['prompt_mode']+'|'+row['model_label']
            layer = int(row['layer'])
            assert str(layer) in payload[short][key]['layers']
            payload[short][key]['default_layer'] = layer
            old_selected[(short,key)] = row
    return payload, old_selected, manifest, alignment
module.source_data = ncc_source_data
module.OUT = DEST
module.DATA = DEST/'data'
# The renderer records the original shared style file as provenance.
read_original = module.read
module.read = lambda p: read_original(ORIGINAL/'appendix_style.py' if p == DEST/'appendix_style.py' else p)
module.main()
path = DEST/'pca_manifest.json'
manifest = json.loads(path.read_text())
manifest.update(layer_reselected=True, selection_file='selection.json', selection_rule='discovery NCC; ties logistic then earlier layer')
path.write_text(json.dumps(manifest,indent=2))
for endpoint, rows in selected.items():
    for row in rows:
        print(endpoint, row['prompt_mode'], row['model_label'], 'L'+str(int(row['layer'])+1), 'NCC', float(row['confirmation_ncc_balanced_accuracy'])*100)
