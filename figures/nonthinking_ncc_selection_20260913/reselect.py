"""Freeze discovery-only NCC layer choices, then evaluate cached endpoint states.

Run: python -s figures/nonthinking_ncc_selection_20260913/reselect.py
No model inference. Historical input reports are immutable.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
REPO = ROOT / 'realistic'
PREVIOUS = ROOT / 'figures/nonthinking_count_readouts_20260912'
DOMAIN = REPO / 'work/domain_transfer_geometry/analysis'
MODELS = [('Qwen3-8B', 'qwen'), ('Gemma4-E4B', 'gemma')]
RNG = 20260806
RULE = 'Maximize discovery seed-grouped NCC accuracy; ties: logistic accuracy, then earlier layer. Scores rounded to 12 decimals only to remove floating-point tie noise.'

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def rows(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2), encoding='utf-8')

def write_csv(name, data):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

def choose(data):
    return max(data, key=lambda r: (round(r['ncc'], 12), round(r['logistic'], 12), -r['layer']))

def main():
    started = time.perf_counter()
    hashes = {}
    def track(path):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    all_layers, selections = [], {}
    for model, short in MODELS:
        choices = {}
        for role, path in [
            ('running_index', PREVIOUS / f'classification_prompt_20_{short}/answer_classifier_metrics.csv'),
            ('answer_query_scan', REPO / f'reports/v4_non-thinking_causal/v4_4_extension/classification/classification_all_{short}/answer_classifier_metrics.csv')]:
            raw = rows(track(path))
            lookup = {(int(r['layer']), r['algorithm']): float(r['accuracy']) for r in raw}
            layer_rows = [dict(model=model, role=role, layer=l, ncc=lookup[l, 'nearest_centroid'],
                               logistic=lookup[l, 'logistic_l2']) for l in sorted({k[0] for k in lookup})]
            all_layers.extend(layer_rows)
            choices[role] = choose(layer_rows)
        raw = rows(track(DOMAIN / 'layer_selection_sweep.csv'))
        layer_rows = [dict(model=model, role='domain_transfer', layer=int(r['layer']),
                          ncc=float(r['cv_ncc_balanced_accuracy']),
                          logistic=float(r['cv_logistic_balanced_accuracy']))
                      for r in raw if r['model_label'] == model and r['mode'] == 'non_thinking']
        all_layers.extend(layer_rows)
        choices['domain_transfer'] = choose(layer_rows)
        selections[model] = choices
    # Persist every choice BEFORE loading any confirmation states or metrics.
    save('selection.json', {'rule': RULE, 'discovery_seeds': list(range(1234, 1254)),
                           'indexing': 'zero-based layer field; paper labels add one',
                           'selected': selections, 'selection_source_sha256': hashes.copy()})
    write_csv('discovery_layer_sweep.csv', all_layers)
    print(json.dumps(selections), flush=True)
    helper = module('refit', PREVIOUS / 'refit_running_index_20_seeds.py')
    metrics, predictions, coordinates, checks = [], [], [], []
    for model, short in MODELS:
        layer = selections[model]['running_index']['layer']
        base = helper.SOURCE / model / 'numeric/representation/capture'
        index = [json.loads(s) for s in track(base / 'capture_index.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        records = sorted((r for r in index if r['design_variant'] == 'v4.4' and r['answer_format'] == 'numeric' and int(r['count']) == 10), key=lambda r: int(r['seed']))
        assert [int(r['seed']) for r in records] == list(range(1234, 1264))
        arrays = []
        for r in records:
            with np.load(track(base / r['shard_path']), allow_pickle=False) as z:
                axis = z['layer_indices'].tolist().index(layer)
                arrays.append(z['span_end'][axis].astype(np.float32))
        x = np.stack(arrays).reshape(300, -1)
        y = np.tile(np.arange(1, 11), 30)
        seeds = np.repeat(np.arange(1234, 1264), 10)
        assert all(r['split'] == 'discovery' for r in records[:20])
        assert all(r['split'] == 'confirmation' for r in records[20:])
        fold_path = helper.OLD / f'classification_all_{short}/answer_classifier_oof_predictions.csv.gz'
        found, _, _ = helper.fit_oof(x[:200], y[:200], seeds[:200], helper.read_fold_map(track(fold_path)))
        for name, key in [('nearest_centroid', 'ncc'), ('logistic_l2', 'logistic')]:
            actual = float(np.mean(found[name] == y[:200]))
            assert abs(actual - selections[model]['running_index'][key]) < 1e-10
            checks.append(dict(model=model, method=name, reproduced_discovery_accuracy=actual))
        pca = PCA(n_components=32, svd_solver='randomized', random_state=RNG)
        train = pca.fit_transform(x[:200])
        test = pca.transform(x[200:])
        scaler = StandardScaler().fit(train)
        a, b = scaler.transform(train), scaler.transform(test)
        continuous = Ridge(alpha=1).fit(a, y[:200]).predict(b)
        classified = NearestCentroid().fit(a, y[:200]).predict(b)
        centroids = np.stack([x[:200][y[:200] == k].mean(0) for k in range(1, 11)]).astype(np.float64)
        centered = centroids - centroids.mean(0)
        sv = np.linalg.svd(centered, compute_uv=False)
        centroid_capture = float(np.square(sv[:3]).sum() / np.square(sv).sum())
        projected_capture = float(np.square(centered @ pca.components_[:3].T).sum() / np.square(centered).sum())
        metrics.append(dict(model=model, layer_one_based=layer+1,
                            discovery_ncc=selections[model]['running_index']['ncc'],
                            confirmation_ncc=float(np.mean(classified == y[200:])),
                            confirmation_ridge_r2=float(r2_score(y[200:], continuous)),
                            centroid_pca3_variance_capture=centroid_capture,
                            state_pca3_centroid_capture=projected_capture,
                            state_pca3_variance_capture=float(pca.explained_variance_ratio_[:3].sum())))
        for i in range(100):
            predictions.append(dict(model=model, layer_one_based=layer+1, seed=int(seeds[200+i]),
                                    running_index=int(y[200+i]), ridge_prediction=float(continuous[i]), ncc_prediction=int(classified[i])))
        for i, z in enumerate(np.vstack([train, test])):
            coordinates.append(dict(model=model, layer_one_based=layer+1, seed=int(seeds[i]),
                                    split='discovery' if i < 200 else 'confirmation', running_index=int(y[i]),
                                    pc1=float(z[0]), pc2=float(z[1]), pc3=float(z[2])))
        print(json.dumps(metrics[-1]), flush=True)
    write_csv('running_index_metrics.csv', metrics)
    write_csv('running_index_confirmation_predictions.csv', predictions)
    write_csv('running_index_pca_coordinates.csv', coordinates)
    # Reuse the domain-analysis implementation, retaining its preprocessing and folds.
    sys.path.insert(0, str(REPO / 'src'))
    from realistic_niah_v5 import domain_transfer_geometry as domain
    payload = json.loads(track(DOMAIN / 'report_payload.json').read_text(encoding='utf-8'))
    for model, short in MODELS:
        selected = selections[model]['domain_transfer']['layer']
        previous = payload['models'][model]['non_thinking']
        if previous['selected_layer'] != selected:
            city_index = REPO / f'work/nonthinking_v44_geometry_300_150_136_166_78/{model}/numeric/representation/answer_query_all_layers_v1/capture_index.jsonl'
            transfer_index = REPO / f'work/domain_transfer_geometry/full/nonthinking/{model}/capture_index.jsonl'
            city = domain.load_city_answer_endpoints(track(city_index), mode='non_thinking')
            transfer = domain.load_transfer_answer_endpoints(track(transfer_index), mode='non_thinking')
            panel = domain.combine_city_and_transfer(domain.subset_by_seeds(city, domain.CONFIRMATION_SEEDS), transfer)
            train_mask = city.metadata['seed'].isin(domain.DISCOVERY_SEEDS).to_numpy()
            cy = city.metadata['gold_count'].to_numpy(dtype=int)
            py = panel.metadata['gold_count'].to_numpy(dtype=int)
            # Reproduce old-layer confirmation scores before replacing them.
            old_layer = previous['selected_layer']
            for topic in ['city', 'flower', 'animal']:
                mask = (panel.metadata['entity_domain'] == topic).to_numpy()
                actual = domain._probe_scores(city.states_by_layer[old_layer][train_mask], cy[train_mask],
                            panel.states_by_layer[old_layer][mask], py[mask], n_components=16)
                expected = previous['metrics']['count_by_evaluation_domain'][topic]
                assert all(abs(actual[k] - expected[k]) < 1e-10 for k in actual), (topic, actual, expected)
                checks.append(dict(model=model, role='domain_transfer_old_layer', layer=old_layer, domain=topic, reproduced_scores=actual))
            fold_scores = []
            cs = city.metadata['seed'].to_numpy(dtype=int)
            for held in np.array_split(np.array(domain.DISCOVERY_SEEDS), 5):
                test_mask = train_mask & np.isin(cs, held)
                fit_mask = train_mask & ~test_mask
                fold_scores.append(domain._probe_scores(city.states_by_layer[selected][fit_mask], cy[fit_mask],
                                   city.states_by_layer[selected][test_mask], cy[test_mask], n_components=16))
            for score, key in [('ncc_balanced_accuracy', 'ncc'), ('logistic_balanced_accuracy', 'logistic')]:
                actual = float(np.mean([r[score] for r in fold_scores]))
                assert abs(actual - selections[model]['domain_transfer'][key]) < 1e-10
                checks.append(dict(model=model, role='domain_transfer_selected_cv', metric=score, reproduced_accuracy=actual))
            previous['metrics'] = domain.evaluate_frozen_layer(panel, training_dataset=city, layer=selected,
                                selection_seeds=domain.DISCOVERY_SEEDS, evaluation_seeds=domain.CONFIRMATION_SEEDS)
            previous['visualization'] = domain.city_anchored_pca3(panel, training_dataset=city, layer=selected, selection_seeds=domain.DISCOVERY_SEEDS)
            del city, transfer, panel
        previous['selected_layer'] = selected
        previous['selection_rule'] = RULE
        print(model, 'domain L', selected+1, previous['metrics']['count_by_evaluation_domain'], flush=True)
    save('domain_report_payload.json', payload)
    save('audit.json', dict(status='PASS', rule=RULE, elapsed_seconds=time.perf_counter()-started,
                           source_sha256=hashes, selected_discovery_reproduction=checks,
                           random_state=RNG, numpy=np.__version__, sklearn=sklearn.__version__,
                           running_index_fit='Discovery-fitted unwhitened PCA32, StandardScaler, Ridge(alpha=1) / NCC(shrinkage=None)',
                           centroid_variance='PCA of ten discovery centroids in original hidden space; distinct from state PCA3',
                           limitation='Reselection requested after earlier results were inspected; reused confirmation seeds are not a new independent replication.'))

if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
