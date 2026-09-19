"""Refit the historical running-index classifiers on discovery seeds only.

Keep the original PCA -> coordinate scaling and classifier settings. Five-fold
GroupKFold excludes an entire seed from all preprocessing and fitting. A small
30-seed reproduction check verifies the cached source against the old results.
"""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import time
import warnings

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / 'realistic/exports/run_20260731_v4_numeric_presentation_v3'
OLD = ROOT / 'realistic/reports/v4_non-thinking_causal/v4_4_extension/classification'
MODELS = [('Qwen3-8B', 'qwen', 36, [0, 8, 10]), ('Gemma4-E4B', 'gemma', 42, [0, 9, 20])]
RNG = 20260806


def read_fold_map(path):
    result = {}
    with gzip.open(path, 'rt', encoding='utf-8-sig') as h:
        for row in csv.DictReader(h):
            seed, fold = int(row['seed']), int(row['fold'])
            if seed in result:
                assert result[seed] == fold
            result[seed] = fold
    return result


def fit_oof(x, y, seeds, fold_map):
    algorithms = ['nearest_centroid', 'logistic_l2']
    predicted = {a: np.empty(len(y), dtype=int) for a in algorithms}
    folds = np.empty(len(y), dtype=int)
    fold_seeds = []
    # Reuse the original fold map: NumPy versions can break equal-size seed
    # ties differently inside GroupKFold. Panel A uses panel B's exact map.
    assignments = np.array([fold_map[int(seed)] for seed in seeds], dtype=int)
    assert sorted(np.unique(assignments).tolist()) == list(range(5))
    for fold in range(5):
        train = np.flatnonzero(assignments != fold)
        test = np.flatnonzero(assignments == fold)
        assert not set(seeds[train]) & set(seeds[test])
        assert sorted(np.unique(y[train]).tolist()) == list(range(1, 11))
        pca = PCA(n_components=32, svd_solver='randomized', random_state=RNG)
        train_z = pca.fit_transform(x[train])
        test_z = pca.transform(x[test])
        scale = StandardScaler()
        train_z = scale.fit_transform(train_z)
        test_z = scale.transform(test_z)
        classifiers = {
            'nearest_centroid': NearestCentroid(),
            'logistic_l2': LogisticRegression(C=1.0, solver='lbfgs', max_iter=2000, random_state=RNG),
        }
        for name, estimator in classifiers.items():
            estimator.fit(train_z, y[train])
            predicted[name][test] = estimator.predict(test_z)
        folds[test] = fold
        fold_seeds.append({'fold': fold, 'train': sorted(np.unique(seeds[train]).tolist()),
                           'test': sorted(np.unique(seeds[test]).tolist())})
    return predicted, folds, fold_seeds


def main():
    started = time.perf_counter()
    sources, checks, results, predictions = {}, [], [], []
    protocol_folds = None
    for model, short, layer_count, check_layers in MODELS:
        base = SOURCE / model / 'numeric/representation/capture'
        index_path = base / 'capture_index.jsonl'
        index = [json.loads(line) for line in index_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        records = sorted((r for r in index if r['design_variant'] == 'v4.4'
                          and r['answer_format'] == 'numeric' and int(r['count']) == 10),
                         key=lambda r: int(r['seed']))
        assert [int(r['seed']) for r in records] == list(range(1234, 1264))
        arrays = []
        for row in records:
            path = base / row['shard_path']
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path, allow_pickle=False) as z:
                assert z['layer_indices'].tolist() == list(range(layer_count))
                arrays.append(z['span_end'].astype(np.float32))
        # One row per seed, layer, endpoint, hidden coordinate.
        states = np.stack(arrays)
        del arrays
        assert states.shape[:3] == (30, layer_count, 10)
        seeds30 = np.repeat(np.arange(1234, 1264), 10)
        labels30 = np.tile(np.arange(1, 11), 30)
        historical_path = OLD / f'classification_prompt_{short}/answer_classifier_metrics.csv'
        with historical_path.open(encoding='utf-8-sig') as h:
            historical = {(int(r['layer']), r['algorithm']): float(r['accuracy']) for r in csv.DictReader(h)}
        old_predictions = historical_path.parent / 'answer_classifier_oof_predictions.csv.gz'
        answer_predictions = OLD / f'classification_all_{short}/answer_classifier_oof_predictions.csv.gz'
        fold_map30 = read_fold_map(old_predictions)
        fold_map20 = read_fold_map(answer_predictions)
        assert sorted(fold_map30) == list(range(1234, 1264))
        assert sorted(fold_map20) == list(range(1234, 1254))
        for path in [historical_path, old_predictions, answer_predictions]:
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for layer in check_layers:
            x = states[:, layer].reshape(300, -1)
            found, _, _ = fit_oof(x, labels30, seeds30, fold_map30)
            for method, pred in found.items():
                value = float(np.mean(pred == labels30))
                old_value = historical[(layer, method)]
                checks.append({'model': model, 'layer_zero_based': layer, 'method': method,
                               'recomputed_accuracy': value, 'historical_accuracy': old_value,
                               'absolute_difference': abs(value - old_value)})
                assert abs(value - old_value) < 1e-10, checks[-1]
        print(f'{model}: 30-seed source check passed (3 layers, 2 classifiers)', flush=True)
        selected = np.arange(20)
        seeds = np.repeat(np.arange(1234, 1254), 10)
        labels = np.tile(np.arange(1, 11), 20)
        assert all(records[j]['split'] == 'discovery' for j in selected)
        for layer in range(layer_count):
            x = states[selected, layer].reshape(200, -1)
            found, folds, fold_seeds = fit_oof(x, labels, seeds, fold_map20)
            if protocol_folds is None:
                protocol_folds = fold_seeds
            assert fold_seeds == protocol_folds
            for method, pred in found.items():
                results.append({'model_label': model, 'role': 'prompt_running', 'layer': layer,
                                'algorithm': method, 'rows': 200, 'seeds': 20,
                                'count_class_count': 10, 'pca_components': 32,
                                'accuracy': float(np.mean(pred == labels)), 'chance_accuracy': .1})
                predictions.extend({'model_label': model, 'layer': layer, 'algorithm': method,
                                    'seed': int(seeds[i]), 'fold': int(folds[i]),
                                    'gold_count': int(labels[i]), 'predicted_count': int(pred[i]),
                                    'correct': int(labels[i] == pred[i])} for i in range(len(labels)))
            if (layer + 1) % 6 == 0 or layer + 1 == layer_count:
                print(f'{model}: {layer + 1}/{layer_count} layers; elapsed {time.perf_counter()-started:.1f}s', flush=True)
        dest = OUT / f'classification_prompt_20_{short}'
        dest.mkdir(exist_ok=True)
        model_results = [r for r in results if r['model_label'] == model]
        with (dest / 'answer_classifier_metrics.csv').open('w', newline='', encoding='utf-8') as h:
            writer = csv.DictWriter(h, fieldnames=list(model_results[0]))
            writer.writeheader()
            writer.writerows(model_results)
    with gzip.open(OUT / 'running_index_20_oof_predictions.csv.gz', 'wt', newline='', encoding='utf-8') as h:
        writer = csv.DictWriter(h, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    audit = {'status': 'PASS', 'seed_range': [1234, 1253], 'seed_count': 20,
             'states_per_model_layer': 200, 'prompt_count': 10,
             'label': 'running index 1--10 at the ten needle endpoints',
             'fit': 'unwhitened PCA32, coordinate StandardScaler, classifier; training-fold-only',
             'nearest_centroid_shrinkage': None, 'logistic_C': 1,
             'random_state': RNG, 'folds': protocol_folds,
             'fold_assignment': 'exact historical answer-query 20-seed fold mapping',
             'source_sha256': sources,
             'historical_30_seed_checks': checks, 'layers_by_model': {'Qwen3-8B': 36, 'Gemma4-E4B': 42},
             'metric_rows': len(results), 'prediction_rows': len(predictions),
             'sklearn_version': sklearn.__version__, 'numpy_version': np.__version__,
             'elapsed_seconds': time.perf_counter() - started}
    (OUT / 'running_index_20_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'metrics': len(results), 'elapsed_seconds': audit['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter('error', ConvergenceWarning)
        main()
