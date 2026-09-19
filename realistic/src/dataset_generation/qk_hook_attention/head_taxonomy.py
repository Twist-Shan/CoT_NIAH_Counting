'''Classify targeted-retrieval, induction, and successor-candidate heads.'''
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

from .analyze_qk_qwen3 import (
    compute_full_attention_matrix,
    load_cache_metadata,
    load_tensor,
    reconstruct_single_head_qk,
)

FAMILIES = ('targeted_retrieval', 'induction', 'successor')


@dataclass(frozen=True)
class FamilyThreshold:
    min_lift: float = 2.0
    min_mass: float = 0.10
    min_queries: int = 2


@dataclass(frozen=True)
class HeadTaxonomyConfig:
    targeted_retrieval: FamilyThreshold = FamilyThreshold()
    induction: FamilyThreshold = FamilyThreshold()
    successor: FamilyThreshold = FamilyThreshold()
    compute_dtype: str = 'float32'
    allow_multi_label: bool = True
    max_full_attention_tokens: int | None = 4096

    @classmethod
    def from_dict(cls, raw: Mapping | None) -> 'HeadTaxonomyConfig':
        raw = dict(raw or {})

        def get(name):
            value = raw.get(name, {})
            return value if isinstance(value, FamilyThreshold) else FamilyThreshold(**dict(value))

        return cls(
            targeted_retrieval=get('targeted_retrieval'),
            induction=get('induction'),
            successor=get('successor'),
            compute_dtype=str(raw.get('compute_dtype', 'float32')),
            allow_multi_label=bool(raw.get('allow_multi_label', True)),
            max_full_attention_tokens=raw.get('max_full_attention_tokens', 4096),
        )


def _attention(value) -> torch.Tensor:
    value = torch.as_tensor(value, dtype=torch.float32)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f'attention must be square [T,T], got {tuple(value.shape)}')
    if not torch.isfinite(value).all() or (value < -1e-7).any():
        raise ValueError('attention must contain finite non-negative probabilities')
    return value


def _positions(values: Iterable[int] | None, length: int) -> list[int]:
    result = list(range(length)) if values is None else sorted({int(x) for x in values})
    if any(x < 0 or x >= length for x in result):
        raise ValueError(f'query positions must lie in [0, {length})')
    return result


def _score(attention, candidates: Mapping[int, Sequence[int]], metric: str) -> dict:
    '''Compare candidate attention mass with a causal uniform baseline.'''
    attention = _attention(attention)
    masses, baselines, evidence = [], [], []
    for query, raw_keys in sorted(candidates.items()):
        query = int(query)
        if query < 0 or query >= attention.shape[0]:
            raise ValueError(f'bad query position {query}')
        keys = sorted({int(k) for k in raw_keys if 0 <= int(k) <= query})
        if not keys:
            continue
        mass = float(attention[query, keys].sum())
        baseline = len(keys) / float(query + 1)
        masses.append(mass)
        baselines.append(baseline)
        evidence.append(
            {
                'query_position': query,
                'key_positions': keys,
                'attention_mass': mass,
                'uniform_baseline': baseline,
                'lift': mass / baseline,
            }
        )
    if not masses:
        return dict(
            metric=metric, mean_mass=0.0, uniform_baseline=0.0, lift=0.0,
            n_queries=0, n_candidate_edges=0, evidence=[]
        )
    mass, baseline = sum(masses) / len(masses), sum(baselines) / len(baselines)
    return dict(
        metric=metric,
        mean_mass=mass,
        uniform_baseline=baseline,
        lift=mass / baseline if baseline else math.nan,
        n_queries=len(masses),
        n_candidate_edges=sum(len(x['key_positions']) for x in evidence),
        evidence=evidence,
    )


def score_targeted_retrieval(attention, *, target_spans, query_positions) -> dict:
    attention = _attention(attention)
    mask = torch.zeros(attention.shape[0], dtype=torch.bool)
    for span in target_spans:
        if len(span) != 2:
            raise ValueError(f'target span must be [start,end), got {span!r}')
        start, end = map(int, span)
        if start < 0 or end > attention.shape[0] or end <= start:
            raise ValueError(f'invalid target span {span!r}')
        mask[start:end] = True
    candidates = {
        query: torch.where(mask[: query + 1])[0].tolist()
        for query in _positions(query_positions, attention.shape[0])
    }
    return _score(attention, candidates, 'target_span_attention')


def induction_candidate_edges(token_ids, query_positions=None) -> dict[int, list[int]]:
    '''Find q to k edges where token[k-1] equals token[q].'''
    ids = [int(x) for x in token_ids]
    return {
        query: keys
        for query in _positions(query_positions, len(ids))
        if (keys := [key for key in range(1, query) if ids[key - 1] == ids[query]])
    }


def score_induction(attention, *, token_ids, query_positions=None) -> dict:
    if len(token_ids) != attention.shape[0]:
        raise ValueError('token_ids length must match attention')
    return _score(
        attention,
        induction_candidate_edges(token_ids, query_positions),
        'induction_match_attention',
    )


def successor_candidate_edges(
    token_ids,
    *,
    successor_token_map=None,
    explicit_pairs=None,
    query_positions=None,
) -> dict[int, list[int]]:
    '''Build predecessor-attention edges for a Q/K successor proxy.

    successor_token_map maps predecessor token IDs to successor token IDs.
    explicit_pairs contains [query_position, predecessor_key_position] pairs and
    is safer for labels that tokenize into multiple pieces.
    '''
    ids = [int(x) for x in token_ids]
    allowed = set(_positions(query_positions, len(ids)))
    result: dict[int, list[int]] = {}
    normalized = {}
    for predecessor, successors in (successor_token_map or {}).items():
        values = successors if isinstance(successors, (list, tuple, set)) else [successors]
        normalized[int(predecessor)] = {int(x) for x in values}
    for query in sorted(allowed):
        keys = [
            key
            for key in range(query)
            if ids[key] in normalized and ids[query] in normalized[ids[key]]
        ]
        if keys:
            result[query] = keys
    for pair in explicit_pairs or []:
        if len(pair) != 2:
            raise ValueError(f'successor pair must be [query,key], got {pair!r}')
        query, key = map(int, pair)
        if query in allowed:
            if key < 0 or key >= query:
                raise ValueError(f'successor pair requires 0 <= key < query: {pair!r}')
            result.setdefault(query, []).append(key)
    return {query: sorted(set(keys)) for query, keys in result.items()}


def score_successor(
    attention,
    *,
    token_ids,
    successor_token_map=None,
    explicit_pairs=None,
    query_positions=None,
) -> dict:
    if len(token_ids) != attention.shape[0]:
        raise ValueError('token_ids length must match attention')
    result = _score(
        attention,
        successor_candidate_edges(
            token_ids,
            successor_token_map=successor_token_map,
            explicit_pairs=explicit_pairs,
            query_positions=query_positions,
        ),
        'successor_predecessor_attention',
    )
    result['evidence_level'] = 'qk_proxy_requires_ov_or_ablation_confirmation'
    return result


def _passes(score, threshold) -> bool:
    return (
        score['n_queries'] >= threshold.min_queries
        and score['mean_mass'] >= threshold.min_mass
        and score['lift'] >= threshold.min_lift
    )


def classify_attention_head(
    attention,
    *,
    token_ids,
    target_spans=(),
    retrieval_query_positions=(),
    induction_query_positions=None,
    successor_query_positions=None,
    successor_token_map=None,
    successor_pairs=None,
    config=None,
) -> dict:
    '''Score and optionally multi-label one attention head.'''
    config = config or HeadTaxonomyConfig()
    scores = {
        'targeted_retrieval': score_targeted_retrieval(
            attention,
            target_spans=target_spans,
            query_positions=retrieval_query_positions,
        ),
        'induction': score_induction(
            attention, token_ids=token_ids, query_positions=induction_query_positions
        ),
        'successor': score_successor(
            attention,
            token_ids=token_ids,
            successor_token_map=successor_token_map,
            explicit_pairs=successor_pairs,
            query_positions=successor_query_positions,
        ),
    }
    labels = [name for name in FAMILIES if _passes(scores[name], getattr(config, name))]
    if labels and not config.allow_multi_label:
        labels = [max(labels, key=lambda name: scores[name]['lift'])]
    primary = max(labels, key=lambda name: scores[name]['lift']) if labels else 'unclassified'
    return {
        'labels': labels,
        'primary_family': primary,
        'scores': scores,
        'successor_interpretation': 'candidate_only_until_ov_or_causal_validation',
    }


def _dtype(name):
    values = {
        'float32': torch.float32,
        'fp32': torch.float32,
        'float16': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
    }
    if name not in values:
        raise ValueError(f'unsupported compute_dtype {name!r}')
    return values[name]


def scan_qk_cache(
    *,
    cache_dir,
    output_dir,
    analysis_spec,
    layers=None,
    heads=None,
    config=None,
    device='cpu',
):
    '''Classify layer/head pairs from an existing Qwen3 Q/K cache.'''
    import pandas as pd

    started = time.perf_counter()
    cache_dir, output_dir = Path(cache_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config or HeadTaxonomyConfig()
    metadata = load_cache_metadata(cache_dir)
    model_config = metadata['model_config']
    layers = [int(x) for x in (layers or metadata.get('target_layers') or [])]
    if not layers:
        raise ValueError('no layers supplied and metadata has no target_layers')
    num_heads = int(model_config['num_attention_heads'])
    heads = [int(x) for x in (heads if heads is not None else range(num_heads))]
    if any(x < 0 or x >= num_heads for x in heads):
        raise ValueError(f'heads must lie in [0, {num_heads})')
    ids = load_tensor(cache_dir / 'input_ids.pt')
    token_ids = (ids[0] if ids.ndim == 2 else ids).tolist()
    if (
        config.max_full_attention_tokens is not None
        and len(token_ids) > int(config.max_full_attention_tokens)
    ):
        raise ValueError(
            f'sequence has {len(token_ids)} tokens, exceeding the configured '
            f'max_full_attention_tokens={config.max_full_attention_tokens}; '
            'use a shorter diagnostic prompt or explicitly raise the limit'
        )
    mask_path = cache_dir / 'attention_mask.pt'
    key_padding_mask = None
    if mask_path.exists():
        mask = load_tensor(mask_path)
        key_padding_mask = (mask[0] if mask.ndim == 2 else mask).bool()

    rows, evidence = [], {}
    device = torch.device(device)
    for layer in layers:
        for head in heads:
            q, k, info = reconstruct_single_head_qk(
                cache_dir=cache_dir,
                layer=layer,
                head=head,
                device=device,
                compute_dtype=_dtype(config.compute_dtype),
            )
            attention = compute_full_attention_matrix(
                q=q, k=k, key_padding_mask=key_padding_mask, scaling=info['scaling']
            )
            result = classify_attention_head(
                attention,
                token_ids=token_ids,
                target_spans=analysis_spec.get('target_spans', []),
                retrieval_query_positions=analysis_spec.get('retrieval_query_positions', []),
                induction_query_positions=analysis_spec.get('induction_query_positions'),
                successor_query_positions=analysis_spec.get('successor_query_positions'),
                successor_token_map=analysis_spec.get('successor_token_map'),
                successor_pairs=analysis_spec.get('successor_pairs'),
                config=config,
            )
            row = dict(
                layer=layer,
                head=head,
                kv_head=info['kv_head'],
                primary_family=result['primary_family'],
                labels='|'.join(result['labels']),
            )
            for family in FAMILIES:
                score = result['scores'][family]
                for field in ('mean_mass', 'uniform_baseline', 'lift', 'n_queries'):
                    row[f'{family}_{field}'] = score[field]
            rows.append(row)
            evidence[f'layer_{layer:02d}_head_{head:02d}'] = result
            del q, k, attention

    frame = pd.DataFrame(rows).sort_values(['layer', 'head']).reset_index(drop=True)
    frame.to_csv(output_dir / 'head_scores.csv', index=False)
    frame[frame.primary_family != 'unclassified'].to_csv(
        output_dir / 'head_labels.csv', index=False
    )
    (output_dir / 'head_evidence.json').write_text(
        json.dumps(evidence, indent=2), encoding='utf-8'
    )
    run_metadata = {
        'schema_version': 'head_taxonomy_v1',
        'cache_dir': str(cache_dir.resolve()),
        'layers': layers,
        'heads': heads,
        'analysis_spec': dict(analysis_spec),
        'config': asdict(config),
        'elapsed_seconds': time.perf_counter() - started,
        'scientific_caveat': (
            'successor is a Q/K proxy; confirm with OV/logit attribution or causal ablation'
        ),
    }
    (output_dir / 'run_metadata.json').write_text(
        json.dumps(run_metadata, indent=2), encoding='utf-8'
    )
    return frame
