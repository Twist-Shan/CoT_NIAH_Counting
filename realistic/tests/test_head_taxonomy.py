import pytest
import torch

from dataset_generation.qk_hook_attention.head_taxonomy import (
    FamilyThreshold,
    HeadTaxonomyConfig,
    classify_attention_head,
    induction_candidate_edges,
    score_induction,
    score_successor,
    score_targeted_retrieval,
    successor_candidate_edges,
)


def causal_uniform(length):
    matrix = torch.zeros(length, length)
    for query in range(length):
        matrix[query, : query + 1] = 1.0 / (query + 1)
    return matrix


def peaked(length, edges):
    matrix = causal_uniform(length) * 0.01
    for query, keys in edges.items():
        matrix[query].zero_()
        matrix[query, keys] = 1.0 / len(keys)
    return matrix


def test_targeted_retrieval_scores_mass_and_uniform_lift():
    attention = peaked(8, {6: [1, 2], 7: [1, 2]})
    score = score_targeted_retrieval(
        attention, target_spans=[[1, 3]], query_positions=[6, 7]
    )
    assert score['mean_mass'] == pytest.approx(1.0)
    assert score['n_queries'] == 2
    assert score['lift'] > 3.0


def test_induction_uses_tokens_after_previous_matching_prefixes():
    token_ids = [5, 8, 3, 5, 9, 5, 7]
    assert induction_candidate_edges(token_ids, [5]) == {5: [1, 4]}
    attention = peaked(len(token_ids), {5: [1, 4]})
    score = score_induction(attention, token_ids=token_ids, query_positions=[5])
    assert score['mean_mass'] == pytest.approx(1.0)
    assert score['n_candidate_edges'] == 2


def test_successor_proxy_supports_token_map_and_explicit_pairs():
    token_ids = [10, 4, 11, 10, 11, 9]
    edges = successor_candidate_edges(
        token_ids,
        successor_token_map={10: 11},
        explicit_pairs=[[5, 1]],
        query_positions=[2, 4, 5],
    )
    assert edges == {2: [0], 4: [0, 3], 5: [1]}
    attention = peaked(len(token_ids), edges)
    score = score_successor(
        attention,
        token_ids=token_ids,
        successor_token_map={10: 11},
        explicit_pairs=[[5, 1]],
        query_positions=[2, 4, 5],
    )
    assert score['mean_mass'] == pytest.approx(1.0)
    assert score['evidence_level'].startswith('qk_proxy')


def test_classifier_is_multilabel_and_marks_successor_as_candidate():
    token_ids = [10, 5, 11, 10, 11, 10, 11]
    attention = peaked(7, {4: [0, 3], 6: [0, 3, 5]})
    threshold = FamilyThreshold(min_lift=1.1, min_mass=0.5, min_queries=2)
    config = HeadTaxonomyConfig(
        targeted_retrieval=threshold,
        induction=FamilyThreshold(min_lift=99, min_mass=1, min_queries=99),
        successor=threshold,
    )
    result = classify_attention_head(
        attention,
        token_ids=token_ids,
        target_spans=[[0, 1], [3, 4], [5, 6]],
        retrieval_query_positions=[4, 6],
        successor_query_positions=[4, 6],
        successor_token_map={10: 11},
        config=config,
    )
    assert result['labels'] == ['targeted_retrieval', 'successor']
    assert result['successor_interpretation'].startswith('candidate_only')


def test_bad_attention_shape_fails_loudly():
    with pytest.raises(ValueError, match='square'):
        score_targeted_retrieval(
            torch.ones(2, 3), target_spans=[[0, 1]], query_positions=[1]
        )
