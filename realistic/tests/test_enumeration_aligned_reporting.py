import pytest

from realistic_niah_v6.aligned_reporting import continuation_summary, relay_point, relay_summary


def row(seed, k, observed, truncated=False):
    return dict(seed=seed, gold_count=10, donor_occurrence_k=k,
                generated_known_city_ordinals_any_surface=observed, generation_truncated=truncated)


def test_failed_prefix_is_never_repaired_and_truncation_remains():
    rows = [row(1, 6, [7, 8]), row(2, 6, [6, 7, 8]), row(3, 6, [], True)]
    r = continuation_summary(rows, 2, draws=200, random_seed=1)
    assert (r['successes'], r['horizon_eligible'], r['conditional_eligible']) == (1, 3, 1)
    assert r['unconditional']['estimate'] == pytest.approx(1/3)
    assert r['conditional']['estimate'] == 1
    assert r['truncated_trials'] == 1


def test_k8_has_two_remaining_records_not_four_failures():
    rows = [row(1, 8, [9, 10]), row(2, 6, [7, 8, 9, 10])]
    r = continuation_summary(rows, 4, draws=200, random_seed=1)
    assert (r['successes'], r['horizon_eligible'], r['horizon_not_applicable']) == (1, 1, 1)
    assert r['unconditional']['estimate'] == 1


def test_no_conditional_survivors_produces_null_not_zero_or_one():
    r = continuation_summary([row(1, 6, []), row(2, 6, [6])], 2, draws=200, random_seed=1)
    assert r['unconditional']['estimate'] == 0
    assert r['conditional']['estimate'] is None and r['conditional']['ci95'] is None


def test_malformed_saved_parse_is_rejected():
    with pytest.raises(ValueError, match='ordinals'):
        continuation_summary([row(1, 6, ['7'])], 1, draws=200, random_seed=1)


def test_relay_sign_reversal_and_unstable_denominator_are_explicit():
    r = relay_point(1, -.15)
    assert r['signed_reduction'] == pytest.approx(1.15)
    assert r['absolute_reduction'] == pytest.approx(.85)
    assert relay_point(0, 1)['signed_reduction'] is None
    with pytest.raises(ValueError, match='finite'):
        relay_point(float('nan'), 1)


def test_relay_weights_seeds_equally_and_bootstraps_paired_effects():
    rows = [dict(seed=1, natural_damage=1, remaining_signed_damage=.5)] * 3
    rows += [dict(seed=2, natural_damage=3, remaining_signed_damage=1.5)]
    r = relay_summary(rows, draws=500, random_seed=2)
    assert r['natural_damage'] == 2  # A row-weighted mean would be 1.5.
    assert r['signed_reduction'] == .5
    assert r['ci95']['signed_reduction'] == [.5, .5]
