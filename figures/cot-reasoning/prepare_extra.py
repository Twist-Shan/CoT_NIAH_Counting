"""Main-figure and supplementary endpoints reconstructed from local evidence."""
import numpy as np
import prepare_data as d

MODELS, REPO, ALIGNED, SCOPE = d.MODELS, d.REPO, d.ALIGNED, d.SCOPE
draws = np.random.default_rng(20260911).integers(0, 10, (10000, 10))
ablation = []
ablation_seed = []
for model in MODELS:
    arms = [r for r in d.ablation_arms if r['model'] == model]
    for outcome in ['next_city_failure', 'final_count_failure']:
        for condition in ['clean', 'layer_matched_random', 'selected_bank']:
            values, baselines = [], []
            for seed in range(1254, 1264):
                original = float(next(r for r in arms if int(r['seed']) == seed and r['condition'] == 'clean')[outcome])
                value = np.mean([float(r[outcome]) for r in arms if int(r['seed']) == seed and r['condition'] == condition])
                values.append(value - original)
                baselines.append(original)
                ablation_seed.append(dict(model=model, seed=seed, outcome=outcome, condition=condition,
                                          original_failure=original, arm_failure=float(value), effect=float(value-original)))
            mean, low, high = d.seed_ci(values, draws)
            ablation.append(dict(model=model, outcome=outcome, condition=condition, mean=mean,
                                  ci95_low=low, ci95_high=high, original_failure=float(np.mean(baselines))))
d.write_csv('ablation_effect_plot.csv', ablation)
d.write_csv('ablation_effect_seed.csv', ablation_seed)

# C. Forward continuation at layers frozen from discovery running-index geometry.
completion_root = REPO / 'work/cot_completion_20260912'
update_root = REPO / 'work/cot_update_ncc_l21_20260915'
update_analysis = d.read_json(update_root / 'analysis_primary/summary.json')
update_manifest = d.read_json(update_root / 'bundle/manifest.json')
assert update_analysis['status'] == 'COMPLETE_PAIRED_AUDIT' and update_analysis['condition_rows'] == 360
assert update_analysis['primary_only']
progress, progress_trials, progress_hops, progress_stepwise_trials = [], [], [], []
example = None
for model in MODELS:
    audited, = [r for r in update_analysis['summaries']
                if r['model'] == model and r['scope'] == 'item_span' and r['direction'] == 'forward']
    layer = update_manifest['models'][model]['layer_zero_based']
    assert layer + 1 == audited['layer']
    rows = []
    for k in [4, 6, 8]:
        p = update_root / 'bundle/runs' / model / 'item_span' / f'forward_k{k}/results/trials.jsonl'
        assert d.read_json(p.parent.parent/'process_status.json')['status'] == 'COMPLETE'
        rows += d.read_jsonl(p)
    assert {r['layer'] for r in rows} == {layer}
    patched = [r for r in rows if r['condition'] == 'donor_to_receiver']
    original = [r for r in rows if r['condition'] == 'receiver_self']
    assert len(patched) == len(original) == 30
    assert len({r['seed'] for r in patched}) == 10
    adopted = [r for r in patched if r['first_generated_known_city_ordinal'] == r['donor_occurrence_k']+1]
    assert all((r['first_generated_known_city_ordinal'] == r['donor_occurrence_k']+1) ==
               (r['generated_known_city_ordinals_any_surface'][:1] == [r['donor_occurrence_k']+1])
               for r in patched)
    original_hits = sum(r['first_generated_known_city_ordinal'] == r['donor_occurrence_k']+1 for r in original)
    for r in rows:
        if r['condition'] not in ['receiver_self', 'donor_to_receiver']:
            continue
        progress_trials.append(dict(model=model, seed=r['seed'], target_occurrence=r['donor_occurrence_k'],
                                    receiver_occurrence=r['receiver_occurrence_j'],
                                    condition='target_patch' if r['condition'] == 'donor_to_receiver' else 'self_patch',
                                    first_city=r['first_generated_known_city_ordinal'],
                                    target_successor=r['donor_occurrence_k']+1))
    # One rule for every step: successful through the preceding step, with an
    # expected next item still present in the prompt. Step 1 starts with all patches.
    previous_successes = len(patched)
    for hop in [1, 2, 3, 4]:
        step_rows = []
        for r in patched:
            k = r['donor_occurrence_k']
            sequence = r['generated_known_city_ordinals_any_surface']
            previous_correct = (k+hop-1 <= r['gold_count'] and
                                sequence[:hop-1] == list(range(k+1, k+hop)))
            enough_items = k+hop <= r['gold_count']
            eligible = previous_correct and enough_items
            success = eligible and sequence[:hop] == list(range(k+1, k+hop+1))
            step_rows.append(dict(model=model, seed=r['seed'], target_occurrence=k, hop=hop,
                                  previous_steps_success=previous_correct, enough_remaining_items=enough_items,
                                  eligible=eligible, success=success))
        assert sum(r['previous_steps_success'] for r in step_rows) == previous_successes
        eligible = sum(r['eligible'] for r in step_rows)
        successes = sum(r['success'] for r in step_rows)
        progress_hops.append(dict(model=model, hop=hop, previous_successes=previous_successes,
                                  excluded_no_remaining_item=previous_successes-eligible,
                                  eligible=eligible, successes=successes,
                                  eligible_seeds=len({r['seed'] for r in step_rows if r['eligible']}),
                                  conditioning='all preceding steps correct and enough remaining items',
                                  analysis='frozen_primary' if hop == 1 else 'conditional_follow_up'))
        expected = audited['continuation_steps'][hop-1]
        assert (successes, eligible) == (expected['success'], expected['eligible'])
        progress_stepwise_trials.extend(step_rows)
        previous_successes = successes
    hop1 = next(h for h in progress_hops if h['model'] == model and h['hop'] == 1)
    assert (hop1['successes'], hop1['eligible']) == (len(adopted), len(patched))
    hop2 = next(h for h in progress_hops if h['model'] == model and h['hop'] == 2)
    progress.append(dict(model=model, condition='natural no-index' if model == MODELS[0] else 'prompt-conditioned no-index',
                         layer=layer, original_hits=original_hits,
                         patched_hits=len(adopted), cells=30, seeds=10,
                         continued_hits=hop2['successes'], continued_eligible=hop2['eligible']))
    assert (original_hits, len(adopted)) == (audited['self_adoption'], audited['target_adoption'])
    if model == MODELS[0]:
        example = next(r for r in patched if r['seed'] == 1307 and r['donor_occurrence_k'] == 6)
        assert example['generated_known_city_ordinals_any_surface'][:4] == [7, 8, 9, 10]
        assert all(city in example['completion_text'] for city in ['Osaka', 'Lima', 'Krakow', 'New York'])
assert [r['layer'] + 1 for r in progress] == [19, 21]
d.write_csv('progress_forward_plot.csv', progress)
d.write_csv('progress_forward_trials.csv', progress_trials)
d.write_csv('progress_forward_hops.csv', progress_hops)
d.write_csv('progress_forward_stepwise_trials.csv', progress_stepwise_trials)

# Supporting evidence for the schematic: the mid-layer routing comparator.
route = d.read_json(REPO/'work/same_site_progress_transplant_20260827/n10_item_span_contextual_l16_v1/frozen_scope_analysis.json')
route_cells = route['cells']
assert len(route_cells) == 20 and {r['layer'] for r in route_cells} == {16}
assert len({r['seed'] for r in route_cells}) == 10
assert {r['donor_occurrence_k'] for r in route_cells} == {6}
assert sum(r['patched_greedy_donor_adoption'] for r in route_cells) == 16
assert all(r['paired_attention_shift'] > 0 and r['paired_logodds_shift'] > 0 for r in route_cells)
d.write_csv('mechanism_midlayer_route.csv', [dict(seed=r['seed'], direction=r['direction'], layer=r['layer'],
    target_occurrence=r['donor_occurrence_k'], receiver_occurrence=r['receiver_occurrence_j'],
    attention_shift=r['paired_attention_shift'], logodds_shift=r['paired_logodds_shift'],
    target_successor_adopted=r['patched_greedy_donor_adoption']) for r in route_cells])

# Supplement A. Clamp clean carriers while maintaining the same head damage.
write, write_seeds = [], []
carrier_damage = []
for model in MODELS:
    p = ALIGNED/'runs'/model/'targeted_counter_write/analysis_confirmation'
    g = d.read_json(p/'claim_gates.json')
    a = d.read_json(p/'audit.json')
    assert g['seed_count'] == 10 and g['outcome_blind'] and g['teacher_forced_trace_tokens']
    seeds = d.read_csv(p/'seed_effects.csv')
    r = next(r for r in g['all_estimands'] if r['estimand'] == 'clean_carrier_restoration')
    assert abs(np.mean([float(s['clean_carrier_restoration']) for s in seeds])-r['mean_effect']) < 1e-12
    for name in ['selected_carrier_deformation', 'selected_carrier_deformation_specificity']:
        v = next(r for r in g['all_estimands'] if r['estimand'] == name)
        assert v['ci_low'] > 0
        assert abs(np.mean([float(s[name]) for s in seeds])-v['mean_effect']) < 1e-12
        carrier_damage.append(dict(model=model, estimand=name, mean=v['mean_effect'], ci95_low=v['ci_low'], ci95_high=v['ci_high']))
    write.append(dict(model=model, mean=r['mean_effect'], ci95_low=r['ci_low'], ci95_high=r['ci_high']))
    write_seeds += [dict(model=model, seed=s['seed'], effect=s['clean_carrier_restoration']) for s in seeds]
d.write_csv('carrier_restore_plot.csv', write)
d.write_csv('carrier_restore_seed.csv', write_seeds)
d.write_csv('mechanism_carrier_damage.csv', carrier_damage)

# Supplement B. Accuracy with prompt records or the entire trace blanked.
blank, blank_seed, blank_cells = [], [], []
blank_draws = np.random.default_rng(20260912).integers(0, 10, (10000, 10))
for model, version in zip(MODELS, ['v2', 'v1']):
    base = REPO/'reports/v5_native_token_level_ablation'/model
    all_banks = []
    for bank in ['tracebank', 'promptbank']:
        if model == MODELS[1]:
            p = completion_root/f'backend_replay/fixed/blank_{bank.removesuffix("bank")}'
            assert d.read_json(p/'process_status.json')['status'] == 'COMPLETE'
            paired = d.read_json(completion_root/f'analysis/blank_{bank.removesuffix("bank")}.json')
            assert paired['status'] == 'COMPLETE_PAIRED_AUDIT' and paired['paired_trials'] == 500
            files = sorted((p/'results/shards').glob('*.jsonl'))
            assert len(files) == 100
            captured = [r for f in files for r in d.read_jsonl(f)]
            assert len(captured) == 500 and {r['status'] for r in captured} == {'ok'}
            all_banks.append(captured)
        else:
            p = base/f'answer_{bank}_top32_confirmation_all20_{version}/analysis_registered_v1'
            audit = d.read_json(p/'analysis_audit.json')
            assert audit['status'] == 'PASS' and int(audit['seed_count']) == 10
            all_banks.append(d.read_csv(p/'token_level_detail.csv'))
    for condition in ['clean', 'prompt_records_blank', 'trace_all_blank']:
        rows = [r for r in all_banks[0] if r['condition'] == condition]
        other = [r for r in all_banks[1] if r['condition'] == condition]
        key = lambda r:(int(r['seed']),int(r['gold_count']))
        assert len(rows) == len(other) == 100
        assert sorted((key(r),d.flag(r['exact_count'])) for r in rows) == sorted((key(r),d.flag(r['exact_count'])) for r in other)
        values = []
        for seed in range(1254,1264):
            ss = [r for r in rows if int(r['seed']) == seed]
            assert len(ss) == 10 and {int(r['gold_count']) for r in ss} == set(range(1,11))
            mean = float(np.mean([d.flag(r['exact_count']) for r in ss]))
            values.append(mean)
            blank_seed.append(dict(model=model, seed=seed, condition=condition, exact_rate=mean))
        mean, lo, hi = d.seed_ci(values, blank_draws)
        blank.append(dict(model=model, condition=condition, mean=mean, ci95_low=lo, ci95_high=hi, prompts=100, seeds=10))
        blank_cells += [dict(model=model, seed=r['seed'], gold_count=r['gold_count'], condition=condition,
                            exact=int(d.flag(r['exact_count']))) for r in rows]
assert np.allclose([r['mean'] for r in blank],[.97,.97,.01,.7,.7,.15])
d.write_csv('answer_sources_plot.csv', blank)
d.write_csv('answer_sources_seed.csv', blank_seed)
d.write_csv('answer_sources_cells.csv', blank_cells)

# Supplement C. Same-trial suffix mediation, with residual damage preserved.
relay_root = completion_root/'terminal_intervals'
relay_summary = d.read_json(relay_root/'summary.json')
pairs = d.read_csv(relay_root/'common_support_pairs.csv')
d.record(relay_root/'common_support_cell_counts.csv')
assert relay_summary['status'] == 'PASS_RECOMPUTED_INTERVALS' and relay_summary['common_pair_count_per_model'] == len(pairs) == 86
d.record(relay_root/'audit.json')
assert relay_summary['common_seed_count'] == 10 and not relay_summary['selection_uses_outcomes']
relay = []
for model in MODELS:
    effects = relay_summary['models'][model]['effects']
    fraction = effects['post_terminal_suffix_explained_fraction']
    natural = effects['patch_damage_natural']
    residual = effects['patch_damage__post_terminal_suffix']
    mediated = effects['specific_mediation__post_terminal_suffix']
    assert residual['ci_low'] > 0 and mediated['ci_low'] > 0
    assert abs((natural['estimate']-residual['estimate'])/natural['estimate']-fraction['estimate']) < 1e-12
    relay.append(dict(model=model, mean=fraction['estimate'], ci95_low=fraction['ci_low'], ci95_high=fraction['ci_high'],
                      natural_damage=natural['estimate'], residual_damage=residual['estimate'],
                      specific_mediation=mediated['estimate'], pairs=86, seeds=10))
d.write_csv('terminal_mediation_plot.csv', relay)
