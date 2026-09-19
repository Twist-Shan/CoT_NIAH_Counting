"""Target-category record geometry and reproducible Broad discovery scores."""
from collections import defaultdict
import math


def sequential_sum(values):
    total = 0.0
    for value in values:
        total += value
    return total


def target_indices(case):
    target = case['target_category']
    if target not in ('city', 'flower'):
        raise ValueError('Unknown target category')
    records = case['records']
    if len(records) != 10 or {r['category'] for r in records} != {'city', 'flower'}:
        raise ValueError('Expected ten records including both categories')
    for r in records:
        if bool(r['is_target']) != (r['category'] == target):
            raise ValueError('Target flags disagree with the question')
    indices = [i for i, r in enumerate(records) if r['category'] == target]
    if len(indices) != int(case['gold']) or len(indices) not in (1, 3, 5, 7, 9):
        raise ValueError('Target count/gold mismatch')
    return indices


def record_geometry(case, rendered_prompt, offsets):
    selected = target_indices(case)
    if rendered_prompt.count(case['passage']) != 1:
        raise ValueError('Passage is not unique in rendered prompt')
    shift = rendered_prompt.index(case['passage'])
    spans = []
    for r in case['records']:
        a, b = r['char_start'], r['char_end']
        if case['passage'][a:b] != r['text']:
            raise ValueError('Record character span mismatch')
        hits = [i for i, (s, e) in enumerate(offsets)
                if e > s and s < shift+b and e > shift+a]
        if not hits or hits != list(range(hits[0], hits[-1]+1)):
            raise ValueError('Record token span is empty or discontinuous')
        spans.append([hits[0], hits[-1]+1])
    return dict(all_record_token_spans=spans, selected_record_indices=selected,
                target_category=case['target_category'], target_record_count=len(selected))


def broad_score(masses):
    """M exp(H) / J, where every term uses only the selected category."""
    if not masses or any(not math.isfinite(x) or x < 0 for x in masses):
        raise ValueError('Invalid record masses')
    total = sequential_sum(masses)
    if total == 0:
        return 0.0
    entropy = -sequential_sum((x/total)*math.log(x/total) for x in masses if x > 0)
    return total*math.exp(entropy)/len(masses)


def score_record_masses(case, masses):
    if len(masses) != len(case['records']):
        raise ValueError('Record mass count mismatch')
    return broad_score([masses[i] for i in target_indices(case)])


def rank_rows(rows):
    """Equal weight to cases within seed, then equal weight to seeds.

    Explicit sequential floating summation and canonical order make the saved
    complete ranking exactly reproducible on Python 3.10 and Python 3.12.
    """
    groups = defaultdict(lambda: defaultdict(list))
    seen, support = set(), None
    for row in sorted(rows, key=lambda r: (r['seed'], r['case_id'])):
        if row['split'] != 'discovery' or row['case_id'] in seen:
            raise ValueError('Duplicate or non-discovery ranking input')
        seen.add(row['case_id'])
        heads = [(int(l), int(h)) for l, h, _ in row['heads']]
        if len(set(heads)) != len(heads) or (support is not None and heads != support):
            raise ValueError('Inconsistent head coverage')
        support = heads
        for layer, head, score in row['heads']:
            if not math.isfinite(score) or score < 0:
                raise ValueError('Invalid Broad score')
            groups[int(layer), int(head)][row['seed']].append(score)
    if not groups:
        raise ValueError('No discovery observations')
    scores = {head: sequential_sum(sequential_sum(v)/len(v) for _, v in sorted(seeds.items()))/len(seeds)
              for head, seeds in groups.items()}
    return [[l, h, scores[l, h]] for l, h in sorted(scores, key=lambda h: (-scores[h], h))]


def canary_cases(plans):
    seed = min(p['seed'] for p in plans if p['split'] == 'confirmation')
    # Exercise city/flower questions at both sparse and dense category counts.
    chosen = [p for p in plans if p['seed'] == seed and int(p['case']['gold']) in (1, 9)]
    if len(chosen) != 4 or {(p['case']['target_category'], int(p['case']['gold'])) for p in chosen} != {
        ('city', 1), ('city', 9), ('flower', 1), ('flower', 9)}:
        raise ValueError('Incomplete canary support')
    return chosen
