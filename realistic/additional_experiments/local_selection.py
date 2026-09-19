"""Task-local discovery rankings and disjoint-first random head controls."""
from collections import Counter, defaultdict
import math
import random


def rank_discovery(rows):
    """Seed-equal mean score, global descending rank, deterministic tie breaking."""
    groups = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in rows:
        if row["split"] != "discovery":
            raise ValueError("Confirmation data cannot enter a discovery ranking")
        identity = (row["seed"], row["case_id"])
        if identity in seen:
            raise ValueError("Duplicate discovery case")
        seen.add(identity)
        for layer, head, score in row["heads"]:
            if not math.isfinite(score) or score < 0:
                raise ValueError("Invalid attention score")
            groups[int(layer), int(head)][int(row["seed"])].append(float(score))
    if not groups:
        raise ValueError("No eligible discovery observations")
    supports = {tuple((s, len(v)) for s, v in sorted(g.items())) for g in groups.values()}
    if len(supports) != 1:
        raise ValueError("Heads do not share the same discovery support")
    scores = {h: sum(sum(v) / len(v) for v in g.values()) / len(g) for h, g in groups.items()}
    ordered = sorted(scores, key=lambda h: (-scores[h], h))
    return [[l, h, scores[l, h]] for l, h in ordered]


def select_heads(ranking, k):
    """True global Top-K; no layer quota or outcome-dependent selection."""
    if not isinstance(k, int) or k < 1 or k > len(ranking):
        raise ValueError("K outside the ranked head set")
    heads = [[int(l), int(h)] for l, h, _ in ranking[:k]]
    if len(set(map(tuple, heads))) != k:
        raise ValueError("Duplicate ranked head")
    return heads


def random_control(heads, widths, seed):
    """Match layer counts while minimizing overlap with selected heads.

    Use only unselected heads if sufficient. Otherwise include every unselected
    head and sample the exact deficit from selected heads. Thus overlap in each
    layer is max(0, 2*n_selected - width). Repeats may overlap one another.
    """
    if len(set(map(tuple, heads))) != len(heads):
        raise ValueError("Duplicate selected head")
    for layer, head in heads:
        if not 0 <= layer < len(widths) or not 0 <= head < widths[layer]:
            raise ValueError("Head outside model")
    rng = random.Random(seed)
    out = []
    for layer, n in sorted(Counter(l for l, _ in heads).items()):
        selected = sorted(h for l, h in heads if l == layer)
        unselected = [h for h in range(widths[layer]) if h not in selected]
        draw = rng.sample(unselected, min(n, len(unselected)))
        deficit = n - len(draw)
        if deficit:
            draw.extend(rng.sample(selected, deficit))
        out.extend([[layer, h] for h in draw])
    return out


def control_audit(heads, widths, controls):
    """Assert disjointness when feasible and minimum unavoidable overlap."""
    selected = set(map(tuple, heads)); details = []
    counts = Counter(l for l, _ in heads)
    for control in controls:
        assert len(control) == len(set(map(tuple, control))) == len(heads)
        assert Counter(l for l, _ in control) == counts
        per_layer = []
        for layer, n in sorted(counts.items()):
            overlap = sum((l, h) in selected for l, h in control if l == layer)
            minimum = max(0, 2*n-widths[layer])
            assert overlap == minimum, (layer, overlap, minimum)
            per_layer.append(dict(layer=layer, width=widths[layer], selected_count=n,
                unselected_count=widths[layer]-n, fallback_required=minimum > 0,
                minimum_overlap=minimum, actual_overlap=overlap))
        details.append(per_layer)
    return details


def target_span(case, rendered_prompt, offsets, target_entity):
    """Map a unique target source record's full literal span to prompt tokens."""
    records = [r for r in case["records"] if r["city"].casefold() == target_entity.casefold()]
    if len(records) != 1:
        raise ValueError("Target must identify exactly one source record")
    if rendered_prompt.count(case["passage"]) != 1:
        raise ValueError("Passage must occur exactly once")
    shift = rendered_prompt.index(case["passage"])
    r = records[0]
    start, end = shift + r["char_start"], shift + r["char_end"]
    indices = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("Target span is empty or discontinuous")
    return indices[0], indices[-1] + 1
