"""Alignment and effect definitions for paired corrupted-state interventions."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields

from realistic_niah_v4.prompts import PromptEncoding
from realistic_niah_v4_4_5.restoration import CorruptionPlan, segment_positions


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def audit_alignment(clean: PromptEncoding, corrupted: PromptEncoding,
                    plan: CorruptionPlan, condition: str) -> dict:
    """Verify exact paired coordinates, untouched context and replacement tokens."""
    if condition not in ('needle', 'ordinary'):
        raise ValueError('Unknown corruption condition')
    for field in fields(clean):
        if field.name != 'input_ids' and getattr(clean, field.name) != getattr(corrupted, field.name):
            raise ValueError(f'Pair metadata mismatch: {field.name}')
    if clean.sequence_length != corrupted.sequence_length:
        raise ValueError('Pair sequence lengths differ')
    if clean.query_position != clean.sequence_length - 1:
        raise ValueError('Expected last-token answer query')
    positions = segment_positions(plan, condition=condition)
    if not positions or len(positions) != len(set(positions)):
        raise ValueError('Empty or overlapping patch positions')
    if min(positions) < 0 or max(positions) >= clean.query_position:
        raise ValueError('Patch touches query or lies outside prompt')
    targets = plan.needle_targets if condition == 'needle' else plan.ordinary_targets
    sources = plan.needle_sources if condition == 'needle' else plan.ordinary_sources
    target_set = set(positions)
    changed = [i for i, (a, b) in enumerate(zip(clean.input_ids, corrupted.input_ids)) if a != b]
    if not changed or not set(changed).issubset(target_set):
        raise ValueError('Corruption is empty or changes tokens outside targets')
    if len(targets) != len(sources):
        raise ValueError('Source/target segment counts differ')
    for (a, b), (c, d) in zip(targets, sources):
        if b - a != d - c or corrupted.input_ids[a:b] != clean.input_ids[c:d]:
            raise ValueError('Replacement tokens or segment lengths differ')
    banks = [set(range(a, b)) for segments in (plan.needle_targets, plan.needle_sources,
              plan.ordinary_targets, plan.ordinary_sources) for a, b in segments]
    if sum(map(len, banks)) != len(set().union(*banks)):
        raise ValueError('Corruption banks overlap')
    lengths = [[b-a for a,b in bank] for bank in (plan.needle_targets, plan.needle_sources,
               plan.ordinary_targets, plan.ordinary_sources)]
    if any(v != lengths[0] for v in lengths):
        raise ValueError('Per-span budgets differ')
    return {'status': 'PASS', 'condition': condition, 'sequence_length': clean.sequence_length,
            'query_position': clean.query_position, 'token_budget': len(positions),
            'changed_tokens': len(changed), 'positions': list(positions),
            'endpoints': [b-1 for a,b in targets], 'segment_lengths': lengths[0],
            'clean_input_sha256': digest(clean.input_ids),
            'corrupted_input_sha256': digest(corrupted.input_ids),
            'attention_mask_sha256': digest(clean.attention_mask)}


def damage_metrics(clean: float, corrupt: float, patched: float, gold: int) -> dict:
    if not all(math.isfinite(v) for v in (clean, corrupt, patched)):
        raise ValueError('Nonfinite outcome')
    denominator = clean - corrupt
    return {'expected_error_increase': abs(patched-gold)-abs(clean-gold),
            'expected_count_drop': clean-patched,
            'clean_corrupt_gap': denominator,
            'normalized_damage': None if abs(denominator) < 1e-8 else (clean-patched)/denominator}
