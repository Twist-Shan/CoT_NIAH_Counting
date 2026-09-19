"""Frozen full-grid definitions and crash-safe cell journals."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

BASELINES = ('clean', 'needle_corrupt', 'ordinary_corrupt')
PATCH_CONDITIONS = ('self_needle_full', 'reverse_needle_full',
                    'reverse_needle_endpoint', 'reverse_ordinary_full', 'restore_needle_full')


def expected_cells(layers, validation_layers=None):
    validation_layers = set(layers if validation_layers is None else validation_layers)
    return {(name, -1) for name in BASELINES} | {
        (name, int(layer)) for layer in layers for name in PATCH_CONDITIONS
        if name.startswith('reverse') or layer in validation_layers}


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def frozen_contract(path: Path, value):
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise RuntimeError('Resume contract changed; use a new output directory')
    else:
        atomic_json(path, value)


def load_journal(path: Path, *, seed: int, count: int, layers, validation_layers=None):
    """Recover only a non-newline-terminated final write; preserve its evidence."""
    if not path.exists():
        return {}
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        backup = path.with_name(path.name+f'.interrupted_tail_{time.time_ns()}')
        backup.write_bytes(raw)
        cut = raw.rfind(b'\n')+1
        with path.open('wb') as handle:
            handle.write(raw[:cut])
            handle.flush()
            os.fsync(handle.fileno())
        raw = raw[:cut]
    rows = {}
    allowed = expected_cells(layers, validation_layers)
    for line in raw.decode('utf-8').splitlines():
        row = json.loads(line)
        key = (row['condition'], row['patch_layer'])
        if row['seed'] != seed or row['gold_count'] != count or key not in allowed:
            raise RuntimeError('Journal contains a foreign cell')
        if key in rows:
            raise RuntimeError('Duplicate journal cell')
        if row.get('cell_audit') != 'PASS':
            raise RuntimeError('Journal contains an unvalidated cell')
        rows[key] = row
    return rows


def journal_digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history_gate(match):
    return (match['expected_count_delta'] <= 0.05 and match['probability_tv'] <= 0.01
            and match['strict_agrees'])


def self_gate(match):
    return (match['expected_count_delta'] <= 1e-5 and match['probability_tv'] <= 1e-6
            and match['strict_agrees'])
