#!/usr/bin/env python3
'''Classify attention heads from an existing Qwen3 Q/K cache.'''
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset_generation.qk_hook_attention.head_taxonomy import (  # noqa: E402
    HeadTaxonomyConfig,
    scan_qk_cache,
)


def _ints(value):
    if value is None or value.lower() == 'all':
        return None
    return [int(x.strip()) for x in value.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--analysis-spec', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--layers', help='Comma-separated layers; default is cache metadata')
    parser.add_argument('--heads', help='Comma-separated heads or all')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    spec = json.loads(args.analysis_spec.read_text(encoding='utf-8'))
    config_raw = (
        json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
    )
    frame = scan_qk_cache(
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        analysis_spec=spec,
        layers=_ints(args.layers),
        heads=_ints(args.heads),
        config=HeadTaxonomyConfig.from_dict(config_raw),
        device=args.device,
    )
    print(frame.to_string(index=False))
    print(f'Wrote taxonomy outputs to {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
