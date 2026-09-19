"""Recompute fixed-layer NCC dynamics from the frozen discovery selection.

Run in the existing Synthetic repository environment, with PYTHONPATH=src:scripts.
Only the additional analysis directory is written; checkpoints remain unchanged.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import time
import torch
import run_v58_unified_additional as source


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    frozen = json.loads(args.selection.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    registry_path = args.run_dir/'analysis/v58_alignment_supplement_20260905/input_registry.csv'
    chosen, registry = source.load_panel(args.run_dir, registry_path)
    torch.set_num_threads(4)
    cfg, vocab, _, _, _ = source.base._load_bundle(args.run_dir, device=args.device)
    expected_steps = sorted(source.AR_STEPS)
    for mode in ['nonthinking', 'thinking']:
        available = source.checkpoint_map(args.run_dir, mode)
        assert set(expected_steps) <= set(available), (mode, expected_steps, list(available))
        folder = args.output/'frozen'/mode/'geometry'
        folder.mkdir(parents=True)
        (folder/'frozen_depths.json').write_text(json.dumps(frozen['selected'][mode]))
        out = args.output/mode
        out.mkdir()
        records = [source.make_record(e, vocab, mode, s, b) for e, s, b in chosen]
        settings = SimpleNamespace(run_dir=args.run_dir, legacy=args.output/'frozen', batch_size=16)
        source.geometry_dynamics(settings, cfg, vocab, records, mode, out)
    manifest = {'status': 'complete', 'selection': frozen['selected'],
                'selection_sha256': digest(args.selection), 'registry_sha256': digest(registry_path),
                'steps': expected_steps, 'script_sha256': digest(Path(__file__)),
                'source_script_sha256': digest(Path(source.__file__)),
                'device': args.device, 'completed_unix': time.time()}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('NCC DYNAMICS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
