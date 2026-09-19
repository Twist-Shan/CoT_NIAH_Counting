"""Read-only remote validation of the frozen pilot and existing model cache."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--cache', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.repo / 'additional_experiments'))
sys.path.insert(0, str(args.repo / 'src'))
started = time.perf_counter()
report = {'repo': str(args.repo), 'cache': str(args.cache), 'checks': []}

try:
    import torch
    import transformers
    from run import verify_frozen, render
    from protocol import read_jsonl
    from realistic_niah_v4.spec import resolve_model_spec
    from realistic_niah_v4.modeling import _tokenizer_compatibility_kwargs

    report['versions'] = {'torch': torch.__version__, 'transformers': transformers.__version__}
    report['nvidia_smi'] = subprocess.run(['nvidia-smi', '-q'], capture_output=True, text=True).stdout
    report['cuda_available'] = torch.cuda.is_available()
    if report['cuda_available']:
        x = torch.ones(8, device='cuda')
        report['cuda_tensor_sum'] = x.sum().item()
        del x
    else:
        report['cuda_tensor_sum'] = None

    inputs = {}
    for name in ('task_transfer_smoke_20260905_v2', 'task_transfer_20260905_v2'):
        frozen = args.repo / 'additional_experiments' / 'runs' / name / 'frozen'
        config = verify_frozen(frozen)
        inputs[name] = (config, read_jsonl(frozen / 'cases.jsonl'))
        report['checks'].append({'check': 'frozen_hashes', 'run': name, 'status': 'PASS'})

    for label in ('Qwen3-8B', 'Gemma4-E4B'):
        checkpoint_started = time.perf_counter()
        spec = resolve_model_spec(label)
        snapshot = args.cache / ('models--' + spec.model_id.replace('/', '--')) / 'snapshots' / spec.revision
        indexes = list(snapshot.glob('*.safetensors.index.json'))
        if len(indexes) == 1:
            index = json.loads(indexes[0].read_text())
            weights = sorted(set(index['weight_map'].values()))
        elif not indexes and (snapshot / 'model.safetensors').is_file():
            weights = ['model.safetensors']
        else:
            raise ValueError(f'Unrecognized or ambiguous weight layout in {snapshot}: {indexes}')
        if not all((snapshot / name).is_file() and (snapshot / name).stat().st_size > 0 for name in weights):
            raise ValueError(f'Missing or empty checkpoint shard: {label}')
        tokenizer = transformers.AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision,
            cache_dir=str(args.cache), local_files_only=True, trust_remote_code=False,
            **_tokenizer_compatibility_kwargs(spec))
        ranges = {}
        for name, (config, cases) in inputs.items():
            lengths = {mode: [] for mode in config['modes']}
            for mode in config['modes']:
                for case in cases:
                    encoding, offsets, prompt = render(case, mode, tokenizer)
                    assert len(offsets) == encoding.sequence_length
                    lengths[mode].append(encoding.sequence_length)
            ranges[name] = {mode: {'n': len(v), 'min': min(v), 'max': max(v)} for mode, v in lengths.items()}
        report['checks'].append({'check': 'cached_checkpoint_and_rendered_prompts', 'model': label,
            'revision': spec.revision, 'status': 'PASS', 'weight_shards': len(weights),
            'weight_bytes': sum((snapshot / name).stat().st_size for name in weights),
            'prompt_tokens': ranges, 'elapsed_seconds': time.perf_counter() - checkpoint_started})
    report['status'] = 'PASS' if report['cuda_available'] else 'CPU_CHECKS_PASS_GPU_BLOCKED'
except Exception:
    report['status'] = 'FAIL'
    report['traceback'] = traceback.format_exc()
finally:
    report['elapsed_seconds'] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'nvidia_smi'}, indent=2), flush=True)
sys.exit(0 if report['status'] == 'PASS' else 2)
