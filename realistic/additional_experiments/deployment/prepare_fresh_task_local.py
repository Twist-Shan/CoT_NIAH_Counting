"""Build Appendix H plans directly from audited new natural generations (CPU).

No historical attention, head banks, intervention outputs, or alignment audit
is required. The generated package discovers its own Broad and Targeted banks.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path[:0] = [str(HERE), str(BASE), str(BASE.parent / 'src')]
from prepare_task_local import copy_snapshot, launcher, read, sha, write
from task_local_inputs import MODES, BROAD_SIZES, TARGET_SIZES, validate_cases, make_plan, expected_points


def verified_capture(path):
    complete = read(path / 'complete.json')
    if complete.get('status') != 'PASS':
        raise ValueError(f'Incomplete natural capture: {path}')
    for name in ('prompt.json', 'generation.json'):
        if complete['files'].get(name) != sha(path / name):
            raise ValueError(f'Natural capture hash mismatch: {path / name}')
    return read(path / 'prompt.json'), read(path / 'generation.json')


def export_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(args, *, tokenizers=None):
    """The injected tokenizer mapping is for CPU tests, never a CLI bypass."""
    from realistic_niah_v4.spec import resolve_model_spec
    from run import summarize_generation

    root = args.output.resolve()
    if root.exists():
        raise FileExistsError('Use a new output directory for a frozen campaign')
    frozen = {'kth': args.kth_frozen.resolve(), 'category': args.category_frozen.resolve()}
    generations = {'kth': args.kth_generations.resolve(), 'category': args.category_generations.resolve()}
    cases, prompts, provenance = {}, {}, {}
    for task, folder in frozen.items():
        manifest = read(folder / 'manifest.json')
        if manifest['status'] != 'PASS':
            raise ValueError(f'{task}: frozen-input audit failed')
        if not {'cases.jsonl', 'user_prompts.jsonl'}.issubset(manifest['files']):
            raise ValueError('Frozen manifest does not cover cases and prompts')
        for name, expected in manifest['files'].items():
            if sha(folder / name) != expected:
                raise ValueError(f'{task}: frozen-input hash mismatch: {name}')
        cases[task] = [json.loads(line) for line in (folder / 'cases.jsonl').read_text(encoding='utf-8').splitlines() if line]
        validate_cases(cases[task], task)
        prompt_rows = [json.loads(line) for line in (folder / 'user_prompts.jsonl').read_text(encoding='utf-8').splitlines() if line]
        prompts[task] = {(r['case_id'], r['mode']): r['user_text'] for r in prompt_rows}
        if len(prompt_rows) != 600 or len(prompts[task]) != 600:
            raise ValueError(f'{task}: expected 600 distinct frozen user prompts')
        provenance[task] = {'manifest_sha256': sha(folder / 'manifest.json'),
                            'cases_sha256': sha(folder / 'cases.jsonl')}
    root.mkdir(parents=True)
    copy_snapshot(root)
    widths, natural, counts, total_points = {}, [], Counter(), 0
    for model in BROAD_SIZES:
        spec = resolve_model_spec(model)
        if tokenizers is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision,
                cache_dir=str(args.cache_dir), local_files_only=True)
        else:
            tokenizer = tokenizers[model]
        for task in ('kth', 'category'):
            source_root = generations[task] / model
            environment = read(source_root / 'environment.json')
            if (environment['model_id'], environment['revision']) != (spec.model_id, spec.revision):
                raise ValueError('Natural generations do not use the registered model revision')
            completed = read(source_root / 'complete.json')
            if completed.get('status') != 'PASS' or completed.get('captures') != 600:
                raise ValueError('A complete 600-trajectory natural run is required for each task/model')
            contract = read(source_root / 'contract.json')
            manifest_hash = contract.get('frozen_manifest_sha256', contract.get('manifest_sha256'))
            if manifest_hash != provenance[task]['manifest_sha256']:
                raise ValueError('Natural run belongs to a different frozen task dataset')
            if task == 'kth':
                widths[model] = environment['head_counts']
                if not widths[model] or any(type(n) is not int or n <= 0 for n in widths[model]):
                    raise ValueError('Invalid saved model head geometry')
            write(root / 'provenance' / f'{task}_{model}.json', dict(
                environment=environment, contract=contract,
                source_complete_sha256=sha(source_root / 'complete.json')))
            plans, registry = [], []
            for case in cases[task]:
                for mode in MODES:
                    source = source_root / 'captures' / mode / case['case_id']
                    prompt, generation = verified_capture(source)
                    if prompt['user_text'] != prompts[task][case['case_id'], mode]:
                        raise ValueError('Saved prompt differs from the frozen task/mode prompt')
                    if prompt['rendered_prompt'].count(case['passage']) != 1:
                        raise ValueError('The task passage must occur exactly once in the rendered prompt')
                    expected_prompt = tokenizer.apply_chat_template(
                        [{'role': 'user', 'content': prompt['user_text']}], tokenize=False,
                        add_generation_prompt=True, enable_thinking=mode == 'native_thinking')
                    if mode == 'nonthinking':
                        expected_prompt += case['answer_prefix']
                    if expected_prompt != prompt['rendered_prompt']:
                        raise ValueError('Rendered prompt differs from the registered model/mode chat template')
                    plan, record = make_plan(case, prompt, generation, tokenizer, model, mode)
                    relative = Path('inputs') / task / model / mode / case['case_id']
                    (root / relative).mkdir(parents=True)
                    for name in ('prompt.json', 'generation.json', 'complete.json'):
                        shutil.copy2(source / name, root / relative / name)
                    plan['source'] = relative.as_posix()
                    plan['source_hashes'] = {name: sha(root / relative / name) for name in ('prompt.json', 'generation.json')}
                    plans.append(plan)
                    registry.append(record)
                    counts['trajectories'] += 1
                    counts[task+'_'+model+'_'+mode+'_broad'] += bool(plan['broad_prefix'])
                    counts[task+'_'+model+'_'+mode+'_targeted'] += bool(plan['target'])
                    scored = summarize_generation(generation, case, mode=mode, prefixed=mode == 'nonthinking')
                    natural.append(dict(model=model, task=task, mode=mode, case_id=case['case_id'],
                        seed=case['seed'], split=plan['split'], level=case['level'], correct=int(scored['correct'])))
            write(root / 'plans' / f'{task}_{model}.json', plans)
            write(root / 'registry' / f'{task}_{model}.json', registry)
            total_points += expected_points(plans, model)
    summary = []
    for model in BROAD_SIZES:
        for task in ('kth', 'category'):
            for mode in MODES:
                for split in ('all', 'discovery', 'confirmation'):
                    subset = [r for r in natural if (r['model'], r['task'], r['mode']) == (model, task, mode)
                              and (split == 'all' or r['split'] == split)]
                    correct = sum(r['correct'] for r in subset)
                    summary.append(dict(model=model, task=task, mode=mode, split=split,
                                        n=len(subset), correct=correct, mean=correct/len(subset)))
    export_csv(root / 'natural/natural_per_case.csv', natural)
    export_csv(root / 'natural/natural_summary.csv', summary)
    write(root / 'input_audit.json', dict(status='PASS', counts=dict(counts), frozen_inputs=provenance,
        source_identity='fresh_run', historical_identity_claimed=False,
        alignment='verified original prompt/output token IDs; unavailable sites retained',
        selection='discovery only; no correctness filtering'))
    cfg = dict(version='task_local_fresh_v1', input_mode='fresh_natural_generations',
        cache=str(args.cache_dir.resolve()), widths=widths, models=list(BROAD_SIZES), tasks=['kth', 'category'],
        target_sizes=TARGET_SIZES, broad_sizes=BROAD_SIZES,
        discovery_seeds=list(range(1234, 1254)), confirmation_seeds=list(range(1254, 1264)),
        control_policy='disjoint_first_minimum_overlap', layer_cap=False,
        broad_max_tokens=64, targeted_max_tokens=256,
        broad_source='nonthinking: ten full prompt-record spans; native: registered item endpoint tokens',
        broad_numerics=dict(epsilon=1e-12, zero_mass_threshold=1e-12, reference='Appendix H; kth_retrieval.broad'),
        target_selection='raw target-source-record attention at middle eligible transition; seed-equal global Top-K',
        old_results_reused=False, natural_outputs_reused=True, expected_full_points=total_points,
        stats=dict(bootstrap=20000, seed=20260907, unit='seed',
                   correction='exact two-sided seed sign-flip; Holm; pointwise bootstrap CIs'))
    launch = launcher(args.python)
    for stage in ('discover', 'canary', 'full'):
        for model in cfg['models']:
            launch.append(f'"$PYTHON" run_task_local.py --model {model} --stage {stage}')
    launch.append('"$PYTHON" analyze_task_local.py --root .')
    (root / 'launch.sh').write_text('\n'.join(launch)+'\n', encoding='utf-8', newline='\n')
    cfg['files'] = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    write(root / 'protocol.json', cfg)
    print(json.dumps(dict(status='PREPARED_NOT_RUN', trajectories=counts['trajectories'],
                         expected_full_points=total_points, package=str(root))))
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('kth-frozen', 'kth-generations', 'category-frozen', 'category-generations', 'cache-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--python', default='python', help='Launcher interpreter; may also be set with PYTHON')
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
