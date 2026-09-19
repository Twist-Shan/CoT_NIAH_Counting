"""Check actual hook inputs and generation positions on frozen audit examples."""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / 'src', ROOT / 'scripts'):
    sys.path.insert(0, str(p))
import pandas as pd
import torch
import audit_v58_retrieval_generation as audit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    align = args.run_dir / 'analysis/v58_alignment_supplement_20260905'
    registry = pd.read_csv(align / 'input_registry.csv')
    lookup = {e.prompt_sha256: e for e in [audit.base.example_from_dict(json.loads(s)) for s in
        (args.run_dir / 'analysis/behavior_confirmation_v58/examples.jsonl').read_text().splitlines() if s.strip()]}
    # One example per count, selected only by the existing registry order.
    keys = registry.loc[registry.split.eq('confirmation')].groupby('count', sort=True).first().key.tolist()
    examples = [lookup[k] for k in keys]
    results = []
    for mode in ['nonthinking', 'thinking']:
        cfg, vocab, _, _, model = audit.base.load_v20_checkpoint_model(args.run_dir, 'rope', mode, step=10000, device='cuda')
        sites = json.loads((align / mode / 'frozen_sites.json').read_text())
        bank = [tuple(x) for x in sites['role_bank']]
        choices = [('selected_top2', bank[:2]), ('control_top4', [tuple(x) for x in sites['controls']['4'][0]])]
        if mode == 'nonthinking':
            choices.append(('control_L1H2', [(1, 2)]))
        items = [audit.base.render_v20(e, vocab, mode) for e in examples]
        stop = items[0].spans.ans_pos + 1 if mode == 'nonthinking' else items[0].spans.think_pos + 1
        for name, heads in choices:
            current, checks = {}, {'forward_calls': 0, 'head_module_calls': 0, 'masked_query_rows': 0}
            def capture_model(module, args, kwargs):
                current['ids'] = kwargs.get('input_ids', args[0] if args else None)
                assert current['ids'] is not None
                checks['forward_calls'] += 1
            model_handle = model.register_forward_pre_hook(capture_model, with_kwargs=True)
            real_edit = audit._local_attention_edit

            @contextlib.contextmanager
            def audited_edit(model, chosen, positions):
                before, captures = {}, []
                layers = sorted({l for l, h in chosen})
                for layer in layers:
                    def save_input(module, inputs, layer=layer):
                        before[layer] = inputs[0].detach().clone()
                    captures.append(model.layers[layer - 1].attention.output.register_forward_pre_hook(save_input))
                try:
                    with real_edit(model, chosen, positions):
                        for layer in layers:
                            def check_input(module, inputs, layer=layer):
                                ids = current['ids']
                                # Infer onset independently from actual input tokens.
                                expected_positions = []
                                for row in ids:
                                    if mode == 'nonthinking':
                                        onset = stop - 1
                                    else:
                                        candidates = [i for i in torch.nonzero(row.eq(vocab.token_to_id['<Sep>'])).flatten().tolist() if i >= stop]
                                        onset = candidates[0] if candidates else None
                                    expected_positions.append(list(range(onset, ids.shape[1])) if onset is not None else [])
                                assert expected_positions == positions, 'Mask range differs from sustained protocol'
                                expected = before[layer].clone()
                                width = cfg.n_embd // cfg.n_head
                                for i, ps in enumerate(expected_positions):
                                    for ll, h in chosen:
                                        if ll == layer:
                                            expected[i, ps, h * width:(h + 1) * width] = 0
                                assert torch.equal(expected, inputs[0]), 'Actual masked vector differs from declared intervention'
                                checks['head_module_calls'] += 1
                                checks['masked_query_rows'] += sum(map(len, positions))
                            captures.append(model.layers[layer - 1].attention.output.register_forward_pre_hook(check_input))
                        yield
                finally:
                    for handle in captures:
                        handle.remove()
            try:
                audit._local_attention_edit = audited_edit
                audit.sustained_generation(model, cfg, vocab, examples, heads, len(examples), mode)
            finally:
                audit._local_attention_edit = real_edit
                model_handle.remove()
            assert checks['head_module_calls'] > 0 and checks['masked_query_rows'] > 0
            assert all(not block.attention.output._forward_pre_hooks for block in model.layers)
            results.append({'mode': mode, 'condition': name, 'heads': heads, **checks,
                            'query_scope_correct': True, 'selected_slices_zero': True,
                            'all_other_slices_unchanged': True, 'hooks_removed_after_run': True})
            print(mode, name, 'hook contract passed', flush=True)
        del model
        torch.cuda.empty_cache()
    audit.write_json(args.output, {'status': 'complete', 'script_sha256': audit.digest(Path(__file__)),
                     'examples_per_condition': len(examples), 'results': results})


if __name__ == '__main__':
    main()
