"""Separate answer-token gating from count discrimination in frozen v58 NT.

All readouts are at the ORIGINAL supplied Ans query, i.e. the first step of
sustained intervention. Later interventions cannot affect this readout. Mean
replacement is fitted on discovery inputs only; it is an additional diagnostic,
not a replacement for the archived full-vocabulary generation endpoint.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / 'src', ROOT / 'scripts'):
    sys.path.insert(0, str(directory))
import numpy as np
import pandas as pd
import torch
import audit_v58_retrieval_generation as audit


@contextlib.contextmanager
def replace_heads(model, heads, positions, kind, means=None, scale=None, verify=False):
    """Replace selected pre-O slices; all other slices are preserved exactly."""
    handles = []
    width = model.config.n_embd // model.config.n_head
    try:
        for layer in sorted({l for l, h in heads}):
            chosen = [h for l, h in heads if l == layer]
            def edit(module, args, layer=layer, chosen=chosen):
                original = args[0]
                value = original.clone()
                keep = torch.ones_like(original, dtype=torch.bool) if verify else None
                for j, q in enumerate(positions):
                    for h in chosen:
                        sl = slice(h * width, (h + 1) * width)
                        replacement = (means[layer - 1, h] if kind == 'mean' else
                                       original[j, q, sl] * scale if kind == 'scale' else
                                       torch.zeros_like(original[j, q, sl]))
                        value[j, q, sl] = replacement
                        if verify:
                            assert torch.equal(value[j, q, sl], replacement)
                            keep[j, q, sl] = False
                if verify:
                    assert torch.equal(value[keep], original[keep])
                return (value,) + args[1:]
            handles.append(model.layers[layer - 1].attention.output.register_forward_pre_hook(edit))
        yield
    finally:
        for handle in handles:
            handle.remove()


def query_batches(examples, vocab, device, batch_size):
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        items = [audit.base.render_v20(e, vocab, 'nonthinking') for e in batch]
        positions = [i.spans.ans_pos for i in items]
        assert len(set(positions)) == 1
        ids = torch.tensor([i.input_ids[:q + 1] for i, q in zip(items, positions)], device=device)
        assert ids[:, -1].eq(vocab.token_to_id['<Ans>']).all()
        yield batch, ids, positions


@torch.inference_mode()
def fit_discovery_means(model, vocab, examples, batch_size):
    captured = {l: [] for l in range(4)}
    width = model.config.n_embd // model.config.n_head
    for batch, ids, positions in query_batches(examples, vocab, next(model.parameters()).device, batch_size):
        handles = []
        try:
            for li, layer in enumerate(model.layers):
                def capture(module, args, li=li):
                    rows = args[0][torch.arange(len(batch), device=ids.device), positions]
                    captured[li].append(rows.view(len(batch), 8, width).cpu())
                handles.append(layer.attention.output.register_forward_pre_hook(capture))
            model(input_ids=ids)
        finally:
            for handle in handles:
                handle.remove()
    contexts = torch.stack([torch.cat(captured[l]) for l in range(4)])
    means = contexts.mean(1)
    rows = []
    for li in range(4):
        weight = model.layers[li].attention.output.weight.detach().cpu()
        for h in range(8):
            x = contexts[li, :, h]
            y = x @ weight[:, h * width:(h + 1) * width].T
            for space, values in [('pre_O', x), ('residual_contribution', y)]:
                mean_energy = values.mean(0).square().sum().item()
                total_energy = values.square().sum(-1).mean().item()
                variance = (values-values.mean(0)).square().sum(-1).mean().item()
                rows.append({'layer':li+1,'head':h,'space':space,
                             'mean_norm':mean_energy**0.5,'rms_norm':total_energy**0.5,
                             'centered_rms_norm':variance**0.5,
                             'mean_energy_fraction':mean_energy/total_energy})
    return means.to(next(model.parameters()).device), contexts, pd.DataFrame(rows)


@torch.inference_mode()
def readout(model, vocab, examples, heads, kind, means, batch_size, scale=None, verify=False):
    device = next(model.parameters()).device
    count_ids = torch.tensor([vocab.token_to_id[vocab.number_token(n)] for n in range(1,11)], device=device)
    rows = []
    for batch, ids, positions in query_batches(examples, vocab, device, batch_size):
        with replace_heads(model, heads, positions, kind, means, scale, verify):
            logits = model(input_ids=ids).logits[:, -1].float()
        assert torch.isfinite(logits).all()
        numeric = logits[:, count_ids]
        probability = logits.softmax(-1)
        log_conditional = numeric.log_softmax(-1)
        for j, e in enumerate(batch):
            target = e.count - 1
            best_id = int(logits[j].argmax())
            alternatives = numeric[j].clone()
            alternatives[target] = -torch.inf
            restricted_pred = int(numeric[j].argmax()) + 1
            restricted_prob = log_conditional[j].exp()
            row = {'key':e.prompt_sha256,'count':e.count,
                   'full_vocab_token':vocab.id_to_token[best_id],
                   'full_vocab_accuracy':float(best_id==int(count_ids[target])),
                   'first_is_number':float(best_id in count_ids),
                   'p_numeric_1to10':float(probability[j,count_ids].sum()),
                   'p_ans':float(probability[j,vocab.token_to_id['<Ans>']]),
                   'p_eos':float(probability[j,vocab.eos_id]),
                   'count_restricted_pred':restricted_pred,
                   'count_restricted_accuracy':float(restricted_pred==e.count),
                   'count_restricted_abs_error':abs(restricted_pred-e.count),
                   'count_restricted_nll':float(-log_conditional[j,target]),
                   'count_restricted_expected':float((restricted_prob*torch.arange(1,11,device=device)).sum()),
                   'count_restricted_margin':float(numeric[j,target]-alternatives.max()),
                   'numeric_vs_ans_margin':float(numeric[j].max()-logits[j,vocab.token_to_id['<Ans>']])}
            row.update({f'count_logit_{n}':float(numeric[j,n-1]) for n in range(1,11)})
            rows.append(row)
    assert all(not b.attention.output._forward_pre_hooks for b in model.layers)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--batch-size',default=50,type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    dest = args.output / 'readouts'
    dest.mkdir()
    run = args.run_dir
    align = run / 'analysis/v58_alignment_supplement_20260905'
    prior = run / 'analysis/v58_top1to8_final_query_20260908'
    source = run / 'analysis/behavior_confirmation_v58/examples.jsonl'
    registry = pd.read_csv(align/'input_registry.csv')
    lookup = {e.prompt_sha256:e for e in [audit.base.example_from_dict(json.loads(s))
              for s in source.read_text(encoding='utf-8').splitlines() if s.strip()]}
    discovery = [lookup[k] for k in registry.loc[registry.split.eq('discovery'),'key']]
    confirmation = [lookup[k] for k in registry.loc[registry.split.eq('confirmation'),'key']]
    assert len(discovery)==200 and len(confirmation)==100
    assert not set(e.prompt_sha256 for e in discovery)&set(e.prompt_sha256 for e in confirmation)
    old = json.loads((prior/'protocol.json').read_text())
    arms = [a for a in old['arms'] if a['mode']=='nonthinking']
    assert len(arms)==496
    seen_single = {tuple(a['heads'][0]) for a in arms if a['top_k']==1}
    for l in range(1,5):
        for h in range(8):
            if (l,h) not in seen_single:
                arms.append({'mode':'nonthinking','arm':'single_head','top_k':1,'repeat':(l-1)*8+h,
                             'heads':[[l,h]],'heads_label':f'L{l}H{h}','overlap':0})
    assert len(arms)==520
    checkpoint = audit.checkpoint_path(run,'nonthinking',10000)
    checkpoint_hash = audit.digest(checkpoint)
    assert checkpoint_hash==old['checkpoint_sha256']['nonthinking']
    protocol = {'status':'frozen_before_new_inference','created_unix':time.time(),
                'script_sha256':audit.digest(Path(__file__)),'checkpoint_sha256':checkpoint_hash,
                'source_sha256':audit.digest(source),'registry_sha256':audit.digest(align/'input_registry.csv'),
                'original_protocol_sha256':audit.digest(prior/'protocol.json'),
                'discovery_keys':[e.prompt_sha256 for e in discovery],
                'confirmation_keys':[e.prompt_sha256 for e in confirmation],
                'arms':arms,'interventions':['zero','mean'],
                'scale_L1H2':[0.,.01,.05,.1,.25,.5,.75,1.],
                'mean_fit':'Unconditional mean clean pre-O head context at supplied Ans query over all 200 frozen discovery inputs. No confirmation fitting or count conditioning.',
                'readout':'Original supplied Ans query. This is exactly the first step of sustained intervention; this experiment does not evaluate subsequent generation under mean replacement.',
                'metrics':'Full-vocabulary first-token accuracy and numeric validity; numeric probability mass; argmax and NLL restricted to known count labels 1..10 on ALL 100 inputs, including invalid full-vocabulary outputs.',
                'selection':'All frozen Top1..8 arms, all 32 individual heads, all listed L1H2 scales; no outcome selection; previous experiments remain unchanged.'}
    audit.write_json(args.output/'protocol.json',protocol)
    torch.set_num_threads(4)
    torch.manual_seed(20260908)
    torch.backends.cuda.matmul.allow_tf32=False
    cfg,vocab,_,_,model=audit.base.load_v20_checkpoint_model(run,'rope','nonthinking',step=10000,device='cuda')
    model.eval()
    means,contexts,stats=fit_discovery_means(model,vocab,discovery,args.batch_size)
    torch.save({'means':means.cpu(),'discovery_contexts':contexts,'keys':protocol['discovery_keys']},args.output/'discovery_head_means.pt')
    stats.to_csv(args.output/'discovery_head_context_stats.csv',index=False)
    clean=readout(model,vocab,confirmation,[],'zero',means,args.batch_size)
    old_clean=pd.read_csv(prior/'nonthinking/clean.csv').set_index('key')
    assert clean.full_vocab_token.tolist()==old_clean.loc[clean.key,'first_token'].tolist()
    clean.to_csv(dest/'clean.csv',index=False)
    # Exact slice checks and independent original implementation comparison.
    small=[next(e for e in confirmation if e.count==n) for n in range(1,11)]
    probe=[[1,2],[2,6]]
    checks={}
    reference=audit.original_query_logits(model,cfg,vocab,small,probe,10)
    for kind in ('zero','mean'):
        together=readout(model,vocab,small,probe,kind,means,10,verify=True)
        alone=readout(model,vocab,small,probe,kind,means,1)
        assert together.full_vocab_token.tolist()==alone.full_vocab_token.tolist()
        cols=[f'count_logit_{n}' for n in range(1,11)]
        error=float(np.max(np.abs(together[cols].values-alone[cols].values)))
        assert error < 1e-3
        if kind=='zero':
            assert together.full_vocab_token.tolist()==reference.full_vocab_token.tolist()
            assert np.max(np.abs(together[cols].values-reference[cols].values)) < 1e-6
        checks[kind]={'selected_and_untouched_slices_verified':True,'batch1_vs10_tokens_equal':True,'batch_logit_max_error':error}
    checks['all_archived_zero_first_tokens_match']=True
    audit.write_json(args.output/'validation.json',checks)
    for kind in ('zero','mean'):
        for i,a in enumerate(arms):
            frame=readout(model,vocab,confirmation,a['heads'],kind,means,args.batch_size)
            for col in ('arm','top_k','repeat','overlap'):
                frame[col]=a[col]
            frame['heads']=a['heads_label']
            frame['intervention']=kind
            if kind=='zero' and a['arm']!='single_head':
                archived=pd.read_csv(prior/'nonthinking/arms'/f"sustained_{a['arm']}_k{a['top_k']}_r{a['repeat']}.csv").set_index('key')
                assert frame.full_vocab_token.tolist()==archived.loc[frame.key,'first_token'].tolist()
            frame.to_csv(dest/f"{kind}_{a['arm']}_k{a['top_k']}_r{a['repeat']}.csv",index=False)
            if i%25==0 or a['arm']=='selected':
                print(kind,i+1,len(arms),a['arm'],a['top_k'],a['repeat'],
                      'full',frame.full_vocab_accuracy.mean(),'numeric',frame.count_restricted_accuracy.mean(),
                      'valid',frame.first_is_number.mean(),flush=True)
    scale_frames=[]
    for scale in protocol['scale_L1H2']:
        f=readout(model,vocab,confirmation,[[1,2]],'scale',means,args.batch_size,scale=scale)
        f['scale']=scale
        scale_frames.append(f)
    pd.concat(scale_frames,ignore_index=True).to_csv(args.output/'L1H2_scaling.csv',index=False)
    checks['archived_zero_conditions_checked']=496
    checks['clean_replay_exact']=True
    checks['discovery_confirmation_disjoint']=True
    audit.write_json(args.output/'validation.json',checks)
    audit.write_json(args.output/'manifest.json',{'status':'complete','readout_rows':104900,
        'files':{str(p.relative_to(args.output)):audit.digest(p) for p in args.output.rglob('*') if p.is_file()}})
    print('NT READOUT DIAGNOSIS COMPLETE',flush=True)


if __name__=='__main__':
    main()
