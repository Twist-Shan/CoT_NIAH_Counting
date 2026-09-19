"""Test whether L1H2 reads a constant background-token value at Ans."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/'src',ROOT/'scripts'):
    sys.path.insert(0,str(p))
import pandas as pd
import torch
import audit_v58_retrieval_generation as audit
import diagnose_v58_nonthinking_readout as diag


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    run=args.run_dir
    prior=run/'analysis/v58_nonthinking_readout_diagnosis_20260908'
    protocol=json.loads((prior/'protocol.json').read_text())
    source=run/'analysis/behavior_confirmation_v58/examples.jsonl'
    lookup={e.prompt_sha256:e for e in [audit.base.example_from_dict(json.loads(s)) for s in source.read_text().splitlines() if s.strip()]}
    cp=audit.checkpoint_path(run,'nonthinking',10000)
    assert audit.digest(cp)==protocol['checkpoint_sha256']
    audit.write_json(args.output/'protocol.json',{
        'status':'frozen_before_probe','created_unix':time.time(),'script_sha256':audit.digest(Path(__file__)),
        'checkpoint_sha256':audit.digest(cp),'source_sha256':audit.digest(source),
        'support':{'discovery':protocol['discovery_keys'],'confirmation':protocol['confirmation_keys']},
        'question':'At the original Ans query, does L1H2 retrieve the nearly constant value vector of background character o?',
        'observational':'All 300 original inputs: attention mass by token identity and target position, head context versus static layer-1 V(o).',
        'causal':'On all 100 confirmation inputs, replace only L1H2 pre-O context by the fixed V(o) computed from the checkpoint and token embedding. No fitted confirmation activations.',
        'scope':'Original supplied Ans query only; conclusions about L1H2 are restricted to this position.'})
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    cfg,vocab,_,_,model=audit.base.load_v20_checkpoint_model(run,'rope','nonthinking',step=10000,device='cuda')
    model.eval()
    o_id=vocab.token_to_id['<CH_006F>']
    layer=model.layers[0]
    value_o=layer.attention.qkv(layer.ln_attention(model.token_embedding(torch.tensor([[o_id]],device='cuda')))).view(1,1,3,8,64)[0,0,2,2]
    means=torch.load(prior/'discovery_head_means.pt',map_location='cpu',weights_only=False)['means'].to('cuda')
    means[0,2]=value_o
    rows=[]
    for split in ['discovery','confirmation']:
        examples=[lookup[k] for k in protocol[split+'_keys']]
        for batch,ids,positions in diag.query_batches(examples,vocab,'cuda',25):
            captured=[]
            def capture(module,args):
                captured.append(args[0][:,-1,128:192].clone())
            handle=layer.attention.output.register_forward_pre_hook(capture)
            try:
                output=model(input_ids=ids,output_attentions=True)
            finally:
                handle.remove()
            attention=output.attentions[0][:,2,-1].float()
            for j,e in enumerate(batch):
                item=audit.base.render_v20(e,vocab,'nonthinking')
                weights=attention[j]
                maxpos=int(weights.argmax())
                residual=(captured[0][j]-value_o).norm()/value_o.norm()
                rows.append({'key':e.prompt_sha256,'split':split,'count':e.count,
                             'o_occurrences':int(ids[j].eq(o_id).sum()),
                             'o_attention_mass':float(weights[ids[j].eq(o_id)].sum()),
                             'target_attention_mass':float(weights[list(item.prompt_needle_positions)].sum()),
                             'answer_self_mass':float(weights[-1]),
                             'max_attention_token':vocab.id_to_token[int(ids[j,maxpos])],
                             'context_vs_static_o_relative_error':float(residual),
                             'first_token':vocab.id_to_token[int(output.logits[j,-1].argmax())]})
    pd.DataFrame(rows).to_csv(args.output/'attention_and_context.csv',index=False)
    examples=[lookup[k] for k in protocol['confirmation_keys']]
    static=diag.readout(model,vocab,examples,[[1,2]],'mean',means,50,verify=True)
    static.to_csv(args.output/'static_o_value_readout.csv',index=False)
    clean=pd.read_csv(prior/'readouts/clean.csv').set_index('key')
    cols=[f'count_logit_{n}' for n in range(1,11)]
    difference=(static.set_index('key')[cols]-clean[cols]).abs().max().max()
    result={'static_o_matches_all_clean_first_tokens':static.full_vocab_token.tolist()==clean.loc[static.key,'full_vocab_token'].tolist(),
            'count_logit_max_absolute_difference':float(difference),
            'static_o_full_accuracy':float(static.full_vocab_accuracy.mean()),
            'static_o_numeric_validity':float(static.first_is_number.mean()),
            'static_o_numeric_probability_mass':float(static.p_numeric_1to10.mean()),
            'pre_O_value_o_norm':float(value_o.norm()),
            'attention_summary':pd.DataFrame(rows).groupby('split')[['o_attention_mass','target_attention_mass','context_vs_static_o_relative_error']].agg(['mean','min','max']).to_json()}
    assert result['static_o_matches_all_clean_first_tokens']
    assert difference<1e-3
    audit.write_json(args.output/'results.json',result)
    audit.write_json(args.output/'manifest.json',{'status':'complete','files':{str(s.relative_to(args.output)):audit.digest(s) for s in args.output.iterdir() if s.is_file()}})
    print(json.dumps(result,indent=2),flush=True)
    print('L1H2 FUNCTION PROBE COMPLETE',flush=True)


if __name__=='__main__':
    main()
