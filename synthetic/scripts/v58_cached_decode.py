"""KV-cached decoding equivalent to fixed-query/sustained pre-O ablation.

The original model performs prefill. Subsequent queries retain the actual
intervened KV history, so no previously damaged state is restored. This helper
does not change the head bank, vocabulary, token budget, or scoring.
"""
from __future__ import annotations
import contextlib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

import run_v58_top1to8_aligned as original
from synthetic_counting_v20.model import _apply_rope


def rope_at(x, position, base):
    width = x.shape[-1]
    inverse = 1.0 / (float(base) ** (torch.arange(0,width,2,device=x.device,dtype=torch.float32)/width))
    pos=torch.as_tensor(position,device=x.device,dtype=torch.float32)
    angles = (pos[...,None]*inverse).to(x.dtype)
    if pos.ndim==0:
        cosine,sine=angles.cos()[None,None,None],angles.sin()[None,None,None]
    else:
        cosine,sine=angles.cos()[:,None,None],angles.sin()[:,None,None]
    even,odd=x[...,0::2],x[...,1::2]
    return torch.stack((even*cosine-odd*sine,even*sine+odd*cosine),-1).flatten(-2)


@torch.inference_mode()
def token_rollout(model,vocab,prefixes,heads,scope,budget):
    assert scope in ('sustained','answer_query_only')
    device=next(model.parameters()).device
    lengths=torch.tensor([len(p) for p in prefixes],device=device)
    n=int(lengths.max())
    ids=torch.full((len(prefixes),n),vocab.pad_id,device=device,dtype=torch.long)
    mask=torch.zeros_like(ids)
    for j,prefix in enumerate(prefixes):
        ids[j,:len(prefix)]=torch.tensor(prefix,device=device)
        mask[j,:len(prefix)]=1
    if n+budget > model.config.n_positions:
        raise ValueError('Token budget exceeds positional capacity')
    caches,handles={},[]
    try:
        for li,layer in enumerate(model.layers):
            attn=layer.attention
            assert attn.intervention is None
            def capture(module,args,output,li=li,attn=attn):
                qkv=output.view(output.shape[0],output.shape[1],3,attn.n_head,attn.head_dim)
                _,key,value=(p.transpose(1,2) for p in qkv.unbind(2))
                shape=(len(prefixes),attn.n_head,n+budget,attn.head_dim)
                stored_key=torch.zeros(shape,device=device,dtype=key.dtype)
                stored_value=torch.zeros_like(stored_key)
                stored_key[:,:,:n]=_apply_rope(key,base=attn.rope_base)
                stored_value[:,:,:n]=value
                caches[li]=[stored_key,stored_value]
            handles.append(attn.qkv.register_forward_hook(capture))
        ctx=original.audit._local_attention_edit(model,heads,[[len(p)-1] for p in prefixes]) if heads else contextlib.nullcontext()
        with ctx:
            logits=model(input_ids=ids,attention_mask=mask).logits[torch.arange(len(prefixes),device=device),lengths-1]
    finally:
        for handle in handles:
            handle.remove()
    seqs=[list(p) for p in prefixes]
    done=torch.zeros(len(prefixes),device=device,dtype=torch.bool)
    by_layer={li:[h for l,h in heads if l==li+1] for li in range(len(model.layers))}
    for step in range(budget):
        next_ids=logits.argmax(-1)
        next_ids=torch.where(done,torch.full_like(next_ids,vocab.eos_id),next_ids)
        was_done=done.cpu().tolist()
        for j,t in enumerate(next_ids.cpu().tolist()):
            if not was_done[j]:
                seqs[j].append(t)
        done |= next_ids.eq(vocab.eos_id)
        if bool(done.all()) or step==budget-1:
            break
        hidden=model.token_embedding(next_ids[:,None])
        positions=lengths+step
        extent=n+step+1
        allowed=torch.arange(extent,device=device)[None,:]<=positions[:,None]
        attention_mask=None if bool(allowed.all()) else allowed[:,None,None,:]
        for li,layer in enumerate(model.layers):
            attn=layer.attention
            qkv=attn.qkv(layer.ln_attention(hidden)).view(len(prefixes),1,3,attn.n_head,attn.head_dim)
            query,key,value=(p.transpose(1,2) for p in qkv.unbind(2))
            query=rope_at(query,positions,attn.rope_base)
            key=rope_at(key,positions,attn.rope_base)
            cache=caches[li]
            cache[0][torch.arange(len(prefixes),device=device),:,positions,:]=key[:,:,0]
            cache[1][torch.arange(len(prefixes),device=device),:,positions,:]=value[:,:,0]
            context=F.scaled_dot_product_attention(query,cache[0][:,:,:extent],cache[1][:,:,:extent],attn_mask=attention_mask,dropout_p=0.,is_causal=False)
            if scope=='sustained' and by_layer[li]:
                context[:,by_layer[li]]=0
            projected=attn.output(context.transpose(1,2).contiguous().view(len(prefixes),1,-1))
            hidden=hidden+projected
            hidden=hidden+layer.mlp(layer.ln_mlp(hidden))
        logits=F.linear(model.final_norm(hidden[:,0]),model.unembedding_weight)
    assert all(not layer.attention.qkv._forward_hooks and not layer.attention.output._forward_pre_hooks for layer in model.layers)
    return seqs


@torch.inference_mode()
def generate(model,cfg,vocab,records,examples,heads,scope,batch_size,budget,verify=False):
    rows=[]
    for start in range(0,len(records),batch_size):
        batch=records[start:start+batch_size]
        seqs=token_rollout(model,vocab,[vocab.encode(r['prefix_tokens']) for r in batch],heads,scope,budget)
        for r,seq in zip(batch,seqs):
            tokens=vocab.decode(seq)
            parsed=original.audit._parse_generation(tokens,vocab,examples[r['key']],r['mode'])
            continuation=tokens[len(r['prefix_tokens']):]
            marker=original.marker_outcome(continuation,r['expected_marker']) if r['mode']=='thinking' else {}
            rows.append({k:v for k,v in r.items() if k not in ('prefix_tokens','clean_full_tokens')} |
                        {'scope':scope,**parsed,**marker,'first_token':continuation[0],
                         'eos_reached':float(tokens[-1]=='<EOS>'),'new_tokens':len(continuation),
                         'generation_capped':float(len(continuation)>=budget and tokens[-1]!='<EOS>'),
                         'answer_and_eos_correct':parsed['ar_accuracy']*float(tokens[-1]=='<EOS>'),
                         'joint_marker_and_count_failure':float(marker['next_marker_correct']==0 and parsed['ar_accuracy']==0) if marker else np.nan})
    return pd.DataFrame(rows),{}
