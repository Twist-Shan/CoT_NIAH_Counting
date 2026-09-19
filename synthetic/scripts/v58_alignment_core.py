"""Small, tested intervention primitives for the v58 alignment supplement."""
from __future__ import annotations

import contextlib
import hashlib
import itertools

import numpy as np
import torch
import torch.nn.functional as F

import run_v58_commit_query as base


def fit_space(x, labels):
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    labels = np.asarray(labels)
    classes = sorted(set(labels.tolist()))
    centroids = torch.stack([x[labels == c].mean(0) for c in classes])
    mean = x.mean(0)
    u = torch.linalg.svd(centroids-centroids.mean(0), full_matrices=False).Vh[:3]
    within = x - torch.stack([centroids[classes.index(c)] for c in labels])
    nuisance = within - (within @ u.T) @ u
    candidates = torch.linalg.svd(nuisance, full_matrices=False).Vh
    orth = []
    for row in itertools.chain(candidates, torch.eye(x.shape[1])):
        row = row - (row @ u.T) @ u
        for previous in orth:
            row = row-(row @ previous)*previous
        if float(row.norm()) > 1e-5:
            orth.append(row/row.norm())
        if len(orth) == 3:
            break
    if len(orth) != 3:
        raise ValueError("Insufficient orthogonal nuisance dimensions")
    v = torch.stack(orth)
    assert float((u @ v.T).abs().max()) < 1e-4
    radii = torch.stack([within[labels == c].square().sum(1).mean().sqrt() for c in classes]).clamp_min(1e-8)
    return {"mean": mean, "u": u, "v": v, "centroids": centroids,
            "classes": classes, "radii": radii}


def removal(x, space, kind):
    u, v, mean = [space[k].to(x) for k in ("u", "v", "mean")]
    centered = x-mean
    aligned = (centered @ u.T) @ u
    if kind == "aligned":
        return aligned
    if kind != "orthogonal":
        raise ValueError(kind)
    orth = (centered @ v.T) @ v
    if float(orth.norm()) < 1e-10:
        orth = v[0]
    orth = orth / orth.norm().clamp_min(1e-12) * aligned.norm()
    assert torch.allclose(orth.norm(), aligned.norm(), atol=2e-4, rtol=2e-5)
    assert float((u @ orth).norm()) < 2e-4 + 2e-5*float(orth.norm())
    return orth


def controls_for(bank, top_k, repeats=3):
    selected = bank[:top_k]
    layers = sorted(set(l for l, _ in selected))
    choices = []
    for layer in layers:
        k = sum(l == layer for l, _ in selected)
        available = [h for h in range(8) if (layer, h) not in selected]
        choices.append(list(itertools.combinations(available, k)))
    products = list(itertools.product(*choices))
    # Frozen, spread deterministic draws; never pretend one complement is 3 controls.
    chosen = np.linspace(0, len(products)-1, min(repeats, len(products)), dtype=int)
    return [[(l, h) for l, group in zip(layers, products[i]) for h in group] for i in chosen]


def make_record(example, vocab, mode, split, block):
    item = base.render_v20(example, vocab, mode)
    seq = item.input_ids[:item.spans.ans_pos+1]
    needles = list(item.prompt_needle_positions)
    data_start = min(needles) - example.needle_positions[0]
    ordinary = [p for p in range(data_start, data_start+len(example.seq_tokens)) if p not in needles]
    available = set(ordinary)
    matched_ordinary = []
    for p in needles:
        nearest = min(available, key=lambda q: (abs(q-p), q))
        matched_ordinary.append(nearest)
        available.remove(nearest)
    return {"example": example, "seq": seq, "q": len(seq)-1, "needles": needles,
            "ordinary": ordinary, "matched_ordinary": matched_ordinary, "markers": list(item.spans.trace_marker_positions),
            "think": item.spans.think_pos, "split": split, "block": block,
            "count": int(example.count), "key": example.prompt_sha256, "mode": mode}


def changed_source(record, scope):
    seq = list(record["seq"])
    positions = record["needles"] if scope == "needles" else record["matched_ordinary"]
    # Ordinary corruption changes identities too; preserve positions and token budget.
    donor_positions = record["ordinary"][-len(positions):]
    for p, d in zip(positions, donor_positions):
        alternatives = [j for j in record["ordinary"] if seq[j] != seq[p]]
        if not alternatives:
            raise ValueError("No distinct ordinary replacement token")
        seq[p] = seq[d] if seq[d] != seq[p] else seq[alternatives[0]]
    assert len(seq) == len(record["seq"])
    return seq


class Engine:
    def __init__(self, model, vocab, bank, batch_size=16):
        self.model, self.vocab, self.bank, self.batch_size = model, vocab, bank, batch_size
        self.device = next(model.parameters()).device
        self.head_width = model.config.n_embd // model.config.n_head
        self.count_ids = [vocab.token_to_id[vocab.number_token(n)] for n in range(1, 11)]
        self.spaces = {}
        self.norm_errors = []

    @contextlib.contextmanager
    def hooks(self, records, actions, bank_writes):
        handles = []
        def hidden_hook(layer):
            def hook(module, inputs, output):
                raw = output[0] if isinstance(output, tuple) else output
                value = raw.clone()
                for i, (r, action) in enumerate(zip(records, actions)):
                    positions = action.get("blank", [])
                    if positions:
                        value[i, positions] = 0
                    for l, ps, vectors in action.get("patch", []):
                        if l == layer and ps:
                            value[i, ps] = vectors.to(value)
                    if action.get("late_layer") == layer:
                        value[i, r["q"]] -= removal(value[i, r["q"]], self.spaces[("answer", layer)], action["late_kind"])
                return (value, *output[1:]) if isinstance(output, tuple) else value
            return hook
        handles.append(self.model.token_embedding.register_forward_hook(hidden_hook(0)))
        for l, block in enumerate(self.model.layers, 1):
            handles.append(block.register_forward_hook(hidden_hook(l)))
            heads = [h for ll, h in self.bank if ll == l]
            def prehook(module, inputs, l=l, heads=heads):
                value = inputs[0].clone()
                for i, (r, action) in enumerate(zip(records, actions)):
                    for ll, h in action.get("mask", []):
                        if ll == l:
                            qs = action.get("mask_queries", [r["q"]])
                            value[i, qs, h*self.head_width:(h+1)*self.head_width] = 0
                    if heads:
                        sub = torch.zeros_like(value[i, r["q"]])
                        for h in heads:
                            sub[h*self.head_width:(h+1)*self.head_width] = value[i, r["q"], h*self.head_width:(h+1)*self.head_width]
                        bank_writes[(i, l)] = F.linear(sub, module.weight)
                return (value, *inputs[1:])
            def posthook(module, inputs, output, l=l):
                value = output.clone()
                for i, (r, action) in enumerate(zip(records, actions)):
                    if action.get("ret_layer") == l:
                        vector = bank_writes[(i, l)]
                        space = self.spaces[("retrieval", l)]
                        delta = removal(vector, space, action["ret_kind"])
                        value[i, r["q"]] -= delta
                        bank_writes[(i, l)] = vector-delta
                return value
            handles.append(block.attention.output.register_forward_pre_hook(prehook))
            handles.append(block.attention.output.register_forward_hook(posthook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    @torch.inference_mode()
    def run(self, records, actions=None, capture=False):
        actions = actions or [{} for _ in records]
        if len(records) != len(actions):
            raise ValueError("Action/record multiplicity mismatch")
        results = []
        for start in range(0, len(records), self.batch_size):
            batch, acts = records[start:start+self.batch_size], actions[start:start+self.batch_size]
            writes = {}
            with self.hooks(batch, acts, writes):
                out = base.forward(self.model, self.vocab, [a.get("seq", r["seq"]) for r, a in zip(batch, acts)], hidden=True)
            for i, r in enumerate(batch):
                logits = out.logits[i, r["q"]].float()
                counts = logits[self.count_ids]
                probs = counts.softmax(0)
                expected = float(probs @ torch.arange(1, 11, device=probs.device, dtype=probs.dtype))
                token = int(logits.argmax())
                pred = self.count_ids.index(token)+1 if token in self.count_ids else 0
                other = counts.clone(); other[r["count"]-1] = -torch.inf
                row = {"key": r["key"], "mode": r["mode"], "split": r["split"], "block": r["block"], "count": r["count"],
                       "predicted_count": pred, "accuracy": float(pred == r["count"]), "answered": float(pred > 0),
                       "expected_count": expected, "expected_abs_error": abs(expected-r["count"]),
                       "greedy_abs_error_penalized": abs(pred-r["count"]) if pred else 10,
                       "margin": float(counts[r["count"]-1]-other.max()),
                       "count_probability_mass": float(torch.exp(torch.logsumexp(counts, 0)-torch.logsumexp(logits, 0)))}
                for l in range(5):
                    q = out.hidden_states[l][i, r["q"]].float().cpu()
                    if ("answer", l) in self.spaces:
                        sp = self.spaces[("answer", l)]
                        c = torch.cdist(q[None], sp["centroids"])[0]
                        row[f"answer_ncc_l{l}"] = float(sp["classes"][int(c.argmin())] == r["count"])
                        row[f"answer_centroid_distance_l{l}"] = float(c[r["count"]-1]/sp["radii"][r["count"]-1])
                    if ("running", l) in self.spaces:
                        h = out.hidden_states[l][i, r["needles"]].float().cpu()
                        sp = self.spaces[("running", l)]
                        c = torch.cdist(h, sp["centroids"])
                        row[f"running_centroid_distance_l{l}"] = float((c.diag()/sp["radii"][:len(h)]).mean())
                for (j, l), vector in writes.items():
                    if j == i and ("retrieval", l) in self.spaces:
                        sp = self.spaces[("retrieval", l)]
                        v = vector.float().cpu()
                        row[f"retrieval_centroid_distance_l{l}"] = float((v-sp["centroids"][r["count"]-1]).norm()/sp["radii"][r["count"]-1])
                if capture:
                    row["hidden"] = [h[i, :len(r["seq"])].float().cpu() for h in out.hidden_states]
                    row["writes"] = {l: v.float().cpu() for (j, l), v in writes.items() if j == i}
                if not all(np.isfinite(v) for k, v in row.items() if isinstance(v, float)):
                    raise ValueError("Nonfinite metric")
                results.append(row)
        return results


@torch.inference_mode()
def rank_discovery(model, vocab, records, batch_size=8):
    broad, targeted = torch.zeros(4, 8), torch.zeros(4, 8)
    targeted_n = 0
    for start in range(0, len(records), batch_size):
        batch = records[start:start+batch_size]
        out = base.forward(model, vocab, [r["seq"] for r in batch], attention=True)
        for i, r in enumerate(batch):
            for l, a in enumerate(out.attentions):
                mass = a[i, :, r["q"], r["needles"]].float()
                total = mass.sum(-1)
                p = mass/total[:, None].clamp_min(1e-20)
                entropy = -(p*p.clamp_min(1e-20).log()).sum(-1)
                broad[l] += (total*entropy.exp()/r["count"]).cpu()
                if r["markers"]:
                    qs = [p-1 for p in r["markers"]]
                    targeted[l] += a[i, :, qs, r["needles"]].float().mean(-1).cpu()
            targeted_n += bool(r["markers"])
    def ordered(scores):
        return sorted([(l+1, h, float(scores[l, h])) for l in range(4) for h in range(8)], key=lambda x: (-x[2], x[0], x[1]))
    return {"broad": ordered(broad/len(records)), "targeted": ordered(targeted/max(1, targeted_n))}


def paired_bootstrap(values, blocks, seed=20260905):
    values, blocks = np.asarray(values), np.asarray(blocks)
    means = np.asarray([values[blocks == b].mean() for b in sorted(set(blocks))])
    rng = np.random.default_rng(seed)
    boot = means[rng.integers(len(means), size=(10000, len(means)))].mean(1)
    return {"effect": float(means.mean()), "ci_low": float(np.quantile(boot, .025)),
            "ci_high": float(np.quantile(boot, .975)), "blocks": len(means), "pairs": len(values)}
