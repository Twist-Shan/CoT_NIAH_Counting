"""Exhaustive layer-matched controls with an explicitly protected head set."""
import itertools
import math


def protected_control_plan(selected, protected=((1,2),), heads_per_layer=8):
    selected=set(map(tuple,selected))
    protected=set(map(tuple,protected))
    if not selected or selected&protected:
        raise ValueError('Selected bank must be nonempty and disjoint from protected heads')
    if any(l<1 or not 0<=h<heads_per_layer for l,h in selected|protected):
        raise ValueError('Invalid head index')
    per_layer=[]
    options=[]
    layers=sorted({l for l,h in selected})
    for l in layers:
        inside=sorted(h for ll,h in selected if ll==l)
        outside=[h for h in range(heads_per_layer) if (l,h) not in selected|protected]
        k=len(inside)
        overlap=max(0,k-len(outside))
        choices=(list(itertools.combinations(outside,k)) if overlap==0 else
                 [tuple(sorted(outside+list(s))) for s in itertools.combinations(inside,overlap)])
        options.append(choices)
        per_layer.append(dict(layer=l,selected_count=k,available_nonselected=len(outside),
                              minimum_overlap=overlap,combinations=len(choices)))
    controls=[]
    for groups in itertools.product(*options):
        bank=sorted((l,h) for l,group in zip(layers,groups) for h in group)
        assert not set(bank)&protected
        controls.append(dict(heads=[list(h) for h in bank],overlap=len(set(bank)&selected)))
    return dict(selected=[list(h) for h in sorted(selected)],protected=[list(h) for h in sorted(protected)],
                layer_counts=per_layer,minimum_overlap=sum(s['minimum_overlap'] for s in per_layer),
                actual_unique_controls=math.prod(map(len,options)),controls=controls)
