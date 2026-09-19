import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_v58_native_continuation import choose_layer,continuation_scores,parse_markers


def frame_with_effects(effects):
    rows=[]
    for layer,effect in enumerate(effects,1):
        for offset in [1,-1]:
            for condition,value in [("clean",0),("self_patch",0),("full_donor_patch",effect)]:
                rows.append({"prompt_sha256":"a","pair_id":str(offset),"offset":offset,"layer":layer,
                             "condition":condition,"successor_identity_distinct":True,"donor_vs_receiver_marker_logodds":value})
    return pd.DataFrame(rows)


def test_earliest_95_percent_plateau_excludes_final_block():
    assert choose_layer(frame_with_effects([9.6,10,8,1000]))["selected_layer"]==1
    assert choose_layer(frame_with_effects([9.4,10,8,1000]))["selected_layer"]==2


def test_no_fallback_for_missing_bidirectional_effect():
    frame=frame_with_effects([10,10,10,10])
    frame.loc[(frame.offset==-1)&(frame.condition=="full_donor_patch"),"donor_vs_receiver_marker_logodds"]=-1
    assert choose_layer(frame)["selected_layer"] is None


def test_repeated_markers_do_not_count_as_transfer():
    assert np.isnan(continuation_scores(["a","a"],["a","a"],["a","a"])["donor_continuation_adoption"])
    result=continuation_scores(["a","b"],["a","b","c"],["a","c","b"])
    assert result["distinguishing_horizon"]==2 and result["donor_continuation_adoption"]==1
    assert result["donor_prefix_h2"]==1 and result["donor_prefix_h3"]==0
    assert continuation_scores(["a"],["a","b"],["a","c"])["donor_continuation_adoption"]==0


def test_parser_stops_at_termination_or_broken_grammar():
    assert parse_markers(["<Sep>","<CH_0061>","</Think>","<Ans>","<10>"])==["<CH_0061>"]
    assert parse_markers(["<Sep>","<CH_0061>","<Sep>","<CH_0062>"])==["<CH_0061>","<CH_0062>"]
