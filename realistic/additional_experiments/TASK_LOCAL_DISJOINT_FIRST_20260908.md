# Task-local selection with minimum-overlap random controls

Protocol: `task_local_disjoint_first_20260908_v3`.

## Cohort and random controls

The historical cohort contains 600 unique inputs and 2,400 natural outputs:
kth retrieval and city/flower category counting, Qwen/Gemma, and both output
modes. Each group uses 30 seeds with 10 cases per seed. Seeds 1234-1253 select
heads; seeds 1254-1263 evaluate interventions. These inputs were used in prior
analyses and do not form a new independent confirmation cohort.

Let H be a layer's head count and n its selected-head count. If H-n >= n,
sample n unselected heads. Otherwise take every unselected head and sample
2n-H selected heads. Thus the minimum overlap is max(0, 2n-H). Match each
selected bank's layer counts; do not cap selected Top-K. The three independent
random banks may overlap one another. Broad bank seeds are 7000-7002;
Targeted bank seeds are 6000-6002.

## Measurements and intervention sites

Compute fresh task/model/mode-specific Broad rankings from discovery inputs.
Non-thinking Broad uses complete original record spans. Native Broad in this
v3 protocol uses final tokens of registered generated records; the later
full-span amendment is specified separately in
[NATIVE_BROAD_FULL_SPAN_20260909.md](NATIVE_BROAD_FULL_SPAN_20260909.md).

Native Targeted retrieval uses raw attention mass over the full source record.
Its query is the next-record event: use the post-marker site for Qwen rank-before traces when
defined, otherwise the frozen P0 rule. Construct fresh task-specific discovery
banks. Qwen Broad uses K = 1, 2, 4, 8, 16, 32, 64, 128, and Targeted uses
K = 32, 64, 80, 96, 112, 128. Gemma uses K = 1, 2, 4, 6, 8 for both assays.

Broad ablates once in prefill at the final-answer query, with at most 64 new
tokens. Score the exact kth city-plus-score answer or category count.
Targeted ablation persists from the registered query through decoding, with
at most 256 new tokens. The first valid semantic record must match the next
registered entity. Later corrections and the final answer do not change this
Targeted score.

If there is no semantic record, a fallback may compare the exact target-token
prefix at offset zero only. Freeze original token IDs and character boundaries;
when retokenization does not preserve them, use stable prefix decoding. Record
every fallback. Flower-score audits must guard against incorrectly crediting
the first record.

## Analysis and reproducibility

Evaluate clean, selected and three random conditions at all K on the same
eligible confirmation endpoints. Keep clean-correct results supplementary.
Use seed-paired effects, bootstrap intervals and exact sign-flip tests, applying
Holm correction within each assay/population across the specified models,
tasks, modes and K values. Preserve the statistical settings in the frozen
configuration and executable analysis.

Rerun every v3 condition; do not combine v2 and v3 outputs. The historical full
panel had 6,739 case-by-K points. Fresh packages determine counts from their
eligible plans instead of forcing historical counts. Preserve input/token,
selection, bank-overlap, endpoint, coverage and scoring audits with CSV/JSON
results. `deployment/prepare_task_local.py` constructs historical packages;
`deployment/prepare_fresh_task_local.py` provides the separate fresh-input
route described in [the execution guide](../../docs/WORKFLOWS.md).

This English specification preserves the protocol's technical rules while
omitting internal report-generation notes. It is not byte-identical to the
historical explanatory document; hashes of regenerated packages will differ.

The 2026-09-09 execution audit corrected an earlier explanatory description
that called both Broad modes full-source-span measurements. The frozen
historical copy remains unchanged in archived packages; the executed native
final-token rule is recorded in
`runs/task_local_disjoint_first_20260908_v3/category_broad_scope_audit_20260909.json`.
