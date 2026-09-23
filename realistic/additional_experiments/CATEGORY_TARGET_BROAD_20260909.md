# Category-target Broad supplement

Protocol: `category_target_broad_20260909_v1`.

This exploratory follow-up measures Non-thinking Broad retrieval over records
of the requested category only. It reuses the existing category-counting inputs;
it is not an independent confirmation cohort. The original prompt, gold answer,
tokens and final-answer query remain unchanged.

## Inputs and score

Each model has 300 inputs, evenly split between city and flower targets.
Discovery uses seeds 1234-1253 (200 inputs); confirmation uses seeds 1254-1263
(100 inputs). Select complete original record spans with
`record.category == case.target_category`, checking `is_target` and the gold
count. Let J be the number of target-category records (1, 3, 5, 7 or 9), M their
total attention mass and H the entropy of the normalized record masses. The
Broad score is B = M * exp(H) / J. Average inputs within each discovery seed,
then weight seeds equally across the combined category task. Recompute the
discovery ranking for this score; do not reuse all-record Broad rankings.

## Interventions and controls

Qwen uses K = 1, 2, 4, 8, 16, 32, 64, 128; Gemma uses K = 1, 2, 4, 6, 8.
Three layer-matched random banks use seeds 7000-7002. Prefer banks disjoint
from selected heads, with only the minimum unavoidable overlap. Zero selected
head outputs before the output projection at the final-answer query in one
prefill pass. Generate at most 64 new tokens. Retain the registered SDPA
backend and model/tokenizer revisions. Regenerate clean, selected and random
conditions and score the exact category count.

The canary uses four cases per model: city/flower counts 1 and 9 at the first
confirmation seed, evaluated at minimum and maximum K. The complete panel has
1,300 case-by-K points. Preserve input, ranking, control and result hashes.

## Analysis

The primary effect is Random minus Selected; Clean minus Selected is secondary.
Use paired seed bootstrap with 20,000 draws, seed 20260907, and pointwise 95%
intervals. City/flower subgroups are descriptive; clean-correct results are
supplementary. Use exact two-sided seed sign-flip tests and Holm correction
over the 13 K comparisons across the two models, separately for the primary
and clean-correct populations. Do not combine this family with the earlier
52-comparison Broad family. Compare old and new scores on matched cases and K
with identical clean outputs. Selection of a peak K after confirmation is
exploratory, without selection-adjusted confidence intervals.

## Reproduction

Run `deployment/prepare_category_target_broad.py` and the associated
`test_category_target_broad.py` checks. Keep the package, discovery, bank,
canary, full-run and numerical-analysis outputs in the run directory.

The local verification option `--score-comparison ulp8` permits at most eight
binary64 ULPs for nonnegative finite scores; zeros must match exactly. Rankings,
Top-K banks and controls must still match exactly. This cross-platform numeric
comparison policy was adopted after completion; it does not establish bitwise
identity with historical discovery. Preserve the numerical-comparison audit
and do not use tolerance to accept a changed ranking or cohort.
