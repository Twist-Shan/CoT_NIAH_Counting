# Answer state patching

The metric is the **proportion of predictions matching the source count after answer state patching**. It is computed directly from generated counts. Invalid outputs remain in the denominator as failures.

The Non-thinking all-layer analysis retains pairs whose source and target baseline answers are both correct and uses the same pairs at every layer. Pairs are pooled; pointwise confidence intervals resample seed clusters and recompute the pooled ratio in each draw. Same-count and self patches are scored against the corresponding different-count source. No control subtraction is applied.

```bash
python figures/answer_state_source_count_20260921/analyze.py --exports realistic/exports --output runs/answer_source_count/data
python figures/answer_state_source_count_20260921/build_controls.py --data runs/answer_source_count/data --output runs/answer_source_count/figures
```

The input contract is `v4_4_causal_v2_overall_clean_correct/clean_correct/{model}/patching/answer_patching/screen/detail.clean_correct.csv.gz` under the supplied exports directory. The analyzer checks eligibility, pair identity, complete layer coverage, and agreement with the archived exact-match flags. Its audit records input SHA256 hashes, sample counts, invalid outputs, and bootstrap settings. The archive directory name `screen` identifies the saved input; it does not change the reported metric.

The main Non-thinking renderer and appendix renderer import the same analyzer. Thinking and structured-enumeration figures use the same source-count matching label and retain their existing exact-match data. Sample selection differs across experiments; equal metric definitions alone do not make their percentages a controlled comparison between modes.
