# Native Broad full-span supplement

Protocol: `native_broad_full_span_20260909_v1`.

This supplement applies to kth retrieval and category counting for Qwen and
Gemma. Native Broad keys cover the entire generated record span, replacing
the final-token keys of the earlier protocol. It retains the original
final-answer query prefix and original generated token IDs.

## Geometry and selection

Use `parse_trace_record`, `align_trace_sites` and `_visible_item_spans` to locate
the registered generated records. `literal_token_start/end` must reproduce the
exact original token/text span. Exclude an unrepresentable span; do not rewrite
tokens or approximate a boundary. J is the number of valid spans. If none are
available, mark the query unavailable. Do not add previously unavailable
queries through this amendment.

`_broad_span_metrics` uses epsilon 1e-12. M is total record attention mass, H
the normalized record-mass entropy, and B = M * exp(H) / J. Category counting
uses all registered records, without filtering to the requested category.
Average within each seed and then equally across discovery seeds.

Each model/task retains its original 300 cases: 200 discovery cases with seeds
1234-1253 and 100 confirmation cases with seeds 1254-1263. Qwen uses
K = 1, 2, 4, 8, 16, 32, 64, 128; Gemma uses K = 1, 2, 4, 6, 8. Random-bank
seeds are 7000-7002; retain layer matching and minimum necessary overlap.

## Execution

Perform a tokenizer-only geometry audit of the 1,200 plans first. Keep a shared
valid cohort across candidate heads, with support from all 20 discovery and
10 confirmation seeds. The canary has 16 case-by-K points and 80 condition
outputs: two cases per model/task group at minimum and maximum K. The maximum
full panel has 2,247 case-by-K points and 11,235 outputs. The actual panel is
determined by geometry availability, never by observed intervention outcomes.
Audit endpoint sets and coverage explicitly.

Intervene once during prefill at the final-answer query, before the output
projection, and generate at most 64 new tokens. Regenerate clean, selected and
all three random conditions on the same panel. Keep the registered backend,
model revisions, prompts and token IDs. Run
`deployment/prepare_native_broad_full_span.py` to construct the package.

## Analysis and verification

Random minus Selected is the main comparison. A peak K selected from
confirmation results is exploratory. Apply exact seed-paired sign-flip tests
and Holm correction across 26 model/task/K comparisons, with a separate family
for the clean-correct population. Use 20,000 paired seed-bootstrap draws with
seed 20260907 and pointwise 95% intervals; these do not adjust for selecting K.

The local `ulp8` comparison permits at most eight binary64 ULPs for finite
nonnegative attention scores, with exact zeros and unchanged rankings, banks
and controls. GPU verification retains its exact-comparison requirement.
Preserve input, rank, control, scoring and numeric-comparison audits alongside
the numerical results. HTML completion pages are not required for reproduction.
