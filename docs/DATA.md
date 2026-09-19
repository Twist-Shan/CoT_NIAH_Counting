# Data and generated artifacts

## Included inputs

- City names: `realistic/data/entities/cities.csv`.
- Fact/query templates: `realistic/data/templates/`.
- Registered public essay URL list:
  `realistic/data/haystacks/paul_graham/ruler_paulgraham_urls.txt`.
- Tiny Shakespeare: `synthetic/src/synthetic_counting_v11/resources/tiny_shakespeare/input.txt`.
  Its 1,115,394 bytes have SHA256
  `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.
- The saved synthetic paper-run configuration, experiment JSON designs, and
  parser file hashes.

## Realistic haystack corpus

Run from the repository root:

```bash
python run.py haystack --output-dir data/haystacks/paul_graham_full
```

The output is inside `realistic/`. The downloader deduplicates by content and
records source URLs, content hashes, and included/excluded files. Public pages
can change or become unavailable. A newly downloaded corpus is not automatically
the original frozen corpus; compare its manifest and hashes before claiming
an exact replication. The freezer accepts an explicit corpus manifest path.

The essay bodies and the original full-grid JSONL files are not in this
source-only snapshot. Generation and validation code is supplied; the selected
tokenizer must be available. The simple tokenizer option in some freezers is
for software checks only and changes the scientific input construction.

Legacy V3.1 provenance labels in `realistic_niah_v3_1.integrity` are anonymized;
the expected file sizes and SHA256 checksums remain unchanged. When supplying
the original frozen V3.1 data, call `validate_frozen_dataset` with
`record_source_revision=True` to replace its local `source_revision.json` with
the anonymous labels. The validator checks the data bytes and audit metadata
before writing that source record.

## Synthetic construction

The supplied text makes the basic synthetic workflow independent of a corpus
download. The v58 sampler uses three-character target sets, counts 1–10,
256-character windows, disjoint corpus regions with guard intervals, and
character permutations. The shared implementation is in
`synthetic/src/synthetic_counting_v20/data.py` and `needle_pool.py`.
The main preset is checked against the saved
configuration by the smoke test.

To use another corpus location, set
`SYNTHETIC_COUNTING_TINY_SHAKESPEARE_PATH`. Changing corpus bytes changes the
experiment; keep and report the checksum.

## Outputs required for later analyses

Trained checkpoints, raw model outputs/token IDs, frozen stimulus manifests,
activation caches, selected-head registries, and analysis tables are generated
artifacts. They are not bundled. Run preceding stages before downstream analyses.
If a script expects a historical `work/` or `runs/` input, provide the newly
generated equivalent using its CLI or restore the expected relative layout.

Some frozen selection/configuration files record hashes of historical artifacts.
Those hashes remain provenance requirements, not proof that the artifacts are
present. Do not bypass a mismatch or silently reuse a selection from a different
cohort, tokenizer, model revision, or attention backend.
