# Third-party resources

- **Tiny Shakespeare** is included at
  `synthetic/src/synthetic_counting_v11/resources/tiny_shakespeare/input.txt`.
  Its public source is the
  [char-rnn dataset](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare).
  The source text, URL, and checksum are recorded beside the file. The counting
  model's setup is informed by [nanoGPT](https://github.com/karpathy/nanoGPT).
- **Paul Graham essays** are downloaded by the corpus preparation command from
  the public URLs registered in the supplied RULER URL list. Essay bodies and
  model-generated traces are not bundled. The essays retain their original
  ownership; this snapshot does not relicense them.
- **Models and libraries** (including Qwen, Gemma, other benchmark checkpoints,
  PyTorch, Transformers, and vLLM) are external dependencies. Model identifiers
  and registered checkpoint revisions are preserved in the experiment code.
- **STIX fonts** are embedded in the editable overview artwork. Their license
  is retained in `figures/aurora_attention_pca_concept/assets/fonts/LICENSE_STIX`.
  Rebuilding math labels uses the STIX font supplied by Matplotlib. Times New
  Roman is not distributed; figure scripts can use an installed copy or the
  bundled DejaVu fallback.
- **Trace parser provenance** is recorded by file checksums in
  `realistic/provenance/NIAH_PARSER_V5.json`. The project-associated repository
  locator and source commit identifier have been anonymized; the three frozen
  parser algorithm files are included. No third-party copyright notice has
  been removed from those files.
