# Reproduction scope

This release includes the synthetic training code and saved main configuration,
realistic data construction and model experiments, parsing and intervention
code, numerical analysis, frozen protocol amendments, tests, and the source map
for 46 manuscript figure assets in `figures/paper_assets.json`.

Large pretrained weights, generated stimuli, natural generations, checkpoints,
activation arrays, result tables and rendered figures are not bundled. Follow
[WORKFLOWS.md](WORKFLOWS.md) and [DATA.md](DATA.md) to construct inputs and run
the required stages. Figure builders consume measurements from those stages;
the source map does not establish that every GPU experiment has been rerun.

Presentation-only HTML/Markdown report builders, their exclusive rendering tests,
one-off completion pages and the unused vendored Plotly bundle are omitted.
Shared numerical helpers formerly located in report-named modules are retained
under their existing import names. Count-chain evidence can be serialized as
audited JSON. Full-panel geometry exports the original `DUAL` coordinate data
to JSON, and the cue appendix can read the analysis JSON directly.

Earlier versioned packages remain where imported by later training, analysis,
tests or figure code. Deleting them based only on version or filename would
break those dependencies. The earlier PCA transfer pilot is outside the
released paper workflow. Protocol amendments retain their distinct cohorts,
selection rules and interpretation limits; they are not interchangeable.

See [VALIDATION.md](VALIDATION.md) for the checks actually performed. CPU tests
and source checks do not certify numerical agreement of a complete GPU rerun.
