# Anonymous review snapshot

The review snapshot is prepared according to the anonymous supplementary-code
principle in the [ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).
The local preparation includes these concrete measures:

- New Git repository with anonymous local identity, no remote, and no inherited
  source commit history.
- Project-associated author/repository locators, server login addresses, and
  private absolute paths removed or replaced by relative artifact locations.
- No credentials, SSH configuration, agent instructions, environment files,
  personal notebooks, run logs, model weights, or activation dumps in the release.
- The manuscript PDF is excluded because the supplied copy identifies its authors.
- External model revisions, public corpus references, and third-party notices
  are retained where needed for reproducibility and attribution.
- The ZIP produced by `tools/package_release.py` uses fixed timestamps and omits
  `.git/` and generated outputs. Hosting platforms may generate a different ZIP
  with their own timestamps and checksum; audit the actual download separately.

`tools/validate_repo.py` checks source syntax, JSON, entry points, new document
links, common credential patterns, and private path/login patterns. A separate
preparation-time scan also checked the author names and project-specific
identifiers; those identifying search terms are not stored in this repository.
The scan is a bounded check, not a guarantee that code cannot be recognized.

Hosting is managed separately from this source snapshot. When using an anonymous
review endpoint, check both its visible file tree and downloadable archive after
each update. A source archive does not establish the identity or remote settings
of an upstream Git repository. Personal-account URLs and original Git history
can identify authors even when current file contents have been cleaned.

Run-generated manifests, reports, and logs are outside this source audit. Some
runners record hostnames, resolved paths, and command lines for local provenance.
Before publishing outputs, inspect them separately, remove machine-specific
identifiers, and retain model revisions, configurations, seeds, and file hashes.
