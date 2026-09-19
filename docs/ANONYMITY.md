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
- The ZIP uses fixed timestamps and omits `.git/` and generated outputs.

`tools/validate_repo.py` checks source syntax, JSON, entry points, new document
links, common credential patterns, and private path/login patterns. A separate
preparation-time scan also checked the author names and project-specific
identifiers; those identifying search terms are not stored in this repository.
The scan is a bounded check, not a guarantee that code cannot be recognized.

The folder and ZIP are local artifacts. No anonymous hosting URL has been
created. If hosting the code for reviewers, use an anonymous download/view
endpoint; a personal account's repository URL or original Git history can
identify authors even when file contents have been cleaned.
