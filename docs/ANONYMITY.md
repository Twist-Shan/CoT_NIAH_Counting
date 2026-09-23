# Anonymous source release

The [ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
require supplementary material and review code links to preserve anonymity.
The checked release contains experiment code, configurations, input
specifications, tests and paper-figure sources. Internal narrative reports,
author-identifying manuscript files, private paths, credentials and generated
logs/caches are excluded.

`tools/validate_repo.py` checks syntax, JSON, entry points, document links,
common credential/private-path patterns and Han characters, including escaped
string values. The release preparation also scans known project identities
and embedded artwork metadata; the identifying search terms stay outside
this repository. Public model IDs, corpus references and required third-party
attribution remain. Third-party authorship is not project authorship.

`tools/package_release.py` exports the validated source with a checksum manifest
and fixed ZIP timestamps, excluding `.git`, bytecode and run artifacts. It
does not anonymize a development repository's commit history or remote. For a
review Git repository, create a new repository from the exported files with
an anonymous commit identity and no inherited history. Never upload the
development `.git` directory or audit backups.

Anonymity checks are bounded and cannot guarantee that recognizable research
content will not suggest an identity. Verify the hosting account, visible
repository metadata and actual downloadable archive separately before sharing
a review URL. Local source checks do not audit a website.

Generated outputs require a separate audit: some experiment runners record
resolved paths, hostnames and command lines for local provenance. Retain
scientific settings, revisions, seeds and data hashes when removing machine
identifiers from outputs that will be published.
