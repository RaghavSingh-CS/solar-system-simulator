# Generated validation results

Each launcher creates a marked, immutable, direct child of this directory. The
collector rejects outside directories and creates a sibling ZIP containing
only allowlisted bounded logs, manifests, statuses, diagnostics, numerical
text records, and executable hashes. Generated run directories and ZIP
archives are intentionally ignored by Git.

Failed and partial runs are valuable diagnostics. Failed collection is
transactional and leaves the run unchanged. Keep it and start a new run instead
of deleting or overwriting an earlier result.
