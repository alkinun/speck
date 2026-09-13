# 137 — Exact append-only Git provenance for every finalist event

## The remaining post-commit boundary

Checkpoint replay proves that the current result derives from the retained frozen run, but it does not
prove how the result entered history. Temporary Git repositories reproduce five states that v2 accepts:
the correct current files are uncommitted, the event commit contains an unrelated file, its subject is
ambiguous, a result is rewritten later with consistent reference hashes, or a transition is modified
only in the worktree.

## Append-only v3 acceptance

V3 first runs every v2 ledger/source/transition/checkpoint replay. It then locates each transition's Git
commit and requires the result and transition to be added together exactly once, their working bytes to
match the commit, the exact count-dependent subject and file set, and a program snapshot containing the
precise accepted prefix and successor. The target lock and final analysis are required only in their
respective boundary commits and must also be write-once. Transition commits must occur in frozen
ancestry order, while unrelated research commits between events remain permitted.

All five counterexamples fail and the exact three-file first-control event passes. V1 and v2 remain
immutable. The frozen runner, collector, analyzer, program, and training process are unchanged.

The distinct terminal-systemd provenance limitation remains deferred exactly as Finding 129 records;
closing it mid-sequence would change the frozen runner. Valid scientific artifacts remain the current
success authority.

## Artifact

- [Append-only Git acceptance qualification](../results/Speck-Paper1/finalist-result-acceptance-qualified-v3.json)

Run it with:

```bash
python -m scripts.paper_finalist_result_acceptance_validate_v3
```
