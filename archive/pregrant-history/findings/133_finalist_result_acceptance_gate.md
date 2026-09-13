# 133 — Append-only finalist result-acceptance gate

## One post-event command

The acceptance verifier checks every result currently referenced by the program. For each, it binds
the result path/hash/status, exact run/arm/pair/order, complete finite 11-source history, matching
transition result and successor, frozen automation hash, and no-polling/no-branch/no-retry flags.

At ledger level it enforces the control-first prefix, six-control target boundary, six-candidate
analysis boundary, status/`next_run`, exact transition inventory, and permanent preservation of the
interrupted attempt plus identical rerun. Reference, transition-edge, source-set, and failure-history
adversaries are rejected.

The verifier passes the live empty ledger without inspecting service state or training outputs. It is
append-only and is not wired into the hash-frozen finalizer. It must run after every automatic commit
and before interpreting a result or completing the corresponding Linear state.

## Failure policy

Any failure blocks interpretation. Retain committed bytes and diagnose the exact invariant; do not
rewrite references/transitions, impute sources, retry automatically, skip, branch, or change thresholds.

## Artifact

- [Acceptance qualification](../results/Speck-Paper1/finalist-result-acceptance-qualified-v1.json)

Run it with:

```bash
python -m scripts.paper_finalist_result_acceptance_validate
```
