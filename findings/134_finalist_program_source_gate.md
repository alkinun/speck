# 134 — Complete finite source coverage moved before commit

## Safe validator-only integration

The core Paper 1 program validator is not hash-pinned by the active automation. The trainer, runner,
analyzer, v2 plan, data, and thresholds are. This permits a strictly stronger evidence check without
changing scientific execution.

`_validate_finalist_evidence` now validates each accepted result's exact run/arm/pair identity, five
validation points, final-history equality, exact 11-source set, and finite non-boolean source losses.
The frozen finalizer already calls this validator after collecting/updating the program and transition,
but before commit and successor scheduling. Missing, extra, or non-finite source evidence therefore
stops the chain before acceptance rather than only blocking later interpretation.

## Preserved independent audit

The append-only result-acceptance verifier remains mandatory after a successful commit. It independently
checks hashes, source coverage, transition edges, target/analysis state, and failure history. Neither
gate permits imputation, deletion, rewriting, retry, skipping, or branching.

## Decision

The strengthened empty program and all existing program tests pass. No live output was accessed and no
frozen scientific or automation identity changed.

## Artifact

- [Program source gate](../results/Speck-Paper1/finalist-program-source-gate-v1.json)
