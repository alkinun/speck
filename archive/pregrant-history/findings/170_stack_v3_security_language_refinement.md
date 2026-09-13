# 170 — Stack v3.1 passes security, language, and tokenizer-partition refinement

## Pinned scanner and license filter

The bounded Stack v3.1 artifact from finding 169 was scanned with the official Gitleaks v8.30.1
Linux binary after its release archive checksum passed. The fully redacted JSON report contains 37
findings across 19 records: 28 `sourcegraph-access-token` and nine `generic-api-key` detections. The
refiner maps report line numbers back to released-content SHA-256 identities and excludes all affected
records without copying secret values into the checked result.

A conservative engineering allowlist admits twelve common permissive SPDX identifiers. It is a data
filter, not legal advice or final rights acceptance. Python comments/docstrings, C-style comments,
Shell/SQL comments, and Markdown are classified with pinned `py3langid==0.3.0` when at least 80
alphabetic characters are present. String literals remain a disclosed limitation.

## Fail-closed expansion

The first 70 MB refinement retained 50,577,172 training bytes and 5,296,197 evaluation bytes, missing
both declared targets. It remains preserved. None of its filters changed. A hash-bound overlay reused
the same twelve local shards and expanded only the raw language-balanced extraction to 85,641,453
bytes; Gitleaks found the same 37 findings in the expanded artifact.

The successor rejected 19 scanner-affected records, 1,318 records outside the engineering license
allowlist, and 3,044 records with non-English extracted prose. It retained 17,172 files across 1,454
repositories: 61,186,107 bytes in the exact downstream tokenizer training partition and 6,422,843 in
evaluation, clearing the 55M and 5.5M targets.

## Decision

Restricted Stack v3.1 can technically supply its tokenizer allocation. Training authority remains
blocked on manual legal/attribution acceptance, cross-source near-duplicate analysis, benchmark
decontamination, and acquisition cleanup/resume qualification. The failed 70 MB refinement and the
passing successor are both permanent evidence; thresholds and filters did not move after output.

Artifacts:

- [Refinement summary](../results/data/stack-v3.1-refinement-20260907.json)
- [Passing refinement plan](../research/flagship/stack_v3_refinement_v2b.json)
- Runtime result: `/mnt/speck-data/speck/source-qualification/stack-v3.1-v2b/report.json`
