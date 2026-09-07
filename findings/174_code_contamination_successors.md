# 174 — Stack-Edu and Common Pile benchmark overlaps are removed

## Result

The frozen HumanEval, MBPP, and BigCodeBench v0.1.4 screen now covers the two bounded code sources
that remained behind Stack v3. The successor changes only the accepted parent-report contract; it
reuses the same 2,278 pinned tasks, lexical normalization, task-unique 13-gram critical rule,
10-gram sensitivity disclosure, and deterministic partition policy.

Stack-Edu contains 32 critical files linked to 23 distinct tasks. Removing 241,650 bytes leaves
9,538 files, 22.28 MB in the training partition, and 2.80 MB in evaluation. Fifteen non-critical
10-gram sensitivity records remain disclosed. Common Pile contains 14 critical files linked to 13
tasks. Removing 81,182 bytes leaves 4,952 files, 10.84 MB in training, and 1.10 MB in evaluation;
three sensitivity-only records remain disclosed. Neither source contains a complete normalized
benchmark field under the frozen exact-match rule.

## Evidence boundary

The earlier bounded cross-source comparison found zero exact released-text and zero verified
near-duplicate matches among Stack-Edu, Stack v3, and Common Pile. These new artifacts are strict
record-removal subsets of those inputs, so the operation cannot introduce a cross-source duplicate.
That monotonic argument applies only to these samples. Production packing still requires global
exact and near-duplicate processing, and future benchmark revisions need their own screen.

## Decision

Use only the v4 Stack-Edu and Common Pile artifacts for tokenizer sampling, after manual rights and
attribution approval. The technical bounded-sample contamination gate is complete; training
authority remains blocked on legal acceptance and production operational gates.

Artifacts:

- [Checked successor summary](../results/data/code-contamination-successors-20260907.json)
- [Stack-Edu frozen plan](../research/flagship/stack_edu_code_contamination_v1.json)
- [Common Pile frozen plan](../research/flagship/common_pile_code_contamination_v1.json)
- Runtime Stack-Edu result: `/mnt/speck-data/speck/source-qualification/stack-edu-v4/report.json`
- Runtime Common Pile result:
  `/mnt/speck-data/speck/source-qualification/common-pile-stackv2-edu-v4/report.json`
