# 200 — Held-out evaluation now has a parser-independent pre-results contract

Magic's pretraining lesson makes production formatting a measured part of the distribution rather
than harmless plumbing. The new bounded contract therefore retains the exact production-formatted
`selection_heldout` file and adds a physically separate alternate extraction. It binds each parser's
ID, revision, artifact hash, view-file hash, source ID, source-document hash, and view-text hash. The
two views are paired by source/document identity, scored separately, and never pooled. The
parser-independent equal-category macro is the ranking view, while every category guardrail must pass
in both views.

The six formal categories remain the decision level. Subdomains are emitted beneath each category
with zero independent decision weight, so reporting detail cannot reweight the macro. Score analysis
retains document BPB, the unweighted six-category paired-delta macro, deterministic document
bootstrap, and the existing +0.01 BPB upper paired 95% category guardrail.

One aggregate training ledger plus selection, D5, and E2 ledgers must be globally disjoint by
source-derived identity, source-document SHA-256, and normalized-content SHA-256. The selection and
audit ledgers must reproduce the original firewall's ordered category commitments, so the sealed
payloads stay unopened. The original firewall and its pinned evidence are unchanged and must still
pass before this successor runs.

Normalized sliding 96-character windows and exact token-shingle Jaccard over every pair inside the
declared bound are two additional fail-closed leakage channels. They cannot substitute for or weaken
the existing exact/near-deduplication, consumer, or opening rules. A deterministic short-document
fixture proves Jaccard can fail independently when no 96-character window exists.

The score schema also binds model and backend identity. A small optional helper compares complete
same-model score reports from two distinct backends against a predeclared per-document NLL tolerance;
no external model was run. Seven new fixture tests, 28 focused integration tests, and the complete CPU
suite pass. No real corpus or external data/model service was accessed, and no audit was opened.

This result has no consumer, selection, training, or audit-opening authority. Real bounds, complete
training-manifest identity coverage, production and alternate parser artifacts, and production
firewall authority remain blocked.

Artifacts: [qualification](../results/data/parser-independent-heldout-tooling-20260908.json) and
[pre-results plan](../research/flagship/heldout_evaluation_plan_v1.json).
