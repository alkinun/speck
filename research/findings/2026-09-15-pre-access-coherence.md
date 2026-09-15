# Pre-access coherence audit — 2026-09-15

The [checked audit](../../results/systems/pre-access-coherence-audit-20260915.json) finds the selected
long-context contracts internally consistent across their checked budget, study, tokenizer, model,
evaluation-length and evidence-status boundaries. Its result is
`pre_access_contracts_consistent_execution_pending`, with training authority false. This is a static
pre-access check, not exhaustive software verification, an independent novelty review or GPU evidence.

The current program uses a 1.2B-class hybrid with 320B default base tokens and a conditional 400B
stretch. The 5,000-hour plan balances 4,111 nonreserve hours plus 889 protected reserve; it represents
52.08 full four-GPU-node days within a conditional 90-calendar-day allocation. The review response
supplied by the owner confirms continued evaluation, not an award or access date.

The study preserves both architectures and both supervision conditions, six core parents and twelve
branches over three paired seeds, plus a disjoint single transfer seed block. The six checked R0
shapes preserve exact parameter/vocabulary accounting; all actual GPU fit, throughput, backward,
resume and DDP fields remain null. All three paper claims remain planned without supporting outcomes.
These checks do not establish attainable training endpoints, model quality or useful context.

The preparation queue and local schedule had pivot notices but still contained active-looking
instructions for the retired wave and a stale code-worker pause. Their [previous bytes](../history/2026-09-15-readiness-cleanup/manifest.json)
are preserved against revision `628c82b`. The [queue](../flagship/PREPARATION_QUEUE.md) and
[schedule](../flagship/LOCAL_PREPARATION_SCHEDULE.md) now lead to bounded R0 qualification, successor
supply/exposure accounting, coherent units and family-separated task/evaluation preparation.
The findings index also distinguishes earlier open tokenizer decisions and schedule forecasts from
current successors. Original dated findings and measured results remain unchanged.

Both finite preparation services were active during this review: FineWeb-Edu exclusion and the
ordered Stack-Edu acquisition. Their progress is not a completed eligible-capacity result. No worker
was restarted or extended, and no new download or model training was launched by this audit.

## Validation and reproduction

Implementation: `db44e5d` for the audit, `a54669a` for the preparation entry-point rewrite. The report
binds selected inputs and its implementation files by SHA-256. Focused validation has 25 tests,
including rejection of budget drift, lost factorial cells, duplicate/overlapping seeds, context and
vocabulary disagreement, invented GPU measurements, premature paper evidence and an assumed access
date. Original entry-point bytes are checked against Git; source identity mapping supports relocated
checkouts and rejects paths outside the original checkout.

Full software validation passed **1,244 tests**, with 10 skipped and 125 deselected, plus formatting,
lint, catalog, archive and source-pin checks. After adding implementation hashes to the report, all
25 focused tests and format/lint checks passed again. These results do not include site-only GH200
or sealed evidence tests.

```bash
uv run --no-sync python -m scripts.readiness_audit
uv run --no-sync python -m scripts.readiness_audit --output /absolute/path/to/new-audit.json
```

The command refuses an existing output. Preserve the checked report when contracts or allocation
status change; publish a successor and update the phase-specific audit when new evidence warrants it.

## Remaining readiness gates

The bounded R0 executor, site dependencies and actual GPU qualification remain pending. Before study
execution, finish measured source capacity/exposure and delivery, coherent units and qualified packing,
family-separated pilot/development/final tasks, pinned benchmarks/scorers, numerical floors/margins,
optimizer settings, attainable endpoints, cost forecasts and the transfer route. Complete independent
novelty/design review and qualify launch compatibility for the new partitions. Preparation decisions,
source-use approvals and final model-launch authority remain separate.
