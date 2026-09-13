# Durable SQLite WAL policy comparison

The [frozen comparison](sqlite_wal_comparison_v1.json) tests one connection-policy change on the
qualified reference-index workload. The preceding [phase timing](SCREEN_CAPACITY.md) attributes about
85% of continuation time to SQLite commit, but does not identify its internal cause.

## Paired experiment

| Policy | WAL autocheckpoint trigger | Required synchronization |
| --- | ---: | --- |
| Baseline | 1,000 pages, nominally 4,096,000 bytes | `synchronous=FULL` |
| Candidate | 65,536 pages, nominally 256 MiB | `synchronous=FULL` |

Both require WAL mode, 4 KiB pages, foreign keys enabled, and the same 10,000-record/source-boundary
application checkpoints. No durability operation is disabled. The policy override is scoped to the
comparison process; this experiment does not change production defaults.

The order is **baseline 1 → candidate 1 → candidate 2 → baseline 2**. Each invocation restores and
fully fsyncs a fresh private copy of the same complete reference checkpoint. Inputs, dedup policy,
source order, and candidate stream remain fixed. Restoration and independent parity-audit time are
recorded separately from the complete preprocessing invocation.

The score includes **final WAL truncation, publication, and final artifact/index verification**.
Moving time from commit to finalization is not a gain by itself. Outputs, removal records, counts,
and logical SQLite tables must match the original qualified pass for every arm. A completed output
must leave no nonempty WAL.

Recommendation requires at least a **10% reduction in complete invocation time in both paired orders**,
all parity checks, hard-crash recovery, and no observed WAL file above **512 MiB** in any measured arm
or recovery run. This is an engineering operating-envelope gate, not a confidence interval.

WAL file sizes are sampled before and after application checkpoint commits, including the final
commit before truncation. Autocheckpoint thresholds are triggers, not hard size caps: transactions
and readers affect when checkpointing can complete. Observed bounded behavior on this workload does
not establish a global or production-scale WAL bound.

## Hard process-crash probe

The candidate receives a separate recovery run. At the first completed candidate-source checkpoint
whose committed document count exceeds the count visible in the main database **without WAL**, the
worker leaves an additional uncommitted SQL update and output tail, records the checkpoint, and exits
with `os._exit(75)`. This bypasses normal connection cleanup and proves that recovery actually needs
committed WAL frames, rather than only an already-flushed main file.

The parent process must recover normally with the candidate settings and reproduce every original
output, removal record, count, and logical SQLite table. Reference exclusion and both controls are
checked again. This tests abrupt process exit, not physical power loss. The uncommitted probe writes
exist only in a private restored runtime; original data and checkpoints remain untouched.

## Execution and retained evidence

From a clean implementation commit and fresh output paths:

```bash
uv run --no-sync python -m scripts.dedup_wal_compare \
  research/flagship/sqlite_wal_comparison_v1.json \
  results/systems/sqlite-wal-comparison-20260913.json
```

The driver retains each restoration, run result, actual SQLite settings, checkpoint-boundary WAL/index
sizes, phase timings, parent logical identities, hard-crash receipt/checkpoint, recovery result, and
progress. SQLite version and implementation revision accompany the report. The result applies to
this local, cache-warmed, bounded continuation and requires separate production-size/site qualification.
