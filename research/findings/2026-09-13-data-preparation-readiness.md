# Production-data readiness: serial schedule and bounded-memory resume

The current calibration supports a conditional preparation branch, not a verified day-21 corpus-ready
date. The [critical-path assessment](../flagship/DATA_PREPARATION.md) records the dependencies, current
machine capacity, mixture-supply calculation, and next bounded implementation step.

## Supported engineering result

The maintained preprocessor now rebuilds its accepted-document resume chain by streaming the ordered
SQLite cursor. The removed `fetchall()` allocated Python objects proportional to the entire accepted
index. The successor preserves both the ordered hash recurrence and row-count check.

The [bounded read-only probe](../../results/systems/production-dedup-resume-streaming-20260913.json)
binds the modified implementation and the original 2,907,049,984-byte calibration database by SHA-256.
It finds identical old/new chains at 10K and 100K rows. At 100K, traced Python peak allocation is
29,002,128 bytes for the old algorithm and 1,808 bytes for streaming. These are Python allocation
measurements; they do not measure total resident memory or production throughput.

Tests additionally check resumed/uninterrupted output parity, count/hash corruption rejection, and
execution through a cursor that rejects materialization. This evidence qualifies the bounded resume
change only; full reorganized-runtime and production-scale qualification remain open. The original
calibration remains tied to its archived implementation.

## Planning consequences

- The historical 150B serial estimate is 454.22 hours. Starting at allocation day 4 projects completion
  at day 22.93, before verification and site delivery. The current day-21 target has no qualified data
  schedule yet.
- Acquisition already counts tokenizer-specific tokens; starting a genuinely tokenizer-independent
  path needs an explicit successor quota definition.
- Covering each E2 category's maximum share at a 150B pool scale requires 214.5B tokens across the
  separate banks. It exceeds the accepted branch and is not a preparation authorization.
- A fixed prior-sized bank can cause materially different per-domain repetition after mixture
  selection. Both stable and decay exposure must be checked against selected-source supply.

These consequences narrow operational readiness. They do not change the model, statistical decisions,
compute allocation, or paper claim status.
