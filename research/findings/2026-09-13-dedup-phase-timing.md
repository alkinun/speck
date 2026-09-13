# SQLite commit dominates the measured candidate continuation

The [v2 timing qualification](../../results/systems/dedup-phase-timing-v2-20260913.json) passes from
clean revision `bfddfba`, using the frozen [replay contract](../flagship/dedup_timing_replay_v2.json).
It restores the previously qualified full-reference checkpoint into a private destination, then
processes exactly the same candidates. All outputs, removal records, and counts match the original
integration; reference preservation, exact/near controls, and zero exact overlap still pass.

The original reference build is not repeated. The parent dataset and checkpoint remain unchanged.
Only the destination contract is rebound; input order, source identities, policy, and the 10,000-record
checkpoint interval remain the same.

## Corrected measurement boundary

The [first diagnostic](../../results/systems/dedup-phase-timing-20260913.json) is retained. It found
large checkpoint costs, but copied prefix files were not explicitly fsynced before timing. V2 flushes
and fsyncs all restored prefixes and the index first, and splits checkpoint operations before assigning
the bottleneck. The difference between these runs is not a pipeline speedup claim.

Durable restoration takes **68.577 seconds**, outside the measured continuation. Restoring committed
state prunes a private database copy; it is not a stress test of rolling back a large uncommitted suffix.
The resulting cache state and allocated index layout are explicit limitations.

## Disjoint wall-time phases

| Phase | Seconds |
| --- | ---: |
| Setup | 0.001 |
| Input identity verification | 0.816 |
| Resume-state and reference-index verification | 1.005 |
| Processing, including checkpoints | 268.983 |
| Publication | 1.147 |
| Final artifact/index verification | 6.330 |
| **Total measured continuation** | **278.281** |

The six candidate sources process 20,770 records and 183,374,777 UTF-8 bytes. Their non-checkpoint work
takes **30.992 seconds** and their checkpoints take **237.788 seconds**. Processing includes reads,
parsing, hashing, candidate comparisons, and index updates; it is not kernel-only time.

Across all checkpoint calls, including final/control checkpoints:

| Checkpoint component | Seconds |
| --- | ---: |
| SQLite `connection.commit()` | **236.737** |
| Output flush/fsync | 0.747 |
| Committed-slice hashing | 0.065 |
| State publication | 0.507 |

These components overlap the phase totals and must not be added to them. SQLite commit alone consumes
about **85%** of total continuation time. This measurement locates the cost in the commit call; it does
not isolate which internal WAL, synchronization, or checkpoint operation causes it.

Main-process peak RSS is 357,007,360 bytes. This replay does not reconstruct benchmark-filter indexes
or run raw acquisition, so it must not be compared as a memory improvement over the earlier full
acquisition/integration process peak.

## Consequence

The next measured optimization target is SQLite commit/journal/checkpoint behavior on the qualified
reference-index workload. A bounded policy comparison should retain durable commits, record WAL/index
space, and require recovery plus identical output/removal/count results. Any gain needs its own
measurement; no production-rate multiplier follows from this diagnostic.

The capacity review separately shows that exact E1/E3 recipes and per-category supply still need to be
resolved. Keep the 150B serial forecast unchanged until both the operating envelope and actual source
mix/capacity are qualified. The earlier streaming-resume memory probe stays bound to its original code
hash; this replay is the execution qualification for the instrumented successor, not a replacement
Python-allocation measurement.
