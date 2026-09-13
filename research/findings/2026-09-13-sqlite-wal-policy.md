# A larger durable WAL checkpoint trigger reduces bounded continuation time

The [WAL comparison](../../results/systems/sqlite-wal-comparison-20260913.json) passes from clean
revision `9bce74ca5bf626a2cb0eaf0eac58b473ac32577e` under the
[frozen two-policy contract](../flagship/sqlite_wal_comparison_v1.json). The workload, reference prefix,
data order, deduplication policy, and application checkpoint rules are unchanged. SQLite version is
3.53.1; both policies use WAL, `synchronous=FULL`, foreign keys, 4 KiB pages, and the same cache setting.

## Paired complete-invocation results

Execution order is baseline 1 → candidate 1 → candidate 2 → baseline 2. The baseline autocheckpoint
trigger is 1,000 pages; the candidate is 65,536 pages, nominally 256 MiB. Every measured invocation
includes final WAL truncation, publication, and normal final artifact/index verification. Durable
private-prefix restoration and independent parity audits are separate qualification costs.

| Pair | Baseline seconds | Candidate seconds | Complete-time reduction |
| --- | ---: | ---: | ---: |
| Baseline first | 279.890 | 156.961 | 43.9% |
| Candidate first | 326.115 | 152.551 | 53.2% |

Both exceed the predeclared 10% gate. Candidate commit totals are 109.568 and 107.058 seconds, versus
238.142 and 249.989 seconds in the baselines. Candidate publication takes 2.170 and 1.692 seconds;
the gain is not obtained by excluding a large final flush from the score.

Observed WAL peaks are 134,670,472 bytes (**128.43 MiB**) for each baseline and 315,872,192 bytes
(**301.24 MiB**) for each candidate. The recovery run has the same 301.24 MiB peak. All remain under
the frozen 512 MiB observed-space gate. These are checkpoint-boundary physical file-size observations;
autocheckpoint thresholds are triggers, not hard global limits.

All four runs match the original outputs, removal records, counts, and logical SQLite tables. Their
published outputs leave no nonempty WAL. Original data/checkpoints remain intact.

## Recovery genuinely depends on committed WAL

The candidate's worker hard-exits with `os._exit(75)` after its first durable candidate checkpoint.
The checkpoint contains **292,808 committed documents**, while an immutable main-file-only read sees
**288,872**. The **3,936-document difference** establishes that committed candidate data still resides
in WAL. The WAL file is 134,670,472 bytes at that point.

Before exiting, the worker leaves an additional uncommitted private reference-row update and a
31-byte uncommitted output tail. Normal recovery preserves the committed data, discards the unfinished
work, and reproduces every original output and logical table. Reference exclusion and the exact/near
controls pass again. The crash receipt and retained checkpoint are stored with the run artifacts.
This is abrupt-process-exit evidence, not a physical power-loss test.

## Recommendation and boundary

The 65,536-page FULL-synchronous policy is recommended **for this qualified local envelope**. The
production default is not changed by the comparison. Bind the policy explicitly in a successor run
contract rather than relying on an implicit connection default.

The measured candidate groups each contain fewer than 10,000 records, so source-end checkpoints drive
their commit boundaries. Larger within-source production transactions, larger indexes, different
storage/site hardware, and sustained WAL growth need separate qualification. Two reversed-order pairs
are an engineering gate, not a population confidence interval. Do not apply these reductions directly
to the 150B preparation estimate or the full three-month calendar.

The E1/E3 source recipes and category capacity gaps remain an independent blocker. They should be
resolved before bulk corpus expansion, alongside explicit operating-policy and storage-budget binding.
