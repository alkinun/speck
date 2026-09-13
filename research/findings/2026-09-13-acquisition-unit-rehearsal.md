# Raw acquisition units and ordered cohort deduplication recover exactly

The [checked qualification](../../results/systems/acquisition-unit-rehearsal-20260913.json) passes from
clean revision `3a717ce794d5a2b35c685c07f8daf37e7486bf12` under the frozen
[unit plan](../flagship/acquisition_unit_rehearsal_v1.json). Raw files are pinned by repository revision,
filename, upstream SHA-256, and byte size. The executed source/security/benchmark-filter settings come
from the archived 2B calibration contract.

## Acquisition output

Twelve windows cover two disjoint 256-row ranges in each of six source files. Row positions refer to
original Parquet records or gzip lines before filtering. Acquisition preserves reader-exposed metadata,
score, source repository/revision, filename, and absolute row alongside the text and content hash.

| Category | Rows yielded by source reader | Records after downstream filters | Retained UTF-8 bytes |
| --- | ---: | ---: | ---: |
| Web | 512 | 510 | 2,064,550 |
| Code | 164 | 148 | 877,857 |
| Math | 509 | 481 | 2,104,228 |
| Synthetic | 512 | 505 | 1,891,832 |
| Science | 506 | 500 | 18,619,525 |
| Reference | 460 | 457 | 1,222,207 |
| **Total** | **2,663** | **2,601** | **26,780,199** |

The downstream filters remove 29 benchmark-contaminated records, 23 raw email/IP records, eight
duplicate-line records, and two Gitleaks-affected records. These counts follow source reader filtering;
they are not the complete rejection breakdown over all physical rows in the raw files.

Both Parquet/web and gzip/code recovery are exercised. Each is interrupted after 17 reader-yielded
rows and resumes from its durable 16-row checkpoint. Recovered manifests, counts, content, metadata,
and security-report identities match uninterrupted execution. Torn uncommitted writes, corrupted
checkpoint state, and raw-file corruption are additionally covered by portable tests.

## Global deduplication over this cohort

The complete cohort is processed in frozen unit order using normalized exact matching, batched MinHash,
verified-near matching, and one SQLite writer. A lower-precedence replay of the first 256-record unit
is a declared control. All 256 replay records are removed; there are no additional exact or near
removals in the natural cohort. This small result cannot estimate large-corpus duplicate yield.

The deduplication pass sees 2,857 records including the control and retains 2,601. It creates a
3,088,384-byte SQLite index. Interruption after processed record 65, recovery from checkpoint 64,
and completion preserve all output hashes, removal-log identity, counts, and logical SQLite tables.
The checked record retains the physical database identities separately.

## Resource and timing evidence

| Measurement | Observed value |
| --- | ---: |
| Immutable raw files downloaded | 6 files / 7,284,804,457 bytes |
| Cold download time | 1,406.919 s |
| Raw hash-verification time across acquisition units | 7.111 s |
| Benchmark-filter setup | 38.490 s |
| Complete acquisition invocation | 1,483.815 s |
| Clean cohort deduplication, with cProfile | 17.236 s |
| Interrupted dedup invocation | 0.362 s |
| Dedup recovery invocation, without profiling | 13.469 s |
| Qualification main-process peak RSS | 4,248,104,960 bytes |

Source-file granularity dominates cold acquisition even though the selected cohort is small. Retained
raw files can support later windows without repeating the download. This measured cold transfer rate
is not a general source-preparation forecast.

The clean dedup profile attributes 7.527 cumulative seconds to 51 checkpoints and 4.720 cumulative
seconds to 2,583 batched-signature calls. Signature time includes 4.203 seconds in `MinHash.update_batch`.
The 64-record checkpoint interval is deliberately short to exercise recovery; production's historical
interval is 10,000 records. Profile cumulative times overlap, and profiling itself adds overhead.
The observed checkpoint share must not be extrapolated directly to production or interpreted as a
measured benefit from changing cadence or adding workers.

The RSS measurement covers the main qualification process, including benchmark indexes and source
decoding, rather than only the SQLite dedup pass. Gitleaks child-process peak memory is not included.
This is a substantially different memory boundary from the earlier dedup-only calibration peak.

## Boundary and next decision

This execution qualifies raw-unit acquisition and ordered **bounded-cohort** deduplication. It does not
run complete tokenizer/selection/audit firewall reference exclusion. The acquired outputs therefore
remain engineering artifacts and cannot enter training or the firewall-excluded source-bank path.
It also does not establish full E1/E3 source-treatment coverage, final-tokenizer supply, or a production
rate. The six equal-row windows are not a representative token mixture; science dominates retained bytes.

Next, measure larger units at the intended checkpoint cadence and integrate the complete firewall
reference-precedence pass. Use stage-specific memory and throughput evidence to choose concurrency.
Keep the current 150B serial projection and unresolved day-21 readiness gate until that production
envelope is qualified.
