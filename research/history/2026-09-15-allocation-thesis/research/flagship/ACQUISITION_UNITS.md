# Bounded upstream acquisition units

The [v1 rehearsal plan](acquisition_unit_rehearsal_v1.json) freezes twelve physical row windows from
six raw source files: `[0, 256)` and `[256, 512)` from each. The repository revisions, readers,
filtering/security rules, benchmark contamination policy, and deny ledger come from the checked 2B
calibration plan. Each raw file also has its exact upstream LFS SHA-256 and byte size. Metadata was
queried at those pinned revisions before this execution.

This upstream rehearsal exercises acquisition and cohort deduplication. It complements the retained,
firewall-excluded [source-bank rehearsal](SOURCE_BANK.md), whose input domain is different.

## Unit and recovery semantics

- Windows address original physical rows, before source score/language/length filters. Parquet row
  numbers and gzip line numbers stay absolute; windows never become "the next N accepted records."
- Acquisition retains text, content hash, score, reader-exposed metadata, repository revision, filename,
  and original row. It makes no tokenizer calls. Fields not exposed by the frozen reader are not
  reconstructed or implicitly retained.
- Each unit owns its configuration, unscanned checkpoint text, final scanned records, security report,
  manifest, and per-invocation attempt records. Completed units can be reused independently.
- Selection checkpoints every 16 reader-yielded rows, including those rejected by the downstream
  security/contamination filter. A checksum binds state and configuration; committed text is verified
  before an uncommitted tail is truncated. Resume begins at the recorded next physical row.
- Source/security/benchmark filters run again for the unfinished portion. Gitleaks operates on a copy
  of completed unscanned text so a scan interruption cannot corrupt acquisition recovery state.
- Retained raw files are verified by full SHA-256 before use. Gzip restart still traverses the compressed
  prefix, and the current Parquet reader iterates earlier batches before its window. Windows bound
  output and filtering work; they do not establish constant-time random access.

The twelve windows require six full raw files, approximately **7.28 GB**. Cold acquisition therefore
includes their full download cost despite small selected windows. Raw files are retained for reuse;
reports distinguish actual downloads from cache hits and hash-verification time. Source-filtered row
counts are not the number of physical records examined, and a completed window is not a source-capacity
measurement.

## Ordered cohort deduplication

The complete bounded cohort is deduplicated in frozen unit order using the existing normalized exact
and verified-near policy, one SQLite writer, batched MinHash signatures, and the streaming resume-chain
verifier. A final lower-precedence replay of the first unit is a declared engineering control. Its
output must be empty; its records are counted separately when interpreting natural-cohort removals.

The clean dedup pass is profiled to locate costs. Its timing includes cProfile overhead. The recovery
pass uses the ordinary path, interrupts after record 65, resumes from the record-64 checkpoint, and
must match outputs, removals, counts, and logical SQLite tables. Physical database page hashes need not
match across resume.

**Full tokenizer/selection/audit firewall reference exclusion is not part of this bounded cohort.**
These outputs are engineering artifacts and cannot enter training or the current firewall-excluded
source-bank materializer. This qualification does not replace the 150B fallback, establish complete
E1/E3 treatment supply, or update the production-throughput projection. The production successor must
include the complete reference precedence and exclusion pass.

## Execution

Prepare or resume the units, retaining per-unit costs as each finishes:

```bash
uv run --no-sync python -m scripts.acquisition_units_prepare \
  research/flagship/acquisition_unit_rehearsal_v1.json \
  /path/to/units /path/to/new-acquisition-report.json
```

From a clean implementation commit, run the complete initial qualification with fresh destinations
and enough command time for downloading and repeatedly hashing the raw files:

```bash
uv run --no-sync python -m scripts.acquisition_units_qualify \
  research/flagship/acquisition_unit_rehearsal_v1.json \
  /mnt/speck-data/speck/acquisition-unit-rehearsal-v1/qualification \
  results/systems/acquisition-unit-rehearsal-20260913.json
```

The driver records stage progress durably, including injected or unexpected failures. It checks
Parquet/web and gzip/code acquisition recovery, all cohort dedup outputs, the exact-replay control,
and logical index parity. Stage-boundary free-space observations and main-process peak RSS are recorded;
these are not continuous disk high-water measurements or the memory peak of the Gitleaks subprocess.

## Completed measurement

The [checked result](../../results/systems/acquisition-unit-rehearsal-20260913.json) passes all twelve
units, Parquet/gzip recovery, dedup recovery, and the exact-replay control. It retains 2,601 natural
cohort records and 26,780,199 UTF-8 bytes. Cold acquisition takes 1,483.82 seconds, including 1,406.92
seconds downloading the six raw files. Profiled cohort dedup takes 17.24 seconds; the recovered pass
takes 13.47 seconds without profiling.

The [finding](../findings/2026-09-13-acquisition-unit-rehearsal.md) records per-category yield, filters,
resource boundaries, and profile interpretation. The short checkpoint interval materially affects
the profile; no production speedup or full firewall-exclusion result is inferred from this rehearsal.

The subsequent [complete integration](FIREWALL_INTEGRATION.md) uses the larger-window
[`acquisition_unit_rehearsal_v2.json`](acquisition_unit_rehearsal_v2.json), retained raw cache, and
10,000-record dedup checkpoints. It qualifies the full reference pass and source-bank handoff; its
results and boundaries are recorded separately from this initial cohort-only measurement.
