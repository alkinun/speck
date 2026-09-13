# Production-data critical path

Working assessment, 2026-09-13. This separates preparation dependencies from the experiment calendar.
It does not replace the selected data, repetition, operations-fallback, or launch contracts.

## 1. Measured starting point

The [2B calibration analysis](../../archive/pregrant-history/results/data/production-data-calibration-2b-20260910.json)
is the rate and byte basis. The owner accepted its conditional 150B-unique-token branch; the 500B
branch remains blocked. The calibration covers six representative sources, not demonstrated full-scale
supply across every E1 candidate or all 30 approved sources. Its token counts use the pinned reference
tokenizer. Re-estimate final-tokenizer yield and per-source supply after D5.

| Serial stage | 150B projected hours | Days | Current execution behavior |
| --- | ---: | ---: | --- |
| Acquisition, filtering, quota counting | 138.48 | 5.77 | sources/files processed serially; tokenizer counts enforce quotas |
| Global exact and verified-near deduplication | 309.71 | 12.90 | ordered accepted-document decisions and one SQLite writer |
| Final tokenization and packing | 6.04 | 0.25 | source-separated output; each source packed serially |
| **Total** | **454.22** | **18.93** | no unmeasured parallel speedup credited |

Acquisition's recorded bytes/second is an end-to-end acquisition/filtering/token-counting rate, not
network bandwidth. The dedup rate includes state verification and finalization of a conservative
observed segment. The packing rate does not represent the entire preparation pipeline.

Current machine observation: 24 logical CPUs; 33,399,128,064 bytes of physical RAM (31.1 GiB); about
5.42 TB available on `/mnt/speck-data`. These are a dated local capacity snapshot, not dedicated grant
resources. The projected 1.99 TB live data working set fits this snapshot; checkpoints, backups,
extension/SFT corpora, source alternatives, and transient files still need their own space.

The tokenizer wrapper uses up to eight threads per batch. There is no qualified source-level parallel
acquirer or multiworker global dedup path in the current rehearsal. CPU count cannot be used as a
throughput multiplier.

## 2. Dependencies and work that can start early

| Work product | Decisions it needs | What can proceed before those decisions |
| --- | --- | --- |
| Source inventory | approved immutable source identities | inspect existing manifests, adapter coverage, file order, metadata, and capacity evidence |
| Bounded source-preparation implementation | fixed filtering, source precedence, firewall exclusions | implement and qualify mechanics on bounded retained training inputs |
| E1/E3 screen corpora | D5, frozen source treatments/incumbent, required firewall exclusions | specify per-arm quotas and reusable source banks; these corpora do not depend on E2's winner |
| E2 candidate corpora | D5 and qualified E1 treatments | prepare source-separated banks once treatments are known; validate each arm's capacity |
| Production 150B source bank | D5, E3 supporting the branch, source treatments and a costed supply envelope | preserve reusable source material and define acquisition boundaries; bulk preparation needs the executable successor |
| Final stable mixture | E2 decision and adequate selected-source supply | pack source-separated shards independently of final weights when tokenizer/source decisions are frozen |
| Final decay mixture | E4 and adequate supply | retain candidate decay-source capacity and account for overlap/repetition with the stable bank |
| Long-document corpus | retained document structure, later D5 and measured length yield | retain book/paper/repository identity and order before raw cleanup |
| Launch receipt and site delivery | final manifests, repetition/mixture decisions, verified shards, storage and runtime | prepare commands and site-transfer checks; measure actual delivery before launch |

The loader already selects from source-separated shards using manifest-defined mixture phases. A
change in mixture weights therefore need not repeat tokenization when the same shards have adequate
supply. The final launch manifest and exposure accounting still change.

Two important constraints on early work:

- **The existing acquisition function is tokenizer-bound.** A tokenizer-independent preparation stage
  needs a versioned byte/whole-document or explicitly reference-token quota definition. Reusing an
  existing reference-token sample does not make its counts final-tokenizer counts.
- **Global deduplication is ordered.** All firewall reference records take precedence, followed by a
  fixed source/record order. Deduplicating source banks independently and concatenating them loses
  cross-source exclusions. Candidate-only sources must not silently remove documents from a later
  selected treatment. Define the retained-source universe and precedence before the pass.

The rehearsal's reduced record schema retains text/hash, URL/host, and one content ID. Long-document
reconstruction may need additional source fields and ordering metadata; preserve those explicitly in
the successor rather than relying on packed shards or this reduced schema.

## 3. The 150B pool is not independent of mixture choice

A pool prepared at the prior weights would contain the following unique-token quantities. The final
column shows a bank large enough for any permitted E2 weight at the same nominal 150B pool scale.
Category maxima need not occur together in one mixture; preparing all alternatives simultaneously
still costs their sum.

| Category | Prior share | Prior-pool tokens (B) | Maximum share | Coverage-bank tokens (B) |
| --- | ---: | ---: | ---: | ---: |
| Web | 55% | 82.5 | 65% | 97.5 |
| Code | 15% | 22.5 | 22% | 33.0 |
| Math | 10% | 15.0 | 18% | 27.0 |
| Synthetic | 10% | 15.0 | 18% | 27.0 |
| Science | 5% | 7.5 | 10% | 15.0 |
| Reference | 5% | 7.5 | 10% | 15.0 |
| **Total** | **100%** | **150.0** | **143%** | **214.5** |

Source alternatives inside each category can require still more preparation. The 214.5B coverage bank
is a planning calculation, not an authorized expansion beyond 150B. Do not build it under the existing
fallback.

Repetition also needs per-source accounting. At the 400B flagship target's approximate 320B stable
phase, a 10% science allocation consumes 32B tokens. A prior-sized 7.5B science bank would therefore
receive 4.27 exposures before decay; reference has the same issue. E3's tested effective epochs are
1/2/4 at its incumbent mixture. The existence of an average repetition allowance does not establish
equivalence for arbitrary per-domain exposure. Check stable plus decay usage against measured supply
and the eventual E3 decision.

Recommended preparation structure: bounded reusable screen banks first, then a costed production bank
with explicit per-category/source capacity, followed by mixture manifests. Freeze enough supply before
the final decision that shortages cannot covertly choose the winner. If capacity fails a frozen arm,
apply its declared failure/fallback rule and retain the failure.

## 4. Calendar arithmetic and the open gate

Use elapsed allocation days consistently: day 4 means 96 hours after access. These calculations use
the historical serial rates and exclude engineering, source exhaustion, extra firewall passes,
final verification, transfer to the site, and queue delays.

| Preparation start | Projected completion | Interpretation |
| --- | ---: | --- |
| Day 0 | Day 18.93 | fits nominally only if all required inputs/decisions already exist |
| Day 4 | Day 22.93 | misses day 21 even with no additional delay |
| Day 11 | Day 29.93 | waiting for final mixture before all preparation is too late |

The latest serial start for day 21 is **day 2.07**, with zero buffer. Starting on day 4 requires at least
**1.113× aggregate throughput**, or **1.175× dedup throughput** if acquisition and packing retain their
measured costs. These are break-even requirements, not measured speedups or adequate production margin.

Formulas, using the `150000000000` projection in the linked calibration:

```text
serial_days = serial_total_hours_approx / 24
finish_day = start_day + serial_days
latest_start_day = 21 - serial_days
required_dedup_speedup_from_day4 = dedup_hours / (17 * 24 - acquisition_hours - packing_hours)
coverage_bank_tokens = 150B * sum(category.max_percent) / 100
source_exposure = (stable_tokens * stable_weight + decay_tokens * decay_weight) / unique_source_tokens
```

**Day-21 data readiness remains unqualified.** Keep that visible alongside the configuration-freeze
target. A credible ready date needs measured end-to-end successor performance, sufficient source
capacity, actual decision dates, and separately costed final verification/transfer.

## 5. Engineering result completed in this review

Resume previously called `fetchall()` on every accepted hash pair before rebuilding the index chain.
It now streams the ordered SQLite cursor while checking the same row count and hash chain. The
[read-only probe](../../results/systems/production-dedup-resume-streaming-20260913.json) verifies the
retained 2B database hash and exact chain parity on 10K and 100K rows:

| Rows | Previous Python peak traced bytes | Streaming Python peak traced bytes |
| --- | ---: | ---: |
| 10,000 | 2,906,584 | 2,216 |
| 100,000 | 29,002,128 | 1,808 |

At the calibration's projected 173.07M retained records for 150B, the old 100K-row allocation ratio
implies about 50.2 GB just for this Python materialization. This extrapolation is separate from total
RSS, SQLite cache, and the dedup runtime working set. Streaming removes the proportional row-list
allocation; it still scans the entire accepted index on resume. Original source/output hashing and
checkpoint identity verification also remain part of restart cost.

Behavioral tests cover uninterrupted/resumed output parity, rejection of altered index counts/hashes,
and a cursor that forbids `fetchall()`. This is a bounded engineering qualification of the resume
change. Retain the old full-pass throughput projection until the successor is measured end to end.

## 6. Bounded end-to-end integration completed; production capacity and rates open

The [source-bank rehearsal](SOURCE_BANK.md) now qualifies the retained-source selection/packing portion:
12.05 MB across all six categories, 3.30M reference tokens, and exact interrupted/uninterrupted payload
parity. It inherits the existing firewall-excluded input corpus. Full E1/E3 source-treatment coverage
and final-tokenizer capacity remain to be qualified.

The [upstream unit rehearsal](ACQUISITION_UNITS.md) now also executes raw acquisition/filtering,
Parquet/gzip recovery, and ordered dedup over a 2,601-record cohort. It measures cold-file download
cost and profiles a diagnostic 64-record checkpoint cadence. Complete firewall reference exclusion
is not run on that original cohort, so those outputs remain engineering-only artifacts.

The follow-up [complete integration](FIREWALL_INTEGRATION.md) now connects larger raw units to the full
twelve-view exclusion pass and bank v2. It preserves all 288,872 references, removes 5,635 candidate
superset matches, and retains 15,135 candidates with zero exact reference overlap. Both exact/near
controls and full-reference checkpoint recovery pass at the 10,000-record cadence. All six fixed
engineering bank quotas pass. Production-scale supply and rate qualification remain separate.

Next, qualify the production capacity and operating cost of the connected path:

The [screen capacity review](SCREEN_CAPACITY.md) now quantifies two concrete gaps. At the documented
prior, the checked 2.056B reference-token bank supplies only a 1.236B balanced pool, below E3's smallest
1.5B unique pool. Exact E1 treatment/blend and E3 incumbent-source recipes remain unfrozen. Separately,
a durable same-input timing replay attributes 236.74 of 278.28 seconds to SQLite commit. These findings
prioritize recipe/supply closure and a measured database-policy comparison; they do not change the
150B forecast yet.

1. Bind source revisions, reader/filter variants, per-arm supply, firewall reference precedence, and
   output identities; reuse retained inputs where their identity and scope match.
2. Freeze a successor with explicit record-level metadata retention and independently resumable
   acquisition units. State whether quotas use final tokens, reference tokens, or bytes.
3. Measure stage time, retained tokens/bytes, source exhaustion, SQLite growth, resume verification,
   and free-space high-water marks on bounded real inputs.
4. Test SQLite commit/journal/checkpoint policy on the qualified workload, preserving durability and
   output/recovery parity while measuring WAL/index space. The phase-separated 10,000-record-cadence
   replay now identifies commit as the dominant cost. The earlier 64-record profile remains a separate
   diagnostic. Reassess worker parallelism only with the measured operating envelope; source
   acquisition and post-dedup packing have separate resource limits.
5. Use those measurements to cost the 150B bank and a real ready date. Preserve the accepted fallback
   and the paused 20B artifacts while this successor is qualified.
