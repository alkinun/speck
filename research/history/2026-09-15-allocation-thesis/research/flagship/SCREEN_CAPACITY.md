# E1/E3 capacity review and processing-cost decomposition

This step connects the experiment token requirements to checked available supply. It also separates
candidate processing from reference setup, checkpointing, resume checks, and final verification.
Neither analysis selects a tokenizer, source treatment, or model configuration.

## Capacity basis

[`screen_capacity_review_v1.json`](screen_capacity_review_v1.json) pins the selected data plan,
human-readable protocol, source registry, and archived measured tokenizer-pilot corpus. The review
reverifies the retained corpus manifest, six training files, and reference tokenizer. Token counts are
the archived measurements; this is not another full tokenization run.

The current corpus is globally deduplicated across its source files. Its category counts may therefore
be added within that one measured corpus. The newer integration bank and alternative tokenizer views
must not be added without proving their union and overlap; their existence does not establish extra
unique-token supply.

For category prior weights `w[c]` and measured reference-token capacities `available[c]`:

```text
balanced_pool_upper_bound = min_c floor(available[c] / w[c])
E3_unique_pool = 6B / effective_epochs       # effective_epochs = 1, 2, 4
required[c] = ceil(E3_unique_pool * w[c])
missing[c] = max(0, required[c] - available[c])
```

Seeds and repeated exposures do not automatically multiply the unique corpus requirement. Packed
alignment, loader lookahead, independent-data requirements if specified later, and final-tokenizer
yield still require explicit allowance before a launch manifest is executable.

The arithmetic is **conditional on the documented category prior and eligible reuse of the measured
bank**. E3's exact incumbent source assignments remain to be frozen. In particular, the representative
Common Pile code/science sources used in operations qualification do not silently replace the research
protocol's source treatments or science incumbent.

## E1 choices that still need executable definitions

The protocol fixes four initial web treatments, three initial treatments for each specialist category,
replication counts, token horizons, and selection rules. It does not yet provide executable per-arm
source/filter/blend manifests. The candidate list includes Ultra-FineWeb variants, DCLM, FineWeb-Edu,
and blends; exact four-arm mapping and blend proportions must be resolved before model outputs.
The specialist blend recipes also need their own explicit source fractions.

For preparation planning, the review reports two **scenarios**, without choosing between them:

- Prior-category substitution: an 8B E1W run allocates 4.4B tokens to its 55% web component; a 2B E1S
  run allocates 300M to a 15% code component or 200M to a 10% math/synthetic component.
- Pure-category treatment: the full 8B or 2B horizon would be supplied by that treatment.

Source availability cannot silently decide this design. Primary-screen sources lacking a capacity
measurement in the reviewed evidence are reported as **unknown**, not zero upstream supply.

Run from a clean identified checkout:

```bash
uv run --no-sync python -m scripts.data_screen_capacity \
  research/flagship/screen_capacity_review_v1.json \
  results/data/screen-capacity-20260913.json
```

## Phase-separated timing replay

[`dedup_timing_replay_v2.json`](dedup_timing_replay_v2.json) binds the completed full-reference
integration. The replay restores its exact committed reference prefix into a new private destination:

1. Verify the parent result, manifest, retained checkpoint, and published SQLite identity.
2. Copy committed output prefixes and a private database copy. Remove only post-checkpoint rows from
   that copy; validate foreign keys, reference count, checkpoint row, and the ordered reference chain.
3. Rebind only the checkpoint's destination contract. Inputs, precedence, policy, candidate stream,
   and 10,000-record checkpoint interval stay identical to the parent pass.
   Flush and fsync copied committed prefixes and the restored index before the measured invocation.
4. Run the same candidate continuation with optional timing instrumentation. Outputs, removal logs,
   and counts must match the original result exactly, and reference controls/exclusion must still pass.

This avoids rebuilding all 288,872 references. Restoration cost is reported separately. It is a
committed-prefix replay, not a stress test of rolling back many uncommitted SQLite rows. Copying warms
caches and preserves an allocated database layout; the resulting rate is not a cold-start or
production-scale throughput forecast.

Timing phases are disjoint: setup, input verification, resume verification, processing, publication,
and final verification. Per-source elapsed time includes its checkpoints; the report also supplies
processing time excluding checkpoints. Checkpoint time across all phases overlaps phase totals and
must never be added to them again. Processing includes reads, parsing, hashing, comparisons, and index
updates; it is not kernel-only time. Interrupted and published-reopen invocations cannot report a
complete fresh processing measurement.

```bash
uv run --no-sync python -m scripts.dedup_timing_replay \
  research/flagship/dedup_timing_replay_v2.json \
  results/systems/dedup-phase-timing-v2-20260913.json
```

Each command binds its clean implementation commit and checked inputs. The timing replay uses a new
runtime directory, retains the parent, and adds no new corpus capacity or training authority.

The optional timing hooks change the maintained preprocessor's file identity. The earlier streaming-
resume memory probe remains bound to its original implementation hash; its record is not rewritten.
The identical-output checkpoint replay is the explicit execution qualification for this instrumented
successor. Its phase measurements do not replace the earlier Python-allocation measurement.

The [v1 diagnostic](../../results/systems/dedup-phase-timing-20260913.json) is retained at its recorded
implementation. It preserves candidate/output parity and reports 31.84 seconds of non-checkpoint
candidate work versus 249.00 seconds inside candidate checkpoints. Its copied prefixes were not
explicitly fsynced before measurement, so deferred restoration writes could enter checkpoint costs.
V2 makes restoration durable first and splits checkpoint time into SQLite commit, output flush/fsync,
slice hashing, and state publication before attributing the bottleneck. This measurement correction
is not a change to the dedup policy or a claimed pipeline speedup.

## Completed capacity review

The [checked review](../../results/data/screen-capacity-20260913.json) finds 2.056B reference tokens in
the verified bank, but only a 1.236B balanced-prior pool ceiling. The smallest E3 unique pool needs
1.5B; its remaining category deficits are 13.215M science and 10.337M reference tokens. Full 6B supply
and the exact scientific source recipes remain open. See the [finding](../findings/2026-09-13-screen-capacity.md)
for all category requirements and the distinction between measured supply and upstream availability.

## Completed timing attribution

The [durable v2 replay](../../results/systems/dedup-phase-timing-v2-20260913.json) preserves the original
outputs, removal records, counts, and exclusion/control results. Restoration takes 68.58 seconds
separately. The measured continuation takes 278.28 seconds, including **236.74 seconds in SQLite commit**.
Non-checkpoint candidate work takes 30.99 seconds. Output flush/fsync and state publication are each
under one second across the run. See the [finding](../findings/2026-09-13-dedup-phase-timing.md) for
disjoint phase totals and checkpoint-overlap accounting.

This identifies a concrete target for a bounded SQLite policy comparison. It does not establish a
production speedup or resolve the independent E1/E3 source-recipe and capacity gaps.

The subsequent [durable WAL comparison](SQLITE_WAL.md) now passes: the 65,536-page trigger reduces
complete continuation time by 43.9% and 53.2% in reversed-order pairs while retaining FULL sync,
output/index parity, and hard-process-exit recovery. This is a local bounded operating recommendation;
full source recipes, capacity, larger transactions/indexes, and site transfer remain to be qualified.

The [UltraData intake](ULTRADATA_INTAKE.md) now supports a conditional candidate revision: new natural
Code/Math L2 datasets replace existing blend slots, while L3 refinement is considered in E4. If admitted,
the source-bank envelope becomes 18B tokens before headroom; the earlier 17.5B figure remains the
preserved v1 proposal. This does not add measured supply or source-use approval.
