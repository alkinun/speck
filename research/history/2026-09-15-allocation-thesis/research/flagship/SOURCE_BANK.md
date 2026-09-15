# Bounded source-bank preparation

This is the first engineering stage toward source-separated E1/E3 corpora. It uses the six retained
pilot-training outputs whose parent pass placed all twelve firewall candidate views first. It reads
the parent manifest and selected training files; it does not open evaluation or sealed-audit payloads.

## Frozen rehearsal

[`source_bank_rehearsal_v2.json`](source_bank_rehearsal_v2.json) binds the completed parent manifest,
pinned Mistral reference tokenizer, six training outputs, 2 MB UTF-8 text quota per category, 16-record
selection checkpoints, and a new runtime destination. Whole documents may overshoot each byte quota.
All original JSONL fields are retained verbatim. No final tokenizer or E1/E3 treatment is selected.

The parent is the firewall-excluded corpus recorded by
[`tokenizer-pilot-corpus-20260911.json`](../../archive/pregrant-history/results/data/tokenizer-pilot-corpus-20260911.json).
Its deduplication/filtering/exclusion evidence is inherited. This rehearsal measures the new bank
selection, reference packing, integrity checks, and recovery; it does not rerun or retime acquisition
and global deduplication. The six representative sources do not cover all E1 treatment alternatives.

The [v1 attempt](../../results/systems/source-bank-rehearsal-interruption-20260913.json) hit an external
120-second command timeout after completing the clean bank and five recovery-source units. Its partial
outputs are retained, but per-invocation timings were not published. V2 changes only the runtime
destination so the same method can be measured in a fresh execution with sufficient command time.

## Behavior

- Select the shortest whole-document prefix meeting each explicit **UTF-8 text-byte** quota.
- Copy raw JSONL records, preserving metadata and source order. Existing missing metadata cannot be
  reconstructed by this copy; book/repository reconstruction remains a separate preparation task.
- Pack those same documents with the pinned reference tokenizer, including BOS/EOS, and record actual
  reference tokens and uint16 shard identities. BOS/EOS does not provide example isolation.
- Resume selection at durable record boundaries after validating the committed prefix and counters.
- Publish packing atomically per source. A failed packing attempt rebuilds that source's unpublished
  packing from retained selected text; it does not redo completed sources.
- Verify original source-file hashes on each invocation and verify published shard hashes on reopen.
- Report input verification, selection, and packing times separately. Each invocation's time excludes
  previous attempts; the qualification records the injected failure's wall time explicitly.

V1 bounds nominal selection to 100 MB per source and raw records to 16 MB. It accepts only named retained
pilot training outputs. These bounds and source restrictions keep the first qualification small; a
production materializer requires a successor with selected tokenizer, real per-arm capacity,
acquisition/filter variants, source universe and precedence, and launch bindings.

## Commands

Prepare or reopen a bank, emitting a new per-invocation report:

```bash
uv run --no-sync python -m scripts.source_bank_prepare \
  research/flagship/source_bank_rehearsal_v2.json \
  --report /path/to/new-invocation-report.json
```

For the initial engineering qualification, start from a clean implementation commit and new runtime
destinations. This command prepares a clean bank, injects a code-source interruption after record 17,
resumes from record 16, compares all six selected and packed payloads, and checks published reopen:

```bash
uv run --no-sync python -m scripts.source_bank_qualify \
  research/flagship/source_bank_rehearsal_v2.json \
  results/systems/source-bank-rehearsal-20260913.json
```

The qualification result binds its clean Git revision, original plan, recovery plan, runtime manifests,
per-source output hashes and measurements. It does not issue model-training or production-operations
authority. Use a successor plan and new destinations for a later qualification; retain the originals.

## Completed v2 measurement

The [checked result](../../results/systems/source-bank-rehearsal-20260913.json) passes all six source
units, injected recovery, and published reopen. It contains 12,049,523 selected UTF-8 bytes and
3,303,264 reference tokens. The clean invocation takes 64.77 seconds, including 48.51 seconds of input
hash verification, 13.57 seconds of selection/checkpointing, and 2.59 seconds of reference packing.
See the [finding](../findings/2026-09-13-source-bank-rehearsal.md) for per-category results and boundaries.

## Excluded acquisition handoff

The [complete firewall integration](FIREWALL_INTEGRATION.md) adds bank **schema v2** for newly acquired
category groups. Its plan includes a `firewall_plan` identity and accepts only the matching
`acquired_train__<category>` outputs after checking complete reference identities, precedence, policy,
and preservation in the parent result. The integrated handoff passes for all six categories.
This schema version is separate from the earlier `source_bank_rehearsal_v2.json` destination successor,
which retains schema v1 and the pilot-training input path.
