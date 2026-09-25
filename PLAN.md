# SpeckLabs: status and next work

Updated 2026-09-25. This is the status and the one work order. The [program design](docs/program.md)
owns the goal, method, experiments and budget; [plan.json](experiments/main-data/plan.json) owns
the numbers; receipts own results.

This release is SpeckLabs' first scaling step, and it targets the data of pretraining and
mid-training: its deliverable is [measured, transferable findings](docs/program.md#goal-the-data-step)
about those pipelines, produced by a model ladder (50m, 130m, 410m) and one 1.2B parent whose branches
carry the decay and mid-training experiments. A fixed SFT recipe probes every branch and yields a
light assistant; post-training research waits for a later step. 5,000 GH200 GPU-hours are
confirmed; access timing and GH200 throughput are not.

## Supply

Ladder runs need a few billion tokens each and can start from current stock. The parent needs far
more. [`supply-gap.json`](experiments/main-data/supply-gap.json) derives retained candidate stock
per bank against the parent's starting mixture at 60B tokens, prepared at 1.25 times exposure:

| Bank | Weight | Preparation target | Retained candidate stock | Coverage | One-pass exposure cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| selected_web | 25% | 18.75B | 0.351B | 1.87% | 1.40B |
| independent_web | 5% | 3.75B | 2.307B | 61.51% | 46.13B |
| natural_code | 35% | 26.25B | 1.157B | 4.41% | 3.31B |
| natural_math | 25% | 18.75B | 1.124B | 6.00% | 4.50B |
| reference_science | 5% | 3.75B | 1.402B | 37.39% | 28.04B |
| refined_web | 5% | 3.75B | 1.489B | 39.71% | 29.79B |

Candidate stock has every gate still open, so it bounds what could be admitted; zero eligible tokens
are established. Two acquisitions will move this table; both are stopped until the workstation is
verified (see the work order):

- **Selected web.** One whole Ultra-FineWeb HQ crawl, `CC-MAIN-2025-43` (27.5M documents), is
  downloaded and sha-verified; its conversion, preprocess and token census are redone on verified
  hardware.
- **Code.** Every licence-eligible Stack-Edu `int_score` 3 row, projected at about 15B tokens from
  the [yield probe](experiments/corpus-audit/stack-edu-yield-probe.json), is partly fetched in
  language tranches and resumes unit by unit.

**Math becomes the binding bank** once those land: 1.1B tokens of stock against a 25% share. The
P3 mixture and P4 repetition families decide how much math the parent actually needs.

## Completed evidence

| Work | Evidence |
| --- | --- |
| Runtime: full-size H100 restart, recovery, generation and export | [H100 receipt](experiments/qualification/h100-result.json) |
| Engineering pilot: 105M tokens, 12,859 tokens/s full trainer, weak development scores | [Run](experiments/pilot/h100-run.json), [scores](experiments/pilot/development-result.json) |
| Throughput recipe: 2.084x on a 318M proxy, compiled restart parity | [Sweep](experiments/qualification/throughput-3090/sweep.json), [base](experiments/qualification/compiled-recovery-descent-3090.json), [SFT](experiments/qualification/compiled-sft-recovery-3090.json) |
| Source use: nine selected sources approved for research training and weight release | [Acceptance](experiments/main-data/source-rights-acceptance.json) |
| Code supply: census of all 42 Stack-Edu files, `int_score` 4+ acquisition (680M tokens), `int_score` 3 yield probe | [Census](experiments/corpus-audit/stack-edu-metadata-census.json), [acquisition](experiments/corpus-audit/stack-edu-acquisition.json), [probe](experiments/corpus-audit/stack-edu-yield-probe.json) |
| Web supply: HQ listing by crawl, stratified audit, extraction follow-up | [Listing](experiments/corpus-audit/ultrafineweb-hq-listing.json), [audit](experiments/corpus-audit/web-hq-stratified.json) |
| Family partitions: six text stocks and the code review cohort, 45 code holds | [Partition](experiments/main-data/family-partition.json), [qualification](experiments/main-data/README.md#qualification) |

## Work order

1. **Now, on the workstation (no grant hours).**
   - **Verify the hardware first.** The i7-13700K shows Raptor Lake instability: 11 segfaults on the
     same two cores since 2026-09-22 and a SQLite index corrupted on a healthy NVMe, which crashed
     the crawl preprocess on 2026-09-24. After a reboot no kernel fault recurred, but data is still
     silently corrupted: three conversions of the same sha-verified crawl shards produced three
     different files; the two kept had 10 and 6 damaged records in 27.5M, all caught by their
     per-record content hashes, and both were deleted. Every data and training job is stopped.
     Update the BIOS to Intel's default power profile, run memtest86+ and a CPU stress test, and
     replace the CPU if it still fails. Then convert the crawl twice and require identical files
     before its preprocess, and rebuild the 50m sweep corpus twice and compare. Re-verified by
     content and kept: the fetched Stack-Edu units, the code conversion, preprocess and token index,
     the HQ web stock and its index, and the dry-run family partition, each matching a second build
     or its per-record content hashes.
   - Run the 50m learning-rate sweep (its own corpus, from the retained HQ sample).
   - Configure the 50m baseline corpus from the verified crawl; P0 and every P family train on it
     and no configuration exists yet. Then run the P0 seed set and choose the metrics that move at
     this scale.
   - Finish both acquisitions: census the crawl and point the supply gap at it; summarize the code
     fetch, preprocess all code and rebuild the family partition over every source.
   - Write the predeclared records for P1 to P7 and prepare their data variants on CPU.
   - Freeze the SFT probe recipe and dataset from the retained assistant stock.
   - Add per-rung throughput runs to the [GH200 packet](experiments/qualification/throughput-gh200.json).
2. **On access,** qualify one worker then four, measure every rung, and convert each budget line
   into run counts.
3. **Pretraining ladder:** LR and P0 at every rung, P1 to P7 and C at 50m, the T transfer re-runs
   and DR; record the predicted parent loss and the parent mixture.
4. **Parent** stable run, then D1 to D3, each probed with the SFT recipe.
5. **Mid-training** M1 to M4 from the selected decayed parent, each probed.
6. **Final evaluation, paper and release.** Final test partitions are scored once, at the end.

## Open decisions

- Whether sources whose licences permit redistribution are released as data, or only as manifests
  from which the corpora can be rebuilt.
- Use of FinePDFs-Edu (ODC-By over Common Crawl).
- Use of Nemotron-CC-v2 (NVIDIA agreement restricted to internal training).
