# 09 — Decisions, open questions, and change log

## Decisions supported by completed experiments

### Keep

- Gated DeltaNet as the recurrent/local backbone.
- Some global attention when content-addressable retrieval matters.
- Separate global and sliding RoPE modules in mixed architectures.
- A two-role design vocabulary: middle integration layers and final retrieval/readout layers.
- FlexAttention for CUDA sliding prefill and a ring buffer for decode.
- Correct causal mean-row FLOPs accounting.
- Complete long-document continuation data and mandatory original-4K regression evaluation.
- Counterfactual needle pairs for internal retrieval regression.

### Retire or stop

- Retire the gated-convolution hybrid from the main research path; it loses about 0.05 nats to GDN.
- Do not use pure GDN as the release architecture; eliminating attention costs about 0.155 nats and
  does not provide long-range associative retrieval.
- Do not run `SpeckLC-150M-Rank-500M` as configured. It spends roughly four times the original
  compute at 4K on a metric already shown to be seed-limited.
- Do not rank the original top four from one 4K seed.
- Do not claim 128K capability from allocation, prefill success, or the current internal passkey.
- Do not launch 128K continuation on the current 16K-minimum document corpus.
- Do not enable undocumented Flex mask guarantees based only on isolated kernel speed.

## Current architectural hypothesis

The former `GGGA`-repeated pattern is not an efficient default. The leading compact design is:

> Predominantly Gated DeltaNet/sliding processing, one global integration layer around the middle,
> and one global retrieval/readout layer near the end.

This is a hypothesis, not a release selection. One seed cannot distinguish the middle-only and
two-layer long losses, and the two-layer model did not improve retrieval retention over final-only
in the internal diagnostic.

## Required next work

1. Replicate `global-1` final, `global-1-mid`, `global-2`, and `global-5` over seeds. The first
   priority is confirming effects larger than the `0.00965`-nat noise range.
2. Build or acquire data with genuine 64K–128K dependencies: long books/papers, repository trees,
   connected documents, and held-out synthetic retrieval/aggregation tasks. Do not pad the current
   data with unrelated documents.
3. Validate multi-token INT8 KV decode quality; current state numbers prove allocation only.
4. Reduce KV heads or add a latent/MLA-style global projection, focusing first on final and middle
   global layers.
5. Add a task-appropriate inference adapter and run pinned independent suites:
   [RULER](https://github.com/NVIDIA/RULER),
   [NoLiMa](https://github.com/adobe-research/NoLiMa), and
   [HELMET](https://github.com/princeton-nlp/HELMET).
6. Promote to 128K only after the data and independent 32K gate pass; re-run original 4K loss at
   every stage.

## Known limitations

- Global-count training uses one seed per point.
- The seed study fixes packed-data order, so it is a lower bound on total variance.
- The long-document validation set contains only 16 documents and 338,711 tokens.
- Global RoPE uses simple fixed linear scaling, which causes measurable short-context erosion.
- Open-vocabulary passkey exact match is zero for every 150M checkpoint.
- Counterfactual directional scores demonstrate causal sensitivity, not robust task completion.
- The model has not passed independent long-context benchmarks.
- No 128K training stage has been run.

## Checkpoint inventory

All paths below have a completion marker and model plus optimizer state:

- Original six: `~/.cache/speck/checkpoints/SpeckLC-150M-MixerScreen-131M-*`, step 2,000
- Noise repeats: `~/.cache/speck/checkpoints/SpeckLC-150M-NoiseFloor-131M-seed-{43,44}`,
  step 2,000
- First 32K pair: `~/.cache/speck/checkpoints/SpeckLC-150M-Context32K-gdn-{local,global}`,
  step 489
- Same-parent frontier:
  `~/.cache/speck/checkpoints/SpeckLC-150M-GlobalCount32K-{global-1,global-1-mid,global-2,global-5}`,
  step 489

The full checkpoint hashes are stored in the checked result summaries, not only in W&B.

## Chronological implementation and result commits

| Commit | Purpose |
| --- | --- |
| `a8d8ff1` | Prepared the later-stopped 500M mixer ranking family |
| `a048326` | Added the standalone 131M mixer screen |
| `4ee5ea4` | Corrected causal attention FLOPs accounting |
| `7d3ded8` | Added FlexAttention sliding prefill |
| `35105d6` | Removed empty attention-cache concatenation |
| `80e7e7c` | Recorded corrected FlexAttention preflight |
| `08057db` | Refreshed both mixer compute ledgers |
| `a2e9e4e` | Prepared the three-seed noise-floor family |
| `55bace7` | Recorded the seed noise floor |
| `e878582` | Hardened long-context evaluation pilots |
| `620bf77` | Added chance-controlled candidate scoring |
| `f015c0d` | Replaced token-mask construction with block metadata |
| `332bc60` | Separated long-context compile warm-up |
| `e88ee32` | Fixed FlexAttention block-width compatibility |
| `84bcc2f` | Recorded the six-model 128K systems frontier |
| `e04d30c` | Added memory-safe context training contracts |
| `bbef983` | Added long-document packed-data derivation |
| `addcdac` | Bound context stages to exact packed data |
| `eb2addf` | Prepared the first matched 32K pair |
| `4595890` | Recorded 32K training preflight |
| `c1fec2e` | Added cross-dataset checkpoint loss evaluation |
| `ce2e6f4` | Added diagnostic RoPE loss overrides |
| `0670ddb` | Added paired counterfactual retrieval evaluation |
| `85ed57e` | Recorded first 32K local/global findings |
| `6c8dcd5` | Added parameter-safe global-layer promotions and scope-specific RoPE |
| `faa1b2a` | Prepared the same-parent global-count frontier |
| `288d81d` | Recorded global-count preflight |
| `f19e5ba` | Added the middle-layer placement control |
| `7c39313` | Recorded the completed global-count frontier |
| `f1a6f49` | Updated core long-context documentation with the frontier |

## Repository state at consolidation

- Test suite: 280 passed
- Training/evaluation processes: none
- GPU: idle
- Worktree before this findings ledger: clean
- Branch: `main`, 28 commits ahead of `origin/main`
- Push status: nothing pushed
- Free disk after checkpoint creation and cache cleanup: approximately 4.9 GiB
