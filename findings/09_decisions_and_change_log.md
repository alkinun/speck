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
- Sigmoid output gating for the recurrent mixer; it improves the matched 131M GDN model by
  0.03734 nats over SiLU and replicates tightly across three seeds.
- KDA/sigmoid/NoPE as the lead long-context research treatment. Its 128K contrastive signal
  replicates 3/3 and strengthens after 32K continuation.
- GDN/sigmoid/RoPE as the short-loss control for subsequent KDA experiments.

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
- Do not convert a trained RoPE checkpoint to NoPE and call it a fair architecture comparison.
  The late switch damaged both long and short loss; train NoPE from the base stage.
- Do not claim that contrastive sensitivity is solved retrieval. All current open-vocabulary exact
  matches are zero.
- Do not launch 128K continuation until a task-appropriate inference adapter and real completion
  benchmarks are available.

## Superseded pre-Kimi architectural hypothesis

The former `GGGA`-repeated pattern is not an efficient default. The leading compact design is:

> Predominantly Gated DeltaNet/sliding processing, one global integration layer around the middle,
> and one global retrieval/readout layer near the end.

This is a hypothesis, not a release selection. One seed cannot distinguish the middle-only and
two-layer long losses, and the two-layer model did not improve retrieval retention over final-only
in the internal diagnostic.

## Current architectural hypothesis

The lead research architecture is a KDA/sigmoid/NoPE recurrent-global hybrid. At five global
layers it is not yet state-efficient enough: resident BF16 state is 481.47 MiB at 128K. The next
frontier question is whether one or two NoPE global layers retain KDA's replicated long-range
signal, reducing 128K state toward 96–192 MiB before INT8 KV compression.

This supersedes the earlier GDN/sliding-first hypothesis for research promotion, not for release.
KDA misses one of three strict base-stage loss ties and has not passed an independent task suite.

## Required next work

1. Add a task-appropriate inference adapter and run pinned independent suites:
   [RULER](https://github.com/NVIDIA/RULER),
   [NoLiMa](https://github.com/adobe-research/NoLiMa), and
   [HELMET](https://github.com/princeton-nlp/HELMET).
2. Replace the one-token contrastive diagnostic with exact-answer retrieval, aggregation, and
   multi-hop tasks whose completion score can gate promotion.
3. Train KDA/sigmoid/NoPE with one and two global layers from the base stage. Compare final-only,
   middle-only, and middle-plus-final placement only after the evaluation adapter is ready.
4. Build or acquire data with genuine 64K–128K dependencies: long books/papers, repository trees,
   connected documents, and held-out synthetic retrieval/aggregation tasks. Do not pad the current
   data with unrelated documents.
5. Repeat the 32K KDA/control stage on additional seeds if real completion metrics validate the
   contrastive result; the current long-document validation contains only 327,680 tokens.
6. Validate multi-token INT8 KV decode quality, fewer KV heads, and latent/MLA-style global
   projections after selecting the minimum viable global-layer count.
7. Promote to 128K only after the data and independent 32K gate pass; re-run original 4K loss at
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
- KDA's three-seed mean base loss is 0.00691 nats worse than the RoPE control. The paired 95%
  interval spans zero, but the strict per-seed tie gate passes only 2/3.
- KDA's 32K long-document loss is 0.01144 nats worse than RoPE on a small held-out split, even
  though it preserves original-4K loss better.
- The current KDA implementation isolates channel-wise decay with a full-rank output gate; it is
  not an exact reproduction of Kimi Linear's low-rank gate.
- Retrieval replication uses an internal deterministic diagnostic, not independent benchmark
  implementations.

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
- Kimi-transfer staircase:
  `~/.cache/speck/checkpoints/SpeckLC-150M-KimiTransfer131M-*`, step 2,000
- Kimi frontier repeats:
  `~/.cache/speck/checkpoints/{gdn-fla-sigmoid-rope,kda-sigmoid-nope}-seed-{43,44}`,
  step 2,000
- Matched Kimi 32K pair:
  `~/.cache/speck/checkpoints/SpeckLC-150M-KimiContext32K-{gdn-sigmoid-rope,kda-sigmoid-nope}`,
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
| `08580a4` | Reviewed Kimi Linear and froze the transfer hypotheses |
| `0936b31` | Made recurrent output gating configurable |
| `f544196` | Added first-class Kimi Delta Attention |
| `ff079bb` | Qualified KDA outputs, states, gradients, and decode on the RTX 3090 |
| `5ac69fd` | Added explicit NoPE global context promotion |
| `04fd8b1` | Recorded the late-switch NoPE loss/retrieval conflict |
| `a57c553` | Added deterministic MQAR, Palindrome, and 64-stack tasks |
| `03cb57b` | Added the synthetic memory training harness |
| `77aa007` | Corrected delta-mixer timescale initialization |
| `3a4f30c` | Recorded the calibrated three-seed MQAR comparison |
| `42366f2` | Recorded replicated MQAR distance/load scaling |
| `d1ed524` | Recorded Palindrome and Stack qualification |
| `f1cf341` | Added the one-intervention Kimi-transfer preparer |
| `0e38c11` | Recorded the 131M Kimi-transfer staircase and 128K curves |
| `ccb759b` | Recorded three-seed loss and retrieval replication |
| `f3f34ef` | Recorded matched 32K continuation, regression, and retrieval findings |

## Repository state at consolidation

- Test suite: 321 passed
- Training/evaluation processes: none
- GPU: idle
- Worktree before this findings ledger: clean
- Branch: `main`, 116 commits ahead of `origin/main`
- Push status: nothing pushed
- Free disk after checkpoint creation and cache cleanup: approximately 29 GiB
