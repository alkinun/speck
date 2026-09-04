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
- Counterfactual needle pairs as a cheap content-sensitivity regression only. Association claims
  require a distractor mutation, exact answer, and candidate-choice checks.
- Sigmoid output gating for the recurrent mixer; it improves the matched 131M GDN model by
  0.03734 nats over SiLU and replicates tightly across three seeds.
- KDA/sigmoid/NoPE as the lead long-context research treatment. With 50% original-language replay,
  4K task adaptation preserves language loss and sustains template-conditioned lookup to 128K. A
  harder audit shows replicated target selection across held-out wording, but not robust eight-record
  exact decoding.
- GDN/sigmoid/RoPE as the short-loss control for subsequent KDA experiments.
- Position-binned and trailing-token loss for long-context comparisons.
- Mixed original-language replay for retrieval and long-context task adaptation.
- Speck Reader Attention as a research mechanism and paper result. Query-only readers can share a
  writer's exact cache with correct eager/checkpointed gradients, and adjacent sharing produces a
  reproducible high-batch decode gain. Readers must bind fresh memories: distance four fails
  retrieval even at fan-out one.

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
- Do not claim that raw contrastive sensitivity is retrieval. The distractor control proved that the
  base checkpoints react to arbitrary record changes without selecting the queried association.
- Do not launch 128K continuation until a task-appropriate inference adapter and real completion
  benchmarks are available.
- Do not promote global attention output gating. Its 32M headwise gain shrinks from 0.01394 to
  0.00234 nats at 131M while costing 3% throughput.
- Do not promote single-template exact retrieval as general retrieval.
- Do not tune more direct-lookup templates, response cues, or scalar candidate-loss weights against
  the current held-out set. Those branches have been measured and do not replicate the exact gate.
- Do not promote the three-cache Reader Attention topology. Strict paired base-loss equivalence and
  strict candidate retrieval each pass only 2/3 seeds, and the symbolic diagnostic loses route-edge
  retrieval while the five-cache control retains both edges perfectly.
- Do not launch MQA/MLA or residual-routing interactions on the reader topology merely because the
  implementation works. Its predeclared capability gate failed; interaction work remains blocked.

## Superseded pre-Kimi architectural hypothesis

The former `GGGA`-repeated pattern is not an efficient default. The leading compact design is:

> Predominantly Gated DeltaNet/sliding processing, one global integration layer around the middle,
> and one global retrieval/readout layer near the end.

This is a hypothesis, not a release selection. One seed cannot distinguish the middle-only and
two-layer long losses, and the two-layer model did not improve retrieval retention over final-only
in the internal diagnostic.

## Current architectural hypothesis

The conservative research architecture remains the five-cache KDA/sigmoid/NoPE recurrent-global
hybrid. It is not state-efficient—resident BF16 state is 481.47 MiB at 128K—but it is the only arm in
the Reader Attention study that has not failed a replicated internal capability gate.

Three-cache Reader Attention remains a useful non-promoted research candidate: it cuts persistent
state `1.66×`, reproduces a `1.31×` eager decode gain at high cached-slot load, and fixes the
retention cliff across 3/3 seeds. It also misses strict paired loss and candidate gates on one seed
each and loses the symbolic route edge. MQA/MLA compression and interaction studies must not use it
as a selected parent. Neither five caches nor three caches is a release architecture: KDA itself
misses one of three strict base-stage loss ties against the RoPE control and no independent task
suite has passed.

## Required next work

1. Define non-inferiority margins, capability floors, serving cost envelopes, and versioned
   evaluation manifests before another architecture promotion decision.
2. Run pinned independent suites:
   [RULER](https://github.com/NVIDIA/RULER),
   [NoLiMa](https://github.com/adobe-research/NoLiMa), and
   [HELMET](https://github.com/princeton-nlp/HELMET).
3. Build or acquire data with genuine 64K–128K dependencies: long books/papers, repository trees,
   connected documents, and held-out synthetic retrieval/aggregation tasks. Do not pad the current
   data with unrelated documents.
4. Add paired target/distractor training that explicitly rewards target-over-distractor invariance;
   ordinary answer loss and a weight-1 candidate-ranking auxiliary are insufficient.
5. Diagnose the Reader Attention route/payload asymmetry on an independent, larger-sample suite
   without tuning against the current 30-case stream. A new result must preserve the failed 3/3 gate.
6. Generalize two-hop intermediates beyond ten one-token nodes and test an objective that binds both
   edges directly. Five caches retrieve both symbolic edges perfectly; three-cache readers preserve
   payload but not route, so constituent qualification is architecture-dependent.
7. Repeat the 32K KDA/control stage on additional seeds if independent metrics validate the
   contrastive result; the current long-document validation contains only 327,680 tokens.
8. Add a bounded KDA reference for K3's `g_min=-5`. Fewer than 0.5% of trained values cross the
   bound, but the observed tail reaches below `-120`; kernel benefit requires a FlashKDA path.
9. Compare five-layer GQA3 with five-layer MQA1 and NoPE MLA only after the held-out exact gate
   passes under a predeclared replication policy. Do not use the failed reader topology as parent.
10. Promote to 128K only after the data and independent 32K gate pass; re-run original 4K loss at
   every stage.

## Known limitations

- Global-count training uses one seed per point.
- The seed study fixes packed-data order, so it is a lower bound on total variance.
- The long-document validation set contains only 16 documents and 338,711 tokens.
- Global RoPE uses simple fixed linear scaling, which causes measurable short-context erosion.
- Unadapted open-vocabulary passkey exact match is zero for every base/context checkpoint.
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
- Exact 128K retrieval remains a single-template, one-seed adaptation result. Hardened 4K tests add
  two-token answers and held-out wording: target specificity replicates, but eight-record exact
  decoding reaches only 80%, 65%, and 64% across curriculum streams.
- Retrieval-only adaptation damages original language loss by 0.808 nats for KDA and 4.637 nats for
  GDN. The successful KDA recipe requires 50% original-language replay.
- Symbolic two-hop route and payload edges reach 99–100%, but composition is 43% before chain
  supervision. A chain intervention passes one stream and fails two; staged distillation also fails.
- Three-cache Reader Attention has a `+0.00578`-nat mean paired loss cost across seeds 42–44, with a
  wide interval spanning zero, but strict per-seed equivalence passes only 2/3 and every mean source
  delta favors five caches.
- Three-cache reader retention and specificity pass 3/3 synthetic streams, but strict final
  candidate accuracy passes only 2/3; seed 43 ends one of thirty cases below the fixed gate.
- On the matched symbolic diagnostic, three-cache readers retain payload at 100% but route retrieval
  falls to 53% with chance specificity. Direct composition is therefore not comparable.
- Same-recipe CUDA adaptation is not bit deterministic: a seed-42 reader repeat changes both model
  hash and trajectory. It passes the gate but cannot replace the original run.
- Controlled systems results are single-process RTX 3090 measurements. Reader high-batch decode
  improves `1.31×` eager and `1.27×`–`1.28×` compiled, but compiled batch-one reader decode regresses
  `61%`–`64%` and measured peak allocation does not fall.
- The filesystem has approximately 8 GiB free after preserving every research checkpoint and
  deleting only explicit redownloadable caches; do not launch another multi-arm checkpoint family
  without a new storage audit.

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
- Structured retrieval adapters:
  `~/.cache/speck/checkpoints/SpeckLC-150M-StructuredRetrievalAdapt-*`, steps 200 or 400
- Attention-gate screens:
  `~/.cache/speck/checkpoints/SpeckLC-150M-AttentionGate{32M-*,131M-headwise}`,
  steps 489 or 2,000
- Reader Attention base arms:
  `~/.cache/speck/checkpoints/SpeckLC-150M-ReaderAttention131M-caches-{1,2,3}` and
  `SpeckLC-150M-ReaderAttention131M-seed{43,44}-caches-3`, step 2,000
- Reader distance controls:
  `~/.cache/speck/checkpoints/SpeckLC-150M-ReaderDistance131M-{caches-4,caches-4-far}`,
  step 2,000

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
| `b407584` | Added position-binned and trailing-token loss evaluation |
| `a5d765a` | Added distractor-controlled association specificity |
| `7c7a270` | Recorded trained KDA decay distributions |
| `7de13cc` | Recorded global attention-sink diagnostics |
| `7eb141e` | Added deterministic structured-retrieval adaptation |
| `11e0546` | Recorded exact retrieval length/load scaling |
| `97b7e99` | Added configurable global attention output gates |
| `7e5ef4c` | Recorded the matched 32M attention-gate screen |
| `e3e498c` | Rejected gate promotion after the 131M confirmation |
| `d7261f5` | Added packed-language replay to retrieval adaptation |
| `fc034c6` | Recorded replay-trained exact retrieval through 128K |
| `2cf8556` | Consolidated exact retrieval and K3-transfer findings |
| `6d6184f` | Added held-out templates and multi-token retrieval evaluation |
| `8c69876` | Recorded the single-template transfer failure |
| `000c92b` | Added template-split, multi-token retrieval adaptation |
| `466c82a` | Recorded the first template-diverse pilot |
| `ae95963` | Added the held-out manifest template |
| `dd730b1` | Recorded the three-template pilot |
| `f646a87` | Kept variable-shape retrieval validation outside compilation |
| `423d47b` | Recorded the first held-out-template pass |
| `94f4670` | Recorded direct eight-record training failure |
| `b48e3b4` | Recorded the two-to-eight-record curriculum |
| `1495f0b` | Rejected that curriculum after three-stream replication |
| `22143cd` | Added joint template, answer, and record-load training |
| `1a97b39` | Recorded joint-load results |
| `7bb7539` | Separated response-cue transfer from template transfer |
| `371f35b` | Recorded the unmatched response-cue control |
| `8ae18e1` | Added matched response-cue adaptation |
| `ffeced5` | Recorded matched response-cue failure |
| `aeb14c4` | Added symbolic two-hop edge and composition supervision |
| `8459e47` | Exposed symbolic auxiliaries in standalone evaluation |
| `22cc72f` | Isolated two-hop composition from near-perfect edge lookup |
| `e0ecfdb` | Added explicit intermediate-chain supervision |
| `ff348c1` | Rejected mixed chain supervision after replication |
| `805566e` | Added staged task curricula |
| `4ee60e5` | Recorded staged chain-to-direct failure |
| `769c95a` | Added optional first-token candidate-ranking loss |
| `1b8de34` | Rejected weight-1 candidate loss on held-out retrieval |
| `87aec06` | Recorded the matched Reader Attention distance base result |
| `37ebf25` | Pre-registered the Reader Distance retrieval decision |
| `1320e39` | Resolved reader-to-writer distance as a retrieval mechanism |
| `3472f5b` | Prepared three-cache seeds 43 and 44 |
| `18c43e7` | Recorded three-seed three-cache base replication |
| `ad16a16` | Pre-registered three-cache retrieval replication |
| `06cd167` | Recorded the failed 3/3 three-cache retrieval promotion gate |
| `e93b18e` | Pre-registered the matched symbolic composition comparison |
| `22f16c5` | Recorded the three-cache symbolic route-edge failure |
| `e12cd04` | Added compiled inference benchmark support |
| `aa6d8c5` | Pre-registered the thermally controlled systems matrix |
| `d19e3c2` | Recorded controlled eager and compiled Reader Attention systems results |

## Repository state at consolidation

- Test suite: 419 passed
- Lint: all checks passed
- Experiment/result JSON: 597 files parse successfully
- Training/evaluation processes: none
- GPU: idle
- Structured-retrieval checkpoint integrity: 23/23 directories have completion markers plus model,
  optimizer, and metadata files
- Reader Attention checkpoint integrity: 7/7 retained base/distance checkpoints match recorded
  model, optimizer, metadata, and timing hashes where applicable
- Worktree after the consolidation commit: clean
- Branch: `main`, 13 commits ahead of `origin/main`
- Push status: nothing pushed
- Free disk after checkpoint creation and deletion of only explicit redownloadable caches:
  approximately 8 GiB
