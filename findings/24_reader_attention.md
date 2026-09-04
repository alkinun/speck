# 24 — Speck Reader Attention and the global cache-count staircase

> **Status: seed-42 cache-count frontier, systems measurements, and matched near/far distance test
> complete.** Reader-to-writer depth distance is independently sufficient to damage retrieval at
> fixed fan-out one, despite unchanged base loss and preserved language loss after adaptation.
> Three-cache base loss is replicated on seeds 42–44; retrieval replication and symbolic composition
> remain unmeasured, so the selection remains provisional.

## Question

Findings [06](06_context32k_local_vs_global.md) and [08](08_global_attention_frontier.md) asked how
many global attention *layers* a compact hybrid needs. The answer bought retrieval but priced it in
resident state: five global layers cost `504,860,160` bytes (481.47 MiB) of BF16 state at 128K while
the entire fifteen-layer KDA recurrence costs `1,543,680` bytes (1.47 MiB). Global key-value cache is
`99.7%` of the state budget of the lead architecture.

That frontier conflated two variables. Each global layer contributes both an *attention read* — a
content-addressable lookup that feeds its own depth — and a *key-value cache* — the stored
representation that makes the lookup possible. Every published ratio study we reviewed, including
Kimi Linear's `3:1` and Nemotron-H's 8%, moves the two together.

This experiment separates them:

> Holding depth, attention placement, parameters, and the number of attention reads fixed, how many
> distinct global key-value caches does a 150M hybrid actually need?

## Mechanism

**Speck Reader Attention** splits the global attention layer into two roles bound to a named memory.

- A **writer** is an ordinary global attention layer. It additionally publishes the exact keys and
  values it attends over under a memory label.
- A **reader** is a query-only attention layer. It owns a query projection, a query norm, and an
  output projection. It owns no key projection, no value projection, no key norm, and no cache. It
  attends over the memory its writer published, with the same causal relation.

The grammar addition is two fields on `AttentionSpec`: `memory` (a label) and `memory_role`
(`none`, `write`, or `read`). `speck/architecture.py` validates, across the whole execution plan,
that every memory has exactly one writer, that every reader follows its writer in depth, and that
reader and writer agree on head dimension, key-value head count, active RoPE dimension, and scope.
Existing architectures omit both fields, take the `none` default, and are byte-identical after a
`from_dict`/`settings()` round trip, so no checkpoint lineage or resume contract changes.

At runtime the model threads one memory dictionary through the block loop. A block returns the
memories it produced as ordinary outputs, so a published key-value pair is a real output of its
activation-checkpointed region rather than an escaped intermediate. Gradients therefore flow from
every reader back into the writer's key and value projections under both eager and checkpointed
execution; both are covered by tests.

Readers are position-free in the lead architecture. When a writer uses RoPE, its cached keys are
already rotated, and a reader rotating its own queries with the same module recovers the correct
relative positions. Reader RoPE is supported and tested but is not the arm we intend to train.

## Why readers need no extra parameters

An obvious worry is that one frozen memory cannot serve several depths that need different views of
the same context. The natural fix would be a per-reader adapter on the shared keys. That fix is
provably empty for any linear adapter.

For a low-rank or full-rank matrix `M`, re-keying the shared cache as `k' = k + Mk` gives

```text
q · (I + M)k = ((I + M)^T q) · k
```

so every linear re-view of the shared keys is exactly a linear re-map of the reader's queries, which
its own query projection already realizes. The same argument applies to a linear re-view of the
shared values, which the reader's output projection already realizes. A reader with a query
projection and an output projection is therefore **already maximally expressive among linear
re-views of a shared cache**, and adding key or value adapters would add parameters with no
representational gain. Only a nonlinear or content-dependent re-view could add capacity, and that
would reintroduce per-layer work proportional to cache length at decode time.

This is recorded as a design decision, not a claim about quality: it says a cheap adapter branch is
not worth an experiment, not that one memory is sufficient. The identity is unit tested.

## Cost accounting

A reader removes `2 · d_model · d_kv` matrix parameters (the key and value projections) and one
`head_dim` key-norm vector. It keeps the full attention score and value-aggregation compute, so its
length-dependent cost is unchanged.

Each arm restores the removed matrix parameters by widening the SwiGLU layer inside the same block.
At the lead geometry this is exact: `2 · 768 · 192 = 294,912` reclaimed parameters equal
`3 · 768 · 128`, so every reader block widens its feed-forward layer from `2,304` to `2,432`. Analytic
training FLOPs per token are then identical across every arm, and parameter counts differ only by the
`64`-element key-norm vector each reader drops.

Exact key-value-head reduction is only available when the writer keeps a head count divisible by
three, because every feed-forward compensation is a multiple of `3 · 768 = 2,304`. The
multi-query arm is therefore explicitly *not* matched and reports its residual.

## The prepared staircase

`scripts/reader_attention_prepare.py` rewrites the lead KDA/sigmoid/NoPE architecture into arms that
share `k` caches across the same five global attention slots at logical layers 3, 7, 11, 15, and 19.
Writers are placed evenly and each remaining slot reads the closest preceding writer.

| Arm | Caches | Reader layers | Parameters | Δ FLOP/token | BF16 state @128K | Fraction | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `caches-5` | 5 | 0 | 153,958,938 | 0 | 504,860,160 B | 1.000 | 1.00× |
| `caches-3` | 3 | 2 | 153,958,810 | 0 | 303,533,568 B | 0.601 | 1.66× |
| `caches-2` | 2 | 3 | 153,958,746 | 0 | 202,870,272 B | 0.402 | 2.49× |
| `caches-1` | 1 | 4 | 153,958,682 | 0 | 102,206,976 B | 0.202 | 4.94× |
| `caches-1-mqa1` | 1 | 4 | 153,957,914 | −4,608 | 35,098,112 B | 0.070 | 14.38× |

BF16 state at the shorter lengths, for the same arms in order: `17,272,320`, `10,980,864`,
`7,835,136`, `4,689,408`, and `2,592,256` bytes at 4K; `127,372,800`, `77,041,152`, `51,875,328`,
`26,709,504`, and `9,932,288` bytes at 32K.

The `caches-5` arm is byte-identical to the source architecture. Its `model.json` round trips to the
same canonical settings, so the completed seed-42 `kda-sigmoid-nope` checkpoint from finding
[16](16_kimi_transfer_131m.md) is a valid `caches-5` result and **must not be retrained**.

The staircase's own state accounting reproduces the independently recorded `504,860,160`-byte
figure from findings [17](17_kimi_frontier_replication.md) and [18](18_kimi_context32k.md) exactly,
which is the cross-check that the new arms are measured on the same basis as the existing frontier.

## What is verified

The repository suite is `415 passed` under the pinned CUDA environment, up from the `383` recorded
at the last consolidation. On the CPU environment it is `410 passed, 5 skipped`. New coverage:

- a reader model is numerically identical to a model whose reader slot is an ordinary attention
  layer holding the writer's key and value projections, with the residual stream frozen between the
  two depths, at both `rope_dim=0` and full RoPE;
- incremental single-token decode and split-prefill both match a full forward for writer/reader
  models;
- a four-slot reader model allocates exactly one attention cache, with byte-for-byte equality to a
  single-global-layer model and one quarter of the four-cache model;
- activation checkpointing preserves reader loss and every gradient;
- gradients reach the writer's key and value projections through the readers;
- parameter and FLOPs accounting drop exactly the key-value projections and keep the full score term;
- the linear-absorption identity;
- architecture validation rejects a reader without a writer, a reader before its writer, more than
  one writer per memory, a writer inside a repeated block group, geometry mismatches, and non-global
  scope;
- the preparer's parameter, FLOPs, cache-count, and state-byte accounting for every arm.

## Preflight

Each arm completed compiled forward, backward, gradient clipping, and a Muon update at the
production geometry before any training was proposed. Hardware and software match the research
contract exactly: RTX 3090, driver 610.43.03, PyTorch `2.9.1+cu128`, FLA `0.5.0`, batch 4,
accumulation 4, sequence length 4,096, torch loss backend, three warm-up and five measured steps.

| Arm | GFLOP/token | tok/s | Median step | Peak allocated |
| --- | ---: | ---: | ---: | ---: |
| `caches-5` | 1.021601 | 45,362.3 | 1.4442 s | 13.87 GiB |
| `caches-1` | 1.021601 | 45,255.6 | 1.4473 s | 13.85 GiB |
| `caches-2` | 1.021601 | 45,154.1 | 1.4518 s | 13.85 GiB |

Three observations. Analytic FLOPs per token are bit-identical across the arms, confirming the
feed-forward compensation. Realized throughput spans `0.46%`, so sharing a cache introduces no
compile regression or graph-break penalty; these five-step measurements prove execution, not stable
throughput, and the full runs remain authoritative. Peak training allocation is unchanged, which is
expected and worth stating plainly: at a 4,096-token training length the key-value cache is not the
dominant allocation, and readers additionally hold the published keys and values live. **The state
reduction in this finding is a decode-time resident-state result, not a training-memory result.**

The `caches-5` preflight also reproduces the earlier KDA/sigmoid/NoPE preflight from finding
[16](16_kimi_transfer_131m.md) — `45,362` versus `45.3K` tok/s and `13.87` versus `13.9` GiB — which
confirms the measurement basis has not drifted.

## Seed-42 discovery result

Both new arms trained to the full budget on the frozen corpus with the same data order, schedule,
and evaluation sample as the existing control.

| Arm | Caches | Readers | Final loss | Versus control | Per reader | Sources worse | BF16 state @128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `caches-5` | 5 | 0 | 2.795380 | — | — | — | 504,860,160 B |
| `caches-3` | 3 | 2 | 2.800078 | +0.004698 | +0.00235 | 11/11 | 303,533,568 B |
| `caches-2` | 2 | 3 | 2.803611 | +0.008230 | +0.00274 | 11/11 | 202,870,272 B |
| `caches-1` | 1 | 4 | 2.803232 | +0.007852 | +0.00196 | 11/11 | 102,206,976 B |

**There is no cliff.** Every arm sits inside the `0.00965`-nat seed range against the control, and
the cost per converted layer is stable between `0.00196` and `0.00274` nats. Sharing degrades
gracefully and roughly in proportion to how many independent caches are removed, rather than
collapsing once some minimum count is crossed. The endpoint buys a `4.94×` resident state reduction
at 128K for an aggregate cost one seed cannot resolve.

Three further conclusions, in decreasing order of confidence.

**One and two caches are indistinguishable.** They differ by `0.000379` nats, an order of magnitude
inside the measured seed range. The cost of sharing is a step from five caches down to two and is
then flat. The marginal second cache buys no measurable language modeling, so one cache dominates
two on every axis: the same loss at half the resident state. `caches-2` should be dropped from the
frontier rather than replicated.

**The deficit versus five caches is real in direction but unresolved in size.** The aggregate
`0.008` nats is inside the `0.00965`-nat seed range, so by the standing convention it is unresolved
on one seed. Its direction is not ambiguous: every one of the eleven validation sources is worse
for both arms. For `caches-1` the deficit is structured rather than scattered, ranging from
`+0.0022` on `ufw_l3_multi_style` to `+0.0191` on `math_multi_style`, with the three mathematics
sources and peS2o taking the largest hits and the general web sources the smallest. Noise would
scatter in sign; this does not. The honest statement is a small real penalty whose magnitude
requires seeds 43 and 44.

**A second writer does not by itself rescue two caches.** `caches-2` adds a writer at logical layer
11, the mid-depth integration point identified in finding [08](08_global_attention_frontier.md),
and recovers nothing. That result previously motivated a stronger rejection of the shallow-writer
explanation, but it did not isolate distance: the later readers still share the new memory at
distance two and fan-out two. The matched distance control below shows that distance four fails even
at fan-out one. The supported statement is therefore narrower: adding one mid-depth writer is
insufficient, while memory freshness remains causal.

Throughput is deliberately not claimed. The three runs executed back to back on an uncooled
consumer card, and the observed spread between analytically FLOP-matched arms was larger than the
effect being measured. A throughput claim requires a thermally controlled, interleaved measurement.

## Three-seed `caches-3` base replication

The selected three-cache topology was trained from scratch at seeds 43 and 44 under the same frozen
packed-data order and compared with the corresponding five-cache KDA/sigmoid/NoPE seeds. This varies
initialization and optimization while retaining the exact data stream.

| Seed | Five-cache loss | Three-cache loss | Paired delta | Sources worse | Strict tie |
| ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 2.795380 | 2.800078 | +0.004698 | 11/11 | pass |
| 43 | 2.793794 | 2.805996 | +0.012202 | 11/11 | fail |
| 44 | 2.800956 | 2.801406 | +0.000450 | 5/11 | pass |
| mean | 2.796710 | 2.802493 | +0.005783 | 11/11 mean deltas | 2/3 |

The mean paired penalty is small and below the `0.00965`-nat reference range, but **strict per-seed
loss equivalence does not replicate**: seed 43 loses by `0.012202` nats. With only three pairs, the
paired 95% t interval is wide (`-0.008999` to `+0.020565`) and spans zero. The correct claim is a
directional mean penalty whose magnitude remains unresolved, not loss non-inferiority.

The source structure supports direction without resolving size. Seeds 42 and 43 are worse on all
eleven sources; seed 44 splits five worse and six better. Nevertheless, every source has a positive
mean paired delta, largest on `math_multi_style` (`+0.00866`), peS2o (`+0.00781`), and
`math_textbook_exercise` (`+0.00747`). All three reader runs complete stably, and realized training
throughput differs by less than `0.1%` across their seeds, so seed 43 is not an execution outlier.

This weakens but does not retire `caches-3`. Its mean loss cost remains small, its state and batched
decode gains are measured, and its seed-42 retrieval matches the five-cache control. The next gate is
therefore retrieval replication on seeds 43 and 44, not a claim that the loss trade-off disappeared.
Machine-readable paired deltas and hashes are checked at
[results/SpeckLC-150M-ReaderAttention131M/replication/caches-3.json](../results/SpeckLC-150M-ReaderAttention131M/replication/caches-3.json).

## Retrieval gate: sharing costs retention, not learnability

Each arm was adapted from its own seed-42 base checkpoint with finding
[22](22_template_diverse_retrieval_adaptation.md)'s joint-load recipe: four training templates,
letters and phrases, two and eight records, 400 steps, 50% original-language replay, and validation
on the held-out `directory` template with unseen phrase answers.

| Arm | Readers | Readers per memory | Peak candidate | Final candidate | Retention | Final specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `caches-5` | 0 | 0 | 0.933 | 0.867 | 0.93 | 1.000 |
| `caches-3` | 2 | 1 | 0.900 | 0.833 | 0.93 | 1.000 |
| `caches-2` | 3 | 2 | 0.900 | 0.633 | 0.70 | 1.000 |
| `caches-1` | 4 | 4 | 0.767 | 0.300 | 0.39 | 0.833 |

**Every arm learns the task.** Peak held-out candidate accuracy moves only from `0.933` to `0.767`
across the entire frontier, and three of the four arms peak at `0.90` or above. Sharing a cache does
not prevent a compact hybrid from acquiring template-robust, distractor-controlled retrieval.

**What collapses is retention.** The ratio of final to peak accuracy falls monotonically as more
readers depend on a single memory: `0.93`, `0.93`, `0.70`, `0.39`. `caches-5` and `caches-3` rise and
hold. `caches-2` reaches `0.90` at step 150 and then oscillates between `0.53` and `0.80`.
`caches-1` reaches `0.767` at step 200 and falls to `0.300`, losing a capability it demonstrably had.

**Language-model loss predicts none of it.** Every arm is inside the `0.00965`-nat seed range on
validation loss, and the loss frontier is smooth and graded at roughly `0.002` nats per converted
layer. A reader hybrid can be loss-equivalent and still lose a retrieval capability it had held
minutes earlier. This is the hidden-cliff failure the [MiniMax-M2 note](../papers/15_minimax_m2.md)
warns about, reproduced under a controlled architecture ablation.

**Target selection survives where value decoding does not.** Association specificity remains perfect
for five, three, and two caches and falls only to `0.833` for one cache, still significant at
`p = 1.6e-4`. Even the collapsed arm knows which record the query names; it stops emitting that
record's value. This is the same selection-versus-decoding split reached by curriculum in findings
[22](22_template_diverse_retrieval_adaptation.md) and [23](23_symbolic_two_hop_composition.md),
reached here by architecture.

A halved adaptation learning rate does not rescue `caches-1`, but the control is underpowered rather
than decisive: at `5e-5` the arm never reaches the `0.767` peak it attains at `1e-4`, so it does not
separate optimization instability from a representational limit.

The two candidate mechanisms are confounded by the even-spacing rule that places writers. Readers
per memory and reader-to-writer depth distance rise together across these arms, so this frontier
cannot yet say whether a memory degrades because too many layers depend on it or because it is read
too far below where it was written. Separating them requires a layout that holds one fixed while
varying the other, and that is the next experiment.

## Depth-distance base result

The matched distance experiment holds four caches, one reader, one reader per memory, parameters,
analytic FLOPs, data order, seed, and the L20 reader position fixed. Only the reader's memory binding
changes: `caches-4` reads the preceding L16 writer at attention-slot distance one, while
`caches-4-far` reads the L4 writer at distance four.

| Arm | Reader distance | Final loss | Versus five caches | Versus near |
| --- | ---: | ---: | ---: | ---: |
| `caches-5` | — | 2.795380 | — | +0.001704 |
| `caches-4` | 1 | 2.793676 | -0.001704 | — |
| `caches-4-far` | 4 | 2.793098 | -0.002282 | -0.000578 |

There is **no base-loss evidence for a depth-distance penalty**. The far reader is nominally
`0.000578` nats better than the near reader, far inside the `0.00965`-nat one-seed resolution floor.
Source-level signs split almost evenly: six favor far and five favor near. Both four-cache arms are
also nominally better than the five-cache control, with eleven of eleven sources better for near
and ten of eleven for far, but those differences are likewise unresolved on one seed.

This result does not decide the retention mechanism. The cache-count frontier already established
that language-model loss does not predict retrieval retention, so a flat near/far base result cannot
exonerate distance. The matched retrieval adaptation is still the decision gate: if far loses a
capability that near holds, distance selectively damages retention; if both hold, fan-out or gradient
coupling through a shared writer becomes the surviving explanation. Machine-readable endpoints and
hashes are checked at
[results/SpeckLC-150M-ReaderDistance131M](../results/SpeckLC-150M-ReaderDistance131M).

**Pre-registered distance decision rule.** Before either distance arm is adapted, `holds` retains the
already-published final candidate-accuracy gate of `>= 0.80` on the same 30 fixed held-out cases.
Near-pass/far-fail supports depth distance; both-pass rejects distance at this diagnostic's resolution
and requires the fixed-distance fan-out control; near-fail makes the experiment inconclusive. Peak,
final-over-peak retention, exact match, and specificity are reported as mechanistic diagnostics, not
substituted post-hoc gates. Each adapter also receives the same 20M-token original-corpus evaluation;
an increase greater than the `0.00965`-nat seed range against its own base checkpoint makes the
comparison inconclusive because adaptation forgetting would be a competing explanation.

### Distance retrieval result

| Arm | Reader distance | Peak candidate | Final candidate | Retention | Final exact | Final specificity | Original-loss delta | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `caches-4` | 1 | 0.933 | 0.933 | 1.00 | 0.900 | 1.000 | +0.004004 | pass |
| `caches-4-far` | 4 | 0.600 | 0.400 | 0.67 | 0.400 | 0.833 | +0.003180 | fail |

The pre-registered outcome is **near-pass/far-fail**, supporting reader-to-writer depth distance.
The final candidate gap is `0.533`, or sixteen of the thirty fixed cases. The adjacent reader
improves through step 400 and finishes at its peak. The far reader peaks at only `0.600` at step 150,
falls to `0.400`, and remains between `0.367` and `0.433` over the final three evaluations. Distance
therefore limits both acquisition and retention in this control; this is not merely a late-training
collapse.

The competing explanations fail their controls. Far has nominally *better* base loss than near by
`0.000578` nats, and both adapters retain original-language loss inside the pre-registered limit:
`+0.004004` near and `+0.003180` far against their own parents. The retrieval split is not predicted
by base loss and is not explained by differential forgetting.

Target sensitivity again survives better than answer decoding. The far arm's target-direction
accuracy remains `1.000`, and specificity is still `0.833` (`p = 1.6e-4`), while candidate and exact
accuracy fall to `0.400`. The model still reacts preferentially to the named record but cannot
reliably emit its value from a four-slot-old memory.

The architectural consequence is periodic **nonlinear memory refresh**. By the absorption lemma, a
linear key or value adapter at the reader cannot restore anything its query/output projections do
not already express. At this scale, the supported topology binds readers to the immediately
preceding writer, as `caches-3` does. The experiment establishes distance one as working and distance
four as failing; it does not locate the threshold between them or exclude an additional fan-out
penalty.

## Measured systems result

Prefill and cached decode were measured with `scripts.inference_benchmark` on the pinned RTX 3090
stack, uncompiled, with preallocated Speck state.

| Cached token slots | Reached as | `caches-5` | `caches-3` | `caches-1` |
| ---: | --- | ---: | ---: | ---: |
| 32,768 | 32K × 1 | 6.872 ms | 6.644 ms (1.03×) | 6.523 ms (1.05×) |
| 131,072 | 32K × 4 | 7.332 ms | 6.954 ms (1.05×) | 6.790 ms (1.08×) |
| 131,072 | 128K × 1 | 6.833 ms | 6.643 ms (1.03×) | 6.563 ms (1.04×) |
| 524,288 | 32K × 16 | 14.805 ms | 11.304 ms (1.31×) | 7.751 ms (1.91×) |
| 524,288 | 128K × 4 | 14.680 ms | 11.163 ms (1.32×) | 7.649 ms (1.92×) |

**Prefill does not improve.** Every arm lies within 2% at both 32K and 128K: `419`–`425` ms and
`3222`–`3286` ms respectively. This is the expected result and it must be stated plainly. Readers
remove key and value projections but still compute the full attention score and value aggregation,
so sharing a cache is not a prefill optimization.

**Single-stream decode barely moves.** At batch 1 the gain is `4`–`5%` at either length. A batch-1
deployment does not benefit from this mechanism.

**Batched decode improves substantially, and the governing variable is total cached token slots.**
At `524,288` slots the maximum-sharing arm decodes `1.9×` faster and the selected three-cache arm
`1.3×` faster. The same speedup appears whether those slots come from 32K at batch 16 or 128K at
batch 4, which is what a bandwidth-bound decode predicts: what matters is how many cached bytes must
be read per step, not how they are arranged. Peak allocation at that point falls from `12.33` to
`11.21` GiB.

The efficiency claim is therefore specific rather than general: no prefill change, negligible
single-stream change, and a near-two-fold batched decode gain at the sharing level that fails the
retrieval gate. The arm that passes the gate keeps a `1.3×` gain.

## Current selection

`caches-3` is the arm to carry forward. It passes the retrieval gate with retention identical to the
five-cache control, its loss deficit of `0.004698` nats is half the endpoint's and inside the seed
range, and it cuts 128K resident state by `1.66×`. `caches-1` remains the more valuable scientific
point precisely because it fails: it establishes that the mechanism has a boundary and locates it.

The distance control strengthens this selection mechanistically: both `caches-3` readers bind the
immediately preceding writer, matching the topology that passes at distance one. It does not yet
justify a universal maximum distance or prove that fan-out is harmless.

Base replication tempers the quality claim: `caches-3` has a `+0.005783`-nat mean paired cost and a
wide interval spanning zero, but it passes the strict per-seed tie on only two of three seeds. It is
the arm to carry into replicated capability gates, not yet a loss-noninferior promotion.

## What is not verified

- Three-cache base loss is replicated across seeds 42–44, but retrieval and systems remain one seed
  per arm.
- Composition is unmeasured.
- Systems numbers are uncompiled single-process measurements on one consumer card, not a serving
  benchmark, and they exclude tokenization, scheduling, and multi-request batching effects.
- Retrieval uses the internal distractor-controlled diagnostic, not RULER, NoLiMa, or HELMET.
- The distance result is one seed on the internal 30-case diagnostic. It establishes a boundary
  between attention-slot distances one and four but does not locate that boundary or rule out an
  additional fan-out penalty. YOCO's half-depth writer at 3B and 1.6T tokens does not override this
  150M result.
- Preflight proves execution and analytic matching only. It contains no quality signal.
- The mechanism has no fused kernel. Readers use the same SDPA path as any global layer.

## Declared protocol and gates

Run order, all at seed 42 first, then seeds 43 and 44 only for arms that pass:

1. `caches-1` — complete.
2. `caches-2` — complete, and retired: indistinguishable from `caches-1` at twice the state.
3. `caches-3` — complete; it lies between the endpoints and shows the cost is graded.
4. `caches-1-mqa1` — only after the capability gates.

One fallback is pre-registered before any result is seen. If `caches-1` loses on loss or retrieval
while `caches-2` holds, the most likely cause is that a memory written at logical layer 3 is too
shallow, not that sharing is wrong. The designed follow-up is a deep-writer arm that keeps five
attention operations, converts the two slots before the writer to sliding-window attention, writes
the single memory at logical layer 11, and reads it at 15 and 19. Its 128K state is one global cache
plus two bounded windows. Declaring it now prevents it from becoming a post-hoc rescue.

Preflight and training use different compile options: the benchmark applies `max_autotune` and
`coordinate_descent_tuning`, while `scripts/base_train.py` adds `aggressive_fusion`. Preflight
throughput is therefore comparable with the earlier preflights in finding
[16](16_kimi_transfer_131m.md), not with full-run throughput.

Every arm reuses the frozen packed corpus manifest
`b84b09e0b701e35d84487cf6f91e6da9c9fb686b7f6efe67b2e2f5f301fda98e`, the same data order, Muon, the
cosine schedule, `131,072,000` tokens, sequence length 4,096, and the same evaluation sample as the
existing staircase.

Promotion gates, in order:

1. **Loss.** Final validation loss versus the `caches-5` three-seed mean of `2.796710`. The measured
   `0.00965`-nat seed range remains the resolution floor; a deficit inside it is unresolved, not a
   tie, until replicated on three seeds.
2. **Retrieval.** The distractor-controlled structured evaluation from findings
   [19](19_retrieval_specificity_and_replay.md) and [22](22_template_diverse_retrieval_adaptation.md),
   after the standard 4K adaptation with 50% original-language replay. Report held-out-template
   candidate accuracy and association specificity at two and eight records. Raw counterfactual
   direction is a sensitivity diagnostic only.
3. **Composition.** The symbolic two-hop diagnostic from finding
   [23](23_symbolic_two_hop_composition.md). Constituent edges already reach 99–100% with five
   caches; composition sits at 43%. Whether shared-memory readers change composition is the
   mechanistic question worth reporting either way.
4. **Systems.** Resident state, 128K prefill latency, and decode latency at batch 1 and at the
   maximum resident batch. State bytes alone are not a serving result.

A negative result is publishable here. If `caches-1` loses materially on loss or retrieval while
`caches-2` holds, the finding is that a compact hybrid needs periodic memory refresh rather than one
global memory, which is a direct and previously untested answer to the question YOCO leaves open.

## Blocking constraint

Approximately 11 GiB of filesystem headroom remains, and the standing decision in
[09](09_decisions_and_change_log.md) is not to launch another checkpoint family at that headroom.
Each 131M-token arm writes one completed checkpoint of roughly 1.3–1.9 GiB. Two arms fit; the full
five-arm, three-seed frontier does not. Resolve the retention question before launching more than
the first two arms.

## Artifacts

- Grammar and validation: [speck/architecture.py](../speck/architecture.py)
- Runtime: [speck/model.py](../speck/model.py)
- Preparer: [scripts/reader_attention_prepare.py](../scripts/reader_attention_prepare.py)
- Preflight records:
  [results/SpeckLC-150M-ReaderAttention131M/preflight](../results/SpeckLC-150M-ReaderAttention131M/preflight)
- Prepared contract:
  [experiments/SpeckLC-150M-ReaderAttention131M/staircase.json](../experiments/SpeckLC-150M-ReaderAttention131M/staircase.json)
- Tests: `tests/test_architecture.py`, `tests/test_model.py`,
  `tests/test_reader_attention_prepare.py`
