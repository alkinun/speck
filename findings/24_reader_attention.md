# 24 — Speck Reader Attention and the global cache-count staircase

> **Status: design, implementation, and static qualification only.** No language model has been
> trained with this mechanism. Every number below is an exact analytic or unit-tested quantity, not
> a measured quality result.

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

## What is not verified

- No training step of any arm has run. There is no loss, retrieval, or throughput result.
- One shared memory is built from layer-3 representations. Whether keys formed that early support
  content-addressable lookup at depth 19 is the central open risk, and it is exactly what the
  staircase measures. YOCO answers it affirmatively at 3B and 1.6T tokens with a half-depth writer;
  that is not evidence at 150M with a writer at the first global slot.
- Preflight proves execution and analytic matching only. It contains no quality signal.
- Decode and prefill latency are unmeasured. Readers reduce cache bytes and key-value projection
  work but each still reads the full cache per step, so a bandwidth-bound decode may improve far less
  than the `4.94×` state reduction suggests. Claiming a latency win requires the systems measurement.
- The mechanism has no fused kernel. Readers use the same SDPA path as any global layer.

## Declared protocol and gates

Run order, all at seed 42 first, then seeds 43 and 44 only for arms that pass:

1. `caches-1` — the maximum-sharing arm and the strongest test of the hypothesis.
2. `caches-2` — one memory refresh at mid-depth, the natural fallback if a single early memory is
   too shallow.
3. `caches-3` and `caches-1-mqa1` — only if the first two justify the frontier.

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
