# Flagship architecture evidence protocol

Status: planning contract v2, 2026-09-10. This document defines how the dense-width KDA/global
flagship architecture is selected and how its efficiency claims are supported.
[`architecture_plan_v2.json`](architecture_plan_v2.json) is the active machine-checked run and budget
contract; `architecture_plan.json` is its immutable predecessor.
Per-arm manifests become immutable before any output from their comparison is inspected.

The objective is not to imitate another model's operators or scale. It is to apply the same standard
of hardware-aware evidence: isolate the mechanisms that survive discovery, confirm them under the
flagship data and training recipe, demonstrate transfer with scale, and measure realized training and
serving cost. The paper may claim only the quality, FLOP, state, memory, latency, throughput, and
energy advantages that this chain directly measures.

## 1. Evidence ladder

Five evidence levels have different authority:

1. **Completed discovery:** the 150M studies establish candidates and negative results. They include
   the mixer screens, [Kimi-transfer staircase](../../findings/16_kimi_transfer_131m.md),
   [three-seed frontier replication](../../findings/17_kimi_frontier_replication.md), long-context
   activation, attention-gate rejection, and Reader Attention frontier. They motivate the grant
   matrix but cannot by themselves select the 350M–1.2B release geometry.
2. **Grant decisions:** D4/D6 calibrate on the qualified incumbent in P1. C0, D2, D3, D7, and D8
   then run on the frozen tokenizer, E2 stable data, optimizer, schedule, and paired data order. They
   isolate the remaining launch settings at 350M.
3. **Integrated validation:** five-seed I1 tests the E2-selected mixture across dense and C0 hybrid models;
   I2 tests the one assembled set of compatible promoted settings against exact C0. These close
   transfer and composition gaps without reopening data or component search.
4. **Scale and horizon transfer:** the I2-confirmed hybrid and matched dense control form a
   60M–750M ladder near 20 tokens per parameter. A separate 350M continuation tests mature-token
   reversal. The fixed 1.2B flagship is excluded from fitting and becomes a held-out realization.
5. **Systems realization:** randomized, interleaved measurements on GH200, RTX 3090, and CPU verify
   whether analytic FLOP/state advantages produce actual training and serving gains.

The completed [whole-architecture proxy evidence](../architecture-promotion-v1/evidence_matrix.json)
already reports a three-pair quality pass for the five-cache KDA/GQA hybrid versus dense, with lower
analytic FLOPs and faster training. Its own evidence record states that it does not isolate KDA,
NoPE, GQA, or the 3:1 ratio. The new program preserves that result as prior evidence and closes the
causal gaps rather than repeating the same comparison.

## 2. Shared control and decision matrix

C0 is the 350M, 10B-token KDA/sigmoid/NoPE 3:1 default at three paired seeds. D2, D3, D7, and D8 may
reuse C0 only when tokenizer, packed shards and order, model parent, optimizer, LR, batch, schedule,
precision, token horizon, evaluation manifest, seed, and analysis hash are identical. Any mismatch
requires a new control and reserve authorization; an approximate historical run is not a control.

| ID | Question | Alternative to C0 | New runs | GPU-h | Role |
| --- | --- | --- | ---: | ---: | --- |
| C0 | Shared launch parent | KDA/sigmoid/NoPE, 3:1 | 3 | 63 | control |
| D2 | Does partial position encoding improve quality without breaking length transfer? | partial RoPE on 32/128 global dimensions | 3 | 63 | launch decision |
| D3 | Can fewer global layers preserve quality and retrieval while reducing state? | 5:1 recurrent/global ratio | 3 | 63 | launch decision |
| D4 | What peak LR and global batch are stable and efficient? | four LR points at about 1M-token batch, 4B tokens | 4 | 33 | calibration |
| D6 | How should optimization transfer with width? | two additional 150M LR/width anchors | 2 | 5 | transfer rule |
| D7 | Is channel-wise KDA necessary under the selected NoPE parent? | FLA-initialized scalar-decay GDN/sigmoid/NoPE | 3 | 63 | causal attribution and launch decision |
| D8 | Is sigmoid the correct KDA output gate? | KDA/SiLU/NoPE | 3 | 63 | causal attribution and launch decision |
| | | | **21** | **353** | |

D7 holds block positions, head and state dimensions, convolution, gate, position treatment, and
training recipe fixed. It reports the residual parameter and analytic-FLOP difference and requires
both fixed-token and fixed-analytic-FLOP views; matched wall-clock is the primary operational view.
If either residual exceeds 1%, the arm remains useful as an operator-package comparison but cannot be
described as a pure decay-rule ablation.

D8 requires a pre-grant code successor because KDA currently hardcodes sigmoid. A missing field must
remain exactly equivalent to explicit sigmoid for old configs and checkpoints. SiLU and sigmoid must
have identical parameters, analytic FLOPs, state geometry, and export behavior before training.

## 3. Statistical and promotion rules

Every launch-setting comparison uses three paired seeds and the frozen architecture-promotion
policy: candidate-minus-control aggregate loss has an upper one-sided 95% bound at or below
+0.01 nats, every source bound is at or below +0.02 nats, and all mandatory 32K/128K capability and
original-4K retention gates pass. Report point estimates, bounds, all individual pairs, analytic cost,
fixed-token cost, fixed-FLOP cost, and fixed-wall-clock cost. No pooled average may hide a failed
seed, source, context length, or hardware envelope.

Non-inferiority only makes an alternative eligible. It promotes according to its declared purpose:

- D2 must strictly improve short-context quality or long-context retention without harming the other.
- D3 must clear the frozen 25% state-reduction threshold or the applicable measured systems threshold.
- D7 promotes GDN only if it is quality/capability non-inferior and improves steady training or
  serving cost by the frozen component threshold (10% for a simple component, 20% if it requires a
  custom runtime); otherwise KDA remains the default and the comparison supplies
  attribution.
- D8 has equal analytic cost, so SiLU promotes only if it shows a multiplicity-corrected improvement
  in either aggregate language loss or the declared long-context primary metric while remaining
  non-inferior everywhere else. A tie keeps sigmoid.

D4 is a screen, not an architecture claim. D6 is judged by held-out scale-prediction error, not by
the fit on its anchors. Decisions and defaults are frozen before day 21; an unresolved result takes
the documented default rather than causing more search.

## 4. Integrated validation before scale

[`integration_plan_v2.json`](integration_plan_v2.json) is a separate 122-hour contract because component
ablation is not evidence that components compose and hybrid-only data selection is not evidence that
data gains transfer.

- **I1:** five-seed 2×2 comparison at 150M/3B: matched dense versus C0 hybrid, each on the balanced
  prior and E2c-selected stable mixture. Report data and architecture main effects, their interaction,
  both held-out parser views, six-category guardrails, and fixed-token/FLOP/time views. I1 cannot
  change the E2 selection; an interaction narrows the transfer claim.
- **I2:** three-seed assembled configuration at 350M/10B versus exact C0. The treatment contains every
  compatible individually promoted D2/D3/D7/D8 setting and nothing else. Failure returns the entire
  flagship to C0; no output-dependent subset search is allowed. If all settings retain defaults, C0
  is already the confirmed assembly and the unused envelope is not spent.
- **I3:** hash-bound capability and systems analysis, including position/trailing loss, 32K/128K
  retrieval and composition, original-4K retention, measured cost, KDA-state boundary resets, and
  middle-versus-final global-cache suppression. These evaluation-only interventions test mechanism
  without creating another trained arm.

The scale ladder cannot launch before I2 resolves. This makes every fitted scale point a measurement
of the architecture that can actually become the flagship.

## 5. Scale and token-horizon reversal program

| Stage | Scale and tokens | New runs | GPU-h |
| --- | --- | ---: | ---: |
| Low anchors | 60M/1.2B and 220M/5B, hybrid and dense | 4 | 14 |
| Mid points | 150M/3B and 350M/7B, hybrid and dense | 4 | 35 |
| S1 reversal check | 750M/15B, hybrid and dense | 2 | 134 |
| S2 mature-token-horizon sentinel | 350M dense/hybrid continued from 7B toward approximately 23B | 2 | 85 |
| | | **12** | **268** |

The primary scale analysis jointly models validation loss against training FLOPs for hybrid and dense,
with uncertainty, residual diagnostics, and leave-one-scale-out sensitivity. It also reports
parameters, tokens, wall-clock, peak allocation, and time to fixed quality. This near-constant-token-
ratio ladder cannot select between distant model/token allocations. The flagship is therefore fixed
before results at 1.2B/400B, with 1.2B/320B as the throughput fallback, and is never used to fit the
curve.

S2 spends the former generic contingency on one predeclared question: whether the 350M hybrid effect
persists after a materially more mature token horizon. The exact endpoint is frozen from measured
GH200 throughput before either output, targeting about 23B tokens with a declared lower fallback when
the 85-hour ceiling requires it. S2 is a reversal sentinel, not population-level equivalence evidence.

## 6. Systems evidence

Analytic savings are necessary but insufficient. Training and serving measurements use pinned model
and artifact hashes, synchronized timers, fixed power limits and software, warmup exclusion, and at
least five randomized/interleaved AB/BA blocks where the hardware supports both arms. Record 1 Hz
power, temperature, clocks, throttling, device memory, and host load; pair each active block with an
idle baseline. Missing telemetry suppresses energy claims rather than being imputed.

Required views, reported separately:

- training tokens/s, achieved FLOP/s, model FLOP utilization, joules/token, peak allocation, and
  time/joules to fixed quality on GH200;
- prefill and decode analytic cost, TTFT/TPOT, throughput, maximum resident batch, peak memory, recurrent state,
  global KV state, workspace, and fragmentation at 4K, 32K, and 128K;
- fixed state, length-growing bytes/token, weights plus state, and runtime HBM state separately;
- persistent prefix bytes, transfer/restore time, and cache-miss recomputation where implemented;
- generated output tokens and wall-clock time to fixed task quality for the instruct release;
- GH200 and RTX 3090 native/Transformers parity plus CPU GGUF parity;
- eager and compiled fallbacks, checkpoint/resume identity, and failures rather than successful runs
  alone.

Time, energy, runtime memory, persistent memory, and output tokens are separate outcomes. A speedup
cannot be inferred from FLOPs, an energy gain cannot be inferred from time, and an analytic state
reduction cannot be substituted for peak allocated memory. DeepSeek-V4.1-Flash's reported 890 global
bytes/token is analytic context only, not a locally measured comparator; Speck's broader state claim
remains against named matched dense-GQA controls.

## 7. Paper evidence map

The architecture section is complete only when it can generate these artifacts from checked results:

1. Exact KDA/global equations, execution diagram, parameter/FLOP/state accounting, and implementation
   identity tests.
2. A completed-discovery intervention staircase, including failed and superseded claims.
3. A D2/D3/D7/D8 forest plot with seed pairs, source bounds, capability gates, and cost deltas.
4. Five-seed I1 data-by-architecture interaction and three-seed I2 assembled-recipe confirmation.
5. Hybrid-versus-dense quality/FLOP scaling curves, S2 mature-horizon reversal, and the flagship held out.
6. State and peak-memory curves over context length with their constant and length-growing terms.
7. Interleaved training and serving time/energy results on the named hardware.
8. Long-context capability, position/trailing loss, and original-4K retention through both extension stages.
9. Negative results: gated convolution, pure recurrence, attention output gating, late NoPE
   conversion, Reader Attention promotion failure, and every losing grant arm.

This is enough for a deep architecture paper because it connects component causality, scale transfer,
and hardware realization. It does not support a claim of architectural novelty for inherited pieces,
or parity with the absolute capabilities or development scale of a frontier laboratory.

## 8. Budget and flexibility

The decision matrix costs 353 GPU-hours and the scale/horizon program 268, for 621 architecture
GPU-hours. Data costs 493 hours and integrated validation costs 122. Together they require 1,236
GPU-hours, or 12.9 fully occupied four-GPU node days, before the flagship freeze. The grant retains
889 hours of protected reserve.

D7 and D8 are mandatory for the corresponding component claims; I1/I2 are mandatory for transfer and
assembly; S2 is mandatory for a mature-horizon claim. If throughput misses plan, cut nonmandatory low
scale anchors only after preserving the 350M and 750M transfer points. If an implementation or evidence
gate fails, keep the default and narrow the paper claim; do not spend reserve inventing a replacement
axis.

Primary methodological references: [DeepSeek-V2](https://arxiv.org/abs/2405.04434) for the combination
of architecture ablation, cache accounting, and measured efficiency, and
[DeepSeek-V3](https://arxiv.org/abs/2412.19437) for carrying previously validated mechanisms into a
new scale while isolating new contributions, and the
[DeepSeek-V4.1-Flash source audit](../../papers/48_deepseek_v4_1_flash.md) for separating prefill,
decode, layer-cache reuse, runtime state, persistent state, and output-token cost without importing its
operator package into grant 1.
