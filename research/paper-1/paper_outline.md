# Paper 1 manuscript outline

## Working title

**Speck: Evidence-Driven Information Routing Across Sequence, Depth, and Width**

The title is provisional. It must change if the promoted novelty is narrower than the tri-axis framing.

## Abstract

The abstract may contain only quantities linked to checked result artifacts. It must state:

1. the efficiency problem and target deployment profiles;
2. the Speck-specific novelty in one sentence;
3. the matched baselines and training scales;
4. the primary quality and cost effects with confidence bounds;
5. the longest trained, effective, and usable context separately; and
6. released code, configurations, kernels, checkpoints, and raw results.

No abstract draft should be treated as stable before the final removal matrix.

## 1. Introduction

- Define quality per training cost and quality per serving cost without a composite vanity score.
- Explain sequence-, depth-, and width-wise information flow.
- State the central claim, component claims, and non-claims.
- List contributions with one result-backed sentence each.
- Distinguish architectural effects from data, optimizer, precision, and post-training effects.

Required figure: one architecture overview with state growth and active-compute annotations.

## 2. Background and design requirements

### 2.1 Exact, local, linear, compressed, and sparse sequence mixing

Define causal global attention, SWA, GDN/KDA, GQA/MQA/MLA, HCA, and CSA in a common notation. Compare
their training/prefill complexity, decode arithmetic, resident state, and information bottlenecks.

### 2.2 Depth-wise information flow

Define PreNorm residual accumulation, Full Attention Residuals, and Block Attention Residuals. State
activation-memory and pipeline-communication implications.

### 2.3 Width-wise conditional compute

Define dense SwiGLU, conventional routed experts, and each Stable LatentMoE intervention separately:
normalized latent projection, bounded activation, and balancing rule.

### 2.4 Design targets

Reference the versioned quality margins, local/throughput/long-context cost envelopes, target hardware,
runtime assumptions, and paper-scale launch gate.

Required table: every candidate operator with parameters, active parameters, FLOPs, state bytes, kernel
status, and expected bottleneck.

## 3. Speck architecture

This section is written only after component promotion.

### 3.1 Sequence hierarchy

- Exact equations and pseudocode for every retained branch.
- Layer grammar, ratio, placement, local window, compression rate, selection budget, positional scheme,
  and attention sink behavior.
- Cache/state transition during prefill, decode, prefix reuse, eviction, and resume.
- Explanation of how the design differs from Kimi Linear, Kimi K3, and DeepSeek-V4.

### 3.2 Depth routing

- Exact residual source set, normalization, initialization, aggregation, block size, and online update.
- Training/checkpointing and incremental-decode state.

### 3.3 Width routing

- Expert geometry, total and active parameters, shared experts, top-k, capacity behavior, balancing,
  activation function, and precision.
- Expert-parallel communication and single-device fallback.

### 3.4 Complexity and resource accounting

Give closed-form expressions and checked numeric examples for 4K, 32K, 128K, and any longer claim.
Separate theoretical arithmetic, measured kernel work, persistent state, peak allocation, and weight
memory.

## 4. Algorithms and systems co-design

### 4.1 Reference operators and correctness

Torch references, forward/state/gradient parity, causal tests, incremental equivalence, determinism,
numerical tolerances, and adversarial shapes.

### 4.2 Training kernels

Chunkwise/recurrent/compressed/sparse/MoE kernels, fusion boundaries, compilation, activation
checkpointing, and achieved utilization.

### 4.3 Inference runtime

Paged heterogeneous state, prefix caching, unfinished compression tails, batch scheduling, quantized
state, continuous batching, and failure recovery.

### 4.4 Distributed execution

Data, tensor, expert, sequence/context, and pipeline parallelism. Report communication volume and
overlap rather than only ideal FLOPs.

Required evidence: reference-vs-kernel parity tables and roofline/bottleneck plots on every claimed
hardware family.

## 5. Experimental methodology

### 5.1 Baselines

Include a conventional dense global-attention Transformer, the conservative five-cache KDA/GQA control,
GDN hybrid, and direct published-parent replications needed to identify the contribution.

### 5.2 Matching and statistical design

Report fixed-token, fixed-analytic-FLOP, fixed-active-parameter, fixed-total-parameter, and
time-to-quality views. Use paired seeds/data orders, one-sided non-inferiority, confidence bounds,
multiplicity correction, predeclared early stops, and all failed runs.

### 5.3 Data and tokenizer

Source revisions, licenses, filtering, deduplication, contamination analysis, mixture weights,
curriculum, document-length distribution, tokenizer fertility, and validation split isolation.

### 5.4 Training

Optimizer roles, schedules, batch/sequence curriculum, precision, initialization, clipping, stability
interventions, rollback policy, hardware, software, energy, checkpoints, and total compute.

### 5.5 Evaluation

Base loss, general knowledge, code, math, reasoning, synthetic mechanisms, independent long-context,
post-training retention if applicable, and realistic serving workloads. Pin prompts, parsers, sample
counts, generation settings, and revisions.

## 6. Component experiments

### 6.1 Sequence axis

Isolate recurrent mixer, exact-cache representation, compression, sparse selection, raw local branch,
global-layer ratio, and placement. Do not compare a full sequence stack until its children pass.

### 6.2 Depth axis

Standard residual versus Full and Block AttnRes, followed by a depth/width cross-sweep. Include hidden
magnitude, update fraction, gradient norm, routing weight/entropy, source age, and activation memory.

### 6.3 Width axis

Dense SwiGLU versus conventional MoE, then add normalized latent projection, bounded activation, and
balancing one at a time. Include expert load, routing margins, dropped tokens, specialization,
communication, total memory, and small-batch serving.

### 6.4 Interaction and removal matrix

Test promoted pairwise interactions before the full composition. Remove every retained component from
the final candidate. A component that no longer earns its complexity is deleted.

Required figures: effect sizes with confidence intervals, not only endpoint score tables.

## 7. Scaling behavior

- At least five compute-optimal points for the final candidate and strongest conventional baseline if a
  scaling-efficiency claim is made.
- Fit the same functional family with disclosed fitting method and uncertainty.
- Report residuals and out-of-fit confirmation points.
- Separate active, total, embedding, and state parameters.
- Test architecture ranking over training horizon, not only model size.

Required figure: loss versus training compute plus time-to-quality on actual hardware.

## 8. Main pretraining results

The paper-scale candidate and controls must share data, tokens, optimizer policy, and evaluation. Report
aggregate and per-domain validation trajectories, downstream tasks, instability events, throughput,
energy, and checkpoint selection. Clearly label any unmatched whole-system comparison.

## 9. Long-context results

- Context curriculum and genuine dependency-length distribution.
- Per-length curves on internal retrieval/composition and pinned RULER, NoLiMa, and HELMET.
- Selector recall, compression fidelity, KDA state behavior, lost-information examples, and
  position-binned loss.
- Allocated, trained, effective, and usable context as four different numbers.

Required figures: quality versus length, TTFT versus length, state versus length, and failure taxonomy.

## 10. Training and serving efficiency

- GPU-hours and energy to fixed quality.
- MFU/utilization and time breakdown by operator.
- Batch-1 and saturation TTFT/TPOT/throughput.
- Weight, expert, KV, recurrent, residual-summary, temporary, and fragmentation memory.
- Eager, compiled, reference, and production-runtime results.
- Sensitivity to prompt/output length, batch/load, precision, and hardware.

Do not transplant speedup numbers between hardware or load profiles.

## 11. Mechanistic analysis

Test why the architecture works:

- recurrence decay/timescale and overwrite distributions;
- dense-attention mass captured by compressed/sparse selections;
- compression reconstruction and retrieval recall;
- depth-source weights, entropy, age, and gradient flow;
- router load, margins, specialization, and outlier dynamics;
- correlations between diagnostics and held-out quality across seeds/scales;
- counterexamples where a seemingly favorable diagnostic predicts failure.

## 12. Post-training transfer

Only if claimed: apply identical SFT/RL/distillation recipes to candidate and control. Report general
capability retention, reasoning learning curves, policy mismatch, and inference-training numerical
agreement. Architecture conclusions must remain separable from post-training data improvements.

## 13. Related work

Organize by mechanism and explicitly identify what is inherited, reimplemented, modified, or novel.

## 14. Limitations, negative results, and broader impact

Include failed components, unresolved intervals, unsupported runtimes, scale boundaries, data gaps,
benchmark contamination risk, energy/resource usage, and the exact claims the experiments cannot make.

## 15. Reproducibility statement

List code revision, environment images, raw configs/results, seeds/data orders, dataset and benchmark
revisions, tokenizer/checkpoint hashes, kernel qualifications, hardware, commands, and artifact licenses.
