# Speck flagship plan v0

Status: draft, 2026-09-06. This document replaces the Paper 1 gate program as the operating plan.
It is intentionally one file. It states defaults, the few decisions still open, the compute split,
the calendar, and the pre-grant checklist.

## Decision record

- The six-pair Paper 1 finalist chain was stopped deliberately on 2026-09-06 after about 7.5 hours
  of dense control pair 0, attempt 2. No checkpoint or result was produced. The three-pair proxy
  (finding 105) stands as the 150M replication evidence.
- The novelty gate and the tri-axis working title are retired. Paper 1 is a recipe-plus-model
  paper: a small, efficient hybrid architecture chosen by matched ablations, trained into a strong
  open model, reported with DeepSeek-level depth on architecture, data, training systems, long
  context, evaluation, serving cost, negative results, and limitations.
- HCA, CSA, AttnRes, Stable LatentMoE, the interaction cube, and the scaling-law program stay in
  `research/paper-1/` as archived future work. They are not flagship inputs.
- HELMET and NoLiMa are dropped from Paper 1. Long-context evaluation is RULER v2 (eleven synthetic
  tasks, 4K to 128K) plus the internal structured-retrieval and composition protocols.
- Kept from the gate program: paired one-sided non-inferiority statistics (0.01-nat margin, 95%
  bound, three seeds) for every ablation that changes the flagship, and the findings ledger as the
  paper appendix.

## Goal

Ship one open model that is the best available at its size on the axis we can actually win:
quality per training FLOP and quality per byte of resident state at long context on consumer
hardware. State plainly that it is not trained on 10T+ tokens and will not top Qwen3-0.6B on MMLU.

## Compute

Grant: 5,000 GH200 GPU-hours on one 4-GPU node. Throughput assumption: 300 to 400 TFLOPS achieved
per GPU in bf16 for models near 1B. FP8 adds roughly 1.3× if it qualifies on the node.

| Run | Tokens | GPU-hours |
| --- | ---: | ---: |
| 150M, 3B tokens | 3B | ~5 |
| 350M, 7B tokens | 7B | ~15–20 |
| 750M, 15B tokens | 15B | ~50–60 |
| 1.2B, 24B tokens | 24B | ~120–140 |
| 1B flagship | 450B | ~2,300–2,600 |

| Phase | GPU-hours | What it produces |
| --- | ---: | --- |
| 1. Decisions at 350M | 900 | The four architecture decisions plus an LR/batch check |
| 2. Reversal check at 750M | 150 | Selected vs dense, one pair |
| 3. Flagship pretraining | 2,400 | The model |
| 4. Extension, anneal, SFT, evals, serving | 450 | 128K model, instruct variant, paper numbers |
| 5. Reserve | 500 | Failures and one surprise |
| 6. Paper ablations on spare capacity | 600 | Reader Attention, MQA/MLA, AttnRes as paper sections, run only during phase 4 |

The window must be at least two months. Three is comfortable. The flagship alone is about 25 days
of wall-clock on all four GPUs.

## Flagship spec v0 (defaults)

Everything below is the launch configuration unless a Phase 1 decision changes it. Geometry
numbers are targets; materialize them with the repository's parameter accounting.

- Architecture: 24 blocks, hidden 2048, 3:1 ratio, 18 KDA blocks and 6 global attention blocks at
  quantile positions. KDA with sigmoid output gate, FLA timescale initialization, conv kernel 4,
  head dim 128, 8 key heads, 16 value heads. Global attention GQA with 16 query heads, 4 KV heads,
  head dim 128, NoPE (`rope_dim: 0`). SwiGLU intermediate 5120. Untied embeddings. Approximately
  1.2B total parameters, about 1.05B non-embedding.
- Tokenizer: current Mistral 32K vocabulary. Pre-grant decision: keep it (default) or train a 64K
  Speck tokenizer with better code coverage. A tokenizer change must happen before any Phase 1 run.
- Training: 450B tokens at sequence length 4096, global batch about 1M tokens, Muon plus AdamW
  roles, weight decay 0.1, clip 1.0, WSD schedule with a 20% decay tail, bf16 with FP8 if
  qualified. Peak LR and batch confirmed by the Phase 1 sweep, not guessed.
- Curriculum: two phases. Stable phase on the web-heavy mixture. Decay phase upweights math, code,
  synthetic textbook, and instruction-style data, following SmolLM2/SmolLM3 and PuRo.
- Context: 4K base, then progressive extension to 32K and 128K on complete long documents with
  original-4K regression evaluation at every stage. NoPE global layers need no RoPE scaling.
- Post-training: constant-LR anneal from three seeds, weight-merged. Then SFT on SpeckChat2-class
  data. DPO only if time remains.

## Open decisions (Phase 1, days 1 to 14)

Each has a default. On day 15 the flagship launches with the winner or the default. No decision
may be added after day 1.

| Decision | Experiment at 350M | Runs | Default |
| --- | --- | ---: | --- |
| Dense vs MoE | Dense 350M vs fine-grained dropless MoE at 350M active, ~1.4B total (many small experts plus one shared expert, bias-based balancing) vs dense 750M. Matched GH200 wall-clock, routing stability over the full 7B tokens. | 3 arms × 3 seeds | Dense |
| NoPE vs partial RoPE in global layers | Same backbone, 7B tokens, loss plus 32K/128K retention. | 2 × 3 | NoPE |
| Global ratio 3:1 vs 5:1 | Same backbone. | 2 × 3 | 3:1 |
| LR and batch | Four-point peak-LR sweep at 1M-token batch on the default architecture. | 4 × 1 | Scaled from the 150M recipe |

Pass rule for changing a default: the alternative must beat it on loss at matched wall-clock with
the upper one-sided 95% bound over three seeds inside the 0.01-nat margin, and must not regress
32K/128K retention. Ties keep the default.

MoE rule: if MoE wins, the flagship becomes about 1B active and 4 to 6B total, the token budget
drops to about 350B to pay the routed overhead, and the serving story becomes laptop and edge,
not phone.

## Data

Target: 500B unique tokens tokenized, globally deduplicated, decontaminated against every
evaluation set, and packed before day 1. About 1 TB at uint16. This is the largest pre-grant task
and the largest schedule risk.

Stable-phase mixture (starting point, weights in percent):

| Source | Weight | Notes |
| --- | ---: | --- |
| Ultra-FineWeb HQ | 35 | Already in the pipeline |
| DCLM baseline | 25 | Already in the pipeline |
| FineWeb-Edu | 10 | New |
| Code (Stack-Edu or The Stack v2 educational subset) | 12 | New. The current pipeline has no code |
| Math (FineMath 4+, MegaMath) | 8 | Partly in the pipeline |
| Cosmopedia v2 and other synthetic textbook | 5 | Already in the pipeline |
| Wikipedia, peS2o | 5 | Already in the pipeline |

Decay-phase mixture: raise math, code, and synthetic to about 45% combined and add
instruction-style pretraining data. Long-document extension data: complete books, papers, and
repository trees only, no concatenated unrelated pages.

## Evaluation

- Short context: MMLU, HellaSwag, ARC, PIQA, WinoGrande, CommonsenseQA, TriviaQA, GSM8K, MATH,
  HumanEval, MBPP, through a pinned lm-evaluation-harness or lighteval revision.
- Long context: RULER v2 at 4K, 8K, 16K, 32K, 64K, 128K, plus the internal 200-case protocols.
- Comparators at matched size: SmolLM2-360M and 1.7B, Qwen3-0.6B and 1.7B, Gemma 3 1B, LFM2-700M
  and 1.2B, Llama 3.2 1B. Report their training token counts next to ours.
- Serving: TTFT, TPOT, throughput, resident state, and peak memory at 4K, 32K, and 128K on GH200,
  RTX 3090, and CPU via GGUF. Report time and energy separately, never a dollar figure without
  measured power.

## Paper skeleton

1. Introduction and claims, including non-claims.
2. Architecture: operators in one notation, state and FLOP accounting.
3. Ablations: the Phase 1 decisions at 350M, three seeds, paired bounds.
4. Scale check: 150M proxy, 350M, 750M, flagship.
5. Data and curriculum.
6. Training systems: arm64 Hopper stack, kernels, FP8, throughput, MFU, failure and resume record.
7. Long-context extension and evaluation.
8. Post-training.
9. Results against comparators, with token budgets shown.
10. Serving cost on three hardware classes.
11. Negative results: Reader Attention, attention gating, late NoPE switch, MoE on the 3090.
12. Limitations and reproducibility: configs, data manifests, checkpoints, the findings ledger.

## Calendar (three-month window)

- Before day 1: data at 500B tokens packed; arm64 Hopper stack tested on a rented GH200 including
  KDA kernels, FlexAttention, Liger, FP8, DDP, and checkpoint resume under a 24-hour job limit;
  routed MoE layer ported to grouped GEMM; flagship config materialized; analysis plan for the
  Phase 1 decisions frozen; tokenizer decision made.
- Days 1 to 14: Phase 1 and Phase 2, four single-GPU jobs in parallel.
- Day 15: decision meeting, launch the flagship the same day.
- Days 15 to 40: flagship. Nothing else runs on the node.
- Days 41 to 55: extension, anneal and merge, SFT, evals, serving, paper ablations on spare GPUs.
- Days 56 to 90: reserve, paper writing, release.

In a two-month window: Phase 1 shrinks to ten days, the reserve to one week, and paper ablations
are cut first.

## Pre-grant work on the 3090 and CPU

- Data pipeline scaled to 500B tokens: storage, download bandwidth, dedup memory, shard throughput.
- Export, GGUF, and serving benchmark path exercised on the current Speck2 checkpoint so the
  serving section has a working harness before the model exists.
- A one-day rented GH200 session for the stack test. This is the cheapest risk removal available.
- No new architecture experiments on the 3090. Anything worth knowing at 150M is a one-hour job on
  the node.

## Risks

1. Data not ready on day 1. Mitigation: start now, ship the stable-phase mixture first, and finish
   the decay-phase mixture during Phase 1.
2. arm64 kernel failures. Mitigation: the rented-node test; fall back to Torch KDA and bf16.
3. MoE instability if it wins. Mitigation: dense default, conventional MoE only, no LatentMoE.
4. 24-hour job limits and preemption. Mitigation: resume path tested pre-grant.
5. Root disk on the 3090 machine is at 98%. Mitigation: retire redundant local checkpoints (see
   the disk audit) before data preparation starts.
