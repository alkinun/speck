# Speck flagship: scope of the model, the paper, and the experiments

Status: v1, 2026-09-06. This is the single operating document for the first flagship model and the
paper that describes it. Everything else under `research/` is either tooling contract
(`architecture-promotion-v1`) or archived evidence (`paper-1/*.json`).

## 1. Mission

Build small language models that are as efficient to train and serve as possible while giving up
as little quality as possible, and prove it with a model people use and a paper people trust.

The first flagship and its paper have to do three things at once:

1. Ship a model that is genuinely good at its size and clearly the best on the axis we choose.
2. Explain every design decision with a controlled experiment, so the paper is the documented
   decision process of the model rather than a report written after the fact.
3. Leave behind a ladder that the next grant extends instead of restarts.

Non-goals for this cycle: a new operator, a novelty claim, mixture-of-experts unless it wins its
one experiment, depth routing, sparse or compressed attention, and any claim we cannot measure on
hardware we own or rent.

## 2. The flagship model

### 2.1 Target

A dense recurrent/global hybrid of about 1.2B total parameters, trained on about 450B tokens on one
4x GH200 node, extended to 128K context, released as base, pre-decay, and instruct checkpoints.

The axis we win on: quality per training FLOP and quality per byte of resident state at long
context, measured on a datacenter GPU, a consumer GPU, and a CPU. We state in the paper that the
model is not trained on 10T+ tokens and will not top 36T-token models on short benchmarks.

### 2.2 Architecture defaults

These are the launch configuration unless an experiment in section 4 changes them. Geometry numbers
are targets; materialize them with the repository's parameter accounting before freezing.

| Component | Default | Source of the default |
| --- | --- | --- |
| Depth and width | 24 blocks, hidden 2048 | scaled from the 150M proxy |
| Mixer ratio | 3:1, 18 KDA blocks and 6 global attention blocks at quantile positions | findings 08, 16 to 18, 105 |
| Recurrent mixer | Kimi Delta Attention, sigmoid output gate, FLA timescale init, conv kernel 4, head dim 128, 8 key heads, 16 value heads | findings 13 to 18 |
| Global attention | GQA, 16 query heads, 4 KV heads, head dim 128, NoPE | findings 16 to 18 |
| Feed-forward | SwiGLU, intermediate 5120 | inherited |
| Embeddings | untied, Mistral 32K vocabulary | inherited; see decision D5 |
| Precision | bf16, FP8 if it qualifies on the node | PuRo-2B evidence |
| Optimizer | Muon for matrices, AdamW for the rest, weight decay 0.1, clip 1.0 | inherited |
| Schedule | WSD, 20% decay tail, global batch about 1M tokens | SmolLM2, PuRo |
| Sequence | 4K base training, then 32K and 128K extension on complete long documents | findings 05, 06, 18 |

### 2.3 Training recipe

- Stable phase on the web-heavy mixture, then a decay phase that upweights math, code, synthetic
  textbook, and instruction-style data.
- Publish the last stable-phase checkpoint. It is the resumable seed for the next grant.
- Context extension in two stages with original-4K regression evaluation at each stage.
- Anneal from three seeds and weight-merge, then supervised fine-tuning on SpeckChat2-class data.
  Preference tuning only if time remains.

## 3. The paper

### 3.1 Thesis

A small model can match dense-attention quality at a fraction of the training FLOPs and a small
fraction of the long-context state by combining a fixed-state recurrent mixer with a few global
attention layers, and every part of that claim can be isolated, replicated, and priced.

### 3.2 Standard

Coverage of Kimi Linear, Kimi K3, and DeepSeek-V4 reports, plus one thing they omit: isolated
component evidence before the combined model. Concretely, each architectural line in the flagship
config maps to one figure with three seeds, a paired one-sided 95% bound against a control, and a
measured cost. If a section changes no number in the config, it is cut. If a config line has no
figure, the paper says it was inherited.

Statistics follow `architecture-promotion-v1`: paired candidate-minus-control differences, a 0.01
nat non-inferiority margin, and per-source guardrails. The measured seed range at 150M (0.00965
nats) is the reason three seeds are mandatory.

### 3.3 Sections and the evidence each needs

1. Introduction: the efficiency problem, the two cost axes, claims, and non-claims.
2. Architecture: every operator in one notation with state and FLOP accounting.
3. Sequence-mixer ablations at 350M: the decisions in section 4, three seeds each.
4. Data ablations at 350M: mixture, code and math fraction, decay-phase composition.
5. Scale: the selected architecture against dense at 150M, 350M, 750M, and 1.2B, a fitted curve
   with uncertainty, and one held-out point.
6. Training systems: arm64 Hopper stack, kernels, FP8, MFU, throughput, failures and resumes.
7. Long context: extension recipe, RULER v2 through 128K, internal retrieval and composition
   protocols, original-4K retention.
8. Post-training: anneal merge and SFT, with the delta they contribute.
9. Results against comparators at matched size, with each comparator's training-token budget.
10. Serving cost: TTFT, TPOT, throughput, resident state, and peak memory at 4K, 32K, and 128K on
    GH200, RTX 3090, and CPU via GGUF. Time and energy reported separately, no dollar figures.
11. Mechanism: why a few global layers suffice (middle versus final role), and what the KDA state
    retains at 128K. One diagnostic figure each.
12. Negative results: Reader Attention, attention output gating, late NoPE conversion, MoE on the
    3090, and whatever loses in section 4.
13. Limitations, future work in one paragraph (depth routing, cache compression, MoE at scale),
    and reproducibility: configs, data manifests, every ladder checkpoint, the findings ledger.

### 3.4 Headline

One sentence of the form "matches model X at 1/N the training compute and 1/M the resident state at
128K, and runs on a laptop." The exact X, N, and M come from section 9 and 10. No draft headline is
stable before those sections exist.

## 4. Experiments

All decision experiments run at 350M parameters and 7B tokens (20 tokens per parameter) on one GPU
each, four in parallel, in the first fourteen days of the allocation. Each has a default. On day 15
the flagship launches with the winner or the default. No decision may be added after day 1.

Pass rule for replacing a default: the alternative must beat it on validation loss at matched
wall-clock with the upper one-sided 95% bound over three seeds inside the 0.01 nat margin, and must
not regress 32K or 128K retention on the built-in curve. Ties keep the default.

| ID | Question | Arms | Runs | Default | Cost (GPU-h) |
| --- | --- | --- | ---: | --- | ---: |
| D1 | Dense or MoE? | dense 350M; fine-grained dropless MoE at 350M active and about 1.4B total with a shared expert and bias-based balancing; dense 750M as the capacity reference | 9 | dense | 200 |
| D2 | NoPE or partial RoPE in the global layers? | NoPE; RoPE on 32 of 128 dims | 6 | NoPE | 120 |
| D3 | 3:1 or 5:1 recurrent to global? | 18+6; 20+4 | 6 | 3:1 | 120 |
| D4 | Peak LR and batch at 1M tokens | four LR points on the default | 4 | scaled from 150M | 80 |
| D5 | Tokenizer | Mistral 32K; a 64K Speck tokenizer with code coverage | pre-grant, on the 3090 at 150M | keep 32K | 0 on the node |
| A1 | Data mixture | three stable-phase mixtures differing in code and math fraction | 9 | section 5 mixture | 180 |
| A2 | Decay-phase composition | two decay mixtures on the same stable checkpoint | 6 | section 5 decay mixture | 60 |
| S1 | Reversal check | selected architecture vs dense at 750M, one pair | 2 | proceed | 120 |

D1 rule: if MoE wins, the flagship becomes about 1B active and 4 to 6B total, the token budget drops
to about 350B to pay the routed overhead, and the serving target becomes laptop and edge rather
than phone. Conventional fine-grained MoE only; no untested operator enters the flagship.

Paper ablations that do not gate the flagship run only on spare GPUs during the extension and
evaluation weeks: Reader Attention at 350M, MQA versus GQA cache representation, and a
scaling-ladder repeat of D2 and D3 at 150M.

## 5. Data

Target: 500B unique tokens tokenized, globally deduplicated, decontaminated against every
evaluation set, and packed before day 1. About 1 TB at uint16. This is the largest pre-grant task
and the largest schedule risk. The current pipeline has prepared 20B tokens and has no code source.

Stable-phase mixture, starting point, weights in percent:

| Source | Weight | Status |
| --- | ---: | --- |
| Ultra-FineWeb HQ | 35 | in the pipeline |
| DCLM baseline | 25 | in the pipeline |
| FineWeb-Edu | 10 | new |
| Code: Stack-Edu or an educational subset of The Stack v2 | 12 | new |
| Math: FineMath 4+, MegaMath | 8 | partly in the pipeline |
| Cosmopedia v2 and similar synthetic textbook | 5 | in the pipeline |
| Wikipedia, peS2o | 5 | in the pipeline |

Decay phase: raise math, code, and synthetic to about 45% combined and add instruction-style
pretraining data. Extension data: complete books, papers, and repository trees only.

## 6. Evaluation

- Short context: MMLU, HellaSwag, ARC, PIQA, WinoGrande, CommonsenseQA, TriviaQA, GSM8K, MATH,
  HumanEval, MBPP, through a pinned lm-evaluation-harness or lighteval revision.
- Long context: RULER v2 at 4K, 8K, 16K, 32K, 64K, and 128K; the internal 200-case structured
  retrieval and symbolic composition protocols; original-4K loss after every extension stage.
- Comparators at matched size: SmolLM2-360M and 1.7B, Qwen3-0.6B and 1.7B, Gemma 3 1B, LFM2-700M
  and 1.2B, Llama 3.2 1B. Their training token counts are printed next to ours.
- Serving: TTFT, TPOT, tokens per second, resident state, and peak memory at 4K, 32K, and 128K on
  GH200, RTX 3090, and CPU via GGUF.
- Out of scope: HELMET and NoLiMa. Their audits are in findings 42 to 54 and 124 to 128.

## 7. Compute and calendar

Grant: 5,000 GH200 GPU-hours on one 4-GPU node. Assumed 300 to 400 achieved TFLOPS per GPU in bf16
near 1B parameters. The window must be at least two months; three is comfortable.

| Phase | GPU-hours | Days |
| --- | ---: | --- |
| Decisions and data ablations (section 4) | 900 | 1 to 14 |
| Reversal check at 750M | 150 | 12 to 14 |
| Flagship pretraining | 2,400 | 15 to 40 |
| Extension, anneal, SFT, evaluation, serving | 450 | 41 to 55 |
| Paper ablations on spare capacity | 600 | 41 to 55 |
| Reserve | 500 | as needed |

Day 15 is a launch date, not a readiness gate. In a two-month window, section 4 shrinks to ten
days and paper ablations are cut first.

## 8. Releases

- Base, pre-decay, 128K-extended, and instruct checkpoints, in native, Transformers, and GGUF form.
- Every scaling-ladder checkpoint at 150M, 350M, 750M, and 1.2B, both arms.
- All experiment configs, data manifests with source revisions, and the packed-data hashes.
- The findings ledger and the raw result JSON.
- A serving benchmark script others can run on their own hardware.

## 9. After this grant

- The pre-decay checkpoint is designed to be continued with more tokens under the same schedule.
- The ladder is designed to be extended upward: the same data pipeline, the same evaluation, the
  same statistics, one more scale.
- The hyperparameter transfer rule fixed in D4 is what makes the next scale cheap.
- Depth routing, cache compression, and MoE at scale are the second paper, gated on the same
  standard as this one.

## 10. Before day 1 (on the 3090 and CPU)

1. Data at 500B tokens packed; stable-phase mixture first, decay mixture during section 4.
2. A one-day rented GH200 session: PyTorch, Triton, FLA KDA kernels, FlexAttention, Liger, FP8,
   DDP, and checkpoint resume under a 24-hour job limit, all on arm64.
3. Routed MoE layer ported to grouped GEMM and tested on that session.
4. Tokenizer decision D5.
5. Flagship config materialized and the section 4 analysis plan frozen.
6. Export, GGUF, and serving benchmark path exercised on the Speck2 checkpoint.
7. Root disk on the 3090 machine cleared to below 80%.

No new architecture experiments on the 3090. Anything worth knowing at 150M is a one-hour job on
the node.
