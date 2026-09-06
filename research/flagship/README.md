# Speck flagship: scope of the model, the paper, and the experiments

Status: v3, 2026-09-06. This is the single operating document for the first flagship model and the
paper that describes it. Everything else under `research/` is either tooling contract
(`architecture-promotion-v1`) or archived evidence (`paper-1/*.json`).

## 1. Mission

Build small language models that are as efficient to train and serve as possible while giving up as
little quality as possible, and prove it with a model people use and a paper people trust.

The first flagship and its paper must do three things at once:

1. Ship a model that is genuinely good at its size and clearly best on the axis we choose.
2. Explain every design decision with a controlled experiment, so the paper is the documented
   decision process of the model rather than a report written afterwards.
3. Leave a ladder the next grant extends instead of restarts.

Data and architecture are treated as equal first-class subjects. The public evidence says data is
the larger lever at this scale: PuRo-2B ranks a within-source quality curriculum at 2.40x
cost-equivalent against 1.34x for FP8 and 1.19x for the optimizer, and SmolLM2's decisive sub-2B
lesson was about data staging. Our own measured architecture effects span 0.005 to 0.05 nats, while
published filter and mixture effects span 0.05 to 0.2 nats and several benchmark points.

Non-goals this cycle: a new operator, a novelty claim, mixture-of-experts, depth routing, sparse or
compressed attention, and any claim we cannot measure on hardware we own or rent. Conditional width
is deferred to a follow-on allocation through sparse upcycling from the released dense checkpoint.

## 2. The flagship model

### 2.1 Target

A dense recurrent/global hybrid trained on one 4x GH200 node, extended to 128K context, released as
base, pre-decay, and instruct checkpoints.

The axis we win on is quality per training FLOP and quality per byte of resident state at long
context, measured on a datacenter GPU, a consumer GPU, and a CPU. The paper states plainly that the
model sees under 1T tokens and will not top 36T-token models on short benchmarks.

### 2.2 Size and token budget

At the same 2,425 GPU-hours we can train either shape:

| Shape | Parameters | Tokens | Tokens per parameter |
| --- | ---: | ---: | ---: |
| A (default) | 1.2B | 400B | 333 |
| B | 600M | 800B | 1,333 |

Shape A has better absolute loss. Shape B is less undertrained relative to its size class, competes
directly with Qwen3-0.6B, SmolLM2-360M, and LFM2-700M, and serves better, which is our axis. This is
decision **F1**, resolved by the scale ladder in section 4.4 no later than day 18, defaulting to A.
If FP8 qualifies on the node, the extra throughput buys tokens, not saved hours.

### 2.3 Architecture defaults

Launch configuration unless an experiment in section 4 changes it. Geometry is a target; materialize
and verify parameter counts with the repository's accounting before freezing.

| Component | Default | Source |
| --- | --- | --- |
| Depth and width | 24 blocks, hidden 2048 | scaled from the 150M proxy |
| Mixer ratio | 3:1, 18 recurrent and 6 global blocks at quantile positions | findings 08, 16 to 18, 105 |
| Recurrent mixer | Kimi Delta Attention, sigmoid output gate, FLA timescale init, conv kernel 4, head dim 128, 8 key heads, 16 value heads | findings 13 to 18 |
| Global attention | GQA, 16 query heads, 4 KV heads, head dim 128, NoPE | findings 16 to 18 |
| Feed-forward | SwiGLU, intermediate 5120 | inherited |
| Embeddings | untied, Mistral 32K vocabulary | inherited, see D5 |
| Precision | bf16, FP8 if it qualifies on the node | PuRo-2B |
| Optimizer | Muon for matrices, AdamW elsewhere, weight decay 0.1, clip 1.0 | inherited |
| Schedule | WSD, 20% decay tail, global batch about 1M tokens | SmolLM2, PuRo |
| Sequence | 4K base, then 32K and 128K extension on complete long documents | findings 05, 06, 18 |

### 2.4 Recipe

- Stable phase on the E2 mixture, decay phase on the E4 mixture.
- Publish the last stable-phase checkpoint. It is the resumable seed for the next grant.
- Context extension in two stages, with original-4K regression evaluation at each stage.
- Anneal from three seeds and weight-merge, then supervised fine-tuning on SpeckChat2-class data.
  Preference tuning only if time remains.

## 3. The paper

### 3.1 Thesis

A small model can match dense-attention quality at a fraction of the training FLOPs and a small
fraction of the long-context state, and the data recipe that makes it good can be isolated with the
same rigor as the architecture. Every part of both claims is replicated and priced.

### 3.2 Standard

Coverage of the Kimi Linear, Kimi K3, and DeepSeek-V4 reports, plus the thing they omit: isolated
component evidence before the combined model, for data as well as architecture. Each line of the
flagship config maps to one figure with seeds, a paired bound, and a measured cost. A section that
changes no number in the config is cut. A config line with no figure is labeled inherited.

### 3.3 Sections and required evidence

1. Introduction: the efficiency problem, the two cost axes, claims and non-claims.
2. Architecture: every operator in one notation, with state and FLOP accounting.
3. Data: sourcing, filtering, deduplication, decontamination, and the mixture design.
4. Data ablations: E1 to E5 with per-domain held-out loss and small-scale benchmarks.
5. Architecture ablations: C0, D2, D3, D4, and D6 with paired non-inferiority bounds.
6. Scale: the selected dense architecture at four scales, a fitted curve with uncertainty, one held-out point,
   and the hyperparameter transfer rule from D6.
7. Training systems: arm64 Hopper stack, kernels, FP8, MFU, throughput, failures and resumes.
8. Long context: extension recipe, RULER v2 through 128K, internal protocols, 4K retention.
9. Post-training: anneal merge and SFT, with the delta each contributes.
10. Results against comparators at matched size, each comparator's token budget printed beside it.
11. Serving cost: TTFT, TPOT, throughput, resident state, peak memory at 4K, 32K, and 128K on
    GH200, RTX 3090, and CPU through GGUF. Time and energy separately, no dollar figures.
12. Mechanism: why a few global layers suffice, and what the recurrent state retains at 128K.
13. Negative results: Reader Attention, attention output gating, late NoPE conversion, and every arm
    that loses in section 4.
14. Limitations, one paragraph of future work, and reproducibility.

### 3.4 Headline

One sentence of the form "matches model X at 1/N the training compute and 1/M the resident state at
128K, and runs on a laptop." X, N, and M come from sections 10 and 11. No draft headline is stable
before those sections exist.

## 4. Experiments

### 4.1 Principles

**Screen at one seed, confirm at three.** The measured seed range at 150M is 0.00965 nats (finding
03). An effect several times that size is visible on one seed; quantifying it needs three. Screens
eliminate, confirmations decide.

**Architecture and data use different statistics.** Architecture ablations are non-inferiority
tests: does the cheaper option lose anything? They use the paired one-sided 95% bound and the
0.01-nat margin from `architecture-promotion-v1`. Data ablations are superiority tests: is the
mixture better? They report effect size, paired bounds, and power, not a margin.

**Never evaluate a mixture on its own data.** One neutral held-out set is defined before any run: an
equal-token slice held out from every candidate source, plus at least one source that appears in no
training mixture. Every arm reports aggregate loss and per-domain loss separately, since mixture
changes trade domains against each other and the aggregate hides it.

**Benchmarks that work at 350M.** HellaSwag, ARC-easy, PIQA, LAMBADA, WinoGrande. MMLU, GSM8K, and
HumanEval are at or near chance at this scale and cannot decide anything; code and math fractions are
judged on held-out domain loss at 350M and confirmed downstream at 750M and at the flagship.

**Known limitation.** Ablations run near 30 tokens per parameter while the flagship runs above 300,
and mixture effects interact with token budget more than architecture effects do. Mitigation: confirm
the winning mixture at 750M, and treat E4 as most transferable because its structure is identical to
the flagship's.

**Shared control.** One default configuration is trained once per seed and serves as the control for
D2 and D3.

### 4.2 Data experiments

Two stages, mirroring how the flagship trains. Stable-phase questions need full runs. Decay-phase
questions branch from one shared stable checkpoint and cost a quarter as much.

| ID | Question | Arms | Scale and tokens | Runs | GPU-h |
| --- | --- | --- | --- | ---: | ---: |
| E1 | Which web filter? | Ultra-FineWeb HQ, DCLM baseline, FineWeb-Edu, blend | 350M, 8B | 8 | 133 |
| E2 | Stable mixture composition | web-heavy, balanced, code and math heavy | 350M, 12B | 9 | 225 |
| E3 | Epoch policy at matched tokens | 1 epoch mixed, 2 epochs higher quality, 4 epochs best | 150M, 6B | 6 | 32 |
| E4 | Decay-phase composition | four mixtures branched from the E2 winner | 350M, +3B | 8 | 50 |
| E5 | Curriculum shape | single-stage uniform, within-source quality-sorted, two-phase | 350M, 12B | 4 | 100 |
| | | | | **35** | **540** |

E1 runs four one-seed screens, then three seeds on the top two. E2 uses three seeds because it sets
the flagship mixture. E3, E4, and E5 use two seeds because their expected effects are large.

E3 is the highest-leverage experiment in the program. If repetition is close to free, as
Muennighoff et al. report up to four epochs, the preparation target drops from 500B unique tokens to
about 150B and the critical path in section 10 shrinks with it. Run it first.

E5 settles a real disagreement in the literature: SmolLM2 found single-stage uniformly high-quality
data better below 2B, while PuRo-2B found within-source quality sorting to be its single largest win.

### 4.3 Architecture decisions

All at 350M and 10B tokens unless noted. Each has a default. Day 21 launches the flagship with the
winner or the default. No decision may be added after day 1.

**Pass rule.** The alternative must beat the default on the neutral held-out set at matched
wall-clock, with the upper one-sided 95% bound over three seeds inside the 0.01-nat margin, and must
not regress 32K or 128K retention on the built-in curve. Ties keep the default.

| ID | Question | Arms | Runs | Default | GPU-h |
| --- | --- | --- | ---: | --- | ---: |
| C0 | Shared control | KDA 3:1 NoPE dense | 3 | | 63 |
| D2 | Global-layer position encoding | partial RoPE on 32 of 128 dims | 3 | NoPE | 63 |
| D3 | Recurrent to global ratio | 5:1 (20 recurrent, 4 global) | 3 | 3:1 | 63 |
| D4 | Peak LR and batch | four LR points at 1M-token batch, 4B tokens each | 4 | scaled from 150M | 33 |
| D6 | Hyperparameter transfer rule | two extra 150M points to fit LR against width | 2 | empirical fit | 5 |
| | | | **15** | | **227** |

**D1 is retired for this allocation.** The first flagship is dense in width. This removes a late
architecture branch plus its unqualified expert optimizer, Hopper routing kernel, distributed-state,
export, and serving dependencies. The identifier is intentionally not reused. Its 132 GPU-hours move
to reserve until measured GH200 throughput and the first complete flagship checkpoint are secure.

D5, the tokenizer, is decided before the grant on the 3090. Default is to keep the Mistral 32K
vocabulary. A 64K Speck vocabulary requires a trainer that does not exist yet, re-preparation of the
corpus, and loss of comparability with every existing checkpoint.

D6 is what makes the next scale cheap and is a paper figure in its own right.

### 4.4 Scale ladder and reversal check

The selected architecture against a dense control at four scales, plus cheap low anchors for the
fit. Resolves F1 in section 2.2 and provides the scaling section.

| Points | Scale and tokens | Runs | GPU-h |
| --- | --- | ---: | ---: |
| Low anchors | 60M/1.2B, 220M/5B | 4 | 14 |
| Mid | 150M/3B, 350M/7B | 4 | 35 |
| Reversal check S1 | 750M/15B, both arms | 2 | 134 |
| Contingency and reruns | | 4 | 107 |
| | | **14** | **290** |

A scaling-efficiency claim additionally requires uncertainty on the fit, residual diagnostics, and
the flagship itself as a held-out confirmation point.

### 4.5 Not gating the flagship

Run only on spare capacity during the extension and evaluation weeks, and only as paper sections:
Reader Attention at 350M, MQA against GQA cache representation, and a 150M repeat of D2 and D3 for
the scale-consistency figure.

### 4.6 Deferred conditional width

No MoE run belongs to this grant or gates the flagship. The repository retains a conventional
single-device routed-SwiGLU reference, routing diagnostics, and expert-masking support so a follow-on
program does not restart from zero. It deliberately retains no active MoE experiment, result-selection
contract, expert-parallel claim, or release path.

The next program begins from the released dense pre-decay checkpoint: sparse-upcycle selected MLPs,
compare against continued dense training at matched wall-clock, and qualify expert-parallel training,
checkpointing, export, and serving before scaling. That work requires its own 10K–50K+ allocation and
paper rather than sharing the first flagship's critical path.

## 5. Data

### 5.1 Targets

Two corpora are needed.

- **Pretraining corpus.** 500B unique tokens tokenized, globally deduplicated, decontaminated
  against every evaluation set, and packed before day 1. About 1.0 TB packed at 2 bytes per token,
  plus 2 to 3 TB of raw parquet during preparation. E3 may cut the unique-token target to about
  150B, so run E3 before committing to the full download.
- **Long-document extension corpus.** Complete books, papers, and repository trees, with source
  token-length filters. Concatenated unrelated documents are a stress condition, not supervision.

Set `speck_base_dir=/mnt/speck-data/speck` so packed shards, the raw download cache, and checkpoints
land on the 5.2 TB volume rather than the 12 GB free on root. Prune raw files as each source
completes; the volume does not hold raw and packed simultaneously at full scale.

### 5.2 Mixture

Starting point for E2's balanced arm, weights in percent. E1 decides the web component and E2
decides these fractions; this table is a prior, not a decision.

| Source | Weight | Status |
| --- | ---: | --- |
| Web, filter chosen by E1 | 55 | Ultra-FineWeb HQ and DCLM in the pipeline, FineWeb-Edu new |
| Code, Stack-Edu or an educational subset of The Stack v2 | 12 | **new, no code source exists today** |
| Math, FineMath 4+ and MegaMath | 8 | partly in the pipeline |
| Synthetic textbook, Cosmopedia v2 and similar | 10 | in the pipeline |
| Reference, Wikipedia and peS2o | 5 | in the pipeline |
| Held in reserve for E2's arms | 10 | |

The absence of any code source is the largest single gap in the current pipeline. Code data improves
reasoning as well as code, so it is not an optional category.

The decay mixture raises math, code, synthetic, and instruction-style data to roughly 45% combined.
E4 chooses among four candidates.

## 6. Evaluation

- **Short context:** MMLU, HellaSwag, ARC, PIQA, WinoGrande, CommonsenseQA, TriviaQA, GSM8K, MATH,
  HumanEval, MBPP, through a pinned lm-evaluation-harness or lighteval revision.
- **Ablation evaluation:** the neutral held-out set from section 4.1, per-domain loss, and the
  small-scale benchmark subset.
- **Long context:** RULER v2 at 4K through 128K, the internal 200-case structured-retrieval and
  symbolic-composition protocols, and original-4K loss after every extension stage.
- **Comparators:** SmolLM2-360M and 1.7B, Qwen3-0.6B and 1.7B, Gemma 3 1B, LFM2-700M and 1.2B,
  Llama 3.2 1B, each with its training-token budget printed alongside.
- **Serving:** TTFT, TPOT, tokens per second, resident state, and peak memory at 4K, 32K, and 128K
  on GH200, RTX 3090, and CPU through GGUF.
- **Out of scope:** HELMET and NoLiMa. Audits are in findings 42 to 54 and 124 to 128.

## 7. Compute and schedule

Grant: 5,000 GH200 GPU-hours on one 4-GPU node. Estimates assume 6ND training FLOPs at 350 achieved
TFLOPS per GPU with a 1.25x overhead factor for experiments and 1.06x for the flagship. The window
must be at least two months; three is comfortable.

| Track | GPU-hours | Share |
| --- | ---: | ---: |
| Data experiments, E1 to E5 | 540 | 11% |
| Dense architecture decisions, C0 and D2 to D6 | 227 | 5% |
| Scale ladder and reversal | 290 | 6% |
| Flagship pretraining | 2,425 | 49% |
| Extension, anneal, SFT, evaluation, serving | 450 | 9% |
| Reserve | 1,068 | 21% |
| | **5,000** | |

The reserve is sized for a first run on unfamiliar hardware with an untested stack, not as slack to
fill with extra arms.

### Schedule and dependencies

| Days | Work | Depends on |
| --- | --- | --- |
| 1 to 3 | D4 and D6 first, since every later run needs the right LR. E3 and the E1 screens in parallel. | |
| 4 to 10 | E1 confirmations, E2, C0, D2, D3 | D4 |
| 11 to 16 | E5, E4, S1 reversal check | E2 winner for E4 |
| 17 to 20 | Scale ladder, analysis, F1 size decision, config freeze | S1 |
| 21 | Flagship launch | all decisions or their defaults |
| 21 to 47 | Flagship, all four GPUs, nothing else on the node | |
| 48 to 60 | Extension, anneal and merge, SFT, evaluation, serving, spare-capacity paper ablations | |

Day 21 is a launch date, not a readiness gate. If the window is two months there is no slack; cut in
this order, decided now rather than under pressure: E5, then the scale ladder to two points, then the
flagship token budget to 320B.

## 8. Releases

- Base, pre-decay, 128K-extended, and instruct checkpoints in native, Transformers, and GGUF form.
- Every scaling-ladder checkpoint at every scale, both arms.
- All experiment configs, data manifests with source revisions, and packed-data hashes.
- The findings ledger and the raw result JSON.
- A serving benchmark script others can run on their own hardware.

## 9. After this grant

- The pre-decay checkpoint is designed to be continued with more tokens under the same schedule.
- The ladder extends upward with the same pipeline, evaluation, and statistics, one scale at a time.
- D6's transfer rule is what keeps the next scale from needing its own sweep.
- Depth routing and cache compression remain possible later sequence/depth work. MoE is a separate
  sparse-upcycling paper and compute proposal rooted in the released dense pre-decay checkpoint.

## 10. Before day 1

On the 3090 and CPU, in priority order. Items 1 and 2 are the critical path.

1. **Storage.** Set `speck_base_dir` to the data volume, delete the out-of-scope 11 GB HELMET
   archive, and bring root below 80%.
2. **Data.** Add a code source. Run a 20B-token rehearsal to measure download bandwidth, dedup
   memory, and shard throughput before committing to the full target. Prepare the stable-phase
   corpus first, the decay candidates during the decision phase, and the long-document corpus in
   parallel. Build the neutral held-out set and the decontamination pass at the same time.
3. **One rented Hopper day.** Scripted checklist: arm64 PyTorch and Triton, FLA KDA kernels,
   FlexAttention, Liger, FP8, four-GPU DDP, and checkpoint resume under a simulated 24-hour job
   limit. Four-GPU training has never run in this repository. Rent GH200 rather than a consumer card:
   the arm64 Grace host is part of the deployment target, and a 5090 does not answer that question.
4. **D5, tokenizer.** Decide within a week.
5. **Comparator table.** Run every comparator in section 6 through the pinned
   harness and measure its serving cost on the 3090 and on CPU. This builds paper sections 10 and 11
   before the flagship exists and establishes the bar.
6. **Configs and plan.** Materialize the 60M, 150M, 220M, 350M, 750M, and 1.2B geometries, verify
   parameter counts, and freeze the section 4 analysis plan as one file.

The 3090 runs only item 5. No new architecture experiment belongs there: anything worth knowing at
150M is an hour on the node.

## 11. Risks

| Risk | Mitigation |
| --- | --- |
| Data not ready on day 1 | Start now; run E3 early to potentially cut the target by 3x; prepare the stable corpus first |
| Volume cannot hold raw and packed together | Prune raw per source as it completes |
| arm64 kernel or four-GPU DDP failure | The rented-node day; fall back to Torch recurrence and bf16 |
| Job time limits and preemption | Resume path tested before the grant |
| Scope expands during the allocation | Dense-width architecture is frozen; 132 retired D1 hours stay in reserve until the flagship is secure |
| Ablations do not transfer to 300+ tokens per parameter | Confirm at 750M; weight E4 highest; state the limitation |
| Two-month window | Pre-decided cut order in section 7 |
