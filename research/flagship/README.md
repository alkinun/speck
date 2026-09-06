# Speck flagship: scope of the model, the paper, and the experiments

Status: v3, 2026-09-06. This is the single operating document for the first flagship model and the
paper that describes it. Everything else under `research/` is either tooling contract
(`architecture-promotion-v1`) or archived evidence (`paper-1/*.json`).

The active operating surface is intentionally small:

- This file freezes scope, defaults, experiments, data, evaluation, and releases.
- [`DATA.md`](DATA.md) defines source qualification, the held-out firewall, the mixture-search
  funnel, promotion rules, and required evidence.
- [`EXECUTION.md`](EXECUTION.md) gives the dependency-based 90-day operating order.
- [`PREGRANT.md`](PREGRANT.md) is the readiness gate before allocated compute starts.
- [`plan.json`](plan.json) is the machine-checked GPU-hour, dependency, reserve, and fallback contract.
- [`data_plan.json`](data_plan.json) is the machine-checked category, run, and selection contract.
- [`targets/`](targets/) contains non-launchable geometry targets; complete launch experiments are
  created only at the day-21 freeze.

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

The current non-launchable Shape-A geometry is
[`targets/shape-a`](targets/shape-a/). It materializes to 1,195,878,432 parameters; the rounded 1.2B
label is the model-size class, not an exact count.

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
4. Data ablations: E1W/E1S and E2–E5 with per-domain held-out loss and small-scale benchmarks.
5. Architecture ablations: C0, D2, D3, D4, and D6 with paired non-inferiority bounds.
6. Scale: the selected dense architecture at four scales, a fitted curve with uncertainty, one
   held-out point, and the hyperparameter transfer rule from D6.
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

The complete protocol is [`DATA.md`](DATA.md). Source qualification is CPU work before the grant.
GPU experiments form a funnel: source/filter screens, cheap space-filling coverage of the constrained
six-category simplex, medium-scale narrowing, and three-seed full-proxy confirmation. Stable-phase
questions need full runs. Decay-phase questions branch from one shared stable checkpoint and cost a
quarter as much.

| ID | Question | Arms | Scale and tokens | Runs | GPU-h |
| --- | --- | --- | --- | ---: | ---: |
| E1W | Which web source/filter? | four treatments, screen then confirm top two | 350M, 8B | 8 | 133 |
| E1S | Which specialist sources? | three treatments each for code, math, and synthetic | 150M, 2B | 15 | 27 |
| E2a | Where is the stable-mixture frontier? | 24 space-filling constrained mixtures | 60M, 1.2B | 24 | 10 |
| E2b | Which proxy winners survive scale? | six diverse Pareto candidates | 150M, 3B | 6 | 16 |
| E2c | Which stable mixture wins? | top three, three seeds each | 350M, 12B | 9 | 225 |
| E3 | Epoch policy at matched tokens | 1 epoch mixed, 2 epochs higher quality, 4 epochs best | 150M, 6B | 6 | 32 |
| E4 | Decay-phase composition | four mixtures branched from the E2 winner | 350M, +3B | 8 | 50 |
| E5 | Curriculum shape | quality-sorted and two-phase; reuse E2c uniform control | 350M, 12B | 4 new | 100 |
| | | | | **80** | **593** |

E1W runs four one-seed screens, then adds two seeds to each finalist. Each E1S category runs three
one-seed treatments, then adds one seed to its top two. E2c uses three seeds because it sets the
flagship mixture. E3, E4, and E5 use two seeds because their expected effects are large. E5's count
is four new runs because the selected E2c mixture is its two-seed uniform control.

The data decision is not a single weighted loss. Selection uses equal-domain bits per UTF-8 byte,
paired confidence intervals, and a hard no-regression guardrail for web, code, math, synthetic,
science, and reference separately. A sealed audit is opened once after selection. Exact rules,
bounds, fallbacks, and artifacts are frozen in [`data_plan.json`](data_plan.json).

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

### 4.5 Outside the active experiment matrix

Reader Attention, MQA/MLA cache alternatives, attention residuals, sparse/compressed attention, and
other archived axes receive no grant-1 runs. Existing completed evidence may appear as background or
negative results, but spare capacity cannot reactivate them.

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

Natural language is English-only. Programming syntax is exempt from language identification, while
code comments, documentation, and notebooks are English-filtered. Starting stable-phase weights and
search bounds are below; these are priors and constraints, not the result.

| Category | Prior | E2 range | Status |
| --- | ---: | ---: | --- |
| Web | 55% | 45–65% | incumbents integrated; E1W chooses treatment |
| Code | 15% | 10–22% | **new; rights, provenance, repository split, and source must qualify** |
| Math | 10% | 8–18% | partly integrated; E1S chooses treatment |
| Synthetic | 10% | 8–18% | integrated; E1S chooses treatment |
| Science | 5% | 3–10% | peS2o incumbent, requalified before use |
| Reference | 5% | 3–10% | English Wikipedia incumbent, requalified before use |

The absence of any code source is the largest single gap in the current pipeline. Code data improves
reasoning as well as code, so it is not an optional category.

E2a generates valid mixtures that sum to 100% inside these bounds rather than relying on three
hand-written blends. E2b and E2c narrow and confirm. The decay prior raises math, code, synthetic,
and instruction-style data to roughly 45% combined; E4 chooses among four candidates. Source gates,
held-out construction, near-duplicate and contamination requirements, selection statistics, and
promotion rules are in [`DATA.md`](DATA.md).

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
| Data experiments, E1W/E1S and E2 to E5 | 593 | 12% |
| Dense architecture decisions, C0 and D2 to D6 | 227 | 5% |
| Scale ladder and reversal | 290 | 6% |
| Flagship pretraining | 2,425 | 49% |
| Extension, anneal, SFT, evaluation, serving | 450 | 9% |
| Reserve | 1,015 | 20% |
| | **5,000** | |

The reserve is sized for a first run on unfamiliar hardware with an untested stack, not as slack to
fill with extra arms. [`plan.json`](plan.json) is authoritative for phase budgets, dependencies,
throughput responses, and cuts; [`EXECUTION.md`](EXECUTION.md) is its readable operating view.

Calendar ranges are advisory. Exit gates and the day-21 configuration freeze are binding. Independent
arms may move within a phase or across GPUs, but work may not cross an unmet dependency or introduce a
new architecture axis.

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

[`PREGRANT.md`](PREGRANT.md) is the complete readiness checklist and records current status. The
critical path is storage headroom, a code-inclusive corpus and neutral held-out set, the 20B data
rehearsal, complete long-document data, and one four-GH200 training/resume qualification.

The RTX 3090 is reserved for comparator serving measurements and representative export rehearsals.
It does not run new architecture searches. A paid allocation does not start while a launch-critical
pre-grant item lacks either a passing artifact or an explicit non-GPU fallback.

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
