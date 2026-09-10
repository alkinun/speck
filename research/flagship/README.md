# Speck flagship: scope of the model, the paper, and the experiments

Status: v6, 2026-09-10. This is the single operating document for the first flagship model and the
paper that describes it. [`../DIRECTION.md`](../DIRECTION.md) holds the multi-grant lab direction;
the promotion protocol, notebook, and `paper-1` archive retain their separate evidence roles.

The active operating surface is intentionally small:

- This file freezes scope, defaults, experiments, data, evaluation, and releases.
- [`PAPER.md`](PAPER.md) freezes the central question, claim ladder, paper spine, required figures,
  headline gate, and scope discipline.
- [`DATA.md`](DATA.md) defines source qualification, the held-out firewall, the mixture-search
  funnel, promotion rules, and required evidence.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) defines causal ablations, shared-control identity, scale
  transfer, systems measurement, and the paper evidence map.
- [`TOKENIZER.md`](TOKENIZER.md) defines balanced sampling, deterministic candidate training,
  static qualification, the matched LM pilot, and migration.
- [`SOURCES.md`](SOURCES.md) records the broad dataset survey, source shortlist, Stack v3 decision,
  and qualification order.
- [`EXECUTION.md`](EXECUTION.md) gives the dependency-based 90-day operating order.
- [`PREGRANT.md`](PREGRANT.md) is the readiness gate before allocated compute starts.
- [`data_calibration_2b_v1/`](data_calibration_2b_v1/) is the active time-bounded six-category
  production calibration. The paused [`data_rehearsal_20b_v1/`](data_rehearsal_20b_v1/) remains a
  resumable fallback; neither has data-selection or training authority.
- [`plan_v2.json`](plan_v2.json) is the active machine-checked GPU-hour, dependency, reserve, and
  fallback contract; `plan.json` is its immutable predecessor.
- [`data_plan_v2.json`](data_plan_v2.json) is the active machine-checked category, run, and selection
  contract; `data_plan.json` preserves the pre-integration predecessor.
- [`architecture_plan_v2.json`](architecture_plan_v2.json) is the active machine-checked architecture,
  scale, mature-horizon, and systems contract; `architecture_plan.json` is its predecessor.
- [`integration_plan_v2.json`](integration_plan_v2.json) is the active 122-GPU-hour crossed
  data-architecture, assembled-recipe, and mechanism contract; `integration_plan.json` is its
  predecessor.
- [`tokenizer_plan_v4.json`](tokenizer_plan_v4.json) preserves the pre-approval tokenizer scope.
- [`tokenizer_plan_v7.json`](tokenizer_plan_v7.json) is the active tokenizer contract; formal sample,
  models, static evaluation, and endpoint nomination pass while the LM pilot and D5 audit remain.
- [`release_and_data_use_policy_v1.json`](release_and_data_use_policy_v1.json) freezes MIT code,
  Apache-2.0 model weights, metadata-only corpus disclosure, attribution, and removal controls.
- [`source_rights_acceptance_v1.json`](source_rights_acceptance_v1.json) records the project owner's
  guarded-use approval of all 30 selected sources without granting production or training authority.
- [`firewall_plan_v4.json`](firewall_plan_v4.json) binds the completed real tokenizer, selection, and
  sealed-audit partitions; the audits remain unopened and model training remains forbidden.
- [`precision_plan_v1.json`](precision_plan_v1.json) freezes bf16 for grant 1 and defers optional FP8
  unless a reviewed GH200-qualified successor passes without consuming protected reserve.
- [`pregrant_blockers_20260910.json`](pregrant_blockers_20260910.json) is the current fail-closed map of
  what remains after the completed operations, firewall, tokenizer-static, and backup work.
- [`embedding_head_contract_v1.json`](embedding_head_contract_v1.json) freezes one physically shared
  token embedding/LM-head parameter and its config, accounting, checkpoint, optimizer, and export rules.
- [`source_registry_v2.json`](source_registry_v2.json) pins the human-approved source revisions and
  tokenizer byte quotas without granting production or training authority.
- [`targets/`](targets/) contains seven exact, machine-checked, non-launchable scale geometries;
  complete launch experiments are created only at the day-21 freeze.

## 1. Mission

Build small language models that maximize intelligence per unit of training compute, serving compute,
and resident memory, and prove the allocation choices with a model people use and a paper people
trust. The multi-grant direction is recorded in [`../DIRECTION.md`](../DIRECTION.md).

The first flagship and its paper must do three things at once:

1. Ship a model that is genuinely good at its size and clearly best on the axis we choose.
2. Explain every design decision with a controlled experiment, so the paper is the documented
   decision process of the model rather than a report written afterwards.
3. Leave a ladder the next grant extends instead of restarts.

The paper asks how a fixed compute and memory budget should be allocated across high-information data,
recurrent processing, and periodic exact attention. Every active experiment must select an input to the
released model, test transfer or composition, validate scale/token-horizon/length transfer, explain a
declared mechanism, or price the released system. Unrelated exploration is excluded.

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

The public target is fixed before results because absolute quality is the first priority:

| Role | Parameters | Tokens | Tokens per parameter |
| --- | ---: | ---: | ---: |
| Target | 1.2B | 400B | 333 |
| Throughput fallback | 1.2B | 320B | 267 |

The retained 600M/800B geometry is not selectable in grant 1. The near-constant-token-ratio scale
ladder cannot identify the separate model-size and token-horizon effects needed to choose between 333
and 1,333 tokens per parameter. It now tests architecture transfer only. If optional FP8 qualifies on
the node, extra throughput buys tokens rather than changing the fixed model size.

The active R11 planning successor is [`targets/scale-targets-v2.json`](targets/scale-targets-v2.json),
with exact accounting in [`targets/ACCOUNTING.md`](targets/ACCOUNTING.md). Under the explicit 32,003-row
D5 fallback, Shape A materializes to 1,195,884,576 shared-head parameters; the rounded 1.2B label is
the model-size class, not an exact count. The old 32,000-row
[`targets/shape-a`](targets/shape-a/) record remains hash-pinned history.

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
| Embeddings | physically tied input/LM head; tokenizer D5 pending; Mistral 32K fallback | [`embedding_head_contract_v1.json`](embedding_head_contract_v1.json), [`TOKENIZER.md`](TOKENIZER.md) |
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

Under fixed training compute and serving memory, high-information data reduces tokens-to-quality while
recurrent layers compress routine sequence processing and periodic exact-attention layers preserve
global access. The data and memory-allocation gains count only if they transfer, compose, survive scale
and token horizon, and materialize in one held-out 1.2B system.

### 3.2 Standard

Coverage of the Kimi Linear, Kimi K3, DeepSeek-V4, and DeepSeek-V4.1-Flash reports, plus the thing
they omit: isolated
component evidence before the combined model, for data as well as architecture. Each line of the
flagship config maps to one figure with seeds, a paired bound, and a measured cost. A section that
changes no number in the config is cut. A config line with no figure is labeled inherited.

### 3.3 Sections and required evidence

1. Resource-allocation problem and controlled experimental framework.
2. Allocating training data through E1–E4 and the sealed audit.
3. Allocating exact memory through dense/hybrid and component comparisons.
4. Transfer and composition through five-seed I1, three-seed I2, scale, and mature horizon.
5. Held-out fixed 1.2B flagship, context extension, and predicted-versus-observed quality.
6. Hardware and release frontier: training, prefill, decode, runtime/persistent state, output-token
   cost, comparators, negative results, limitations, and reproducibility.

### 3.4 Headline

One sentence of the form "the fixed 1.2B model matches model X at 1/N the training compute and 1/M
the resident state at 128K under a named hardware envelope." X, N, and M come from checked quality,
context, and systems results. “Runs on a laptop” is included only after a qualified recurrent GGUF or
equivalent CPU runtime exists. No draft headline is stable before those results.

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
| | | | | **76** | **493** |

E1W runs four one-seed screens, then adds two seeds to each finalist. Each E1S category runs three
one-seed treatments, then adds one seed to its top two. E2c uses three seeds because it sets the
flagship mixture. E3 and E4 use two seeds because their expected effects are large.

The data decision is not a single weighted loss. Selection uses equal-domain bits per UTF-8 byte,
paired confidence intervals, and a hard no-regression guardrail for web, code, math, synthetic,
science, and reference separately. A sealed audit is opened once after selection. Exact rules,
bounds, fallbacks, and artifacts are frozen in [`data_plan_v2.json`](data_plan_v2.json).

E3 is the highest-leverage experiment in the program. If repetition is close to free, as
Muennighoff et al. report up to four epochs, the preparation target drops from 500B unique tokens to
about 150B and the critical path in section 10 shrinks with it. Run it first.

E5 curriculum shape was retired before outputs. Stable training defaults to uniform sampling within
the E2-selected source treatments, followed by the E4-selected decay mixture. This is an operational
default, not a curriculum claim. Its former 100-hour envelope now closes two direct threats to the
paper: architecture-specific data gains and non-composing individually selected mechanisms.

### 4.3 Architecture decisions

The complete protocol and evidence map are in [`ARCHITECTURE.md`](ARCHITECTURE.md). All launch
decisions run at 350M and 10B tokens unless noted. Each has a default. Day 21 launches the flagship
with the winner or the default. No decision may be added after day 1.

**Pass rule.** An alternative is eligible only when its upper one-sided 95% bound over three seeds is
inside the 0.01-nat aggregate margin at matched wall-clock, every source bound is inside the 0.02-nat
guardrail, and 32K/128K plus original-4K retention gates pass. Non-inferiority makes an alternative
eligible; it promotes only by its declared quality, state, or systems benefit. Ties keep the default.

| ID | Question | Arms | Runs | Default | GPU-h |
| --- | --- | --- | ---: | --- | ---: |
| C0 | Shared control | KDA 3:1 NoPE dense | 3 | | 63 |
| D2 | Global-layer position encoding | partial RoPE on 32 of 128 dims | 3 | NoPE | 63 |
| D3 | Recurrent to global ratio | 5:1 (20 recurrent, 4 global) | 3 | 3:1 | 63 |
| D4 | Peak LR and batch | four LR points at 1M-token batch, 4B tokens each | 4 | scaled from 150M | 33 |
| D6 | Hyperparameter transfer rule | two extra 150M points to fit LR against width | 2 | empirical fit | 5 |
| D7 | Recurrent operator attribution | FLA-initialized scalar-decay GDN/sigmoid/NoPE | 3 | KDA | 63 |
| D8 | KDA output gate | SiLU | 3 | sigmoid | 63 |
| | | | **21** | | **353** |

**D1 is retired for this allocation.** The first flagship is dense in width. This removes a late
architecture branch plus its unqualified expert optimizer, Hopper routing kernel, distributed-state,
export, and serving dependencies. The identifier is intentionally not reused. Its 132 GPU-hours move
out of MoE: 126 hours fund D7 and D8, the two missing dense causal comparisons, and 6 remain in the
overall reserve.

D7 asks whether KDA itself is needed under an otherwise matched sigmoid/NoPE parent. D8 closes the
known gate-attribution gap: KDA currently hardcodes sigmoid, while the clean natural-language
sigmoid-versus-SiLU result is on GDN and only one seed. Both reuse C0 only under exact parent, data,
training, seed, and analysis identity.

D5 is decided before grant experiments. The deterministic trainer and static evaluator now exist;
final six-category inputs are not yet frozen. Compare Mistral 32K with Speck BPE vocabularies of
32,000, 32,768, and 40,960 pieces, statically advance two custom candidates, then run the matched 60M
local-3090 pilot in [`TOKENIZER.md`](TOKENIZER.md). Static fertility cannot select the tokenizer.
Mistral remains the fallback if no custom candidate passes BPB, category, systems, and audit gates.
No 64K arm fits the uint16-plus-chat and parameter-efficiency contract.

D6 is what makes the next scale cheap and is a paper figure in its own right.

### 4.4 Integrated validation

The active [`integration_plan_v2.json`](integration_plan_v2.json) strengthens the retired E5 envelope
with 22 newly mandatory hours for the paper's central interaction:

| ID | Question | Design | New training runs | GPU-h |
| --- | --- | --- | ---: | ---: |
| I1 | Does the E2-selected data improvement transfer between architectures? | 150M/3B, dense versus C0 hybrid crossed with balanced-prior versus selected data, five seeds | 20 | 54 |
| I2 | Do individually promoted D2/D3/D7/D8 settings compose? | 350M/10B assembled recipe versus exact C0, three seeds | at most 3 | 63 |
| I3 | Do capability, systems, and mechanism interventions agree with the component decisions? | Frozen held-out, state/cache interventions, 32K/128K, 4K retention, cost and energy reports | 0 | 5 |
| | | | **at most 23** | **122** |

I1 reports data, architecture, and interaction effects but cannot reopen E2 selection. I2 applies every
compatible individually promoted setting exactly once. If that assembly fails any aggregate, source,
long-context, retention, or declared-benefit gate, the flagship uses complete C0; no favorable subset
search is allowed. If no alternative promotes, C0 already satisfies I2 and the unused hours are not
spent. This confirmation completes before the scale ladder so every scale point measures the same
launchable architecture.

### 4.5 Scale ladder and token-horizon reversal checks

The I2-confirmed architecture is compared with a dense control along a near-constant-token-ratio
ladder. This tests architecture transfer; it does not select model size. The fixed flagship remains
excluded from fitting and becomes a held-out realization point. S2 asks whether the hybrid effect
reverses after a materially more mature token horizon.

| Points | Scale and tokens | Runs | GPU-h |
| --- | --- | ---: | ---: |
| Low anchors | 60M/1.2B, 220M/5B | 4 | 14 |
| Mid | 150M/3B, 350M/7B | 4 | 35 |
| Reversal check S1 | 750M/15B, both arms | 2 | 134 |
| Mature-horizon check S2 | 350M dense/hybrid continued from 7B toward approximately 23B; exact endpoint frozen from pre-output GH200 throughput | 2 | 85 |
| | | **12** | **268** |

A scaling-efficiency claim additionally requires uncertainty on the fit, residual diagnostics, the S2
horizon result, and the flagship itself as a held-out realization point. S2 is a reversal sentinel,
not population-level equivalence evidence.

### 4.6 Outside the active experiment matrix

Reader Attention, MQA/MLA cache alternatives, attention residuals, sparse/compressed attention, and
other archived axes receive no grant-1 runs. Existing completed evidence may appear as background or
negative results, but spare capacity cannot reactivate them.

### 4.7 Deferred features

No MoE, depth-routing, compressed-attention, or asymmetric-compute run belongs to this grant or gates
the flagship. Retained reference code has no active experiment, result-selection, scaling, serving, or
follow-on planning authority.

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
land on the data volume. On 2026-09-09 root has about 125 GiB free and the data volume about 5.1 TiB
available. Prune raw files as each source completes; the volume does not hold raw and packed
simultaneously at full scale.

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
  symbolic-composition protocols, position-wise and trailing-token loss on complete documents, and
  original-4K loss after every extension stage.
- **Comparators:** SmolLM2-360M and 1.7B, Qwen3-0.6B and 1.7B, Gemma 3 1B, LFM2-700M and 1.2B,
  Llama 3.2 1B, each with its training-token budget printed alongside.
- **Serving:** TTFT, TPOT, tokens per second, resident state, and peak memory at 4K, 32K, and 128K
  on GH200, RTX 3090, and CPU through GGUF.
- **Out of scope:** HELMET and NoLiMa. Audits are in findings 42 to 54 and 124 to 128.

## 7. Compute and schedule

Grant: 5,000 GH200 GPU-hours on one 4-GPU node. One full node hour consumes four GPU-hours. Existing
estimates use 350 TFLOP/s per GPU plus separate planning overheads; P0 must replace that potentially
ambiguous combination with measured exact-shape end-to-end tokens/s and checkpoint/evaluation cost.
The 400B target remains only when it fits the authorized flagship envelope; otherwise use the frozen
320B fallback. The window must be at least two months; three is comfortable.

| Track | GPU-hours | Share |
| --- | ---: | ---: |
| Data experiments, E1W/E1S and E2 to E4 | 493 | 10% |
| Dense architecture decisions, C0 and D2 to D8 | 353 | 7% |
| Integrated validation, I1 to I3 | 122 | 2% |
| Scale ladder and reversal | 268 | 5% |
| Flagship pretraining | 2,425 | 49% |
| Extension, anneal, SFT, evaluation, serving | 450 | 9% |
| Reserve | 889 | 18% |
| | **5,000** | |

The reserve is sized for a first run on unfamiliar hardware with an untested stack, not as slack to
fill with extra arms. [`plan_v2.json`](plan_v2.json) is authoritative for phase budgets, dependencies,
throughput responses, and cuts; [`EXECUTION.md`](EXECUTION.md) is its readable operating view.

Calendar ranges are advisory. Exit gates and the day-21 configuration freeze are binding. Independent
arms may move within a phase or across GPUs, but work may not cross an unmet dependency or introduce a
new architecture axis.

## 8. Releases

- Base, pre-decay, 128K-extended, and instruct checkpoints in native, Transformers, and GGUF form,
  with model weights under Apache-2.0; repository source code remains MIT.
- Every scaling-ladder checkpoint at every scale, both arms.
- All experiment configs, source/data manifests, revisions, filters, aggregate statistics, attribution,
  and packed-data hashes. Raw source text and derived packed shards are not redistributed.
- The findings ledger and the raw result JSON.
- A serving benchmark script others can run on their own hardware.

## 9. After this grant

Preserve the pre-decay checkpoint, contracts, and complete evidence ladder. Do not choose or plan a
follow-on program until Grant 1 results, release use, and the independent audit identify the next
highest-value question.

## 10. Before day 1

[`PREGRANT.md`](PREGRANT.md) is the complete readiness checklist and records current status. The
critical path is a code-inclusive corpus and neutral held-out set, tokenizer qualification, a real
production-data calibration and conservative 20B plus 150B/500B projections, complete long-document
data, and one four-GH200 training/resume qualification. The complete 2B calibration passes all six
stages; R3 closes by project-owner fallback through a conditional 150B branch. The 20B attempt remains
paused and 500B remains blocked pending E3 plus an operations/storage successor. R4's twelve real
primary/unseen input views pass global near deduplication and the final R4 partitions pass every gate.

The RTX 3090 is reserved for the D5 tokenizer pilot, comparator serving measurements, and
representative export rehearsals. It does not run new architecture searches. A paid allocation does
not start while a launch-critical pre-grant item lacks either a passing artifact or an explicit
non-GPU fallback.

## 11. Risks

| Risk | Mitigation |
| --- | --- |
| Data not ready on day 1 | Start now; run E3 early to potentially cut the target by 3x; prepare the stable corpus first |
| Volume cannot hold raw and packed together | Prune raw per source as it completes |
| arm64 kernel or four-GPU DDP failure | The rented-node day; fall back to Torch recurrence and bf16 |
| Job time limits and preemption | Resume path tested before the grant |
| Scope expands during the allocation | Dense-width architecture and the I1/I2 integration questions are frozen; no new axis may use reserve |
| Data gains do not transfer across architectures | I1 measures the interaction and narrows the claim without reopening E2 |
| Individually selected mechanisms do not compose | I2 falls back to complete C0 without post-hoc subset search |
| Ablations do not transfer to 300+ tokens per parameter | Confirm at 750M; weight E4 highest; state the limitation |
| Two-month window | Pre-decided cut order in section 7 |
