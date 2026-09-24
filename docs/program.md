# SpeckLabs first program: design

2026-09-24. This document owns the design of the first program: its goal, method, experiments,
compute budget and release. [plan.json](../experiments/main-data/plan.json) owns the numbers,
[PLAN.md](../PLAN.md) owns status and the work order, and the [paper outline](paper.md) owns what
the report must show. No GPU run starts from this document.

## Goal: the data step

SpeckLabs scales in steps, each much larger than the last, and each step starts from what the
previous one measured. This first step targets data. Its product is a set of **measured, reusable
findings about the data pipeline of each training stage**: which sources, filters, mixtures,
schedules and derived data help, by how much, at what preparation cost, and what did not work,
published as the most open account we can give. A strong model is welcome but is not the objective;
5,000 GPU-hours cannot buy a state-of-the-art model at any size, but they can buy many careful
experiments.

A finding is only useful to the next step if it can be carried to a larger run. Each one records:

- the stage, the contrast and the controls, with negative and inconclusive results kept;
- the rule in reusable form (a filter, threshold, weight or schedule), not just the chosen dataset;
- the preparation pipeline and cost per accepted token, so it can be rerun on more data;
- the scales at which it was measured, and whether it held across them;
- the limits that stop it transferring, such as supply, licence or the fixed architecture.

Rigel's six-phase recipe shows why this matters: it publishes the mixture of each phase, but no
experiment attributes any gain to a phase or source, so none of it can be scaled with confidence.

## Method

**A model ladder.** Most experiments run on small models, where they are cheap enough to repeat
with several seeds and to train long. Selected contrasts are re-run on larger rungs, so the paper
can report how often a small-scale conclusion held at larger scale. That transfer rate is itself
the most reusable result for the next step. Every rung is the same architecture family, generated
from one [shape rule](../experiments/ladder/shapes.py), so rungs differ only in scale.

**Tuned per rung.** Before any family runs on a rung, a short learning-rate sweep on the baseline
sets that rung's learning rate; hyperparameters then stay fixed within the rung, so a data effect
is never a mistuned optimizer.

**Noise before effects.** Each rung then trains its baseline with several seeds. The seed-to-seed
spread of every metric sets the smallest effect that rung can detect; no finding is reported below
it.

**Predeclared contrasts.** Before a family runs, record its question, arms, controls, primary
metric, minimum useful effect and decision rule. Arms within a comparison share initial model
tensors (checked by hash), tokenizer, context, objective, schedule and token budget. Report every
arm, including failed and inconclusive ones, at the same detail.

**Metrics that move at the scale measured.** Small models sit at the floor of capability
benchmarks. The ladder's primary metrics are held-out loss per source and domain on fixed
family-disjoint validation packs, plus benchmarks that show signal early (cloze and log-likelihood
multiple choice, code pass@k at large k). Generative capability benchmarks become primary only for
the parent and post-trained models. Final test partitions stay untouched until the end.

**Length as a scale axis.** Data conclusions can reverse with training length: aggressive filtering
tends to win short runs and lose long ones once the filtered pool repeats. The ladder therefore
varies tokens per parameter as well as parameters.

**No gate relaxed for small runs.** Every run trains only on data that passed the same pipeline:
source use, deduplication, benchmark firewall and family holds, as [Data](data.md) describes.

**One parent, many branches.** The 1.2B parent trains once, on a warmup-stable-decay schedule, with
its stable checkpoints preserved. Decay, mid-training and post-training experiments branch from it.
The released base and assistant are the best branches, so there is no separate production track.

## The ladder

| Rung | Shape | Parameters | Tokens per run at 50 per parameter | Projected GPU-hours per run | Grant runs |
| --- | --- | ---: | ---: | ---: | ---: |
| 50m | width 512, 12 layers | 52,636,068 | 2.6B | 0.9 | 120 |
| 130m | width 768, 16 layers | 131,699,272 | 6.6B | 5.8 | 100 |
| 410m | width 1280, 20 layers | 410,024,470 | 20.5B | 56.6 | 14 |
| Parent | width 2048, 24 layers | 1,195,884,576 | 60B | 483.4 | 1 |

Projections use 6 × parameters FLOPs per token at 25% utilization of a 989.5-TFLOPs GPU. They size
the plan only: the GH200 qualification measures each rung, and run counts then follow from the
budget lines. The 50m rung also runs on the workstation RTX 3090 before access, which costs no
grant hours.

## Experiments by stage

Each family below is a question, not a commitment to every arm. Arms are fixed in its predeclared
record before it runs, within the stage's budget line.

### Pretraining (the ladder)

| Family | Question | Arms (illustrative) |
| --- | --- | --- |
| P0 Noise | How large is seed-to-seed variation per metric and rung? | Baseline with 5 seeds at 50m, 3 at 130m, 2 at 410m |
| P1 Quality floor | How strict should classifier floors be? | Stack-Edu score 3 versus 4+; web HQ floors; FineMath 3+ versus 4+ |
| P2 Deduplication | How much does exact and near deduplication matter? | None, exact only, exact plus near |
| P3 Domain mixture | What code, math and general shares serve code and math without losing general ability? | Five mixtures around the starting 35/25/40 |
| P4 Repetition | How many passes over a scarce bank are nearly as good as fresh data? | 1, 2, 4 and 8 passes at fixed exposure |
| P5 Source choice | Which source to use per bank? | FineWeb-Edu versus Ultra-FineWeb HQ; Stack-Edu versus Stack v3 |
| P6 Synthetic share | How much synthetic educational text helps? | 0, 5, 15 and 30% |
| P7 Length | Do P1 and P4 conclusions survive longer training? | 12.5, 50 and 200 tokens per parameter |

The best and worst arms of P1 to P6 are re-run on the next rung. Before the parent launches, fit
the ladder and record a predicted parent loss; the parent then tests that prediction.

### Decay

Branches of the parent's stable run, replicated cheaply at 130m.

| Family | Question | Arms (illustrative) |
| --- | --- | --- |
| D1 Decay data | Does a late shift to higher-quality math, code and web beat ordinary decay? | Unchanged mixture, quality-enriched, enriched plus verified derived data |
| D2 Decay length | How long should decay be? | 10% versus 20% of the run |
| D3 Branch point | Does the decay effect depend on how long the base trained? | An early and a late stable checkpoint |

### Mid-training

From the selected decayed parent, at 4K unless stated.

| Family | Question | Arms (illustrative) |
| --- | --- | --- |
| M1 Replay | How much general replay keeps short-task ability? | 10, 25 and 50% replay |
| M2 Repository context | Does repository-ordered code beat file-level code? | File-level versus repository-ordered with dependencies |
| M3 Context data | What data extends context usefully? | At 16K: repacked baseline, long documents with distant dependencies, long chain-of-thought QA |

A 32K continuation runs only if 16K helps and runtime qualifies. Longer contexts are later work.

### Post-training

Every family runs on our parent and on an open external base of similar size (candidate: OLMo 2 1B),
because post-training outcomes depend heavily on the parent's existing ability. The external base
is a control, never a released Speck model.

| Family | Question | Arms (illustrative) |
| --- | --- | --- |
| S1 SFT scale | How does quality grow with SFT conversations? | 30K, 100K, 300K and 1M |
| S2 Verification | Does outcome-verified data beat an unfiltered pool of equal size? | Verified versus unfiltered |
| S3 Reasoning traces | Do longer reasoning traces help a small model? | Short versus long traces on the same tasks |
| R1 RL prompts | Does filtering prompts to a pass-rate band beat the full pool? | Full pool versus band-filtered |
| R2 Reward | How strict should rewards be? | Strict tests versus partial credit |
| X1 Self-distillation | Does verified self-generated data beat anchor-only continuation? | Anchor only versus self-distillation plus anchor |

## Compute

Four GH200s, 5,000 GPU-hours. Every line is a ceiling, and each operation is charged to one line.

| Line | GPU-hours |
| --- | ---: |
| Runtime qualification | 120 |
| Pretraining ladder | 1,600 |
| Parent stable run | 700 |
| Decay experiments | 200 |
| Mid-training experiments | 500 |
| Post-training experiments | 800 |
| Evaluation | 500 |
| Reserve | 580 |
| **Total** | **5,000** |

About three quarters of the budget is experiments. The parent line covers 60B tokens at the
projected rate with margin; the only 1.2B rate measured so far is the H100 pilot's 12,859 tokens/s,
an eager, checkpointed recipe at about 10% utilization, which is an upper bound on cost rather than
an estimate. [plan.json](../experiments/main-data/plan.json) records the rules for a faster or
slower measured rate; a surplus buys experiments, never a longer parent run.

## Model

The architecture is fixed by declaration: a KDA/NoPE-GQA hybrid with three KDA layers per global
attention layer, dense SwiGLU, tied embeddings and the frozen Mistral 32K tokenizer. It is not
compared with alternatives, so the paper makes no architecture claim. MoE, attention variants,
other sizes as products and contexts beyond 32K belong to later programs. The [model notes](model.md)
describe the reference and its runtime.

## Release

Release everything the licences allow: the paper, all pipeline and training code, every ladder,
parent and branch checkpoint, hash-bound data manifests from which the corpora can be rebuilt,
every evaluation output, the full compute ledger and every negative result. Whether source data or
derived shards are released too is an [open decision](../PLAN.md#open-decisions).
