# SpeckLabs first program: design

2026-09-24. This document owns the design of the first program: its goal, method, experiments,
compute budget and release. [plan.json](../experiments/main-data/plan.json) owns the numbers,
[PLAN.md](../PLAN.md) owns status and the work order, and the [paper outline](paper.md) owns what
the report must show. No GPU run starts from this document.

## Goal: the data step

SpeckLabs scales in steps, each much larger than the last, and each step starts from what the
previous one measured. This first step targets the data of **pretraining and mid-training**. Its
product is a set of **measured, reusable findings about those pipelines**: which sources, filters,
mixtures, repetition, schedules and context data help, by how much, at what preparation cost, and
what did not work, published as the most open account we can give. A strong model is welcome but
is not the objective; 5,000 GPU-hours cannot buy a state-of-the-art model at any size, but they
can buy many careful experiments.

Post-training research (SFT data, RL, self-distillation) waits for a later step with a stronger
base, because at this scale its outcomes mostly measure the parent's weakness. Here SFT appears only
as a fixed **probe**: one frozen recipe that turns base checkpoints into comparable downstream
scores, and that yields a light assistant for the release.

A finding is only useful to the next step if it can be carried to a larger run. Each one records:

- the stage, the contrast and the controls, with negative and inconclusive results kept;
- the rule in reusable form (a filter, threshold, weight or schedule), not just the chosen dataset;
- the preparation pipeline and cost per accepted token, so it can be rerun on more data;
- the scales at which it was measured, and whether it held across them;
- the limits that stop it transferring, such as supply, licence or the fixed architecture.

[Rigel's](../experiments/main-data/frontier-data-research.json) six-phase recipe shows why this matters: it publishes the mixture of each phase, but no
experiment attributes any gain to a phase or source, so none of it can be scaled with confidence.

## Method

**A model ladder.** Most experiments run on small models, where they are cheap enough to repeat
with several seeds and to train long. Selected contrasts are re-run on larger rungs, so the paper
can report how often a small-scale conclusion held at larger scale. That transfer rate is itself
the most reusable result for the next step. Every rung is the same architecture family, generated
from one [shape rule](../experiments/ladder/shapes.py), so rungs differ only in scale.

**Tuned per rung.** Before any family runs on a rung, a short learning-rate sweep on the baseline
sets that rung's learning rate; hyperparameters then stay fixed within the rung, so a data effect
is never a mistuned optimizer. The parent's learning rate is extrapolated from the three sweeps.

**Noise before effects.** Each rung then trains its baseline with several seeds. The seed-to-seed
spread of every metric sets the smallest effect that rung can detect; no finding is reported below
it.

**Predeclared contrasts.** Before a family runs, its [record](../experiments/ladder/records) states
the question, arms, controls, primary metric, minimum useful effect, decision rule and cost. Arms
within a comparison share initial model tensors (checked by hash), tokenizer, context, objective,
schedule and token budget. Every arm is reported, including failed and inconclusive ones.

**Metrics that move at the scale measured.** Small models sit at the floor of generative
benchmarks. Ladder runs are scored by held-out loss per source on fixed family-disjoint validation
packs and by an early-signal suite (log-likelihood multiple choice and arithmetic). Parent branches
add generative benchmarks after the SFT probe. Final test partitions stay untouched until the end.

**Length as a scale axis.** Data conclusions can reverse with training length: aggressive filtering
tends to win short runs and lose long ones once the filtered pool repeats. The ladder therefore
varies tokens per parameter as well as parameters.

**No gate relaxed for small runs.** Every run trains only on data that passed the same pipeline:
source use, deduplication, benchmark firewall and family holds, as [Data](data.md) describes.

**One parent, many branches.** The 1.2B parent trains once through the stable phase of a
warmup-stable-decay schedule, with its stable checkpoints preserved. Decay branches start from them, mid-training branches from the chosen
decay, and the SFT probe from each branch. The released base and assistant are the best branches,
so there is no separate production track.

## The ladder

| Rung | Shape | Parameters | Tokens per run at 50 per parameter | Projected GPU-hours per run |
| --- | --- | ---: | ---: | ---: |
| 50m | width 512, 12 layers | 52,636,068 | 2.6B | 0.9 |
| 130m | width 768, 16 layers | 131,699,272 | 6.6B | 5.8 |
| 410m | width 1280, 20 layers | 410,024,470 | 20.5B | 56.6 |
| Parent | width 2048, 24 layers | 1,195,884,576 | 60B | 483.4 |

Projections use 6 × parameters FLOPs per token at 25% utilization of a 989.5-TFLOPs GPU, with
unmeasured cost factors of 1.3 at 16K and 1.7 at 32K context. They size the plan only: the GH200
qualification measures each rung, and run counts then follow from the budget lines. The 50m rung
also runs on the workstation RTX 3090 before access, which costs no grant hours.

## Experiments

Run counts are at the rung's default horizon unless a multiple is given. Arms are fixed in each
family's record before it runs, within its budget line; unspent hours go to follow-up contrasts that
the results call for, recorded the same way.

### Pretraining: the ladder (1,900 GPU-hours)

| Family | Question | Arms | Runs | Projected GPU-hours |
| --- | --- | --- | --- | ---: |
| LR | Which learning rate per rung? | Four rates at 50m and 130m, three at 410m, each at a quarter horizon | 50m 4, 130m 4, 410m 3 at 0.25 | 49.3 |
| P0 | How large is seed-to-seed variation per metric and rung? | Baseline seeds | 50m 5, 130m 3, 410m 3 | 192.1 |
| P1 | How strict should classifier floors be? | Stack-Edu score 3 versus 4+; a stricter web HQ floor; FineMath 3+ versus 4+ | 50m 4 | 3.7 |
| P2 | How much do exact and near deduplication matter? | No deduplication; exact only | 50m 2 | 1.9 |
| P3 | Which code, math and general shares? | Six mixtures around the starting 35/25/40 | 50m 6 | 5.6 |
| P4 | How many passes over a scarce bank match fresh data? | Code at 2, 4 and 8 passes; math at 4 | 50m 4 | 3.7 |
| P5 | Which source per bank? | FineWeb-Edu for HQ web; Stack v3 for Stack-Edu | 50m 2 | 1.9 |
| P6 | How much synthetic educational text? | 0, 15 and 30% | 50m 3 | 2.8 |
| P7 | Do filtering and repetition conclusions survive longer training? | Baseline and the strictest P1 arm at 12.5 and 200 tokens per parameter | 50m and 130m, 2 each at 0.25 and at 4 | 57.6 |
| C | Do 50m effects repeat on a new seed? | Second seeds of the arms that cross the minimum detectable effect | 50m 10 | 9.3 |
| T | Do 50m rankings hold at larger scale? | Best and worst arm of P1 to P6 | 130m 12; 410m 24 (two seeds each) | 1,429.3 |
| DR | Do decay conclusions hold below the parent? | One stable run to 80%, then three D1 decay arms | 130m and 410m, 1 at 0.8 and 3 at 0.2 each | 87.5 |

Before the parent launches, fit loss against size and tokens over the ladder and record a predicted
parent loss; the parent tests that prediction.

### Decay (300 GPU-hours)

Branches of the parent's stable run. The parent's final stable checkpoint is at 60B tokens.

| Family | Question | Arms | Projected GPU-hours |
| --- | --- | --- | ---: |
| D1 | Does a late shift to higher-quality math, code and web beat ordinary decay? | Unchanged mixture, quality-enriched, enriched plus verified derived data; 6B tokens each | 145.0 |
| D2 | Does a longer decay help? | The D1 winner at 12B tokens | 96.7 |
| D3 | Does the decay effect depend on how long the base trained? | Unchanged and enriched from the 30B checkpoint, 3B tokens each | 48.3 |

### Mid-training (800 GPU-hours)

From the selected decayed parent. Every arm keeps short-task replay and is scored for short-task
retention.

| Family | Question | Arms | Projected GPU-hours |
| --- | --- | --- | ---: |
| M1 | How much general replay keeps short-task ability while adding code and math capability? | 10, 25 and 50% replay, two seeds, 5B tokens each | 241.7 |
| M2 | Does repository-ordered code beat file-level code? | Repository-ordered with dependencies against the M1 winner, 5B tokens | 40.3 |
| M3 | What data extends context usefully at 16K? | Repacked baseline, long documents with distant dependencies, long chain-of-thought QA; two seeds, 5B tokens each | 314.2 |
| M4 | Does 32K add useful context beyond 16K? | The M3 winner continued at 32K for 3B tokens, only if 16K helped and runtime qualifies | 41.1 |

Useful context is measured by retrieval across positions, loss on a fixed suffix as related context
grows, and multi-file tasks, never by the configured maximum length.

### SFT probe (100 GPU-hours)

One frozen recipe: 75,000 conversations (about 150M tokens) from the retained assistant stock,
fixed data, masks, schedule and serialization. It runs twice (two data-order seeds) on every decay
and mid-training arm, 40 runs at a projected 1.2 GPU-hours each, and once on each 410m transfer and
seed run, 27 runs at 0.4 each. The probe's own data is never varied here; its scores are
comparisons between the checkpoints it probes.

## Compute

Four GH200s, 5,000 GPU-hours. Every line is a ceiling, and each operation is charged to one line.

| Line | GPU-hours |
| --- | ---: |
| Runtime qualification | 120 |
| Pretraining ladder | 1,900 |
| Parent stable run | 700 |
| Decay experiments | 300 |
| Mid-training experiments | 800 |
| SFT probe | 100 |
| Evaluation | 500 |
| Reserve | 580 |
| **Total** | **5,000** |

More than three quarters of the budget is experiments. The parent line covers 60B tokens at the
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
parent and branch checkpoint, the light SFT assistant, hash-bound data manifests from which the
corpora can be rebuilt, every evaluation output, the full compute ledger and every negative result.
Whether source data or derived shards are released too is an
[open decision](../PLAN.md#open-decisions).
