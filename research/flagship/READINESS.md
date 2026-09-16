# Readiness assessment — 2026-09-16

**The program is planned at the design/budget level; it is not fully specified for execution or ready
for flagship training.** The application is still under review according to the last owner-supplied
response. No award or access date has been confirmed here. The [catalog](../catalog.json) selects
contracts and the [queue](PREPARATION_QUEUE.md) orders remaining work.

## What is settled, and what is still open

| Area | Settled or checked | Required next |
| --- | --- | --- |
| Model | 1.2B KDA/GQA hybrid, frozen Mistral tokenizer, 320B default and 400B measured-fit stretch | Exact-shape GH200 qualification and achievable end-to-end token horizon |
| Research | Dense/hybrid × standard/targeted supervision, three paired seeds; one preselected scale/horizon transfer check | Independent novelty/design challenge, task pilots and final attainable endpoints |
| Compute | 5,000-hour balanced envelope; dependencies, cuts and reserve rules recorded | Measured rates, complete per-run costs and allocation-specific calendar |
| Runtime | CPU optimization, two-rank Gloo, persisted checkpoint/RNG restart and bounded failure handling checked | arm64/CUDA/NCCL, independent numerical/cache parity, production-loader recovery, Slurm/requeue and sustained throughput |
| Data | Several real filtered text/token stocks and preserved finite acquisition work | Production supply/exposures, joint eligibility, coherent units, family partitions and immutable packed manifests |
| Evaluation | Claims, paired analysis, multiplicity and broad guardrail structure selected | Actual disjoint task/benchmark/scorer identities, numerical floors/margins, output budgets and real heldout bundles |
| Final assistant | Owner requires `<think>...</think>` followed by the answer | Separate-codebase recipe reconciliation, template/parser/loss-mask choices, reasoning limits, cost and export qualification |
| Paper | Methods structure, closest-work review, claims explicitly pre-results | Controlled outcomes, useful-model evidence, independent review and reproducible results |

The 1,274-test software pass does not establish model quality, experimental power, available training
supply or GH200 feasibility. Some settings intentionally depend on R0/R1 measurements; those stages
must complete before confirmatory outputs. A conditional endpoint is not an already frozen run manifest.

## Selected compute envelope

| Work | GPU-hours |
| --- | ---: |
| R0: hardware/runtime qualification | 70 |
| R1: recipe/task/endpoint calibration | 80 |
| R2: replicated controlled study | 360 |
| R3: one transfer check | 180 |
| R4: study evaluation and analysis | 60 |
| B0: broad base pretraining | 2,425 |
| L0: context and instruction/reasoning capability envelope | 700 |
| V1–V3: final evaluation, serving and export | 236 |
| Protected reserve | 889 |
| **Total** | **5,000** |

These are caps and planned allocations, not measured runtime forecasts. The 700-hour capability
split remains the existing planning envelope; an undecided reasoning recipe has not been demonstrated
to fit it. Any required reallocation needs a costed successor before affected training. Four allocated
GPUs consume four GPU-hours per wall hour: 5,000 hours equal about 52.1 full-node days within the
90-calendar-day window if awarded. Dependencies dominate overlapping target dates on the single node.

## Current operational blockers

Stack-Edu stopped at its working/free-space guard after 1,856 units and 436,048,483 tokens before
full exclusion. The [failure observation](../../results/systems/stack-edu-storage-stop-20260916.json)
preserves progress and logs. It is not completed source capacity; storage/recovery reconciliation is
needed before resuming. FineWeb follow-up was still active at this assessment. Neither worker
completion would by itself finish production datasets or authorize training.

## Next sequence

1. Reconcile the stopped Stack-Edu attempt and verify completed finite source outputs as they finish.
2. Finish source-capacity/exposure planning and materialize coherent, family-separated pilot inputs.
3. Pin evaluation/scorer identities and complete the pre-results novelty, matching and analysis review.
4. Complete local runtime checks; qualify site-specific hardware, scheduler and throughput after access.
5. Use bounded R1 measurements to freeze numerical gates, optimization settings, endpoints, transfer
   route, per-run costs and launch manifests before R2 confirmation.
6. Reconcile the separate post-training work with the owner-selected reasoning response format and
   capability budget before those stages. Preserve base and intermediate checkpoints separately.

The [response decision](release_behavior_decision_v1.json) is a product-policy addendum, not a new
operator, vocabulary allocation or training method. The long-context research focus remains subject
to its existing pre-results gates; changing domain requires a coherent successor before confirmation.
