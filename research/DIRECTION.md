# SpeckLabs: efficient long-context intelligence

## Mission

Build open general-purpose models that learn, retain, access and reason over information with less
training compute, serving compute and memory. Model size grows with available resources and validated
scaling evidence; small models are the current experimental scale, not a permanent lab boundary.

The first allocation should deliver a useful model, a scientifically consequential result, and reusable
assets that support the next allocation. Efficiency means quality at a declared resource budget or cost
at a declared quality level. Report trade-offs rather than requiring every metric to improve together.

## First flagship

A 1.2B dense-width KDA/global-GQA model distinguished by understanding and reasoning across supplied
long context. Documents and conversation histories are the primary workloads. Technical/code
comprehension supports the result; ordinary assistance, writing, instructions, math and code remain
quality guardrails. The release is general-purpose and English-first.

32K useful capability is the first milestone, 64K an intermediate measurement, and 128K the target.
Advertise only the highest tested length passing both primary workload families and retained general
quality. Allocating a context window or finding a needle is insufficient.

## First paper

**Learning to Use Long Context Efficiently: Data and Memory Trade-offs in a 1.2B Hybrid Language Model.**

Study how memory architecture and dependency-requiring supervision interact at fixed resource budgets.
A replicated dense/hybrid × standard/targeted-supervision study is the central experiment. One focused
transfer check and the flagship establish the limits and practical consequence of the result.
The 1.2B flagship has no full-horizon dense counterfactual; it cannot establish that unmeasured effect.

The [selected execution plan](flagship/plan_v4.json) gives focused research 750 GPU-hours, base training
2,425, capability development 700, final evaluation/serving 236, and protected reserve 889. The total
remains 5,000 GH200 GPU-hours. Broad base-mixture and component searches are retired from this allocation.

## What compounds

Every release leaves continuation checkpoints, reusable source banks, learning/capability curves,
categorized failures, qualified inference, and reproducible comparisons. The next allocation chooses
among continued training, larger capacity and a method improvement using these results. No next
architecture or model size is preselected.

A useful external case combines a reproducible technical advantage, held-out demonstrations, independent
replication or trials, honest all-in costs, and a costed next experiment. Funding or external adoption is
an intended outcome, not a guaranteed result or substitute for evidence. No outreach is authorized by
this document.

## Scientific boundary

KDA, GQA, the 3:1 ratio, long-context continuation and synthetic supervision are inherited mechanisms.
The contribution must be a new controlled finding or useful realized frontier. Null interactions and
failed targets remain results. See [the paper contract](flagship/PAPER.md) and
[transition decision](flagship/PIVOT.md).
