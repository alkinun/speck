# HeadKV-R2: head-level cache allocation from retrieval-reasoning profiles

- **Paper:** [arXiv:2410.19258v4](https://arxiv.org/abs/2410.19258v4)
- **Version reviewed:** v4, 23 October 2025; accepted at ICLR 2025
- **Code/data:** [FYYFU/HeadKV](https://github.com/FYYFU/HeadKV)
- **Primary topic:** allocate a global KV-cache budget across layer/head cells using a static
  retrieval-plus-reasoning importance profile

## Mechanism

HeadKV estimates one importance score per attention head from synthetic needle examples, then pools a
portion of every head's nominal cache budget and redistributes that global pool in proportion to the
normalized scores. It uses SnapKV's trailing-window attention and pooling to select token identities
inside each allocated head budget.

The retrieval baseline scores a head when its top-attended context token lies in the generated answer
span. HeadKV-R2 constructs examples containing a reasoning statement, a correct answer, and a matched
incorrect answer. During generation it sums attention weights whose top-ranked input positions lie in
the complete correct-answer span. The profile is static after construction; it is loaded once at model
initialization and does not adapt to each downstream example.

The paper's appendix makes an important boundary explicit: although the constructed prompt contains a
reasoning path and distractor, the importance equation deliberately focuses on the correct answer to
retain a normalized maximum score. It does not measure whether every prerequisite reasoning statement
or distinct source survives.

## Evidence boundary

Profiles use two manually constructed retrieval-reasoning examples, five context lengths, and ten
insertion depths for 100 examples per model. LongBench single-/multi-document QA and LooGLE long-
dependency QA evaluate Llama-3-8B-Instruct and Mistral-7B-Instruct. The allocation-pool hyperparameter
is selected from eight values using 15% validation splits of the six LongBench tasks.

Reasoning-in-a-Haystack uses bAbI QA1–QA5 statements inserted into PG19, cache size 128, and contexts to
8K for Llama and 32K for Mistral. HeadKV-R2 averages 56.84 versus FullKV 57.04 on Llama and 43.09 versus
45.29 on Mistral. The paper notes that answers occur in the inserted needle, so retrieval-only HeadKV-R
also performs strongly; on Mistral it is slightly better than R2 on this aggregate.

Memory/latency use Mistral at 32K with FlashAttention and generation lengths through 4,096. The plotted
latency includes prefill plus decode and peak memory is averaged over three trials. The authors report
comparable systems cost to other compression baselines and flag per-head management/synchronization as
a multi-GPU limitation; no Speck-hardware ratio transfers.

## What matters for Speck

HeadKV-R2 is a mandatory baseline for any claim that reasoning-aware head allocation is new. It also
occupies global cross-layer/head redistribution, static task-profiled importance, whole-answer attention
scoring, and Reasoning-in-a-Haystack evaluation.

Speck N2 can remain distinct only as a narrower diagnostic claim: label every independently required
route and payload source, measure their conjunction per example, show incremental held-out prediction
beyond HeadKV-R2/answer-span importance and other routing metrics, and recover outcomes through fixed-
budget single-source restoration. Merely adding reasoning examples or evaluating multi-hop QA is already
overlapped.

## Bottom line

Reasoning-aware head profiling and allocation are not novel Speck contributions. HeadKV-R2 does not,
however, test conjunctive survival of separately annotated prerequisite sources or causal restoration;
those distinctions remain hypotheses, not established novelty.
