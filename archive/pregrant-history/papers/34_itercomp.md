# IterCOMP: iterative evidence sufficiency and missing-hop compression

- **Paper:** [arXiv:2608.13588v1](https://arxiv.org/abs/2608.13588v1)
- **Version reviewed:** v1, 12 July 2026; ACL 2026 main conference
- **Code:** no dedicated implementation declared on the arXiv record
- **Primary topic:** training-free prompt compression that repeatedly judges whether all evidence needed
  for multi-hop QA is present and asks for missing information

## Mechanism

IterCOMP splits retrieved documents into sentences and scores each against the current question with
frozen BGE-M3 semantic and lexical signals. It retains segments above a global percentile threshold.
An LLM then judges whether the accumulated evidence is sufficient for the original question. If not, it
names the missing information as a follow-up question, reruns filtering with that question, and
accumulates new evidence until answerable or a five-iteration limit.

The preliminary experiment defines `Full` as containing sub-answers for every required hop and `Partial`
as containing only an initial subset. The missing-information target is the next annotated subquestion.
This directly operationalizes conjunctive evidence sufficiency, premature stopping when a source is
missing, and targeted missing-source recovery.

## Evidence boundary

The sufficiency study samples 400 MuSiQue examples at each of two, three, and four hops. GPT-4o derives
sub-QA pairs from dataset decompositions. Four judge models obtain only 63–84 overall F1; full-evidence
accuracy falls with hop count (for example, Llama-3.1-8B from 86.5% to 47.3%), showing that an LLM
sufficiency judgment is not a reliable gold diagnostic.

Main evaluation uses dev sets of MuSiQue, 2WikiMultiHopQA, and HotpotQA with Llama-3-8B as reader. The
semantic/lexical weight is 0.6; relevance percentiles 85/90 are empirically aligned to oracle compression.
IterCOMP reaches F1 27.36/39.69/51.78 at retained-token ratios 0.14/0.37/0.19, above the compared prompt-
compression baselines but below gold-document oracles. MuSiQue F1 declines from 30.19 at two hops to
20.72 at four, while iterations and compressed tokens rise.

Ablations make iterative refinement the largest contributor; removing it gives 17.39 versus 27.36 F1,
and removing answerability judgment gives 23.63. However, the reported API cost reductions and speedups
explicitly cover only the final reader call, excluding BGE scoring and up to five LLM controller calls.
The paper flags controller error propagation, iterative overhead, hyperparameter sensitivity, and lack
of non-Wikipedia/domain evidence. The exact controller/API revisions and a dedicated code artifact are
not declared.

## What matters for Speck

The notions that all hop evidence is necessary, partial evidence is unanswerable, a compressor should
detect missing information, and restoration should target the missing hop are prior art. N2 cannot claim
the conjunction, sufficiency test, or missing-source recovery workflow.

A potentially distinct study would replace a fallible, costly LLM judgment with preregistered source
identities and a cheap internal-state diagnostic, prove incremental held-out prediction beyond IterCOMP's
Full/Partial answerability score, and intervene at equal physical state rather than growing the prompt
through extra retrieval. That is a narrower experimental distinction, not current novelty.

## Bottom line

IterCOMP is the closest direct overlap so far. Only a quantitative, mechanistic, pre-output predictor and
equal-budget causal law across KV/sparse architectures might remain; the general all-required-source
idea is no longer defensible as novel.
