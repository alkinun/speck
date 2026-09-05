# BRIEF: multi-hop evidence fusion by proposition compression

- **Paper:** [ACL Anthology 2025.findings-naacl.301](https://aclanthology.org/2025.findings-naacl.301/)
- **arXiv version reviewed:** [2410.15277v2](https://arxiv.org/abs/2410.15277v2), 15 February
  2025
- **Code/data:** [JasonForJoy/BRIEF](https://github.com/JasonForJoy/BRIEF)
- **Primary topic:** query-aware abstractive compression of retrieved documents into a multi-hop
  evidence summary

## Mechanism

BRIEF retrieves documents, concatenates them with the query, and uses a fine-tuned T5-large compressor
to produce a short natural-language summary for a frozen reader LM. Unlike KV eviction, it changes the
reader's input text and pays a separate 770M-parameter compression pass.

Its training targets are built from atomic, decontextualized propositions. A candidate multi-hop
question is decomposed into single-hop subquestions. For each subquestion, the pipeline measures whether
an individual proposition increases the reader's target-answer log likelihood, rejects the example if
any hop lacks a helpful proposition, and requires the selected propositions to come from distinct
documents. The target summary is the concatenation of the helpful proposition for every hop.

This directly operationalizes complete multi-source evidence before training a compressor. It is not a
per-example diagnostic of an internal sparse model, but the general idea that every hop-specific source
must be preserved for multi-hop compression is already present.

## Evidence boundary

The training corpus combines single-hop and synthetic multi-hop tuples from TriviaQA, Natural Questions,
HotpotQA, and MuSiQue. NQ/TriviaQA composition reaches three hops; HotpotQA and MuSiQue include up to
four. The pipeline uses open-source models for question composition/decomposition and propositionizing,
then trains the compressor by next-token likelihood on the concatenated proposition targets.

Evaluation uses Contriever over a December 2018 Wikipedia corpus split into non-overlapping 100-word
documents, T5-large as compressor, and Flan-UL2 as the principal 20B reader. Results report answer EM/F1
and a word-count compression ratio. On the 500-example HotpotQA subset, BRIEF reports 19.19x compression
with 31.20 EM / 42.07 F1, versus full top-five documents at 32.80 / 43.90 and RECOMP at 10.02x with
28.20 / 37.91. Compression ratio is not KV bytes, runtime, energy, or training cost.

The compressor is query-aware and relies on retrieved documents already containing sufficient evidence.
Helpful propositions are scored individually against the known target answer during synthetic-data
construction, which is not an online causal-necessity test. Generated summaries may paraphrase or merge
source identity, and final EM/F1 does not expose which required proposition was lost.

## What matters for Speck

BRIEF must be a cross-paradigm baseline for N2. The following are no longer plausible novelty claims:

- decomposing multi-hop questions into constituent hops;
- labeling one helpful proposition per hop;
- rejecting training examples without complete hop evidence;
- requiring evidence from distinct documents; or
- training compression targets that concatenate every hop's proposition.

Any surviving Speck contribution must be narrower: a prospectively frozen *diagnostic law* inside
native KV/sparse representations that adds held-out predictive value beyond BRIEF-style proposition
coverage and routing/cache baselines, plus a fixed-physical-budget intervention showing that restoring
one independently necessary missing source recovers the model output. Even that remains an unproven
distinction pending newer evidence-compression work.

## Bottom line

Constituent-complete evidence construction for multi-hop compression is prior art. N2 cannot claim the
conjunction itself; only incremental predictive and causal behavior under internal model compression may
remain distinct, subject to the 2026 successor landscape.
