# 82 — STEC adjacent evidence-verification baseline

## Classification

STEC is a current, relevant external baseline but not a direct implementation of the residual N2
internal-state question. It runs after multiple search trajectories and candidate answers already exist,
whereas N2 would have to predict failure from compressed model state before output.

For each normalized answer candidate, STEC compresses its trajectories into raw variants, supporting
evidence, contradicting evidence, a reasoning path, and group statistics. A verifier compares candidates
at that common granularity, checking coverage of question entities/relations/constraints and coherence
from evidence to answer, then selects only among existing candidates.

## Evidence

With eight Search-R1 trajectories and Qwen2.5-7B, STEC averages 0.347 EM across HotpotQA, 2Wiki,
MuSiQue, and Bamboogle, versus 0.301 for Search-R1 and 0.339 for verifier-only. The evidence-compression
ablation therefore adds 0.008 macro EM, with a slight MuSiQue regression and no paired uncertainty or
multiplicity analysis. Results across 3B/7B base/instruct models are directionally positive on average
but include per-dataset reversals.

The paper reports no state/token compression ratio, latency, FLOPs, energy, or complete search-plus-
compression-plus-verification cost. It declares no dedicated code/data artifact or immutable model,
retriever, corpus, and prompt identity.

## Decision

STEC-style structured evidence verification is mandatory on future agentic/multi-trajectory evaluations.
It does not restore novelty to all-required-source conjunctions, nor does it itself falsify or establish
a cheap internal-state predictor. The v3 N2 concept rejection and all architecture/training blocks remain
unchanged.

## Artifacts

- [STEC full-text note](../papers/35_stec.md)
- [Novelty landscape v4](../research/paper-1/novelty_landscape_v4.json)
