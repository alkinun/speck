# Sparse Prefix Caching for hybrid and recurrent models

- **Paper:** [arXiv:2605.05219](https://arxiv.org/pdf/2605.05219)
- **Version reviewed:** v1, 17 April 2026
- **Primary topic:** distribution-aware placement of exact recurrent-state prefix checkpoints

## Problem and algorithm

An attention layer needs its per-token KV prefix, but a recurrent/SSM layer can resume from one exact
state at a checkpoint. Sparse Prefix Caching stores recurrent states at selected prefix depths. On a hit,
it restores the deepest cached recurrent checkpoint and exactly recomputes the remaining suffix while
reusing the attention KV cache.

For an overlap-depth distribution and `M` recurrent checkpoints across a prefix of length `N`, the paper
minimizes expected suffix recomputation. Uniform overlap and worst-case latency favor balanced gaps.
For a general distribution, prefix-sum costs and a monotone convex-hull optimization yield an exact
`O(NM)` dynamic program. An exponentially weighted histogram handles distribution drift; the reported
configuration uses decay 0.99 and refreshes the placement every ten requests.

## Systems boundary

All hybrid experiments retain the full attention KV cache. Methods differ only in recurrent checkpoint
locations. Positions are clipped to runtime block boundaries. Admission and eviction use a deliberately
fixed last-K policy to isolate within-entry placement; full trie optimization and integrated eviction are
outside the main formulation.

The prototype reports QuALITY, NarrativeQA simulations, and real system-prompt plus ShareGPT request
groups. Timing uses one deterministic run per configuration on an RTX 2080 Super host, with chunked
prefill and block-granular execution. Production-quality recurrent-state extraction/restoration and
joint admission/eviction remain future work.

## What matters for Speck

Sparse or distribution-aware recurrent prefix checkpoints are not a novel Speck systems method by
themselves. This paper supplies a direct exact baseline for any KDA checkpoint-placement claim.

Speck's heterogeneous state is richer—KDA, exact/compressed/sparse attention, incomplete compression
tails, and possibly depth/expert state—but extra state types alone do not create novelty. A distinct
systems contribution would need a new joint objective or algorithm that handles those coupled states,
prefix trees, eviction, latency/capacity constraints, and realized production execution while comparing
against this dynamic program.

## Limitations for transfer

- The main optimization is one prefix/edge; general branching tries and eviction are not jointly solved.
- Attention KV remains dense and complete.
- Timing is a prototype single deterministic run, not an online p99 study.
- Hardware, block sizes, overlap distributions, and fixed last-K capacity do not transfer as constants.

## Bottom line

The exact recurrent checkpoint-placement problem is prior art. Speck should use it as the baseline and
claim only a demonstrably new coupled heterogeneous-state result, if one survives full review.
