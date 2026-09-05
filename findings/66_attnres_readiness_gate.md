# 66 — Attention Residuals readiness gate

## Module-graph correction

Speck has 20 logical blocks but two residual updates per block: sequence mixing and feed-forward. Full
and Block Attention Residuals must therefore operate over 40 ordered residual modules, not 20 block
labels. At the final module, Full AttnRes can see at most 40 sources: the embedding plus 39 completed
module updates.

The approximately eight-block candidate partitions the 40 updates into eight blocks of five modules.
Within its final block, its largest source set contains nine values: the embedding, seven completed
block sums, and the current partial block. This is the actual bounded-depth state being compared.

## Initial isolation

Four arms are required at one fixed sequence parent, depth, and width:

- standard PreNorm;
- token-independent learned static depth weights;
- Full AttnRes; and
- eight-block AttnRes.

The static arm is essential: it tests whether a gain requires token-content-dependent source weights
rather than merely learning a better fixed residual mixture. AttnRes fixes one softmax depth head,
RMS-normalized keys, unnormalized values, and one zero-initialized pseudo-query per module based on the
cited controlled ablations. These weights remain token-dependent because normalized source keys depend
on the token.

Correctness covers explicit equations, source order, zero initialization, forward/gradient parity,
online block updates across arbitrary chunks, prefill/decode, recomputation, checkpoint/resume, export,
and activation/workspace accounting. AttnRes adds no persistent sequence KV or recurrent state, but its
training activations grow with sequence length and retained depth sources.

## Required successors

If eight-block AttnRes passes, block counts 4, 8, and 12 use deterministic quantile boundaries over 40
modules; Full AttnRes remains the uncompressed control. A selected residual must then face matched
three-depth-by-three-width grids for both standard and routed residuals. This prevents comparison only
at the standard residual's shape optimum.

Fixed-token language quality and every source guardrail remain constraints. Content routing must beat
the static control on a frozen diagnostic, and fixed-compute loss or time-to-quality must clear a 10%
paired systems threshold. Activation, recomputation, and runtime overhead remain hard gates.

## Decision

No residual, block count, or geometry is selected. Implementation, training, and promotion remain
blocked until the sequence parent is fixed; the next action is only a materialized 40-module source
graph and small-tensor reference equations.

## Artifact

- [AttnRes readiness gate](../research/paper-1/attnres_readiness_v1.json)
