# 63 — CSA selector-readiness gate before implementation

## Question

When does compressed sparse attention become a justified next mechanism rather than a bundle of
high-resolution compression, selector, sparse kernel, and coarse-coverage changes?

## Factorization

CSA is now explicitly downstream of the selected recurrent backbone, exact-cache representation, and
dense coarse HCA path. Those remain fixed. CSA alone supplies selective higher-resolution access; KDA
remains local, and a raw SWA branch is a later intervention. No cell may credit CSA while changing any
of those components.

The causal contract names immutable source spans, excludes future entries before top-k, separates
unfinished compression/selection tails, shares contiguous selected blocks across query heads, applies
causal masking to the current block, fixes deterministic ties, and requires one equivalent softmax
normalization. Full/full, sparse/sparse, and sparse-prefill/full-decode are separate reported modes.

## Selector sequence

The first gate is an oracle, not a learned router. Frozen dense-attention probability is aggregated by
candidate causal block; oracle top-block selection tests whether a block size and budget can retain the
needed evidence. If the oracle fails, no learned selector is trained for that cell.

Only oracle-passing cells compare a mean-key parameterless control and then a learned block indexer.
Primary selector evidence includes captured dense probability mass and recall of every required route,
payload, and distractor span—not top-k index overlap alone. Token-level selection requires a successor
contract proving a block-level failure and token-level oracle success at a realizable kernel budget.

## Conditional grid and cost truth

The mechanism grid crosses high-resolution compression 2/4/8, selection blocks 32/64/128 raw tokens,
and attended budgets 512/2,048/8,192 raw-token equivalents. Oracle screening eliminates infeasible
cells before router training and freezes the smallest passing frontier on held-out cases.

Selected attention is `O(Lk)`, but a dense index scan over `L/m` entries remains `O(L²/m)`. HCA's
separate dense-summary term also remains. State accounting includes compressed entries, index keys,
scales, dispatch metadata, and unfinished tails. A custom runtime must realize 20% primary improvement
and 25% state reduction under sparse-prefill/sparse-decode; analytic sparsity or prefill-only speed is
insufficient.

## Decision

No granularity, compression, budget, or selector is selected. Implementation, training, token-level
selection, and promotion remain unauthorized. After parent and HCA selection, the only authorized next
step is the frozen oracle block-mass feasibility matrix.

## Artifact

- [CSA readiness gate](../research/paper-1/csa_readiness_v1.json)
