# Attention Residuals

- **Paper:** [arXiv:2603.15031](https://arxiv.org/pdf/2603.15031)
- **Version reviewed:** v1, 16 March 2026
- **Code:** [MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)
- **Primary topic:** content-dependent information flow over model depth

## Central claim

Standard PreNorm residuals feed every layer the unit-weighted sum of all earlier layer outputs. As depth
increases, hidden-state magnitude grows and each new normalized update becomes a smaller fraction of the
stream, a problem the paper calls **PreNorm dilution**. Attention Residuals (AttnRes) replace this fixed
sum with softmax attention over earlier layer outputs.

The operation attends over **depth for each token independently**. It does not add another sequence
attention operation and does not enlarge the token KV cache.

## Full AttnRes

For layer `l`, the sources are the token embedding and every earlier module output. Each source is both a
key and value. Keys receive RMSNorm, and the query is one learned vector `w_l` of hidden dimension `d`.
The layer input is the softmax-weighted sum of those values.

The pseudo-query is input-independent, but its dot product with each token's normalized earlier output
makes the resulting weights token-dependent. This distinction matters: the paper's static-weight control
does not improve on baseline, while content-dependent softmax weighting does.

Full AttnRes costs `O(L^2 d)` arithmetic and stores `O(Ld)` values per token across depth. Depth is small
relative to sequence length, so arithmetic is minor. At scale, the real cost is retaining and communicating
earlier activations under recomputation and pipeline parallelism.

## Block AttnRes

Block AttnRes groups modules and sums outputs within each block. A layer attends over the embedding,
completed block sums, and—after the first module in a block—the current block's partial sum. With `N`
blocks, storage and pipeline communication fall from `O(Ld)` to `O(Nd)`.

The paper typically holds `N` near eight. Cross-stage caching sends only newly completed summaries, and
a two-phase schedule batches all inter-block queries before walking the intra-block dependencies. Online
softmax merges the two pieces exactly. The reported inference overhead is below `2%` on typical loads.

Initialize every pseudo-query to zero. This starts the model at a uniform average over available sources
and avoids early training volatility.

## Evidence

- Scaling sweeps cover 194M–528M active-parameter models. At 5.6 PFLOP/s-days, Block AttnRes reaches
  fitted loss `1.692` versus baseline `1.714`, which the authors equate to a `1.25×` compute advantage.
- Full and Block variants have similar scaling slopes; at the largest small-scale point their loss differs
  by only `0.001`.
- A matched 48B-total/3B-active Kimi Linear model is trained on 1.4T tokens. The only architectural
  change is AttnRes. It improves every reported downstream task; the largest cited gains include
  GPQA-Diamond `+7.5`, MATH `+3.6`, and HumanEval `+3.1` points.
- AttnRes bounds the growth of hidden outputs and distributes gradients more evenly across depth in the
  paper's diagnostics.
- Softmax beats sigmoid aggregation; one shared depth head beats multi-head; RMSNorm on keys is
  necessary; and static DenseFormer-style weights provide no gain.
- Under fixed compute and parameters, AttnRes shifts the best tested shape from width/depth ratio about
  `60` to about `45`, favoring a deeper, narrower model.

## What matters for Speck

AttnRes is orthogonal to the KDA-versus-GDN and global-attention questions. Its parameter cost is roughly
one RMSNorm and one `d`-vector per module, making it attractive for Speck's deep/narrow small-model
regime.

A disciplined test should compare standard residuals, Full AttnRes, and an approximately eight-block
variant at fixed parameters and tokens. It must include:

- validation loss over at least three seeds;
- hidden/output and gradient magnitude by depth;
- throughput and activation memory at both 4K and the promoted long length;
- a depth/width cross-sweep, because the baseline's optimal shape may not remain optimal;
- export and cached-decode feasibility before promotion.

## Limitations and cautions

- No mainstream consumer runtime currently treats AttnRes as a standard residual path.
- The 48B result is one Kimi Linear MoE family; interaction with dense convolutional or GQA-heavy models
  is not established.
- Block summaries irreversibly merge within-block outputs. The paper shows a graceful trade-off, not
  equivalence to Full AttnRes.
- Prefill storage is sequence-length dependent: the paper's 128K, eight-block example is 15 GB before
  sharding/chunking optimizations.

## Bottom line

AttnRes is a credible, cheap depth-axis improvement with unusually clear ablations. Test it after the
sequence mixer is stable, and judge it together with a new depth/width optimum and runtime cost rather
than dropping it into an already tuned baseline.
