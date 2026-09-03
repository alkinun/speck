# DeepSeek-V3.2 and DeepSeek Sparse Attention

- **Paper:** [arXiv:2512.02556](https://arxiv.org/pdf/2512.02556)
- **Version reviewed:** v1, 2 December 2025
- **Primary topic:** retrofitting fine-grained sparse attention into MLA

## Central claim

DeepSeek Sparse Attention (DSA) adds a lightweight indexer to an existing MLA checkpoint and lets each
query attend to only the top 2,048 cached token latents. DeepSeek-V3.2 is continued from a 128K
DeepSeek-V3.1-Terminus checkpoint and reports short/long quality parity with much lower deployed
long-context cost.

This is important evidence that sparsity can be introduced after dense pretraining—but the adaptation is
nearly one trillion tokens, not a cheap patch.

## Mechanism

The Lightning Indexer maps a query to several small indexer heads and one scalar weight per head. Each
past token has a compact index key. The score sums weighted ReLU query/key products across indexer
heads. The top-k cached MLA latents selected by that score enter the normal softmax attention operation.

For kernel efficiency, every selected MLA latent is shared across all query heads: the core is the MQA
mode of MLA. This differs from the earlier MHA-style reconstruction and is another potential quality
change to isolate.

The main sparse attention falls from `O(L^2)` to `O(Lk)`. The indexer itself remains `O(L^2)`, but its
head count/dimension and FP8 compute are much smaller than the full MLA path. DSA therefore lowers the
quadratic coefficient rather than making the complete operator strictly linear.

## Conversion recipe

1. Start from the already 128K V3.1-Terminus model.
2. Keep dense attention, freeze the base model, and train only the indexer for 1,000 steps. The target is
   the L1-normalized sum of dense attention scores across heads, optimized with KL divergence. This uses
   2.1B tokens.
3. Enable top-2,048 selection and optimize the entire model for 15,000 steps at 128K. Continue training
   the indexer against dense-attention targets restricted to its selected set. This uses 943.7B tokens.
4. Detach indexer inputs from the LM graph: the indexer learns only from its KL objective, while the base
   model learns only from language modeling loss.
5. Retain the same sparse path through post-training.

## Evidence

The paper compares V3.2-Exp with V3.1-Terminus under the same post-training strategy and reports no
substantial regression on its short- or long-context suite. Chatbot Arena Elo is similar; external
AA-LCR and Fiction.liveBench results are also cited as non-regressing or improved.

The service-cost plot is measured on H800 clusters and shows a widening prefill/decode advantage toward
128K. The paper's stronger final reasoning/agent scores also depend on specialist distillation, a mixed RL
stage exceeding 10% of pretraining cost, and 85,000 synthesized prompts across more than 1,800 agent
environments. They do not isolate DSA.

## What matters for Speck

DSA is the clearest recipe for retrofitting sparsity:

- distill an indexer from dense attention before pruning;
- retain a long joint adaptation stage;
- keep the selector objective separate from LM gradients;
- use a selection granularity that the production kernel can reuse across heads.

For Speck, compare token-level DSA with block-level MoBA. Token selection has higher retrieval precision;
block selection has much better memory locality. Evaluate selector recall directly: how often does top-k
contain every causal token the dense model materially uses on multi-hop examples?

## Limitations and cautions

- `k=2,048` is a large fixed budget; benefits are modest at short length and grow with context.
- The quadratic indexer remains and needs a highly optimized low-precision implementation.
- Nearly 946B adaptation tokens and a 128K base checkpoint make this an expensive retrofit.
- Reported parity lacks seed intervals and uses changing real-world evaluation snapshots.
- Full-model capability gains mostly reflect post-training scale, not the sparse operator.

## Bottom line

DeepSeek-V3.2 shows that MLA can be converted to fine-grained sparse attention without obvious quality
loss when given a distillation warm-up and massive continuation budget. It is a later retrofit strategy,
not the cheapest path for Speck's first model.
