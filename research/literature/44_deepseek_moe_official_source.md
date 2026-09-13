# DeepSeekMoE official source audit for a conventional dropless reference

## Pinned primary sources

- [DeepSeekMoE official repository](https://github.com/deepseek-ai/DeepSeek-MoE/tree/66edeee5a4f75cbd76e0316229ad101805a90e01)
  at commit `66edeee5a4f75cbd76e0316229ad101805a90e01` contains the report, code/model
  licenses, and fine-tuning entry point.
- [DeepSeekMoE paper](https://arxiv.org/abs/2401.06066) defines generic MoE, fine-grained expert
  segmentation, and shared expert isolation. The repository PDF has SHA-256
  `a64b509e1410d09cf5b70570788d27790985fb9808ce436b9da8de73f2eed651`.
- [Official 16B Hugging Face release](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/tree/521d2bc4fb69a3f3ae565310fcc3b65f97af2580)
  at commit `521d2bc4fb69a3f3ae565310fcc3b65f97af2580` provides the training-capable
  custom model and exact released configuration.
- [DeepSeek-V3 official repository](https://github.com/deepseek-ai/DeepSeek-V3/tree/9b4e9788e4a3a731f7567338ed15d3ec549ce03b)
  at commit `9b4e9788e4a3a731f7567338ed15d3ec549ce03b` provides a later expert-parallel
  inference reference.

## Conventional and DeepSeekMoE semantics

The report's generic equations route a token to the top-k of a full softmax over expert affinities.
Only selected expert outputs are weighted, using their original full-softmax probabilities; those
selected values are not renormalized. Fine-grained segmentation splits each conventional expert into
smaller experts and proportionally increases both the pool and active count at fixed expert parameters
and arithmetic. Shared expert isolation then moves some active experts outside the router and adds their
full outputs to the routed sum.

For the released 16B model, every FFN except the first becomes MoE. It has width 2,048, 64 routed
experts with intermediate width 1,408, six routed experts per token, and two always-active shared
experts. The paper explicitly says no tokens are dropped: all experts for a layer are placed on one
device, while pipeline parallelism distributes layers. A small expert-level auxiliary balance factor is
used to prevent routing collapse. This is evidence for single-device-per-layer dropless semantics, not
for expert-parallel dropless execution.

## What the official 16B code provides

The pinned Hugging Face configuration uses softmax routing and `norm_topk_prob=false`, matching the
paper's unrenormalized selected full-softmax weights. Its training branch repeats each input token k
times, evaluates each selected expert, reshapes, weights, and sums all selected outputs. There is no
capacity factor, padding capacity, overflow fallback, or token-drop branch in the MoE implementation.
The shared path is one SwiGLU MLP whose intermediate width is multiplied by the shared-expert count;
its output is added to the routed path.

The training gate computes either sequence-level or batch-level auxiliary balance loss. A custom
autograd identity returns the MoE output unchanged in the forward pass and injects the auxiliary-loss
gradient during backward. The inference branch instead sorts flattened expert identities, executes
grouped expert batches, and combines duplicate token contributions with `scatter_reduce(sum)` under
`no_grad`.

This is a useful single-device training semantic reference, but it has no upstream parity test between
the training repeat/interleave path and the inference sort/scatter path. It does not specify stable
top-k ties, an expert token-order contract, checkpoints across a new dispatcher, or expert-parallel
gradient behavior.

## What the V3 code does and does not add

The pinned V3 inference code adds sigmoid/bias routing, group-limited selection, local expert ranges,
and an all-reduce of each rank's routed output. Every rank receives the same hidden states and computes
only its resident experts before output reduction. The top-level Transformer forward is explicitly
inference-only. This demonstrates a simple expert-sharded inference decomposition; it is not a token
all-to-all training dispatcher and does not provide backward, optimizer, capacity, or training-state
semantics.

## License boundary

Both official GitHub code repositories carry MIT code licenses. The official Hugging Face custom model
file carries an Apache-2.0 header. Speck may use the equations and write a clean-room reference while
preserving applicable notices if code is reused. Model weights have a separate DeepSeek model license;
no weights are needed, downloaded, executed, or authorized by this audit.

## Speck decision

The primary conventional/DeepSeekMoE equations and a single-device dropless training path are now
pinned. They do not select the released expert geometry or its unrenormalized routing for Speck. They
also do not close deterministic ties, train/inference parity, expert-parallel dispatch/backward,
resource, parent, or active-finalist blockers. No local conventional-MoE implementation or training is
authorized by this source audit.
