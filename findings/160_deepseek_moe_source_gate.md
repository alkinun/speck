# 160 — Conventional dropless MoE has a single-device source, not an EP reference

## Source result

The official DeepSeekMoE report, 16B Hugging Face code, and later DeepSeek-V3 inference code are now
pinned by immutable revisions and hashes. Eleven files revalidated from their official network
locations.

DeepSeekMoE equations define full-softmax top-k routing, fine-grained expert segmentation, and shared
expert isolation. Selected routed weights retain their original full-softmax values rather than being
renormalized. The released 16B configuration uses 64 routed/six selected/two shared experts, but these
dimensions and routing choice are source facts, not Speck selections.

The 16B report explicitly uses no token dropping because every layer's experts are colocated on one
device. Its official custom model has a differentiable dropless path: repeat each token k times,
evaluate every selected assignment, weight/sum the results, add the shared path, and inject auxiliary-
balance gradients through an identity autograd function. The separate inference path groups by expert
and scatter-reduces token outputs.

## Boundary found

No upstream test proves parity between those train and inference paths. Top-k tie behavior is not
specified. The 16B code has no expert-parallel token dispatcher or distributed backward contract.
DeepSeek-V3's official code shards resident experts and all-reduces routed outputs, but its top-level
forward is inference-only and every rank receives the hidden input; this is not a released all-to-all
training implementation.

The conventional source gate is therefore narrower than “DeepSeek has MoE training.” Exact equations
and single-device-per-layer dropless semantics qualify; deterministic ties, Speck routing weights,
train/inference parity, expert-parallel dispatch/backward, resources, parents, and hardware do not.
Local implementation and training remain unauthorized while the active finalist is running.

## Artifacts

- [Official-source note](../papers/44_deepseek_moe_official_source.md)
- [Primary-source audit](../results/Speck-Paper1/deepseek-moe-primary-source-audit-v1.json)
