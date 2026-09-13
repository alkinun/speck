# 155 — Sigmoid gate evidence is conditional and KDA-SiLU is unimplemented

## Evidence boundary

The clean seed-42 GDN/RoPE result changes exactly fifteen recurrent gate activations from SiLU to
sigmoid and improves language loss by 0.037337 nats at equal parameters/FLOPs. It is one seed. Later
sigmoid-arm replications show stability, not a replicated treatment effect.

Synthetic MQAR reverses the GDN result: SiLU passes 3/3 and sigmoid 0/3, while KDA/sigmoid passes 3/3.
The missing KDA/SiLU cell is not currently expressible—KDA's spec has no activation field and its
forward path hardcodes sigmoid.

## Conditional design

Gate choice therefore waits for an independently selected mixer and positional parent. On that exact
parent, six seed×data-order cells compare only SiLU and sigmoid. An existing sigmoid arm may be reused
only if every parent identity and acceptance gate matches; otherwise both arms are new. No result may
generalize gate choice to another mixer, position, or task.

If KDA is selected, a post-finalist code successor must add a default-sigmoid field and prove field-
absent/explicit-sigmoid config normalization, strict checkpoint loading, sigmoid numerical/export/
geometry identity, and SiLU Torch/FLA correctness. The active model code is not changed mid-sequence.

## Artifact

- [Conditional gate qualification](../results/Speck-Paper1/recurrent-gate-v1-qualified.json)
