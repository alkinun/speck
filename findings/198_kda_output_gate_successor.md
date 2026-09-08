# 198 — KDA output gating is selectable without changing sigmoid checkpoints

## Successor boundary

This is a code successor to the KDA implementation recorded by [finding 11](11_kda_implementation_and_qualification.md)
and the conditional readiness boundary in [finding 155](155_recurrent_gate_conditional_readiness.md).
Their source hashes, claims, and checked CUDA result remain historical evidence and were not changed.
The predecessor architecture/model hashes remain
`47384b48bbb558729b2f9a0476b47d17e10b089076a9fb2bfb82d183110220e7` and
`72620f04bcd07ff6d6b4d2bd61908195ea208d80e12a65e5ce874e72f7319c27`.

Code successor `b8bb539d657cf6c4a37f409fb809fee85dbe5049` adds
`KimiDeltaAttentionSpec.output_gate_activation`. Sigmoid is the default and is omitted from canonical
serialization, so both a field-absent config and an explicit-sigmoid config normalize to the legacy
config shape. SiLU is explicit in serialized configs. The activation dispatch occurs after per-head
RMSNorm, downstream of the common Torch/FLA recurrence, and adds no parameters or state.

## Compatibility and verification

CPU tests prove exact default/explicit-sigmoid forward and backward identity, strict legacy state-dict
loading, both activations' finite backward passes and cached/full-forward parity, unchanged parameter/
state/FLOP geometry, config round trips, and native/Transformers export parity. The full CPU suite
passes with **832 passed, 10 skipped** when the dataset-build and RULER groups are installed. Ruff
lint and formatting pass on all changed Python files.

The CUDA integration cases cover both activations through FLA 0.5.0 chunk forward/backward and fused
recurrent decode. They are skipped on the CPU research host and still require execution during the
R7 GH200 stack qualification. The original checked FLA recurrence result at
[`results/KimiLinearTransfer/kda_kernel_qualification.json`](../results/KimiLinearTransfer/kda_kernel_qualification.json)
remains untouched; it qualifies the shared recurrence, not this new full-layer SiLU choice.
