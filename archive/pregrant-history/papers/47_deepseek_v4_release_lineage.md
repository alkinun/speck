# DeepSeek-V4 initial-to-current sequence-code lineage

## Compared revisions

- Initial report-aligned release:
  [`efc855127ecba8ece36817f9e4cdeeae03b10200`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/efc855127ecba8ece36817f9e4cdeeae03b10200)
- Current official release at audit time:
  [`60d8d70770c6776ff598c94bb586a859a38244f1`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1)

Eight small source/config/license files were compared byte-for-byte at both revisions. The technical
report itself remains pinned from the initial revision; it was removed from the current repository tree.

## Stable sequence architecture

The complete `Compressor` through `Attention` source segment is byte-identical: 14,015 bytes with
SHA-256 `1c5d1900c08af592fa541ee9085eccae3c28436e82ffe4201d1304475588168b`.
This covers HCA/CSA pooling, overlap state, the Lightning Indexer, causal compressed-entry masks, raw
window indices, concatenated KV identities, query/KV normalization, partial RoPE, and grouped output.

The complete sparse-attention kernel segment is also byte-identical: 3,439 bytes with SHA-256
`de29320545180b4028a15c676594866a1c01be9e785a95bd30bc4907bb3abea8`.
The sequence factorization and causal-state corrections therefore remain valid.

## Material precision correction

The generic in-place activation quantizer changed. Its initial 2,771-byte segment has SHA-256
`815f83d02b649e952901d4d49e0c08a3460bf1aff840de5c62ab48cacdc42951`; the current
2,765-byte segment has SHA-256
`deeeef06479551a90c559dd17c94d86eef26a51e5b78bb502e9aed296212d568`.

Both versions first replace `out_dtype` with the input dtype for in-place quantize/dequantize. The
initial kernel then also casts normalized values through that same output dtype before rescaling. For a
BF16 input this performs BF16 rounding, not the intended FP8 rounding. The current kernel explicitly
casts the normalized value through FP8 before converting back and multiplying by its scale.

This path is called in-place for the non-RoPE dimensions of both exact raw-window KV and main HCA/CSA
compressed KV. It is not used by the separately defined FP4 Lightning-Indexer query/key simulation.
Thus architecture, selection, causal state, and union-softmax semantics are stable, while the initial
release's low-precision numerical behavior is superseded.

## Unrelated changes

The root config adds `expert_dtype="fp4"`, aligning it with the already unchanged inference config.
The model adds SwiGLU clamping to the shared MoE expert. Neither change touches sequence attention.
Generation code, inference config/readme, and MIT license remain byte-identical. The root README changed
for release documentation.

## Speck consequence

The initial report remains the primary equation source and the shared architecture/state segment remains
qualified. Any future precision reference must use the current explicit FP8 cast as the intended
quantize/dequantize source, while treating its actual behavioral and hardware parity as unqualified until
locally tested. Full-precision HCA/CSA/local correctness must precede every precision intervention.

No current or initial upstream code is executed, no model weight is accessed, and no active experiment
or Speck implementation is changed by this lineage audit.
