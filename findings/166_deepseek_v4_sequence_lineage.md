# 166 — V4 architecture is stable, but initial FP8 simulation is superseded

The report-aligned initial DeepSeek-V4-Flash release and current official release were compared across
eight source/config/license files each. The HCA/CSA compressor-indexer-attention segment is byte-
identical, as is the sparse attention kernel. The qualified sequence factorization and dependency DAG
therefore remain valid.

One kernel change is materially numerical. In the initial in-place activation quantizer, `out_dtype`
is first rebound to the input dtype and then reused for the normalized-value rounding cast. For BF16
inputs, that performs BF16 rather than intended FP8 rounding. Current code explicitly casts through
FP8 before converting back and rescaling.

The affected call sites are the non-RoPE dimensions of both raw sliding-window KV and main HCA/CSA
compressed KV. Lightning-Indexer FP4 simulation uses a separate function and is unchanged. The current
root config also adds FP4 expert dtype, and shared-expert clamping was fixed; both are width/runtime
changes outside sequence attention.

The initial report retains equation authority and its architecture/causal-state implementation remains
the pinned semantic source. Its low-precision numerical behavior does not. A future precision-specific
successor must use the current FP8 cast, but full-precision operator correctness comes first and no
low-precision behavior is locally qualified yet.

The current repository removed the report file, so the immutable initial report pin remains necessary
for provenance. No upstream code or weights were executed, and no active experiment, DAG, implementation,
training, or promotion authority changed.

## Artifacts

- [Lineage note](../papers/47_deepseek_v4_release_lineage.md)
- [Lineage audit](../results/Speck-Paper1/deepseek-v4-sequence-lineage-v1.json)
