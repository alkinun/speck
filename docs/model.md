# Reference model and bounded architecture study

The first release studies data and training, supported by a bounded architecture/efficiency study.
Freeze the chosen backbone before data experiments. These notes support the
[program overview](program.md) and [paper](report.md): explain the attention and size choices,
their implementation, measured costs and limitations. Architectural novelty is not the paper's
central contribution. Broader design searches belong to later, separately budgeted releases.

## Reference architecture

The reference **1.2B-total, all-active model** is defined in the
[qualification configuration](../experiments/qualification/model.json). Freeze the selected geometry
and tokenizer for data comparisons after the architecture decision; qualify implementation changes
independently and preserve this historical reference configuration.

- 1,195,884,576 total/active parameters; 24 layers, model width 2048.
- Three KDA recurrent blocks followed by one global GQA block, repeated six times.
- Dense SwiGLU feed-forward layers throughout, intermediate width 5120; no expert routing.
- Tied embeddings and the frozen Mistral tokenizer, with 32,003 embedding rows including role IDs.
- Start at 4K; qualify 16K then 32K. Approximately 128K is a stretch within an explicit cost revision.

Dense here describes the all-active feed-forward computation. The KDA/GQA token-mixing hybrid
is the reference backbone for the study. Its recurring layers and global layers are not an MoE mechanism.
KDA is inherited from [Kimi Linear](https://arxiv.org/abs/2510.26692v2); distinguish inherited building
blocks from Speck's configuration, implementation and evidence. No attention ratio or 128K capability
is proven by the engineering pilot. Global layers still retain length-growing caches and quadratic
attention work. Architectural novelty or superiority requires supporting evidence.

## What the paper should explore

Explain the rationale for the 1.2B size under the available training/inference budget, the recurring
KDA/global-attention pattern, NoPE global layers, dense feed-forward computation and tokenization.
Report actual parameter counts, attention/cache behavior, supported context and precision. These
are design choices to document, not experimentally established optima.

Use measured training throughput, memory, prefill/decode cost and length-dependent behavior to
discuss trade-offs. The 200-hour study compares the hybrid with one matched attention baseline near
1.2B; freeze exact geometry, positional policy, corpus, horizon and endpoints before execution.
Report actual parameter counts and FLOPs rather than assuming equal parameters imply equal cost.
Measure training cost/memory and prefill/decode latency, throughput and cache memory over declared
lengths and batch sizes. Separate implementation tuning from architectural effects. A single size
cannot establish a scaling law, and superiority requires matched evidence.
Numerical/restart qualification supports implementation claims. Data controls support data claims.

Keep the full geometry and runtime settings reproducible, with detailed kernel/export material in
the appendix where appropriate. The [evaluation guide](evaluation.md) defines efficiency evidence;
the [program overview](program.md#model-and-runtime) owns runtime qualification and budget decisions.

## Later architecture research

MoE, attention residuals, broad attention-layout searches and model-size sweeps remain future work.
The first release permits only the bounded reference/control study described above. A measured blocker requiring a structural change must be recorded as a change to the
selected model, with its data-comparison implications addressed explicitly.

Existing runtime variants serve historical checkpoint/export compatibility and behavioral fixtures.
Their presence is not a research roadmap. Preserve the current data manifests, stage checkpoints,
learning curves and cost records so future architecture studies have an established training baseline.
