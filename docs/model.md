# Selected model and architectural context

The first release studies data and training on a fixed model. These notes support the
[program overview](program.md) and [paper](report.md): explain the attention and size choices,
their implementation, measured costs and limitations. Architectural novelty is not the paper's
central contribution. Broader design searches belong to later, separately budgeted releases.

## Selected flagship architecture

The selected **1.2B-total, all-active model** is defined in the
[qualification configuration](../experiments/qualification/model.json). Keep its geometry and
tokenizer fixed for data comparisons; qualify implementation changes independently.

- 1,195,884,576 total/active parameters; 24 layers, model width 2048.
- Three KDA recurrent blocks followed by one global GQA block, repeated six times.
- Dense SwiGLU feed-forward layers throughout, intermediate width 5120; no expert routing.
- Tied embeddings and the frozen Mistral tokenizer, with 32,003 embedding rows including role IDs.
- Start at 4K context; extend toward approximately 128K only through measured qualification.

Dense here describes the all-active feed-forward computation. The KDA/GQA token-mixing hybrid
remains the selected backbone. Its recurring layers and global layers are not an MoE mechanism.
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
discuss trade-offs. A single size cannot establish a scaling law; no architecture control is planned,
so the paper cannot attribute quality gains to the KDA/GQA ratio or superiority over another backbone.
Numerical/restart qualification supports implementation claims. Data controls support data claims.

Keep the full geometry and runtime settings reproducible, with detailed kernel/export material in
the appendix where appropriate. The [evaluation guide](evaluation.md) defines efficiency evidence;
the [program overview](program.md#model-and-runtime) owns runtime qualification and budget decisions.

## Later architecture research

MoE, attention residuals, attention-layout searches and model-size sweeps are future lab work as
compute and model sizes grow. No such comparison arm or implementation project is scheduled in the
first release. A measured blocker requiring a structural change must be recorded as a change to the
selected model, with its data-comparison implications addressed explicitly.

Existing runtime variants serve historical checkpoint/export compatibility and behavioral fixtures.
Their presence is not a research roadmap. Preserve the current data manifests, stage checkpoints,
learning curves and cost records so future architecture studies have an established training baseline.
