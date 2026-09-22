# Reference model and deferred architecture study

The first release studies data and training across all six stages. The backbone below is the
**fixed substrate for that work, frozen by declaration rather than selected over a control**: the
bounded architecture/efficiency comparison was deferred to a later allocation on 2026-09-21 and its
200 hours were released. The design in [proposed control and decision](#proposed-control-and-decision)
is preserved intact for that allocation and is **not run here**, so this release supports no
architecture superiority or parity claim. These notes support the
[program overview](program.md) and [paper](report.md): explain the attention and size choices,
their implementation, reference-only measured costs and limitations. Architectural novelty is not
the paper's contribution at all in this release. Broader design searches belong to later,
separately budgeted releases.

## Reference architecture

The reference **1.2B-total, all-active model** is defined in the
[qualification configuration](../experiments/qualification/model.json). The geometry and tokenizer
are already frozen for the data comparisons — by declaration, with no architecture decision pending
on this allocation. Qualify implementation changes independently and preserve this reference
configuration.

- 1,195,884,576 total/active parameters; 24 layers, model width 2048.
- Three KDA recurrent blocks followed by one global GQA block, repeated six times.
- Dense SwiGLU feed-forward layers throughout, intermediate width 5120; no expert routing.
- Tied embeddings and the frozen Mistral tokenizer, with 32,003 embedding rows including role IDs.
- Start at 4K; qualify 16K then 32K. Defer 64K/128K until a later explicit cost and quality revision.

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
discuss trade-offs. The comparison against a matched attention baseline near 1.2B is **deferred to
a later allocation**; its design is preserved unmodified in the
[packet](../experiments/main-data/architecture-study-packet.json). On this allocation the hybrid is
the declared substrate, profiled on its own, so no comparative claim is available. When the study
is eventually funded, freeze exact geometry, positional policy, corpus, horizon and endpoints
before execution.
Report actual parameter counts and FLOPs rather than assuming equal parameters imply equal cost.
Measure training cost/memory and prefill/decode latency, throughput and cache memory over declared
lengths and batch sizes. Separate implementation tuning from architectural effects. A single size
cannot establish a scaling law, and superiority requires matched evidence.
Numerical/restart qualification supports implementation claims. Data controls support data claims.

Keep the full geometry and runtime settings reproducible, with detailed kernel/export material in
the appendix where appropriate. The [evaluation guide](evaluation.md) defines efficiency evidence;
the [program overview](program.md#model-and-runtime) owns runtime qualification and budget decisions.

## Proposed control and decision

The proposed control uses 24 global GQA layers, width 2048, 16 query heads / four KV heads,
head dimension 128, full RoPE (theta 1,000,000), and SwiGLU width 5888. Keep tied 32,003-row
embeddings, the tokenizer, normalization and initialization policy. A meta-device construction using
the current model code counts **1,185,527,808 all-active parameters**, 0.866% below the reference.
Choose the closest parameter match on a 256-wide feed-forward grid; this is arithmetic, not a
quality-based geometry sweep. The control remains proposed and has not run or qualified on a GPU.

This is a comparison of complete backbone designs: the wider feed-forward layers and positional
policy are explicit differences. It cannot isolate KDA, NoPE or feed-forward width. Equal parameters
also do not equalize FLOPs. Use the same eligible corpus/order, token horizon, optimizer policy,
precision, update size and schedule; disclose shape-dependent initialization/optimizer differences.
The common seed does not make differently shaped model tensors identical.

The [architecture study packet](../experiments/main-data/architecture-study-packet.json) and its
[offline validator](../experiments/main-data/check_architecture_study.py) preserve the proposed 200h
envelope **for a later allocation**; it draws zero hours from this one. It assigns 120h to one seed
pair (60h per arm), 40h to training/inference profiling and 40h to qualification, evaluation and
recovery. When that allocation funds it, select a common token horizon from
measured cost before running either arm. Measure fixed-pack loss and development regressions, then
training throughput/memory and prefill/decode at batch sizes one/eight, prefix lengths
512/2048/3840 and a 256-token output cap within 4K. Freeze timing repetitions, warmup and environment;
longer synthetic timings establish cost only, not learned long-context ability.

Before results, declare the acceptable quality regression and which deployment costs matter.
Choose a qualified, affordable design that passes those gates; if trade-offs are inconclusive and
the reference passes, retain it. If neither passes, stop and document a scope revision. One seed
supports a bounded design decision, not general architectural superiority. Remeasure the selected
backbone for data-study costing; preserve the original H100 configurations and receipts. The
[numeric plan](../experiments/main-data/plan.json) owns the proposed geometry/caps, and the
[research design](../experiments/main-data/README.md#measurements-and-selection) defines common
measurement, failure and claim rules.

## Later architecture research

MoE, attention residuals, broad attention-layout searches and model-size sweeps remain future work,
alongside the deferred reference/control comparison. **This release runs no architecture study at
all**, so it can support no superiority or parity claim. A measured blocker requiring a structural
change must be recorded as a change to the selected model, with its data-comparison implications
addressed explicitly.

Existing runtime variants serve historical checkpoint/export compatibility and behavioral fixtures.
Their presence is not a research roadmap. Preserve the current data manifests, stage checkpoints,
learning curves and cost records so future architecture studies have an established training baseline.
