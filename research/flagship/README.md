# Flagship charter

Build a small model whose data and memory-allocation choices are supported by controlled experiments,
and release the model together with its evidence. The paper asks whether high-information data and
periodic exact attention improve quality per training compute and resident state, and whether those
gains transfer and compose in a held-out flagship.

## Model

- **Size:** 1.2B dense-width parameters; 400B-token target and 320B throughput fallback.
- **Backbone:** 24 blocks, hidden width 2048, 3:1 KDA/global GQA, quantile global positions.
- **Defaults:** sigmoid KDA output gate, NoPE global attention, SwiGLU, physically tied embeddings/head.
- **Training:** bf16, Muon matrices plus AdamW elsewhere, WSD schedule, approximately 1M-token batches.
- **Context:** 4K base training, followed by coherent-document 32K and 128K continuation.
- **Tokenizer:** D5 remains open; Mistral 32K is the fallback.
- **Releases:** pre-decay, base, extended, and Instruct checkpoints with measured quality and systems cost.

Defaults change only through the named comparisons in the selected protocols. MoE, sparse/compressed
attention, depth routing, and new operators are outside this allocation.

## Protocols

| Area | Maintained explanation |
| --- | --- |
| Data treatments, mixture, repetition, decay | [DATA.md](DATA.md) |
| Architecture, shared controls, transfer, systems | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Tokenizer decision | [TOKENIZER.md](TOKENIZER.md) |
| Budget and dependency order | [EXECUTION.md](EXECUTION.md) |
| Long-document continuation | [CONTEXT_EXTENSION.md](CONTEXT_EXTENSION.md) |
| Instruct development | [POST_TRAINING.md](POST_TRAINING.md) |
| Paper evidence standard | [PAPER.md](PAPER.md) |

The [catalog](../catalog.json) selects exact contract versions. Use the
[status record](../status.json) for readiness and next actions, and the
[archive](../../archive/README.md) for predecessor plans and completed research.
