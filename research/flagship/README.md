# Flagship: efficient long-context intelligence

**Selected direction, 2026-09-15.** Build a general-purpose model that understands and reasons across
supplied documents and histories efficiently. This replaces the allocation-thesis experiment program;
see [PIVOT.md](PIVOT.md) for preserved evidence, retired work and migration.

## Model

- 1.2B-class dense-width, 24 blocks, hidden width 2048, inherited 3:1 KDA/global GQA.
- Sigmoid KDA output gate, NoPE global attention, SwiGLU, physically tied embeddings/head.
- Frozen Mistral 32K tokenizer; no new tokenizer search or D5 opening.
- BF16, Muon matrices plus AdamW elsewhere, WSD. Freeze LR/batch after hardware calibration.
- 320B broad-base token target; 400B only when the 2,425-hour base envelope and calendar support it.
- 4K base training, useful 32K milestone, 64K evaluation and 128K target.
- Broad general assistance remains a release requirement. Response mode is a measured development
  choice; thinking-only is not selected.

[model_plan_v1.json](model_plan_v1.json) records model scope. Exact parameter count and tensors are
revalidated from the retained geometry and frozen tokenizer before launch.

## Science and release

Cross dense/hybrid architecture with standard/dependency-requiring supervision at three paired seeds,
then perform one preselected scale or horizon transfer check. Develop the flagship and measure its
quality-cost frontier against public models and retrieval-based alternatives. No scaling-law or
full-horizon dense-comparison claim is implied by one flagship.

Primary demonstrations: document/evidence understanding and conversation-history understanding.
Supporting evaluation: technical/code comprehension and short/general capability. Release checkpoints,
qualified native/Transformers inference, one accelerated serving path, reproducible evaluation and a
paper. CPU/GGUF is optional until the recurrent path is implemented and qualified.

| Area | Contract |
| --- | --- |
| Budget and dependencies | [EXECUTION.md](EXECUTION.md), [plan_v4.json](plan_v4.json) |
| Controlled experiment | [STUDY.md](STUDY.md), [long_context_study_v1.json](long_context_study_v1.json) |
| Architecture and systems | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Data | [LONG_CONTEXT_DATA.md](LONG_CONTEXT_DATA.md), [data_plan_v3.json](data_plan_v3.json) |
| Context/instruction stages | [CONTEXT_EXTENSION.md](CONTEXT_EXTENSION.md), [POST_TRAINING.md](POST_TRAINING.md) |
| Evaluation and comparators | [EVALUATION.md](EVALUATION.md) |
| Paper claims | [PAPER.md](PAPER.md) |

The [catalog](../catalog.json) selects contracts; [status](../status.json) records readiness. No new
training, measurement, source acquisition or release is performed by this planning transition.
