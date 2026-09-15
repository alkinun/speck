# R0 readiness before compute access

The application remains under review according to the response supplied by the owner on
2026-09-15. The team said no further action was needed from the applicant. This is not an
allocation award or an access date. Calendar targets in plan_v4 begin from actual allocation
availability; local preparation continues while the review is pending.

## Exact shapes prepared

The [bound shape plan](r0_shape_preparation_v1.json) and
[checked result](../../results/systems/r0-shape-preparation-20260915.json) expand the retained
24-layer, width-2048 geometry under the selected model, architecture, tokenizer and R0 allocation.
All six cases were constructed with the actual model implementation on PyTorch's meta device,
which records tensor shapes without allocating the model's tensor storage.

| Architecture | Positions checked | Instantiated parameters | Recurrent / global layers |
| --- | --- | ---: | --- |
| Selected hybrid | 4,096 / 32,768 / 131,072 | 1,195,884,576 | 18 / 6 |
| Matched dense control | 4,096 / 32,768 / 131,072 | 1,072,281,600 | 0 / 24 |

Dense replaces recurrent positions with global GQA while preserving width, depth, FFN, global
head geometry and NoPE. It is not an equal-parameter comparator. These dense configurations
support implementation qualification; they do not add a full-budget dense flagship experiment.
Both models have independent layer weights and one physically tied embedding/head parameter.
Exact optimizer membership covers every parameter once, with the shared embedding in AdamW
without weight decay. The optimizer state totals are estimates under explicit dtype assumptions.

The model reserves 32,003 vocabulary rows while the frozen Mistral tokenizer emits 32,000 IDs.
The three reserved rows remain unused as synthetic inputs. The [bounded executor](R0_EXECUTOR.md)
preserves this distinction rather than rebuilding the model with the base vocabulary and changing
its parameter count. The generic benchmark's tokenizer-derived model construction is not, by itself,
a qualified executor for these effective-vocabulary shapes.

The result includes analytic training FLOPs and batch-one inference state geometry. Recurrent
matrices remain FP32; the reported KV/convolution state uses BF16. These are neither measured
peak training memory nor evidence of GPU fit, useful context or speedup. Every GPU fit,
throughput, backward, resume and DDP result is explicitly null.

Generation and exact result reopen passed at implementation revision `791e30e`. Reproduction
requires that revision and the bound original input paths; a changed implementation/contract
must publish a successor result. The result embeds all six complete model configurations.

```bash
uv run --no-sync python -m scripts.prepare_r0_shapes \
  research/flagship/r0_shape_preparation_v1.json /absolute/path/to/new-result.json
```

Software validation passed: **1,219 tests**, 10 skipped and 125 deselected, plus formatting,
lint, research catalog, archive and source-pin checks. The ten new tests check independent
parameter/state equations, unchanged control geometry, optimizer coverage, physical ties,
reserved vocabulary, contract drift and refusal to publish changed input identities.

## Remaining work before R0 execution

The [bounded synthetic executor](R0_EXECUTOR.md) now binds the exact shapes, deterministic per-rank
inputs, precision/backend/optimizer settings, warmup/measured steps, deadlines and conservative
budget reservations. Its [fresh-process CPU qualification](../../results/systems/r0-fresh-process-local-qualification-20260915.json)
checks dense/KDA persisted checkpoint/RNG replay in new workers, two-rank Gloo, failures and rank
cleanup. It has not executed the actual GPU cases. CUDA restart, cached-generation/reference parity,
hard interruption, scheduler integration and production-data throughput still require qualification.

At the allocated site, qualify arm64 dependencies and KDA kernels, forward/backward and the
optimizer, cached-generation parity, checkpoint/resume and four-GPU behavior. Record startup,
compilation, checkpoint/storage overhead and actual end-to-end throughput against the R0
70-GPU-hour ceiling. Freeze site-specific scheduler and recovery commands once they are known.
128K failure follows the selected length fallback; it does not silently add a new trainer.

In parallel, prioritize measured source-capacity/exposure planning, coherent document/history
identities, family-separated task pilots and pinned evaluation exclusions. These inputs prepare
R1/R2; finishing old acquisition quotas does not authorize the retired experiment matrix.
No model training or new download was launched by the shape preparation.
