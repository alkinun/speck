# Architecture for the long-context flagship

Selected contracts: [model_plan_v1.json](model_plan_v1.json) and
[architecture_plan_v3.json](architecture_plan_v3.json).

Keep the inherited 1.2B 24-block/2048-width, 3:1 KDA/global-GQA model with sigmoid output gating,
NoPE globals, SwiGLU, tied embeddings and BF16/Muon. The frozen Mistral tokenizer is selected.
The old independent position, ratio, decay-operator and output-gate sweeps are retired.

## What is compared

The [study](STUDY.md) compares dense global GQA against the complete hybrid package under two
supervision treatments. Match depth, residual width, FFN, tokenizer, attention geometry, paired seeds,
document order and optimization controls, and disclose residual parameter/FLOP differences. A credible
bounded LR calibration for each architecture precedes the common freeze. This is no isolated KDA/NoPE
or ratio claim. The dense model is a control, not a second full-budget flagship.

## Hardware first

R0 qualifies actual 1.2B shapes at 4K, 32K and 128K with forward/backward, optimizer, saving/resume,
FLA recurrent/chunk parity, cached versus uncached behavior, four-GPU DDP and storage. DDP does not
pool four GPUs' memory to fit one long example. A failed 128K fit selects a lower-length path; no new
context-parallel implementation is implicitly funded. A broken baseline training path blocks its run.

Keep the existing CPU reference for correctness. Native/Transformers plus one accelerated GH200 serving
path are release requirements. RTX 3090 is a secondary reference when qualified; CPU/GGUF is optional
because the recurrent export/runtime is incomplete.

## Efficiency boundaries

KDA carries fixed state; the complete hybrid still carries a length-growing exact-attention cache.
Its global-attention prefill remains quadratic in context length. Fewer global layers reduce a
coefficient, not the asymptotic order of that component. Measure state, weights, workspace and peak
allocation separately. Report training, prefill, decode and complete task cost, including output tokens.

The inherited short proxy's lower FLOPs and faster time-to-quality motivate the study; they do not
prove the new data intervention, usable long context, or full-horizon flagship advantage. The Reader
Attention failures remain useful evidence about stale representations. See [prior findings](../../archive/pregrant-history/findings/README.md).

New operators, MoE, depth routing, compressed/sparse attention, multimodality and mandatory quantized
cache research stay outside this allocation. A correctness-driven change requires a pre-results
successor and re-cost, not an unrecorded alteration of the control.
