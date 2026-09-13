# Evaluation and benchmarking

Evaluation implementations live under `speck.evaluation`. Each report should identify the checkpoint,
data, scoring contract, producing revision, and hardware. Current outputs live in `results/` or the
runtime store; completed measurements remain in the [archive](../archive/pregrant-history/results/README.md).

## Held-out checkpoint loss

```bash
uv run --no-sync python -m scripts.checkpoint_loss_eval PATH_TO_EXPERIMENT \
  --checkpoint-dir CHECKPOINT_DIRECTORY --eval-tokens 65536 --no-compile
```

Use `--data-experiment` for a separately configured corpus with the same tokenizer. The installed
`speck evaluate` command is equivalent. Parser-independent held-out construction and analysis use
`scripts.heldout_evaluation_build` and `scripts.heldout_evaluation_analyze` with their selected contracts.

## Context and capability

| Command | Purpose |
| --- | --- |
| `scripts.long_context_eval` | Exact-length passkey and systems diagnostics |
| `scripts.position_loss_eval` | Position-binned and trailing-token loss |
| `scripts.structured_retrieval_eval` | Controlled retrieval and composition |
| `scripts.ruler_source_prepare`, `scripts.ruler_case_prepare` | Pinned offline RULER sources and cases |
| `scripts.evaluation_server` | Local OpenAI-compatible evaluation endpoint |
| `scripts.logprob_parity` | Offline per-token cross-backend parity |

Each command exposes `--help`. An experiment used by `long_context_eval` must supply its
`long_context.json`. [Long-context tooling](long_context.md) describes the capability boundaries.
HELMET/NoLiMa preparation belongs to the historical program and is indexed in the archive.

## Quality and serving

`scripts.open_slm_eval` uses pinned benchmark tooling; install the `open-slm` dependency group in its
own environment when needed. `scripts.instruct_eval` and `scripts.sft_compare` provide local Instruct
diagnostics. These small diagnostic sets do not replace a full held-out benchmark suite.

Use `scripts.benchmark` for optimization-step cost and `scripts.inference_benchmark` for prefill/decode
measurements. Report hardware, precision, batch, sequence length, startup/compile overhead, and steady
timing separately. Compare native and exported logits before interpreting backend speed differences.

Historical score tables and detailed release-specific commands are in the
[original evaluation guide](../archive/pregrant-history/docs/evaluation.md).
