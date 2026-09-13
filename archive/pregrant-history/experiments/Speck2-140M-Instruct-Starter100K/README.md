# Speck2 starter-data SFT pilot

Completed pre-compute trial using the pinned Speck2-140M base, final-response-only supervision,
AdamW at 1e-4, 65,536-token optimizer batches, and a 4K ceiling. Native preparation retains 99,364
training and 1,991 validation conversations; final one-epoch checkpoint is step **1,554**.
See the [pilot outcome](../../research/notebook/2026-09-12-instruct-data-pilot.md) and
[consolidated result](../../results/Speck2-Instruct-Data-Pilot/summary.json).

## Reproduce

Build the [100K dataset](../Speck-Instruct-Starter/README.md), then run from the repository root:

```bash
export speck_base_dir=/mnt/speck-data/speck
uv run --no-sync python -m scripts.sft_prepare experiments/Speck2-140M-Instruct-Starter100K \
  --source-dir "$speck_base_dir/data/instruct-starter-100k"
UV_PROJECT_ENVIRONMENT=.venv-gpu uv sync --extra gpu --locked
UV_PROJECT_ENVIRONMENT=.venv-gpu WANDB_MODE=offline OMP_NUM_THREADS=8 uv run --no-sync \
  python -m scripts.sft_train experiments/Speck2-140M-Instruct-Starter100K
UV_PROJECT_ENVIRONMENT=.venv-gpu uv run --no-sync python -m scripts.sft_compare \
  experiments/Speck2-140M-Instruct-Starter100K --step 1554 \
  --output-dir "$speck_base_dir/evaluations/Speck2-Instruct-Starter100K"
```

The diagnostic contains 215 fixed scored cases and 12 qualitative prompts. Both models use native
BF16 inference, identical chat serialization, greedy decoding, and at most 128 output tokens.
`--prepare-only` writes cases before training; `--validation-data-dir` selects common held-out loss
when comparing different training corpora. Use fresh evaluation output directories for reruns.

The existing [BananaMind and Open SLM wrappers](../../docs/evaluation.md) provide raw-continuation
checks. BananaMind accepts this SFT directory directly. For Open SLM, export with
`scripts.model_publish --no-upload --expected-epochs 1`, then pass the parity-checked directory as
`--local-model` with this experiment's `open_slm.json`. That config's Hub model is the release baseline.

Variants: [100K, two epochs](../Speck2-140M-Instruct-Starter100K-2Epochs/README.md) and
[500K, one epoch](../Speck2-140M-Instruct-Starter500K/README.md). Evaluate final checkpoints; the
released recipe differs in source volume, target masking, sequence length, and exposure.
