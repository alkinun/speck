# Releasing Models

These workflows are for maintainers publishing Speck artifacts. They default to repositories in the
`specklabs` Hugging Face organization and can create remote commits. Authenticate with a token that
has write access and review every source revision, destination, and generated artifact before
uploading.

Run local validation with `--no-upload` first.

## Transformers Export

Export and validate the canonical one-epoch instruction checkpoint as a BF16 Transformers
repository without uploading:

```bash
uv run --extra cpu --group transformers python -m scripts.model_publish \
  --expected-epochs 1 \
  --no-upload
```

The command defaults to the latest completed checkpoint under
`~/.cache/speck/checkpoints/Speck1.1-140M-Instruct` and writes a generated release under
`~/.cache/speck/releases`. Use `--checkpoint-dir`, `--step`, `--repo`, and `--output-dir` to make
the source and destination explicit. Omit `--no-upload` only after reviewing the local export.

Published likelihood evaluation supports binary right-padded batches when `use_cache=False`. Left
padding, mask gaps, and cached padded inference remain unsupported.

## Code-Only Compatibility Update

Apply and validate the tracked padding compatibility code against an immutable base-model source
without changing weights:

```bash
uv run --extra cpu --group transformers python -m scripts.model_code_publish --no-upload
```

The publisher verifies the source revision, model-weight LFS checksum, Auto class loading,
parameter count, padded-batch logit parity, generated code hashes, and unchanged weights. Omit
`--no-upload` only after local validation succeeds and the configured source revision still matches
the intended remote parent.

## GGUF Variants

GGUF publication requires Git, CMake, a working C/C++ toolchain, network access, and enough local
space for BF16 plus every requested quantization. Build and smoke-test locally first:

```bash
uv run --extra cpu python -m scripts.gguf_publish --no-upload
```

The default workflow creates BF16, Q4_K_M, Q5_K_M, and Q8_0 variants from the public instruction
model, builds a pinned llama.cpp revision, and smoke-tests every artifact with llama.cpp. Generated
weights and the checkout stay under `~/.cache/speck`.

Repeat `--quantization <type>` to select variants, use `--llama-cpp <path>` for an existing checkout,
and set `--jobs` to control build and inference concurrency. `--resume` validates and reuses existing
artifacts after an interruption. Omit `--no-upload` only when all requested files pass.

## Release Safety

- Prefer explicit immutable source revisions over `main`.
- Preserve generated manifests and command output with the release record.
- Never use `--force` until the target path has been checked manually.
- Confirm destination repositories before removing `--no-upload`.
- Treat model-card-only migration scripts as one-shot operations tied to their pinned parent
  commits; verify current Hub heads before attempting them.
