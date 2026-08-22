# Architecture Search

Speck's search command runs a resumable, deterministic architecture study. It compares candidate quality, latency, and memory under one fixed data and runtime contract.

## Prerequisites

Install the GPU environment and prepare the tokenizer and packed data used by the baseline experiment:

```bash
uv sync --extra gpu
python -m scripts.tokenizer_prepare experiments/Speck1-140M
python -m scripts.data_prepare experiments/Speck1-140M
```

The checked-in search profile requires CUDA. Ensure the configured packed dataset is complete, its manifest and shards are readable, and the study filesystem has enough space for a generation's checkpoints. Search does not use W&B.

## Run a Study

The public entry point is:

```bash
python -m scripts.search run experiments/Speck1-140M --name <name>
```

A new study also requires at least one cumulative limit:

```bash
python -m scripts.search run experiments/Speck1-140M \
  --name evolution-01 \
  --hours 3 \
  --generations 2 \
  --device cuda
```

`--hours`, `--generations`, and `--device` are optional for an existing study. The device defaults to `cuda`. For a new study, supply `--hours`, `--generations`, or both. A later run may increase a cumulative limit but may not decrease it.

Study names must be one path component. Studies are stored at `~/.cache/speck/search/<name>`, or under `$speck_base_dir/search/<name>` when the cache root is overridden.

## Provenance and Runtime Contract

Study creation records comparison-sensitive inputs and rejects later changes. The immutable contract includes:

- The absolute experiment path and copied `search.json` settings.
- The model, tokenizer, and data configurations.
- The tokenizer fingerprint, vocabulary size, and special-token IDs.
- The packed-data path and manifest fingerprint.
- The device identifier, device type and name, PyTorch and CUDA versions, parameter and compute dtypes, deterministic-algorithm setting, and cuBLAS workspace configuration.

Every run verifies the current configuration and data against this provenance. Packed shards are checksum-verified when the coordinator starts. Resume fails if the experiment, search settings, tokenizer, dataset manifest, data path, or runtime contract has changed.

The study uses deterministic seeds, deterministic PyTorch algorithms, fixed validation slices, and fixed batch order. Candidate IDs break score ties deterministically. Hardware timing can still contain normal measurement noise, so comparisons are only valid within the recorded profile contract.

## Resuming and Ownership

Run the same command with the same name to continue from saved state:

```bash
python -m scripts.search run experiments/Speck1-140M --name evolution-01
```

Candidate checkpoints include model, optimizer, training metadata, and loader position. Interrupted candidate training resumes from its retained checkpoint. Study state and candidate results are written atomically, and elapsed time is cumulative across coordinator runs.

Each study has a nonblocking file lock. Only one coordinator or finalizer may own a study at a time; a second process exits with `study already has a running coordinator`. The lock provides single-host filesystem ownership and is not a distributed worker queue.

## Status

Show a concise status report:

```bash
python -m scripts.search status evolution-01
```

Use `--json` for the full machine-readable snapshot:

```bash
python -m scripts.search status evolution-01 --json
```

Status includes the lifecycle phase, cumulative elapsed time, generation, current candidate, status and rung counts, lane leaders, and retained checkpoint bytes.

## Generations and Promotions

Generation zero contains the normalized baseline, configured controlled mutations, and random candidates. Later generations mutate selected archive parents and add configured random immigrants. Architectures must remain within the parameter, logical-depth, and operation bounds in `search.json`.

Each generation advances through three training rungs:

1. `screen` trains feasible candidates to the first rung and promotes four across quality, balanced, and efficiency lanes.
2. `develop` trains those candidates to the second rung and promotes three.
3. `confirm` trains the remaining candidates to the third rung and adds them to the confirmed archive.

Quality combines current validation NLL and a deterministic learning-curve projection. Efficiency ranks measured prefill and decode latency, parameter bytes, sequence-state bytes, and peak VRAM. Balanced scoring averages quality and efficiency ranks. Failed, non-finite, or out-of-memory candidates are recorded and excluded from promotion.

After confirmation, the archive is rescored at common rungs. Checkpoints are retained for the top two confirmed candidates in each lane, with duplicates removed; other search checkpoints are pruned. A missing retained checkpoint can be rebuilt deterministically when needed.

## Candidate Checks and Profiling

Before training, each candidate must pass a forward and backward feasibility check with finite loss and gradients. Full-sequence logits must also match token-by-token cached decoding within the configured tolerances.

Each feasible candidate is profiled before rung training. The profile records static parameter, FLOP, and sequence-state estimates; eager prefill and decode latency distributions; and CUDA memory use. These measurements participate in efficiency and balanced scoring.

## Finalize

Finalize after the study has confirmed candidates and lane leaders:

```bash
python -m scripts.search finalize evolution-01
```

`--device` is available and defaults to `cuda`; it must satisfy the study's CUDA runtime contract.

Finalization performs the following work for the distinct quality, balanced, and efficiency leaders selected from the confirmed archive:

1. Ensures each leader has its final search-rung checkpoint, rebuilding it if necessary.
2. Trains a continuation from that checkpoint to `final_tokens`.
3. Trains an independent run from initialization to `final_tokens` with the configured seed offset.
4. Evaluates both runs on the monitor slice and the separate final validation slice.
5. Profiles the continuation checkpoint on GPU in eager and compiled modes, verifies compiled output equivalence, and profiles it on CPU under the recorded host contract.
6. Recomputes quality, efficiency, and balanced roles among the verified candidates.

The report records both the pre-finalization and verified role assignments, including any role changes. Finalization writes `finalists.json` and changes the study status and phase to `finalized`. It does not copy finalists into `experiments/` or upload checkpoints.

Finalization is resumable at the run and profile level: completed final runs and compatible profiles are reused. A CPU profile is repeated when its host contract differs.

## Output Files

The useful study files are:

```text
search.json                              Immutable copied search settings.
state.json                               Lifecycle, limits, provenance, and current work.
candidates/<id>/architecture.json        Materialized candidate architecture.
candidates/<id>/result.json              Rungs, NLL curve, profiles, scores, and errors.
candidates/<id>/checkpoint/              Retained search checkpoint, when selected.
candidates/<id>/final/continuation/       Continued final run and checkpoint.
candidates/<id>/final/independent/        Independent final run and checkpoint.
candidates/<id>/final/profile.json        Final eager, compiled, and CPU profiles.
finalists.json                            Verified finalist report and role assignments.
```

The underscore-prefixed subcommands, including `_check`, `_profile`, `_train`, `_rebuild`, and `_final_*`, are coordinator worker internals. Do not invoke them directly.
