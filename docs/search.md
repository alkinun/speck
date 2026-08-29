# Architecture Search

Speck's architecture search compares model structures under a fixed training, evaluation, and
profiling contract. A study is resumable, but its experiment inputs and comparison-sensitive
settings are immutable after the first run.

## Prerequisites

- Run commands from the repository root on Linux or another platform that provides `fcntl`.
- Install the CUDA environment with `uv sync --extra gpu`.
- Prepare the experiment's tokenizer and packed dataset before starting a study.
- Use a CUDA device compatible with the profile contract in the experiment's `search.json`.
- Reserve enough time and disk space for candidate checkpoints under `~/.cache/speck/search`.

The checked `experiments/Speck1-140M/search.json` contract uses deterministic CUDA execution,
BF16 computation, 2,048-token sequences, and three successive training rungs. Search verifies the
packed-data manifest and every shard before doing candidate work.

Prepare the baseline inputs:

```bash
uv run --extra gpu python -m scripts.tokenizer_prepare experiments/Speck1-140M
uv run --extra gpu python -m scripts.data_prepare experiments/Speck1-140M
```

Data preparation is the expensive step. See the repository README for its storage requirements
and resume behavior.

## Start or Resume

A new study needs a time limit, a generation limit, or both:

```bash
uv run --extra gpu python -m scripts.search run experiments/Speck1-140M \
  --name evolution-01 \
  --hours 3
```

`--hours` and `--generations` are cumulative study limits, not an allowance added by each
invocation. Resume with the same command and raise either limit when more work is required:

```bash
uv run --extra gpu python -m scripts.search run experiments/Speck1-140M \
  --name evolution-01 \
  --hours 6
```

A resumed study must use the same experiment, search settings, model/data/tokenizer inputs, packed
dataset manifest, and runtime contract. Limits may be increased but not decreased. Only one
coordinator can hold a study lock at a time.

The coordinator checkpoints state after each material change. Re-running `run` resumes the stored
phase and candidate rather than creating a new study. Runtime failures are recorded on the study
and candidate; out-of-memory and non-finite failures are classified separately in candidate
results.

## Lifecycle

Each generation moves through these phases:

1. `planning` creates the baseline, mutated candidates, and random immigrants.
2. `screen` trains every viable candidate to the first rung and promotes the leaders.
3. `develop` extends promoted candidates to the second rung and promotes again.
4. `confirm` trains the remaining candidates to the final search rung, scores them, and adds them
   to the confirmed archive.
5. `complete` records a finished generation. The next generation starts only when the configured
   budget permits it.

Candidate scores expose three lanes:

- `quality` prioritizes validation loss.
- `efficiency` prioritizes latency and memory.
- `balanced` combines quality and efficiency ranks.

The checked configuration screens at 2,097,152 tokens, develops at 8,388,608 tokens, confirms at
33,554,432 tokens, and reserves 100,663,296-token runs for final verification.

## Inspect Progress

Show a concise status report:

```bash
uv run --extra gpu python -m scripts.search status evolution-01
```

The report includes the current phase and candidate, candidate counts by status and rung, current
lane leaders, the latest validation point, and retained checkpoint bytes. Use JSON for automation
or detailed diagnosis:

```bash
uv run --extra gpu python -m scripts.search status evolution-01 --json
```

`stopped` means a cumulative time or generation limit was reached. It does not necessarily mean a
generation reached `complete`. Increase a limit and run the study again to continue an incomplete
phase.

## Finalize

Finalization requires a confirmed candidate for all three score lanes. In practice, let at least
one generation finish its `confirm` phase before finalizing. A study stopped partway through a
generation may not have eligible finalists yet.

```bash
uv run --extra gpu python -m scripts.search finalize evolution-01
```

For each unique lane leader, finalization:

1. Restores or rebuilds the retained search checkpoint.
2. Runs both continued and independent training to `final_tokens`.
3. Profiles eager and compiled GPU inference.
4. Profiles CPU inference under the configured thread contract.
5. Ranks final quality, efficiency, and balanced results.

The command is resumable: completed final runs and compatible profiles are reused.

## Study Files

Studies default to `~/.cache/speck/search/<name>`. Set `speck_base_dir` before the first command to
move the cache root.

```text
<study>/
  search.json                 Immutable materialized search settings.
  state.json                  Limits, provenance, phase, and current progress.
  finalists.json              Final report, created by finalize.
  candidates/
    000001/
      architecture.json       Materialized candidate architecture.
      result.json             Rungs, validation curve, profile, scores, and errors.
      checkpoint/             Retained search checkpoint files when applicable.
      final/
        continuation/         Final run resumed from the search checkpoint.
        independent/          Final run trained independently.
        profile.json          Final GPU and CPU measurements.
```

Candidate checkpoints are pruned as a study advances. The confirmed archive retains the leading
checkpoint candidates needed for finalization and can deterministically rebuild a required final
rung checkpoint when necessary.

## Reproducibility Contract

Search comparisons depend on more than architecture JSON. A study records and validates:

- Model, data, and tokenizer configurations.
- Tokenizer fingerprint and special-token IDs.
- Packed-data manifest fingerprint, resolved directory, and shard checksums.
- Torch and CUDA versions, device identity, dtypes, deterministic settings, and cuBLAS workspace
  configuration.
- Search, training, evaluation, promotion, and profiling settings.

Start a new named study when any of these inputs intentionally changes. Do not edit persisted study
files by hand.
