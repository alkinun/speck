# SpeckGym v0

SpeckGym tests whether procedural pre-pretraining changes Speck's language-learning curve without
changing its model architecture, parameter count, inference memory, or inference speed. The suite
uses the pinned Speck1.5 language corpus and the unchanged 140,652,288-parameter architecture.

## Experiment Contract

All runs use 2,048-token sequences, 65,536-token optimizer batches, Muon, and the Speck1.5 training
settings. The requested 500M-token budget rounds to the same 500,039,680 actual tokens and 7,630
updates for every run.

The selected formal positive control uses 500 updates at the standard Speck batch size. This makes
the procedural phase 32,768,000 tokens rather than the originally considered approximately 8M
tokens. Runs B-E consequently use 467,232,000 requested language tokens, rounded to 467,271,680
actual tokens over 7,130 language updates.

| Run | First 500 updates | Remaining updates |
| --- | --- | ---: |
| A | No separate warm-up; continuous Speck1.5 language training | 7,630 total |
| B | IID abstract symbols matched to SpeckGym's unigram distribution | 7,130 language |
| C | SpeckGym blocks shuffled token-by-token | 7,130 language |
| D | k-Shuffle Dyck formal language | 7,130 language |
| E | Mixed SpeckGym computational curriculum | 7,130 language |

Every warm-up still constructs the full 32K tied embedding and output head. Only 128 deterministic
abstract token IDs carry procedural symbols, but unused rows remain present and participate in the
full softmax. Warm-up parameter count and output computation therefore match baseline language
training.

At the transition, Speck retains residual cores, internal width adapters, and final normalization.
It reinitializes the tied token matrix, first embedding-to-hidden adapter, and final
hidden-to-embedding projection. A new optimizer and LR schedule are created, and the Speck1.5 data
loader starts at offset zero. This operation is checkpoint initialization, not checkpoint resume.

## Procedural Corpora

Prepare all B-E corpora on CPU:

```bash
uv run --extra cpu python -m scripts.speckgym_prepare experiments/SpeckGym-v0
```

Artifacts are written under `~/.cache/speck/data/SpeckGym-v0`. Each corpus uses a versioned direct
token manifest and checksummed `uint16` shards. Generation is deterministic and requires neither
network access nor a GPU after the configured tokenizer has been prepared.

SpeckGym E contains equal-weight sources for:

- nested hierarchy and direct-parent retrieval;
- randomized key/value binding;
- ordered state updates;
- set union;
- multi-step function composition.

Examples use freshly sampled entity, value, and rule assignments. Each source progresses through
three difficulty levels over the warm-up. Run C applies a deterministic permutation to every
2,048-token E block, preserving its exact token multiset while destroying order. Run B samples IID
symbols from E's aggregate unigram distribution.

Run D follows the main k-Shuffle Dyck construction in Hu et al., [Between Circuits and
Chomsky](https://aclanthology.org/2025.acl-long.478/): 64 bracket pairs, 128 symbols,
`p_open=0.5`, maximum depth 16, and 2,048-token truncation without draining unmatched open
brackets. Its data construction and 500-update duration reproduce the selected formal control;
optimization and interface-reset behavior intentionally follow the common Speck experiment rather
than the paper's Pythia setup.

## Training

Train baseline A directly:

```bash
uv run --extra gpu python -m scripts.speckgym_train A language
```

For each of B-E, finish the warm-up before starting language training:

```bash
uv run --extra gpu python -m scripts.speckgym_train B warmup
uv run --extra gpu python -m scripts.speckgym_train B language
```

Replace `B` with `C`, `D`, or `E` for the other runs. Both phases support `--resume <phase-step>`.
Language-phase resume restores the complete language checkpoint and does not need the warm-up
checkpoint to remain available.

W&B uses one group per A-E run and separate jobs for warm-up and language phases. Logged token and
step axes are global across the experiment, while native checkpoint filenames use phase-local
steps.

| Requested tokens | Actual tokens | A step | B-E language step |
| ---: | ---: | ---: | ---: |
| 50M | 50,003,968 | 763 | 263 |
| 100M | 100,007,936 | 1,526 | 1,026 |
| 250M | 250,019,840 | 3,815 | 3,315 |
| 500M | 500,039,680 | 7,630 | 7,130 |

Each milestone forces fresh language validation before its checkpoint is published. Metadata records
the requested and actual global token positions, source checkpoint hashes, transferred/reset tensor
keys, optimizer time, evaluation time, checkpoint time, and active process time. Active time includes
initialization, shard verification, compilation, evaluation, and checkpointing, but excludes downtime
between resumed processes.

## Evaluation

Run the native held-out hierarchy, retrieval, binding, state, set-union, and composition evaluation:

```bash
uv run --extra gpu python -m scripts.speckgym_eval E 500000000 procedural
```

Cases use evaluation-only seeds, randomized textual symbols, four raw-continuation choices, and mean
conditional token log probability. Reports include per-family accuracy, overall accuracy, chance
level, winning margins, case fingerprints, and checkpoint hashes.

Run HellaSwag, ARC-Easy, ARC-Challenge, and PIQA through the pinned external harness:

```bash
uv run --extra gpu --group open-slm python -m scripts.speckgym_eval \
  E 500000000 standard
```

This exports the native checkpoint locally and retains the existing model-code, lm-eval, and dataset
revision guards. Use `standard --limit 2` only for a smoke test; limited results cannot be summarized.

Combine available quality and timing artifacts:

```bash
uv run --extra cpu python -m scripts.speckgym_eval E 500000000 summary --device cpu
```

Outputs default to `~/.cache/speck/evaluations/SpeckGym-v0/<run>/<requested-tokens>`. Generated
models, reports, checkpoints, and corpora remain outside the repository.

## Cross-Run Report

Once runs have been evaluated, aggregate them into one comparison:

```bash
uv run --extra cpu python -m scripts.speckgym_report
```

This reads every `summary.json` under `~/.cache/speck/evaluations/SpeckGym-v0` and writes
`report.md` and `report.json` beside them. Runs that have not reached a milestone are reported as
missing rather than omitted, so the report is readable while a sweep is still in flight. The
markdown covers the language-learning curve at all four milestones with deltas against baseline A,
final standard-task and per-family procedural accuracy, token-budget parity across arms, and
training cost.
