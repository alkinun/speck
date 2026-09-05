# 107 — Finalist materialization boundary

## Frozen identity

The finalist family is fixed as `Speck-Paper1-Finalist-131M`, with configs under a new experiment root
and checkpoints under `/mnt/speck-data/speck/paper1-finalist-131m`. It contains exactly six pairs and
twelve runs: seeds 42/43/44 crossed with data offsets 0 and 1,610,612,736. Every run uses
1,539,833,856 tokens and one final checkpoint.

Both model arms and every proxy parent config are SHA-pinned. Data, model, tokenizer, runtime, optimizer,
learning rate, schedule, and batch geometry remain unchanged. Only the longer token horizon, exact
23,496-step terminal save, 5,874-step quartile evaluation cadence, family/run identities, and paired
seed/data offsets change.

## Fail-closed qualification

Materialization must fail rather than overwrite an experiment, checkpoint, result, target, or analysis
path. The two data windows must be disjoint and replay exactly at start/quartile/end. Storage requires
the named ext4 UUID and 25.77 GB free without deletion. The exact configs then require a fresh no-output
CUDA preflight and collector/analysis parity. RULER v2, NoLiMa, and HELMET remain separate gates.

## Decision

The contract authorizes implementing, testing, materializing, and qualifying configs only. It does not
authorize automatic or manual finalist training, component attribution, promotion, novelty, or
paper-scale work.

## Artifact

- [Finalist materialization v1](../research/paper-1/finalist_materialization_v1.json)
