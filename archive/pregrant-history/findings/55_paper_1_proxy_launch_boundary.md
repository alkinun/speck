# 55 — Paper 1 proxy-launch and release-claim boundary

## Question

Must every external long-context suite be executable before the frozen paired language-model proxy
can train, or is that requirement circular because those suites need the resulting checkpoints?

## Frozen boundary

The answer is now explicit and machine-checked. Proxy training requires the paired language-loss
estimand, six-run fixed sample, data windows, control-only target lock, model materialization, RTX 3090
preflight, storage, evaluation definitions, thresholds, contamination handling, prompt-versioning rule,
and missing-suite outcome to be frozen before any new model result.

It does not require acquisition or execution of external suites that consume a trained checkpoint.
Those suites do not disappear: RULER, NoLiMa, and HELMET remain mandatory release gates, and the active
manifest continues to score an unavailable required suite as a failed gate. This breaks a sequencing
cycle without weakening the final evidence standard.

## Live qualification

The committed qualifier rehashed all six frozen prerequisites, found all six checkpoint targets and
result records absent, verified the exact RTX 3090 at 0% utilization and 47 C, and requalified the
dedicated ext4 volume with 5.65 TB free. The memory-heavy HELMET transfer was stopped without deleting
its partial archive; available host memory rose to 25.3 GB before qualification.

The execution order is immutable: train and collect all three dense controls, lock the time-to-quality
target from those controls, and only then create or inspect the three KDA/GQA candidate result records.
There are no interim efficacy or futility decisions.

## Decision

The six-run paired proxy is authorized to execute. Release claims, long-context capability claims,
component attribution, architecture promotion, and paper-scale pretraining remain blocked. The first
authorized cell is dense pair 0, seed 42, packed-data offset zero.

## Artifacts

- [Frozen proxy-launch contract](../research/paper-1/proxy_launch_v1.json)
- [Live qualification](../results/Speck-Paper1/proxy-launch-qualified.json)
- [Qualification runner](../scripts/paper_proxy_launch_qualify.py)
