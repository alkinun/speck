# Experiment directory status

No first-grant flagship experiment is launchable yet. Its scope and planning targets live under
[`research/flagship/`](../research/flagship/); a complete experiment appears here only after data,
training, hardware, checkpoint, and analysis contracts are frozen.

## Release and continuation recipes

- `Speck1-140M`, `Speck1.5-140M`, and `Speck2-140M` are retained general-model recipes.
- Their `-Instruct` variants are retained supervised-fine-tuning recipes.

These are reproducible baselines, not the first-grant flagship architecture.

## Completed Instruct data pilot

The [pre-compute Speck2 pilot](../research/notebook/2026-09-12-instruct-data-pilot.md) exercised
the [100K compiler recipe](Speck-Instruct-Starter/README.md), its
[capacity-adjusted 500K variant](Speck-Instruct-Starter500K/README.md), and
[native SFT/evaluation](Speck2-140M-Instruct-Starter100K/README.md). These are retained reproduction
inputs; the pilot is complete and did not promote a new release.

## Active evidence inputs

The Kimi transfer, replication, 32K continuation, noise-floor, global-count, Reader Attention, and
`Speck-Paper1-Baselines-131M` families are completed or bounded research evidence. They constrain the
flagship defaults but should not be relaunched merely because their configs remain checked in.

## Historical experiments

Other `SpeckLC-150M-*` and synthetic-memory directories preserve earlier screens, diagnostics, and
negative results. Use [`findings/README.md`](../findings/README.md) for the small current evidence set
and [`findings/ARCHIVE.md`](../findings/ARCHIVE.md) for full provenance.

## Launch rule

Before running a directory, require all of the following:

1. It is named by the current [`flagship execution plan`](../research/flagship/EXECUTION.md), not only
   by an archived finding.
2. Its data and tokenizer artifacts exist and match their manifests.
3. Its analysis, checkpoint, hardware, and stopping rules are frozen before output.
4. Its expected GPU-hour charge fits the current phase and leaves protected reserve untouched.

Planning-only geometry under `research/flagship/targets/` intentionally fails this launch rule.

Experiment directories are reproducibility inputs, not lab notes or conclusions. Record chronological
context under `research/notebook/`, machine-readable outcomes under `results/`, distilled conclusions
under `findings/`, and paper use under `paper/claims.json`. See
[`research/WORKFLOW.md`](../research/WORKFLOW.md).
