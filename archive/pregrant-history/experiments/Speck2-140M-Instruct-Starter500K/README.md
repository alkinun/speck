# Speck2 capacity-adjusted 500K SFT

Completed [pilot](../../research/notebook/2026-09-12-instruct-data-pilot.md) using the
[500K dataset](../Speck-Instruct-Starter500K/README.md). The base, tokenizer, masking, optimizer,
4K ceiling, batch geometry, and one-epoch schedule policy inherit from the
[100K run](../Speck2-140M-Instruct-Starter100K/README.md).

- Native data: **496,537 training / 9,922 validation** conversations; 3,463/78 overlength rows
  rejected without truncation.
- Final checkpoint: **8,039 steps**, **205,997,024 supervised tokens**.
- Use this experiment with the shared reproduction commands and source directory
  `/mnt/speck-data/speck/data/instruct-starter-500k-v2`.
- For comparable diagnostic loss, pass
  `--validation-data-dir /mnt/speck-data/speck/data/Speck2-Instruct-Starter-4K-v3` to `sft_compare`.
  The new mixture's validation loss is recorded separately during training.
- Use final step 8,039 and `--expected-epochs 1` for export, then the same BananaMind/Open SLM protocol.

[Consolidated results](../../results/Speck2-Instruct-Data-Pilot/summary.json) show better diagnostics
but lower Open SLM aggregates. Source rebalancing, exclusions, and token exposure confound a
data-size-only interpretation.
