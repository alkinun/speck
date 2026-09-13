# Speck2 starter-data two-epoch follow-up

Completed [pilot](../../research/notebook/2026-09-12-instruct-data-pilot.md) using the same prepared
100K data and settings as the [one-epoch run](../Speck2-140M-Instruct-Starter100K/README.md).
It starts from the pinned base with a fresh two-epoch cosine schedule.

- Final checkpoint: **3,108 steps**, **75,243,246 supervised tokens**.
- Follow the one-epoch reproduction commands with this experiment directory and final step.
- Export with `--expected-epochs 2`; use the inherited diagnostic and Open SLM configurations.
- [Consolidated results](../../results/Speck2-Instruct-Data-Pilot/summary.json) record the mixed outcome.
