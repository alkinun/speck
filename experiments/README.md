# Runnable experiments

This directory contains maintained examples and future complete launch manifests.

- `Speck1-140M` and `Speck1-140M-Instruct` are retained base/SFT examples from the earlier release.
- `python -m scripts.smoke` exercises the maintained CPU pipeline on generated fixture data.
- No first-grant flagship experiment is launchable here yet. See [current status](../research/status.json).

Completed screens, pilots, and additional release recipes are in the
[experiment archive](../archive/pregrant-history/experiments/README.md). Reproduce them from their
recorded implementation revision using `python -m scripts.archive restore DESTINATION`.

New launch directories bind the model, tokenizer, data, training, analysis, and implementation
identities. Draft research plans belong under `research/flagship`.
