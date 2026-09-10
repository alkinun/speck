# 209 — The 2B calibration closes P0 operations through the 150B branch

After reviewing finding 208's complete technical result, the SpeckLabs project owner accepts the 2B
six-category calibration as the P0 non-GPU production-operations fallback through a 150B unique-token
branch. The paused 20B rehearsal is not resumed before D5/E3 because it would spend about 2.5 serial
days tokenizing data that the selected tokenizer and repetition policy may replace.

The 150B branch remains conditional: E3 must support repetition, final preparation must clean acquired
intermediates after deduplication, and source capacity, runtime, index growth, and free space must be
checked at durable boundaries. The current 500B branch remains blocked because its projected 6.62 TB
live working set exceeds the data volume and its roughly 63 serial days are operationally unacceptable.
It requires demonstrated E3 necessity and a versioned cleanup/parallelization or storage successor.

This decision closes PREGRANT R3 by explicit fallback, not by claiming that a full 20B rehearsal ran.
It does not issue model-training authority. Real firewall partitions, D5, final tokenizer-bound corpus
preparation, and a data-launch verifier successor binding the calibration fallback plus final production
outputs remain required.

Artifact: [project-owner fallback decision](../research/flagship/production_data_calibration_fallback_v1.json).
