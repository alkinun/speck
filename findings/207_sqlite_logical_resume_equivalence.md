# 207 — Resume equivalence is logical database identity, not SQLite page identity

The 2B calibration completed acquisition, global exact/verified-near deduplication, and 2.000008B
packed tokens before the injected real-data resume probe stopped its final gate. The resumed and
uninterrupted 2,000-record probes have identical cleaned JSONL, removal output, counts, table row
counts, table contents, and successful SQLite integrity checks. Their SQLite files have equal size but
different SHA-256 values because interruption changes internal page layout.

The corrected prospective gate requires exact output, removal, and count identity plus deterministic
logical hashes over ordered `docs`, `bands`, and `checkpoints` rows. The observed logical hashes are
identical for all three tables. Both physical hashes remain in the result as diagnostics; neither is
substituted for logical content identity.

This is stricter on scientific state and weaker only on irrelevant SQLite serialization. The
production index is an internal rebuildable lookup artifact, not released model data. The successor
does not change source records, deduplication decisions, packed shards, or the frozen calibration
plan. Resume/cleanup, firewall disjointness, scale projection, and final calibration authority remain
pending until rerun.

Artifact: [logical SQLite resume qualification](../results/data/production-resume-logical-sqlite-qualification-20260910.json).
