# 196 — Checkpoint replacement now preserves its predecessor

The cleanup follow-up fixes the two highest-priority deferred runtime findings without rewriting
their historical evidence records.

Checkpoint replacement now serializes and flushes every payload before touching a completed step. A
small transaction journal records the predecessor files, the completion marker is hidden during the
multi-file switch, and any catchable publication failure restores the old model, optimizer, metadata,
timing, and marker. Checkpoint discovery also recovers a transaction interrupted by an uncatchable
exception after publication started. The established filenames and load contract remain compatible.
Failure injection covers both tensor writes, metadata serialization, timing collection, partial
`os.replace`, interrupted publication, and successful removal of stale optional timing.

The production near-duplicate reader now opens a source only on a real cache miss. Repeated candidate
lookups reuse one handle, and all candidate handles close when comparison fails. Deduplication,
resume, output, and authority semantics are unchanged.

The default documented formatting check now uses `ruff-format.toml`, which excludes exactly 17
source-pinned files from formatting while retaining lint and test coverage. `pyproject.toml` remains
byte-identical to its prior data-evidence dependency.

These are CPU fixture qualifications. They do not qualify distributed checkpointing, the GH200
stack, or the 20B data path and grant no training authority.

Artifact: [runtime cleanup successor](../results/runtime-cleanup-successor-20260908.json).
