# 129 — Full finalist automation transition simulation

## Twelve-cell state machine

The frozen v2 `finalize` function was executed against temporary paths for all six controls and six
candidates. External systemd, checkpoint collection, git, and scheduling calls were replaced with
recording doubles; the runner's state-mutation and transition-writing code executed unchanged.

The simulation produces exactly 12 collections, validations, transitions, and commits; 11 successor
schedules; one target lock after control six and before candidate one; and one analysis after candidate
six. It reaches `complete` with no next run and preserves both the interrupted attempt and identical
rerun record through every transition. Every transition retains no polling, branching, or retry and
binds the frozen automation hash.

An obsolete prelaunch-only test also asserted that the real checkpoint root must not exist. It was
replaced with a temporary-path non-mutation check, so the suite continues to test state-check purity
after legitimate scheduled checkpoints begin appearing during the active sequence.

## Artifact-first success boundary

`base_train` writes its final summary atomically only after all steps and tracking finish. The finalizer
then disables the trigger, waits on the training PID, validates the collected scientific artifact, and
commits before scheduling. Collection, validation, or scheduling failures stop rather than skip or
retry. This supports the current artifact-first authority without sampling intermediate metrics.

## Residual hardening—not a live patch

Two future improvements are identified. The transition should record terminal systemd result/exit
provenance, and finalization should explicitly recheck stored `next_run` in addition to deriving it from
counts. The current launch already checked both, the repository/program are validated, and the summary
ordering limits the first risk to post-summary cleanup. Neither warrants invalidating the active frozen
runner. A disposable systemd integration fixture is also still distinct from this temporary-state
simulation.

## Decision

Current v2 transition ordering qualifies and remains byte-identical. No interruption, patch, retry,
branch, skip, or result inspection is authorized. Runner hardening is deferred until the sequence is
complete.

## Artifact

- [Transition audit](../results/Speck-Paper1/finalist-automation-transition-audit-v1.json)
