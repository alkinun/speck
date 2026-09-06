# Deferred mixture-of-experts support

Mixture-of-experts is not part of the first flagship grant. Speck retains a compact conventional
reference so a later sparse-upcycling program can start from tested model semantics rather than
rebuilding routing from scratch.

## Retained substrate

- `RoutedSwiGLUSpec` architecture grammar with active-parameter accounting.
- Token-choice, dropless top-k routing with selected-score normalization.
- A direct expert loop on CPU and a grouped-BF16 CUDA path using `torch._grouped_mm`.
- Load-balancing and router z-loss terms in base training.
- Per-layer routing entropy, utilization spread, zero-load counts, and periodic per-expert
  weight/gradient norms.
- Checkpoint-compatible Muon expert-bank updates.
- Held-out loss sensitivity from masking one routed layer at a time through
  `scripts.expert_masking`.

These paths remain covered by the architecture, model, training, and expert-masking unit tests. No
checked flagship experiment uses them.

## Deliberately absent

- An active dense-versus-MoE experiment or promotion rule.
- Loss-free or quantile balancing.
- Expert parallelism, FSDP, all-to-all dispatch, or distributed expert checkpoints.
- Transformers or GGUF MoE export and supported serving.
- A claim that analytic activated FLOPs predict realized wall-clock cost.

The archived September 2026 pilots, stopped screen checkpoint, logs, and packed DCLM-Edu stream live
outside the repository under `/mnt/speck-data/speck/archive/moe-deferred-20260906/` on the originating
research machine. Git history retains the removed screen tooling.

## Follow-on program

Begin from the released dense pre-decay flagship checkpoint. Freeze which dense MLPs are copied into
experts, then compare sparse upcycling with continued dense training under the same added data and a
matched wall-clock budget. Expert-parallel training, checkpoint/resume, export, and serving must
qualify before a scaled run. This is a separate 10K–50K+ compute proposal and paper, not a contingency
inside the first 5K-hour grant.
