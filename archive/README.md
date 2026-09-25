# History

Retired plans, papers, results, tools and records live in Git, not in the active tree:

- **`pre-cleanup-2026-09-25`**: the tree before the pre-grant cleanup. It still has the RL and
  self-distillation code, the RL pilot, the synthetic R0 diagnostic, the H100 rental packet, the
  one-off corpus audits and their receipts, and the research-review notes.
- **`pre-simplification-2026-09-17`**: the complete earlier checkout. [snapshot.json](snapshot.json)
  records its exact commit and tree.

```bash
git show pre-cleanup-2026-09-25:docs/research.md
git show pre-simplification-2026-09-17:research/status.json
python -m scripts.archive locate scripts/tokenizer_pilot_screen.py
python -m scripts.archive check
python -m scripts.archive restore /path/to/speck-history
```

Run historical workflows in their own checkout with that checkout's lockfile. Historical evidence
keeps its original meaning and limits and does not govern current work; [PLAN.md](../PLAN.md) does.
Clone with full history (CI fetches it).
