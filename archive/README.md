# Historical work

The complete pre-simplification checkout is preserved at **pre-simplification-2026-09-17**;
[snapshot.json](snapshot.json) records its exact commit and tree. It includes all earlier plans,
papers, results, source approvals, code, tests, the previous physical archive, and the three completed
preparation receipts. Runtime datasets/checkpoints remain in their existing external locations.

Historical work is stored in Git instead of duplicated throughout the active tree. To inspect a file:

```bash
git show pre-simplification-2026-09-17:research/status.json
python -m scripts.archive locate scripts/tokenizer_pilot_screen.py
```

To restore the complete original environment in a separate checkout:

```bash
python -m scripts.archive check
python -m scripts.archive restore /path/to/speck-history
```

Install dependencies using that checkout's lockfile when reproducing its workflows. Verify external
input identities before continuing old preparation; current sources are not replacements for frozen
implementations. Existing historical evidence retains its original meaning and limitations.

The current direction is [PLAN.md](../PLAN.md). The old long-context paper and experiment funnels are
not active alternatives. Reusable behavioral tests may read frozen configurations, but historical
experiment suites run in their original checkout. Fetch full Git history when cloning/using CI.
