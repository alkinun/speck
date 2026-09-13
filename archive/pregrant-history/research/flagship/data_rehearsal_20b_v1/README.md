# Flagship 20B production data rehearsal v1

This is the first real, rights-bound, non-training rehearsal of the flagship data path. It prepares
20B packed training tokens under the balanced six-category prior using one operational primary source
per category:

| Category | Source | Packed-token target |
| --- | --- | ---: |
| Web | Ultra-FineWeb English v1.4 | 11B |
| Code | Common Pile Stack v2 educational code | 3B |
| Math | FineMath 4+ | 2B |
| Synthetic | Cosmopedia v2 | 2B |
| Science | Common Pile PubMed | 1B |
| Reference | FineWiki English | 1B |

The rehearsal uses the pinned Mistral tokenizer fallback because D5 has not run. This does not select
the final tokenizer or data mixture. E1–E4 retain scientific selection authority.

## Contracts

- [`production_plan.json`](production_plan.json) freezes source readers, quotas, filtering, a 5%
  pre-training holdout pool, benchmark contamination, Gitleaks, exact/near deduplication, packing, and
  output paths.
- [`deny_ledger_v1.json`](deny_ledger_v1.json) is the initial human-reviewed removal ledger. It is empty
  because no verified removal request is currently registered; future entries require a successor.
- [`orchestration.json`](orchestration.json) binds the six stage commands and every static input by
  SHA-256.

## Stages

1. Resolve immutable source revisions and shuffled file lists.
2. Acquire source files with source-native filters, conservative local PII/repetition/host checks,
   63,652-task contamination removal, fully redacted Gitleaks exclusion, durable source-file resume,
   and raw cleanup.
3. Apply global normalized exact deduplication and disk-backed MinHash candidate plus verified
   Jaccard near-deduplication, with heldout candidates taking precedence over training records.
4. Pack exactly the six source quotas into checksummed uint16 shards.
5. Inject a real-data preprocessing interruption, compare resumed with uninterrupted outputs, reopen
   completed packing, and verify raw cleanup.
6. Verify training/holdout disjointness and publish hash-only firewall candidate commitments.

Passing this rehearsal permits issuing a production-operations qualification record. It does not
authorize tokenizer selection, model training, sealed-audit access, or corpus redistribution.

Run or resume only from a clean checkout containing the exact identities in `orchestration.json`:

```bash
speck_base_dir=/mnt/speck-data/speck \
  uv run --extra cpu python -m scripts.data_rehearsal \
  research/flagship/data_rehearsal_20b_v1/orchestration.json \
  --issue-operations-record /mnt/speck-data/speck/data-rehearsal-20b-v1/operations.json
```
