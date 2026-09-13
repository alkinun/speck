# 2026-09-13 — Tokenizer pilot screen execution

## Context

The three tokenizer-pilot seed-42 arms were authorized after corrected fixed-FLOP accounting, real
input verification, checkpoint/resume qualification, CUDA document-NLL parity, and full-loop fixture
qualification. This entry tracks execution without opening D5 or authorizing confirmation.

## Work performed

Completed the Mistral 32K baseline through 18,311 optimizer steps and 1,200,029,696 aligned training
tokens. The run produced 20 complete six-category evaluation boundaries, a final checkpoint, an
analyzer-compatible result, and a compact summary. Mechanical interruptions were resumed from exact
completed checkpoints; evaluations newer than their checkpoint lineage were preserved separately and
excluded from scientific use.

## Decisions

The baseline arm passes completion and schema gates with fixed-document macro BPB 1.0954040, 4.71
active hours, 70,766 tokens/s including evaluation time, and 7.54 GB peak allocated memory. This is a
baseline result, not a tokenizer decision. Continue with the two already authorized custom seed-42
arms. Confirmation remains blocked until all three screen arms complete and measured overhead is
reconciled against the 30-hour ceiling.

## Evidence and links

- [Mistral completion record](../../results/data/tokenizer-pilot-mistral-seed42-20260913.json)
- [Execution record](../flagship/tokenizer_pilot_executions_v1/mistral-32k-seed-42.json)
- [First interruption](../../results/systems/tokenizer-pilot-mistral-interruption-20260912.json)
- [Second interruption](../../results/systems/tokenizer-pilot-mistral-interruption-v2-20260912.json)

## Open questions

The relative custom-tokenizer quality and runtime are unknown. Baseline active time is materially above
the synthetic throughput-only projection because full document evaluation and interruption recovery
are included differently; the screen stop rule requires measured reprojection after all three arms.

## Next actions

Run the 40,960 compression endpoint and exact-32K compact endpoint from their immutable execution
records. After all three arms complete, run only the frozen screen analysis and overhead projection.
Do not launch confirmation or open D5 beforehand.
