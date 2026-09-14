# Completed tokenizer screen: retain the frozen Mistral fallback

All three seed-42 screens are complete. The [checked review](../../results/data/tokenizer-screen-20260914.json)
verifies paired evaluation documents, common backbone/stream, matched analytic-FLOP targets, and
summary/result identities. All 61 checkpoint-bound evaluation-file hashes were also checked in the
retained timing-evidence receipt. The compact run exited successfully after 4h 43m 16.998s service wall time.

| Tokenizer | Fixed-document macro BPB | Fixed-FLOP macro BPB | Completed active GPU-hours |
| --- | ---: | ---: | ---: |
| Mistral 32K | 1.095404003 | 1.095404003 | 4.710480 |
| Custom 32K | 1.103594371 | 1.066491016 | 4.718507 |
| Custom 40,960 | 1.109140944 | 1.107121722 | 4.589378 |

The frozen rule nominates **custom 32K** among the two custom endpoints. This nomination does not
establish replicated custom superiority or inferiority. Its improved fixed-FLOP result is retained
alongside the worse fixed-document result; neither substitutes for the required confirmation matrix.
The recovered 40,960 report still explicitly lacks peak GPU allocation.

## Budget disposition

The three completed active durations sum to **14.018364 GPU-hours**. The required two additional
baseline and two selected-custom runs have an active-duration projection of **18.857974 GPU-hours**.
Thus even the lower bound on the v10 projection is **32.876338 GPU-hours**, above the **30-hour ceiling**.

The maintained analyzer applies a monotone consequence of the unchanged stop rule: missing setup,
final publication, qualification, and failed/replayed work cannot reduce this projection. The new
lower-bound field can establish a stop but can never authorize confirmation. This is a projection
under the frozen measured-cost rule, not a guaranteed runtime for unexecuted seeds.

**All-attempt spending is still incomplete.** Mistral's interrupted initial session/replay costs and
GPU probes outside the retained service preflights need reconciliation. The full budget field remains
`null`; these omissions are not assigned zero or hidden in a reported total. This accounting task
remains necessary for disclosure but cannot reverse the budget stop.

The registered consequence is to retain Mistral and leave D5 unopened. No confirmation or audit was
launched. The separate [artifact decision](../flagship/tokenizer_decision_v1.json) freezes the pinned
Mistral tokenizer at SHA-256 `dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055`.
The artifact is copied into `/mnt/speck-data/speck/tokenizer-final-mistral-v1` and reopened through the
ordinary tokenizer loader. Base vocabulary is 32,000; the existing model-accounting reservation of
three role IDs is preserved without deciding post-training behavior.

## Consequence for prepared stock

The frozen base tokenizer has exactly the same bytes as the reference counter used for Math L2 and
FineWiki. The **384,788,210** and **582,070,378** counts therefore also hold under the selected base
tokenizer, including BOS/EOS. No retokenization is needed merely to establish that identity.
This does not establish joint-background eligibility, packing alignment, or a complete launch corpus.

Reproduce the review with the hash-bound runtime review listed in the checked result:

```bash
uv run --no-sync python -m scripts.tokenizer_pilot_screen \
  /mnt/speck-data/speck/tokenizer-screen-review-20260914/review.json /path/to/new-analysis.json
```

The fallback materializer independently recomputes the review, verifies the static prerequisite's
baseline manifest/model, and requires new artifact and decision paths. Its implementation is pinned
inside the decision. It cannot use incomplete, within-budget accounting to freeze an outcome.
