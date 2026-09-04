# 22 — Template-diverse retrieval adaptation

## Goal

Finding 21 showed that the replay-trained KDA checkpoint learned durable archive-template lookup,
not template-robust retrieval. This series changes the training split itself: surface forms, answer
sets, and random example streams are explicit experimental axes, and validation uses a template
that training never sees.

All runs start from the seed-42 post-32K KDA/sigmoid/NoPE checkpoint. They retain 50% replay from
the original packed language corpus, AdamW at `1e-4`, exact 4K inputs, batch 4, accumulation 4, and
two-record multi-key lookup. Candidate accuracy is a ten-way test; association specificity compares
the queried mutation with an unrelated-record mutation.

## Pilot 1 — Two training templates and two answer sets

Training alternates archive and registry syntax, each with both `A`–`J` and arbitrary two-token
phrases. The held-out evaluation uses ledger prose with phrases. The 200-step run contains 1,600
retrieval examples (400 per surface/answer condition), 2,400 supervised answer tokens, 6.55M
retrieval input tokens, and 6.55M language replay tokens.

| Step | Exact | Candidate | Specificity accuracy | Specificity score |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0% | 20.0% | 43.3% | −0.009 |
| 20 | 33.3% | 36.7% | 60.0% | 0.114 |
| 60 | 63.3% | 63.3% | **80.0%** | **1.185** |
| 120 | **66.7%** | **66.7%** | 66.7% | 0.900 |
| 200 | 50.0% | 50.0% | 56.7% | 0.435 |

Candidate accuracy becomes decisively above chance, so diverse prompting teaches partial transfer.
The result does not pass the 80% candidate-plus-specificity gate. Training loss falls effectively to
zero after roughly 70 steps while held-out performance oscillates and declines. More steps on the
same finite template family are not justified.

The pilot also establishes that multi-token optimization and validation work end to end. Exact match
requires both tokens and token accuracy ends at 75.0%.

## Decision after pilot 1

- Do not replicate this recipe yet; it fails the held-out-template gate.
- Do not increase steps without increasing relation-language diversity.
- Since archive-only adaptation already transfers to unseen phrases, spend the next supervision on
  more surface forms and reserve phrases for validation.
- Add a fourth template, train archive/registry/ledger with letters, and hold the fourth template plus
  phrases out together.

## Artifacts

- Report:
  [results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse/kda-seed42.json](../results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse/kda-seed42.json)
- Checkpoint: `SpeckLC-150M-StructuredRetrievalAdapt-kda-template-diverse-seed42`, step 200
- Model SHA-256: `3a7c48541f2467e566050f55265b64647e50e41347278e1afbc2557c88019980`
- Metadata SHA-256: `8fdee220931c81fa0a6f2bbdb00e58f05dbe83ccea3331f4234d212dc80bddb4`
- Optimizer SHA-256: `7abecf2d5df08ab9ae2ff9b222184c951e308a7af0af7d381e3b28f2adf58a16`

