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

## Pilot 2 — Three training templates, held-out answers

Pilot 2 trains archive, registry, and ledger with letters only, then holds out both manifest syntax
and phrase answers. It again uses 1,600 retrieval examples and 6.55M replay tokens. On the fixed
30-case adaptation validation stream, the final manifest/phrase result is 23.3% exact, 33.3%
candidate, and 96.7% specificity. The high specificity is significant (`p = 2.89e-8`): the model
selects the queried manifest association more strongly than an unrelated association even though it
often fails to emit the unseen phrase.

An additional seed-0 evaluator factorial separates template transfer from answer transfer:

| Evaluation condition | Exact | Candidate | Specificity accuracy | Specificity score |
| --- | ---: | ---: | ---: | ---: |
| Archive / letters (seen/seen) | 100% | 100% | 100% | 12.500 |
| Ledger / letters (seen/seen) | 100% | 100% | 100% | 12.992 |
| Manifest / letters (held-out/seen) | **93.3%** | **90.0%** | **96.7%** | 5.532 |
| Manifest / phrases (held-out/held-out) | 40.0% | 50.0% | **93.3%** | 1.737 |

This is the first evidence of template-robust target selection. The remaining failure is joint
generalization to a new surface form and a new output vocabulary. Pilot 3 will retain three training
templates but restore phrase supervision; the manifest remains fully held out.

## Pilot 3 — Three templates and both answer sets

Pilot 3 trains archive/registry/ledger with both letters and phrases. Manifest/phrases remains held
out as a surface-form split, while the exact validation answers and random records are unseen. At
step 200, the fixed adaptation validation stream passes the predeclared 80% gate:

| Metric | Result |
| --- | ---: |
| Exact match | **86.7%** |
| Candidate accuracy | **86.7%** |
| Token accuracy | 93.3% |
| Specificity accuracy | **96.7%** |
| Specificity score | 4.599 |

An independent seed-0 evaluator confirms 86.7% exact/candidate and 90.0% specificity with two
records. It also exposes a load boundary:

| Held-out manifest/phrases load | Exact | Candidate | Specificity accuracy |
| --- | ---: | ---: | ---: |
| 2 records | **86.7%** | **86.7%** | **90.0%** |
| 8 records | 43.3% | 43.3% | **93.3%** |

The model usually responds more to the queried record than to the distractor even at eight records,
but cannot reliably make the correct ten-way choice. Template and value generalization are now
demonstrated at the trained load; distractor-load generalization is not.

The 20M-token original-corpus loss is `2.805700`, only `+0.006011` nats from the post-32K parent
(`2.799689`) and inside the measured `0.00965`-nat seed range. The 50% replay recipe continues to
prevent material forgetting under the harder task mixture.

## Decision after pilot 3

- Treat the two-record held-out-template gate as passed on one training stream.
- Do not run long-length or architecture comparisons yet; eight-record exact accuracy is below gate.
- Train the same template/value mixture directly at eight records, then replicate distinct training
  streams only if the eight-record manifest gate passes.

## Artifacts

- Report:
  [results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse/kda-seed42.json](../results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse/kda-seed42.json)
- Checkpoint: `SpeckLC-150M-StructuredRetrievalAdapt-kda-template-diverse-seed42`, step 200
- Model SHA-256: `3a7c48541f2467e566050f55265b64647e50e41347278e1afbc2557c88019980`
- Metadata SHA-256: `8fdee220931c81fa0a6f2bbdb00e58f05dbe83ccea3331f4234d212dc80bddb4`
- Optimizer SHA-256: `7abecf2d5df08ab9ae2ff9b222184c951e308a7af0af7d381e3b28f2adf58a16`
- Pilot 2 report:
  [results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse/kda-template3-seed42.json](../results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse/kda-template3-seed42.json)
- Pilot 2 evaluator controls:
  [results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse](../results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse)
- Pilot 2 model SHA-256: `9b070cba51a9d6d9799393e93edc8ceca0d80801c95d57c1bac11ecd2a2490c7`
- Pilot 2 metadata SHA-256: `eb204400eea17ed7307875b71df0967a12bd996e8134a70601866d73461d52bc`
- Pilot 2 optimizer SHA-256: `2f941cbb69455f3a51df246bf192b6329b9b92648bcd7f856ba39f2581dfc6cc`
- Pilot 3 report:
  [results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse/kda-template3-mixed-seed42.json](../results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse/kda-template3-mixed-seed42.json)
- Pilot 3 short-loss report:
  [results/SpeckLC-150M-StructuredRetrievalAdapt/short-loss/kda-template3-mixed-seed42.json](../results/SpeckLC-150M-StructuredRetrievalAdapt/short-loss/kda-template3-mixed-seed42.json)
- Pilot 3 model SHA-256: `cae56a8a4f7af2ad553904957a0e2053332e1c7d22f6b4c1d4940a27372da047`
- Pilot 3 metadata SHA-256: `4dd437e9cb15d87d4415e0d42ca1d2476a1785273ca351f5fd9d7b85ceb90cd1`
- Pilot 3 optimizer SHA-256: `0ef72600387eddda75f6fcc158c4b11578fb4d7cd19a7d9c22635faa124d576e`
