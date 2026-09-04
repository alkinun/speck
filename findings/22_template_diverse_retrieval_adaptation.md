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

## Pilot 4 — Direct eight-record training

Pilot 4 changes only the training and validation load from two records to eight. The 1,600 retrieval
examples remain balanced across three training templates and two answer sets. Learning is much
slower: specificity appears before correct decoding, and the final checkpoint does not pass.

| Step | Exact | Candidate | Specificity accuracy | Specificity score |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0% | 6.7% | 60.0% | −0.000 |
| 120 | 20.0% | 16.7% | 56.7% | 0.013 |
| 160 | 50.0% | 50.0% | 96.7% | 1.939 |
| 180 | **60.0%** | **60.0%** | 96.7% | 2.272 |
| 200 | 50.0% | 50.0% | **100%** | 2.501 |

The perfect final specificity shows that the target record is distinguished from the distractor,
but value selection remains unreliable. A curriculum is now the controlled next test: initialize
from pilot 3, which already learned template-robust two-record lookup, and add eight-record examples
at lower learning rate with language replay.

## Pilot 5 — Two-to-eight-record curriculum

The curriculum initializes from pilot 3 and adds 100 steps / 800 retrieval examples at eight
records, with `5e-5` learning rate and the same 50% language replay. Training examples use a new
random stream. The held-out manifest/phrase validation crosses both gates by step 30 and ends at:

| Metric | Fixed validation, n=30 | Independent seed-0, n=30 | Independent seed-0, n=100 |
| --- | ---: | ---: | ---: |
| Exact match | **80.0%** | 76.7% | **80.0%** |
| Candidate accuracy | **83.3%** | 76.7% | **80.0%** |
| Token accuracy | 90.0% | 88.3% | 90.0% |
| Specificity accuracy | **96.7%** | **93.3%** | **96.0%** |
| Specificity score | 5.422 | 5.577 | 5.894 |

The first independent 30-case estimate misses the 80% choice gate by one example. Expanding that
same deterministic stream to 100 cases gives exactly 80/100, while specificity remains 96/100. The
result is therefore a narrow pass, not a high-margin one. It warrants full training-stream
replication before any architecture claim.

An eager-training launch of the same curriculum failed before step 1: backward requested another
1.95 GiB with only 0.62 GiB free on the RTX 3090. No output checkpoint or report was created. Batch-4
adaptation therefore retains the compiled path; eager validation remains safe and avoids dynamic
shape recompilation.

## Three-stream replication

Seeds 43 and 44 vary both `torch.manual_seed` and the actual synthetic example streams at each
stage. Parent checkpoint, replay order, validation stream, optimizer, and budgets remain fixed.

| Training stream | Stage-1 fixed 2-record candidate / specificity | Stage-1 independent 8-record candidate / specificity | Curriculum fixed candidate / specificity | Curriculum independent n=100 candidate / specificity |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 86.7% / 96.7% | 43.3% / 93.3% | **83.3% / 96.7%** | **80.0% / 96.0%** |
| 43 | 93.3% / 100% | 63.3% / 96.7% | 70.0% / 100% | 65.0% / 95.0% |
| 44 | 90.0% / 96.7% | 50.0% / 90.0% | 56.7% / 100% | 64.0% / 93.0% |

The stage-1 template result is replicated: all three streams exceed the 80% two-record gate on the
fixed held-out template. Eight-record candidate accuracy is not replicated. The curriculum passes
only stream 42 and averages 69.7% on the independent 100-case evaluations. In contrast, specificity
remains 93–96% on all 300 independent cases.

This localizes the failure. KDA consistently identifies which association matters, across unseen
wording and load, but the trained readout does not reliably select the corresponding value among
eight bindings. The next recipe must increase structural diversity during training, not tune more
steps against the manifest validation set.

## Pilot 6 — Joint template, value, and load diversity

This run replaces the curriculum with one 400-step distribution spanning four training templates,
letters and phrases, and both two- and eight-record cases. Each of the 16 conditions receives exactly
200 examples. A fifth directory template and its native `Signal:` response cue are held out.

The fixed 30-case validation ends at 90.0% exact/candidate and 96.7% specificity. A larger independent
sample is more conservative:

| Held-out directory condition, n=100 | Exact | Candidate | Specificity accuracy |
| --- | ---: | ---: | ---: |
| Phrases | 78.0% | 77.0% | 95.0% |
| Letters | 80.0% | 80.0% | 99.0% |

Joint load training improves the independent phrase candidate result from the replicated curriculum
mean of 69.7% to 77.0%, but still misses the strict 80% gate. Nearly identical letter and phrase
results show that multi-token emission is no longer the main issue. The held-out template also
changes the response cue (`Answer:`, `PAYLOAD:`, `Seal:`, and `Marker:` in training versus `Signal:`
in validation). The next diagnostic will standardize only that output cue while retaining all
held-out record and question wording. This separates associative retrieval from small-model
instruction/readout calibration.

### Response-cue diagnostic

Changing only held-out directory's native `Signal:` cue to the archive-familiar `Answer:` cue makes
the result worse:

| Held-out directory condition, n=100 | Native cue candidate / specificity | `Answer:` candidate / specificity |
| --- | ---: | ---: |
| Letters | 80.0% / 99.0% | 61.0% / 98.0% |
| Phrases | 77.0% / 95.0% | 49.0% / 97.0% |

The target-selection signal remains, but decoding is co-adapted to the relation wording and output
cue. Standardizing only evaluation creates a mismatched prompt; it is not a valid shortcut. The next
control must train and evaluate every template with the same response channel.

### Matched standardized-cue training

A final 400-step control trains all four templates, both loads, and both answer sets with `Answer:`
and evaluates held-out directory wording with the same cue. It also fails the gate: final exact and
candidate accuracy are 60.0%, while specificity remains 93.3%. The best fixed-validation candidate
accuracy is only 70.0% at step 120.

This rules out the response token itself as the main cause. At this scale, native semantic cues help
readout, but neither native nor standardized cues make eight-record open decoding robust across
held-out wording. Direct-lookup prompt tuning stops here. The replicated result is strong
target-specific association selection; exact value emission at eight-record load remains an open
optimization/data problem.

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
- Pilot 4 report:
  [results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse/kda-template3-mixed-r8-seed42.json](../results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse/kda-template3-mixed-r8-seed42.json)
- Pilot 4 model SHA-256: `47e21c5290a73a424494b331765a959eceba1cba06509fb02671bca2af38eaf8`
- Pilot 4 metadata SHA-256: `d6a1ce6b3218d178c093d4ebe21f8100fa5c00d305fb7a04f235c9180ab3daf9`
- Pilot 4 optimizer SHA-256: `2440198c0002350b86ed7223200497fa21a99e8e6a06d1e2b1c048761a4b6189`
- Curriculum report:
  [results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse/kda-curriculum-r8-seed42.json](../results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse/kda-curriculum-r8-seed42.json)
- Curriculum n=100 audit:
  [results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse/eval-kda-curriculum-r8-manifest-phrases-4k-n100.json](../results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse/eval-kda-curriculum-r8-manifest-phrases-4k-n100.json)
- Curriculum model SHA-256: `a34678360e296e3588d421795359d90845df8e2628331afd11a1b9eb0f62697e`
- Curriculum metadata SHA-256: `fd27404f04c0bdfdf89320c923d58cfb186fea21efa952a5e582485efb971b3a`
- Curriculum optimizer SHA-256: `7b1984d1f819d6322c378fb636ef7a8b3cbe52895130ef943bd57e858f9e8177`
- Seed-43 stage-1 model / metadata / optimizer SHA-256:
  `dadbe884a19825bad89f8669c7861028db294ea85ea097b03472382cf2f4b6ae`,
  `e306960b5ba7d68d6f584f0c1a8a8b306c78420303d7ec51cbb82070a7b603fc`,
  `2a0de7cf5b081680114d46e49f75537fb8448fb987e58f7d900b7ab4bb6ec389`
- Seed-43 curriculum model / metadata / optimizer SHA-256:
  `1c6f2c471f612c49d65380be2ffd0712576ae6cf2cd93852d1da2155d9fbe7f9`,
  `f3cd3a2bee7f859ac818025033d4476607a6b0629fd68c0f3d321447ec1906f2`,
  `56f74d59c290808063bb65e595ee92547324f5c1b2e9269a3c93b12d0f740da3`
- Seed-44 stage-1 model / metadata / optimizer SHA-256:
  `b60e9277eed913b07db7ff2ee4742d96ef2341f70133bd4af2a98ea8c7ca2a9d`,
  `d6965c2d681b79cb395187d81623b22bac3508c68af09525f0ef91dc7554e5cb`,
  `f5801faa7f9b17e34f04d8f71847e70d311c98608a9148acb092e29104ce4199`
- Seed-44 curriculum model / metadata / optimizer SHA-256:
  `468b037877e851faed48cfc62591a08b93f245e4e52d7f8981d823b089fbdddb`,
  `7e444a1898cd622f6425c70951dd98e573a12260cebe844d41c9c8dc7a63a4b1`,
  `71dad761a9cf5614f2ff2314a9912f7302bc977c2d46a559f62e2e432c71723d`
- Joint-load report:
  [results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse/kda-joint-load-seed42.json](../results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse/kda-joint-load-seed42.json)
- Joint-load model / metadata / optimizer SHA-256:
  `0bbdd1348bc9c574152994bcc193f26473a88b47a028ad345e900bf526093d25`,
  `4b433d207eab990ba1055da93aa2bef8daf24aafe718f386a31834c32ab35477`,
  `30aed47d5d177d1dff75d5233398f129cd2ea1eb0f3375a43a5160ff715c68b9`
- Response-cue control reports are stored beside the joint-load reports with the
  `answer-cue.json` suffix.
- Matched-cue report:
  [results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse/kda-joint-load-answercue-seed42.json](../results/SpeckLC-150M-StructuredRetrievalAdapt/template-diverse/kda-joint-load-answercue-seed42.json)
- Matched-cue model / metadata / optimizer SHA-256:
  `84f5e31593956b161c0159b6a3e4b32f3a734795e82839b47a0f212764d058b9`,
  `13d04711e337d888bf44f1da4c95e9b4b379fbd1651061fda17ebebec1fed239`,
  `5dd1b5e12786f572e3e1107a4ae80d0df497c40e9992404138dfcbe6d8a38683`
