# Data

The goal and next work are defined in [PLAN.md](../PLAN.md), with the complete outline in
[the program overview](program.md). The engineering pilot is complete. Prepare the main corpus
against the working mixture and actual eligible supply; frozen pilot shares are historical.

## Pipeline

Pin source revisions and permitted use → acquire/filter → deduplicate and exclude evaluation
material → partition → tokenize/pack → verify → train. Source-separated shards allow later mixture
changes without retokenizing. Recheck tokenizer fingerprints and hashes whenever reusing stock.

| Responsibility | Code / command |
| --- | --- |
| Source readers, configuration, packing, resume | `speck/data/{acquisition,configuration,packing,dataset}.py` |
| Global disk-backed deduplication | `scripts.production_data_preprocess` (`--batched-minhash` is optional) |
| Secret filtering, near duplicates, contamination | `scripts.text_gitleaks_filter`, `text_near_duplicates`, `text_contamination` |
| Source-use review | `scripts.source_rights_review` |
| Checked retained-stock tokenization | `scripts.tokenize_stock` |
| Distributed loading | `speck/data/loader.py` |

For a complete experiment directory containing `tokenizer.json` and `data.json`:

```bash
uv run --no-sync python -m scripts.tokenizer_prepare PATH_TO_EXPERIMENT
uv run --no-sync python -m scripts.data_prepare PATH_TO_EXPERIMENT
```

These can download substantial data. Use `make smoke` for the offline fixture. Each preparation
command accepts `--help`; the old finite source-acquisition jobs run only from the
[historical checkout](../archive/README.md), where their original plans and code are preserved.

## Retained material

The runtime store is `/mnt/speck-data/speck` on the maintainer's machine; portable code defaults to
`~/.cache/speck`, overridden by `speck_base_dir`. Existing corpora and caches were not moved or deleted.
The frozen tokenizer is under `tokenizer-final-mistral-v1` on that volume. Its model SHA-256 is
`dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055`.

FineWeb's cache manifest is under `document-token-stock-e1s-v1/fineweb_edu`; its receipt reports
2.307B tokens and a passing reopen. The 476.775M-token Stack-Edu acquisition still needs full
exclusion. Neither those receipts nor the completed specialist stocks establish a joint eligible
pilot corpus on their own. Reopen manifests, preserve source rights, count overlapping banks once,
and verify cross-source/validation separation before reuse. The complete receipts are in the Git snapshot.

The finite [first pilot](../experiments/pilot/README.md) has now been selected, jointly excluded,
packed, and reopened from those retained inputs. It stores 105,652,323 training tokens including
loader reserve and whole-document overshoot, plus 799,536 validation tokens. The scheduled training
exposure is 104,857,600 tokens. Its packed manifest is at
`/mnt/speck-data/speck/data/flagship-pilot-105m/manifest.json`; selection, exclusion, and packing
records are under `/mnt/speck-data/speck/flagship-preparation-20260917/pilot`.
The [compact preparation receipt](../experiments/pilot/preparation.json) binds those manifests and
records complete one- and four-rank CPU loader scans, exact fresh-process replay, and source/language
exposure. Benchmark matching removed 3,838 candidate documents before joint exclusion; those are
conservative matches, not proven leakage. Joint exclusion subsequently removed 19 code documents.
These finite pilot counts do not establish eligible supply for the main training horizon.

## Main-corpus headroom

The [initial content audit](../experiments/corpus-audit/README.md) now adds a reproducible
924-document review packet, 34 qualitative excerpt reviews, a full FineMath host census, and
document-length measurements. It identifies concrete selection questions; it does not certify
corpus quality. Main training needs a labeled quality/coverage audit and one costed data comparison
alongside the supply and integrity checks below. The frozen engineering pilot remains unchanged.
The [follow-up decisions](../experiments/corpus-audit/README.md#follow-up-decisions--2026-09-18)
keep FineMath, prepare a reversible 8,563-document topic-directory exclusion candidate, and
prioritize consistency checks on English L3 Q&A before any synthetic-source replacement.
No blanket numeric-deduplication rule has been adopted. The main mixture below is a preparation
hypothesis; it has not been admitted or materialized as a training corpus.
The [web candidate shortlist](../experiments/corpus-audit/README.md#candidate-decisions) explicitly
prioritizes natural Ultra-FineWeb qualification following the
[matched paper results](research.md#openbmb-web-data-review--2026-09-19), with FineWeb-Edu as
the control, DCLM baseline/DCLM-Edu as independent candidates, and synthetic L3 assessed separately.
Their historical pins match current repository heads as checked September 19. None is in the
frozen pilot; the main mixture still needs a comparable content/coverage audit and source eligibility.

The [coding plan](coding.md) now prioritizes a checked-exercise comparison. A pinned, bounded
UltraData-Code preview establishes schema and serialization questions, not training eligibility
or correctness. The [natural-code cohort](../experiments/corpus-audit/natural-code-cohort.json)
now has 16 immutable byte matches and ten preliminary review candidates; none is admitted.
Next independently check the three practical candidates named in [coding](coding.md).

The [September 18 reopen](../experiments/pilot/supply.json) verifies all shard hashes and every document-index span for the five
retained token stocks used by the pilot. It counts the larger local `document-token-stock-v2`
peS2o-v3 bank once; the earlier local stock version is not additional supply. Stack-Edu below is the archived acquisition receipt, before full exclusion.

| Source | Retained tokens | Pilot share |
| --- | ---: | ---: |
| FineWeb-Edu | 2,306,703,052 | 50% |
| Stack-Edu | 476,774,847 | 15% |
| FineMath 4+ | 1,124,167,472 | 15% |
| Cosmopedia v2 | 1,489,288,743 | 10% |
| peS2o v3 | 820,097,493 | 5% |
| FineWiki English | 582,070,378 | 5% |

These are 6,799,101,985 source tokens before joint eligibility, not a 6.8B training manifest.
At the frozen pilot weights, total code supply gives a 3.18B-token single-pass mixture upper bound;
the 5% Go share within code lowers it to **2,755,093,600 total mixture tokens**. Validation reserves,
benchmark exclusions, and joint duplicates reduce that ceiling. FineWeb's corresponding ceiling
is 4.61B. A long run needs additional approved source acquisition, explicitly permitted repetition,
or a separately frozen mixture; existing specialist stocks cannot fill arbitrary missing code/web
quotas. Natural UltraData-Math remains separate and is not silently added to this mixture.

At uint16 storage, each billion packed tokens requires about 2 GB for token IDs, before indexes,
validation, preparation intermediates, and duplicate databases. Keep raw acquisition, exclusion
outputs, packed data, and recovery checkpoints separately budgeted. The completed rental used the small pilot directory and
assistant pack; main-run storage must be measured separately.
Choose the main horizon from measured all-in throughput and eligible supply,
then run the same source-rights, joint-exclusion, partition, pack, and full-loader checks at that scale.

## Recipe direction — 2026-09-19

Prioritize data quality, coverage and learnability over additional architecture searches. Keep the
current hybrid while qualifying data and the planned extension from 4K toward approximately 128K.
This is a direction for the main recipe; the pilot weights are not inherited as optimized weights.
The [main mixture and scale plan](../experiments/main-data/README.md) retains a desired 320B base target and 400B stretch:
35% code, 25% math, 30% natural web, 5% reference/science and 5% refined educational web. Prepare
400B eligible unique tokens as selection headroom for 320B exposure (500B for the stretch); the plan distinguishes proposed exposure from
materialized supply and reserves separate context-extension/post-training budgets. These are working
weights to qualify, not proven optima or source admission. The 100B case remains the current
measurement-based budget-fit scenario; 320–400B requires more effective throughput or compute.
The weak [pilot completions](../experiments/pilot/completion-preview.json) establish an immature
endpoint, not a causal verdict on its datasets or architecture.

### What has actually been explored

| Component | Evidence already available | Decision |
| --- | --- | --- |
| FineWeb-Edu, FineMath 4+, Cosmopedia v2, peS2o, FineWiki | Retained stocks, pilot preparation, index census and stratified excerpt audit | Keep useful content and controls; improve selection rather than discard all existing stock |
| Stack-Edu | Retained multilingual code; 16-file audit with exact upstream matches, 13 notices and ten preliminary review candidates | Independently test the three selected practical candidates; freeze expanded exclusions/family splits before derivation |
| Natural Ultra-FineWeb | Paper reviewed, historical/current revision checked; comparable local natural-web content audit still pending | Leading candidate for the main web component |
| DCLM baseline / DCLM-Edu | Historical configurations and current revisions checked | Independent comparison and coverage options; no claimed local quality ranking |
| Ultra-FineWeb-L3 | 96 retained Q&A/multi-style records, length measurements and eight excerpt reviews | Promising synthetic component; check source/answer consistency before replacing Cosmopedia |
| Natural UltraData-Math | Retained index census, length profile and excerpt inspection | Compare with FineMath; qualify selected/refined tiers separately |
| UltraData-Code L2/L3 | Bounded preview, isolated L3 tests and lineage audit | All 16 L3 preview rows remain held; seven passing supplied tests do not establish independent correctness |
| UltraX-Preview; ZGCM-1 | Release/report review | Secondary references, not admitted training supply |

The flagship target is always-thinking agentic coding, general coding, math reasoning and tool
use. High-quality code and reasoning must be present during pretraining, not deferred entirely to
SFT. Prioritize these forms in the main corpus:

- Natural implementation code with useful tests, documentation, API usage and project context;
  preserve Python and other target languages, including JavaScript/TypeScript.
- Correct worked math, derivations and scientific explanations spanning elementary foundations
  through harder problems; inspect intermediate reasoning as well as final answers.
- Checked code explanations, algorithm derivations, debugging/repair examples and exercises with
  independent tests. Preserve mistakes only when clearly identified and followed by valid correction.
- Coherent repository/document bundles for context extension. Complete observed tool trajectories
  belong in the separately serialized reasoning/agent training stage; planning prose alone does
  not establish an agent's ability to inspect, edit, test and recover.

Selected natural web and reference material provide language, knowledge and task diversity. Do not
equate educational score, reasoning length or a passing generated test with quality. Main code/math
exposure should reflect the target capabilities; the pilot's 15%/15% shares are not final quotas.

Post-training is more developed than an empty shortlist: the retained 500,000-row stock already
includes 220,000 UltraData-SFT-2605 reasoning conversations, 110,000 UltraData-SFT-Agent-2609
trajectories, 120,000 Glaive reasoning rows and 50,000 SYNTHETIC-2-SFT-verified rows. These are
source/format inventory counts, not 500,000 independently verified answers. See the
[assistant recipe](assistant.md#main-assistant-data-direction--2026-09-19) for the missing balance
and the existing stock's length-selection bias.

### Focused candidate queue

[recipe-review.json](../experiments/corpus-audit/recipe-review.json) pins the seven newly reviewed
public cards and snapshots existing local SFT receipts. New cards were inspected; new corpus
payloads were not acquired. These revision pins are separate from older stock identities.

| Priority | Source | Specific question |
| --- | --- | --- |
| First: natural web | Natural Ultra-FineWeb versus retained FineWeb-Edu | Does the selected English path/threshold preserve broad topics and useful unique supply? Keep DCLM independent. |
| First: code | Retained Stack-Edu plus checked exercises | Can we improve practical coding with verified task/solution pairs while retaining languages, libraries, tests and documentation? Follow the existing code plan. |
| First: thinking assistant | UltraData-SFT `think`, retained agent traces and selected [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) reasoning components | Qualify code/math reasoning and complete tool trajectories; retain brief reasoning for simple tasks, with no non-thinking mode. |
| First: math | UltraData-Math L2/L3 and [Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) `4plus` against FineMath | Compare extraction, worked-solution correctness and unique source coverage. Nemotron's published continuation comparison uses an 8B model, not ours. |
| Next: documents | English [FinePDFs-Edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu), alongside retained papers/reference material | Inspect educational document coverage, extraction/figure dependencies and intact long-document supply for context extension. Educational filtering alone does not certify long-range coherence. |
| Secondary task/reference pool | [Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT), UltraData-SFT `no_think` | Candidate tasks/reference answers only; any derived thinking examples need verified reasoning. Do not directly import a non-thinking response mode. |
| Later reward training | [UltraData-RL-2609](https://huggingface.co/datasets/openbmb/UltraData-RL-2609) | Task/reference pool for math, STEM, code and document QA; not ready-made successful SFT trajectories. Use only after a useful SFT baseline and checked rewards. |

The intended pretraining composition is selected broad natural text, meaningful code exposure,
math/science, reference/documents and a controlled refined/synthetic component. Preserve everyday,
nontechnical topics and varied prose as well as difficult educational material. Retain source-family
identity across original pages, rewrites, Q&A and instruction derivatives; several dataset names
can represent the same underlying information. Review source-use evidence per component.

### Next deliverables and decisions

1. Produce a bounded natural Ultra-FineWeb/FineWeb-Edu comparison packet using the existing audit
   pipeline. Report accepted unique tokens, language/domain and length distributions, extraction
   defects, overlap and source eligibility. Inspect differing cutoff choices before acquisition.
2. In assistant preparation, audit retained code/math reasoning and tool trajectories, including
   useful brief reasoning. Include complete examples in each available length band: <=4K, 4–16K,
   16–32K and 32–128K. Keep longer examples separately. Prioritize answer correctness, useful
   reasoning, tool-result consistency, concise ordinary assistance and source identity.
3. Complete the already planned practical-code qualification. Admit only traceable material with
   independent checks; do not turn every source on the shortlist into a separate GPU experiment.
4. Recount usable supply and freeze token-based domain, source, length and reasoning-depth weights
   for the relevant phase. SFT needs both supervised-token and total-context counts. Final
   percentages depend on those measurements; the working allocation guides preparation.
   Training stock is not the same as exposure.
5. Price one controlled comparison from a useful common base with a fixed evaluation protocol.
   Track capability versus trained tokens and GPU-hours, plus preparation/verification cost and
   regressions by capability. Better data can improve learning per token without raising tokens/s.

Plan broad base training, then measured context extension toward approximately 128K with short
replay, followed by an SFT recipe spanning short and long interactions. Exact extension stages,
budgets and any later preference/RL phase remain to be frozen. Long-context qualification must
measure retrieval across positions, cross-document reasoning, sustained generation and short-task
retention, alongside memory/runtime. A configured maximum alone does not establish usable context.

## Artifact discipline

Keep source revisions, filters, counts, hashes, tokenizer identity, data order, and output locations
in each run's manifests. Keep runtime data/checkpoints/logs outside Git, and back up irreplaceable
checkpoints before dependent work. Preserve failed attempts. Corpus text and packed shards are not
redistributed as model-release artifacts. Exact/near-duplicate and benchmark exclusion remain
requirements even though their old administrative workflow has been retired.
