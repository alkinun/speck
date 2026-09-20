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
The [first natural-web inspection](../experiments/corpus-audit/NATURAL_WEB.md) now prioritizes
the newer L1-derived HQ route for its observed URL/WARC metadata. Six hash-verified shards
provide 144,876 inspected records; default English exposes no equivalent original-page identity.
High-scoring extraction defects and template-heavy examples reinforce separate quality controls.
This is source qualification progress, not a demonstrated learning-quality win or training admission.
The [follow-up inventory and DCLM preview](../experiments/corpus-audit/WEB_INVENTORY_DCLM.md)
completes the HQ listing and the 959 MB acquisition: 290,761 documents, 192 stratified samples and
24 reviewed texts/excerpts. High-score extraction defects and lower-score coverage candidates leave
cutoff selection open. Both DCLM variants expose URLs and
document IDs in their partial viewer indexes. DCLM-Edu's integer and continuous score cutoffs
select different records; any stricter candidate must name the exact field and operator.
The [extraction follow-up](../experiments/corpus-audit/web-filter-validation.json) joins three
problematic HQ texts to their exact archived captures, confirming missing text and mixed page
boundaries alongside image dependency. A fresh 32-document comparison shows false alarms and
missed defects; all candidate flags remain review-only. Sample token retention is measured,
but eligible full-corpus supply and a better score cutoff remain unestablished.

The [coding plan](coding.md) treats checked exercises as one candidate for the pretraining recipe
comparison before the main base allocation. Its [completed qualification summary](coding.md#completed-code-qualification)
links the provenance, practical CPU and bundle checks. Natural-code eligibility and independent
exercise verification remain separate; no inspected candidate is admitted by these checks.

The [retained-code census](../experiments/corpus-audit/code-supply.json) verifies 714,369 files /
476,774,847 tokens before joint eligibility: only 1.59% of the proposed 30B natural-code exposure.
The [fixed Stack-Edu cohort](../experiments/corpus-audit/code-yield-result.json) has 138 files /
459,615 tokens and 31 known-family holds. The [broader Stack v3 preflight](../experiments/corpus-audit/stack-v3-broader.json)
acquired sixteen groups, reconciling 29,347 repository rows / 379,942 entries, then selected
44 repositories / 80 files. Screening records four content flags and eight known-family holds; the rest
remain unresolved. Earlier Stack v3 origin checks explain twelve selected transformations, not
all supplied-text mismatches or source-use applicability.

The [current cohort assessment](../experiments/corpus-audit/stylesheet-cohort-review.json) records
**176 full texts / 412,974 tokens reviewed**, **44 family-held records**, and **no unheld records still
unread**. Original sampling factors and source labels are preserved; observed language, dialect
and file role are recorded separately. A family hold alone does not establish an exact benchmark
match or justify replacing a sampled record. All 174 currently unheld records are read; the other
two full reads are now held, so reading and hold counts overlap.

The [coding guide](coding.md#common-cohort-review) summarizes the frozen readings, redaction
checks and package/family evidence. The final stylesheet readings distinguish page, template and
component roles. The [notice follow-up](../experiments/corpus-audit/code-notice-provenance.json)
adds seven verified host origins and reconfirms one, recovering Objective-C repository and Eclipse
notice context. Eclipse declares separate code/non-code licenses; template attribution and restrictive
Java wording remain unresolved. The [pinned-origin follow-up](../experiments/corpus-audit/pinned-code-origins.json)
adds 42 verified origins. The subsequent [retained-origin recovery](../experiments/corpus-audit/retained-code-origins.json)
adds ten exact-match Stack-Edu origins, bringing coverage to 28/104 unheld Stack-Edu and 68/70 unheld
Stack v3 records. GitHub quota exhaustion leaves 75 Stack-Edu records unresolved, separately from
one prior Stack-Edu failure and two pinned Stack v3 source 404s. The C++-only successful prefix is
not a representative recovery rate. Pause quota-blocked requests until access changes and continue
independent web/math qualification. Fixed-cohort reading is complete; source-use and broader family
qualification remain open, including four path-verified Stack v3 ancestor-notice searches.
Parser success and passing test assertions cannot certify behavior, semantic preservation or eligibility.
Complete source-use, provenance and family checks in the
[qualification packet](../experiments/main-data/QUALIFICATION.md) before selecting finite experiment
inventories. No source ranking, eligible yield or bulk-filter adoption follows from this completed reading pass.

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

The first model and paper center on data quality, coverage and learnability through pretraining,
mid-training and post-training. Keep the selected model fixed; attention/size rationale and measured
trade-offs have a bounded 200-hour study before backbone freeze. Broader architecture research
belongs to later releases; data experiments have a separate 900-hour allocation.
The [program lifecycle](program.md#training-lifecycle) defines capability continuation, context
16K/32K context qualification with 128K stretch, thinking SFT and conditional RL with separate data objectives.
This is a direction for the main recipe; the pilot weights are not inherited as optimized weights.
The [main mixture and scale plan](../experiments/main-data/README.md) uses a 100B working base horizon:
35% code, 25% math, 30% natural web, 5% reference/science and 5% refined educational web. Prepare
125B eligible unique tokens as selection headroom for 100B exposure; the plan distinguishes proposed exposure from
materialized supply and reserves separate context-extension/post-training budgets. These are working
weights to qualify, not proven optima or source admission. The H100 reference projects to 2,160 base GPU-hours;
GH200 qualification must confirm the cost or reduce the horizon. The old 320–400B ambitions are
deferred scale comparisons, not first-allocation targets.
The weak [pilot completions](../experiments/pilot/completion-preview.json) establish an immature
endpoint, not a causal verdict on its datasets or architecture.

### What has actually been explored

| Component | Evidence already available | Decision |
| --- | --- | --- |
| FineWeb-Edu, FineMath 4+, Cosmopedia v2, peS2o, FineWiki | Retained stocks, pilot preparation, index census and stratified excerpt audit | Keep useful content and controls; improve selection rather than discard all existing stock |
| Stack-Edu | Full retained census: 714,369 files / 476.8M tokens; four-repository linkage follow-up and bounded earlier execution checks | Measure expansion yield by language/practical role; require explicit linkage, independent tests and family exclusions |
| Natural Ultra-FineWeb | HQ inventory, twelve-shard census, stratified review and matching-capture extraction follow-up complete | Qualification candidate; no stricter cutoff or automatic review-flag rejection |
| DCLM baseline / DCLM-Edu | Pinned partial-viewer schema/content previews; score predicates distinguished | Source-file evidence needed for concrete coverage decisions; no local quality ranking |
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
source/format inventory counts, not 500,000 independently verified answers. The
[data-readiness closeout](../experiments/corpus-audit/data-readiness.json) adds full format checks,
sampled tool-aware lengths and RL prompt links. See the
[assistant recipe](assistant.md#main-assistant-data-direction--2026-09-19) for the missing balance
and the existing stock's length-selection bias.

### Focused candidate queue

[recipe-review.json](../experiments/corpus-audit/recipe-review.json) pins the ten newly reviewed
public cards, snapshots existing local SFT receipts and links bounded viewer-sample diagnostics.
The viewer samples are separate from bulk acquisition and older retained-stock identities;
card revision pins do not establish the revision served by the live viewer.

| Priority | Source | Specific question |
| --- | --- | --- |
| First: natural web | Natural Ultra-FineWeb versus retained FineWeb-Edu | Does the selected English path/threshold preserve broad topics and useful unique supply? Keep DCLM independent. |
| First: code | Retained Stack-Edu, Stack v3 and qualified checked exercises | Can we improve practical coding with verified task/solution pairs while retaining languages, libraries, tests and documentation? Follow the existing code plan. |
| First: thinking assistant | UltraData-SFT `think`, retained agent traces and selected [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) reasoning components | Qualify code/math reasoning and complete tool trajectories; retain brief reasoning for simple tasks, with no non-thinking mode. |
| First: math | UltraData-Math L2/L3 and [Nemotron-CC-Math-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1) `4plus` against FineMath | Compare extraction, worked-solution correctness and unique source coverage. Nemotron's published continuation comparison uses an 8B model, not ours. |
| Next: additional math candidates | Synthetic [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) and filtered web [InfiWebMath 3+](https://huggingface.co/datasets/HuggingFaceTB/finemath/tree/main/infiwebmath-3plus) | Measure source provenance, teacher provenance for generated solutions, independent correctness, contamination, length tails and cross-source overlap. These remain review candidates outside the retained training inventory. |
| Next: staged mixture | The [Cagliostro v3 recipe](research.md#cagliostro-v3-review--2026-09-20) and its late math reweighting | Test phase-conditioned weights only as a predeclared replacement or later ablation with schedule, total exposure and validation fixed. Do not add a fourth screening arm or attribute a cooldown result to data alone. |
| Next: documents | English [FinePDFs-Edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu), alongside retained papers/reference material | Inspect educational document coverage, extraction/figure dependencies and intact long-document supply for context extension. Educational filtering alone does not certify long-range coherence. |
| Secondary task/reference pool | [Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT), UltraData-SFT `no_think` | Candidate tasks/reference answers only; any derived thinking examples need verified reasoning. Do not directly import a non-thinking response mode. |
| Later reward training | [UltraData-RL-2609](https://huggingface.co/datasets/openbmb/UltraData-RL-2609) | Task/reference pool for math, STEM, code and document QA; not ready-made successful SFT trajectories. Use only after a useful SFT baseline and checked rewards. |

The bounded [OpenMathInstruct-2 sample receipt](../experiments/corpus-audit/openmathinstruct2-sample.json)
checks 80 fixed-offset rows for schema, source strata, lengths and boxed-answer self-consistency.
It finds zero exact normalized matches against the pinned 1,319-row GSM8K evaluation file, while
10 sampled rows carry GSM8K-derived source labels. This is diagnostic evidence only: MATH, derived
variants, semantic overlap, independent correctness and source-use decisions remain open.

The [FineMath/InfiWebMath diagnostic](../experiments/corpus-audit/finemath-config-sample.json)
checks 64 rows from each configuration and finds no exact normalized text or literal URL overlap
between these samples. FineMath exposes top-level crawl/WARC and language fields absent from the
InfiWebMath schema; missing fields do not establish missing provenance in nested metadata.
The sampled continuous-score minima are 3.5 for FineMath 4+ and 2.546875 for InfiWebMath 3+:
configuration labels must not be implemented as equivalent continuous-score cutoffs.
Reported token counts use upstream metadata, not the Speck tokenizer. These small, fixed-offset
samples establish neither population overlap nor relative quality. Next, compare pinned source
files, join held-out families and review complete worked solutions with independent checks.

The filtered [InfiWebMath arithmetic diagnostic](../experiments/corpus-audit/infiwebmath-4plus-arithmetic.json)
adds a conservative independent numeric check over 3,725 qualified sample records. It parses 276
standalone numeric equalities: 260 agree within displayed rounding and 16 are flagged across six
records. The [manual review](../experiments/corpus-audit/infiwebmath-4plus-arithmetic-review.json)
classifies the flags as binary notation, intentional counterexamples, puzzle notation, extraction
truncation and one likely source transcription error. These classifications are triage evidence,
not a correctness rate; symbolic, unit-bearing and multi-line reasoning remain unchecked. The
source-readiness matrix records InfiWebMath as a separate non-admitted candidate with source-use,
family, contamination, correctness and supply gates open.

The shared [stage-conditioned data-design contract](../experiments/main-data/data-design-contract.json)
now binds the pretraining, mid-training and post-training packets. It standardizes manifest fields for
stage, source family, transformation, quality, coverage, dependency, contamination and lineage, while
keeping natural, grounded, synthetic, repository-event and agent-trajectory banks distinct. This is a
design requirement for the next source-mapping pass; it does not change the working mixture, admit a
source or authorize a run.

The [source-mapping receipt](../experiments/main-data/frontier-data-source-mapping.json) now records
which existing audits support each reviewed hypothesis and which readiness gates remain open. It is
an evidence index, not a source-admission decision.

The [candidate manifest preflight](../experiments/main-data/candidate-manifest-preflight.json) applies
the shared fields to all 12 candidates. It records five pinned identities, zero complete manifests and
zero established eligible tokens, so the next work is qualification rather than training.

The first source-specific application is the [natural-web candidate manifest](../experiments/main-data/natural-web-candidate-manifest.json).
It records the FineWeb-Edu control, the Ultra-FineWeb HQ route, their bounded exact-overlap result and
the evidence still required before the web contrast can enter screening.

The [natural-code candidate manifest](../experiments/main-data/natural-code-candidate-manifest.json)
binds the retained Stack-Edu census, the bounded Stack v3 cohort and the unavailable checked-code
route. It keeps natural code separate from verified exercises and records the current zero-eligible-
token boundary.

The [math candidate manifest](../experiments/main-data/math-candidate-manifest.json) keeps natural
worked material, filtered web math, refined/generated solutions and unavailable sources separate. The
InfiWebMath arithmetic review remains a triage diagnostic; it does not certify solution correctness or
eligible supply.

The [post-training candidate manifest](../experiments/main-data/post-training-candidate-manifest.json)
binds the 500K-row assistant census and 44,369 reward-prompt rows. Structural serializability,
independent outcomes, tool trajectories and fixed-policy RL feasibility remain separate gates.

The intended pretraining composition is selected broad natural text, meaningful code exposure,
math/science, reference/documents and a controlled refined/synthetic component. Preserve everyday,
nontechnical topics and varied prose as well as difficult educational material. Retain source-family
identity across original pages, rewrites, Q&A and instruction derivatives; several dataset names
can represent the same underlying information. Review source-use evidence per component.

## Data preparation closeout — 2026-09-20

The [data closeout receipt](../experiments/main-data/data-closeout.json) freezes the evidence-only
preparation pass. Retained inventories, web/code/math qualification packets, assistant and reward
format audits, candidate card reviews, bounded diagnostics and the three design-only study packets
are complete for the current local evidence. No source is admitted: rights, family partitions,
independent correctness, finite eligible supply and GH200 cost remain explicit gates.

The supplied frontier-data research is now recorded in the structured synthesis and linked from the
closeout. The next data change is source-specific mapping of those hypotheses into pinned candidate
manifests. Preserve existing receipts, attach new claims to a sampling frame and baseline, and
revise source readiness only when the five gates are supported. The closeout does not authorize new
acquisition, repetition, GPU work or training.

### Next deliverables and decisions

1. Complete origin/source-use and semantic-quality assessment on the
   [frozen stratified sample](../experiments/corpus-audit/code-yield-result.json). Its first pass
   recovers/screens 138 files across 11 languages; no-hit records remain unresolved. Natural code
   needs traceable, useful content; verified exercises additionally need test linkage and independent
   oracles. Record weighted yield only after assessing the gates. The 0.477B retained tokens
   do not establish the proposed 30B natural-code exposure or the separate checked-exercise bank.
   For web, resolve source-use evidence, source families, overlap and source-aware extraction
   repair before counting accepted unique tokens. Keep candidate flags review-only and the HQ
   cutoff unchanged. Exact sample-token retention is not full-corpus yield. Extend DCLM previews
   into comparable source-file evidence only for a concrete coverage/eligibility question before
   freezing allocations; no unqualified split or stricter cutoff inherits a quality advantage.
2. Use the completed format/context audit to verify retained code/math reasoning and tool trajectories, including
   useful brief reasoning. Include complete examples in each available length band: <=4K, 4–16K,
   16–32K and 32–128K. Keep longer examples separately. Prioritize answer correctness, useful
   reasoning, tool-result consistency, concise ordinary assistance and source identity.
3. Extend the completed bounded practical-code checks to corpus-scale source eligibility and
   benchmark/family exclusions. Admit only traceable material with independent checks; do not
   turn every source on the shortlist into a separate GPU experiment.
4. Recount usable supply and freeze token-based domain, source, length and reasoning-depth weights
   for the relevant phase. SFT needs both supervised-token and total-context counts. Final
   percentages depend on those measurements; the working allocation guides preparation.
   Training stock is not the same as exposure.
5. Price and run a bounded baseline/candidate comparison from matched fresh initializations before
   main pretraining. Freeze the contrast after source qualification and record the recipe decision
   before launch. Track held-out source loss and development capability versus tokens/GPU-hours,
   preparation cost and regressions. Audit post-training sources now; evaluate their training
   recipes on useful parent checkpoints before committing the downstream stage budgets.

Plan broad pretraining, capability-focused mid-training within the base horizon, then measured
context mid-training toward 16K/32K, with 128K stretch, with short replay, followed by an SFT recipe spanning
short and long interactions. Preserve source-family identity and exclusions across every stage,
including derived exercises, teacher traces and RL prompts. Exact continuation/extension stages,
budgets and any later preference/RL phase remain to be frozen. Long-context qualification must
measure retrieval across positions, cross-document reasoning, sustained generation and short-task
retention, alongside memory/runtime. A configured maximum alone does not establish usable context.

## Artifact discipline

Keep source revisions, filters, counts, hashes, tokenizer identity, data order, and output locations
in each run's manifests. Keep runtime data/checkpoints/logs outside Git, and back up irreplaceable
checkpoints before dependent work. Preserve failed attempts. Corpus text and packed shards are not
redistributed as model-release artifacts. Exact/near-duplicate and benchmark exclusion remain
requirements even though their old administrative workflow has been retired.
