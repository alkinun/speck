# Data

This document owns the data contract for the 5,000 GPU-hour program: the shared pipeline, where
natural, derived and synthetic records may enter, the retained inventory, and the code route. The
goal and next work are defined in [PLAN.md](../PLAN.md), with the complete outline in
[the program overview](program.md), which owns stage boundaries and budgets. The engineering pilot
is complete. Prepare the main corpus against the working mixture and actual eligible supply; frozen
pilot shares are historical. No document here authorizes a training run.

## Pipeline

Pin source revisions and permitted use → acquire/filter → deduplicate and exclude evaluation
material → partition → tokenize/pack → verify → train. Source-separated shards allow later mixture
changes without retokenizing. Recheck tokenizer fingerprints and hashes whenever reusing stock.

Every record follows the same auditable path, at every stage:

1. identify the source, licence or use constraint, family, acquisition receipt and transformation history;
2. extract with a source-specific parser, retaining the original text or artifact identity;
3. remove exact duplicates, near duplicates, template copies and semantic repeats, while preserving
   a family graph for held-out splits;
4. apply quality, language, safety, benchmark-contamination and source-coverage filters;
5. attach lineage, quality dimensions, derivation cost and accepted-token accounting;
6. partition by source family before tokenization and packing;
7. verify manifests, token counts, packing, sampling weights, checksums and resumability;
8. train only from the hash-bound manifest, reporting exposure, replay, discarded data and
   evaluation results separately.

AI-assisted filtering may recommend keep/remove decisions, but pretraining retains the original
source and records the model, prompt, version and decision. Generated text is a derived record and
must never be indistinguishable from source text in a manifest.

**Where derived text may enter.** The broad production pretraining mixture is natural and
source-traceable. Late pretraining decay is the *only* initial pretraining experiment that can admit
derived text, and it must stay separately identified by lineage, teacher/generator, verification
result and cost. Synthetic material is most acceptable in post-training, where it is generated
inside pinned, testable environments or used for self-distillation; it remains separately attributed
and earns its place on held-out transfer, not training reward. Derived data never compensates for
missing natural supply.

**Admission gates.** Before compute is committed, each stage must have a pinned manifest, a
family-disjoint evaluation split, a contamination report, accepted-token or example accounting,
reproducible checksums, measured throughput, a retry/recovery policy and a declared stop rule. The
order is: close source and manifest readiness including the natural-code gap; run the pretraining
data screen and freeze one recipe; qualify changed-data continuation and run the bounded
mid-training comparisons; qualify SFT outcome checks and RL infrastructure; launch production only
from the selected, hash-bound recipes.

| Responsibility | Code / command |
| --- | --- |
| Source readers, configuration, packing, resume | `speck/data/{acquisition,configuration,packing,dataset}.py` |
| Global disk-backed deduplication | `scripts.production_data_preprocess` (batched MinHash is the default; `--per-shingle-minhash` is a slower bitwise-identical fallback) |
| Secret filtering, near duplicates, contamination | `scripts.text_gitleaks_filter`, `text_near_duplicates`, `text_contamination` |
| Source-use review | `scripts.source_rights_review` (validates a pending human template; never makes an approval decision). Decided 2026-09-22: [acceptance record](../experiments/main-data/source-rights-acceptance.json) |
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

Code supply is the binding constraint on the whole horizon and has its own section below; see
[code priority and qualification](#code-priority-and-qualification) for the census, the cohorts and
the gates. In short: the retained stock covers 1.36% of the natural-code preparation target, and
checked code stopped being a declared bank in the
[2026-09-22 re-freeze](../experiments/main-data/README.md#base-mixture).

The fixed code cohort's reading pass is complete. Its holds, origin and notice recovery, and
family-partition state are recorded once in the
[qualification packet](../experiments/main-data/QUALIFICATION.md), which the conditional source-use
decision for both code routes depends on. Parser success and passing test assertions cannot certify
behavior, semantic preservation or eligibility, and no source ranking, eligible yield or bulk-filter
adoption follows from the reading pass.

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

These are 6,799,101,985 source tokens before joint eligibility, not a 6.8B training manifest; the
[supply gap](../experiments/main-data/supply-gap.json) adds the distinct HQ web stock and assigns
stock to the six current banks.
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

The first model and paper center on data quality, coverage and learnability through all six
training stages. The selected model is fixed by declaration; the bounded 200-hour attention/size
study was deferred on 2026-09-21 to a later allocation, which released 180 of its hours to data
research and 20 to reference-only efficiency profiling. Architecture research belongs to later
releases; data experiments have a 1,230-hour allocation.
The [program lifecycle](program.md#training-lifecycle) defines capability continuation, 16K/32K
context qualification, thinking SFT and conditional RL with separate data objectives. 64K/128K is
deferred until a later measured revision.
This is a direction for the main recipe; the pilot weights are not inherited as optimized weights.
The [main mixture and scale plan](../experiments/main-data/README.md) uses an 80B working base horizon:
35% code, 25% math, 30% natural web, 5% reference/science and 5% refined educational web. Prepare
100B eligible unique tokens as selection headroom for 80B exposure; the plan distinguishes proposed exposure from
materialized supply and reserves separate context-extension/post-training budgets. These are working
weights to qualify, not proven optima or source admission. The H100 reference projects to 1,728 base GPU-hours;
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
This queue predates the 2026-09-22 re-freeze. The
[source registry](../experiments/main-data/source-registry.json) now fixes the nine selected
sources; checked exercises, UltraData-Math L2 and Nemotron-CC-Math are not selected, so their rows
below are review questions for a later revised freeze, not current candidates.

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
[source-readiness matrix](../experiments/main-data/source-readiness.json) records InfiWebMath as a
separate non-admitted candidate and owns its gate status.

Design records for this evidence:

- The [data-design contract](../experiments/main-data/data-design-contract.json) fixes the
  manifest fields every stage records: stage, source family, transformation, quality, coverage,
  dependency, contamination and lineage.
- The [source-mapping receipt](../experiments/main-data/frontier-data-source-mapping.json) indexes
  which audits support each reviewed hypothesis; it closes no gate.
- The [source-readiness matrix](../experiments/main-data/source-readiness.json) owns per-source
  gates and manifest-field completeness; it finds zero complete manifests and zero eligible tokens.
- The [natural-web](../experiments/main-data/natural-web-candidate-manifest.json),
  [natural-code](../experiments/main-data/natural-code-candidate-manifest.json),
  [math](../experiments/main-data/math-candidate-manifest.json) and
  [post-training](../experiments/main-data/post-training-candidate-manifest.json) candidate
  manifests add domain evidence and comparison contracts for those sources.

The [post-training research synthesis](../experiments/main-data/post-training-research.json) keeps
general SFT, reasoning SFT, agent trajectories, RL prompts, preference/critique data and on-policy
teacher feedback as separate banks. It retains the always-thinking release baseline while defining
a bounded check for hybrid or token-budget control. Correctness remains primary in RL; any efficiency
preference is delayed, soft and task-conditioned, with truncation, shortcut and verifier-hacking checks.

The intended pretraining composition is selected broad natural text, meaningful code exposure,
math/science, reference/documents and a controlled refined/synthetic component. Preserve everyday,
nontechnical topics and varied prose as well as difficult educational material. Retain source-family
identity across original pages, rewrites, Q&A and instruction derivatives; several dataset names
can represent the same underlying information. Carry each component's recorded source-use
conditions.

## Data preparation closeout — 2026-09-20

The [data closeout receipt](../experiments/main-data/data-closeout.json) freezes the evidence-only
preparation pass. Retained inventories, web/code/math qualification packets, assistant and reward
format audits, candidate card reviews, bounded diagnostics and the three design-only study packets
are complete for the current local evidence. No source is admitted: rights, family partitions,
independent correctness, finite eligible supply and GH200 cost remain explicit gates.

The supplied frontier-data research is recorded in the structured synthesis and linked from the
closeout. Attach new claims to a sampling frame and baseline, and revise source readiness only when
the five gates are supported. The closeout does not authorize new acquisition, repetition, GPU work
or training.

### Order of work

[PLAN.md](../PLAN.md#immediate-order-of-work) owns the order of data work. The methods it applies
are in this guide: count accepted unique tokens only after family, overlap and extraction review;
keep candidate flags review-only; include complete assistant examples in each length band (<=4K,
4–16K, 16–32K and 32–128K); keep training stock, supervised tokens and exposure as separate counts;
and choose recipes from matched fresh initializations before main pretraining.

Plan broad pretraining, then separately budgeted capability/context/agentic mid-training toward
16K/32K, using short replay, followed by an SFT recipe spanning
short and long interactions. Preserve source-family identity and exclusions across every stage,
including derived exercises, teacher traces and RL prompts. Exact continuation/extension stages,
budgets and any later preference/RL phase remain to be frozen. Long-context qualification must
measure retrieval across positions, cross-document reasoning, sustained generation and short-task
retention, alongside memory/runtime. A configured maximum alone does not establish usable context.

## Code priority and qualification

Agentic coding, general coding and math reasoning are the primary first-release targets. Code
quality is a pretraining requirement, not only a post-training concern: natural code, tests,
documentation and correct worked explanations should establish useful foundations before
reasoning-SFT. Preserve practical API use, debugging and repository relationships alongside
algorithmic exercises. The pilot's code weight is an engineering setting, not the target.
Long-context preparation retains coherent repository units and dependencies for 16K/32K
qualification; an arbitrary concatenation of unrelated files is not repository-level supervision.
Split original repositories and derived tasks together to protect held-out repair and agent
evaluations.

**Qualification rules.** Natural source code needs immutable identity, applicable source-use
evidence, intact useful content, family/benchmark exclusions and joint deduplication. It does *not*
need to pass invented tests to be natural-code material. Verified exercises additionally require
clear specifications, same-revision implementation/test linkage, bound dependencies, independent
oracles and deliberately wrong controls; tests generated alongside a solution establish
self-consistency only. Keep natural-code and verified-exercise outcomes separate throughout
preparation and experiments. Preserve original bytes and notices, upstream content IDs,
repository/commit/file identities and consumed-text hashes. Encoding changes, redaction and notebook
cleanup require explicit provenance; never disable exact-byte checks to accommodate an unexplained
mismatch. Related originals, forks, rewrites, patches and exercises belong to the same
exclusion/partition family. A clean bounded screen or a publisher licence label alone authorizes
nothing.

**Supply position.** The [retained-stock census](../experiments/corpus-audit/code-supply.json)
verifies 714,369 files / 476,774,847 tokens before joint eligibility across 11 languages: 1.36% of
the 35B natural-code preparation target, which makes natural code the bank where the whole horizon
binds. That stock supports at most **1.36B total one-pass tokens at 35% natural code**, before
exclusions, validation and other bank constraints. The
[corpus-audit record](../experiments/corpus-audit/README.md) owns the per-batch narrative of the
yield preflight, the Stack v3 cohort and the reading pass; the
[qualification packet](../experiments/main-data/QUALIFICATION.md) owns the current cohort holds and
origin/notice recovery. Repetition is not an approved way to fill the gap.

**First comparison to prepare.** The
[research design](../experiments/main-data/README.md#research-before-the-main-run) owns the
screening/confirmation matrix, controls, cost caps and decision rules. Screening permits one
code-bank candidate, one natural-web candidate and one calibrated AI-generation/provenance-filter
candidate, each against the same qualified baseline. The code candidate is the natural-source
contrast (Stack-Edu versus Stack v3). Keep total code share, non-intervened language coverage,
non-code banks and serialization fixed. Checked code is no longer a declared bank, so do not add it
back as an extra arm or fill a code quota by repeating a tiny bank. A source swap does not test
repository packing, and checked-code substitution would not isolate synthesis, selection and
verification separately.

**Evidence for a coding claim.** Keep the pilot's compiled HumanEval+ metric as a continuity check;
33 development tasks cannot establish broad coding strength. Prepare a separate pinned protocol
before admitting new data:

| Dimension | Next evidence |
| --- | --- |
| Python generation | HumanEval+ continuity plus MBPP+ through a declared [EvalPlus](https://github.com/evalplus/evalplus) protocol |
| Language coverage | Selected [MultiPL-E](https://github.com/nuprl/MultiPL-E) languages, reporting each separately |
| Harder generation and repair | A fixed release/date window of [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench), with explicit test variant and output limits |
| Practical use | Bounded held-out tasks for library use, debugging and repair, scored against independent hidden tests |
| General usefulness | Existing math, instruction following, knowledge/reasoning checks, and per-source held-out loss |

These are planned, not implemented or scored. Freeze task-family partitions and exclusion identities
before data selection, fit prompts and output within the context budget, pin Python/library
versions, and re-run reference models under the same protocol. Report denominators, uncertainty,
execution failures, output tokens and runtime; do not tune against the final partition or equate our
compiled metric with the official leaderboard.

**Repository change data.** Qualify 8–16 repository repair cases before any bulk route. Retain
origin, licence evidence, immutable parent/fix commits, issue specification, changed files, patch
and environment/dependency identities. Exclude benchmark/task families before acquisition,
deduplicate commits that also occur in pull requests, and group related files by repository for
partitioning. Prevent post-fix state or hidden-test answers from leaking into task inputs. Run
checks only in the existing sandbox: at least one relevant test must fail before the fix and pass
after it, while declared regression tests keep passing. Verify empty-patch failure and
reference-patch success repeatedly; reject flaky or underspecified cases, and record
environment/setup failures separately from incorrect solutions. Before training on change examples,
specify how pre-change context is loss-masked and patches are supervised, and audit
processed-context and supervised-token totals independently — the current plain pretraining pack
does not implement that objective.

## Artifact discipline

Keep source revisions, filters, counts, hashes, tokenizer identity, data order, and output locations
in each run's manifests. Keep runtime data/checkpoints/logs outside Git, and back up irreplaceable
checkpoints before dependent work. Preserve failed attempts. Corpus text and packed shards are not
redistributed as model-release artifacts. Exact/near-duplicate and benchmark exclusion remain
requirements even though their old administrative workflow has been retired.
