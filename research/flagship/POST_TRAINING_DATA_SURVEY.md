# Flagship instruction-data survey

Status: research shortlist v0, 2026-09-11; [SPE-176](https://linear.app/openspecklabs/issue/SPE-176).
The owner selected Base + Instruct; Think and dual-mode models are deferred. This survey informs
[POST_TRAINING.md](POST_TRAINING.md). It does not approve a corpus, prescribe final
sampling weights, or authorize training. The early SpeckChat sources remain useful candidates.

## 1. Scope and evidence

Reviewed both checked-in builders and pinned release cards, all ten distinct upstream SpeckChat
sources, their important parents, and newer instruction/grounding/preference alternatives. The saved
inventory covers 32 dataset repositories. Supplemental component cards, including SciRIFF-train-mix
and Tulu Personas IF, and references reviewed during pipeline design supply additional context.

Collected 158 viewer rows and 13 direct JSONL rows for bounded illustrative inspection. Viewer
responses have `x-revision` receipts; nonempty pages matched the card revision. Direct samples use
immutable file URLs. Inspected schemas and selected response excerpts, with targeted checks of the
examples below. This is not a representative quality audit or a measured acceptance-rate comparison.
Some discovery offsets use approximate card counts, and some endpoints failed or returned empty pages;
all retained response pages and errors remain in the inventory. No missing page is counted as inspected
data. Header examples and different configurations are not pooled into population estimates.

Artifacts: `~/.cache/speck/evaluations/posttraining-dataset-survey-20260911/`; identities and artifact
digest: [post_training_dataset_survey_v0.json](post_training_dataset_survey_v0.json). These are source
discovery snapshots, not production data manifests. No source text is added to Git.

## 2. Reconstructing the actual SpeckChat mixtures

### SpeckChat1: 300,000 rows

Authority: [builder](../../scripts/speckchat1_prepare.py) and pinned published card.

| Source | Rows | Role |
| --- | ---: | --- |
| `teknium/OpenHermes-2.5` | 180,000 | broad synthetic instruction mixture |
| `Magpie-Align/Magpie-Air-MT-300K-v0.1` | 98,523 | synthetic multi-turn conversations |
| `HuggingFaceH4/no_robots` | 10,000 | human-written demonstrations |
| `allenai/coconot` | 11,477 | contextual noncompliance, uncertainty, unsupported requests |

The builder randomly samples OpenHermes/Magpie, normalizes messages, and retains only outer source
names. It imports No Robots `train+test`, and CoCoNot `original/train`. It does not implement a
corpus-wide near-duplicate firewall, source-specific correctness checks, or length qualification.
Later SFT preparation has its own chat validation and truncation; do not confuse that with verified
demonstration quality. OpenHermes' internal source/category provenance is discarded by conversion.

The 500 No Robots test examples were exposed to the historical SpeckChat1 training path. They are
not independent evaluation for those models. A fresh flagship has separate lineage, but should use
its own family-held-out audit rather than reuse this test set as cross-generation proof.

### SpeckChat2: 500,000 rows

Authority: [builder](../../scripts/speckchat2_prepare.py) and pinned published card.

| Source | Rows | Actual transformation |
| --- | ---: | --- |
| `OpenLeecher/lmsys_chat_1m_clean` | 200,000 | first user prompt + DeepSeek-V3 response |
| `Magpie-Align/Magpie-Llama-3.1-Pro-MT-500K-v0.1` | 130,000 | English, good/excellent inputs, exactly two exchanges |
| `enPurified/Hermes-3-Dataset-enPurified-openai-messages` | 85,000 | prose-filtered Hermes; removes two generic system prompts |
| `enPurified/ultrachat_200k_sft-enPurified-openai-messages` | 65,000 | prose-filtered UltraChat conversations |
| `Magpie-Align/Magpie-Reasoning-V1-150K` | 10,000 | easy/medium: 4K math, 4K code, 2K reasoning examples |
| `HuggingFaceH4/no_robots` | 8,000 | train only, generation-category cap |
| `HuggingFaceTB/everyday-conversations-llama3.1-2k` | 2,000 | strips repetitive opening greetings where matched |

SpeckChat2 adds pinned row provenance, reward/category metadata, normalized prompt deduplication,
source quotas, and source-specific filters. Every accepted conversation must fit 2,048 Mistral-chat
tokens with at most 1,536 assistant tokens. Overlength rows are rejected by the builder, not shortened.

Reward bands are quota-filling preferences, not universal correctness thresholds. LMSYS can fall
through from nonnegative rewards to lower bands; missing scores enter the first band. Magpie MT also
uses bands and category caps. Actual band utilization was not measured in this survey. Prompt hashing
omits assistant turns and catches normalized exact prompts, not semantic variants or all shared
upstream families. Repeated greetings are deliberately limited, not evidence that greetings are useless.

Conclusion: use these source selections as experience, then return to pinned upstream rows. Retaining
the old 2K-filtered output would lose longer useful conversations and preserve old size-specific quotas.

## 3. Disposition of every historical source

### 3.1 OpenLeecher/LMSYS — priority core candidate

[Dataset](https://huggingface.co/datasets/OpenLeecher/lmsys_chat_1m_clean), 273,402 published rows.
Real user prompts were deduplicated, English-filtered, reanswered by DeepSeek-V3 and Phi-3-mini,
reward-scored with Skywork-Reward-Gemma-2-27B-v0.2, and categorized. This contributes realistic prompt
distribution alongside synthetic Magpie conversations. Keep DeepSeek answers that pass fresh checks;
use prompts with weak answers as targeted regeneration candidates.

Important semantics: `grounded` means a determinate answer, including math/trivia, not an answer
grounded in a supplied document. `agreement` is model-judged answer agreement, not external truth.
The card documents classifier errors. Its prompt filters also suppress many repetitive constraint,
extraction, and QA templates, so it cannot supply all precise-instruction coverage. Reward scores are
ranking signals with potential verbosity effects. Existing metadata is valuable, but not a verifier.

Use primarily in capability SFT and short replay; evidence-bearing subsets can support short grounded
training after inspection. The dataset card lacks a complete top-level license declaration; preserve
upstream LMSYS and response-generation terms in the source record.

### 3.2 Magpie Llama-3.1 Pro MT — priority core candidate

[Dataset](https://huggingface.co/datasets/Magpie-Align/Magpie-Llama-3.1-Pro-MT-500K-v0.1), 500K rows,
generated with Llama-3.1-70B-Instruct. It has category, difficulty, input-quality, language, and reward
metadata. Strong candidate for dialogue, explanations, follow-up handling, and broad instruction SFT.

The upstream process selected long responses before extending conversations. Do not reproduce a
longest-is-best filter. Stratify length and category, inspect the second turn, and retain longer
conversations that SpeckChat2 could not fit. The MT and related single-turn/filtered releases share
first-turn families; adding them does not create independent coverage. Follow the stated Llama 3.1
terms rather than treating the derived text as unconditionally Apache-licensed.

### 3.3 Hermes 3 / enPurified Hermes — compare original against prose subset

[Original](https://huggingface.co/datasets/NousResearch/Hermes-3-Dataset), 958,829 viewer rows;
[enPurified](https://huggingface.co/datasets/enPurified/Hermes-3-Dataset-enPurified-openai-messages),
117,877. The latter deliberately removes math, code, logic/MCQ formats, short answers, and many symbols
using heuristics, including lexical-richness thresholds. That can produce useful prose, but it is not
a general correctness filter and may remove precisely the JSON, concise answers, and symbolic tasks
the flagship needs. A sampled retained row still contains an MCQ-form question: filtering is imperfect.

Keep the prose subset as a bounded writing/conversation source. Audit the original as a candidate for
recovering structured responses, system following, concise outputs, and broader skills. Preserve
meaningful system prompts and inspect special scratchpad/tool formats before conversion. The current
original card declares Apache-2.0, while the derivative says `other` and points upstream; retain the
actual snapshot rather than propagating the old placeholder declaration without investigation.

### 3.4 UltraChat / enPurified UltraChat — useful secondary diversity

[Canonical H4 version](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k): 207,865 `train_sft`
rows; [prose subset](https://huggingface.co/datasets/enPurified/ultrachat_200k_sft-enPurified-openai-messages):
143,969. The original ChatGPT-generated conversations cover broad topics and follow-ups; the H4
version already removes some boilerplate and corrects capitalization. The prose subset additionally
removes code blocks, math, short responses, and other material under heuristic quality rules.

Sampled conversations include both useful passage transformations and open-world factual claims.
Use selected components for dialogue and transformations; inspect dated claims and formulaic follow-ups.
Prefer the canonical source for provenance and targeted filtering, with the existing purified subset
as a comparator. H4's `train_gen` is a possible later prompt pool; its separation does not replace
global deduplication. Preserve all `test_*` splits outside training.

### 3.5 OpenHermes 2.5 — selectively retain, recover component identities

[Dataset](https://huggingface.co/datasets/teknium/OpenHermes-2.5): 1,001,551 viewer rows. It combines
Airoboros, Camel, LMSYS, Evol-Instruct, Glaive, MetaMath, SlimOrca, Platypus, ShareGPT, and other sources.
This is useful breadth and response-style diversity, not one homogeneous source. The card even notes
a lost source reference for one component. Keep internal source/category metadata when available.

Use inspected contributions for general instructions, transformations, and technical explanations.
Do not retain a 60% share merely because SpeckChat1 did. Reverify math/code; overlap with dedicated
math/code, LMSYS, SmolTalk, and Dolci sources must be handled at upstream-family level.

### 3.6 Magpie Air MT — reserve source, not the main dialogue backbone

[Dataset](https://huggingface.co/datasets/Magpie-Align/Magpie-Air-MT-300K-v0.1): 300K Llama-3-8B-generated
conversations. Its original filter prioritizes medium-or-harder inputs and the longest responses.
It is a useful historical baseline and possible diversity supplement. First compare acceptance and
usefulness against Pro MT and newer data; source age or teacher size alone does not prove inferiority.
The card explicitly warns that Air single-turn and MT releases share first turns.

### 3.7 No Robots — valuable human source, conditional for the public recipe

[Dataset](https://huggingface.co/datasets/HuggingFaceH4/no_robots): 9,500 train / 500 test; human-authored,
with writing, brainstorming, QA, transformations, and some code. Valuable for natural instruction
styles and human demonstrations; small enough to inspect closely. Its CC-BY-NC-4.0 declaration needs
an explicit source decision under the flagship release policy before inclusion. This is a real stated
term, not a conclusion that a dataset license automatically determines the model-weight license.
Do not copy or paraphrase its content as a way of erasing provenance. Independently authored human
examples can fill the same skill roles if it is not selected.

### 3.8 CoCoNot — small, targeted ambiguity/correction contribution

[Dataset](https://huggingface.co/datasets/allenai/coconot): 11,477 original training rows; includes
incomplete, unsupported, indeterminate, humanizing, and safety-related requests. Select examples
that teach clarification, limits of available information, and appropriate responses. A blanket
rejection policy would be the wrong lesson; counterbalance with answerable training cases.

Keep `original/test` and `contrast/test` for evaluation only if their families remain unseen. The
927-row `pref/train` is a later candidate; it contrasts appropriate compliance against over-refusal.
The pinned card mentions both ImpACT and ODC-BY terms; resolve component/source terms explicitly.

### 3.9 Everyday Conversations — retain a small behavior anchor

[Dataset](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k): 2,260
training conversations plus 119 test examples, generated by Llama-3.1-70B. It teaches short natural
exchanges and simple topics. Keep a small contribution and diversify boilerplate openings. It is
neither a broad knowledge corpus nor a reason to oversample a few thousand examples repeatedly.

### 3.10 Magpie Reasoning V1 — targeted, conditional supplementary source

[Dataset](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V1-150K): Qwen2-72B instructions
and Llama-3-70B responses across math, reasoning, and code. These are worked-answer demonstrations,
not a requirement to release a Think model. Reverify solutions, calibrate difficulty, and avoid
upstream longest-response selection. The pinned card's license section explicitly includes CC-BY-NC
in addition to model terms, which the old builder's short declaration did not fully convey. Keep it
conditional rather than making it a dependency of the released recipe.

## 4. New components worth adding first

### SmolTalk: targeted skills rather than a second giant mixture

[Dataset](https://huggingface.co/datasets/HuggingFaceTB/smoltalk).
Prioritize `smol-constraints`, `smol-rewrite`, `smol-summarize`, and inspected `smol-magpie-ultra`.
The former three directly fill precise-instruction and transformation gaps. Smol-Magpie-Ultra is a
useful peer to Pro MT, generated using Llama-3.1-405B and curated for the SmolLM work. Judge it on
accepted useful examples, not teacher size. The card's rounded constraint count is 36K; the inspected
train viewer reports 34,424. Source/config/split identity matters more than rounded marketing counts.

SmolTalk already contains OpenHermes, Everyday, LongAlign, math, code, and system-chat components.
Select components once. For SmolTalk2, use selected `no_think` components and restore any system
content carried in `chat_template_kwargs.custom_instructions`. `no_think` does not mean terse,
automatically correct, or JSON-only: inspected Table-GPT examples still request worked reasoning.

### Nemotron IF/Chat v2 and v3: promising modern complement, different adapters

[v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v2) has a
`reasoning_off` split suitable for initial inspection. Its card reports about 2M rows across the
release, not 2M verified non-thinking training examples. Inspected rows include practical rewriting
and technical help, but also a concatenated multi-question prompt and a broad performance claim.
Newer generation does not remove the need for prompt/answer checks.

[v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v3) reports roughly
637K chat and 249K instruction-following samples. Chat uses real user seeds, GLM-5 responses, and
pairwise ranking by Qwen3-Nemotron GenRM; IF uses GPT-OSS-120B. Particularly promising for broader
real-world prompts and precise constraints.

Adapter requirements, confirmed in direct pinned rows:

- Some initial system/user contents are `null` because external seeds must be restored with the
  provided reconstruction script and appropriate source access. Do not invent missing prompts.
- Honor `metadata.train_turns`; in chat only the final assistant turn is the selected target. Earlier
  sampled assistant turns are conditioning context, not approved demonstrations.
- For Instruct, omit separate `reasoning_content` from both targets and serialized conversation
  history; inspect that the remaining answers are self-contained. Audit supervision per split/row.
- Filter actual language: inspected chat rows include Arabic despite the top-level English tag.
- Group by `seed_dataset` and `seed_prompt_sha256`; successive conversation extensions repeat seeds.

The existing SFT mask supervises all assistant turns. Per-message target eligibility is a concrete
prerequisite for using v3 correctly. Empty optional system messages can be normalized away; missing
user content cannot. Treat v3 as a priority candidate with explicit adapter work, not plug-and-play.

### Dolci Instruct SFT: component-level comparison and gap filling

[Dataset](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT): 2,152,112 rows with source/domain
metadata. Prioritize inspected precise-IF, Python, logic, verifiable-reasoning, and general-chat
components. It already includes OpenThoughts, CoCoNot, FLAN, Tulu personas, Table-GPT, SciRIFF, and tool
data. Whole-mixture addition would duplicate other selections and introduce unintended objectives.
Filter tool-role examples unless the corresponding interface is explicitly supported. It is also a
useful independent recipe baseline, not evidence that its original 7B sampling weights fit Speck.

### Math: Orca-Math plus selected Numina, rather than competition traces alone

[Orca-Math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k): 200,035 grade-school
word problems with GPT-4-Turbo answers. Strong candidate for practical quantitative reasoning and
self-contained worked solutions. It supplies demonstration text, not a separately verified answer
oracle. Recompute answers where possible and audit problem-family overlap.

[NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT): about 859K training rows with
source labels. Start with an inspected `cn_k12` slice plus a limited harder slice. The card explicitly
lists GSM8K, MATH, AMC/AIME, synthetic derivatives, and Orca-Math as upstream sources. Preserve source
labels and remove benchmark test/derivative overlap before use. Official training splits can be used
under a declared in-distribution benchmark protocol; they are not evidence of unseen-family transfer.

### Code: a compact validated baseline plus a larger selectively verified pool

[Self-OSS-Instruct](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k):
50,661 execution-validated Python demonstrations with seed functions and concept metadata. Start here
for a bounded, inspectable code component. The final schema does not expose a standalone complete
test suite per row, so execution-validation provenance is not automatically an RL-ready verifier.

[OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct): the card describes 5M
generic/algorithmic examples with tests, execution status, and judge scores. The inspected viewer is
explicitly partial (1.4M visible rows); this is not a revised corpus count. Both failed solutions and
incorrect tests are present in inspected examples. Filter and reverify before SFT; retain suitable
failures for repair/preference research only when the task and oracle are sound. Judge scores alone
cannot override contradictory execution evidence.

## 5. Grounded-stage candidates

| Candidate | Role | Selection requirement |
| --- | --- | --- |
| `THUDM/LongAlign-10k` | long-document QA and summarization; short subset can appear in first SFT | English, source provenance, evidence-bearing answer, actual Speck token length |
| `THUDM/LongCite-45k` | evidence-linked QA and optional citations | English subset, validate cited support, citation-format adaptation, length buckets |
| `allenai/SciRIFF` | scientific QA, extraction, classification, entailment, summaries | select training task families, document split, component terms |
| `LipengCS/Table-GPT` | table extraction, matching, sorting, filtering, structured outputs | select inferable transformations; preserve train/test task boundary |
| selected ChatQA components | short/medium grounded QA, unanswerable questions, table arithmetic | source-specific terms/splits and correct document/answer serialization |
| new Speck document tasks | multi-document, positional diversity, absent evidence, practical schemas | source-controlled documents, verifiable targets, independent task templates |

### LongAlign and LongCite

[LongAlign](https://huggingface.co/datasets/THUDM/LongAlign-10k) is 10K bilingual examples described as
8K-64K tokens under its original setup. The pinned first English example contains about 256K
characters and asks for a fact near the document end. It is useful long retrieval supervision, not
automatically multi-hop comprehension. The next inspected row is Chinese. The card does not provide
a simple blanket data-license declaration, and embedded documents retain source-specific identity.

[LongCite](https://huggingface.co/datasets/THUDM/LongCite-45k) reports 44,600 bilingual examples up to
128,000 **words**, not Speck tokens. The inspected format uses `<C...>` document chunks and
`<statement>...<cite>...</cite></statement>` responses. Initial inspected rows are Chinese and shorter
than the maximum. English sampling and length-yield measurement remain pending. If citations are
retained, make them an explicitly requested output style; do not force citation XML into every answer.
If converted to ordinary QA, retain evidence mappings offline and check that conversion preserves support.

Neither dataset alone provides a qualified 128K English curriculum. Measure retained length coverage,
evidence depth, document source, and task complexity before assigning the long-stage budget.

### SciRIFF

[SciRIFF](https://huggingface.co/datasets/allenai/SciRIFF) has 54 scientific tasks and 4K/8K/16K views
with explicit source/task metadata. Use it for grounded reading breadth and structured scientific
outputs, not as 128K data. The 16K view has 72,646 training rows, but maximum length is not actual
example length. The paper's [train mix](https://huggingface.co/datasets/allenai/SciRIFF-train-mix)
contains about 35K scientific rows plus matching Tulu V2 replay, totaling 70,714. If constructing our
own mixture, reproduce its intended training-task selection without blindly adding its replay or
holding out only by row. Do not concatenate alternate context views of the same instance.

### Table-GPT

[Table-GPT](https://huggingface.co/datasets/LipengCS/Table-GPT) exposes training versus held-out task
families. It is useful for structured manipulation that prose-only filtering removes. Initially favor
sorting, filtering, extraction, and matching with checkable answers. Some augmentation/imputation tasks
require guessing unknown facts; do not mix them indiscriminately into evidence-only QA. One inspected
example asks for an absent ship name; this is not equivalent to extracting a supplied value. Preserve
the intended distinction between grounded transformation and predictive completion.

### ChatQA

[ChatQA-Training-Data](https://huggingface.co/datasets/nvidia/ChatQA-Training-Data) provides separate
SQuAD, DROP, Quoref, ROPES, TAT-QA, and other components. Inspected rows store `document` and `answers`
outside `messages`; a messages-only adapter would lose both evidence and supervision. Some TAT-QA
answers are arithmetic expressions, not evaluated final numbers. Build source-specific adapters.
Use these as short/medium grounding candidates, not long-context proof. Its synthetic conversational
component explicitly says non-commercial; NarrativeQA and other source-specific issues also need
their own resolution. Do not adopt the full ChatQA collection as one uniformly licensed source.

## 6. Preference and RL data

- [UltraFeedback cleaned](https://huggingface.co/datasets/allenai/ultrafeedback_binarized_cleaned):
  60,829 `train_prefs` rows; removed TruthfulQA prompts and identified faulty annotations, retains
  upstream source tags. Useful compact baseline and prompt source. Cleaning does not certify every pair.
- [Dolci Instruct DPO](https://huggingface.co/datasets/allenai/Dolci-Instruct-DPO): 259,922 rows,
  includes delta-based and judge-ranked pairs plus multi-turn data. Strong comparison candidate; adapt
  only relevant message fields and recheck factual/constraint quality. A chosen response can still be
  wrong or violate a constraint even when it is better than the rejected response.
- [Tulu 3 8B preferences](https://huggingface.co/datasets/allenai/llama-3.1-tulu-3-8b-preference-mixture):
  272,898 pairs with source IDs; useful recipe precedent. Its card explicitly notes non-commercial
  components. It overlaps sources already in SFT/SmolTalk2; choose components rather than the whole mix.
- [Dolci Instruct RL](https://huggingface.co/datasets/allenai/Dolci-Instruct-RL): 169,964 prompts with
  heterogeneous fields for answers, difficulty, constraints, and source. Rebuild prompts from text,
  not its tokenizer-specific `input_ids_prompt`. Verify each reward source and recalibrate difficulty
  for Speck. The IF component has benchmark-style/train derivations; preserve the evaluation boundary.
- The primary eventual preference set should include Speck's own verified failure pairs. Public
  preference data does not cover the exact mistakes of this tokenizer, architecture, and checkpoint.

## 7. Concrete observations that change the adapter/filter design

These are illustrative counterexamples, not quality-rate estimates. Row indices are zero-based.

| Observation | Reproducible location | Consequence |
| --- | --- | --- |
| SpeckChat1 says the longest common substring of `hello` and `hallo` is `allo`; the correct longest substring is `llo` | published revision, train row 48687 | do not equate fluent old demonstrations with correctness |
| Prose-filtered Hermes still contains an MCQ-format prompt | pinned derivative, train row 54641 | card-described filters are not exhaustive guarantees |
| OpenCodeInstruct includes failed-test rows, and tests that confuse separate object instances | pinned source, train row 115566 (`BankAccount`) | verify the verifier as well as the solution |
| OpenCodeInstruct BMI test expects a numeric result inconsistent with the formula and compares a raising call to the string `ValueError` | train row 999 | naive pass-rate filtering can reject correct code because of a bad oracle |
| Dolci DPO chosen text contains commas despite a no-commas instruction | train row 999 | rerun exact constraint checks; chosen is a relative label, not perfect SFT gold |
| Nemotron v3 has withheld prompts, Arabic rows, repeated seed hashes, separate reasoning, and final-turn masks | pinned `data/chat.jsonl`, first three rows | restore prompts; language filter; family split; per-message supervision |
| LongAlign's first English example is long but asks for a trailing fact | pinned `long.jsonl`, first row | diversify evidence position and task complexity, not only length |

Do not execute arbitrary downloaded test code in the survey process. Later code-data qualification
uses the existing project's controlled execution environment with bounded resources and known dependencies.

## 8. Recommended first qualification wave

The current [training plan](POST_TRAINING.md) uses guided curation and milestone evaluation. The
candidate comparisons below are optional source-inspection questions, not mandatory training arms
or a requirement to run every sampling/review quantity before starting data preparation.

Organize around twelve decision-relevant candidate groups rather than downloading every complete mix:

1. OpenLeecher DeepSeek responses.
2. Magpie Pro MT versus Smol-Magpie-Ultra.
3. Original Hermes 3 versus the existing prose subset.
4. Canonical UltraChat versus its prose subset.
5. Smol-Constraints plus selected rewrite/summarize components.
6. Nemotron v2 `reasoning_off` and v3 IF/chat with correct adapters.
7. Selected Dolci Instruct IF/Python/general components.
8. Orca-Math plus source-stratified Numina.
9. Self-OSS-Instruct plus reverified OpenCodeInstruct.
10. LongAlign English plus LongCite English.
11. Selected SciRIFF training tasks and Table-GPT transformations.
12. Small Everyday and conditional CoCoNot/No Robots contributions.

For each selected component, prepare a bounded sample of roughly 1-2K rows, stratified by available
source/skill/length metadata, with a separate probability sample if estimating acceptance rates.
Inspect at least a proposed 100-200 examples per major candidate, oversampling flagged failures.
Record stratum weights; an oversampled problem set cannot estimate raw corpus quality without weighting.
Measure accepted-token yield, correctness, prompt validity, verbosity, target density, source overlap,
length coverage, and conversion fidelity. This qualification wave is proposed, not completed here.

First SFT should combine realistic prompts (LMSYS/Nemotron), synthetic dialogue (Magpie), selected
Hermes/UltraChat breadth, explicit constraints/transformations, and verified math/code. Grounded SFT
should add actual documents, evidence, tables, and scientific tasks while replaying general skills.
Use capability quotas from the pipeline draft as priors; do not freeze per-source weights before
measuring unique accepted tokens and overlap. Old source names remain candidates without inheriting
old sampling proportions, length restrictions, or training authority.

## 9. Highest-value original Speck data

Public data already supplies abundant general conversation. The strongest gaps to fill ourselves are:

- diverse evidence-linked questions over source-controlled long documents;
- multi-document comparisons and distractor-heavy questions with known answers;
- matched answerable/unanswerable cases;
- varied system/format constraints and realistic multi-turn corrections;
- code debugging with independently checked tests;
- preference pairs reflecting actual Speck failures.

This creates a fresh pipeline while preserving the useful source-selection work behind both SpeckChat
datasets. The contribution is the verified, balanced, traceable selection and its measured effects.
