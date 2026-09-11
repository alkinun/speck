# Flagship Instruct: data and training plan

Status: current approximate design, 2026-09-11. Coordination: [SPE-176](https://linear.app/openspecklabs/issue/SPE-176).
This supersedes the experiment-heavy proposal summarized in the [design notebook](../notebook/2026-09-11-post-training-design.md). The first release
focuses on Instruct; dedicated Think and dual-mode behavior are deferred. This is guided model
development with milestone evaluations, not another mandatory ablation matrix. Exact source weights,
training settings, and launch manifests follow prepared data and measured throughput.

## 1. The three-stage recipe

```text
Qualified extended base
    -> broad instruction SFT
    -> high-quality mixed-length finishing SFT
    -> preference tuning
    -> final Instruct evaluation and export
```

The [context-extension plan](CONTEXT_EXTENSION.md) supplies the base checkpoint. Instruct gives direct
answers to simple questions and self-contained worked solutions where useful, with no separate
thinking section. Use full-model SFT and assistant-target loss. Keep meaningful system instructions,
natural multi-turn exchanges, and grounded examples throughout.

No mandatory four-arm recipe screen, RL-versus-SFT comparison, or three-branch weight merge belongs to
this recipe. RL is outside the committed first-release pipeline. A later specific need can motivate a
separately costed decision; it is not a prerequisite for a good SFT/DPO release.

## 2. Approximate data allocation

### Broad SFT: capability mixture

Shares are **assistant target tokens**. Also track processed input-plus-target tokens, padding, and
source exposure so long responses or long prompts cannot silently dominate training. Each example has
one primary quota category and any number of secondary skill tags.

| Primary capability | Initial share | Candidate components |
| --- | ---: | --- |
| General assistance and multi-turn | 25% | OpenLeecher LMSYS/DeepSeek; Magpie Pro MT; selected Hermes/UltraChat; Nemotron chat |
| Precise instructions and structured outputs | 15% | Smol-Constraints; selected Nemotron/Dolci IF; independent schema/constraint tasks |
| Grounded reading and text transformations | 25% | Smol-Rewrite/Smol-Summarize; short/medium document QA; selected Table-GPT/SciRIFF |
| Math and quantitative reasoning | 15% | Orca-Math; source-stratified Numina; inspected concise worked solutions |
| Code and debugging | 15% | Self-OSS-Instruct; reverified OpenCodeInstruct; selected Dolci Python |
| Missing information and correction | 5% | selected CoCoNot; independently authored clarification/absent-evidence/correction tasks |
| Total | 100% | |

Everyday Conversations supplies a small part of general dialogue. No Robots and Magpie Reasoning V1
remain conditional supplementary sources because of the specific terms documented in the survey.
The same skill roles can be filled with independently authored or other suitable examples.

These are starting proportions, not optimal weights or six separate training runs. Use source/skill
inspection and a compact development dashboard to adjust the mixture, recording changes. Keep broad
assistance and transformations substantial while covering useful math/code, format, and evidence tasks.

### Finishing SFT: quality and length mixture

Select stronger demonstrations, grounded tasks, and corrections to recurring development failures.
Retain short general examples rather than finishing solely on a narrow long-QA or math distribution.

| Length/content group | Initial share of processed tokens | Sources |
| --- | ---: | --- |
| Short/medium high-quality replay, up to 8K | 50% | best broad-SFT examples, verified code/math, precise IF, correction tasks |
| Grounded examples, over 8K through 32K | 35% | English LongAlign/LongCite; suitable SciRIFF/document tasks; Speck-generated tasks |
| Grounded examples, over 32K through 128K | 15% | qualified English LongAlign/LongCite and source-controlled long-document tasks |
| Total | 100% | |

Within the long groups, include document QA, extraction, summaries, comparisons, distractors, and
absent-answer cases. Measure actual Speck token lengths. SciRIFF/Table-GPT mainly contribute shorter
grounding; their names do not imply 128K examples. Citation-style data should use an explicitly
requested output format; preserve evidence mappings if converting it to ordinary QA.

### Preference data

Start with selected `allenai/ultrafeedback_binarized_cleaned` and/or `allenai/Dolci-Instruct-DPO` pairs.
Add pairs from the selected Speck SFT checkpoint so the data addresses its actual mistakes. An initial
target is **20-50K accepted pairs in total**, including a bounded generation pool of **5-25K prompts**
with up to four candidates per prompt. These quantities overlap; the generated prompts are not an
additional guaranteed 100K-pair dataset.

Favor clear differences in correctness, evidence support, instruction compliance, completeness,
termination, and useful concision. Discard ambiguous comparisons, control for answer length, and
recheck exact constraints. A public `chosen` label is not a correctness certificate. Generated pairs
share the same train/development/audit family partition as all other data.

## 3. Sources and provenance

Use the [dataset survey](POST_TRAINING_DATA_SURVEY.md) for component-level reasoning and the
[snapshot receipt](post_training_dataset_survey_v0.json) for inspected revisions. Those snapshots
are discovery evidence, not a prepared training corpus.

Revisit the good upstream SpeckChat components without inheriting the old row quotas or 2K restriction:

- `OpenLeecher/lmsys_chat_1m_clean`: realistic first-user prompts and selected DeepSeek-V3 responses.
- `Magpie-Align/Magpie-Llama-3.1-Pro-MT-500K-v0.1`: dialogue and follow-ups; balance response length.
- `NousResearch/Hermes-3-Dataset` and selected original/`enPurified` Hermes/UltraChat components:
  broad response styles, meaningful system instructions, writing and transformations.
- `HuggingFaceTB/smoltalk` / `smoltalk2`: select named components; do not duplicate their included
  OpenHermes, Everyday, math/code, or LongAlign examples through another mixture.
- `nvidia/Nemotron-SFT-Instruction-Following-Chat-v2` (`reasoning_off`) and selected v3 chat/IF:
  newer demonstrations with source-specific adapters.
- `microsoft/orca-math-word-problems-200k`, `AI-MO/NuminaMath-CoT`,
  `bigcode/self-oss-instruct-sc2-exec-filter-50k`, and `nvidia/OpenCodeInstruct`: inspected specialist data.
- `THUDM/LongAlign-10k`, `THUDM/LongCite-45k`, `allenai/SciRIFF`, `LipengCS/Table-GPT`, and selected
  ChatQA components: evidence-bearing data with correct document and answer reconstruction.

Actual terms and inherited source identities remain attached to each component. Existing metadata-only
corpus disclosure rules apply; any release of synthetic training text needs its own compatible source
and release-policy treatment. Dataset-list membership is not approval of every underlying record.

## 4. One reusable data pipeline

1. **Ingest and normalize:** messages, source/teacher revision, document/problem family, evidence,
   answer checks, skill/difficulty/length tags, and per-message supervision eligibility.
2. **Deduplicate and partition:** cluster shared upstream prompts, document versions, repositories,
   and conversation extensions; keep all variants in one partition. Exclude benchmark test and audit
   material from demonstrations, generation seeds, preferences, and local examples.
3. **Verify and inspect:** check answers, execute code in a controlled environment, validate tests and
   schemas, inspect evidence and summaries, and sample-check open-ended responses. Human review guides
   selection; fluent text or a teacher score does not replace task correctness.
4. **Balance and tokenize:** select capability and length coverage, count both target and processed
   tokens, and package length buckets with immutable manifests. Keep unrelated examples isolated until
   recurrent/attention reset semantics for packing are qualified.
5. **Refresh selectively:** after broad SFT, group development failures by skill and add useful
   finishing examples or preference pairs. Record the changes; keep the final audit independent.

Critical adapter details:

- Nemotron v3 may require restoration of withheld initial prompts. Honor `metadata.train_turns`;
  earlier assistant messages can be context without being targets. Omit separate `reasoning_content`
  from Instruct history and targets, and verify the retained final answer stands alone.
- Restore system instructions stored in SmolTalk2 chat-template metadata.
- ChatQA may store the document and answer outside `messages`; include both correctly.
- Keep Numina/upstream math family labels and code test provenance. Verify the verifier, not only code.
- Reject or rebuild overlength grounded examples rather than truncating away their evidence.

## 5. Approximate training sizes and development cadence

| Stage | Data preparation target | Initial training exposure | Training behavior |
| --- | --- | --- | --- |
| Broad SFT | about 1-2B unique processed tokens, with enough candidate headroom for filtering | about 1.5-3B processed tokens | mostly up-to-8K conversations, one pass first; a second pass only if useful and affordable |
| Finishing SFT | about 100-300M unique processed tokens | about 200-500M processed tokens | smaller low-LR continuation with mixed lengths and short replay |
| Preferences | about 20-50K accepted pairs | approximately one pass initially | one conservative DPO recipe, explicit reference checkpoint and scoring |

These are planning ranges, not mandatory minimums or evidence of available data. Unique tokens count
text once across views; exposure counts repeats. Long grounded tokens cost more than short SFT tokens.
Pin actual stage data/settings before starting the stage and record any development-driven successor.

Use a short engineering rehearsal to verify masks, local-parent initialization, memory, finite updates,
and checkpoint/resume. Start with an established full-model SFT optimizer/low-LR recipe; tune only when
observed behavior justifies it. No mandatory hyperparameter grid or seed campaign is attached to this
development recipe. Claims remain stage-wise observed behavior, not isolated algorithm superiority.

Evaluate after broad SFT, finishing SFT, preference tuning, and export, using the same capability
dashboard: general ability, instruction/format following, math/code, grounding, context retention,
repetition/stopping, and output cost. Include inspected real conversations as well as benchmarks.
Optional within-run checks diagnose drift without turning every checkpoint into a benchmark sweep.
If a continuation regresses, retain the earlier qualified checkpoint and fix the identified problem
within the budget; do not assume the last checkpoint is the best one.

## 6. Budget and operational work

Working **130-GPU-hour post-training envelope**, replacing the earlier proposal's recipe screens and
RL comparison with direct model development:

| Work | Initial GPU-hours |
| --- | ---: |
| Engineering rehearsal | 10 |
| Broad SFT | 60 |
| Finishing SFT | 30 |
| Preference generation/scoring/training on GH200 | 20 |
| Targeted correction / repair | 10 |
| Total | 130 |

This uses the proposed P6 redistribution: 200 context + 130 post-training + 60 quality evaluation +
60 serving/export = 450 GPU-hours. It replaces the old 80-hour annealing/merge commitment plus 50-hour
SFT line; an executable successor to `plan_v2.json` must reconcile that work list before launch.
The 5,000-hour ceiling and 889-hour protected reserve do not change. These are spending ceilings,
not promises that every token range above fits.

Use local 5090/3090 machines for suitable data generation, scoring, inspection, and engineering work.
Choose teachers by task quality and accepted useful data per hour. Record local GPU time separately;
all GH200 generation, scoring, saving, and training count against the grant once. Benchmarking uses
the separate quality envelope. Do not make a large teacher deployment or online RL stack a dependency.

Next concrete artifacts: selected component manifests, correct source adapters and target masks,
prepared capability/length buckets, local-checkpoint SFT support, a reference-scored preference set,
measured stage throughput, and a compact final evaluation manifest. Data preparation is not complete
merely because this approximate design is documented.
