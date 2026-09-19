# Assistant data and serialization

Thinking SFT and conditional verified-reward RL form the post-training part of the
[data/training program](program.md#training-lifecycle), following pretraining and mid-training.
The conversation targets below are for SFT. RL needs a separately qualified inventory of task
prompts, reference answers/tests, verifiers and policy outcomes; it is not counted as SFT rows.

The final product is an always-thinking assistant focused on agentic coding, general coding,
mathematical reasoning and tool-mediated tasks. It has no supported non-thinking toggle or purely
instruct mode. Short tasks may need brief reasoning; difficult tasks may need deeper reasoning and
multiple tool steps. A possible hybrid effort policy must preserve the thinking contract. This is
a release/training objective, not an established capability of the current base checkpoint.

The completed hardware rehearsal used complete 4K conversations, the frozen 32K base tokenizer,
and chat format v2 with its existing three role IDs. Context-only assistant turns retain weight
zero. Supervision covers assistant content and EOS; system, user, and tool-result content is masked.
Nothing is truncated to manufacture a fitting example.

`speck.tokenization.tools.adapt_conversation` defines `speck_tools_v1`. It converts explicit
OpenAI-style function definitions, calls, and results to text before the existing chat tokenizer:

- Definitions and protocol instructions are appended to the system message.
- Assistant calls use `<tool_calls>[{"id":"1","name":"calculate","arguments":{...}}]</tool_calls>`.
- Results use a user turn containing `<tool_results>[{"id":"1","name":"calculate","content":"..."}]</tool_results>`.
- Multiple calls keep their IDs; all must receive one result before the next turn. Consecutive
  results share a single user turn. IDs cannot be reused within a conversation.
- Reasoning remains in `<think>...</think>`. A separate reasoning field is converted explicitly;
  ambiguous combinations, undeclared tools, unresolved calls, duplicate JSON keys, unsupported
  fields, and nonfinite JSON numbers are rejected. Prose accompanying structured calls must be
  a reasoning block; other source formats require a separately reviewed adapter.

Use this adapter at inference before `apply_chat_template`, and use `parse_tool_calls` on the
assistant response. The generic tokenizer template deliberately rejects raw tool fields. Exported
tokenizers preserve chat v2; applications must also implement this named protocol. No untrusted
tool definition is executed by the adapter. Production applications must validate arguments against
their schemas and independently authorize tool side effects.

The finite local rehearsal selects 32 text and 32 tool conversations for training, and eight of
each for validation. Selection is deterministic from a bounded candidate pool after a full source
census. Exact conversation identities are deduplicated; normalized first-user prompts determine the
split. All five frozen benchmarks are checked with the existing exact/ngram exclusion scanner.
The receipt records every selected identity, source, length, supervised token count, rejection, and
input hash. These small balanced counts are engineering coverage, not a final assistant mixture.

```bash
python -m scripts.sft_rehearsal /path/to/generator-train-*.arrow \
  --tokenizer /path/to/tokenizer.model --prepared-evaluation /path/to/evaluation.json \
  --pool 2048 --output /external/assistant-rehearsal
```

The deterministic tool environment in `speck.evaluation.tools` exercises successful calculation,
no-tool responses, missing information, tool failure, correction, malformed calls, and premature
answers. Its scripted golden episodes verify the environment; they are not model capability scores.
The broader stock still needs source-rights review, teacher-answer verification, semantic deduplication,
and a measured reasoning/output-budget recipe before a release-quality SFT run.

## Main assistant data direction — 2026-09-19

The working target is **1.5M unique qualified training conversations**, with a **1–2M range**;
the existing 500K are starting stock, not a final-size constraint. The
[scale plan](../experiments/main-data/README.md#post-training-scale) targets 600K code reasoning,
375K math reasoning, 375K agent/tool trajectories and 150K supporting thinking/instruction tasks.
Assign one primary category per conversation, exclude holdouts and task-family duplicates, and
choose final training weights from supervised/total-context tokens rather than these row quotas.
The initial cost model assumes one pass. Inventory lengths do not set the qualified context ceiling
or the average conversation size.

Context production qualifies 16K/32K within 300 hours; approximately 128K is a stretch.
Longer inventory remains separate from the qualified training ceiling. The 4K rehearsal above is an initial
engineering phase, not a maximum length policy for future SFT. Preserve complete long reasoning,
document QA and agent trajectories for extension and later SFT. Keep short interactions represented
throughout. Do not truncate solutions, detach tool results or delete long examples merely because
they cannot run in the initial phase. Measure length after our chat/tool serialization, including
prompt, observations, reasoning and final answer; characterize <=4K, 4–16K, 16–32K, 32–128K and
over-target records separately. Current validated runtime remains 4K until extension is qualified.

The [inventory receipt](../experiments/corpus-audit/recipe-review.json) binds the earlier audit of
500,000 retained conversations and its upstream build report:

| Retained source | Rows | Current evidence |
| --- | ---: | --- |
| UltraData-SFT-2605: Code / Math / Knowledge / IF, all `think` | 220,000 | Source census and sampled serialization/length audit |
| UltraData-SFT-Agent-2609: four agent/tool subsets | 110,000 | Source census; selected examples pass the later tool-aware rehearsal |
| glaiveai/reasoning-v1-20m | 120,000 | Source census and sampled serialization/length audit |
| PrimeIntellect/SYNTHETIC-2-SFT-verified | 50,000 | Source census, publisher reward filter and sampled serialization/length audit |

The OpenBMB stock pins matched the release heads in the September 19 review. The first audit rejected tool
formats before `speck_tools_v1` existed; those zero-fit entries are not current long-context
measurements. The later successful 64-training/16-validation rehearsal demonstrates selected
format compatibility, not the quality or compatibility of all 110,000 tool trajectories.

For the sampled text records, 89/256 Code and 64/256 Knowledge examples fit 4K, while 164/256
Math and 247/256 IF examples fit. These are early-phase compatibility counts, not quality
rejections. Moreover, the upstream local build already excluded length tails using a 98th-percentile
policy and 200,000 conversation / 64,000 thinking character limits. It therefore cannot represent
the full long-context source distribution. Keep this stock intact; qualify a separate acquisition
of missing long examples at pinned sources with token-based budgets. Do not inherit those old
character limits as a 128K policy or claim discarded examples are still retained.

Prioritize these additions and checks:

1. Prioritize the retained [UltraData-SFT-2605](https://huggingface.co/datasets/openbmb/UltraData-SFT-2605)
   `think` math/code/instruction examples and independently checked reasoning. Balance brief and
   deep reasoning by task difficulty. The earlier proposal to add `no_think` as a response mode is
   superseded by the user's always-thinking requirement. Such sources may supply candidate tasks
   or reference answers, but need separately generated/verified reasoning before reasoning-SFT
   admission. Do not fabricate a rationale or insert an empty thinking block to relabel a record.
2. Inspect selected [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) SFT
   reasoning components for multi-turn instructions and supporting general skills. Audit reasoning
   usefulness and outcome correctness. Its Mid/SFT/Preference sets have different purposes and shared
   upstream sources. Non-thinking components can be task/reference candidates only, not an alternate
   final response mode. Keep source tags and do not concatenate the collections blindly.
3. Use [Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) as a second
   secondary task/reference pool with source/category labels, rather than directly adopting its
   non-thinking assistant targets. Check component terms, duplication and benchmark overlap;
   any new reasoning supervision needs its own correctness and cost checks.
4. Continue [UltraData-SFT-Agent-2609](https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609)
   checks with complete tool traces, loss masks and task outcomes. Preserve long trajectories;
   teacher-reported success is not equivalent to replay in our actual environment. Include when
   to ask for clarification, avoid unnecessary calls and recover from tool errors.
5. Reserve [UltraData-RL-2609](https://huggingface.co/datasets/openbmb/UltraData-RL-2609) for later
   verified-reward work. Its questions, reference answers and code test cases are not successful
   assistant traces; generating and verifying such traces has a separate cost. No RL launch follows
   from listing it. Keep simple-task reasoning brief without switching reasoning off.

The first main SFT recipe should prioritize code/math reasoning and complete agent/tool trajectories,
with supporting general tasks following the same thinking protocol. Balance brief/deep reasoning
by supervised tokens, while accounting for total context cost, final-answer tokens and length coverage.
Freeze quantities after the content audit. The [main data work order](data.md#recipe-direction--2026-09-19)
keeps one bounded comparison at a time and protects independent development/final evaluations.

## Stage and budget ownership

The [program overview](program.md#thinking-sft) connects this data/format contract to length-bucketed
SFT and conditional reward training. Proposed subdivisions are 500 GPU-hours SFT, 200 RL and 100
on-allocation teacher/verification work within the existing 800-hour post-training envelope. These
are planning bounds, not measured costs. RL and long-context runtime remain unqualified; retain
the SFT thinking assistant if reward training does not justify its cost or causes regressions.
