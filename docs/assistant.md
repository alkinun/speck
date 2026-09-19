# Assistant rehearsal contract

The first hardware rehearsal uses complete 4K conversations, the frozen 32K base tokenizer,
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

The planned context extension targets approximately 128K. The 4K rehearsal above is an initial
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

The OpenBMB stock pins still match the reviewed release heads. The first audit rejected tool
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

1. Inspect [UltraData-SFT-2605](https://huggingface.co/datasets/openbmb/UltraData-SFT-2605)
   `no_think` variants alongside retained `think` rows. The release provides both modes for core
   math/code/knowledge/instruction domains. Verify answers and source overlap before weighting;
   a publisher's training-validation claim does not independently verify every retained answer.
2. Inspect selected [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) SFT
   components for everyday conversation, writing, rewriting, summarization, tabular understanding
   and multi-turn instructions. Its Mid/SFT/Preference sets have different purposes and shared
   upstream sources. Keep source tags and modes; do not concatenate the collections blindly.
3. Use [Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) as a second
   general-assistant comparison with source/category labels. Check component terms, duplication
   and benchmark overlap. It is a candidate, not an established winner over retained sources.
4. Continue [UltraData-SFT-Agent-2609](https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609)
   checks with complete tool traces, loss masks and task outcomes. Preserve long trajectories;
   teacher-reported success is not equivalent to replay in our actual environment. Include when
   to ask for clarification, avoid unnecessary calls and recover from tool errors.
5. Reserve [UltraData-RL-2609](https://huggingface.co/datasets/openbmb/UltraData-RL-2609) for later
   verified-reward work. Its questions, reference answers and code test cases are not successful
   assistant traces; generating and verifying such traces has a separate cost. No RL launch follows
   from listing it. Do not make lengthy reasoning the default for simple requests.

The first main SFT recipe should balance direct assistance, reasoning, practical code and tool
interactions by supervised tokens, while accounting for total context cost and length coverage.
Freeze quantities after the content audit. The [main data work order](data.md#recipe-direction--2026-09-19)
keeps one bounded comparison at a time and protects independent development/final evaluations.
