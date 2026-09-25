# Assistant data

In this release, assistant conversations serve one purpose: the fixed
[SFT probe](program.md#sft-probe-100-gpu-hours), one frozen recipe applied unchanged to every decay
and mid-training arm and to the 410m transfer and seed runs. The probe of the selected branch is the
released light assistant. The probe's data is never varied.

## Serialization

Chat format v2 with the frozen 32K tokenizer and its three role IDs. Supervision covers assistant
content and EOS; system, user and tool-result content is masked, and `weight: 0` assistant turns are
context. Nothing is truncated to make an example fit.

`speck.tokenization.tools.adapt_conversation` defines `speck_tools_v1`, which converts
OpenAI-style function definitions, calls and results to text before the chat tokenizer:

- Definitions and protocol instructions are appended to the system message.
- Calls: `<tool_calls>[{"id":"1","name":"calculate","arguments":{...}}]</tool_calls>`.
- Results: a user turn with `<tool_results>[{"id":"1","name":"calculate","content":"..."}]</tool_results>`.
  Every call gets one result before the next turn; IDs are unique within a conversation.
- Reasoning stays in `<think>...</think>`. Undeclared tools, unresolved calls, duplicate JSON keys,
  unsupported fields and prose beside structured calls are rejected.

Apply the adapter before `apply_chat_template` at inference and `parse_tool_calls` on the response.
The chat template rejects raw tool fields. `speck.evaluation.tools` is a small deterministic tool
environment for checks, not a capability score.

## Stock

The [recipe review](../experiments/corpus-audit/recipe-review.json) records the 500,000 retained
conversations:

| Source | Rows |
| --- | ---: |
| UltraData-SFT-2605: Code / Math / Knowledge / IF, all `think` | 220,000 |
| UltraData-SFT-Agent-2609: four agent/tool subsets | 110,000 |
| glaiveai/reasoning-v1-20m | 120,000 |
| PrimeIntellect/SYNTHETIC-2-SFT-verified | 50,000 |

The [data-readiness receipt](../experiments/corpus-audit/data-readiness.json) has the format and
context-fit census: 424,463 of 500,000 rows pass the adapter. Most rejections are agent
trajectories with prose beside calls or unresolved calls; they stay out rather than being
rewritten. About 90,000 conversations repeat an opening prompt, some across sources, and each such
group must fall in a single split.

## Freezing the probe

Before any branch is probed: finish the source-use review, answer verification and semantic
deduplication of the stock; assign one primary category per conversation (code reasoning, math
reasoning, agent/tool, supporting instruction tasks); exclude holdouts and task-family duplicates;
weight by supervised and total tokens rather than rows; then freeze data, masks, schedule and
serialization. Keep conversations whole. Do not truncate solutions, detach tool results, invent
rationales or insert empty thinking blocks. The tools are in [Training](training.md#sft-probe).
`scripts.sft_rehearsal` builds the small rehearsal set the GH200 bundle carries.
