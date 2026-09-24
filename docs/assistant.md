# Assistant data and serialization

In this release, assistant conversations serve one purpose: the fixed
[SFT probe](program.md#sft-probe-100-gpu-hours), a single frozen recipe drawn from the retained
stock below and applied unchanged to every decay and mid-training arm and to the 410m transfer and
seed runs. Its scores compare the checkpoints it probes; the probe of the chosen branch is the
released light assistant. The probe's data is never varied here. SFT data studies, RL and
self-distillation are research for a later step.

The assistant is always-thinking, focused on agentic coding, general coding, mathematical reasoning
and tool-mediated tasks. A non-thinking or low/medium/high budget control is not a supported
release feature; the [post-training research synthesis](../experiments/main-data/post-training-research.json)
records the evidence for studying it later.

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
`--subset source:subset`, `--kind` and the per-kind counts restrict the same selection to one
stock slice, as the [RL pilot](../experiments/rl-pilot/README.md) does for its math SFT.

```bash
python -m scripts.sft_rehearsal /path/to/generator-train-*.arrow \
  --tokenizer /path/to/tokenizer.model --prepared-evaluation /path/to/evaluation.json \
  --pool 2048 --output /external/assistant-rehearsal
```

The deterministic tool environment in `speck.evaluation.tools` exercises successful calculation,
no-tool responses, missing information, tool failure, correction, malformed calls, and premature
answers. Its scripted golden episodes verify the environment; they are not model capability scores.
The stock still needs source-rights review, answer verification and semantic deduplication before
the probe recipe is frozen.

## Assistant stock

The [inventory receipt](../experiments/corpus-audit/recipe-review.json) binds the earlier audit of
500,000 retained conversations and its upstream build report:

| Retained source | Rows | Current evidence |
| --- | ---: | --- |
| UltraData-SFT-2605: Code / Math / Knowledge / IF, all `think` | 220,000 | Source census and sampled serialization/length audit |
| UltraData-SFT-Agent-2609: four agent/tool subsets | 110,000 | Source census; selected examples pass the later tool-aware rehearsal |
| glaiveai/reasoning-v1-20m | 120,000 | Source census and sampled serialization/length audit |
| PrimeIntellect/SYNTHETIC-2-SFT-verified | 50,000 | Source census, publisher reward filter and sampled serialization/length audit |

The [data-readiness receipt](../experiments/corpus-audit/data-readiness.json) now applies the
same `speck_tools_v1` adapter used by training to the original deterministic sample: 256 rows per
subset, with every prior sample identity preserved. Complete conversations fitting each ceiling are:

| Subset | 4K | 16K | 32K | 128K | Format rejections / 256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SYNTHETIC-2 verified | 80 | 226 | 256 | 256 | 0 |
| Glaive reasoning | 256 | 256 | 256 | 256 | 0 |
| UltraData Code/think | 89 | 227 | 256 | 256 | 0 |
| UltraData Math/think | 164 | 239 | 256 | 256 | 0 |
| UltraData Knowledge/think | 64 | 249 | 256 | 256 | 0 |
| UltraData IF/think | 247 | 256 | 256 | 256 | 0 |
| Code-Agent | 0 | 27 | 86 | 195 | 61 |
| General-Agent | 0 | 0 | 1 | 1 | 255 |
| Search-Agent | 0 | 23 | 56 | 96 | 160 |
| Tool-Use | 40 | 90 | 92 | 98 | 158 |

A separate full-stock structural pass accepts **424,463 of 500,000 rows** under the current adapter
and records the first format failure for 75,537. This is format compatibility only, independent of
context fit, thinking quality, source use and answer correctness. Exact conversation copies are absent,
but 89,956 repeated normalized first-user prompts and 448 prompt groups spanning sources require
shared split handling. Prompt equality is a conservative link, not proof of identical tasks.

The context-fit table reports sample counts, not full-stock token totals, answer-quality scores or
qualified runtime lengths. Lengths include schemas, observations and context-only turns; supervision
counts remain separate. Most rejected agent samples contain prose beside structured calls or end with unresolved
calls. Preserve those records for an explicit adapter/completeness decision; do not silently turn
prose into reasoning, fabricate observations or truncate trajectories. The historical audit's blanket
tool rejection is superseded for current planning; its receipt remains unchanged.

The local build already excluded length tails using a 98th-percentile policy and 200,000 conversation /
64,000 thinking character limits. It cannot represent the full upstream long-context distribution.
The new sample includes complete 32–128K agent examples, but does not recover previously discarded
examples or establish 128K capability. Keep the stock intact.

The SFT probe draws only from this retained stock. Assign one primary category per conversation
(code reasoning, math reasoning, agent/tool trajectories, supporting thinking/instruction tasks),
exclude holdouts and task-family duplicates, and weight by supervised and total-context tokens
rather than row counts. Keep complete conversations: do not truncate solutions, detach tool results,
fabricate a rationale or insert an empty thinking block. Freeze the probe's data, masks, schedule
and serialization after the content audit, before any branch is probed.

Candidate additions (SmolTalk2, Dolci-Instruct-SFT, missing long agent trajectories) are reviewed
in the [recipe review](../experiments/corpus-audit/recipe-review.json) and wait for a later step.

## Reward data, kept for a later step

RL is not part of this release. The [data-readiness receipt](../experiments/corpus-audit/data-readiness.json)
binds all 32,412 Math and 11,872 Knowledge rows from the pinned
[UltraData-RL-2609](https://huggingface.co/datasets/openbmb/UltraData-RL-2609) release, plus schema
probes of 25 Code and 60 Long-Context rows. It records 416 repeated normalized prompt copies in 333
groups (eight with differing references, preserved for review, not relabelled) and 43 Knowledge
prompts that match retained SFT first-user prompts. These are prompt/reference inventories, not
successful assistant traces. The [post-training audit protocol](../experiments/main-data/post-training-audit-protocol.json)
fixes the review order for that later work and authorizes no SFT or RL run.
