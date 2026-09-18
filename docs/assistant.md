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
