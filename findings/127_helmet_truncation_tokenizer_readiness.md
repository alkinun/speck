# 127 — HELMET truncation-tokenizer replacement gate

## Exact semantics depend on gated bytes

HELMET references `meta-llama/Llama-2-7b-hf` without a revision at three loader call sites. The
tokenizer controls minimum-length filtering, NarrativeQA's strict 131,072-token preselection, and
character-boundary truncation for 25 long-QA/summarization entries. It also separately defines the
paper's RULER generation lengths for 15 archive-local task-length cells.

The truncator counts the suffix with AutoTokenizer defaults, tokenizes the full document with offset
mappings, indexes `target - suffix_tokens`, slices the original Unicode string at that token's end
offset, and appends the suffix. BOS/special-token behavior, SentencePiece normalization, offsets,
library versions, and retokenization across the new boundary therefore all matter. Matching aggregate
token counts is insufficient.

## Authority and oracle boundary

The exact tokenizer is absent locally, anonymous repository identity resolution remains rejected, and
the official model page requires accepting the Llama 2 license before downloading the tokenizer. No
organizational acceptance or intended-use scope decision is recorded. This audit makes no legal
conclusion about permissibility.

Without authorized oracle bytes, testing another tokenizer on fabricated fixtures cannot prove
semantic preservation. A future replacement protocol must pin only the exact tokenizer artifacts and
compare token IDs, counts, offsets, keep/drop decisions, cut positions, output strings, and retokenized
lengths on adversarial fixtures plus qualified real data across every target length.

## Decision

No access retry, acceptance, payload acquisition, fixture-only replacement, prompt materialization,
RULER regeneration, candidate evaluation, or manifest change is authorized. The exact tokenizer and
all dependent entries remain failed gates until an authorized oracle comparison is possible.

## Artifact

- [Truncation-tokenizer readiness](../results/Speck-Architecture-Promotion-v1/helmet-truncation-tokenizer-readiness.json)

Validate it offline with:

```bash
python -m scripts.helmet_truncation_tokenizer_validate \
  --readiness results/Speck-Architecture-Promotion-v1/helmet-truncation-tokenizer-readiness.json \
  --runtime-protocol research/architecture-promotion-v1/helmet_runtime_dependencies_v1.json \
  --runtime-audit results/Speck-Architecture-Promotion-v1/helmet-runtime-dependency-audit.json \
  --synthetic-readiness results/Speck-Architecture-Promotion-v1/helmet-synthetic-reconstruction-readiness.json
```
