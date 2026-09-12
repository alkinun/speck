# Instruct starter compiler

Configurable English SFT data compiler. The default recipe produces 100,000 training conversations
and 2,000 validation conversations from 17 pinned source components. This was exercised in the
completed [pre-compute pilot](../../research/notebook/2026-09-12-instruct-data-pilot.md).

## Build

```bash
uv sync --extra cpu --group dev --group dataset-build --locked
uv run --no-sync python -m scripts.instruct_prepare --dry-run
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python -m scripts.instruct_prepare \
  --output-dir /mnt/speck-data/speck/data/instruct-starter-100k
```

- `--recipe PATH` selects a mixture; `--samples N` scales exact source quotas and validation size.
  `--validation-samples N` overrides the latter. Weights describe rows, not tokens.
- Optional source `validation_weight` allocates held-out rows independently; it defaults to the
  training weight. Stable prompt-hash assignment still separates the splits.
- `--exclude PATH` accepts JSONL with `text`, `prompt` (text/messages), or `messages`, plus optional
  `family_ids`. Explicit prompts are excluded from every user turn; near matching uses the first
  substantive prompt. Supply the intended evaluation exclusions before building.
- `--resume` reuses hash-checked completed sources after an interruption. Recipe, compiler,
  tokenizer, packages, and exclusions must match. A quota shortfall stops publication and records
  `shortfall.json` in the `.building` directory. Completed datasets are never overwritten.

## Output and selection

Outputs are `train.parquet`, `validation.parquet`, `identity.json`, and `summary.json`. Parquet rows
contain conversational `prompt`/`completion`, source and family identities, category, upstream
verification provenance, and reference token counts. Only the final assistant response is supervised;
earlier assistant turns remain context. Native preparation accepts `prompt_completion_v1` with
`--source-dir`; see the [SFT pilot](../Speck2-140M-Instruct-Starter100K/README.md).

The recipe controls source labels/scores and category caps. Shared checks enforce supported roles,
English prose, complete code fences, no inline thinking markup, limited repetition, and an 8K
reference-token ceiling without truncation. The constraints adapter also checks selected literal
instructions. Separate reasoning fields are omitted; upstream verification labels are not independent
correctness guarantees.

Deduplication combines exact first-substantive prompts and recorded source families with MinHash
candidates verified by word-shingle Jaccard similarity, globally across sources and splits. It is not
exhaustive semantic deduplication. The pinned reference tokenizer supplies length estimates; native
SFT rechecks lengths with its own tokenizer. No dataset or model is uploaded by the compiler.

The [500K recipe](../Speck-Instruct-Starter500K/README.md) demonstrates capacity-aware allocations
using the same source pool and filters.
