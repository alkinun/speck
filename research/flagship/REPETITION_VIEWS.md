> **Scope update, 2026-09-15:** the [long-context pivot](PIVOT.md) supersedes old experiment budgets, quotas and selection dependencies in this note. Retained mechanics and measurements remain evidence at their recorded identities. Use [LONG_CONTEXT_DATA.md](LONG_CONTEXT_DATA.md) and [FIRST_WAVE.md](FIRST_WAVE.md) for new preparation; do not automatically expand the retired wave.

# Explicit repeated source streams for E3

The registered E3 comparison remains 1, 2 and 4 effective epochs at a nominal 6B-token
horizon, with seeds 42 and 43. The incumbent and matched code-language allocation are selected
by the active first-wave preparation contracts. No model has been trained for this comparison.

[`speck/data/repetition.py`](../../speck/data/repetition.py) provides the source-stream
materializer. It consumes a verified document token cache and an **explicit ordered list of
selected stock ordinals**, then writes three ordinary uint16 streams for the existing loader.
It does not choose documents or shuffle them. Seed is bound as provenance for the supplied order;
changing seed without supplying a different order does not create another randomized dataset.

The ordered largest pool must contain distinct document ordinals and distinct released text
hashes. Its exact token total equals this source's requested exposure. The half-size and
quarter-size prefixes must end at whole-document boundaries, and all three pools must align
to the declared distributed microbatch stride. Arbitrary stock prefixes usually do not satisfy
these conditions: the upstream jointly eligible pool selector must supply suitable documents.
Cross-source deduplication and selection qualification remain upstream requirements.

Each variant writes its unchanged pool order exactly 1, 2 or 4 times. This defines the supported
materialization policy; it does not introduce fresh shuffling on later epochs. All variants
contain the same number of training input tokens, followed by **one additional token from the
start of that variant's own pool** for next-token lookahead. The extra token is not a further
training input, unique document or unique supply. With complete cycles, shifted targets also
cover each pool token position the same number of times. BOS/EOS serialization does not isolate
attention or examples across document boundaries.

The compact pool index preserves original stock ordinal, content ID/hash, original token span,
UTF-8 length and new stream offset. An occurrence in zero-based epoch `e` begins at
`e * unique_pool_tokens + view_token_start`. Pool prefixes give exact nested membership; the
occurrence rule avoids writing a second large index containing repeated copies of every record.

A plan contains:

```json
{
  "format": "speck_repetition_source_plan",
  "format_version": 1,
  "stock_manifest": {"path": "<verified-cache>/manifest.json", "sha256": "<hash>"},
  "document_order": {"path": "<frozen-order>.jsonl", "sha256": "<hash>"},
  "source_id": "<source-id>",
  "seed": 42,
  "exposure_tokens": 192,
  "global_stride": 8,
  "effective_epochs": [1, 2, 4],
  "shard_tokens": 17,
  "output_directory": "<new-output-directory>",
  "training_authority": false
}
```

The small numbers illustrate the schema and are not E3 launch settings. Each line of the
order file is one JSON integer stock ordinal. A real plan must use the source's exact exposure
from the actual mixture schedule and `sequence_length * local_batch_size * world_size`.
The source exposure must be divisible by four times that stride. The **nominal 6B horizon is
not divisible by 2,048**; final horizon, sequence/batch geometry, exact mixture scheduling and
source quotas must therefore be resolved together in execution contracts. No horizon, quota,
training budget or nominal experiment plan is rounded or revised by this materializer.

```bash
uv run --no-sync python -m scripts.materialize_repetition /absolute/path/to/plan.json
uv run --no-sync python -m scripts.materialize_repetition /absolute/path/to/plan.json --resume
```

The output binds the plan, stock manifest, order file, tokenizer identity and materializer
implementation hash. One process owns its output directory. Every token shard is published
with its checksum only after its payload is flushed; interrupted attempts remain separate files.
Resume requires unchanged ownership and reconstructs every completed shard's expected periodic
bytes from the original stock before reuse. Corruption is an error, not a reason to regenerate
over the earlier evidence. The final manifest appears only after all three variants are complete.

This is **source-stream machinery, not a completed E3 dataset or training manifest**. CPU fixtures
exercise exact exposures, whole-document nesting, shard boundaries, partial-write preservation,
corruption rejection and the unchanged loader algorithm on two ranks, comparing resumed with
uninterrupted input/target batches across repetition boundaries. The loader fixture supplies a
minimal schedule explicitly; it does not qualify the packed-manifest exporter or launch receipt.

Remaining work is to freeze real jointly eligible pools and order for both seeds, choose aligned
execution geometry, bind per-source counts to the actual schedule, export repeated occurrence
provenance into the production dataset manifest without presenting repeated documents as new
unique supply, and complete evaluation/hardware/launch qualification. No real E3 materialization
or model-training job has been launched by this implementation.

Implementation validation passed: **1,206 tests**, 10 skipped and 125 deselected, plus
formatting, lint, research catalog, archive and source-pin checks. The ten new tests cover
this component; they are software validation, not E3 model results or real-pool qualification.
