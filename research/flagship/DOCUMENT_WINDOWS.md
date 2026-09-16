# Finite engineering document windows

`speck/data/document_windows.py` provides a small, separately named engineering view of the frozen
document-token stocks. It does not replace the production packed loader or authorize model training.
The explicit input order contains document ordinals and within-document token offsets. Preparation
rechecks the complete document index and token payload hashes, then binds every selected span to its
original content digest and available upstream ID. Missing upstream IDs remain missing.

Each batch row has exactly `sequence_length + 1` tokens from one document. Its inputs and shifted
targets cannot cross a document boundary, even when the physical token shards split that document.
There is no padding, drop-last, wraparound or implicit repetition. Input windows from the same
document must be disjoint; the target lookahead may be the next window's first input token. Repeated
content under different source ordinals is rejected. Batch size and world size must divide the finite
window count exactly. The serialized global batch cursor binds the full view and both dimensions;
each rank takes its own consecutive rows, and exhaustion stops instead of starting another epoch.

The [FineWiki engineering plan](finewiki_document_window_preparation_v1.json) selects the first 32K+1
tokens of eight documents, in the longest-first order already reported by the
[index census](../../results/data/document-length-census-v2-20260916.json). This is a deliberately
biased mechanics input, not a training recipe, capability task or heldout evaluation. It processes
262,144 input positions if completed. The raw long-document probe is separate; no unqualified raw
candidates enter this view.

```bash
uv run --no-sync python -m scripts.qualify_document_windows \
  research/flagship/finewiki_document_window_preparation_v1.json \
  results/data/finewiki-document-windows-20260916.json
```

The command refuses existing output, reads every window on CPU, preserves each payload hash, stops at
the finite end, and checks a serialized middle cursor using a newly constructed reader in the same
process. Eight focused tests cover original ordering, physical shard crossings, shifted targets,
rank-disjoint mapping, resume, exhaustion, invalid spans, duplicate exposure and changed bytes.
Reader construction performs a full stock verification; concurrent large-scale rank startup is not
performance-qualified. This implementation is for bounded mechanics checks.

Call a model without persistent recurrent/KV state between independent batches. Data boundaries alone
do not qualify the model's reset behavior. Model forward/backward, distributed execution, optimizer
recovery, the production launcher, source-family partitions and scientific launch manifests still
need their own checks. The view explicitly leaves family and model-isolation qualification false.
