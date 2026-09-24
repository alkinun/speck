# Export and release

Export only a completed, identified checkpoint. Keep the base and assistant artifacts distinct.
What is released is set by the [program](program.md#release); release notes and model cards follow
the [paper outline](paper.md). The released base is the selected branch (decay or mid-training) and
the assistant is its SFT probe. Identify each model's branch lineage (parent checkpoint, decay and
mid-training branches, SFT probe), actual data/exposure, cost and measured capabilities.
The architecture was [not compared](program.md#model) with alternatives, so cite no architecture
result; scope claims to the experiments actually run.
The maintained path bundles native model code behind a Transformers wrapper and checks logit parity.
An export is not evidence of tool-use capability or accelerated serving support.

```bash
uv run --no-sync python -m scripts.model_publish \
  --checkpoint-dir CHECKPOINT_DIRECTORY --repo OWNER/MODEL --no-upload
```

Use `scripts.base_checkpoint_export --help` for base checkpoints. Supply the intended source,
destination, and output directory explicitly. Review the generated files and parity results before
uploading. The pinned compatibility-code provenance refers to an older published Speck checkpoint; the
new artifact must identify its own source checkpoint and producing revision.

The export vendors architecture, model layers/state, and optimizer definitions and records their
hashes, including the maintained exact SentencePiece backend. Both export entry points check
native/exported logits, generation, text token IDs, and (for assistants) complete chat token IDs.
The tokenizer check writes `tokenizer_parity.json`; weighted assistant context and the checkpoint's
chat format version are included in that check.

Base exports use the original prepared tokenizer, bound to the checkpoint's recorded fingerprint.
Use `--tokenizer-dir` if those same bytes have moved. Older checkpoints without an explicit tokenizer
fingerprint additionally require their original hash-bound packed manifest. A tokenizer is never
substituted from an older published model.
Right-padded likelihood batches are supported without caching; cached padded inference is not.
A production tool parser and accelerated KDA backend still require their own qualification.

Source code is MIT; released weights are Apache-2.0, shipped with the complete licence and an
accurate model card. Release model/tokenizer metadata and permitted artifacts. Whether source text
or packed training shards are released is an [open decision](../PLAN.md#open-decisions); until it is
recorded, upload neither. Preserve checkpoints, producing revisions, data provenance, evaluation,
limitations, and costs. Old GGUF and one-shot release migrations are recoverable in
[history](../archive/README.md); they are not the current KDA export path.
