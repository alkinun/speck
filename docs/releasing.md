# Export and release

Export only a completed, identified checkpoint. Keep the base and assistant artifacts distinct.
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
hashes. Check native/exported logits, generation, padding support, tokenizer, and chat template.
Right-padded likelihood batches are supported without caching; cached padded inference is not.
A production tool parser and accelerated KDA backend still require their own qualification.

Source remains MIT; intended new weights use Apache-2.0 with a complete license and accurate model
card. Release model/tokenizer metadata and permitted artifacts; do not upload source corpus text or
packed training shards. Preserve checkpoints, producing revisions, data provenance, evaluation,
limitations, and costs. Old GGUF and one-shot release migrations are recoverable in
[history](../archive/README.md); they are not the current KDA export path.
