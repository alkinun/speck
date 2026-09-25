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
uploading. Exports need no Hub download.

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

## Licences

Everything is released as openly as its inputs allow:

| Artifact | Licence | Shipped as |
| --- | --- | --- |
| Source code, including the model code bundled in exports | MIT | [LICENSE](../LICENSE); `LICENSE.code` in exports |
| Model weights and configuration | Apache-2.0 | `LICENSE` in exports |
| Tokenizer (Mistral-7B-v0.1) | Apache-2.0, upstream | `LICENSE.tokenizer` in exports |
| Paper, results, receipts and data manifests | MIT, with the code | this repository |

Every export writes all three licence files. The model card is written separately and declares
`license: apache-2.0`, the training sources and their licences, and the conditions of the
[source-use decision](../experiments/main-data/source-rights-acceptance.json). Whether source text
or packed training shards are released is an [open decision](../PLAN.md#open-decisions); until it
is recorded, upload neither. Old GGUF and one-shot release migrations are in
[Git history](../README.md#history).
