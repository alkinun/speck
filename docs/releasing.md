# Export and release

Export only a completed, identified checkpoint, and keep base and assistant artifacts distinct. The
[program](program.md#release) sets what is released. The released base is the selected decay or
mid-training branch and the assistant is its SFT probe; each model card identifies that lineage,
the training data and exposure, cost and measured capabilities, following the [paper outline](paper.md).

```bash
uv run --no-sync python -m scripts.model_publish \
  --checkpoint-dir CHECKPOINT_DIRECTORY --repo OWNER/MODEL --no-upload
```

`scripts.base_checkpoint_export` exports base checkpoints the same way. An export bundles the native
model code behind a Transformers wrapper and checks native against exported logits, generation, text
token IDs and, for assistants, chat token IDs, writing `tokenizer_parity.json`. Base exports use the
checkpoint's original tokenizer, bound to its recorded fingerprint; pass `--tokenizer-dir` if those
bytes have moved. Review the generated files and parity results before uploading.

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
is recorded, upload neither.
