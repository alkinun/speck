# Flagship tokenizer training and qualification

Status: tooling complete; the bounded code core is technically qualified but rights-blocked; five
non-code categories and two code supplements remain pending, 2026-09-07. The first flagship should
use a Speck-trained tokenizer only if it clears this protocol. Mistral 32K remains the frozen
fallback until the final source-balanced sample and matched language-model pilot are complete.

[`tokenizer_plan.json`](tokenizer_plan.json) freezes the research decision. The executable config is
created under `experiments/Speck-Tokenizer-v1/` only after every input file has an immutable source
card and SHA-256 hash. This avoids presenting placeholders as launchable inputs.
[`source_registry.json`](source_registry.json) proposes the per-source quotas that sum to each
category target; a failed source is replaced and the registry versioned before sampling begins.

## 1. Scope

Train an English-first SentencePiece tokenizer on equal UTF-8 bytes from web, code, math, synthetic,
science, and reference data. Programming syntax is retained as-is; surrounding natural language is
English-filtered before sampling. The sample is disjoint from data-mixture selection and benchmark
evaluation data.

The custom candidates are BPE vocabularies of 32,768, 40,960, and 49,152 pieces. Mistral 32K is the
baseline and fallback. A 64K candidate is excluded: the packed loader uses uint16 IDs, chat adds three
role tokens, and Shape A has untied 2,048-wide input/output matrices, so larger vocabularies consume
parameters rapidly.

All custom candidates use the same explicit trainer settings:

- byte fallback and 100% character coverage;
- identity normalization, no whitespace collapse, and no dummy prefix;
- deterministic preselected input order, no SentencePiece subsampling or shuffle, and one trainer
  thread;
- `<unk>=0`, `<s>=1`, `</s>=2`, and no padding token;
- exact model/vocabulary/sample hashes and path-independent model serialization.

The provisional code slice uses the decontaminated Stack v3.1 artifact and v4 Stack-Edu/Common Pile
artifacts recorded in [finding 174](../../findings/174_code_contamination_successors.md), plus
still-unprepared Python-Edu and Python language-design prose. None may enter the sample until its
rights disposition is accepted and the executable input manifest pins the final hashes.

## 2. Sample contract

The production sample contains at least 100,000,000 training bytes and 10,000,000 static-evaluation
bytes from each of the six categories: 600 MB train and 60 MB evaluation before JSONL framing.
Whole documents are retained, so each category may overshoot slightly.

Each local input is declared explicitly with an ID, format, text column, path, SHA-256, and separate
training/evaluation byte quotas. Input quotas must sum exactly to the category target, so one early
file cannot silently crowd every other source out. Supported formats are plain text, JSONL, gzip
JSONL, and Parquet. Input order and category order are binding.
Documents are filtered by length, globally exact-deduplicated under NFKC/lowercase/whitespace
normalization, and assigned to train or evaluation by a seeded content hash. A document can never
cross partitions or categories. The manifest records input identity, accepted documents/bytes,
overshoot, rejections, and output hashes.

This tokenizer deduplication protects the qualification sample only. It does not replace the exact
and near-duplicate requirements for the full flagship corpus.

## 3. Running the pipeline

After source qualification, create `experiments/Speck-Tokenizer-v1/tokenizer_experiment.json` using
the frozen values in [`tokenizer_plan.json`](tokenizer_plan.json) plus explicit input records. Relative
input and output paths resolve against the config directory.

Build the balanced sample:

```bash
uv run --extra cpu python -m scripts.tokenizer_sample_prepare \
  experiments/Speck-Tokenizer-v1/tokenizer_experiment.json
```

Train all candidates and pin the baseline:

```bash
uv run --extra cpu python -m scripts.tokenizer_train \
  experiments/Speck-Tokenizer-v1/tokenizer_experiment.json \
  --prepare-baselines
```

Evaluate static metrics:

```bash
uv run --extra cpu python -m scripts.tokenizer_evaluate \
  experiments/Speck-Tokenizer-v1/tokenizer_experiment.json
```

`--restart` deletes only an incomplete `.building` directory. Completed samples and tokenizers are
never overwritten. Changing an input hash, trainer setting, candidate, output directory, or evaluation
setting changes the plan fingerprint and invalidates reuse.

Outputs are isolated under the configured directory:

```text
sample/
  manifest.json
  train-<category>.jsonl
  eval-<category>.jsonl
candidates/<id>/
  tokenizer.model
  tokenizer.vocab
  manifest.json
baselines/<id>/
  tokenizer.model
  manifest.json
evaluation.json
```

## 4. Static qualification

Report per category and macro-average:

- tokens per KiB, bytes per token, and whitespace-token fertility;
- document-level p50/p90/p99 tokens per KiB;
- unknown tokens and exact probe roundtrip;
- vocabulary size including the three chat roles;
- Shape-A embedding plus LM-head parameters;
- difference from Mistral 32K.

The probe set includes consecutive whitespace, indentation, newlines, source code, mathematical
notation, ASCII, and non-ASCII text. Static evaluation has no selection authority. It eliminates
broken candidates and nominates the two custom Pareto candidates; it cannot show which tokenizer
makes a better language model.

## 5. Matched language-model pilot

Train Mistral 32K and the two nominated custom tokenizers with the same 60M backbone and document
stream. Run one screen seed for all three, then add two seeds to Mistral and the best custom candidate:
seven local RTX 3090 runs and a 30-hour local-GPU ceiling. The fixed-document view ends where Mistral
has consumed 1.2B tokens; the fixed-FLOP view stops each tokenizer at the analytic training FLOPs of
that baseline run. Measured throughput defines a secondary fixed-wall-clock view, never the stopping
point after results are known.

Primary quality is macro bits per UTF-8 byte over the six-category held-out set. Also report every
category, fixed-document and fixed-FLOP views, total parameters, training/validation throughput,
peak memory, and time to fixed BPB. Candidate-minus-Mistral must have an upper paired 95% bound no
worse than +0.01 BPB in aggregate and +0.02 BPB in every category. Among eligible candidates choose
the lowest measured compute to fixed quality; a statistical and systems tie chooses the smaller
vocabulary. If no custom candidate passes, retain Mistral 32K.

The final model is opened once on a separately frozen tokenizer audit slice after ranking. Failure
keeps Mistral; it does not start another tokenizer search.

## 6. Freeze and migration

The D5 decision record must contain the selected model and manifest hashes, source/sample identities,
static report, pilot configs/results, audit outcome, parameter cost, and rejection reasons. Then:

1. copy the selected model into a versioned Speck tokenizer artifact;
2. update flagship `tokenizer.json` only after the artifact is immutable;
3. repack all training, validation, long-context, and SFT data from text;
4. regenerate tokenizer-dependent evaluation vocabularies and contamination hashes;
5. verify uint16 packing, chat-role IDs, native/Transformers export, and roundtrip generation;
6. forbid comparison of token-normalized loss across different tokenizers—use BPB or downstream
   metrics.

No existing checkpoint is converted to the new vocabulary. D5 is frozen before E1–E5 and C0 launch.
