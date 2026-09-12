# Flagship tokenizer training and qualification

Status: production operations and the real six-category firewall pass; the 600 MB/60 MB formal sample,
three custom models, pinned Mistral baseline, static evaluation, and endpoint nomination are complete
while both sealed audits remain unopened. Formal static results advance 40,960 as the compression
endpoint and exact-32K as the compact endpoint. The first flagship should use a Speck-trained tokenizer only if it clears this
protocol. Mistral 32K remains the fallback until the formal production sample and LM pilot are complete.

[`tokenizer_plan_v7.json`](tokenizer_plan_v7.json) is active and binds the formal sample, models, static
evaluation, and nomination without granting final selection. [`tokenizer_plan_v6.json`](tokenizer_plan_v6.json)
is its executable-input predecessor, [`tokenizer_plan_v5.json`](tokenizer_plan_v5.json) is its human-rights predecessor, and
[`tokenizer_plan_v4.json`](tokenizer_plan_v4.json) is the immutable scope predecessor that
preserves the complete v3 research decision and physically tied cost contract while replacing the
retired E1–E5
downstream binding with E1–E4 plus I1/I2. [`tokenizer_plan_v3.json`](tokenizer_plan_v3.json) remains
the immutable predecessor used by the already-checked scale-v2 accounting; candidate geometry and
costs are identical. [`tokenizer_plan_v2.json`](tokenizer_plan_v2.json)
preserves the corrected candidate plan with its now-superseded untied-cost assumption;
[`tokenizer_plan.json`](tokenizer_plan.json) and its 32,768/40,960/49,152 candidate set remain the
immutable v1 predecessor. The executable v6 config is
[`tokenizer_d5_v1/experiment.json`](tokenizer_d5_v1/experiment.json); it adopts, without repartitioning,
the firewall's already disjoint train/evaluation files.
[`source_registry_v2.json`](source_registry_v2.json) freezes the approved per-source quotas that sum
to each category target; a failed or newly prohibited source is replaced and requalified through a
new successor before sampling begins.
All final bounded successors are identity-bound in
[`tokenizer_inputs_blocked_v1.json`](tokenizer_inputs_blocked_v1.json). The manifest verifies all 30
files and exact 600/60 MB quotas but remains a pre-approval predecessor and deliberately non-executable
until production and real firewall gates pass. The Mistral payload is pinned independently in
[`mistral_tokenizer_baseline.json`](mistral_tokenizer_baseline.json).

## 1. Scope

Train an English-first SentencePiece tokenizer on equal UTF-8 bytes from web, code, math, synthetic,
science, and reference data. Programming syntax is retained as-is; surrounding natural language is
English-filtered before sampling. The sample is disjoint from data-mixture selection and benchmark
evaluation data.

The active custom candidates are BPE vocabularies of exactly 32,000, 32,768, and 40,960 pieces.
Mistral 32K is the baseline and fallback. The v1 49,152 candidate is preserved but not carried into
v2 and v3: corrected local evidence shows too little incremental compression for its
embedding/head cost. It remains absent from v4 and v5.
A 64K candidate remains excluded because the packed loader uses uint16 IDs, chat adds three role
tokens, and Shape A has one physically shared 2,048-wide input/LM-head matrix. At Shape A, 32,003,
32,771, and 40,963 effective rows cost exactly 65,542,144, 67,115,008, and 83,892,224 parameters.

All custom candidates use the same explicit trainer settings:

- byte fallback and 100% character coverage;
- identity normalization, no whitespace collapse, and no dummy prefix;
- explicit whitespace-only-piece support so indentation and repeated spaces can merge losslessly;
- deterministic preselected input order, no SentencePiece subsampling or shuffle, and one trainer
  thread;
- `<unk>=0`, `<s>=1`, `</s>=2`, and no padding token;
- exact model/vocabulary/sample hashes and path-independent model serialization.

The code slice uses the decontaminated Stack v3.1, Stack-Edu, Common Pile, Python-Edu, and Python PEP
artifacts recorded through [finding 175](../../findings/175_code_tokenizer_supplements.md). Their
55/20/10/10/5 allocation and duplicate precedence are technically frozen. None may enter the sample
until its rights disposition is accepted and the executable input manifest pins the final hashes.

The web slice has a viable 35/30/25/10 Ultra-FineWeb/FineWeb-Edu/DCLM/FineWeb-base bounded selection
through [finding 176](../../findings/176_web_tokenizer_bounded_sources.md). Its source identity,
language/quality/local safety, Gitleaks, partition-yield, and bounded-overlap gates pass. It remains
excluded from the executable sample until the evaluation firewall supplies immutable contamination
payloads and the upstream rights review is accepted.

## 2. Sample contract

The production sample contains at least 100,000,000 training bytes and 10,000,000 static-evaluation
bytes from each of the six categories: 600 MB train and 60 MB evaluation before JSONL framing.
Whole documents are retained, so each category may overshoot slightly.

The active config declares each firewall train/evaluation path and SHA-256 directly. Preparation
reverifies the production manifest and consumer authorization, validates every record/category/detail,
and copies the partitions without resampling. Their global deduplication and assignment are inherited
from the firewall. The older raw-input path remains available for fixtures and historical studies.

The tokenizer files must be produced or authorized by the hash-bound
[`firewall_plan_v4.json`](firewall_plan_v4.json) contract. `tokenizer_train` is the only partition available
to tokenizer training and `tokenizer_eval` is the only partition available to static evaluation.
Neither selection held-out nor either sealed audit identity may enter tokenizer or model training.
Fixture partitions never grant real consumer authority.

This tokenizer deduplication protects the qualification sample only. It does not replace the exact
and near-duplicate requirements for the full flagship corpus.

## 3. Running the pipeline

The exact executable config is frozen at `tokenizer_d5_v1/experiment.json`.
The matched LM pilot is governed by
[`tokenizer_pilot_plan_v3.json`](tokenizer_pilot_plan_v3.json). Its training corpus reruns global
exact/verified-near deduplication with all twelve prepared firewall candidate views taking precedence;
this conservative superset keeps both audits unopened while excluding every possible firewall record.
The completed pass retains 2.056B Mistral-reference tokens for the 1.2B pilot.

Build the balanced sample:

```bash
uv run --extra cpu python -m scripts.tokenizer_sample_prepare \
  research/flagship/tokenizer_d5_v1/experiment.json
```

Train all candidates and pin the baseline:

```bash
uv run --extra cpu python -m scripts.tokenizer_train \
  research/flagship/tokenizer_d5_v1/experiment.json \
  --prepare-baselines
```

Evaluate static metrics:

```bash
uv run --extra cpu python -m scripts.tokenizer_evaluate \
  research/flagship/tokenizer_d5_v1/experiment.json
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
- Shape-A physically shared embedding/LM-head parameters (`effective_vocab_size * 2,048`);
- difference from Mistral 32K.

The probe set includes consecutive whitespace, indentation, newlines, source code, mathematical
notation, ASCII, and non-ASCII text. Static evaluation has no selection authority. It eliminates
broken candidates and nominates the two custom Pareto candidates; it cannot show which tokenizer
makes a better language model.

The full 32,768/40,960/49,152 plus Mistral plumbing passes on deterministic generated fixtures:
requested vocabulary size, path-independent model identity, uint16 capacity, zero unknowns, probe
roundtrip, and parameter accounting. Fixture fertility cannot advance a candidate because the custom
models saw that generated alphabetic distribution and Mistral did not. The real comparison begins
only after the blocked 600/60 MB input is authorized and materialized.

A separate non-authoritative local study on 60.38 MB train and 6.61 MB held out explains the first
real-text custom deficit. Mistral enables SentencePiece `allow_whitespace_only_pieces`; the original
custom settings implicitly disabled it, heavily fragmenting indentation and repeated spaces in code
and markup. Enabling only that setting gives an exact-32,000-piece BPE 264.88 equal-category
tokens/KiB versus Mistral's 285.51 at identical embedding/head parameter cost. It improves all six
categories and all 30 bounded sources, repeats exactly at 32,768 pieces, and still leads after a
whitespace-collapse diagnostic. Tripling data, paragraph chunking, code reweighting, dummy-prefix
matching, and unigram did not explain or improve the original control. The versioned v2–v5 plans
contain this treatment; neither plan nor the local result has D5, LM-quality, audit-opening, or
launch authority.

If more than two custom candidates remain Pareto-valid, the pre-results
[`tokenizer_static_nomination_policy_v2.json`](tokenizer_static_nomination_policy_v2.json) chooses two
endpoints: best macro compression, then lowest parameter cost among the remaining frontier. The v3–v5
tied accounting halves every v2 embedding/head value without changing candidate ordering: moving
from 32,768 to 40,960 pieces adds exactly 16,777,216 Shape-A parameters rather than 33,554,432.
Frozen tie rules prefer the other objective, smaller vocabulary, then ID. Fewer than two valid distinct
Pareto candidates stops without improvisation. Static nomination advances LM-pilot candidates only;
it has no final D5 selection authority.

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

[`tokenizer_pilot_plan.json`](tokenizer_pilot_plan.json) freezes the executable analysis details. The
custom must pass the +0.01 macro and +0.02 category upper bounds in both fixed-document and fixed-FLOP
views. Paired document BPB deltas are averaged over seeds before a deterministic 10,000-replicate
bootstrap. Compute-to-quality uses Mistral seed 42's fixed-document final macro BPB; fixed wall-clock
is secondary only. The analyzer reports parameters, throughput, peak memory, active time, seed
dispersion, and a provisional two-finalist D5 handoff without opening the audit.

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

No existing checkpoint is converted to the new vocabulary. D5 is frozen before E1–E4, I1/I2, and C0
launch.
