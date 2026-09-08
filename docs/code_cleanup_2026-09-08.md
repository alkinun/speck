# Code cleanup audit — 2026-09-08

## Review scope

The repository-wide inventory covered 218 tracked Python files: 54 library modules, 81 scripts,
and 83 test modules. Review combined AST/reference and duplicate-body scans, configured linting,
additional bug-pattern linting, source-hash dependency checks across tracked JSON/Markdown, and
manual inspection of model/cache, training, data-loading, evaluation, export, and data-processing
paths. The baseline suite passed with 631 tests and 8 skips.

No clearly unreferenced top-level functions were identified for deletion. The actionable findings
were duplicated utilities, redundant branches, validation order, and reproducible edge-case bugs.
The static scan is not proof that every runtime path is exercised.

## Applied cleanup

| Area | Finding and change |
| --- | --- |
| File I/O | Consolidated 20 hashing/JSON-writing implementations across 18 modules into `speck/io.py`. Hash digests and JSON serialization conventions are preserved. JSON reports now use independent temporary files, retain the old report on failed publication, and clean up temporary files. |
| Chat validation | Local tokenization and the evaluation API now share one conversation validator. Assistant-first conversations, non-string roles, and reserved control tokens are rejected consistently before generation. |
| Evaluation output | Overlapping stop strings now use the earliest match in the original decoded output, independent of request order. Completion usage counts actual generated token IDs rather than re-tokenizing trimmed text. |
| Evaluation HTTP lifecycle | Invalid UTF-8 is reported as a client error. The serving socket is closed on exit, and startup reports the bound port. |
| Packed loading | Empty reads return an empty array of the requested dtype. Negative counts are rejected before slicing. Batch geometry is checked before opening datasets or calculating strides. Identical epoch-wrap branches were merged. |
| INT8 cache | FP16 scale underflow could erase small values; rounding a scale downward could saturate the largest value. Scales are now rounded upward to a nonzero representable value and used consistently for quantization. Cache storage geometry is unchanged. |
| Model maintenance | Removed a redundant empty-cache branch and merged identical DeltaNet/KDA state-allocation code. Vocabulary growth clears both total- and active-parameter expectations. |
| SFT configuration | Empty/malformed sequence buckets and invalid numeric settings now produce configuration errors. Boolean counts, fractional intervals, NaN, and infinity are rejected before runtime initialization. |
| Research validation | Probability validation now checks types before making numeric comparisons, so malformed contracts raise `ValueError` instead of incidental `TypeError`. |
| Formatting | Applied the configured formatter to unpinned tracked Python files and removed imports made unused by the refactor. |

Regression tests cover failed and concurrent JSON writes, multi-shard reads, malformed requests over
HTTP, generation accounting, quantization at several magnitudes, vocabulary resizing, and invalid
configuration inputs. Existing model, loader, training, export, and research-contract tests cover
the shared-helper migrations and behavior-preserving refactors.

Final verification in the installed CPU development environment:

- `.venv/bin/pytest -q`: **676 passed, 8 skipped**, including 45 additional test cases.
- `.venv/bin/ruff check .`: passed.
- Ruff formatting check on all 47 added/modified Python files: passed.
- `git diff --check`: passed.
- Source-pin comparison: all 80 currently pinned Python files preserved exactly.

## Follow-up refinement

After committing the initial cleanup, local inference and instruction evaluation were migrated to
`speck/generation.py`. Their duplicated loops both computed unused logits after emitting the final
requested token. The shared loop performs one prompt prefill and only the decode calls needed for
subsequent tokens. It preserves greedy/top-k decoding and excludes EOS from returned token IDs.

Sampling settings are validated before checkpoint loading in the inference CLI and before cache
allocation in the shared helper. Regression coverage includes token-budget and EOS stopping,
top-k clamping, invalid inputs, and cached-generation parity with a full-prefix model reference.

Follow-up verification: **696 passed, 8 skipped** in the full CPU suite, with 20 additional test
cases. Lint and changed-file formatting checks pass, and all 80 source-pinned Python files remain
byte-identical.

## Deferred findings in hash-pinned code

The requested provenance policy is to preserve source pins. All 80 Python files whose current
SHA-256 appears in checked-in JSON/Markdown remain byte-identical to the pre-cleanup revision.
Recorded research outputs were not rehashed to imply that they were produced by revised code.

### High priority: failed checkpoint overwrite loses the completed checkpoint

`speck/checkpoint.py:save` deletes the existing model, optimizer, metadata, and completion marker
before writing replacement temporary files. A local failure-injection check saved step 1, made the
next `torch.save` raise `OSError`, and retried step 1. `latest()` then returned `None`: the original
completed checkpoint was lost. Deletion also removes data files before removing their completion
marker.

Requalify this module with an explicit overwrite contract and failure-injection tests covering
serialization and publication failures. A completed checkpoint must remain recoverable when a
replacement fails.

### Performance/resource handling: repeated opening of cached deduplication sources

`speck/production_data.py:_candidate_text` uses
`handles.setdefault(source_index, Path(...).open("rb"))`. Python evaluates the `open()` call on
every lookup, even when the handle already exists. A local fixture confirmed two opens for two
lookups of the same source. Replace this with an explicit cache-miss check during requalification;
test that repeated lookups reuse one open handle and that handles close on failure.

### Maintainability: duplicated data-pipeline helpers and long validators

The source-qualified data modules retain many identical `_sha256`, `_write_json`, and digest
validation implementations. Migrate those to shared utilities when requalifying the pipelines;
preserve each writer's serialization and durability requirements, which are not all identical.

Several large functions also warrant dedicated decomposition: `validate_external_suite` and
`_validate_evaluations` mix independent suite-specific validations, while `preprocess_sources` and
source samplers combine acquisition, filtering, attribution, and publication. Extract domain-level
steps with contract tests rather than splitting solely to meet a line-count target.

## Formatting constraint

The repository-wide formatting check still reports 17 pre-existing failures in pinned files:

- `scripts/ruler_case_prepare.py`
- `speck/code_contamination.py`, `speck/code_contamination_successor.py`,
  `speck/code_near_duplicates.py`
- `speck/common_pile_code.py`, `speck/common_pile_peps.py`, `speck/gitleaks_filter.py`
- `speck/python_edu.py`, `speck/stack_edu.py`, `speck/stack_v3.py`,
  `speck/stack_v3_expand.py`, `speck/stack_v3_refine.py`
- `speck/text_near_duplicates.py`, `speck/tokenizer_experiment.py`, `speck/web_sample.py`
- `tests/test_data_launch.py`, `tests/test_production_data.py`

Those require source-evidence requalification before formatting. The formatter configuration and
evidence assertions remain enabled. GPU-specific kernels and remote publication were not exercised
by this CPU cleanup validation.
