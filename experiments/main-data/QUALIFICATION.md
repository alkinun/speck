# Main-data qualification packet

2026-09-23. CPU preparation only. The model, working mixture, pilot and evaluation remain unchanged.
[qualification-rules.json](qualification-rules.json) owns version-one eligibility rules;
[qualification-inputs.json](qualification-inputs.json) pins the bounded exclusion inputs and source
snapshot. Candidate partitions and a clean content screen never authorize training.

## Completed preparation

Current state only; the [corpus-audit record](../corpus-audit/README.md) keeps the per-batch
history and each receipt owns its detail. None of this establishes eligible yield or admission.

| Evidence | What it establishes now |
| --- | --- |
| [Retained-code census](../corpus-audit/code-supply.json) | 714,369 files / 476,774,847 tokens reopened from all 1,999 archives; benchmark names and prior content holds cover 123 files / 209,751 tokens by propagation, not a whole-stock content screen |
| [Stratified preflight](../corpus-audit/code-yield-result.json) and [Stack v3 broader preflight](../corpus-audit/stack-v3-broader.json) | Fixed review cohorts of 138 retained Stack-Edu and 80 Stack v3 files (218 records), replayed exactly offline |
| [Reading closeout](../corpus-audit/stylesheet-cohort-review.json) | 176 files / 412,974 tokens read; all 173 currently unheld records are read |
| [Joint partitions](family-partition.json) | 45 cohort family holds after the exact firewall match `strategist922/bigpipe` `SPEC/Acl.md`; candidate 90/5/5 family buckets |
| [Origin recovery](../corpus-audit/code-origin-recovery-20260922.json) | Verified host origins for 58/103 unheld Stack-Edu and 68/70 unheld Stack v3 records; 44 Stack-Edu lookups quota-blocked, one earlier failure, two Stack v3 404s |
| [Practical-code checks](../corpus-audit/practical-code-checks.json) | Python and JS/TS feasibility method and a 13,236-file multilingual coverage sample; not a corpus-wide correctness rate |
| [Bounded screen](qualification-result.json) | 37 files against 22 lanes / 18,958 rows: seven content flags, 22 files quarantined by family, 15 with candidate partitions only |

Exclusion inputs are acquired at immutable revisions and checked against publisher hashes: MBPP+
(378 rows), fourteen MultiPL-E variants (HumanEval/MBPP in C++, Java, JavaScript, TypeScript, Go,
Rust and Shell), SWE-bench Verified (500 rows) and LiveCodeBench `release_v6`. They are exclusion
inputs, not training data or newly scored benchmarks; other languages need their own coverage if
selected. SWE-bench Verified contributes twelve repository names to the family hold list. Aliases and
derived/duplicate links propagate holds through the whole connected component, and unresolved
origins quarantine it. Each benchmark lane keeps its own content index; opaque/private test payloads
are never unpacked or executed, and translated rows are not independent tasks.

The exact/fragment matcher is a conservative screen, not proof of contamination or its absence.
Improving it needs versioned controls and a new receipt; test-derived output reports match counts,
not held-out task text.

## Reproduce the bounded screen

From the project environment, with the external artifact store available:

```bash
PYTHONPATH=. python experiments/main-data/check_qualification.py \
  experiments/main-data/qualification-inputs.json
```

The command verifies hashes, scans 37 source files, assigns candidate family partitions, and emits
a JSON summary. The 2026-09-20 replay reproduced seven content-flagged files and 22 quarantined
files. It performs no acquisition, corpus execution, training admission or model scoring.
Hash/row-count mismatches fail closed. New records require a new input snapshot and receipt.

## Remaining exclusion work

LiveCodeBench is pinned to `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, `release_v6`. All six
raw files (4,485,994,821 bytes) are retained and checked against the pinned publisher hashes.
The [projection receipt](livecodebench-exclusion.json) binds the compact public-text index to every
raw row. Its file set is checked against literal release metadata in the pinned loader, without
executing that loader. Task identities use platform plus question ID. Private-test strings remain
opaque and are omitted from the index; task text, starter code and public tests are retained.
The verified projection contains 1,055 tasks from 2023-05-07 through 2025-04-06 in 2,281,423 bytes.
Observed cumulative counts match the card's releases v1–v5 (400, 511, 612, 713, 880), followed by
1,055 at v6. Positive/negative exclusion controls pass, and the first 511 projected rows exactly
match the independently produced two-file schema-check output. No benchmark tasks were scored.

Rebuild into a fresh directory with the pinned raw files and preparation manifest retained locally:

```bash
PYTHONPATH=. python -m scripts.livecodebench_exclusion \
  /mnt/speck-data/speck/data-qualification-20260919/livecodebench/projection-inputs.json \
  /mnt/speck-data/speck/data-qualification-20260919/livecodebench/acquisition.json \
  /mnt/speck-data/speck/data-qualification-20260919/livecodebench/projection-repeat
```

This closes public-text coverage of this release. A final scoring window, subsequent releases,
private-test matching and external solution-mirror coverage are separate decisions. Do not claim
comprehensive contamination removal or a fresh live evaluation from this fixed historical release.

SWE-bench Verified is the initial repository hold list, not a complete agentic-coding protocol.
Freeze any additional repair/agent benchmarks before related acquisition or synthesis. Resolve
repository aliases from evidence; the splitter cannot discover forks, copied tasks or near-duplicates
by itself. Freeze the complete graph, deduplication configuration and hashes before assigning the
production inventory. Adding edges can change partitions. The reviewed feasibility cohorts are not
a fresh blind evaluation set.

## Next bounded data packet

The fixed-cohort reading pass is complete; do not open another reading sample without a concrete
unresolved coverage question. Current origin evidence, disjoint per cohort in its own frame:

| Cohort | Family-held files / observed tokens | Verified host files / observed tokens | Attempted unresolved files / observed tokens | Unattempted files / observed tokens |
| --- | --- | --- | --- | --- |
| Retained Stack-Edu | 35 / 172,401 | 58 / 136,124 | 45 / 151,090 | 0 / 0 |
| Stack v3 | 10 / 10,652 | 68 / 121,397 | 2 / 1,162 | 0 / 0 |

These are evidence categories, not eligible supply or pass rates, and the two cohorts do not support
pooled yield or raw pass-rate comparisons. Each gate needs its own evidence:

| Qualification gate | Evidence needed before a decision |
| --- | --- |
| Identity, origins and notices | Resolve host/upstream revisions, consumed transformations and the applicable notices the signed source-use decision requires; retain explicit unresolved outcomes |
| Families and exclusions | Extend known links to copied/transformed families and near-duplicates; freeze graph and scoring coverage before partitioning |
| Intended use | Record document role and contextual limitations; require independent oracles only for claimed verified exercises |
| Finite supply | Count deduplicated eligible tokens, exclusions and acquisition costs separately per bank; use supply and runtime to bound the shared study horizon |

Open code items:

- Resume the 44 quota-blocked Stack-Edu lookups only after quota recovery; keep the earlier failure
  and the two Stack v3 404s outside training absent new evidence, without substituting revisions.
- Four path-verified Stack v3 files need bounded ancestor-notice searches. Host verification does
  not resolve copied or vendored provenance.
- Resolve the Eclipse dependency applicability, the Java restrictive wording and the Ororus
  template attribution with host/upstream context. File headers and dataset licence metadata do not
  authorize use on their own; a test path or snapshot is not an independent result oracle.
- Keep all 45 family holds, including GoLLIE, vendored Pylint, the metadata-identified LeetCode
  collection and the Borealis `svcomp` record. Metadata-only holds are conservative; do not inspect
  their task text to tune exclusions. Held records stay in their original denominators.
- Extend the graph beyond known repository/alias/dependency links: no exact consumed-byte duplicates
  among the 218 records does not settle near-duplicates or transformed copies. Keep package artifact
  hashes, host origins and consumed identities separate.
- The pair-aware Python diagnostic passes fourteen controls on CPython 3.10.20 and 3.14.6 and
  confirms one redaction-induced syntax failure; it is not a content filter. Parser success cannot
  validate behavior, redacted assertions or contamination. Do not restore redacted source text.

Remaining comparisons among the [selected sources](source-registry.json):

| Lane | Candidates | Required comparison |
| --- | --- | --- |
| Natural web | Ultra-FineWeb HQ; FineWeb-Edu control | Source provenance, boilerplate, topic/language diversity, duplication and retained tokens at the unchanged cutoff |
| Math | FineMath 4+; filtered InfiWebMath 4+ | Intact questions/solutions, conservative arithmetic triage, checkable correctness, topic/difficulty coverage and shared source families |
| Natural code | Retained Stack-Edu; Stack v3 | Multilingual practical roles, source/test/docs linkage, immutable origins, dependency cost and eligible token yield |

Not selected on 2026-09-22: DCLM baseline/Edu, UltraData-Math L2, Nemotron-CC-Math 4plus,
source-resolved UltraData-Code L2, and generated web/math L3 or checked-code derivatives. Each
enters only through a recorded replacement and a revised freeze. The HQ route
(`data/ultrafineweb_l1_en_hq`, [variant inspection](../corpus-audit/web-variants.json)) is selected;
its inventory is 6,000 files / 477.97 GB compressed, and extraction flags from the
[follow-up](../corpus-audit/web-filter-validation.json) remain review-only.

### Retained inventory closeout — 2026-09-19

The [data-readiness receipt](../corpus-audit/data-readiness.json) owns the full measurements. Still
current: all 290,761 retained HQ documents hold 351,718,255 tokens, and after within-HQ and exact
FineWeb-Edu overlap **351,237,925 HQ tokens** remain distinct from the control, bounding a 25% bank at
**1,404,951,700 total one-pass tokens** before further exclusions. Natural code bounds its 35% share
at **1,362,213,848** tokens, the binding constraint. These are stock upper bounds, not an executable
common horizon; never treat the 30/40/60-hour arm caps as guaranteed token supply. The closeout
also covers math index reconciliation, context length bands, the 500K SFT structural census
(424,463 compatible rows) and the bounded RL inventory; the
[assistant contract](../../docs/assistant.md) owns SFT fit counts.

For each new source, pin release/card/serialization first, then sample deterministically across
length, domain/language and upstream-score bands with a shared rubric, reporting denominators and
weights. Keep correctness checks separate from extraction judgments; assistant review is not
independent human annotation. Record raw, unique, eligible and rejected tokens with reasons,
source-family coverage and CPU/storage/teacher cost separately, using the frozen Mistral tokenizer.
Overlapping parent/filtered releases count once. The work order is in
[PLAN.md](../../PLAN.md#immediate-order-of-work).
