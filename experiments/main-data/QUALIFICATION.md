# Main-data qualification packet

2026-09-19. CPU preparation only. The model, working mixture, pilot and evaluation remain unchanged.
[qualification-rules.json](qualification-rules.json) owns version-one eligibility rules;
[qualification-inputs.json](qualification-inputs.json) pins the bounded exclusion inputs and source
snapshot. Candidate partitions and a clean content screen never authorize training.

## Completed preparation

- Common full-read coverage is **101 files / 159,514 tokens** across the fixed 218 records.
  The [latest shell/build/config review](../corpus-audit/build-config-cohort-review.json) adds 24
  readings: twelve scripts, eight authored configs, two generated build files and two localization
  files. It reuses one verified origin and one unavailable-original outcome, with no new checks.
  There are 43 held records and 76 unheld records still unread; source labels and weights remain intact.
  The [coding guide](../../docs/coding.md#common-cohort-review) links the frozen earlier readings.
  The [family/provenance assessment](../corpus-audit/code-family-provenance.json) remains partial;
  the GoLLIE examples encountered earlier are exposed material when freezing final scoring coverage.
  No eligible yield, source ranking or admission follows.

- The [Stack v3 feasibility probe](../corpus-audit/stack-v3-feasibility.json) inventories two
  revisions and samples 2,699 repository rows / 47,826 files from the current corrected pin.
  The [origin follow-up](../corpus-audit/stack-v3-origins.json) links twelve originals to pinned Git
  blobs and characterizes their transformations and ancestor notices. Source-use applicability,
  exclusions and eligible yield remain unresolved. This is separate from the frozen retained-stock audit.
  The [broader preflight](../corpus-audit/stack-v3-broader.json) acquired all sixteen frozen groups
  (510.02 MiB), reconciling 29,347 rows / 379,942 files. The fixed 44-repository / 80-file cohort
  has four content flags and eight known-family holds; 72 files remain unresolved. Acquisition,
  selection and screening replay identically offline; content/source-use review remains open.
- The [expansion packet](../corpus-audit/code-expansion.json) pins both candidate inventories,
  verifies one new Python metadata shard and recovers 16 blobs. Thirteen length-matched files
  contain 9,541 tokens; four content flags and unresolved eligibility prevent admission. The
  [origin follow-up](../corpus-audit/code-application-origins.json) recovers all four application
  revisions/notices but establishes no independently verified examples.
- The [stratified preflight](../corpus-audit/code-yield-result.json) recovers 138 files / 459,615
  tokens across 11 languages and 72 strata. Thirty content flags hold 31 sampled files by known
  family. Exact replay succeeds; full source-use/quality/behavioral assessment remains open.
- The [immutable bundle follow-up](../corpus-audit/code-bundles.json) recovers 19 linked files /
  5,650 tokens across four repositories. Four content flags propagate to 15 quarantined files.
  Static review finds weak Python assertions and a contradictory Go test expectation; the Go
  candidate partition is not admission. No corpus code ran. Co-presence alone fails as a bundle
  criterion; practical coverage, explicit linkage and independent oracles are the next gates.
- The [retained-code census](../corpus-audit/code-supply.json) reopens all 1,999 v1/v2 archives,
  reproducing 714,369 files / 476,774,847 tokens. It records role hints and named-repository
  co-presence, with no populated commit fields. The twelve benchmark names hold 84 files;
  including prior content-held components/aliases gives 123 files / 209,751 tokens. This is
  propagation of existing evidence, not a whole-stock run of the newer content screen or admission.
- The [practical-code receipt](../corpus-audit/practical-code-checks.json) closes the Python and
  JS/TS feasibility checks and the 13,236-file multilingual coverage sample. Its exact replay
  scripts, source snapshots and outputs remain outside Git. These small checks establish a method,
  not a corpus-wide correctness rate or enough flagship supply.
- MBPP+ (378 rows), fourteen MultiPL-E variants (HumanEval/MBPP in C++, Java, JavaScript,
  TypeScript, Go, Rust and Shell), and SWE-bench Verified (500 rows) are acquired at immutable
  revisions and checked against the publisher's LFS content hashes. They are exclusion inputs,
  not training data or newly scored benchmarks. C, SQL and additional languages need their own
  evaluation coverage if selected; the current list does not imply all-language coverage.
- SWE-bench Verified contributes twelve repository names to a conservative family hold list.
  Source-verified fork/rename aliases and derived/duplicate relationships propagate holds through
  the whole connected component. Unresolved origins/parents quarantine the component.
- A separate content index per benchmark lane preserves matching signals shared across translated
  tasks. Source prompts, code, public tests and patches are included where named in the manifest.
  Opaque/private test payloads are not unpacked or executed. Family grouping of translations is
  still required when preparing a scoring protocol; row counts are not independent task counts.

The existing exact/fragment matcher is a conservative screen, not proof of semantic contamination
or its absence. Any language/domain-specific improvement requires versioned controls and a new
receipt. Test-derived output should report match counts, not held-out task text.

The [bounded result](qualification-result.json) screens 37 files against 22 lanes / 18,958 rows
(including translated tasks). Seven files trigger content flags; family propagation quarantines
22 files. The other 15 have candidate partitions only and remain outside training. The `frans`
test file and task-queue README trigger additional flags even though their implementation-only
pilot screen was clear. This is a reason to screen complete bundles, not a claim of proven leakage.
The added LiveCodeBench lane flags one already-held file; the total held-file count remains 22.

## Reproduce the bounded screen

From the project environment, with the external artifact store available:

```bash
PYTHONPATH=. python experiments/main-data/check_qualification.py \
  experiments/main-data/qualification-inputs.json
```

The command verifies hashes, scans 37 source files, assigns candidate family partitions, and emits
a JSON summary. It performs no acquisition, corpus execution, training admission or model scoring.
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

Reuse retained packets and current receipts. Common full-read coverage now includes 101 records;
continue the 76 currently unheld records outside that coverage with the same origin/notice,
intended-use, content and family gates. Keep this sampling frame fixed; reconcile observed file roles
and unresolved eligibility into the dataset inventory rather than replacing difficult outcomes.
Use the updated 43-record family holds, including GoLLIE, vendored Pylint and the metadata-identified
LeetCode collection. That last hold is conservative,
not a proven task match; do not inspect solution text to tune exclusions. Extend the partial graph
beyond known repository/alias/dependency links; no exact consumed-byte duplicates in the 218 records does not
settle near-duplicates or transformed copies. Arcade 2.5.7 and stringutils 0.3.0 installed identities
are verified, but complete upstream revision/notice applicability remains open. Keep package
artifact hashes, host origins and consumed identities separate; ancestor notices alone are insufficient.
Held records stay in their original denominators; do not replace them or expose held-out task text.
The external assessment retains every original identity/weight, observed token count, existing
hold and unresolved outcome. Resolve conservative matches and source-family links before admission;
natural code does not require independent tests, while claimed verified exercises do.

The pair-aware Python diagnostic passes fourteen controls on each of CPython 3.10.20 and 3.14.6;
it confirms the known redaction-induced syntax failure but is not a general content filter.
Require a verified parseable original and exact placeholder-only changes before assigning that
diagnosis. Preserve notices and consumed identities; do not restore redacted source text.
Parser success cannot validate runtime behavior, embedded doctests, redacted assertion meaning
or contamination. Legacy syntax and intentionally invalid fixtures need context. Low-information
examples need a declared selection rule rather than ad hoc removal.

Report each cohort against its own sampling frame and weights. The retained Stack-Edu and release
Stack v3 samples do not support raw pass-percentage comparisons or pooled eligible yield.
Use completed gates to prepare finite candidate inventories for the
[proposed study](README.md#research-before-the-main-run); the current partial review cannot select a
corpus winner or supply the full confirmation horizon. Remaining source comparisons are:

| Lane | Candidates | Required comparison |
| --- | --- | --- |
| Natural web | Ultra-FineWeb English/HQ; FineWeb-Edu control; DCLM baseline/Edu | Separate selection thresholds, source provenance, boilerplate, topic/language diversity, duplication and retained tokens |
| Math | FineMath 4+; UltraData-Math L2; then Nemotron-CC-Math 4plus | Intact questions/solutions, checkable correctness, topic/difficulty coverage and shared source families |
| Natural code | Retained Stack-Edu; Stack v3; source-resolved UltraData-Code L2 | Multilingual practical roles, source/test/docs linkage, immutable origins, dependency cost and eligible token yield |
| Generated material | Web/math L3 and checked code derivatives | Source grounding, independent answer/test verification, teacher lineage and rejection rates; held separately from natural stock |

The [pinned UltraFineWeb variant inspection](../corpus-audit/web-variants.json) distinguishes
`data/ultrafineweb_en` (the default English split, described as FineWeb-derived) from
`data/ultrafineweb_l1_en_hq` (the publisher's newer L1-derived selected route, claiming crawl coverage
through CC-MAIN-2025-51). Inspect both against FineWeb-Edu before choosing a bank. The separate
`ultrafineweb_en_v1_4` directory remains unqualified; its name alone does not establish selection
semantics or superiority. That initial inspection acquired only metadata/card bytes.

The subsequent [bounded natural-web inspection](../corpus-audit/NATURAL_WEB.md) verifies six HQ
shards (144,876 documents), 48 default viewer rows and the existing 64-document FineWeb-Edu
control. HQ actually provides `uid`, `content`, JSON `meta` and `dataset_index`; its metadata
retains URLs and WARC identities throughout the inspected shards. Default English has only
`content`, `score`, `source`. Prioritize HQ for further provenance-resolved qualification;
quality superiority and eligible supply remain unestablished. Eleven assistant-reviewed examples
include missing equations despite a high classifier score, template-heavy text and navigation
tails. The [follow-up inventory and DCLM preview](../corpus-audit/WEB_INVENTORY_DCLM.md) now
closes the HQ inventory at 6,000 files / 477.97 GB compressed and completes a 959 MB, 12-shard
sample across the full inventory: 290,761 documents, 192 sampled and 24 reviewed texts/excerpts.
The [extraction follow-up](../corpus-audit/web-filter-validation.json) recovers three exact
archived captures and compares flags on 32 fresh documents. Candidate flags remain review-only;
source-aware repair, source eligibility and usable-token counts remain open. The retained-code census
above closes inventory accounting; immutable linkage and expanded supply remain the next code work.
The inventory follow-up also distinguishes DCLM-Edu's `edu_int_score >= 3` from
`edu_score >= 3`; the two predicates retain different preview records. DCLM viewer indexes
are partial, so their samples establish schema/content questions only. Score/domain-stratified
comparison, source-use review and joint deduplication remain open. No corpus-wide quality
ranking follows from these sampling frames.

For each new source, pin release/card/serialization first, then take a deterministic sample across
length, domain/language and upstream-score bands before labeling. Use the same bands and review
rubric for comparisons; report sample denominators and sampling weights. Keep independent correctness
checks separate from extraction/format judgments. Do not treat assistant review as independent human
annotation. Preserve hard but useful examples, rare domains, non-English coverage and long documents.

Record raw tokens, deduplicated unique tokens, eligible tokens, rejected counts/reasons, source-family
coverage and CPU/storage/teacher cost separately. Use the frozen Mistral tokenizer and retain exact
serialization. Overlapping parent/filtered releases count once. The working eight-bank weights in
[plan.json](plan.json) remain hypotheses; the chat suggestion of an 80/20 staged mixture has not
become a launch recipe. No 100B qualified supply or GH200 compute-feasibility claim follows from this packet.

The deliverable is one source-comparison table with auditable accepted/rejected examples and a
costed acquisition route. Only then freeze the eligible banks and a bounded baseline/candidate
comparison from matched fresh initializations. Run it before main pretraining and record the
starting-recipe decision before freezing the production manifest. Neither source inspection nor
a later continuation comparison substitutes for this gate.
Post-training still needs reasoning/tool verification and complete long examples; its 1.5M working
target is not supplied by these pretraining checks.
