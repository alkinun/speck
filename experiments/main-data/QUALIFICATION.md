# Main-data qualification packet

2026-09-20. CPU preparation only. The model, working mixture, pilot and evaluation remain unchanged.
[qualification-rules.json](qualification-rules.json) owns version-one eligibility rules;
[qualification-inputs.json](qualification-inputs.json) pins the bounded exclusion inputs and source
snapshot. Candidate partitions and a clean content screen never authorize training.

## Completed preparation

- Common full-read coverage is **176 files / 412,974 tokens** across the fixed 218 records.
  The [stylesheet closeout](../corpus-audit/stylesheet-cohort-review.json) adds the final three readings,
  distinguishing page, template and component roles. All 174 currently unheld records are now read;
  two earlier reads are now held, so the 176 readings overlap the 44 family holds. Original labels,
  weights and document boundaries remain intact; this batch adds no verified origins.
  The [coding guide](../../docs/coding.md#common-cohort-review) links the frozen earlier readings.
  The [family/provenance assessment](../corpus-audit/code-family-provenance.json) remains partial;
  the GoLLIE examples encountered earlier are exposed material when freezing final scoring coverage.
  No eligible yield, source ranking or admission follows.

- The [notice follow-up](../corpus-audit/code-notice-provenance.json) recovers pinned host bytes for
  eight purpose-selected records in four repositories: seven new origins and one reconfirmed origin.
  Four complete trees and four distinct Git-verified ancestor notices recover the Objective-C root
  license and Eclipse's code/non-code declaration. The Java files and CSS template retain unresolved
  notice/attribution questions. It brought linked origin coverage to 18/104 unheld Stack-Edu
  and 26/70 unheld Stack v3 records before the complete pinned-origin follow-up below.
  All assessment fields and holds remain unchanged; these counts do not establish eligible yield.

- The [pinned-origin follow-up](../corpus-audit/pinned-code-origins.json) attempts all 44 remaining
  unheld Stack v3 records at pinned commits and verifies 42 new origins. Two source files return 404.
  Four successful identities use path-specific metadata after recursive trees exceed the cap; their
  ancestor-notice searches remain incomplete. Fifteen distinct verified notices cover 22 selected
  records, without deciding applicability. A metadata-only inventory reconciles the existing cohort.

- The [retained-origin recovery](../corpus-audit/retained-code-origins.json) attempts bounded path
  histories for all 85 previously unattempted unheld Stack-Edu records, which lack dataset commit IDs.
  Ten exact-match origins and ten ancestor notices are verified. GitHub quota exhaustion blocks
  74 history requests and one tree check after source bytes match. These infrastructure outcomes
  do not establish content rejection or source quality; the ten successes are a C++-only prefix.
  The original dataset snapshot is not reconstructed by finding a matching later-visible revision.

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

The fixed-cohort reading pass is complete. Reuse its retained evidence for source-use and inventory
qualification; do not open another reading sample without a concrete unresolved coverage question.
The closeout reconciles the original observed tokens without treating held/read counts as disjoint:

| Cohort | Original files / tokens | Unheld, read files / tokens | Held, previously read files / tokens | Held, unread files / tokens |
| --- | --- | --- | --- | --- |
| Retained Stack-Edu | 138 / 459,615 | 104 / 287,668 | 1 / 596 | 33 / 171,351 |
| Stack v3 | 80 / 133,211 | 70 / 122,559 | 1 / 2,151 | 9 / 8,501 |

Neither cohort has unheld unread records. These are observed sample counts, not eligible-token
estimates or comparable quality rates. The next deliverable uses the existing inventory and receipts:

| Qualification gate | Evidence needed before a decision |
| --- | --- |
| Identity and source use | Resolve host/upstream revisions, consumed transformations and applicable notices; retain explicit unresolved outcomes |
| Families and exclusions | Extend known links to copied/transformed families and near-duplicates; freeze graph and scoring coverage before partitioning |
| Intended use | Record document role and contextual limitations; require independent oracles only for claimed verified exercises |
| Finite supply | Count deduplicated eligible tokens, exclusions and acquisition costs separately per bank; use supply and runtime to bound the shared study horizon |

Keep the original sampling frames fixed and reconcile roles and unresolved outcomes into the
inventory. Reading completion does not close any of these gates by itself.
The derived origin inventory reports disjoint evidence states, retaining each source's own frame:

| Cohort | Family-held files / observed tokens | Verified host files / observed tokens | Attempted unresolved files / observed tokens | Unattempted files / observed tokens |
| --- | --- | --- | --- | --- |
| Retained Stack-Edu | 34 / 171,947 | 28 / 52,329 | 76 / 235,339 | 0 / 0 |
| Stack v3 | 10 / 10,652 | 68 / 121,397 | 2 / 1,162 | 0 / 0 |

These are evidence categories, not eligible supply or pass rates. Of the 76 unresolved Stack-Edu
records, 75 are quota-blocked in the latest batch and one preserves an earlier failed attempt. All
records have an attempted outcome, but 74 new history requests returned no history data. Resume
quota-blocked work only after quota recovery or an available authenticated route, retaining prior
responses; do not reinterpret the C++-only successful prefix as representative. The acquisition now
stops fresh API requests after observed quota exhaustion; that guard was added after this batch.
Keep the two Stack v3 404s outside training absent new evidence; do not substitute revisions or
repeat unchanged failures. Continue independent web/math qualification while API recovery is blocked.
Four Stack v3 path-verified files need ancestor-notice searches using bounded path traversal.
Host verification does not resolve copied or vendored source provenance.
The recovered Eclipse NOTICE explains its code/non-code labels; dependency applicability remains open.
Resolve the Java restrictive wording and Ororus template attribution alongside host/upstream context;
file headers and dataset license metadata do not independently authorize source use. Review embedded application data in
dumps separately from schema quality; a test path or snapshot is not an independent result oracle.
Use the updated 44-record family holds, including GoLLIE, vendored Pylint and the metadata-identified
LeetCode collection and Borealis `svcomp` record. The metadata-only holds are conservative, not proven
matches to the intended scoring suite; do not inspect their task text to tune exclusions. Extend the
partial graph beyond known repository/alias/dependency links; no exact consumed-byte duplicates in
the 218 records does not settle near-duplicates or transformed copies. Arcade 2.5.7 and stringutils 0.3.0 installed identities
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
[proposed study](README.md#research-before-the-main-run); the completed reading pass cannot select a
corpus winner or supply the full confirmation horizon. Remaining source comparisons are:

| Lane | Candidates | Required comparison |
| --- | --- | --- |
| Natural web | Ultra-FineWeb English/HQ; FineWeb-Edu control; DCLM baseline/Edu | Separate selection thresholds, source provenance, boilerplate, topic/language diversity, duplication and retained tokens |
| Math | FineMath 4+; UltraData-Math L2; filtered InfiWebMath 4+; then Nemotron-CC-Math 4plus | Intact questions/solutions, conservative arithmetic triage, checkable correctness, topic/difficulty coverage and shared source families |
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

### Retained inventory closeout — 2026-09-19

The [data-readiness receipt](../corpus-audit/data-readiness.json) binds the completed CPU work,
replay scripts, metadata ledgers, failures and source identities. It adds no training admissions.

| Lane | Completed measurement | Consequence for preparation |
| --- | --- | --- |
| Natural web | All 290,761 retained HQ documents: 351,718,255 tokens; 162 exact duplicate copies / 162,768 tokens; 313 unique contents / 317,562 tokens shared with retained FineWeb-Edu | Count shared content once. Cutoff and review-only flags remain unchanged; no quality ranking follows. |
| Retained non-code union | Six existing banks plus HQ: 5,661,264 byte-distinct documents / 7,058,353,273 tokens | Includes alternative Math L2 stock; excludes code. This is exact-content accounting, not an eligible production union or a replacement for the historical pilot-supply total. |
| Math | All 753,071 FineMath and 169,058 L2 texts match their indices. No exact cross-source copies; one normalized full-document link | Normalization may change math meaning; do not automatically remove that pair. Every retained L2 row lacks host metadata, so source-family attribution and correctness remain open. |
| Context inventory | Complete-document length bands for all seven non-code banks; existing science/reference stocks contain longer documents | Length alone cannot establish coherent bundles, useful context or a qualified runtime ceiling. Inspect retained material before opening the 298.67 GB English FinePDFs inventory. |
| Thinking/tool SFT | All 500K rows checked structurally; 424,463 compatible, 75,537 first-failure rejections. Original 2,560-row sample retokenized through the training adapter | Keep unsupported and incomplete trajectories intact. Shared prompts need split links; sampled lengths do not establish full-stock token totals or task success. |
| Conditional RL | Full 32,412 Math / 11,872 Knowledge rows; 25 Code / 60 Long-Context prefix records | Reconcile prompt/reference links before verifier work. All inspected long-context queries exceed 32K before responses; none enters initial 16K RL unchanged. |

The external finite inventory accounts for FineWeb-Edu as one pool even when it supplies both web
roles. After within-HQ and exact FineWeb overlap accounting, **351,237,925 HQ tokens** remain distinct
from the retained control, bounding a 25% candidate bank at **1,404,951,700 total one-pass tokens**
before further exclusions. Natural code separately bounds 30% exposure at **1,589,249,490** tokens.
These are stock upper bounds, not an executable common horizon: qualified checked-code and
refined-math inventories are still missing. Obtain qualified supply or explicitly revise the bounded
research recipe/horizon. Never silently replace refined shares, repeat scarce sources or treat the
40/90-hour arm caps as guaranteed token supply. Compute reservations and main working weights are unchanged.

The pinned downstream inventories reconcile against their release metadata. The Nemotron 4plus
listing has 46 files / 62.19 GB but unusable masked LFS hashes; no shard was acquired or verified.
The RL code files total about 184 GB, making bounded test-payload inspection preferable to an
unpriced full download. The [assistant contract](../../docs/assistant.md) owns complete fit counts,
reference discrepancies and the retained stock's missing long tails. The source cards' verification
claims do not replace our independent answer/environment checks.

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

The retained-data table and acquisition routes are now recorded; source-use, family and correctness
decisions still gate accepted/rejected inventory and usable yield. Then freeze the eligible banks and
a bounded baseline/candidate comparison from matched fresh initializations. Run it before main pretraining and record the
starting-recipe decision before freezing the production manifest. Neither source inspection nor
a later continuation comparison substitutes for this gate.
Post-training still needs reasoning/tool verification and complete long examples; its 1.5M working
target is not supplied by these pretraining checks.
