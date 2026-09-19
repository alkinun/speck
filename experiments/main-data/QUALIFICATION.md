# Main-data qualification packet

2026-09-19. CPU preparation only. The model, working mixture, pilot and evaluation remain unchanged.
[qualification-rules.json](qualification-rules.json) owns version-one eligibility rules;
[qualification-inputs.json](qualification-inputs.json) pins the bounded exclusion inputs and source
snapshot. Candidate partitions and a clean content screen never authorize training.

## Completed preparation

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

The [bounded result](qualification-result.json) screens 37 files against 21 lanes / 17,903 rows
(including translated tasks). Seven files trigger content flags; family propagation quarantines
22 files. The other 15 have candidate partitions only and remain outside training. The `frans`
test file and task-queue README trigger additional flags even though their implementation-only
pilot screen was clear. This is a reason to screen complete bundles, not a claim of proven leakage.

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

LiveCodeBench is pinned to `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, including hashes and sizes
of six raw files totaling 4,485,994,821 bytes. Payloads were not downloaded. Before deriving tasks,
choose a declared release/date window and acquire an authenticated text/identity projection or
bounded payload route; verify complete row coverage and build its exclusion adapter. Reading an
arbitrary prefix of a large file is not complete exclusion. Do not execute its dataset script or
deserialize opaque private-test objects during inspection.

SWE-bench Verified is the initial repository hold list, not a complete agentic-coding protocol.
Freeze any additional repair/agent benchmarks before related acquisition or synthesis. Resolve
repository aliases from evidence; the splitter cannot discover forks, copied tasks or near-duplicates
by itself. Freeze the complete graph, deduplication configuration and hashes before assigning the
production inventory. Adding edges can change partitions. The reviewed feasibility cohorts are not
a fresh blind evaluation set.

## Next bounded data packet

Reuse the retained 924-document packet and current version receipts rather than repeating the same
spot checks. Prepare comparable source-level evidence in the following order:

| Lane | Candidates | Required comparison |
| --- | --- | --- |
| Natural web | Ultra-FineWeb English/HQ; FineWeb-Edu control; DCLM baseline/Edu | Separate selection thresholds, source provenance, boilerplate, topic/language diversity, duplication and retained tokens |
| Math | FineMath 4+; UltraData-Math L2; then Nemotron-CC-Math 4plus | Intact questions/solutions, checkable correctness, topic/difficulty coverage and shared source families |
| Natural code | Retained Stack-Edu; source-resolved UltraData-Code L2 | Multilingual practical roles, source/test/docs linkage, immutable origins, dependency cost and eligible token yield |
| Generated material | Web/math L3 and checked code derivatives | Source grounding, independent answer/test verification, teacher lineage and rejection rates; held separately from natural stock |

The [pinned UltraFineWeb variant inspection](../corpus-audit/web-variants.json) distinguishes
`data/ultrafineweb_en` (the default English split, described as FineWeb-derived) from
`data/ultrafineweb_l1_en_hq` (the publisher's newer L1-derived selected route, claiming crawl coverage
through CC-MAIN-2025-51). Inspect both against FineWeb-Edu before choosing a bank. The separate
`ultrafineweb_en_v1_4` directory remains unqualified; its name alone does not establish selection
semantics or superiority. Only metadata/card bytes were acquired. The card's default fields
`content`, `score`, `source` must not be assumed to describe the other variants. Some directory
listings are partial, and no full shard inventory, schema, corpus quality or eligible supply is claimed.

For each new source, pin release/card/serialization first, then take a deterministic sample across
length, domain/language and upstream-score bands before labeling. Use the same bands and review
rubric for comparisons; report sample denominators and sampling weights. Keep independent correctness
checks separate from extraction/format judgments. Do not treat assistant review as independent human
annotation. Preserve hard but useful examples, rare domains, non-English coverage and long documents.

Record raw tokens, deduplicated unique tokens, eligible tokens, rejected counts/reasons, source-family
coverage and CPU/storage/teacher cost separately. Use the frozen Mistral tokenizer and retain exact
serialization. Overlapping parent/filtered releases count once. The working eight-bank weights in
[plan.json](plan.json) remain hypotheses; the chat suggestion of an 80/20 staged mixture has not
become a launch recipe. No 320B supply or compute feasibility claim follows from this packet.

The deliverable is one source-comparison table with auditable accepted/rejected examples and a
costed acquisition route. Only then freeze the eligible banks and one equal-budget data comparison.
Post-training still needs reasoning/tool verification and complete long examples; its 1.5M working
target is not supplied by these pretraining checks.
