# 171 — Stack v3.1 code benchmark overlaps are removed before tokenizer use

## Frozen evaluation payloads

The scan pins 2,278 tasks from the official HumanEval, MBPP, and BigCodeBench v0.1.4 payloads before
reading overlap results. Text is normalized with NFKC and lowercased lexical code tokens. The primary
rule requires at least three informative 13-grams unique to one benchmark task; normalized exact
fields of at least 100 characters also qualify. Ten-gram results are a separately reported
sensitivity view.

## Result

The refined Stack v3.1 sample contains 17,172 files. The primary rule removes 125 files and 1,790,425
bytes linked to 72 unique tasks. There are no complete normalized field matches. The task links are
dominated by BigCodeBench, with smaller MBPP and HumanEval counts. This is conservative overlap
detection, not evidence that benchmark material was intentionally copied: common library and testing
idioms can match task-unique n-grams. Removal is nevertheless cheaper than weakening evaluation
integrity.

Sixty-six additional files meet only the 10-gram sensitivity rule and remain disclosed in the runtime
report. After primary removal, 17,047 files remain. The exact tokenizer partition retains 59,883,015
training bytes and 5,935,510 evaluation bytes, above the declared 55M and 5.5M requirements.

## Decision

The v3 decontaminated artifact becomes the Stack v3 tokenizer input parent. This closes the bounded
HumanEval/MBPP/BigCodeBench gate only; it does not establish that the 15.9 TB corpus or future
benchmark versions are contamination-free. Manual rights acceptance, cross-source near-duplicates,
and acquisition cleanup/resume still block training authority.

Artifacts:

- [Checked summary](../results/data/stack-v3.1-code-contamination-20260907.json)
- [Frozen scan plan](../research/flagship/code_contamination_v1.json)
- Runtime result: `/mnt/speck-data/speck/source-qualification/stack-v3.1-v3/report.json`
