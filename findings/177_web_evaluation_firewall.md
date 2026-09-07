# 177 — The bounded web slice passes the flagship benchmark firewall

## Frozen before inspection

The executable firewall freezes 20 immutable payloads containing 63,652 tasks. It covers HellaSwag,
ARC Easy and Challenge, PIQA, WinoGrande XL, LAMBADA OpenAI English, MMLU, CommonsenseQA, TriviaQA,
GSM8K, all seven MATH test subjects, HumanEval, MBPP, and BigCodeBench v0.1.4. Every payload has an
exact revision, split, path, SHA-256, extraction fields, and expected task count.

The general-text matcher normalizes Unicode with NFKC and lowercase tokenization. Critical overlap is
either two task-unique 13-gram matches or a complete eligible normalized benchmark field. Exact-field
matching is indexed independently, so a short complete field cannot be missed merely because it was
not first nominated by a 13-gram. Task-unique 10-gram matches are sensitivity disclosures, not
automatic removals. The contract and implementation were committed at `9c67d05` before the web scan.

## Result

Across 31,092 bounded records, the scanner removes 92 records and 1,196,228 UTF-8 bytes. Sixty-eight
removed records contain a complete benchmark field, 25 meet the primary n-gram rule, and one meets
both. These records link to 82 unique tasks; record-task links are concentrated in TriviaQA, MMLU,
ARC Easy, and HellaSwag and are reported per source and benchmark. Fifty-five sensitivity-only
records linked to 27 unique tasks remain retained and disclosed.

The successors retain 8,497 FineWeb-Edu, 11,851 Ultra-FineWeb, 6,415 DCLM, and 4,237 FineWeb-base
records. Each still exceeds its frozen 30/3, 35/3.5, 25/2.5, and 10/1 MB train/evaluation quotas. A
second complete scan of all four successors finds zero critical records. All recorded output hashes
were recomputed. Because these outputs are strict record-removal subsets of the inputs, they also
preserve the earlier bounded zero exact and zero verified near-duplicate cross-source result.

## Boundary

Only the four bounded selections and the 20 pinned payload revisions are covered. Future benchmark
revisions and production-scale corpora require another scan. Training authority remains blocked on
human acceptance of dataset and Common Crawl rights, production global exact/near deduplication, and
acquisition cleanup/resume qualification. Use only the `web-contamination-v1` successors after those
gates close; never use the pre-decontamination inputs.

Artifacts:

- [Checked contamination summary](../results/data/web-contamination-20260907.json)
- [Frozen firewall plan](../research/flagship/web_contamination_v1.json)
- Runtime result: `/mnt/speck-data/speck/source-qualification/web-contamination-v1/report.json`
