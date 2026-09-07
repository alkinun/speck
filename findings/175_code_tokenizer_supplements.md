# 175 — The five-source tokenizer code slice passes technical qualification

## Python-Edu

The first pinned 322 MB metadata shard contains 3,839,224 score records but no file-level license
field. A bounded SWH fetch examines 12,201 rows, obtains 10,496 blobs with zero missing, failed, or
SHA-1-mismatched contents, and accepts 8,948 files from 5,666 repositories after size, English-prose,
repository-cap, decode, and high-confidence-secret filters. Metadata length differs for 969 valid
blobs and is reported rather than treated as content identity.

Gitleaks removes five files with eight generic-key findings. The frozen 2,278-task benchmark screen
then removes 247 files / 657,371 bytes linked to 149 tasks and discloses 229 sensitivity-only files.
No complete normalized benchmark field matches. The final source retains 14.06 MB train and 1.29 MB
evaluation against its 10M/1M quota. Technical gates pass, but absent file-level license metadata is
an explicit human rights blocker.

## Python enhancement proposals

The pinned Common Pile source expands from a 3.72 MB gzip to 11.32 MB of text. Of 655 public-domain
metadata rows, 645 pass the English-prose gate. Gitleaks finds nothing. Benchmark decontamination
removes nine documents / 133,545 bytes linked to eight tasks and discloses three sensitivity-only
documents, leaving 9.96 MB train and 0.97 MB evaluation against 5M/0.5M. The source-level public-domain
selection still requires human confirmation before use.

## Five-source blend

The final bounded overlap pass covers 40,869 decontaminated documents, with 39,725 MinHash
signatures. It finds zero exact released-text matches and zero verified 0.80-Jaccard near duplicates.
Precedence is PEP prose, Python-Edu, Stack-Edu, Stack v3.1, then Common Pile educational code. Every
55/20/10/10/5 train/evaluation quota survives.

This completes the bounded technical code slice, not training authority. Production-scale global
deduplication, acquisition cleanup/resume, future evaluation payloads, and manual rights/attribution
acceptance remain open.

Artifacts:

- [Checked summary](../results/data/code-tokenizer-supplements-20260907.json)
- [Five-source overlap plan](../research/flagship/code_cross_source_duplicates_v2.json)
- [Python-Edu sample plan](../research/flagship/python_edu_sample_v1.json)
- [PEP sample plan](../research/flagship/common_pile_peps_sample_v1.json)
- Runtime overlap result: `/mnt/speck-data/speck/source-qualification/code-cross-source-dedup-v2/report.json`
