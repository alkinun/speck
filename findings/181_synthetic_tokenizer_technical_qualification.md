# 181 — Four synthetic sources pass bounded technical qualification

The frozen 35/35/15/15 MB allocation covers Cosmopedia v2, Ultra-FineWeb-L3 Multi-Style,
Ultra-FineWeb-L3 QA, and MegaMath QA, with the MegaMath quota split equally across its Llama-3.3 and
Qwen-2.5 directory families. Sampling records generator family, explicitly undisclosed revisions,
transformation, seed-source status, prompt/seed/answer hashes when exposed, and seed URL/domain when
available. English, size, repeated-line/5-gram, model-identity phrase, template-prefix, seed-domain,
PII, and local-secret gates apply before the dedicated scan.

The initial Cosmopedia result exposed that `seed_data` contains labels such as `fineweb`, not the seed
text described by the card. That result is preserved and forbidden. The corrected successor records
the label separately and hashes/audits the prompt that embeds the web excerpt. Its maximum observed
generated-text/prompt 5-shingle Jaccard is 0.218, below the frozen 0.80 unchanged-seed rejection rule.

Gitleaks reports seven fully redacted findings across six records, all removed. Frozen four-source
precedence finds zero exact and zero verified ≥0.80 near duplicates among 56,814 security-clean
records. The 20-payload/63,652-task firewall then removes 353 records (903,431 bytes): 54 exact-field,
300 primary 13-gram, and one meeting both. The 294 retained sensitivity records remain disclosed. A
full successor rescan finds zero critical matches.

The corrected final slice contains 56,461 records, 119.19 MB train text, and 11.91 MB evaluation text;
every per-source quota survives. Technical qualification does not resolve undisclosed generator
revisions, seed rights, Common Crawl terms, or Ultra-FineWeb-L3's redistribution wording. Production
global deduplication and cleanup/resume remain open. No tokenizer sampling or training is authorized.

Artifacts:

- [Checked synthetic summary](../results/data/synthetic-tokenizer-sources-20260907.json)
- [Corrected Cosmopedia plan](../research/flagship/synthetic_cosmopedia_v2_v2.json)
- Runtime overlap: `/mnt/speck-data/speck/source-qualification/synthetic-cross-source-dedup-v2/report.json`
- Runtime contamination: `/mnt/speck-data/speck/source-qualification/synthetic-contamination-v2/report.json`
