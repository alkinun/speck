# 54 — HELMET seeded-demonstration repair qualification

## Question

Can NarrativeQA and Multi-LexSum's unseeded two-shot demonstration selection be repaired with a
minimal, reviewable change whose behavior is controlled by the already declared evaluation seed?

## Patch boundary

The pinned upstream `data.py` has SHA-256 `559977d8…`. The patch changes exactly two calls:

- `all_data["train"].shuffle()` becomes `shuffle(seed=seed)` in NarrativeQA; and
- `train_data.shuffle()` becomes `shuffle(seed=seed)` in Multi-LexSum.

No dataset row, test ordering, shot count, template, answer, length filter, truncation, or metric is
changed. The patch has SHA-256 `ddb855f0…`; the patched source has SHA-256 `eb085cae…`.

## Behavioral qualification

Both loaders ran on frozen eight-row training and three-row evaluation fixtures with two shots. Two
independent processes used different `PYTHONHASHSEED` values while Hub and Datasets networking were
disabled.

At loader seed 42, each loader produced byte-identical demonstration and complete-prompt hashes across
both processes. Re-running at seed 43 changed both identities. Static analysis simultaneously moved
the targeted unseeded-shuffle count from one to zero in each loader. Thus the existing loader seed now
controls demonstration identity rather than leaving it to entropy or cache state.

## Decision

The two-line repair qualifies on synthetic fixtures. It is the required implementation if either
rights-blocked dataset is later retained.

The active evaluation manifest is not changed to use those datasets, real-data prompts are not
qualified, and candidate execution remains unauthorized. A successor manifest must pin this patch
before data access or candidate outputs. NarrativeQA and Multi-LexSum rights, payload, tokenizer,
judge, and contamination blockers are unchanged.

## Artifacts

- [Frozen repair protocol](../research/architecture-promotion-v1/helmet_seeded_demos_v1.json)
- [Exact two-line patch](../research/architecture-promotion-v1/patches/helmet_seeded_demos.patch)
- [Fixture qualification](../results/Speck-Architecture-Promotion-v1/helmet-seeded-demos-qualified.json)
- [Qualification runner](../scripts/helmet_seeded_demos_qualify.py)
