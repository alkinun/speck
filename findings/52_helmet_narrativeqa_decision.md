# 52 — HELMET NarrativeQA embedded-work and prompt-path decision

## Question

Does Apache-2.0 on NarrativeQA's metadata repository authorize HELMET's embedded full-story payload,
and are the five NarrativeQA prompts reproducible and locally scoreable?

## Repository versus embedded payload

The [original NarrativeQA repository](https://github.com/google-deepmind/narrativeqa) contains
metadata, Wikipedia summaries, questions/answers, and links to full stories. Its download script fetches
those stories from external sites. The Hugging Face conversion at `2e643e7…` instead embeds full story
text in 35 Parquet files totaling 3,232,805,701 bytes. No Parquet payload was acquired.

The exact original `documents.csv` contains 1,572 works: 783 Gutenberg books and 789 movie scripts.
Of the scripts, 767 links target IMSDb, 15 DailyScript, and 7 AwesomeFilm. The repository's Apache
license can govern its own code/metadata contributions; it does not by itself demonstrate rights to
every externally linked story later embedded by the conversion.

## Work-level rights boundary

[IMSDb's disclaimer](https://imsdb.com/disclaimer.html) and
[DailyScript](https://www.dailyscript.com/) describe scripts as educational-purpose material rather
than granting commercial evaluation rights. Many individual script files carry the same marking.

[Project Gutenberg's license guidance](https://www.gutenberg.org/policy/license) says most works are
unrestricted under U.S. copyright law, but requires users outside the United States to check their own
country's law and distinguishes separately permitted copyrighted books. Speck's current operating
jurisdiction is not the United States, and no 783-work jurisdiction-specific review exists.

Thus neither the dataset-level Apache label nor one source site's general status qualifies the embedded
full-text collection as a whole.

## HELMET prompt path

HELMET defines five NarrativeQA cells from 8K through 128K with two demonstrations. Test ordering is
seeded, but the two training question-answer demonstrations come from
`all_data["train"].shuffle().select(...)` without a seed. Exact prompts can therefore vary with runtime
or cache state.

The same loader requires the gated Llama 2 tokenizer to select documents longer than 131K tokens and
truncate each context. The primary NarrativeQA metric is `gpt-4-score`, using the separately
unqualified proprietary judge. Rights, prompt selection, truncation identity, and scoring are four
independent gates.

## Decision

Metadata and the exact 3.23GB pointer-tree plan qualify. Embedded-work rights, payload acquisition,
evaluation/commercial scope, prompt determinism, tokenizer identity, and model-judge reproducibility do
not. Zero NarrativeQA payload files were downloaded and no external contact was attempted.

Required next work is either complete work-level and jurisdiction-specific clearance or freeze a
pre-results replacement; then seed/replay demonstrations, qualify the truncation tokenizer and judge
or replacement metric, pin every payload, materialize exact prompts, and run contamination checks.

NarrativeQA cannot currently support the RULER-v2 source-document guardrail. This operational decision
is not legal advice or a final ownership determination.

## Artifacts

- [Frozen decision protocol](../research/architecture-promotion-v1/helmet_narrativeqa_decision_v1.json)
- [Blocked decision](../results/Speck-Architecture-Promotion-v1/helmet-narrativeqa-decision.json)
- [Decision runner](../scripts/helmet_narrativeqa_decision.py)
- [HELMET runtime inventory](../results/Speck-Architecture-Promotion-v1/helmet-runtime-dependency-audit.json)
