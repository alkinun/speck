# Checked reference-token supply is below the conditional E3 requirements

The [capacity review](../../results/data/screen-capacity-20260913.json), produced by clean revision
`edac779`, reverifies the retained corpus, source files, and reference-tokenizer identities. Its counts
come from the archived measured pilot corpus; it does not retokenize it or combine overlapping banks.
The [review contract](../flagship/screen_capacity_review_v1.json) binds the data protocol, plan, registry,
evidence, and assumptions.

## Category capacity, not aggregate tokens, determines the usable pool

The checked bank has **2,056,192,176 Mistral-reference tokens**. Under the documented category prior,
science limits a balanced pool to **1,235,700,560 tokens**, before alignment/lookahead allowance.

| Category | Available reference tokens | Required for a 6B one-epoch prior pool | Shortfall |
| --- | ---: | ---: | ---: |
| Web | 1,263,779,846 | 3,300,000,000 | 2,036,220,154 |
| Code | 294,750,097 | 900,000,000 | 605,249,903 |
| Math | 174,544,300 | 600,000,000 | 425,455,700 |
| Synthetic | 196,670,235 | 600,000,000 | 403,329,765 |
| Science | 61,785,028 | 300,000,000 | 238,214,972 |
| Reference | 64,662,670 | 300,000,000 | 235,337,330 |

For E3's fixed 6B training horizon, the 1/2/4-epoch arms need 6B/3B/1.5B unique-token pools. None is
fully supplied by this bank at the prior weights. Even the four-epoch pool is short **13,214,972 science
tokens** and **10,337,330 reference tokens**. The other categories can cover that smallest pool.
Seed count and repeated exposures do not automatically multiply unique-token storage requirements.

These are conditional reference-token calculations. E3's exact incumbent source assignments and D5's
final tokenizer remain to be frozen. The representative Common Pile code/science sources used in the
operations bank are not automatically the scientific incumbents. Other prepared data is not counted
without checked eligibility, final-tokenizer measurements, and union/dedup evidence.

## E1 still needs exact treatment recipes

Seven registry primary-screen sources have no capacity measurement in this reviewed corpus:
Ultra-FineWeb L1-HQ, FineWeb-Edu, DCLM, restricted Stack v3, Stack-Edu, MegaMath Web-Pro, and
Ultra-FineWeb-L3 Multi-Style. Their entries are unknown, not claims of zero upstream availability.

The current protocol fixes the experiment funnel but lacks executable per-arm source/filter/blend
manifests. In particular, freeze the four web treatments, specialist blends, whether treatments
replace their category at the prior weight, and the exact E3 incumbent mixture before preparing launch
corpora. Under prior substitution, an 8B web-screen run needs 4.4B web tokens; a 2B specialist run needs
300M code or 200M math/synthetic tokens. Pure-category runs would require the full run horizon from that
treatment. These scenarios are reported without selecting a design.

The next preparation action is therefore to freeze those recipes and the tokenizer, then materialize
the per-source deficits with packing headroom. The bounded integration proves mechanics; it does not
close this supply gap or authorize E1/E3 training.
