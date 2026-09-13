# UltraData intake makes the source and serialization choices concrete

The [intake result](../../results/data/ultradata-intake-20260913.json) comes from clean revision
`8ac56a4` under the [frozen seven-view plan](../flagship/ultradata_intake_v1.json). It captures 96 rows
per view from first/middle/last 32-row windows, **672 records total**, with matching pinned dataset
revision headers and retained response hashes. All requested views completed without truncated cells.
No dataset content is executed or used for model training.

## Language and schema observations

| View | Complete rows | Confident-English diagnostic | Other diagnostic outcomes |
| --- | ---: | ---: | --- |
| Code L2, Python comments/docstrings | 96 | 45 | 38 insufficient-prose; 13 other/low-confidence |
| Code L3, task prose | 96 | 96 | analysis prose also 96/96 |
| Math L2-preview | 96 | 93 | 3 other/low-confidence |
| Math L3 QA | 96 | 96 | |
| Math L3 Conversation | 96 | 96 | |
| Math L3 Multi-Style | 96 | 96 | |
| Math L3 Textbook-Exercise | 96 | 96 | |

These are sample diagnostics, not corpus-wide English yield or proof that every non-English label is
correct. In particular, code with insufficient prose is not automatically non-English, and formulas
can confuse language identification. The bounded detector uses at most 50K characters, at least 80
alphabetic characters, and an English probability threshold of 0.8.

Math L2-preview's sample labels are 4 (72 rows) and 5 (24 rows). Preserve the released labels; this
observation does not justify another cutoff or define the full classifier rubric. The upstream L2
selector README was still marked forthcoming when checked. The observed content schema has no URL
or parent identifier. Math L3 supplies `uid` and text, with lineage semantics still unresolved.

Code L2 retains repository name, relative path and UUID, but no explicit repository revision/license
field. A manually inspected sampled path includes `third_party`, so algorithmic selection does not
itself satisfy the current non-vendor/permissive-code policy. No SPDX marker was found in these 96
contents; this is not a conclusion about licenses across the corpus.

## Code L3 serialization is a real treatment choice

Every sampled row has task, analysis, solution and test fields. The released `content` contains the
task and solution verbatim in all 96 rows, but never the complete analysis field verbatim. This agrees
with the card's task/solution description. Its formats are code comment (37), Markdown (32), and no
fence (27).

`full_content` contains generated tests and varies between standard, commented, solution-first, and
test-first orders. Full-analysis exact containment passes in 76 rows; commenting/formatting prevents
equating an unsuccessful substring check with semantic absence. `raw_content` is not JSON in these
samples. The full serialization has 503,257 UTF-8 bytes versus 257,144 for the released `content`.

Use released `content` as the initial reproduction-oriented L3 treatment. If reasoning-text exposure
is the intended intervention, define task+analysis+solution serialization separately and account for
its different token cost. Do not silently select `full_content` and claim it is the same experiment;
generated tests are also not verified tests. This observation is useful to the separate thinking-model
post-training work without selecting its final response format.

## Preparation decisions

The [candidate revision](../flagship/first_wave_candidate_revision_v2.json) proposes:

- **E1 code:** retain restricted Stack v3 and Stack-Edu; replace their blend slot with UltraData-Code L2,
  conditional on source/provenance and matched-language qualification.
- **E1 math:** retain FineMath and MegaMath; replace their blend slot with UltraData-Math L2-preview.
- **E4 refinement:** consider Code L3 released content and Math L3 QA inside the existing decay budget.
  Textbook/exercise is a further math candidate, not an automatically added arm. Later-stage placement
  is better aligned with the mature-parent evidence reviewed in the dataset cards.

The first-wave count remains **29 logical slots / 192 GPU-hours**, or 230 with D4/D6. If both natural
challengers are admitted, the source-capacity envelope becomes **18B tokens** before headroom: the
parent's 17.5B plus 300M Code L2 and 200M Math L2. Source overlap, final-tokenizer counts and arm-specific
views still need measurement. The current incumbent is preserved as a proposal/control, not promoted
from vendor results.

## What remains before admission

The new sources are not in the current source-use record. Code needs a workable repository/file
license and lineage policy; Math needs a source-use decision with its provenance limitations explicit.
Both need normal full-shard acquisition, English filtering, complete exclusion, and measured supply.
L3 additionally needs parent/problem-family handling: zero sample-ID overlap across these windows
does not establish unrelated ancestry. No answer-correctness or contamination qualification is claimed
by this intake.

If a challenger cannot qualify before its source-screen outputs, the revision specifies the existing
approved blend as the operational fallback. The failure and fallback must be recorded before results,
so source availability cannot silently change the scientific comparison after it begins.
