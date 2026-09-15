# Targeted code metadata and supply followup

The [v3 metadata intake](../../results/data/stack-edu-metadata-acquisition-v3-20260915.json)
completed with **28 complete files / 110,704,408 physical rows / 11,774,288,820 compressed bytes**.
The original 26-file prefix is preserved. The only additions are the third JavaScript and third
Python files at the existing approved Stack-Edu revision and unchanged source/language policy.

The [updated census](../../results/data/code-metadata-feasibility-v3-20260915.json) independently
rehashed all 28 files, checked every physical row's language and each durable intake receipt,
and recomputed policy eligibility in **128.64 seconds** during concurrent local preparation.
The added JavaScript file has 54,172 metadata-eligible rows / 107,524,283 declared content bytes;
the added Python file has 101,845 / 202,439,757. Across the three files per language, the totals
are 163,403 eligible JavaScript rows / 330,831,192 bytes and 305,889 Python rows / 603,556,622 bytes.
There are no duplicate eligible blob IDs within either language's combined metadata view.
These are metadata counts, not verified code-text or token capacity; cross-language overlap and
full reference/candidate exclusion remain unqualified.

The [v3 content probe](../flagship/code_supply_probe_v3.json) selects 32 eligible rows in each
of four equal eligible-order strata per new file, seed 42: **256 samples total**. Earlier first-file
and fifteen-added-file observations remain preserved. The executable rejects old-file reuse,
changed file order/selection, mismatched metadata generation and changed language/source policy.
It retains the qualified bounded fetch, content/security/Gitleaks/Mistral checks, complete cache
reopen and checksum-verified archival copy, without repeating interruption or warm benchmarks.

The [completed probe](../../results/data/code-supply-probe-v3-20260915.json) retained **183
documents / 109,170 Mistral tokens** after content/security filtering. All 256 requests reached
durable publication and completed reopen; missing/rejected content contributes zero to the yield
estimate. Fetch took **7.94 seconds**, including cache/persistence work, not a controlled network
speedup. The working cache, attempts, indices and verified archive remain preserved.

The [combined analysis](../../results/data/code-supply-comparison-v2-20260915.json) covers the
28 metadata files exactly once across the three probes. It checks sample metadata identities
and census eligibility counts, sums file-position estimates, and combines sampling variances.
The latest archive/index hashes are checked again; prior archive verification is inherited from
the bound earlier receipts rather than rerun. Selected language estimates are:

| Language | Estimated tokens before full exclusion | Sampling SE | E1S headroom target |
| --- | ---: | ---: | ---: |
| Python | 123.387M | 9.852M | 90M |
| JavaScript | 71.800M | 5.564M | 54M |
| Java | 45.826M | 1.566M | 54M |
| TypeScript | 31.683M | 3.525M | 36M |

The new files remove the Python/JavaScript point-estimate shortfalls. Full exclusion losses are
still unknown, and these standard errors are not simultaneous confidence bounds or guaranteed
capacity. Java/TypeScript estimates are unchanged, leaving shortfalls of 8.174M and 4.317M
against headroom. These observations do not support the much larger full-wave code background.
No sample estimate is promoted into the paper's measured-stock table.

This preparation does not change source-use approval, category or language weights, first-wave
slots, quality/license filters, or model-launch authority. Full code-stock production remains
paused pending a per-language and shared-background supply decision. Java and TypeScript already
use all released shards; adding Python/JavaScript files cannot resolve their constraints.
