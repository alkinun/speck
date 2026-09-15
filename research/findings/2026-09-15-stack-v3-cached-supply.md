# Restricted Stack v3 cached content yield

The [bound probe](../flagship/stack_v3_cached_supply_probe_v1.json) completed without new downloads.
It rehashed all twelve complete cached source files and reproduced their prior metadata census.
From language-specific eligible indices in declared file/repository/file order, four equal
eligible-order strata supplied 32 seed-42 samples each: **1,408 files total**. Positions were
fixed before content outcomes. The [result](../../results/data/stack-v3-cached-supply-probe-20260915.json)
retained **1,088 documents / 1,170,568 Mistral tokens**, including BOS/EOS, before full exclusion.

Released-content length, source-specific high-confidence secrets and English-prose policy,
shared character/security/benchmark checks and fresh Gitleaks filtering apply. Insufficient prose
retains the original syntax exemption. Released content IDs are preserved without assuming
they are plain SHA-1 of transformed text; every accepted sample has its own SHA-256. Original
samples, rejected outcomes, indices, scanner records, attribution and source positions remain
preserved. No old tokenizer-sampler repository cap or sample partition is imposed on this probe.

The [completion review](../../results/systems/stack-v3-cached-supply-verification-20260915.json)
reconstructs every selected position, re-encodes every retained document and recomputes each
estimate, sampling standard error and outcome total. All agree. The worker also verified every
payload in its 34,785,280-byte archive; review rechecks archive and inventory hashes. Acquisition,
indexing, content checks and counting measured 97.49 seconds before archival work, during concurrent
local preparation. This is not a controlled storage benchmark or training throughput result.

| Language | Retained / sampled | Estimated tokens in these twelve files | Sampling SE | Current E1S headroom target |
| --- | ---: | ---: | ---: | ---: |
| Python | 104 / 128 | 5.750M | 1.142M | 90M |
| C++ | 112 / 128 | 6.202M | 1.124M | 54M |
| C | 114 / 128 | 1.356M | 0.243M | 18M |
| Java | 104 / 128 | 8.477M | 1.264M | 54M |
| JavaScript | 103 / 128 | 2.083M | 0.309M | 54M |
| TypeScript | 104 / 128 | 0.955M | 0.132M | 36M |
| Rust | 113 / 128 | 1.316M | 0.299M | 18M |
| Go | 99 / 128 | 2.320M | 0.340M | 18M |
| Shell | 105 / 128 | 0.424M | 0.046M | 3.6M |
| SQL | 91 / 128 | 0.313M | 0.103M | 3.6M |
| Markdown | 39 / 128 | 3.927M | 1.880M | 10.8M |

Rejected samples contribute zero to stratified yield estimates. These estimates and SE describe
only the twelve-file view before full reference/candidate exclusion. They are not measured
whole-file token capacity, simultaneous confidence bounds or predictions for the newly inspected
sixteen files. Every language remains short in point estimates. Markdown's 77 non-English-prose
rejections out of 128 samples illustrate why metadata bytes cannot stand in for filtered tokens.

The next complete-file intake should be explicitly limited and justified by the preserved
metadata view, followed by actual content qualification. JavaScript and TypeScript remain major
restricted-source supply gaps, while Stack-Edu's corresponding complete-release constraints and
the larger shared-background problem remain separate. No language/background recipe revision,
source approval, model launch or promotion into the paper's measured-stock table occurs here.
