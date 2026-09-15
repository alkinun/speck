# Cosmopedia stock and token cache complete

The [five-file result](../../results/data/cosmopedia-stock-preparation-20260914.json) completed
on 2026-09-15 at 11:44 UTC; the original launch-date filename is preserved. It retains
**1,851,034 documents / 6,927,573,615 UTF-8 text bytes / 1,489,288,743 Mistral tokens** including
BOS/EOS. The 800M nominal requirement and **960M preparation target** both pass; headroom above
the latter is **529,288,743 tokens (55.13%)**. Filters and targets were unchanged.

The [completion review](../../results/systems/cosmopedia-completion-verification-20260915.json)
checks all five complete raw files and acquisition units, configuration/source-use identities,
published text/index/removal hashes, and recomputes exclusion analysis. Acquisition retained
1,866,001 candidates from 1,881,445 physical rows / 5,879,393,188 compressed bytes. Exclusion
removed 762 exact and 96 verified-near reference matches, plus 12,923 exact and 1,186 verified-near
candidate duplicates. Reference outputs remain intact, exact/near controls pass, and final exact
candidate/reference overlap is zero.

The [cache result](../../results/data/cosmopedia-token-stock-20260914.json) reproduces the text
document and token counts. Its runtime plan binds the actual completed result and frozen tokenizer.
The cache at `/mnt/speck-data/speck/document-token-stock-v1/cosmopedia_v2` passes complete shard
and contiguous document-index reopen. Review also re-encodes beginning/middle/end documents and
documents crossing physical token-shard boundaries. Verification cost is separate from construction.

Natural postfilter proportions remain under the declared corpus policy. The stock contains
1,817,844 distinct full-prompt hashes and 33,190 documents beyond the first occurrence of an
identical prompt. Template-prefix byte HHI is approximately 0.000012296. Prompt identity is not
original seed ancestry; generator and seed revisions remain undisclosed, and seed-source labels
are not domains. These diagnostics and filtering do not establish factual correctness or model quality.

Acquisition measured 12,265.88 seconds; the exclusion invocation measured **59,059.41 seconds
(16.41 hours)** during concurrent local preparation. SQLite commit accounts for **48,855.77
seconds (82.72%)**, already included in exclusion time. The observed WAL peak is **603,308,112
bytes (575.36 MiB)**, above the older 512 MiB observation but within the declared 2 GiB gate.
Token-cache construction measured **264.42 seconds**; surrounding verification adds cost. These
are invocation measurements, not isolated throughput, a controlled speedup, or GH200 forecasts.

This source-specific stock replaces older overlapping Cosmopedia capacity in new paper assets;
banks are not summed. Joint background/treatment exclusion, membership, order, exact quotas,
evaluation and launch contracts remain open. The finite followup sequence has advanced to the
FineWeb-Edu E1S text tranche. No model training or new source-use approval occurred.
