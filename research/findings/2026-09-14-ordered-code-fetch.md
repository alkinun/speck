# Ordered code fetching: bounded qualification and supply estimates

At frozen implementation `66d5fe7`, the
[registered bounded probe](../flagship/ordered_code_fetch_qualification_v1.json) completed.
The [result](../../results/systems/ordered-code-fetch-qualification-20260914.json) binds the
original eleven-file stock plan, deterministic eligible indices and 1,408 sampled inputs.
All eleven index eligible counts match the earlier complete metadata census. Every sample
is accounted for as retained or rejected, and final journal/output/inventory identities passed
post-publication inspection.

Indexing took **21.50 seconds**. Fetching all **1,408 samples took 68.18 seconds**, including a
deliberate interruption after 64 ordered records and the resumed invocation. A clean warm-cache
replay took 0.71 seconds and produced the identical journal hash; completed-result reopening
also passed. These timings do not establish a controlled speedup against the old run: the
sample, storage, cache state and workload differ, and content/security processing is separate.

The existing content, English-prose, benchmark/security and Gitleaks checks retained
**1,039 documents / 788,811 Mistral tokens**. These are bounded sample observations before
full reference/candidate exclusion, not qualified source stocks. No language quota pass or
training launch was declared. All new working artifacts were copied into a checksummed
355,368,960-byte archival tar on the data drive; every archived payload was reopened and
verified. Original cached inputs and the intentionally paused source-stock checkpoint remain
preserved separately. The NVMe working copy remains available.

## Feasibility implication

The stratified sample estimates for the current first metadata file are:

| Language | Estimated tokens before full exclusion | Sampling SE | E1S headroom target |
| --- | ---: | ---: | ---: |
| C | 23.37M | 2.12M | 18M |
| C++ | 18.65M | 2.13M | 54M |
| Java | 4.16M | 0.38M | 54M |
| JavaScript | 25.62M | 3.77M | 54M |
| Python | 41.61M | 7.47M | 90M |
| TypeScript | 31.68M | 3.52M | 36M |
| Rust | 36.39M | 3.64M | 18M |
| Go | 24.00M | 2.53M | 18M |
| Shell | 294.45M | 37.55M | 3.6M |
| SQL | 55.94M | 7.03M | 3.6M |
| Markdown | 50.76M | 10.46M | 10.8M |

Sampling SE accounts for within-stratum finite-population sampling only. It does not account
for full exclusion loss, differences in additional shards or uncertainty in future recipes.
The estimates must not populate the paper's measured source-capacity table. The E1S language
targets are one quarter of the existing Stack-Edu maximum-stock targets, preserving the
currently selected common language shares and 20% headroom.

TypeScript is particularly important: the pinned release has one metadata file for that
language, and the estimate is below its E1S headroom requirement before full exclusion. An
uncertain estimate is not proof of failure, but additional same-language shards cannot be
assumed available. Full acquisition should remain paused pending a capacity/recipe decision.
Additional Java/C++/JavaScript/Python metadata is being acquired under the explicit
[v2 metadata successor](../flagship/stack_edu_metadata_acquisition_v2.json). Those files still
need complete verification and a census; do not multiply one file's estimated yield by the
number of files as though it were observed capacity.

The ordered index/fetch components now have bounded real evidence. Production stock integration,
full exclusion, any recipe revision, joint source membership and launch manifests remain open.
The next decision should use complete-source feasibility and preserve matched comparisons;
no language reweighting, relaxed filters or background substitution has been adopted.

Validation: full quality run **1,148 passed, 10 skipped, 125 deselected**; focused ordered-fetch
tests rerun after the final cache-reservation/identity guards; formatting, lint, catalog, archive
and source-pin checks passed. Fixtures cover parallel/order/duplicate behavior, interruption and
warm replay, corrupted checkpoints, space gates, fallback identity reuse, transient failures,
metadata membership parity, deterministic strata and source-plan successor boundaries.
