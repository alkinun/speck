# Revised local preparation priorities — 2026-09-14

The owner requested a more practical plan after the serial Stack-Edu path projected days of
work. This schedule selects operational priorities under existing delegation. Frozen scientific
plans, source-use approvals, tokenizer, language weights and headroom targets remain unchanged.
Implementation/data-view changes below require explicit bound successors before execution.

## 1. Finish useful running work; pause the unqualified long code path

FineWeb-Edu raw downloads are now complete: all fourteen file receipts passed and agree with
the [published result](../../results/data/fineweb-edu-raw-acquisition-20260914.json). FineMath
exclusion and Cosmopedia preparation continue. Verify each completed text result and build its
token cache only from hash-bound qualified text. Raw downloads
must not become eligible-capacity claims. Avoid starting another large exclusion pass on the
shared HDD while current passes are active.

Stack-Edu was intentionally interrupted with SIGINT. The service is inactive; metadata, cached
blobs, fetch attempts, the durable cursor, interrupted tail and frozen checkout are preserved.
Its launch is not a completed stock. The [pause receipt](../../results/systems/stack-edu-intentional-pause-20260914.json)
binds before/after snapshots and the log. Do not restart it blindly into the same long schedule.

## 2. Establish source-language feasibility before more code fetching

The [complete metadata census](../../results/data/code-metadata-feasibility-20260914.json)
scanned all 38,669,178 currently pinned rows in 66.25 seconds. Under unchanged filters, the
first Java shard contains only 12,206 eligible rows / 26.16MB declared content; C++ contains
85.44MB and Python 199.26MB. Their full-stock targets are 216M, 216M and 360M tokens respectively.
These byte counts are not token capacity; security/exclusion can only reduce usable content.
The original one-file-per-language choice did not establish a feasible supply envelope.

Expanded metadata verification and a deterministic added-file content/token probe are now
complete; see the [combined finding](../findings/2026-09-14-code-supply-probe.md). Java and
TypeScript remain below E1S headroom in the estimates despite complete released metadata.
JavaScript and Python have additional released files. Full exclusion loss remains unmeasured. Confirm capacity for every language before committing to a large transfer.
Do not infer all-language token/byte ratios from the observed C prefix.

If the approved source cannot support the full matched-language background, record a pre-results
recipe/source-capacity successor. Options to assess include a different already-approved shared
code background while retaining Stack-Edu as a matched specialist treatment, or a justified
common-language allocation revision. These are proposals, not adopted recipe changes. No lowering
quality/license filters, silent source substitution, or counting surplus in another language.

## 3. Qualify a more efficient fetch path on real eligible inputs

The [operating observation](../../results/systems/code-acquisition-operating-observation-20260914.json)
found a median four eligible fetches per physical-row batch despite 32 configured workers.
Each batch waits before the next is submitted. Fetch timing includes verification/persistence,
and all small durable files share the rotational data disk with exclusion. No controlled
network/storage speedup has been measured.

Build a deterministic eligible-row index with original row IDs and rejection accounting; feed
bounded concurrent requests across metadata batches. Preserve physical source order when
publishing accepted records regardless of completion order. Reuse verified existing blobs and
retain every missing/failed attempt. Preserve quota stopping, lookahead of prefetched records,
checkpoint replay and all source-policy identities.

Use a bounded NVMe working area for small cache/checkpoint operations; approximately 110GB was
free at inspection. Declare a space ceiling and free-space floor before launch. Keep durable
checksummed payloads and explicit copy/publication receipts to the archival data drive; no
memory-only durability or unrecorded physical migration. The exact cache/backend and successor
plan remain implementation work, not an existing migration.

Run one bounded real-workload qualification, targeting 30–60 minutes of investigation rather
than broad infrastructure benchmarking. Compare fetch throughput, verified bytes/tokens per
hour, errors/retries, disk cost, bounded memory and exact clean/resumed output parity. Forecast
the next concrete capacity tranche from those observations before launching it. Worker count
alone is not evidence of speedup; the old week-long linear extrapolation is not a deadline.

## 4. Materialize one complete first-wave dataset before maximal source stocks

Prioritize the nominal 150M-model / 2B-token E1S path for the first end-to-end dataset milestone,
subject to code-supply and evaluation contracts. Its 55/15/10/10/5/5 category allocation requires:

| Category | Nominal per-run tokens | Preparation with 20% headroom |
| --- | ---: | ---: |
| Web | 1.1B | 1.32B |
| Code | 300M | 360M |
| Math | 200M | 240M |
| Synthetic | 200M | 240M |
| Science | 100M | 120M |
| Reference | 100M | 120M |

These are category envelopes, not a final recipe or proof that alternatives are ready. Only the
tested category changes; source and language quotas must be derived for each actual arm. The
existing science, reference and math stocks already exceed these individual category envelopes,
but joint eligibility remains unqualified. Code and synthetic supply still require verification.
A deterministic subset/successor can use completed compatible acquisitions; do not declare a
partially finished original run complete or sum overlapping stocks.

Freeze shared-background precedence, heldout exclusions, whole-document memberships, tokenizer,
order/seed and exact quotas before materializing the first complete dataset. Qualify the real
loader's alignment/lookahead and interruption/resume on that dataset. Additional arm assembly
must preserve matched alternatives and blend proportions. This milestone does not authorize
model training, open sealed evaluations or bypass GH200 qualification.

The [FineWeb-Edu E1S tranche](fineweb_edu_e1s_stock_preparation_v2.json) now binds three
complete already-downloaded files and the 1.32B target, with identical per-document policy and
unit configurations. Its actual yield is pending. Schedule this tranche after the current large
exclusion passes; retain the original fourteen-file / 5.28B plan for later expansion.

Then grow to the 350M / 8B E1W stocks and the complete existing first-wave capacity envelope.
The program's 29 logical slots, 192-GPU-hour data budget and fixed 1.2B flagship are unchanged.
Defer deeper post-training. The next estimate should name a concrete qualified input milestone;
there is not yet an evidence-backed all-datasets completion date.

## Bounded qualification completed

The [ordered-fetch result](../../results/systems/ordered-code-fetch-qualification-20260914.json)
now provides real evidence for the index/queue and NVMe working path: 1,408 samples fetched in
68.18s including deliberate interruption/resume, identical warm replay, completed reopen and
verified archival copy. Content/security checks retained 788,811 tokens. These are sample
observations, not full exclusion or source capacity. See the [finding](../findings/2026-09-14-ordered-code-fetch.md)
for stratified estimates and limits. The [expanded metadata and content probe](../findings/2026-09-14-code-supply-probe.md)
is now complete too. Resolve the remaining per-language/background supply constraint before
production integration or a recorded recipe successor.

The [finite followup sequence](LOCAL_PREPARATION_FOLLOWUPS.md) is now
[running in a frozen checkout](../../results/systems/local-preparation-followups-v2-launch-20260914.json).
It waits for checked FineMath/Cosmopedia completion, builds and verifies their caches, then
processes the E1S FineWeb tranche and cache. Do not launch duplicates. Future results remain
unmeasured until the sequence publishes them; failures/shortfalls stop dependent work.
