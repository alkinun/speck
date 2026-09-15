# Autonomous pre-grant preparation queue

## Current preparation assignments

The owner requested autonomous, step-by-step progress on 2026-09-14. The agent's
[v2 preparation choice](first_wave_proposal_v2.json) makes the next acquisition targets concrete:

- Retain the proposed six-source incumbent and category weights.
- Replace the math blend slot with the admitted natural UltraData-Math L2-preview challenger.
- Use the already-declared approved code blend fallback for preparation. UltraData-Code has not
  supplied the repository/file-license mapping required by the restricted-code policy in the reviewed
  release; it is not admitted through this choice.
- Preserve all 29 first-wave slots, confirmation seeds, 130B token exposures and 192 GPU-hours.

The compiled source envelope is **17.7B tokens before headroom**, rather than 18B for the conditional
two-UltraData revision. This is an agent preparation choice under delegation, not an assertion of a
separate human recipe approval. It precedes all first-wave model outputs. Changes require a recorded
successor before the affected category's outputs. Exact language, filter, membership, training and
analysis contracts remain necessary before launch.

```bash
uv run --no-sync python -m scripts.first_wave_preparation \
  research/flagship/first_wave_proposal_v2.json research/flagship/first_wave_preparation_v2.json
```

The compiler validates the additive Math approval against the original acceptance; it does not edit
the old source registry or frozen tokenizer corpus. All other sources still require their existing
approvals, category assignment and eligibility.

## Execution order and exit criteria

1. **Science background:** complete the pinned peS2o shard and record selected-tokenizer yield against
   the 400M nominal requirement. If short, pin the next complete shard and report the new combined
   exclusion result; do not substitute PubMed or relax document-license/English criteria.
2. **Reusable token stocks:** tokenize qualified Math L2 and FineWiki with document span indices.
   Verify original counts, every output hash and cross-shard document probes. Apply the same step to
   the qualified science stock. These caches allow subsequent whole-document membership selection.
3. **Remaining incumbent stock:** prepare source-identical FineMath (800M), Cosmopedia (800M),
   Stack-Edu (1.2B), and FineWeb-Edu (4.4B), with per-source complete-shard manifests and qualified
   filters. Each target is an eventual eligible capacity, not permission to count overlapping banks.
4. **Alternative source stock:** prepare Ultra-FineWeb and DCLM (4.4B each), restricted Stack v3 (300M),
   MegaMath Web-Pro (200M), and the already-approved Ultra-FineWeb-L3 Multi-Style (200M). Preserve
   matched language allocations and treatment identities in blends.
5. **Per-arm assembly:** give shared background precedence over tested-category candidates so a
   treatment cannot silently alter the control background. Preserve alternative-source membership;
   exclude within each actual arm, then select document ordinals and assemble the fixed token order.
   Verify complete source quotas plus loader lookahead before publishing an experiment manifest.
6. **Exact execution and analysis:** bind proxy shapes, D4/D6 settings, seed/order, tokenizer, training
   horizon and fixed selection/confirmation/failure rules. Qualify the actual GH200 node and use the
   measurements to validate each run's budget before launch.

## Concrete E3 integration issue

The current packed loader deliberately prohibits training-source wraparound before the requested
training horizon. E3's 2×/4× exposures therefore need an explicit repeated-view materializer or a
qualified declared repetition contract; simply giving the loader a smaller pool would fail rather
than produce the registered experiment. Bind nested pool membership, boundary alignment, +1-token
lookahead, exact exposure and resume identities before E3 execution. No implicit repetition or new
loader default is introduced by the current preparation choice.

The queue continues through local preparation while hardware-dependent work awaits grant access.
Tokenizer historical spending reconciliation remains a disclosure task. The already-determined
budget fallback is fixed. Post-training choices remain with the later-stage work.

## Executable stock handoffs

- Science's [two-shard successor](pes2o_stock_preparation_v2.json) passes at 820.10M tokens with
  verified token stock. The earlier 403.56M bank is an exact included prefix. Joint experiment-view
  eligibility follows this source-specific preparation and headroom pass.
- Math L2 and FineWiki [document-indexed token caches](TOKEN_STOCK.md) are complete and verified.
- The [FineMath preparation](FINEMATH_STOCK.md) records the completed eight-shard stock and an eleven-shard successor targeting 960M.
  Its natural-domain corpus-selection rule is explicit; small tokenizer-sample diversity caps are
  not silently treated as full-corpus limits.

## Current execution and next work

- FineMath's [completed eleven-shard stock](../findings/2026-09-15-finemath-headroom.md) has
  **1,124,167,472 tokens** and a verified document cache. The 960M target passes by 164,167,472.
  Its earlier eight-shard retained text is a verified exact prefix, not additional supply. Bind
  joint experiment-view eligibility next; do not restart the completed source/cache jobs.
- Cosmopedia's [completed five-shard stock](../findings/2026-09-15-cosmopedia-stock.md) has
  **1,489,288,743 tokens** and a verified document cache, passing 960M headroom by 529,288,743.
  Its [prompt-lineage and corpus-selection policy](COSMOPEDIA_STOCK.md) and natural diagnostics
  remain explicit. Joint eligibility follows; do not restart completed source/cache jobs.
- Code's [matched eleven-language requirements](CODE_LANGUAGES.md) now bind identical token shares
  for each source, the equal blend and shared background. Stack-Edu's three earlier
  [SWH access probes](../../results/data/stack-edu-access-review-20260914.json) pass, and its
  [complete metadata intake](stack_edu_metadata_acquisition_v1.json) is
  [complete and verified](../../results/data/stack-edu-metadata-acquisition-20260914.json):
  38,669,178 rows in eleven pinned files. The
  [real-prefix replay qualification](../../results/systems/stack-edu-stock-prefix-20260914.json)
  passed; [full code stock preparation](stack_edu_stock_preparation_v1.json) is
  [intentionally paused](../../results/systems/stack-edu-intentional-pause-20260914.json).
  The ordered fetcher is qualified and the expanded metadata/content probes are complete.
  Resolve remaining per-language supply constraints before continuing the long run. Preserve
  source-specific license/prose/vendor rules and their differences; metadata is not code-token supply.
- FineWeb-Edu's [complete fourteen-file stock view](FINEWEB_EDU_STOCK.md) is pinned.
  The original qualified final shard is preserved and verified in the new cache; its real reader/web
  policy passed a bounded prefix check. [Raw-only acquisition](../../results/data/fineweb-edu-raw-acquisition-20260914.json)
  completed with all fourteen files verified. The three-file E1S text tranche is now running
  at 1.32B selected-Mistral tokens; retain the eventual full-stock 5.28B target. Raw files are not eligible supply.
- Continue remaining alternative-source stock, followed by joint per-arm assembly
  and execution contracts. Tokenizer historical expenditure remains a disclosure task; no tokenizer
  confirmations, D5 opening, model training, or deeper post-training work is launched here.

## Revised operational priority

Follow [LOCAL_PREPARATION_SCHEDULE.md](LOCAL_PREPARATION_SCHEDULE.md): establish code-supply
feasibility first, qualify a bounded eligible-request/NVMe fetch path, and assemble a complete
E1S-sized dataset before maximal stocks. Existing recipes and source-use approvals remain in
force until an explicit successor is adopted. FineMath and Cosmopedia caches are complete; FineWeb-Edu E1S text is running and raw
acquisition is complete; Stack-Edu is intentionally paused with its evidence intact.

The [ordered code-fetch qualification](../findings/2026-09-14-ordered-code-fetch.md) and
[expanded metadata census](../findings/2026-09-14-expanded-code-metadata.md) are complete. All
26 files / 101,229,394 physical rows passed verification. The targeted v3 metadata successor below is now complete.
The [added-file content probe](../findings/2026-09-14-code-supply-probe.md) also completed.
Combined estimates still flag Java, TypeScript, JavaScript and Python against E1S headroom,
before full exclusion. Java/TypeScript have no further released shards. Code-stock production
remains paused pending a recorded supply/recipe decision; no language changes are adopted.

The [FineWeb-Edu E1S tranche](fineweb_edu_e1s_stock_preparation_v2.json) binds three complete
downloaded files and 1.32B tokens with unchanged filters. The [finite followup sequence](LOCAL_PREPARATION_FOLLOWUPS.md)
is [running in a frozen checkout](../../results/systems/local-preparation-followups-v2-launch-20260914.json):
FineMath and Cosmopedia caches are complete; it now processes the E1S FineWeb tranche and cache. Do not launch duplicates. Future results remain
unmeasured until publication; failures/shortfalls stop dependent work. Preserve the fourteen-file
5.28B full-stock requirement and reuse identical acquisitions on later expansion.

The [targeted JS/Python metadata successor](stack_edu_metadata_acquisition_v3.json) is
[complete](../../results/data/stack-edu-metadata-acquisition-v3-20260915.json): 28 complete files,
110,704,408 physical rows and 11,774,288,820 compressed bytes. It preserves the original 26-file
prefix and adds two complete shards. The updated census is in progress. The
[bound v3 supply probe](code_supply_probe_v3.json) samples only the two new files (256 eligible
rows total), preserving previous observations and avoiding a repeated fetch benchmark.
Java/TypeScript constraints and the production code-stock pause remain in force.
