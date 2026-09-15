# Matched code-language preparation

The [selected preparation prior](code_language_preparation_v1.json) binds the existing first-wave
assignments and frozen Mistral tokenizer. It keeps the eleven languages in the restricted Stack v3
qualification and applies the same token proportions independently to Stack-Edu and Stack v3.
The code category remains 15% of the complete mixture. No arm, seed, model size, exposure or GPU-hour
budget changes. This is an agent preparation decision before code-model outputs, not separately
signed human recipe approval or evidence of optimal language weights.

| Language | Code-token share | Stack-Edu preparation target | Stack v3 preparation target |
| --- | ---: | ---: | ---: |
| C | 5% | 72,000,000 | 18,000,000 |
| C++ | 15% | 216,000,000 | 54,000,000 |
| Go | 5% | 72,000,000 | 18,000,000 |
| Java | 15% | 216,000,000 | 54,000,000 |
| JavaScript | 15% | 216,000,000 | 54,000,000 |
| Markdown | 3% | 43,200,000 | 10,800,000 |
| Python | 25% | 360,000,000 | 90,000,000 |
| Rust | 5% | 72,000,000 | 18,000,000 |
| SQL | 1% | 14,400,000 | 3,600,000 |
| Shell | 1% | 14,400,000 | 3,600,000 |
| TypeScript | 10% | 144,000,000 | 36,000,000 |

Targets total **1.44B Stack-Edu / 360M restricted Stack v3**, including 20% headroom over their
1.2B / 300M nominal reusable source capacities. These are requirements, not measured supply.
A surplus in one language cannot cover a deficit in another. The equal-source blend is 50/50
within every language; shared code background uses the same language proportions. The smaller
Shell/SQL/Markdown allocations remain part of the comparison rather than being dropped when scarce.

These token shares are explicitly new; the older tokenizer sampler's byte quotas are preserved as
historical inputs and are not silently repurposed. Source-specific license, attribution, English
prose, security and vendor/fork rules still need executable stock contracts. Stack v3 supplies
vendor/fork metadata that Stack-Edu's reviewed schema does not; matched language shares do not
claim those upstream views are identical. No additional source-use approval is inferred.

The [compiled requirements](code_language_requirements_v1.json) cover 29 logical slots and 22
source-language capacity targets. Ten finalist confirmations retain null recipes and quotas until
their registered selection. E3 entries describe unique pool requirements, not repeated exposures;
the repeated-view materializer remains a separate dependency. Whole-document allocation, joint
background precedence, alignment/lookahead, splits and launch receipts remain pending.

```bash
uv run --no-sync python -m scripts.code_language_preparation \
  research/flagship/code_language_preparation_v1.json \
  research/flagship/code_language_requirements_v1.json
```

## Stack-Edu metadata acquisition

The [metadata plan](stack_edu_metadata_acquisition_v1.json) pins one complete file per language:
38,669,178 metadata rows / 3,880,693,478 bytes at the approved revision. Its
[manifest](../../results/data/stack-edu-metadata-manifest-20260914.json) records immutable LFS IDs,
physical row counts and required top-level fields. First-row-group checks are bounded diagnostics;
the acquisition verifies full file hashes, schema, counts and every row's language.

The original Rust/Go/SQL metadata files are preserved and copied only after identity verification.
The other eight files were acquired in full. Each completed file receives a durable receipt;
resume verifies those bytes and requires the original execution revision/plan. This stage does not
fetch code blobs, qualify individual licenses, or establish usable token capacity. The verified
metadata supplies concrete inputs to the next resumable blob builder.

```bash
uv run --no-sync python -m scripts.prepare_stack_edu_metadata \
  research/flagship/stack_edu_metadata_acquisition_v1.json \
  results/data/stack-edu-metadata-acquisition-20260914.json
```

## Stack-Edu code stock policy

The [stock plan](stack_edu_stock_preparation_v1.json) binds the completed metadata result.
Its builder retains SHA-1-verified, bounded gzip SWH payloads and checksummed request
receipts. Verified 404 responses are explicit missing inputs; exhausted transient retries stop the
unit without advancing its durable row cursor. Successful fetches remain reusable. One collector
owns the cache; each batch dispatches each blob ID once. Metadata order, whole-document token
counts, and checkpoints determine the candidate prefix independently of network completion order.
Interrupted output tails and scanner attempts remain preserved.

Each language's candidate prefix targets twice its nominal requirement before Gitleaks and full
reference/candidate exclusion. This fixed acquisition allowance is not measured eligible capacity.
Final frozen-Mistral counts must satisfy every language's 20% headroom target as well as the total.
Exhausting a complete metadata file or losing too much supply requires an explicit successor;
no target, language weight, or filter changes automatically.

The original score >=4, permissive detected-license allowlist, ASCII/UTF-8, size, secret and English
prose rules remain. Insufficient prose retains the original syntax exemption. Declared blob length
must equal verified content length; text must preserve the decoded UTF-8 bytes. Common document
security, benchmark and Gitleaks checks also apply. Paths containing `node_modules`, `third-party`,
`third_party`, or `vendor` as components are excluded. This is an explicit preparation path rule;
it does not establish complete vendor/fork removal where upstream metadata is absent. Restricted
Stack v3 preparation must also apply this common path rule alongside its released vendor/fork flags.

The old 500KB per-language/repository tokenizer-sampler cap is not imposed on full stock. Natural
postfilter repository proportions are retained and their byte concentration is reported. The
choice is fixed before model outputs and does not assert an optimal code distribution. Released
repository/path/license attribution and blob identities remain attached to each accepted document;
missing commits and upstream ancestry remain disclosed. Source-use approval, these agent-selected
preparation rules, and final training-launch authority remain distinct.

## Current supply evidence

The [28-file census and targeted content probe](../findings/2026-09-15-targeted-code-supply.md)
are complete. Combined estimates now exceed E1S headroom in Python (123.387M against 90M)
and JavaScript (71.800M against 54M), before full exclusion. Java remains at 45.826M against
54M, and TypeScript at 31.683M against 36M; both already cover all released shards. The larger
Stack-Edu incumbent-background requirement remains a separate unresolved constraint. Original
stock preparation remains paused; no language/background revision has been adopted. Samples
and metadata bytes are not measured source capacity.

Restricted Stack v3's twelve cached files also do not establish the required matched supply.
The [bounded metadata-range qualification](stack_v3_metadata_range_qualification_v1.json)
will compare one pinned remote metadata projection against its complete verified local file.
Its purpose is to inspect eligibility before committing to larger text transfers. Partial ranges
are not complete source files; source stock still requires full-file hashes and all content,
security and exclusion checks. No quality/license rule, source approval or language share changes.
