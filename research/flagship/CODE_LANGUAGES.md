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
The other eight files must be acquired in full. Each completed file receives a durable receipt;
resume verifies those bytes and requires the original execution revision/plan. This stage does not
fetch code blobs, qualify individual licenses, or establish usable token capacity. The verified
metadata supplies concrete inputs to the next resumable blob builder.

```bash
uv run --no-sync python -m scripts.prepare_stack_edu_metadata \
  research/flagship/stack_edu_metadata_acquisition_v1.json \
  results/data/stack-edu-metadata-acquisition-20260914.json
```
