# 51 — HELMET Multi-LexSum rights and prompt-determinism decision

## Question

Can HELMET's five Multi-LexSum summarization cells be acquired and frozen under the intended
organizational scope, and are their prompts reproducible as written?

## Component rights

The exact `v20230518` conversion is pinned at revision `80d2f662…`: four Parquet files totaling
836,396,018 bytes. No payload file was downloaded.

The [dataset authors](https://github.com/multilexsum/dataset) specify different rights for different
components:

- the database layer is ODC-By and requires attribution;
- summaries and metadata are CC BY-NC 4.0;
- source court documents are stated to be public domain; and
- loader code is Apache-2.0.

ODC-By explicitly separates database rights from rights in individual contents. CC BY-NC defines
NonCommercial around whether use is primarily intended for commercial advantage or compensation.
The authors separately direct commercial users to `info@clearinghouse.net` for a license that limits
summary reposting.

## How HELMET uses the restricted component

The public-domain documents cannot be isolated as the only consumed material. The pinned HELMET
configs use two shots. `load_multi_lexsum` places two training `summary/short` values into every prompt
as demonstrations and uses validation `summary/short` as the reference answer. The primary
`gpt-4-f1` metric also sends references and model outputs through the separately unqualified judge.

Accordingly, the CC BY-NC summaries are directly reproduced and processed during evaluation; they are
not merely metadata adjacent to public-domain inputs. The project's intended organizational scope has
not been established as strictly noncommercial, and no separate commercial license exists in the
evidence ledger.

## Prompt reproducibility failure

The same data path contains one `train_data.shuffle()` call without a seed. Because two training
summaries become demonstrations, the exact prompt can vary with runtime/cache state even though the
outer evaluation seed is 42. Dataset hashes alone therefore cannot freeze the benchmark prompt.

A repair must bind demonstration selection to an explicit seed and prove case/prompt replay before
candidate outputs exist. This is a scientific manifest change, not merely an infrastructure fix.

## Decision

Metadata and the exact 836MB payload plan qualify. Payload acquisition, evaluation use, commercial
scope, prompt determinism, the Llama 2 truncation tokenizer, and the proprietary judge do not. No
dataset payload was acquired and no license contact was attempted.

Required next steps are an authorized strictly noncommercial-scope decision or separate commercial
license, followed by a versioned seeded-demonstration repair, immutable payload qualification,
tokenizer/judge qualification, exact prompt replay, and contamination scanning. Removing Multi-LexSum
from a successor manifest remains an alternative only before candidate results.

This operational decision is not legal advice or a final ownership determination.

## Artifacts

- [Frozen decision protocol](../research/architecture-promotion-v1/helmet_multilexsum_decision_v1.json)
- [Blocked decision](../results/Speck-Architecture-Promotion-v1/helmet-multilexsum-decision.json)
- [Decision runner](../scripts/helmet_multilexsum_decision.py)
- [HELMET runtime inventory](../results/Speck-Architecture-Promotion-v1/helmet-runtime-dependency-audit.json)
