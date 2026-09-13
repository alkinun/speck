# 136 — Retained-checkpoint replay closes the result-provenance gap

## Collection was strong; independent acceptance was incomplete

The frozen collector already revalidates the materialized config manifest, exact seed/data window and
geometry, complete checkpoint metadata, summary/history equality, timing evidence, and model,
optimizer, and metadata hashes. The v1 post-commit verifier checked the resulting file hash, run labels,
source completeness, and transition graph, but did not independently replay those collector checks.

Adversarial fixtures demonstrate the boundary: after updating the result reference and transition hash,
v1 accepts a wrong plan hash, wrong experiment path, or wrong claimed model hash. It also accepts model
bytes changed after collection because the result file itself is unchanged.

## Append-only v2 acceptance

The v2 command first runs every v1 ledger check, then derives each experiment and checkpoint directory
from the pinned qualification, reruns the frozen collector, and requires exact equality for every stable
result field. Only the fresh recollection timestamp is excluded; the original timestamp must still be
timezone-aware. Exact replay passes, and all four v1 counterexamples fail.

V1, the active runner, collector, analyzer, program, and experiment remain unchanged. Retained checkpoint
bytes are now mandatory until v2 acceptance passes after each automatic commit.

## Artifact

- [Checkpoint-replay acceptance qualification](../results/Speck-Paper1/finalist-result-acceptance-qualified-v2.json)

Run it with:

```bash
python -m scripts.paper_finalist_result_acceptance_validate_v2
```
