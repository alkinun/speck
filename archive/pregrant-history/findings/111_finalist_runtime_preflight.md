# 111 — Exact finalist CUDA runtime preflight

## Qualified path

Both exact finalist arm templates pass one compiled CUDA/BF16 4×4,096 forward, backward, gradient
clip, and Muon update on the RTX 3090. Dense loss/gradient are finite with 10.24 GiB peak allocation;
candidate loss/gradient are finite with 14.14 GiB peak. Both remain below the frozen 16 GiB envelope.

Temporary Transformers exports validate and produce exact full and incremental parity for both arms.
The previously powered, disjoint, trained-topology cache-equivalence v3 result remains the behavioral
pass/fail authority. No checkpoint, result, target-lock, or final analysis path exists after preflight.

## Retained negative diagnostic

The short random-weight native full-versus-cached diagnostic again fails elementwise tolerance for both
arms. Dense has four mismatched logits with 100% token argmax agreement. Candidate has 12,765 mismatched
logits, 0.04199 relative RMS error, and 87.5% argmax agreement in this run. Under the frozen v2 decision,
this random-weight path is retained as a diagnostic without pass/fail authority because the powered
trained-topology common-history contract is authoritative. The worse candidate argmax result is still a
mandatory free-running/runtime risk and cannot be hidden by the preflight pass.

## Decision

Exact-config compiled training feasibility, export, and trained-topology behavioral cache reference
qualify. RULER v2, NoLiMa, and HELMET remain unqualified or unexecuted, so training and automatic launch
remain false. This result has no attribution, promotion, novelty, serving, or paper-scale authority.

## Artifact

- [Finalist preflight v1](../results/Speck-Paper1/finalist-preflight-v1.json)
