# 157 — Official K3 sources close the Stable LatentMoE specification gap only

## Source result

The official Moonshot repository and Hugging Face release are pinned by immutable revisions and file
hashes. The report now supplies the exact normalized LatentMoE, SiTU-GLU, exact Quantile Balancing,
and histogram Quantile Balancing definitions needed for clean-room reference work. A fresh network
validation reproduced seven official file hashes and checked the released configuration and forward
semantics.

The code boundary matters. The released inference path implements SiTU, sigmoid routing, selected raw-
score renormalization, latent down/expert/norm/up flow, and the shared path, but explicitly rejects
training. It contains neither the Quantile Balancing update nor the histogram estimator. Official
inference code is therefore not a training, distributed-dispatch, backward, or checkpoint reference.

## Readiness consequence

V2 preserves v1's exact six-stage order: conventional MoE, latent projection, latent normalization,
bounded activation, balancing, then expert geometry. It authorizes four isolated CPU references only:
normalized LatentMoE, SiTU-GLU, exact Quantile Balancing, and a 1,000-bin global-step histogram
emulator. Each must be independently qualified on tiny disposable fixtures before composition.

The released 7,168/3,584 widths, 896 experts, top-16 routing, and two shared experts are source-fidelity
facts, not selected Speck geometry. The unspecified optional EMA is excluded from the initial reference.

Sequence/depth parents, a dropless conventional MoE, training/distributed semantics, resource/storage
budgets, routing intervention rules, and the single-device/expert-parallel hardware envelope remain
blocked. The active finalist program still pins v1; architecture/model code, registration, training,
and promotion are unchanged.

## Artifacts

- [Official K3 source note](../papers/43_kimi_k3_official_release.md)
- [Primary-source audit](../results/Speck-Paper1/kimi-k3-primary-source-audit-v1.json)
- [Stable LatentMoE readiness v2](../research/paper-1/stable_latentmoe_readiness_v2.json)
- [Qualification](../results/Speck-Paper1/stable-latentmoe-readiness-v2-qualified.json)
