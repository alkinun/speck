# 103 — N1 claim table after cross-task KL selection

## Claim correction

Generic-text KL selection with downstream recall evaluation, cross-scale evidence, and GDN-to-GLA
transfer is now a mandatory baseline. Teacher-dependent clustered placements and explicit spacing
interventions are direct prior art.

This removes the residual's held-out-task, cross-mixer, cross-scale, clustering, and spacing stories.
Only teacher-free prediction of jointly trained from-scratch interactions remains. That is procedural,
not an architecture mechanism, and has no demonstrated causal or systems value.

## Decision

N1 retirement is now evidence-favored and requires independent review. It has no experiment authority.
There are zero established architecture-novelty candidates; N2 remains deferred. The frozen baseline
continues independently and cannot satisfy the novelty gate.

## Artifact

- [Novelty claim-overlap v6](../research/paper-1/novelty_claim_overlap_v6.json)
