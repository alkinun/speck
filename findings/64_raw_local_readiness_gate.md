# 64 — Raw-local branch readiness gate

## Question

After HCA and CSA, does “add a local window” define one controlled intervention, or does it still hide
placement, projection, normalization, overlap, and execution-mode choices?

## First conditional formulation

The gate fixes a clean first formulation without authorizing it. One exact raw-token ring is attached
to each of the same five global integration slots; the fifteen KDA-only layers do not gain attention.
It shares the precise CSA query representation. Raw-local and selected-precise logits enter one causal
softmax over a deterministic deduplicated union, so window utility is not confounded with an independent
branch scale or learned output gate. HCA's coarse branch remains unchanged.

A raw token covered by both the local ring and selected precise access appears once at its highest
resolution. Parallel independently normalized sums, learned local/global gates, attention in every KDA
layer, and prefill-only local execution are not authorized by this version.

## Causal and evidence boundary

The ring retains exactly the most recent `min(history, W)` causal raw-token identities. Full prefill,
arbitrary chunks, one-token decode, prefix reuse, eviction, serialization, and resume must preserve
chronology, deduplication, and output within frozen tolerances. Raw coverage cannot make partial HCA or
CSA entries visible early.

The conditional window grid is 64, 128, 256, and 512, plus the no-local control. Mechanism discovery
selects the smallest window passing every aggregate, source, within-block, boundary-straddling,
unfinished-block, recent-distractor, trailing-loss, and local/distant-composition floor. Confirmation
uses independent paired cells.

Local state is an added cost, not a reduction claim: five times `W` times the selected exact-cache bytes,
plus ring metadata and scales. A retained branch must recover enough quality to justify that state,
clear the 10% simple-component systems threshold, and preserve the parent's main realized benefit.

## Decision

Fusion, window, and placement remain unqualified. Implementation, training, and promotion are blocked
until every parent qualifies; the only next action is reference union/deduplication/state correctness.

## Artifact

- [Raw-local readiness gate](../research/paper-1/raw_local_readiness_v1.json)
