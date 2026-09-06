# 154 — NoPE requires the missing mixer-by-position factorial

## What is actually known

Seed 42 has a clean from-scratch GDN comparison: changing exactly five global `rope_dim` values from
32 to 0 leaves parameters/FLOPs fixed but worsens final language loss by 0.024367 nats. It greatly
strengthens distant-content sensitivity, while exact retrieval remains zero. Late conversion likewise
fails language retention.

The three-seed frontier is not NoPE replication: it compares GDN/RoPE with KDA/NoPE. KDA/RoPE is absent,
and GDN/NoPE is absent on seeds 43/44. The missing KDA/RoPE config is expressible with the same
153,958,938 parameters and 1,021,601,280 FLOPs/token as KDA/NoPE.

## Frozen 2×2 design

V1 crosses GDN/KDA with RoPE32/NoPE0 over the six finalist seed×data-order cells. It estimates both
within-mixer positional effects and their interaction; pooled six-cell output is descriptive. Holm
covers the two positional contrasts, while aggregate/source language non-inferiority and all capability
floors remain intersection gates. A sensitivity-only gain cannot pay for a language failure.

The still-unseen six KDA/NoPE finalist runs are preregistered for reuse only after v3 acceptance. This
saves duplicate training while locking the design before outcomes. The other three arms require 18 new
runs. NoPE can be selected only within the independently selected mixer; interaction forbids a generic
NoPE claim.

## Active boundary

The design is unregistered and training-blocked. It changes neither the active execution nor program.

## Artifacts

- [NoPE factorial](../research/paper-1/nope_factorial_v1.json)
- [Qualification](../results/Speck-Paper1/nope-factorial-v1-qualified.json)
