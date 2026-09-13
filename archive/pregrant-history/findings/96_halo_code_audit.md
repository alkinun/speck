# 96 — HALO source and result-log audit

## Available evidence

The official repository is pinned at `229e3013740266394640af8cff7f24cd35795ec9`. Its 271 files
include 31 Python files, 12 shell scripts, eight JSON configs, and 206 per-layer evaluation logs across
the 2B/4B/8B sweeps. Stage 1/2/3 training, layer selection, model code, and architecture configs are
present. The 2B config resolves the paper's rendered-table ambiguity: it has seven full-attention and
twenty-one Lightning Attention layers.

## Blocking differences

There is no root license or test suite. The implementation uses epsilon `1e-10` and clips importance
at 50, while the paper specifies epsilon `1e-6` without that clipping. Requirements are unpinned,
training/data paths are local or moving, FineWeb-Edu order is not identified, and linked checkpoint
revisions/custom code have not been audited. No source, dependency, or model was imported or executed.

## Decision

Source identity and the presence of result logs are qualified, but rights, paper/code semantic identity,
portable environment, data/checkpoint identity, behavior, reuse, and reproduction are blocked. HALO is
a mandatory baseline; its upstream code cannot be executed or reused for N1.

## Artifact

- [HALO code audit v1](../research/paper-1/halo_code_audit_v1.json)
