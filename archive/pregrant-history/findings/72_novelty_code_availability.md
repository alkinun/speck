# 72 — Novelty-baseline code and license availability

## Method

Official repository HEADs were resolved and immutable recursive trees inspected. Only declared license
files were fetched and hashed; no repository was cloned, installed, imported, or executed.

## Results

- **FlashMorph** at `b9b6353` contains six files: a README and five images. It has no code-like or
  license file. The paper's declared repository cannot reproduce the method at this snapshot.
- **Sparse Prefix Caching** declares no official implementation. Its equations can support a clean-room
  contract, but performance is not reproduced.
- **Ada-KV** at `04497ab` has 38 files, 15 code-like Python/CUDA/shell files, a root MIT license for
  Yuan Feng (`32c2e620…`), and a separate MIT CUDA license for 66RING (`9a429046…`). It is the only
  current code-plus-root-rights path eligible for a deeper dependency/fixture/offline audit. Execution is
  not yet authorized.
- **SqueezeAttention** at `a1933d1` has 460 files and 333 code-like files, but no root license. Its only
  detected license is Apache-2.0 inside the vendored `helm/` tree; that does not license root changes.
- **Budgeted Attention Allocation** declares no official code.
- **Alternating Sparse Attention** provides paper pseudocode and cites the base NSA kernel repository,
  not a dedicated ASA implementation.

Thus three immutable repositories exist, two contain code, one has an affirmative root license covering
its code, and only one paper-specific code/rights path is currently eligible for deeper audit. Public
readability is not treated as reuse or distribution permission.

## Decision

Only Ada-KV may advance to a non-executing dependency/provenance/fixture audit. FlashMorph and papers
without code require a release or clean-room reference contract; SqueezeAttention requires affirmative
root licensing. None changes the failed novelty gate or authorizes architecture freeze.

## Artifact

- [Code availability audit](../research/paper-1/novelty_code_availability_v1.json)
