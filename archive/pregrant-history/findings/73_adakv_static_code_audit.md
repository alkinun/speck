# 73 — Ada-KV immutable static code audit

## Scope

Ada-KV revision `04497ab` was inspected through its immutable Git tree and selected raw source files.
The repository was not cloned, installed, imported, or executed. Root and CUDA MIT licenses qualify
source identity and audit access, not behavioral correctness.

## Environment blockers

The project metadata is not a portable lock. `pyproject.toml` leaves several runtime dependencies
unbounded. `freeze_requirements.txt` contains machine-local conda file URLs and installs Ada-KV through
SSH at older revision `67a0fc2`, not the audited HEAD. Both files pin Transformers 4.44.2, while runtime
`check_version` identifies only 4.37 as tested and merely warns on mismatch.

`make i` builds and installs a native CUDA extension before editable package installation. That side
effect is not allowed in the research environment without a separate source/build/fixture contract.

## Runtime and evaluation blockers

The library globally monkeypatches Llama/Mistral FlashAttention, model-forward, and generation-input
methods. Its flattened per-head cache imports `tiny_api_cuda` dynamically; source comments still mark
the interface incomplete, and one cache length method returns the sentinel value one instead of retained
length. GQA selection can expand KV heads and choose mean or max aggregation, which must be frozen.

LongBench prediction loads models/tokenizers with `trust_remote_code=True`, no immutable model revision,
editable local paths, and dataset configs without pinned source revisions. It seeds Python, NumPy, and
Torch to 42, but the 38-file tree contains no dedicated tests. These properties block network-denied,
versioned reproduction.

## Decision

Upstream execution and Speck reuse remain unauthorized. The next valid step is a clean-room small-case
reference derived from the paper equations: exhaustive budget conservation/mass-bound tests, deterministic
ties/GQA/question-mode semantics, and comparison of all-required-source recall against retained mass and
L1 attention-output error. Only afterward may an isolated disposable upstream fixture be considered.

## Artifact

- [Ada-KV static code audit](../research/paper-1/adakv_code_audit_v1.json)
