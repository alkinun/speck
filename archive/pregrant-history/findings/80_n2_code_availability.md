# 80 — HeadKV and routing-artifact availability audit

## HeadKV immutable tree

The official HeadKV repository is pinned at revision `0862a0955fe82e9ff611d59541918e02c5def625`
(10 March 2025). A metadata-only tree inspection found 426 files: 37 source-code and 376 data-like
files. No working tree was checked out, and no third-party module or binary was imported or executed.

The only license is `csrc/LICENSE`, the MIT file for the 66RING-derived CUDA subtree. No root license
covers HeadKV's Python, shell scripts, head profiles, or embedded data. This blocks reuse and execution
regardless of source visibility.

## Reproduction blockers

The tree has no requirements, lock, pyproject, or environment file. Its seven-line `setup.py` declares
the package name `adakv`, version 0.0.1, and no dependencies. The monkeypatch layer names Transformers
4.37 as tested but only warns on mismatch, then replaces global Llama/Mistral forward and generation
methods. Cache code imports `tiny_api_cuda`; two prebuilt Linux x86_64 eggs for Python 3.8/3.12 are
committed without a locked build chain.

Model identity is partial: some Mistral tokenizer loads pin a revision, but model weights and Llama
loads do not. Profile construction can enable remote code. Four committed head-score profiles are not
bound to exact code/model/tokenizer/prompt/output manifests. Embedded LongBench JSONL and BABILong Arrow
copies lack a source-revision, derivation, checksum, or rights manifest. `test.sh` launches an experiment;
there is no assertion-based behavior suite.

The allocation code also floor-divides a budget pool by floating beta and independently rounds head
capacities without a post-round conservation assertion. A clean-room global layer/head apportionment
would need its own frozen oracle; the current local within-layer reference does not validate this rule.

## Routing-paper artifact boundary

The 2026 attention-dynamics paper names NVIDIA KVPress and Expected Attention but declares no dedicated
code repository, dataset bundle, per-example results, or immutable dependency revisions on its arXiv
record. Current KVPress HEAD is a moving observation, not the historical experiment identity, and cannot
be substituted silently.

## Decision

HeadKV remains a mandatory conceptual baseline, but upstream execution, reuse, embedded profiles/data,
and behavioral claims are blocked. The routing study is not presently reproducible from declared
artifacts. Neither result authorizes an N2 experiment, changes novelty status, or advances architecture
freeze.

## Artifacts

- [HeadKV static code audit](../research/paper-1/headkv_code_audit_v1.json)
- [Novelty code-availability v2](../research/paper-1/novelty_code_availability_v2.json)
