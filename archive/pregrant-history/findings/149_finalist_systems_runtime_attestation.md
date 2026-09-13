# 149 — Runtime identity and attestation builder qualifies with mocks

## Exact identity vector

The builder requires a clean 40-hex git commit, exact GPU UUID and driver row, Python, PyTorch, CUDA
runtime, flash-linear-attention, and Triton versions, plus model-config and benchmark-engine hashes.
Empty GPU-environment versions fail.

The local CPU environment is intentionally ineligible: Python 3.10.20 and Torch 2.9.1+cpu are present,
while CUDA, flash-linear-attention, and Triton are absent. That confirms attestation must run inside the
future frozen GPU environment rather than borrowing the test environment's identity.

## Hash-bound runtime probe

A qualified probe must bind protocol, engine result, trace, run, and PID; report at least one compiled
graph; retain zero graph breaks, eager fallbacks, and OOM; and provide positive CUDA forward/backward/
optimizer times whose sum does not exceed wall time. The attestation preserves its path and hash and is
accepted by the synthetic trial assembler.

Nine mock tests pass. The builder is qualified, but the live probe/counter producer, live GPU identity,
orchestrator, pipeline, and execution remain blocked.

## Artifact

- [Runtime-attestation qualification](../results/Speck-Paper1/finalist-systems-runtime-attestation-qualified-v1.json)
