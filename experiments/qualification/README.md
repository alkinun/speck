# Hardware qualification

[model.json](model.json) is the exact 1.2B parent at 4K. [throughput-gh200.json](throughput-gh200.json)
is the GH200 throughput packet, checked against the benchmark CLI by
[check_throughput_packet.py](check_throughput_packet.py). The access procedure is the
[GH200 runbook](../../docs/compute-qualification.md). [throughput-h100.json](throughput-h100.json)
is the packet for the pre-access [H100 rental](../../docs/throughput-rental.md).

## Completed

| Check | Receipt |
| --- | --- |
| RTX 3090: full-size optimization and fresh-process restart | [local-result.json](local-result.json) |
| One H100: real-data base and assistant recovery, kernels, generation, export parity | [h100-result.json](h100-result.json) |
| One H100: training, validation, checkpoint, SFT and decode timing | [timing-result.json](timing-result.json) |
| RTX 3090: throughput recipe on a 318M proxy (its configuration, `experiments/throughput-proxy`, is at tag `pre-cleanup-2026-09-25`) | [throughput-3090/sweep.json](throughput-3090/sweep.json) |
| RTX 3090: compiled restart parity for base and SFT | [base](compiled-recovery-descent-3090.json), [SFT](compiled-sft-recovery-3090.json) |
| One H100: parent and rung throughput, `liger_aligned` A/B, Hopper profile | [throughput-h100/sweep.json](throughput-h100/sweep.json) |

## Open

ARM64 GH200 execution and per-rung throughput, four-worker communication and restart (eager and
compiled), sustained throughput with checkpoint and validation overhead, and Slurm requeue.
