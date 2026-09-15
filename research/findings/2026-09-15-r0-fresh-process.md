# Fresh-process R0 checkpoint and RNG qualification

The [successor receipt](../../results/systems/r0-fresh-process-local-qualification-20260915.json)
at `2bd1676` records 12 shape-request bindings and real fresh-process recovery in tiny CPU dense/KDA
models, including two Gloo ranks. Initial workers save model, optimizer, next-microbatch cursor and
RNG state, publish an uninterrupted next-step reference and exit. Newly started workers verify those
identities and reproduce the next loss, model/optimizer tensors and RNG continuation. The eight
physical initial/restart rank reports have distinct producer/replay PIDs. These are engineering
fixtures, not trained research models or GPU performance evidence.

The explicit [v2 preparation plan](../flagship/r0_execution_preparation_v2.json) retains the checked
shapes, numerical tolerances and 70-GPU-hour envelope. Both worker generations are reserved before
launch: two 900-second deadlines plus ten-second termination grace per phase cost a conservative
2.0222 hours with four allocated GPUs. Failure of the initial phase prevents restart and keeps the
full reservation. Observed time includes both generations and their gap; scheduler-total cost remains
unmeasured. Repeated cancellation is ignored during rank cleanup so cleanup can finish.

The reference binds checkpoint payloads, cursor, request and rank identity, persisted RNG and the
completed producer result. Corrupted RNG and altered reference fixtures are rejected. A dedicated
probe exercises Torch CPU, Python and NumPy RNG, plus CUDA RNG only on a CUDA execution. CPU fixtures
cannot qualify that CUDA branch. V2 reports process-restart parity and leaves the v1 same-process
parity field null. It requires a complete producer reference; it is not hard-crash or requeue evidence.

The retained focused run passed 30 tests. After the final predecessor-plan validation addition,
full quality passed 1,274 tests, 10 skipped and 125 deselected, including all 30 focused cases, plus
format, lint, catalog and archive checks. Full-suite R0 fixture copies and runtime inventories were
reopened by SHA-256. An initial broad fixture-copy attempt stopped on deliberately unreadable sealed
test fixtures; its partial copies are preserved. No permissions were changed. The successful copy
selects only R0 fixtures; no real heldout bundle or audit was opened.

The four previously qualified implementation/test files are intentional evidence-bound successors.
Their [preserved bytes](../history/2026-09-15-r0-fresh-process/manifest.json) match the predecessor Git
revision, and the original qualification remains unchanged. The maintained `lc_hardware` status is
updated through an [explicit checked successor](../../results/systems/r0-fresh-process-status-successor-20260915.json);
other status rows remain unchanged. This is progress within R0, not a claim that R0 is complete.

Actual GH200/arm64 kernels, CUDA RNG, four-GPU NCCL, independent numerical and cached-generation
parity, hard interruption, scheduler/requeue, production-loader resume and sustained throughput
remain pending. No scientific training, site execution or additional source acquisition was launched.
The proposal remains under review with no confirmed access date. See [R0_EXECUTOR.md](../flagship/R0_EXECUTOR.md)
for the inspect-only command and site execution boundaries.
