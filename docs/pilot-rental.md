# One-H100 pilot launch

This workflow runs the frozen 104,857,600-token pilot, exports its final base checkpoint, and scores
one complete development partition. The earlier H100 rehearsal and timing runs are preserved
separately. No final-test evaluation, SFT, architecture change, microbatch-four trial, or automatic
retry is included. The implementation is `scripts.pilot_rental`.

The [local readiness receipt](../experiments/pilot/rental-readiness.json) records the built archive,
source commit and validation. Relocation and offline grader checks passed locally. The complete
GPU workflow completed training, then failed offline export template enumeration. The
[execution receipt](../experiments/pilot/h100-run.json) preserves that failure and the separate
recovery implementations. Recovery uses staged template files and the original checkpoint under
the remaining six-hour deadline; it does not retrain. Export checks passed; development evaluation
started after removing supervisor rank variables that incorrectly triggered distributed startup.
The endpoint later refused SSH and the gateway reported container not found. The final checkpoint
and CPU export reconstruction are verified locally. The original eight-hour watcher ended at
2026-09-19 04:11 UTC without observing recovery. Its terminal evidence is
preserved in the execution receipt. The user subsequently restarted one H100 and supplied access.
The retained disk contains only 84 partial GSM8K evaluation rows; the old evaluator is no longer
running. A fresh complete development pass now uses `development-restored-20260919/`, supervised
under `recovery-after-restart-20260919/`, with a new four-hour reservation ending September 19
at 09:59:06 UTC. Training and export are reused. The previous partial attempt is preserved;
the frozen evaluator has no resume support. Existing backup/grading services and the milestone
watcher have been resumed against direct SSH port 13431, without duplicate workers.
The original six-hour execution deadline has expired and is not reused. Do not interpret lost SSH
as proof that billing stopped or that all remote artifacts are backed up. Use the recorded implementation commit
for each phase even when documentation on main advances.


Export parity now compares native and exported BF16 logits on matching full and cached paths,
then checks cached/full-pass semantics in FP32 on the same rounded release weights. The report
retains BF16 cross-path drift separately: trained weights exposed ordinary reduced-precision
shape-dependent rounding that the earlier combined comparison conflated with wrapper drift.
The trained checkpoint matched its export exactly on both CPU and CUDA BF16 paths in the retained
diagnostic; CPU FP32 cached/full error was 4.77e-6. This does not establish long-context accuracy.

Use persistent job supervision for local transfers and deferred grading: this run's original
exec workers ended between task turns. Their states and partial files were preserved before
migration to systemd user services. The phase watcher also checks worker liveness. Reclaim remote
checkpoint files only after matching their complete local backup against remote hashes; remove
the remote completion marker before weights. The step100 reclaim is recorded in the execution receipt.

The migrated Runpod container denies user-namespace creation. For that host, pass
`--defer-code-grading` to both `check` and `run`: model generation and non-code scoring stay on the
GPU host, while generated Python is never executed there. Remote completion is explicitly
`awaiting_local_code_grading`. Copy its `development` directory and the attempt's `prepared.json`
to the local sandbox host, then finalize with `scripts.code_grade` using the same protocol and a
local prepared-input file. The finalizer verifies source hashes, task identities and complete code
coverage before scoring. It preserves the original pending result and writes a separate result.
This split changes execution location, not tasks, decoding, code tests, or their denominator.

## Local packet preparation

After committing a clean checkout, build the private archive on the retained-data machine:

```bash
uv run --no-sync python -m scripts.pilot_rental prepare \
  --output /mnt/speck-data/speck/h100-pilot-launch-20260918 \
  --data /mnt/speck-data/speck/data/flagship-pilot-105m \
  --tokenizer /mnt/speck-data/speck/tokenizer-final-mistral-v1 \
  --evaluation /mnt/speck-data/speck/flagship-preparation-20260917/evaluation.json \
  --nltk-data /home/alkin/nltk_data
```

Preparation verifies frozen configuration, packed-manifest, evaluation-partition and tokenizer
identities and all packed shards. It copies the packed corpus, tokenizer, benchmark files and
partitions, English NLTK resource, and the three pinned export-template files from the local cache.
It bundles committed source, hashes every payload, and writes an archive SHA-256 in the adjacent
`.transfer.json`. It neither allocates a GPU nor starts training. A partial build is preserved;
use a new output path after investigating any failure. Keep this archive private.

The packet is portable. On the rental, check its archive SHA-256 against the local transfer receipt
before extracting it. Read `branch` from `packet.json` to select the bundled source branch:

```bash
cd /workspace/h100-pilot-launch-20260918
# Replace PACKET_BRANCH with packet.json's branch, not a remote branch.
git clone --branch PACKET_BRANCH code.bundle code
cd code
uv sync --locked --python 3.10 --extra gpu --extra linear --group transformers --group capability
uv run --no-sync python -m scripts.pilot_rental check .. --output /workspace/pilot-preview
```

The check verifies the clean bundled commit and every transported payload, then writes exact
relocated configurations, command arrays, and environment settings to the fresh preview directory.
There is no GPU execution. Input files remain offline during the actual run; dependency installation
happens during setup. The packet deliberately excludes credentials and local virtual environments.

Use an x86_64 rental with exactly one full H100 allocated, at least 200 GB SSD capacity, and at least
128 GiB free at preflight. Run as a non-root account with write access to the packet's parent and
output directories. Install bubblewrap and enable Linux user namespaces before launch; the code
grader refuses an unsandboxed fallback. Check `nvidia-smi` for other workloads. The runtime checks
Python 3.10, CUDA 12.8, PyTorch 2.9.1, Triton 3.5.1, FLA 0.5.0, and Liger 0.8.2. The pinned scorer
revisions and golden grader checks must pass before training. A different GPU/runtime needs its
own qualification; this command does not qualify GH200 or distributed execution.

## Budget and execution

Run in a persistent terminal with a complete console log. Replace `ACCOUNTED_EXTERNAL_HOURS` with
the cumulative allocated GPU-hours outside this ledger: earlier pilot-related work, setup, transfers,
and idle time. Do not count ledger attempts twice. Record provider instance ID, rate, start time,
and storage/network charges beside the ledger. Never assume that unknown earlier usage is zero.
When changing providers or instances, restore the same cumulative ledger before another attempt.

```bash
uv run --no-sync python -m scripts.pilot_rental run .. \
  --ledger /workspace/pilot-runs \
  --prior-gpu-hours ACCOUNTED_EXTERNAL_HOURS
```

For a host without namespaces, append `--defer-code-grading`; finalize on the local host after copy:

```bash
.venv-open-slm/bin/python -m scripts.code_grade \
  --protocol experiments/pilot/evaluation.json \
  --prepared /mnt/speck-data/speck/flagship-preparation-20260917/evaluation.json \
  --source-prepared /external/copied-attempt/prepared.json \
  --pending /external/copied-attempt/development \
  --output /external/copied-attempt/graded-development
```

One attempt reserves **six GPU-hours** against the pilot's existing **50-hour cumulative ceiling**.
The ledger retains the entire reservation after success or failure, or the observed time if greater;
unused time is conservatively unavailable to automatic retries. External cumulative usage may
increase between invocations and cannot decrease. A filesystem lock excludes concurrent attempts.
Changed result receipts or unresolved attempts block execution. A killed supervisor leaves its
reservation unresolved; inspect process and provider state before any explicit reconciliation.
Do not delete or replace the ledger to bypass those checks.

Each attempt gets a fresh directory. Its phases run sequentially under one six-hour execution
deadline, allowing termination grace within that window:

1. Check hardware, memory, software and sandbox prerequisites.
2. Qualify the frozen graders with known answers, faults, timeouts and isolation probes.
3. Train 800 steps with the original 4K, microbatch-one, 131,072-token global batch and schedule.
4. Export step 800 locally and check native/Transformers/tokenizer parity.
5. Run the complete development partition with the frozen decoding budgets, in base mode.

The launcher changes paths and disables external tracking; scientific pilot settings are preserved.
Every phase has its command and worker log. Any nonzero exit, timeout, or interrupt stops progression;
existing trainer checks reject nonfinite states and input/checkpoint inconsistencies. A timeout
terminates the process group and preserves the last durable checkpoints; it cannot guarantee a new
checkpoint at the deadline. There is no automatic resume. Diagnose a failure before preparing a
separate, reviewed recovery launch within the remaining budget.

The measured training/validation/checkpoint projection is 2.20 H100 hours; one development backend
pass adds approximately 2.15 hours. Six hours leaves working margin for export, grading and runtime
variation, but completion is not guaranteed. Installation, initial packet verification, transfer,
idle time and provider billing are outside the process deadline and must be accounted separately.
The supervisor never stops or deletes the provider instance.

## Completion and backup

A successful attempt has `result.json` with `status: completed`, a matching entry in `ledger.json`,
a final `checkpoints/run_summary.json` with 800 completed steps and 104,857,600 global tokens, a
parity-checked export, and development outputs/results. Inspect source losses, gradients, generated
samples, task denominators, failures and all-in cost before selecting a larger training horizon.
Low capability scores at this short endpoint are diagnostic, not a release-quality judgment.

Before stopping/deleting the rental, copy the **entire ledger directory**, including failed attempts,
to durable local storage. This retains all eight periodic checkpoints (about 73.25 GiB), the final
export, grader/development outputs, bound configurations, environment, commands and logs. Preserve
the original packet and transfer receipt locally. Create a SHA-256 file inventory on the rental
and verify it against the copied files locally; do not infer backup success from a completed copy
command alone. If bandwidth requires keeping fewer checkpoints, preserve the final model/optimizer,
recovery metadata, all small evidence and every failed artifact needed for diagnosis first, and
record exactly what was omitted. The user controls rental shutdown after backup verification.

With every attempt stopped and no writer active, produce the inventory on the rental:

```bash
cd /workspace/pilot-runs
rg --files --hidden -g '!SHA256SUMS' -0 | sort -z | xargs -0 sha256sum > SHA256SUMS
sha256sum --check SHA256SUMS
```

Copy that entire directory through the rental's SSH transport. On the durable local copy, run
`sha256sum --check SHA256SUMS` again from its root, and require every file to pass before shutdown.
Retain both verification logs and account for inventory/copy time as provider usage.
