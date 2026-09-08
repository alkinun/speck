# Slurm operations

The flagship Slurm layer turns a frozen wave manifest into one-GPU arrays or one-node four-GPU
scripts. It validates and records operations; it does not choose scientific winners, advance a
phase, or authorize protected-reserve use. Runtime files default to `~/.cache/speck/slurm` and must
remain outside the checkout.

## Wave contract

A wave is a strict `speck_slurm_wave` version-1 JSON object. Unknown fields fail validation. Paths
are resolved relative to the wave file; absolute paths are appropriate when the wave itself lives
outside the checkout. Each job binds the exact Git commit and at least one `config` and one `data`
file by SHA-256. Additional `checkpoint`, `authority`, and `tool` identities are supported.
Preflight binds the commit's Git tree and rejects staged or unstaged changes to tracked files.
Untracked files, including local `.opencode`-style harness state, are ignored unless a job declares
them as an explicit identity.

The following abbreviated one-job manifest shows every field:

```json
{
  "format": "speck_slurm_wave",
  "format_version": 1,
  "wave_id": "p1-e1w-a",
  "created_at_utc": "2026-10-01T00:00:00Z",
  "budget": {
    "total_gpu_hours": 5000,
    "mandatory_gpu_hours": 4111,
    "reserve_gpu_hours": 889
  },
  "plan": {
    "path": "/shared/speck/research/flagship/plan.json",
    "sha256": "<64 lowercase hex characters>"
  },
  "repository": {
    "path": "/shared/speck",
    "commit": "<40 lowercase hex characters>",
    "require_clean": true
  },
  "jobs": [
    {
      "id": "e1w-screen",
      "phase": "P1",
      "kind": "train",
      "allocation": "mandatory",
      "resources": {
        "nodes": 1,
        "gpus": 1,
        "cpus_per_task": 16,
        "memory_mb": 120000,
        "walltime_minutes": 240,
        "signal_seconds": 300
      },
      "array": {"indices": [0, 1, 2, 3], "max_parallel": 4},
      "command": [
        "uv", "run", "--extra", "gpu", "torchrun", "--nproc-per-node=1",
        "-m", "scripts.slurm_base_train", "/shared/experiments/e1w-{array_index}",
        "--output-dir", "/shared/checkpoints/e1w-{array_index}",
        "--slurm-requeue-resume"
      ],
      "working_directory": "/shared/speck",
      "identities": [
        {"role": "config", "path": "/shared/experiments/e1w-0/train.json", "sha256": "<sha256>"},
        {"role": "data", "path": "/shared/data/e1w/manifest.json", "sha256": "<sha256>"}
      ],
      "max_retries": 1,
      "depends_on": []
    }
  ]
}
```

`{array_index}` must be a complete command argument. In a real heterogeneous array, bind every
arm's config files in `identities`, not just the first arm shown above. A four-GPU job sets `gpus` to
4, sets `array` to `null`, removes the placeholder, and normally puts
`torchrun --nproc-per-node=4` in its immutable command. Every `train` job must use
`torchrun -m scripts.slurm_base_train --slurm-requeue-resume`; the historical base trainer remains
the single owner of the training loop, while the Slurm subclass supplies only an optimizer-boundary
stop hook. SFT requeue is explicitly unsupported and fails wave validation.

Maximum commitment is wall time × GPUs × array tasks × (1 + `max_retries`). Mandatory and reserve
commitments are tracked independently against 4,111 and 889 GPU-hours. Initial submissions reserve
their maximum mandatory commitment and duplicate submission of the same manifest is rejected.
Reserve jobs may be validated and rendered for human review, but `submit` and `retry` always refuse
them. Mandatory jobs use their scientific phase; reserve jobs use P7, so the pools cannot be blurred.
Reserve jobs must set `max_retries` to zero, preventing a manually started reserve script from
self-requeueing. Every further reserve attempt needs renewed human handling. There is deliberately no
promotion command.

The budget commitment is written before the first `sbatch`, and every returned scheduler ID is
written as a separate immutable event before the next job is submitted. If a later submission fails,
the wave stays committed and cannot be blindly rerun; inspect `submission-events/<manifest-sha256>`
and recover or cancel the partial wave manually.

Dependencies are accepted only on `collect` and `eval` jobs and render as `afterok`. Training jobs
cannot depend on another job: an operator must inspect the frozen scientific result and create a new
wave after making any promotion decision.

## Commands

Run from the repository root. `validate` is structural and hash-checks the execution plan;
`preflight` additionally requires the exact clean Git checkout and verifies all bound files.

```bash
python -m scripts.slurm_ops validate /shared/manifests/p1-e1w-a.json
python -m scripts.slurm_ops preflight /shared/manifests/p1-e1w-a.json
python -m scripts.slurm_ops render /shared/manifests/p1-e1w-a.json
```

No account or partition is guessed. If the site requires them, pass confirmed values explicitly;
they are operational rendering inputs and do not alter the scientific manifest.

```bash
python -m scripts.slurm_ops render /shared/manifests/p1-e1w-a.json \
  --account CONFIRMED_ACCOUNT --partition CONFIRMED_PARTITION
python -m scripts.slurm_ops submit /shared/manifests/p1-e1w-a.json \
  --account CONFIRMED_ACCOUNT --partition CONFIRMED_PARTITION
```

Record scheduler state using the immutable submission record printed by `submit`:

```bash
python -m scripts.slurm_ops collect \
  ~/.cache/speck/slurm/submissions/<manifest-sha256>/<timestamp>.json
python -m scripts.slurm_ops summary --date 2026-10-01
```

After an authorized person submits an all-reserve wave manually, register its scheduler IDs so the
same collection and protected accounting apply. The authorization file and hash must already be an
`authority` identity on every job. Registration invokes no Slurm command.

```bash
python -m scripts.slurm_ops register-reserve /shared/manifests/recovery.json \
  --job recover=123456 --authorization /shared/approvals/reserve-001.json \
  --authorization-sha256 <sha256>
```

The collector asks `sacct` for job ID, state, exit code, elapsed seconds, allocated TRES, restart
count, and timestamps. It excludes `.batch`/`.extern` steps and array-parent rows when task rows are
present. Results classify as `success`, `active`, `mechanical_retryable`,
`resource_configuration_failure`, `operator_cancelled`, `application_failure`, or `unknown`.
GPU-hours are elapsed seconds × allocated GPUs. The daily report keeps observed mandatory and
reserve use separate and also shows conservative commitments.

A retry is allowed only when the supplied observation belongs to the same manifest/submission and
every non-success row for the logical job is a mechanical failure (`BOOT_FAIL`, `NODE_FAIL`,
`PREEMPTED`, `REVOKED`, or `TIMEOUT`). It resubmits the unchanged script, restricts arrays to failed
task indices, carries an attempt offset, and refuses to exceed `max_retries`. It does not retry OOM,
application, cancellation, or scientific failures.

```bash
python -m scripts.slurm_ops retry /shared/manifests/p1-e1w-a.json e1w-screen \
  ~/.cache/speck/slurm/submissions/<sha>/<submission>.json \
  ~/.cache/speck/slurm/sacct/<sha>/<observation>.json
```

## Timeout and requeue contract

Rendered scripts request `USR1` before timeout. The batch-shell trap writes a per-job sentinel rather
than signaling the `torchrun` launcher; all trainer ranks poll that sentinel at optimizer boundaries.
`scripts.slurm_base_train --slurm-requeue-resume` also installs a direct `SIGUSR1` handler inside a
Slurm job, for sites that deliver the signal to tasks. The handler only sets a flag. After the current
optimizer step finishes, all ranks coordinate while rank zero writes the normal exact-resume
checkpoint. The trainer exits 99.
The wrapper waits for checkpoint completion, asks `scontrol requeue` only while the same manifest's
retry bound remains, and otherwise preserves exit 99 as a terminal failure. The attempt number is
persisted outside Git before `scontrol requeue`, so the bound does not rely only on a site's restart
environment variable. On a scheduler or operator retry, the trainer resolves the latest complete
checkpoint only when the retry offset or Slurm restart count is nonzero. Normal runs still require
explicit `--resume STEP` and never resume implicitly.

The rendered script exports the manifest's retry ceiling. Missing, malformed, or exhausted retry
counters fail closed before checkpoint discovery. A requeue checkpoint and summary are always marked
partial, including a signal received on the last optimizer step. They use
`partial_run_summary_<step>.json` with the `speck_base_partial_run_summary` contract; only a run with
final-step validation and finite parameters may publish canonical `run_summary.json`. Partial
checkpoints cannot be exported. On resume, the first ten optimizer steps of that process remain
startup/autotune overhead rather than steady-state training time.

Rendered one-node jobs also export their expected local world size (one or four). Runtime startup
requires a complete integer `RANK`/`LOCAL_RANK`/`WORLD_SIZE` tuple, agreement with that one-node
contract, and local ranks within visible CUDA devices before NCCL initialization. Generic multi-node
execution is outside this Slurm v1 layer: the shared runtime can validate `LOCAL_WORLD_SIZE` and local
GPU range, but it does not infer node topology, rendezvous correctness, or cross-node fabric health.
Every initialized rank all-gathers the packed-manifest and tokenizer identity before model
construction and rejects disagreement.

## Cluster confirmations required before qualification

Confirm these values with the site rather than adding guesses to checked files:

- account, partition/QOS, allocation dates, and maximum wall time;
- whether `--gres=gpu:4` receives the exclusive four-GH200 node and whether four concurrent
  one-GPU array tasks are supported;
- CPU and memory limits, local/shared filesystem paths, quotas, and checkpoint retention;
- permission for a batch job to run `scontrol requeue` and the site's batch-shell `USR1`
  delivery/requeue behavior;
- `sacct` retention and support for `AllocTRES`, `ElapsedRaw`, and `RestartCnt`;
- the arm64 CUDA, NCCL, PyTorch, Torch/FLA kernel, network, and container/module environment.

Run an unpaid or explicitly authorized short rehearsal before relying on signal/requeue behavior.
The layer does not establish GH200 correctness, DDP reliability, achieved TFLOP/s, data readiness,
or P0 exit authority by itself.
