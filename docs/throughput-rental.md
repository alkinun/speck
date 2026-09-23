# H100 throughput rental

Measure the **flagship 1.2B speedup over the frozen pilot recipe** on rented H100 time, before
grant access. The packet is [`throughput-h100.json`](../experiments/qualification/throughput-h100.json);
this document is how it is executed. It spends **zero grant GPU-hours** and is charged to the
external rental ledger, exactly as the completed pilot was.

Why it exists: the [RTX 3090 pass](../experiments/qualification/throughput-3090/sweep.json) selected
a configuration but could only measure it on a 318M proxy, and no checkpointing-off flagship
configuration fit that 24 GiB card. Without this rental the
[GH200 confirmation sweep](../experiments/qualification/throughput-gh200.json) would be the first
place a flagship speedup is ever seen, inside a 120-hour reservation that cannot be re-spent.

It does **not** replace the GH200 phase. `device_batch_size` is immutable on resume and depends on
96 GiB rather than 80 GiB, so it is still frozen on the grant hardware. Absolute tokens/s used to
re-anchor horizons also come from the GH200, not from here.

## Preflight evidence

[`throughput-h100-preflight.json`](../experiments/qualification/throughput-h100-preflight.json)
records the earlier local preflight. Three defects were found and fixed before any machine was rented:

- The `--profile` hole in the old command template was filled with a boolean. The flag takes a
  trace path, so the profile run would have died in `argparse` after the whole ladder was paid for.
- The `h100-best-endtoend` run switches to end-to-end mode, which loads a packed manifest, but
  nothing supplied `--data-dir`. It would have fallen back to a cache path that does not exist on a
  fresh host, so the loader-inclusive check would have failed.
- Nothing bound the packet to the CLI at all.

Every run now carries a machine-generated `argv`, checked against the real
`speck.training.benchmark` parser by
[`check_throughput_packet.py`](../experiments/qualification/check_throughput_packet.py), which runs
in `make plan-check` and in the test suite. The ladder was also executed locally on the RTX 3090 at
the baseline configuration and in end-to-end mode against the real packed pilot pack, confirming the
1,195,884,576-parameter reference steps at sequence length 4096 and that the manifest relocates.
Those local numbers describe a 3090 and establish no throughput result.

## What the operator provides

Nothing below can be done from this repository:

1. An **80 GiB H100 SXM** instance with one allocated GPU. Record instance ID, hourly rate,
   storage/network charges and intended deletion time.
2. SSH host, port and user.
3. A declared **rental spend ceiling**, so the packet's stop condition has a number to compare to.

The operator controls creation and deletion. Stopping a benchmark does not stop billing.

## Build and transfer the bundle

The builder refuses to run against a dirty working tree, so commit first.

```bash
uv run --no-sync python -m scripts.gh200_check bundle \
  --output /mnt/speck-data/speck/h100-throughput-20260922 \
  --data /mnt/speck-data/speck/data/flagship-pilot-105m \
  --tokenizer /mnt/speck-data/speck/tokenizer-final-mistral-v1 \
  --assistant /mnt/speck-data/speck/gh200-readiness-20260918/assistant-rehearsal-2
```

This writes `h100-throughput-20260922.tar.gz` plus a transfer receipt carrying its SHA-256 and the
source commit. The payload is about 240 MB, almost all of it the packed pilot pack that only the
end-to-end run reads. Copy both files, verify the SHA-256 **after** transfer, then extract into a
private directory on the instance's persistent disk. Do not publish the corpus payload.

## On the instance

From the extracted bundle root, clone its recorded branch, install the pinned GPU dependencies,
and bind the transported inputs before printing benchmark commands:

```bash
speck_bundle_branch=$(python3 -c 'import json; print(json.load(open("bundle.json"))["branch"])')
git clone --branch "$speck_bundle_branch" code.bundle code
cd code
uv sync --locked --python 3.10 --extra gpu --extra linear --group dev --group transformers
uv run --no-sync python -m scripts.gh200_check bind ..
export TORCHINDUCTOR_CACHE_DIR=$PWD/../inductor-cache   # preserve across runs
export TRITON_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR/triton # also preserve FLA kernel choices
export CUBLAS_WORKSPACE_CONFIG=:4096:8               # required by the deterministic recipe
mkdir -p ../results/throughput-h100
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv
```

Run the remaining commands from `code/`. Binding verifies the bundled commit and input hashes,
then creates `../relocated-base` with the transported tokenizer and packed-data paths. Both rental
packets use that experiment; the source `experiments/pilot` contains workstation paths. Keep
results and caches outside the checkout so measurement receipts continue to record a clean source.

Confirm the card is an 80 GiB H100 before spending anything. Then establish the noise band: run the
first configuration **five times** and keep the spread. A delta smaller than that band is not a
result. Repeat the first command with a unique label and output filename for each copy. The packet
uses ten warmup steps and a thirty-step measured window; keep the full window after compilation has
settled rather than shortening it to the old short probe.

After the five baseline runs, create the machine-checked noise summary before reading any ladder
delta:

```bash
uv run --no-sync python -m scripts.throughput_summary \
  ../results/throughput-h100/h100-pilot-baseline-r*.json \
  --output ../results/throughput-h100/baseline-summary.json
```

Use fresh version 2 receipts from this checkout. The helper rejects mixed hardware, software,
source, data or runtime settings, unstable runs, copied receipts and invalid measurements. It
refuses to overwrite an existing summary. The operator still verifies the physical instance and
clock/power stability; matching recorded device names alone cannot establish those.

Print the exact ladder from the packet rather than retyping it:

```bash
uv run --no-sync python experiments/qualification/check_throughput_packet.py \
  experiments/qualification/throughput-h100.json --print-commands
```

Run the eight configurations in the packet's order. The first three isolate the pilot baseline, the
checkpointing-off gain and the compile gain; the next three are the microbatch ladder; the last two
measure packed-loader overhead and take the Hopper kernel profile at the selected microbatch. The
two `selected` runs take the microbatch and accumulation that the ladder chose. Substitute those
values in the copied commands, preserving 131,072 tokens per update. Keep the committed packet
unchanged: editing it during the sweep would mark subsequent receipts dirty. `--print-commands`
only prints commands; its selected entries contain preflight probe values identified by an
operator-substitution comment. Each benchmark receipt records the arguments actually executed.

Record SM clock, temperature and power around every run.

## Stop conditions

Stop and diagnose rather than explore:

- Clock drift beyond 5%, unstable step times, or a graph-break count differing from the Ampere
  result without explanation.
- Out-of-memory at a microbatch the 80 GiB card was expected to hold. Record the failing
  configuration and stop the ladder there.
- Rental spend reaching the declared ceiling. Preserve every receipt and stop rather than extending.

## What to bring back

Copy the whole `../results/throughput-h100/` directory and the profile trace back before deleting the
instance, then record:

- Flagship speedup over the baseline measured on this same host, with tokens per update fixed.
- Estimated MFU with its timing boundary, FLOP estimate and dense BF16 peak denominator.
- Loader-inclusive/compute throughput ratio. This does not replace the pilot's full-trainer
  derate: neither benchmark mode measures startup, validation or checkpoint saves.
- Memory headroom, repeated-run spread, warmup cost and graph-break diagnostics.
- The selected Hopper trace and the next optimization supported by it.

Use the [performance plan](performance.md) for the compact result handoff. A separate sustained
trainer run at the selected cadence must measure the full-trainer derate; leave it unmeasured
if the rental only executes this benchmark ladder.

After GH200 and distributed qualification, apply the predeclared
[re-anchoring rules](program.md#compute-and-allocation); a surplus does **not** buy a longer base run.

Verify the instance is actually deleted. Provider billing state is separate from local backups.
