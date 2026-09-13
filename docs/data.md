# Data preparation

Data preparation produces source-separated packed shards with tokenizer, source, mixture, and checksum
metadata. Runtime artifacts belong under `speck_base_dir`, normally `~/.cache/speck`.

## Prepare a maintained example

```bash
uv run --no-sync python -m scripts.tokenizer_prepare experiments/Speck1-140M
uv run --no-sync python -m scripts.data_prepare experiments/Speck1-140M
```

This is the retained 5B-token release recipe and requires substantial network and disk capacity.
Use `python -m scripts.smoke` for a tiny offline example.

## Configuration and implementation

An experiment's `tokenizer.json` binds a prepared tokenizer. Its `data.json` specifies source revisions,
readers, filters, mixture phases, validation partitions, deduplication, and shard geometry.
`output_dir` selects an explicit destination; `output_name` selects a named dataset under the cache.

| Responsibility | Implementation |
| --- | --- |
| Configuration, quotas, and disk estimates | [configuration.py](../speck/data/configuration.py) |
| Acquisition and document readers | [acquisition.py](../speck/data/acquisition.py) |
| Bounded uint16 token shards | [packing.py](../speck/data/packing.py) |
| Preparation, recovery, and manifest verification | [dataset.py](../speck/data/dataset.py) |
| Deterministic distributed loading | [loader.py](../speck/data/loader.py) |
| Source-specific adapters | [sources/](../speck/data/sources/) |

Source manifests retain revisions, file order, filtering statistics, document indexes, and shard
identities. Resume validates committed source state before continuing. Training checks the prepared
tokenizer and shard identities; changing a corpus requires a new dataset/launch identity.

## Flagship pipeline

The flagship pipeline additionally has global deduplication, deny-ledger processing, benchmark
decontamination, evaluation partitions, operational calibration, and a data-launch receipt.
The selected [data protocol](../research/flagship/DATA.md) defines these requirements;
[current status](../research/status.json) records which stages remain.

Key commands provide `--help`: `production_data_preprocess`, `production_calibration`,
`firewall_inputs_prepare`, `data_firewall_calibrated_build`, `data_launch_preflight`, and
`tokenizer_pilot_full_train`. Frozen pre-grant contracts execute from their preserved checkout until
a new implementation qualification exists.

The [bounded source-bank rehearsal](../research/flagship/SOURCE_BANK.md) qualifies source-separated
byte selection, metadata preservation, reference-tokenizer packing, and recovery on retained inputs.
Use `source_bank_prepare` for per-invocation preparation reports and `source_bank_qualify` for a fresh
clean-checkout recovery qualification. Its outputs remain engineering artifacts until the full
source-treatment, tokenizer, capacity, and launch requirements are met.

The [upstream acquisition-unit rehearsal](../research/flagship/ACQUISITION_UNITS.md) adds fixed physical
row windows, original-row metadata, independent acquisition recovery, and ordered cohort deduplication.
Use `acquisition_units_prepare` for preparation and `acquisition_units_qualify` for the bounded
clean-checkout qualification. Complete reference exclusion is qualified by the integration below.

The [complete exclusion integration](../research/flagship/FIREWALL_INTEGRATION.md), run through
`firewall_integrate`, connects larger units to all twelve reference views and bank schema v2. Its
bounded real-data qualification passes reference preservation, exact/near controls, recovery, and
all six bank handoffs. Training-scale capacity and the final tokenizer remain separate requirements.

The [screen-capacity and timing review](../research/flagship/SCREEN_CAPACITY.md) provides
`data_screen_capacity` for conditional per-category E1/E3 supply accounting and `dedup_timing_replay`
for a private, durable reference-checkpoint replay with phase/checkpoint-component timings. Use the
v2 replay plan; the initial timing diagnostic is retained at its original revision.

For source surveys, see the [literature library](../research/literature/README.md). Earlier curriculum
details remain in the [original guide](../archive/pregrant-history/docs/data.md).
