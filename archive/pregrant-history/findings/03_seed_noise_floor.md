# 03 — Seed noise floor

## Question

Can one-seed validation differences below 0.02 nats rank the top four mixer variants?

## Design

- Architecture: `gdn-global`
- Seeds: 42, 43, 44
- Sequence length: 4,096
- Training tokens: 131,072,000
- Batch tokens: 65,536
- Final validation tokens: 19,988,480
- Packed-data manifest:
  `b84b09e0b701e35d84487cf6f91e6da9c9fb686b7f6efe67b2e2f5f301fda98e`
- Data stream and token offset: fixed
- Varied factor: training seed, primarily model initialization

Seed 42 is the completed mixer-screen run. Seeds 43 and 44 were new matched runs. Because packed
data order was fixed, this measures initialization/optimization sensitivity and is a lower bound on
total run-to-run variance.

## Results

| Seed | Validation loss | PPL | tok/s | GPU-h | Model SHA-256 | W&B run |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 42 | 2.819378 | 16.7664 | 47,724.3 | 0.8178 | `426f0f221aee…` | `v3onm3is` |
| 43 | 2.829024 | 16.9289 | 47,758.4 | 0.8122 | `29096e94c566…` | `7ofh2ac8` |
| 44 | 2.820822 | 16.7907 | 47,717.9 | 0.8124 | `8fc947137781…` | `4tjumvfg` |

Aggregate:

- Mean: `2.8230745`
- Median: `2.8208222`
- Population standard deviation: `0.0042478`
- Sample standard deviation: `0.0052025`
- Minimum–maximum range: `0.0096459`

## Interpretation

- The seed range is 61.4% of the complete original top-four range (`0.0157`).
- It exceeds the apparent original `gdn-local` advantage over `gdn-global` (`0.00888`).
- A one-seed difference below roughly 0.01 nats is not distinguishable at this budget.
- The large GDN-versus-convolution and hybrid-versus-pure-GDN gaps remain far outside this noise
  floor.
- Data-order replication remains undone, so the true noise floor may be larger.

## Checkpoints

- Seed 42: `~/.cache/speck/checkpoints/SpeckLC-150M-MixerScreen-131M-gdn-global/model_002000.pt`
- Seed 43: `~/.cache/speck/checkpoints/SpeckLC-150M-NoiseFloor-131M-seed-43/model_002000.pt`
- Seed 44: `~/.cache/speck/checkpoints/SpeckLC-150M-NoiseFloor-131M-seed-44/model_002000.pt`

## Artifacts

- Checked aggregate and full hashes:
  [results/SpeckLC-150M-NoiseFloor-131M/summary.json](../results/SpeckLC-150M-NoiseFloor-131M/summary.json)
- Experiment family:
  [experiments/SpeckLC-150M-NoiseFloor-131M](../experiments/SpeckLC-150M-NoiseFloor-131M)
- Preparation utility: [scripts/noise_floor_prepare.py](../scripts/noise_floor_prepare.py)
