# Falcon-H1: A Family of Hybrid-Head Language Models

- **Paper:** [arXiv:2507.22448](https://arxiv.org/pdf/2507.22448)
- **Version reviewed:** v1, 30 July 2025
- **Code and models:** [tiiuae/falcon-h1](https://github.com/tiiuae/falcon-h1)
- **Primary topic:** parallel attention/Mamba-2 heads and hardware-aware model shape

## Central claim

Falcon-H1 runs attention and Mamba-2 heads in parallel on the same layer input, concatenates their
outputs, and applies a shared output projection. This gives every layer both a high-resolution retrieval
path and a fixed-state recurrent path while allowing their channel budgets to be tuned independently.

The final block is semi-parallel: attention and SSM form the token mixer in parallel, followed by the
MLP. This layout performs better in the paper's architecture screen than making all three branches
parallel or placing all three sequentially.

## Architecture search findings

The controlled screen uses an approximately 1.2B, 60-layer model trained for 70B tokens. Attention,
SSM, and MLP channel allocations are varied at roughly fixed parameters.

- More attention channels consistently worsen loss in the tested region; the minimum tested attention
  allocation, `1/8`, is preferred.
- The best semi-parallel allocation is approximately SSM:attention:MLP `2:1:5` in the paper's chunk
  units. This is an internal-width allocation, not a layer ratio like Kimi's 3:1.
- For a fixed `state_dimension × group_count` budget, fewer groups and larger state dimension improve
  loss, while throughput peaks near state dimension 16. The released models choose state 128 or 256 as
  a quality/speed compromise.
- SSM head sizes below 32 underutilize the GPU; 64 or larger maintain high throughput. Larger heads also
  slightly improve the reported loss.
- A custom causal-convolution sweep over kernels 2–32 finds only small quality changes and a systems
  trade-off; the released configurations do not treat kernel 4 as sacred.
- SSD chunk sizes 128–256 form the useful throughput plateau; the paper selects 256.

## Released family

| Model | Parameters | Layers | Attention Q/KV heads | SSM heads | Context | Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H1-0.5B | 0.52B | 36 | 8/2 | 24 | 16K | 2.5T |
| H1-1.5B | 1.55B | 24 | 8/2 | 48 | 128K | 3T |
| H1-1.5B-Deep | 1.55B | 66 | 6/2 | 24 | 128K | 3T |
| H1-3B | 3.15B | 32 | 10/2 | 32 | 128K | 2.5T |
| H1-7B | 7.59B | 44 | 12/2 | 24 | 256K | about 12T |
| H1-34B | 33.6B | 72 | 20/4 | 32 | 256K | about 18T |

The family uses unusually large RoPE bases, reaching `1e11` in later configurations. Training length is
increased progressively, and the paper reports that the large base improves extension, but this choice is
entangled with model size and continuation stages.

The matched 1.5B shape experiment is also informative: a deep/narrow 66-layer model improves quality
over a 24-layer model at equal parameters, but training and inference throughput fall roughly 25–30%.

## What matters for Speck

Falcon-H1 is the main alternative to layerwise interleaving. A parallel hybrid guarantees that every
representation transformation can access both recent/exact and compressed/global information, avoiding
a bottleneck where one layer type must repair another several layers later.

The right Speck ablation compares:

- layerwise 3:1 KDA/global attention;
- parallel KDA+attention heads in every block;
- a parameter-matched and a FLOP-matched version of each;
- at least two attention-channel shares, including a very small fraction.

Cache bytes may be larger than expected because attention is present in every block even if only a few
heads are allocated. Cross-layer KV sharing is therefore especially relevant to the parallel design.

## Limitations and cautions

- The architecture report changes tokenizers, data horizons, shape, RoPE, and scale across released
  models. Use the 1.2B internal screen for attribution.
- A small attention **channel** fraction in every layer is not equivalent to a small attention **layer**
  fraction. Kernel launches and per-layer KV metadata can erase theoretical savings.
- The 256K context is a supported/evaluated model property, not proof of uniform hard-task quality at
  every length.
- Deep/narrow improvements must be valued against serial latency on Speck's target device.

## Bottom line

Falcon-H1 makes parallel hybrid heads a serious ablation arm. Its strongest local lesson is that very
little attention width may suffice, but the comparison must include real cache layout and kernel overhead,
not only parameter ratios.
