# DeepSeek-V4.1-Flash: asymmetric compute and compressed global memory

## Pinned release

- Model repository: [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- Audited revision: `df42c109f1defefcbfcedbe7d905718a12266e40`
- Technical report: [`DeepSeek_V41_Tech_Report.pdf`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/DeepSeek_V41_Tech_Report.pdf)
- Report bytes: `1,809,802`
- Report SHA-256: `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`
- License: MIT for the released repository and weights

The report describes 552B logical backbone parameters plus 196B Engram conditional-memory
parameters. The Hub API counts about 484.6B stored tensor elements because routed-expert FP4 values
are packed; this is not the logical model size. The release artifact occupies about 510 GB. “Flash”
therefore means low active computation and cache cost rather than a small total checkpoint.

## Reported architecture

The language backbone is a 40-layer causal stack split into a 20-layer causal encoder and a 20-layer
decoder. It activates about 8B parameters per prompt token during prefill and 16B per generated token
during decode. For long prompts, the encoder processes the complete sequence, decoder global memory
is derived from the final encoder representation, and only a bounded recent segment is replayed through
the decoder to construct local sliding-window state.

Every layer retains a 128-token local sliding-window path. The first two layers are local-only. The
remaining layers use Compressed Sparse Attention 2 (CSA2), which separates three roles:

- **Full:** produce main KV and indexer K, and compute a new Top-K selection;
- **Reindex:** reuse main KV and indexer K but compute a fresh depth-specific selection;
- **Reuse:** reuse both the representation and the latest selection.

The released configuration writes main global KV at layers `2, 8, 14, 20` and refreshes sparse
selection at layers `2, 8, 14, 20, 24, 28, 32, 36`. Encoder global entries compress two tokens into
one; decoder entries use ratio one. The decoder's first Full layer builds a candidate pool of at most
2,048 eight-position blocks, and later Reindex layers select Top-512 entries only within that bounded
pool. The first decoder index remains linear in context length; deeper indexers become bounded.

Main global KV uses E2M1 FP4 with one E4M3 scale per 16 channels after RoPE. Local SWA KV remains
FP8. The report attributes a global-cache footprint of 890 bytes per token to cross-layer reuse,
sequence compression, latent entry size, and precision together. SWA Bounded Replay omits local state
from long-lived persistent storage and approximately reconstructs it from only the most recent window,
reducing reported host/SSD persistent state to about one eighth of DeepSeek-V4-Flash.

Additional components are Single-Pass mHC residual mixing, 196B parameters of deterministic n-gram
Engram lookup memory, 384 routed plus one shared expert per MoE layer with six routed experts active,
and a separately trained DSpark speculative decoder. Training uses head-wise Muon for query/key
matrices, ordinary Muon for other backbone matrices, AdamW for non-matrix parameters, and a
momentum-plus-Sinkhorn update for large embeddings and prediction tables.

## Training and post-training evidence

The model is trained on 45T multimodal tokens with a 100.6M-token batch. Sparse attention is active
from scratch at 64K; context extends to 1M at 34T tokens. The learning rate warms for 2,000 steps,
holds at `2.6e-4` through 28T, decays to `2.6e-5` through 40T, then remains constant through 45T.

The report describes post-training as conventional SFT, RL, and on-policy distillation. It attributes
the substantive improvement to automated construction of verifiable tasks and environments,
difficulty calibration, failure-driven data, heterogeneous agent scaffolds, and more than forty
teachers—not a new RL algorithm. A learned effort scalar changes the output-token penalty and exposes
a deployment-time quality/cost frontier.

Reported agent results at maximum effort are unusually strong, including 74.2 on DeepSWE v1.1 and
90.6 on Terminal-Bench 2.1. They are official same-release claims rather than independent
reproductions. The capability profile is not uniformly dominant: V4-Pro remains materially better on
some knowledge, math, and long-context evaluations, and maximum effort often spends roughly 2.5 times
the output tokens of low effort.

## Released-code boundary

The public `inference/` directory is explicitly a readable reference rather than the production
serving system. It implements CSA2 cache sharing, hierarchical selection, local attention, Engram,
MoE, mHC, vision, quantized kernels, and the DSpark forward path. Its simple Transformer loop still
processes all backbone layers during ordinary prefill; production CED skipping and bounded replay are
described in the report but not reproduced by that loop. The DSpark confidence forward exists, while
the confidence-scheduled speculative generation engine is out of scope.

The reported deployment additionally relies on fused and distributed systems such as FlashMLA,
DeepGEMM, DeepSelect, TileKernels, Mega-mHC, encoder/prefill/decode disaggregation, RDMA Engram
access, and persistent cache infrastructure. The release supports architectural inspection and weight
use, not reproduction of the 45T training run or complete production economics.

## Speck interpretation

The transferable result is a resource taxonomy rather than a grant-1 operator:

1. Sequence state factors across entry width, retained positions, distinct layer caches, precision,
   and persistence policy.
2. Representation refresh, selection refresh, and attention reads are separate depth roles.
3. Prompt ingestion and token generation need not activate the same depth.
4. Recompute can dominate persistence when approximate recovery is trained and measured.
5. Output tokens are an efficiency axis alongside training FLOPs, state, and latency.

Speck Reader Attention independently found that fresh queries cannot always compensate for a stale
global representation: distance-one readers retained capability while a distance-four reader failed.
V4.1's periodic Full layers, more frequent Reindex layers, and exact local path are consistent with
that boundary. This strengthens the mechanism finding without reversing Reader Attention's failed
promotion gate.

Grant 1 should not add CED, CSA2, mHC, Engram, DSpark, multimodality, or MoE. It should instead:

- state memory claims against matched dense GQA rather than all architectures;
- report prefill, decode, runtime HBM state, persistent state, and output-token cost separately;
- preserve the 1.2B dense-width KDA/global flagship;
- keep asymmetric compute and periodically refreshed compressed memory as unprioritized background;
- optionally test FP4 global-KV storage on existing checkpoints without making it a launch gate.

DeepSeek reports about 890 global bytes per token, while the planned 1.2B Speck geometry uses about
12,288 BF16 or 6,240 INT8-plus-scale global bytes per token, plus 19.3 MB of fixed KDA state. Speck's
defensible advantage is therefore not globally minimal bytes per token. It is a roughly fourfold
reduction against matched dense GQA at a radically smaller total model footprint that can be measured
on accessible hardware.
