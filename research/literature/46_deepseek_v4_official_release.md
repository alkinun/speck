# DeepSeek-V4 official hybrid-attention source audit

## Pinned release

The [initial official DeepSeek-V4-Flash release](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/efc855127ecba8ece36817f9e4cdeeae03b10200)
at commit `efc855127ecba8ece36817f9e4cdeeae03b10200` contains the technical report,
configuration, inference implementation, TileLang kernels, and MIT license. This immutable revision is
aligned to the report rather than a later post-release code update.

The report PDF is 4,479,901 bytes with SHA-256
`fa4a3490e2dcc03c9da61b04a8be471795e9966ebbbf292a3899fa62683a330e`.
The official inference model and kernel hashes are
`6380e70f3690227155ae17fa40b99c8e6900623d00925fd8a8ccc9588e02e818` and
`c4dc859a2208f9182ae00f6267cf8655deba51dcd92dfd14d0e65401658593e4`.

## Architecture factorization

CSA and HCA are separate attention-layer types used in an interleaved schedule, not two parallel
branches in one layer. V4-Flash begins with two pure sliding-window layers and then alternates CSA/HCA;
V4-Pro begins with two HCA layers and then alternates. Every CSA and HCA layer also includes a
128-token raw sliding-window path.

For each layer, the raw local entries and that layer's compressed entries are concatenated into one KV
index space and normalized together by the sparse-attention kernel, with one learned sink logit per
query head. HCA includes every causally completed compressed entry. CSA includes only indexer-selected
completed compressed entries. HCA, CSA, and local coverage therefore interact both within each layer
and through the interleaved depth schedule.

## Exact HCA compressor

HCA is more than a generic block summary. Equations 20–23 project each source hidden state separately
into a KV vector and a channel-wise compression-score vector. A learned position-by-channel bias is
added inside each non-overlapping `m'`-token block; a per-channel softmax over block positions weights
the KV vectors. Each completed pooled entry is RMS-normalized, receives partial RoPE, and serves as
both key and value in MQA. Queries use a low-rank down/up path, and outputs use a two-stage grouped
projection.

The source `m'=128` geometry is not a Speck optimum. A clean isolation must retain block mean and
static scalar/channel controls, then add the exact dynamic channel-gated compressor as its own arm
before selecting a rate.

## Exact CSA compressor and selector

CSA equations 9–12 produce two KV projections and two channel-wise score projections. Each emitted
entry combines `m` positions from the current block's `a` half with `m` positions from the previous
block's `b` half, using a single per-channel softmax over `2m` positions. The preceding half is zero/
negative-infinity padded for the first entry. Entries are emitted every `m` tokens, but after the first,
each has a `2m`-token support overlapping its neighbor.

The Lightning Indexer builds a separate compressed index key with the same overlap rule. A shared
query latent produces multiple indexer heads; ReLU query-key affinities are combined by learned
per-query head weights, then one top-k set of compressed entry indices is shared by the main MQA heads.
CSA therefore selects compressed representations, not raw token blocks. Its unfinished current entry is
excluded; the raw local window supplies current-block causality.

## State and causal behavior in released code

The inference compressor maintains float32 projected-KV and score tails. HCA keeps one `m'`-token
non-overlap tail. CSA keeps both the prior block half and current block half needed for overlapping
entries. A compressed entry becomes visible at the query that completes its current source block;
earlier queries mask it. Completed entries live in a separate compressed cache, while the local window
uses a raw ring.

The released reference supports one full prefill call at `start_pos=0` and one-token incremental calls
afterward. Its nonzero-start paths use `squeeze(1)` and scalar ring writes, so they are not an arbitrary
chunked-prefill implementation. The non-persistent buffers carry no request identity, serialization,
eviction, prefix-hit, or checkpoint/resume contract.

## Correction to Speck's current gates

- HCA v1 correctly requires causal non-overlap and tail state, but its compressor isolation lacks the
  exact source dynamic channel-gated arm. An HCA-without-local experiment must be labeled `HCA core`,
  because the published operator includes local attention.
- CSA v1 must call selected objects overlapping compressed entries, not contiguous raw blocks. HCA is
  held fixed through an interleaved layer schedule; it is not a simultaneous branch in a CSA layer.
- Raw-local v1's deduplication rule is wrong for this source. Raw and compressed entries remain distinct
  even when their source-token supports overlap. Only duplicate exact cache-entry identities may be
  removed. The local window belongs at each selected HCA/CSA layer unless placement itself is the
  intervention.

These corrections do not select V4's rates, schedules, dimensions, positions, indexer, precision, or
local window for Speck.

## Code and license boundary

The official implementation is an inference reference with a top-level `torch.inference_mode` forward.
It contains no training compressor/indexer objective, backward contract, arbitrary chunked prefill,
prefix-cache identity, or isolated parity suite. The TileLang sparse kernel demonstrates one normalized
union with a sink but is hardware-specific and unqualified locally. The release is MIT licensed.

Only append-only readiness corrections are authorized now. Local HCA/CSA/raw-local implementation,
training, and promotion remain blocked by the active finalist and parent gates.
