# 02 — Correctness and kernel calibration

## Defects found

### Sliding attention was slower than global attention

The former cacheless sliding path passed an explicit boolean mask to SDPA. That disqualified the
fast causal FlashAttention path. Its 2,048-token query chunks were also poorly matched to a
2,048-token window: the second chunk at sequence length 4,096 could span 4,095 keys. Realized score
area was about 12.6M entries, compared with about 8.4M for global causal attention and 6.3M for an
ideal local window.

An empty K/V prefix was concatenated on cacheless sliding calls, allocating and retaining needless
full K and V tensors. Decode itself already used a correct window-bounded ring buffer.

### FLOPs accounting charged the wrong context

The original formula used full sequence length for global attention and `min(length, window)` for
sliding attention. A causal kernel instead pays the mean attended row length.

For length `L` and effective context `C=min(L,W)`, the mean is:

```text
[C(C+1)/2 + (L-C)C] / L
```

For global causal attention, `C=L`, giving `(L+1)/2`. Tests now compare the analytic value directly
with the exact causal/window predicate.

| Variant | Recorded F/token | Corrected F/token | Recorded matched tokens | Corrected matched tokens |
| --- | ---: | ---: | ---: | ---: |
| `full-global` | 1.697710 G | 1.320315 G | 85,989,910 | 101,202,716 |
| `full-local` | 1.320223 G | 1.225897 G | 110,576,750 | 108,997,294 |
| `gdn-global` | 1.113784 G | 1.019436 G | 131,072,000 | 131,072,000 |
| `gdn-local` | 1.019412 G | 0.995831 G | 143,205,955 | 134,178,838 |
| `conv-global` | 1.105075 G | 1.010726 G | 132,104,981 | 132,201,407 |
| `pure-gdn` | 0.919142 G | 0.919142 G | 158,828,423 | 145,374,049 |

Both checked sweep ledgers were updated and are covered by a test that rebuilds every model on the
meta device and recomputes the entries.

## Implemented kernel path

- CUDA cacheless/initial prefill uses compiled FlexAttention with a block mask.
- Stateful decode or unsupported small test heads retain the readable SDPA reference.
- Forward and backward parity are checked against masked SDPA with GQA and prefix offsets.
- Empty cache prefixes bypass K/V concatenation.
- Sliding and global operations use separate rotary modules in mixed models.
- Global RoPE scaling does not compress sliding-window relative distances.

PyTorch's generic `create_block_mask` first materializes the token-level boolean relation. At 128K
that temporary would exceed 16 GiB. The replacement constructs exact block membership directly.
FlexAttention requires full supported block-index widths for all compiled model layouts; an initial
compressed-width implementation passed simple parity but caused a CUDA illegal access at a
non-block-aligned 4,095-token prefill. That implementation was rejected and a model-level boundary
regression test was added. The supported 128K block metadata occupies about 16 MiB.

## Rejected optimization

Marking the causal-window mask as `ROWS_GUARANTEED_SAFE` and `BLOCKS_ARE_CONTIGUOUS` improved an
isolated Flex kernel by about 3.5%. The real compiled 20-layer training step then produced a
non-finite loss. The flags were removed. Isolated microbenchmark improvements are not accepted
without full-model training validation.

## Corrected 4K preflight

| Variant | Device batch | tok/s | Peak allocated |
| --- | ---: | ---: | ---: |
| `conv-global` | 8 | 56,702.3 | 20,570 MiB |
| `pure-gdn` | 4 | 51,958.0 | 13,679 MiB |
| `gdn-global` | 4 | 48,535.8 | 13,243 MiB |
| `gdn-local` | 4 | 47,721.2 | 13,245 MiB |
| `full-global` | 4 | 41,273.1 | 11,907 MiB |
| `full-local` | 4 | 40,306.8 | 11,911 MiB |

Relative to the old preflight, `gdn-local` improved 1.47× and `full-local` improved 2.53×. Their
safe microbatches increased from 2 and 1 respectively to 4. Local did not become faster than global
at 4K: with `L=2W`, the modest score reduction roughly balances Flex overhead. The expected local
advantage appears only when `L >> W`.

## Environment repair

The project venv initially contained `torch 2.9.1+cpu`. It was restored with the `gpu` and `linear`
extras to PyTorch 2.9.1+cu128 and FLA 0.5.0. During the later frontier, regenerable temporary
Inductor artifacts and the UV download/build cache were cleared for disk headroom. No checkpoint,
dataset, or result was deleted; those caches must be regenerated or redownloaded if needed.

## Artifacts and commits

- Raw preflight: [results/hardware/rtx3090-flexattention-preflight](../results/hardware/rtx3090-flexattention-preflight)
- `4ee5ea4` — corrected causal attention FLOPs
- `7d3ded8` — FlexAttention sliding prefill
- `35105d6` — empty K/V concat removal
- `80e7e7c` — corrected preflight
- `08057db` — corrected sweep ledgers
- `f015c0d`, `e88ee32` — scalable then supported block-mask construction
