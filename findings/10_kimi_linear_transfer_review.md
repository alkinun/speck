# 10 — Kimi Linear transfer review and revised experiment order

## Status and sources

This is a literature-to-experiment review, not a Speck model result. It was completed on
2026-09-03 from version 2 of the
[Kimi Linear paper](https://arxiv.org/pdf/2510.26692), the
[official Kimi Linear repository](https://github.com/MoonshotAI/Kimi-Linear), and the
[open KDA implementation in FLA](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda).

The paper is unusually relevant because it studies almost the same high-level family as our
current frontier: mostly finite-state delta-rule layers, periodic global attention, 4K base
pretraining, and later context activation. Its evidence should change our experiment order, but
its conclusions are not Speck results until they survive our scale, data, and noise controls.

## What the paper proposes

Kimi Delta Attention (KDA) replaces Gated DeltaNet's one scalar decay per head and token with a
separate decay for every key channel. The recurrent state remains a fixed `d_k × d_v` matrix per
value head. Its update is a constrained diagonal-plus-low-rank transition: channel-wise decay is
followed by the delta-rule correction. The paper supplies a specialized chunkwise algorithm rather
than paying the cost of a general DPLR recurrence.

The complete Kimi Linear mixer has three other important choices:

- three KDA layers followed by one global MLA layer, repeated uniformly;
- sigmoid output gating after per-head RMSNorm;
- no positional encoding in global attention; KDA's learned recurrence is responsible for
  position and recency.

Short convolutions remain inside KDA's Q/K/V parameterization. This does not contradict our
decision to retire the standalone gated-convolution hybrid: the paper ablates a small local
convolution inside a delta-rule layer, not convolution as the primary sequence mixer.

## Evidence reported by Kimi

### Component ablation

The paper's first scaling-law model has 16 layers and 16 heads. The following points were trained
under the same stated FLOPs budget and hyperparameters:

| Variant | Train PPL | Validation PPL |
| --- | ---: | ---: |
| KDA:MLA `3:1` | 9.23 | 5.65 |
| full MLA `0:1` | 9.45 | 5.77 |
| KDA:MLA `1:1` | 9.29 | 5.66 |
| KDA:MLA `7:1` | 9.23 | 5.70 |
| KDA:MLA `15:1` | 9.34 | 5.82 |
| `3:1`, no output gate | 9.25 | 5.67 |
| `3:1`, SiLU output gate | 9.43 | 5.81 |
| `3:1`, no short convolution | 9.29 | 5.70 |

The SiLU-versus-sigmoid result is directly relevant to Speck. Our current GDN applies SiLU to its
output gate, while the paper applies sigmoid to both KDA and its GDN-H baseline. Consequently, a
naive Speck GDN-versus-KDA comparison would change both decay granularity and output gating.

### Memory tests

On two-layer, two-head, head-dimension-128 models, the authors train Palindrome, multi-query
associative recall (MQAR), and 64-stack state tracking at lengths 256 through 2,048. KDA has the
best reported accuracy and converges faster than GDN on the copying and recall tasks. This is a
better mixer qualification suite than our current single-query passkey diagnostic because it
separates exact copying, many simultaneous associations, and mutable state.

### Matched large-model results

The main comparison uses 48B-total/3B-activated MoE models, 1.4T shared pretraining tokens, a
4,096-token base context, and the same stated optimizer and continuation recipe. Kimi Linear beats
full MLA and hybrid GDN-H on most reported short-context tasks. At 128K, the paper reports:

| Model | RULER | MRCR | HELMET-ICL | RepoQA | Reported mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLA | 81.3 | 22.6 | 88.0 | 63.0 | 52.2 |
| GDN-H | 80.5 | 23.9 | 85.5 | 63.0 | 51.2 |
| Kimi Linear with RoPE | 78.8 | 22.0 | 88.0 | 66.5 | 51.8 |
| Kimi Linear with NoPE | 84.3 | 29.6 | 90.0 | 68.5 | 54.5 |

The mean also contains LongBench V2, Frames, and two Long Code Arena subsets. NoPE is not a cosmetic
simplification in these results: the RoPE KDA hybrid is worse than full MLA on the reported mean,
while the NoPE version is best.

The fixed KDA state enables the systems result. The paper reports near-equal KDA and GDN-H prefill
latency, a 2.9× hybrid-over-MLA prefill speedup at 1M, and up to 6.3× decoding throughput at 1M when
the saved KV memory permits a larger batch.

## Where Speck already agrees

1. **Pure recurrence is not enough.** The paper explicitly retains global attention for exact
   retrieval. Our pure-GDN loss and retrieval results independently reach the same conclusion.
2. **A 3:1 layerwise hybrid is credible.** Kimi selects the same 15-recurrent/5-global count as our
   original 20-layer hybrid. Our `global-5` point also has the best measured 32K loss and retrieval
   retention, although at substantial state cost.
3. **Layerwise mixing is operationally clean.** It matches Speck's existing architecture grammar,
   cache accounting, and global-layer promotion machinery.
4. **The recurrent state is the position-aware component.** This directly addresses our measured
   RoPE extension asymmetry and the `global-5` short-context erosion under 8× scaling.
5. **Internal short convolution remains useful.** Speck already uses kernel-size-4 depthwise
   convolution inside every GDN layer, matching the component the paper finds beneficial.

## What is new for Speck

| Axis | Current Speck GDN hybrid | Kimi Linear | Required isolation |
| --- | --- | --- | --- |
| recurrent decay | scalar per value head | channel-wise per key dimension | GDN-sigmoid vs KDA-sigmoid |
| output gate | full-rank SiLU | low-rank sigmoid | GDN-SiLU vs GDN-sigmoid |
| global positions | partial RoPE, scaled on promotion | NoPE | same-parent RoPE vs NoPE |
| recurrent head dimension | 64 | 128 | do not assume scale transfer |
| global operator | GQA | MLA | keep GQA fixed for the first isolation |
| training scale | 150M dense, 131M base tokens | 3B activated MoE, 1.4T comparison tokens | replicate at Speck scale |

The installed CUDA environment already contains `fla.ops.kda` in
`flash-linear-attention==0.5.0`, including chunkwise training and fused recurrent kernels with
grouped value-head support. KDA therefore does not require us to invent a kernel. It still requires
an auditable Torch reference, forward/backward parity tests, parameter/FLOPs accounting, state and
cache integration, and a pinned API contract around FLA.

### Local kernel feasibility probe

On 2026-09-03, the installed chunkwise KDA kernel was compared with a direct GDN recurrence by
repeating one scalar log-decay across every KDA channel. The shape was batch 1, length 64, 2 key
heads, 4 value heads, and 32-dimensional key/value heads in float32 on the RTX 3090.

Under the model's intended Q/K L2 normalization and `1/sqrt(32)` query scale:

- maximum/mean absolute output error: `4.9952e-4` / `8.1196e-5`;
- maximum/mean absolute final-state error: `2.5166e-3` / `4.8415e-4`;
- combined test loss: `0.2432141`;
- gradients for Q, K, V, decay, and beta were all finite.

An earlier deliberate stress probe used unnormalized Gaussian Q/K and query scale 1.0. The
recurrence became unstable, produced infinite loss, and had non-finite gradients. That probe is not
representative of the Kimi or Speck parameterization; it confirms why Q/K normalization must be a
tested invariant. The successful result proves kernel availability and the scalar-decay reduction
at one small shape only. It does not replace full-precision, bf16, decode-cache, or production-shape
qualification.

Machine-readable record:
[KDA kernel smoke result](../results/KimiLinearTransfer/kda_kernel_smoke.json).

## What should not be imported as fact

- The paper reports no seed intervals for the architecture comparisons. We retain the measured
  `0.00965`-nat Speck screening range and require repeat seeds for promoted claims.
- The long-context activation data and schedule are referred to through the proprietary K2 recipe,
  not specified well enough to reproduce. Architecture cannot substitute for genuine
  long-dependency supervision.
- The main model is about twenty times larger in activated parameters and sees more than ten
  thousand times our base-stage tokens. Its optimal global ratio may not be ours.
- The paper ablates global-layer count but not placement. It does not test our observed distinction
  between middle integration and final retrieval/readout layers.
- Its MLA cache and projection geometry differ from Speck GQA. The reported 75% cache reduction is
  a ratio for their architecture, not our byte count.
- The synthetic plots use the best learning rate from a grid. Our replication must preserve every
  attempted learning rate and seed, not only the winning curve.

## Revised experiment order

### K0 — Implementation qualification, no pretraining

1. Add an explicit output-gate activation to the GDN specification, defaulting to current SiLU so
   existing checkpoints and configs remain unchanged.
2. Add KDA as a distinct operation kind. Implement a slow Torch recurrence and bind CUDA execution
   to the installed FLA chunk and recurrent kernels.
3. Prove that KDA with an identical decay in every channel reduces numerically to GDN. Test forward,
   final-state, cached decode, backward, serialization, parameter count, state bytes, and FLOPs.
4. Extend context-stage promotion with an explicit global `rope_dim`; `0` must produce NoPE without
   touching sliding layers.
5. Pin and qualify the exact FLA KDA API. We will sigmoid `beta` outside the kernel and will not rely
   on silently accepted keyword arguments.

No language-model training starts until these checks pass.

### K1 — Same-parent NoPE control

Run one new `global-5-nope` 32K continuation from the exact original `gdn-local` parent used by the
completed frontier:

- global layers: 3, 7, 11, 15, 19;
- global `rope_dim=0`, sliding layers unchanged;
- same 32M tokens, data manifest, optimizer state, seed, and schedule as existing `global-5`;
- compare initial/final 32K loss, original 4K regression, paired retrieval curve, throughput, and
  resident state against the completed RoPE `global-5` control.

This is the cheapest direct test of the paper's strongest long-context finding. Because position
encoding changes only at the context branch, it measures continuation adaptability, not the final
answer for from-scratch NoPE pretraining.

Outcome: complete. NoPE greatly strengthened paired retrieval sensitivity through 128K but ended
worse on both 32K and original-4K language loss. See
[12 — Same-parent NoPE context activation](12_nope_context_activation.md).

### K2 — Synthetic memory factorization

Build deterministic Palindrome, MQAR, and 64-stack generators following the paper's public task
definitions. Compare:

1. GDN with SiLU output gate;
2. GDN with sigmoid output gate;
3. KDA with sigmoid output gate.

Use the paper's two-layer/two-head/head-dimension-128 geometry, lengths 256, 512, 1,024, and 2,048,
and declared learning-rate grid. Use a one-seed sweep only to select viable learning rates, then
repeat the selected configuration over three fixed seeds. Report all runs. This separates the gate
activation effect from the channel-wise decay effect before any expensive language-model screen.

Synthetic outcome: complete. KDA-sigmoid and GDN-SiLU pass 3/3 on calibrated MQAR, but KDA is more
reliable at replicated length-2,048 endpoints. On Palindrome, KDA passes 2/3 and GDN 0/3. Both pass
Stack 3/3 with tied medians. See [13 — Synthetic MQAR](13_synthetic_mqar.md),
[14 — MQAR length scaling](14_mqar_length_scaling.md), and
[15 — Palindrome and Stack](15_palindrome_and_stack.md).

### K3 — 4K language-model staircase

Use the same 20-layer `3:1` layout, tokenizer, packed data order, sequence length, Muon recipe, and
131,072,000-token budget. Compare each row only with the preceding row so every step has one primary
architectural intervention:

| Step | Recurrent layer | Output gate | Global position encoding |
| --- | --- | --- | --- |
| existing control | GDN | SiLU | RoPE |
| gate control | GDN | sigmoid | RoPE |
| position control | GDN | sigmoid | NoPE |
| KDA candidate | KDA | sigmoid | NoPE |

Keep the recurrent and global head geometry fixed initially. Record both exact parameter count and
corrected training FLOPs; compensate in the FFN only if KDA changes the total enough to make the
comparison material. A discovery seed may screen the staircase, but any promoted difference must
be repeated across three seeds. Add the crossed KDA-with-RoPE cell only if the NoPE and KDA effects
appear non-additive.

### K4 — Re-open the 32K frontier with the winning mixer

Only after K2 and K3 pass, train the winning recurrent mixer with:

- five uniformly distributed NoPE global layers, matching the paper's ratio;
- the two-layer middle-plus-final design suggested by our placement experiment;
- a final-only layer as the minimum-state retrieval control.

Rank these on seed-repeated long loss, original-4K retention, MQAR-style retrieval, state bytes,
prefill FLOPs, prefill latency, and decode latency. The purpose is to learn whether better recurrent
memory lowers the number of global layers needed, not merely to copy `3:1`.

### K5 — Data and independent evaluation gate before 128K

Do not launch 128K continuation until we have training examples with genuine 64K–128K dependencies
and an inference adapter that can run pinned RULER, NoLiMa, and HELMET suites. Add MRCR, RepoQA, and
repository-scale code evaluation where license and model interface permit. Continue to treat our
counterfactual diagnostic as an internal causal-sensitivity test, not a public capability score.

## Immediate decision

Do not spend the next GPU hours replicating the old GDN-SiLU/RoPE frontier. First qualify KDA and
sigmoid gating, then run the single same-parent NoPE control. If those effects reproduce, the old
frontier becomes a baseline rather than the architecture to replicate.

There is also an operational storage gate: only about 4.5 GiB is currently free, while one complete
150M continuation checkpoint uses about 1.3 GiB. Existing checkpoints remain untouched. Before a
multi-run sweep, provision additional space or define an explicit checkpoint-retention policy; one
NoPE control still fits safely.
