# 20 — K3 diagnostics and global attention gating

## Questions

Three direct transfers from the literature review were tested:

1. Would Kimi K3's `g_min=-5` decay floor materially change trained Speck KDA behavior?
2. Do Speck's ungated global layers exhibit the attention-sink mechanism reported by gated attention?
3. Does sigmoid gating after SDPA improve KDA/NoPE at fixed parameters and FLOPs?

## Trained KDA decay distribution

Speck currently uses Kimi Linear's unbounded mapping
`g = -exp(A) * softplus(z)`. K3 uses `g = -5 * sigmoid(exp(A) * z)` so reciprocal cumulative
decays remain within BF16 range and dense Tensor Core tiles can replace the slow diagonal path.

| Checkpoint/input | Mean `g` | Minimum | Fraction `< -5` | Fraction `< -80` |
| --- | ---: | ---: | ---: | ---: |
| Base KDA, 4K | −0.3110 | −120.48 | 0.4465% | 0.000205% |
| Base KDA, 32K | −0.3111 | −127.40 | 0.4393% | 0.000130% |
| Post-32K KDA, 32K | −0.3004 | −122.37 | 0.3905% | 0.000110% |

The floor would touch fewer than 0.5% of values but removes a real extreme tail. The first KDA layer
is the outlier, with about 1.5% below `−5`. This is primarily a numerical/kernel opportunity, not a
reason to expect a large quality gain. Speck's present FLA kernel accepts bounded `g`, but K3's speedup
requires its redesigned FlashKDA diagonal-tile path and will not appear from the parameterization alone.

## Attention-sink diagnostic

The diagnostic reconstructs actual normalized and rotated Q/K rows for 32 evenly sampled queries in
the second half of a 32K sequence. It avoids materializing the full quadratic score matrix.

| Checkpoint | First-token mass | Enrichment over uniform | First token is argmax |
| --- | ---: | ---: | ---: |
| Base KDA/NoPE at 32K | 0.0041% | 0.92× | 0.00% |
| Post-32K KDA/NoPE | 0.0170% | 3.69× | 0.10% |
| Post-32K GDN/RoPE | 0.0018% | 0.37× | 0.00% |

KDA develops a small isolated first-token enrichment after continuation, concentrated in the global
layer at depth 7. But the absolute mass and argmax frequency are tiny, and the first 128 tokens are
underweighted overall. Speck does not reproduce the gated-attention paper's extreme sink. A gate may
still add nonlinearity; sink removal is not a strong local rationale.

## Parameter- and FLOP-matched 32M screen

All variants have 153,958,938 parameters and 1.021601280 GFLOP/token:

- ungated uses FFN width 2,304;
- headwise sigmoid projects 768→12 and uses FFN width 2,303;
- elementwise sigmoid projects 768→768 and uses FFN width 2,240.

The FFN reductions exactly pay for each gate projection.

| Variant | Final loss | Versus ungated | tok/s |
| --- | ---: | ---: | ---: |
| Ungated | 3.881081 | — | 44,387 |
| Headwise | **3.867141** | **−0.013940** | 43,086 |
| Elementwise | 3.874083 | −0.006999 | 44,128 |

Headwise beats the large-model paper's preferred elementwise gate at this scale. It improves every
validation source, while elementwise remains inside the old 0.00965-nat seed range. Only headwise was
promoted.

## 131M confirmation

The treatment and existing ungated KDA/NoPE control use identical seed, data order, schedule,
parameters, and analytical FLOPs.

| Variant | Final loss | tok/s | GPU-h |
| --- | ---: | ---: | ---: |
| Ungated | 2.795380 | 44,384 | 0.876 |
| Headwise | **2.793038** | 43,039 | 0.901 |

The gain shrinks to 0.002342 nats, well inside seed variance, while throughput falls 3.0%. At matched
31.98M, 63.96M, 95.94M, and 127.93M milestones, the gate is better by only 0.0009–0.0030 nats.
The short-screen result was schedule-sensitive and did not confirm at the target budget.

## Decision

- Keep attention gating supported and tested in the architecture grammar.
- Do not enable it in the lead KDA/NoPE architecture.
- Do not run elementwise at 131M.
- Treat the K3 decay floor as future kernel work; first add a bounded reference and require numerical
  parity, but do not spend a language-model run expecting a large quality gain.
- Move the experimental budget to exact retrieval with replay and global-cache compression.

## Artifacts

- 32M contracts and result:
  [experiments/SpeckLC-150M-AttentionGate32M](../experiments/SpeckLC-150M-AttentionGate32M),
  [results/SpeckLC-150M-AttentionGate32M](../results/SpeckLC-150M-AttentionGate32M)
- 131M confirmation:
  [experiments/SpeckLC-150M-AttentionGate131M](../experiments/SpeckLC-150M-AttentionGate131M),
  [results/SpeckLC-150M-AttentionGate131M](../results/SpeckLC-150M-AttentionGate131M)
- KDA decay and sink diagnostics:
  [results/SpeckLC-150M-KimiContext32K/diagnostics](../results/SpeckLC-150M-KimiContext32K/diagnostics)
