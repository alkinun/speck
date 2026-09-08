# R11 scale-target accounting

Status: complete planning geometry, not launchable. The compact source contract is
[`scale-targets-v1.json`](scale-targets-v1.json); deterministic expanded accounting is
[`accounting-v1.json`](accounting-v1.json). Check it with:

```bash
python -m scripts.scale_targets --check
```

| Target | Geometry (E/H, depth, FFN) | Recurrent/global | Total = active parameters | 4K analytic FLOPs/token | 6ND (separate) |
| --- | --- | ---: | ---: | ---: | ---: |
| 60M | 512/512, 16, 1408 | 12/4 | 60,561,712 | 415,552,512 | 363,370,272 |
| 150M | 640/768, 20, 2304 | 15/5 | 153,960,858 | 1,021,612,800 | 923,765,148 |
| 220M | 1024/1024, 16, 3072 | 12/4 | 220,786,016 | 1,429,088,256 | 1,324,716,096 |
| 350M | 1152/1152, 20, 3200 | 15/5 | 351,030,008 | 2,262,614,016 | 2,106,180,048 |
| 600M flagship option | 1536/1536, 20, 3968 | 15/5 | 592,344,884 | 3,765,178,368 | 3,554,069,304 |
| 750M | 1536/1536, 24, 4352 | 18/6 | 743,449,560 | 4,714,030,080 | 4,460,697,360 |
| 1.2B flagship option | 2048/2048, 24, 5120 | 18/6 | 1,195,884,576 | 7,513,092,096 | 7,175,307,456 |

All totals use the explicit D5 fallback: Mistral's 32,000 pieces plus three reserved chat roles,
or 32,003 rows. D5 has not selected a tokenizer. The artifact prices every active v2 candidate at
every width. It reports both the planned **untied** embedding-plus-head cost (`2*V*E`) and current
instantiated **shared** storage (`V*E`); exact target totals follow the latter. Untying the head is a
known launch-freeze integration decision and must not be hidden by calling logical matrices physical
parameters.

The parameter equations independently count every projection, normalization vector, convolution,
decay parameter, adapter, and shared embedding/head, then compare with models instantiated on the
PyTorch meta device. Analytic training cost follows the repository implementation:

`6 * linear-projection MACs + KDA rule FLOPs + causal global-attention FLOPs`.

The conventional `6ND` value is only a separately printed comparator. It is never added to or used
in place of analytic cost. State records separate fixed FP32 recurrent matrices and BF16 convolution
history from length-growing global KV, with exact BF16 and int8-plus-FP16-scale reports at 4K and
128K. Optimizer estimates use exact Muon/AdamW membership and state their precision assumptions.

These files contain no data, tokenizer artifact, optimizer choice, learning rate, batch, schedule,
seed, or launch command. Every future experiment still requires D5, architecture and analysis
freezes, an authorized data manifest, and `train.requires_data_launch_authority=true` with a valid
data-launch receipt.
