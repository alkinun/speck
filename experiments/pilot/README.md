# Engineering pilot

A completed engineering run of the 1.2B parent architecture at 4K context on one H100. Its
configuration is reused elsewhere: the GH200 bundle and throughput packet run
it, and the ladder inherits its tokenizer.

| Result | Receipt |
| --- | --- |
| 800 steps / 104,857,600 tokens; final validation loss 3.379 | [Execution receipt](h100-run.json) |
| 2,619 development tasks; weak base capability | [Development receipt](development-result.json) |
| Checkpoints, exports and grading outputs verified locally | [Backup receipt](backup-result.json) |
| Retained token stock behind the corpus | [Supply receipt](supply.json) |

Scores are on frozen custom development subsets with greedy decoding and no SFT.

## Recipe

800 optimizer steps of 131,072 tokens; FP32 parameters and optimizer state, BF16 activations,
activation checkpointing, Liger loss, Muon/AdamW, deterministic kernels, no compile. Peak learning
rate 3e-4, 40 warmup steps, cosine decay to 10%. The engineering mixture was FineWeb-Edu 50%,
Stack-Edu 15%, FineMath 4+ 15%, Cosmopedia v2 10%, peS2o v3 5% and FineWiki 5%. The preparation
code is at revision `87ddab4d`.

[evaluation.json](evaluation.json) is the pinned five-task development protocol;
[Evaluation](../../docs/evaluation.md#generative-evaluation) describes how to run it.
