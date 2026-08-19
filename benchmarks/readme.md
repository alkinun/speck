# benchmarks

results use an rtx 3090, pytorch 2.9.1, cuda 12.8, five warmup steps, and twenty measured steps.

| revision | change | tokens/s | peak memory |
| --- | --- | ---: | ---: |
| `39109de` | baseline | 95,710 | 3.75 gb |
| `0822189` | device batch 16 | 111,930 | 12.02 gb |
| `4980cb3` | fused adamw, rejected | 111,594 | 12.02 gb |
| `bb6d08f` | max autotune | 118,032 | 12.01 gb |
| `9908fb1` | fused cross entropy | 119,813 | 9.97 gb |
| `20c67be` | device batch 32 | 123,892 | 18.97 gb |
| `92b8c7b` | packed qkv, rejected | 124,509 | 20.56 gb |
| `828b4d3` | packed mlp inputs, rejected | 122,298 | 19.08 gb |
| `35d5f0d` | optimized | 124,469 | 18.97 gb |
| `6f62b12` | weight cache, rejected | 118,561 | 11.98 gb |

the optimized end-to-end result is 127,963 tokens/s.

```bash
python -m scripts.benchmark --mode compute --steps 20 --warmup-steps 5 --peak-tflops 142
```
