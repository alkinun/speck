# benchmarks

results use an rtx 3090, pytorch 2.9.1, cuda 12.8, five warmup steps, and twenty measured steps.

| revision | change | tokens/s | peak memory |
| --- | --- | ---: | ---: |
| `39109de` | baseline | 95,710 | 3.75 gb |
| `0822189` | device batch 16 | 111,930 | 12.02 gb |

```bash
python -m scripts.benchmark --mode compute --steps 20 --warmup-steps 5 --peak-tflops 142
```
