# benchmarks

results in this directory are historical and apply only to the model configuration recorded by each result file. rerun benchmarks after architecture or training changes.

## architecture 86 batch geometry

matched end-to-end measurements on an rtx 3090 use packed ultra-fineweb data, 10 warmup updates, and 30 measured updates. both geometries keep the optimizer batch at 16,384 tokens.

| device batch | accumulation | tokens/s | median step | peak allocated |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 40,225 | 0.4074 s | 4.27 gib |
| 8 | 1 | 42,943 | 0.3815 s | 8.64 gib |

batch 8 improves end-to-end throughput by 6.8%. a matched 300-update quality smoke test over 4,915,200 tokens produced final validation losses of 5.587219 at batch 2 and 5.587439 at batch 8, a 0.004% difference.

## smoke test

run a short local benchmark before relying on throughput or memory numbers:

```bash
python -m scripts.benchmark experiments/speck00-200m \
  --mode compute \
  --steps 20 \
  --warmup-steps 5 \
  --peak-tflops 142
```
