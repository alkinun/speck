# benchmarks

results in this directory are historical and apply only to the model configuration recorded by each result file. rerun benchmarks after architecture or training changes.

## smoke test

run a short local benchmark before relying on throughput or memory numbers:

```bash
python -m scripts.benchmark experiments/speck00-200m \
  --mode compute \
  --steps 20 \
  --warmup-steps 5 \
  --peak-tflops 142
```
