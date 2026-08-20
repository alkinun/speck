# benchmarks

results in this directory apply only to `speck00-200m`.

## smoke test

an rtx 3090 compiled training step with device batch 2, sequence length 2048, and accumulation 4 reaches 35,727 synthetic tokens/s and 35,916 end-to-end tokens/s with 5.04 gb peak allocated memory.

```bash
python -m scripts.benchmark --mode compute --steps 20 --warmup-steps 5 --peak-tflops 142
```
