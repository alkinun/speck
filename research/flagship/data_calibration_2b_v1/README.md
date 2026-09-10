# Flagship 2B production-data calibration v1

This is a time-bounded replacement for running the complete 20B operations rehearsal before D5 and
E3. It executes the same real source, filtering, contamination, security, exact/near deduplication,
packing, interruption/resume, cleanup, and firewall-disjointness path at one tenth of the former token
target:

| Category | Source | Packed-token target |
| --- | --- | ---: |
| Web | Ultra-FineWeb English v1.4 | 1.1B |
| Code | Common Pile Stack v2 educational code | 300M |
| Math | FineMath 4+ | 200M |
| Synthetic | Cosmopedia v2 | 200M |
| Science | Common Pile PubMed | 100M |
| Reference | FineWiki English | 100M |

The calibration uses the Mistral fallback tokenizer and a 5% heldout-candidate partition. It measures
real throughput, disk, SQLite index size, peak RSS, cleanup, and resume, then prints a conservative
20B linear byte projection. It cannot issue the `speck_production_data_operations_qualification`
record and cannot authorize training. Final corpus preparation after D5/E3 must confirm source
capacity and any non-linear memory/runtime behavior.

The paused 20B v1 attempt remains under `/mnt/speck-data/speck/data-rehearsal-20b-v1` with its last
verified source-file checkpoint. It can be resumed later if calibration evidence or final-corpus risk
requires it.

Run or resume:

```bash
speck_base_dir=/mnt/speck-data/speck \
  uv run --extra cpu python -m scripts.production_calibration \
  research/flagship/data_calibration_2b_v1/production_plan.json
```
