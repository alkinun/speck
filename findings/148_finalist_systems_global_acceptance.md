# 148 — Global six-block systems acceptance qualifies synthetically

## Write-once-style file graph validation

Before analysis, every block now rehashes its two referenced trial files and requires their decoded bytes
to equal the trials embedded in the block. Frozen block/order/pair/role/position/arm identities, complete
software keys, paired batch SHA, failure reason, and no-retry state are mandatory.

A retained failed block cannot mask corruption elsewhere. Every present complete block is fully checked
before the validator returns the missing/failed no-claim state.

## Twelve-trial identity boundary

For a complete set, git commit, Python, PyTorch, CUDA runtime/driver, FLA, Triton, and engine hash must be
one value across all twelve trials. Model config must be one value within each arm and differ across arms.
Only after all six blocks pass does the validator invoke the frozen systems analyzer.

Seven synthetic adversaries cover the complete joint path, file drift, global software drift, invalid
model-role identity, failed and missing blocks, and the masking attack. Runtime identity production,
orchestration, automatic failure artifacts, live pipeline qualification, and execution remain blocked.

## Artifact

- [Global acceptance qualification](../results/Speck-Paper1/finalist-systems-acceptance-qualified-v1.json)
