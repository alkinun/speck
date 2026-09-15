# Flagship paper

[claims.json](claims.json) connects the paper's four claims to selected contracts and evidence.
The [paper contract](../research/flagship/PAPER.md) defines the question and required outputs.

The [working manuscript](manuscript/main.md) now contains the methods, evidence boundaries, completed
Math L2 preparation, and the planned results structure. Claim states remain pre-results.

Regenerate and verify the checked preparation tables and SVG without optional plotting dependencies:

```bash
python paper/analysis/preparation.py
python paper/analysis/preparation.py --check
python paper/analysis/selected_stock.py
python paper/analysis/selected_stock.py --check
```

The generator verifies [pinned inputs](analysis/preparation-inputs.json), reconciles preparation
attrition, and records generator/input/output hashes in [the asset receipt](analysis/preparation-assets.json).
Source-capacity comparisons match source IDs; unknown supply is explicit, and overlapping banks are
not summed. Outputs currently describe preparation only.

The additive [selected-stock table](tables/selected-stock-v1/first-wave-source-capacity.md) records
the completed FineWiki stock and byte identity between the counting and frozen base tokenizers.
Its [separate receipt](tables/selected-stock-v1/assets.json) preserves the earlier assets and updates
the measurement without summing overlapping banks.

The [current capacity table](tables/source-capacity-v5/source-capacity.md) incorporates the combined
science stock, completed eleven-shard FineMath and five-shard Cosmopedia stocks, and v2 preparation assignments.
Cosmopedia has 1,489,288,743 tokens and a verified cache, exceeding 960M headroom by 529,288,743. FineMath
now has 1,124,167,472 tokens and a verified cache, exceeding its 960M headroom target by 164,167,472.
The original eight-shard text is an exact included prefix. Its renderer accepts a hash-bound input manifest and
requires a new output directory, preserving every earlier table:

```bash
python paper/analysis/source_capacity.py paper/analysis/source-capacity-v5.json paper/tables/source-capacity-v5
python paper/analysis/source_capacity.py paper/analysis/source-capacity-v5.json paper/tables/source-capacity-v5 --check
```

Asset layout:

- `analysis/`: scripts reading checked results and generating tables/figures;
- `figures/` and `tables/`: outputs with source identities and generating commands;
- `manuscript/`: paper and supplement source;
- `references/`: publication bibliography, linked to the [literature library](../research/literature/README.md).

Keep important numbers traceable to results, including archived results. Claim status reflects evidence;
two claims currently have prior evidence and two are planned. Preserve negative and unresolved outcomes.
Validate references with `python -m scripts.research_catalog`.
