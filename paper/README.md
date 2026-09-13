# Flagship paper

[claims.json](claims.json) connects the paper's four claims to selected contracts and evidence.
The [paper contract](../research/flagship/PAPER.md) defines the question and required outputs.

Add assets as they are implemented:

- `analysis/`: scripts reading checked results and generating tables/figures;
- `figures/` and `tables/`: outputs with source identities and generating commands;
- `manuscript/`: paper and supplement source;
- `references/`: publication bibliography, linked to the [literature library](../research/literature/README.md).

Keep important numbers traceable to results, including archived results. Claim status reflects evidence;
two claims currently have prior evidence and two are planned. Preserve negative and unresolved outcomes.
Validate references with `python -m scripts.research_catalog`.
