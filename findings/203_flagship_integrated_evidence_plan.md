# 203 — The flagship now validates transfer and assembly before scale

On 2026-09-09 the pre-results flagship plan retired E5 curriculum shape and reassigned its 100
GH200 GPU-hours to integrated validation. No E5 output exists. The immutable predecessor remains
[`data_plan.json`](../research/flagship/data_plan.json), SHA-256
`9adda80ed0c53f2eef376191223c8df8e444f448cf05c6326432a5b283fa4363`; the active successor is
[`data_plan_v2.json`](../research/flagship/data_plan_v2.json).

The replacement asks two questions that directly gate the paper and release. I1 is a three-seed 2×2
comparison crossing dense/C0-hybrid architectures with balanced-prior/E2-selected data at 150M/3B.
It reports data and architecture main effects plus their interaction without reopening E2. I2 compares
one assembled set of every compatible individually promoted D2/D3/D7/D8 setting against exact C0 at
350M/10B and three seeds. If the assembly fails, the complete C0 default launches; no post-results
subset search is allowed. I3 reserves the remaining five hours for integrated capability and systems
analysis. The scale ladder begins only after I2 resolves.

The paper's claim and narrative contract is now explicit in
[`PAPER.md`](../research/flagship/PAPER.md). Every active experiment must select a release input, test
transfer or composition, validate scale/length transfer, or price the released system. MoBA, MLA,
sparse/compressed attention, MoE, depth routing, and unrelated exploration remain outside grant 1.

This is a planning decision, not experimental evidence or training authority. Data rights, real 20B
rehearsal, tokenizer D5, GH200 qualification, per-arm manifests, and every existing launch gate remain
open.
