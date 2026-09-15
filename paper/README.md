# Long-context flagship paper

**Learning to Use Long Context Efficiently: Data and Memory Trade-offs in a 1.2B Hybrid Language Model.**

[claims.json](claims.json) contains three **planned** claims: useful-context training, its interaction
with architecture, and a realized quality-cost frontier. No claim has supported new outcomes.
The [paper contract](../research/flagship/PAPER.md), [study](../research/flagship/STUDY.md) and
[working manuscript](manuscript/main.md) define scope and inference boundaries.

Prior claims/manuscript are byte-preserved in the [transition history](../research/history/README.md).
The existing preparation tables and figures remain historical source evidence, with their original
quotas, generators and receipts. They do not establish new study capacity or long-context performance.

## Regenerate retained preparation assets

```bash
python paper/analysis/preparation.py --check
python paper/analysis/selected_stock.py --check
python paper/analysis/source_capacity.py paper/analysis/source-capacity-v5.json paper/tables/source-capacity-v5 --check
```

`analysis/` contains scripts and immutable input manifests, `figures/` and `tables/` contain generated
assets, `manuscript/` contains the paper, and `references/` contains bibliography assets. New scientific
figures require new checked model/evaluation results and analysis receipts. Preserve every historical
output; do not relabel old quota coverage as new capacity.

Before release, regenerate main results, verify claims and limitations, test inference parity, and
prepare two held-out demonstrations plus a short technical opportunity brief. External publication and
outreach are separate actions; this workspace does not send them automatically.
