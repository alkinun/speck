# Model ladder

The rungs of the [model ladder](../../docs/program.md#the-ladder). Each `model.json` is written by
[shapes.py](shapes.py) from one shape rule that also reproduces the 1.2B reference, so the rungs
differ only in scale. Regenerate or verify them with:

```bash
PYTHONPATH=. python experiments/ladder/shapes.py
PYTHONPATH=. python experiments/ladder/shapes.py --check
```

Rung sizes and run counts live in [plan.json](../main-data/plan.json); `make plan-check` fails if a
configuration drifts from the rule or from the plan.
