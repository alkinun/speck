# Model ladder

The rungs of the [model ladder](../../docs/program.md#the-ladder). Each `model.json` is written by
[shapes.py](shapes.py) from one shape rule that also reproduces the 1.2B parent, so the rungs
differ only in scale. Regenerate or verify them with:

```bash
PYTHONPATH=. python experiments/ladder/shapes.py
PYTHONPATH=. python experiments/ladder/shapes.py --check
```

Rung sizes and run counts live in [plan.json](../main-data/plan.json); `make plan-check` fails if a
configuration drifts from the rule or from the plan.

[records/](records) holds each family's predeclared record: question, arms, controls, primary
metric, decision rule and cost, written before the family runs and updated only with its result.
