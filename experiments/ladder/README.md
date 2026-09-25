# Model ladder

The rungs of the [model ladder](../../docs/program.md#the-ladder). Each `model.json` is written by
[shapes.py](shapes.py) from one shape rule that also reproduces the 1.2B parent, so the rungs
differ only in scale. Regenerate or verify them with:

```bash
PYTHONPATH=. python experiments/ladder/shapes.py
PYTHONPATH=. python experiments/ladder/shapes.py --check
```

Every ladder corpus schedules sources per sequence (see [Training](../../docs/training.md#base-training)),
so a run's data does not depend on its device batch size. [train.json](train.json) is the training
recipe every rung shares; a rung's own `train.json` adds its
device batch size, run name and token budget, and its learning-rate sweep sets `lr`. Rung sizes and
run counts live in [plan.json](../main-data/plan.json); `make plan-check` fails if a
configuration drifts from the rule or from the plan.

[records/](records) holds each family's predeclared record: question, arms, controls, primary
metric, decision rule, cost and prerequisites. A record stays a `draft` until its prerequisites are
met, is declared before its family runs, and is then updated only with its result.
