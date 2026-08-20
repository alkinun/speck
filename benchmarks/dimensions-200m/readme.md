# 200m dimension sweep

four approximately 50m-parameter llama models train on the same 200,015,872 tokens with seed 42, a 16,384-token batch, muon, and the production learning-rate schedule.

validation uses the same fixed 1,048,576-token sample at 0m, 50m, 100m, 150m, and 200m training tokens. runs do not create checkpoints or report to wandb or hugging face.

| shape | parameters | validation loss | tokens/s | decision |
| --- | ---: | ---: | ---: | --- |
| 6x768 | 50,095,872 | pending | pending | pending |
| 10x640 | 49,984,640 | pending | pending | pending |
| 15x512 | 49,823,232 | pending | pending | pending |
| 24x384 | 50,055,552 | pending | pending | pending |
