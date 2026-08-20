# 200m dimension sweep

four approximately 50m-parameter llama models train on the same 200,015,872 tokens with seed 42, a 16,384-token batch, muon, and the production learning-rate schedule.

validation uses the same fixed 1,048,576-token sample at 0m, 50m, 100m, 150m, and 200m training tokens. runs do not create checkpoints or report to wandb or hugging face.

| shape | parameters | validation loss | tokens/s | decision |
| --- | ---: | ---: | ---: | --- |
| 6x768 | 50,095,872 | 3.65317 | 130,126 | reject |
| 10x640 | 49,984,640 | 3.62202 | 113,662 | reject |
| 15x512 | 49,823,232 | 3.61270 | 107,563 | runner-up |
| 24x384 | 50,055,552 | 3.61347 | 89,714 | keep |

## validation trajectory

| training tokens | 6x768 | 10x640 | 15x512 | 24x384 |
| ---: | ---: | ---: | ---: | ---: |
| 50,003,968 | 4.75098 | 4.81615 | 4.88650 | 5.01166 |
| 100,007,936 | 3.89505 | 3.88262 | 3.92065 | 3.96338 |
| 150,011,904 | 3.72552 | 3.69583 | 3.69360 | 3.70818 |
| 200,015,872 | 3.65317 | 3.62202 | 3.61270 | 3.61347 |

15x512 has the lowest measured loss by 0.00077, which is not a meaningful gap on this validation sample. 24x384 reduced loss by 0.09471 over the final 50m tokens versus 0.08090 for 15x512 and was still closing the gap as warmup approached completion. select 24x384 for projected 10b-token quality; select 15x512 instead if wall time is the priority.
