# quality

results use 49,807,360 training tokens and 98,304 fixed validation tokens.

| loss | validation loss | tokens/s | decision |
| --- | ---: | ---: | --- |
| torch | 7.10396 | 117,365 | keep |
| cce exact | 7.10487 | 100,519 | reject |
| cce filtered | 7.48695 | 124,274 | reject |

## optimizer

| optimizer | validation loss | tokens/s | decision |
| --- | ---: | ---: | --- |
| adamw | 7.10396 | 117,365 | reject |
| muon | 6.44581 | 116,945 | keep |

## sequence

| schedule | validation loss | tokens/s | decision |
| --- | ---: | ---: | --- |
| full context | 6.44581 | 116,945 | keep |
| equal thirds | 6.50428 | 129,936 | reject |
