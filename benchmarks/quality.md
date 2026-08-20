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
| quarter stages | 6.50510 | 126,229 | reject |
| 1024 warmup | 6.47792 | 120,418 | reject |

## batch

| schedule | validation loss | tokens/s | decision |
| --- | ---: | ---: | --- |
| fixed 524k | 6.44581 | 116,945 | reject |
| 131k to 262k to 524k | 5.60311 | 116,163 | reject |
| fixed 262k | 5.75040 | 116,099 | reject |
| fixed 131k | 5.18771 | 114,341 | reject |
| fixed 65k | 4.60778 | 111,524 | reject |
| fixed 32k | 4.14256 | 105,951 | reject |
| fixed 16k | 3.86934 | 89,493 | keep |
| fixed 8k | 3.79849 | 74,198 | reject |

## architecture

| shape | parameters | validation loss | tokens/s | decision |
| --- | ---: | ---: | ---: | --- |
| 24x384 | 50,055,552 | 3.86934 | 89,493 | reject |
| 15x512 | 49,823,232 | 3.82319 | 107,613 | reject |
| 10x640 | 49,984,640 | 3.80685 | 113,455 | keep |
| 6x768 | 50,095,872 | 3.82444 | 129,888 | reject |
| 4x896 | 49,782,656 | 3.87434 | 142,710 | reject |

## selected batch

| batch tokens | validation loss | tokens/s | decision |
| ---: | ---: | ---: | --- |
| 16,384 | 3.82444 | 129,888 | keep |
| 8,192 | 3.80461 | 110,559 | reject |

## mlp

| mlp | parameters | validation loss | tokens/s | decision |
| --- | ---: | ---: | ---: | --- |
| swiglu | 50,095,872 | 3.82444 | 129,888 | keep |
| relu squared | 50,095,872 | 3.84696 | 131,965 | reject |

## production

| parameters | batch tokens | validation loss | tokens/s |
| ---: | ---: | ---: | ---: |
| 49,984,640 | 16,384 | 3.80685 | 113,455 |
