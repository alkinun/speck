# Stage-5 RL pilot on a real SFT parent (RTX 3090)

Engineering pilot for the [reward trainer](../../docs/training.md#reward-training) that the R1 and
R2 [post-training families](../../docs/program.md#post-training) use, run on the workstation. It spends no grant GPU-hours and admits no data. The pilot's base checkpoint (step 800,
the one [development scoring](../pilot/development-result.json) measured) is pinned by hash in
`sft.json`. It first gets a short math SFT, then `rl.json` runs group-relative RL from that SFT
checkpoint.

Inputs are selected from local stock with the frozen five-benchmark exclusion:

- **SFT:** 6,000 train and 256 validation `openbmb/UltraData-SFT-2605` Math/think conversations
  that fit whole in 4K, chosen by `scripts.sft_rehearsal --subset`.
- **RL:** 2,000 UltraData-RL-2609 Math prompts from `scripts.rl_prompts`, deduplicated, with
  conflicting references dropped and no prompt shared with the SFT set.
- **Arithmetic:** `arithmetic/` runs the same trainer on 1,000 synthetic problems from
  `arithmetic_prompts.py`, as a check of whether easier prompts give signal.

```bash
uv run --no-sync python -m scripts.sft_train experiments/rl-pilot
uv run --no-sync python -m scripts.rl_train experiments/rl-pilot
```

[`result.json`](result.json) records the inputs, measurements and restart checks:

- **SFT:** 375 steps, 9.3M supervised tokens in 1.7 hours. Validation loss went from 2.048 at step
  100 to 1.697.
- **Memory:** a group of 8 with Muon and checkpointing peaks at 18.8 GiB, flat from 512 to 1,536
  new tokens.
- **Reward:** the first 40-step run solved 2 of 160 prompt groups, which gave two real updates. The
  deterministic rerun and the arithmetic run solved none. About half the math completions reached
  the 1,536-token cap.
- **Restart:** a resumed run reproduces rollouts and state exactly. Without `deterministic`, the
  first update after resume differs slightly and the runs diverge. With it, backward passes are
  bitwise reproducible, at about 28% more time per step.

The base saw only 105M tokens, so this qualifies memory, rollout cost, zero-signal accounting and
restart on a full-size parent, not RL benefit.
