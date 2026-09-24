"""Write deterministic synthetic arithmetic RL prompts in the UltraData-RL Math format.

python experiments/rl-pilot/arithmetic_prompts.py OUTPUT.jsonl --count 2000

The pilot's SFT parent solves none of the UltraData-RL Math prompts, so no group carries signal.
These small integer problems give the update path a chance of nonzero advantages; they measure
engineering behaviour, not mathematical capability.
"""

import argparse
import json
import random

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def prompts(count, seed="speck-rl-arithmetic-v1"):
    generator = random.Random(seed)
    for index in range(count):
        a, b = generator.randint(2, 99), generator.randint(2, 99)
        operator = generator.choice("+-*")
        answer = {"+": a + b, "-": a - b, "*": a * b}[operator]
        yield {
            "uuid": f"arithmetic_{index:05d}",
            "query": f"What is {a} {operator} {b}?" + SUFFIX,
            "ground_truth": str(answer),
            "source": seed,
            "domain": "Math",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--count", type=int, default=2000)
    args = parser.parse_args()
    with open(args.output, "w") as handle:
        for row in prompts(args.count):
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
