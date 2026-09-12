"""Run the frozen tokenizer-pilot CUDA document-NLL parity qualification."""

import argparse
import json
from pathlib import Path

from speck.io import atomic_json
from speck.tokenizer_pilot_cuda_evaluation import (
    load_document_nll_policy,
    qualify_document_nll,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"tokenizer pilot document NLL qualification exists: {args.output}")
    result = qualify_document_nll(load_document_nll_policy(args.policy))
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
