"""Compile the matched language envelopes for all first-wave code components."""

import argparse
from pathlib import Path

from speck.experiments.code_languages import load_code_languages
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = load_code_languages(args.plan)
    root = repository_root(__file__)
    result["implementation"] = [
        {"path": name, "sha256": file_sha256(root / name)}
        for name in ("speck/experiments/code_languages.py", "scripts/code_language_preparation.py")
    ]
    if "code_preparation_decision" in result["inputs"]:
        name = "speck/experiments/code_preparation_decision.py"
        result["implementation"].append({"path": name, "sha256": file_sha256(root / name)})
    durable_json(args.output, result)
    print(
        f"Compiled {len(result['logical_slot_code_quotas'])} logical slots and 22 source-language targets."
    )


if __name__ == "__main__":
    main()
