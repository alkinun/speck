"""Apply the frozen equal-category BPB analysis to two held-out score reports."""

import argparse
import json
from pathlib import Path

from speck.heldout_evaluation import analyze_heldout_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"held-out analysis already exists: {output}")
    baseline = json.loads(Path(args.baseline).read_text())
    candidate = json.loads(Path(args.candidate).read_text())
    result = analyze_heldout_scores(args.manifest, baseline, candidate)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "Eligible under both separate views."
        if result["eligible_under_both_separate_views"]
        else "At least one category/view guardrail failed."
    )


if __name__ == "__main__":
    main()
