"""Review three completed tokenizer screens against the measured 30-GPU-hour budget."""

import argparse
import json
from pathlib import Path

from speck.provenance.io import file_sha256
from speck.provenance.repository import repository_root
from speck.tokenization.pilot_screen import load_screen_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="Hash-bound screen review manifest")
    parser.add_argument("output", help="New analysis JSON; existing files are never overwritten")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"tokenizer screen analysis already exists: {output}")
    result = load_screen_report(args.manifest)
    root = repository_root(__file__)
    result["implementation"] = [
        {"path": str(root / name), "sha256": file_sha256(root / name)}
        for name in (
            "speck/tokenization/pilot.py",
            "speck/tokenization/pilot_screen.py",
            "speck/provenance/io.py",
            "speck/provenance/repository.py",
            "scripts/tokenizer_pilot_screen.py",
        )
    ]
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
