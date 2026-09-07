"""Assess or canonicalize a human-completed source-rights template."""

import argparse
import json
from pathlib import Path

from speck.source_rights import assess_rights_template, finalize_human_acceptance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("template")
    parser.add_argument("--finalize")
    args = parser.parse_args()
    path = Path(args.template).resolve()
    template = json.loads(path.read_text(encoding="utf-8"))
    if args.finalize:
        result = finalize_human_acceptance(template, args.finalize, config_dir=path.parent)
    else:
        result = assess_rights_template(template, config_dir=path.parent, verify_evidence=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
