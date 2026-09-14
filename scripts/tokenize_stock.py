"""Build or verify document-indexed token shards from one checked source stock."""

import argparse
import subprocess
from pathlib import Path

from speck.data.stock_tokens import load_stock_token_plan, tokenize_stock
from speck.provenance.io import durable_json, file_sha256
from speck.provenance.repository import repository_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    root = repository_root(__file__)
    if args.report.exists():
        raise FileExistsError(args.report)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("stock tokenization requires a clean implementation")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    result = tokenize_stock(load_stock_token_plan(args.plan))
    result.update(
        format="speck_stock_tokenization_result",
        format_version=1,
        repository_revision=revision,
        plan={"path": str(args.plan.resolve()), "sha256": file_sha256(args.plan)},
        training_authority=False,
    )
    durable_json(args.report, result)
    print(result)


if __name__ == "__main__":
    main()
