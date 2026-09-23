"""Select a finite, benchmark-excluded stage-5 prompt file from verifiable JSONL rows."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from speck.evaluation.protocol import BenchmarkExclusion
from speck.provenance.io import atomic_json, file_sha256
from speck.training.rl import REWARDS


def _normalized(text):
    return " ".join(text.lower().split())


def sft_prompts(paths):
    """Normalized first user turns of prepared SFT Parquet files, so RL cannot reuse them."""
    prompts = set()
    for path in paths:
        for messages in pq.read_table(path, columns=["messages"])["messages"].to_pylist():
            messages = [json.loads(m) if isinstance(m, str) else m for m in messages]
            user = next((m["content"] for m in messages if m["role"] == "user"), "")
            prompts.add(_normalized(user))
    return prompts


def prepare(paths, output, prepared, *, limit, seed, exclude_sft=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inputs, groups, rejected = [], defaultdict(list), Counter()
    for path in sorted(Path(path).resolve() for path in paths):
        inputs.append({"path": str(path), "sha256": file_sha256(path)})
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["domain"] not in REWARDS:
                rejected["unverifiable_domain"] += 1
                continue
            groups[_normalized(row["query"])].append(row)
    excluded_prompts = sft_prompts(exclude_sft)
    exclusion = BenchmarkExclusion(prepared)
    kept = []
    for prompt, rows in groups.items():
        rejected["duplicate_copy"] += len(rows) - 1
        if len({json.dumps(row["ground_truth"], sort_keys=True) for row in rows}) > 1:
            rejected["conflicting_reference"] += 1
        elif prompt in excluded_prompts:
            rejected["sft_prompt"] += 1
        elif exclusion.matches(rows[0]["query"]):
            rejected["benchmark_match"] += 1
        else:
            kept.append(rows[0])
    kept.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['uuid']}".encode()).hexdigest())
    if len(kept) < limit:
        raise ValueError(f"only {len(kept)} eligible prompts for a limit of {limit}")
    selected = kept[:limit]
    path = output / "prompts.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
    receipt = {
        "format": "speck_rl_prompts",
        "format_version": 1,
        "inputs": inputs,
        "excluded_sft": [{"path": str(p), "sha256": file_sha256(p)} for p in exclude_sft],
        "benchmarks": [benchmark["id"] for benchmark in prepared["benchmarks"]],
        "unique_prompts": len(groups),
        "eligible_prompts": len(kept),
        "selected_prompts": len(selected),
        "rejections": dict(rejected),
        "domains": dict(Counter(row["domain"] for row in selected)),
        "prompts": {"path": str(path), "sha256": file_sha256(path)},
        "boundary": (
            "Exact normalized-prompt deduplication, conflicting-reference removal, SFT prompt "
            "exclusion and the frozen benchmark scanner only. Reference answers are unverified "
            "and semantic contamination can remain."
        ),
    }
    atomic_json(output / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared-evaluation", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--seed", default="speck-rl-prompts-v1")
    parser.add_argument("--exclude-sft", action="append", default=[])
    args = parser.parse_args(argv)
    receipt = prepare(
        args.inputs,
        args.output,
        json.loads(Path(args.prepared_evaluation).read_text()),
        limit=args.limit,
        seed=args.seed,
        exclude_sft=args.exclude_sft,
    )
    print(json.dumps({k: v for k, v in receipt.items() if k != "inputs"}, indent=2))


if __name__ == "__main__":
    main()
