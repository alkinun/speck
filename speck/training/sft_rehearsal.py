"""Select a finite, complete-conversation assistant rehearsal from retained local stock."""

import argparse
import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from speck.evaluation.protocol import BenchmarkExclusion
from speck.provenance.io import atomic_json, file_sha256
from speck.tokenization.chat import ChatFormatError, ChatTokenizer, decode_chat_record
from speck.tokenization.tokenizer import Tokenizer
from speck.tokenization.tools import PROTOCOL, adapt_conversation
from speck.training.sft_audit import iter_local_rows
from speck.training.sft_data import prepare_sft_dataset


def prepare(
    paths,
    output,
    tokenizer,
    prepared,
    *,
    train_per_kind=32,
    val_per_kind=8,
    pool=256,
    kinds=("text", "tools"),
    subsets=None,
    name="gh200-assistant-rehearsal",
):
    """Select complete conversations; `subsets` restricts rows to `source:subset` labels."""
    if min(train_per_kind, val_per_kind) < 1 or pool < max(train_per_kind, val_per_kind):
        raise ValueError(
            "positive sample counts and a sufficiently large candidate pool are required"
        )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    candidates, census, inputs = {}, Counter(), []
    serial = 0
    for path in sorted(Path(path).resolve() for path in paths):
        before = path.stat()
        inputs.append({"path": str(path), "bytes": before.st_size, "sha256": file_sha256(path)})
        for raw in iter_local_rows(path):
            row = decode_chat_record(raw)
            if subsets and f"{row.get('source')}:{row.get('subset')}" not in subsets:
                continue
            kind = (
                "tools"
                if row["tools"] or any(m.get("tool_calls") for m in row["messages"])
                else "text"
            )
            census[kind] += 1
            content = {key: row[key] for key in ("messages", "tools")}
            identity = hashlib.sha256(
                json.dumps(content, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            prompt = next(
                (m.get("content", "") for m in row["messages"] if m.get("role") == "user"), ""
            )
            prompt = " ".join(prompt.lower().split())
            split = (
                "val"
                if int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16) % 5 == 0
                else "train"
            )
            heap = candidates.setdefault((kind, split), [])
            priority = int(identity, 16)
            candidate = (-priority, serial, identity, row)
            if len(heap) < pool:
                heapq.heappush(heap, candidate)
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, candidate)
            serial += 1
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("SFT input changed during selection")
        print(f"Censused {serial:,} rows: {path.name}", flush=True)
    exclusion = BenchmarkExclusion(prepared)
    selected, identities, rejected, seen = {"train": [], "val": []}, [], Counter(), set()
    for kind in kinds:
        for split, count in (("train", train_per_kind), ("val", val_per_kind)):
            accepted = 0
            for _, _, identity, row in sorted(candidates.get((kind, split), []), reverse=True):
                if identity in seen:
                    rejected["duplicate"] += 1
                    continue
                seen.add(identity)
                try:
                    adapted = adapt_conversation(row)
                    tokens, mask = tokenizer.encode_messages(adapted["messages"])
                except ChatFormatError as error:
                    rejected[f"format: {error}"] += 1
                    continue
                if len(tokens) > 4097:
                    rejected["over_4096_complete"] += 1
                    continue
                if not any(mask):
                    rejected["no_supervised_tokens"] += 1
                    continue
                if exclusion.matches("\n".join(m["content"] for m in adapted["messages"])):
                    rejected["benchmark_match"] += 1
                    continue
                selected[split].append(
                    {"messages": adapted["messages"], "tools": [], "source": kind}
                )
                identities.append(
                    {
                        "sha256": identity,
                        "split": split,
                        "kind": kind,
                        "source": row.get("source"),
                        "subset": row.get("subset"),
                        "tokens": len(tokens),
                        "supervised_tokens": sum(mask),
                    }
                )
                accepted += 1
                if accepted == count:
                    break
            if accepted != count:
                atomic_json(
                    output / "selection-failure.json",
                    {
                        "status": "insufficient_candidates",
                        "kind": kind,
                        "split": split,
                        "accepted": accepted,
                        "required": count,
                        "inputs": inputs,
                        "census": dict(census),
                        "rejections": dict(rejected),
                        "selected": identities,
                    },
                )
                raise ValueError(
                    f"insufficient complete eligible {kind}/{split} candidates: {accepted}/{count}"
                )
    source = output / "source"
    source.mkdir()
    files = []
    for split, rows in selected.items():
        path = source / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        files.append({"filename": path.name, "split": split, "sha256": file_sha256(path)})
    dataset = {
        "format": "messages_v1",
        "name": name,
        "files": files,
        "expected_samples": sum(map(len, selected.values())),
        "validation_samples": len(selected["val"]),
    }
    manifest = prepare_sft_dataset(dataset, tokenizer, [4096], output / "packed", source_dir=source)
    result = {
        "format": "speck_sft_rehearsal",
        "format_version": 1,
        "status": "prepared",
        "protocol": PROTOCOL,
        "adapter_sha256": file_sha256(Path(__file__).parents[1] / "tokenization/tools.py"),
        "tokenizer": tokenizer.metadata(),
        "inputs": inputs,
        "census": dict(census),
        "candidate_pool_per_kind_split": pool,
        "subsets": sorted(subsets) if subsets else None,
        "rejections": dict(rejected),
        "selected": identities,
        "dataset": dataset,
        "packed_manifest_sha256": file_sha256(output / "packed/manifest.json"),
        "boundary": "Engineering rehearsal only. Source labels do not establish teacher correctness or release rights. Exact conversation deduplication, prompt-based split assignment, and benchmark exclusions are applied; semantic duplicates and contamination can remain.",
    }
    atomic_json(output / "dataset.json", dataset)
    atomic_json(output / "receipt.json", result)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prepared-evaluation", required=True)
    parser.add_argument("--pool", type=int, default=256)
    parser.add_argument("--train-per-kind", type=int, default=32)
    parser.add_argument("--val-per-kind", type=int, default=8)
    parser.add_argument("--kind", action="append", choices=("text", "tools"))
    parser.add_argument("--subset", action="append", help="source:subset label to keep")
    parser.add_argument("--name", default="gh200-assistant-rehearsal")
    args = parser.parse_args(argv)
    prepare(
        args.inputs,
        args.output,
        ChatTokenizer(Tokenizer(args.tokenizer)),
        json.loads(Path(args.prepared_evaluation).read_text()),
        pool=args.pool,
        train_per_kind=args.train_per_kind,
        val_per_kind=args.val_per_kind,
        kinds=tuple(args.kind or ("text", "tools")),
        subsets=set(args.subset or ()),
        name=args.name,
    )


if __name__ == "__main__":
    main()
