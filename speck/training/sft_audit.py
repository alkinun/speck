"""Measure local SFT stock without truncation, training, or corpus redistribution."""

import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from speck.provenance.io import file_sha256
from speck.tokenization.chat import ChatFormatError


def iter_local_rows(path):
    """Read Parquet or Hugging Face Arrow streams, including List(Json()) columns."""

    path = Path(path)
    if path.suffix == ".parquet":
        with pq.ParquetFile(path) as source:
            for batch in source.iter_batches(batch_size=128):
                yield from batch.to_pylist()
    elif path.suffix == ".arrow":
        with pa.memory_map(str(path), "r") as source, pa.ipc.open_stream(source) as reader:
            for batch in reader:
                yield from batch.to_pylist()
    else:
        raise ValueError("SFT audit inputs must be .arrow or .parquet files")


def decode_conversation(row):
    """Decode HF Json features without requiring the datasets runtime."""

    if "messages" not in row:
        raise ChatFormatError(
            "input row is missing messages; index caches are not conversation data"
        )
    result = dict(row)
    for key in ("messages", "tools"):
        values = row.get(key) or []
        if not isinstance(values, list):
            raise ChatFormatError(f"{key} must be a list")
        try:
            result[key] = [
                json.loads(value) if isinstance(value, str) else value for value in values
            ]
        except (ValueError, TypeError) as error:
            raise ChatFormatError(f"invalid JSON in {key}") from error
        if any(not isinstance(value, dict) for value in result[key]):
            raise ChatFormatError(f"{key} entries must be objects")
    result["messages"] = [
        {key: value for key, value in message.items() if value is not None}
        for message in result["messages"]
    ]
    return result


def conversation_identity(row):
    """Use full conversation content, independent of input file ordering."""

    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _quantiles(values):
    if not values:
        return None
    return {
        "min": min(values),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


def audit_sft(paths, tokenizer, *, samples_per_subset=256, seed=42, lengths=(4096,), progress=None):
    """Census all rows; tokenize a deterministic bounded sample within each source/subset."""

    if type(samples_per_subset) is not int or samples_per_subset < 1:
        raise ValueError("samples_per_subset must be positive")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not lengths or any(type(length) is not int or length < 1 for length in lengths):
        raise ValueError("context lengths must be positive integers")
    paths = sorted(Path(path).resolve() for path in paths)
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("provide distinct SFT input files")
    groups, inputs = {}, []
    seen = 0
    for path in paths:
        before = path.stat()
        identity = {"path": str(path), "sha256": file_sha256(path), "bytes": before.st_size}
        for raw in iter_local_rows(path):
            row = decode_conversation(raw)
            key = json.dumps([row.get("source", "unknown"), row.get("subset", "unknown")])
            group = groups.setdefault(key, {"counts": Counter(), "sample": []})
            counts = group["counts"]
            counts["rows"] += 1
            messages = row["messages"]
            counts["with_tools"] += bool(row["tools"])
            counts["with_tool_calls"] += any(message.get("tool_calls") for message in messages)
            counts["with_tool_results"] += any(
                message.get("role") == "tool" for message in messages
            )
            counts["with_context_only_assistant"] += any(
                message.get("role") == "assistant" and message.get("weight", 1) == 0
                for message in messages
            )
            counts["with_thinking"] += any(
                "<think>" in (message.get("content") or "") for message in messages
            )
            digest = conversation_identity(row)
            priority = int(hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest(), 16)
            # The serial breaks ties for repeated identical records without comparing dicts.
            candidate = (-priority, seen, digest, row)
            sample = group["sample"]
            if len(sample) < samples_per_subset:
                heapq.heappush(sample, candidate)
            elif priority < -sample[0][0]:
                heapq.heapreplace(sample, candidate)
            seen += 1
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"SFT audit input changed while reading: {path}")
        inputs.append(identity)
        if progress is not None:
            progress(f"Censused {seen:,} rows; {path.name}")
    summaries = []
    for key, group in sorted(groups.items()):
        failures, fits = Counter(), {str(length): 0 for length in sorted(set(lengths))}
        token_lengths, supervised_lengths, identities = [], [], []
        for _, _, digest, row in sorted(group["sample"], key=lambda item: item[2]):
            identities.append(digest)
            try:
                if row["tools"]:
                    raise ChatFormatError("tool definitions require a tool-aware chat format")
                tokens, mask = tokenizer.encode_messages(row["messages"])
                if not any(mask):
                    raise ChatFormatError("conversation has no assistant target")
            except ChatFormatError as error:
                failures[str(error)] += 1
                continue
            token_lengths.append(len(tokens))
            supervised_lengths.append(sum(mask))
            for length in fits:
                fits[length] += len(tokens) <= int(length) + 1
        source, subset = json.loads(key)
        summaries.append(
            {
                "source": source,
                "subset": subset,
                "census": dict(group["counts"]),
                "sampled_rows": len(group["sample"]),
                "sample_sha256": hashlib.sha256("\n".join(identities).encode()).hexdigest(),
                "serializable_rows": len(token_lengths),
                "rejections": dict(failures),
                "tokens": _quantiles(token_lengths),
                "supervised_tokens": _quantiles(supervised_lengths),
                "complete_rows_fitting_context": fits,
            }
        )
        if progress is not None:
            progress(f"Tokenized sample: {source} / {subset}")
    return {
        "format": "speck_sft_stock_audit",
        "format_version": 1,
        "inputs": inputs,
        "tokenizer": tokenizer.metadata(),
        "settings": {
            "samples_per_subset": samples_per_subset,
            "seed": seed,
            "lengths": list(lengths),
        },
        "rows": seen,
        "subsets": summaries,
        "interpretation": (
            "Census counts cover every input row. Token lengths and context-fit counts cover only "
            "the deterministic per-subset sample under the recorded chat format. Unsupported rows "
            "are counted as rejections, not assigned hypothetical token lengths. This checks "
            "format and length, not answer correctness, benchmark contamination, or split eligibility."
        ),
    }
