"""Finite engineering views with one source document per independent batch row."""

import json
from pathlib import Path

import numpy as np
import torch

from speck.data.document_lengths import census
from speck.data.loader import PackedTokenSource, manifest_fingerprint
from speck.data.stock_tokens import _identity, verify_token_stock
from speck.provenance.io import durable_json, file_sha256


def _compile(plan_path):
    """Bind explicit, non-overlapping windows; never select families, split data or repeat it."""
    plan_path = Path(plan_path).resolve()
    spec = json.loads(plan_path.read_text())
    if (
        set(spec)
        != {
            "format",
            "format_version",
            "stock_manifest",
            "windows",
            "sequence_length",
            "output_directory",
            "training_authority",
        }
        or spec["format"] != "speck_document_window_preparation"
        or spec["format_version"] != 1
        or spec["training_authority"] is not False
        or type(spec["sequence_length"]) is not int
        or not 1 <= spec["sequence_length"] <= 131072
    ):
        raise ValueError("unsupported bounded document-window preparation")
    stock_id = _identity(spec["stock_manifest"], plan_path.parent)
    order_id = _identity(spec["windows"], plan_path.parent)
    stock_path = Path(stock_id["path"])
    output = (plan_path.parent / spec["output_directory"]).resolve()
    if output.is_relative_to(stock_path.parent) or any(
        p.is_relative_to(output) for p in (plan_path, stock_path, Path(order_id["path"]))
    ):
        raise ValueError("window output overlaps protected inputs")
    stock = json.loads(stock_path.read_text())
    census(stock_id)  # Hash, complete index geometry and original content identities.
    verify_token_stock(stock_path.parent, stock["plan"])
    with Path(order_id["path"]).open() as handle:
        requested = [json.loads(line) for line in handle]
    if not requested or any(
        not isinstance(row, dict)
        or set(row) != {"ordinal", "document_offset"}
        or any(type(v) is not int or v < 0 for v in row.values())
        for row in requested
    ):
        raise ValueError("windows require explicit nonnegative document ordinals and offsets")
    wanted = {row["ordinal"] for row in requested}
    documents = {}
    with (stock_path.parent / stock["documents"]["path"]).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["ordinal"] in wanted:
                documents[row["ordinal"]] = row
    if set(documents) != wanted:
        raise ValueError("requested window document is absent")
    length = spec["sequence_length"]
    spans, owners, windows = {}, {}, []
    for request in requested:
        row = documents[request["ordinal"]]
        start = request["document_offset"]
        if start + length + 1 > row["token_count"]:
            raise ValueError("window and lookahead must stay inside one document")
        digest = row["released_content_sha256"]
        if owners.setdefault(digest, row["ordinal"]) != row["ordinal"]:
            raise ValueError("duplicate document content under different ordinals")
        spans.setdefault(row["ordinal"], []).append((start, start + length))
        windows.append(
            {
                **request,
                "content_id": row["content_id"],
                "released_content_sha256": digest,
                "document_tokens": row["token_count"],
                "source_token_start": row["token_start"] + start,
            }
        )
    for intervals in spans.values():
        intervals.sort()
        if any(b[0] < a[1] for a, b in zip(intervals, intervals[1:])):
            raise ValueError("input windows overlap or repeat; no repetition policy is selected")
    result = {
        "format": "speck_engineering_document_window_view",
        "format_version": 1,
        "plan": {"path": str(plan_path), "sha256": file_sha256(plan_path)},
        "stock_manifest": stock_id,
        "window_order": order_id,
        "source_id": stock["plan"]["source_id"],
        "tokenizer": stock["plan"]["tokenizer"],
        "sequence_length": length,
        "input_tokens": len(windows) * length,
        "windows": windows,
        "training_authority": False,
        "family_partition_qualified": False,
        "model_isolation_qualified": False,
        "boundary": "Engineering-only view. One complete length+1 span per independent batch row; no padding, wraparound, cross-document lookahead or implicit repetition. Call models without persistent recurrent/KV state between batches. Family/edition identity, train/evaluation separation, model reset behavior and scientific launch remain unqualified.",
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(__file__)},
    }
    return output, result, stock


def prepare_document_windows(plan_path):
    output, result, _ = _compile(plan_path)
    output.mkdir(parents=True, exist_ok=False)
    durable_json(output / "manifest.json", result)
    return result


class DocumentWindowReader:
    """Read finite rank-disjoint batches and resume from an exact global batch cursor."""

    def __init__(self, view_identity, *, batch_size, world_size=1, rank=0):
        if (
            any(type(v) is not int or v < 1 for v in (batch_size, world_size))
            or type(rank) is not int
            or not 0 <= rank < world_size
        ):
            raise ValueError("invalid document-window distributed geometry")
        identity = _identity(view_identity, Path.cwd())
        self.view = json.loads(Path(identity["path"]).read_text())
        if (
            self.view.get("format") != "speck_engineering_document_window_view"
            or self.view.get("format_version") != 1
            or self.view.get("training_authority") is not False
        ):
            raise ValueError("unsupported document-window view")
        plan_id = _identity(self.view["plan"], Path(identity["path"]).parent)
        _, expected, stock = _compile(plan_id["path"])
        if self.view != expected:
            raise ValueError("document-window view differs from its verified preparation")
        self.fingerprint = manifest_fingerprint(self.view)
        self.batch_size, self.world_size, self.rank = batch_size, world_size, rank
        self.stride = batch_size * world_size
        if not self.view["windows"] or len(self.view["windows"]) % self.stride:
            raise ValueError(
                "window count must fill complete distributed batches; no implicit drop"
            )
        stock_id = _identity(self.view["stock_manifest"], Path(identity["path"]).parent)
        stock_path = Path(stock_id["path"])
        self.source = PackedTokenSource(
            stock_path.parent,
            {
                "id": self.view["source_id"],
                "splits": {
                    "train": {
                        "shards": stock["shards"],
                        "tokens": stock["token_count"],
                    }
                },
            },
            "train",
        )
        self.batches = len(self.view["windows"]) // self.stride

    def state(self, batch):
        if type(batch) is not int or not 0 <= batch <= self.batches:
            raise ValueError("document-window cursor outside finite view")
        return {
            "format": "speck_document_window_batch_cursor",
            "format_version": 1,
            "view_sha256": self.fingerprint,
            "global_batch": batch,
            "batch_size": self.batch_size,
            "world_size": self.world_size,
        }

    def read(self, state):
        batch = state.get("global_batch")
        if state != self.state(batch):
            raise ValueError("document-window resume identity or geometry changed")
        if batch == self.batches:
            raise StopIteration("finite document-window view exhausted")
        begin = batch * self.stride + self.rank * self.batch_size
        rows = self.view["windows"][begin : begin + self.batch_size]
        length = self.view["sequence_length"]
        values = np.stack([self.source.read(row["source_token_start"], length + 1) for row in rows])
        return (
            torch.from_numpy(values[:, :-1].copy()),
            torch.from_numpy(values[:, 1:].copy()),
            rows,
            self.state(batch + 1),
        )
