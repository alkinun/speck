"""Materialize explicit nested whole-document repetition streams from frozen token stocks."""

import bisect
import fcntl
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

from speck.data.loader import PackedTokenSource, manifest_fingerprint
from speck.data.stock_tokens import _identity, verify_token_stock
from speck.provenance.io import durable_json, file_sha256

_EPOCHS = (1, 2, 4)


def load_repetition_plan(path):
    """Validate an explicit order and exact nested boundaries; never choose or round pools."""
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (
        set(value)
        != {
            "format",
            "format_version",
            "stock_manifest",
            "document_order",
            "source_id",
            "seed",
            "exposure_tokens",
            "global_stride",
            "effective_epochs",
            "shard_tokens",
            "output_directory",
            "training_authority",
        }
        or value["format"] != "speck_repetition_source_plan"
        or value["format_version"] != 1
        or value["training_authority"] is not False
        or value["effective_epochs"] != list(_EPOCHS)
        or any(
            type(value[k]) is not int or value[k] < 1
            for k in ("exposure_tokens", "global_stride", "shard_tokens")
        )
        or type(value["seed"]) is not int
        or value["seed"] < 0
        or value["shard_tokens"] > 8_000_000
        or value["exposure_tokens"] % (4 * value["global_stride"])
    ):
        raise ValueError("repetition requires exact 1/2/4 pools aligned to distributed batches")
    stock_id = _identity(value["stock_manifest"], path.parent)
    order_id = _identity(value["document_order"], path.parent)
    stock_path = Path(stock_id["path"])
    stock = json.loads(stock_path.read_text())
    if (
        stock.get("format") != "speck_document_token_stock"
        or stock.get("format_version") != 1
        or stock.get("training_authority") is not False
        or stock["plan"]["source_id"] != value["source_id"]
    ):
        raise ValueError("repetition requires a source-identical frozen document token stock")
    verify_token_stock(stock_path.parent, stock["plan"])
    order = [json.loads(line) for line in Path(order_id["path"]).read_text().splitlines()]
    if (
        not order
        or any(type(i) is not int or i < 0 for i in order)
        or len(order) != len(set(order))
    ):
        raise ValueError("document order must list distinct nonnegative stock ordinals")
    wanted = set(order)
    selected = {}
    with (stock_path.parent / stock["documents"]["path"]).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["ordinal"] in wanted:
                selected[row["ordinal"]] = row
    if set(selected) != wanted:
        raise ValueError("document order refers to absent stock ordinals")
    rows, ends, hashes = [], [], set()
    total = 0
    for ordinal in order:
        row = selected[ordinal]
        digest = row["released_content_sha256"]
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or digest in hashes
        ):
            raise ValueError("nested pool contains duplicate or invalid document content hashes")
        hashes.add(digest)
        rows.append({**row, "view_token_start": total})
        total += row["token_count"]
        ends.append(total)
    if total != value["exposure_tokens"] or any(total // e not in ends for e in _EPOCHS):
        raise ValueError("nested pool boundaries must exactly match whole-document ends")
    output = (path.parent / value["output_directory"]).resolve()
    if any(
        p.is_relative_to(output) for p in (path, stock_path, Path(order_id["path"]))
    ) or output.is_relative_to(stock_path.parent):
        raise ValueError("repetition output overlaps protected inputs")
    config = {
        **value,
        "plan": {"path": str(path), "sha256": file_sha256(path)},
        "stock_manifest": stock_id,
        "document_order": order_id,
        "output_directory": str(output),
        "tokenizer": stock["plan"]["tokenizer"],
        "materializer_sha256": file_sha256(__file__),
    }
    return config, stock, rows, ends


def periodic_chunk(source, rows, ends, pool_tokens, start, count):
    """Read a declared periodic prefix without including any document outside its pool."""
    values = np.empty(count, dtype="<u2")
    written = 0
    while written < count:
        position = (start + written) % pool_tokens
        index = bisect.bisect_right(ends, position)
        row = rows[index]
        offset = position - row["view_token_start"]
        take = min(count - written, row["token_count"] - offset, pool_tokens - position)
        values[written : written + take] = source.read(
            row["token_start"] + offset, take, dtype=np.uint16
        )
        written += take
    return values


def _shard(directory, config_hash, start, values):
    directory.mkdir(parents=True, exist_ok=True)
    owner = {"config_sha256": config_hash, "start_token": start, "tokens": len(values)}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        file = (directory / manifest["path"]).resolve()
        if (
            any(manifest.get(k) != v for k, v in owner.items())
            or file.parent != directory.resolve()
            or file.stat().st_size != len(values) * 2
            or file_sha256(file) != manifest["sha256"]
            or not np.array_equal(np.fromfile(file, dtype="<u2"), values)
        ):
            raise ValueError("repetition shard identity, size or periodic bytes changed")
        return manifest, False
    # Each attempt is immutable. An interrupted payload never gets overwritten or truncated.
    if shutil.disk_usage(directory).free < values.nbytes + 1048576:
        raise ValueError("insufficient free space for the next repetition shard")
    payload = directory / f"attempt-{len(list(directory.glob('attempt-*'))):05d}.bin"
    with payload.open("xb") as handle:
        handle.write(values.tobytes())
        handle.flush()
        os.fsync(handle.fileno())
    manifest = {**owner, "path": payload.name, "sha256": file_sha256(payload)}
    durable_json(manifest_path, manifest)
    return manifest, True


def materialize_repetition(path, *, resume=False, interrupt_after_shards=None):
    """Write three ordinary uint16 source streams, keeping one explicit final lookahead token."""
    config, stock, rows, ends = load_repetition_plan(path)
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = output / "config.json"
        if owner.exists():
            if not resume or json.loads(owner.read_text()) != config:
                raise ValueError("resume requires the identical repetition plan and inputs")
        else:
            if resume or set(p.name for p in output.iterdir()) != {"owner.lock"}:
                raise ValueError("cannot resume absent repetition or overwrite unowned output")
            durable_json(owner, config)
        config_hash = manifest_fingerprint(config)
        source = PackedTokenSource(
            Path(config["stock_manifest"]["path"]).parent,
            {
                "id": config["source_id"],
                "splits": {"train": {"tokens": stock["token_count"], "shards": stock["shards"]}},
            },
            "train",
        )
        index_path = output / "pool-documents.jsonl"
        index_bytes = "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
        ).encode()
        if index_path.exists():
            if index_path.read_bytes() != index_bytes:
                raise ValueError("repetition pool document identities changed")
        else:
            # Publish via a temporary attempt so a torn index can be preserved and retried.
            temporary = (
                output / f"pool-index-attempt-{len(list(output.glob('pool-index-attempt-*'))):05d}"
            )
            with temporary.open("xb") as handle:
                handle.write(index_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, index_path)
        variants = {}
        new_shards = 0
        for epochs in _EPOCHS:
            pool = config["exposure_tokens"] // epochs
            shards = []
            for number, start in enumerate(
                range(0, config["exposure_tokens"] + 1, config["shard_tokens"])
            ):
                count = min(config["shard_tokens"], config["exposure_tokens"] + 1 - start)
                values = periodic_chunk(source, rows, ends, pool, start, count)
                directory = output / f"epochs-{epochs}" / f"shard-{number:05d}"
                shard, created = _shard(directory, config_hash, start, values)
                shards.append(
                    {
                        "path": str((directory / shard["path"]).relative_to(output)),
                        "tokens": count,
                        "sha256": shard["sha256"],
                    }
                )
                new_shards += created
                if interrupt_after_shards is not None and new_shards >= interrupt_after_shards:
                    raise RuntimeError("injected repetition materialization interruption")
            variants[str(epochs)] = {
                "effective_epochs": epochs,
                "unique_pool_tokens": pool,
                "unique_pool_documents": bisect.bisect_left(ends, pool) + 1,
                "input_exposure_tokens": config["exposure_tokens"],
                "tokens": config["exposure_tokens"] + 1,
                "shards": shards,
                "lookahead": {"tokens": 1, "pool_offset": 0, "additional_training_inputs": 0},
            }
        manifest = {
            "format": "speck_repeated_source_tokens",
            "format_version": 1,
            "status": "materialized_source_streams_not_training_manifest",
            "config": config,
            "config_sha256": config_hash,
            "pool_documents": {"path": index_path.name, "sha256": file_sha256(index_path)},
            "variants": variants,
            "training_authority": False,
            "semantics": "Each variant repeats its exact whole-document prefix in unchanged explicit order. Epoch e occurrence of base document d starts at e * unique_pool_tokens + d.view_token_start. The one final token is pool offset zero for next-token lookahead, not another training input. BOS/EOS preserves serialization, not attention isolation. No split, joint eligibility, mixture or launch receipt is selected here.",
        }
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("completed repetition manifest changed")
        else:
            durable_json(manifest_path, manifest)
        return manifest
