"""Tokenize verified single-source stock into immutable shards and document span indices."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from speck.data.exclusion import verify_excluded_parent
from speck.data.packing import TokenShardWriter
from speck.provenance.io import durable_json, file_sha256
from speck.tokenization.tokenizer import Tokenizer


def _identity(identity, base):
    if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
        raise ValueError("stock token inputs require path and sha256")
    path = (base / identity["path"]).resolve()
    if not path.is_file() or file_sha256(path) != identity["sha256"]:
        raise ValueError(f"stock token input identity mismatch: {path}")
    return {"path": str(path), "sha256": identity["sha256"]}


def load_stock_token_plan(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (
        set(value)
        != {
            "format",
            "format_version",
            "source_id",
            "category",
            "stock_result",
            "tokenizer_decision",
            "shard_tokens",
            "output_directory",
            "training_authority",
        }
        or value["format"] != "speck_stock_tokenization_plan"
        or value["format_version"] != 1
        or value["training_authority"] is not False
        or not isinstance(value["source_id"], str)
        or Path(value["source_id"]).name != value["source_id"]
        or value["category"] not in ("web", "code", "math", "synthetic", "science", "reference")
        or type(value["shard_tokens"]) is not int
        or not 1 <= value["shard_tokens"] <= 100_000_000
    ):
        raise ValueError("unsupported stock tokenization plan")
    stock_id = _identity(value["stock_result"], path.parent)
    stock_path = Path(stock_id["path"])
    stock = json.loads(stock_path.read_text())
    if stock.get("training_authority") is not False or stock.get("format") not in (
        "speck_admitted_math_preparation_result",
        "speck_reference_stock_preparation_result",
        "speck_science_stock_preparation_result",
        "speck_finemath_stock_preparation_result",
        "speck_fineweb_edu_stock_preparation_result",
        "speck_cosmopedia_stock_preparation_result",
        "speck_stack_edu_stock_preparation_result",
    ):
        raise ValueError("tokenization requires a supported single-source preparation result")
    prepared_plan_id = _identity(stock["plan"], stock_path.parent)
    if json.loads(Path(prepared_plan_id["path"]).read_text())["source_id"] != value["source_id"]:
        raise ValueError("stock source differs from the requested source")
    parent_id = _identity(stock["analysis"]["parent_manifest"], stock_path.parent)
    parent_path = Path(parent_id["path"])
    parent = json.loads(parent_path.read_text())
    verify_excluded_parent(parent, stock["analysis"]["references"])
    if stock["analysis"]["exact_reference_overlap"] != 0:
        raise ValueError("stock contains reference overlap")
    category = value["category"]
    if any(row["records"] for key, row in stock["analysis"]["retained"].items() if key != category):
        raise ValueError("stock is not a single-category source")
    entry = parent["outputs"][f"acquired_train__{category}"]
    input_path = (parent_path.parent / entry["path"]).resolve()
    if not input_path.is_relative_to(parent_path.parent) or input_path == parent_path:
        raise ValueError("stock text escapes its parent directory")
    text_id = _identity({"path": str(input_path), "sha256": entry["sha256"]}, path.parent)
    decision_id = _identity(value["tokenizer_decision"], path.parent)
    decision = json.loads(Path(decision_id["path"]).read_text())
    if decision.get("status") != "tokenizer_selected_and_frozen":
        raise ValueError("stock tokenization requires the frozen base tokenizer")
    model = Path(decision["tokenizer"]["directory"]) / "tokenizer.model"
    tokenizer_id = _identity(
        {"path": str(model), "sha256": decision["tokenizer_fingerprint"]}, path.parent
    )
    if stock["reference_capacity"]["tokenizer"]["sha256"] != tokenizer_id["sha256"]:
        raise ValueError("stock count does not bind the selected tokenizer")
    output = (path.parent / value["output_directory"]).resolve()
    for protected in (path, stock_path, parent_path, input_path, model, Path(decision_id["path"])):
        if protected.is_relative_to(output):
            raise ValueError("stock output would contain a protected input")
    return {
        **value,
        "plan": {"path": str(path), "sha256": file_sha256(path)},
        "stock_result": stock_id,
        "tokenizer_decision": decision_id,
        "tokenizer": tokenizer_id,
        "parent_manifest": parent_id,
        "input": text_id,
        "expected_documents": stock["reference_capacity"]["documents"],
        "expected_tokens": stock["reference_capacity"]["tokens"],
        "output_directory": str(output),
    }


def verify_token_stock(directory, plan):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest["plan"] != plan
        or manifest["status"] != "complete_document_token_cache_not_training_view"
    ):
        raise ValueError("token stock owner/plan mismatch")
    for entry in [manifest["documents"], *manifest["shards"]]:
        path = directory / entry["path"]
        if path.parent != directory or file_sha256(path) != entry["sha256"]:
            raise ValueError("token stock payload identity mismatch")
        if "tokens" in entry and path.stat().st_size != 2 * entry["tokens"]:
            raise ValueError("token stock shard size mismatch")
    if sum(row["tokens"] for row in manifest["shards"]) != plan["expected_tokens"]:
        raise ValueError("token stock token count mismatch")
    offset = count = 0
    with (directory / manifest["documents"]["path"]).open() as handle:
        for raw in handle:
            row = json.loads(raw)
            if row["ordinal"] != count or row["token_start"] != offset or row["token_count"] < 2:
                raise ValueError("token stock document index is not contiguous")
            offset += row["token_count"]
            count += 1
    if (offset, count) != (plan["expected_tokens"], plan["expected_documents"]):
        raise ValueError("token stock document coverage mismatch")
    return manifest


def _read_span(directory, shards, start, count):
    arrays = []
    for shard in shards:
        size = shard["tokens"]
        if start >= size:
            start -= size
            continue
        take = min(count, size - start)
        values = np.memmap(directory / shard["path"], mode="r", dtype="<u2")
        arrays.append(np.array(values[start : start + take]))
        count -= take
        start = 0
        if not count:
            return np.concatenate(arrays).tolist()
    raise ValueError("document span exceeds packed stock")


def tokenize_stock(plan):
    started = time.perf_counter()
    output = Path(plan["output_directory"])
    if output.exists():
        manifest = verify_token_stock(output, plan)
        return {
            "manifest": {
                "path": str(output / "manifest.json"),
                "sha256": file_sha256(output / "manifest.json"),
            },
            "documents": manifest["document_count"],
            "tokens": manifest["token_count"],
            "reused": True,
            "elapsed_seconds": time.perf_counter() - started,
        }
    staging = output.with_name(output.name + ".building")
    # An interrupted unpublished build is retained; no implicit deletion or reuse.
    staging.mkdir(parents=True, exist_ok=False)
    durable_json(staging / "plan.json", plan)
    tokenizer = Tokenizer(plan["tokenizer"]["path"])
    if tokenizer.fingerprint() != plan["tokenizer"]["sha256"] or tokenizer.vocab_size > 65536:
        raise ValueError("tokenizer identity/uint16 geometry mismatch")
    if file_sha256(plan["input"]["path"]) != plan["input"]["sha256"]:
        raise ValueError("source stock changed after planning")
    writer = TokenShardWriter(staging, "tokens", plan["shard_tokens"])
    probes, batch = [], []
    characters = documents = 0
    probe_ordinals = {0, plan["expected_documents"] // 2, plan["expected_documents"] - 1}
    index_path = staging / "documents.jsonl"
    try:
        with Path(plan["input"]["path"]).open() as source, index_path.open("w") as index:

            def flush():
                nonlocal documents
                encoded = tokenizer.encode_batch([row["text"] for row in batch], bos=True, eos=True)
                for row, tokens in zip(batch, encoded, strict=True):
                    digest = hashlib.sha256(row["text"].encode()).hexdigest()
                    if digest != row["released_content_sha256"]:
                        raise ValueError("source document text/hash mismatch")
                    span = {
                        "ordinal": documents,
                        "content_id": row["content_id"],
                        "released_content_sha256": digest,
                        "token_start": writer.total_tokens,
                        "token_count": len(tokens),
                        "utf8_bytes": len(row["text"].encode()),
                    }
                    if documents in probe_ordinals:
                        probes.append((span, tokens))
                    writer.write(tokens)
                    index.write(json.dumps(span, sort_keys=True, separators=(",", ":")) + "\n")
                    documents += 1

            for raw in source:
                row = json.loads(raw)
                batch.append(row)
                characters += len(row["text"])
                if len(batch) >= 128 or characters >= 1_000_000:
                    flush()
                    batch, characters = [], 0
            if batch:
                flush()
        writer.finish()
    except BaseException:
        writer.finish()
        raise
    if (documents, writer.total_tokens) != (plan["expected_documents"], plan["expected_tokens"]):
        raise ValueError("packed counts differ from the checked stock measurement")
    for span, tokens in probes:
        if _read_span(staging, writer.shards, span["token_start"], span["token_count"]) != tokens:
            raise ValueError("packed document span differs from tokenizer output")
    manifest = {
        "format": "speck_document_token_stock",
        "format_version": 1,
        "status": "complete_document_token_cache_not_training_view",
        "plan": plan,
        "document_count": documents,
        "token_count": writer.total_tokens,
        "documents": {"path": index_path.name, "sha256": file_sha256(index_path)},
        "shards": writer.shards,
        "probe_ordinals": sorted(probe_ordinals),
        "serialization": "unchanged text with BOS and EOS for every document",
        "training_authority": False,
        "boundary": "Token cache in source order with whole-document span index. No mixture, training order, train/validation split or joint treatment/background eligibility is selected here. Later views reference source ordinals and token spans under this immutable parent.",
    }
    durable_json(staging / "manifest.json", manifest)
    verify_token_stock(staging, plan)
    staging.replace(output)
    return {
        "manifest": {
            "path": str(output / "manifest.json"),
            "sha256": file_sha256(output / "manifest.json"),
        },
        "documents": documents,
        "tokens": writer.total_tokens,
        "reused": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
