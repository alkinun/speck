"""Pack a shared post-endpoint continuation for equal-FLOP tokenizer pilot views."""

import hashlib
import json
import os
import shutil
from pathlib import Path

from speck.dataset import TokenShardWriter
from speck.io import atomic_json, file_sha256
from speck.tokenizer import Tokenizer

FORMAT = "speck_tokenizer_pilot_continuation"
FORMAT_VERSION = 1
REFERENCE_CONTINUATION_TOKENS = 72_000_000
CATEGORIES = ("web", "code", "math", "synthetic", "science", "reference")


def _identity(value, root, context):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{context} must contain path and sha256")
    path = Path(value["path"]).expanduser()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file() or file_sha256(path) != value["sha256"]:
        raise ValueError(f"{context} identity mismatch")
    return {"path": str(path), "sha256": value["sha256"]}


def validate_continuation_plan(value, *, config_dir=None):
    root = Path(config_dir or ".").resolve()
    expected = {
        "format",
        "format_version",
        "status",
        "fixed_stream",
        "reference_tokenizer_id",
        "tokenizers",
        "categories",
        "shard_tokens",
        "output_directory",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("tokenizer pilot continuation fields are invalid")
    if (
        value["format"] != FORMAT
        or value["format_version"] != FORMAT_VERSION
        or value["status"] != "materialization_authorized_not_model_training_authority"
    ):
        raise ValueError("unsupported tokenizer pilot continuation contract")
    fixed_stream = _identity(value["fixed_stream"], root, "fixed stream")
    tokenizers = value["tokenizers"]
    if not isinstance(tokenizers, list) or len(tokenizers) < 2:
        raise ValueError("tokenizer pilot continuation requires at least two tokenizers")
    normalized_tokenizers = []
    tokenizer_ids = []
    for item in tokenizers:
        if not isinstance(item, dict) or set(item) != {
            "id",
            "model",
            "minimum_additional_tokens",
        }:
            raise ValueError("tokenizer pilot continuation tokenizer fields are invalid")
        tokenizer_id = item["id"]
        if (
            not isinstance(tokenizer_id, str)
            or not tokenizer_id
            or Path(tokenizer_id).name != tokenizer_id
        ):
            raise ValueError("tokenizer pilot continuation ID must be a path component")
        minimum = item["minimum_additional_tokens"]
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
            raise ValueError("minimum additional tokenizer tokens must be non-negative")
        tokenizer_ids.append(tokenizer_id)
        normalized_tokenizers.append(
            {
                **item,
                "model": _identity(item["model"], root, f"tokenizer {tokenizer_id}"),
            }
        )
    if len(tokenizer_ids) != len(set(tokenizer_ids)):
        raise ValueError("tokenizer pilot continuation IDs must be unique")
    if value["reference_tokenizer_id"] not in tokenizer_ids:
        raise ValueError("continuation reference tokenizer is not declared")
    categories = value["categories"]
    if (
        not isinstance(categories, list)
        or tuple(item.get("id") for item in categories) != CATEGORIES
    ):
        raise ValueError("continuation categories must use the frozen order")
    normalized_categories = []
    for item in categories:
        if not isinstance(item, dict) or set(item) != {
            "id",
            "source_id",
            "input",
            "reference_token_target",
        }:
            raise ValueError("continuation category fields are invalid")
        target = item["reference_token_target"]
        if isinstance(target, bool) or not isinstance(target, int) or target < 1:
            raise ValueError("continuation reference target must be positive")
        normalized_categories.append(
            {**item, "input": _identity(item["input"], root, f"category {item['id']}")}
        )
    if sum(item["reference_token_target"] for item in categories) != REFERENCE_CONTINUATION_TOKENS:
        raise ValueError("continuation reference targets have the wrong total")
    shard_tokens = value["shard_tokens"]
    if isinstance(shard_tokens, bool) or not isinstance(shard_tokens, int) or shard_tokens < 1:
        raise ValueError("continuation shard size must be positive")
    output = Path(value["output_directory"]).expanduser()
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    normalized = {
        **value,
        "fixed_stream": fixed_stream,
        "tokenizers": normalized_tokenizers,
        "categories": normalized_categories,
        "output_directory": str(output),
    }
    normalized["plan_fingerprint"] = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return normalized


def load_continuation_plan(path):
    path = Path(path).resolve()
    return validate_continuation_plan(json.loads(path.read_text()), config_dir=path.parent)


def _validated_plan(plan):
    if "plan_fingerprint" not in plan:
        return validate_continuation_plan(plan)
    payload = {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if plan["plan_fingerprint"] != fingerprint:
        raise ValueError("normalized tokenizer pilot continuation fingerprint mismatch")
    return plan


def _last_jsonl(path):
    with Path(path).open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell() - 1
        while position >= 0:
            handle.seek(position)
            if handle.read(1) == b"\n" and position < handle.seek(0, os.SEEK_END) - 1:
                break
            position -= 1
        handle.seek(position + 1)
        return json.loads(handle.readline())


def _verify_published(plan, output, manifest):
    if manifest.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise ValueError("published continuation belongs to another plan")
    for tokenizer in manifest["tokenizers"]:
        for category in tokenizer["categories"]:
            for shard in category["shards"]:
                path = output / tokenizer["id"] / category["id"] / shard["path"]
                if not path.is_file() or file_sha256(path) != shard["sha256"]:
                    raise ValueError("published continuation shard identity mismatch")
    for category in manifest["continuation"]["categories"]:
        path = output / category["index"]["path"]
        if not path.is_file() or file_sha256(path) != category["index"]["sha256"]:
            raise ValueError("published continuation index identity mismatch")


def materialize_continuation(plan, *, restart=False):
    """Continue after the frozen whole-document boundary and pack all tokenizers."""

    plan = _validated_plan(plan)
    fixed_path = Path(plan["fixed_stream"]["path"])
    fixed = json.loads(fixed_path.read_text())
    if (
        fixed.get("format") != "speck_tokenizer_pilot_stream_result"
        or fixed.get("status")
        != "whole_document_stream_and_tokenizer_packs_complete_not_training_authority"
        or fixed.get("model_training_authority") is not False
    ):
        raise ValueError("fixed tokenizer pilot stream is incomplete")
    fixed_categories = {item["id"]: item for item in fixed["document_stream"]["categories"]}
    fixed_tokenizers = {item["id"]: item for item in fixed["tokenizers"]}
    for tokenizer in plan["tokenizers"]:
        fixed_tokenizer = fixed_tokenizers.get(tokenizer["id"])
        if fixed_tokenizer is None or fixed_tokenizer["model"] != tokenizer["model"]:
            raise ValueError(f"continuation tokenizer differs from fixed stream: {tokenizer['id']}")
    start_rows = {}
    for category in plan["categories"]:
        fixed_category = fixed_categories.get(category["id"])
        if (
            fixed_category is None
            or fixed_category["source_id"] != category["source_id"]
            or fixed_category["source"] != category["input"]
        ):
            raise ValueError(f"continuation category differs from fixed stream: {category['id']}")
        index_path = fixed_path.parent / fixed_category["index"]["path"]
        if not index_path.is_file() or file_sha256(index_path) != fixed_category["index"]["sha256"]:
            raise ValueError(f"fixed stream index identity mismatch: {category['id']}")
        start_rows[category["id"]] = _last_jsonl(index_path)["source_row"]
    output = Path(plan["output_directory"])
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        _verify_published(plan, output, manifest)
        return manifest
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete continuation exists: {staging}")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    tokenizers = {item["id"]: Tokenizer(item["model"]["path"]) for item in plan["tokenizers"]}
    reference = tokenizers[plan["reference_tokenizer_id"]]
    writers = {
        tokenizer_id: {
            category["id"]: TokenShardWriter(
                staging / tokenizer_id / category["id"], "train", plan["shard_tokens"]
            )
            for category in plan["categories"]
        }
        for tokenizer_id in tokenizers
    }
    continuation_hasher = hashlib.sha256(bytes.fromhex(fixed["document_stream"]["sha256"]))
    category_results = []
    try:
        for category in plan["categories"]:
            category_id = category["id"]
            index_path = staging / f"documents-{category_id}.jsonl"
            documents = reference_tokens = utf8_bytes = 0
            tokenizer_tokens = {tokenizer_id: 0 for tokenizer_id in tokenizers}
            with (
                Path(category["input"]["path"]).open(encoding="utf-8") as source,
                index_path.open("w", encoding="utf-8") as index,
            ):

                def write_batch(batch):
                    nonlocal documents, reference_tokens, utf8_bytes
                    reference_encoded = reference.encode_batch(
                        [item[2] for item in batch], bos=True, eos=True
                    )
                    selected = 0
                    projected = reference_tokens
                    for tokens in reference_encoded:
                        projected += len(tokens)
                        selected += 1
                        if projected >= category["reference_token_target"]:
                            break
                    batch = batch[:selected]
                    encoded = {
                        plan["reference_tokenizer_id"]: reference_encoded[:selected],
                        **{
                            tokenizer_id: tokenizer.encode_batch(
                                [item[2] for item in batch], bos=True, eos=True
                            )
                            for tokenizer_id, tokenizer in tokenizers.items()
                            if tokenizer_id != plan["reference_tokenizer_id"]
                        },
                    }
                    for position, (row, content_sha256, text) in enumerate(batch):
                        starts = {
                            tokenizer_id: writers[tokenizer_id][category_id].total_tokens
                            for tokenizer_id in tokenizers
                        }
                        for tokenizer_id, token_batches in encoded.items():
                            tokens = token_batches[position]
                            writers[tokenizer_id][category_id].write(tokens)
                            tokenizer_tokens[tokenizer_id] += len(tokens)
                        result = {
                            "document_id": content_sha256,
                            "source_row": row,
                            "utf8_bytes": len(text.encode()),
                            "reference_tokens": len(
                                encoded[plan["reference_tokenizer_id"]][position]
                            ),
                            "token_spans": {
                                tokenizer_id: {
                                    "start": starts[tokenizer_id],
                                    "end": starts[tokenizer_id]
                                    + len(encoded[tokenizer_id][position]),
                                }
                                for tokenizer_id in tokenizers
                            },
                        }
                        index.write(
                            json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
                        )
                        continuation_hasher.update(
                            category_id.encode() + b"\0" + bytes.fromhex(content_sha256)
                        )
                        documents += 1
                        utf8_bytes += result["utf8_bytes"]
                        reference_tokens += result["reference_tokens"]

                batch = []
                characters = 0
                for row, line in enumerate(source):
                    if row <= start_rows[category_id]:
                        continue
                    record = json.loads(line)
                    text = record.get("text")
                    content_sha256 = record.get("released_content_sha256")
                    if (
                        not isinstance(text, str)
                        or not isinstance(content_sha256, str)
                        or hashlib.sha256(text.encode()).hexdigest() != content_sha256
                    ):
                        raise ValueError(f"invalid continuation source record: {category_id}:{row}")
                    batch.append((row, content_sha256, text))
                    characters += len(text)
                    if len(batch) >= 512 or characters >= 1_000_000:
                        write_batch(batch)
                        batch = []
                        characters = 0
                        if reference_tokens >= category["reference_token_target"]:
                            break
                if batch and reference_tokens < category["reference_token_target"]:
                    write_batch(batch)
                index.flush()
                os.fsync(index.fileno())
            if reference_tokens < category["reference_token_target"]:
                raise RuntimeError(f"continuation category exhausted: {category_id}")
            category_results.append(
                {
                    "id": category_id,
                    "start_after_source_row": start_rows[category_id],
                    "documents": documents,
                    "utf8_bytes": utf8_bytes,
                    "reference_token_target": category["reference_token_target"],
                    "reference_tokens": reference_tokens,
                    "reference_overshoot_tokens": reference_tokens
                    - category["reference_token_target"],
                    "tokenizer_tokens": tokenizer_tokens,
                    "index": {
                        "path": index_path.name,
                        "sha256": file_sha256(index_path),
                        "bytes": index_path.stat().st_size,
                        "records": documents,
                    },
                }
            )
        tokenizer_results = []
        for declaration in plan["tokenizers"]:
            tokenizer_id = declaration["id"]
            categories = []
            for category in plan["categories"]:
                writer = writers[tokenizer_id][category["id"]]
                writer.finish()
                categories.append(
                    {
                        "id": category["id"],
                        "tokens": writer.total_tokens,
                        "shards": writer.shards,
                    }
                )
            tokens = sum(category["tokens"] for category in categories)
            if tokens < declaration["minimum_additional_tokens"]:
                raise RuntimeError(f"continuation is too short for {tokenizer_id}")
            tokenizer_results.append(
                {
                    "id": tokenizer_id,
                    "model": declaration["model"],
                    "minimum_additional_tokens": declaration["minimum_additional_tokens"],
                    "tokens": tokens,
                    "categories": categories,
                }
            )
        manifest = {
            "format": "speck_tokenizer_pilot_continuation_result",
            "format_version": FORMAT_VERSION,
            "status": "shared_equal_flop_continuation_complete_not_training_authority",
            "plan_fingerprint": plan["plan_fingerprint"],
            "fixed_stream": plan["fixed_stream"],
            "continuation": {
                "sha256": continuation_hasher.hexdigest(),
                "hash_chain_parent": fixed["document_stream"]["sha256"],
                "reference_tokenizer_id": plan["reference_tokenizer_id"],
                "minimum_reference_tokens": REFERENCE_CONTINUATION_TOKENS,
                "actual_reference_tokens": sum(
                    item["reference_tokens"] for item in category_results
                ),
                "whole_documents": True,
                "categories": category_results,
            },
            "tokenizers": tokenizer_results,
            "gates": {
                "fixed_stream_identity": "pass",
                "source_continuity": "pass",
                "whole_document_selection": "pass",
                "shared_document_identity": "pass",
                "minimum_equal_flop_tokens": "pass",
                "packed_shard_integrity": "pass",
                "model_training_authority": "blocked",
            },
            "model_training_authority": False,
            "final_tokenizer_selection_authority": False,
        }
        atomic_json(staging / "manifest.json", manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output)
        _verify_published(plan, output, manifest)
        return manifest
    except BaseException:
        for by_category in writers.values():
            for writer in by_category.values():
                writer.finish()
        raise
