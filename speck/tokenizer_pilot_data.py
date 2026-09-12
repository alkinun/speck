"""Materialize one whole-document stream under multiple tokenizer identities."""

import hashlib
import json
import os
import shutil
from pathlib import Path

from speck.dataset import TokenShardWriter
from speck.io import atomic_json, file_sha256
from speck.tokenizer import Tokenizer

FORMAT = "speck_tokenizer_pilot_stream"
FORMAT_VERSION = 1
REFERENCE_TOKENS = 1_200_000_000


def _identity(value, root, context):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{context} must contain path and sha256")
    path = Path(value["path"]).expanduser()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file() or file_sha256(path) != value["sha256"]:
        raise ValueError(f"{context} identity mismatch")
    return {"path": str(path), "sha256": value["sha256"]}


def validate_pilot_stream_plan(value, *, config_dir=None):
    root = Path(config_dir or ".").resolve()
    expected = {
        "format",
        "format_version",
        "status",
        "seed",
        "dedup_manifest",
        "reference_tokenizer_id",
        "tokenizers",
        "categories",
        "shard_tokens",
        "output_directory",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("tokenizer pilot stream plan fields are invalid")
    if (
        value["format"] != FORMAT
        or value["format_version"] != FORMAT_VERSION
        or value["status"] != "materialization_authorized_not_model_training_authority"
    ):
        raise ValueError("unsupported tokenizer pilot stream contract")
    if isinstance(value["seed"], bool) or not isinstance(value["seed"], int):
        raise ValueError("tokenizer pilot stream seed must be an integer")
    dedup_manifest = _identity(value["dedup_manifest"], root, "dedup manifest")
    tokenizers = value["tokenizers"]
    if not isinstance(tokenizers, list) or len(tokenizers) < 2:
        raise ValueError("tokenizer pilot stream requires at least two tokenizers")
    normalized_tokenizers = []
    tokenizer_ids = []
    for item in tokenizers:
        if not isinstance(item, dict) or set(item) != {"id", "model"}:
            raise ValueError("tokenizer pilot declaration fields are invalid")
        tokenizer_id = item["id"]
        if (
            not isinstance(tokenizer_id, str)
            or not tokenizer_id
            or Path(tokenizer_id).name != tokenizer_id
        ):
            raise ValueError("tokenizer pilot ID must be a path component")
        tokenizer_ids.append(tokenizer_id)
        normalized_tokenizers.append(
            {"id": tokenizer_id, "model": _identity(item["model"], root, tokenizer_id)}
        )
    if len(tokenizer_ids) != len(set(tokenizer_ids)):
        raise ValueError("tokenizer pilot IDs must be unique")
    if value["reference_tokenizer_id"] not in tokenizer_ids:
        raise ValueError("reference tokenizer is not declared")
    categories = value["categories"]
    expected_categories = ("web", "code", "math", "synthetic", "science", "reference")
    if (
        not isinstance(categories, list)
        or tuple(item.get("id") for item in categories) != expected_categories
    ):
        raise ValueError("tokenizer pilot categories must use the frozen six-category order")
    normalized_categories = []
    for item in categories:
        if not isinstance(item, dict) or set(item) != {
            "id",
            "source_id",
            "input",
            "reference_token_target",
        }:
            raise ValueError("tokenizer pilot category fields are invalid")
        target = item["reference_token_target"]
        if isinstance(target, bool) or not isinstance(target, int) or target < 1:
            raise ValueError("reference token target must be positive")
        normalized_categories.append(
            {**item, "input": _identity(item["input"], root, f"category {item['id']}")}
        )
    if sum(item["reference_token_target"] for item in categories) != REFERENCE_TOKENS:
        raise ValueError("tokenizer pilot reference targets must sum to 1.2B")
    shard_tokens = value["shard_tokens"]
    if isinstance(shard_tokens, bool) or not isinstance(shard_tokens, int) or shard_tokens < 1:
        raise ValueError("tokenizer pilot shard size must be positive")
    output = Path(value["output_directory"]).expanduser()
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    normalized = {
        **value,
        "dedup_manifest": dedup_manifest,
        "tokenizers": normalized_tokenizers,
        "categories": normalized_categories,
        "output_directory": str(output),
    }
    normalized["plan_fingerprint"] = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return normalized


def load_pilot_stream_plan(path):
    path = Path(path).resolve()
    return validate_pilot_stream_plan(json.loads(path.read_text()), config_dir=path.parent)


def _validated_plan(plan):
    if "plan_fingerprint" not in plan:
        return validate_pilot_stream_plan(plan)
    payload = {key: value for key, value in plan.items() if key != "plan_fingerprint"}
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if plan["plan_fingerprint"] != fingerprint:
        raise ValueError("normalized tokenizer pilot stream fingerprint mismatch")
    return plan


def _verify_published(plan, output, manifest):
    if manifest.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise ValueError("published tokenizer pilot stream belongs to another plan")
    for tokenizer in manifest["tokenizers"]:
        for category in tokenizer["categories"]:
            for shard in category["shards"]:
                path = output / tokenizer["id"] / category["id"] / shard["path"]
                if not path.is_file() or file_sha256(path) != shard["sha256"]:
                    raise ValueError("published tokenizer pilot shard identity mismatch")
    for category in manifest["document_stream"]["categories"]:
        path = output / category["index"]["path"]
        if not path.is_file() or file_sha256(path) != category["index"]["sha256"]:
            raise ValueError("published tokenizer pilot document index identity mismatch")


def materialize_pilot_stream(plan, *, restart=False):
    """Select whole documents by reference-token quotas and pack every tokenizer."""

    plan = _validated_plan(plan)
    dedup_manifest = Path(plan["dedup_manifest"]["path"])
    dedup = json.loads(dedup_manifest.read_text())
    if (
        dedup.get("format") != "speck_production_text_preprocess_result"
        or dedup.get("gates", {}).get("global_exact_deduplication") != "pass"
        or dedup.get("gates", {}).get(
            "disk_backed_Minhash_candidates_and_verified_near_deduplication"
        )
        != "pass"
    ):
        raise ValueError("tokenizer pilot dedup manifest is incomplete")
    for category in plan["categories"]:
        entry = dedup.get("outputs", {}).get(category["source_id"])
        expected_path = dedup_manifest.parent / entry.get("path", "") if entry else None
        if (
            entry is None
            or expected_path.resolve() != Path(category["input"]["path"]).resolve()
            or entry.get("sha256") != category["input"]["sha256"]
        ):
            raise ValueError(
                f"tokenizer pilot category differs from dedup output: {category['id']}"
            )
    output = Path(plan["output_directory"])
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        _verify_published(plan, output, manifest)
        return manifest
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete tokenizer pilot stream exists: {staging}")
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
    stream_hasher = hashlib.sha256()
    category_results = []
    try:
        for category in plan["categories"]:
            category_id = category["id"]
            source_path = Path(category["input"]["path"])
            index_path = staging / f"documents-{category_id}.jsonl"
            documents = reference_tokens = utf8_bytes = 0
            tokenizer_tokens = {tokenizer_id: 0 for tokenizer_id in tokenizers}
            with (
                source_path.open(encoding="utf-8") as source,
                index_path.open("w", encoding="utf-8") as index,
            ):

                def write_batch(batch):
                    nonlocal documents, reference_tokens, utf8_bytes
                    texts = [item[2] for item in batch]
                    reference_encoded = reference.encode_batch(texts, bos=True, eos=True)
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
                        stream_hasher.update(
                            category_id.encode() + b"\0" + bytes.fromhex(content_sha256)
                        )
                        documents += 1
                        utf8_bytes += result["utf8_bytes"]
                        reference_tokens += result["reference_tokens"]

                batch = []
                characters = 0
                for row, line in enumerate(source):
                    record = json.loads(line)
                    text = record.get("text")
                    content_sha256 = record.get("released_content_sha256")
                    if (
                        not isinstance(text, str)
                        or not isinstance(content_sha256, str)
                        or hashlib.sha256(text.encode()).hexdigest() != content_sha256
                    ):
                        raise ValueError(
                            f"invalid tokenizer pilot source record: {category_id}:{row}"
                        )
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
                raise RuntimeError(f"tokenizer pilot category exhausted: {category_id}")
            category_results.append(
                {
                    "id": category_id,
                    "source_id": category["source_id"],
                    "source": category["input"],
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
        for tokenizer in plan["tokenizers"]:
            tokenizer_id = tokenizer["id"]
            categories = []
            for category in plan["categories"]:
                category_id = category["id"]
                writer = writers[tokenizer_id][category_id]
                writer.finish()
                categories.append(
                    {
                        "id": category_id,
                        "tokens": writer.total_tokens,
                        "shards": writer.shards,
                    }
                )
            tokenizer_results.append(
                {
                    "id": tokenizer_id,
                    "model": tokenizer["model"],
                    "tokens": sum(category["tokens"] for category in categories),
                    "categories": categories,
                }
            )
        actual_reference_tokens = sum(category["reference_tokens"] for category in category_results)
        manifest = {
            "format": "speck_tokenizer_pilot_stream_result",
            "format_version": FORMAT_VERSION,
            "status": "whole_document_stream_and_tokenizer_packs_complete_not_training_authority",
            "plan_fingerprint": plan["plan_fingerprint"],
            "seed": plan["seed"],
            "dedup_manifest": plan["dedup_manifest"],
            "document_stream": {
                "sha256": stream_hasher.hexdigest(),
                "order": "frozen category order then retained source row order",
                "reference_tokenizer_id": plan["reference_tokenizer_id"],
                "minimum_reference_tokens": REFERENCE_TOKENS,
                "actual_reference_tokens": actual_reference_tokens,
                "whole_documents": True,
                "categories": category_results,
            },
            "tokenizers": tokenizer_results,
            "gates": {
                "input_identity": "pass",
                "whole_document_selection": "pass",
                "six_category_reference_targets": "pass",
                "shared_document_identity": "pass",
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
