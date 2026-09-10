"""Build, train, and evaluate reproducible SentencePiece tokenizer candidates."""

import gzip
import hashlib
import json
import math
import os
import re
import shutil
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import sentencepiece as sentencepiece
from huggingface_hub import hf_hub_download
from sentencepiece import sentencepiece_model_pb2

from speck.data_firewall import authorize_consumer
from speck.tokenizer import Tokenizer

FORMAT = "speck_tokenizer_experiment"
FORMAT_VERSION = 1
SAMPLE_FORMAT = "speck_tokenizer_sample"
TRAINING_FORMAT = "speck_tokenizer_candidate"
EVALUATION_FORMAT = "speck_tokenizer_evaluation"
_INPUT_FORMATS = {"jsonl", "jsonl_gzip", "parquet", "text"}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SPECIAL_IDS = {"unk_id": 0, "bos_id": 1, "eos_id": 2, "pad_id": -1}


def _sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{name} must be a single non-empty path component")
    return value


def _exact_keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly: {', '.join(sorted(expected))}")


def _resolve_path(path, config_dir):
    path = Path(path).expanduser()
    return (config_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _validate_sample(sample, config_dir):
    if not isinstance(sample, dict):
        raise ValueError("sample must be an object")
    training_bytes = _integer(
        sample.get("training_bytes_per_category"), "training_bytes_per_category", 1
    )
    evaluation_bytes = _integer(
        sample.get("evaluation_bytes_per_category"), "evaluation_bytes_per_category", 1
    )
    min_chars = _integer(sample.get("min_chars"), "sample.min_chars", 1)
    max_chars = _integer(sample.get("max_chars"), "sample.max_chars", 1)
    if min_chars > max_chars:
        raise ValueError("sample.min_chars cannot exceed sample.max_chars")
    if sample.get("kind") == "production_firewall":
        _exact_keys(
            sample,
            {
                "kind",
                "training_bytes_per_category",
                "evaluation_bytes_per_category",
                "min_chars",
                "max_chars",
                "firewall_manifest",
                "firewall_manifest_sha256",
                "categories",
            },
            "production firewall sample",
        )
        manifest_path = _resolve_path(sample["firewall_manifest"], config_dir)
        if not isinstance(sample["firewall_manifest_sha256"], str) or not _SHA256.fullmatch(
            sample["firewall_manifest_sha256"]
        ):
            raise ValueError("firewall manifest sha256 must be lowercase hexadecimal")
        categories = sample["categories"]
        if not isinstance(categories, list) or not categories:
            raise ValueError("sample.categories must be a non-empty list")
        normalized_categories = []
        category_ids = []
        for category_index, category in enumerate(categories):
            _exact_keys(category, {"id", "train", "eval"}, f"category {category_index}")
            category_id = _identifier(category["id"], f"category {category_index} id")
            category_ids.append(category_id)
            normalized = {"id": category_id}
            for split in ("train", "eval"):
                item = category[split]
                _exact_keys(
                    item,
                    {"path", "sha256", "format", "text_column"},
                    f"category {category_id} {split}",
                )
                if item["format"] != "jsonl" or not isinstance(item["text_column"], str):
                    raise ValueError(
                        "firewall tokenizer partitions must be JSONL with a text column"
                    )
                if not isinstance(item["sha256"], str) or not _SHA256.fullmatch(item["sha256"]):
                    raise ValueError(
                        "firewall tokenizer partition sha256 must be lowercase hexadecimal"
                    )
                normalized[split] = {**item, "path": str(_resolve_path(item["path"], config_dir))}
            normalized_categories.append(normalized)
        if len(category_ids) != len(set(category_ids)):
            raise ValueError("category IDs must be unique")
        return {
            "kind": "production_firewall",
            "training_bytes_per_category": training_bytes,
            "evaluation_bytes_per_category": evaluation_bytes,
            "min_chars": min_chars,
            "max_chars": max_chars,
            "firewall_manifest": str(manifest_path),
            "firewall_manifest_sha256": sample["firewall_manifest_sha256"],
            "categories": normalized_categories,
        }

    _exact_keys(
        sample,
        {
            "training_bytes_per_category",
            "evaluation_bytes_per_category",
            "evaluation_modulus",
            "evaluation_remainders",
            "min_chars",
            "max_chars",
            "categories",
        },
        "sample",
    )
    modulus = _integer(sample["evaluation_modulus"], "evaluation_modulus", 2)
    remainders = sample["evaluation_remainders"]
    if (
        not isinstance(remainders, list)
        or not remainders
        or any(isinstance(value, bool) or not isinstance(value, int) for value in remainders)
        or len(remainders) != len(set(remainders))
        or any(value < 0 or value >= modulus for value in remainders)
        or len(remainders) == modulus
    ):
        raise ValueError("evaluation_remainders must be unique values inside evaluation_modulus")
    categories = sample["categories"]
    if not isinstance(categories, list) or not categories:
        raise ValueError("sample.categories must be a non-empty list")
    normalized_categories = []
    category_ids = []
    for category_index, category in enumerate(categories):
        _exact_keys(category, {"id", "inputs"}, f"category {category_index}")
        category_id = _identifier(category["id"], f"category {category_index} id")
        category_ids.append(category_id)
        inputs = category["inputs"]
        if not isinstance(inputs, list) or not inputs:
            raise ValueError(f"category {category_id} inputs must be a non-empty list")
        normalized_inputs = []
        input_ids = []
        for input_index, item in enumerate(inputs):
            _exact_keys(
                item,
                {
                    "id",
                    "path",
                    "format",
                    "text_column",
                    "sha256",
                    "training_bytes",
                    "evaluation_bytes",
                },
                f"category {category_id} input {input_index}",
            )
            input_id = _identifier(item["id"], f"category {category_id} input id")
            input_ids.append(input_id)
            if item["format"] not in _INPUT_FORMATS:
                raise ValueError(f"input {input_id} has unsupported format")
            text_column = item["text_column"]
            if item["format"] == "text":
                if text_column is not None:
                    raise ValueError(f"text input {input_id} must use null text_column")
            elif not isinstance(text_column, str) or not text_column:
                raise ValueError(f"input {input_id} requires a text_column")
            if not isinstance(item["path"], str) or not item["path"]:
                raise ValueError(f"input {input_id} path must be a non-empty string")
            if not isinstance(item["sha256"], str) or not _SHA256.fullmatch(item["sha256"]):
                raise ValueError(f"input {input_id} sha256 must be lowercase hexadecimal")
            input_training_bytes = _integer(
                item["training_bytes"], f"input {input_id} training_bytes", 1
            )
            input_evaluation_bytes = _integer(
                item["evaluation_bytes"], f"input {input_id} evaluation_bytes", 1
            )
            normalized_inputs.append(
                {
                    **item,
                    "path": str(_resolve_path(item["path"], config_dir)),
                    "training_bytes": input_training_bytes,
                    "evaluation_bytes": input_evaluation_bytes,
                }
            )
        if len(input_ids) != len(set(input_ids)):
            raise ValueError(f"category {category_id} input IDs must be unique")
        if sum(item["training_bytes"] for item in normalized_inputs) != training_bytes:
            raise ValueError(f"category {category_id} input training bytes must sum to its target")
        if sum(item["evaluation_bytes"] for item in normalized_inputs) != evaluation_bytes:
            raise ValueError(
                f"category {category_id} input evaluation bytes must sum to its target"
            )
        normalized_categories.append({"id": category_id, "inputs": normalized_inputs})
    if len(category_ids) != len(set(category_ids)):
        raise ValueError("category IDs must be unique")
    return {
        "training_bytes_per_category": training_bytes,
        "evaluation_bytes_per_category": evaluation_bytes,
        "evaluation_modulus": modulus,
        "evaluation_remainders": sorted(remainders),
        "min_chars": min_chars,
        "max_chars": max_chars,
        "categories": normalized_categories,
    }


def validate_experiment_config(config, *, config_dir=None):
    """Validate and normalize one executable tokenizer experiment configuration."""

    config_dir = Path(config_dir or ".").resolve()
    top_level = {
        "format",
        "format_version",
        "seed",
        "output_dir",
        "sample",
        "trainer",
        "candidates",
        "baselines",
        "evaluation",
    }
    _exact_keys(config, top_level, "tokenizer experiment")
    if config["format"] != FORMAT or config["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported tokenizer experiment format")
    seed = _integer(config["seed"], "seed")
    if not isinstance(config["output_dir"], str) or not config["output_dir"]:
        raise ValueError("output_dir must be a non-empty path")
    output_dir = _resolve_path(config["output_dir"], config_dir)

    normalized_sample = _validate_sample(config["sample"], config_dir)

    trainer = config["trainer"]
    trainer_keys = {
        "model_type",
        "character_coverage",
        "byte_fallback",
        "normalization_rule_name",
        "remove_extra_whitespaces",
        "add_dummy_prefix",
        "split_digits",
        "split_by_unicode_script",
        "split_by_whitespace",
        "split_by_number",
        "max_sentence_length",
        "num_threads",
        "hard_vocab_limit",
    }
    optional_trainer_keys = {"allow_whitespace_only_pieces"}
    if (
        not isinstance(trainer, dict)
        or not trainer_keys <= set(trainer)
        or set(trainer) - trainer_keys - optional_trainer_keys
    ):
        raise ValueError(
            "trainer must contain the frozen keys and only supported optional settings"
        )
    if trainer["model_type"] not in {"bpe", "unigram"}:
        raise ValueError("trainer.model_type must be bpe or unigram")
    coverage = trainer["character_coverage"]
    if (
        isinstance(coverage, bool)
        or not isinstance(coverage, (int, float))
        or not 0.98 <= coverage <= 1
    ):
        raise ValueError("trainer.character_coverage must be in [0.98, 1]")
    for name in trainer_keys - {
        "model_type",
        "character_coverage",
        "normalization_rule_name",
        "max_sentence_length",
        "num_threads",
    }:
        if not isinstance(trainer[name], bool):
            raise ValueError(f"trainer.{name} must be boolean")
    if "allow_whitespace_only_pieces" in trainer and not isinstance(
        trainer["allow_whitespace_only_pieces"], bool
    ):
        raise ValueError("trainer.allow_whitespace_only_pieces must be boolean")
    if (
        not isinstance(trainer["normalization_rule_name"], str)
        or not trainer["normalization_rule_name"]
    ):
        raise ValueError("trainer.normalization_rule_name must be non-empty")
    _integer(trainer["max_sentence_length"], "trainer.max_sentence_length", 1)
    _integer(trainer["num_threads"], "trainer.num_threads", 1)

    candidates = config["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list")
    normalized_candidates = []
    candidate_ids = []
    for index, candidate in enumerate(candidates):
        _exact_keys(candidate, {"id", "vocab_size"}, f"candidate {index}")
        candidate_id = _identifier(candidate["id"], f"candidate {index} id")
        vocab_size = _integer(candidate["vocab_size"], f"candidate {candidate_id} vocab_size", 259)
        candidate_ids.append(candidate_id)
        normalized_candidates.append({"id": candidate_id, "vocab_size": vocab_size})
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be unique")

    baselines = config["baselines"]
    if not isinstance(baselines, list):
        raise ValueError("baselines must be a list")
    normalized_baselines = []
    baseline_ids = []
    for index, baseline in enumerate(baselines):
        required_baseline = {"id", "repo", "revision", "filename", "expected_vocab_size"}
        if (
            not isinstance(baseline, dict)
            or not required_baseline <= set(baseline)
            or set(baseline) - required_baseline - {"sha256"}
        ):
            raise ValueError(f"baseline {index} fields are invalid")
        baseline_id = _identifier(baseline["id"], f"baseline {index} id")
        baseline_ids.append(baseline_id)
        for key in ("repo", "revision", "filename"):
            if not isinstance(baseline[key], str) or not baseline[key]:
                raise ValueError(f"baseline {baseline_id} {key} must be non-empty")
        if "sha256" in baseline and (
            not isinstance(baseline["sha256"], str) or not _SHA256.fullmatch(baseline["sha256"])
        ):
            raise ValueError(f"baseline {baseline_id} sha256 must be lowercase hexadecimal")
        normalized_baselines.append(
            {
                **baseline,
                "expected_vocab_size": _integer(
                    baseline["expected_vocab_size"],
                    f"baseline {baseline_id} expected_vocab_size",
                    1,
                ),
            }
        )
    if len(baseline_ids) != len(set(baseline_ids)) or set(baseline_ids) & set(candidate_ids):
        raise ValueError("baseline and candidate IDs must be globally unique")

    evaluation = config["evaluation"]
    _exact_keys(
        evaluation,
        {"embedding_width", "tied_embeddings", "chat_added_tokens", "probe_strings"},
        "evaluation",
    )
    embedding_width = _integer(evaluation["embedding_width"], "embedding_width", 1)
    chat_added_tokens = _integer(evaluation["chat_added_tokens"], "chat_added_tokens")
    if not isinstance(evaluation["tied_embeddings"], bool):
        raise ValueError("evaluation.tied_embeddings must be boolean")
    probes = evaluation["probe_strings"]
    if not isinstance(probes, list) or not probes or any(not isinstance(x, str) for x in probes):
        raise ValueError("evaluation.probe_strings must be a non-empty string list")
    for tokenizer_id, vocab_size in [
        *((item["id"], item["vocab_size"]) for item in normalized_candidates),
        *((item["id"], item["expected_vocab_size"]) for item in normalized_baselines),
    ]:
        if vocab_size + chat_added_tokens > 65536:
            raise ValueError(f"tokenizer {tokenizer_id} plus chat tokens exceeds uint16 capacity")

    normalized = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "seed": seed,
        "output_dir": str(output_dir),
        "sample": normalized_sample,
        "trainer": dict(trainer),
        "candidates": normalized_candidates,
        "baselines": normalized_baselines,
        "evaluation": {
            "embedding_width": embedding_width,
            "tied_embeddings": evaluation["tied_embeddings"],
            "chat_added_tokens": chat_added_tokens,
            "probe_strings": list(probes),
        },
    }
    normalized["plan_fingerprint"] = _fingerprint(normalized)
    return normalized


def load_experiment_config(path):
    """Load a JSON tokenizer experiment relative to its own directory."""

    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        return validate_experiment_config(json.load(handle), config_dir=path.parent)


def _validated_config(config):
    if "plan_fingerprint" not in config:
        return validate_experiment_config(config)
    payload = {key: value for key, value in config.items() if key != "plan_fingerprint"}
    if config["plan_fingerprint"] != _fingerprint(payload):
        raise ValueError("normalized tokenizer experiment fingerprint mismatch")
    return config


def _iter_jsonl(path, text_column, *, compressed=False):
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path} row {row}") from error
            if not isinstance(value, dict) or text_column not in value:
                raise ValueError(f"missing {text_column} in {path} row {row}")
            text = value[text_column]
            if not isinstance(text, str):
                raise ValueError(f"non-string {text_column} in {path} row {row}")
            yield row, text


def _iter_parquet(path, text_column):
    parquet = pq.ParquetFile(path)
    if text_column not in parquet.schema_arrow.names:
        raise ValueError(f"missing {text_column} in {path}")
    field = parquet.schema_arrow.field(text_column)
    if not (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)):
        raise ValueError(f"non-string {text_column} in {path}")
    row = 0
    for batch in parquet.iter_batches(columns=[text_column], batch_size=2048):
        for text in batch.column(0).to_pylist():
            if text is not None:
                yield row, text
            row += 1


def _iter_input(item):
    path = Path(item["path"])
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer input not found: {path}")
    actual_hash = _sha256(path)
    if actual_hash != item["sha256"]:
        raise ValueError(f"tokenizer input checksum mismatch: {path}")
    if item["format"] == "text":
        with path.open(encoding="utf-8") as handle:
            for row, line in enumerate(handle):
                yield row, line.rstrip("\n")
        return
    if item["format"] == "jsonl":
        yield from _iter_jsonl(path, item["text_column"])
        return
    if item["format"] == "jsonl_gzip":
        yield from _iter_jsonl(path, item["text_column"], compressed=True)
        return
    if item["format"] == "parquet":
        yield from _iter_parquet(path, item["text_column"])
        return
    raise ValueError(f"unsupported tokenizer input format: {item['format']}")


def _dedup_digest(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _partition(config, category_id, content_hash):
    payload = f"{config['seed']}\0{category_id}\0{content_hash}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    sample = config["sample"]
    return (
        "eval"
        if value % sample["evaluation_modulus"] in sample["evaluation_remainders"]
        else "train"
    )


def _prepare_firewall_sample(config, *, restart=False):
    sample = config["sample"]
    manifest_path = Path(sample["firewall_manifest"])
    if not manifest_path.is_file() or _sha256(manifest_path) != sample["firewall_manifest_sha256"]:
        raise ValueError("tokenizer firewall manifest identity mismatch")
    firewall = json.loads(manifest_path.read_text())
    required_gates = {
        "input_identity",
        "global_content_disjointness",
        "equal_category_targets",
        "sealed_audits_unopened",
    }
    if (
        firewall.get("format") != "speck_data_firewall_manifest"
        or firewall.get("status") != "production_firewall_complete_rights_and_operations_bound"
        or firewall.get("authority", {}).get("mode") != "production"
        or any(firewall.get("gates", {}).get(gate) != "pass" for gate in required_gates)
    ):
        raise ValueError("tokenizer production firewall is incomplete")
    firewall_categories = {category["id"]: category for category in firewall["categories"]}
    declared_ids = [category["id"] for category in sample["categories"]]
    if declared_ids != [category["id"] for category in firewall["categories"]]:
        raise ValueError("tokenizer categories differ from the production firewall")
    root = Path(config["output_dir"])
    output = root / "sample"
    staging = root / "sample.building"
    if output.exists():
        raise FileExistsError(f"tokenizer sample already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete tokenizer sample exists: {staging}; pass --restart")
        shutil.rmtree(staging)
    destinations = {"train": "tokenizer_train", "eval": "tokenizer_eval"}
    paths = {split: [] for split in destinations}
    for category in sample["categories"]:
        firewall_category = firewall_categories[category["id"]]
        for split, destination in destinations.items():
            entry = firewall_category["outputs"][destination]
            item = category[split]
            path = Path(item["path"])
            if (
                path.resolve() != (manifest_path.parent / entry["path"]).resolve()
                or item["sha256"] != entry["sha256"]
            ):
                raise ValueError(f"tokenizer {category['id']} {split} differs from firewall")
            paths[split].append(path)
    authorize_consumer(manifest_path, "tokenizer_training", paths["train"])
    authorize_consumer(manifest_path, "tokenizer_static_evaluation", paths["eval"])
    staging.mkdir(parents=True)
    category_manifests = []
    try:
        for category in sample["categories"]:
            category_id = category["id"]
            splits = {}
            for split, destination in destinations.items():
                item = category[split]
                source_path = Path(item["path"])
                output_path = staging / f"{split}-{category_id}.jsonl"
                documents = utf8_bytes = 0
                with (
                    source_path.open(encoding="utf-8") as source,
                    output_path.open("w", encoding="utf-8") as target,
                ):
                    for row, line in enumerate(source):
                        record = json.loads(line)
                        text = record.get(item["text_column"])
                        if (
                            not isinstance(text, str)
                            or not text.strip()
                            or not sample["min_chars"] <= len(text) <= sample["max_chars"]
                            or record.get("category") != category_id
                            or record.get("partition_detail") != destination
                        ):
                            raise ValueError(
                                f"invalid tokenizer firewall record: {category_id}:{split}:{row}"
                            )
                        target.write(line)
                        documents += 1
                        utf8_bytes += len(text.encode())
                    target.flush()
                    os.fsync(target.fileno())
                required = sample[
                    "training_bytes_per_category"
                    if split == "train"
                    else "evaluation_bytes_per_category"
                ]
                firewall_entry = firewall_categories[category_id]["outputs"][destination]
                if (
                    utf8_bytes < required
                    or utf8_bytes != firewall_entry["utf8_bytes"]
                    or documents != firewall_entry["documents"]
                ):
                    raise ValueError(
                        f"tokenizer firewall statistics mismatch: {category_id}:{split}"
                    )
                splits[split] = {
                    "path": output_path.name,
                    "sha256": _sha256(output_path),
                    "file_bytes": output_path.stat().st_size,
                    "documents": documents,
                    "utf8_bytes": utf8_bytes,
                    "target_utf8_bytes": required,
                    "overshoot_bytes": utf8_bytes - required,
                    "firewall_source": item,
                }
            category_manifests.append({"id": category_id, "splits": splits})
        manifest = {
            "format": SAMPLE_FORMAT,
            "format_version": FORMAT_VERSION,
            "plan_fingerprint": config["plan_fingerprint"],
            "seed": config["seed"],
            "dedup": {
                "normalization": "NFKC+lower+whitespace",
                "scope": "inherited_global_production_firewall",
            },
            "partition": {
                "kind": "production_firewall",
                "manifest": str(manifest_path),
                "manifest_sha256": sample["firewall_manifest_sha256"],
            },
            "categories": category_manifests,
        }
        _write_json(staging / "manifest.json", manifest)
        root.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
        return manifest
    except Exception:
        raise


def prepare_sample(config, *, restart=False):
    """Build a deterministic, category-balanced, disjoint tokenizer sample."""

    config = _validated_config(config)
    if config["sample"].get("kind") == "production_firewall":
        return _prepare_firewall_sample(config, restart=restart)
    root = Path(config["output_dir"])
    output = root / "sample"
    staging = root / "sample.building"
    if output.exists():
        raise FileExistsError(f"tokenizer sample already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(f"incomplete tokenizer sample exists: {staging}; pass --restart")
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    targets = {
        "train": config["sample"]["training_bytes_per_category"],
        "eval": config["sample"]["evaluation_bytes_per_category"],
    }
    seen = set()
    category_manifests = []
    try:
        for category in config["sample"]["categories"]:
            category_id = category["id"]
            handles = {
                split: (staging / f"{split}-{category_id}.jsonl").open("w", encoding="utf-8")
                for split in ("train", "eval")
            }
            stats = {
                split: {"documents": 0, "utf8_bytes": 0, "overshoot_bytes": 0} for split in handles
            }
            rejected = {"duplicate": 0, "length": 0, "filled_partition": 0}
            input_manifests = []
            try:
                for item in category["inputs"]:
                    input_targets = {
                        "train": item["training_bytes"],
                        "eval": item["evaluation_bytes"],
                    }
                    input_stats = {
                        split: {"documents": 0, "utf8_bytes": 0, "overshoot_bytes": 0}
                        for split in handles
                    }
                    input_rejected = {"duplicate": 0, "length": 0, "filled_partition": 0}
                    for row, text in _iter_input(item):
                        if all(
                            input_stats[split]["utf8_bytes"] >= input_targets[split]
                            for split in input_stats
                        ):
                            break
                        if (
                            not text.strip()
                            or not config["sample"]["min_chars"]
                            <= len(text)
                            <= config["sample"]["max_chars"]
                        ):
                            rejected["length"] += 1
                            input_rejected["length"] += 1
                            continue
                        digest = _dedup_digest(text)
                        if digest in seen:
                            rejected["duplicate"] += 1
                            input_rejected["duplicate"] += 1
                            continue
                        split = _partition(config, category_id, digest)
                        if input_stats[split]["utf8_bytes"] >= input_targets[split]:
                            rejected["filled_partition"] += 1
                            input_rejected["filled_partition"] += 1
                            continue
                        seen.add(digest)
                        encoded_bytes = len(text.encode("utf-8"))
                        record = {
                            "category": category_id,
                            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "input": item["id"],
                            "row": row,
                            "text": text,
                        }
                        handles[split].write(
                            json.dumps(
                                record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                            )
                            + "\n"
                        )
                        stats[split]["documents"] += 1
                        stats[split]["utf8_bytes"] += encoded_bytes
                        input_stats[split]["documents"] += 1
                        input_stats[split]["utf8_bytes"] += encoded_bytes
                    missing = [
                        split
                        for split in input_stats
                        if input_stats[split]["utf8_bytes"] < input_targets[split]
                    ]
                    if missing:
                        values = ", ".join(
                            f"{split}={input_stats[split]['utf8_bytes']:,}/"
                            f"{input_targets[split]:,} bytes"
                            for split in missing
                        )
                        raise RuntimeError(
                            f"input {item['id']} exhausted before sample target: {values}"
                        )
                    for split in input_stats:
                        input_stats[split]["overshoot_bytes"] = (
                            input_stats[split]["utf8_bytes"] - input_targets[split]
                        )
                    input_manifests.append(
                        {
                            **item,
                            "rejected": input_rejected,
                            "splits": input_stats,
                        }
                    )
            finally:
                for handle in handles.values():
                    handle.flush()
                    os.fsync(handle.fileno())
                    handle.close()
            missing = [split for split in stats if stats[split]["utf8_bytes"] < targets[split]]
            if missing:
                values = ", ".join(
                    f"{split}={stats[split]['utf8_bytes']:,}/{targets[split]:,} bytes"
                    for split in missing
                )
                raise RuntimeError(
                    f"category {category_id} exhausted before sample target: {values}"
                )
            split_manifests = {}
            for split in ("train", "eval"):
                path = staging / f"{split}-{category_id}.jsonl"
                stats[split]["overshoot_bytes"] = stats[split]["utf8_bytes"] - targets[split]
                split_manifests[split] = {
                    **stats[split],
                    "path": path.name,
                    "file_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "target_utf8_bytes": targets[split],
                }
            category_manifests.append(
                {
                    "id": category_id,
                    "inputs": input_manifests,
                    "rejected": rejected,
                    "splits": split_manifests,
                }
            )
        manifest = {
            "format": SAMPLE_FORMAT,
            "format_version": FORMAT_VERSION,
            "plan_fingerprint": config["plan_fingerprint"],
            "seed": config["seed"],
            "dedup": {"normalization": "NFKC+lower+whitespace", "scope": "global"},
            "partition": {
                "modulus": config["sample"]["evaluation_modulus"],
                "evaluation_remainders": config["sample"]["evaluation_remainders"],
            },
            "categories": category_manifests,
        }
        _write_json(staging / "manifest.json", manifest)
        root.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
        return manifest
    except Exception:
        raise


def _load_sample(config):
    sample_dir = Path(config["output_dir"]) / "sample"
    manifest_path = sample_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"tokenizer sample is not prepared: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != SAMPLE_FORMAT
        or manifest.get("format_version") != FORMAT_VERSION
        or manifest.get("plan_fingerprint") != config["plan_fingerprint"]
    ):
        raise ValueError("tokenizer sample does not match the experiment plan")
    for category in manifest["categories"]:
        for split in ("train", "eval"):
            entry = category["splits"][split]
            path = sample_dir / entry["path"]
            if not path.is_file() or _sha256(path) != entry["sha256"]:
                raise ValueError(f"tokenizer sample checksum mismatch: {path}")
    return sample_dir, manifest


def _iter_sample_texts(sample_dir, sample_manifest, split):
    for category in sample_manifest["categories"]:
        path = sample_dir / category["splits"][split]["path"]
        with path.open(encoding="utf-8") as handle:
            for row, line in enumerate(handle):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid tokenizer sample JSON in {path} row {row}"
                    ) from error
                if record.get("category") != category["id"] or not isinstance(
                    record.get("text"), str
                ):
                    raise ValueError(f"invalid tokenizer sample record in {path} row {row}")
                yield record["text"]


def _canonicalize_sentencepiece_model(path):
    model = sentencepiece_model_pb2.ModelProto()
    model.ParseFromString(Path(path).read_bytes())
    model.trainer_spec.model_prefix = "tokenizer"
    del model.trainer_spec.input[:]
    Path(path).write_bytes(model.SerializeToString(deterministic=True))


def train_candidate(config, candidate_id, *, restart=False):
    """Train one path-independent SentencePiece candidate from the prepared sample."""

    config = _validated_config(config)
    candidates = {candidate["id"]: candidate for candidate in config["candidates"]}
    if candidate_id not in candidates:
        raise ValueError(f"unknown tokenizer candidate: {candidate_id}")
    candidate = candidates[candidate_id]
    sample_dir, sample_manifest = _load_sample(config)
    root = Path(config["output_dir"]) / "candidates"
    output = root / candidate_id
    staging = root / f"{candidate_id}.building"
    if output.exists():
        raise FileExistsError(f"tokenizer candidate already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(
                f"incomplete tokenizer candidate exists: {staging}; pass --restart"
            )
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    model_prefix = staging / "tokenizer"
    settings = {
        **config["trainer"],
        **_SPECIAL_IDS,
        "unk_piece": "<unk>",
        "bos_piece": "<s>",
        "eos_piece": "</s>",
        "vocab_size": candidate["vocab_size"],
        "input_sentence_size": 0,
        "shuffle_input_sentence": False,
        "train_extremely_large_corpus": True,
        "minloglevel": 1,
    }
    try:
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=_iter_sample_texts(sample_dir, sample_manifest, "train"),
            model_prefix=str(model_prefix),
            **settings,
        )
        model_path = model_prefix.with_suffix(".model")
        vocab_path = model_prefix.with_suffix(".vocab")
        _canonicalize_sentencepiece_model(model_path)
        tokenizer = Tokenizer(model_path)
        if (
            config["trainer"]["hard_vocab_limit"]
            and tokenizer.vocab_size != candidate["vocab_size"]
        ):
            raise ValueError("trained tokenizer did not produce the requested vocabulary size")
        if (tokenizer.unk_id, tokenizer.bos_id, tokenizer.eos_id) != (0, 1, 2):
            raise ValueError("trained tokenizer special IDs do not match the Speck contract")
        manifest = {
            "format": TRAINING_FORMAT,
            "format_version": FORMAT_VERSION,
            "id": candidate_id,
            "plan_fingerprint": config["plan_fingerprint"],
            "sample_manifest_sha256": _sha256(sample_dir / "manifest.json"),
            "sentencepiece_version": sentencepiece.__version__,
            "trainer": settings,
            "model": {
                "path": model_path.name,
                "sha256": _sha256(model_path),
                "vocab_size": tokenizer.vocab_size,
                "unk_id": tokenizer.unk_id,
                "bos_id": tokenizer.bos_id,
                "eos_id": tokenizer.eos_id,
            },
            "vocab": {"path": vocab_path.name, "sha256": _sha256(vocab_path)},
        }
        _write_json(staging / "manifest.json", manifest)
        root.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
        return manifest
    except Exception:
        raise


def prepare_baseline(config, baseline_id, *, restart=False):
    """Download and pin one declared baseline tokenizer into the experiment."""

    config = _validated_config(config)
    baselines = {baseline["id"]: baseline for baseline in config["baselines"]}
    if baseline_id not in baselines:
        raise ValueError(f"unknown tokenizer baseline: {baseline_id}")
    baseline = baselines[baseline_id]
    root = Path(config["output_dir"]) / "baselines"
    output = root / baseline_id
    staging = root / f"{baseline_id}.building"
    if output.exists():
        raise FileExistsError(f"tokenizer baseline already exists: {output}")
    if staging.exists():
        if not restart:
            raise FileExistsError(
                f"incomplete tokenizer baseline exists: {staging}; pass --restart"
            )
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        downloaded = hf_hub_download(
            repo_id=baseline["repo"],
            filename=baseline["filename"],
            revision=baseline["revision"],
            local_dir=staging,
        )
        model_path = Path(downloaded)
        if "sha256" in baseline and _sha256(model_path) != baseline["sha256"]:
            raise ValueError(f"baseline {baseline_id} checksum mismatch")
        tokenizer = Tokenizer(model_path)
        if tokenizer.vocab_size != baseline["expected_vocab_size"]:
            raise ValueError(f"baseline {baseline_id} vocabulary size mismatch")
        shutil.rmtree(staging / ".cache", ignore_errors=True)
        manifest = {
            "format": TRAINING_FORMAT,
            "format_version": FORMAT_VERSION,
            "id": baseline_id,
            "kind": "downloaded_baseline",
            "plan_fingerprint": config["plan_fingerprint"],
            "repo": baseline["repo"],
            "revision": baseline["revision"],
            "model": {
                "path": baseline["filename"],
                "sha256": _sha256(model_path),
                "vocab_size": tokenizer.vocab_size,
                "unk_id": tokenizer.unk_id,
                "bos_id": tokenizer.bos_id,
                "eos_id": tokenizer.eos_id,
            },
        }
        _write_json(staging / "manifest.json", manifest)
        root.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
        return manifest
    except Exception:
        raise


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _load_tokenizers(config):
    root = Path(config["output_dir"])
    declarations = [
        *(("candidate", item) for item in config["candidates"]),
        *(("baseline", item) for item in config["baselines"]),
    ]
    loaded = []
    for kind, declaration in declarations:
        directory = (
            root / ("candidates" if kind == "candidate" else "baselines") / declaration["id"]
        )
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"tokenizer {declaration['id']} is not prepared")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("format") != TRAINING_FORMAT
            or manifest.get("format_version") != FORMAT_VERSION
            or manifest.get("id") != declaration["id"]
            or manifest.get("plan_fingerprint") != config["plan_fingerprint"]
        ):
            raise ValueError(f"tokenizer {declaration['id']} does not match the experiment plan")
        model = manifest["model"]
        model_path = directory / model["path"]
        if not model_path.is_file() or _sha256(model_path) != model["sha256"]:
            raise ValueError(f"tokenizer {declaration['id']} model checksum mismatch")
        tokenizer = Tokenizer(model_path)
        expected_vocab = (
            declaration["vocab_size"]
            if kind == "candidate" and config["trainer"]["hard_vocab_limit"]
            else declaration.get("expected_vocab_size")
        )
        if model.get("vocab_size") != tokenizer.vocab_size or (
            expected_vocab is not None and tokenizer.vocab_size != expected_vocab
        ):
            raise ValueError(f"tokenizer {declaration['id']} vocabulary identity mismatch")
        loaded.append((kind, declaration["id"], tokenizer, manifest))
    return loaded


def _summarize_tokenizer(tokenizer, texts):
    documents = utf8_bytes = characters = words = tokens = unknown_tokens = 0
    ratios = []
    roundtrip_mismatches = []
    maximum_token_id = -1
    for text in texts:
        token_ids = tokenizer.encode(text)
        size = len(text.encode("utf-8"))
        documents += 1
        utf8_bytes += size
        characters += len(text)
        words += len(text.split())
        tokens += len(token_ids)
        unknown_tokens += sum(token_id == tokenizer.unk_id for token_id in token_ids)
        if token_ids:
            maximum_token_id = max(maximum_token_id, max(token_ids))
        ratios.append(1024 * len(token_ids) / size)
        if tokenizer.decode(token_ids) != text and len(roundtrip_mismatches) < 20:
            roundtrip_mismatches.append(hashlib.sha256(text.encode()).hexdigest())
    return {
        "documents": documents,
        "utf8_bytes": utf8_bytes,
        "characters": characters,
        "whitespace_words": words,
        "tokens": tokens,
        "unknown_tokens": unknown_tokens,
        "unknown_token_rate": unknown_tokens / tokens if tokens else None,
        "bytes_per_token": utf8_bytes / tokens if tokens else None,
        "tokens_per_word": tokens / words if words else None,
        "tokens_per_kib": 1024 * tokens / utf8_bytes if utf8_bytes else None,
        "document_tokens_per_kib_p50": _percentile(ratios, 0.50),
        "document_tokens_per_kib_p90": _percentile(ratios, 0.90),
        "document_tokens_per_kib_p99": _percentile(ratios, 0.99),
        "maximum_token_id": maximum_token_id,
        "roundtrip_mismatch_count_capped": len(roundtrip_mismatches),
        "roundtrip_mismatch_sha256_first_20": roundtrip_mismatches,
    }


def evaluate_tokenizers(config):
    """Evaluate every prepared baseline and candidate on the disjoint sample split."""

    config = _validated_config(config)
    sample_dir, sample_manifest = _load_sample(config)
    loaded = _load_tokenizers(config)
    category_ids = [category["id"] for category in sample_manifest["categories"]]
    results = []
    parameter_multiplier = 1 if config["evaluation"]["tied_embeddings"] else 2
    for kind, tokenizer_id, tokenizer, manifest in loaded:
        per_category = {}
        for category in sample_manifest["categories"]:
            per_category[category["id"]] = _summarize_tokenizer(
                tokenizer,
                _iter_sample_texts(
                    sample_dir,
                    {"categories": [category]},
                    "eval",
                ),
            )
        probes = _summarize_tokenizer(tokenizer, config["evaluation"]["probe_strings"])
        total_vocab = tokenizer.vocab_size + config["evaluation"]["chat_added_tokens"]
        result = {
            "id": tokenizer_id,
            "kind": kind,
            "model_sha256": manifest["model"]["sha256"],
            "vocab_size": tokenizer.vocab_size,
            "vocab_size_with_chat_tokens": total_vocab,
            "embedding_and_head_parameters": (
                total_vocab * config["evaluation"]["embedding_width"] * parameter_multiplier
            ),
            "per_category": per_category,
            "macro": {
                "bytes_per_token": sum(
                    per_category[category]["bytes_per_token"] for category in category_ids
                )
                / len(category_ids),
                "tokens_per_kib": sum(
                    per_category[category]["tokens_per_kib"] for category in category_ids
                )
                / len(category_ids),
                "tokens_per_word": sum(
                    per_category[category]["tokens_per_word"] for category in category_ids
                )
                / len(category_ids),
            },
            "probes": probes,
            "static_hard_gates": {
                "uint16_with_chat_tokens": total_vocab <= 65536,
                "zero_unknown_tokens": probes["unknown_tokens"] == 0
                and all(value["unknown_tokens"] == 0 for value in per_category.values()),
                "probe_roundtrip_exact": probes["roundtrip_mismatch_count_capped"] == 0,
            },
        }
        results.append(result)
    baseline_results = [result for result in results if result["kind"] == "baseline"]
    reference = baseline_results[0] if baseline_results else None
    for result in results:
        if reference is None:
            result["delta_vs_primary_baseline"] = None
        else:
            result["delta_vs_primary_baseline"] = {
                "tokens_per_kib": result["macro"]["tokens_per_kib"]
                - reference["macro"]["tokens_per_kib"],
                "embedding_and_head_parameters": result["embedding_and_head_parameters"]
                - reference["embedding_and_head_parameters"],
            }
    report = {
        "format": EVALUATION_FORMAT,
        "format_version": FORMAT_VERSION,
        "plan_fingerprint": config["plan_fingerprint"],
        "sample_manifest_sha256": _sha256(sample_dir / "manifest.json"),
        "primary_baseline": reference["id"] if reference else None,
        "selection_authority": False,
        "selection_note": "static metrics nominate candidates; the matched LM pilot decides",
        "tokenizers": results,
    }
    output_path = Path(config["output_dir"]) / "evaluation.json"
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != report:
            raise FileExistsError(f"different tokenizer evaluation already exists: {output_path}")
    else:
        _write_json(output_path, report)
    return report
