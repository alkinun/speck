"""Bounded, non-authoritative tokenizer diagnostics and treatment training."""

import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import sentencepiece
from sentencepiece import sentencepiece_model_pb2

from speck.tokenizer import Tokenizer

FORMAT_VERSION = 1
TRAIN_SPEC_FORMAT = "speck_tokenizer_study_train_spec"
TRAIN_RESULT_FORMAT = "speck_tokenizer_study_model"
EVAL_SPEC_FORMAT = "speck_tokenizer_study_evaluation_spec"
EVAL_RESULT_FORMAT = "speck_tokenizer_study_evaluation"
_SPECIAL_IDS = {"unk_id": 0, "bos_id": 1, "eos_id": 2, "pad_id": -1}


def _sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_sample(sample_directory):
    sample_directory = Path(sample_directory).resolve()
    manifest_path = sample_directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"sample manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "speck_tokenizer_sample" or manifest.get("format_version") != 1:
        raise ValueError("unsupported tokenizer sample manifest")
    for category in manifest["categories"]:
        for split in ("train", "eval"):
            entry = category["splits"][split]
            path = sample_directory / entry["path"]
            if not path.is_file() or _sha256(path) != entry["sha256"]:
                raise ValueError(f"sample split identity mismatch: {path}")
    return sample_directory, manifest, _sha256(manifest_path)


def _iter_records(sample_directory, manifest, split):
    for category in manifest["categories"]:
        category_id = category["id"]
        path = sample_directory / category["splits"][split]["path"]
        with path.open(encoding="utf-8") as handle:
            for row, line in enumerate(handle):
                record = json.loads(line)
                if (
                    record.get("category") != category_id
                    or not isinstance(record.get("input"), str)
                    or not isinstance(record.get("text"), str)
                    or not isinstance(record.get("content_sha256"), str)
                ):
                    raise ValueError(f"invalid sample record: {path}:{row + 1}")
                yield category_id, record


def _split_utf8(text, maximum_bytes):
    if len(text.encode()) <= maximum_bytes:
        return [text]
    chunks = []
    current = []
    current_bytes = 0
    for character in text:
        size = len(character.encode())
        if current and current_bytes + size > maximum_bytes:
            chunks.append("".join(current))
            current = []
            current_bytes = 0
        current.append(character)
        current_bytes += size
    if current:
        chunks.append("".join(current))
    return chunks


def _paragraph_chunks(text, maximum_bytes):
    """Preserve every character while preventing training pieces from spanning long documents."""

    chunks = []
    current = ""
    for line in text.splitlines(keepends=True):
        if current and len((current + line).encode()) > maximum_bytes:
            chunks.extend(_split_utf8(current, maximum_bytes))
            current = ""
        if len(line.encode()) > maximum_bytes:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_utf8(line, maximum_bytes))
        else:
            current += line
        if not line.strip() and current:
            chunks.extend(_split_utf8(current, maximum_bytes))
            current = ""
    if current:
        chunks.extend(_split_utf8(current, maximum_bytes))
    if "".join(chunks) != text:
        raise AssertionError("paragraph chunking changed training text")
    return [chunk for chunk in chunks if chunk]


def _canonicalize_model(path):
    model = sentencepiece_model_pb2.ModelProto()
    model.ParseFromString(Path(path).read_bytes())
    model.trainer_spec.model_prefix = "tokenizer"
    del model.trainer_spec.input[:]
    Path(path).write_bytes(model.SerializeToString(deterministic=True))


class _TrainingStream:
    def __init__(self, sample_directory, manifest, targets, unit, maximum_chunk_bytes):
        self.sample_directory = sample_directory
        self.manifest = manifest
        self.targets = targets
        self.unit = unit
        self.maximum_chunk_bytes = maximum_chunk_bytes
        self.raw_bytes = 0
        self.units = 0
        self.documents = 0
        self.by_source = defaultdict(lambda: {"documents": 0, "utf8_bytes": 0})
        self.hasher = hashlib.sha256()

    def __iter__(self):
        accepted = defaultdict(int)
        seen_sources = set()
        for category, record in _iter_records(self.sample_directory, self.manifest, "train"):
            source = record["input"]
            key = f"{category}/{source}"
            if key not in self.targets:
                continue
            seen_sources.add(key)
            if accepted[key] >= self.targets[key]:
                continue
            text = record["text"]
            size = len(text.encode())
            accepted[key] += size
            self.raw_bytes += size
            self.documents += 1
            self.by_source[key]["documents"] += 1
            self.by_source[key]["utf8_bytes"] += size
            units = (
                [text]
                if self.unit == "document"
                else _paragraph_chunks(text, self.maximum_chunk_bytes)
            )
            for unit in units:
                payload = unit.encode()
                self.hasher.update(len(payload).to_bytes(8, "big"))
                self.hasher.update(payload)
                self.units += 1
                yield unit
        missing = {
            key: {"accepted": accepted[key], "target": target}
            for key, target in self.targets.items()
            if accepted[key] < target
        }
        if missing or seen_sources != set(self.targets):
            raise RuntimeError(f"training stream exhausted before source targets: {missing}")


def _validate_train_spec(spec):
    required = {
        "format",
        "format_version",
        "id",
        "sample_directory",
        "output_directory",
        "source_byte_targets",
        "model_type",
        "vocab_size",
        "add_dummy_prefix",
        "training_unit",
        "maximum_chunk_bytes",
    }
    optional = {"trainer_overrides"}
    if not isinstance(spec, dict) or not required <= set(spec) or set(spec) - required - optional:
        raise ValueError("invalid tokenizer study train spec keys")
    overrides = spec.get("trainer_overrides", {})
    if (
        spec["format"] != TRAIN_SPEC_FORMAT
        or spec["format_version"] != FORMAT_VERSION
        or not isinstance(spec["id"], str)
        or not spec["id"]
        or spec["model_type"] not in {"bpe", "unigram"}
        or not isinstance(spec["vocab_size"], int)
        or not 256 < spec["vocab_size"] < 65_533
        or not isinstance(spec["add_dummy_prefix"], bool)
        or spec["training_unit"] not in {"document", "paragraph_4k"}
        or not isinstance(spec["maximum_chunk_bytes"], int)
        or spec["maximum_chunk_bytes"] < 256
        or not isinstance(spec["source_byte_targets"], dict)
        or not spec["source_byte_targets"]
        or not isinstance(overrides, dict)
        or set(overrides)
        - {
            "allow_whitespace_only_pieces",
            "character_coverage",
            "max_sentence_length",
            "train_extremely_large_corpus",
        }
        or any(
            not isinstance(key, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for key, value in spec["source_byte_targets"].items()
        )
    ):
        raise ValueError("invalid tokenizer study train spec")
    if (
        "allow_whitespace_only_pieces" in overrides
        and not isinstance(overrides["allow_whitespace_only_pieces"], bool)
    ) or (
        "train_extremely_large_corpus" in overrides
        and not isinstance(overrides["train_extremely_large_corpus"], bool)
    ):
        raise ValueError("invalid tokenizer study train spec")
    if "character_coverage" in overrides and (
        isinstance(overrides["character_coverage"], bool)
        or not isinstance(overrides["character_coverage"], (int, float))
        or not 0.98 <= overrides["character_coverage"] <= 1.0
    ):
        raise ValueError("invalid tokenizer study train spec")
    if "max_sentence_length" in overrides and (
        isinstance(overrides["max_sentence_length"], bool)
        or not isinstance(overrides["max_sentence_length"], int)
        or overrides["max_sentence_length"] < spec["maximum_chunk_bytes"]
    ):
        raise ValueError("invalid tokenizer study train spec")
    return spec


def train_study_model(spec):
    """Train one immutable exploratory model from a source-balanced sample view."""

    spec = _validate_train_spec(spec)
    sample_directory, manifest, sample_hash = _load_sample(spec["sample_directory"])
    output = Path(spec["output_directory"]).resolve()
    staging = output.with_name(output.name + ".building")
    if output.exists() or staging.exists():
        raise FileExistsError(f"study model output already exists: {output}")
    staging.mkdir(parents=True)
    stream = _TrainingStream(
        sample_directory,
        manifest,
        spec["source_byte_targets"],
        spec["training_unit"],
        spec["maximum_chunk_bytes"],
    )
    settings = {
        "model_type": spec["model_type"],
        "vocab_size": spec["vocab_size"],
        "character_coverage": 1.0,
        "byte_fallback": True,
        "normalization_rule_name": "identity",
        "remove_extra_whitespaces": False,
        "add_dummy_prefix": spec["add_dummy_prefix"],
        "split_digits": True,
        "split_by_unicode_script": True,
        "split_by_whitespace": True,
        "split_by_number": True,
        "max_sentence_length": 1_000_000,
        "num_threads": 1,
        "hard_vocab_limit": True,
        **_SPECIAL_IDS,
        "unk_piece": "<unk>",
        "bos_piece": "<s>",
        "eos_piece": "</s>",
        "input_sentence_size": 0,
        "shuffle_input_sentence": False,
        "train_extremely_large_corpus": True,
        "minloglevel": 2,
    }
    settings.update(spec.get("trainer_overrides", {}))
    prefix = staging / "tokenizer"
    try:
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=iter(stream), model_prefix=str(prefix), **settings
        )
        model_path = prefix.with_suffix(".model")
        vocab_path = prefix.with_suffix(".vocab")
        _canonicalize_model(model_path)
        tokenizer = Tokenizer(model_path)
        if tokenizer.vocab_size != spec["vocab_size"]:
            raise ValueError("study tokenizer did not reach requested vocabulary size")
        if (tokenizer.unk_id, tokenizer.bos_id, tokenizer.eos_id) != (0, 1, 2):
            raise ValueError("study tokenizer special IDs differ from contract")
        result = {
            "format": TRAIN_RESULT_FORMAT,
            "format_version": FORMAT_VERSION,
            "status": "local_exploratory_model_no_selection_authority",
            "id": spec["id"],
            "spec": spec,
            "sample_manifest_sha256": sample_hash,
            "training_stream": {
                "sha256": stream.hasher.hexdigest(),
                "raw_utf8_bytes": stream.raw_bytes,
                "documents": stream.documents,
                "units": stream.units,
                "by_source": dict(sorted(stream.by_source.items())),
            },
            "sentencepiece_version": sentencepiece.__version__,
            "settings": settings,
            "model": {
                "path": model_path.name,
                "sha256": _sha256(model_path),
                "vocab_size": tokenizer.vocab_size,
            },
            "vocab": {"path": vocab_path.name, "sha256": _sha256(vocab_path)},
            "selection_authority": False,
            "launch_authority": False,
        }
        _write_json(staging / "manifest.json", result)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output)
        return result
    except Exception:
        raise


def _percentile(values, fraction):
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _summarize_documents(documents):
    total_bytes = sum(document["utf8_bytes"] for document in documents)
    total_tokens = sum(document["tokens"] for document in documents)
    ratios = [1024 * document["tokens"] / document["utf8_bytes"] for document in documents]
    return {
        "documents": len(documents),
        "utf8_bytes": total_bytes,
        "tokens": total_tokens,
        "tokens_per_kib": 1024 * total_tokens / total_bytes,
        "document_tokens_per_kib_p50": _percentile(ratios, 0.50),
        "document_tokens_per_kib_p90": _percentile(ratios, 0.90),
        "document_tokens_per_kib_p99": _percentile(ratios, 0.99),
    }


def _piece_structure(model_path, token_counts):
    model = sentencepiece_model_pb2.ModelProto()
    model.ParseFromString(Path(model_path).read_bytes())
    normal_ids = [index for index, piece in enumerate(model.pieces) if piece.type == 1]
    byte_ids = {index for index, piece in enumerate(model.pieces) if piece.type == 6}
    normal_pieces = [model.pieces[index].piece for index in normal_ids]
    whitespace_ids = {
        index
        for index in normal_ids
        if model.pieces[index].piece
        and all(character.isspace() or character == "▁" for character in model.pieces[index].piece)
    }
    multi_space_ids = {
        index for index in whitespace_ids if model.pieces[index].piece.count("▁") > 1
    }
    used_ids = set(token_counts)
    return {
        "normal_pieces": len(normal_ids),
        "normal_pieces_used_on_evaluation": len(set(normal_ids) & used_ids),
        "normal_piece_utilization_on_evaluation": len(set(normal_ids) & used_ids) / len(normal_ids),
        "byte_pieces_used_on_evaluation": len(byte_ids & used_ids),
        "byte_piece_token_occurrences": sum(token_counts[index] for index in byte_ids),
        "whitespace_only_pieces": len(whitespace_ids),
        "whitespace_piece_token_occurrences": sum(token_counts[index] for index in whitespace_ids),
        "multi_space_pieces": len(multi_space_ids),
        "multi_space_piece_token_occurrences": sum(
            token_counts[index] for index in multi_space_ids
        ),
        "pieces_containing_newline": sum("\n" in piece or "\r" in piece for piece in normal_pieces),
        "punctuation_only_pieces": sum(
            bool(piece) and all(not character.isalnum() for character in piece.replace("▁", ""))
            for piece in normal_pieces
        ),
        "maximum_piece_characters": max(map(len, normal_pieces)),
    }


def _bootstrap_delta(documents, repetitions, seed):
    rng = random.Random(seed)
    category_values = defaultdict(list)
    for document in documents:
        category_values[document["category"]].append(document["delta_tokens_per_kib"])
    categories = sorted(category_values)
    macros = []
    for _ in range(repetitions):
        means = []
        for category in categories:
            values = category_values[category]
            means.append(sum(values[rng.randrange(len(values))] for _ in values) / len(values))
        macros.append(sum(means) / len(means))
    return {
        "repetitions": repetitions,
        "seed": seed,
        "macro_delta_document_mean_p025": _percentile(macros, 0.025),
        "macro_delta_document_mean_p50": _percentile(macros, 0.50),
        "macro_delta_document_mean_p975": _percentile(macros, 0.975),
    }


def evaluate_study(spec):
    """Compare arbitrary study models on one immutable held-out sample."""

    required = {
        "format",
        "format_version",
        "id",
        "sample_directory",
        "output",
        "baseline_id",
        "models",
        "embedding_width",
        "chat_added_tokens",
        "bootstrap_repetitions",
        "bootstrap_seed",
        "probe_strings",
    }
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError("invalid tokenizer study evaluation spec keys")
    if spec["format"] != EVAL_SPEC_FORMAT or spec["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported tokenizer study evaluation spec")
    sample_directory, manifest, sample_hash = _load_sample(spec["sample_directory"])
    records = list(_iter_records(sample_directory, manifest, "eval"))
    model_results = []
    document_results = {}
    for declaration in spec["models"]:
        model_path = Path(declaration["path"]).resolve()
        if not model_path.is_file() or _sha256(model_path) != declaration["sha256"]:
            raise ValueError(f"study model identity mismatch: {model_path}")
        tokenizer = Tokenizer(model_path)
        token_counts = Counter()
        unknowns = 0
        roundtrip_mismatches = 0
        by_category = defaultdict(list)
        by_source = defaultdict(list)
        per_document = []
        for category, record in records:
            text = record["text"]
            token_ids = tokenizer.encode(text)
            token_counts.update(token_ids)
            unknowns += sum(token_id == tokenizer.unk_id for token_id in token_ids)
            roundtrip_mismatches += tokenizer.decode(token_ids) != text
            document = {
                "category": category,
                "source": record["input"],
                "content_sha256": record["content_sha256"],
                "utf8_bytes": len(text.encode()),
                "tokens": len(token_ids),
            }
            per_document.append(document)
            by_category[category].append(document)
            by_source[f"{category}/{record['input']}"].append(document)
        probe_unknowns = probe_mismatches = 0
        for probe in spec["probe_strings"]:
            ids = tokenizer.encode(probe)
            probe_unknowns += sum(token_id == tokenizer.unk_id for token_id in ids)
            probe_mismatches += tokenizer.decode(ids) != probe
        per_category = {
            category: _summarize_documents(documents)
            for category, documents in sorted(by_category.items())
        }
        macro = sum(value["tokens_per_kib"] for value in per_category.values()) / len(per_category)
        total_vocab = tokenizer.vocab_size + spec["chat_added_tokens"]
        diagnostic_views = {}
        for view, transform in (
            ("collapse_all_whitespace", lambda text: re.sub(r"\s+", " ", text)),
            ("collapse_horizontal_whitespace", lambda text: re.sub(r"[ \t]+", " ", text)),
        ):
            transformed_categories = defaultdict(list)
            for category, record in records:
                text = transform(record["text"])
                transformed_categories[category].append(
                    {"utf8_bytes": len(text.encode()), "tokens": len(tokenizer.encode(text))}
                )
            summarized = {
                category: _summarize_documents(documents)
                for category, documents in sorted(transformed_categories.items())
            }
            diagnostic_views[view] = {
                "macro_tokens_per_kib": sum(
                    value["tokens_per_kib"] for value in summarized.values()
                )
                / len(summarized),
                "per_category": summarized,
            }
        model_results.append(
            {
                "id": declaration["id"],
                "kind": declaration["kind"],
                "model_sha256": declaration["sha256"],
                "vocab_size": tokenizer.vocab_size,
                "vocab_size_with_chat_tokens": total_vocab,
                "embedding_and_head_parameters": total_vocab * spec["embedding_width"] * 2,
                "macro_tokens_per_kib": macro,
                "per_category": per_category,
                "per_source": {
                    source: _summarize_documents(documents)
                    for source, documents in sorted(by_source.items())
                },
                "piece_structure": _piece_structure(model_path, token_counts),
                "diagnostic_views": diagnostic_views,
                "hard_gates": {
                    "uint16_with_chat_tokens": total_vocab <= 65_536,
                    "zero_unknown_tokens": unknowns == 0 and probe_unknowns == 0,
                    "roundtrip_exact": roundtrip_mismatches == 0 and probe_mismatches == 0,
                },
            }
        )
        document_results[declaration["id"]] = per_document
    by_id = {result["id"]: result for result in model_results}
    if spec["baseline_id"] not in by_id:
        raise ValueError("study baseline is missing")
    baseline = by_id[spec["baseline_id"]]
    baseline_documents = {
        document["content_sha256"]: document for document in document_results[spec["baseline_id"]]
    }
    comparisons = {}
    for result in model_results:
        if result["id"] == spec["baseline_id"]:
            continue
        deltas = []
        for document in document_results[result["id"]]:
            reference = baseline_documents[document["content_sha256"]]
            if (
                document["category"] != reference["category"]
                or document["source"] != reference["source"]
                or document["utf8_bytes"] != reference["utf8_bytes"]
            ):
                raise ValueError("study evaluation documents are not exactly paired")
            deltas.append(
                {
                    **document,
                    "delta_tokens_per_kib": 1024
                    * (document["tokens"] - reference["tokens"])
                    / document["utf8_bytes"],
                }
            )
        source_deltas = {
            source: result["per_source"][source]["tokens_per_kib"]
            - baseline["per_source"][source]["tokens_per_kib"]
            for source in result["per_source"]
        }
        comparisons[result["id"]] = {
            "macro_tokens_per_kib_delta": result["macro_tokens_per_kib"]
            - baseline["macro_tokens_per_kib"],
            "macro_relative_delta": result["macro_tokens_per_kib"]
            / baseline["macro_tokens_per_kib"]
            - 1,
            "embedding_and_head_parameter_delta": result["embedding_and_head_parameters"]
            - baseline["embedding_and_head_parameters"],
            "category_deltas": {
                category: value["tokens_per_kib"]
                - baseline["per_category"][category]["tokens_per_kib"]
                for category, value in result["per_category"].items()
            },
            "source_deltas": source_deltas,
            "diagnostic_view_macro_deltas": {
                view: values["macro_tokens_per_kib"]
                - baseline["diagnostic_views"][view]["macro_tokens_per_kib"]
                for view, values in result["diagnostic_views"].items()
            },
            "bootstrap": _bootstrap_delta(
                deltas, spec["bootstrap_repetitions"], spec["bootstrap_seed"]
            ),
            "top_regression_documents": [
                {
                    key: document[key]
                    for key in (
                        "category",
                        "source",
                        "content_sha256",
                        "utf8_bytes",
                        "delta_tokens_per_kib",
                    )
                }
                for document in sorted(
                    deltas, key=lambda value: value["delta_tokens_per_kib"], reverse=True
                )[:20]
            ],
        }
    result = {
        "format": EVAL_RESULT_FORMAT,
        "format_version": FORMAT_VERSION,
        "status": "local_exploratory_evaluation_no_selection_authority",
        "id": spec["id"],
        "sample_manifest_sha256": sample_hash,
        "baseline_id": spec["baseline_id"],
        "models": model_results,
        "comparisons": comparisons,
        "selection_authority": False,
        "launch_authority": False,
    }
    output = Path(spec["output"]).resolve()
    if output.exists():
        raise FileExistsError(f"study evaluation output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, result)
    return result
