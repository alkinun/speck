"""document-aligned data plans for calibrated search runs."""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from speck.dataloader import manifest_fingerprint
from speck.dataset import load_manifest
from speck.search.protocol import canonical_json, content_digest


@dataclass(frozen=True)
class DocumentRecord:
    content_hash: str
    split: str
    start_token: int
    end_token: int
    source: str
    score: float | None = None

    def __post_init__(self):
        if len(self.content_hash) != 64:
            raise ValueError("document content hashes must be sha256 values")
        if self.split not in {"train", "val"}:
            raise ValueError("document split must be train or val")
        if self.start_token < 0 or self.end_token <= self.start_token:
            raise ValueError("document token ranges must be positive")
        if not self.source:
            raise ValueError("document sources cannot be empty")

    @property
    def tokens(self):
        return self.end_token - self.start_token

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class TokenSpan:
    content_hash: str
    start_token: int
    end_token: int

    def __post_init__(self):
        if len(self.content_hash) != 64:
            raise ValueError("span content hashes must be sha256 values")
        if self.start_token < 0 or self.end_token <= self.start_token:
            raise ValueError("span token ranges must be positive")

    @property
    def tokens(self):
        return self.end_token - self.start_token


@dataclass(frozen=True)
class SegmentPartition:
    name: str
    split: str
    spans: tuple[TokenSpan, ...]

    def __post_init__(self):
        if not self.name or self.name.lower() != self.name:
            raise ValueError("partition names must be lowercase")
        if self.split not in {"train", "val"}:
            raise ValueError("partition split must be train or val")
        if not self.spans:
            raise ValueError("partitions cannot be empty")
        ordered = sorted(self.spans, key=lambda span: (span.start_token, span.end_token))
        for left, right in zip(ordered, ordered[1:]):
            if left.end_token > right.start_token:
                raise ValueError("partition spans cannot overlap")

    @property
    def tokens(self):
        return sum(span.tokens for span in self.spans)


@dataclass(frozen=True)
class SegmentPlan:
    dataset_digest: str
    data_seed: int
    partitions: tuple[SegmentPartition, ...]
    format_version: int = 1

    def __post_init__(self):
        if not self.dataset_digest:
            raise ValueError("segment plans need a dataset digest")
        if self.data_seed < 0:
            raise ValueError("segment plan seeds cannot be negative")
        if not self.partitions:
            raise ValueError("segment plans cannot be empty")
        names = tuple(partition.name for partition in self.partitions)
        if len(set(names)) != len(names):
            raise ValueError("partition names must be unique")
        used = {"train": [], "val": []}
        for partition in self.partitions:
            used[partition.split].extend(
                (span.start_token, span.end_token, partition.name)
                for span in partition.spans
            )
        for split, ranges in used.items():
            ranges.sort()
            for left, right in zip(ranges, ranges[1:]):
                if left[1] > right[0]:
                    raise ValueError(
                        f"segment plan partitions overlap in the {split} split"
                    )

    @property
    def digest(self):
        return content_digest(self)

    def export(self):
        return json.loads(canonical_json(self))

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["partitions"] = tuple(
            SegmentPartition(
                name=partition["name"],
                split=partition["split"],
                spans=tuple(TokenSpan(**span) for span in partition["spans"]),
            )
            for partition in value["partitions"]
        )
        return cls(**value)


def load_document_index(data_dir, manifest=None):
    data_dir = Path(data_dir)
    manifest = manifest or load_manifest(data_dir)
    settings = manifest.get("document_index")
    if settings is None:
        raise ValueError("packed dataset has no document index")
    path = data_dir / settings["path"]
    digest = hashlib.sha256()
    records = []
    with path.open("rb") as source:
        for raw_line in source:
            digest.update(raw_line)
            records.append(DocumentRecord.from_dict(json.loads(raw_line)))
    if len(records) != settings["records"]:
        raise ValueError("document index record count does not match")
    if digest.hexdigest() != settings["sha256"]:
        raise ValueError("document index checksum does not match")
    return tuple(records)


def _take_documents(documents, requested_tokens):
    selected = []
    tokens = 0
    while documents and tokens < requested_tokens:
        document = documents.pop()
        selected.append(
            TokenSpan(
                document.content_hash,
                document.start_token,
                document.end_token,
            )
        )
        tokens += document.tokens
    if tokens < requested_tokens:
        raise ValueError("document index cannot satisfy the segment token budget")
    return tuple(selected)


def build_segment_plan(
    records,
    dataset_digest,
    data_seed,
    train_tokens,
    validation_tokens,
):
    if train_tokens < 1:
        raise ValueError("training segment tokens must be positive")
    if not validation_tokens or any(tokens < 1 for tokens in validation_tokens.values()):
        raise ValueError("validation segment tokens must be positive")
    train = [record for record in records if record.split == "train"]
    validation = [record for record in records if record.split == "val"]
    random.Random(data_seed).shuffle(train)
    random.Random(data_seed ^ 0x5DEECE66D).shuffle(validation)
    partitions = [
        SegmentPartition(
            "train",
            "train",
            _take_documents(train, train_tokens),
        )
    ]
    for name, tokens in sorted(validation_tokens.items()):
        partitions.append(
            SegmentPartition(
                name,
                "val",
                _take_documents(validation, tokens),
            )
        )
    return SegmentPlan(dataset_digest, data_seed, tuple(partitions))


def build_segment_plan_from_dataset(
    data_dir,
    data_seed,
    train_tokens,
    validation_tokens,
):
    manifest = load_manifest(data_dir)
    return build_segment_plan(
        load_document_index(data_dir, manifest),
        manifest_fingerprint(manifest),
        data_seed,
        train_tokens,
        validation_tokens,
    )
