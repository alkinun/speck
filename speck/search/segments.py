"""document-aligned data plans for calibrated search runs."""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from speck.common import dist_info
from speck.dataloader import PackedTokenSplit, manifest_fingerprint
from speck.dataset import load_manifest
from speck.search.protocol import canonical_json, content_digest, derive_seed


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


def load_segment_plan(path):
    path = Path(path).expanduser()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("segment plan must be an object")
    return SegmentPlan.from_dict(value)


def validate_segment_plan(plan, records, required_partitions=()):
    if not isinstance(plan, SegmentPlan):
        raise TypeError("segment validation needs a segment plan")
    partitions = {partition.name for partition in plan.partitions}
    missing = set(required_partitions) - partitions
    if missing:
        raise ValueError(
            f"segment plan is missing partitions: {', '.join(sorted(missing))}"
        )
    documents = {
        (
            record.split,
            record.start_token,
            record.end_token,
            record.content_hash,
        )
        for record in records
    }
    if len(documents) != len(records):
        raise ValueError("document index contains duplicate token ranges")
    for partition in plan.partitions:
        for span in partition.spans:
            identity = (
                partition.split,
                span.start_token,
                span.end_token,
                span.content_hash,
            )
            if identity not in documents:
                raise ValueError(
                    f"segment partition {partition.name} does not match the document index"
                )
    return True


def _ordered_spans(partition, data_seed, epoch):
    spans = list(partition.spans)
    seed = derive_seed(data_seed, "segment_order", partition.name, epoch)
    random.Random(seed).shuffle(spans)
    return tuple(spans)


class SegmentTokenReader:
    def __init__(self, packed, spans):
        self.packed = packed
        self.spans = spans
        self.total_tokens = sum(span.tokens for span in spans)

    def read(self, start, count):
        if start < 0 or count < 0 or start + count > self.total_tokens:
            raise IndexError("segment token read is out of range")
        if count == 0:
            return np.empty(0, dtype=np.int64)
        pieces = []
        position = start
        remaining = count
        for span in self.spans:
            if position >= span.tokens:
                position -= span.tokens
                continue
            take = min(remaining, span.tokens - position)
            pieces.append(self.packed.read(span.start_token + position, take))
            remaining -= take
            if remaining == 0:
                break
            position = 0
        if remaining:
            raise IndexError("segment token read is incomplete")
        if len(pieces) == 1:
            return pieces[0]
        return np.concatenate(pieces)


def segment_loader(
    tokenizer,
    plan,
    partition_name,
    data_seed,
    batch_size,
    sequence_length,
    *,
    device="cuda",
    resume_state_dict=None,
    data_dir=None,
):
    if not isinstance(plan, SegmentPlan):
        raise TypeError("segment loaders need a segment plan")
    if data_seed < 0 or min(batch_size, sequence_length) < 1:
        raise ValueError("segment loader dimensions and seeds must be positive")
    rank, _, world_size = dist_info()
    if rank != 0 or world_size != 1:
        raise ValueError("segment loaders currently require world size one")
    data_dir = Path(data_dir)
    manifest = load_manifest(data_dir)
    if tokenizer.vocab_size != manifest["tokenizer"]["vocab_size"]:
        raise ValueError("packed dataset vocabulary does not match tokenizer")
    if tokenizer.fingerprint() != manifest["tokenizer"]["fingerprint"]:
        raise ValueError("packed dataset was created with a different tokenizer")
    manifest_digest = manifest_fingerprint(manifest)
    if plan.dataset_digest != manifest_digest:
        raise ValueError("segment plan belongs to a different packed dataset")
    validate_segment_plan(
        plan,
        load_document_index(data_dir, manifest),
        (partition_name,),
    )
    partitions = {
        partition.name: partition for partition in plan.partitions
    }
    if partition_name not in partitions:
        raise ValueError(f"unknown segment partition: {partition_name}")
    partition = partitions[partition_name]
    packed = PackedTokenSplit(data_dir, partition.split, manifest)
    required = batch_size * sequence_length + 1
    if required > partition.tokens:
        raise ValueError("segment partition is smaller than one batch")
    epoch = 0
    logical_offset = 0
    resuming = resume_state_dict is not None
    if resume_state_dict is not None:
        expected = {
            "batch_size": batch_size,
            "data_seed": data_seed,
            "format_version": 1,
            "manifest": manifest_digest,
            "partition": partition_name,
            "plan": plan.digest,
            "sequence_length": sequence_length,
            "world_size": 1,
        }
        changed = [
            name
            for name, value in expected.items()
            if resume_state_dict.get(name) != value
        ]
        if changed:
            raise ValueError(
                f"cannot resume with changed segment state: {', '.join(changed)}"
            )
        epoch = resume_state_dict["epoch"]
        logical_offset = resume_state_dict["logical_offset"]
        if epoch < 0 or logical_offset < 0:
            raise ValueError("segment resume cursor cannot be negative")
    device = torch.device(device)
    stride = batch_size * sequence_length
    while True:
        spans = _ordered_spans(partition, data_seed, epoch)
        reader = SegmentTokenReader(packed, spans)
        if logical_offset + stride + 1 > reader.total_tokens:
            if resuming:
                raise ValueError("segment resume cursor is outside its partition")
            logical_offset = 0
            epoch += 1
            continue
        permutation = content_digest(
            tuple(
                (span.content_hash, span.start_token, span.end_token)
                for span in spans
            )
        )
        if resuming and resume_state_dict.get("permutation") != permutation:
            raise ValueError("segment resume permutation does not match")
        resuming = False
        state = {
            "batch_size": batch_size,
            "data_seed": data_seed,
            "epoch": epoch,
            "format_version": 1,
            "logical_offset": logical_offset,
            "manifest": manifest_digest,
            "partition": partition_name,
            "permutation": permutation,
            "plan": plan.digest,
            "sequence_length": sequence_length,
            "world_size": 1,
        }
        flat = torch.from_numpy(reader.read(logical_offset, required))
        rows = flat.unfold(0, sequence_length + 1, sequence_length)
        inputs = rows[:, :-1].contiguous()
        targets = rows[:, 1:].contiguous()
        if device.type == "cuda":
            inputs = inputs.pin_memory().to(device, non_blocking=True)
            targets = targets.pin_memory().to(device, non_blocking=True)
        else:
            inputs = inputs.to(device)
            targets = targets.to(device)
        yield inputs, targets, state
        logical_offset += stride


def segment_evaluation_batches(
    tokenizer,
    plan,
    partition_name,
    batch_size,
    sequence_length,
    *,
    device="cuda",
    data_dir=None,
):
    if not isinstance(plan, SegmentPlan):
        raise TypeError("segment evaluation needs a segment plan")
    if min(batch_size, sequence_length) < 1:
        raise ValueError("segment evaluation dimensions must be positive")
    rank, _, world_size = dist_info()
    if rank != 0 or world_size != 1:
        raise ValueError("segment evaluation currently requires world size one")
    data_dir = Path(data_dir)
    manifest = load_manifest(data_dir)
    if tokenizer.vocab_size != manifest["tokenizer"]["vocab_size"]:
        raise ValueError("packed dataset vocabulary does not match tokenizer")
    if tokenizer.fingerprint() != manifest["tokenizer"]["fingerprint"]:
        raise ValueError("packed dataset was created with a different tokenizer")
    if plan.dataset_digest != manifest_fingerprint(manifest):
        raise ValueError("segment plan belongs to a different packed dataset")
    validate_segment_plan(
        plan,
        load_document_index(data_dir, manifest),
        (partition_name,),
    )
    partitions = {
        partition.name: partition for partition in plan.partitions
    }
    if partition_name not in partitions:
        raise ValueError(f"unknown segment partition: {partition_name}")
    partition = partitions[partition_name]
    reader = SegmentTokenReader(
        PackedTokenSplit(data_dir, partition.split, manifest),
        partition.spans,
    )
    if reader.total_tokens < 2:
        raise ValueError("segment evaluation needs at least two tokens")
    device = torch.device(device)
    offset = 0
    remaining = reader.total_tokens - 1
    while remaining:
        if remaining >= sequence_length:
            rows = min(batch_size, remaining // sequence_length)
            length = sequence_length
        else:
            rows = 1
            length = remaining
        targets = rows * length
        flat = torch.from_numpy(reader.read(offset, targets + 1))
        windows = flat.unfold(0, length + 1, length)
        inputs = windows[:, :-1].contiguous()
        labels = windows[:, 1:].contiguous()
        if device.type == "cuda":
            inputs = inputs.pin_memory().to(device, non_blocking=True)
            labels = labels.pin_memory().to(device, non_blocking=True)
        else:
            inputs = inputs.to(device)
            labels = labels.to(device)
        yield inputs, labels
        offset += targets
        remaining -= targets


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
