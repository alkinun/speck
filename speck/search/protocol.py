"""immutable protocol identities for calibrated architecture search."""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass


architecture_schema_version = 3
configuration_schema_version = 3
study_semantics_version = 3
scheduler_algorithm_version = 1
worker_protocol_version = 3
artifact_manifest_version = 1
report_schema_version = 1


def canonical_json(value):
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def content_digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def derive_seed(study_seed, *parts):
    payload = canonical_json((study_seed, *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class VersionSet:
    architecture_schema: int = architecture_schema_version
    configuration_schema: int = configuration_schema_version
    study_semantics: int = study_semantics_version
    scheduler_algorithm: int = scheduler_algorithm_version
    worker_protocol: int = worker_protocol_version
    artifact_manifest: int = artifact_manifest_version
    report_schema: int = report_schema_version

    def __post_init__(self):
        if any(value < 1 for value in asdict(self).values()):
            raise ValueError("protocol versions must be positive")

    @property
    def digest(self):
        return content_digest(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class SeedBundle:
    index: int
    initialization_seed: int
    data_seed: int
    numerical_seed: int
    numerical_repeat: int = 0
    initialization_index: int | None = None
    data_index: int | None = None

    def __post_init__(self):
        if self.initialization_index is None:
            object.__setattr__(self, "initialization_index", self.index)
        if self.data_index is None:
            object.__setattr__(self, "data_index", self.index)
        if min(
            self.index,
            self.numerical_repeat,
            self.initialization_index,
            self.data_index,
        ) < 0:
            raise ValueError("seed bundle indices cannot be negative")
        seeds = (self.initialization_seed, self.data_seed, self.numerical_seed)
        if any(seed < 0 for seed in seeds):
            raise ValueError("seeds cannot be negative")

    @classmethod
    def create(cls, study_seed, index, numerical_repeat=0):
        return cls.create_panel(
            study_seed,
            index,
            index,
            numerical_repeat,
        )

    @classmethod
    def create_panel(
        cls,
        study_seed,
        initialization_index,
        data_index,
        numerical_repeat=0,
    ):
        diagonal = initialization_index + data_index
        index = diagonal * (diagonal + 1) // 2 + data_index
        return cls(
            index=index,
            initialization_seed=derive_seed(
                study_seed,
                "initialization",
                initialization_index,
            ),
            data_seed=derive_seed(study_seed, "data", data_index),
            numerical_seed=derive_seed(
                study_seed,
                "numerical",
                initialization_index,
                data_index,
                numerical_repeat,
            ),
            numerical_repeat=numerical_repeat,
            initialization_index=initialization_index,
            data_index=data_index,
        )

    @property
    def digest(self):
        return content_digest(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    direction: str
    role: str
    required_for_selection: bool = True

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9_.]+", self.name):
            raise ValueError("objective names must be lowercase identifiers")
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("objective direction must be minimize or maximize")
        if self.role not in {"quality", "efficiency", "safety", "reporting"}:
            raise ValueError("invalid objective role")
        if self.role == "reporting" and self.required_for_selection:
            raise ValueError("reporting objectives cannot be required for selection")

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class ObjectiveSet:
    name: str
    objectives: tuple[ObjectiveSpec, ...]

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9_]+", self.name):
            raise ValueError("objective set names must be lowercase identifiers")
        if not self.objectives:
            raise ValueError("objective sets cannot be empty")
        names = tuple(objective.name for objective in self.objectives)
        if len(set(names)) != len(names):
            raise ValueError("objective names must be unique")
        if not any(objective.required_for_selection for objective in self.objectives):
            raise ValueError("objective sets need a selection objective")

    @property
    def selection(self):
        return tuple(
            objective
            for objective in self.objectives
            if objective.required_for_selection
        )

    @property
    def digest(self):
        return content_digest(self)

    @classmethod
    def from_dict(cls, value):
        return cls(
            name=value["name"],
            objectives=tuple(
                ObjectiveSpec.from_dict(item) for item in value["objectives"]
            ),
        )


@dataclass(frozen=True)
class TrainingProtocol:
    name: str
    dataset_digest: str
    tokenizer_digest: str
    segment_plan_digest: str
    sequence_length: int
    batch_tokens: int
    device_batch_size: int
    optimizer: str
    learning_rate: float
    minimum_learning_rate_scale: float
    warmup_steps: int
    weight_decay: float
    gradient_clip: float
    checkpoint_tokens: tuple[int, ...]
    device_type: str = "cuda"
    dtype: str = "float32"
    compile_model: bool = False
    world_size: int = 1
    evaluation_partition: str = "monitor"
    evaluation_batch_size: int = 1

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9_]+", self.name):
            raise ValueError("training protocol names must be lowercase identifiers")
        digests = (
            self.dataset_digest,
            self.tokenizer_digest,
            self.segment_plan_digest,
        )
        if any(not value for value in digests):
            raise ValueError("training protocol digests cannot be empty")
        integers = (
            self.sequence_length,
            self.batch_tokens,
            self.device_batch_size,
        )
        if any(value < 1 for value in integers):
            raise ValueError("training dimensions must be positive")
        device_tokens = self.device_batch_size * self.sequence_length
        if self.batch_tokens % device_tokens:
            raise ValueError("batch tokens must contain complete device batches")
        floats = (
            self.learning_rate,
            self.minimum_learning_rate_scale,
            self.weight_decay,
            self.gradient_clip,
        )
        if any(not math.isfinite(value) or value < 0 for value in floats):
            raise ValueError("training values must be finite and nonnegative")
        if self.learning_rate == 0 or self.gradient_clip == 0:
            raise ValueError("learning rate and gradient clip must be positive")
        if not 0 <= self.minimum_learning_rate_scale <= 1:
            raise ValueError("minimum learning rate scale must be between zero and one")
        if self.warmup_steps < 0:
            raise ValueError("warmup steps cannot be negative")
        if not self.optimizer:
            raise ValueError("optimizer cannot be empty")
        if self.device_type not in {"cuda", "cpu"}:
            raise ValueError("training device type must be cuda or cpu")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("training dtype must be float32 or bfloat16")
        if self.world_size != 1:
            raise ValueError("v3 quality training currently requires world size one")
        if not re.fullmatch(r"[a-z0-9_]+", self.evaluation_partition):
            raise ValueError("evaluation partitions must be lowercase identifiers")
        if self.evaluation_batch_size < 1:
            raise ValueError("evaluation batch sizes must be positive")
        if not self.checkpoint_tokens:
            raise ValueError("checkpoint tokens cannot be empty")
        if tuple(sorted(set(self.checkpoint_tokens))) != self.checkpoint_tokens:
            raise ValueError("checkpoint tokens must be unique and increasing")
        if any(
            tokens < self.batch_tokens or tokens % self.batch_tokens
            for tokens in self.checkpoint_tokens
        ):
            raise ValueError("checkpoint tokens must align to optimizer batches")

    @property
    def target_tokens(self):
        return self.checkpoint_tokens[-1]

    @property
    def digest(self):
        return content_digest(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["checkpoint_tokens"] = tuple(value["checkpoint_tokens"])
        return cls(**value)
