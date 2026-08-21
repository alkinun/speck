"""versioned configuration for multi-fidelity architecture search."""

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass

from speck.search.architecture import SearchSpace
from speck.search.evaluate import InferenceSettings, QualitySettings, QuantizationSettings


@dataclass(frozen=True)
class ValidationSlice:
    name: str
    offset_tokens: int = 0
    objective: bool = True

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9_]+", self.name):
            raise ValueError("validation slice names must be lowercase identifiers")
        if self.offset_tokens < 0:
            raise ValueError("validation slice offsets cannot be negative")


@dataclass(frozen=True)
class RungSettings:
    name: str
    architecture_limit: int
    seed_count: int
    train_tokens: int
    sequence_length: int
    eval_every_tokens: int
    eval_tokens: int
    inference_samples: int

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9_]+", self.name):
            raise ValueError("rung names must be lowercase identifiers")
        values = (
            self.architecture_limit,
            self.seed_count,
            self.train_tokens,
            self.sequence_length,
            self.eval_every_tokens,
            self.eval_tokens,
            self.inference_samples,
        )
        if any(value < 1 for value in values):
            raise ValueError("rung settings must be positive")


@dataclass(frozen=True)
class QualityBase:
    data_dir: str
    batch_tokens: int
    device_batch_size: int
    eval_batch_size: int
    lr: float
    min_lr: float
    warmup_steps: int
    weight_decay: float
    grad_clip: float
    optimizer: str
    compile: bool = False
    batch_curriculum: bool = False

    def settings(self, rung):
        return QualitySettings(
            data_dir=self.data_dir,
            train_tokens=rung.train_tokens,
            batch_tokens=self.batch_tokens,
            device_batch_size=self.device_batch_size,
            sequence_length=rung.sequence_length,
            eval_every_tokens=rung.eval_every_tokens,
            eval_batch_size=self.eval_batch_size,
            eval_tokens=rung.eval_tokens,
            lr=self.lr,
            min_lr=self.min_lr,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            optimizer=self.optimizer,
            compile=self.compile,
            batch_curriculum=self.batch_curriculum,
        )


@dataclass(frozen=True)
class SearchSettings:
    format_version: int
    seed: int
    max_architectures: int
    initial_population: int
    population_size: int
    cohort_size: int
    confidence_z: float
    space: SearchSpace
    quality: QualityBase
    validation_slices: tuple[ValidationSlice, ...]
    inference: InferenceSettings
    quantization: QuantizationSettings
    rungs: tuple[RungSettings, ...]
    operator_probability_floor: float = 0.04
    operator_prior_success: float = 1.0
    operator_prior_failure: float = 1.0
    crossover_probability: float = 0.15
    max_generation_attempts: int = 100
    max_worker_retries: int = 1
    worker_timeout_seconds: float = 7200

    def __post_init__(self):
        if self.format_version != 2:
            raise ValueError("unsupported search configuration version")
        if min(
            self.max_architectures,
            self.initial_population,
            self.population_size,
            self.cohort_size,
        ) < 1:
            raise ValueError("search sizes must be positive")
        if self.initial_population > self.max_architectures:
            raise ValueError("initial population exceeds architecture budget")
        if self.cohort_size > self.max_architectures:
            raise ValueError("cohort size exceeds architecture budget")
        if not self.rungs or self.rungs[0].architecture_limit != self.max_architectures:
            raise ValueError("first rung must contain the full architecture budget")
        for previous, following in zip(self.rungs, self.rungs[1:]):
            if following.architecture_limit >= previous.architecture_limit:
                raise ValueError("rung architecture limits must decrease")
            if following.seed_count < previous.seed_count:
                raise ValueError("rung seed counts cannot decrease")
            if following.train_tokens <= previous.train_tokens:
                raise ValueError("rung training budgets must increase")
            if following.sequence_length < previous.sequence_length:
                raise ValueError("rung sequence lengths cannot decrease")
        if len({rung.name for rung in self.rungs}) != len(self.rungs):
            raise ValueError("rung names must be unique")
        if not self.validation_slices or len({item.name for item in self.validation_slices}) != len(self.validation_slices):
            raise ValueError("validation slices must be nonempty and unique")
        if not any(item.objective for item in self.validation_slices):
            raise ValueError("at least one validation slice must be an objective")
        operators = 8
        if not 0 <= self.operator_probability_floor < 1 / operators:
            raise ValueError("operator probability floor is too large")
        if self.operator_prior_success <= 0 or self.operator_prior_failure <= 0:
            raise ValueError("operator priors must be positive")
        if not 0 <= self.crossover_probability < 1:
            raise ValueError("crossover probability must be between zero and one")
        if self.max_generation_attempts < 1 or self.max_worker_retries < 0:
            raise ValueError("invalid retry settings")
        if not math.isfinite(self.worker_timeout_seconds) or self.worker_timeout_seconds <= 0:
            raise ValueError("worker timeout must be finite and positive")
        for rung in self.rungs:
            rung_quality = self.quality.settings(rung)
            if rung_quality.sequence_length > 4096:
                raise ValueError("rung sequence exceeds supported model context")

    @classmethod
    def from_dict(cls, settings):
        values = dict(settings)
        quality = dict(values.pop("quality"))
        quality["data_dir"] = os.path.expanduser(quality["data_dir"])
        return cls(
            space=SearchSpace.from_dict(values.pop("space")),
            quality=QualityBase(**quality),
            validation_slices=tuple(
                ValidationSlice(**item) for item in values.pop("validation_slices")
            ),
            inference=InferenceSettings.from_dict(values.pop("inference")),
            quantization=QuantizationSettings.from_dict(values.pop("quantization")),
            rungs=tuple(RungSettings(**item) for item in values.pop("rungs")),
            **values,
        )

    def export(self):
        return json.loads(json.dumps(asdict(self)))


def deterministic_seed(study_seed, *parts):
    payload = json.dumps((study_seed, *parts), separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
