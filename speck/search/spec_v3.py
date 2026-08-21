"""configuration schema for calibrated version three searches."""

import json
import math
import os
import re
from dataclasses import asdict, dataclass

from speck.search.architecture_v3 import V3SearchSpace
from speck.search.protocol import (
    ObjectiveSet,
    ObjectiveSpec,
    TrainingProtocol,
    configuration_schema_version,
)


@dataclass(frozen=True)
class SegmentPlanSettings:
    path: str
    expected_digest: str | None = None

    def __post_init__(self):
        if not self.path:
            raise ValueError("segment plan path cannot be empty")
        if self.expected_digest is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.expected_digest
        ):
            raise ValueError("expected segment plan digest must be lowercase sha256")


@dataclass(frozen=True)
class QualityProtocolTemplate:
    name: str
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

    def __post_init__(self):
        self.resolve("dataset", "tokenizer", "segments")

    def resolve(self, dataset_digest, tokenizer_digest, segment_plan_digest):
        return TrainingProtocol(
            name=self.name,
            dataset_digest=dataset_digest,
            tokenizer_digest=tokenizer_digest,
            segment_plan_digest=segment_plan_digest,
            sequence_length=self.sequence_length,
            batch_tokens=self.batch_tokens,
            device_batch_size=self.device_batch_size,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            minimum_learning_rate_scale=self.minimum_learning_rate_scale,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            gradient_clip=self.gradient_clip,
            checkpoint_tokens=self.checkpoint_tokens,
        )


@dataclass(frozen=True)
class CalibrationSettings:
    noise_architectures: int
    broad_architectures: int
    anchor_architectures: int
    initialization_seeds: int
    data_seeds: int
    numerical_repeats: int
    bootstrap_samples: int = 1_000

    def __post_init__(self):
        values = asdict(self)
        if any(value < 1 for value in values.values()):
            raise ValueError("calibration settings must be positive")
        if self.noise_architectures > self.broad_architectures:
            raise ValueError("noise panel cannot exceed the broad panel")
        if self.anchor_architectures > self.broad_architectures:
            raise ValueError("anchor panel cannot exceed the broad panel")


@dataclass(frozen=True)
class PlannerSettings:
    total_cost: float
    cost_unit: str
    max_actions_per_event: int
    posterior_samples: int
    surrogate_models: int
    surrogate_ridge: float = 1e-3

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9_]+", self.cost_unit):
            raise ValueError("planner cost units must be lowercase identifiers")
        if not math.isfinite(self.total_cost) or self.total_cost <= 0:
            raise ValueError("planner total cost must be finite and positive")
        if min(
            self.max_actions_per_event,
            self.posterior_samples,
            self.surrogate_models,
        ) < 1:
            raise ValueError("planner counts must be positive")
        if not math.isfinite(self.surrogate_ridge) or self.surrogate_ridge <= 0:
            raise ValueError("surrogate ridge must be finite and positive")


@dataclass(frozen=True)
class ProfileTemplate:
    name: str
    backend: str
    device: str
    dtype: str
    cache_dtype: str
    batch_size: int
    prompt_tokens: int
    generated_tokens: int
    warmup_requests: int
    measured_requests: int
    process_repetitions: int

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9_]+", self.name):
            raise ValueError("profile names must be lowercase identifiers")
        if not re.fullmatch(r"[a-z0-9_]+", self.backend):
            raise ValueError("profile backends must be lowercase identifiers")
        counts = (
            self.batch_size,
            self.prompt_tokens,
            self.generated_tokens,
            self.measured_requests,
            self.process_repetitions,
        )
        if any(value < 1 for value in counts) or self.warmup_requests < 0:
            raise ValueError("profile counts are invalid")
        if not self.device or not self.dtype or not self.cache_dtype:
            raise ValueError("profile runtime values cannot be empty")


@dataclass(frozen=True)
class V3SearchSettings:
    format_version: int
    seed: int
    segment_plan: SegmentPlanSettings
    quality: QualityProtocolTemplate
    calibration: CalibrationSettings
    planner: PlannerSettings
    space: V3SearchSpace
    objective_sets: tuple[ObjectiveSet, ...]
    profiles: tuple[ProfileTemplate, ...]

    def __post_init__(self):
        if self.format_version != configuration_schema_version:
            raise ValueError("unsupported v3 search configuration version")
        if self.seed < 0:
            raise ValueError("search seed cannot be negative")
        if not self.objective_sets or not self.profiles:
            raise ValueError("v3 search needs objective sets and profiles")
        objective_names = tuple(value.name for value in self.objective_sets)
        profile_names = tuple(value.name for value in self.profiles)
        if len(set(objective_names)) != len(objective_names):
            raise ValueError("objective set names must be unique")
        if len(set(profile_names)) != len(profile_names):
            raise ValueError("profile names must be unique")
        profile_devices = {profile.device.split(":", 1)[0] for profile in self.profiles}
        if not {"cuda", "cpu"} <= profile_devices:
            raise ValueError("v3 search needs native gpu and cpu profile templates")

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        segment_plan = dict(values.pop("segment_plan"))
        segment_plan["path"] = os.path.expanduser(segment_plan["path"])
        quality = dict(values.pop("quality"))
        quality["checkpoint_tokens"] = tuple(quality["checkpoint_tokens"])
        objective_sets = tuple(
            ObjectiveSet(
                item["name"],
                tuple(ObjectiveSpec(**objective) for objective in item["objectives"]),
            )
            for item in values.pop("objective_sets")
        )
        return cls(
            segment_plan=SegmentPlanSettings(**segment_plan),
            quality=QualityProtocolTemplate(**quality),
            calibration=CalibrationSettings(**values.pop("calibration")),
            planner=PlannerSettings(**values.pop("planner")),
            space=V3SearchSpace.from_dict(values.pop("space")),
            objective_sets=objective_sets,
            profiles=tuple(
                ProfileTemplate(**item) for item in values.pop("profiles")
            ),
            **values,
        )

    def export(self):
        return json.loads(json.dumps(asdict(self)))
