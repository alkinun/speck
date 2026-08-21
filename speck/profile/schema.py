"""versioned records for inference profiling."""

import math
import re
from dataclasses import asdict, dataclass

from speck.search.protocol import content_digest


profile_schema_version = 2


@dataclass(frozen=True)
class BackendIdentity:
    name: str
    version: str
    options_digest: str

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9_]+", self.name):
            raise ValueError("backend names must be lowercase identifiers")
        if not self.version or not self.options_digest:
            raise ValueError("backend identity values cannot be empty")

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class ProfileScenario:
    name: str
    backend: BackendIdentity
    device: str
    dtype: str
    cache_dtype: str
    batch_size: int
    prompt_tokens: int
    generated_tokens: int
    warmup_requests: int
    measured_requests: int
    process_repetitions: int = 1
    schema_version: int = profile_schema_version

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9_]+", self.name):
            raise ValueError("profile scenario names must be lowercase identifiers")
        values = (
            self.batch_size,
            self.prompt_tokens,
            self.generated_tokens,
            self.measured_requests,
            self.process_repetitions,
        )
        if any(value < 1 for value in values) or self.warmup_requests < 0:
            raise ValueError("profile scenario counts are invalid")
        if not self.device or not self.dtype or not self.cache_dtype:
            raise ValueError("profile device and dtype values cannot be empty")

    @property
    def digest(self):
        return content_digest(self)

    def export(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["backend"] = BackendIdentity.from_dict(value["backend"])
        return cls(**value)


@dataclass(frozen=True)
class SampleSummary:
    samples: tuple[float, ...]
    mean: float
    p50: float
    p95: float
    minimum: float
    maximum: float

    def __post_init__(self):
        if not self.samples:
            raise ValueError("sample summaries cannot be empty")
        if any(not math.isfinite(value) or value < 0 for value in self.samples):
            raise ValueError("latency samples must be finite and nonnegative")

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["samples"] = tuple(value["samples"])
        return cls(**value)


@dataclass(frozen=True)
class ProfileResult:
    scenario_digest: str
    architecture_digest: str
    model_prefill_ms: SampleSummary
    first_decode_ms: SampleSummary
    decode_ms: SampleSummary
    request_ms: SampleSummary
    weight_bytes: int
    state_bytes: int
    peak_memory_bytes: int
    schema_version: int = profile_schema_version

    def __post_init__(self):
        if not self.scenario_digest or not self.architecture_digest:
            raise ValueError("profile result identities cannot be empty")
        if min(self.weight_bytes, self.state_bytes, self.peak_memory_bytes) < 0:
            raise ValueError("profile memory values cannot be negative")

    @property
    def digest(self):
        return content_digest(self)

    def export(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for name in (
            "model_prefill_ms",
            "first_decode_ms",
            "decode_ms",
            "request_ms",
        ):
            value[name] = SampleSummary.from_dict(value[name])
        return cls(**value)
