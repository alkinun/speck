import torch

from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    StageConfig,
)
from speck.profile.backends.torch_native import TorchNativeBackend
from speck.profile.protocol import profile_session
from speck.profile.schema import ProfileScenario
from speck.profile.stats import nearest_rank, summarize


def config():
    return ArchitectureConfig(
        (
            BlockGroup(
                BlockConfig(
                    8,
                    (StageConfig((AttentionSpec(4, 1, "sliding", 3),)),),
                )
            ),
        ),
        8,
        vocab_size=16,
        max_position_embeddings=16,
    )


def scenario(backend):
    return ProfileScenario(
        name="cpu_short",
        backend=backend.identity,
        device="cpu",
        dtype="float32",
        cache_dtype="float32",
        batch_size=1,
        prompt_tokens=4,
        generated_tokens=3,
        warmup_requests=1,
        measured_requests=3,
    )


def test_nearest_rank_percentiles_are_exact():
    samples = tuple(range(1, 101))
    assert nearest_rank(samples, 0.5) == 50
    assert nearest_rank(samples, 0.95) == 95
    summary = summarize((3, 1, 2))
    assert summary.p50 == 2
    assert summary.p95 == 3


def test_torch_backend_uses_resident_model_dtype():
    backend = TorchNativeBackend()
    profile = scenario(backend)
    artifact = backend.prepare(config(), profile)
    session = backend.load(artifact, profile)
    assert {parameter.dtype for parameter in session.model.parameters()} == {
        torch.float32
    }
    assert artifact.weight_bytes == sum(
        parameter.numel() * parameter.element_size()
        for parameter in session.model.parameters()
    )


def test_native_profile_runs_growing_decode_and_keeps_raw_samples():
    backend = TorchNativeBackend()
    profile = scenario(backend)
    architecture = config()
    artifact = backend.prepare(architecture, profile)
    session = backend.load(artifact, profile)
    prompt = torch.randint(0, 16, (1, profile.prompt_tokens))
    generated = torch.randint(0, 16, (1, profile.generated_tokens))
    result = profile_session(
        session,
        profile,
        architecture.digest,
        prompt,
        generated,
        artifact.weight_bytes,
    )
    assert len(result.model_prefill_ms.samples) == profile.measured_requests
    assert len(result.first_decode_ms.samples) == profile.measured_requests
    assert len(result.decode_ms.samples) == (
        profile.measured_requests * (profile.generated_tokens - 1)
    )
    assert result.state_bytes == 2 * 1 * 3 * 4 * 4
    assert result.weight_bytes == artifact.weight_bytes
