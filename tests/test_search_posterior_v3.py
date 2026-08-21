from types import SimpleNamespace

from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    StageConfig,
    SwiGLUSpec,
)
from speck.search.architecture_v3 import V3SearchSpace, sample_architecture
from speck.search.artifacts import ArtifactStore
from speck.search.posterior_v3 import build_posterior_shadow
from speck.search.protocol import (
    ObjectiveSet,
    ObjectiveSpec,
    SeedBundle,
    TrainingProtocol,
)
from speck.search.study_v3 import V3Study


def base():
    return ArchitectureConfig(
        (
            BlockGroup(
                BlockConfig(
                    8,
                    (
                        StageConfig((AttentionSpec(4, 1),)),
                        StageConfig((SwiGLUSpec(16),)),
                    ),
                )
            ),
        ),
        8,
        vocab_size=16,
        max_position_embeddings=16,
    )


def space():
    return V3SearchSpace(
        1,
        3,
        (8, 12, 16),
        (16, 24),
        (4,),
        (1, 2),
        (4, 8),
        (3,),
        (8, 16),
    )


def objective_sets():
    return (
        ObjectiveSet(
            "gpu_short",
            (
                ObjectiveSpec("quality.target_nll", "minimize", "quality"),
                ObjectiveSpec("gpu_short.decode_p95", "minimize", "efficiency"),
            ),
        ),
        ObjectiveSet(
            "cpu_short",
            (
                ObjectiveSpec("quality.target_nll", "minimize", "quality"),
                ObjectiveSpec("cpu_short.decode_p95", "minimize", "efficiency"),
            ),
        ),
    )


def settings():
    return SimpleNamespace(
        seed=42,
        objective_sets=objective_sets(),
        calibration=SimpleNamespace(
            anchor_architectures=2,
            anchor_tokens=16,
            bootstrap_samples=20,
            broad_tokens=8,
            noise_architectures=2,
            noise_tokens=4,
        ),
        planner=SimpleNamespace(
            posterior_samples=100,
            surrogate_models=8,
            surrogate_ridge=0.001,
        ),
    )


def test_posterior_shadow_is_calibrated_persisted_and_replayable(tmp_path):
    configs = [base()]
    for seed in range(20):
        candidate = sample_architecture(base(), space(), seed)
        if candidate.digest not in {value.digest for value in configs}:
            configs.append(candidate)
        if len(configs) == 5:
            break
    assert len(configs) == 5
    study = V3Study(tmp_path / "study.sqlite3")
    study.initialize({}, {})
    for objectives in objective_sets():
        study.add_objective_set(objectives)
    for config in configs:
        study.add_architecture(config)
    protocol = TrainingProtocol(
        "calibration",
        "dataset",
        "tokenizer",
        "segments",
        4,
        4,
        1,
        "adamw",
        0.001,
        0.1,
        0,
        0.1,
        1.0,
        (4, 8, 16),
        device_type="cpu",
    )
    for index, config in enumerate(configs):
        quality = 2.0 + (index - 2) ** 2 * 0.05
        for set_index, objectives in enumerate(objective_sets()):
            study.add_observation(
                config.digest,
                objectives.digest,
                "quality.target_nll",
                quality,
                tokens=8,
                source="quality_evaluation",
            )
            study.add_observation(
                config.digest,
                objectives.digest,
                objectives.selection[1].name,
                1.0 + ((index * 3 + set_index) % 5),
                source="profile",
            )
        if index < 2:
            for seed_index in range(2):
                seeds = SeedBundle.create_panel(42, seed_index, 0, 0)
                run_id = study.add_run(config.digest, protocol, seeds)
                for set_index, objectives in enumerate(objective_sets()):
                    study.add_observation(
                        config.digest,
                        objectives.digest,
                        "quality.target_nll",
                        quality + seed_index * 0.02 + set_index * 0.001,
                        run_id=run_id,
                        tokens=4,
                        source="quality_evaluation",
                    )
    first = build_posterior_shadow(
        study,
        settings(),
        configs,
        tmp_path / "artifacts",
        anchor_cost=10.0,
    )
    assert first["created"]
    assert len(first["anchors"]) == 2
    stored = study.posterior_report(first["evidence_digest"])
    assert stored["anchors"] == first["anchors"]
    artifact = study.artifact(first["artifact_digest"])
    assert ArtifactStore(tmp_path / "artifacts").verify(artifact)
    repeated = build_posterior_shadow(
        study,
        settings(),
        configs,
        tmp_path / "artifacts",
        anchor_cost=10.0,
    )
    assert repeated == {**first, "created": False}
    study.close()
