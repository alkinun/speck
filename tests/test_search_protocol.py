import pytest

from speck.search.protocol import (
    ObjectiveSet,
    ObjectiveSpec,
    SeedBundle,
    TrainingProtocol,
    VersionSet,
    content_digest,
)


def training_protocol(**overrides):
    values = {
        "name": "ultrafineweb_calibration",
        "dataset_digest": "dataset",
        "tokenizer_digest": "tokenizer",
        "segment_plan_digest": "segments",
        "sequence_length": 8,
        "batch_tokens": 32,
        "device_batch_size": 2,
        "optimizer": "muon",
        "learning_rate": 0.001,
        "minimum_learning_rate_scale": 0.1,
        "warmup_steps": 4,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "checkpoint_tokens": (32, 64, 128),
    }
    values.update(overrides)
    return TrainingProtocol(**values)


def test_version_set_has_a_stable_digest():
    versions = VersionSet()
    assert versions.digest == content_digest(versions)
    assert versions.architecture_schema == 3
    assert versions.study_semantics == 3


def test_seed_bundles_separate_randomness_sources():
    first = SeedBundle.create(42, 0)
    repeated = SeedBundle.create(42, 0, numerical_repeat=1)
    second = SeedBundle.create(42, 1)
    assert first.initialization_seed == repeated.initialization_seed
    assert first.data_seed == repeated.data_seed
    assert first.numerical_seed != repeated.numerical_seed
    assert first.initialization_seed != second.initialization_seed
    assert first.data_seed != second.data_seed


def test_objective_sets_separate_selection_and_reporting():
    objectives = ObjectiveSet(
        "gpu_short",
        (
            ObjectiveSpec("quality.target_nll", "minimize", "quality"),
            ObjectiveSpec("gpu.decode_p95", "minimize", "efficiency"),
            ObjectiveSpec(
                "gpu.decode_p50",
                "minimize",
                "reporting",
                required_for_selection=False,
            ),
        ),
    )
    assert [objective.name for objective in objectives.selection] == [
        "quality.target_nll",
        "gpu.decode_p95",
    ]
    assert len(objectives.digest) == 64


def test_reporting_objectives_cannot_enter_selection():
    with pytest.raises(ValueError, match="reporting"):
        ObjectiveSpec("gpu.decode", "minimize", "reporting")


def test_training_protocol_requires_resumable_checkpoints():
    protocol = training_protocol()
    assert protocol.target_tokens == 128
    assert protocol.digest == content_digest(protocol)
    with pytest.raises(ValueError, match="align"):
        training_protocol(checkpoint_tokens=(32, 63))


def test_training_protocol_rejects_geometry_changes_hidden_in_fidelity():
    with pytest.raises(ValueError, match="complete device batches"):
        training_protocol(batch_tokens=24)
