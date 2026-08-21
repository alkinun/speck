import pytest

from speck.model import Config, LayerConfig
from speck.search.study import SearchStudy


def config(hidden=8):
    return Config(
        vocab_size=16,
        layers=(LayerConfig(hidden, 16, 1),),
        head_dim=4,
    )


def test_study_separates_architectures_rungs_and_trials(tmp_path):
    study = SearchStudy(tmp_path / "study.sqlite3")
    assert study.initialize({"format_version": 2}, {"device": "cpu"})
    parent = study.add_architecture(
        config(), {"parameters": 100}, 0, 0, 1, {"operator": "seed"}
    )
    child = study.add_architecture(
        config(12),
        {"parameters": 120},
        0,
        1,
        2,
        {"operator": "change_hidden_size"},
        parents=(("primary", parent),),
    )
    assert parent is not None and child is not None
    assert study.add_architecture(
        config(), {"parameters": 100}, 1, 0, 3, {"operator": "duplicate"}
    ) is None
    assert study.add_rung(parent, 0, (11, 12))
    assert study.add_rung(parent, 1, (11, 12, 13))
    assert not study.add_rung(parent, 0, (11, 12))
    assert [(trial["rung"], trial["seed_index"]) for trial in study.trials(architecture_id=parent)] == [
        (1, 0), (1, 1), (1, 2), (0, 0), (0, 1)
    ]
    assert study.lineage(child)[0]["id"] == parent
    study.close()


def test_study_rejects_stale_attempt_results(tmp_path):
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize({"format_version": 2}, {})
    architecture = study.add_architecture(
        config(), {"parameters": 100}, 0, 0, 1, {"operator": "seed"}
    )
    assert architecture is not None
    study.add_rung(architecture, 0, (11,))
    trial = study.trials()[0]
    attempt = study.start_attempt(trial["id"])
    study.complete_attempt(trial["id"], attempt, {"objectives": {"quality": 2.0}})
    with pytest.raises(RuntimeError, match="stale"):
        study.complete_attempt(trial["id"], attempt, {"objectives": {"quality": 1.0}})
    assert study.trial(trial["id"])["result"]["objectives"]["quality"] == 2.0
    study.close()


def test_study_recovers_trials_without_consuming_retry(tmp_path):
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize({"format_version": 2}, {})
    architecture = study.add_architecture(
        config(), {"parameters": 100}, 0, 0, 1, {"operator": "seed"}
    )
    assert architecture is not None
    study.add_rung(architecture, 0, (11,))
    trial = study.trials()[0]
    study.start_attempt(trial["id"])
    assert study.recover_running() == 1
    assert study.trial(trial["id"])["status"] == "pending"
    assert study.failed_attempt_count(trial["id"]) == 0
    study.close()


def test_study_records_architectures_and_promotions_atomically(tmp_path):
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize({"format_version": 2}, {})
    architecture = study.add_architecture_with_rung(
        config(),
        {"parameters": 100},
        0,
        0,
        1,
        {"operator": "seed"},
        (),
        (),
        0,
        (11,),
    )
    assert architecture is not None
    assert len(study.trials(architecture_id=architecture)) == 1
    study.update_rung(architecture, 0, "complete", {"objectives": {}})
    assert study.add_rung(architecture, 1, (12,))
    assert not study.promote(
        architecture,
        0,
        1,
        (12,),
        {"reason": "test"},
    )
    assert study.rung(architecture, 0)["status"] == "complete"

    promoted = study.add_architecture_with_rung(
        config(12),
        {"parameters": 120},
        0,
        1,
        2,
        {"operator": "change_hidden_size"},
        (),
        (("primary", architecture),),
        0,
        (11,),
    )
    assert promoted is not None
    study.update_rung(promoted, 0, "complete", {"objectives": {}})
    assert study.promote(promoted, 0, 1, (12,), {"reason": "test"})
    assert study.rung(promoted, 0)["status"] == "promoted"
    assert study.rung(promoted, 1)["status"] == "active"
    study.close()
