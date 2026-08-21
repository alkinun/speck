import sqlite3

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


def test_study_tracks_process_identity_and_interruptions(tmp_path):
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
    trial = study.trials()[0]
    attempt = study.start_attempt(trial["id"])
    study.set_attempt_payload(attempt, "digest")
    study.set_attempt_process(attempt, 123, 456, "boot")
    assert study.running_attempts()[0]["pid_start_time"] == 456
    study.interrupt_attempt(trial["id"], attempt, "coordinator stopped")
    assert study.trial(trial["id"])["status"] == "pending"
    assert study.failed_attempt_count(trial["id"]) == 0
    study.close()


def test_study_rejects_legacy_schema_before_creating_tables(tmp_path):
    database = tmp_path / "study.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("create table candidates(id integer primary key)")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="legacy"):
        SearchStudy(database)
    connection = sqlite3.connect(database)
    tables = {
        row[0]
        for row in connection.execute(
            "select name from sqlite_master where type = 'table'"
        )
    }
    connection.close()
    assert "architectures" not in tables


def test_study_opens_initialized_database_read_only(tmp_path):
    database = tmp_path / "study.sqlite3"
    study = SearchStudy(database)
    study.initialize({"format_version": 2}, {"device": "cpu"})
    study.close()
    readonly = SearchStudy(database, readonly=True)
    assert readonly.study()["provenance"] == {"device": "cpu"}
    readonly.close()


def test_study_migrates_format_two_schema_one(tmp_path):
    database = tmp_path / "study.sqlite3"
    study = SearchStudy(database)
    study.initialize({"format_version": 2}, {})
    study.close()
    connection = sqlite3.connect(database)
    connection.execute("alter table attempts drop column payload_digest")
    connection.execute("alter table attempts drop column pid_boot_id")
    connection.execute("alter table attempts drop column pid_start_time")
    connection.execute(
        "update metadata set value = '1' where key = 'schema_version'"
    )
    connection.commit()
    connection.close()
    migrated = SearchStudy(database)
    columns = {
        row["name"]
        for row in migrated.connection.execute("pragma table_info(attempts)")
    }
    assert {"pid_start_time", "pid_boot_id", "payload_digest"} <= columns
    migrated.close()


def test_study_rejects_unrelated_nonempty_database(tmp_path):
    database = tmp_path / "study.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("create table unrelated(id integer primary key)")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="not an architecture search"):
        SearchStudy(database)
