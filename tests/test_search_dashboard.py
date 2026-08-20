from scripts.search_dashboard import candidate, snapshot
from speck.model import Config, LayerConfig
from speck.search.store import StudyStore


def test_dashboard_reads_study_without_writing(tmp_path):
    database = tmp_path / "study.sqlite3"
    store = StudyStore(database)
    store.initialize({"max_evaluations": 4}, {"device": "cpu"})
    config = Config(
        vocab_size=16,
        layers=(LayerConfig(8, 16, 1),),
        head_dim=4,
    )
    candidate_id = store.add_candidate(config, 1, {"operator": "seed"})
    assert candidate_id is not None
    attempt = store.start_attempt(candidate_id)
    store.complete_attempt(candidate_id, attempt, {
        "objectives": {"quality.validation_nll": 2.0},
        "model": {"parameters": 123},
        "quality": {
            "train_curve": [{"tokens": 4, "loss": 2.1}],
            "validation_curve": [{"tokens": 4, "loss": 2.0}],
        },
    })
    store.update_selection(
        (candidate_id,),
        (candidate_id,),
        {},
    )
    store.close()

    state = snapshot(database)
    assert state["counts"] == {"completed": 1}
    assert state["frontier"][0]["parameters"] == 123
    assert state["objectives"] == ["quality.validation_nll"]
    detail = candidate(database, candidate_id)
    assert detail["mutation"] == {"operator": "seed"}
    assert detail["result"]["quality"]["train_curve"][0]["loss"] == 2.1
