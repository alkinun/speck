from speck.model import Config, LayerConfig
from speck.search.evolution import SelectionMetrics
from speck.search.store import StudyStore


def config(hidden=8):
    return Config(
        vocab_size=16,
        layers=(LayerConfig(hidden, 16, 1),),
        head_dim=4,
    )


def test_store_tracks_candidates_ancestry_and_frontier(tmp_path):
    store = StudyStore(tmp_path / "study.sqlite3")
    assert store.initialize({"population_size": 2}, {"device": "cpu"})
    assert not store.initialize({"population_size": 2}, {"device": "cpu"})
    parent = store.add_candidate(config(), 1, {"operator": "seed"})
    child = store.add_candidate(
        config(12),
        2,
        {"operator": "change_hidden_size"},
        ({"kind": "repair_layer"},),
        parent,
    )
    assert parent is not None and child is not None
    assert store.add_candidate(config(), 3, {"operator": "duplicate"}) is None

    attempt = store.start_attempt(parent)
    store.complete_attempt(
        parent,
        attempt,
        {"objectives": {"quality": 2.0, "latency": 1.0}},
    )
    child_attempt = store.start_attempt(child)
    store.fail_attempt(child, child_attempt, "oom")
    assert store.candidate(child)["status"] == "failed"
    assert store.lineage(child)[0]["id"] == parent

    metrics = {parent: SelectionMetrics(0, 1.0, 0.5)}
    store.update_selection((parent,), (parent,), metrics)
    assert store.population() == [parent]
    assert store.frontier()[0]["id"] == parent
    assert store.summary()["candidates"] == {"completed": 1, "failed": 1}
    store.close()


def test_store_recovers_running_attempt(tmp_path):
    store = StudyStore(tmp_path / "study.sqlite3")
    store.initialize({}, {})
    candidate_id = store.add_candidate(config(), 1, {"operator": "seed"})
    assert candidate_id is not None
    store.start_attempt(candidate_id)
    assert store.recover_running() == 1
    assert store.candidate(candidate_id)["status"] == "pending"
    store.close()
