from scripts.search_dashboard import _pareto_ranks, candidate, dashboard, snapshot
from speck.model import Config, LayerConfig
from speck.search.store import StudyStore
from speck.search.study import SearchStudy


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
    assert state["generated"] == state["screened"] == 1
    assert state["data_revision"] == state["updated_at"]
    assert state["candidates"][0]["comparison_status"] == "completed"
    assert state["frontier"][0]["parameters"] == 123
    assert state["objectives"] == ["quality.validation_nll"]
    detail = candidate(database, candidate_id)
    assert detail["mutation"] == {"operator": "seed"}
    assert detail["result"]["quality"]["train_curve"][0]["loss"] == 2.1


def test_dashboard_recomputes_every_pareto_rank():
    assert _pareto_ranks({
        1: {"quality": 1.0, "latency": 1.0},
        2: {"quality": 2.0, "latency": 2.0},
        3: {"quality": 3.0, "latency": 3.0},
    }) == {1: 0, 2: 1, 3: 2}


def test_dashboard_projects_v2_architectures_rungs_and_trials(tmp_path):
    database = tmp_path / "study.sqlite3"
    study = SearchStudy(database)
    study.initialize(
        {
            "format_version": 2,
            "max_architectures": 4,
            "rungs": [
                {"name": "screen", "architecture_limit": 4},
                {"name": "verify", "architecture_limit": 1},
            ],
            "validation_slices": [{"name": "main", "objective": True}],
        },
        {},
    )
    config = Config(
        vocab_size=16,
        layers=(LayerConfig(8, 16, 1),),
        head_dim=4,
    )
    architecture_id = study.add_architecture_with_rung(
        config,
        {"parameters": 123},
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
    study.complete_attempt(trial["id"], attempt, {
        "objectives": {"quality.validation_nll.main": 2.0},
        "quality": {
            "train_curve": [{"tokens": 4, "loss": 2.1}],
            "validation_curve": [{"tokens": 4, "loss": 2.0}],
        },
    })
    study.update_rung(
        architecture_id,
        0,
        "complete",
        {
            "trials": [trial["id"]],
            "objectives": {
                "quality.validation_nll.main": {
                    "n": 1,
                    "mean": 2.0,
                    "stdev": 0.0,
                    "lower": 1.9,
                    "upper": 2.1,
                }
            },
        },
        rank=0,
        crowding=1.0,
        novelty=1.0,
    )
    assert study.promote(
        architecture_id,
        0,
        1,
        (22,),
        {"reason": "test"},
    )
    verify_trial = study.trials(status="pending")[0]
    verify_attempt = study.start_attempt(verify_trial["id"])
    study.complete_attempt(verify_trial["id"], verify_attempt, {
        "objectives": {"quality.validation_nll.main": 1.8},
        "quality": {
            "train_curve": [{"tokens": 8, "loss": 1.9}],
            "validation_curve": [{"tokens": 8, "loss": 1.8}],
        },
    })
    study.update_rung(
        architecture_id,
        1,
        "complete",
        {
            "trials": [verify_trial["id"]],
            "objectives": {
                "quality.validation_nll.main": {
                    "n": 1,
                    "mean": 1.8,
                    "stdev": 0.0,
                    "lower": 1.7,
                    "upper": 1.9,
                }
            },
        },
        rank=0,
        crowding=1.0,
        novelty=1.0,
    )
    study.close()

    state = snapshot(database)
    assert state["format_version"] == 2
    assert state["architecture_counts"] == {"completed": 1}
    assert state["trial_counts"] == {"completed": 2}
    assert state["screened"] == 1
    assert state["frontier_rung"] == 0
    assert not state["frontier_closed"]
    assert state["available_rungs"] == [0, 1]
    assert state["rungs"][1]["trial_counts"] == {"completed": 1}
    assert state["frontier"][0]["parameters"] == 123
    assert state["objectives"] == ["quality.validation_nll.main"]
    verify_state = snapshot(database, rung=1)
    assert verify_state["frontier_rung"] == 1
    assert verify_state["candidates"][0]["comparison_status"] == "completed"
    assert verify_state["frontier"][0]["objectives"] == {
        "quality.validation_nll.main": 1.8
    }
    detail = candidate(database, architecture_id)
    assert detail["mutation"] == {"operator": "seed"}
    assert detail["rungs"][0]["aggregate"]["objectives"]
    assert detail["trials"][0]["seed"] == 22
    assert detail["result_trial"] == {
        "id": verify_trial["id"],
        "rung": 1,
        "seed_index": 0,
        "seed": 22,
    }
    assert detail["result"]["quality"]["train_curve"][0]["loss"] == 1.9

    study = SearchStudy(database)
    study.update_rung(architecture_id, 1, "failed")
    study.close()
    failed_state = snapshot(database)
    assert failed_state["architecture_counts"] == {"failed": 1}
    assert failed_state["candidates"][0]["comparison_status"] == "completed"


def test_dashboard_asset_has_responsive_operator_views():
    for landmark in (
        'id="overview-grid"',
        'id="rung-pipeline"',
        'id="recommendations"',
        'id="trend-chart"',
        'id="pareto-chart"',
        'id="history-table"',
        'id="inspector"',
    ):
        assert landmark in dashboard
    assert "@media (max-width: 760px)" in dashboard
    assert "data-architecture" in dashboard
    assert "?rung=" in dashboard
    assert "AbortController" in dashboard
    assert "scrollIntoView" in dashboard
    assert "role=\"button\"" in dashboard
