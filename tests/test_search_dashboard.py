import pytest

from scripts.search_dashboard import _pareto_ranks, candidate, dashboard, snapshot
from speck.architecture import ArchitectureConfig, BlockConfig, BlockGroup, StageConfig, SwiGLUSpec
from speck.model import Config, LayerConfig
from speck.search.protocol import ObjectiveSet, ObjectiveSpec
from speck.search.store import StudyStore
from speck.search.study import SearchStudy
from speck.search.study_v3 import V3Study


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


def test_dashboard_projects_v3_objective_sets_horizons_and_native_blocks(tmp_path):
    database = tmp_path / "study.sqlite3"
    objectives = ObjectiveSet(
        "gpu_short",
        (
            ObjectiveSpec("quality.target_nll", "minimize", "quality"),
            ObjectiveSpec("gpu_short.throughput", "maximize", "efficiency"),
            ObjectiveSpec(
                "quality.procedural_score",
                "maximize",
                "reporting",
                required_for_selection=False,
            ),
        ),
    )
    revised_objectives = ObjectiveSet(
        "gpu_short",
        (ObjectiveSpec("quality.target_nll", "minimize", "quality"),),
    )
    config = {
        "format_version": 3,
        "calibration": {
            "noise_tokens": 4,
            "broad_architectures": 2,
            "broad_tokens": 8,
            "anchor_tokens": 16,
        },
        "objective_sets": [
            {
                "name": objectives.name,
                "objectives": [
                    {
                        "name": item.name,
                        "direction": item.direction,
                        "role": item.role,
                        "required_for_selection": item.required_for_selection,
                    }
                    for item in objectives.objectives
                ],
            }
        ],
        "quality": {"checkpoint_tokens": [4, 8, 16]},
        "planner": {"total_cost": 100},
    }
    first = ArchitectureConfig(
        (BlockGroup(BlockConfig(8, (StageConfig((SwiGLUSpec(16),)),))),),
        embedding_size=8,
        vocab_size=16,
        max_position_embeddings=16,
    )
    second = ArchitectureConfig(
        (BlockGroup(BlockConfig(12, (StageConfig((SwiGLUSpec(24),)),))),),
        embedding_size=8,
        vocab_size=16,
        max_position_embeddings=16,
    )
    study = V3Study(database)
    study.initialize_bundle(
        config,
        {"git": {"revision": "test"}},
        objective_sets=(objectives,),
        architecture=first,
        static={"logical_depth": 1, "parameters": 100, "unique_parameter_blocks": 1},
        operation={"operator": "baseline"},
    )
    study.add_architecture(
        second,
        {"logical_depth": 1, "parameters": 120, "unique_parameter_blocks": 1},
        {
            "operator": "broad_sample",
            "parents": [{"digest": first.digest, "role": "mutation"}],
        },
    )
    study.add_objective_set(revised_objectives)
    study.add_observation(
        first.digest,
        objectives.digest,
        "quality.target_nll",
        9.0,
        tokens=4,
        source="measured",
    )
    study.add_observation(
        first.digest,
        objectives.digest,
        "quality.target_nll",
        20.0,
        source="measured",
    )
    for architecture, nll, throughput in (
        (first, 1.0, 20.0),
        (second, 2.0, 10.0),
    ):
        study.add_observation(
            architecture.digest,
            objectives.digest,
            "quality.target_nll",
            nll,
            tokens=8,
            source="quality_evaluation",
        )
        study.add_observation(
            architecture.digest,
            objectives.digest,
            "gpu_short.throughput",
            throughput,
            source="profile",
        )
    action_id = study.add_action(
        "probe",
        1.0,
        1.0,
        {"architecture_digest": second.digest},
    )
    claimed = study.claim_action("dashboard-secret-test")
    assert claimed["id"] == action_id
    assert claimed["claim_token"]
    study.close()

    before = database.read_bytes()
    state = snapshot(database)
    assert database.read_bytes() == before
    assert state["format_version"] == 3
    assert state["comparison_tokens"] == 8
    assert state["active_objective_set"]["name"] == "gpu_short"
    assert state["active_objective_set"]["digest"] == objectives.digest
    assert len(state["objective_sets"]) == 2
    assert all(" / " in item["label"] for item in state["objective_sets"])
    assert state["objective_directions"]["gpu_short.throughput"] == "maximize"
    assert state["architecture_budget"] == 2
    assert state["frontier_closed"]
    assert len(state["frontier"]) == 1
    assert state["frontier"][0]["architecture_hash"] == first.digest
    assert state["frontier"][0]["objectives"]["quality.target_nll"] == 1.0
    assert state["frontier"][0]["pareto_rank"] == 0
    second_summary = next(
        item for item in state["candidates"] if item["architecture_hash"] == second.digest
    )
    assert second_summary["pareto_rank"] == 1
    assert second_summary["parents"] == [state["frontier"][0]["id"]]

    revised_state = snapshot(database, objective_set=revised_objectives.digest)
    assert revised_state["active_objective_set"]["digest"] == revised_objectives.digest
    assert revised_state["comparison_tokens"] is None
    with pytest.raises(ValueError, match="ambiguous"):
        snapshot(database, objective_set="gpu_short")

    detail = candidate(database, second_summary["id"])
    assert detail["format_version"] == 3
    assert detail["config"]["blocks"][0]["block"]["stages"][0]["branches"][0]["kind"] == "swiglu"
    assert detail["parent_roles"] == [{"id": state["frontier"][0]["id"], "role": "mutation"}]
    assert detail["actions"][0]["owner"] == "dashboard-secret-test"
    assert "claim_token" not in detail["actions"][0]
    assert detail["observations"]


def test_dashboard_asset_has_responsive_operator_views():
    for landmark in (
        'id="overview-grid"',
        'id="rung-pipeline"',
        'id="recommendations"',
        'id="trend-chart"',
        'id="pareto-chart"',
        'id="history-table"',
        'id="inspector"',
        'id="objective-set-select"',
        'id="horizon-select"',
    ):
        assert landmark in dashboard
    assert "@media (max-width: 760px)" in dashboard
    assert "data-architecture" in dashboard
    assert "parameters.set('rung'" in dashboard
    assert "AbortController" in dashboard
    assert "scrollIntoView" in dashboard
    assert "role=\"button\"" in dashboard
