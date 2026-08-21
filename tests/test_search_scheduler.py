from speck.model import Config, LayerConfig
from speck.search.architecture import mutate
from speck.search.scheduler import _add_architecture, advance
from speck.search.spec import SearchSettings
from speck.search.study import SearchStudy


def settings():
    return SearchSettings.from_dict({
        "format_version": 2,
        "seed": 7,
        "max_architectures": 8,
        "initial_population": 2,
        "population_size": 4,
        "cohort_size": 2,
        "confidence_z": 1.645,
        "space": {
            "min_layers": 1,
            "max_layers": 3,
            "hidden_size_min": 8,
            "hidden_size_max": 12,
            "hidden_size_step": 4,
            "intermediate_size_min": 16,
            "intermediate_size_max": 24,
            "intermediate_size_step": 8,
            "kv_heads": [1, 2],
        },
        "quality": {
            "data_dir": "~/data",
            "batch_tokens": 8,
            "device_batch_size": 1,
            "eval_batch_size": 1,
            "lr": 0.001,
            "min_lr": 0.1,
            "warmup_steps": 1,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "optimizer": "adamw",
        },
        "validation_slices": [{"name": "main"}],
        "inference": {"contexts": [4], "warmup_samples": 0, "samples": 1},
        "quantization": {"bits": 4, "group_size": 4},
        "rungs": [
            {
                "name": "screen",
                "architecture_limit": 8,
                "seed_count": 1,
                "train_tokens": 16,
                "sequence_length": 4,
                "eval_every_tokens": 8,
                "eval_tokens": 8,
                "inference_samples": 1,
            },
            {
                "name": "verify",
                "architecture_limit": 2,
                "seed_count": 2,
                "train_tokens": 32,
                "sequence_length": 4,
                "eval_every_tokens": 16,
                "eval_tokens": 16,
                "inference_samples": 2,
            },
        ],
    })


def baseline():
    return Config(
        vocab_size=16,
        layers=(LayerConfig(8, 16, 1),),
        head_dim=4,
        max_position_embeddings=8,
    )


def result(trial):
    value = float(trial["architecture_id"] + trial["seed_index"] / 10)
    return {
        "objectives": {
            "quality.validation_nll.main": value,
            "memory.kv_cache_bytes_per_token": value,
            "memory.quantized_weight_bytes": value,
            "prefill.ms.context_4": value,
            "decode.ms_per_token.context_4": value,
            "memory.inference_peak_bytes.context_4": value,
        }
    }


def complete_pending(study):
    for trial in study.trials(status="pending"):
        attempt = study.start_attempt(trial["id"])
        study.complete_attempt(trial["id"], attempt, result(trial))


def finish(study, reverse=False, fail=None):
    for _ in range(128):
        advance(study, baseline(), settings())
        pending = study.trials(status="pending")
        if pending:
            trial = pending[-1] if reverse else pending[0]
            attempt = study.start_attempt(trial["id"])
            if fail is not None and fail(trial):
                study.fail_attempt(trial["id"], attempt, "test failure")
            else:
                study.complete_attempt(trial["id"], attempt, result(trial))
        if study.study()["status"] != "running":
            return
    raise AssertionError("study did not reach a terminal state")


def test_scheduler_runs_all_rungs_and_recommends(tmp_path):
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize(settings().export(), {})
    for _ in range(32):
        advance(study, baseline(), settings())
        complete_pending(study)
        if study.study()["status"] == "completed":
            break
    assert len(study.architectures()) == 8
    assert len(study.rungs(rung=1)) == 2
    assert len(study.trials(rung=0)) == 8
    assert len(study.trials(rung=1)) == 4
    promoted = sorted(
        (
            study.rung(item["architecture_id"], 0)["decision"]
            for item in study.rungs(rung=1)
        ),
        key=lambda item: item["event"],
    )
    assert [item["lane"] for item in promoted] == ["conservative", "optimistic"]
    assert [item["bound"] for item in promoted] == ["upper", "lower"]
    assert [len(item["eligible_architecture_ids"]) for item in promoted] == [4, 8]
    assert len({trial["seed"] for trial in study.trials(rung=0)}) == 1
    assert len({trial["seed"] for trial in study.trials(rung=1)}) == 2
    assert study.study()["status"] == "completed"
    assert study.study()["recommendations"]["balanced"]["id"]
    study.close()


def test_scheduler_is_reproducible(tmp_path):
    snapshots = []
    for name, reverse in (("first", False), ("second", True)):
        study = SearchStudy(tmp_path / f"{name}.sqlite3")
        study.initialize(settings().export(), {})
        finish(study, reverse=reverse)
        snapshots.append({
            "hashes": [
                item["architecture_hash"] for item in study.architectures()
            ],
            "promotions": sorted(
                (
                    study.rung(item["architecture_id"], 0)["decision"]["event"],
                    item["architecture_id"],
                )
                for item in study.rungs(rung=1)
            ),
            "outcomes": [
                (item["architecture_id"], item["operator"], item["success"])
                for item in study.outcomes()
            ],
        })
        study.close()
    assert snapshots[0] == snapshots[1]


def test_scheduler_continues_after_source_and_final_failures(tmp_path):
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize(settings().export(), {})
    failed_final = []

    def fail(trial):
        if trial["rung"] == 0 and trial["architecture_id"] == 1:
            return True
        if trial["rung"] == 1 and not failed_final:
            failed_final.append(trial["architecture_id"])
        return (
            trial["rung"] == 1
            and trial["architecture_id"] == failed_final[0]
            and trial["seed_index"] == 0
        )

    finish(study, fail=fail)
    assert study.study()["status"] == "completed"
    assert len(study.architectures()) == 8
    assert len(study.rungs(rung=1)) == 2
    assert len([item for item in study.rungs(rung=1) if item["aggregate"]]) == 1
    assert study.study()["recommendations"]["balanced"]["id"]
    study.close()


def test_scheduler_fails_when_no_initial_architecture_succeeds(tmp_path):
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize(settings().export(), {})
    finish(study, fail=lambda trial: trial["rung"] == 0)
    assert study.study()["status"] == "failed"
    assert len(study.architectures()) == settings().initial_population
    assert not study.trials(status="pending")
    study.close()


def test_initial_mutation_replays_from_repaired_parent(tmp_path):
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize(settings().export(), {})
    unrepaired = Config(
        vocab_size=16,
        layers=(LayerConfig(8, 17, 1),),
        head_dim=4,
        max_position_embeddings=8,
    )
    advance(study, unrepaired, settings())
    parent, child = study.architectures()
    replay = mutate(
        Config.from_dict(parent["config"]),
        settings().space,
        child["generation_seed"],
        child["operator"],
    )
    assert replay.config.settings() == child["config"]
    assert replay.repairs == tuple(child["repairs"])
    study.close()


def test_scheduler_fills_an_interrupted_cohort_before_dispatch(tmp_path):
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize(settings().export(), {})
    advance(study, baseline(), settings())
    complete_pending(study)
    advance(study, baseline(), settings())
    complete_pending(study)
    advance(study, baseline(), settings())
    complete_pending(study)
    assert _add_architecture(study, settings(), baseline(), 4)
    assert len(study.architectures()) == 5
    assert len(study.trials(status="pending")) == 1
    advance(study, baseline(), settings())
    assert len(study.architectures()) == 6
    assert len(study.trials(status="pending")) == 2
    study.close()


def test_scheduler_persists_architecture_space_exhaustion(tmp_path):
    constrained = settings().export()
    constrained["max_architectures"] = 2
    constrained["initial_population"] = 1
    constrained["cohort_size"] = 1
    constrained["population_size"] = 1
    constrained["space"].update({
        "min_layers": 1,
        "max_layers": 1,
        "hidden_size_min": 8,
        "hidden_size_max": 8,
        "intermediate_size_min": 16,
        "intermediate_size_max": 16,
        "kv_heads": [1],
        "min_attention_layers": 1,
        "max_attention_layers": 1,
    })
    constrained["rungs"] = [{
        **constrained["rungs"][0],
        "architecture_limit": 2,
    }]
    fixed = SearchSettings.from_dict(constrained)
    study = SearchStudy(tmp_path / "study.sqlite3")
    study.initialize(fixed.export(), {})
    advance(study, baseline(), fixed)
    complete_pending(study)
    advance(study, baseline(), fixed)
    assert study.study()["status"] == "failed"
    assert not advance(study, baseline(), fixed)
    study.close()
