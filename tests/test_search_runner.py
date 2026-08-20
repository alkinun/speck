from speck.model import Config, LayerConfig
from speck.search.runner import (
    SearchSettings,
    _ingest_output,
    generate_offspring,
    run_search,
    search_objectives,
    seed_candidates,
    update_selection,
)
from speck.search.store import StudyStore


def settings():
    return SearchSettings.from_dict({
        "population_size": 2,
        "initial_population": 2,
        "max_evaluations": 4,
        "seed": 7,
        "space": {
            "min_layers": 1,
            "max_layers": 2,
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
            "train_tokens": 16,
            "batch_tokens": 8,
            "device_batch_size": 1,
            "sequence_length": 4,
            "eval_every_tokens": 8,
            "eval_batch_size": 1,
            "eval_tokens": 4,
            "lr": 0.001,
            "min_lr": 0.1,
            "warmup_steps": 1,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "optimizer": "adamw",
            "compile": False,
        },
        "inference": {"contexts": [4, 8], "warmup_samples": 0, "samples": 1},
        "quantization": {"bits": 4, "group_size": 4},
    })


def baseline():
    return Config(
        vocab_size=16,
        layers=(LayerConfig(8, 16, 1),),
        head_dim=4,
        max_position_embeddings=10,
    )


def result(value):
    return {
        "objectives": {
            "quality.validation_nll": value,
            "memory.kv_cache_bytes_per_token": value,
            "memory.quantized_weight_bytes": value,
            "prefill.ms.context_4": value,
            "decode.ms_per_token.context_4": value,
            "memory.inference_peak_bytes.context_4": value,
            "prefill.ms.context_8": value,
            "decode.ms_per_token.context_8": value,
            "memory.inference_peak_bytes.context_8": value,
        }
    }


def test_search_settings_and_objectives():
    search = settings()
    assert search.quality.data_dir.endswith("/data")
    assert len(search_objectives(search)) == 9
    assert search.export()["inference"]["contexts"] == [4, 8]


def test_seed_selection_and_offspring(tmp_path):
    store = StudyStore(tmp_path / "study.sqlite3")
    store.initialize(settings().export(), {})
    seed_candidates(store, baseline(), settings())
    candidates = store.candidates()
    assert len(candidates) == 2
    for index, candidate in enumerate(candidates):
        attempt = store.start_attempt(candidate["id"])
        store.complete_attempt(candidate["id"], attempt, result(index + 1))
    metrics, selected = update_selection(store, settings())
    assert len(metrics) == 2
    assert len(selected) == 2
    offspring = generate_offspring(store, settings())
    assert store.candidate(offspring)["parents"]
    assert store.candidate(offspring)["status"] == "pending"
    store.close()


def test_search_lifecycle_and_resume(tmp_path, monkeypatch):
    store = StudyStore(tmp_path / "study.sqlite3")
    store.initialize(settings().export(), {})

    def evaluate(store, study_dir, candidate, tokenizer, search, device):
        attempt = store.start_attempt(candidate["id"])
        store.complete_attempt(candidate["id"], attempt, result(candidate["id"]))
        return True

    monkeypatch.setattr(
        "speck.search.runner.evaluate_candidate_process", evaluate
    )
    run_search(store, tmp_path, baseline(), {}, settings(), "cpu")
    assert store.summary()["candidates"] == {"completed": 4}
    assert store.summary()["study"]["status"] == "completed"
    assert len(store.frontier()) == 1
    assert any(
        store.candidate(candidate["id"])["parents"]
        for candidate in store.candidates()[1:]
    )

    run_search(store, tmp_path, baseline(), {}, settings(), "cpu")
    assert len(store.candidates()) == 4
    store.close()


def test_structured_worker_failure_retries_once(tmp_path):
    store = StudyStore(tmp_path / "study.sqlite3")
    store.initialize(settings().export(), {})
    candidate_id = store.add_candidate(baseline(), 1, {"operator": "seed"})
    assert candidate_id is not None

    first_attempt = store.start_attempt(candidate_id)
    output = {
        "status": "failed",
        "candidate_id": candidate_id,
        "attempt_id": first_attempt,
        "error_type": "RuntimeError",
        "error": "transient",
    }
    assert not _ingest_output(
        store, candidate_id, first_attempt, output, settings()
    )
    assert store.candidate(candidate_id)["status"] == "pending"

    second_attempt = store.start_attempt(candidate_id)
    output["attempt_id"] = second_attempt
    assert not _ingest_output(
        store, candidate_id, second_attempt, output, settings()
    )
    assert store.candidate(candidate_id)["status"] == "failed"
    store.close()


def test_evicted_candidate_does_not_reenter_population(tmp_path):
    store = StudyStore(tmp_path / "study.sqlite3")
    store.initialize(settings().export(), {})
    candidate_ids = []
    for index, (hidden, intermediate) in enumerate(
        ((8, 16), (12, 16), (8, 24)), start=1
    ):
        candidate = Config(
            vocab_size=16,
            layers=(LayerConfig(hidden, intermediate, 1),),
            head_dim=4,
            max_position_embeddings=10,
        )
        candidate_id = store.add_candidate(
            candidate,
            index,
            {"operator": "seed" if index == 1 else "change_hidden_size"},
        )
        assert candidate_id is not None
        attempt = store.start_attempt(candidate_id)
        store.complete_attempt(candidate_id, attempt, result(index))
        update_selection(store, settings())
        candidate_ids.append(candidate_id)

    evicted = candidate_ids[2]
    assert evicted not in store.population()

    fourth = Config(
        vocab_size=16,
        layers=(LayerConfig(12, 24, 1),),
        head_dim=4,
        max_position_embeddings=10,
    )
    fourth_id = store.add_candidate(
        fourth, 4, {"operator": "change_ffn_width"}
    )
    assert fourth_id is not None
    attempt = store.start_attempt(fourth_id)
    store.complete_attempt(fourth_id, attempt, result(4))
    update_selection(store, settings())
    assert evicted not in store.population()
    assert store.candidate(evicted)["pareto_rank"] is None
    store.close()
