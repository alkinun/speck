import json
import math
import subprocess
import time
from pathlib import Path

import pytest
import torch

import scripts.search as search_script
from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    GatedCausalConvSpec,
    StageConfig,
    SwiGLUSpec,
)
from speck.checkpoint import save
from speck.dataloader import manifest_fingerprint
from speck.model import SpeckForCausalLM
from speck.search import (
    MUTATIONS,
    CandidatePlan,
    SearchSettings,
    StudyStore,
    aggregate_final_runs,
    atomic_json,
    first_incomplete_candidate,
    initial_generation,
    later_generation,
    loader_state,
    materialize_generation,
    mutate_architecture,
    normalize_baseline,
    open_study,
    parameter_count,
    percentile_ranks,
    project_learning_curve,
    prune_checkpoints,
    random_architecture,
    score_candidates,
    select_finalists,
    select_parent,
    select_promotions,
    sequence_state_bytes,
    status_snapshot,
    validate_architecture,
    validation_slices,
)

root = Path(__file__).parents[1]
experiment = root / "experiments" / "speck00-200m"


def tiny_settings():
    values = json.loads((experiment / "search.json").read_text(encoding="utf-8"))
    values.update(
        seed=7,
        parameter_bounds=[1, 1_000_000],
        logical_depth_bounds=[2, 8],
        baseline_normalization={
            "expand_unshared_repetitions": True,
            "width_replacements": {},
        },
        rungs=[8, 16, 32],
        final_tokens=64,
        widths=[8, 12, 16],
        head_dimensions=[4, 8],
        kv_heads=[1, 2],
        sliding_windows=[2, 4, 8],
        depth_probabilities={"4": 1.0},
        embedding_width_probabilities={"8": 0.4, "12": 0.3, "16": 0.3},
        attention_head_dimension_probabilities={"4": 0.75, "8": 0.25},
        kv_head_probabilities={"1": 0.6, "2": 0.4},
    )
    values["training"] = {
        "sequence_length": 1,
        "device_batch_size": 2,
        "batch_tokens": 2,
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "warmup_tokens": 2,
        "schedule_tokens": 64,
        "minimum_learning_rate_scale": 0.1,
        "deterministic": True,
        "cublas_workspace_config": ":4096:8",
        "checkpoints": [2, 4, 8, 16, 32],
    }
    values["evaluation"] = {
        "monitor_offset": 0,
        "monitor_tokens": 2,
        "final_offset": 2,
        "final_tokens": 4,
    }
    values["profile"].update(
        device="cpu",
        parameter_dtype="float32",
        compute_dtype="float32",
        seed=7,
        warmups=1,
        requests=2,
        prompt_lengths=[2, 4],
        generated_tokens=2,
    )
    values["final_profile"].update(
        warmups=1,
        gpu_requests=2,
        cpu_requests=2,
        compile_mode="default",
    )
    return SearchSettings.from_dict(values)


def rich_architecture():
    return ArchitectureConfig(
        (
            BlockGroup(
                BlockConfig(
                    8,
                    (
                        StageConfig((AttentionSpec(4, 1, "sliding", 4),)),
                        StageConfig((SwiGLUSpec(16),)),
                    ),
                ),
                repeat=2,
                weight_sharing="all",
            ),
            BlockGroup(
                BlockConfig(
                    12,
                    (
                        StageConfig((GatedCausalConvSpec(12, 3),)),
                        StageConfig((SwiGLUSpec(36),)),
                    ),
                )
            ),
            BlockGroup(BlockConfig(16, (StageConfig((AttentionSpec(8, 1),)),))),
            BlockGroup(BlockConfig(12, (StageConfig((SwiGLUSpec(24),)),))),
        ),
        8,
        vocab_size=16,
        max_position_embeddings=16,
    )


def profile(value):
    return {
        "latency": {
            "prefill_512": {"p50_seconds": value},
            "prefill_2048": {"p50_seconds": value},
            "decode_2048": {"p50_seconds": value},
        },
        "static": {
            "weight_bytes": value,
            "state_bytes": {"2048": value},
        },
        "memory": {"peak_vram_bytes": value},
    }


def record(candidate_id, architecture, score=0.0):
    return {
        "candidate_id": f"{candidate_id:06d}",
        "digest": architecture.digest,
        "architecture": architecture.settings(),
        "status": "ready",
        "trained_tokens": 16,
        "scores": {
            "quality": score,
            "balanced": score,
            "efficiency": score,
        },
        "scores_by_rung": {
            "16": {
                "quality": score,
                "balanced": score,
                "efficiency": score,
            }
        },
    }


def cached_logits(model, tokens):
    state = model.state(length=tokens.size(1))
    return torch.cat(
        [model(tokens[:, index : index + 1], state=state) for index in range(tokens.size(1))],
        dim=1,
    )


def test_search_configuration_is_complete():
    settings = SearchSettings.from_dict(
        json.loads((experiment / "search.json").read_text(encoding="utf-8"))
    )
    assert tuple(settings["mutation_probabilities"]) == MUTATIONS
    assert sum(settings["mutation_probabilities"].values()) == pytest.approx(1)
    assert sum(settings["parent_lane_probabilities"].values()) == pytest.approx(1)


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_every_mutation_changes_one_valid_genome(mutation):
    settings = tiny_settings()
    parent = rich_architecture()
    result = mutate_architecture(parent, settings, 100 + MUTATIONS.index(mutation), mutation)
    assert result.name == mutation
    assert result.architecture.digest != parent.digest
    validate_architecture(result.architecture, settings)


def test_parameter_and_state_accounting_match_the_model():
    settings = tiny_settings()
    architecture = rich_architecture()
    model = SpeckForCausalLM(architecture)
    assert parameter_count(architecture) == model.parameter_count()
    state = model.state(length=8, dtype=torch.bfloat16)
    assert sequence_state_bytes(architecture, 8) == state.allocated_bytes()
    metrics = validate_architecture(architecture, settings)
    assert settings["logical_depth_bounds"][0] <= metrics["logical_depth"]
    assert metrics["parameters"] <= settings["parameter_bounds"][1]


def test_random_architectures_are_deterministic_and_feasible():
    settings = tiny_settings()
    baseline = rich_architecture()
    first = [random_architecture(baseline, settings, seed) for seed in range(20)]
    second = [random_architecture(baseline, settings, seed) for seed in range(20)]
    assert [value.digest for value in first] == [value.digest for value in second]
    assert all(validate_architecture(value, settings) for value in first)


def test_initial_generation_is_deterministic_and_unique():
    settings = tiny_settings()
    baseline = rich_architecture()
    first = initial_generation(baseline, settings)
    second = initial_generation(baseline, settings)
    assert len(first) == settings["generation_size"]
    assert [plan.architecture.digest for plan in first] == [
        plan.architecture.digest for plan in second
    ]
    assert len({plan.architecture.digest for plan in first}) == len(first)
    with pytest.raises(RuntimeError, match="duplicates"):
        initial_generation(
            baseline,
            settings,
            {normalize_baseline(baseline, settings).digest},
        )


def test_later_generation_and_parent_tournament_are_deterministic():
    settings = tiny_settings()
    baseline = rich_architecture()
    plans = initial_generation(baseline, settings)
    archive = [
        record(index, plan.architecture, index / len(plans))
        for index, plan in enumerate(plans[:8], 1)
    ]
    first_parent = select_parent(archive, settings, 55)
    second_parent = select_parent(archive, settings, 55)
    assert first_parent[0]["candidate_id"] == second_parent[0]["candidate_id"]
    assert first_parent[1] == second_parent[1]
    first = later_generation(baseline, archive, settings, 1)
    second = later_generation(baseline, archive, settings, 1)
    assert [plan.architecture.digest for plan in first] == [
        plan.architecture.digest for plan in second
    ]
    assert len(first) == settings["generation_size"]
    assert len({plan.architecture.digest for plan in first}) == len(first)


def test_learning_curves_forecast_sustained_improvement():
    fast_plateau = [
        {"tokens": 1, "nll": 5.0},
        {"tokens": 2, "nll": 4.0},
        {"tokens": 4, "nll": 4.0},
    ]
    slow_sustained = [
        {"tokens": 1, "nll": 5.0},
        {"tokens": 2, "nll": 4.2},
        {"tokens": 4, "nll": 3.4},
    ]
    fast = project_learning_curve(fast_plateau, 8, -0.65)
    slow = project_learning_curve(slow_sustained, 8, -0.65)
    assert slow["projected_nll"] < fast["projected_nll"]
    noisy = project_learning_curve(
        [
            {"tokens": 1, "nll": 5.0},
            {"tokens": 2, "nll": 3.0},
            {"tokens": 4, "nll": 4.0},
        ],
        8,
        -1.0,
    )
    assert noisy["effective_slope"] != pytest.approx(noisy["measured_slope"])
    assert min(noisy["measured_slope"], -1.0) <= noisy["effective_slope"]
    assert noisy["effective_slope"] <= max(noisy["measured_slope"], -1.0)


def test_candidate_scoring_and_lane_promotions():
    curves = (
        [5.0, 4.5, 4.0],
        [5.2, 4.6, 3.9],
        [5.1, 4.8, 4.5],
        [5.3, 4.9, 4.6],
    )
    records = []
    for index, curve in enumerate(curves, 1):
        records.append(
            {
                "candidate_id": str(index),
                "status": "ready",
                "nll_curve": [
                    {"tokens": tokens, "nll": nll} for tokens, nll in zip((1, 2, 4), curve)
                ],
                "profile": profile(5 - index),
            }
        )
    scored = score_candidates(records, 8)
    assert all({"quality", "balanced", "efficiency"} <= set(value["scores"]) for value in scored)
    promoted = select_promotions(
        scored,
        {"quality": 2, "balanced": 1, "efficiency": 1},
    )
    assert len(promoted) == 4
    assert len({value["candidate_id"] for value in promoted}) == 4
    assert percentile_ranks({"a": 1, "b": 1, "c": 2}) == {
        "a": 0.25,
        "b": 0.25,
        "c": 1.0,
    }


def test_atomic_records_resume_and_checkpoint_pruning(tmp_path):
    path = tmp_path / "result.json"
    atomic_json(path, {"status": "pending"})
    assert json.loads(path.read_text()) == {"status": "pending"}
    assert not path.with_name("result.json.tmp").exists()

    settings = tiny_settings()
    store = open_study(tmp_path / "study", tmp_path, settings, generations=1)
    plans = initial_generation(rich_architecture(), settings)
    materialize_generation(store, plans, 0, settings)
    store.update_result("000001", status="completed")
    store.update_result("000002", status="running")
    assert first_incomplete_candidate(store.results())["candidate_id"] == "000002"

    checkpoint = store.candidate_path("000001") / "checkpoint"
    checkpoint.mkdir()
    for step in (1, 2):
        for name in (f"model_{step:06d}.pt", f"optimizer_{step:06d}.pt"):
            (checkpoint / name).write_bytes(b"value")
        (checkpoint / f"metadata_{step:06d}.json").write_text("{}")
        (checkpoint / f"complete_{step:06d}").write_text("complete\n")
    removed = prune_checkpoints(checkpoint, {2})
    assert removed > 0
    assert not (checkpoint / "complete_000001").exists()
    assert (checkpoint / "complete_000002").exists()


def test_partial_study_initialization_and_provenance_resume(tmp_path):
    settings = tiny_settings()
    directory = tmp_path / "study"
    (directory / "candidates").mkdir(parents=True)
    atomic_json(directory / "search.json", settings.settings())
    provenance = {"inputs": {"fixture": "one"}, "runtime": {"device": "cpu"}}
    store = open_study(
        directory,
        tmp_path,
        settings,
        generations=1,
        provenance=provenance,
    )
    assert store.state()["provenance"] == provenance
    with pytest.raises(ValueError, match="inputs or runtime"):
        open_study(
            directory,
            tmp_path,
            settings,
            generations=1,
            provenance={"inputs": {"fixture": "two"}},
        )
    state = store.state()
    state["active_since"] = time.time() - 10
    store.write_state(state)
    assert status_snapshot(store)["elapsed_seconds"] >= 9
    state = store.state()
    state["elapsed_seconds"] = 5.0
    state["active_since"] = time.time() - 100
    atomic_json(store.state_path, state)
    open_study(
        directory,
        tmp_path,
        settings,
        generations=2,
        provenance=provenance,
    )
    resumed = store.state()
    assert resumed["active_since"] is None
    assert resumed["elapsed_seconds"] == 5.0


def test_completed_checkpoint_reconciles_result(tmp_path, monkeypatch):
    settings = tiny_settings()
    runtime = search_script._runtime_contract(settings, torch.device("cpu"))
    store = open_study(
        tmp_path / "study",
        tmp_path,
        settings,
        generations=1,
        provenance={"inputs": {}, "runtime": runtime},
    )
    architecture = rich_architecture()
    materialize_generation(
        store,
        (CandidatePlan(architecture, None, None),),
        0,
        settings,
    )
    store.update_result("000001", status="running")
    torch.manual_seed(settings["seed"])
    model = SpeckForCausalLM(architecture)
    model.init_weights()
    optimizer = model.optimizer(
        settings["training"]["learning_rate"],
        settings["training"]["weight_decay"],
        settings["training"]["optimizer"],
    )
    manifest = {"fixture": "packed"}
    metadata = {
        "format_version": 1,
        "step": 4,
        "trained_tokens": 8,
        "config": architecture.settings(),
        "architecture_digest": architecture.digest,
        "manifest": manifest_fingerprint(manifest),
        "data_state": {"offset": 8},
        "training": settings["training"],
        "seed": settings["seed"],
        "run": None,
        "nll_curve": [
            {"tokens": 2, "nll": 5.0},
            {"tokens": 4, "nll": 4.5},
            {"tokens": 8, "nll": 4.0},
        ],
        "final_nll": None,
        "training_seconds": 1.0,
    }
    save(
        store.candidate_path("000001") / "checkpoint",
        4,
        model.state_dict(),
        optimizer.state_dict(),
        metadata,
    )

    class Tokenizer:
        vocab_size = 16
        bos_id = 1
        eos_id = 2

    monkeypatch.setattr(
        search_script,
        "_context",
        lambda study, candidate: (
            store,
            settings,
            store.state(),
            {"data": {}, "tokenizer": {}},
            store.candidate_path(candidate),
            architecture,
        ),
    )
    monkeypatch.setattr(search_script, "get_tokenizer", lambda **config: Tokenizer())
    monkeypatch.setattr(search_script, "load_manifest", lambda path: manifest)
    result = search_script.train_candidate(
        store.directory,
        "000001",
        8,
        "cpu",
    )
    stored = next(value for value in store.results() if value["candidate_id"] == "000001")
    assert result == {"complete": True, "trained_tokens": 8}
    assert stored["status"] == "ready"
    assert stored["rung"] == 8
    assert stored["nll_curve"] == metadata["nll_curve"]


def test_rebuilt_checkpoint_installs_with_validated_boundary(tmp_path):
    source = tmp_path / "rebuild" / "checkpoint"
    destination = tmp_path / "candidate" / "checkpoint"
    source.mkdir(parents=True)
    step = 4
    digest = "architecture"
    (source / "model_000004.pt").write_bytes(b"model")
    (source / "optimizer_000004.pt").write_bytes(b"optimizer")
    atomic_json(
        source / "metadata_000004.json",
        {"trained_tokens": 8, "architecture_digest": digest},
    )
    (source / "complete_000004").write_text("complete\n")
    assert search_script._checkpoint_ready(source, step, digest, 8)
    search_script._install_checkpoint(source, destination, step)
    assert search_script._checkpoint_ready(destination, step, digest, 8)


def test_monitor_and_final_slices_are_fixed_and_disjoint():
    settings = tiny_settings()
    slices = validation_slices(settings)
    assert slices == {
        "monitor": {"offset": 0, "tokens": 2},
        "final": {"offset": 2, "tokens": 4},
    }
    state = loader_state("digest", 2, 1, 2)
    assert state["global_offset"] == slices["final"]["offset"]
    with pytest.raises(ValueError, match="align"):
        loader_state("digest", 1, 1, 2)


def test_search_genome_cached_decode_matches_full_sequence():
    torch.manual_seed(3)
    model = SpeckForCausalLM(rich_architecture())
    model.init_weights()
    model.eval()
    tokens = torch.randint(0, 16, (1, 8))
    assert torch.allclose(model(tokens), cached_logits(model, tokens), atol=1e-5)


def test_non_finite_and_oom_failures_are_classified():
    non_finite = subprocess.CalledProcessError(
        1,
        ["worker"],
        stderr="non-finite training loss",
    )
    oom = subprocess.CalledProcessError(
        1,
        ["worker"],
        stderr="cuda out of memory",
    )
    assert search_script._failure(non_finite)["type"] == "non_finite"
    assert search_script._failure(oom)["type"] == "oom"


def test_worker_failure_is_recorded_and_excluded(tmp_path, monkeypatch):
    settings = tiny_settings()
    store = open_study(tmp_path / "study", tmp_path, settings, generations=1)
    materialize_generation(
        store,
        (CandidatePlan(rich_architecture(), None, None),),
        0,
        settings,
    )
    error = subprocess.CalledProcessError(
        1,
        ["worker"],
        stderr="cuda out of memory",
    )
    monkeypatch.setattr(
        search_script,
        "run_child",
        lambda command: (_ for _ in ()).throw(error),
    )
    result = search_script._run_candidate(store, "000001", 8, "cpu")
    assert result["status"] == "failed"
    assert result["error"]["type"] == "oom"
    assert select_finalists(
        [
            {
                "candidate_id": "eligible",
                "status": "confirmed",
                "scores": {"quality": 0, "balanced": 0, "efficiency": 0},
            },
            result,
        ]
    ) == {
        "quality": "eligible",
        "balanced": "eligible",
        "efficiency": "eligible",
    }


def test_archive_rescoring_compares_generations(tmp_path):
    settings = tiny_settings()
    store = open_study(tmp_path / "study", tmp_path, settings, generations=2)
    first = rich_architecture()
    second = mutate_architecture(first, settings, 2, "change_embedding_width").architecture
    materialize_generation(
        store,
        (CandidatePlan(first, None, None),),
        0,
        settings,
    )
    materialize_generation(
        store,
        (CandidatePlan(second, None, None),),
        1,
        settings,
    )
    for candidate_id, offset in (("000001", 1.0), ("000002", 0.0)):
        result = next(value for value in store.results() if value["candidate_id"] == candidate_id)
        candidate_profile = dict(result["profile"])
        candidate_profile.update(profile(1.0))
        store.update_result(
            candidate_id,
            status="confirmed",
            trained_tokens=32,
            rung=32,
            nll_curve=[
                {"tokens": tokens, "nll": offset + 5 - math.log2(tokens) / 4}
                for tokens in (2, 4, 8, 16, 32)
            ],
            profile=candidate_profile,
        )
    search_script._rescore_archive(store, 16, 32)
    values = {result["candidate_id"]: result for result in store.results()}
    assert (
        values["000002"]["scores_by_rung"]["16"]["quality"]
        < values["000001"]["scores_by_rung"]["16"]["quality"]
    )


def write_tiny_experiment(path):
    path.mkdir()
    (path / "model.json").write_text(
        json.dumps(rich_architecture().settings()),
        encoding="utf-8",
    )
    (path / "search.json").write_text(
        json.dumps(tiny_settings().settings()),
        encoding="utf-8",
    )


def fake_study_inputs(path):
    model = json.loads((Path(path) / "model.json").read_text(encoding="utf-8"))
    return {"model": model}, {"fixture": "tiny"}


def fake_child(command):
    action = command[3]
    store = StudyStore(command[4])
    candidate_id = command[5]
    result = next(value for value in store.results() if value["candidate_id"] == candidate_id)
    if action == "_check":
        store.update_result(candidate_id, feasibility={"status": "passed"})
        return {"status": "passed"}
    if action == "_profile":
        candidate_profile = dict(result["profile"])
        value = int(candidate_id)
        candidate_profile.update(profile(value))
        store.update_result(candidate_id, profile=candidate_profile)
        return candidate_profile
    target = int(command[6])
    curve = list(result["nll_curve"])
    for tokens in (2, 4, 8, 16, 32):
        if tokens <= target and not any(point["tokens"] == tokens for point in curve):
            curve.append(
                {
                    "tokens": tokens,
                    "nll": 5 - int(candidate_id) / 100 - 0.1 * math.log2(tokens),
                }
            )
    store.update_result(
        candidate_id,
        status="ready",
        rung=target,
        trained_tokens=target,
        nll_curve=sorted(curve, key=lambda point: point["tokens"]),
    )
    checkpoint = store.candidate_path(candidate_id) / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    step = target // store.settings()["training"]["batch_tokens"]
    (checkpoint / f"model_{step:06d}.pt").write_bytes(b"model")
    (checkpoint / f"optimizer_{step:06d}.pt").write_bytes(b"optimizer")
    atomic_json(
        checkpoint / f"metadata_{step:06d}.json",
        {
            "trained_tokens": target,
            "architecture_digest": result["digest"],
        },
    )
    (checkpoint / f"complete_{step:06d}").write_text("complete\n")
    return {"complete": True, "trained_tokens": target}


def test_one_tiny_study_and_stable_status(tmp_path, monkeypatch):
    tiny_experiment = tmp_path / "experiment"
    write_tiny_experiment(tiny_experiment)
    studies = tmp_path / "studies"
    monkeypatch.setattr(search_script, "study_directory", lambda name: studies / name)
    monkeypatch.setattr(search_script, "_study_inputs", fake_study_inputs)
    monkeypatch.setattr(search_script, "_verify_inputs", lambda inputs: None)
    monkeypatch.setattr(search_script, "run_child", fake_child)
    monkeypatch.setattr(search_script, "_check_generation_space", lambda *args: None)
    snapshot = search_script.run_study(
        tiny_experiment,
        "tiny",
        hours=None,
        generations=1,
        device="cpu",
    )
    assert snapshot["status"] == "stopped"
    assert snapshot["counts"]["status"] == {"completed": 13, "confirmed": 3}
    store = StudyStore(studies / "tiny")
    assert status_snapshot(store) == status_snapshot(store)
    assert "generation 0" in search_script.human_status(snapshot)


def test_final_role_selection_and_two_seed_aggregation(tmp_path, monkeypatch):
    tiny_experiment = tmp_path / "experiment"
    write_tiny_experiment(tiny_experiment)
    studies = tmp_path / "studies"
    monkeypatch.setattr(search_script, "study_directory", lambda name: studies / name)
    monkeypatch.setattr(search_script, "_study_inputs", fake_study_inputs)
    monkeypatch.setattr(search_script, "_verify_inputs", lambda inputs: None)
    monkeypatch.setattr(search_script, "run_child", fake_child)
    monkeypatch.setattr(search_script, "_check_generation_space", lambda *args: None)
    search_script.run_study(
        tiny_experiment,
        "tiny",
        hours=None,
        generations=1,
        device="cpu",
    )
    store = StudyStore(studies / "tiny")
    confirmed = [result for result in store.results() if result["status"] == "confirmed"]
    lane_scores = (
        {"quality": 0.0, "balanced": 2.0, "efficiency": 2.0},
        {"quality": 2.0, "balanced": 0.0, "efficiency": 1.0},
        {"quality": 1.0, "balanced": 1.0, "efficiency": 0.0},
    )
    for result, scores in zip(confirmed, lane_scores):
        store.update_result(result["candidate_id"], scores=scores)
    roles = select_finalists(store.results())
    assert len(set(roles.values())) == 3

    def final_child(command):
        action = command[3]
        local_store = StudyStore(command[4])
        candidate_id = command[5]
        candidate = local_store.candidate_path(candidate_id)
        if action == "_final_train":
            run_name = command[6]
            target = int(command[7])
            value = int(candidate_id) / 10 + (0.01 if run_name == "independent" else 0)
            run = {
                "format_version": 1,
                "run": run_name,
                "seed": 7 if run_name == "continuation" else 8,
                "status": "completed",
                "trained_tokens": target,
                "nll_curve": [{"tokens": target, "nll": value}],
                "final_nll": value + 0.1,
            }
            atomic_json(candidate / "final" / run_name / "result.json", run)
            return run
        profile_path = candidate / "final" / "profile.json"
        if action == "_final_cpu_profile":
            final_profile = json.loads(profile_path.read_text(encoding="utf-8"))
            final_profile["cpu"] = {"contract": search_script._cpu_contract(local_store.settings())}
            atomic_json(profile_path, final_profile)
            return final_profile
        value = int(candidate_id)
        final_profile = {
            "format_version": 1,
            "eager_gpu": {
                "latency": {
                    "prefill_512": {"p50_seconds": value},
                    "prefill_2048": {"p50_seconds": value},
                    "decode_2048": {"p50_seconds": value},
                },
                "memory": {"peak_vram_bytes": value},
            },
            "compiled_gpu": {},
            "compilation_seconds": 1.0,
            "outputs_equivalent": True,
        }
        atomic_json(profile_path, final_profile)
        return final_profile

    monkeypatch.setattr(search_script, "run_child", final_child)
    report = search_script.finalize_study("tiny", "cpu")
    assert report["roles_before"] == roles
    assert set(report["candidates"]) == set(roles.values())
    assert all(
        set(candidate["verification"]["runs"]) == {"continuation", "independent"}
        for candidate in report["candidates"].values()
    )
    assert (studies / "tiny" / "finalists.json").is_file()
    run = {
        "nll_curve": [{"tokens": 64, "nll": 2.0}],
        "final_nll": 2.1,
    }
    aggregate = aggregate_final_runs(
        {"continuation": run, "independent": run},
        64,
    )
    assert aggregate["mean_monitor_nll"] == 2.0
    assert aggregate["mean_final_nll"] == 2.1
