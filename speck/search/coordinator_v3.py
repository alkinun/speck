"""deterministic calibration bootstrap coordination for version three search."""

import math

from speck.profile.schema import ProfileScenario
from speck.search.architecture_v3 import (
    parameter_count,
    quantized_weight_bytes,
    sample_architecture,
)
from speck.search.artifacts import ArtifactStore
from speck.search.profile_worker import backend_plugin
from speck.search.posterior_v3 import build_posterior_shadow
from speck.search.protocol import SeedBundle, TrainingProtocol, derive_seed
from speck.search.segments import load_segment_plan


def _static_metrics(config):
    return {
        "logical_depth": config.logical_depth,
        "parameters": parameter_count(config),
        "q4_weight_bytes": quantized_weight_bytes(config),
        "unique_parameter_blocks": config.unique_parameter_blocks,
    }


def _ensure_broad_panel(study, settings, baseline):
    configs = [baseline]
    known = {baseline.digest}
    attempt = 0
    maximum_attempts = max(100, settings.calibration.broad_architectures * 100)
    while len(configs) < settings.calibration.broad_architectures:
        if attempt >= maximum_attempts:
            raise ValueError("search space could not produce the broad calibration panel")
        seed = derive_seed(settings.seed, "broad_architecture", attempt)
        candidate = sample_architecture(baseline, settings.space, seed)
        attempt += 1
        if candidate.digest in known:
            continue
        slot = len(configs)
        study.add_architecture(
            candidate,
            _static_metrics(candidate),
            {
                "attempt": attempt - 1,
                "operator": "broad_sample",
                "seed": seed,
                "slot": slot,
            },
        )
        known.add(candidate.digest)
        configs.append(candidate)
    return tuple(configs)


def _panel_runs(study, settings, configs, protocol):
    values = []
    canonical = SeedBundle.create_panel(settings.seed, 0, 0, 0)
    for slot, config in enumerate(configs):
        run_id = study.add_run(config.digest, protocol, canonical)
        values.append((slot, run_id, settings.calibration.broad_tokens, canonical))
        if slot >= settings.calibration.noise_architectures:
            continue
        for initialization_index in range(settings.calibration.initialization_seeds):
            for data_index in range(settings.calibration.data_seeds):
                for repeat in range(settings.calibration.numerical_repeats):
                    seeds = SeedBundle.create_panel(
                        settings.seed,
                        initialization_index,
                        data_index,
                        repeat,
                    )
                    if seeds.digest == canonical.digest:
                        continue
                    run_id = study.add_run(config.digest, protocol, seeds)
                    values.append(
                        (slot, run_id, settings.calibration.noise_tokens, seeds)
                    )
    return tuple(values)


def _profile_scenarios(settings):
    objective_names = {value.name for value in settings.objective_sets}
    scenarios = []
    for template in settings.profiles:
        if template.name not in objective_names:
            continue
        plugin = backend_plugin(template.backend)
        scenarios.append(
            ProfileScenario(
                name=template.name,
                backend=plugin.identity,
                device=template.device,
                dtype=template.dtype,
                cache_dtype=template.cache_dtype,
                batch_size=template.batch_size,
                prompt_tokens=template.prompt_tokens,
                generated_tokens=template.generated_tokens,
                warmup_requests=template.warmup_requests,
                measured_requests=template.measured_requests,
                process_repetitions=template.process_repetitions,
            )
        )
    return tuple(scenarios)


def _apply_checkpoint_retention(
    study,
    settings,
    configs,
    panel_runs,
    runs,
    active_runs,
    anchors,
    artifact_root,
):
    if artifact_root is None:
        return ()
    store = ArtifactStore(artifact_root)
    pruned = []
    for slot, run_id, target_tokens, seeds in panel_runs:
        if run_id in active_runs:
            continue
        run = runs[run_id]
        checkpoints = study.checkpoints(run_id)
        for checkpoint in checkpoints:
            digest = checkpoint.artifact.digest
            if (
                digest != run["checkpoint_digest"]
                and not study.checkpoint_payload_pruned(digest)
                and study.quality_evaluation(run_id, digest) is not None
            ):
                study.prune_checkpoint_payload(
                    run_id,
                    digest,
                    store,
                    "superseded",
                )
                pruned.append(digest)
        current = run["checkpoint_digest"]
        if (
            current is None
            or study.checkpoint_payload_pruned(current)
            or study.quality_evaluation(run_id, current) is None
        ):
            continue
        canonical = (
            seeds.initialization_index == 0
            and seeds.data_index == 0
            and seeds.numerical_repeat == 0
        )
        architecture_digest = configs[slot].digest
        reason = None
        if not canonical and run["tokens"] >= target_tokens:
            reason = "noise_trajectory_complete"
        elif anchors and architecture_digest not in anchors and run["tokens"] >= target_tokens:
            reason = "unselected_broad_complete"
        elif (
            architecture_digest in anchors
            and run["tokens"] >= settings.calibration.anchor_tokens
        ):
            reason = "anchor_trajectory_complete"
        if reason is not None:
            study.prune_checkpoint_payload(
                run_id,
                current,
                store,
                reason,
                archive_run=True,
            )
            pruned.append(current)
    return tuple(pruned)


def coordinate_bootstrap(
    study,
    settings,
    *,
    quality_tokens_per_cost,
    evaluation_tokens_per_cost,
    profile_cost,
    artifact_root=None,
):
    costs = (quality_tokens_per_cost, evaluation_tokens_per_cost, profile_cost)
    if any(not math.isfinite(value) or value <= 0 for value in costs):
        raise ValueError("coordinator rates and costs must be finite and positive")
    stored = study.study()
    baseline = study.architecture(stored["provenance"]["model_digest"])["config"]
    protocol = TrainingProtocol.from_dict(
        stored["provenance"]["resolved_protocol"]
    )
    plan = load_segment_plan(stored["provenance"]["segment_plan"]["path"])
    partition = next(
        (
            value
            for value in plan.partitions
            if value.name == protocol.evaluation_partition
        ),
        None,
    )
    if partition is None:
        raise ValueError("coordinator evaluation partition is missing")
    evaluation_tokens = partition.tokens - 1
    evaluation_action_cost = evaluation_tokens / evaluation_tokens_per_cost
    configs = _ensure_broad_panel(study, settings, baseline)
    panel_runs = _panel_runs(study, settings, configs, protocol)
    runs = {run["id"]: run for run in study.runs()}
    posterior = study.posterior_report()
    anchors = set(posterior["anchors"]) if posterior is not None else set()
    actions = study.actions()
    active = tuple(
        action for action in actions if action["status"] in {"pending", "running"}
    )
    active_runs = {
        action["run_id"] for action in active if action["run_id"] is not None
    }
    pruned = _apply_checkpoint_retention(
        study,
        settings,
        configs,
        panel_runs,
        runs,
        active_runs,
        anchors,
        artifact_root,
    )
    if pruned:
        runs = {run["id"]: run for run in study.runs()}
    existing_profiles = {
        (
            action["architecture_digest"],
            action["profile_scenario_digest"],
            action["profile_repetition"],
        )
        for action in actions
        if action["kind"] == "profile"
        and action["status"] in {"pending", "running", "completed"}
    }
    objective_sets = {
        value.name: value for value in settings.objective_sets
    }
    quality_objectives = tuple(
        value.digest
        for value in settings.objective_sets
        if any(
            objective.name == "quality.target_nll"
            for objective in value.objectives
        )
    )
    candidates = []
    for slot, run_id, target_tokens, seeds in panel_runs:
        run = runs[run_id]
        canonical = (
            seeds.initialization_index == 0
            and seeds.data_index == 0
            and seeds.numerical_repeat == 0
        )
        if canonical and configs[slot].digest in anchors:
            target_tokens = settings.calibration.anchor_tokens
        if run_id in active_runs:
            continue
        if run["checkpoint_digest"] is not None and study.quality_evaluation(
            run_id, run["checkpoint_digest"]
        ) is None:
            candidates.append(
                {
                    "cost": evaluation_action_cost,
                    "key": (0, slot, run_id),
                    "kind": "evaluate",
                    "run_id": run_id,
                }
            )
        elif run["tokens"] < target_tokens:
            next_tokens = next(
                tokens
                for tokens in protocol.checkpoint_tokens
                if run["tokens"] < tokens <= target_tokens
            )
            candidates.append(
                {
                    "cost": (next_tokens - run["tokens"])
                    / quality_tokens_per_cost,
                    "key": (1, slot, run_id),
                    "kind": "continue",
                    "run_id": run_id,
                }
            )
    for slot, config in enumerate(configs):
        for scenario in _profile_scenarios(settings):
            objectives = objective_sets[scenario.name]
            for repetition in range(scenario.process_repetitions):
                identity = (config.digest, scenario.digest, repetition)
                if identity in existing_profiles:
                    continue
                candidates.append(
                    {
                        "architecture_digest": config.digest,
                        "cost": profile_cost,
                        "key": (2, slot, scenario.name, repetition),
                        "kind": "profile",
                        "objective_set_digest": objectives.digest,
                        "repetition": repetition,
                        "scenario": scenario,
                    }
                )
    slots = max(0, settings.planner.max_actions_per_event - len(active))
    committed_cost = sum(action["estimated_cost"] for action in actions)
    available_cost = max(0.0, settings.planner.total_cost - committed_cost)
    scheduled = []
    for candidate in sorted(candidates, key=lambda value: value["key"]):
        if len(scheduled) >= slots:
            break
        if candidate["cost"] > available_cost:
            continue
        if candidate["kind"] == "continue":
            action_id = study.add_quality_action(
                candidate["run_id"],
                2.0,
                candidate["cost"],
            )
        elif candidate["kind"] == "evaluate":
            action_id = study.add_evaluation_action(
                candidate["run_id"],
                quality_objectives,
                evaluation_tokens,
                3.0,
                candidate["cost"],
            )
        else:
            scenario = candidate["scenario"]
            prompt_seed = derive_seed(
                settings.seed,
                "profile",
                candidate["architecture_digest"],
                scenario.digest,
                candidate["repetition"],
            )
            action_id = study.add_profile_action(
                candidate["architecture_digest"],
                scenario,
                candidate["objective_set_digest"],
                candidate["repetition"],
                prompt_seed,
                1.0,
                candidate["cost"],
            )
        scheduled.append(action_id)
        available_cost -= candidate["cost"]
    panel_complete = all(
        runs[run_id]["tokens"] >= target
        and runs[run_id]["checkpoint_digest"] is not None
        and study.quality_evaluation(run_id, runs[run_id]["checkpoint_digest"])
        is not None
        for _, run_id, target, _ in panel_runs
    )
    profiles_complete = not any(
        candidate["kind"] == "profile" for candidate in candidates
    ) and not any(
        action["kind"] == "profile" for action in active
    )
    shadow = None
    if panel_complete and profiles_complete and posterior is None:
        if artifact_root is None:
            phase = "awaiting_anchor_posterior"
        else:
            remaining_checkpoints = sum(
                tokens > settings.calibration.broad_tokens
                for tokens in protocol.checkpoint_tokens
            )
            shadow = build_posterior_shadow(
                study,
                settings,
                configs,
                artifact_root,
                anchor_cost=remaining_checkpoints
                * evaluation_action_cost
                + (
                    settings.calibration.anchor_tokens
                    - settings.calibration.broad_tokens
                )
                / quality_tokens_per_cost,
            )
            posterior = study.posterior_report(shadow["evidence_digest"])
            anchors = set(posterior["anchors"])
            phase = "anchors"
    elif not panel_complete or not profiles_complete:
        phase = "bootstrap"
    else:
        anchor_runs = tuple(
            (run_id, runs[run_id])
            for slot, run_id, _, seeds in panel_runs
            if configs[slot].digest in anchors
            and seeds.initialization_index == 0
            and seeds.data_index == 0
            and seeds.numerical_repeat == 0
        )
        anchors_complete = all(
            run["tokens"] >= settings.calibration.anchor_tokens
            and run["checkpoint_digest"] is not None
            and study.quality_evaluation(run_id, run["checkpoint_digest"])
            is not None
            for run_id, run in anchor_runs
        )
        phase = "anchor_complete" if anchors_complete else "anchors"
    result = {
        "active_actions": len(active),
        "anchors": tuple(sorted(anchors)),
        "architectures": len(configs),
        "available_cost": available_cost,
        "phase": phase,
        "posterior_report": posterior["digest"] if posterior is not None else None,
        "pruned_checkpoint_payloads": pruned,
        "runs": len(panel_runs),
        "scheduled_actions": scheduled,
        "shadow": shadow,
    }
    study.record_event("coordinator_tick", result)
    return result
