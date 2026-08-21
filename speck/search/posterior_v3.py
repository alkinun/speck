"""calibrated shadow posterior and anchor selection for version three search."""

import math
from dataclasses import asdict

import numpy as np

from speck.search.architecture_v3 import (
    architecture_feature_names,
    architecture_features,
)
from speck.search.artifacts import ArtifactStore
from speck.search.calibration import calibration_report, frontier_calibration
from speck.search.planner import ActionProposal, plan_actions, posterior_information
from speck.search.posterior import CandidatePosterior, posterior_pareto
from speck.search.protocol import content_digest, derive_seed
from speck.search.surrogate import BootstrapRidgeSurrogate, cross_fitted_predictions


posterior_report_format_version = 1


def _objective_evidence(study, configs, objectives, training_tokens):
    targets = []
    evidence = []
    for config in configs:
        row = []
        architecture_evidence = {}
        observations = study.observations(config.digest, objectives.digest)
        for objective in objectives.selection:
            if objective.role == "quality":
                matching = tuple(
                    value
                    for value in observations
                    if value["objective_name"] == objective.name
                    and value["tokens"] == training_tokens
                    and value["source"] == "quality_evaluation"
                )
            else:
                matching = tuple(
                    value
                    for value in observations
                    if value["objective_name"] == objective.name
                    and value["source"] == "profile"
                )
            if not matching:
                raise ValueError(
                    f"posterior evidence is missing {objectives.name}:{objective.name} "
                    f"for {config.digest}"
                )
            values = tuple(float(value["value"]) for value in matching)
            row.append(float(np.mean(values)))
            architecture_evidence[objective.name] = {
                "observation_ids": tuple(value["id"] for value in matching),
                "values": values,
            }
        targets.append(row)
        evidence.append(
            {
                "architecture_digest": config.digest,
                "objectives": architecture_evidence,
            }
        )
    return np.asarray(targets, dtype=np.float64), tuple(evidence)


def _novelty(features):
    scale = features.std(axis=0)
    scale[scale == 0] = 1
    normalized = (features - features.mean(axis=0)) / scale
    values = []
    for index, row in enumerate(normalized):
        distances = tuple(
            float(np.linalg.norm(row - other) / math.sqrt(features.shape[1]))
            for other_index, other in enumerate(normalized)
            if other_index != index
        )
        values.append(min(distances) if distances else 0.0)
    return tuple(values)


def _sample_variance(values):
    return float(np.var(values, ddof=1)) if len(values) > 1 else 0.0


def _noise_components(study, configs, objectives, training_tokens):
    results = {}
    for objective in objectives.selection:
        if objective.role != "quality":
            results[objective.name] = {
                "data": 0.0,
                "initialization": 0.0,
                "numerical": 0.0,
                "total": 0.0,
            }
            continue
        architecture_components = []
        for config in configs:
            records = []
            for observation in study.observations(config.digest, objectives.digest):
                if (
                    observation["objective_name"] != objective.name
                    or observation["tokens"] != training_tokens
                    or observation["source"] != "quality_evaluation"
                    or observation["run_id"] is None
                ):
                    continue
                seeds = study.run(observation["run_id"])["seed_bundle"]
                records.append(
                    (
                        seeds.initialization_index,
                        seeds.data_index,
                        seeds.numerical_repeat,
                        float(observation["value"]),
                    )
                )
            if len(records) < 2:
                raise ValueError(
                    f"noise evidence is incomplete for {objectives.name}:"
                    f"{objective.name} on {config.digest}"
                )

            def grouped_mean(position):
                groups = {}
                for record in records:
                    groups.setdefault(record[position], []).append(record[3])
                return tuple(float(np.mean(values)) for values in groups.values())

            repeat_groups = {}
            for initialization, data, _, value in records:
                repeat_groups.setdefault((initialization, data), []).append(value)
            architecture_components.append(
                {
                    "data": _sample_variance(grouped_mean(1)),
                    "initialization": _sample_variance(grouped_mean(0)),
                    "numerical": float(
                        np.mean(
                            tuple(
                                _sample_variance(values)
                                for values in repeat_groups.values()
                            )
                        )
                    ),
                    "total": _sample_variance(tuple(record[3] for record in records)),
                }
            )
        results[objective.name] = {
            name: float(np.mean(tuple(value[name] for value in architecture_components)))
            for name in ("data", "initialization", "numerical", "total")
        }
    return results


def build_posterior_shadow(
    study,
    settings,
    configs,
    artifact_root,
    *,
    anchor_cost,
):
    configs = tuple(configs)
    if len(configs) < 3:
        raise ValueError("posterior calibration needs at least three architectures")
    if not math.isfinite(anchor_cost) or anchor_cost <= 0:
        raise ValueError("posterior anchor costs must be finite and positive")
    if settings.calibration.anchor_architectures > len(configs):
        raise ValueError("posterior anchor count exceeds available architectures")
    digests = tuple(config.digest for config in configs)
    if len(set(digests)) != len(digests):
        raise ValueError("posterior architectures must be unique")
    features = np.asarray(
        [architecture_features(config) for config in configs],
        dtype=np.float64,
    )
    groups = np.arange(len(configs))
    posterior_by_set = {}
    evidence_by_set = {}
    calibration_by_set = {}
    frontier_by_set = {}
    normalization_by_set = {}
    noise_by_set = {}
    surrogate_states = {}
    aggregate_probability = np.zeros(len(configs), dtype=np.float64)
    aggregate_information = np.zeros(len(configs), dtype=np.float64)
    aggregate_rank = np.zeros(len(configs), dtype=np.float64)
    for set_index, objectives in enumerate(settings.objective_sets):
        names = tuple(value.name for value in objectives.selection)
        raw_targets, evidence = _objective_evidence(
            study,
            configs,
            objectives,
            settings.calibration.broad_tokens,
        )
        target_mean = raw_targets.mean(axis=0)
        target_scale = raw_targets.std(axis=0)
        target_scale[target_scale == 0] = 1
        targets = (raw_targets - target_mean) / target_scale
        normalization_by_set[objectives.name] = {
            "mean": tuple(float(value) for value in target_mean),
            "objective_names": names,
            "scale": tuple(float(value) for value in target_scale),
        }
        noise = _noise_components(
            study,
            configs[: settings.calibration.noise_architectures],
            objectives,
            settings.calibration.noise_tokens,
        )
        noise_by_set[objectives.name] = noise
        seed = derive_seed(settings.seed, "surrogate", objectives.digest)
        folds = min(5, len(configs))
        cross_fitted = cross_fitted_predictions(
            features,
            targets,
            groups,
            names,
            folds=folds,
            models=settings.planner.surrogate_models,
            ridge=settings.planner.surrogate_ridge,
            seed=seed,
        )
        calibration_by_set[objectives.name] = {
            name: calibration_report(
                dict(zip(digests, cross_fitted[:, objective_index])),
                dict(zip(digests, targets[:, objective_index])),
                bootstrap_samples=settings.calibration.bootstrap_samples,
                seed=derive_seed(seed, "calibration", name),
            ).export()
            for objective_index, name in enumerate(names)
        }
        directions = {
            objective.name: objective.direction for objective in objectives.selection
        }
        predicted_points = {
            digest: {
                name: float(cross_fitted[row_index, objective_index])
                for objective_index, name in enumerate(names)
            }
            for row_index, digest in enumerate(digests)
        }
        observed_points = {
            digest: {
                name: float(targets[row_index, objective_index])
                for objective_index, name in enumerate(names)
            }
            for row_index, digest in enumerate(digests)
        }
        frontier_by_set[objectives.name] = asdict(
            frontier_calibration(predicted_points, observed_points, directions)
        )
        surrogate = BootstrapRidgeSurrogate(
            names,
            models=settings.planner.surrogate_models,
            ridge=settings.planner.surrogate_ridge,
            seed=seed,
        ).fit(features, targets, groups)
        candidates = surrogate.predict(features, digests)
        normalized_noise = np.asarray(
            [
                noise[name]["total"] / (target_scale[index] ** 2)
                for index, name in enumerate(names)
            ],
            dtype=np.float64,
        )
        candidates = tuple(
            CandidatePosterior(
                candidate.architecture_digest,
                candidate.objective_names,
                candidate.mean,
                tuple(
                    tuple(float(value) for value in row)
                    for row in (
                        np.asarray(candidate.covariance)
                        + np.diag(normalized_noise)
                    )
                ),
            )
            for candidate in candidates
        )
        metrics = posterior_pareto(
            candidates,
            objectives,
            samples=settings.planner.posterior_samples,
            seed=derive_seed(settings.seed, "pareto", objectives.digest),
        )
        metric_by_digest = {value.architecture_digest: value for value in metrics}
        for index, candidate in enumerate(candidates):
            metric = metric_by_digest[candidate.architecture_digest]
            aggregate_probability[index] += metric.nondominated_probability
            aggregate_rank[index] += metric.expected_rank
            aggregate_information[index] += posterior_information(
                candidate.covariance,
                1.0,
            )
        posterior_by_set[objectives.name] = {
            "candidates": tuple(asdict(value) for value in candidates),
            "metrics": tuple(asdict(value) for value in metrics),
        }
        evidence_by_set[objectives.name] = evidence
        surrogate_states[objectives.name] = surrogate.state()
    divisor = len(settings.objective_sets)
    aggregate_probability /= divisor
    aggregate_rank /= divisor
    aggregate_information /= divisor
    novelty = _novelty(features)
    proposals = tuple(
        ActionProposal(
            "anchor",
            digest,
            anchor_cost,
            float(aggregate_probability[index]),
            float(aggregate_information[index]),
            novelty[index],
            {
                "expected_rank": float(aggregate_rank[index]),
                "training_tokens": settings.calibration.anchor_tokens,
            },
        )
        for index, digest in enumerate(digests)
    )
    decision = plan_actions(
        proposals,
        available_cost=anchor_cost * settings.calibration.anchor_architectures,
        max_actions=settings.calibration.anchor_architectures,
        seed=derive_seed(settings.seed, "anchor_selection"),
    )
    anchors = tuple(
        value.proposal.architecture_digest for value in decision.selected
    )
    if len(anchors) != settings.calibration.anchor_architectures:
        raise RuntimeError("posterior planner did not fill the anchor panel")
    evidence_definition = {
        "architecture_digests": digests,
        "broad_tokens": settings.calibration.broad_tokens,
        "evidence": evidence_by_set,
        "feature_names": architecture_feature_names,
        "features": tuple(tuple(float(value) for value in row) for row in features),
        "objective_sets": tuple(asdict(value) for value in settings.objective_sets),
    }
    evidence_digest = content_digest(evidence_definition)
    report = {
        "anchors": anchors,
        "calibration": calibration_by_set,
        "decision": decision.export(),
        "evidence": evidence_definition,
        "evidence_digest": evidence_digest,
        "format_version": posterior_report_format_version,
        "frontier_calibration": frontier_by_set,
        "normalization": normalization_by_set,
        "noise": noise_by_set,
        "posterior": posterior_by_set,
        "surrogates": surrogate_states,
    }
    report_digest = content_digest(report)
    artifact = ArtifactStore(artifact_root).put_json("posterior_shadow", report)
    definition = {
        "anchors": anchors,
        "decision_digest": decision.digest,
        "format_version": posterior_report_format_version,
        "report_digest": report_digest,
    }
    created = study.register_posterior_report(
        report_digest,
        evidence_digest,
        artifact,
        definition,
        anchors,
    )
    return {
        "anchors": anchors,
        "artifact_digest": artifact.digest,
        "created": created,
        "evidence_digest": evidence_digest,
        "report_digest": report_digest,
    }
