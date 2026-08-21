"""deterministic multi-fidelity evolution and promotion policy."""

import math
import random
from dataclasses import asdict

from speck.model import Config
from speck.search.architecture import (
    available_mutations,
    crossover,
    kv_bytes_per_token,
    mutate,
    mutation_operators,
    parameter_count,
    repair,
)
from speck.search.evaluate import quantized_weight_bytes
from speck.search.evolution import (
    EvaluatedArchitecture,
    ObjectiveEstimate,
    OperatorOutcome,
    aggregate_trials,
    estimated_candidates,
    nondominated_sort,
    operator_probabilities,
    select_parent,
    select_survivors,
    selection_metrics,
)
from speck.search.spec import deterministic_seed, objective_names


class ArchitectureSpaceExhausted(RuntimeError):
    pass


def trial_seeds(settings, rung):
    return tuple(
        deterministic_seed(settings.seed, "trial", rung, seed_index)
        for seed_index in range(settings.rungs[rung].seed_count)
    )


def static_metrics(config, settings):
    quantization = quantized_weight_bytes(config, settings.quantization)
    return {
        "parameters": parameter_count(config),
        "memory.kv_cache_bytes_per_token": kv_bytes_per_token(
            config, settings.inference.cache_dtype_bytes
        ),
        "memory.quantized_weight_bytes": quantization["total_bytes"],
    }


def _evaluated(study, rung):
    values = []
    for item in study.rungs(rung=rung):
        if item["aggregate"] is None:
            continue
        architecture = study.architecture(item["architecture_id"])
        estimates = {
            name: ObjectiveEstimate(**estimate)
            for name, estimate in item["aggregate"]["objectives"].items()
        }
        values.append(
            EvaluatedArchitecture(
                architecture["id"],
                architecture["architecture_hash"],
                Config.from_dict(architecture["config"]),
                estimates,
            )
        )
    return tuple(values)


def _aggregate_ready(study, settings):
    changed = False
    names = objective_names(settings)
    for item in study.rungs(status="active"):
        trials = study.trials(
            architecture_id=item["architecture_id"], rung=item["rung"]
        )
        if any(trial["status"] in {"pending", "running"} for trial in trials):
            continue
        if any(trial["status"] == "failed" for trial in trials):
            study.update_rung(
                item["architecture_id"],
                item["rung"],
                "failed",
                decision={
                    "reason": "trial_failed",
                    "trials": [
                        trial["id"]
                        for trial in trials
                        if trial["status"] == "failed"
                    ],
                },
            )
            changed = True
            continue
        architecture = study.architecture(item["architecture_id"])
        static = {
            name: value
            for name, value in architecture["static"].items()
            if name in names
        }
        try:
            estimates = aggregate_trials(
                tuple(trial["result"] for trial in trials),
                static,
                names,
                settings.confidence_z,
            )
        except (KeyError, TypeError, ValueError) as error:
            study.update_rung(
                item["architecture_id"],
                item["rung"],
                "failed",
                decision={
                    "reason": "invalid_trial_result",
                    "error": str(error),
                },
            )
            changed = True
            continue
        aggregate = {
            "trials": [trial["id"] for trial in trials],
            "objectives": {
                name: asdict(estimate) for name, estimate in estimates.items()
            },
        }
        study.update_rung(
            item["architecture_id"], item["rung"], "complete", aggregate
        )
        changed = True
    return changed


def _refresh_rung(study, settings, rung):
    candidates = _evaluated(study, rung)
    if not candidates:
        return {}, ()
    point_candidates = estimated_candidates(candidates)
    selected, metrics, frontier = select_survivors(
        point_candidates,
        min(settings.population_size, len(point_candidates)),
        objective_names(settings),
        settings.space,
    )
    by_id = {item["architecture_id"]: item for item in study.rungs(rung=rung)}
    for candidate in candidates:
        current = by_id[candidate.id]
        values = metrics[candidate.id]
        study.update_rung(
            candidate.id,
            rung,
            current["status"],
            rank=values.rank,
            crowding=values.crowding,
            novelty=values.novelty,
        )
    return metrics, selected


def _weighted_choice(probabilities, seed):
    rng = random.Random(seed)
    point = rng.random()
    cumulative = 0.0
    for name in sorted(probabilities):
        cumulative += probabilities[name]
        if point <= cumulative:
            return name
    return sorted(probabilities)[-1]


def _operator_probabilities(study, settings, available):
    outcomes = tuple(
        OperatorOutcome(item["operator"], bool(item["success"]))
        for item in study.outcomes()
    )
    return operator_probabilities(
        outcomes,
        available,
        settings.operator_prior_success,
        settings.operator_prior_failure,
        settings.operator_probability_floor,
    )


def _parent_population(study, settings):
    fallback = ()
    for rung in reversed(range(len(settings.rungs))):
        candidates = _evaluated(study, rung)
        if len(candidates) > len(fallback):
            fallback = candidates
        required = min(
            settings.population_size,
            settings.rungs[rung].architecture_limit,
        )
        if len(candidates) >= required:
            break
    else:
        candidates = fallback
    if not candidates:
        return (), {}
    point = estimated_candidates(candidates)
    selected, metrics, _ = select_survivors(
        point,
        min(settings.population_size, len(point)),
        objective_names(settings),
        settings.space,
    )
    by_id = {candidate.id: candidate for candidate in point}
    return tuple(by_id[candidate_id] for candidate_id in selected), metrics


def _add_architecture(study, settings, baseline, ordinal):
    cohort, slot = divmod(ordinal, settings.cohort_size)
    for attempt in range(settings.max_generation_attempts):
        seed = deterministic_seed(settings.seed, "architecture", ordinal, attempt)
        if ordinal == 0:
            config, repairs = repair(baseline, settings.space)
            operation = {"operator": "seed", "seed": seed}
            parents = ()
        elif ordinal < settings.initial_population:
            baseline_row = study.architectures()[0]
            repaired_baseline = Config.from_dict(baseline_row["config"])
            operator = mutation_operators[(ordinal + attempt - 1) % len(mutation_operators)]
            try:
                result = mutate(repaired_baseline, settings.space, seed, operator)
            except ValueError:
                continue
            config, operation, repairs = result.config, result.mutation, result.repairs
            parents = (("primary", baseline_row["id"]),)
        else:
            population, metrics = _parent_population(study, settings)
            if not population:
                return None
            primary = select_parent(population, metrics, seed)
            rng = random.Random(seed)
            use_crossover = len(population) > 1 and rng.random() < settings.crossover_probability
            if use_crossover:
                secondary = rng.choice(
                    [candidate for candidate in population if candidate.id != primary.id]
                )
                try:
                    result = crossover(
                        primary.config, secondary.config, settings.space, seed
                    )
                except ValueError:
                    continue
                parents = (("primary", primary.id), ("secondary", secondary.id))
            else:
                available = available_mutations(primary.config, settings.space)
                if not available:
                    continue
                probabilities = _operator_probabilities(study, settings, available)
                operator = _weighted_choice(probabilities, seed)
                try:
                    result = mutate(primary.config, settings.space, seed, operator)
                except ValueError:
                    continue
                parents = (("primary", primary.id),)
            config, operation, repairs = result.config, result.mutation, result.repairs
        architecture_id = study.add_architecture_with_rung(
            config,
            static_metrics(config, settings),
            cohort,
            slot,
            seed,
            operation,
            repairs,
            parents,
            0,
            trial_seeds(settings, 0),
        )
        if architecture_id is None:
            continue
        return architecture_id
    raise ArchitectureSpaceExhausted("architecture space is exhausted")


def _ensure_architectures(study, settings, baseline):
    architectures = study.architectures()
    count = len(architectures)
    if count < settings.initial_population:
        target = settings.initial_population
    elif count < settings.max_architectures and count % settings.cohort_size:
        target = min(
            settings.max_architectures,
            count + settings.cohort_size - count % settings.cohort_size,
        )
    else:
        if study.trials(status="pending") or study.trials(status="running"):
            return False
        if study.rungs(rung=0, status="active"):
            return False
        remainder = count % settings.cohort_size
        increment = settings.cohort_size - remainder if remainder else settings.cohort_size
        target = min(settings.max_architectures, count + increment)
    changed = False
    while len(study.architectures()) < target:
        ordinal = len(study.architectures())
        try:
            architecture_id = _add_architecture(
                study, settings, baseline, ordinal
            )
        except ArchitectureSpaceExhausted as error:
            changed |= study.record_event(
                "architecture_space_exhausted",
                "architecture_space_exhausted",
                {"ordinal": ordinal, "error": str(error)},
            )
            changed |= study.finalize("failed")
            break
        if architecture_id is None:
            break
        changed = True
    return changed


def _rung_closed(study, settings, rung):
    rows = study.rungs(rung=rung)
    if any(item["status"] == "active" for item in rows):
        return False
    if rung == 0:
        return len(rows) >= settings.rungs[0].architecture_limit
    if not _rung_closed(study, settings, rung - 1):
        return False
    expected = min(
        settings.rungs[rung].architecture_limit,
        len(_evaluated(study, rung - 1)),
    )
    return len(rows) >= expected


def _promotion_order(candidates, settings, event):
    names = objective_names(settings)
    lane = event % 4
    bound = "upper" if lane == 0 else "lower" if lane == 1 else "mean"
    point = estimated_candidates(candidates, bound)
    metrics, _ = selection_metrics(point, names, settings.space)
    order = sorted(
        candidates,
        key=lambda candidate: (
            metrics[candidate.id].rank,
            -metrics[candidate.id].novelty
            if lane == 3
            else -metrics[candidate.id].crowding,
            candidate.architecture_hash,
        ),
    )
    by_id = {candidate.id: candidate for candidate in point}
    return order, metrics, bound, by_id


def _promote(study, settings):
    changed = False
    for rung in range(len(settings.rungs) - 1):
        source_rows = study.rungs(rung=rung)
        if any(item["status"] == "active" for item in source_rows):
            continue
        current = _evaluated(study, rung)
        next_settings = settings.rungs[rung + 1]
        source_closed = _rung_closed(study, settings, rung)
        available = (
            next_settings.architecture_limit
            if source_closed
            else len(source_rows)
            * next_settings.architecture_limit
            // settings.rungs[rung].architecture_limit
        )
        unlocked = min(
            next_settings.architecture_limit, available, len(current)
        )
        existing = {
            item["architecture_id"] for item in study.rungs(rung=rung + 1)
        }
        needed = unlocked - len(existing)
        while needed > 0:
            event = len(existing)
            order, metrics, bound, point_by_id = _promotion_order(
                current, settings, event
            )
            chosen = next(
                candidate for candidate in order if candidate.id not in existing
            )
            lane = ("conservative", "optimistic", "mean", "diversity")[event % 4]
            values = metrics[chosen.id]
            crowding = (
                values.crowding if math.isfinite(values.crowding) else None
            )
            decision = {
                "reason": "confidence_lane",
                "event": event,
                "lane": lane,
                "bound": bound,
                "source_rung": rung,
                "architecture_hash": chosen.architecture_hash,
                "eligible_architecture_ids": sorted(
                    candidate.id for candidate in current
                ),
                "objectives": point_by_id[chosen.id].objectives,
                "metrics": {
                    "rank": values.rank,
                    "crowding": crowding,
                    "novelty": values.novelty,
                },
            }
            promoted = study.promote(
                chosen.id,
                rung,
                rung + 1,
                trial_seeds(settings, rung + 1),
                decision,
            )
            if not promoted:
                raise RuntimeError("could not record an atomic promotion")
            existing.add(chosen.id)
            needed -= 1
            changed = True
        if source_closed:
            for item in study.rungs(rung=rung, status="complete"):
                study.update_rung(
                    item["architecture_id"],
                    rung,
                    "stopped",
                    decision={"reason": "successive_halving"},
                    rank=item["pareto_rank"],
                    crowding=item["crowding"],
                    novelty=item["novelty"],
                )
                changed = True
    return changed


def _credit_outcomes(study, settings):
    if study.rungs(rung=0, status="active"):
        return False
    credited = {item["architecture_id"] for item in study.outcomes()}
    candidates = _evaluated(study, 0)
    if not candidates:
        return False
    point = estimated_candidates(candidates)
    selected, _, frontier = select_survivors(
        point,
        min(settings.population_size, len(point)),
        objective_names(settings),
        settings.space,
    )
    successful = set(selected) | set(frontier)
    changed = False
    for item in study.rungs(rung=0):
        architecture_id = item["architecture_id"]
        if architecture_id in credited:
            continue
        architecture = study.architecture(architecture_id)
        if architecture["operator"] not in mutation_operators:
            continue
        changed |= study.record_outcome(
            architecture_id,
            architecture["operator"],
            architecture_id in successful,
        )
    return changed


def _recommendations(study, settings):
    final_rung = len(settings.rungs) - 1
    candidates = _evaluated(study, final_rung)
    if not candidates:
        return None
    points = estimated_candidates(candidates)
    names = objective_names(settings)
    frontier = nondominated_sort(points, names)[0]
    minima = {name: min(candidate.objectives[name] for candidate in points) for name in names}
    maxima = {name: max(candidate.objectives[name] for candidate in points) for name in names}

    def regret(candidate, selected_names):
        values = [
            (candidate.objectives[name] - minima[name]) / (maxima[name] - minima[name])
            for name in selected_names
            if maxima[name] > minima[name]
        ]
        return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0

    quality_names = tuple(name for name in names if name.startswith("quality."))
    efficiency_names = tuple(name for name in names if name not in quality_names)
    quality = min(
        frontier,
        key=lambda item: (regret(item, quality_names), regret(item, names), item.id),
    )
    efficiency = min(
        frontier,
        key=lambda item: (regret(item, efficiency_names), regret(item, names), item.id),
    )
    balanced = min(frontier, key=lambda item: (regret(item, names), item.id))

    def summary(candidate):
        architecture = study.architecture(candidate.id)
        return {
            "id": candidate.id,
            "architecture_hash": architecture["architecture_hash"],
            "objectives": candidate.objectives,
            "config": architecture["config"],
        }

    return {
        "rung": final_rung,
        "quality": summary(quality),
        "efficiency": summary(efficiency),
        "balanced": summary(balanced),
        "frontier": [summary(candidate) for candidate in frontier],
    }


def rung_frontier(study, settings, rung=None):
    if rung is None:
        available = [
            index
            for index in range(len(settings.rungs))
            if _evaluated(study, index)
        ]
        if not available:
            return {
                "rung": None,
                "closed": False,
                "scheduled": 0,
                "completed": 0,
                "architectures": [],
            }
        closed = [
            index for index in available if _rung_closed(study, settings, index)
        ]
        rung = closed[-1] if closed else available[0]
    if not 0 <= rung < len(settings.rungs):
        raise ValueError("rung is outside the search configuration")
    candidates = _evaluated(study, rung)
    details = {
        "rung": rung,
        "name": settings.rungs[rung].name,
        "closed": _rung_closed(study, settings, rung),
        "scheduled": len(study.rungs(rung=rung)),
        "completed": len(candidates),
    }
    if not candidates:
        return {**details, "architectures": []}
    points = estimated_candidates(candidates)
    front = nondominated_sort(points, objective_names(settings))[0]
    values = []
    for candidate in front:
        architecture = study.architecture(candidate.id)
        aggregate = study.rung(candidate.id, rung)["aggregate"]
        values.append({
            "id": candidate.id,
            "architecture_hash": architecture["architecture_hash"],
            "config": architecture["config"],
            "objectives": candidate.objectives,
            "estimates": aggregate["objectives"],
            "operation": architecture["operation"],
            "parents": architecture["parents"],
        })
    return {**details, "architectures": values}


def advance(study, baseline, settings):
    if study.study()["status"] in {"completed", "failed"}:
        return False
    changed = _aggregate_ready(study, settings)
    for rung in range(len(settings.rungs)):
        if _evaluated(study, rung):
            _refresh_rung(study, settings, rung)
    changed |= _credit_outcomes(study, settings)
    changed |= _promote(study, settings)
    changed |= _ensure_architectures(study, settings, baseline)
    final_rung = len(settings.rungs) - 1
    if _rung_closed(study, settings, final_rung):
        recommendations = _recommendations(study, settings)
        status = "completed" if recommendations is not None else "failed"
        changed |= study.finalize(status, recommendations)
    elif (
        not study.trials(status="pending")
        and not study.trials(status="running")
        and len(study.architectures()) < settings.max_architectures
        and not _evaluated(study, 0)
    ):
        changed |= study.finalize("failed")
    return changed
