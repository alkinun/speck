"""Compile a reviewable first-wave recipe proposal into comparisons and data demands."""

import json
from fractions import Fraction
from pathlib import Path

from speck.provenance.io import file_sha256


def _apportion(total, weights):
    denominator = sum(weights.values())
    exact = {key: Fraction(total * weight, denominator) for key, weight in weights.items()}
    counts = {key: value.numerator // value.denominator for key, value in exact.items()}
    remainder = total - sum(counts.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - counts[key]), key))
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def compile_first_wave(proposal, data_plan, registry, source_use):
    if (
        proposal.get("format") != "speck_first_wave_recipe_proposal"
        or proposal.get("format_version") != 1
        or proposal.get("status") != "proposed_for_review_not_frozen"
        or proposal.get("training_authority") is not False
        or proposal.get("comparison")
        != "replace only the tested category inside the fixed complete prior mixture"
    ):
        raise ValueError("first-wave recipe must be an explicitly unlaunched proposal")
    weights = {item["id"]: item["prior_percent"] for item in data_plan["categories"]}
    if weights != {
        "web": 55,
        "code": 15,
        "math": 10,
        "synthetic": 10,
        "science": 5,
        "reference": 5,
    }:
        raise ValueError("first wave must preserve the documented category prior")
    if (
        source_use.get("status") != "all_sources_human_approved"
        or source_use.get("automated_approval_made") is not False
    ):
        raise ValueError("first wave requires the existing source-use record")
    approved = set(source_use["approved_source_ids"])
    sources = {item["id"]: item for item in registry["sources"]}

    def validate_recipe(recipe):
        if set(recipe) != set(weights):
            raise ValueError("recipe must cover the six fixed categories")
        for category, mixture in recipe.items():
            if not isinstance(mixture, dict) or not mixture:
                raise ValueError("empty source treatment")
            for source, weight in mixture.items():
                if (
                    source not in approved
                    or source not in sources
                    or sources[source]["category"] != category
                    or sources[source].get("priority")
                    in ("tokenizer_only", "hold_terms", "reference_only")
                    or isinstance(weight, bool)
                    or not isinstance(weight, int)
                    or weight <= 0
                ):
                    raise ValueError(f"invalid/unapproved treatment source or weight: {source}")

    incumbent = proposal["incumbent"]
    validate_recipe(incumbent)
    expected_arms = {"web": 4, "code": 3, "math": 3, "synthetic": 3}
    if set(proposal["treatments"]) != set(expected_arms):
        raise ValueError("first-wave treatment categories differ from the data plan")
    if (
        proposal["screen_seed"] != 42
        or proposal["web_confirmation_seeds"] != [43, 44]
        or proposal["specialist_confirmation_seeds"] != [43]
        or proposal["repetition_seeds"] != [42, 43]
        or proposal["effective_epochs"] != [1, 2, 4]
    ):
        raise ValueError("first-wave seed/epoch proposal differs from the declared matrix")
    experiments = {item["id"]: item for item in data_plan["experiments"]}
    slots = []
    recipes = {"incumbent": incumbent}
    demands = []

    def capacity(recipe, horizon):
        counts = {}
        for category, tokens in _apportion(horizon, weights).items():
            counts.update(_apportion(tokens, recipe[category]))
        return counts

    for category, arms in proposal["treatments"].items():
        if len(arms) != expected_arms[category] or len({arm["id"] for arm in arms}) != len(arms):
            raise ValueError("treatment count/identity differs from the experiment funnel")
        family = "E1W" if category == "web" else "E1S"
        experiment = experiments[family]
        for arm in arms:
            recipe = {**incumbent, category: arm["sources"]}
            validate_recipe(recipe)
            recipe_id = f"{category}:{arm['id']}"
            recipes[recipe_id] = recipe
            required = capacity(recipe, experiment["tokens"])
            demands.append(required)
            slots.append(
                {
                    "id": f"{family}-{category}-{arm['id']}-seed42",
                    "family": family,
                    "stage": "screen",
                    "seed": 42,
                    "parameters": experiment["parameters"],
                    "training_tokens": experiment["tokens"],
                    "recipe": recipe_id,
                    "required_source_tokens": required,
                }
            )
        seeds = (
            proposal["web_confirmation_seeds"]
            if category == "web"
            else proposal["specialist_confirmation_seeds"]
        )
        for rank in (1, 2):
            for seed in seeds:
                slots.append(
                    {
                        "id": f"{family}-{category}-finalist{rank}-seed{seed}",
                        "family": family,
                        "stage": "confirmation",
                        "seed": seed,
                        "parameters": experiment["parameters"],
                        "training_tokens": experiment["tokens"],
                        "recipe": None,
                        "selection_ref": {"category": category, "finalist_rank": rank},
                    }
                )
    for epochs in proposal["effective_epochs"]:
        horizon = experiments["E3"]["tokens"]
        if horizon % epochs:
            raise ValueError("repetition horizon must divide into whole-token pools")
        required = capacity(incumbent, horizon // epochs)
        demands.append(required)
        for seed in proposal["repetition_seeds"]:
            slots.append(
                {
                    "id": f"E3-epochs{epochs}-seed{seed}",
                    "family": "E3",
                    "stage": "repetition",
                    "seed": seed,
                    "parameters": experiments["E3"]["parameters"],
                    "training_tokens": horizon,
                    "effective_epochs": epochs,
                    "unique_pool_tokens": horizon // epochs,
                    "recipe": "incumbent",
                    "required_source_tokens": required,
                }
            )
    for family in ("E1W", "E1S", "E3"):
        if sum(slot["family"] == family for slot in slots) != experiments[family]["new_runs"]:
            raise ValueError("logical run count differs from the selected data plan")
    envelope = {}
    for demand in demands:
        for source, count in demand.items():
            envelope[source] = max(envelope.get(source, 0), count)
    return {
        "format": "speck_first_wave_preparation_proposal",
        "format_version": 1,
        "status": "proposed_for_review_not_launchable",
        "category_weights_percent": weights,
        "recipes": recipes,
        "logical_slots": slots,
        "budget_gpu_hours": {
            family: experiments[family]["gpu_hours"] for family in ("E1W", "E1S", "E3")
        },
        "logical_training_tokens": sum(slot["training_tokens"] for slot in slots),
        "source_capacity_envelope": [
            {
                "source_id": source,
                "category": sources[source]["category"],
                "repo": sources[source]["repo"],
                "revision": sources[source]["revision"],
                "required_token_capacity": count,
            }
            for source, count in sorted(envelope.items())
        ],
        "source_capacity_total_tokens": sum(envelope.values()),
        "capacity_boundary": "D5-token requirements before alignment/lookahead and selection-loss headroom. Maximum source demand over proposed recipes assumes eligible reusable views; totals are not measured globally unique data. Seeds/confirmations do not multiply the stock requirement. Every actual arm still requires globally deduplicated, firewall-excluded inputs and preserved treatment membership.",
        "control_reuse": "Only after exact materialized model/data/order/optimizer/schedule/seed/evaluation identities match; no savings assumed in these ceilings.",
        "remaining_before_launch": [
            "recipe review and freeze",
            "D5 selection",
            "per-source reader/filter/language and membership contracts",
            "per-arm corpus identities and yield",
            "fixed proxy/training manifests and D4/D6 calibration dependency",
            "selection/confirmation analysis and tie/fallback manifests",
            "GH200 qualification",
        ],
        "training_authority": False,
    }


def load_first_wave(path):
    path = Path(path).resolve()
    proposal = json.loads(path.read_text())
    inputs = {}
    values = {}
    for name in ("data_plan", "source_registry", "source_use"):
        identity = proposal[name]
        target = (path.parent / identity["path"]).resolve()
        if file_sha256(target) != identity["sha256"]:
            raise ValueError(f"first-wave {name} identity mismatch")
        inputs[name] = {"path": str(target), "sha256": identity["sha256"]}
        values[name] = json.loads(target.read_text())
    result = compile_first_wave(
        proposal, values["data_plan"], values["source_registry"], values["source_use"]
    )
    result["inputs"] = {"proposal": {"path": str(path), "sha256": file_sha256(path)}, **inputs}
    return result
