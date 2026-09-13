"""Calculate conditional E1/E3 requirements from checked reference-token capacity."""


def screen_capacity(data_plan, registry, capacity_record):
    categories = data_plan["categories"]
    weights = {item["id"]: item["prior_percent"] for item in categories}
    if (
        set(weights) != {"web", "code", "math", "synthetic", "science", "reference"}
        or sum(weights.values()) != 100
    ):
        raise ValueError("capacity review requires the six-category prior")
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in weights.values()):
        raise ValueError("capacity prior weights must be positive integer percentages")
    experiments = {item["id"]: item for item in data_plan["experiments"]}
    measured = capacity_record["pilot_training_capacity"]["per_source"]
    by_source = {item["id"]: item for item in registry["sources"]}
    available = {key: 0 for key in weights}
    measured_sources = []
    for key, value in measured.items():
        if not key.startswith("pilot_train__"):
            raise ValueError("capacity source is not a retained pilot-training output")
        source_id = key.removeprefix("pilot_train__")
        source = by_source[source_id]
        tokens = value["mistral_tokens"]
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("invalid measured reference-token capacity")
        available[source["category"]] += tokens
        measured_sources.append(
            {
                "source_id": source_id,
                "category": source["category"],
                "reference_tokens": tokens,
                "documents": value["documents"],
                "registry_priority": source.get("priority"),
            }
        )
    if (
        sum(available.values())
        != capacity_record["pilot_training_capacity"]["total_mistral_reference_tokens"]
    ):
        raise ValueError("capacity total differs from its per-source measurements")
    limits = {category: tokens * 100 // weights[category] for category, tokens in available.items()}
    limiting = min(limits, key=limits.get)
    horizon = experiments["E3"]["tokens"]
    repetition = []
    for epochs in (1, 2, 4):
        if horizon % epochs:
            raise ValueError("E3 horizon does not divide into the declared epoch arms")
        unique = horizon // epochs
        required = {category: (unique * weight + 99) // 100 for category, weight in weights.items()}
        gaps = {category: max(0, required[category] - available[category]) for category in weights}
        repetition.append(
            {
                "effective_epochs": epochs,
                "training_tokens_per_run": horizon,
                "unique_pool_tokens": unique,
                "required_by_category": required,
                "missing_by_category": gaps,
                "capacity_sufficient": not any(gaps.values()),
            }
        )
    scenarios = {}
    for name in ("E1W", "E1S"):
        tokens = experiments[name]["tokens"]
        scenarios[name] = {
            "training_tokens_per_run": tokens,
            "conditional_prior_substitution_category_tokens": {
                category: (tokens * weight + 99) // 100 for category, weight in weights.items()
            },
            "pure_category_treatment_tokens_if_chosen": tokens,
            "scientific_recipe_frozen": False,
        }
    known = {item["source_id"] for item in measured_sources}
    unmeasured = [
        {
            "source_id": source["id"],
            "category": source["category"],
            "measured_tokens_in_this_evidence": None,
        }
        for source in registry["sources"]
        if source["priority"] == "primary_screen" and source["id"] not in known
    ]
    return {
        "format": "speck_data_screen_capacity_review",
        "format_version": 1,
        "status": "reference_capacity_gaps_and_unfrozen_recipes",
        "reference_tokenizer": capacity_record["pilot_training_capacity"]["tokenizer"],
        "measured_sources": measured_sources,
        "available_by_category": available,
        "total_available_reference_tokens": sum(available.values()),
        "balanced_prior_pool_upper_bound_tokens": limits[limiting],
        "limiting_category": limiting,
        "E3_conditional_prior_requirements": repetition,
        "E1_conditional_requirements": scenarios,
        "primary_screen_sources_without_capacity_in_this_evidence": unmeasured,
        "boundary": "Archived measured reference-token bank, conditional on the documented category prior and reuse eligibility. Not final-tokenizer counts, frozen E1/E3 source assignments, distinct data per seed, prepared launch shards, or total upstream source capacity. Other banks are not added without union/dedup evidence. Bounds exclude packing/alignment/lookahead headroom.",
        "training_authority": False,
    }
