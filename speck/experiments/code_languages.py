"""Compile matched code-language requirements without changing first-wave source assignments."""

import json
from pathlib import Path

from speck.experiments.code_preparation_decision import (
    bound_json,
    load_decision,
    validate_language_successor,
)
from speck.provenance.io import file_sha256

CODE_SOURCES = {"stack_edu", "stack_v3_train_permissive"}
LANGUAGES = {
    "Python",
    "C++",
    "C",
    "Java",
    "JavaScript",
    "TypeScript",
    "Rust",
    "Go",
    "Shell",
    "SQL",
    "Markdown",
}


def compile_code_languages(spec, wave):
    weights = spec.get("language_weights_percent", {})
    if (
        spec.get("format") != "speck_code_language_preparation"
        or spec.get("format_version") != 1
        or spec.get("status") != "selected_for_preparation_not_launchable"
        or spec.get("training_authority") is not False
        or spec.get("headroom_percent") != 20
        or set(weights) != LANGUAGES
        or any(type(weight) is not int or weight <= 0 for weight in weights.values())
        or sum(weights.values()) != 100
        or spec.get("source_ids") != sorted(CODE_SOURCES)
    ):
        raise ValueError("unsupported matched code-language preparation")
    if wave.get("status") != "selected_for_preparation_not_launchable":
        raise ValueError("code allocation requires selected first-wave preparation")
    demand = {
        row["source_id"]: row["required_token_capacity"]
        for row in wave["source_capacity_envelope"]
        if row["category"] == "code"
    }
    if set(demand) != CODE_SOURCES:
        raise ValueError("code source identities differ from the selected comparison")

    def allocate(tokens, percent):
        if type(tokens) is not int or tokens <= 0 or tokens * percent % 100:
            raise ValueError("language token quota would require undeclared rounding")
        return tokens * percent // 100

    slots = []
    maxima = {source: {language: 0 for language in weights} for source in CODE_SOURCES}
    for slot in wave["logical_slots"]:
        if slot["recipe"] is None:
            slots.append(
                {
                    "id": slot["id"],
                    "recipe": None,
                    "source_language_tokens": None,
                    "selection_ref": slot["selection_ref"],
                    "status": "apply_common_language_weights_after_registered_finalist_selection",
                }
            )
            continue
        sources = wave["recipes"][slot["recipe"]]["code"]
        if not set(sources).issubset(CODE_SOURCES):
            raise ValueError("code slot contains a different source")
        quotas = {}
        for source in sources:
            tokens = slot["required_source_tokens"][source]
            quotas[source] = {
                language: allocate(tokens, weight) for language, weight in weights.items()
            }
            for language, count in quotas[source].items():
                maxima[source][language] = max(maxima[source][language], count)
        slots.append({"id": slot["id"], "recipe": slot["recipe"], "source_language_tokens": quotas})
    capacity = []
    for source, total in sorted(demand.items()):
        for language, weight in weights.items():
            nominal = allocate(total, weight)
            if maxima[source][language] != nominal:
                raise ValueError("source capacity envelope does not match maximum logical demand")
            capacity.append(
                {
                    "source_id": source,
                    "language": language,
                    "nominal_tokens": nominal,
                    "preparation_target_tokens": allocate(nominal, 100 + spec["headroom_percent"]),
                }
            )
    return {
        "format": "speck_code_language_preparation_result",
        "format_version": 1,
        "status": "matched_requirements_compiled_not_measured_supply",
        "language_weights_percent": weights,
        "source_language_capacity_envelope": capacity,
        "logical_slot_code_quotas": slots,
        "boundary": "The same selected-tokenizer language proportions apply independently to each code source in every arm, including both halves of the equal blend and shared code background. Quotas are nominal reusable-pool requirements, not measured eligible stock or repeated E3 exposure. Unselected confirmations retain null recipes/quotas until registered finalist selection. Whole-document assembly, alignment/lookahead, joint exclusion and launch authority remain pending. A short language pool is not replaced by surplus in another language; headroom is required per language.",
        "training_authority": False,
    }


def load_code_languages(path):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    inputs, values = {}, {}
    for key in ("first_wave", "tokenizer_decision"):
        identity = spec[key]
        target = (path.parent / identity["path"]).resolve()
        if file_sha256(target) != identity["sha256"]:
            raise ValueError(f"code language {key} identity mismatch")
        inputs[key] = {"path": str(target), "sha256": identity["sha256"]}
        values[key] = json.loads(target.read_text())
    if values["tokenizer_decision"]["status"] != "tokenizer_selected_and_frozen":
        raise ValueError("code language requirements need the selected tokenizer")
    if "code_preparation_decision" in spec:
        decision = load_decision(spec["code_preparation_decision"], path.parent)
        previous = bound_json(spec["supersedes"], path.parent)
        validate_language_successor(spec, previous, values["first_wave"], decision)
        inputs["code_preparation_decision"] = spec["code_preparation_decision"]
        inputs["supersedes"] = spec["supersedes"]
    result = compile_code_languages(spec, values["first_wave"])
    result["inputs"] = {"plan": {"path": str(path), "sha256": file_sha256(path)}, **inputs}
    result["tokenizer_sha256"] = values["tokenizer_decision"]["tokenizer_fingerprint"]
    return result
