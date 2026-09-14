"""Compile additive source-use preparation choices through the preserved first-wave design."""

import json
from pathlib import Path

from speck.data.rights import load_source_use_extension
from speck.experiments.first_wave import compile_first_wave
from speck.provenance.io import file_sha256


def compile_preparation(proposal, data_plan, registry, source_use, *, extensions=()):
    if (
        proposal.get("format_version") != 2
        or proposal.get("status") != "selected_for_preparation_not_launchable"
    ):
        raise ValueError("expected v2 preparation-only source assignments")
    approved = set(source_use["approved_source_ids"])
    sources = {row["id"]: row for row in registry["sources"]}
    for extension in extensions:
        if (
            extension.get("status") != "human_approved_source_extension"
            or extension.get("decision") != "approve"
            or extension.get("automated_approval_made") is not False
            or extension.get("training_authority") is not False
            or extension.get("scope_details") != source_use.get("scope_details")
        ):
            raise ValueError("invalid additive source approval")
        source = extension["source"]
        if source["id"] in sources or source["id"] in approved:
            raise ValueError("source extension cannot replace an existing source")
        sources[source["id"]] = {**source, "priority": "primary_screen"}
        approved.add(source["id"])
    # These effective in-memory views combine existing human decisions. Neither
    # original registry nor acceptance is overwritten or reissued.
    result = compile_first_wave(
        {**proposal, "format_version": 1, "status": "proposed_for_review_not_frozen"},
        data_plan,
        {**registry, "sources": list(sources.values())},
        {**source_use, "approved_source_ids": sorted(approved)},
    )
    return {
        **result,
        "format_version": 2,
        "status": "selected_for_preparation_not_launchable",
        "source_use_boundary": "Effective eligibility is the union of the hash-bound parent acceptance and explicit additive human decisions; original records remain unchanged.",
    }


def load_preparation(path):
    path = Path(path).resolve()
    proposal = json.loads(path.read_text())
    inputs, values = {}, {}
    for name in ("data_plan", "source_registry", "source_use"):
        identity = proposal[name]
        target = (path.parent / identity["path"]).resolve()
        if file_sha256(target) != identity["sha256"]:
            raise ValueError(f"first-wave {name} identity mismatch")
        inputs[name] = {"path": str(target), "sha256": identity["sha256"]}
        values[name] = json.loads(target.read_text())
    extensions = []
    for identity in proposal["source_use_extensions"]:
        target = (path.parent / identity["path"]).resolve()
        if file_sha256(target) != identity["sha256"]:
            raise ValueError("first-wave source extension identity mismatch")
        extension = load_source_use_extension(target)
        if extension["parent_acceptance"]["sha256"] != inputs["source_use"]["sha256"]:
            raise ValueError("source extension belongs to another acceptance")
        extensions.append(extension)
    inputs["source_use_extensions"] = [extension["identity"] for extension in extensions]
    result = compile_preparation(
        proposal,
        values["data_plan"],
        values["source_registry"],
        values["source_use"],
        extensions=extensions,
    )
    result["inputs"] = {"proposal": {"path": str(path), "sha256": file_sha256(path)}, **inputs}
    return result
