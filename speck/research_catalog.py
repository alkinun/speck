"""Validate Speck's research compendium catalog and paper claim registry."""

import json
import re
from collections import Counter
from pathlib import Path

from speck.io import file_sha256

CATALOG_FORMAT = "speck_research_catalog"
CLAIMS_FORMAT = "speck_paper_claim_registry"
FORMAT_VERSION = 1
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
CLAIM_ID_PATTERN = re.compile(r"^C-[A-Z][A-Z0-9-]*$")
ENTRY_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9][a-z0-9-]*\.md$")
COLLECTION_AUTHORITIES = {
    "active_scope",
    "external_evidence",
    "historical_only",
    "method_implementation",
    "process_record",
    "publication_output",
    "reproducibility_input",
    "scientific_evidence",
}
MUTABILITIES = {
    "append_only",
    "generated_from_evidence",
    "immutable_history",
    "normal_version_control",
    "versioned_successors",
}
CLAIM_STATUSES = {
    "planned",
    "prior_evidence_only",
    "partially_supported",
    "supported",
    "refuted",
    "retired",
}
NOTEBOOK_HEADINGS = (
    "## Context",
    "## Work performed",
    "## Decisions",
    "## Evidence and links",
    "## Open questions",
    "## Next actions",
)


def _load_object(path, context):
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {context} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain a JSON object: {path}")
    return value


def _exact_keys(value, expected, context):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{context} must contain exactly: {', '.join(sorted(expected))}")


def _repository_root(path):
    path = Path(path).resolve()
    for parent in (path.parent, *path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ValueError(f"cannot resolve repository root from {path}")


def _repository_path(root, value, context, *, kind=None):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{context} must be a non-empty repository-relative path")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{context} escapes the repository: {value}")
    if kind == "file" and not path.is_file():
        raise ValueError(f"{context} file does not exist: {value}")
    if kind == "directory" and not path.is_dir():
        raise ValueError(f"{context} directory does not exist: {value}")
    return path


def _unique_ids(values, context, pattern=ID_PATTERN):
    identifiers = [value.get("id") if isinstance(value, dict) else None for value in values]
    if any(
        not isinstance(identifier, str) or not pattern.fullmatch(identifier)
        for identifier in identifiers
    ):
        raise ValueError(f"every {context} requires a valid unique id")
    duplicates = sorted(
        identifier for identifier, count in Counter(identifiers).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate {context} ids: {', '.join(duplicates)}")
    return identifiers


def validate_claim_registry(path, *, repository_root=None):
    """Validate paper claims and every current contract/evidence reference."""

    path = Path(path).resolve()
    root = Path(repository_root).resolve() if repository_root else _repository_root(path)
    registry = _load_object(path, "paper claim registry")
    _exact_keys(
        registry,
        {"format", "format_version", "paper_id", "status", "contract", "claims"},
        "paper claim registry",
    )
    if registry["format"] != CLAIMS_FORMAT or registry["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported paper claim registry format")
    _repository_path(root, registry["contract"], "paper contract", kind="file")
    claims = registry["claims"]
    if not isinstance(claims, list) or not claims:
        raise ValueError("paper claim registry requires claims")
    identifiers = _unique_ids(claims, "paper claim", CLAIM_ID_PATTERN)
    for claim in claims:
        claim_id = claim["id"]
        _exact_keys(
            claim,
            {
                "id",
                "title",
                "status",
                "question",
                "decision_rule",
                "contracts",
                "evidence",
                "planned_outputs",
                "paper_sections",
                "headline_eligible",
            },
            f"paper claim {claim_id}",
        )
        for field in ("title", "question", "decision_rule"):
            if not isinstance(claim[field], str) or not claim[field].strip():
                raise ValueError(f"paper claim {claim_id} requires {field}")
        if claim["status"] not in CLAIM_STATUSES:
            raise ValueError(f"paper claim {claim_id} has unsupported status")
        if not isinstance(claim["headline_eligible"], bool):
            raise ValueError(f"paper claim {claim_id} headline_eligible must be boolean")
        for field in ("contracts", "evidence"):
            values = claim[field]
            if not isinstance(values, list) or (field == "contracts" and not values):
                raise ValueError(f"paper claim {claim_id} {field} must be a list")
            for index, value in enumerate(values):
                _repository_path(
                    root, value, f"paper claim {claim_id} {field}[{index}]", kind="file"
                )
        if (
            claim["status"]
            in {
                "prior_evidence_only",
                "partially_supported",
                "supported",
                "refuted",
            }
            and not claim["evidence"]
        ):
            raise ValueError(f"paper claim {claim_id} status requires evidence")
        if claim["status"] == "supported" and (
            not any(value.startswith("findings/") for value in claim["evidence"])
            or not any(value.startswith("results/") for value in claim["evidence"])
        ):
            raise ValueError(
                f"supported paper claim {claim_id} requires both a finding and result evidence"
            )
        for field in ("planned_outputs", "paper_sections"):
            values = claim[field]
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) or not value.strip() for value in values)
            ):
                raise ValueError(f"paper claim {claim_id} requires non-empty {field}")
    return {
        "path": str(path.relative_to(root)),
        "sha256": file_sha256(path),
        "paper_id": registry["paper_id"],
        "status": registry["status"],
        "claims": len(claims),
        "claim_statuses": dict(sorted(Counter(claim["status"] for claim in claims).items())),
        "claim_ids": identifiers,
        "headline_ready": [
            claim["id"]
            for claim in claims
            if claim["headline_eligible"] and claim["status"] == "supported"
        ],
    }


def _validate_notebook(path):
    entries = sorted(entry for entry in path.glob("*.md") if entry.name != "README.md")
    for entry in entries:
        if not ENTRY_PATTERN.fullmatch(entry.name):
            raise ValueError(f"research notebook entry has invalid name: {entry.name}")
        text = entry.read_text(encoding="utf-8")
        missing = [heading for heading in NOTEBOOK_HEADINGS if heading not in text]
        if missing:
            raise ValueError(
                f"research notebook entry {entry.name} is missing headings: {', '.join(missing)}"
            )
    return entries


def _collection_file_count(path):
    count = 0
    for file in path.rglob("*"):
        relative = file.relative_to(path)
        if (
            file.is_file()
            and file.suffix != ".pyc"
            and "__pycache__" not in relative.parts
            and not any(part.startswith(".") for part in relative.parts)
        ):
            count += 1
    return count


def validate_research_catalog(path, *, repository_root=None):
    """Validate the central catalog, collection paths, lifecycle, notebook, and claims."""

    path = Path(path).resolve()
    root = Path(repository_root).resolve() if repository_root else _repository_root(path)
    catalog = _load_object(path, "research catalog")
    _exact_keys(
        catalog,
        {
            "format",
            "format_version",
            "project",
            "status",
            "active_program",
            "collections",
            "systems",
            "lifecycle",
            "paper",
        },
        "research catalog",
    )
    if catalog["format"] != CATALOG_FORMAT or catalog["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported research catalog format")

    active = catalog["active_program"]
    _exact_keys(active, {"id", "path", "charter", "execution", "paper_contract"}, "active program")
    if not ID_PATTERN.fullmatch(active["id"]):
        raise ValueError("active program id is invalid")
    for field in ("path",):
        _repository_path(root, active[field], f"active program {field}", kind="directory")
    for field in ("charter", "execution", "paper_contract"):
        _repository_path(root, active[field], f"active program {field}", kind="file")

    collections = catalog["collections"]
    if not isinstance(collections, list) or not collections:
        raise ValueError("research catalog requires collections")
    collection_ids = _unique_ids(collections, "research collection")
    collection_status = {}
    notebook_path = None
    for collection in collections:
        collection_id = collection["id"]
        _exact_keys(
            collection,
            {"id", "name", "path", "index", "role", "authority", "mutability"},
            f"research collection {collection_id}",
        )
        directory = _repository_path(
            root, collection["path"], f"research collection {collection_id} path", kind="directory"
        )
        _repository_path(
            root, collection["index"], f"research collection {collection_id} index", kind="file"
        )
        if collection["authority"] not in COLLECTION_AUTHORITIES:
            raise ValueError(f"research collection {collection_id} has unsupported authority")
        if collection["mutability"] not in MUTABILITIES:
            raise ValueError(f"research collection {collection_id} has unsupported mutability")
        files = _collection_file_count(directory)
        collection_status[collection_id] = {"path": collection["path"], "files": files}
        if collection_id == "notebook":
            notebook_path = directory
    if active["id"] not in collection_ids:
        raise ValueError("active program must be a catalog collection")
    if notebook_path is None:
        raise ValueError("research catalog requires the notebook collection")

    systems = catalog["systems"]
    if not isinstance(systems, list) or not systems:
        raise ValueError("research catalog requires research systems")
    _unique_ids(systems, "research system")
    for system in systems:
        _exact_keys(
            system, {"id", "name", "location", "role", "authority"}, f"system {system['id']}"
        )
        if any(not isinstance(system[field], str) or not system[field] for field in system):
            raise ValueError(f"research system {system['id']} requires string values")

    lifecycle = catalog["lifecycle"]
    if not isinstance(lifecycle, list) or not lifecycle:
        raise ValueError("research catalog requires lifecycle stages")
    _unique_ids(lifecycle, "research lifecycle stage")
    for stage in lifecycle:
        _exact_keys(stage, {"id", "order", "name", "record", "exit_gate"}, f"stage {stage['id']}")
    if [stage["order"] for stage in lifecycle] != list(range(1, len(lifecycle) + 1)):
        raise ValueError("research lifecycle order must be contiguous")

    paper = catalog["paper"]
    _exact_keys(paper, {"workspace", "contract", "claims"}, "paper catalog")
    _repository_path(root, paper["workspace"], "paper workspace", kind="directory")
    _repository_path(root, paper["contract"], "paper catalog contract", kind="file")
    claims_path = _repository_path(root, paper["claims"], "paper claims", kind="file")
    claims = validate_claim_registry(claims_path, repository_root=root)
    entries = _validate_notebook(notebook_path)

    return {
        "format": "speck_research_catalog_validation",
        "format_version": FORMAT_VERSION,
        "catalog": {"path": str(path.relative_to(root)), "sha256": file_sha256(path)},
        "project": catalog["project"],
        "status": catalog["status"],
        "active_program": active["id"],
        "collections": collection_status,
        "systems": len(systems),
        "lifecycle_stages": len(lifecycle),
        "notebook_entries": len(entries),
        "paper": claims,
    }
