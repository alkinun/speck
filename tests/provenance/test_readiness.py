"""Reject plausible cross-contract drift without mistaking static checks for GPU evidence."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from speck.provenance.readiness import audit_readiness, check_contracts, relocated_source_path
from speck.provenance.repository import repository_root

ROOT = repository_root()


def read(path):
    return json.loads((ROOT / path).read_text())


@pytest.fixture
def inputs():
    selected = read("research/catalog.json")["active_contracts"]
    return {
        "values": {role: read(row["path"]) for role, row in selected.items()},
        "claims": read("paper/claims.json"),
        "status": read("research/status.json"),
        "shapes": read("results/systems/r0-shape-preparation-20260915.json"),
    }


def test_current_audit_binds_inputs_but_leaves_execution_pending():
    result = audit_readiness(ROOT)
    assert result["status"] == "pre_access_contracts_consistent_execution_pending"
    assert not result["training_authority"]
    assert result["allocated_node_days_if_awarded"] == pytest.approx(5000 / 4 / 24)
    assert len(result["pending"]) >= 6
    assert len(result["inputs"]) == len({r["path"] for r in result["inputs"]})
    for row in result["inputs"]:
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("values", "model", "flagship", "base_tokens"), 400_000_000_000, "horizons differ"),
        (("values", "model", "flagship", "pretraining_gpu_hours_ceiling"), 2400, "base budget"),
        (("values", "capability", "gpu_hours_ceiling"), 701, "capability subtotal"),
        (("values", "execution", "budget", "research_gpu_hours"), 749, "research subtotal"),
        (("values", "integration", "core", "branch_runs"), 9, "parent/branch"),
        (("values", "integration", "core", "paired_seeds"), [42, 42, 44], "duplicate paired"),
        (("values", "integration", "transfer", "paired_seeds"), [42], "overlaps core"),
        (("values", "heldout_evaluation", "lengths", 1), 16384, "context lengths"),
        (("values", "data", "base", "category_percent", "web"), 54, "category weights"),
        (
            ("values", "data", "long_context", "general_replay_fraction_of_assistant_tokens"),
            0.8,
            "conserve exposure",
        ),
        (("shapes", "cases", 0, "model_vocab_size"), 32000, "vocabulary differs"),
        (("shapes", "cases", 0, "gpu_fit_pass"), True, "cannot certify GPU"),
        (("shapes", "cases", 0, "actual_gpu_peak_bytes"), 0, "cannot certify GPU"),
        (("claims", "claims", 0, "status"), "supported", "paper status exceeds"),
        (("claims", "claims", 0, "evidence"), ["shape-only.json"], "paper status exceeds"),
        (("status", "allocation_status", "compute_access_date"), "2026-09-18", "access date"),
        (("values", "integration", "execution_ready"), True, "not yet execution-qualified"),
        (("values", "model", "training_authority"), True, "not training authority"),
    ],
)
def test_rejects_cross_contract_or_evidence_overclaim(inputs, path, replacement, message):
    target = inputs
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(ValueError, match=message):
        check_contracts(**inputs)


def test_matching_factor_edits_cannot_hide_a_missing_cell(inputs):
    for role in ("integration", "architecture"):
        factors = inputs["values"][role]["factors"]["training"]
        factors[1] = factors[0]
    with pytest.raises(ValueError, match="four factorial cells"):
        check_contracts(**inputs)


def test_duplicate_shape_cannot_substitute_for_a_missing_length(inputs):
    cases = inputs["shapes"]["cases"]
    cases[-1] = cases[0]
    with pytest.raises(ValueError, match="shape coverage"):
        check_contracts(**inputs)


def test_source_mapping_supports_arbitrarily_named_original_checkouts():
    assert relocated_source_path(
        "/temporary/research-checkout/speck/model/state.py",
        "/temporary/research-checkout/research/flagship/plan.json",
        "research/flagship/plan.json",
    ) == Path("speck/model/state.py")


@pytest.mark.parametrize("source", ["/unrelated/state.py", "/old/repo/../state.py"])
def test_source_mapping_rejects_paths_outside_original_checkout(source):
    with pytest.raises(ValueError):
        relocated_source_path(source, "/old/repo/research/plan.json", "research/plan.json")


def test_preparation_cleanup_preserves_exact_previous_entrypoints():
    snapshot = read("research/history/2026-09-15-readiness-cleanup/manifest.json")
    assert len(snapshot["files"]) == 2
    for row in snapshot["files"]:
        original = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"{snapshot['revision']}:{row['original_path']}"]
        )
        assert hashlib.sha256(original).hexdigest() == row["sha256"]
        assert (ROOT / row["preserved_path"]).read_bytes() == original
