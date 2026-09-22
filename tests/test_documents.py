"""Prose figures, mixture facts and links must track the numeric plan, in both directions.

The repository's budget revisions have repeatedly left stale figures in Markdown that
`check_program_plan.py` could not see, because it validates arithmetic inside `plan.json`
rather than the documents that restate it. Three separate revisions did it: a budget move
left stale GPU-hours, the 2026-09-21 architecture deferral left a research-envelope table
still billing the deferred study, and the 2026-09-22 mixture re-freeze left three copies of
the superseded eight-bank table. These tests pin the guard that closes that gap: it must
accept the tree as committed, and it must reject a contradiction whichever side of the pair
moved.
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_documents", ROOT / "experiments/main-data/check_documents.py"
)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


@pytest.fixture
def plan():
    return json.loads((ROOT / "experiments/main-data/plan.json").read_text())


def validate_plan(tmp_path, plan):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return checker.validate(path)


def test_committed_documents_agree_with_the_plan():
    result = checker.validate()
    assert result["status"] == "prose_figures_links_and_mixture_agree_with_numeric_plan"
    assert result["training_admitted"] is False
    # A guard that binds nothing would also pass; require it to be doing real work.
    assert result["bound_figures_checked"] > 40
    assert result["links_resolved"] > 100


def test_moving_a_reservation_without_the_documents_is_rejected(tmp_path, plan):
    """The realistic failure: plan.json is revised and the prose is forgotten."""
    reservations = plan["compute"]["reservations_gpu_hours"]
    reservations["capability_and_agentic_mid_training"] += 50
    reservations["protected_recovery_and_evaluation"] -= 50
    with pytest.raises(ValueError, match="mid-training production"):
        validate_plan(tmp_path, plan)


def test_moving_a_research_cap_without_the_documents_is_rejected(tmp_path, plan):
    research = plan["compute"]["data_experiments_breakdown_gpu_hours"]
    research["mid_training"] -= 60
    research["post_training"] += 60
    with pytest.raises(ValueError, match="mid-training research|data research split"):
        validate_plan(tmp_path, plan)


def test_post_training_subdivision_drift_is_rejected(tmp_path, plan):
    """The subdivision that lost its final self-SFT line once already."""
    post = plan["compute"]["proposed_post_training_breakdown_gpu_hours"]
    post["sft"] += 50
    post["final_self_sft"] -= 50
    with pytest.raises(ValueError, match="post-training SFT subdivision"):
        validate_plan(tmp_path, plan)


def test_a_slash_compound_is_checked_as_one_claim(tmp_path, plan):
    """ "700/360/170-hour" is a single claim about three reservations, not three loose numbers."""
    splits = checker._splits(plan)
    assert "data research split" in [description for description, _, _ in splits]
    plan["compute"]["data_experiments_breakdown_gpu_hours"]["pretraining"] += 10
    with pytest.raises(ValueError, match="data research split"):
        validate_plan(tmp_path, plan)


def test_the_history_pointer_is_not_governed():
    """Archived documents preserve superseded figures on purpose."""
    assert "archive/" in checker.SKIP
    assert not any("archive/" in str(path) for path in checker._documents())


def test_moving_a_bank_weight_without_the_documents_is_rejected(tmp_path, plan):
    """The 2026-09-22 failure: the mixture is re-frozen and a rendered table is forgotten."""
    for bank in plan["main_pretraining"]["mixture"]:
        if bank["id"] == "natural_code":
            bank["weight_percent"] = 30
    with pytest.raises(ValueError, match="bank 'natural_code' is given"):
        validate_plan(tmp_path, plan)


def test_moving_the_domain_split_without_the_documents_is_rejected(tmp_path, plan):
    plan["main_pretraining"]["code_percent"] = 30
    plan["main_pretraining"]["math_percent"] = 30
    with pytest.raises(ValueError, match="domain split stated as"):
        validate_plan(tmp_path, plan)


def test_dropping_a_bank_without_the_documents_is_rejected(tmp_path, plan):
    """Bank count is restated in prose; removing a bank must not leave 'six banks' behind."""
    plan["main_pretraining"]["mixture"] = plan["main_pretraining"]["mixture"][:-1]
    with pytest.raises(ValueError, match="banks, plan declares"):
        validate_plan(tmp_path, plan)


def test_a_research_envelope_table_row_is_adjudicated(tmp_path, plan):
    """The row form has no label beside the figure, which is how 600/300/150 survived."""
    plan["compute"]["data_experiments_breakdown_gpu_hours"]["pretraining"] = 600
    with pytest.raises(ValueError, match="table row states 700 GPU-hours"):
        validate_plan(tmp_path, plan)


def test_a_one_pass_bound_must_match_the_derived_supply_gap(tmp_path, plan, monkeypatch):
    """One-pass bounds are derived, so a typed copy must never outlive its source."""
    supply = json.loads((ROOT / "experiments/main-data/supply-gap.json").read_text())
    for bank in supply["banks"]:
        if bank["id"] == "natural_code":
            bank["maximum_one_pass_exposure_from_retained_stock"] = 1_589_000_000
    path = tmp_path / "supply-gap.json"
    path.write_text(json.dumps(supply))
    monkeypatch.setattr(checker, "SUPPLY_GAP", path)
    with pytest.raises(ValueError, match="one-pass bound"):
        checker.validate()


def _isolated_tree(tmp_path, monkeypatch, documents):
    """Point the checker at a small synthetic tree so document-side faults can be tested."""
    for name, text in documents.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(checker, "_documents", lambda: sorted(tmp_path.rglob("*.md")))


def test_a_stranded_link_is_rejected(tmp_path, monkeypatch):
    """Consolidating a document must not leave a reference pointing at the old path."""
    _isolated_tree(tmp_path, monkeypatch, {"a.md": "# A\n\nSee [b](b.md).\n"})
    with pytest.raises(ValueError, match="link target does not exist"):
        checker.validate()


def test_a_stranded_anchor_is_rejected(tmp_path, monkeypatch):
    """Renaming a heading must not leave a reference pointing at the old anchor."""
    _isolated_tree(
        tmp_path,
        monkeypatch,
        {
            "a.md": "# A\n\nSee [b](b.md#gone).\n",
            "b.md": "# B\n\n## Still here\n",
        },
    )
    with pytest.raises(ValueError, match="no such heading anchor"):
        checker.validate()


def test_a_heading_anchor_matches_github_slugging(tmp_path, monkeypatch):
    """An em dash is dropped and its surrounding spaces each become a hyphen."""
    assert checker._slug("Cagliostro v3 review — 2026-09-20") == "cagliostro-v3-review--2026-09-20"
    _isolated_tree(
        tmp_path,
        monkeypatch,
        {
            "a.md": "# A\n\nSee [b](b.md#review--2026-09-20).\n",
            "b.md": "# B\n\n## Review — 2026-09-20\n",
        },
    )
    assert checker.validate()["training_admitted"] is False


def test_a_retired_bank_may_not_reappear_as_a_mixture_row(tmp_path, monkeypatch):
    """Prose may discuss checked_code; a weighted table row would re-bank its dropped share."""
    _isolated_tree(
        tmp_path,
        monkeypatch,
        {
            "a.md": (
                "# A\n\n| Bank | Share |\n| --- | ---: |\n"
                "| `checked_code` — checked code and repair | 5% |\n"
            )
        },
    )
    with pytest.raises(ValueError, match="re-freeze removed it"):
        checker.validate()
