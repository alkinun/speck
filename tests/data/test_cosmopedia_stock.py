import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import speck.data.acquisition_units as units
import speck.data.cosmopedia_stock as cosmo
from speck.provenance.io import file_sha256
from tests.data.test_acquisition_units import fixture


def policy():
    root = Path(__file__).resolve().parents[2]
    return cosmo.load_cosmopedia_preparation(
        root / "research/flagship/cosmopedia_stock_preparation_v1.json"
    )["base"]["cosmopedia_policy"]


def english(monkeypatch):
    monkeypatch.setattr(
        cosmo, "_language_identifier", lambda: SimpleNamespace(classify=lambda text: ("en", 0.99))
    )


def document():
    return {
        "content": (
            "Astronomers compare distant stellar spectra to identify chemical elements. "
            "A prism separates incoming light into wavelengths that reveal absorption features. "
            "Laboratory measurements provide a reference for those observations, while telescope "
            "calibration reduces systematic errors. Researchers estimate uncertainty before "
            "drawing conclusions about the composition and temperature of a star."
        ),
        "metadata": {
            "prompt": "Explain the physical science of stars.",
            "style": "textbook",
            "seed_source_label": "fineweb",
            "audience": "students",
        },
    }


def test_prompt_is_hashed_separately_from_seed_label_without_copying(monkeypatch):
    english(monkeypatch)
    doc = document()
    reason, metadata = cosmo.cosmopedia_document(doc, policy())
    assert reason is None
    assert metadata["seed_source_label"] == "fineweb"
    assert (
        metadata["prompt_sha256"] == hashlib.sha256(doc["metadata"]["prompt"].encode()).hexdigest()
    )
    assert metadata["seed_sha256"] == metadata["prompt_sha256"]
    assert "prompt" not in metadata
    assert metadata["generator"]["revision"] == "not_disclosed_by_dataset_card"
    assert metadata["seed_source"]["revision"] == "not_disclosed_by_dataset_card"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("prompt_copy", "synthetic_unchanged_seed"),
        ("missing_label", "synthetic_missing_lineage"),
        ("missing_prompt", "synthetic_missing_lineage"),
        ("model_identity", "synthetic_model_identity_phrase"),
        ("repetition", "synthetic_repeated_ngrams"),
    ],
)
def test_qualified_synthetic_rejections(monkeypatch, change, reason):
    english(monkeypatch)
    doc = document()
    if change == "prompt_copy":
        doc["metadata"]["prompt"] = doc["content"]
    elif change == "missing_label":
        doc["metadata"]["seed_source_label"] = ""
    elif change == "missing_prompt":
        doc["metadata"].pop("prompt")
    elif change == "model_identity":
        doc["content"] += " As an AI language model, I can discuss this."
    else:
        doc["content"] = (
            "These scientific methods provide important evidence for researchers. " * 15
        )
    assert cosmo.cosmopedia_document(doc, policy()) == (reason, None)


def test_diagnostics_report_reused_prompts_and_natural_template_shares(tmp_path, monkeypatch):
    english(monkeypatch)
    _, metadata = cosmo.cosmopedia_document(document(), policy())
    rows = [{"text": "aaaa", "metadata": metadata}, {"text": "bb", "metadata": metadata}]
    third = copy.deepcopy(metadata)
    third.update(prompt_sha256="b" * 64, template_prefix_sha256="c" * 64, style="story")
    rows.append({"text": "cc", "metadata": third})
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = cosmo.cosmopedia_diagnostics(path)
    assert report["documents"] == 3
    assert report["documents_beyond_first_identical_prompt"] == 1
    assert report["template_byte_hhi"] == 0.625
    assert report["styles"][0]["byte_share"] == 0.75
    assert report["seed_domain_coverage"] == "unavailable_in_released_schema"


def test_synthetic_acquisition_resume_preserves_lineage_and_binds_policy(tmp_path, monkeypatch):
    english(monkeypatch)
    plan, unit = fixture(tmp_path, monkeypatch)
    unit["category"] = "synthetic"
    unit["reader"]["metadata_columns"].update(
        prompt="prompt", style="style", seed_source_label="seed_data"
    )
    raw_path = units._raw_local_path(
        Path(plan["raw_directory"]),
        unit["reader"],
        unit["reader"]["revision"],
        unit["raw"]["filename"],
    )
    rows = pq.read_table(raw_path).to_pylist()
    for i, row in enumerate(rows):
        row.update(
            prompt=row["text"] if i == 4 else "Describe this topic for a student.",
            style="textbook",
            seed_data="fineweb",
        )
    pq.write_table(pa.Table.from_pylist(rows), raw_path)
    unit["raw"].update(sha256=file_sha256(raw_path), bytes=raw_path.stat().st_size)
    unit["expected_file_rows"] = len(rows)
    plan["base"]["cosmopedia_policy"] = policy()
    plan["base"]["cosmopedia_policy"]["filters"]["min_document_bytes"] = 1
    with pytest.raises(RuntimeError, match="injected"):
        units.acquire_unit(plan, unit, tmp_path / "resumed", {}, interrupt_after_rows=3)
    changed = copy.deepcopy(plan)
    changed["base"]["cosmopedia_policy"]["maximum_seed_text_jaccard"] = 0.9
    with pytest.raises(ValueError, match="owner/config"):
        units.acquire_unit(changed, unit, tmp_path / "resumed", {})
    resumed = units.acquire_unit(plan, unit, tmp_path / "resumed", {})
    clean = units.acquire_unit(plan, unit, tmp_path / "clean", {})
    assert resumed["manifest"] == clean["manifest"]
    assert clean["manifest"]["retained_records"] == 3
    assert clean["manifest"]["rejections"]["synthetic_unchanged_seed"] == 1
    output = (tmp_path / "clean" / unit["id"] / "records.jsonl").read_text()
    assert "Describe this topic" not in output
    for row in map(json.loads, output.splitlines()):
        assert row["metadata"]["seed_source_label"] == "fineweb"
        assert row["content_id"] == f"example:{unit['raw']['sha256']}:{row['source_row']}"
    bad_category = copy.deepcopy(unit)
    bad_category["category"] = "web"
    with pytest.raises(ValueError, match="only govern synthetic"):
        units._unit_config(plan, bad_category)


def test_bound_plan_preserves_document_filters_and_declares_sample_cap_difference():
    root = Path(__file__).resolve().parents[2]
    plan = cosmo.load_cosmopedia_preparation(
        root / "research/flagship/cosmopedia_stock_preparation_v1.json"
    )
    original = json.loads(
        (
            root / "archive/pregrant-history/research/flagship/synthetic_cosmopedia_v2_v2.json"
        ).read_text()
    )
    filters = plan["base"]["cosmopedia_policy"]["filters"]
    for key, value in original["filters"].items():
        assert filters[key] == (
            None
            if key in {"maximum_bytes_per_template_prefix", "maximum_bytes_per_seed_domain"}
            else value
        )
    assert len(plan["units"]) == 5
    assert sum(unit["stop_row"] for unit in plan["units"]) == 1881445
    assert (
        plan["reference_tokenizer"]["sha256"]
        == "dadfd56d766715c61d2ef780a525ab43b8e6da4de6865bda3d95fdef5e134055"
    )
