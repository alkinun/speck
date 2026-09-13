import hashlib
import json
from pathlib import Path

import pytest

import speck.data.source_bank as bank
from speck.provenance.io import file_sha256


class ByteTokenizer:
    def __init__(self, path):
        self.repeats = int(Path(path).read_text())

    def encode_batch(self, texts, **kwargs):
        return [[1, *list(text.encode()) * self.repeats, 2] for text in texts]


def make_plan(tmp_path, monkeypatch, name="clean", target=70):
    monkeypatch.setattr(bank, "Tokenizer", ByteTokenizer)
    pool = tmp_path / "pool"
    pool.mkdir(exist_ok=True)
    outputs = {}
    ordered = [
        {"id": f"firewall_reference__{category}_{role}", "precedence": index + 1}
        for index, (category, role) in enumerate(
            (category, role) for category in bank.CATEGORIES for role in ("primary", "unseen")
        )
    ]
    for category in bank.CATEGORIES:
        key = f"pilot_train__{category}"
        source = pool / f"{key}.jsonl"
        records = []
        for i in range(20):
            text = f"{category} document {i}: α and β"
            records.append(
                {
                    "text": text,
                    "released_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "metadata": {"repository": "fixture", "order": i},
                }
            )
        source.write_text("".join(json.dumps(record) + "\n" for record in records))
        outputs[key] = {"path": source.name, "sha256": file_sha256(source)}
        ordered.append(
            {
                "id": key,
                "precedence": len(ordered) + 1,
                "text_field": "text",
                "content_sha256_field": "released_content_sha256",
            }
        )
    parent = pool / "manifest.json"
    parent.write_text(
        json.dumps(
            {
                "format": "speck_production_text_preprocess_result",
                "sources": ordered,
                "outputs": outputs,
                "gates": {
                    "global_exact_deduplication": "pass",
                    "disk_backed_Minhash_candidates_and_verified_near_deduplication": "pass",
                },
            }
        )
    )
    tokenizer = tmp_path / "reference.model"
    tokenizer.write_text("1")
    value = {
        "format": "speck_bounded_source_bank_plan",
        "format_version": 1,
        "purpose": "engineering_rehearsal_not_training_data",
        "parent_manifest": {"path": str(parent), "sha256": file_sha256(parent)},
        "reference_tokenizer": {"path": str(tokenizer), "sha256": file_sha256(tokenizer)},
        "sources": [
            {
                "category": category,
                "parent_source_id": f"pilot_train__{category}",
                "target_utf8_bytes": target,
            }
            for category in bank.CATEGORIES
        ],
        "checkpoint_records": 2,
        "shard_tokens": 64,
        "output_directory": str(tmp_path / name),
    }
    path = tmp_path / f"{name}-plan.json"
    path.write_text(json.dumps(value))
    return path


def edit_plan(path, edit):
    value = json.loads(path.read_text())
    edit(value)
    path.write_text(json.dumps(value))


def test_preserves_whole_document_metadata_and_reopens_identically(tmp_path, monkeypatch):
    plan = make_plan(tmp_path, monkeypatch)
    result = bank.prepare_source_bank(plan)
    for item in result["sources"]:
        path = tmp_path / "clean" / item["category"] / "selected.jsonl"
        original = tmp_path / "pool" / f"pilot_train__{item['category']}.jsonl"
        assert original.read_bytes().startswith(path.read_bytes())
        texts = [json.loads(line)["text"] for line in path.read_text().splitlines()]
        assert sum(len(text.encode()) for text in texts) >= 70
        assert sum(len(text.encode()) for text in texts[:-1]) < 70
        assert item["packing"]["reference_tokens"] == sum(len(text.encode()) + 2 for text in texts)
    reopened = bank.prepare_source_bank(plan)
    assert reopened["manifest"] == result["manifest"]
    assert reopened["sources"] == result["sources"]
    assert result["training_authority"] is False


def test_record_boundary_resume_matches_uninterrupted_selection_and_packing(tmp_path, monkeypatch):
    plan = make_plan(tmp_path, monkeypatch, "resumed", target=140)
    with pytest.raises(RuntimeError, match="injected"):
        bank.prepare_source_bank(plan, interrupt_source="code", interrupt_after_records=3)
    selected = tmp_path / "resumed/code/selected.jsonl"
    with selected.open("ab") as handle:
        handle.write(b"uncommitted torn write")
    resumed = bank.prepare_source_bank(plan)
    clean = bank.prepare_source_bank(make_plan(tmp_path, monkeypatch, target=140))
    assert resumed["sources"] == clean["sources"]


def test_packing_failure_rebuilds_only_unpublished_source_packing(tmp_path, monkeypatch):
    plan = make_plan(tmp_path, monkeypatch)

    class BrokenTokenizer(ByteTokenizer):
        def encode_batch(self, texts, **kwargs):
            yield [1, 3, 2]
            raise RuntimeError("injected packing failure")

    monkeypatch.setattr(bank, "Tokenizer", BrokenTokenizer)
    with pytest.raises(RuntimeError, match="packing failure"):
        bank.prepare_source_bank(plan)
    selected = tmp_path / "clean/web/selected.jsonl"
    before = selected.read_bytes()
    monkeypatch.setattr(bank, "Tokenizer", ByteTokenizer)
    result = bank.prepare_source_bank(plan)
    assert selected.read_bytes() == before
    assert len(result["sources"]) == 6
    assert not (tmp_path / "clean/web/packed.building").exists()


def test_byte_selection_is_independent_of_reference_tokenizer(tmp_path, monkeypatch):
    first = bank.prepare_source_bank(make_plan(tmp_path, monkeypatch))
    second_plan = make_plan(tmp_path, monkeypatch, "other-tokenizer")
    tokenizer = tmp_path / "second.model"
    tokenizer.write_text("2")
    edit_plan(
        second_plan,
        lambda value: value.update(
            reference_tokenizer={"path": str(tokenizer), "sha256": file_sha256(tokenizer)}
        ),
    )
    second = bank.prepare_source_bank(second_plan)
    assert [item["selection"] for item in first["sources"]] == [
        item["selection"] for item in second["sources"]
    ]
    assert (
        first["sources"][0]["packing"]["reference_tokens"]
        != second["sources"][0]["packing"]["reference_tokens"]
    )


@pytest.mark.parametrize("where", ["input", "selected", "shard"])
def test_corruption_is_rejected(tmp_path, monkeypatch, where):
    plan = make_plan(tmp_path, monkeypatch)
    result = bank.prepare_source_bank(plan)
    if where == "input":
        path = tmp_path / "pool/pilot_train__web.jsonl"
    elif where == "selected":
        path = tmp_path / "clean/web/selected.jsonl"
    else:
        path = tmp_path / "clean/web/packed" / result["sources"][0]["packing"]["shards"][0]["path"]
    path.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="identity|truncated|hash|differs"):
        bank.prepare_source_bank(plan)


def test_exhaustion_does_not_publish_success(tmp_path, monkeypatch):
    plan = make_plan(tmp_path, monkeypatch, target=10000)
    with pytest.raises(RuntimeError, match="exhausted"):
        bank.prepare_source_bank(plan)
    assert not (tmp_path / "clean/manifest.json").exists()


def test_reference_views_and_changed_plan_cannot_enter_bank(tmp_path, monkeypatch):
    plan = make_plan(tmp_path, monkeypatch)
    edit_plan(
        plan,
        lambda value: value["sources"][0].update(
            parent_source_id="firewall_reference__web_primary"
        ),
    )
    with pytest.raises(ValueError, match="only retained pilot training"):
        bank.prepare_source_bank(plan)
    plan = make_plan(tmp_path, monkeypatch)
    bank.prepare_source_bank(plan)
    edit_plan(plan, lambda value: value["sources"][0].update(target_utf8_bytes=71))
    with pytest.raises(ValueError, match="different plan"):
        bank.prepare_source_bank(plan)


def test_resume_rejects_corrupted_quota_counter(tmp_path, monkeypatch):
    plan = make_plan(tmp_path, monkeypatch, target=140)
    with pytest.raises(RuntimeError, match="injected"):
        bank.prepare_source_bank(plan, interrupt_source="web", interrupt_after_records=3)
    state_path = tmp_path / "clean/web/selection-state.json"
    state = json.loads(state_path.read_text())
    state["utf8_bytes"] = 10000
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="counters disagree"):
        bank.prepare_source_bank(plan)
