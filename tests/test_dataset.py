import gzip
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import speck.dataloader as dataloader
import speck.dataset as dataset
from speck.dataloader import (
    loader_state_for_offset,
    packed_loader,
    scheduled_source,
    source_selection_counts,
)


class FakeTokenizer:
    vocab_size = 32000
    bos_id = 1
    eos_id = 2

    def encode(self, text, bos=False, eos=False):
        tokens = [byte + 3 for byte in text.encode()]
        return ([1] if bos else []) + tokens + ([2] if eos else [])

    def encode_batch(self, texts, bos=False, eos=False):
        return [self.encode(text, bos, eos) for text in texts]

    def fingerprint(self):
        return "test-tokenizer"


def source_config(source_id):
    return {
        "id": source_id,
        "repo": f"test/{source_id}",
        "revision": None,
        "tree_path": "data",
        "content_column": "text",
        "metadata_columns": {"fixture": "fixture"},
        "filters": {},
    }


def settings(train_tokens=400, validation_tokens=32):
    return {
        "sources": [source_config("a"), source_config("b")],
        "mixture": {
            "phases": [
                {"end_tokens": train_tokens // 2, "weights": {"a": 75, "b": 25}},
                {"end_tokens": train_tokens, "weights": {"a": 25, "b": 75}},
            ]
        },
        "requested_train_tokens": train_tokens,
        "validation_tokens_per_source": validation_tokens,
        "validation_fraction": 0.1,
        "filtering": {"min_chars": 0, "max_chars": 10_000},
        "dedup": {
            "normalization": "NFKC+lower+whitespace",
            "hash": "blake2b-128",
            "scope": "global",
        },
        "shards": {"tokens": 37, "maximum_loader_microbatch_tokens": 0},
        "seed": 7,
    }


def single_source_settings(train_tokens=60, validation_tokens=60, max_chars=10_000):
    return {
        "sources": [source_config("a")],
        "mixture": {"phases": [{"end_tokens": train_tokens, "weights": {"a": 100}}]},
        "requested_train_tokens": train_tokens,
        "validation_tokens_per_source": validation_tokens,
        "validation_fraction": 0.1,
        "filtering": {"min_chars": 0, "max_chars": max_chars},
        "dedup": {
            "normalization": "NFKC+lower+whitespace",
            "hash": "blake2b-128",
            "scope": "global",
        },
        "shards": {"tokens": 37, "maximum_loader_microbatch_tokens": 0},
        "seed": 7,
    }


def documents(source_id, duplicate=None):
    values = [
        {
            "content": f"val-{source_id}-000-" + "v" * 30,
            "metadata": {"fixture": source_id},
        }
    ]
    if duplicate is not None:
        values.append(
            {
                "content": duplicate,
                "score": 0.9,
                "metadata": {"fixture": source_id},
            }
        )
    values.extend(
        {
            "content": f"train-{source_id}-{index:03d}-" + source_id * 30,
            "score": 1.0,
            "metadata": {"fixture": source_id},
        }
        for index in range(30)
    )
    return values


@pytest.fixture
def packed_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset,
        "_is_validation_document",
        lambda content, seed, fraction: content.startswith("val-"),
    )
    tokenizer = FakeTokenizer()
    path = tmp_path / "packed"
    manifest = dataset.prepare_dataset(
        **settings(),
        output_dir=path,
        tokenizer=tokenizer,
        check_disk=False,
        document_iterators={
            "a": documents("a", "Ａ  Foo\tBAR"),
            "b": documents("b", "a foo   bar"),
        },
    )
    return path, tokenizer, manifest


def source_values(path, source, split):
    pieces = [
        np.fromfile(path / shard["path"], dtype="<u2")
        for shard in source["splits"][split]["shards"]
    ]
    return np.concatenate(pieces)


def test_production_mixture_derives_exact_targets_and_small_reserve():
    root = Path(__file__).parents[1]
    experiment = root / "experiments" / "Speck1-140M"
    config = json.loads((experiment / "data.json").read_text(encoding="utf-8"))
    train = json.loads((experiment / "train.json").read_text(encoding="utf-8"))
    assert train["train_tokens"] == 5_000_000_000
    assert train["batch_tokens"] == 65_536
    output_dir = config.pop("output_dir")
    assert output_dir is None
    config.pop("seed")
    validated = dataset.validate_data_settings(**config)
    assert validated["quotas"] == {
        "finemath_4plus": 475_000_000,
        "ultra_fineweb": 1_975_000_000,
        "dclm": 1_550_000_000,
        "cosmopedia_v2": 670_000_000,
        "ultrafineweb_l3": 330_000_000,
    }
    assert validated["train_reserve_tokens_per_source"] == 262_144
    assert config["sources"][0]["id"] == "finemath_4plus"
    assert all("train_tokens" not in source for source in config["sources"])
    assert all("hf_config" not in source for source in config["sources"])
    cosmopedia = next(source for source in config["sources"] if source["id"] == "cosmopedia_v2")
    assert "seed_data" not in cosmopedia["metadata_columns"]
    schedule_manifest = {
        "requested_train_tokens": config["requested_train_tokens"],
        "mixture": {"phases": validated["phases"]},
        "sources": [{"id": source["id"]} for source in config["sources"]],
    }
    consumed_tokens = (
        (config["requested_train_tokens"] + train["batch_tokens"] - 1)
        // train["batch_tokens"]
        * train["batch_tokens"]
    )
    assert consumed_tokens == 5_000_003_584
    for world_size in (1, 2, 4, 8):
        stride = train["device_batch_size"] * train["sequence_length"] * world_size
        assert train["batch_tokens"] % stride == 0
        assert consumed_tokens % stride == 0
        counts = source_selection_counts(schedule_manifest, "train", consumed_tokens, stride)
        for source_id, count in counts.items():
            assert count * stride + 1 <= (
                validated["quotas"][source_id] + validated["train_reserve_tokens_per_source"]
            )
        for phase_index, phase in enumerate(validated["phases"][:-1]):
            end = phase["end_tokens"]
            before = ((end - 1) // stride) * stride
            after = ((end + stride - 1) // stride) * stride
            assert scheduled_source(schedule_manifest, "train", before, stride)[1] == phase_index
            assert scheduled_source(schedule_manifest, "train", after, stride)[1] == phase_index + 1


def test_mixture_validation_rejects_bad_phases():
    config = settings()
    config.pop("seed")
    config["mixture"]["phases"][0]["weights"]["a"] = 74
    with pytest.raises(ValueError, match="sum to 100"):
        dataset.validate_data_settings(**config)
    config = settings()
    config.pop("seed")
    config["mixture"]["phases"][-1]["end_tokens"] -= 4
    with pytest.raises(ValueError, match="final mixture phase"):
        dataset.validate_data_settings(**config)


def test_half_percentage_weights_produce_exact_quotas_and_schedule():
    sources = [source_config("a"), source_config("b")]
    mixture = {"phases": [{"end_tokens": 200, "weights": {"a": 99.5, "b": 0.5}}]}
    quotas, phases = dataset.derive_source_quotas(sources, mixture, 200)
    assert quotas == {"a": 199, "b": 1}
    manifest = {
        "requested_train_tokens": 200,
        "mixture": {"phases": phases},
        "sources": [{"id": "a"}, {"id": "b"}],
    }
    assert source_selection_counts(manifest, "train", 200, 1) == {"a": 199, "b": 1}
    cycle = dataloader._smooth_cycle((("a", 99.5), ("b", 0.5)))
    assert len(cycle) == 200
    assert cycle.count("a") == 199
    assert cycle.count("b") == 1


def test_weighted_schedule_reduces_ratios_and_bounds_exact_cycles():
    cycle = dataloader._smooth_cycle((("a", 75), ("b", 25)))
    assert len(cycle) == 4
    assert cycle.count("a") == 3
    assert cycle.count("b") == 1

    with pytest.raises(ValueError, match="cycle exceeds 100,000 batches"):
        dataloader._smooth_cycle((("a", 50.000001), ("b", 49.999999)))


def test_disk_preflight_reports_required_and_available_bytes(tmp_path):
    config = single_source_settings()
    config.pop("seed")
    validated = dataset.validate_data_settings(**config)
    estimate = dataset.estimate_disk_requirement(validated, config["requested_train_tokens"])
    assert estimate["components"]["packed_uint16_bytes"] == 240
    with pytest.raises(OSError, match="required .* available"):
        dataset.disk_preflight(
            tmp_path / "packed",
            validated,
            config["requested_train_tokens"],
            disk_usage=lambda path: SimpleNamespace(free=1),
        )


def test_repository_tree_is_revision_pinned_recursive_and_deterministic():
    class Api:
        def __init__(self):
            self.info_calls = []
            self.tree_calls = []

        def dataset_info(self, repo, revision=None):
            self.info_calls.append((repo, revision))
            return SimpleNamespace(sha="abc123")

        def list_repo_tree(self, repo, **kwargs):
            self.tree_calls.append((repo, kwargs))
            return [
                SimpleNamespace(path="data/nested/two.parquet"),
                SimpleNamespace(path="data/readme.md"),
                SimpleNamespace(path="data/one.parquet"),
            ]

    api = Api()
    source = source_config("a")
    first = dataset.discover_source_files(source, 17, api)
    second_api = Api()
    second = dataset.discover_source_files(source, 17, second_api)
    assert first == second
    assert sorted(first["files"]) == ["data/nested/two.parquet", "data/one.parquet"]
    assert api.info_calls == [("test/a", None)]
    assert api.tree_calls == [
        (
            "test/a",
            {
                "path_in_repo": "data",
                "recursive": True,
                "revision": "abc123",
                "repo_type": "dataset",
            },
        )
    ]


def test_repository_tree_verifies_explicit_gzip_jsonl_files():
    files = [
        "data/v2/train-00010-of-00020.json.gz",
        "data/v2/train-00011-of-00020.json.gz",
    ]

    class Api:
        def dataset_info(self, repo, revision=None):
            return SimpleNamespace(sha="abc123")

        def list_repo_tree(self, repo, **kwargs):
            return [
                *(SimpleNamespace(path=path) for path in files),
                SimpleNamespace(path="data/v2/train-00000-of-00020.json.gz"),
            ]

    source = {
        **source_config("papers"),
        "file_format": "jsonl_gzip",
        "files": files,
    }
    resolved = dataset.discover_source_files(source, 17, Api())
    assert sorted(resolved["files"]) == files


def test_download_uses_revision_pinned_hf_xet(tmp_path, monkeypatch):
    calls = {}

    def download(**kwargs):
        calls.update(kwargs)
        cache = Path(kwargs["cache_dir"])
        blob = cache / "blobs" / "shard"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"parquet")
        path = cache / "snapshots" / "abc123" / "data" / "shard.parquet"
        path.parent.mkdir(parents=True)
        path.symlink_to(blob)
        return str(path)

    monkeypatch.setattr(dataset, "hf_hub_download", download)
    destination = tmp_path / "raw" / "shard.parquet"
    dataset._download_file(
        "https://huggingface.co/datasets/openbmb/Ultra-FineWeb/resolve/abc123/data/shard.parquet",
        destination,
        "test shard",
    )
    assert destination.read_bytes() == b"parquet"
    assert calls == {
        "repo_id": "openbmb/Ultra-FineWeb",
        "filename": "data/shard.parquet",
        "repo_type": "dataset",
        "revision": "abc123",
        "cache_dir": destination.parent / ".shard.download",
    }


def test_download_retries_xet_runtime_errors_without_discarding_cache(tmp_path, monkeypatch):
    attempts = 0

    def download(**kwargs):
        nonlocal attempts
        attempts += 1
        cache = Path(kwargs["cache_dir"])
        partial = cache / "partial"
        if attempts == 1:
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"partial")
            raise RuntimeError("CAS Client Error: 503 Service Unavailable")
        assert partial.read_bytes() == b"partial"
        blob = cache / "blobs" / "shard"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"complete")
        return str(blob)

    monkeypatch.setattr(dataset, "hf_hub_download", download)
    monkeypatch.setattr(dataset.time, "sleep", lambda delay: None)
    destination = tmp_path / "raw" / "shard.parquet"
    dataset._download_file(
        "https://huggingface.co/datasets/example/data/resolve/abc123/data/shard.parquet",
        destination,
        "test shard",
        attempts=2,
    )
    assert attempts == 2
    assert destination.read_bytes() == b"complete"


def test_stream_reads_needed_columns_and_only_keeps_raw_when_explicit(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture.parquet"
    pq.write_table(
        pa.table({"content": ["kept text"], "score": ["0.9"], "unused": ["not requested"]}),
        fixture,
    )
    cache = tmp_path / "raw"

    def download(url, destination, description, repo=None):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixture, destination)

    monkeypatch.setattr(dataset, "_download_file", download)
    source = {
        **source_config("a"),
        "content_column": "content",
        "score_column": "score",
        "filters": {"min_score": 0.8},
    }
    rows = list(
        dataset.iter_documents(
            source=source,
            revision="abc123",
            files=["data/nested/shard.parquet"],
            filtering={"min_chars": 0, "max_chars": 100},
            cache_dir=cache,
            keep_raw=True,
        )
    )
    assert rows[0]["content"] == "kept text"
    assert rows[0]["score"] == 0.9
    assert len(list(cache.iterdir())) == 1
    list(
        dataset.iter_documents(
            source=source,
            revision="abc123",
            files=["data/nested/shard.parquet"],
            filtering={"min_chars": 0, "max_chars": 100},
            cache_dir=cache,
        )
    )
    assert list(cache.iterdir()) == []


def test_streams_gzip_jsonl_source_files(tmp_path, monkeypatch):
    fixture = tmp_path / "papers.json.gz"
    with gzip.open(fixture, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": "paper-1",
                    "source": "s2orc/train",
                    "text": "Full paper text with enough content.",
                    "version": "v2",
                }
            )
            + "\n"
        )

    def download(url, destination, description, repo=None):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixture, destination)

    monkeypatch.setattr(dataset, "_download_file", download)
    source = {
        **source_config("papers"),
        "file_format": "jsonl_gzip",
        "files": ["data/v2/train-00010-of-00020.json.gz"],
        "metadata_columns": {"id": "id", "source": "source", "version": "version"},
    }
    cache = tmp_path / "raw"
    rows = list(
        dataset.iter_documents(
            source=source,
            revision="abc123",
            files=source["files"],
            filtering={"min_chars": 0, "max_chars": 100},
            cache_dir=cache,
        )
    )
    assert rows == [
        {
            "content": "Full paper text with enough content.",
            "score": None,
            "metadata": {"id": "paper-1", "source": "s2orc/train", "version": "v2"},
            "file": "data/v2/train-00010-of-00020.json.gz",
            "row": 0,
        }
    ]
    assert list(cache.iterdir()) == []


def test_score_and_language_filter_columns_are_required(tmp_path, monkeypatch):
    fixture = tmp_path / "missing.parquet"
    pq.write_table(pa.table({"content": ["text"]}), fixture)

    def download(url, destination, description, repo=None):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixture, destination)

    monkeypatch.setattr(dataset, "_download_file", download)
    score_source = {
        **source_config("score"),
        "content_column": "content",
        "score_column": "score",
        "filters": {"min_score": 0.8},
    }
    with pytest.raises(ValueError, match="missing configured columns.*score"):
        list(
            dataset.iter_documents(
                source=score_source,
                revision="abc123",
                files=["data/score.parquet"],
                filtering={"min_chars": 0, "max_chars": 100},
                cache_dir=tmp_path / "score_raw",
            )
        )
    language_source = {
        **source_config("language"),
        "content_column": "content",
        "language_column": "language",
        "filters": {"language": "en"},
    }
    with pytest.raises(ValueError, match="missing configured columns.*language"):
        list(
            dataset.iter_documents(
                source=language_source,
                revision="abc123",
                files=["data/language.parquet"],
                filtering={"min_chars": 0, "max_chars": 100},
                cache_dir=tmp_path / "language_raw",
            )
        )


def test_source_extensions_validate_explicit_contracts():
    source = {**source_config("score"), "score_column": "score"}
    with pytest.raises(ValueError, match="score_operator requires min_score"):
        dataset._validate_source({**source, "filters": {"score_operator": ">"}})
    with pytest.raises(ValueError, match="unsupported score_operator"):
        dataset._validate_source({**source, "filters": {"min_score": 0.8, "score_operator": "=="}})
    with pytest.raises(ValueError, match="min_score must be numeric"):
        dataset._validate_source({**source, "filters": {"min_score": float("nan")}})
    with pytest.raises(ValueError, match="language_detector requires a language filter"):
        dataset._validate_source({**source_config("language"), "language_detector": "py3langid"})
    with pytest.raises(ValueError, match="cannot use both"):
        dataset._validate_source(
            {
                **source_config("language"),
                "language_column": "language",
                "language_detector": "py3langid",
                "filters": {"language": "en"},
            }
        )
    with pytest.raises(ValueError, match="invalid files"):
        dataset._validate_source(
            {
                **source_config("compressed"),
                "file_format": "jsonl_gzip",
                "files": ["data/shard.parquet"],
            }
        )


def test_strict_score_and_detected_language_filters(tmp_path, monkeypatch):
    fixture = tmp_path / "filtered.parquet"
    pq.write_table(
        pa.table(
            {
                "content": ["english equal", "english above", "chinese above", "not finite"],
                "score": ["0.8", "0.81", "0.82", "nan"],
            }
        ),
        fixture,
    )

    def download(url, destination, description, repo=None):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixture, destination)

    monkeypatch.setattr(dataset, "_download_file", download)
    monkeypatch.setattr(
        dataset,
        "_detect_language",
        lambda content, detector: "zh" if content.startswith("chinese") else "en",
    )
    base = {
        **source_config("filtered"),
        "content_column": "content",
        "score_column": "score",
        "metadata_columns": {},
    }
    inclusive = list(
        dataset.iter_documents(
            source={**base, "filters": {"min_score": 0.8}},
            revision="abc123",
            files=["data/filter.parquet"],
            filtering={"min_chars": 0, "max_chars": 100},
            cache_dir=tmp_path / "inclusive",
        )
    )
    assert [row["content"] for row in inclusive] == [
        "english equal",
        "english above",
        "chinese above",
    ]
    strict = list(
        dataset.iter_documents(
            source={
                **base,
                "language_detector": "py3langid",
                "filters": {
                    "language": "en",
                    "min_score": 0.8,
                    "score_operator": ">",
                },
            },
            revision="abc123",
            files=["data/filter.parquet"],
            filtering={"min_chars": 0, "max_chars": 100},
            cache_dir=tmp_path / "strict",
        )
    )
    assert [row["content"] for row in strict] == ["english above"]


def test_py3langid_detects_long_english_and_chinese_text():
    english = "This chapter explains algebra with detailed examples and careful reasoning. " * 8
    chinese = "本章通过详细的例子和严谨的推理来解释代数的基本概念。" * 8
    assert dataset._detect_language(english, "py3langid") == "en"
    assert dataset._detect_language(chinese, "py3langid") == "zh"


def test_resolve_data_dir_preserves_default_and_supports_isolated_names(tmp_path):
    assert dataset.resolve_data_dir() == dataset.default_data_dir / "packed"
    assert dataset.resolve_data_dir(output_name="Speck1.5-140M") == (
        dataset.default_data_dir / "Speck1.5-140M"
    )
    assert dataset.resolve_data_dir(tmp_path / "explicit", "ignored") == tmp_path / "explicit"
    with pytest.raises(ValueError, match="single non-empty path component"):
        dataset.resolve_data_dir(output_name="nested/name")


def test_prepare_writes_nested_source_manifest_and_normalized_global_dedup(
    packed_dataset,
):
    path, _, manifest = packed_dataset
    assert manifest["format_version"] == 3
    assert manifest["requested_train_tokens"] == 400
    assert manifest["mixture"]["source_quotas"] == {"a": 200, "b": 200}
    assert manifest["splits"]["train"]["tokens"] == sum(
        source["splits"]["train"]["tokens"] for source in manifest["sources"]
    )
    for source in manifest["sources"]:
        assert source["revision"] == "injected"
        assert source["splits"]["train"]["requested_tokens"] == 200
        assert source["splits"]["val"]["requested_tokens"] == 32
        assert source["splits"]["train"]["tokens"] >= 200
        assert source["splits"]["val"]["tokens"] >= 32
        assert (path / source["document_index"]["path"]).is_file()
        for split in ("train", "val"):
            assert all(
                shard["path"].startswith(f"sources/{source['id']}/")
                for shard in source["splits"][split]["shards"]
            )

    duplicate_hash = dataset.dedup_hash("a foo bar").hex()
    records = []
    for source in manifest["sources"]:
        records.extend(
            json.loads(line)
            for line in (path / source["document_index"]["path"])
            .read_text(encoding="utf-8")
            .splitlines()
        )
    matching = [record for record in records if record["dedup_hash"] == duplicate_hash]
    assert len(matching) == 1
    assert matching[0]["source_id"] == "a"
    assert manifest["dedup"]["accepted_hashes"] == manifest["documents"]
    assert (path / manifest["dedup"]["path"]).stat().st_size == manifest["documents"] * 16
    for source in manifest["sources"]:
        journal = source["dedup_journal"]
        assert journal["hashes"] == source["documents"]
        assert journal["end_byte"] - journal["start_byte"] == source["documents"] * 16


def test_full_preferred_split_discards_instead_of_rerouting(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset,
        "_is_validation_document",
        lambda content, seed, fraction: content.startswith("val-"),
    )
    values = [
        {"content": "train-kept-" + "a" * 30},
        {"content": "train-discarded-" + "b" * 30},
        {"content": "val-kept-" + "c" * 30},
    ]
    path = tmp_path / "partition"
    manifest = dataset.prepare_dataset(
        **single_source_settings(train_tokens=20, validation_tokens=20),
        output_dir=path,
        tokenizer=FakeTokenizer(),
        check_disk=False,
        document_iterators={"a": values},
    )
    source = manifest["sources"][0]
    records = [
        json.loads(line)
        for line in (path / source["document_index"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["split"] for record in records] == ["train", "val"]
    discarded = hashlib.sha256(values[1]["content"].encode()).hexdigest()
    assert discarded not in {record["content_hash"] for record in records}


def test_tokenizer_batches_are_bounded_by_documents_and_characters(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset,
        "_is_validation_document",
        lambda content, seed, fraction: content.startswith("val-"),
    )

    class TrackingTokenizer(FakeTokenizer):
        def __init__(self):
            self.calls = []

        def encode_batch(self, texts, bos=False, eos=False):
            self.calls.append((len(texts), sum(map(len, texts))))
            return [[self.bos_id, 3, self.eos_id] for _ in texts]

    tokenizer = TrackingTokenizer()
    values = []
    for index in range(20):
        for split in ("train", "val"):
            prefix = f"{split}-{index:03d}-"
            values.append({"content": prefix + "x" * (100_000 - len(prefix))})
    dataset.prepare_dataset(
        **single_source_settings(train_tokens=60, validation_tokens=60, max_chars=100_000),
        output_dir=tmp_path / "bounded",
        tokenizer=tokenizer,
        check_disk=False,
        document_iterators={"a": values},
    )
    assert len(tokenizer.calls) == 2
    assert all(documents <= 1024 for documents, _ in tokenizer.calls)
    assert all(characters <= 2_000_000 for _, characters in tokenizer.calls)


def test_deterministic_phased_selection_and_source_separation(packed_dataset):
    path, tokenizer, manifest = packed_dataset
    selected = [scheduled_source(manifest, "train", offset, 4)[0] for offset in range(0, 400, 4)]
    assert selected[:8] == ["a", "a", "b", "a", "a", "a", "b", "a"]
    assert selected[:50].count("a") == 38
    assert selected[:50].count("b") == 12
    assert selected[50:].count("a") == 13
    assert selected[50:].count("b") == 37
    assert selected == [
        scheduled_source(manifest, "train", offset, 4)[0] for offset in range(0, 400, 4)
    ]

    loader = packed_loader(tokenizer, 1, 4, "train", device="cpu", data_dir=path)
    inputs, targets, state = next(loader)
    source = next(value for value in manifest["sources"] if value["id"] == state["selected_source"])
    values = source_values(path, source, "train")
    offset = state["source_offsets"][state["selected_source"]]
    assert inputs.flatten().tolist() == values[offset : offset + 4].tolist()
    assert targets.flatten().tolist() == values[offset + 1 : offset + 5].tolist()
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_exact_resume_before_at_and_after_phase_change(packed_dataset):
    path, tokenizer, manifest = packed_dataset
    loader = packed_loader(tokenizer, 1, 4, "train", device="cpu", data_dir=path)
    batches = {}
    offsets = {196, 200, 204, 396, 400, 404, 800, 804}
    for _ in range(202):
        batch = next(loader)
        if batch[2]["global_consumed_tokens"] in offsets:
            batches[batch[2]["global_consumed_tokens"]] = batch
    assert batches[196][2]["phase"] == 0
    assert batches[200][2]["phase"] == 1
    assert batches[396][2]["phase"] == 1
    assert batches[400][2]["phase"] == 1
    for offset, expected in batches.items():
        state = loader_state_for_offset(manifest, "train", offset, 4, 1)
        resumed = packed_loader(
            tokenizer,
            1,
            4,
            "train",
            device="cpu",
            data_dir=path,
            resume_state_dict=state,
        )
        actual = next(resumed)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
        assert actual[2] == expected[2] == state

    tampered = dict(batches[200][2])
    tampered["selected_source"] = "b" if tampered["selected_source"] == "a" else "a"
    with pytest.raises(ValueError, match="selected_source"):
        next(
            packed_loader(
                tokenizer,
                1,
                4,
                "train",
                device="cpu",
                data_dir=path,
                resume_state_dict=tampered,
            )
        )


def test_validation_schedule_is_equal_and_absolute_offsets_are_disjoint(packed_dataset):
    path, tokenizer, manifest = packed_dataset
    selected = [scheduled_source(manifest, "val", offset, 4)[0] for offset in range(0, 40, 4)]
    assert selected == ["a", "b"] * 5
    monitor_end = loader_state_for_offset(manifest, "val", 16, 4, 1)
    assert monitor_end["selected_source"] == "a"
    assert monitor_end["source_offsets"] == {"a": 8, "b": 8}
    loader = packed_loader(
        tokenizer,
        1,
        4,
        "val",
        device="cpu",
        data_dir=path,
        resume_state_dict=monitor_end,
    )
    assert next(loader)[2] == monitor_end


def test_distributed_ranks_agree_on_source_and_state_and_read_rank_slices(
    packed_dataset, monkeypatch
):
    path, tokenizer, manifest = packed_dataset
    monkeypatch.setattr(dataloader, "dist_info", lambda: (0, 0, 2))
    rank0 = packed_loader(tokenizer, 1, 4, "train", device="cpu", data_dir=path)
    inputs0, _, state0 = next(rank0)
    monkeypatch.setattr(dataloader, "dist_info", lambda: (1, 1, 2))
    rank1 = packed_loader(tokenizer, 1, 4, "train", device="cpu", data_dir=path)
    inputs1, _, state1 = next(rank1)
    assert state0 == state1
    source = next(
        value for value in manifest["sources"] if value["id"] == state0["selected_source"]
    )
    values = source_values(path, source, "train")
    offset = state0["source_offsets"][state0["selected_source"]]
    assert inputs0.flatten().tolist() == values[offset : offset + 4].tolist()
    assert inputs1.flatten().tolist() == values[offset + 4 : offset + 8].tolist()
    for _ in range(30):
        _, _, state0 = next(rank0)
        _, _, state1 = next(rank1)
        assert state0 == state1


def test_ddp_resume_is_exact_around_unaligned_phase_boundary(packed_dataset, monkeypatch):
    path, tokenizer, manifest = packed_dataset
    for offset, phase in ((192, 0), (208, 1)):
        state = loader_state_for_offset(manifest, "train", offset, 4, 1, world_size=4)
        assert state["phase"] == phase
        rank_states = []
        for rank in (0, 3):
            monkeypatch.setattr(dataloader, "dist_info", lambda rank=rank: (rank, rank, 4))
            loader = packed_loader(
                tokenizer,
                1,
                4,
                "train",
                device="cpu",
                data_dir=path,
                resume_state_dict=state,
            )
            rank_states.append(next(loader)[2])
        assert rank_states == [state, state]


def test_manifest_rejects_a_different_tokenizer(packed_dataset):
    path, tokenizer, _ = packed_dataset

    class OtherTokenizer(FakeTokenizer):
        def fingerprint(self):
            return "other-tokenizer"

    with pytest.raises(ValueError, match="different tokenizer"):
        next(packed_loader(OtherTokenizer(), 1, 4, "train", device="cpu", data_dir=path))
    assert tokenizer.fingerprint() == "test-tokenizer"


def test_shard_checksum_detects_nested_corruption(packed_dataset):
    path, _, manifest = packed_dataset
    shard = manifest["sources"][0]["splits"]["train"]["shards"][0]
    shard_path = path / shard["path"]
    with shard_path.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))
    with pytest.raises(ValueError, match="checksum mismatch"):
        dataset.verify_shards(path, manifest)


def test_remote_source_resumes_at_completed_parquet_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset,
        "_is_validation_document",
        lambda content, seed, fraction: content.startswith("val-"),
    )
    source = source_config("a")
    files = ["data/one.parquet", "data/two.parquet"]

    class Api:
        def dataset_info(self, repo, revision=None):
            return SimpleNamespace(sha="abc123")

        def list_repo_tree(self, repo, **kwargs):
            return [SimpleNamespace(path=path) for path in files]

    order = dataset.discover_source_files(source, 7, Api())["files"]
    fixtures = {}
    first_rows = ["val-first-" + "v" * 99_990]
    first_rows.extend(f"first-{index:02d}-" + "a" * 99_991 for index in range(10))
    second_rows = [f"second-ok-{index:02d}-" + "b" * 99_987 for index in range(20)]
    second_rows.extend(f"crash-{index:02d}-" + "c" * 99_991 for index in range(11))
    for filename, rows in zip(order, (first_rows, second_rows)):
        path = tmp_path / Path(filename).name
        pq.write_table(pa.table({"text": rows}), path)
        fixtures[filename] = path

    downloads = []

    def download(url, destination, description, repo=None):
        filename = next(filename for filename in fixtures if url.endswith(filename))
        downloads.append(filename)
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixtures[filename], destination)

    monkeypatch.setattr(dataset, "_download_file", download)

    class CrashingTokenizer(FakeTokenizer):
        def __init__(self, crash):
            self.crash = crash
            self.texts = []

        def encode_batch(self, texts, bos=False, eos=False):
            self.texts.extend(texts)
            if self.crash and any(text.startswith("crash-") for text in texts):
                raise RuntimeError("injected tokenizer crash")
            return [[self.bos_id, 3, self.eos_id] for _ in texts]

    config = single_source_settings(
        train_tokens=120,
        validation_tokens=3,
        max_chars=100_000,
    )
    output = tmp_path / "file-resume"
    first_tokenizer = CrashingTokenizer(True)
    with pytest.raises(RuntimeError, match="injected tokenizer crash"):
        dataset.prepare_dataset(
            **config,
            output_dir=output,
            tokenizer=first_tokenizer,
            api=Api(),
            check_disk=False,
        )

    source_stage = tmp_path / "file-resume.building" / "sources" / "a.building"
    progress = json.loads((source_stage / "source_progress.json").read_text(encoding="utf-8"))
    assert progress["next_file_index"] == 1
    assert progress["next_file_path"] == order[1]
    assert progress["document_index"]["records"] == 11
    assert progress["dedup_journal"]["hashes"] == 11
    committed = {
        shard["path"]: dataset._file_hash(source_stage / shard["path"])
        for split in progress["splits"].values()
        for shard in split["shards"]
    }
    assert downloads == order

    resumed_tokenizer = CrashingTokenizer(False)
    manifest = dataset.prepare_dataset(
        **config,
        output_dir=output,
        tokenizer=resumed_tokenizer,
        api=Api(),
        check_disk=False,
    )
    assert downloads == [order[0], order[1], order[1]]
    assert not any(text.startswith(("first-", "val-first-")) for text in resumed_tokenizer.texts)
    for name, checksum in committed.items():
        assert dataset._file_hash(output / "sources" / "a" / name) == checksum
    assert manifest["sources"][0]["documents"] == 41
    assert manifest["sources"][0]["files_completed"] == 1
    assert manifest["sources"][0]["final_file"] == order[1]


def test_staged_dedup_slice_integrity_is_enforced(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    (directory / "documents.jsonl").write_bytes(b"")
    dedup_path = tmp_path / "dedup.bin"
    dedup_path.write_bytes(b"x" * 16)
    progress = {
        "format_version": 1,
        "source_id": "a",
        "file_list_sha256": "files",
        "next_file_index": 0,
        "next_file_path": "data/file.parquet",
        "splits": {
            "train": {"tokens": 0, "documents": 1, "shards": []},
            "val": {"tokens": 0, "documents": 0, "shards": []},
        },
        "document_index": {
            "path": "documents.jsonl",
            "bytes": 0,
            "records": 1,
            "sha256": hashlib.sha256(b"").hexdigest(),
        },
        "dedup_journal": {
            "start_byte": 0,
            "end_byte": 16,
            "hashes": 1,
            "sha256": hashlib.sha256(b"wrong").hexdigest(),
        },
    }
    with pytest.raises(ValueError, match="dedup slice checksum"):
        dataset._recover_source_progress(
            directory,
            progress,
            {"files": ["data/file.parquet"], "file_list_sha256": "files"},
            dedup_path,
            0,
        )


def test_completed_source_is_reused_when_staged_build_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset,
        "_is_validation_document",
        lambda content, seed, fraction: content.startswith("val-"),
    )
    config = settings(train_tokens=80, validation_tokens=20)
    output = tmp_path / "retry"
    with pytest.raises(RuntimeError, match="source b was exhausted"):
        dataset.prepare_dataset(
            **config,
            output_dir=output,
            tokenizer=FakeTokenizer(),
            check_disk=False,
            document_iterators={"a": documents("a"), "b": [{"content": "train-b-only"}]},
        )
    assert (tmp_path / "retry.building" / "sources" / "a" / "source.json").is_file()

    class MustNotIterate:
        def __iter__(self):
            raise AssertionError("completed source a was rebuilt")

    manifest = dataset.prepare_dataset(
        **config,
        output_dir=output,
        tokenizer=FakeTokenizer(),
        check_disk=False,
        document_iterators={"a": MustNotIterate(), "b": documents("b")},
    )
    assert [source["id"] for source in manifest["sources"]] == ["a", "b"]
    assert output.is_dir()
