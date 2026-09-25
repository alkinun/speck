"""An opt-in per-sequence mixture schedule makes data order independent of batch geometry."""

import random
from collections import Counter

import numpy as np
import pytest
import torch

import speck.data.dataset as dataset
import speck.data.loader as dataloader
from speck.data.loader import (
    loader_state_for_offset,
    packed_loader,
    scheduled_source,
    sequence_schedule,
    source_selection_counts,
)
from tests.data.test_dataset import (
    FakeTokenizer,
    documents,
    settings,
    source_config,
    source_values,
)

LENGTH = 4
BATCH_TOKENS = 32
GEOMETRIES = [
    (batch_size, world_size)
    for batch_size in (1, 2, 4, 8)
    for world_size in (1, 2, 4)
    if BATCH_TOKENS % (batch_size * world_size * LENGTH) == 0
]


def sequence_settings(length=LENGTH, **overrides):
    config = settings(**overrides)
    config["schedule"] = {"version": 1, "unit": "sequence", "sequence_length": length}
    return config


@pytest.fixture
def sequence_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset,
        "_is_validation_document",
        lambda content, seed, fraction: content.startswith("val-"),
    )
    tokenizer = FakeTokenizer()
    path = tmp_path / "packed"
    manifest = dataset.prepare_dataset(
        **sequence_settings(),
        output_dir=path,
        tokenizer=tokenizer,
        check_disk=False,
        document_iterators={"a": documents("a"), "b": documents("b")},
    )
    return path, tokenizer, manifest


def schedule_manifest(phases, requested, source_ids, length):
    return {
        "requested_train_tokens": requested,
        "mixture": {
            "phases": phases,
            "schedule": {"version": 1, "unit": "sequence", "sequence_length": length},
        },
        "sources": [{"id": source_id} for source_id in source_ids],
    }


def geometry_steps(
    monkeypatch, path, tokenizer, batch_size, world_size, steps, resume_state_dict=None
):
    """Return each optimizer step's rows in microbatch, rank, and row order, with rank-0 states."""

    microbatches = steps * BATCH_TOKENS // (batch_size * world_size * LENGTH)
    by_rank = []
    for rank in range(world_size):
        monkeypatch.setattr(dataloader, "dist_info", lambda rank=rank: (rank, rank, world_size))
        loader = packed_loader(
            tokenizer,
            batch_size,
            LENGTH,
            device="cpu",
            data_dir=path,
            resume_state_dict=resume_state_dict,
        )
        by_rank.append([next(loader) for _ in range(microbatches)])
    rows = []
    for index in range(microbatches):
        for batches in by_rank:
            inputs, targets = batches[index][:2]
            assert torch.equal(inputs[:, 1:], targets[:, :-1])
            rows.extend(torch.cat((inputs, targets[:, -1:]), 1).tolist())
        states = [batches[index][2] for batches in by_rank]
        assert all(state == states[0] for state in states)
    step_rows = BATCH_TOKENS // LENGTH
    steps_rows = [rows[step * step_rows : (step + 1) * step_rows] for step in range(steps)]
    return steps_rows, [batch[2] for batch in by_rank[0]]


def expected_steps(path, manifest, first_step, steps):
    values = {source["id"]: source_values(path, source, "train") for source in manifest["sources"]}
    step_rows = BATCH_TOKENS // LENGTH
    return [
        [
            values[source_id][cursor * LENGTH : (cursor + 1) * LENGTH + 1].tolist()
            for source_id, cursor in sequence_schedule(manifest, step * step_rows, step_rows)
        ]
        for step in range(first_step, first_step + steps)
    ]


def test_schedule_is_validated_and_recorded_only_when_requested(sequence_dataset):
    path, _, manifest = sequence_dataset
    assert manifest["mixture"]["schedule"] == {
        "version": 1,
        "unit": "sequence",
        "sequence_length": LENGTH,
    }
    assert manifest["preparation"]["train_reserve_tokens_per_source"] == 2 * 2 * LENGTH
    assert manifest["preparation"]["reserve_basis"] == "2 * phase_count * schedule.sequence_length"
    assert dataset.load_manifest(path) == manifest

    config = settings()
    config.pop("seed")
    assert dataset.validate_data_settings(**config)["schedule"] is None
    for schedule in (
        "sequence",
        {"version": 2, "unit": "sequence", "sequence_length": 4},
        {"version": True, "unit": "sequence", "sequence_length": 4},
        {"version": 1, "unit": "batch", "sequence_length": 4},
        {"version": 1, "unit": "sequence", "sequence_length": 0},
        {"version": 1, "unit": "sequence"},
    ):
        with pytest.raises(ValueError, match="schedule"):
            dataset.validate_data_settings(**config, schedule=schedule)


def test_default_manifests_keep_the_per_microbatch_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset,
        "_is_validation_document",
        lambda content, seed, fraction: content.startswith("val-"),
    )
    manifest = dataset.prepare_dataset(
        **settings(),
        output_dir=tmp_path / "packed",
        tokenizer=FakeTokenizer(),
        check_disk=False,
        document_iterators={"a": documents("a"), "b": documents("b")},
    )
    assert set(manifest["mixture"]) == {"phases", "source_quotas"}
    assert manifest["preparation"]["reserve_basis"] == (
        "(phase_count + 1) * maximum_loader_microbatch_tokens"
    )
    config = settings()
    config.pop("seed")
    config["shards"]["maximum_loader_microbatch_tokens"] = 64
    assert dataset.validate_data_settings(**config)["train_reserve_tokens_per_source"] == 192


@pytest.mark.parametrize("batch_size,world_size", GEOMETRIES)
def test_optimizer_steps_read_the_same_sequences_at_every_geometry(
    sequence_dataset, monkeypatch, batch_size, world_size
):
    path, tokenizer, manifest = sequence_dataset
    steps = manifest["requested_train_tokens"] // BATCH_TOKENS
    actual, states = geometry_steps(monkeypatch, path, tokenizer, batch_size, world_size, steps)
    assert actual == expected_steps(path, manifest, 0, steps)
    stride = batch_size * world_size * LENGTH
    for index, state in enumerate(states):
        expected = loader_state_for_offset(
            manifest, "train", index * stride, LENGTH, batch_size, world_size
        )
        assert state == expected
        # Cursors and diagnostics depend only on the global token offset.
        reference = loader_state_for_offset(manifest, "train", index * stride, LENGTH, 1)
        assert {**state, "batch_size": 1, "world_size": 1} == reference


def test_sequence_cursors_count_exact_mixture_shares_and_switch_at_token_endpoints():
    weights = {"a": 35, "b": 20, "c": 15, "d": 10, "e": 10, "f": 5, "g": 5}
    second = {"a": 5, "b": 5, "c": 10, "d": 10, "e": 15, "f": 20, "g": 35}
    length = 3
    phases = [{"end_tokens": 200, "weights": weights}, {"end_tokens": 800, "weights": second}]
    manifest = schedule_manifest(phases, 800, list(weights), length)
    rows = sequence_schedule(manifest, 0, 400)
    # Sequence 66 starts at token 198 in the first phase; sequence 67 starts at 201.
    assert scheduled_source(manifest, "train", 66 * length, length)[1] == 0
    assert scheduled_source(manifest, "train", 67 * length, length)[1] == 1
    first = Counter(source_id for source_id, _ in rows[:60])
    assert first == {source_id: 60 * weight // 100 for source_id, weight in weights.items()}
    later = Counter(source_id for source_id, _ in rows[67 : 67 + 200])
    assert later == {source_id: 2 * weight for source_id, weight in second.items()}
    cursors = Counter()
    for sequence, (source_id, cursor) in enumerate(rows):
        assert cursor == cursors[source_id]
        cursors[source_id] += 1
        if sequence % 37 == 0:
            assert source_selection_counts(manifest, "train", (sequence + 1) * length, length) == {
                source_id: cursors[source_id] for source_id in weights
            }
    assert sequence_schedule(manifest, 123, 50) == rows[123:173]


def test_resume_matches_an_uninterrupted_run_across_geometry_changes(sequence_dataset, monkeypatch):
    path, tokenizer, manifest = sequence_dataset
    steps = manifest["requested_train_tokens"] // BATCH_TOKENS
    uninterrupted, states = geometry_steps(monkeypatch, path, tokenizer, 2, 1, steps)
    for resume_step in (3, 6, 7):
        state = states[resume_step * BATCH_TOKENS // (2 * LENGTH)]
        assert state["global_consumed_tokens"] == resume_step * BATCH_TOKENS
        for batch_size, world_size in ((2, 1), (8, 1), (1, 4), (4, 2)):
            resumed, resumed_states = geometry_steps(
                monkeypatch,
                path,
                tokenizer,
                batch_size,
                world_size,
                steps - resume_step,
                resume_state_dict=state,
            )
            assert resumed == uninterrupted[resume_step:]
            assert {**resumed_states[0], "batch_size": 2, "world_size": 1} == state

    state = states[7 * BATCH_TOKENS // (2 * LENGTH)]
    with pytest.raises(ValueError, match="source_offsets"):
        next(
            packed_loader(
                tokenizer,
                2,
                LENGTH,
                device="cpu",
                data_dir=path,
                resume_state_dict={**state, "source_offsets": {"a": 0, "b": 0}},
            )
        )
    with pytest.raises(ValueError, match="4-token sequences"):
        next(packed_loader(tokenizer, 1, 2 * LENGTH, device="cpu", data_dir=path))


def test_phase_boundary_splits_a_microbatch_by_sequence(sequence_dataset):
    _, _, manifest = sequence_dataset
    # The phase ends at token 200, sequence 50; a microbatch of eight starts at 192.
    state = loader_state_for_offset(manifest, "train", 192, LENGTH, 8)
    assert state["phase"] == 0
    phases = [
        scheduled_source(manifest, "train", sequence * LENGTH, LENGTH)[1]
        for sequence in range(48, 56)
    ]
    assert phases == [0, 0, 1, 1, 1, 1, 1, 1]
    sources = [source_id for source_id, _ in sequence_schedule(manifest, 0, 100)]
    assert Counter(sources[:48]) == {"a": 36, "b": 12}
    assert Counter(sources[50:98]) == {"a": 12, "b": 36}


def test_validation_keeps_the_per_microbatch_round_robin(sequence_dataset):
    path, tokenizer, manifest = sequence_dataset
    loader = packed_loader(tokenizer, 2, LENGTH, "val", device="cpu", data_dir=path)
    selected = [next(loader)[2]["selected_source"] for _ in range(4)]
    assert selected == ["a", "b", "a", "b"]
    assert loader_state_for_offset(manifest, "val", 16, LENGTH, 2)["source_offsets"] == {
        "a": 8,
        "b": 8,
    }


def test_sequence_reserve_covers_every_scheduled_sequence():
    random.seed(0)
    for _ in range(300):
        source_ids = ["a", "b", "c", "d"][: random.randint(1, 4)]
        length = random.randint(1, 50)
        ends = sorted(random.sample(range(1, 4000), random.randint(1, 4)))
        ends = [end * 100 for end in ends]
        phases = []
        for end in ends:
            cuts = sorted(random.randint(0, 100) for _ in source_ids[1:])
            shares = [high - low for low, high in zip([0, *cuts], [*cuts, 100], strict=True)]
            phases.append(
                {"end_tokens": end, "weights": dict(zip(source_ids, shares, strict=True))}
            )
        config = settings(train_tokens=ends[-1])
        config.pop("seed")
        config["sources"] = [source_config(source_id) for source_id in source_ids]
        config["mixture"] = {"phases": phases}
        config["schedule"] = {"version": 1, "unit": "sequence", "sequence_length": length}
        validated = dataset.validate_data_settings(**config)
        manifest = schedule_manifest(validated["phases"], ends[-1], source_ids, length)
        final = -(-ends[-1] // length) * length
        counts = source_selection_counts(manifest, "train", final, length)
        for source_id, count in counts.items():
            assert count * length + (1 if count else 0) <= (
                validated["quotas"][source_id] + validated["train_reserve_tokens_per_source"]
            )


def test_prepared_reserve_serves_the_schedule_and_wraps_only_after_it(
    sequence_dataset, monkeypatch
):
    path, tokenizer, manifest = sequence_dataset
    for source in manifest["sources"]:
        train = source["splits"]["train"]
        assert train["tokens"] >= train["requested_tokens"] + train["reserve_tokens"]
    loader = packed_loader(tokenizer, 2, LENGTH, device="cpu", data_dir=path)
    sequences = manifest["requested_train_tokens"] // LENGTH
    states = [next(loader)[2] for _ in range(sequences // 2)]
    assert not any(any(state["source_epochs"].values()) for state in states)
    # Past the finite schedule each source repeats from its start rather than failing.
    later = [next(loader)[2] for _ in range(sequences)]
    assert all(any(state["source_epochs"].values()) for state in later[-4:])

    starved = dict(manifest)
    tight = [dict(source) for source in manifest["sources"]]
    tight[0]["splits"] = {**tight[0]["splits"], "train": dict(tight[0]["splits"]["train"])}
    tight[0]["splits"]["train"]["tokens"] = 100
    starved["sources"] = tight
    with pytest.raises(ValueError, match="too small for the training schedule"):
        dataloader._validate_training_capacity(starved, LENGTH)


def test_masked_and_unmasked_sources_share_a_microbatch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset,
        "_is_validation_document",
        lambda content, seed, fraction: content.startswith("val-"),
    )
    length = 64
    config = sequence_settings(length=length, train_tokens=16 * length)
    config["mixture"] = {"phases": [{"end_tokens": 16 * length, "weights": {"a": 50, "b": 50}}]}
    config["sources"][1]["packing"] = {"row_tokens": length, "open_rows": 2}
    records = [{"content": f"val-{index}-" + "v" * 20} for index in range(4)]
    records += [{"content": f"record-{index:02d}-" + "y" * 30} for index in range(40)]
    path = tmp_path / "mixed"
    manifest = dataset.prepare_dataset(
        **config,
        output_dir=path,
        tokenizer=FakeTokenizer(),
        check_disk=False,
        document_iterators={"a": documents("a"), "b": records},
    )
    inputs, targets, _ = next(
        packed_loader(FakeTokenizer(), 4, length, device="cpu", data_dir=path)
    )
    sources = [source_id for source_id, _ in sequence_schedule(manifest, 0, 4)]
    assert set(sources) == {"a", "b"}
    for row, source_id in enumerate(sources):
        source = next(value for value in manifest["sources"] if value["id"] == source_id)
        cursor = sources[:row].count(source_id)
        values = source_values(path, source, "train").astype(np.int64)
        window = slice(cursor * length + 1, (cursor + 1) * length + 1)
        assert inputs[row].tolist() == values[cursor * length : (cursor + 1) * length].tolist()
        expected = values[window]
        if source_id == "b":
            shards = source["splits"]["train"]["mask_shards"]
            mask = np.concatenate([np.fromfile(path / shard["path"], "u1") for shard in shards])
            expected = np.where(mask[window] == 0, -100, expected)
            assert (expected == -100).any()
        assert targets[row].tolist() == expected.tolist()
