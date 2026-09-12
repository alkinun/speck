import json

import pyarrow.parquet as pq
import pytest

from scripts import instruct_prepare as prepare


class FixtureTokenizer:
    newline_ids = ()

    def metadata(self):
        return {"fingerprint": "fixture"}

    def encode_messages(self, messages, add_generation_prompt=False):
        prepare.validate_messages(messages, add_generation_prompt)
        size = 1 + sum(len(message["content"]) + 4 for message in messages)
        size += 4 * add_generation_prompt
        return [1] * size, [False] * size


def source(key="fixture", **changes):
    return {
        "id": key,
        "repo": f"fixture/{key}",
        "revision": "a" * 40,
        "split": "train",
        "weight": 1,
        "kind": "messages",
        "skill": "general",
        **changes,
    }


def conversation(index, **changes):
    return {
        "id": str(index),
        "messages": [
            {"role": "user", "content": f"Explain topic {index}."},
            {"role": "assistant", "content": f"An explanation of topic {index}."},
        ],
        **changes,
    }


def recipe(**changes):
    return {
        **prepare.load_recipe(prepare.DEFAULT_RECIPE),
        "samples": 8,
        "validation_samples": 2,
        "sources": [source()],
        "max_source_rows": 100,
        **changes,
    }


def test_starter_recipe_and_exact_rescaling():
    value = prepare.load_recipe(prepare.DEFAULT_RECIPE)
    assert len(value["sources"]) == 17
    assert sum(prepare.quotas(value["sources"], 100000).values()) == 100000
    for size in (1, 17, 103, 1001):
        counts = prepare.quotas(value["sources"], size)
        assert sum(counts.values()) == size
        assert counts == prepare.quotas(value["sources"], size)
    assert prepare.load_recipe(prepare.DEFAULT_RECIPE, samples=1000)["validation_samples"] == 20


@pytest.mark.parametrize(
    "change",
    [
        {"samples": 0},
        {"samples": True},
        {"validation_samples": -1},
        {"near_duplicate_threshold": float("nan")},
        {"sources": [source(revision="main")]},
        {"sources": [source(), source()]},
        {"sources": [source(kind="unknown")]},
        {"sources": [source("train")]},
        {"sources": [source(typo="unsupported")]},
        {"sources": [source(validation_weight=-1)]},
        {"sources": [source(validation_weight=True)]},
        {"sources": [source(validation_weight=0)]},
    ],
)
def test_invalid_recipes_fail_before_download(tmp_path, change):
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe(**change)))
    with pytest.raises(ValueError):
        prepare.load_recipe(path)


def test_normalization_preserves_system_and_omits_separate_reasoning():
    row = {
        "conversations": [
            {"from": "system", "value": "  Answer briefly.  "},
            {"from": "human", "value": "A\r\nquestion"},
            {"from": "gpt", "value": "The answer.", "reasoning_content": "Not a target"},
        ]
    }
    messages = prepare.adapt(row, source(messages_field="conversations"))
    assert messages[0] == {"role": "system", "content": "Answer briefly."}
    assert messages[1]["content"] == "A\nquestion"
    assert messages[-1] == {"role": "assistant", "content": "The answer."}


def test_edit_targets_only_the_final_edited_response():
    context = conversation(1)["messages"] + [{"role": "user", "content": "Clarify that."}]
    row = {"context": context, "edited_response": "Improved answer."}
    item = prepare.candidate(row, source(kind="edit"), 0, FixtureTokenizer(), lambda _: True)
    assert item["prompt"] == context
    assert item["completion"] == [{"role": "assistant", "content": "Improved answer."}]
    assert item["turns"] == 2
    assert item["target_tokens"] == len("Improved answer.")


@pytest.mark.parametrize(
    "row",
    [
        {"messages": [{"role": "user", "content": None}, {"role": "assistant", "content": "A"}]},
        {"messages": [{"role": "user", "content": "Q"}, {"role": "tool", "content": "A"}]},
        conversation(1, metadata={"train_turns": [False, False]}),
        {
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "<think>hidden</think>A"},
            ]
        },
    ],
)
def test_invalid_or_unsupported_conversations_are_rejected(row):
    with pytest.raises(ValueError):
        prepare.adapt(row, source())


def test_source_filters_and_science_replay():
    spec = source(include={"language": ["EN"]}, minimum={"reward": 0})
    with pytest.raises(ValueError, match="source filter"):
        prepare.adapt(conversation(1, language="FR", reward=2), spec)
    with pytest.raises(ValueError, match="source score"):
        prepare.adapt(conversation(1, language="EN", reward=-1), spec)
    with pytest.raises(ValueError, match="non-science"):
        prepare.adapt(conversation(1, dataset="oasst1"), source(kind="science"))
    assert prepare.adapt(conversation(1, dataset="science.qasper"), source(kind="science"))


@pytest.mark.parametrize(
    "prompt,response,reason",
    [
        ("Use [keyword] [frequency] times.", "Some answer.", "unresolved"),
        ("Use all lowercase letters.", "Incorrect capital.", "lowercase"),
        ("Write a summary with no commas.", "First, this.", "comma"),
        ("Your response should contain less than 3 words.", "One two three", "word-count"),
    ],
)
def test_observed_constraint_failures_are_rejected(prompt, response, reason):
    with pytest.raises(ValueError, match=reason):
        prepare.check_simple_constraints(prompt, response)


def test_literal_placeholders_and_valid_constraints_are_preserved():
    prepare.check_simple_constraints(
        "Use [address] as a placeholder, in exactly 2 words.", "address: [address]"
    )
    prepare.check_simple_constraints("Use all lowercase letters.", "a correct response.")


def test_human_edit_clipboard_artifact_is_removed_without_changing_code_strings():
    row = {
        "context": conversation(0)["messages"][:-1],
        "edited_response": "```python\nCopy code\nprint('Copy code')\n```",
    }
    answer = prepare.adapt(row, source(kind="edit"))[-1]["content"]
    assert answer == "```python\nprint('Copy code')\n```"


def test_english_filter_checks_answers_but_accepts_code_only_targets():
    english = prepare.EnglishFilter(0.8)
    question = "Please write a Python function that returns the sum of the integers in a list."
    assert english(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "```python\ndef total(xs):\n    return sum(xs)\n```"},
        ]
    )
    assert not english(
        [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": "Cette réponse est entièrement écrite en français et ne répond pas dans la langue demandée par la personne.",
            },
        ]
    )


def test_exact_family_and_verified_near_deduplication():
    dedup = prepare.Deduplicator(0.9)
    text = " ".join(f"word{i}" for i in range(80))
    dedup.add(["family"], dedup.prepare(text))
    assert dedup.duplicate(["family"], dedup.prepare("unrelated"))
    assert dedup.duplicate(["other"], dedup.prepare(text + " extra"))
    assert not dedup.duplicate(["other"], dedup.prepare("different short prompt"))


def test_global_exclusion_and_no_truncation(tmp_path):
    config = recipe(max_tokens=150, validation_samples=0)
    dedup = prepare.Deduplicator(0.9)
    exclusion = tmp_path / "exclude.jsonl"
    exclusion.write_text(json.dumps({"text": "Explain topic 0."}) + "\n")
    prepare.add_exclusions([exclusion], dedup)
    rows = [
        conversation(0),
        conversation(
            2,
            messages=conversation(2)["messages"] + conversation(0)["messages"],
        ),
        {
            "messages": [
                {"role": "user", "content": "Long evidence " * 100},
                {"role": "assistant", "content": "Answer."},
            ]
        },
        conversation(1),
    ]
    selected, report = prepare.select_source(
        source(),
        config,
        {"train": 1, "validation": 0},
        FixtureTokenizer(),
        lambda _: True,
        dedup,
        rows,
    )
    assert len(selected) == 1 and selected[0]["source_id"] == "1"
    assert report["rejected"]["excluded user turn"] == 2
    assert report["rejected"]["over context limit"] == 1


def test_full_categories_are_skipped_before_expensive_processing():
    calls = []
    spec = source(category_field="category", category_cap=0.5)
    rows = [
        conversation(1, category="a"),
        conversation(2, category="a"),
        conversation(3, category="b"),
    ]
    selected, stats = prepare.select_source(
        spec,
        recipe(validation_samples=0),
        {"train": 2, "validation": 0},
        FixtureTokenizer(),
        lambda messages: calls.append(messages) or True,
        prepare.Deduplicator(0.9),
        rows,
    )
    assert len(calls) == 2
    assert [row["category"] for row in selected] == ["a", "b"]
    assert stats["rejected"]["category cap"] == 1


def test_independent_validation_allocations_preserve_training_and_disjointness(tmp_path):
    config = recipe(sources=[source("a", validation_weight=0), source("b", validation_weight=1)])

    def rows(spec, _):
        return (conversation(f"{spec['id']}-{i}") for i in range(100))

    output = tmp_path / "dataset"
    prepare.build(
        config,
        output,
        rows_factory=rows,
        tokenizer=FixtureTokenizer(),
        english=lambda _: True,
    )
    training = pq.read_table(output / "train.parquet").to_pylist()
    validation = pq.read_table(output / "validation.parquet").to_pylist()
    assert [sum(row["source_key"] == key for row in training) for key in ("a", "b")] == [4, 4]
    assert len(validation) == 2 and all(row["source_key"] == "b" for row in validation)
    assert {family for row in training for family in row["family_ids"]}.isdisjoint(
        family for row in validation for family in row["family_ids"]
    )


def test_compilation_is_deterministic_and_outputs_disjoint_splits(tmp_path):
    def rows(*_):
        return (conversation(i) for i in range(100))

    config = recipe()
    first = prepare.build(
        config,
        tmp_path / "first",
        rows_factory=rows,
        tokenizer=FixtureTokenizer(),
        english=lambda _: True,
    )
    second = prepare.build(
        config,
        tmp_path / "second",
        rows_factory=rows,
        tokenizer=FixtureTokenizer(),
        english=lambda _: True,
    )
    assert first == second
    assert first["train"]["samples"] == 8 and first["validation"]["samples"] == 2
    train = pq.read_table(tmp_path / "first/train.parquet").to_pylist()
    val = pq.read_table(tmp_path / "first/validation.parquet").to_pylist()
    assert not {r["family_ids"][0] for r in train} & {r["family_ids"][0] for r in val}
    assert all(len(r["completion"]) == 1 for r in train + val)
    assert {p.name for p in (tmp_path / "first").iterdir()} == {
        "train.parquet",
        "validation.parquet",
        "identity.json",
        "summary.json",
    }
    with pytest.raises(FileExistsError):
        prepare.build(
            config,
            tmp_path / "first",
            rows_factory=rows,
            tokenizer=FixtureTokenizer(),
            english=lambda _: True,
        )


def test_resume_reuses_only_hash_checked_completed_sources(tmp_path):
    config = recipe(sources=[source("one"), source("two")])
    output = tmp_path / "dataset"

    def interrupted(spec, _):
        if spec["id"] == "two":
            raise RuntimeError("network interrupted")
        return (conversation(i) for i in range(100))

    with pytest.raises(RuntimeError, match="network interrupted"):
        prepare.build(
            config,
            output,
            rows_factory=interrupted,
            tokenizer=FixtureTokenizer(),
            english=lambda _: True,
        )
    assert not output.exists()

    def resumed(spec, _):
        assert spec["id"] == "two"
        return (conversation(i) for i in range(100, 200))

    cached = tmp_path / "dataset.building/one.parquet"
    original = cached.read_bytes()
    cached.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt completed source"):
        prepare.build(
            config,
            output,
            resume=True,
            rows_factory=resumed,
            tokenizer=FixtureTokenizer(),
            english=lambda _: True,
        )
    cached.write_bytes(original)
    report = prepare.build(
        config,
        output,
        resume=True,
        rows_factory=resumed,
        tokenizer=FixtureTokenizer(),
        english=lambda _: True,
    )
    assert report["train"]["samples"] == 8


def test_shortfall_does_not_publish_or_relax_filters(tmp_path):
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match="quota shortfall"):
        prepare.build(
            recipe(),
            output,
            rows_factory=lambda *_: [conversation(1)],
            tokenizer=FixtureTokenizer(),
            english=lambda _: True,
        )
    assert not output.exists()
    assert (tmp_path / "dataset.building/shortfall.json").exists()
    with pytest.raises(ValueError, match="identical recipe"):
        prepare.build(
            recipe(seed=3),
            output,
            resume=True,
            tokenizer=FixtureTokenizer(),
            english=lambda _: True,
        )


def test_dry_run_needs_no_output_directory():
    assert prepare.arguments(["--dry-run", "--samples", "1000"]).output_dir is None
    with pytest.raises(SystemExit):
        prepare.arguments([])
