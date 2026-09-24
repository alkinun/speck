"""Prompt selection must drop duplicates, conflicts, SFT reuse and benchmark matches."""

import json

import pyarrow as pa
import pyarrow.parquet as pq

from speck.training import rl_prompts


class NoBenchmarks:
    def __init__(self, prepared):
        pass

    def matches(self, text):
        return ["bench"] if "benchmark" in text else []


def test_selection_rejects_each_ineligible_kind(tmp_path, monkeypatch):
    monkeypatch.setattr(rl_prompts, "BenchmarkExclusion", NoBenchmarks)
    rows = [
        {"uuid": "a", "query": "Add 1 and 1.", "ground_truth": "2", "domain": "Math"},
        {"uuid": "b", "query": "add 1  and 1.", "ground_truth": "2", "domain": "Math"},
        {"uuid": "c", "query": "Two answers?", "ground_truth": "1", "domain": "Math"},
        {"uuid": "d", "query": "two answers?", "ground_truth": "3", "domain": "Math"},
        {"uuid": "e", "query": "Seen in SFT.", "ground_truth": "4", "domain": "Math"},
        {"uuid": "f", "query": "A benchmark item.", "ground_truth": "5", "domain": "Math"},
        {"uuid": "g", "query": "Explain history.", "ground_truth": "x", "domain": "Knowledge"},
    ]
    source = tmp_path / "rows.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    sft = tmp_path / "train.parquet"
    messages = [json.dumps({"role": "user", "content": "seen in sft."})]
    pq.write_table(pa.table({"messages": [messages]}), sft)
    receipt = rl_prompts.prepare(
        [source], tmp_path / "out", {"benchmarks": []}, limit=1, seed="s", exclude_sft=[sft]
    )
    assert receipt["rejections"] == {
        "unverifiable_domain": 1,
        "duplicate_copy": 2,
        "conflicting_reference": 1,
        "sft_prompt": 1,
        "benchmark_match": 1,
    }
    selected = [
        json.loads(line) for line in (tmp_path / "out/prompts.jsonl").read_text().splitlines()
    ]
    assert [row["uuid"] for row in selected] == ["a"]
