import json

import pyarrow as pa
import pyarrow.parquet as pq

from speck.provenance.io import file_sha256
from speck.tokenization.chat import ChatTokenizer
from speck.training import self_distill
from speck.training.sft_data import prepare_sft_dataset
from tests.training.test_rl import AnyIdTokenizer, parent_checkpoint


def test_verified_samples_and_anchors_become_a_local_sft_dataset(tmp_path, monkeypatch):
    tokenizer = ChatTokenizer(AnyIdTokenizer(tmp_path / "tokenizer.model"))
    monkeypatch.setattr(self_distill, "get_chat_tokenizer", lambda **config: tokenizer)
    prompts = tmp_path / "prompts.jsonl"
    rows = [{"domain": "Math", "query": f"{index}*2", "ground_truth": "0"} for index in range(12)]
    prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    anchor = tmp_path / "anchor.parquet"
    anchors = [
        {
            "messages": [
                {"role": "user", "content": f"hi {index}"},
                {"role": "assistant", "content": f"hello {index}"},
            ],
            "source": "anchor",
        }
        for index in range(20)
    ]
    pq.write_table(pa.Table.from_pylist(anchors), anchor)
    settings = {
        "prompt_files": [str(prompts)],
        "samples_per_prompt": 4,
        "keep_per_prompt": 1,
        "max_prompt_tokens": 32,
        "max_new_tokens": 24,
        "temperature": 1.0,
        "seed": 3,
        "validation_fraction": 0.5,
        "parent": parent_checkpoint(tmp_path, tokenizer),
        "anchor_files": [{"path": str(anchor), "sha256": file_sha256(anchor)}],
        "output_dir": str(tmp_path / "self-sft"),
    }

    def reward(prompt, text):
        return float(len(text) % 2 == 0)

    receipt = self_distill.build({"tokenizer": {}, "self_distill": settings}, reward=reward)
    counts = receipt["counts"]
    assert counts["prompts"] == 12 and counts["samples"] == 48
    # keep_per_prompt is one, so every prompt with an accepted sample contributes exactly one.
    assert counts["self_distilled_conversations"] == counts["prompts_with_accepted"]
    assert counts["anchor_conversations"] == 20
    dataset = receipt["dataset"]
    assert dataset["expected_samples"] == counts["self_distilled_conversations"] + 20
    manifest = prepare_sft_dataset(
        dataset, tokenizer, [64], tmp_path / "packed", source_dir=tmp_path / "self-sft"
    )
    assert manifest["splits"]["train"]["samples"] + manifest["splits"]["val"]["samples"] > 0
