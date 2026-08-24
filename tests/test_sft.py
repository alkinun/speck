import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import torch

import speck.sft as sft_module
from speck.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    BlockConfig,
    BlockGroup,
    StageConfig,
    SwiGLUSpec,
)
from speck.chat import ChatTokenizer
from speck.model import SpeckForCausalLM
from speck.sft import (
    _truncate_conversation,
    prepare_sft_dataset,
    sft_loader,
    sft_optimization_step,
    sft_plan,
    verify_sft_dataset,
)


class BaseTokenizer:
    vocab_size = 300
    bos_id = 1
    eos_id = 2

    def __init__(self, model_path):
        self.model_path = str(model_path)

    def encode(self, text):
        return [byte + 3 for byte in text.encode()]

    def decode(self, tokens):
        return bytes(token - 3 for token in tokens).decode()

    def fingerprint(self):
        return "base-tokenizer"


def test_long_conversation_keeps_latest_user_and_assistant(tmp_path):
    path = tmp_path / "tokenizer.model"
    path.write_bytes(b"sentencepiece")
    tokenizer = ChatTokenizer(BaseTokenizer(path))
    tokens, mask = tokenizer.encode_messages(
        [
            {"role": "user", "content": "old " * 30},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new " * 30},
            {"role": "assistant", "content": "new answer"},
        ]
    )

    truncated, truncated_mask, changed = _truncate_conversation(tokens, mask, tokenizer, 48)

    assert changed
    assert len(truncated) == len(truncated_mask) <= 48
    assert truncated[:2] == [tokenizer.bos_id, tokenizer.role_ids["user"]]
    assert truncated[2 : 2 + len(tokenizer.newline_ids)] == list(tokenizer.newline_ids)
    assert tokenizer.role_ids["assistant"] in truncated
    assert any(truncated_mask)


def test_oversized_assistant_keeps_a_canonical_minimal_user(tmp_path):
    path = tmp_path / "tokenizer.model"
    path.write_bytes(b"sentencepiece")
    tokenizer = ChatTokenizer(BaseTokenizer(path))
    tokens, mask = tokenizer.encode_messages(
        [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "answer " * 100},
        ]
    )

    truncated, truncated_mask, changed = _truncate_conversation(tokens, mask, tokenizer, 48)

    expected_prefix = [tokenizer.bos_id, tokenizer.role_ids["user"]]
    expected_prefix.extend(tokenizer.base.encode("\n..."))
    expected_prefix.extend([tokenizer.eos_id, *tokenizer.newline_ids])
    expected_prefix.extend([tokenizer.role_ids["assistant"], *tokenizer.newline_ids])
    assert changed
    assert truncated[: len(expected_prefix)] == expected_prefix
    assert len(truncated) == len(truncated_mask) == 48
    assert any(truncated_mask)


def test_prepare_and_load_masked_sft_data(tmp_path, monkeypatch):
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"sentencepiece")
    tokenizer = ChatTokenizer(BaseTokenizer(tokenizer_path))
    parquet_path = tmp_path / "data.parquet"
    short = {
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ],
        "source": "test/source",
    }
    rows = [short, short, short, short, short]
    rows.append(
        {
            "messages": [
                {"role": "user", "content": "Question" * 2},
                {"role": "assistant", "content": "Answer" * 2},
            ],
            "source": "test/source",
        }
    )
    rows.append(
        {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": ""},
            ],
            "source": "test/source",
        }
    )
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    monkeypatch.setattr(sft_module, "hf_hub_download", lambda *args, **kwargs: parquet_path)
    config = {
        "repo": "test/dataset",
        "revision": "a" * 40,
        "files": ["data.parquet"],
        "expected_samples": 7,
        "validation_samples": 1,
    }
    output = tmp_path / "packed"

    manifest = prepare_sft_dataset(config, tokenizer, [16, 64], output)

    assert manifest["splits"]["train"]["samples"] == 5
    assert manifest["splits"]["train"]["input_samples"] == 6
    assert manifest["splits"]["train"]["rejected_samples"] == 1
    assert manifest["splits"]["val"]["samples"] == 1
    assert manifest["splits"]["train"]["supervised_tokens"] > 0
    assert manifest["splits"]["train"]["buckets"]["16"]["samples"] == 4
    assert manifest["splits"]["train"]["buckets"]["64"]["samples"] == 1
    verify_sft_dataset(output, manifest)
    plan = sft_plan(manifest, "train", device_tokens=64)
    assert plan["cycle_microbatches"] == 2
    assert plan["buckets"][16]["batch_size"] == 4
    assert plan["buckets"][64]["batch_size"] == 1
    loader = sft_loader(tokenizer, 64, data_dir=output, device="cpu")
    short_inputs, short_targets, short_state = next(loader)
    long_inputs, long_targets, long_state = next(loader)
    assert short_inputs.shape == short_targets.shape == (4, 16)
    assert long_inputs.shape == long_targets.shape == (1, 64)
    assert short_state["global_consumed_microbatches"] == 0
    assert long_state["global_consumed_microbatches"] == 1
    assert (short_targets == -100).any() and (short_targets != -100).any()
    resumed = sft_loader(
        tokenizer,
        64,
        resume_state_dict=long_state,
        data_dir=output,
        device="cpu",
    )
    resumed_inputs, resumed_targets, resumed_state = next(resumed)
    assert torch.equal(resumed_inputs, long_inputs)
    assert torch.equal(resumed_targets, long_targets)
    assert resumed_state == long_state


def test_sft_loader_masks_target_positions(tmp_path):
    class Tokenizer:
        eos_id = 2

        def metadata(self):
            return {"fingerprint": "test"}

    tokens = torch.arange(15, dtype=torch.int64).numpy().astype("<u2")
    mask = torch.zeros(15, dtype=torch.uint8).numpy()
    mask[[2, 4]] = 1
    token_path = tmp_path / "train.tokens.bin"
    mask_path = tmp_path / "train.mask.bin"
    token_path.write_bytes(tokens.tobytes())
    mask_path.write_bytes(mask.tobytes())

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    bucket = {
        "samples": 3,
        "tokens": 15,
        "supervised_tokens": 2,
        "truncated_samples": 0,
        "sequence_length": 4,
        "token_file": token_path.name,
        "token_sha256": digest(token_path),
        "mask_file": mask_path.name,
        "mask_sha256": digest(mask_path),
    }
    sft_module._write_json(
        tmp_path / "manifest.json",
        {
            "format": "speck_sft",
            "format_version": 3,
            "dataset": {
                "repo": "test",
                "revision": "a" * 40,
                "files": ["test"],
                "sequence_lengths": [4],
            },
            "tokenizer": Tokenizer().metadata(),
            "splits": {
                split: {
                    "samples": 3,
                    "tokens": 15,
                    "supervised_tokens": 2,
                    "buckets": {"4": bucket},
                }
                for split in ("train", "val")
            },
        },
    )

    loader = sft_loader(Tokenizer(), 8, data_dir=tmp_path, device="cpu")
    inputs, targets, _ = next(loader)
    padded_inputs, padded_targets, _ = next(loader)

    assert inputs[0].tolist() == [0, 1, 2, 3]
    assert targets[0].tolist() == [-100, 2, -100, 4]
    assert padded_inputs.shape == padded_targets.shape == (2, 4)
    assert padded_inputs[1].tolist() == [2, 2, 2, 2]
    assert padded_targets[1].tolist() == [-100, -100, -100, -100]


def test_sft_plan_keeps_constant_tokens_and_exact_resume_geometry():
    manifest = {
        "splits": {
            "train": {
                "buckets": {
                    "4": {"samples": 16},
                    "8": {"samples": 6},
                }
            }
        }
    }

    plan = sft_plan(
        manifest,
        "train",
        device_tokens=8,
        world_size=2,
        accumulation=2,
    )

    assert plan["cycle_microbatches"] == 8
    assert plan["real_microbatches"] == 7
    assert plan["dummy_microbatches"] == 1
    assert plan["context_tokens"] == 128
    assert plan["buckets"][4]["batch_size"] == 2
    assert plan["buckets"][8]["batch_size"] == 1
    assert all(
        sequence_length * plan["buckets"][sequence_length]["batch_size"] == 8
        for sequence_length, _ in plan["schedule"]
    )
    assert sum(bucket["scheduled_microbatches"] for bucket in plan["buckets"].values()) == 8
    assert sum(bucket["used_samples"] for bucket in plan["buckets"].values()) == 22
    assert sum(bucket["padded_samples"] for bucket in plan["buckets"].values()) == 4


def test_sft_optimization_counts_supervised_tokens():
    config = ArchitectureConfig(
        (
            BlockGroup(
                BlockConfig(
                    8,
                    (
                        StageConfig((AttentionSpec(4, 1),)),
                        StageConfig((SwiGLUSpec(16),)),
                    ),
                )
            ),
        ),
        embedding_size=8,
        vocab_size=16,
        max_position_embeddings=8,
    )
    model = SpeckForCausalLM(config)
    model.init_weights()
    optimizer = model.optimizer(lr=1e-3)
    inputs = torch.randint(0, 16, (1, 4))
    first_targets = torch.tensor([[1, -100, 2, -100]])
    second_targets = torch.tensor([[-100, 3, -100, -100]])
    first = (inputs, first_targets, {"batch": 0})
    second = (inputs, second_targets, {"batch": 1})
    third = (inputs, second_targets, {"batch": 2})

    loss, grad_norm, next_batch, supervised = sft_optimization_step(
        model,
        tuple(model.parameters()),
        optimizer,
        iter([second, third]),
        first,
        accumulation=2,
        grad_clip=1.0,
        lr=1e-3,
    )

    assert supervised == 3
    assert torch.isfinite(loss) and torch.isfinite(grad_norm)
    assert next_batch[2] == {"batch": 2}
