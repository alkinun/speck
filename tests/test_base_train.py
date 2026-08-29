import json
import shutil
from types import SimpleNamespace

from scripts import base_train
from speck.architecture import (
    ArchitectureConfig,
    BlockConfig,
    BlockGroup,
    StageConfig,
    SwiGLUSpec,
)
from speck.checkpoint import load_metadata, load_timing
from speck.speckgym import prepare_speckgym


class FakeTokenizer:
    vocab_size = 256
    bos_id = 1
    eos_id = 2

    def fingerprint(self):
        return "base-train-test-tokenizer"


def procedural_config():
    return {
        "batch_tokens": 40,
        "total_requested_tokens": 160,
        "checkpoint_tokens": [120, 160],
        "procedural": {
            "seed": 7,
            "updates": 2,
            "sequence_length": 8,
            "symbol_count": 128,
            "validation_sequences": 4,
            "reserve_sequences": 4,
            "shard_tokens": 32,
            "formal": {
                "citation": "test",
                "reference": "test",
                "k": 64,
                "p_open": 0.5,
                "max_depth": 16,
            },
        },
    }


def training_config(data_dir, output_dir, *, phase, offset=0, initialization=None):
    model = ArchitectureConfig(
        (BlockGroup(BlockConfig(12, (StageConfig((SwiGLUSpec(24),)),))),),
        embedding_size=8,
        vocab_size=256,
        max_position_embeddings=8,
    )
    return {
        "data": {"output_dir": str(data_dir)},
        "tokenizer": {},
        "model": model.export(),
        "train": {
            "batch_tokens": 40,
            "checkpoint_tokens": [] if phase == "procedural_warmup" else [120, 160],
            "device_batch_size": 1,
            "eval_every": 0,
            "eval_tokens": 8,
            "final_eval_tokens": 8,
            "global_token_offset": offset,
            "grad_clip": 1.0,
            "initialization": initialization,
            "log_every": 1,
            "lr": 1e-3,
            "min_lr": 0.1,
            "optimizer": "adamw",
            "output_dir": str(output_dir),
            "run": "dummy",
            "save_every": 0,
            "sequence_length": 8,
            "train_tokens": 80,
            "training_phase": phase,
            "wandb_group": "test",
            "wandb_project": "test",
            "warmup_steps": 3,
            "weight_decay": 0.1,
        },
    }


def test_procedural_checkpoint_initializes_a_fresh_language_phase(tmp_path, monkeypatch):
    tokenizer = FakeTokenizer()
    data_root = tmp_path / "data"
    prepare_speckgym(procedural_config(), tokenizer, output_dir=data_root)
    data_dir = data_root / "B-RandomSymbols"
    monkeypatch.setattr(base_train, "get_tokenizer", lambda **settings: tokenizer)
    cli = SimpleNamespace(
        device="cpu",
        resume=None,
        no_compile=True,
        experiment=str(tmp_path / "experiment"),
    )

    warmup_dir = tmp_path / "warmup"
    base_train.train(training_config(data_dir, warmup_dir, phase="procedural_warmup"), cli)
    warmup = load_metadata(warmup_dir, 2)
    assert warmup["global_tokens"] == 80
    assert warmup["training_phase"] == "procedural_warmup"
    assert warmup["initialization"] is None

    language_dir = tmp_path / "language"
    base_train.train(
        training_config(
            data_dir,
            language_dir,
            phase="language",
            offset=80,
            initialization={
                "kind": "backbone_checkpoint",
                "checkpoint_dir": str(warmup_dir),
                "step": 2,
            },
        ),
        cli,
    )
    language = load_metadata(language_dir, 2)
    assert language["global_step"] == 4
    assert language["global_tokens"] == 160
    assert language["milestone_tokens"] == 160
    assert language["validation_step"] == 2
    assert language["validation_global_tokens"] == 160
    assert language["initialization"]["policy"] == "reset_token_interface"
    assert language["initialization"]["reset"] == [
        "adapters.0.weight",
        "embed_tokens.weight",
        "lm_head.weight",
        "output_projection.weight",
    ]
    summary = json.loads((language_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["global_tokens"] == 160
    assert summary["optimizer_seconds"] > 0
    checkpoint_timing = load_timing(language_dir, 2)
    assert checkpoint_timing["checkpoint_seconds"] > 0
    assert checkpoint_timing["active_seconds"] >= language["timing"]["active_seconds"]

    shutil.rmtree(warmup_dir)
    cli.resume = 1
    base_train.train(
        training_config(
            data_dir,
            language_dir,
            phase="language",
            offset=80,
            initialization={
                "kind": "backbone_checkpoint",
                "checkpoint_dir": str(warmup_dir),
                "step": 2,
            },
        ),
        cli,
    )
    resumed = load_metadata(language_dir, 2)
    assert resumed["initialization"] == language["initialization"]
