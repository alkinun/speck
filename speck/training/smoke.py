"""Exercise the CPU data/train/resume/evaluate path with deterministic local inputs."""

import contextlib
import io
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import sentencepiece
import torch

from speck.data.dataset import prepare_dataset
from speck.evaluation.loss import arguments as evaluation_arguments
from speck.evaluation.loss import run as evaluate
from speck.export.pretrained import native_pretrained_source
from speck.model import build_model
from speck.provenance.io import atomic_json, file_sha256
from speck.tokenization.chat import ChatTokenizer
from speck.tokenization.tokenizer import Tokenizer
from speck.training.base import arguments, train
from speck.training.checkpoint import load_model
from speck.training.rl import RLTrainer
from speck.training.sft import SFTTrainer
from speck.training.sft_data import prepare_sft_dataset


def run_smoke(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    experiment = directory / "experiment"
    experiment.mkdir()
    tokenizer_dir = directory / "tokenizer"
    tokenizer_dir.mkdir()
    documents = [
        f"Document {index}: the small model reads a different sequence of words and numbers {index * 17}."
        for index in range(300)
    ]
    model_bytes = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(documents),
        model_writer=model_bytes,
        vocab_size=64,
        model_type="bpe",
        hard_vocab_limit=False,
        num_threads=1,
        shuffle_input_sentence=False,
        minloglevel=2,
    )
    model_path = tokenizer_dir / "tokenizer.model"
    model_path.write_bytes(model_bytes.getvalue())
    tokenizer = Tokenizer(model_path)
    atomic_json(tokenizer_dir / "tokenizer_metadata.json", {"fingerprint": tokenizer.fingerprint()})
    data = {
        "sources": [
            {
                "id": "fixture",
                "repo": "local/fixture",
                "revision": None,
                "tree_path": "data",
                "content_column": "text",
                "filters": {},
            }
        ],
        "mixture": {"phases": [{"end_tokens": 1024, "weights": {"fixture": 100}}]},
        "requested_train_tokens": 1024,
        "validation_tokens_per_source": 128,
        "validation_fraction": 0.2,
        "filtering": {"min_chars": 0, "max_chars": 10000},
        "dedup": {
            "normalization": "NFKC+lower+whitespace",
            "hash": "blake2b-128",
            "scope": "global",
        },
        "shards": {"tokens": 256, "maximum_loader_microbatch_tokens": 32},
        "seed": 7,
        "output_dir": str(directory / "data"),
    }
    prepare_dataset(
        **data,
        tokenizer=tokenizer,
        check_disk=False,
        document_iterators={"fixture": ({"content": text, "metadata": {}} for text in documents)},
    )
    blocks = []
    for mixer in (
        {
            "kind": "kimi_delta_attention",
            "key_head_dim": 8,
            "value_head_dim": 8,
            "num_key_heads": 1,
            "num_value_heads": 2,
            "conv_kernel_size": 4,
        },
        {"kind": "attention", "head_dim": 8, "num_key_value_heads": 1},
    ):
        blocks.append(
            {
                "block": {
                    "hidden_size": 16,
                    "stages": [
                        {"branches": [mixer]},
                        {"branches": [{"kind": "swiglu", "intermediate_size": 32}]},
                    ],
                }
            }
        )
    settings = {
        "batch_tokens": 64,
        "device_batch_size": 2,
        "sequence_length": 16,
        "train_tokens": 256,
        "eval_tokens": 64,
        "final_eval_tokens": 64,
        "eval_every": 2,
        "save_every": 2,
        "log_every": 1,
        "grad_clip": 1.0,
        "lr": 0.001,
        "lr_schedule": "cosine",
        "min_lr": 0.1,
        "optimizer": "adamw",
        "run": "dummy",
        "output_dir": None,
        "wandb_project": None,
        "warmup_steps": 0,
        "weight_decay": 0.1,
        "checkpoint_tokens": [128],
        "seed": 42,
    }
    configs = {
        "data": data,
        "tokenizer": {"directory": str(tokenizer_dir)},
        "train": settings,
        "model": {
            "blocks": blocks,
            "embedding_size": 16,
            "max_position_embeddings": 64,
            "vocab_size": tokenizer.vocab_size + 3,
        },
    }
    with torch.device("meta"):
        count = build_model(configs["model"], tokenizer.vocab_size).parameter_count()
    configs["model"].update(expected_parameters=count, expected_active_parameters=count)
    for name, value in configs.items():
        atomic_json(experiment / f"{name}.json", value)
    common = [str(experiment), "--device", "cpu", "--no-compile"]
    uninterrupted, resumed = directory / "uninterrupted", directory / "resumed"
    with (directory / "training.log").open("w") as log, contextlib.redirect_stdout(log):
        train(configs, arguments(common + ["--output-dir", str(uninterrupted)]))
        train(
            configs, arguments(common + ["--output-dir", str(resumed), "--stop-at-tokens", "128"])
        )
        train(configs, arguments(common + ["--output-dir", str(resumed), "--resume", "2"]))
    expected = load_model(uninterrupted, 4, "cpu")
    actual = load_model(resumed, 4, "cpu")
    if expected.keys() != actual.keys():
        raise AssertionError("resumed checkpoint parameter names changed")
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    evaluation = evaluate(
        evaluation_arguments(
            [
                str(experiment),
                "--checkpoint-dir",
                str(resumed),
                "--device",
                "cpu",
                "--no-compile",
                "--eval-tokens",
                "64",
                "--output",
                str(directory / "evaluation.json"),
            ]
        )
    )
    if not math.isfinite(evaluation["loss"]):
        raise AssertionError("smoke evaluation loss is not finite")
    mid_training_result = run_mid_training_smoke(directory, configs, tokenizer, resumed)
    sft_result = run_sft_smoke(directory, configs, tokenizer, resumed)
    rl_result = run_rl_smoke(directory, configs, directory / "assistant")
    result = {
        "status": "pass",
        "mid_training": mid_training_result,
        "sft": sft_result,
        "rl": rl_result,
        "resume": "exact_parameter_parity",
        "steps": 4,
        "evaluated_tokens": evaluation["evaluated_tokens"],
        "loss": evaluation["loss"],
    }
    atomic_json(directory / "summary.json", result)
    return result


def run_mid_training_smoke(directory, configs, tokenizer, parent):
    """Branch the base parent onto best-fit-packed chat rows with assistant-only loss masks."""

    def record(index):
        messages = [
            {"role": "user", "content": f"{index}"},
            {"role": "assistant", "content": f"{index * 3}"},
        ]
        return {"content": json.dumps({"messages": messages}), "metadata": {}}

    length = configs["train"]["sequence_length"]
    data = {
        **configs["data"],
        "sources": [
            {
                **configs["data"]["sources"][0],
                "packing": {"row_tokens": length, "open_rows": 4},
                "record_format": "messages",
            }
        ],
        "mixture": {"phases": [{"end_tokens": 256, "weights": {"fixture": 100}}]},
        "requested_train_tokens": 256,
        "validation_tokens_per_source": 2 * length,
        "output_dir": str(directory / "mid-training-data"),
    }
    chat = ChatTokenizer(tokenizer)
    prepare_dataset(
        **data,
        tokenizer=chat,
        check_disk=False,
        document_iterators={"fixture": (record(index) for index in range(400))},
    )
    experiment = directory / "mid-training-experiment"
    experiment.mkdir()
    mid_configs = {
        **configs,
        "data": data,
        "tokenizer": {**configs["tokenizer"], "chat_format_version": 2},
        "train": {
            **configs["train"],
            "training_phase": "data_continuation",
            "train_tokens": 128,
            "checkpoint_tokens": [],
        },
    }
    for name, value in mid_configs.items():
        atomic_json(experiment / f"{name}.json", value)
    common = [str(experiment), "--device", "cpu", "--no-compile", "--branch-kind", "data"]
    branch = ["--branch-from", str(parent), "--branch-step", "2"]
    first, second = directory / "mid-training", directory / "mid-training-repeat"
    with (directory / "mid-training.log").open("w") as log, contextlib.redirect_stdout(log):
        train(mid_configs, arguments(common + branch + ["--output-dir", str(first)]))
        train(mid_configs, arguments(common + branch + ["--output-dir", str(second)]))
    expected, actual = load_model(first, 2, "cpu"), load_model(second, 2, "cpu")
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    return {
        "steps": 2,
        "parent": "base_checkpoint_step_2",
        "packing": "best_fit_rows",
        "masks": "assistant",
        "repeat": "exact_parameter_parity",
    }


def run_sft_smoke(directory, configs, tokenizer, parent):
    """Exercise local base-to-SFT initialization, weighted packing, and resume."""

    source = directory / "sft-source"
    source.mkdir()
    row = {
        "messages": [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2", "weight": 0},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "4", "weight": 1},
        ],
        "tools": [],
        "source": "fixture",
    }
    files = []
    for split, count in (("train", 4), ("val", 2)):
        path = source / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist([row] * count), path)
        files.append({"filename": path.name, "split": split, "sha256": file_sha256(path)})
    dataset = {
        "format": "messages_v1",
        "name": "smoke-sft",
        "files": files,
        "expected_samples": 6,
        "validation_samples": 2,
    }
    chat = ChatTokenizer(tokenizer)
    packed = directory / "sft-data"
    prepare_sft_dataset(dataset, chat, [64], packed, source_dir=source)
    output = directory / "assistant"
    settings = {
        "batch_tokens": 128,
        "device_batch_size": 2,
        "sequence_length": 64,
        "sequence_lengths": [64],
        "dataset": dataset,
        "data_dir": str(packed),
        "epochs": 1,
        "eval_every": 1,
        "save_every": 1,
        "log_every": 1,
        "keep_checkpoints": 3,
        "grad_clip": 1.0,
        "lr": 0.001,
        "min_lr": 0.1,
        "optimizer": "adamw",
        "run": "dummy",
        "wandb_project": None,
        "warmup_steps": 0,
        "weight_decay": 0.1,
        "output_dir": str(output),
        "pretrained": native_pretrained_source(parent, 4),
    }
    experiment = directory / "sft-experiment"
    experiment.mkdir()
    sft_configs = {"model": configs["model"], "tokenizer": configs["tokenizer"], "sft": settings}
    for name, value in sft_configs.items():
        atomic_json(experiment / f"{name}.json", value)
    cli = SimpleNamespace(experiment=str(experiment), device="cpu", resume=None, no_compile=True)
    with (directory / "sft.log").open("w") as log, contextlib.redirect_stdout(log):
        SFTTrainer(sft_configs, cli).run()
        resumed = directory / "assistant-resumed"
        resumed.mkdir()
        for name in (
            "model_000001.pt",
            "optimizer_000001.pt",
            "metadata_000001.json",
            "complete_000001",
        ):
            shutil.copy2(output / name, resumed / name)
        resumed_configs = {**sft_configs, "sft": {**settings, "output_dir": str(resumed)}}
        SFTTrainer(resumed_configs, SimpleNamespace(**{**vars(cli), "resume": 1})).run()
    expected, actual = load_model(output, 2, "cpu"), load_model(resumed, 2, "cpu")
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    return {
        "steps": 2,
        "parent": "native_checkpoint",
        "resume": "exact_parameter_parity",
        "chat_format_version": 2,
    }


def run_rl_smoke(directory, configs, parent):
    """Run verifier-rewarded group policy steps from the SFT checkpoint and resume them."""

    prompts = directory / "rl-prompts.jsonl"
    rows = [
        {"domain": "Math", "query": f"{index} + {index}", "ground_truth": str(2 * index)}
        for index in range(4)
    ]
    prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    settings = {
        "prompt_files": [str(prompts)],
        "group_size": 4,
        "activation_checkpointing": False,
        "prompts_per_step": 2,
        "max_prompt_tokens": 48,
        "max_new_tokens": 8,
        "temperature": 1.0,
        "steps": 2,
        "lr": 0.001,
        "min_lr": 0.1,
        "warmup_steps": 0,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "optimizer": "adamw",
        "save_every": 1,
        "seed": 11,
        "parent": native_pretrained_source(parent, 2),
        "output_dir": str(directory / "rl"),
    }
    rl_configs = {"tokenizer": configs["tokenizer"], "rl": settings}
    resumed = directory / "rl-resumed"
    with (directory / "rl.log").open("w") as log, contextlib.redirect_stdout(log):
        RLTrainer(rl_configs).run()
        resumed.mkdir()
        for name in (
            "model_000001.pt",
            "optimizer_000001.pt",
            "metadata_000001.json",
            "complete_000001",
        ):
            shutil.copy2(directory / "rl" / name, resumed / name)
        RLTrainer({**rl_configs, "rl": {**settings, "output_dir": str(resumed)}}).run()
    expected, actual = load_model(directory / "rl", 2, "cpu"), load_model(resumed, 2, "cpu")
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    return {
        "steps": 2,
        "parent": "sft_checkpoint",
        "reward": "checked_math",
        "resume": "exact_parameter_parity",
    }
