"""Exercise the CPU data/train/resume/evaluate path with deterministic local inputs."""

import contextlib
import io
import math
from pathlib import Path

import sentencepiece
import torch

from speck.data.dataset import prepare_dataset
from speck.evaluation.loss import arguments as evaluation_arguments
from speck.evaluation.loss import run as evaluate
from speck.provenance.io import atomic_json
from speck.tokenization.tokenizer import Tokenizer
from speck.training.base import arguments, train
from speck.training.checkpoint import load_model


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
        "model": {"blocks": blocks, "embedding_size": 16, "max_position_embeddings": 32},
    }
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
    result = {
        "status": "pass",
        "resume": "exact_parameter_parity",
        "steps": 4,
        "evaluated_tokens": evaluation["evaluated_tokens"],
        "loss": evaluation["loss"],
    }
    atomic_json(directory / "summary.json", result)
    return result
