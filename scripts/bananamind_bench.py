"""Run the official BananaMind Base Bench 1.1 runner with Speck support."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from speck.architecture import ArchitectureConfig
from speck.chat import ChatTokenizer
from speck.checkpoint import latest, load_model
from speck.common import base_dir
from speck.config import load_experiment
from speck.model import SpeckForCausalLM
from speck.tokenizer import get_tokenizer

DATASET_ID = "BananaMind/BananaMind-Base-Bench-1.1"
DATASET_REVISION = "d4aade51312889e8580963e1ce960c6eaef1a450"
RUNNER_SHA256 = "973a81d09d1c4075d031e1369b4278c52a7813d1ab3b11b33eef665d3247bf2c"


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _official_runner():
    path = Path(
        hf_hub_download(
            DATASET_ID,
            "benchmark.py",
            repo_type="dataset",
            revision=DATASET_REVISION,
        )
    )
    if hashlib.sha256(path.read_bytes()).hexdigest() != RUNNER_SHA256:
        raise ValueError("official BananaMind benchmark runner checksum mismatch")
    spec = importlib.util.spec_from_file_location("bananamind_base_bench_1_1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load the official benchmark runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _speck_experiment(value):
    path = Path(value).expanduser()
    return path if (path / "model.json").is_file() and (path / "train.json").is_file() else None


def _parse_speck_options(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--speck-checkpoint-step", type=int)
    parser.add_argument("--speck-checkpoint-dir")
    options, remaining = parser.parse_known_args(argv)
    if options.speck_checkpoint_step is not None and options.speck_checkpoint_step < 0:
        parser.error("--speck-checkpoint-step must be non-negative")
    return options.speck_checkpoint_step, options.speck_checkpoint_dir, remaining


def _pin_dataset_revision(args):
    if args.dataset_id == DATASET_ID and args.dataset_revision == "main":
        args.dataset_revision = DATASET_REVISION
    return args


def _resolve_speck_run(args, checkpoint_step, checkpoint_directory=None):
    experiment = _speck_experiment(args.model)
    if experiment is None:
        if checkpoint_step is not None or checkpoint_directory is not None:
            raise ValueError(
                "Speck checkpoint options require --model to specify a Speck experiment"
            )
        return None
    if args.tokenizer is not None:
        raise ValueError("Speck evaluations use the checkpoint tokenizer; omit --tokenizer")

    configs = load_experiment(experiment, "tokenizer", "train")
    checkpoint_dir = (
        Path(checkpoint_directory).expanduser()
        if checkpoint_directory is not None
        else Path(
            configs["train"].get("output_dir")
            or Path(base_dir()) / "checkpoints" / configs["train"]["run"]
        ).expanduser()
    )
    step = latest(checkpoint_dir) if checkpoint_step is None else checkpoint_step
    if step is None:
        raise FileNotFoundError(f"no checkpoint found in {checkpoint_dir}")

    complete_path = checkpoint_dir / f"complete_{step:06d}"
    model_path = checkpoint_dir / f"model_{step:06d}.pt"
    metadata_path = checkpoint_dir / f"metadata_{step:06d}.json"
    if not complete_path.is_file() or not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint {step} is incomplete in {checkpoint_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("step") != step:
        raise ValueError(f"checkpoint metadata step does not match {step}")

    tokenizer_config = metadata.get("resolved", {}).get("tokenizer")
    if not isinstance(tokenizer_config, dict):
        raise ValueError("checkpoint metadata does not identify its tokenizer")
    tokenizer = get_tokenizer(**configs["tokenizer"])
    if metadata.get("training_phase") == "sft":
        checkpoint_tokenizer = ChatTokenizer(tokenizer)
        if tokenizer_config != checkpoint_tokenizer.metadata():
            raise ValueError("SFT checkpoint tokenizer metadata does not match the experiment")
    else:
        if configs["tokenizer"] != tokenizer_config:
            raise ValueError("experiment tokenizer configuration differs from the checkpoint")
        checkpoint_tokenizer = tokenizer
    model_config = metadata["config"]
    expected = (
        model_config["vocab_size"],
        model_config["bos_token_id"],
        model_config["eos_token_id"],
    )
    actual = (
        checkpoint_tokenizer.vocab_size,
        checkpoint_tokenizer.bos_id,
        checkpoint_tokenizer.eos_id,
    )
    if actual != expected:
        raise ValueError("checkpoint model and tokenizer IDs do not match")

    identity = {
        "experiment": str(experiment.resolve()),
        "checkpoint_directory": str(checkpoint_dir.resolve()),
        "checkpoint_step": step,
        "checkpoint_sha256": _file_sha256(model_path),
        "metadata_sha256": _file_sha256(metadata_path),
        "tokenizer_sha256": checkpoint_tokenizer.fingerprint(),
        "scoring_tokenizer_sha256": tokenizer.fingerprint(),
        "tokenizer_repository": configs["tokenizer"].get("repo"),
        "tokenizer_revision": configs["tokenizer"].get("revision"),
        "scoring_format": "raw_continuation",
    }
    return {
        "checkpoint_dir": checkpoint_dir,
        "identity": identity,
        "metadata": metadata,
        "step": step,
        "tokenizer": tokenizer,
    }


def _add_run_identity(signature, args, official):
    device = official.resolve_device(args.device)
    signature.update(
        {
            "runner_sha256": RUNNER_SHA256,
            "device": str(device),
            "dtype": str(official.resolve_dtype(args.dtype, device)).removeprefix("torch."),
            "batch_size": args.batch_size,
            "threads": args.threads,
        }
    )
    if args._speck_run is not None:
        signature["speck"] = args._speck_run["identity"]
    return signature


def _add_report_identity(report_path, args):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["runner_sha256"] = RUNNER_SHA256
    report["batch_size"] = args.batch_size
    report["threads"] = args.threads
    if args._speck_run is not None:
        report["speck"] = args._speck_run["identity"]
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main():
    official = _official_runner()
    hf_runner = official.ModelRunner
    official_parse_args = official.parse_args
    official_run_signature = official.run_signature
    official_write_outputs = official.write_outputs

    def parse_args():
        checkpoint_step, checkpoint_directory, remaining = _parse_speck_options(sys.argv[1:])
        original_argv = sys.argv[:]
        try:
            sys.argv[1:] = remaining
            args = official_parse_args()
        finally:
            sys.argv[:] = original_argv
        _pin_dataset_revision(args)
        args._speck_run = _resolve_speck_run(args, checkpoint_step, checkpoint_directory)
        return args

    def run_signature(args, data_sha256):
        return _add_run_identity(official_run_signature(args, data_sha256), args, official)

    def write_outputs(args, runner, data_path, data_sha256, results, *, official):
        paths = official_write_outputs(
            args,
            runner,
            data_path,
            data_sha256,
            results,
            official=official,
        )
        _add_report_identity(paths[0], args)
        return paths

    class ModelRunner(hf_runner):
        def __init__(self, args, token):
            run = args._speck_run
            self._is_speck = run is not None
            if run is None:
                super().__init__(args, token)
                return

            self.args = args
            self.device = official.resolve_device(args.device)
            self.dtype = official.resolve_dtype(args.dtype, self.device)
            torch.set_num_threads(args.threads)
            torch.set_float32_matmul_precision("high")

            if self.device.type == "cuda" and self.dtype != torch.bfloat16:
                raise ValueError("Speck CUDA inference supports only BF16")
            metadata = run["metadata"]
            model_state = load_model(run["checkpoint_dir"], run["step"], "cpu")
            self.model = SpeckForCausalLM(ArchitectureConfig.from_dict(metadata["config"]))
            self.model.load_state_dict(model_state)
            if self.device.type == "cuda":
                self.model.to(self.device)
            else:
                self.model.to(self.device, dtype=self.dtype)
            self.model.eval()
            self.tokenizer = run["tokenizer"]
            self.context_length = official.infer_context_length(
                self.model, self.tokenizer, args.max_context
            )
            self.pad_token_id = self.tokenizer.eos_id
            self.bos_token_id = self.tokenizer.bos_id
            print(
                f"Loading Speck checkpoint {run['step']:,} on {self.device} as "
                f"{str(self.dtype).removeprefix('torch.')}; "
                f"scoring raw continuations with context length {self.context_length}; "
                f"add_bos={args.add_bos}.",
                flush=True,
            )

        def encode(self, text):
            if not self._is_speck:
                return super().encode(text)
            return self.tokenizer.encode(text, bos=False, eos=False)

        def forward(self, input_ids, attention_mask):
            if not self._is_speck:
                return super().forward(input_ids, attention_mask)
            # Speck's attention and convolution layers are causal, so future pads are unscored.
            return self.model(input_ids)

    official.parse_args = parse_args
    official.run_signature = run_signature
    official.write_outputs = write_outputs
    official.ModelRunner = ModelRunner
    official.main()


if __name__ == "__main__":
    main()
