"""Export a completed Speck SFT checkpoint and publish it to Hugging Face."""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download
from safetensors.torch import load_file, save_file

from speck.architecture import ArchitectureConfig, AttentionSpec, SwiGLUSpec
from speck.checkpoint import latest, load_model
from speck.common import base_dir

DEFAULT_CHECKPOINT = Path(base_dir()) / "checkpoints" / "Speck1.1-140M-Instruct"
DEFAULT_REPO = "specklabs/Speck1.1-140M-Instruct"
CODE_REPO = "specklabs/Speck1-140M-Instruct"
CODE_REVISION = "16ad80599d499490b70317770a84a18466719bba"
CODE_FILES = (
    "LICENSE",
    "LICENSE.tokenizer",
    "configuration_speck.py",
    "modeling_speck.py",
    "tokenization_speck.py",
)
TOKENIZER_FILES = (
    "chat_template.jinja",
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "tokenizer_metadata.json",
)
PADDING_SOURCE = Path(__file__).resolve().parents[1] / "speck" / "transformers_padding.py"
PADDING_DESTINATION = "padding_speck.py"
MODEL_IMPORT = "from .configuration_speck import SpeckConfig\n"
PATCHED_MODEL_IMPORT = MODEL_IMPORT + "from .padding_speck import validate_right_padding\n"
MODEL_FORWARD_SETUP = """        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if output_attentions:
            raise ValueError("Speck does not expose attention weights")
        if output_hidden_states:
            raise ValueError("Speck does not expose per-layer hidden states")
        if attention_mask is not None and not torch.all(attention_mask == 1):
            raise ValueError("Speck does not support padded inputs")

        use_cache = self.config.use_cache if use_cache is None else use_cache
        if past_key_values is not None and not isinstance(
            past_key_values, SequenceState
        ):
            raise TypeError("Speck requires its native SequenceState cache")
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        batch_size = (
            input_ids.size(0) if input_ids is not None else inputs_embeds.size(0)
        )
        if use_cache and past_key_values is None:
            past_key_values = self.state(
                batch_size=batch_size,
                device=self.device,
                dtype=self.dtype,
            )

        length = input_ids.size(1) if input_ids is not None else inputs_embeds.size(1)
"""
PATCHED_MODEL_FORWARD_SETUP = """        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if output_attentions:
            raise ValueError("Speck does not expose attention weights")
        if output_hidden_states:
            raise ValueError("Speck does not expose per-layer hidden states")

        values = input_ids if input_ids is not None else inputs_embeds
        batch_size, length = values.shape[:2]
        has_padding = validate_right_padding(attention_mask, batch_size, length)
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if has_padding and use_cache:
            raise ValueError("right-padded inputs require use_cache=False")
        if past_key_values is not None and not isinstance(
            past_key_values, SequenceState
        ):
            raise TypeError("Speck requires its native SequenceState cache")
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        if use_cache and past_key_values is None:
            past_key_values = self.state(
                batch_size=batch_size,
                device=self.device,
                dtype=self.dtype,
            )

"""
MODEL_POSITION_CHECK = """        if position_ids is not None:
            expected = expected_positions.unsqueeze(0).expand(batch_size, -1)
            if not torch.equal(position_ids, expected):
                raise ValueError(
                    "position_ids does not match the unpadded Speck sequence"
                )
"""
PATCHED_MODEL_POSITION_CHECK = """        if position_ids is not None:
            expected = expected_positions.unsqueeze(0).expand(batch_size, -1)
            valid = (
                attention_mask.bool()
                if has_padding
                else torch.ones_like(expected, dtype=torch.bool)
            )
            if position_ids.shape != expected.shape or not torch.equal(
                position_ids[valid], expected[valid]
            ):
                raise ValueError("position_ids does not match the Speck sequence")
"""


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="completed SFT checkpoint directory (default: %(default)s)",
    )
    parser.add_argument(
        "--step", type=int, default=None, help="checkpoint step; defaults to latest"
    )
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, help="destination Hugging Face model repository"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="export directory; defaults to ~/.cache/speck/releases/<repo>",
    )
    parser.add_argument(
        "--expected-epochs",
        type=int,
        default=None,
        help="reject a checkpoint trained for a different number of epochs",
    )
    parser.add_argument("--private", action="store_true", help="create a private repository")
    parser.add_argument("--no-upload", action="store_true", help="export without uploading")
    parser.add_argument("--force", action="store_true", help="replace an existing local export")
    return parser.parse_args()


def load_metadata(checkpoint_dir, step):
    complete = checkpoint_dir / f"complete_{step:06d}"
    path = checkpoint_dir / f"metadata_{step:06d}.json"
    if not complete.is_file() or not path.is_file():
        raise FileNotFoundError(f"checkpoint {step} is incomplete")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("training_phase") != "sft" or metadata.get("step") != step:
        raise ValueError("checkpoint is not a matching SFT checkpoint")
    resolved = metadata.get("resolved", {})
    epochs = resolved.get("epochs")
    if metadata.get("data_state", {}).get("epoch") != epochs or resolved.get("steps") != step:
        raise ValueError("checkpoint has not completed its configured training epochs")
    return metadata


def release_config(metadata):
    architecture = ArchitectureConfig.from_dict(metadata["config"])
    settings = architecture.export()
    attention = next(
        branch
        for invocation in architecture.execution_plan
        for stage in invocation.block.stages
        for branch in stage.branches
        if isinstance(branch, AttentionSpec)
    )
    intermediate_sizes = {
        branch.intermediate_size
        for invocation in architecture.execution_plan
        for stage in invocation.block.stages
        for branch in stage.branches
        if isinstance(branch, SwiGLUSpec)
    }
    hidden_sizes = {invocation.block.hidden_size for invocation in architecture.execution_plan}
    if len(intermediate_sizes) != 1 or len(hidden_sizes) != 1:
        raise ValueError("Transformers release requires uniform hidden and SwiGLU dimensions")
    expected_parameters = metadata["resolved"].get("parameters")
    if not isinstance(expected_parameters, int) or expected_parameters < 1:
        raise ValueError("checkpoint metadata has no parameter count")
    settings.update(
        {
            "architectures": ["SpeckForCausalLM"],
            "auto_map": {
                "AutoConfig": "configuration_speck.SpeckConfig",
                "AutoModelForCausalLM": "modeling_speck.SpeckForCausalLM",
            },
            "dtype": "bfloat16",
            "expected_parameters": expected_parameters,
            "head_dim": attention.head_dim,
            "hidden_act": "silu",
            "hidden_size": hidden_sizes.pop(),
            "intermediate_size": intermediate_sizes.pop(),
            "is_decoder": True,
            "is_encoder_decoder": False,
            "model_type": "speck",
            "num_attention_heads": architecture.execution_plan[0].block.hidden_size
            // attention.head_dim,
            "num_hidden_layers": architecture.logical_depth,
            "num_key_value_heads": attention.num_key_value_heads,
            "num_logits_to_keep": 1,
            "pad_token_id": None,
            "tie_word_embeddings": True,
            "transformers_version": "5.1.0",
            "use_cache": True,
        }
    )
    return settings


def release_state(state):
    required = {"embed_tokens.weight", "lm_head.weight"}
    if not required <= set(state):
        raise ValueError("checkpoint is missing tied token embeddings")
    if not torch.equal(state["embed_tokens.weight"], state["lm_head.weight"]):
        raise ValueError("checkpoint input and output embeddings are not tied")
    return {
        name: tensor.detach().to(torch.bfloat16).contiguous()
        for name, tensor in state.items()
        if name != "lm_head.weight"
    }


def patch_modeling_source(source):
    replacements = (
        (MODEL_IMPORT, PATCHED_MODEL_IMPORT, "configuration import"),
        (MODEL_FORWARD_SETUP, PATCHED_MODEL_FORWARD_SETUP, "forward setup"),
        (MODEL_POSITION_CHECK, PATCHED_MODEL_POSITION_CHECK, "position check"),
    )
    for original, replacement, label in replacements:
        if source.count(original) != 1:
            raise ValueError(f"pinned Transformers source has unexpected {label}")
        source = source.replace(original, replacement)
    compile(source, "modeling_speck.py", "exec")
    return source


def prepare_release_code(code_dir, output_dir):
    for filename in CODE_FILES:
        source = code_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Transformers source is missing {filename}")
        if filename == "modeling_speck.py":
            patched = patch_modeling_source(source.read_text(encoding="utf-8"))
            (output_dir / filename).write_text(patched, encoding="utf-8")
        else:
            shutil.copy2(source, output_dir / filename)
    shutil.copy2(PADDING_SOURCE, output_dir / PADDING_DESTINATION)


def prepare_export(checkpoint_dir, step, output_dir, metadata):
    building = output_dir.with_name(output_dir.name + ".building")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    try:
        state = load_model(checkpoint_dir, step, "cpu")
        exported = release_state(state)
        expected_parameters = metadata["resolved"]["parameters"]
        actual_parameters = sum(tensor.numel() for tensor in exported.values())
        if actual_parameters != expected_parameters:
            raise ValueError(
                f"checkpoint has {actual_parameters:,} parameters, expected {expected_parameters:,}"
            )
        save_file(exported, building / "model.safetensors", metadata={"format": "pt"})
        del exported, state

        config = release_config(metadata)
        (building / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        generation = {
            "_from_model_config": True,
            "bos_token_id": config["bos_token_id"],
            "eos_token_id": config["eos_token_id"],
            "transformers_version": "5.1.0",
        }
        (building / "generation_config.json").write_text(
            json.dumps(generation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        tokenizer_dir = checkpoint_dir / "tokenizer"
        for filename in TOKENIZER_FILES:
            source = tokenizer_dir / filename
            if not source.is_file():
                raise FileNotFoundError(f"checkpoint tokenizer is missing {filename}")
            shutil.copy2(source, building / filename)
        tokenizer_config_path = building / "tokenizer_config.json"
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        tokenizer_config["auto_map"] = {
            "AutoTokenizer": ["tokenization_speck.SpeckTokenizer", None]
        }
        tokenizer_config["tokenizer_class"] = "SpeckTokenizer"
        tokenizer_config_path.write_text(
            json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        code_dir = Path(
            snapshot_download(
                repo_id=CODE_REPO,
                revision=CODE_REVISION,
                allow_patterns=list(CODE_FILES),
            )
        )
        prepare_release_code(code_dir, building)
        if (building / "README.md").exists():
            raise RuntimeError("model-card-free export unexpectedly contains README.md")
        os.replace(building, output_dir)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise


def validate_export(output_dir, metadata):
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    if config != release_config(metadata):
        raise ValueError("exported config does not match checkpoint metadata")
    state = load_file(output_dir / "model.safetensors", device="cpu")
    if "lm_head.weight" in state or not state:
        raise ValueError("exported Safetensors tied-weight layout is invalid")
    if any(tensor.dtype != torch.bfloat16 for tensor in state.values()):
        raise ValueError("exported model is not entirely BF16")
    if not (output_dir / PADDING_DESTINATION).is_file():
        raise ValueError("export is missing Transformers padding support")
    if (output_dir / "README.md").exists():
        raise ValueError("export must not contain a model card")


def main():
    args = arguments()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    step = args.step if args.step is not None else latest(checkpoint_dir)
    if step is None:
        raise FileNotFoundError(f"no completed checkpoint in {checkpoint_dir}")
    metadata = load_metadata(checkpoint_dir, step)
    epochs = metadata["resolved"]["epochs"]
    if args.expected_epochs is not None and epochs != args.expected_epochs:
        raise ValueError(f"checkpoint trained for {epochs} epochs, expected {args.expected_epochs}")

    output_dir = args.output_dir or Path(base_dir()) / "releases" / args.repo.replace("/", "--")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"export already exists (use --force): {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    prepare_export(checkpoint_dir, step, output_dir, metadata)
    validate_export(output_dir, metadata)
    unit = "epoch" if epochs == 1 else "epochs"
    print(f"Exported step {step:,} ({epochs} {unit}) to {output_dir}")
    if args.no_upload:
        return

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    files = sorted(path for path in output_dir.iterdir() if path.is_file())
    if any(path.name == "README.md" for path in files):
        raise RuntimeError("refusing to upload a model card")
    commit = api.create_commit(
        repo_id=args.repo,
        repo_type="model",
        operations=[
            CommitOperationAdd(path_in_repo=path.name, path_or_fileobj=path) for path in files
        ],
        commit_message=f"Publish checkpoint step {step}",
    )
    print(commit.commit_url)


if __name__ == "__main__":
    main()
