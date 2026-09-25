"""Export a completed Speck SFT checkpoint and publish it to Hugging Face."""

import argparse
import ast
import hashlib
import json
import os
import shutil
from importlib.resources import files
from pathlib import Path

import torch
from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download
from safetensors.torch import load_file, save_file

from speck.model import build_model
from speck.model.architecture import (
    ArchitectureConfig,
    AttentionSpec,
    SwiGLUSpec,
)
from speck.operations.runtime import base_dir
from speck.provenance.io import atomic_json
from speck.tokenization.chat import ChatTokenizer
from speck.tokenization.tokenizer import Tokenizer
from speck.training.checkpoint import (
    checkpoint_identity,
    is_assistant_checkpoint,
    latest,
    load_model,
)

CODE_REPO = "specklabs/Speck1-140M-Instruct"
CODE_REVISION = "16ad80599d499490b70317770a84a18466719bba"
LICENSE_FILES = ("LICENSE", "LICENSE.tokenizer")
TOKENIZER_FILES = (
    "chat_template.jinja",
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "tokenizer_metadata.json",
)
PACKAGE_SOURCE = Path(str(files("speck")))
PADDING_SOURCE = PACKAGE_SOURCE / "transformers_padding.py"
ARCHITECTURE_SOURCE = PACKAGE_SOURCE / "model" / "architecture.py"
NATIVE_MODEL_SOURCE = PACKAGE_SOURCE / "model" / "__init__.py"
NATIVE_SOURCES = (
    PACKAGE_SOURCE / "training" / "optimizers.py",
    PACKAGE_SOURCE / "model" / "state.py",
    PACKAGE_SOURCE / "model" / "layers.py",
    NATIVE_MODEL_SOURCE,
)
CURRENT_CONFIGURATION_SOURCE = PACKAGE_SOURCE / "transformers_configuration.py"
CURRENT_MODELING_SOURCE = PACKAGE_SOURCE / "transformers_modeling.py"
CURRENT_TOKENIZATION_SOURCE = PACKAGE_SOURCE / "transformers_tokenization.py"
PADDING_DESTINATION = "padding_speck.py"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="completed SFT checkpoint directory",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="checkpoint step; defaults to the latest completed step",
    )
    parser.add_argument("--repo", required=True, help="destination Hugging Face model repository")
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
    if not is_assistant_checkpoint(metadata) or metadata.get("step") != step:
        raise ValueError("checkpoint is not a matching assistant checkpoint")
    if metadata["training_phase"] == "rl":
        if metadata.get("settings", {}).get("steps") != step:
            raise ValueError("checkpoint has not completed its configured RL steps")
        return metadata
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
        f"native.{name}": tensor.detach().to(torch.bfloat16).contiguous()
        for name, tensor in state.items()
        if name != "lm_head.weight"
    }


def prepare_current_release_code(output_dir):
    """Ship the current native implementation behind a small Transformers wrapper."""

    shutil.copy2(ARCHITECTURE_SOURCE, output_dir / "architecture_speck.py")
    sections = []
    identities = {}
    bundled = {"speck.model.state", "speck.model.layers", "speck.training.optimizers"}
    for path in NATIVE_SOURCES:
        source = path.read_text(encoding="utf-8")
        identities[str(path.relative_to(PACKAGE_SOURCE))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        lines = source.splitlines(keepends=True)
        for node in reversed(ast.parse(source).body):
            if isinstance(node, ast.ImportFrom):
                if node.module in bundled:
                    lines[node.lineno - 1 : node.end_lineno] = []
                elif node.module == "speck.model.architecture":
                    names = ", ".join(alias.name for alias in node.names)
                    lines[node.lineno - 1 : node.end_lineno] = [
                        f"from .architecture_speck import ({names})\n"
                    ]
                elif node.module and node.module.startswith("speck."):
                    raise ValueError(f"unbundled native dependency: {node.module}")
        sections.append("".join(lines))
    native = "\n\n".join(sections)
    compile(native, "native_speck.py", "exec")
    (output_dir / "native_speck.py").write_text(native, encoding="utf-8")
    shutil.copy2(CURRENT_CONFIGURATION_SOURCE, output_dir / "configuration_speck.py")
    shutil.copy2(CURRENT_MODELING_SOURCE, output_dir / "modeling_speck.py")
    shutil.copy2(CURRENT_TOKENIZATION_SOURCE, output_dir / "tokenization_speck.py")
    shutil.copy2(PADDING_SOURCE, output_dir / PADDING_DESTINATION)
    for path in (
        ARCHITECTURE_SOURCE,
        CURRENT_CONFIGURATION_SOURCE,
        CURRENT_MODELING_SOURCE,
        CURRENT_TOKENIZATION_SOURCE,
        PADDING_SOURCE,
    ):
        identities[str(path.relative_to(PACKAGE_SOURCE))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    (output_dir / "native_sources.json").write_text(
        json.dumps(identities, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def prepare_export(checkpoint_dir, step, output_dir, metadata, state=None):
    building = output_dir.with_name(output_dir.name + ".building")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    try:
        state = load_model(checkpoint_dir, step, "cpu") if state is None else state
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
        provenance = {
            "format": "speck_export_source",
            "format_version": 1,
            "type": "sft_checkpoint",
            "checkpoint": checkpoint_identity(checkpoint_dir, step),
        }
        (building / "speck_source.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
                allow_patterns=list(LICENSE_FILES),
            )
        )
        for filename in LICENSE_FILES:
            shutil.copy2(code_dir / filename, building / filename)
        prepare_current_release_code(building)
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
    if "native.lm_head.weight" in state or not state:
        raise ValueError("exported Safetensors tied-weight layout is invalid")
    invalid_dtypes = [name for name, tensor in state.items() if tensor.dtype != torch.bfloat16]
    if invalid_dtypes:
        raise ValueError("exported model has invalid compute dtypes")
    if not (output_dir / PADDING_DESTINATION).is_file():
        raise ValueError("export is missing Transformers padding support")
    if not (output_dir / "speck_source.json").is_file():
        raise ValueError("export is missing checkpoint provenance")
    required_code = {
        "architecture_speck.py",
        "configuration_speck.py",
        "modeling_speck.py",
        "native_speck.py",
        PADDING_DESTINATION,
    }
    missing_code = sorted(name for name in required_code if not (output_dir / name).is_file())
    if missing_code:
        raise ValueError(f"export is missing current model code: {missing_code}")
    if (output_dir / "README.md").exists():
        raise ValueError("export must not contain a model card")


def validate_parity(output_dir, state, metadata):
    """Gate an exported checkpoint on native/Transformers logit identity."""

    from transformers import AutoModelForCausalLM

    architecture = ArchitectureConfig.from_dict(metadata["config"])
    native = build_model(
        architecture.export(),
        architecture.vocab_size,
        architecture.bos_token_id,
        architecture.eos_token_id,
    )
    native.load_state_dict(state)
    native.to(torch.bfloat16)
    native.eval()
    exported = AutoModelForCausalLM.from_pretrained(
        output_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    exported.eval()
    expected_parameters = native.parameter_count()
    exported_parameters = sum(parameter.numel() for parameter in exported.parameters())
    if exported_parameters != expected_parameters:
        raise ValueError(
            f"Transformers export has {exported_parameters:,} parameters, "
            f"expected {expected_parameters:,}"
        )
    generator = torch.Generator().manual_seed(42)
    tokens = torch.randint(0, architecture.vocab_size, (2, 8), generator=generator)
    with torch.no_grad():
        native_logits = native(tokens)
        exported_logits = exported(input_ids=tokens, use_cache=False).logits
    torch.testing.assert_close(exported_logits, native_logits, rtol=2e-2, atol=2e-2)
    prompt = tokens[:1, :4]
    with torch.no_grad():
        prefill = exported(input_ids=prompt, use_cache=True)
        next_token = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
        incremental = exported(
            input_ids=next_token,
            past_key_values=prefill.past_key_values,
            use_cache=True,
        ).logits[:, -1]
        reference = native(torch.cat((prompt, next_token), dim=1))[:, -1]
    # Compare the same execution path for wrapper identity. BF16 GEMMs with
    # different sequence shapes can round differently even within the native model.
    with torch.no_grad():
        native_cache = native.state(batch_size=1, device=prompt.device, dtype=torch.bfloat16)
        native(prompt, state=native_cache)
        native_incremental = native(next_token, state=native_cache)[:, -1]
    torch.testing.assert_close(incremental, native_incremental, rtol=2e-2, atol=2e-2)
    generated = exported.generate(
        prompt,
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=architecture.eos_token_id,
    )
    # Independently check cache/full-pass semantics in FP32 on the same rounded
    # release weights. Retain the observed BF16 cross-path drift in the report.
    native.float()
    exported.float()
    with torch.no_grad():
        fp32_prefill = exported(input_ids=prompt, use_cache=True)
        fp32_incremental = exported(
            input_ids=next_token,
            past_key_values=fp32_prefill.past_key_values,
            use_cache=True,
        ).logits[:, -1]
        fp32_reference = native(torch.cat((prompt, next_token), dim=1))[:, -1]
    torch.testing.assert_close(fp32_incremental, fp32_reference, rtol=1e-4, atol=1e-4)
    report = {
        "format": "speck_export_parity",
        "format_version": 2,
        "passed": True,
        "parameters": expected_parameters,
        "compute_dtype": "bfloat16",
        "logits_max_absolute_error": (exported_logits - native_logits).abs().max().item(),
        "incremental_logits_max_absolute_error": (incremental - reference).abs().max().item(),
        "native_export_cached_max_absolute_error": (incremental - native_incremental)
        .abs()
        .max()
        .item(),
        "fp32_cached_full_max_absolute_error": (fp32_incremental - fp32_reference)
        .abs()
        .max()
        .item(),
        "comparison_policy": {
            "native_export_bf16": {"rtol": 2e-2, "atol": 2e-2, "paths": ["full", "cached"]},
            "cached_full_fp32": {"rtol": 1e-4, "atol": 1e-4},
            "bf16_cached_full": "Measured separately; not the wrapper identity comparison.",
        },
        "generation_smoke_new_tokens": generated.size(1) - prompt.size(1),
        "tokens_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
    }
    path = output_dir / "speck_parity.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def validate_tokenizer_parity(output_dir, metadata, *, base_fingerprint=None):
    """Check exported text/chat token IDs against the tokenizer used during training."""

    from transformers import AutoTokenizer

    output_dir = Path(output_dir)
    base = Tokenizer(output_dir / "tokenizer.model")
    chat_metadata = metadata.get("resolved", {}).get("tokenizer", {})
    is_chat = is_assistant_checkpoint(metadata)
    expected = chat_metadata["base_fingerprint"] if is_chat else base_fingerprint
    if expected is None or base.fingerprint() != expected:
        raise ValueError("export tokenizer differs from the checkpoint")
    exported = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    for token, identity in (
        (exported.bos_token, exported.bos_token_id),
        (exported.eos_token, exported.eos_token_id),
    ):
        if not token or exported.decode([identity], skip_special_tokens=False) != token:
            raise ValueError("exported control token decoding loses its visible spelling")
        if exported.decode([identity], skip_special_tokens=True) != "":
            raise ValueError("exported control tokens are not skipped when requested")
    texts = [
        "Hello, world!",
        "\nA second line.\n",
        "def f(x):\n    return x + 1",
        "İstanbul café 日本語",
    ]
    for text in texts:
        if exported.encode(text, add_special_tokens=True) != base.encode(text, bos=True):
            raise ValueError("exported text token IDs differ from native tokenization")
    chat_cases = 0
    if is_chat:
        native = ChatTokenizer.from_metadata(base, chat_metadata)
        if json.loads((output_dir / "tokenizer_metadata.json").read_text()) != chat_metadata:
            raise ValueError("exported chat metadata differs from the checkpoint")
        if (output_dir / "chat_template.jinja").read_text() != native.chat_template:
            raise ValueError("exported chat template differs from the checkpoint")
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "One plus one?"},
            {"role": "assistant", "content": "Context", "weight": 0},
            {"role": "user", "content": "Try again."},
            {"role": "assistant", "content": "Two.", "weight": 1},
        ]
        if native.format_version == 1:
            messages[2].pop("weight")
        for conversation, generation in ((messages, False), (messages[:-1], True)):
            expected_ids, _ = native.encode_messages(conversation, add_generation_prompt=generation)
            actual = exported.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=generation, return_dict=True
            )
            if actual["input_ids"] != expected_ids:
                raise ValueError("exported chat token IDs differ from native tokenization")
            chat_cases += 1
    report = {
        "format": "speck_tokenizer_export_parity",
        "format_version": 1,
        "passed": True,
        "base_fingerprint": base.fingerprint(),
        "text_cases": len(texts),
        "control_token_cases": 2,
        "chat_cases": chat_cases,
        "chat_format_version": chat_metadata.get("format_version") if is_chat else None,
    }
    atomic_json(output_dir / "tokenizer_parity.json", report)
    return report


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
    state = load_model(checkpoint_dir, step, "cpu")
    prepare_export(checkpoint_dir, step, output_dir, metadata, state)
    validate_export(output_dir, metadata)
    validate_parity(output_dir, state, metadata)
    validate_tokenizer_parity(output_dir, metadata)
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
