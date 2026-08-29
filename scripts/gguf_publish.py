"""Build, validate, and publish llama.cpp-compatible Speck GGUF files."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
    snapshot_download,
)
from safetensors.torch import load_file, save_file

from speck.common import base_dir

SOURCE_REPO = "specklabs/Speck1-140M-Instruct"
DESTINATION_REPO = "specklabs/Speck1-140M-Instruct-GGUF"
DEFAULT_QUANTIZATIONS = ("Q4_K_M", "Q5_K_M", "Q8_0")
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_REVISION = "2e88c49c90f0add8796f633fea8c3d65b975f295"
SOURCE_FILES = (
    "LICENSE",
    "LICENSE.tokenizer",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.model",
    "tokenizer_config.json",
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default=SOURCE_REPO, help="source Hugging Face model repository"
    )
    parser.add_argument(
        "--destination",
        default=DESTINATION_REPO,
        help="destination Hugging Face model repository",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="source revision, resolved to a commit before conversion (default: %(default)s)",
    )
    parser.add_argument(
        "--quantization",
        dest="quantizations",
        action="append",
        help="llama.cpp quantization; repeat for multiple values (default: Q4_K_M, Q5_K_M, Q8_0)",
    )
    parser.add_argument(
        "--llama-cpp",
        type=Path,
        default=None,
        help="existing llama.cpp checkout; otherwise use a pinned checkout in the Speck cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="artifact directory; defaults to ~/.cache/speck/gguf/<model>-<revision>",
    )
    parser.add_argument(
        "--private", action="store_true", help="create a private destination repository"
    )
    parser.add_argument("--no-upload", action="store_true", help="build locally without uploading")
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="maximum build, quantization, and inference workers (default: %(default)s)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse and validate existing local artifacts, building only missing variants",
    )
    parser.add_argument("--force", action="store_true", help="replace existing local artifacts")
    return parser.parse_args()


def run(command, *, cwd=None, capture=False, timeout=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    return subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        check=True,
        capture_output=capture,
        stdin=subprocess.DEVNULL,
        text=capture,
        timeout=timeout,
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config):
    if config.get("architectures") != ["SpeckForCausalLM"]:
        raise ValueError("source must use the SpeckForCausalLM architecture")

    blocks = config.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("source config has no Speck blocks")

    hidden_size = None
    intermediate_size = None
    head_dim = None
    kv_heads = None
    layer_types = []
    kernel_sizes = []
    conv_inner_sizes = []
    for index, group in enumerate(blocks):
        if group.get("repeat", 1) != 1 or group.get("weight_sharing", "none") != "none":
            raise ValueError("GGUF conversion requires unshared, non-repeated Speck blocks")
        block = group.get("block", {})
        stages = block.get("stages", [])
        if len(stages) != 2 or any(len(stage.get("branches", [])) != 1 for stage in stages):
            raise ValueError(f"block {index} must contain one operator followed by one SwiGLU")

        width = block.get("hidden_size")
        hidden_size = width if hidden_size is None else hidden_size
        if width != hidden_size:
            raise ValueError("GGUF conversion requires a uniform residual width")

        operator = stages[0]["branches"][0]
        feed_forward = stages[1]["branches"][0]
        if feed_forward.get("kind") != "swiglu":
            raise ValueError(f"block {index} does not end in SwiGLU")
        current_intermediate = feed_forward.get("intermediate_size")
        intermediate_size = current_intermediate if intermediate_size is None else intermediate_size
        if current_intermediate != intermediate_size:
            raise ValueError("GGUF conversion requires a uniform SwiGLU width")

        kind = operator.get("kind")
        if kind == "attention":
            layer_types.append("full_attention")
            current_head_dim = operator.get("head_dim")
            current_kv_heads = operator.get("num_key_value_heads")
            head_dim = current_head_dim if head_dim is None else head_dim
            kv_heads = current_kv_heads if kv_heads is None else kv_heads
            if current_head_dim != head_dim or current_kv_heads != kv_heads:
                raise ValueError("GGUF conversion requires uniform attention dimensions")
            if operator.get("scope", "global") != "global":
                raise ValueError("GGUF conversion only supports global Speck attention")
        elif kind == "gated_causal_conv":
            layer_types.append("conv")
            kernel_sizes.append(operator.get("kernel_size"))
            conv_inner_sizes.append(operator.get("inner_size"))
        else:
            raise ValueError(f"unsupported Speck operator in block {index}: {kind!r}")

    if hidden_size is None or head_dim is None or kv_heads is None:
        raise ValueError("source requires both attention and convolution blocks")
    if hidden_size % head_dim:
        raise ValueError("residual width must be divisible by the attention head dimension")
    if any(not isinstance(size, int) or not 1 <= size <= hidden_size for size in conv_inner_sizes):
        raise ValueError("convolution inner widths must fit in the residual width")
    if any(not isinstance(size, int) or size < 2 for size in kernel_sizes):
        raise ValueError("convolution kernel sizes must be at least two")

    return {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "head_dim": head_dim,
        "kv_heads": kv_heads,
        "layer_types": layer_types,
        "conv_kernel_size": max(kernel_sizes),
    }


def _take(state, consumed, name):
    try:
        tensor = state[name]
    except KeyError as error:
        raise ValueError(f"source checkpoint is missing {name}") from error
    consumed.add(name)
    return tensor


def transform_state(state, config):
    """Transform Speck weights into the operator-equivalent llama.cpp LFM2 layout."""

    layout = validate_config(config)
    hidden_size = layout["hidden_size"]
    consumed = set()
    transformed = {}

    embeddings = _take(state, consumed, "embed_tokens.weight")
    input_adapter = _take(state, consumed, "adapters.0.weight")
    output_adapter = _take(state, consumed, "output_projection.weight")
    transformed["model.embed_tokens.weight"] = (embeddings.float() @ input_adapter.float().T).to(
        embeddings.dtype
    )
    transformed["lm_head.weight"] = (embeddings.float() @ output_adapter.float()).to(
        embeddings.dtype
    )
    transformed["model.embedding_norm.weight"] = _take(state, consumed, "norm.weight")

    for index, (group, layer_type) in enumerate(zip(config["blocks"], layout["layer_types"])):
        source = f"cores.group_{index}_repeat_0"
        target = f"model.layers.{index}"
        operator = f"{source}.stages.0.branches.0"
        feed_forward = f"{source}.stages.1.branches.0"
        transformed[f"{target}.operator_norm.weight"] = _take(
            state, consumed, f"{operator}.norm.weight"
        )
        transformed[f"{target}.post_attention_layernorm.weight"] = _take(
            state, consumed, f"{feed_forward}.norm.weight"
        )
        for source_name, target_name in (
            ("gate_proj", "gate_proj"),
            ("up_proj", "up_proj"),
            ("down_proj", "down_proj"),
        ):
            transformed[f"{target}.mlp.{target_name}.weight"] = _take(
                state, consumed, f"{feed_forward}.operation.{source_name}.weight"
            )

        if layer_type == "full_attention":
            for source_name, target_name in (
                ("q_proj", "q_proj"),
                ("k_proj", "k_proj"),
                ("v_proj", "v_proj"),
                ("o_proj", "out_proj"),
                ("q_norm", "q_norm"),
                ("k_norm", "k_norm"),
            ):
                transformed[f"{target}.self_attn.{target_name}.weight"] = _take(
                    state, consumed, f"{operator}.operation.{source_name}.weight"
                )
            continue

        spec = group["block"]["stages"][0]["branches"][0]
        inner_size = spec["inner_size"]
        kernel_size = spec["kernel_size"]
        input_projection = _take(state, consumed, f"{operator}.operation.input_projection.weight")
        padded_input = input_projection.new_zeros(3 * hidden_size, hidden_size)
        for chunk in range(3):
            start = chunk * inner_size
            padded_input[chunk * hidden_size : chunk * hidden_size + inner_size] = input_projection[
                start : start + inner_size
            ]
        transformed[f"{target}.conv.in_proj.weight"] = padded_input

        output_projection = _take(state, consumed, f"{operator}.operation.output_projection.weight")
        padded_output = output_projection.new_zeros(hidden_size, hidden_size)
        padded_output[:, :inner_size] = output_projection
        transformed[f"{target}.conv.out_proj.weight"] = padded_output

        kernel = _take(state, consumed, f"{operator}.operation.kernel")
        padded_kernel = kernel.new_zeros(hidden_size, 1, layout["conv_kernel_size"])
        padded_kernel[:inner_size, :, -kernel_size:] = kernel
        transformed[f"{target}.conv.conv.weight"] = padded_kernel

    unexpected = sorted(set(state) - consumed)
    if unexpected:
        raise ValueError(f"source checkpoint has unmapped tensors: {unexpected}")
    return transformed, layout


def transformed_config(config, layout):
    return {
        "architectures": ["Lfm2ForCausalLM"],
        "block_auto_adjust_ff_dim": False,
        "block_dim": layout["hidden_size"],
        "block_ff_dim": layout["intermediate_size"],
        "block_ffn_dim_multiplier": 1.0,
        "block_multiple_of": 1,
        "bos_token_id": config["bos_token_id"],
        "conv_L_cache": layout["conv_kernel_size"],
        "eos_token_id": config["eos_token_id"],
        "head_dim": layout["head_dim"],
        "hidden_size": layout["hidden_size"],
        "intermediate_size": layout["intermediate_size"],
        "layer_types": layout["layer_types"],
        "max_position_embeddings": config["max_position_embeddings"],
        "model_type": "lfm2",
        "norm_eps": config["rms_norm_eps"],
        "num_attention_heads": layout["hidden_size"] // layout["head_dim"],
        "num_heads": layout["hidden_size"] // layout["head_dim"],
        "num_hidden_layers": len(layout["layer_types"]),
        "num_key_value_heads": layout["kv_heads"],
        "rope_theta": config["rope_theta"],
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "vocab_size": config["vocab_size"],
    }


def prepare_staging(source_dir, staging_dir):
    config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    state = load_file(source_dir / "model.safetensors", device="cpu")
    transformed, layout = transform_state(state, config)
    del state
    save_file(transformed, staging_dir / "model.safetensors")
    del transformed

    for filename in SOURCE_FILES:
        if filename in {"README.md", "config.json", "model.safetensors"}:
            continue
        source = source_dir / filename
        if source.is_file():
            shutil.copy2(source, staging_dir / filename)
    (staging_dir / "config.json").write_text(
        json.dumps(transformed_config(config, layout), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return layout


def ensure_llama_cpp(requested_path, jobs):
    if requested_path is not None:
        checkout = requested_path.expanduser().resolve()
        if not (checkout / "gguf-py" / "gguf").is_dir():
            raise FileNotFoundError(f"not a llama.cpp checkout: {checkout}")
    else:
        checkout = Path(base_dir()) / "tools" / f"llama.cpp-{LLAMA_CPP_REVISION[:12]}"
        if not checkout.exists():
            checkout.parent.mkdir(parents=True, exist_ok=True)
            run(["git", "clone", "--filter=blob:none", "--no-checkout", LLAMA_CPP_REPO, checkout])
            run(["git", "fetch", "--depth", "1", "origin", LLAMA_CPP_REVISION], cwd=checkout)
            run(["git", "checkout", "--detach", LLAMA_CPP_REVISION], cwd=checkout)
        revision = run(["git", "rev-parse", "HEAD"], cwd=checkout, capture=True).stdout.strip()
        if revision != LLAMA_CPP_REVISION:
            raise RuntimeError(f"cached llama.cpp has unexpected revision {revision}")

    build_dir = checkout / "build"
    quantizer = build_dir / "bin" / "llama-quantize"
    cli = build_dir / "bin" / "llama-cli"
    completion = build_dir / "bin" / "llama-completion"
    if not quantizer.is_file() or not cli.is_file() or not completion.is_file():
        run(
            [
                "cmake",
                "-S",
                checkout,
                "-B",
                build_dir,
                "-DCMAKE_BUILD_TYPE=Release",
                "-DGGML_NATIVE=OFF",
                "-DLLAMA_CURL=OFF",
                "-DLLAMA_BUILD_TESTS=OFF",
            ]
        )
        run(
            [
                "cmake",
                "--build",
                build_dir,
                "--target",
                "llama-quantize",
                "llama-cli",
                "llama-completion",
                "-j",
                str(jobs),
            ]
        )
    return checkout, quantizer, cli, completion


def convert_bf16(staging_dir, output, llama_cpp, model_name):
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    sys.path.insert(0, str(llama_cpp))
    try:
        import gguf
        from conversion.lfm2 import LFM2Model

        class SpeckLFM2Model(LFM2Model):
            model_arch = gguf.MODEL_ARCH.LFM2

            def set_vocab(self):
                self._set_vocab_sentencepiece()

        with torch.inference_mode():
            model = SpeckLFM2Model(
                staging_dir,
                gguf.LlamaFileType.MOSTLY_BF16,
                output,
                model_name=model_name,
            )
            model.write()
    finally:
        sys.path.pop(0)
        sys.path.pop(0)


def smoke_test(cli, completion, model, jobs, conversational):
    command = [
        cli if conversational else completion,
        "--model",
        model,
        "--prompt",
        "What is 2 + 2?" if conversational else "The meaning of life is",
        "--n-predict",
        "4",
        "--temp",
        "0",
        "--ctx-size",
        "256",
        "--threads",
        str(jobs),
        "--threads-batch",
        str(jobs),
        "--simple-io",
    ]
    if conversational:
        command.extend(("--conversation", "--single-turn"))
    else:
        command.append("--no-display-prompt")
    run(command, capture=True, timeout=120)


def transformed_parameter_count(config, layout):
    expected = config.get("expected_parameters")
    embedding_size = config.get("embedding_size")
    vocab_size = config.get("vocab_size")
    if not all(isinstance(value, int) for value in (expected, embedding_size, vocab_size)):
        raise ValueError("source config must declare integer parameter and embedding sizes")

    hidden_size = layout["hidden_size"]
    total = expected
    total -= vocab_size * embedding_size + 2 * embedding_size * hidden_size
    total += 2 * vocab_size * hidden_size
    for group, layer_type in zip(config["blocks"], layout["layer_types"]):
        if layer_type != "conv":
            continue
        spec = group["block"]["stages"][0]["branches"][0]
        inner_size = spec["inner_size"]
        kernel_size = spec["kernel_size"]
        total -= 4 * inner_size * hidden_size + inner_size * kernel_size
        total += 4 * hidden_size * hidden_size + hidden_size * layout["conv_kernel_size"]
    return total


def size_label(size):
    units = ("B", "KB", "MB", "GB")
    value = float(size)
    for unit in units:
        if value < 1000 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1000
    raise AssertionError("unreachable")


def write_release_files(
    output_dir,
    source,
    destination,
    source_revision,
    llama_revision,
    artifacts,
    stored_parameters,
    conversational,
):
    manifest = {
        "format_version": 1,
        "source": {"repo": source, "revision": source_revision},
        "conversion": {
            "runtime_architecture": "lfm2",
            "llama_cpp_revision": llama_revision,
            "quantizations": [artifact["quantization"] for artifact in artifacts],
            "stored_parameters": stored_parameters,
        },
        "files": {
            artifact["path"].name: {
                "bytes": artifact["path"].stat().st_size,
                "sha256": sha256(artifact["path"]),
            }
            for artifact in artifacts
        },
    }
    manifest_path = output_dir / "conversion.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = "\n".join(
        f"| [{item['path'].name}](./{item['path'].name}) | {item['quantization']} | "
        f"{size_label(item['path'].stat().st_size)} |"
        for item in artifacts
    )
    model_name = source.rsplit("/", 1)[-1]
    tags = "  - gguf\n" + ("  - conversational\n" if conversational else "")
    usage = (
        f"llama-cli -hf {destination}:Q4_K_M -cnv"
        if conversational
        else f'llama-completion -hf {destination}:Q4_K_M -p "The meaning of life is" -n 64'
    )
    card = f"""---
license: mit
base_model: {source}
library_name: llama.cpp
pipeline_tag: text-generation
tags:
{tags.rstrip()}
---

# {model_name} GGUF

llama.cpp-compatible GGUF builds of [{source}](https://huggingface.co/{source}), pinned to
source revision `{source_revision}`.

| File | Quantization | Size |
| --- | --- | ---: |
{rows}

## Usage

```bash
{usage}
```

The source Speck architecture and llama.cpp's LFM2 runtime implement the same alternating
attention/short-convolution operators. Conversion folds the 640-to-768 input and 768-to-640
output adapters into the embeddings, zero-pads the 384-wide convolution channels to 768, and
left-pads 3-tap causal kernels to 5 taps. These transformations preserve the model function
apart from normal floating-point and quantization rounding.

The GGUF graph stores {stored_parameters:,} parameters because the source's tied 640-wide embedding and
two adapters become separate 768-wide input and output matrices. This compatibility transform
does not add layers or model capacity.

The conversion was built with llama.cpp revision `{llama_revision}`. Exact checksums and
conversion provenance are in [`conversion.json`](./conversion.json).
"""
    card_path = output_dir / "README.md"
    card_path.write_text(card, encoding="utf-8")
    return card_path, manifest_path


def main():
    args = arguments()
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    if args.resume and args.force:
        raise ValueError("--resume and --force are mutually exclusive")
    quantizations = tuple(args.quantizations or DEFAULT_QUANTIZATIONS)
    if not quantizations or any(not value.strip() for value in quantizations):
        raise ValueError("at least one non-empty quantization is required")
    quantizations = tuple(value.upper() for value in quantizations)
    if len(set(quantizations)) != len(quantizations):
        raise ValueError("quantizations must be unique")

    api = HfApi()
    source_info = api.model_info(args.source, revision=args.revision)
    source_revision = source_info.sha
    if source_revision is None:
        raise RuntimeError("Hugging Face did not resolve the source revision")
    source_dir = Path(
        snapshot_download(
            repo_id=args.source,
            revision=source_revision,
            allow_patterns=list(SOURCE_FILES),
        )
    )
    source_config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    layout = validate_config(source_config)
    stored_parameters = transformed_parameter_count(source_config, layout)
    tokenizer_config_path = source_dir / "tokenizer_config.json"
    tokenizer_config = (
        json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        if tokenizer_config_path.is_file()
        else {}
    )
    conversational = bool(tokenizer_config.get("chat_template"))
    output_dir = args.output_dir or (
        Path(base_dir()) / "gguf" / f"{args.source.rsplit('/', 1)[-1]}-{source_revision[:12]}"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = args.source.rsplit("/", 1)[-1]
    intended = [output_dir / f"{basename}-BF16.gguf"] + [
        output_dir / f"{basename}-{quantization}.gguf" for quantization in quantizations
    ]
    existing = [path for path in intended if path.exists()]
    if existing and not (args.force or args.resume):
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"local artifacts already exist (use --resume or --force): {names}")
    if args.force:
        for path in existing:
            path.unlink()

    llama_cpp, quantizer, cli, completion = ensure_llama_cpp(args.llama_cpp, args.jobs)
    llama_revision = run(["git", "rev-parse", "HEAD"], cwd=llama_cpp, capture=True).stdout.strip()
    bf16 = intended[0]
    if not bf16.is_file():
        with tempfile.TemporaryDirectory(prefix="speck-gguf-", dir=output_dir) as temporary:
            staging_dir = Path(temporary)
            prepare_staging(source_dir, staging_dir)
            convert_bf16(staging_dir, bf16, llama_cpp, basename)

    artifacts = [{"path": bf16, "quantization": "BF16"}]
    smoke_test(cli, completion, bf16, args.jobs, conversational)
    for quantization, output in zip(quantizations, intended[1:]):
        if not output.is_file():
            run([quantizer, bf16, output, quantization, str(args.jobs)])
        smoke_test(cli, completion, output, args.jobs, conversational)
        artifacts.append({"path": output, "quantization": quantization})

    card, manifest = write_release_files(
        output_dir,
        args.source,
        args.destination,
        source_revision,
        llama_revision,
        artifacts,
        stored_parameters,
        conversational,
    )
    if args.no_upload:
        print(f"Artifacts are ready in {output_dir}")
        return

    api.create_repo(
        repo_id=args.destination,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    operations = [
        CommitOperationAdd(path_in_repo=item["path"].name, path_or_fileobj=item["path"])
        for item in artifacts
    ]
    operations.extend(
        [
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card),
            CommitOperationAdd(path_in_repo="conversion.json", path_or_fileobj=manifest),
        ]
    )
    for filename in ("LICENSE", "LICENSE.tokenizer"):
        path = source_dir / filename
        if path.is_file():
            operations.append(CommitOperationAdd(path_in_repo=filename, path_or_fileobj=path))
    commit = api.create_commit(
        repo_id=args.destination,
        repo_type="model",
        operations=operations,
        commit_message=f"Publish GGUF variants from {source_revision[:12]}",
    )
    print(commit.commit_url)


if __name__ == "__main__":
    main()
