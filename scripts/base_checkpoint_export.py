"""Export a pretraining checkpoint for local Transformers evaluation."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

from scripts.model_publish import (
    patch_generation_source,
    release_config,
    release_state,
    validate_export,
)
from speck.architecture import ArchitectureConfig
from speck.checkpoint import checkpoint_identity, latest, load_model
from speck.model import build_model

TEMPLATE_REPO = "specklabs/Speck1-140M"
TEMPLATE_REVISION = "155b759545645cc694545fab85cd7d4c385fd965"
TEMPLATE_FILES = (
    "LICENSE",
    "LICENSE.tokenizer",
    "configuration_speck.py",
    "modeling_speck.py",
    "padding_speck.py",
    "tokenization_speck.py",
    "tokenizer.model",
    "tokenizer_config.json",
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_checkpoint_metadata(checkpoint_dir, step):
    path = checkpoint_dir / f"metadata_{step:06d}.json"
    complete = checkpoint_dir / f"complete_{step:06d}"
    if not path.is_file() or not complete.is_file():
        raise FileNotFoundError(f"checkpoint {step} is incomplete")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("step") != step or metadata.get("training_phase") == "sft":
        raise ValueError("checkpoint is not a matching pretraining checkpoint")
    return metadata


def load_source(checkpoint_dir, step):
    step = step if step is not None else latest(checkpoint_dir)
    if step is None:
        raise FileNotFoundError(f"no completed checkpoint in {checkpoint_dir}")
    metadata = load_checkpoint_metadata(checkpoint_dir, step)
    provenance = {
        "format": "speck_export_source",
        "format_version": 1,
        "type": "checkpoint",
        "checkpoint": checkpoint_identity(checkpoint_dir, step),
    }
    return load_model(checkpoint_dir, step, "cpu"), metadata, f"step {step:,}", provenance


def export(state, output_dir, metadata, provenance):
    building = output_dir.with_name(output_dir.name + ".building")
    shutil.rmtree(building, ignore_errors=True)
    building.mkdir(parents=True)
    try:
        template = Path(
            snapshot_download(
                repo_id=TEMPLATE_REPO,
                revision=TEMPLATE_REVISION,
                allow_patterns=list(TEMPLATE_FILES),
            )
        )
        for filename in TEMPLATE_FILES:
            source = template / filename
            if filename == "modeling_speck.py":
                patched = patch_generation_source(source.read_text(encoding="utf-8"))
                (building / filename).write_text(patched, encoding="utf-8")
            else:
                shutil.copy2(source, building / filename)

        save_file(release_state(state), building / "model.safetensors", metadata={"format": "pt"})
        config = release_config(metadata)
        (building / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        generation = {
            "_from_model_config": True,
            "bos_token_id": config["bos_token_id"],
            "eos_token_id": config["eos_token_id"],
            "transformers_version": config["transformers_version"],
        }
        (building / "generation_config.json").write_text(
            json.dumps(generation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (building / "speck_source.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(building, output_dir)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)
        raise


def validate_parity(output_dir, state, metadata):
    """Gate evaluation on native/Transformers logits and parameter identity."""

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
    report = {
        "format": "speck_export_parity",
        "format_version": 1,
        "passed": True,
        "parameters": expected_parameters,
        "compute_dtype": "bfloat16",
        "logits_max_absolute_error": (exported_logits - native_logits).abs().max().item(),
        "tokens_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
    }
    path = output_dir / "speck_parity.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main():
    args = arguments()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    state, metadata, source, provenance = load_source(checkpoint_dir, args.step)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"export already exists (use --force): {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    export(state, output_dir, metadata, provenance)
    validate_export(output_dir, metadata)
    validate_parity(output_dir, state, metadata)
    if json.loads((output_dir / "speck_source.json").read_text(encoding="utf-8")) != provenance:
        raise ValueError("exported source provenance does not match its input")
    print(f"Exported {source} to {output_dir}")


if __name__ == "__main__":
    main()
