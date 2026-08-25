"""Validate and publish a code-only Transformers compatibility update."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, snapshot_download

from scripts.model_publish import PADDING_DESTINATION, PADDING_SOURCE, patch_modeling_source
from speck.common import base_dir

DEFAULT_REPO = "specklabs/Speck1-140M"
DEFAULT_SOURCE_REVISION = "32675011a75e3bb3f180983a0014de10d1fa6693"
DEFAULT_WEIGHTS_SHA256 = "199a77b5564868d0b2a03e4f59eb9e58615a2ece947c0618c4289c0ef1c6daf0"
WEIGHTS_FILE = "model.safetensors"
CODE_FILES = ("modeling_speck.py", PADDING_DESTINATION)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face model repository")
    parser.add_argument(
        "--source-revision",
        default=DEFAULT_SOURCE_REVISION,
        help="immutable revision containing the Transformers source to patch",
    )
    parser.add_argument(
        "--expected-weights-sha256",
        default=DEFAULT_WEIGHTS_SHA256,
        help="refuse to update a repository with different model weights",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="generated code directory under the Speck cache by default",
    )
    parser.add_argument("--no-upload", action="store_true", help="validate without uploading")
    parser.add_argument("--force", action="store_true", help="replace an existing output directory")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def weight_sha256(files):
    weight = next((item for item in files if item.path == WEIGHTS_FILE), None)
    if weight is None or weight.lfs is None:
        raise ValueError(f"repository has no LFS metadata for {WEIGHTS_FILE}")
    return weight.lfs.sha256


def prepare_code_update(source_path, output_dir):
    output_dir.mkdir(parents=True)
    patched = patch_modeling_source(source_path.read_text(encoding="utf-8"))
    (output_dir / "modeling_speck.py").write_text(patched, encoding="utf-8")
    shutil.copy2(PADDING_SOURCE, output_dir / PADDING_DESTINATION)
    return {filename: sha256(output_dir / filename) for filename in CODE_FILES}


def shadow_snapshot(snapshot, code_dir, parent):
    shadow = Path(tempfile.mkdtemp(prefix="speck-model-code-", dir=parent))
    for source in snapshot.iterdir():
        if source.is_file() and source.name not in CODE_FILES:
            (shadow / source.name).symlink_to(source)
    for filename in CODE_FILES:
        shutil.copy2(code_dir / filename, shadow / filename)
    return shadow


def validate_transformers_snapshot(snapshot, expected_parameters):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("validation requires transformers==5.1.0") from error

    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        trust_remote_code=True,
        dtype=torch.float32,
    ).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != expected_parameters:
        raise ValueError(f"loaded model has {parameters:,} parameters, expected {expected_parameters:,}")
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("model and tokenizer vocabulary sizes differ")

    first = torch.tensor([[1, 4, 5, 6]])
    second = torch.tensor([[1, 7]])
    batch = torch.tensor([[1, 4, 5, 6], [1, 7, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    positions = torch.tensor([[0, 1, 2, 3], [0, 1, 1, 1]])
    with torch.inference_mode():
        first_logits = model(first, use_cache=False).logits
        second_logits = model(second, use_cache=False).logits
        batch_logits = model(
            batch,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
        ).logits
    torch.testing.assert_close(batch_logits[0], first_logits[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(batch_logits[1, :2], second_logits[0], rtol=1e-4, atol=1e-4)

    try:
        model(batch, attention_mask=mask, use_cache=True)
    except ValueError as error:
        if "require use_cache=False" not in str(error):
            raise
    else:
        raise ValueError("right-padded cached inference was not rejected")


def main():
    args = arguments()
    if len(args.source_revision) != 40:
        raise ValueError("source revision must be a full commit hash")
    api = HfApi()
    source_info = api.model_info(args.repo, revision=args.source_revision)
    if source_info.sha != args.source_revision:
        raise ValueError("source revision did not resolve exactly")
    current_info = api.model_info(args.repo)
    if current_info.sha is None:
        raise RuntimeError("Hugging Face did not resolve the current repository revision")
    current_files = list(
        api.list_repo_tree(
            args.repo,
            revision=current_info.sha,
            repo_type="model",
            recursive=True,
            expand=True,
        )
    )
    current_weights = weight_sha256(current_files)
    if current_weights != args.expected_weights_sha256:
        raise ValueError(
            f"repository weights are {current_weights}, expected {args.expected_weights_sha256}"
        )

    source_path = Path(
        hf_hub_download(args.repo, "modeling_speck.py", revision=args.source_revision)
    )
    current_path = Path(hf_hub_download(args.repo, "modeling_speck.py", revision=current_info.sha))
    if source_path.read_bytes() != current_path.read_bytes():
        raise ValueError("current Transformers source differs from the pinned source revision")

    output_dir = args.output_dir or (
        Path(base_dir()) / "releases" / f"{args.repo.replace('/', '--')}-code"
    )
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"output exists (use --force): {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    code_hashes = prepare_code_update(source_path, output_dir)

    snapshot = Path(snapshot_download(args.repo, revision=current_info.sha))
    shadow = shadow_snapshot(snapshot, output_dir, output_dir.parent)
    try:
        config = json.loads((shadow / "config.json").read_text(encoding="utf-8"))
        validate_transformers_snapshot(shadow, config["expected_parameters"])
    finally:
        shutil.rmtree(shadow)
    print(f"Validated code-only update against {args.repo}@{current_info.sha}")
    if args.no_upload:
        return

    commit = api.create_commit(
        repo_id=args.repo,
        repo_type="model",
        operations=[
            CommitOperationAdd(path_in_repo=filename, path_or_fileobj=output_dir / filename)
            for filename in CODE_FILES
        ],
        commit_message="Support right-padded evaluation batches",
        parent_commit=current_info.sha,
    )
    remote_files = list(
        api.list_repo_tree(
            args.repo,
            revision=commit.oid,
            repo_type="model",
            recursive=True,
            expand=True,
        )
    )
    if weight_sha256(remote_files) != args.expected_weights_sha256:
        raise RuntimeError("model weights changed during the code-only update")
    for filename, expected in code_hashes.items():
        remote = hf_hub_download(args.repo, filename, revision=commit.oid, force_download=True)
        if sha256(remote) != expected:
            raise RuntimeError(f"uploaded {filename} does not match the validated source")
    print(commit.commit_url)


if __name__ == "__main__":
    main()
