"""commit local checkpoints to a hugging face model repository."""

import json
import os

from huggingface_hub import CommitOperationAdd, HfApi

from speck.common import base_dir


def upload(repo, checkpoint_dir, step, metadata, private=False, optimizer=False):
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    prefix = f"{step:06d}"
    operations = [
        CommitOperationAdd(
            path_in_repo="pytorch_model.bin",
            path_or_fileobj=os.path.join(checkpoint_dir, f"model_{prefix}.pt"),
        ),
        CommitOperationAdd(
            path_in_repo="config.json",
            path_or_fileobj=json.dumps(metadata["resolved"]["model"], indent=2).encode(),
        ),
        CommitOperationAdd(
            path_in_repo="training/state.json",
            path_or_fileobj=json.dumps(metadata, indent=2).encode(),
        ),
        CommitOperationAdd(
            path_in_repo="configuration_speck.py",
            path_or_fileobj=os.path.join(os.path.dirname(__file__), "export", "configuration_speck.py.txt"),
        ),
        CommitOperationAdd(
            path_in_repo="modeling_speck.py",
            path_or_fileobj=os.path.join(os.path.dirname(__file__), "export", "modeling_speck.py.txt"),
        ),
        CommitOperationAdd(
            path_in_repo="tokenizer_config.json",
            path_or_fileobj=json.dumps({
                "bos_token": "<s>",
                "eos_token": "</s>",
                "legacy": False,
                "model_max_length": metadata["resolved"]["model"]["max_position_embeddings"],
                "tokenizer_class": "LlamaTokenizer",
                "unk_token": "<unk>",
            }, indent=2).encode(),
        ),
    ]
    tokenizer_dir = os.path.join(base_dir(), "tokenizer")
    for local_name, remote_name in (
        ("tokenizer.model", "tokenizer.model"),
        ("tokenizer_metadata.json", "artifacts/tokenizer_metadata.json"),
    ):
        operations.append(CommitOperationAdd(
            path_in_repo=remote_name,
            path_or_fileobj=os.path.join(tokenizer_dir, local_name),
        ))
    if optimizer:
        operations.append(CommitOperationAdd(
            path_in_repo="training/optimizer.pt",
            path_or_fileobj=os.path.join(checkpoint_dir, f"optimizer_{prefix}.pt"),
        ))
    commit = api.create_commit(
        repo_id=repo,
        repo_type="model",
        operations=operations,
        commit_message=f"checkpoint step {step}",
    )
    return commit.commit_url
