"""Normalize and publish the three Speck Transformers model cards."""

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from scripts.model_code_publish import weight_sha256
from speck.common import base_dir

LEADERBOARD_REVISION = "2eafcfc647b667e67f3b0288e9b67da497a78052"
BANANAMIND_REVISION = "d4aade51312889e8580963e1ce960c6eaef1a450"
CARD_SPECS = (
    {
        "repo": "specklabs/Speck1-140M",
        "revision": "57fe6b558b654ef91d5888ee8835048a4a0e9231",
        "weights_sha256": "199a77b5564868d0b2a03e4f59eb9e58615a2ece947c0618c4289c0ef1c6daf0",
    },
    {
        "repo": "specklabs/Speck1-140M-Instruct",
        "revision": "580a1ac03c8d198639ff470180c11827336664bd",
        "weights_sha256": "536eb6750ba41b8d6d88c02c09cdb5ba7411b58226b37c1883f1899e88a1ca4a",
    },
    {
        "repo": "specklabs/Speck1.1-140M-Instruct",
        "revision": "8a25b02e9049a29db49984968f4dc5a95c16980a",
        "weights_sha256": "5462ed5bc24361e19d506973a9267d02eda02965ff1c1c682f49c57c80752e48",
    },
)

COMPARISON_TABLE = """| Model | Params | Training tokens | Open SLM Int Index | BananaMind Base Bench 1.1 Elo | CPU prefill | CPU decode | RTX 3090 prefill | RTX 3090 decode | BF16 memory @2K | BF16 state @2K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BananaMind-2-Pro | 139M | 100B | 24.96 | 1131 | 2,190 tok/s | 43.0 tok/s | 64,060 tok/s | 140.3 tok/s | 325.1 MiB | 60.0 MiB |
| SmolLM2-135M | 135M | ~2T | 27.13 | 1119 | 2,201 tok/s | 47.4 tok/s | 64,814 tok/s | 157.7 tok/s | 301.6 MiB | 45.0 MiB |
| GPT-X2.5-135M | 135M | 75B | 25.17 | 1106 | 2,042 tok/s | 47.2 tok/s | 55,346 tok/s | 125.0 tok/s | 302.6 MiB | 45.0 MiB |
| Supra2-100M-Base | 101M | 30B | 19.41 | 1030 | **3,362 tok/s** | **56.0 tok/s** | **113,326 tok/s** | **298.1 tok/s** | **216.0 MiB** | 24.0 MiB |
| **Speck1-140M** | **141M** | **5B** | **18.15** | **965** | 2,252 tok/s | 55.1 tok/s | 74,323 tok/s | 247.3 tok/s | 281.3 MiB | **12.0 MiB** |
| **Speck1-140M-Instruct** | **141M** | **5B + 317M SFT** | **17.75** | **1001** | 2,285 tok/s | 55.3 tok/s | 73,398 tok/s | 246.7 tok/s | 280.3 MiB | **12.0 MiB** |
| **Speck1.1-140M-Instruct** | **141M** | **5B + 559M SFT** | **17.90** | **1002** | 2,315 tok/s | **56.9 tok/s** | 74,941 tok/s | 243.6 tok/s | 280.3 MiB | **12.0 MiB** |"""

INTELLIGENCE_INDEX_INPUTS = {
    "BananaMind-2-Pro": (42.78, 53.58, 27.82, 67.52, 38.20),
    "SmolLM2-135M": (43.22, 58.63, 29.69, 68.44, 39.20),
    "GPT-X2.5-135M": (40.57, 51.81, 29.18, 69.42, 38.40),
    "Supra2-100M-Base": (35.98, 47.81, 24.83, 65.40, 36.90),
    "Speck1-140M": (35.03, 46.68, 25.94, 63.87, 36.60),
    "Speck1-140M-Instruct": (35.22, 45.66, 25.85, 63.60, 36.10),
    "Speck1.1-140M-Instruct": (35.64, 46.93, 26.02, 64.15, 33.70),
}

EVALUATION_SECTION = f"""## Evaluation

The quality columns combine the
[Open SLM Leaderboard](https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard)
at revision `{LEADERBOARD_REVISION}` and
[BananaMind Base Bench 1.1](https://huggingface.co/datasets/BananaMind/BananaMind-Base-Bench-1.1)
at revision `{BANANAMIND_REVISION}`. No chat template or generation was used for the three
Speck evaluations.

### Benchmarks and speed

{COMPARISON_TABLE}

`Open SLM Int Index` means the chance-normalized Intelligence Index reported by the Open SLM
Leaderboard. `BananaMind Base Bench 1.1 Elo` means the overall Elo reported by BananaMind Base
Bench 1.1. Speed and memory values are local batch-1 measurements described below. Reference
models saw 6-400x more pretraining tokens, so this is a parameter-adjacent comparison, not a
compute-matched one."""


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(base_dir()) / "model-cards",
        help="local generated-card directory",
    )
    parser.add_argument("--no-upload", action="store_true", help="validate without uploading")
    parser.add_argument("--force", action="store_true", help="replace existing local cards")
    return parser.parse_args()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _remove_front_matter_key(card, key):
    if not card.startswith("---\n"):
        raise ValueError("model card has no YAML front matter")
    end = card.find("\n---\n", 4)
    if end < 0:
        raise ValueError("model card front matter is not terminated")
    lines = card[4:end].splitlines()
    start = next((index for index, line in enumerate(lines) if line == f"{key}:"), None)
    if start is not None:
        stop = start + 1
        while stop < len(lines) and (not lines[stop] or lines[stop][0].isspace()):
            stop += 1
        del lines[start:stop]
    return "---\n" + "\n".join(lines).rstrip() + "\n---\n" + card[end + 5 :]


def update_card(card):
    card = _remove_front_matter_key(card, "model-index")
    removed_rows = (
        "| BananaMind Base Bench Elo |",
        "| Direct instruction probe |",
    )
    card = "\n".join(
        line for line in card.splitlines() if not any(row in line for row in removed_rows)
    )
    card = card.replace(
        "Direct forward passes support unpadded batches.",
        "Direct forward passes support right-padded batches when `use_cache=False`.",
    )
    card = "\n".join(
        line
        for line in card.splitlines()
        if not line.startswith("- The direct probe answered none of five")
    )
    card = "\n".join(
        "The released checkpoint is training step 8,534."
        if line.startswith("The released checkpoint is training step 8,534.")
        else line
        for line in card.splitlines()
    )

    evaluation_start = card.find("## Evaluation\n")
    inference_start = card.find("## Inference speed\n", evaluation_start)
    if evaluation_start < 0 or inference_start < 0:
        raise ValueError("model card has no replaceable evaluation section")
    card = card[:evaluation_start] + EVALUATION_SECTION + "\n\n" + card[inference_start:]
    return card.rstrip() + "\n"


def _validate_intelligence_indexes():
    def normalize(value, chance):
        return 100 * (value - chance) / (100 - chance)

    displayed_indexes = {}
    for row in COMPARISON_TABLE.splitlines()[2:]:
        columns = [column.strip().strip("*") for column in row.strip("|").split("|")]
        displayed_indexes[columns[0]] = float(columns[3])

    if displayed_indexes.keys() != INTELLIGENCE_INDEX_INPUTS.keys():
        raise ValueError("comparison table models do not match the Int Index inputs")
    for model, scores in INTELLIGENCE_INDEX_INPUTS.items():
        hellaswag, arc_easy, arc_challenge, piqa, arithmark_3 = scores
        combined_arc = (arc_easy + arc_challenge) / 2
        calculated = (
            normalize(hellaswag, 25)
            + normalize(combined_arc, 25)
            + normalize(piqa, 50)
            + 0.65 * normalize(arithmark_3, 25)
        ) / 3.65
        if round(calculated, 2) != displayed_indexes[model]:
            raise ValueError(f"{model} Int Index does not match its benchmark scores")


def validate_card(card):
    _validate_intelligence_indexes()
    forbidden = (
        "model-index:",
        "| Category |",
        "| Elo |",
        "| Accuracy |",
        "Weighted acc.",
        "| BananaMind Base Bench Elo |",
        "Direct instruction probe",
        "| HellaSwag |",
        "| ARC-Easy |",
        "| ARC-Challenge |",
        "| PIQA |",
        "| ArithMark-3 |",
        "| ArithMark-2 |",
    )
    for value in forbidden:
        if value in card:
            raise ValueError(f"model card still contains {value!r}")
    if card.count("## Evaluation\n") != 1 or card.count("## Inference speed\n") != 1:
        raise ValueError("model card evaluation sections are not unique")
    evaluation = card[card.index("## Evaluation\n") : card.index("## Inference speed\n")]
    if evaluation.count("|---|---:") != 1:
        raise ValueError("model card must contain one evaluation table")
    if COMPARISON_TABLE not in evaluation:
        raise ValueError("model card does not contain the canonical comparison table")


def build_cards(output_dir, force=False):
    api = HfApi()
    built = []
    for spec in CARD_SPECS:
        current = api.model_info(spec["repo"])
        if current.sha != spec["revision"]:
            raise RuntimeError(
                f"{spec['repo']} moved from {spec['revision']} to {current.sha}; "
                "review and update the pinned card source"
            )
        files = list(
            api.list_repo_tree(
                spec["repo"],
                revision=spec["revision"],
                repo_type="model",
                recursive=True,
                expand=True,
            )
        )
        if weight_sha256(files) != spec["weights_sha256"]:
            raise RuntimeError(f"{spec['repo']} model weights do not match the release")
        source = Path(hf_hub_download(spec["repo"], "README.md", revision=spec["revision"]))
        card = update_card(source.read_text(encoding="utf-8"))
        validate_card(card)
        destination = output_dir / spec["repo"].replace("/", "--") / "README.md"
        if destination.exists() and not force:
            raise FileExistsError(f"generated card exists (use --force): {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(card, encoding="utf-8")
        built.append((spec, destination))
    return built


def publish_cards(built):
    api = HfApi()
    for spec, path in built:
        commit = api.create_commit(
            repo_id=spec["repo"],
            repo_type="model",
            operations=[CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=path)],
            commit_message="Simplify benchmark comparison table",
            parent_commit=spec["revision"],
        )
        files = list(
            api.list_repo_tree(
                spec["repo"],
                revision=commit.oid,
                repo_type="model",
                recursive=True,
                expand=True,
            )
        )
        if weight_sha256(files) != spec["weights_sha256"]:
            raise RuntimeError(f"{spec['repo']} model weights changed during card update")
        remote = hf_hub_download(
            spec["repo"],
            "README.md",
            revision=commit.oid,
            force_download=True,
        )
        if sha256(remote) != sha256(path):
            raise RuntimeError(f"{spec['repo']} uploaded card differs from local card")
        print(commit.commit_url)


def main():
    args = arguments()
    output_dir = args.output_dir.expanduser().resolve()
    built = build_cards(output_dir, args.force)
    for spec, path in built:
        print(f"Validated {spec['repo']} card at {path}")
    if not args.no_upload:
        publish_cards(built)


if __name__ == "__main__":
    main()
