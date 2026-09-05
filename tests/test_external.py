import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from speck.external import validate_external_suite

root = Path(__file__).parents[1]
external = root / "research" / "architecture-promotion-v1" / "external"


@pytest.mark.parametrize("name", ("ruler_v1.json", "nolima.json", "helmet.json"))
def test_checked_external_suite_contract_is_valid_and_blocked(name):
    config = validate_external_suite(external / name)
    assert "blocked" in config["status"]
    assert "blocked" in config["data"]["status"]
    assert "blocked" in config["model_adapter"]["status"]


def test_external_suite_rejects_an_unpinned_revision(tmp_path):
    copied = tmp_path / "external"
    shutil.copytree(external, copied)
    path = copied / "helmet.json"
    value = deepcopy(json.loads(path.read_text(encoding="utf-8")))
    value["upstream"]["revision"] = "main"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="full commit"):
        validate_external_suite(path)


def test_ruler_case_lengths_must_partition_the_suite(tmp_path):
    repository = tmp_path / "repository"
    copied = repository / "research/architecture-promotion-v1/external"
    shutil.copytree(external, copied)
    result_dir = repository / "results/Speck-Architecture-Promotion-v1"
    result_dir.mkdir(parents=True)
    for name in ("ruler-source-manifest.json", "ruler-cases-4096-qualified.json"):
        shutil.copy2(root / "results/Speck-Architecture-Promotion-v1" / name, result_dir / name)
    patch_dir = repository / "research/architecture-promotion-v1/patches"
    patch_dir.mkdir()
    shutil.copy2(
        root / "research/architecture-promotion-v1/patches/ruler_qa_required_docs.patch",
        patch_dir / "ruler_qa_required_docs.patch",
    )
    path = copied / "ruler_v1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["data"]["case_generation"]["remaining_lengths"].remove(8192)
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="do not partition"):
        validate_external_suite(path)
