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
