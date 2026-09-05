import json

import yaml

import scripts.helmet_runtime_dependency_audit as audit
from scripts.helmet_runtime_dependency_audit import extract_code_calls, protocol_identity


def test_extract_code_calls_tracks_remote_revisions_and_tokenizers():
    source = '''
load_dataset("dataset/a")
load_dataset("dataset/b", revision="abc")
AutoTokenizer.from_pretrained("model/tokenizer")
'''

    assert extract_code_calls(source) == {
        "load_dataset_literals": {"dataset/a": 1, "dataset/b": 1},
        "load_dataset_revision_bound_literals": {"dataset/b": 1},
        "tokenizer_literals": {"model/tokenizer": 1},
    }


def test_config_partition_treats_code_prompt_as_local(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "cite.yaml").write_text(
        yaml.safe_dump(
            {
                "datasets": "alce_asqa_30",
                "test_files": "data/alce/asqa.json",
                "demo_files": "prompts/asqa.json",
                "input_max_length": "8192",
            }
        ),
        encoding="utf-8",
    )
    (configs / "longqa.yaml").write_text(
        yaml.safe_dump(
            {
                "datasets": "narrativeqa_7892",
                "test_files": "",
                "demo_files": "",
                "input_max_length": "8192",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "CONFIGS", ("configs/cite.yaml", "configs/longqa.yaml"))

    assert [entry["mode"] for entry in audit.config_entries(tmp_path)] == [
        "archive_local",
        "runtime_loaded",
    ]


def test_protocol_identity_excludes_only_registration_fields():
    base = {"format": "protocol", "status": "frozen", "value": 1}
    registered = {**base, "status": "executed", "result": {"sha256": "abc"}}
    assert protocol_identity(base) == protocol_identity(registered)
    changed = json.loads(json.dumps(registered))
    changed["value"] = 2
    assert protocol_identity(changed) != protocol_identity(registered)
