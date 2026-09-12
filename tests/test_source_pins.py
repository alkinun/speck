import argparse
import hashlib
import subprocess

from scripts.source_pin_check import run
from speck.source_pins import changed_evidence_pins, changed_source_pins, source_pin_inventory


def git(repository, *args):
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repository(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    source = tmp_path / "speck" / "model.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n")
    other = tmp_path / "scripts" / "helper.py"
    other.parent.mkdir()
    other.write_text("VALUE = 2\n")
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nname = "fixture"\n')
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    project_digest = hashlib.sha256(project.read_bytes()).hexdigest()
    results = tmp_path / "results" / "qualification.json"
    results.parent.mkdir()
    results.write_text(
        f'{{"source_sha256": "{source_digest}", "pyproject_sha256": "{project_digest}"}}\n'
    )
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "fixture")
    return source, results, other


def test_inventory_finds_python_hash_references(tmp_path):
    source, results, _ = repository(tmp_path)

    inventory = source_pin_inventory(tmp_path)

    assert [entry["path"] for entry in inventory] == ["speck/model.py"]
    assert inventory[0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert inventory[0]["references"] == (str(results.relative_to(tmp_path)),)


def test_changed_pins_use_base_tree_even_if_reference_is_also_changed(tmp_path):
    source, results, other = repository(tmp_path)
    source.write_text("VALUE = 3\n")
    results.write_text("{}\n")
    other.write_text("VALUE = 4\n")

    changed = changed_source_pins(tmp_path)

    assert [entry["path"] for entry in changed] == ["speck/model.py"]
    assert changed[0]["references"] == ("results/qualification.json",)


def test_new_python_files_are_not_misclassified_as_historical_pins(tmp_path):
    repository(tmp_path)
    (tmp_path / "new.py").write_text("VALUE = 5\n")

    assert changed_source_pins(tmp_path) == ()


def test_changed_evidence_pins_include_non_python_contract_inputs(tmp_path):
    repository(tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "changed"\n')

    changed = changed_evidence_pins(tmp_path)

    assert [entry["path"] for entry in changed] == ["pyproject.toml"]
    assert changed[0]["references"] == ("results/qualification.json",)


def test_cli_returns_failure_only_for_evidence_bound_changes(tmp_path, capsys):
    source, _, other = repository(tmp_path)
    args = argparse.Namespace(
        repository=tmp_path,
        base="HEAD",
        target=None,
        inventory=False,
        all_files=False,
        json=False,
    )
    other.write_text("VALUE = 4\n")
    assert run(args) == 0
    assert "No evidence-bound tracked-file changes" in capsys.readouterr().out

    source.write_text("VALUE = 3\n")
    assert run(args) == 1
    assert "speck/model.py" in capsys.readouterr().out


def test_cli_checks_evidence_bound_non_python_files(tmp_path, capsys):
    repository(tmp_path)
    args = argparse.Namespace(
        repository=tmp_path,
        base="HEAD",
        target=None,
        inventory=False,
        all_files=False,
        json=False,
    )
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "changed"\n')

    assert run(args) == 1
    assert "pyproject.toml" in capsys.readouterr().out
