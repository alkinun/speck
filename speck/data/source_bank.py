"""Prepare bounded source-separated engineering banks from retained pilot training text."""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from speck.data.packing import TokenShardWriter
from speck.provenance.io import durable_json, file_sha256
from speck.tokenization.tokenizer import Tokenizer

CATEGORIES = ("web", "code", "math", "synthetic", "science", "reference")


def _fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _integer(value, name, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _identity(value, root):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError("source bank identities require path and sha256")
    path = (root / Path(value["path"]).expanduser()).resolve()
    if not path.is_file() or file_sha256(path) != value["sha256"]:
        raise ValueError(f"source bank input identity mismatch: {path}")
    return {"path": str(path), "sha256": value["sha256"]}


def load_bank_plan(path):
    """Validate the bounded byte-quota contract and inherited exclusion lineage."""

    path = Path(path).resolve()
    value = json.loads(path.read_text())
    version = value.get("format_version")
    additional = {"firewall_plan"} if version == 2 else set()
    if (
        set(value)
        != {
            "format",
            "format_version",
            "purpose",
            "parent_manifest",
            "reference_tokenizer",
            "sources",
            "checkpoint_records",
            "shard_tokens",
            "output_directory",
        }
        | additional
        or value["format"] != "speck_bounded_source_bank_plan"
        or version not in (1, 2)
        or value["purpose"] != "engineering_rehearsal_not_training_data"
    ):
        raise ValueError("unsupported bounded source bank plan")
    parent_identity = _identity(value["parent_manifest"], path.parent)
    parent_path = Path(parent_identity["path"])
    parent = json.loads(parent_path.read_text())
    if (
        parent.get("format") != "speck_production_text_preprocess_result"
        or parent.get("gates", {}).get("global_exact_deduplication") != "pass"
        or parent.get("gates", {}).get(
            "disk_backed_Minhash_candidates_and_verified_near_deduplication"
        )
        != "pass"
    ):
        raise ValueError("source bank requires a checked globally deduplicated parent")
    ordered = parent.get("sources", [])
    references = {
        f"firewall_reference__{category}_{role}"
        for category in CATEGORIES
        for role in ("primary", "unseen")
    }
    if {item["id"] for item in ordered[:12]} != references or [
        item["precedence"] for item in ordered
    ] != list(range(1, len(ordered) + 1)):
        raise ValueError("parent must give all twelve firewall reference views precedence")
    firewall_identity = None
    if version == 2:
        from speck.data.firewall_integration import reference_sources, verify_excluded_parent

        reference_binding = reference_sources(value["firewall_plan"], path.parent)
        verify_excluded_parent(parent, reference_binding)
        firewall_identity = reference_binding["firewall_plan"]
    source_map = {item["id"]: item for item in ordered[12:]}
    sources = value["sources"]
    if not isinstance(sources, list) or [item.get("category") for item in sources] != list(
        CATEGORIES
    ):
        raise ValueError("source bank requires the six ordered categories")
    normalized = []
    for item in sources:
        if set(item) != {"category", "parent_source_id", "target_utf8_bytes"}:
            raise ValueError("invalid source bank source declaration")
        key = item["parent_source_id"]
        prefix = "pilot_train__" if version == 1 else "acquired_train__"
        if not isinstance(key, str) or not key.startswith(prefix) or key not in source_map:
            raise ValueError(
                "only retained pilot training outputs may enter a v1 bank"
                if version == 1
                else "only excluded acquisition outputs may enter a v2 bank"
            )
        if version == 2 and key != f"acquired_train__{item['category']}":
            raise ValueError("v2 bank category differs from its acquired source group")
        _integer(item["target_utf8_bytes"], "byte target", 100_000_000)
        entry = parent["outputs"][key]
        input_path = (parent_path.parent / entry["path"]).resolve()
        if not input_path.is_relative_to(parent_path.parent) or input_path == parent_path:
            raise ValueError("parent output path escapes the retained corpus")
        normalized.append(
            {
                **item,
                "input": {"path": str(input_path), "sha256": entry["sha256"]},
                "text_field": source_map[key]["text_field"],
                "content_sha256_field": source_map[key]["content_sha256_field"],
            }
        )
    if len({item["parent_source_id"] for item in normalized}) != len(CATEGORIES):
        raise ValueError("source bank inputs must be distinct")
    tokenizer = _identity(value["reference_tokenizer"], path.parent)
    _integer(value["checkpoint_records"], "checkpoint records", 10_000)
    _integer(value["shard_tokens"], "shard tokens", 10_000_000)
    output = (path.parent / Path(value["output_directory"]).expanduser()).resolve()
    # Keep managed outputs away from every retained input, including the plan itself.
    for protected in (
        path,
        parent_path,
        Path(tokenizer["path"]),
        *(Path(item["input"]["path"]) for item in normalized),
    ):
        if protected.is_relative_to(output):
            raise ValueError("source bank output contains a protected input")
    plan = {
        **value,
        "parent_manifest": parent_identity,
        "reference_tokenizer": tokenizer,
        "sources": normalized,
        "output_directory": str(output),
    }
    if firewall_identity is not None:
        plan["firewall_plan"] = firewall_identity
    plan["plan_fingerprint"] = _fingerprint(plan)
    return plan


def _select(plan, source, directory, *, interrupt_after_records=None):
    """Copy a whole-document prefix with durable record-boundary recovery."""

    target = source["target_utf8_bytes"]
    state_path = directory / "selection-state.json"
    selected_path = directory / "selected.jsonl"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "records": 0,
            "utf8_bytes": 0,
            "input_offset": 0,
            "output_bytes": 0,
            "output_sha256": hashlib.sha256(b"").hexdigest(),
        }
    )
    if (
        any(
            isinstance(state.get(key), bool)
            or not isinstance(state.get(key), int)
            or state[key] < 0
            for key in ("records", "utf8_bytes", "input_offset", "output_bytes")
        )
        or state["input_offset"] != state["output_bytes"]
    ):
        raise ValueError("invalid source bank selection checkpoint")
    hasher = hashlib.sha256()
    with selected_path.open("r+b" if selected_path.exists() else "w+b") as output:
        remaining = state["output_bytes"]
        records = utf8_bytes = 0
        with Path(source["input"]["path"]).open("rb") as retained:
            while remaining:
                chunk = output.readline(min(remaining, 16_000_001))
                if not chunk:
                    raise ValueError("source bank committed selection is truncated")
                if len(chunk) > 16_000_000 or retained.read(len(chunk)) != chunk:
                    raise ValueError("source bank committed selection differs from retained input")
                record = json.loads(chunk)
                records += 1
                utf8_bytes += len(record[source["text_field"]].encode())
                hasher.update(chunk)
                remaining -= len(chunk)
        if (records, utf8_bytes) != (state["records"], state["utf8_bytes"]):
            raise ValueError("source bank selection checkpoint counters disagree with text")
        if hasher.hexdigest() != state["output_sha256"]:
            raise ValueError("source bank committed selection hash mismatch")
        output.truncate(state["output_bytes"])
        output.seek(state["output_bytes"])

        def checkpoint():
            output.flush()
            os.fsync(output.fileno())
            state["output_bytes"] = output.tell()
            state["output_sha256"] = hasher.hexdigest()
            durable_json(state_path, state)

        with Path(source["input"]["path"]).open("rb") as handle:
            handle.seek(state["input_offset"])
            while state["utf8_bytes"] < target:
                raw = handle.readline(16_000_001)
                if len(raw) > 16_000_000:
                    raise ValueError("source bank record exceeds the 16MB engineering bound")
                if not raw:
                    checkpoint()
                    raise RuntimeError(f"source exhausted before byte quota: {source['category']}")
                record = json.loads(raw)
                text = record.get(source["text_field"])
                if (
                    not isinstance(text, str)
                    or not text
                    or hashlib.sha256(text.encode()).hexdigest()
                    != record.get(source["content_sha256_field"])
                ):
                    raise ValueError("source bank record content identity is invalid")
                output.write(raw)
                hasher.update(raw)
                state["records"] += 1
                state["utf8_bytes"] += len(text.encode())
                state["input_offset"] = handle.tell()
                if state["records"] % plan["checkpoint_records"] == 0:
                    checkpoint()
                if state["records"] == interrupt_after_records:
                    raise RuntimeError("injected source bank selection interruption")
        checkpoint()
    return state


def _pack(plan, source, directory, selection, tokenizer):
    """Publish packing per source; incomplete packing is rebuilt from verified text."""

    final = directory / "packed"
    staging = directory / "packed.building"
    if final.exists():
        manifest = json.loads((final / "manifest.json").read_text())
        if (
            manifest.get("selection_sha256") != selection["output_sha256"]
            or manifest.get("tokenizer_sha256") != plan["reference_tokenizer"]["sha256"]
        ):
            raise ValueError("packed source belongs to different text or tokenizer")
        for shard in manifest["shards"]:
            if file_sha256(final / shard["path"]) != shard["sha256"]:
                raise ValueError("packed source shard identity mismatch")
        return manifest
    # This directory is owned by the bank's checked fingerprint; only unpublished
    # per-source packing is disposable. Selected text and other sources are retained.
    if staging.exists():
        shutil.rmtree(staging)
    writer = TokenShardWriter(staging, "train", plan["shard_tokens"])
    batch = []
    characters = 0
    try:
        with (directory / "selected.jsonl").open() as handle:
            for raw in handle:
                text = json.loads(raw)[source["text_field"]]
                batch.append(text)
                characters += len(text)
                if len(batch) >= 128 or characters >= 1_000_000:
                    for tokens in tokenizer.encode_batch(batch, bos=True, eos=True):
                        writer.write(tokens)
                    batch, characters = [], 0
            if batch:
                for tokens in tokenizer.encode_batch(batch, bos=True, eos=True):
                    writer.write(tokens)
        writer.finish()
    except BaseException:
        writer.finish()
        raise
    manifest = {
        "selection_sha256": selection["output_sha256"],
        "tokenizer_sha256": plan["reference_tokenizer"]["sha256"],
        "reference_tokens": writer.total_tokens,
        "shards": writer.shards,
    }
    durable_json(staging / "manifest.json", manifest)
    staging.replace(final)
    return manifest


def prepare_source_bank(plan_path, *, interrupt_source=None, interrupt_after_records=None):
    """Prepare/reopen six bounded banks, preserving metadata and inherited exclusions."""

    started = time.perf_counter()
    plan = load_bank_plan(plan_path)
    output = Path(plan["output_directory"])
    owner = output / "plan.json"
    if output.exists():
        if not owner.is_file() or json.loads(owner.read_text()) != plan:
            raise ValueError("source bank output is unowned or has a different plan")
    else:
        output.mkdir(parents=True)
        durable_json(owner, plan)
    tokenizer = Tokenizer(plan["reference_tokenizer"]["path"])
    records = []
    timings = []
    for source in plan["sources"]:
        directory = output / source["category"]
        directory.mkdir(exist_ok=True)
        start = time.perf_counter()
        _identity(source["input"], Path("/"))
        verified = time.perf_counter()
        selection = _select(
            plan,
            source,
            directory,
            interrupt_after_records=(
                interrupt_after_records if source["category"] == interrupt_source else None
            ),
        )
        selected = time.perf_counter()
        packing = _pack(plan, source, directory, selection, tokenizer)
        finished = time.perf_counter()
        records.append(
            {
                "category": source["category"],
                "parent_source_id": source["parent_source_id"],
                "target_utf8_bytes": source["target_utf8_bytes"],
                "selection": selection,
                "packing": packing,
            }
        )
        timings.append(
            {
                "category": source["category"],
                "input_verification_seconds": verified - start,
                "selection_seconds": selected - verified,
                "packing_seconds": finished - selected,
            }
        )
    manifest = {
        "format": "speck_bounded_source_bank",
        "format_version": 1,
        "status": "complete_engineering_bank_not_training_data",
        "plan_fingerprint": plan["plan_fingerprint"],
        "parent_manifest": plan["parent_manifest"],
        "reference_tokenizer": plan["reference_tokenizer"],
        "sources": records,
        "training_authority": False,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("source bank published manifest changed")
    else:
        durable_json(manifest_path, manifest)
    return {
        "manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
        "sources": records,
        "timings": timings,
        "elapsed_seconds": time.perf_counter() - started,
        "timing_scope": "this invocation only; inherited acquisition/global dedup and prior interrupted work excluded",
        "training_authority": False,
    }
