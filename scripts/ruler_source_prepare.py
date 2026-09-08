"""Build or verify the offline, content-pinned RULERv1 source bundle."""

import argparse
import glob
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from speck.io import atomic_json, file_sha256

PINNED_PACKAGES = {
    "beautifulsoup4": "4.15.0",
    "html2text": "2025.4.15",
    "nltk": "3.10.3",
    "pyyaml": "6.0.3",
    "tenacity": "9.1.4",
    "tqdm": "4.70.0",
    "transformers": "5.1.0",
    "wonderwords": "3.0.1",
}
GENERATOR_REVISION = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
NEEDLE_REVISION = "021385d68d3202e37893e9d3cd29011c569abe30"
URL_LIST_SHA256 = "9862c06b4303b7bb6fe4c4a88671c51d2d36ec65757db54730c703d9cf474099"
ASSETS = {
    "english_words": {
        "url": "https://media.githubusercontent.com/media/NVIDIA/RULER/"
        f"{GENERATOR_REVISION}/scripts/data/synthetic/json/english_words.json",
        "path": "inputs/english_words.json",
        "license": "Apache-2.0 repository; word-list provenance not separately stated",
        "expected_sha256": "affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca",
        "expected_bytes": 8_564_991,
    },
    "squad_v2_dev": {
        "url": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
        "path": "inputs/squad.json",
        "license": "CC-BY-SA-4.0",
    },
    "hotpotqa_dev_distractor": {
        "url": "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/"
        "7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json",
        "path": "inputs/hotpotqa.json",
        "license": "CC-BY-SA-4.0",
    },
}
WONDERWORDS_ASSETS = {
    "adjectivelist.txt": "66814d46b7e292c83e839d12fe2fa083c8f66779b14b294e1be4e1342c5d4131",
    "nounlist.txt": "8d88b3ebc2e2969ed92ebe49cdb0c8a87c2ad3b27493d7afdff608f2e51c5bc9",
    "profanitylist.txt": "67c668a797704ecbffdf4725ea3ee1a905a983968d2a3877b60a3971433f514e",
    "verblist.txt": "9fbf5e4e69b8869aebd88e6d545312ff9f878a747327ede5f6dbbba5c27b08ad",
}
NLTK_ARCHIVES = {
    "punkt.zip": "51c3078994aeaf650bfc8e028be4fb42b4a0d177d41c012b6a983979653660ec",
    "punkt_tab.zip": "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106",
}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-checkout", type=Path, required=True)
    parser.add_argument("--needle-checkout", type=Path, required=True)
    parser.add_argument("--nltk-data", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def git_revision(directory):
    return subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def package_versions():
    versions = {name: importlib.metadata.version(name) for name in PINNED_PACKAGES}
    if versions != PINNED_PACKAGES:
        raise ValueError(f"RULER source package versions changed: {versions}")
    return versions


def fetch(url, path):
    path = Path(path)
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_paul_html(raw):
    import html2text
    from bs4 import BeautifulSoup

    content = raw.decode("unicode_escape", "utf-8")
    specific_tag = BeautifulSoup(content, "html.parser").find("font")
    if specific_tag is None:
        raise ValueError("Paul Graham page has no canonical font body")
    converter = html2text.HTML2Text()
    converter.ignore_images = True
    converter.ignore_tables = True
    converter.escape_all = True
    converter.reference_links = False
    converter.mark_code = False
    return converter.handle(str(specific_tag))


def _entry(path, bundle, **values):
    return {
        **values,
        "path": path.relative_to(bundle).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _copy_checked(source, destination, expected_sha256=None):
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    if expected_sha256 and file_sha256(destination) != expected_sha256:
        raise ValueError(f"RULER pinned asset changed: {destination.name}")


def _paul_sources(generator, needle, bundle):
    url_list = generator / "scripts/data/synthetic/json/PaulGrahamEssays_URLs.txt"
    if file_sha256(url_list) != URL_LIST_SHA256:
        raise ValueError("RULER Paul Graham URL list changed")
    urls = url_list.read_text(encoding="utf-8").splitlines()
    entries = []
    failures = []
    for index, url in enumerate(urls):
        name = url.rsplit("/", 1)[-1]
        try:
            if ".html" in url:
                raw_path = bundle / "paul_graham/raw_html" / f"{index:03d}-{name}"
                canonical_path = (
                    bundle / "paul_graham/canonical_html" / name.replace(".html", ".txt")
                )
                fetch(url, raw_path)
                parsed = parse_paul_html(raw_path.read_bytes())
                if not canonical_path.is_file():
                    canonical_path.parent.mkdir(parents=True, exist_ok=True)
                    canonical_path.write_text(parsed, encoding="utf-8")
                elif canonical_path.read_text(encoding="utf-8") != parsed:
                    raise ValueError("canonical essay body changed from retained cache")
                entries.append(
                    {
                        "index": index,
                        "url": url,
                        "kind": "paulgraham.com_html",
                        "license": "upstream copyright; local research cache only",
                        "raw": _entry(raw_path, bundle),
                        "canonical": _entry(canonical_path, bundle),
                    }
                )
            else:
                source = needle / "needlehaystack/PaulGrahamEssays" / name
                canonical_path = bundle / "paul_graham/canonical_repo" / name
                if not source.is_file():
                    raise FileNotFoundError(f"pinned needle source is missing {name}")
                _copy_checked(source, canonical_path)
                entries.append(
                    {
                        "index": index,
                        "url": url,
                        "kind": "pinned_needle_repository",
                        "repository_revision": NEEDLE_REVISION,
                        "license": "MIT repository; underlying essay copyright not relicensed",
                        "canonical": _entry(canonical_path, bundle),
                    }
                )
        except Exception as error:
            failures.append({"index": index, "url": url, "error": str(error)})
    if failures:
        raise RuntimeError(f"RULER Paul Graham acquisition failed: {failures}")
    repo_files = sorted(glob.glob(str(bundle / "paul_graham/canonical_repo/*.txt")))
    html_files = sorted(glob.glob(str(bundle / "paul_graham/canonical_html/*.txt")))
    text = ""
    for path in (*repo_files, *html_files):
        text += Path(path).read_text(encoding="utf-8")
    output = bundle / "inputs/PaulGrahamEssays.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps({"text": text})
    if not output.is_file():
        output.write_text(encoded, encoding="utf-8")
    elif output.read_text(encoding="utf-8") != encoded:
        raise ValueError("RULER consolidated Paul Graham artifact changed")
    return entries, _entry(
        output,
        bundle,
        license="mixed upstream essay rights; local research cache only; redistribution forbidden",
    )


def _package_assets(bundle, nltk_data):
    import wonderwords

    source = Path(wonderwords.__file__).parent / "assets"
    entries = []
    for name, expected in WONDERWORDS_ASSETS.items():
        destination = bundle / "packages/wonderwords" / name
        _copy_checked(source / name, destination, expected)
        entries.append(_entry(destination, bundle, package="wonderwords==3.0.1"))
    for name, expected in NLTK_ARCHIVES.items():
        destination = bundle / "packages/nltk" / name
        _copy_checked(Path(nltk_data) / "tokenizers" / name, destination, expected)
        entries.append(_entry(destination, bundle, package="nltk==3.10.3"))
    return entries


def prepare(args):
    generator = args.generator_checkout.expanduser().resolve()
    needle = args.needle_checkout.expanduser().resolve()
    bundle = args.bundle.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    if git_revision(generator) != GENERATOR_REVISION:
        raise ValueError("RULER generator checkout is not at the pinned revision")
    if git_revision(needle) != NEEDLE_REVISION:
        raise ValueError("Needle repository is not at the pinned revision")
    versions = package_versions()
    bundle.mkdir(parents=True, exist_ok=True)
    paul_sources, paul_output = _paul_sources(generator, needle, bundle)
    assets = []
    for identifier, spec in ASSETS.items():
        path = bundle / spec["path"]
        fetch(spec["url"], path)
        entry = _entry(
            path,
            bundle,
            id=identifier,
            url=spec["url"],
            license=spec["license"],
        )
        if spec.get("expected_sha256") and (
            entry["sha256"] != spec["expected_sha256"] or entry["bytes"] != spec["expected_bytes"]
        ):
            raise ValueError(f"RULER pinned network asset changed: {identifier}")
        assets.append(entry)
    package_assets = _package_assets(bundle, args.nltk_data.expanduser().resolve())
    identity_payload = {
        "paul_output": paul_output,
        "assets": assets,
        "package_assets": package_assets,
        "package_versions": versions,
    }
    report = {
        "format": "speck_ruler_source_manifest",
        "format_version": 1,
        "status": "offline_sources_complete_unredistributable",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bundle_root": str(bundle),
        "generator": {
            "repository": "https://github.com/NVIDIA/RULER.git",
            "revision": GENERATOR_REVISION,
            "url_list_sha256": URL_LIST_SHA256,
        },
        "needle_repository": {
            "repository": "https://github.com/gkamradt/LLMTest_NeedleInAHaystack.git",
            "revision": NEEDLE_REVISION,
        },
        "package_versions": versions,
        "paul_graham_sources": paul_sources,
        "paul_graham_output": paul_output,
        "assets": assets,
        "package_assets": package_assets,
        "bundle_identity_sha256": hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "licenses": {
            "ruler_code": "Apache-2.0",
            "needle_repository": "MIT",
            "squad_v2": "CC-BY-SA-4.0",
            "hotpotqa": "CC-BY-SA-4.0",
            "paul_graham": "upstream copyright; local research use only; no redistribution",
        },
        "release_policy": "manifest and hashes may be published; source payloads and generated text containing upstream content remain outside the MIT repository",
    }
    atomic_json(manifest_path, report)
    atomic_json(bundle / "ruler-source-manifest.json", report)
    return report


def check(manifest_path):
    manifest_path = Path(manifest_path).expanduser().resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        value.get("format") != "speck_ruler_source_manifest"
        or value.get("status") != "offline_sources_complete_unredistributable"
    ):
        raise ValueError("RULER source manifest is invalid")
    bundle = Path(value["bundle_root"])
    entries = [
        value["paul_graham_output"],
        *value["assets"],
        *value["package_assets"],
        *[source["canonical"] for source in value["paul_graham_sources"]],
        *[source["raw"] for source in value["paul_graham_sources"] if "raw" in source],
    ]
    for entry in entries:
        path = bundle / entry["path"]
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or file_sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"RULER source bundle file changed: {entry['path']}")
    return value


def main(argv=None):
    args = arguments(argv)
    if args.check:
        report = check(args.manifest)
    else:
        report = prepare(args)
    print(f"RULER sources: {report['status']} ({report['bundle_identity_sha256'][:12]})")


if __name__ == "__main__":
    main()
