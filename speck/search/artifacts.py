"""content-addressed artifacts and immutable lineage manifests."""

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from speck.search.protocol import artifact_manifest_version, canonical_json, content_digest


def file_digest(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactRecord:
    kind: str
    digest: str
    size: int
    uri: str
    media_type: str = "application/octet-stream"

    def __post_init__(self):
        if not self.kind or self.kind.lower() != self.kind:
            raise ValueError("artifact kinds must be lowercase")
        if len(self.digest) != 64:
            raise ValueError("artifact digests must be sha256 values")
        if self.size < 0:
            raise ValueError("artifact sizes cannot be negative")
        if Path(self.uri).is_absolute() or ".." in Path(self.uri).parts:
            raise ValueError("artifact uris must be relative")
        if not self.media_type:
            raise ValueError("artifact media types cannot be empty")


@dataclass(frozen=True)
class ArtifactEdge:
    parent_digest: str
    child_digest: str
    relation: str

    def __post_init__(self):
        if self.parent_digest == self.child_digest:
            raise ValueError("artifact lineage cannot contain self edges")
        if not self.relation or self.relation.lower() != self.relation:
            raise ValueError("artifact relations must be lowercase")


@dataclass(frozen=True)
class ArtifactManifest:
    artifacts: tuple[ArtifactRecord, ...]
    edges: tuple[ArtifactEdge, ...] = ()
    version: int = artifact_manifest_version

    def __post_init__(self):
        if self.version < 1:
            raise ValueError("artifact manifest versions must be positive")
        digests = tuple(artifact.digest for artifact in self.artifacts)
        if len(set(digests)) != len(digests):
            raise ValueError("artifact manifests cannot contain duplicate digests")
        known = set(digests)
        for edge in self.edges:
            if edge.parent_digest not in known or edge.child_digest not in known:
                raise ValueError("artifact lineage must reference manifest artifacts")

    @property
    def digest(self):
        return content_digest(self)

    def export(self):
        return json.loads(canonical_json(self))


class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _relative_path(self, digest):
        return Path("objects") / digest[:2] / digest

    def path(self, artifact):
        path = (self.root / artifact.uri).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("artifact path escapes its store")
        return path

    def _record(self, kind, digest, size, media_type):
        return ArtifactRecord(
            kind=kind,
            digest=digest,
            size=size,
            uri=self._relative_path(digest).as_posix(),
            media_type=media_type,
        )

    def put_bytes(self, kind, value, media_type="application/octet-stream"):
        digest = hashlib.sha256(value).hexdigest()
        artifact = self._record(kind, digest, len(value), media_type)
        destination = self.path(artifact)
        if destination.exists():
            self.verify(artifact)
            return artifact
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".staged-")
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(value)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return artifact

    def put_json(self, kind, value):
        return self.put_bytes(
            kind,
            canonical_json(value).encode(),
            "application/json",
        )

    def put_file(self, kind, source, media_type="application/octet-stream"):
        source = Path(source)
        digest = file_digest(source)
        artifact = self._record(kind, digest, source.stat().st_size, media_type)
        destination = self.path(artifact)
        if destination.exists():
            self.verify(artifact)
            return artifact
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".staged-")
        try:
            with source.open("rb") as input_file, os.fdopen(descriptor, "wb") as output:
                shutil.copyfileobj(input_file, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return artifact

    def verify(self, artifact):
        path = self.path(artifact)
        if not path.is_file():
            raise FileNotFoundError(f"artifact is missing: {artifact.digest}")
        if path.stat().st_size != artifact.size:
            raise ValueError(f"artifact size does not match: {artifact.digest}")
        if file_digest(path) != artifact.digest:
            raise ValueError(f"artifact digest does not match: {artifact.digest}")
        return True

    def read_bytes(self, artifact):
        self.verify(artifact)
        return self.path(artifact).read_bytes()

    def put_manifest(self, manifest):
        return self.put_json("artifact_manifest", manifest.export())
