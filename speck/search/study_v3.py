"""normalized transactional state for version three search studies."""

import json
import math
import secrets
import sqlite3
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from speck.architecture import ArchitectureConfig
from speck.search.artifacts import ArtifactEdge, ArtifactRecord
from speck.search.protocol import (
    ObjectiveSet,
    SeedBundle,
    TrainingProtocol,
    VersionSet,
    canonical_json,
    content_digest,
    study_semantics_version,
)


database_schema_version = 1


def _now():
    return datetime.now(timezone.utc)


def _timestamp(value=None):
    return (value or _now()).isoformat()


def _decode(value):
    return json.loads(value) if value is not None else None


class V3Study:
    def __init__(self, path, readonly=False):
        self.path = Path(path)
        if readonly:
            self.connection = sqlite3.connect(
                f"{self.path.resolve().as_uri()}?mode=ro",
                uri=True,
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("pragma foreign_keys = on")
        self.connection.execute("pragma busy_timeout = 5000")
        if not readonly:
            self.connection.execute("pragma journal_mode = wal")
            self._create_schema()
        self._validate_schema()

    def close(self):
        self.connection.close()

    def _create_schema(self):
        self.connection.executescript(
            """
            create table if not exists metadata (
                key text primary key,
                value text not null
            );
            create table if not exists study (
                id integer primary key check (id = 1),
                config_json text not null,
                provenance_json text not null,
                versions_json text not null,
                status text not null,
                created_at text not null,
                updated_at text not null
            );
            create table if not exists objective_sets (
                digest text primary key,
                name text not null,
                definition_json text not null,
                created_at text not null
            );
            create table if not exists architectures (
                digest text primary key,
                architecture_json text not null,
                static_json text not null,
                operation_json text,
                created_at text not null
            );
            create table if not exists runs (
                id integer primary key autoincrement,
                architecture_digest text not null references architectures(digest),
                protocol_digest text not null,
                seed_bundle_digest text not null,
                protocol_json text not null,
                seed_bundle_json text not null,
                status text not null,
                checkpoint_digest text,
                steps integer not null default 0,
                tokens integer not null default 0,
                created_at text not null,
                updated_at text not null,
                unique(architecture_digest, protocol_digest, seed_bundle_digest)
            );
            create table if not exists observations (
                id integer primary key autoincrement,
                run_id integer references runs(id),
                architecture_digest text not null references architectures(digest),
                objective_set_digest text not null references objective_sets(digest),
                objective_name text not null,
                value real not null,
                variance real,
                tokens integer,
                source text not null,
                artifact_digest text,
                created_at text not null
            );
            create table if not exists actions (
                id integer primary key autoincrement,
                kind text not null,
                status text not null,
                priority real not null,
                estimated_cost real not null,
                payload_json text not null,
                decision_digest text not null,
                owner text,
                claim_token text,
                lease_expires_at text,
                result_json text,
                error text,
                created_at text not null,
                claimed_at text,
                completed_at text
            );
            create table if not exists events (
                sequence integer primary key autoincrement,
                kind text not null,
                payload_json text not null,
                payload_digest text not null,
                created_at text not null
            );
            create table if not exists artifacts (
                digest text primary key,
                kind text not null,
                size integer not null,
                uri text not null,
                media_type text not null,
                created_at text not null
            );
            create table if not exists artifact_edges (
                parent_digest text not null references artifacts(digest),
                child_digest text not null references artifacts(digest),
                relation text not null,
                primary key(parent_digest, child_digest, relation)
            );
            create index if not exists actions_schedule
                on actions(status, priority desc, id);
            create index if not exists actions_leases
                on actions(status, lease_expires_at);
            create index if not exists observations_architecture
                on observations(architecture_digest, objective_set_digest, objective_name);
            create index if not exists runs_status on runs(status, id);
            """
        )
        values = {
            "database_schema_version": str(database_schema_version),
            "study_semantics_version": str(study_semantics_version),
        }
        for key, value in values.items():
            self.connection.execute(
                "insert or ignore into metadata(key, value) values (?, ?)",
                (key, value),
            )
        self.connection.commit()

    def _validate_schema(self):
        metadata = {
            row["key"]: row["value"]
            for row in self.connection.execute("select key, value from metadata")
        }
        expected = {
            "database_schema_version": database_schema_version,
            "study_semantics_version": study_semantics_version,
        }
        for key, value in expected.items():
            if int(metadata.get(key, 0)) != value:
                raise ValueError(f"unsupported v3 study {key}")

    def initialize(self, config, provenance, versions=None):
        versions = versions or VersionSet()
        values = (
            canonical_json(config),
            canonical_json(provenance),
            canonical_json(versions),
        )
        row = self.connection.execute("select * from study where id = 1").fetchone()
        if row is not None:
            stored = (
                row["config_json"],
                row["provenance_json"],
                row["versions_json"],
            )
            if stored != values:
                raise ValueError("v3 study identity changed")
            return False
        now = _timestamp()
        with self.connection:
            self.connection.execute(
                "insert into study values (1, ?, ?, ?, 'running', ?, ?)",
                (*values, now, now),
            )
            self._record_event("study_initialized", {"versions": asdict(versions)})
        return True

    def study(self):
        row = self.connection.execute("select * from study where id = 1").fetchone()
        if row is None:
            raise ValueError("v3 study is not initialized")
        return {
            "config": _decode(row["config_json"]),
            "provenance": _decode(row["provenance_json"]),
            "versions": _decode(row["versions_json"]),
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _record_event(self, kind, payload):
        encoded = canonical_json(payload)
        cursor = self.connection.execute(
            "insert into events(kind, payload_json, payload_digest, created_at) values (?, ?, ?, ?)",
            (kind, encoded, content_digest(payload), _timestamp()),
        )
        return cursor.lastrowid

    def record_event(self, kind, payload):
        with self.connection:
            return self._record_event(kind, payload)

    def events(self, after=0):
        return [
            {
                "sequence": row["sequence"],
                "kind": row["kind"],
                "payload": _decode(row["payload_json"]),
                "payload_digest": row["payload_digest"],
                "created_at": row["created_at"],
            }
            for row in self.connection.execute(
                "select * from events where sequence > ? order by sequence",
                (after,),
            )
        ]

    def add_objective_set(self, objectives):
        if not isinstance(objectives, ObjectiveSet):
            raise TypeError("objective definition must be an objective set")
        encoded = canonical_json(objectives)
        with self.connection:
            row = self.connection.execute(
                "select definition_json from objective_sets where digest = ?",
                (objectives.digest,),
            ).fetchone()
            if row is not None:
                if row["definition_json"] != encoded:
                    raise ValueError("objective set digest collision")
                return False
            self.connection.execute(
                "insert into objective_sets values (?, ?, ?, ?)",
                (objectives.digest, objectives.name, encoded, _timestamp()),
            )
            self._record_event(
                "objective_set_added",
                {"digest": objectives.digest, "name": objectives.name},
            )
        return True

    def add_architecture(self, config, static=None, operation=None):
        if not isinstance(config, ArchitectureConfig):
            raise TypeError("v3 study architectures need an architecture config")
        encoded = canonical_json(config.settings())
        with self.connection:
            row = self.connection.execute(
                "select architecture_json, static_json from architectures where digest = ?",
                (config.digest,),
            ).fetchone()
            if row is not None:
                if (
                    row["architecture_json"] != encoded
                    or row["static_json"] != canonical_json(static or {})
                ):
                    raise ValueError("architecture digest collision")
                return False
            self.connection.execute(
                "insert into architectures values (?, ?, ?, ?, ?)",
                (
                    config.digest,
                    encoded,
                    canonical_json(static or {}),
                    canonical_json(operation) if operation is not None else None,
                    _timestamp(),
                ),
            )
            self._record_event("architecture_added", {"digest": config.digest})
        return True

    def add_run(self, architecture_digest, protocol, seed_bundle):
        if not isinstance(protocol, TrainingProtocol) or not isinstance(
            seed_bundle, SeedBundle
        ):
            raise TypeError("run identity needs a training protocol and seed bundle")
        now = _timestamp()
        try:
            with self.connection:
                cursor = self.connection.execute(
                    """
                    insert into runs(
                        architecture_digest, protocol_digest, seed_bundle_digest,
                        protocol_json, seed_bundle_json, status, created_at, updated_at
                    ) values (?, ?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (
                        architecture_digest,
                        protocol.digest,
                        seed_bundle.digest,
                        canonical_json(protocol),
                        canonical_json(seed_bundle),
                        now,
                        now,
                    ),
                )
                run_id = cursor.lastrowid
                self._record_event(
                    "run_added",
                    {"architecture_digest": architecture_digest, "run_id": run_id},
                )
                return run_id
        except sqlite3.IntegrityError:
            row = self.connection.execute(
                """
                select id, protocol_json, seed_bundle_json from runs
                where architecture_digest = ?
                    and protocol_digest = ? and seed_bundle_digest = ?
                """,
                (architecture_digest, protocol.digest, seed_bundle.digest),
            ).fetchone()
            if row is None:
                raise
            if (
                row["protocol_json"] != canonical_json(protocol)
                or row["seed_bundle_json"] != canonical_json(seed_bundle)
            ):
                raise ValueError("run identity digest collision")
            return row["id"]

    def update_run(self, run_id, status, steps, tokens, checkpoint_digest=None):
        if status not in {"pending", "running", "paused", "completed", "failed"}:
            raise ValueError("invalid run status")
        if steps < 0 or tokens < 0:
            raise ValueError("run progress cannot be negative")
        with self.connection:
            current = self.connection.execute(
                "select steps, tokens from runs where id = ?",
                (run_id,),
            ).fetchone()
            if current is None:
                raise KeyError(run_id)
            if steps < current["steps"] or tokens < current["tokens"]:
                raise ValueError("run progress cannot move backwards")
            self.connection.execute(
                """
                update runs set status = ?, steps = ?, tokens = ?,
                    checkpoint_digest = coalesce(?, checkpoint_digest), updated_at = ?
                where id = ?
                """,
                (status, steps, tokens, checkpoint_digest, _timestamp(), run_id),
            )
            self._record_event(
                "run_updated",
                {"run_id": run_id, "status": status, "steps": steps, "tokens": tokens},
            )

    def add_observation(
        self,
        architecture_digest,
        objective_set_digest,
        objective_name,
        value,
        *,
        run_id=None,
        variance=None,
        tokens=None,
        source="measured",
        artifact_digest=None,
    ):
        if not math.isfinite(value):
            raise ValueError("observation values must be finite")
        if variance is not None and (not math.isfinite(variance) or variance < 0):
            raise ValueError("observation variance must be finite and nonnegative")
        row = self.connection.execute(
            "select definition_json from objective_sets where digest = ?",
            (objective_set_digest,),
        ).fetchone()
        if row is None:
            raise KeyError(objective_set_digest)
        names = {
            item["name"] for item in _decode(row["definition_json"])["objectives"]
        }
        if objective_name not in names:
            raise ValueError("observation objective is not in its objective set")
        with self.connection:
            cursor = self.connection.execute(
                """
                insert into observations(
                    run_id, architecture_digest, objective_set_digest,
                    objective_name, value, variance, tokens, source,
                    artifact_digest, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    architecture_digest,
                    objective_set_digest,
                    objective_name,
                    value,
                    variance,
                    tokens,
                    source,
                    artifact_digest,
                    _timestamp(),
                ),
            )
            self._record_event(
                "observation_added",
                {"id": cursor.lastrowid, "objective_name": objective_name},
            )
        return cursor.lastrowid

    def observations(self, architecture_digest=None, objective_set_digest=None):
        clauses = []
        values = []
        if architecture_digest is not None:
            clauses.append("architecture_digest = ?")
            values.append(architecture_digest)
        if objective_set_digest is not None:
            clauses.append("objective_set_digest = ?")
            values.append(objective_set_digest)
        where = f" where {' and '.join(clauses)}" if clauses else ""
        return [
            dict(row)
            for row in self.connection.execute(
                f"select * from observations{where} order by id",
                values,
            )
        ]

    def add_action(self, kind, priority, estimated_cost, payload):
        if not kind or kind.lower() != kind:
            raise ValueError("action kinds must be lowercase")
        if not math.isfinite(priority) or not math.isfinite(estimated_cost):
            raise ValueError("action values must be finite")
        if estimated_cost <= 0:
            raise ValueError("action cost must be positive")
        decision_digest = content_digest(
            {"kind": kind, "payload": payload, "priority": priority}
        )
        with self.connection:
            cursor = self.connection.execute(
                """
                insert into actions(
                    kind, status, priority, estimated_cost, payload_json,
                    decision_digest, created_at
                ) values (?, 'pending', ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    priority,
                    estimated_cost,
                    canonical_json(payload),
                    decision_digest,
                    _timestamp(),
                ),
            )
            action_id = cursor.lastrowid
            self._record_event(
                "action_added",
                {"action_id": action_id, "decision_digest": decision_digest},
            )
        return action_id

    def release_expired_actions(self, now=None):
        now = _timestamp(now)
        with self.connection:
            rows = self.connection.execute(
                """
                select id from actions where status = 'running'
                    and lease_expires_at < ?
                """,
                (now,),
            ).fetchall()
            for row in rows:
                self.connection.execute(
                    """
                    update actions set status = 'pending', owner = null,
                        claim_token = null, lease_expires_at = null,
                        claimed_at = null where id = ?
                    """,
                    (row["id"],),
                )
                self._record_event("action_released", {"action_id": row["id"]})
        return len(rows)

    def claim_action(self, owner, lease_seconds=300):
        if not owner or lease_seconds <= 0:
            raise ValueError("action claims need an owner and positive lease")
        now = _now()
        token = secrets.token_hex(16)
        self.connection.execute("begin immediate")
        try:
            row = self.connection.execute(
                """
                select id from actions where status = 'pending'
                order by priority desc, id limit 1
                """
            ).fetchone()
            if row is None:
                self.connection.rollback()
                return None
            self.connection.execute(
                """
                update actions set status = 'running', owner = ?, claim_token = ?,
                    lease_expires_at = ?, claimed_at = ? where id = ?
                """,
                (
                    owner,
                    token,
                    _timestamp(now + timedelta(seconds=lease_seconds)),
                    _timestamp(now),
                    row["id"],
                ),
            )
            self._record_event(
                "action_claimed",
                {"action_id": row["id"], "owner": owner},
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return self.action(row["id"])

    def heartbeat_action(self, action_id, claim_token, lease_seconds=300):
        if lease_seconds <= 0:
            raise ValueError("action heartbeat lease must be positive")
        expires = _timestamp(_now() + timedelta(seconds=lease_seconds))
        with self.connection:
            cursor = self.connection.execute(
                """
                update actions set lease_expires_at = ?
                where id = ? and status = 'running' and claim_token = ?
                """,
                (expires, action_id, claim_token),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("stale action heartbeat")
        return expires

    def action(self, action_id):
        row = self.connection.execute(
            "select * from actions where id = ?",
            (action_id,),
        ).fetchone()
        if row is None:
            raise KeyError(action_id)
        value = dict(row)
        value["payload"] = _decode(value.pop("payload_json"))
        value["result"] = _decode(value.pop("result_json"))
        return value

    def finish_action(self, action_id, claim_token, result=None, error=None):
        status = "failed" if error is not None else "completed"
        with self.connection:
            cursor = self.connection.execute(
                """
                update actions set status = ?, result_json = ?, error = ?,
                    completed_at = ?, lease_expires_at = null
                where id = ? and status = 'running' and claim_token = ?
                """,
                (
                    status,
                    canonical_json(result) if result is not None else None,
                    error,
                    _timestamp(),
                    action_id,
                    claim_token,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("stale action completion")
            self._record_event(
                "action_finished",
                {"action_id": action_id, "status": status},
            )

    def register_artifact(self, artifact):
        if not isinstance(artifact, ArtifactRecord):
            raise TypeError("artifact registration needs an artifact record")
        encoded = (
            artifact.digest,
            artifact.kind,
            artifact.size,
            artifact.uri,
            artifact.media_type,
        )
        with self.connection:
            row = self.connection.execute(
                "select * from artifacts where digest = ?",
                (artifact.digest,),
            ).fetchone()
            if row is not None:
                stored = (
                    row["digest"],
                    row["kind"],
                    row["size"],
                    row["uri"],
                    row["media_type"],
                )
                if stored != encoded:
                    raise ValueError("artifact digest collision")
                return False
            self.connection.execute(
                "insert into artifacts values (?, ?, ?, ?, ?, ?)",
                (*encoded, _timestamp()),
            )
            self._record_event("artifact_registered", {"digest": artifact.digest})
        return True

    def add_artifact_edge(self, edge):
        if not isinstance(edge, ArtifactEdge):
            raise TypeError("artifact lineage needs an artifact edge")
        known = {
            row["digest"]
            for row in self.connection.execute(
                "select digest from artifacts where digest in (?, ?)",
                (edge.parent_digest, edge.child_digest),
            )
        }
        if known != {edge.parent_digest, edge.child_digest}:
            raise KeyError("artifact lineage references an unknown artifact")
        try:
            with self.connection:
                self.connection.execute(
                    "insert into artifact_edges values (?, ?, ?)",
                    (edge.parent_digest, edge.child_digest, edge.relation),
                )
                self._record_event(
                    "artifact_edge_added",
                    asdict(edge),
                )
            return True
        except sqlite3.IntegrityError:
            return False
