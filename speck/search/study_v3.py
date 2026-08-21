"""normalized transactional state for version three search studies."""

import json
import math
import re
import secrets
import sqlite3
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from speck.architecture import ArchitectureConfig
from speck.profile.schema import ProfileResult, ProfileScenario
from speck.search.artifacts import ArtifactEdge, ArtifactRecord
from speck.search.checkpoints import RunCheckpoint
from speck.search.protocol import (
    ObjectiveSet,
    SeedBundle,
    TrainingProtocol,
    VersionSet,
    canonical_json,
    content_digest,
    study_semantics_version,
    worker_protocol_version,
)


database_schema_version = 6


def _now():
    return datetime.now(timezone.utc)


def _timestamp(value=None):
    return (value or _now()).isoformat()


def _decode(value):
    return json.loads(value) if value is not None else None


def _profile_metrics(scenario, result):
    prefix = scenario.name
    device_type = scenario.device.split(":", 1)[0]
    peak_name = "peak_vram" if device_type == "cuda" else "peak_rss"
    return {
        f"{prefix}.decode_p50": result.decode_ms.p50,
        f"{prefix}.decode_p95": result.decode_ms.p95,
        f"{prefix}.first_decode_p50": result.first_decode_ms.p50,
        f"{prefix}.first_decode_p95": result.first_decode_ms.p95,
        f"{prefix}.{peak_name}": float(result.peak_memory_bytes),
        f"{prefix}.prefill_p50": result.model_prefill_ms.p50,
        f"{prefix}.prefill_p95": result.model_prefill_ms.p95,
        f"{prefix}.request_p50": result.request_ms.p50,
        f"{prefix}.request_p95": result.request_ms.p95,
        "memory.state_bytes": float(result.state_bytes),
        "memory.weight_bytes": float(result.weight_bytes),
    }


def _preflight(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    try:
        connection = sqlite3.connect(
            f"{path.resolve().as_uri()}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        tables = {
            row["name"]
            for row in connection.execute(
                "select name from sqlite_master where type = 'table'"
            )
        }
        if "metadata" not in tables:
            raise ValueError("existing database is not a v3 study")
        metadata = {
            row["key"]: row["value"]
            for row in connection.execute("select key, value from metadata")
        }
    except sqlite3.DatabaseError as error:
        raise ValueError("existing database is not a v3 study") from error
    finally:
        if "connection" in locals():
            connection.close()
    expected = {
        "database_schema_version": database_schema_version,
        "study_semantics_version": study_semantics_version,
    }
    for key, value in expected.items():
        if int(metadata.get(key, 0)) != value:
            raise ValueError(f"unsupported v3 study {key}")


class V3Study:
    def __init__(self, path, readonly=False):
        self.path = Path(path)
        _preflight(self.path)
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
                run_id integer references runs(id),
                architecture_digest text references architectures(digest),
                profile_scenario_digest text,
                profile_backend text,
                required_device_type text,
                profile_repetition integer,
                evaluation_checkpoint_digest text,
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
            create table if not exists run_checkpoints (
                artifact_digest text primary key references artifacts(digest),
                run_id integer not null references runs(id),
                parent_digest text references run_checkpoints(artifact_digest),
                steps integer not null,
                tokens integer not null,
                format_version integer not null,
                created_at text not null,
                unique(run_id, tokens)
            );
            create table if not exists profile_results (
                action_id integer primary key references actions(id),
                artifact_digest text not null references artifacts(digest),
                architecture_digest text not null references architectures(digest),
                scenario_digest text not null,
                repetition integer not null,
                result_digest text not null,
                created_at text not null,
                unique(architecture_digest, scenario_digest, repetition)
            );
            create table if not exists planning_decisions (
                digest text primary key,
                definition_json text not null,
                event_sequence integer not null references events(sequence),
                created_at text not null
            );
            create table if not exists planning_actions (
                decision_digest text not null references planning_decisions(digest),
                position integer not null,
                action_id integer not null unique references actions(id),
                primary key(decision_digest, position)
            );
            create table if not exists quality_evaluations (
                action_id integer primary key references actions(id),
                artifact_digest text not null references artifacts(digest),
                run_id integer not null references runs(id),
                checkpoint_digest text not null references run_checkpoints(artifact_digest),
                evaluated_tokens integer not null,
                nll real not null,
                created_at text not null,
                unique(run_id, checkpoint_digest)
            );
            create table if not exists posterior_reports (
                digest text primary key,
                evidence_digest text not null unique,
                artifact_digest text not null references artifacts(digest),
                definition_json text not null,
                created_at text not null
            );
            create table if not exists posterior_anchors (
                report_digest text not null references posterior_reports(digest),
                position integer not null,
                architecture_digest text not null references architectures(digest),
                primary key(report_digest, position),
                unique(report_digest, architecture_digest)
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
            create unique index if not exists actions_active_run
                on actions(run_id)
                where run_id is not null and status in ('pending', 'running');
            create unique index if not exists actions_profile_identity
                on actions(architecture_digest, profile_scenario_digest, profile_repetition)
                where kind = 'profile' and status in ('pending', 'running', 'completed');
            create unique index if not exists actions_evaluation_identity
                on actions(run_id, evaluation_checkpoint_digest)
                where kind = 'evaluate'
                    and status in ('pending', 'running', 'completed');
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
        return self.initialize_bundle(
            config,
            provenance,
            versions=versions,
        )

    def initialize_bundle(
        self,
        config,
        provenance,
        *,
        versions=None,
        objective_sets=(),
        architecture=None,
        static=None,
        operation=None,
        artifacts=(),
    ):
        versions = versions or VersionSet()
        values = (
            canonical_json(config),
            canonical_json(provenance),
            canonical_json(versions),
        )
        self.connection.execute("begin immediate")
        try:
            row = self.connection.execute("select * from study where id = 1").fetchone()
            initialized = row is None
            if row is not None:
                stored = (
                    row["config_json"],
                    row["provenance_json"],
                    row["versions_json"],
                )
                if stored != values:
                    raise ValueError("v3 study identity changed")
            else:
                now = _timestamp()
                self.connection.execute(
                    "insert into study values (1, ?, ?, ?, 'running', ?, ?)",
                    (*values, now, now),
                )
                self._record_event(
                    "study_initialized",
                    {"versions": asdict(versions)},
                )
            for artifact in artifacts:
                if not isinstance(artifact, ArtifactRecord):
                    raise TypeError("study bundle artifacts need artifact records")
                self._register_artifact(artifact)
            for objectives in objective_sets:
                self._add_objective_set(objectives)
            if architecture is not None:
                self._add_architecture(architecture, static, operation)
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return initialized

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
        created_at = _timestamp()
        cursor = self.connection.execute(
            "insert into events(kind, payload_json, payload_digest, created_at) values (?, ?, ?, ?)",
            (kind, encoded, content_digest(payload), created_at),
        )
        self.connection.execute(
            "update study set updated_at = ? where id = 1",
            (created_at,),
        )
        return cursor.lastrowid

    def record_event(self, kind, payload):
        with self.connection:
            return self._record_event(kind, payload)

    def commit_planning_decision(self, decision_digest, definition, actions):
        if not re.fullmatch(r"[0-9a-f]{64}", decision_digest):
            raise ValueError("planning decision digests must be lowercase sha256")
        actions = tuple(actions)
        encoded = canonical_json(definition)
        self.connection.execute("begin immediate")
        try:
            row = self.connection.execute(
                "select definition_json from planning_decisions where digest = ?",
                (decision_digest,),
            ).fetchone()
            if row is not None:
                if row["definition_json"] != encoded:
                    raise ValueError("planning decision digest collision")
                action_ids = tuple(
                    item["action_id"]
                    for item in self.connection.execute(
                        """
                        select action_id from planning_actions
                        where decision_digest = ? order by position
                        """,
                        (decision_digest,),
                    )
                )
                self.connection.rollback()
                return action_ids
            event = self._record_event(
                "planning_decision",
                {
                    "decision": definition,
                    "decision_digest": decision_digest,
                },
            )
            self.connection.execute(
                "insert into planning_decisions values (?, ?, ?, ?)",
                (decision_digest, encoded, event, _timestamp()),
            )
            action_ids = []
            for position, action in enumerate(actions):
                payload = {
                    **action["payload"],
                    "planning_decision_digest": decision_digest,
                    "planning_event": event,
                }
                action_id = self._insert_action(
                    action["kind"],
                    action["priority"],
                    action["estimated_cost"],
                    payload,
                )
                self.connection.execute(
                    "insert into planning_actions values (?, ?, ?)",
                    (decision_digest, position, action_id),
                )
                action_ids.append(action_id)
            self.connection.commit()
            return tuple(action_ids)
        except Exception:
            self.connection.rollback()
            raise

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

    def _add_objective_set(self, objectives):
        if not isinstance(objectives, ObjectiveSet):
            raise TypeError("objective definition must be an objective set")
        encoded = canonical_json(objectives)
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

    def add_objective_set(self, objectives):
        with self.connection:
            return self._add_objective_set(objectives)

    def objective_set(self, digest):
        row = self.connection.execute(
            "select definition_json from objective_sets where digest = ?",
            (digest,),
        ).fetchone()
        if row is None:
            raise KeyError(digest)
        return ObjectiveSet.from_dict(_decode(row["definition_json"]))

    def _add_architecture(self, config, static=None, operation=None):
        if not isinstance(config, ArchitectureConfig):
            raise TypeError("v3 study architectures need an architecture config")
        encoded = canonical_json(config.settings())
        row = self.connection.execute(
            """
            select architecture_json, static_json, operation_json
            from architectures where digest = ?
            """,
            (config.digest,),
        ).fetchone()
        encoded_operation = (
            canonical_json(operation) if operation is not None else None
        )
        if row is not None:
            if (
                row["architecture_json"] != encoded
                or row["static_json"] != canonical_json(static or {})
                or row["operation_json"] != encoded_operation
            ):
                raise ValueError("architecture digest collision")
            return False
        self.connection.execute(
            "insert into architectures values (?, ?, ?, ?, ?)",
            (
                config.digest,
                encoded,
                canonical_json(static or {}),
                encoded_operation,
                _timestamp(),
            )
        )
        self._record_event("architecture_added", {"digest": config.digest})
        return True

    def add_architecture(self, config, static=None, operation=None):
        with self.connection:
            return self._add_architecture(config, static, operation)

    def architecture(self, digest):
        row = self.connection.execute(
            "select * from architectures where digest = ?",
            (digest,),
        ).fetchone()
        if row is None:
            raise KeyError(digest)
        return {
            "config": ArchitectureConfig.from_dict(_decode(row["architecture_json"])),
            "created_at": row["created_at"],
            "digest": row["digest"],
            "operation": _decode(row["operation_json"]),
            "static": _decode(row["static_json"]),
        }

    def architectures(self):
        return [
            self.architecture(row["digest"])
            for row in self.connection.execute(
                "select digest from architectures order by created_at, digest"
            )
        ]

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

    def run(self, run_id):
        row = self.connection.execute(
            "select * from runs where id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        value = dict(row)
        value["protocol"] = TrainingProtocol.from_dict(
            _decode(value.pop("protocol_json"))
        )
        value["seed_bundle"] = SeedBundle.from_dict(
            _decode(value.pop("seed_bundle_json"))
        )
        return value

    def runs(self, status=None):
        where = " where status = ?" if status is not None else ""
        values = (status,) if status is not None else ()
        return [
            self.run(row["id"])
            for row in self.connection.execute(
                f"select id from runs{where} order by id",
                values,
            )
        ]

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

    def _insert_action(
        self,
        kind,
        priority,
        estimated_cost,
        payload,
        *,
        run_id=None,
        architecture_digest=None,
        profile_scenario_digest=None,
        profile_backend=None,
        required_device_type=None,
        profile_repetition=None,
        evaluation_checkpoint_digest=None,
    ):
        if not kind or kind.lower() != kind:
            raise ValueError("action kinds must be lowercase")
        if not math.isfinite(priority) or not math.isfinite(estimated_cost):
            raise ValueError("action values must be finite")
        if estimated_cost <= 0:
            raise ValueError("action cost must be positive")
        decision_digest = content_digest(
            {"kind": kind, "payload": payload, "priority": priority}
        )
        cursor = self.connection.execute(
            """
            insert into actions(
                run_id, architecture_digest, profile_scenario_digest,
                profile_backend, required_device_type, profile_repetition,
                evaluation_checkpoint_digest,
                kind, status, priority, estimated_cost, payload_json,
                decision_digest, created_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                architecture_digest,
                profile_scenario_digest,
                profile_backend,
                required_device_type,
                profile_repetition,
                evaluation_checkpoint_digest,
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

    def add_action(self, kind, priority, estimated_cost, payload):
        with self.connection:
            return self._insert_action(
                kind,
                priority,
                estimated_cost,
                payload,
            )

    def add_quality_action(
        self,
        run_id,
        priority,
        estimated_cost,
        target_tokens=None,
    ):
        with self.connection:
            row = self.connection.execute(
                "select * from runs where id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            if row["status"] not in {"pending", "paused"}:
                raise ValueError("quality actions need a pending or paused run")
            if row["checkpoint_digest"] is not None:
                evaluated = self.connection.execute(
                    """
                    select 1 from quality_evaluations
                    where run_id = ? and checkpoint_digest = ?
                    """,
                    (run_id, row["checkpoint_digest"]),
                ).fetchone()
                if evaluated is None:
                    raise ValueError(
                        "quality continuation needs the current checkpoint evaluation"
                    )
            protocol = TrainingProtocol.from_dict(_decode(row["protocol_json"]))
            remaining = tuple(
                tokens for tokens in protocol.checkpoint_tokens if tokens > row["tokens"]
            )
            if not remaining:
                raise ValueError("quality run has no remaining checkpoints")
            expected_target = remaining[0]
            if target_tokens is None:
                target_tokens = expected_target
            if target_tokens != expected_target:
                raise ValueError("quality actions must target the next checkpoint")
            payload = {
                "expected_checkpoint_digest": row["checkpoint_digest"],
                "run_id": run_id,
                "starting_steps": row["steps"],
                "starting_tokens": row["tokens"],
                "target_tokens": target_tokens,
                "worker_protocol_version": worker_protocol_version,
            }
            try:
                return self._insert_action(
                    "continue",
                    priority,
                    estimated_cost,
                    payload,
                    run_id=run_id,
                    architecture_digest=row["architecture_digest"],
                    required_device_type=protocol.device_type,
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("quality run already has an active action") from error

    def add_profile_action(
        self,
        architecture_digest,
        scenario,
        objective_set_digest,
        repetition,
        prompt_seed,
        priority,
        estimated_cost,
    ):
        if not isinstance(scenario, ProfileScenario):
            raise TypeError("profile actions need a profile scenario")
        if not 0 <= repetition < scenario.process_repetitions:
            raise ValueError("profile repetition is outside its scenario")
        if prompt_seed < 0:
            raise ValueError("profile prompt seeds cannot be negative")
        device_type = scenario.device.split(":", 1)[0]
        with self.connection:
            if self.connection.execute(
                "select 1 from architectures where digest = ?",
                (architecture_digest,),
            ).fetchone() is None:
                raise KeyError(architecture_digest)
            if self.connection.execute(
                "select 1 from objective_sets where digest = ?",
                (objective_set_digest,),
            ).fetchone() is None:
                raise KeyError(objective_set_digest)
            payload = {
                "architecture_digest": architecture_digest,
                "objective_set_digest": objective_set_digest,
                "prompt_seed": prompt_seed,
                "repetition": repetition,
                "scenario": scenario.export(),
                "scenario_digest": scenario.digest,
                "worker_protocol_version": worker_protocol_version,
            }
            try:
                return self._insert_action(
                    "profile",
                    priority,
                    estimated_cost,
                    payload,
                    architecture_digest=architecture_digest,
                    profile_scenario_digest=scenario.digest,
                    profile_backend=scenario.backend.name,
                    required_device_type=device_type,
                    profile_repetition=repetition,
                )
            except sqlite3.IntegrityError as error:
                row = self.connection.execute(
                    """
                    select id, payload_json, priority, estimated_cost from actions
                    where architecture_digest = ? and profile_scenario_digest = ?
                        and profile_repetition = ?
                        and status in ('pending', 'running', 'completed')
                    """,
                    (architecture_digest, scenario.digest, repetition),
                ).fetchone()
                if row is None:
                    raise
                if (
                    row["payload_json"] != canonical_json(payload)
                    or row["priority"] != priority
                    or row["estimated_cost"] != estimated_cost
                ):
                    raise ValueError(
                        "profile repetition already has a different action"
                    ) from error
                return row["id"]

    def add_evaluation_action(
        self,
        run_id,
        objective_set_digests,
        expected_evaluation_tokens,
        priority,
        estimated_cost,
    ):
        objective_set_digests = tuple(sorted(set(objective_set_digests)))
        if not objective_set_digests:
            raise ValueError("quality evaluation needs objective sets")
        if expected_evaluation_tokens < 1:
            raise ValueError("quality evaluation token counts must be positive")
        with self.connection:
            row = self.connection.execute(
                "select * from runs where id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            if row["status"] not in {"paused", "completed"}:
                raise ValueError("quality evaluation needs a checkpointed run")
            if row["checkpoint_digest"] is None:
                raise ValueError("quality evaluation needs a checkpoint")
            for digest in objective_set_digests:
                objectives = self.connection.execute(
                    "select definition_json from objective_sets where digest = ?",
                    (digest,),
                ).fetchone()
                if objectives is None:
                    raise KeyError(digest)
                names = {
                    item["name"]
                    for item in _decode(objectives["definition_json"])["objectives"]
                }
                if "quality.target_nll" not in names:
                    raise ValueError(
                        "quality evaluation objective sets need quality.target_nll"
                    )
            payload = {
                "architecture_digest": row["architecture_digest"],
                "checkpoint_digest": row["checkpoint_digest"],
                "expected_evaluation_tokens": expected_evaluation_tokens,
                "objective_set_digests": objective_set_digests,
                "run_id": run_id,
                "training_tokens": row["tokens"],
                "worker_protocol_version": worker_protocol_version,
            }
            try:
                return self._insert_action(
                    "evaluate",
                    priority,
                    estimated_cost,
                    payload,
                    run_id=run_id,
                    architecture_digest=row["architecture_digest"],
                    required_device_type=TrainingProtocol.from_dict(
                        _decode(row["protocol_json"])
                    ).device_type,
                    evaluation_checkpoint_digest=row["checkpoint_digest"],
                )
            except sqlite3.IntegrityError as error:
                existing = self.connection.execute(
                    """
                    select id, payload_json, priority, estimated_cost from actions
                    where run_id = ? and evaluation_checkpoint_digest = ?
                        and kind = 'evaluate'
                        and status in ('pending', 'running', 'completed')
                    """,
                    (run_id, row["checkpoint_digest"]),
                ).fetchone()
                if existing is None:
                    raise
                if (
                    existing["payload_json"] != canonical_json(payload)
                    or existing["priority"] != priority
                    or existing["estimated_cost"] != estimated_cost
                ):
                    raise ValueError(
                        "quality checkpoint already has a different evaluation action"
                    ) from error
                return existing["id"]

    def release_expired_actions(self, now=None):
        now = _timestamp(now)
        with self.connection:
            rows = self.connection.execute(
                """
                select id, run_id, kind from actions where status = 'running'
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
                if row["run_id"] is not None and row["kind"] == "continue":
                    self.connection.execute(
                        """
                        update runs set status = case when tokens = 0 then 'pending'
                            else 'paused' end, updated_at = ? where id = ?
                        """,
                        (now, row["run_id"]),
                    )
                self._record_event("action_released", {"action_id": row["id"]})
        return len(rows)

    def claim_action(
        self,
        owner,
        lease_seconds=300,
        kind=None,
        backend=None,
        device_type=None,
    ):
        if not owner or lease_seconds <= 0:
            raise ValueError("action claims need an owner and positive lease")
        if kind is not None and (not kind or kind.lower() != kind):
            raise ValueError("action claim filters must be lowercase")
        if backend is not None and (not backend or backend.lower() != backend):
            raise ValueError("profile backend filters must be lowercase")
        if device_type is not None and device_type not in {"cpu", "cuda"}:
            raise ValueError("profile device filters must be cpu or cuda")
        now = _now()
        token = secrets.token_hex(16)
        self.connection.execute("begin immediate")
        try:
            where = "status = 'pending'"
            values = []
            if kind is not None:
                where += " and kind = ?"
                values.append(kind)
            if backend is not None:
                where += " and profile_backend = ?"
                values.append(backend)
            if device_type is not None:
                where += " and required_device_type = ?"
                values.append(device_type)
            row = self.connection.execute(
                f"""
                select id, run_id, kind from actions where {where}
                order by priority desc, id limit 1
                """,
                values,
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
            if row["run_id"] is not None and row["kind"] == "continue":
                cursor = self.connection.execute(
                    """
                    update runs set status = 'running', updated_at = ?
                    where id = ? and status in ('pending', 'paused')
                    """,
                    (_timestamp(now), row["run_id"]),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("quality run is not claimable")
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
        now = _now()
        expires = _timestamp(now + timedelta(seconds=lease_seconds))
        with self.connection:
            cursor = self.connection.execute(
                """
                update actions set lease_expires_at = ?
                where id = ? and status = 'running' and claim_token = ?
                    and lease_expires_at >= ?
                """,
                (expires, action_id, claim_token, _timestamp(now)),
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

    def actions(self, status=None):
        where = " where status = ?" if status is not None else ""
        values = (status,) if status is not None else ()
        return [
            self.action(row["id"])
            for row in self.connection.execute(
                f"select id from actions{where} order by id",
                values,
            )
        ]

    def finish_action(self, action_id, claim_token, result=None, error=None):
        status = "failed" if error is not None else "completed"
        now = _timestamp()
        with self.connection:
            action = self.connection.execute(
                """
                select run_id, kind, profile_scenario_digest,
                    evaluation_checkpoint_digest from actions
                where id = ? and status = 'running' and claim_token = ?
                    and lease_expires_at >= ?
                """,
                (action_id, claim_token, now),
            ).fetchone()
            if action is None:
                raise RuntimeError("stale action completion")
            normalized_result = (
                action["kind"] == "continue" and action["run_id"] is not None
            ) or action["profile_scenario_digest"] is not None or (
                action["evaluation_checkpoint_digest"] is not None
            )
            if normalized_result and error is None:
                raise ValueError("normalized actions need an atomic result completion")
            self.connection.execute(
                """
                update actions set status = ?, result_json = ?, error = ?,
                    completed_at = ?, lease_expires_at = null
                where id = ?
                """,
                (
                    status,
                    canonical_json(result) if result is not None else None,
                    error,
                    now,
                    action_id,
                ),
            )
            if action["kind"] == "continue":
                self.connection.execute(
                    """
                    update runs set status = case when tokens = 0 then 'pending'
                        else 'paused' end, updated_at = ? where id = ?
                    """,
                    (now, action["run_id"]),
                )
            self._record_event(
                "action_finished",
                {"action_id": action_id, "status": status},
            )

    def commit_quality_checkpoint(self, action_id, claim_token, checkpoint):
        if not isinstance(checkpoint, RunCheckpoint):
            raise TypeError("quality commits need a run checkpoint")
        now = _now()
        timestamp = _timestamp(now)
        self.connection.execute("begin immediate")
        try:
            row = self.connection.execute(
                """
                select actions.*,
                    runs.architecture_digest as run_architecture_digest,
                    runs.protocol_digest,
                    runs.seed_bundle_digest, runs.protocol_json,
                    runs.checkpoint_digest as run_checkpoint_digest,
                    runs.steps as run_steps, runs.tokens as run_tokens
                from actions join runs on runs.id = actions.run_id
                where actions.id = ? and actions.kind = 'continue'
                    and actions.status = 'running' and actions.claim_token = ?
                    and actions.lease_expires_at >= ?
                """,
                (action_id, claim_token, timestamp),
            ).fetchone()
            if row is None:
                raise RuntimeError("stale quality checkpoint completion")
            payload = _decode(row["payload_json"])
            identity = (
                row["run_architecture_digest"],
                row["protocol_digest"],
                row["seed_bundle_digest"],
            )
            checkpoint_identity = (
                checkpoint.architecture_digest,
                checkpoint.protocol_digest,
                checkpoint.seed_bundle_digest,
            )
            if identity != checkpoint_identity:
                raise ValueError("quality checkpoint identity does not match its run")
            expected_parent = row["run_checkpoint_digest"]
            if (
                checkpoint.parent_digest != expected_parent
                or payload["expected_checkpoint_digest"] != expected_parent
            ):
                raise ValueError("quality checkpoint parent does not match its run")
            if (
                payload["starting_steps"] != row["run_steps"]
                or payload["starting_tokens"] != row["run_tokens"]
            ):
                raise ValueError("quality checkpoint action started from stale progress")
            protocol = TrainingProtocol.from_dict(_decode(row["protocol_json"]))
            if checkpoint.tokens != payload["target_tokens"]:
                raise ValueError("quality checkpoint did not reach its action target")
            remaining = tuple(
                tokens
                for tokens in protocol.checkpoint_tokens
                if tokens > row["run_tokens"]
            )
            if not remaining or checkpoint.tokens != remaining[0]:
                raise ValueError("quality checkpoint is not the next protocol checkpoint")
            if checkpoint.tokens % protocol.batch_tokens:
                raise ValueError("quality checkpoint tokens do not align to its protocol")
            expected_steps = checkpoint.tokens // protocol.batch_tokens
            if checkpoint.steps != expected_steps:
                raise ValueError("quality checkpoint steps do not match its tokens")
            if expected_parent is not None:
                parent = self.connection.execute(
                    """
                    select run_id, steps, tokens from run_checkpoints
                    where artifact_digest = ?
                    """,
                    (expected_parent,),
                ).fetchone()
                if parent is None or parent["run_id"] != row["run_id"]:
                    raise ValueError("quality checkpoint parent is not in its run")
                if (
                    checkpoint.steps <= parent["steps"]
                    or checkpoint.tokens <= parent["tokens"]
                ):
                    raise ValueError("quality checkpoint progress did not advance")
            self._register_artifact(checkpoint.artifact)
            self.connection.execute(
                """
                insert into run_checkpoints(
                    artifact_digest, run_id, parent_digest, steps, tokens,
                    format_version, created_at
                ) values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.artifact.digest,
                    row["run_id"],
                    checkpoint.parent_digest,
                    checkpoint.steps,
                    checkpoint.tokens,
                    checkpoint.format_version,
                    timestamp,
                ),
            )
            if checkpoint.parent_digest is not None:
                self._add_artifact_edge(
                    ArtifactEdge(
                        checkpoint.parent_digest,
                        checkpoint.artifact.digest,
                        "continued_from",
                    )
                )
            run_status = (
                "completed"
                if checkpoint.tokens == protocol.target_tokens
                else "paused"
            )
            cursor = self.connection.execute(
                """
                update runs set status = ?, checkpoint_digest = ?, steps = ?,
                    tokens = ?, updated_at = ? where id = ? and steps = ?
                    and tokens = ? and checkpoint_digest is ?
                """,
                (
                    run_status,
                    checkpoint.artifact.digest,
                    checkpoint.steps,
                    checkpoint.tokens,
                    timestamp,
                    row["run_id"],
                    row["run_steps"],
                    row["run_tokens"],
                    expected_parent,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("quality run changed during checkpoint commit")
            result = {
                "checkpoint_digest": checkpoint.artifact.digest,
                "steps": checkpoint.steps,
                "tokens": checkpoint.tokens,
            }
            self.connection.execute(
                """
                update actions set status = 'completed', result_json = ?,
                    completed_at = ?, lease_expires_at = null where id = ?
                """,
                (canonical_json(result), timestamp, action_id),
            )
            self._record_event(
                "quality_checkpoint_committed",
                {"action_id": action_id, "run_id": row["run_id"], **result},
            )
            self._record_event(
                "run_updated",
                {"run_id": row["run_id"], "status": run_status, **result},
            )
            self._record_event(
                "action_finished",
                {"action_id": action_id, "status": "completed"},
            )
            self.connection.commit()
            return run_status
        except Exception:
            self.connection.rollback()
            raise

    def commit_profile_result(
        self,
        action_id,
        claim_token,
        scenario,
        result,
        artifact,
    ):
        if not isinstance(scenario, ProfileScenario):
            raise TypeError("profile commits need a profile scenario")
        if not isinstance(result, ProfileResult):
            raise TypeError("profile commits need a profile result")
        if not isinstance(artifact, ArtifactRecord) or artifact.kind != "profile_result":
            raise TypeError("profile commits need a profile result artifact")
        now = _now()
        timestamp = _timestamp(now)
        self.connection.execute("begin immediate")
        try:
            row = self.connection.execute(
                """
                select * from actions where id = ? and kind = 'profile'
                    and status = 'running' and claim_token = ?
                    and lease_expires_at >= ?
                """,
                (action_id, claim_token, timestamp),
            ).fetchone()
            if row is None:
                raise RuntimeError("stale profile result completion")
            payload = _decode(row["payload_json"])
            stored_scenario = ProfileScenario.from_dict(payload["scenario"])
            if (
                scenario != stored_scenario
                or scenario.digest != row["profile_scenario_digest"]
                or scenario.backend.name != row["profile_backend"]
                or scenario.device.split(":", 1)[0] != row["required_device_type"]
            ):
                raise ValueError("profile scenario does not match its action")
            if (
                result.scenario_digest != scenario.digest
                or result.architecture_digest != row["architecture_digest"]
            ):
                raise ValueError("profile result identity does not match its action")
            objectives = self.connection.execute(
                "select definition_json from objective_sets where digest = ?",
                (payload["objective_set_digest"],),
            ).fetchone()
            if objectives is None:
                raise KeyError(payload["objective_set_digest"])
            names = {
                item["name"]
                for item in _decode(objectives["definition_json"])["objectives"]
            }
            metrics = {
                name: value
                for name, value in _profile_metrics(scenario, result).items()
                if name in names
            }
            if not metrics:
                raise ValueError("profile result has no objectives in its objective set")
            self._register_artifact(artifact)
            self.connection.execute(
                """
                insert into profile_results(
                    action_id, artifact_digest, architecture_digest,
                    scenario_digest, repetition, result_digest, created_at
                ) values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action_id,
                    artifact.digest,
                    row["architecture_digest"],
                    scenario.digest,
                    row["profile_repetition"],
                    result.digest,
                    timestamp,
                ),
            )
            observation_ids = []
            for name, value in sorted(metrics.items()):
                cursor = self.connection.execute(
                    """
                    insert into observations(
                        run_id, architecture_digest, objective_set_digest,
                        objective_name, value, variance, tokens, source,
                        artifact_digest, created_at
                    ) values (null, ?, ?, ?, ?, null, null, 'profile', ?, ?)
                    """,
                    (
                        row["architecture_digest"],
                        payload["objective_set_digest"],
                        name,
                        value,
                        artifact.digest,
                        timestamp,
                    ),
                )
                observation_ids.append(cursor.lastrowid)
                self._record_event(
                    "observation_added",
                    {"id": cursor.lastrowid, "objective_name": name},
                )
            action_result = {
                "artifact_digest": artifact.digest,
                "observation_ids": observation_ids,
                "result_digest": result.digest,
            }
            self.connection.execute(
                """
                update actions set status = 'completed', result_json = ?,
                    completed_at = ?, lease_expires_at = null where id = ?
                """,
                (canonical_json(action_result), timestamp, action_id),
            )
            self._record_event(
                "profile_result_committed",
                {
                    "action_id": action_id,
                    "architecture_digest": row["architecture_digest"],
                    **action_result,
                },
            )
            self._record_event(
                "action_finished",
                {"action_id": action_id, "status": "completed"},
            )
            self.connection.commit()
            return tuple(observation_ids)
        except Exception:
            self.connection.rollback()
            raise

    def commit_quality_evaluation(
        self,
        action_id,
        claim_token,
        nll,
        evaluated_tokens,
        artifact,
    ):
        if not math.isfinite(nll) or nll < 0:
            raise ValueError("quality evaluation nll must be finite and nonnegative")
        if evaluated_tokens < 1:
            raise ValueError("quality evaluation tokens must be positive")
        if (
            not isinstance(artifact, ArtifactRecord)
            or artifact.kind != "quality_evaluation"
        ):
            raise TypeError("quality evaluation commits need an evaluation artifact")
        now = _now()
        timestamp = _timestamp(now)
        self.connection.execute("begin immediate")
        try:
            row = self.connection.execute(
                """
                select actions.*, runs.checkpoint_digest as run_checkpoint_digest,
                    runs.tokens as run_tokens
                from actions join runs on runs.id = actions.run_id
                where actions.id = ? and actions.kind = 'evaluate'
                    and actions.status = 'running' and actions.claim_token = ?
                    and actions.lease_expires_at >= ?
                """,
                (action_id, claim_token, timestamp),
            ).fetchone()
            if row is None:
                raise RuntimeError("stale quality evaluation completion")
            payload = _decode(row["payload_json"])
            if (
                row["evaluation_checkpoint_digest"]
                != payload["checkpoint_digest"]
                or row["run_checkpoint_digest"] != payload["checkpoint_digest"]
                or row["run_tokens"] != payload["training_tokens"]
            ):
                raise ValueError("quality evaluation checkpoint changed")
            if evaluated_tokens != payload["expected_evaluation_tokens"]:
                raise ValueError("quality evaluation did not cover its full partition")
            self._register_artifact(artifact)
            self.connection.execute(
                """
                insert into quality_evaluations(
                    action_id, artifact_digest, run_id, checkpoint_digest,
                    evaluated_tokens, nll, created_at
                ) values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action_id,
                    artifact.digest,
                    row["run_id"],
                    payload["checkpoint_digest"],
                    evaluated_tokens,
                    nll,
                    timestamp,
                ),
            )
            observation_ids = []
            for objective_set_digest in payload["objective_set_digests"]:
                cursor = self.connection.execute(
                    """
                    insert into observations(
                        run_id, architecture_digest, objective_set_digest,
                        objective_name, value, variance, tokens, source,
                        artifact_digest, created_at
                    ) values (?, ?, ?, 'quality.target_nll', ?, null, ?,
                        'quality_evaluation', ?, ?)
                    """,
                    (
                        row["run_id"],
                        row["architecture_digest"],
                        objective_set_digest,
                        nll,
                        payload["training_tokens"],
                        artifact.digest,
                        timestamp,
                    ),
                )
                observation_ids.append(cursor.lastrowid)
                self._record_event(
                    "observation_added",
                    {
                        "id": cursor.lastrowid,
                        "objective_name": "quality.target_nll",
                    },
                )
            action_result = {
                "artifact_digest": artifact.digest,
                "evaluated_tokens": evaluated_tokens,
                "nll": nll,
                "observation_ids": observation_ids,
            }
            self.connection.execute(
                """
                update actions set status = 'completed', result_json = ?,
                    completed_at = ?, lease_expires_at = null where id = ?
                """,
                (canonical_json(action_result), timestamp, action_id),
            )
            self._record_event(
                "quality_evaluation_committed",
                {"action_id": action_id, "run_id": row["run_id"], **action_result},
            )
            self._record_event(
                "action_finished",
                {"action_id": action_id, "status": "completed"},
            )
            self.connection.commit()
            return tuple(observation_ids)
        except Exception:
            self.connection.rollback()
            raise

    def quality_evaluation(self, run_id, checkpoint_digest):
        row = self.connection.execute(
            """
            select * from quality_evaluations
            where run_id = ? and checkpoint_digest = ?
            """,
            (run_id, checkpoint_digest),
        ).fetchone()
        return dict(row) if row is not None else None

    def register_posterior_report(
        self,
        report_digest,
        evidence_digest,
        artifact,
        definition,
        anchors,
    ):
        for name, digest in (
            ("report", report_digest),
            ("evidence", evidence_digest),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"posterior {name} digests must be lowercase sha256")
        if not isinstance(artifact, ArtifactRecord) or artifact.kind != "posterior_shadow":
            raise TypeError("posterior reports need a shadow artifact")
        anchors = tuple(anchors)
        if not anchors or len(set(anchors)) != len(anchors):
            raise ValueError("posterior anchors must be nonempty and unique")
        encoded = canonical_json(definition)
        self.connection.execute("begin immediate")
        try:
            row = self.connection.execute(
                "select * from posterior_reports where evidence_digest = ?",
                (evidence_digest,),
            ).fetchone()
            if row is not None:
                stored_anchors = tuple(
                    item["architecture_digest"]
                    for item in self.connection.execute(
                        """
                        select architecture_digest from posterior_anchors
                        where report_digest = ? order by position
                        """,
                        (row["digest"],),
                    )
                )
                stored = (
                    row["digest"],
                    row["artifact_digest"],
                    row["definition_json"],
                    stored_anchors,
                )
                expected = (report_digest, artifact.digest, encoded, anchors)
                if stored != expected:
                    raise ValueError("posterior evidence already has a different report")
                self.connection.rollback()
                return False
            known = {
                item["digest"]
                for item in self.connection.execute(
                    f"select digest from architectures where digest in ({','.join('?' for _ in anchors)})",
                    anchors,
                )
            }
            if known != set(anchors):
                raise KeyError("posterior anchors reference unknown architectures")
            self._register_artifact(artifact)
            self.connection.execute(
                "insert into posterior_reports values (?, ?, ?, ?, ?)",
                (
                    report_digest,
                    evidence_digest,
                    artifact.digest,
                    encoded,
                    _timestamp(),
                ),
            )
            for position, architecture_digest in enumerate(anchors):
                self.connection.execute(
                    "insert into posterior_anchors values (?, ?, ?)",
                    (report_digest, position, architecture_digest),
                )
            self._record_event(
                "posterior_report_registered",
                {
                    "anchors": anchors,
                    "artifact_digest": artifact.digest,
                    "evidence_digest": evidence_digest,
                    "report_digest": report_digest,
                },
            )
            self.connection.commit()
            return True
        except Exception:
            self.connection.rollback()
            raise

    def posterior_report(self, evidence_digest=None):
        if evidence_digest is None:
            row = self.connection.execute(
                "select * from posterior_reports order by created_at desc, digest desc limit 1"
            ).fetchone()
        else:
            row = self.connection.execute(
                "select * from posterior_reports where evidence_digest = ?",
                (evidence_digest,),
            ).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["definition"] = _decode(value.pop("definition_json"))
        value["anchors"] = tuple(
            item["architecture_digest"]
            for item in self.connection.execute(
                """
                select architecture_digest from posterior_anchors
                where report_digest = ? order by position
                """,
                (row["digest"],),
            )
        )
        return value

    def _register_artifact(self, artifact):
        encoded = (
            artifact.digest,
            artifact.kind,
            artifact.size,
            artifact.uri,
            artifact.media_type,
        )
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

    def register_artifact(self, artifact):
        if not isinstance(artifact, ArtifactRecord):
            raise TypeError("artifact registration needs an artifact record")
        with self.connection:
            return self._register_artifact(artifact)

    def artifact(self, digest):
        row = self.connection.execute(
            "select * from artifacts where digest = ?",
            (digest,),
        ).fetchone()
        if row is None:
            raise KeyError(digest)
        return ArtifactRecord(
            kind=row["kind"],
            digest=row["digest"],
            size=row["size"],
            uri=row["uri"],
            media_type=row["media_type"],
        )

    def checkpoint(self, digest):
        row = self.connection.execute(
            """
            select run_checkpoints.*, runs.architecture_digest,
                runs.protocol_digest, runs.seed_bundle_digest
            from run_checkpoints join runs on runs.id = run_checkpoints.run_id
            where run_checkpoints.artifact_digest = ?
            """,
            (digest,),
        ).fetchone()
        if row is None:
            raise KeyError(digest)
        return RunCheckpoint(
            artifact=self.artifact(digest),
            architecture_digest=row["architecture_digest"],
            protocol_digest=row["protocol_digest"],
            seed_bundle_digest=row["seed_bundle_digest"],
            steps=row["steps"],
            tokens=row["tokens"],
            parent_digest=row["parent_digest"],
            format_version=row["format_version"],
        )

    def checkpoints(self, run_id):
        return tuple(
            self.checkpoint(row["artifact_digest"])
            for row in self.connection.execute(
                """
                select artifact_digest from run_checkpoints
                where run_id = ? order by tokens
                """,
                (run_id,),
            )
        )

    def _add_artifact_edge(self, edge):
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
            self.connection.execute(
                "insert into artifact_edges values (?, ?, ?)",
                (edge.parent_digest, edge.child_digest, edge.relation),
            )
        except sqlite3.IntegrityError:
            return False
        self._record_event("artifact_edge_added", asdict(edge))
        return True

    def add_artifact_edge(self, edge):
        if not isinstance(edge, ArtifactEdge):
            raise TypeError("artifact lineage needs an artifact edge")
        with self.connection:
            return self._add_artifact_edge(edge)
