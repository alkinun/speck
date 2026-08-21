"""transactional state for multi-fidelity architecture search."""

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from speck.model import Config
from speck.search.architecture import architecture_hash, canonical_settings


format_version = 2
schema_version = 1


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class SearchStudy:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("pragma foreign_keys = on")
        self.connection.execute("pragma journal_mode = wal")
        self._create_schema()

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
                status text not null,
                recommendations_json text,
                created_at text not null,
                updated_at text not null
            );
            create table if not exists architectures (
                id integer primary key autoincrement,
                architecture_hash text not null unique,
                architecture_json text not null,
                static_json text not null,
                cohort integer not null,
                slot integer not null,
                generation_seed integer not null,
                operator text not null,
                operation_json text not null,
                repairs_json text not null,
                created_at text not null,
                unique(cohort, slot)
            );
            create table if not exists architecture_parents (
                child_id integer not null references architectures(id) on delete cascade,
                parent_id integer not null references architectures(id),
                role text not null check (role in ('primary', 'secondary')),
                primary key(child_id, role),
                unique(child_id, parent_id)
            );
            create table if not exists architecture_rungs (
                id integer primary key autoincrement,
                architecture_id integer not null references architectures(id) on delete cascade,
                rung integer not null,
                status text not null,
                aggregate_json text,
                decision_json text,
                pareto_rank integer,
                crowding real,
                novelty real,
                created_at text not null,
                completed_at text,
                unique(architecture_id, rung)
            );
            create table if not exists trials (
                id integer primary key autoincrement,
                architecture_rung_id integer not null references architecture_rungs(id) on delete cascade,
                architecture_id integer not null references architectures(id) on delete cascade,
                rung integer not null,
                seed_index integer not null,
                seed integer not null,
                status text not null,
                result_json text,
                error text,
                created_at text not null,
                started_at text,
                completed_at text,
                unique(architecture_id, rung, seed_index)
            );
            create table if not exists attempts (
                id integer primary key autoincrement,
                trial_id integer not null references trials(id) on delete cascade,
                status text not null,
                pid integer,
                error text,
                started_at text not null,
                completed_at text
            );
            create table if not exists operator_outcomes (
                architecture_id integer primary key references architectures(id) on delete cascade,
                operator text not null,
                success integer not null,
                credited_at text not null
            );
            create table if not exists events (
                id integer primary key autoincrement,
                event_key text not null unique,
                kind text not null,
                payload_json text not null,
                created_at text not null
            );
            """
        )
        stored_format = self.connection.execute(
            "select value from metadata where key = 'search_format_version'"
        ).fetchone()
        if stored_format is None:
            legacy = self.connection.execute(
                "select name from sqlite_master where type = 'table' and name = 'candidates'"
            ).fetchone()
            if legacy is not None:
                raise ValueError("legacy search study requires a new study name")
            self.connection.executemany(
                "insert into metadata(key, value) values (?, ?)",
                (
                    ("search_format_version", str(format_version)),
                    ("schema_version", str(schema_version)),
                ),
            )
        elif int(stored_format["value"]) != format_version:
            raise ValueError("unsupported search study format")
        stored_schema = self.connection.execute(
            "select value from metadata where key = 'schema_version'"
        ).fetchone()
        if stored_schema is None or int(stored_schema["value"]) != schema_version:
            raise ValueError("unsupported search database schema")
        self.connection.commit()

    def initialize(self, config, provenance):
        row = self.connection.execute("select * from study where id = 1").fetchone()
        if row is not None:
            if json.loads(row["config_json"]) != config:
                raise ValueError("study configuration changed")
            if json.loads(row["provenance_json"]) != provenance:
                raise ValueError("study provenance changed")
            return False
        now = _now()
        self.connection.execute(
            "insert into study values (1, ?, ?, 'running', null, ?, ?)",
            (_json(config), _json(provenance), now, now),
        )
        self.connection.commit()
        return True

    def study(self):
        row = self.connection.execute("select * from study where id = 1").fetchone()
        if row is None:
            raise ValueError("study is not initialized")
        return {
            "config": json.loads(row["config_json"]),
            "provenance": json.loads(row["provenance_json"]),
            "status": row["status"],
            "recommendations": (
                json.loads(row["recommendations_json"])
                if row["recommendations_json"]
                else None
            ),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def set_status(self, status):
        self.connection.execute(
            "update study set status = ?, updated_at = ? where id = 1",
            (status, _now()),
        )
        self.connection.commit()

    def set_recommendations(self, recommendations):
        self.connection.execute(
            "update study set recommendations_json = ?, updated_at = ? where id = 1",
            (_json(recommendations), _now()),
        )
        self.connection.commit()

    def finalize(self, status, recommendations=None):
        recommendations_json = (
            _json(recommendations) if recommendations is not None else None
        )
        current = self.connection.execute(
            "select status, recommendations_json from study where id = 1"
        ).fetchone()
        if current is None:
            raise ValueError("study is not initialized")
        if (
            current["status"] == status
            and current["recommendations_json"] == recommendations_json
        ):
            return False
        with self.connection:
            self.connection.execute(
                """
                update study set status = ?, recommendations_json = ?, updated_at = ?
                where id = 1
                """,
                (status, recommendations_json, _now()),
            )
        return True

    def _insert_architecture(
        self,
        config,
        static,
        cohort,
        slot,
        generation_seed,
        operation,
        repairs,
        parents,
    ):
        cursor = self.connection.execute(
            """
            insert into architectures(
                architecture_hash, architecture_json, static_json,
                cohort, slot, generation_seed, operator,
                operation_json, repairs_json, created_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                architecture_hash(config),
                _json(canonical_settings(config)),
                _json(static),
                cohort,
                slot,
                generation_seed,
                operation["operator"],
                _json(operation),
                _json(repairs),
                _now(),
            ),
        )
        architecture_id = cursor.lastrowid
        for role, parent_id in parents:
            self.connection.execute(
                "insert into architecture_parents values (?, ?, ?)",
                (architecture_id, parent_id, role),
            )
        return architecture_id

    def add_architecture(
        self,
        config,
        static,
        cohort,
        slot,
        generation_seed,
        operation,
        repairs=(),
        parents=(),
    ):
        try:
            with self.connection:
                return self._insert_architecture(
                    config,
                    static,
                    cohort,
                    slot,
                    generation_seed,
                    operation,
                    repairs,
                    parents,
                )
        except sqlite3.IntegrityError:
            return None

    def add_architecture_with_rung(
        self,
        config,
        static,
        cohort,
        slot,
        generation_seed,
        operation,
        repairs,
        parents,
        rung,
        seeds,
    ):
        try:
            with self.connection:
                architecture_id = self._insert_architecture(
                    config,
                    static,
                    cohort,
                    slot,
                    generation_seed,
                    operation,
                    repairs,
                    parents,
                )
                self._insert_rung(architecture_id, rung, seeds)
                return architecture_id
        except sqlite3.IntegrityError:
            return None

    def _architecture(self, row):
        parent_rows = self.connection.execute(
            """
            select parent_id, role from architecture_parents
            where child_id = ? order by role
            """,
            (row["id"],),
        ).fetchall()
        return {
            "id": row["id"],
            "architecture_hash": row["architecture_hash"],
            "config": json.loads(row["architecture_json"]),
            "static": json.loads(row["static_json"]),
            "cohort": row["cohort"],
            "slot": row["slot"],
            "generation_seed": row["generation_seed"],
            "operator": row["operator"],
            "operation": json.loads(row["operation_json"]),
            "repairs": json.loads(row["repairs_json"]),
            "parents": [
                {"id": parent["parent_id"], "role": parent["role"]}
                for parent in parent_rows
            ],
            "created_at": row["created_at"],
        }

    def architecture(self, architecture_id):
        row = self.connection.execute(
            "select * from architectures where id = ?", (architecture_id,)
        ).fetchone()
        if row is None:
            raise KeyError(architecture_id)
        return self._architecture(row)

    def architectures(self):
        rows = self.connection.execute(
            "select * from architectures order by cohort, slot"
        ).fetchall()
        return [self._architecture(row) for row in rows]

    def _insert_rung(self, architecture_id, rung, seeds):
        now = _now()
        cursor = self.connection.execute(
            """
            insert into architecture_rungs(
                architecture_id, rung, status, created_at
            ) values (?, ?, 'active', ?)
            """,
            (architecture_id, rung, now),
        )
        architecture_rung_id = cursor.lastrowid
        self.connection.executemany(
            """
            insert into trials(
                architecture_rung_id, architecture_id, rung,
                seed_index, seed, status, created_at
            ) values (?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                (
                    architecture_rung_id,
                    architecture_id,
                    rung,
                    seed_index,
                    seed,
                    now,
                )
                for seed_index, seed in enumerate(seeds)
            ),
        )

    def add_rung(self, architecture_id, rung, seeds):
        try:
            with self.connection:
                self._insert_rung(architecture_id, rung, seeds)
        except sqlite3.IntegrityError:
            return False
        return True

    def promote(self, architecture_id, source_rung, destination_rung, seeds, decision):
        now = _now()
        try:
            with self.connection:
                source = self.connection.execute(
                    """
                    update architecture_rungs
                    set status = 'promoted', decision_json = ?,
                        completed_at = coalesce(completed_at, ?)
                    where architecture_id = ? and rung = ? and status = 'complete'
                    """,
                    (_json(decision), now, architecture_id, source_rung),
                )
                if source.rowcount != 1:
                    raise RuntimeError("source rung is not promotable")
                self._insert_rung(architecture_id, destination_rung, seeds)
        except sqlite3.IntegrityError:
            return False
        return True

    def _rung(self, row):
        return {
            "id": row["id"],
            "architecture_id": row["architecture_id"],
            "rung": row["rung"],
            "status": row["status"],
            "aggregate": json.loads(row["aggregate_json"]) if row["aggregate_json"] else None,
            "decision": json.loads(row["decision_json"]) if row["decision_json"] else None,
            "pareto_rank": row["pareto_rank"],
            "crowding": _clean(row["crowding"]),
            "novelty": row["novelty"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
        }

    def rungs(self, rung=None, status=None):
        clauses = []
        values = []
        if rung is not None:
            clauses.append("rung = ?")
            values.append(rung)
        if status is not None:
            clauses.append("status = ?")
            values.append(status)
        where = f" where {' and '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"select * from architecture_rungs{where} order by rung, architecture_id",
            values,
        ).fetchall()
        return [self._rung(row) for row in rows]

    def rung(self, architecture_id, rung):
        row = self.connection.execute(
            "select * from architecture_rungs where architecture_id = ? and rung = ?",
            (architecture_id, rung),
        ).fetchone()
        if row is None:
            raise KeyError((architecture_id, rung))
        return self._rung(row)

    def update_rung(
        self,
        architecture_id,
        rung,
        status,
        aggregate=None,
        decision=None,
        rank=None,
        crowding=None,
        novelty=None,
    ):
        completed_at = _now() if status in {"complete", "promoted", "stopped", "failed"} else None
        with self.connection:
            cursor = self.connection.execute(
                """
                update architecture_rungs
                set status = ?, aggregate_json = coalesce(?, aggregate_json),
                    decision_json = coalesce(?, decision_json), pareto_rank = ?,
                    crowding = ?, novelty = ?, completed_at = coalesce(?, completed_at)
                where architecture_id = ? and rung = ?
                """,
                (
                    status,
                    _json(aggregate) if aggregate is not None else None,
                    _json(decision) if decision is not None else None,
                    rank,
                    crowding,
                    novelty,
                    completed_at,
                    architecture_id,
                    rung,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError((architecture_id, rung))

    def _trial(self, row):
        return {
            "id": row["id"],
            "architecture_rung_id": row["architecture_rung_id"],
            "architecture_id": row["architecture_id"],
            "rung": row["rung"],
            "seed_index": row["seed_index"],
            "seed": row["seed"],
            "status": row["status"],
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error": row["error"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
        }

    def trial(self, trial_id):
        row = self.connection.execute(
            "select * from trials where id = ?", (trial_id,)
        ).fetchone()
        if row is None:
            raise KeyError(trial_id)
        return self._trial(row)

    def trials(self, status=None, architecture_id=None, rung=None):
        clauses = []
        values = []
        for name, value in (
            ("status", status),
            ("architecture_id", architecture_id),
            ("rung", rung),
        ):
            if value is not None:
                clauses.append(f"{name} = ?")
                values.append(value)
        where = f" where {' and '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"select * from trials{where} order by rung desc, architecture_id, seed_index",
            values,
        ).fetchall()
        return [self._trial(row) for row in rows]

    def start_attempt(self, trial_id):
        now = _now()
        with self.connection:
            claim = self.connection.execute(
                """
                update trials set status = 'running', started_at = ?,
                    completed_at = null, error = null
                where id = ? and status = 'pending'
                """,
                (now, trial_id),
            )
            if claim.rowcount != 1:
                raise RuntimeError("trial is not pending")
            cursor = self.connection.execute(
                "insert into attempts(trial_id, status, started_at) values (?, 'running', ?)",
                (trial_id, now),
            )
        return cursor.lastrowid

    def set_attempt_pid(self, attempt_id, pid):
        with self.connection:
            cursor = self.connection.execute(
                "update attempts set pid = ? where id = ? and status = 'running'",
                (pid, attempt_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("attempt is not running")

    def complete_attempt(self, trial_id, attempt_id, result):
        now = _now()
        with self.connection:
            attempt = self.connection.execute(
                """
                update attempts set status = 'completed', completed_at = ?
                where id = ? and trial_id = ? and status = 'running'
                """,
                (now, attempt_id, trial_id),
            )
            trial = self.connection.execute(
                """
                update trials set status = 'completed', result_json = ?,
                    completed_at = ?, error = null
                where id = ? and status = 'running'
                """,
                (_json(result), now, trial_id),
            )
            if attempt.rowcount != 1 or trial.rowcount != 1:
                raise RuntimeError("stale trial result")

    def fail_attempt(self, trial_id, attempt_id, error, retry=False):
        now = _now()
        trial_status = "pending" if retry else "failed"
        completed_at = None if retry else now
        with self.connection:
            attempt = self.connection.execute(
                """
                update attempts set status = 'failed', completed_at = ?, error = ?
                where id = ? and trial_id = ? and status = 'running'
                """,
                (now, error, attempt_id, trial_id),
            )
            trial = self.connection.execute(
                """
                update trials set status = ?, completed_at = ?, error = ?
                where id = ? and status = 'running'
                """,
                (trial_status, completed_at, error, trial_id),
            )
            if attempt.rowcount != 1 or trial.rowcount != 1:
                raise RuntimeError("stale trial failure")

    def failed_attempt_count(self, trial_id):
        row = self.connection.execute(
            "select count(*) as count from attempts where trial_id = ? and status = 'failed'",
            (trial_id,),
        ).fetchone()
        return row["count"]

    def running_attempts(self):
        return [
            dict(row)
            for row in self.connection.execute(
                "select id, trial_id, pid from attempts where status = 'running' order by id"
            )
        ]

    def running_attempt(self, trial_id):
        row = self.connection.execute(
            """
            select id from attempts where trial_id = ? and status = 'running'
            order by id desc limit 1
            """,
            (trial_id,),
        ).fetchone()
        return row["id"] if row else None

    def recover_running(self):
        now = _now()
        with self.connection:
            self.connection.execute(
                """
                update attempts set status = 'interrupted', completed_at = ?,
                    error = 'coordinator interrupted' where status = 'running'
                """,
                (now,),
            )
            cursor = self.connection.execute(
                """
                update trials set status = 'pending', error = 'coordinator interrupted'
                where status = 'running'
                """
            )
        return cursor.rowcount

    def record_outcome(self, architecture_id, operator, success):
        try:
            self.connection.execute(
                "insert into operator_outcomes values (?, ?, ?, ?)",
                (architecture_id, operator, int(success), _now()),
            )
            self.connection.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def outcomes(self):
        return [dict(row) for row in self.connection.execute("select * from operator_outcomes order by architecture_id")]

    def record_event(self, event_key, kind, payload):
        try:
            self.connection.execute(
                "insert into events(event_key, kind, payload_json, created_at) values (?, ?, ?, ?)",
                (event_key, kind, _json(payload), _now()),
            )
            self.connection.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def lineage(self, architecture_id):
        seen = set()
        pending = [architecture_id]
        values = []
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            architecture = self.architecture(current)
            values.append(architecture)
            pending.extend(parent["id"] for parent in architecture["parents"])
        return sorted(values, key=lambda item: item["id"])

    def summary(self):
        trial_counts = {
            row["status"]: row["count"]
            for row in self.connection.execute(
                "select status, count(*) as count from trials group by status"
            )
        }
        rung_counts = {
            str(row["rung"]): {"total": row["total"], "complete": row["complete"]}
            for row in self.connection.execute(
                """
                select rung, count(*) as total,
                    sum(case when status in ('complete','promoted','stopped') then 1 else 0 end) as complete
                from architecture_rungs group by rung order by rung
                """
            )
        }
        return {
            "study": self.study(),
            "architectures": len(self.architectures()),
            "trials": trial_counts,
            "rungs": rung_counts,
        }
