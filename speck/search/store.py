"""transactional study storage for architecture search."""

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from speck.model import Config
from speck.search.architecture import architecture_hash, canonical_settings
from speck.search.evolution import (
    EvaluatedCandidate,
    SelectionMetrics,
    nondominated_sort,
)


schema_version = 3


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class StudyStore:
    def __init__(self, path, readonly=False):
        self.path = Path(path)
        if readonly:
            self.connection = sqlite3.connect(
                f"{self.path.resolve().as_uri()}?mode=ro", uri=True
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        try:
            if readonly:
                self._validate_schema()
            else:
                self.connection.execute("pragma foreign_keys = on")
                self.connection.execute("pragma journal_mode = wal")
                self._create_schema()
        except Exception:
            self.connection.close()
            raise

    def close(self):
        self.connection.close()

    def _validate_schema(self):
        candidates = self.connection.execute(
            "select name from sqlite_master where type = 'table' and name = 'candidates'"
        ).fetchone()
        row = self.connection.execute(
            "select value from metadata where key = 'schema_version'"
        ).fetchone()
        if (
            candidates is None
            or row is None
            or int(row["value"]) not in {1, 2, schema_version}
        ):
            raise ValueError("unsupported search database schema")

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
                created_at text not null,
                updated_at text not null
            );
            create table if not exists candidates (
                id integer primary key autoincrement,
                architecture_hash text not null unique,
                architecture_json text not null,
                status text not null,
                seed integer not null,
                mutation_json text not null,
                repairs_json text not null,
                result_json text,
                error text,
                in_population integer not null default 0,
                is_frontier integer not null default 0,
                pareto_rank integer,
                crowding real,
                novelty real,
                selection_pending integer not null default 0,
                created_at text not null,
                started_at text,
                completed_at text
            );
            create table if not exists parents (
                candidate_id integer not null references candidates(id) on delete cascade,
                parent_id integer not null references candidates(id),
                primary key (candidate_id, parent_id)
            );
            create table if not exists attempts (
                id integer primary key autoincrement,
                candidate_id integer not null references candidates(id) on delete cascade,
                status text not null,
                started_at text not null,
                completed_at text,
                pid integer,
                error text
            );
            """
        )
        row = self.connection.execute(
            "select value from metadata where key = 'schema_version'"
        ).fetchone()
        if row is None:
            self.connection.execute(
                "insert into metadata(key, value) values ('schema_version', ?)",
                (str(schema_version),),
            )
        elif int(row["value"]) == 1:
            self.connection.execute(
                "alter table candidates add column selection_pending integer not null default 0"
            )
            self.connection.execute("alter table attempts add column pid integer")
            self.connection.execute(
                "update metadata set value = ? where key = 'schema_version'",
                (str(schema_version),),
            )
        elif int(row["value"]) == 2:
            self.connection.execute("alter table attempts add column pid integer")
            self.connection.execute(
                "update metadata set value = ? where key = 'schema_version'",
                (str(schema_version),),
            )
        elif int(row["value"]) != schema_version:
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
            "insert into study values (1, ?, ?, 'running', ?, ?)",
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
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def set_study_status(self, status):
        self.connection.execute(
            "update study set status = ?, updated_at = ? where id = 1",
            (status, _now()),
        )
        self.connection.commit()

    def add_candidate(
        self,
        config,
        seed,
        mutation,
        repairs=(),
        parent_id=None,
    ):
        try:
            cursor = self.connection.execute(
                """
                insert into candidates(
                    architecture_hash, architecture_json, status, seed,
                    mutation_json, repairs_json, created_at
                ) values (?, ?, 'pending', ?, ?, ?, ?)
                """,
                (
                    architecture_hash(config),
                    _json(canonical_settings(config)),
                    seed,
                    _json(mutation),
                    _json(repairs),
                    _now(),
                ),
            )
        except sqlite3.IntegrityError:
            return None
        candidate_id = cursor.lastrowid
        if parent_id is not None:
            self.connection.execute(
                "insert into parents(candidate_id, parent_id) values (?, ?)",
                (candidate_id, parent_id),
            )
        self.connection.commit()
        return candidate_id

    def candidate(self, candidate_id):
        row = self.connection.execute(
            "select * from candidates where id = ?", (candidate_id,)
        ).fetchone()
        if row is None:
            raise KeyError(candidate_id)
        parent_rows = self.connection.execute(
            "select parent_id from parents where candidate_id = ? order by parent_id",
            (candidate_id,),
        ).fetchall()
        return self._candidate_dict(row, [parent["parent_id"] for parent in parent_rows])

    def _candidate_dict(self, row, parents=()):
        return {
            "id": row["id"],
            "architecture_hash": row["architecture_hash"],
            "config": json.loads(row["architecture_json"]),
            "status": row["status"],
            "seed": row["seed"],
            "mutation": json.loads(row["mutation_json"]),
            "repairs": json.loads(row["repairs_json"]),
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error": row["error"],
            "parents": list(parents),
            "in_population": bool(row["in_population"]),
            "is_frontier": bool(row["is_frontier"]),
            "pareto_rank": row["pareto_rank"],
            "crowding": (
                row["crowding"]
                if row["crowding"] is None or math.isfinite(row["crowding"])
                else None
            ),
            "novelty": row["novelty"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
        }

    def candidates(self, status=None):
        if status is None:
            rows = self.connection.execute("select * from candidates order by id").fetchall()
        else:
            rows = self.connection.execute(
                "select * from candidates where status = ? order by id", (status,)
            ).fetchall()
        values = []
        for row in rows:
            parent_rows = self.connection.execute(
                "select parent_id from parents where candidate_id = ? order by parent_id",
                (row["id"],),
            ).fetchall()
            values.append(
                self._candidate_dict(
                    row, [parent["parent_id"] for parent in parent_rows]
                )
            )
        return values

    def evaluated_candidates(self):
        values = []
        for candidate in self.candidates("completed"):
            result = candidate["result"]
            values.append(
                EvaluatedCandidate(
                    id=candidate["id"],
                    config=Config.from_dict(candidate["config"]),
                    objectives=result["objectives"],
                )
            )
        return values

    def start_attempt(self, candidate_id):
        now = _now()
        with self.connection:
            claim = self.connection.execute(
                """
                update candidates
                set status = 'running', started_at = ?, completed_at = null, error = null
                where id = ? and status = 'pending'
                """,
                (now, candidate_id),
            )
            if claim.rowcount != 1:
                raise RuntimeError("candidate is not pending")
            cursor = self.connection.execute(
                "insert into attempts(candidate_id, status, started_at) values (?, 'running', ?)",
                (candidate_id, now),
            )
        return cursor.lastrowid

    def complete_attempt(self, candidate_id, attempt_id, result):
        now = _now()
        with self.connection:
            self.connection.execute(
                "update attempts set status = 'completed', completed_at = ? where id = ?",
                (now, attempt_id),
            )
            self.connection.execute(
                """
                update candidates
                set status = 'completed', result_json = ?, completed_at = ?,
                    error = null, selection_pending = 1
                where id = ?
                """,
                (_json(result), now, candidate_id),
            )

    def fail_attempt(self, candidate_id, attempt_id, error, retry=False):
        now = _now()
        status = "pending" if retry else "failed"
        completed_at = None if retry else now
        with self.connection:
            self.connection.execute(
                "update attempts set status = 'failed', completed_at = ?, error = ? where id = ?",
                (now, error, attempt_id),
            )
            self.connection.execute(
                "update candidates set status = ?, completed_at = ?, error = ? where id = ?",
                (status, completed_at, error, candidate_id),
            )

    def recover_running(self):
        now = _now()
        with self.connection:
            self.connection.execute(
                """
                update attempts set status = 'interrupted', completed_at = ?, error = 'coordinator interrupted'
                where status = 'running'
                """,
                (now,),
            )
            cursor = self.connection.execute(
                "update candidates set status = 'pending', error = 'coordinator interrupted' where status = 'running'"
            )
        return cursor.rowcount

    def attempt_count(self, candidate_id):
        row = self.connection.execute(
            "select count(*) as count from attempts where candidate_id = ?",
            (candidate_id,),
        ).fetchone()
        return row["count"]

    def failed_attempt_count(self, candidate_id):
        row = self.connection.execute(
            """
            select count(*) as count from attempts
            where candidate_id = ? and status = 'failed'
            """,
            (candidate_id,),
        ).fetchone()
        return row["count"]

    def set_attempt_pid(self, attempt_id, pid):
        with self.connection:
            cursor = self.connection.execute(
                "update attempts set pid = ? where id = ? and status = 'running'",
                (pid, attempt_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("attempt is not running")

    def running_attempts(self):
        rows = self.connection.execute(
            "select id, candidate_id, pid from attempts where status = 'running' order by id"
        ).fetchall()
        return [dict(row) for row in rows]

    def running_attempt(self, candidate_id):
        row = self.connection.execute(
            """
            select id from attempts
            where candidate_id = ? and status = 'running'
            order by id desc limit 1
            """,
            (candidate_id,),
        ).fetchone()
        return row["id"] if row else None

    def update_selection(self, population_ids, frontier_ids, metrics):
        with self.connection:
            self.connection.execute(
                """
                update candidates
                set in_population = 0, is_frontier = 0,
                    pareto_rank = null, crowding = null, novelty = null,
                    selection_pending = 0
                """
            )
            for candidate_id, values in metrics.items():
                self.connection.execute(
                    """
                    update candidates
                    set pareto_rank = ?, crowding = ?, novelty = ?
                    where id = ?
                    """,
                    (values.rank, values.crowding, values.novelty, candidate_id),
                )
            self.connection.executemany(
                "update candidates set in_population = 1 where id = ?",
                ((candidate_id,) for candidate_id in population_ids),
            )
            self.connection.executemany(
                "update candidates set is_frontier = 1 where id = ?",
                ((candidate_id,) for candidate_id in frontier_ids),
            )

    def population(self):
        rows = self.connection.execute(
            "select id from candidates where in_population = 1 order by id"
        ).fetchall()
        return [row["id"] for row in rows]

    def selection_pending(self):
        rows = self.connection.execute(
            """
            select id from candidates
            where status = 'completed' and selection_pending = 1
            order by id
            """
        ).fetchall()
        return [row["id"] for row in rows]

    def frontier(self):
        candidates = self.evaluated_candidates()
        if not candidates:
            return []
        objective_names = tuple(candidates[0].objectives)
        front = nondominated_sort(candidates, objective_names)[0]
        return [self.candidate(candidate.id) for candidate in front]

    def lineage(self, candidate_id):
        seen = set()
        pending = [candidate_id]
        lineage = []
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            candidate = self.candidate(current)
            lineage.append(candidate)
            pending.extend(candidate["parents"])
        return sorted(lineage, key=lambda candidate: candidate["id"])

    def summary(self):
        rows = self.connection.execute(
            "select status, count(*) as count from candidates group by status"
        ).fetchall()
        return {
            "study": self.study(),
            "candidates": {row["status"]: row["count"] for row in rows},
            "population": len(self.population()),
            "frontier": len(self.frontier()),
        }
