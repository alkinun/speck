"""Give checkpoint rollback an indexed foreign-key lookup without changing logical data."""

import time
from contextlib import contextmanager

import speck.data.production_data as production_data


def ensure_recovery_index(connection):
    started = time.perf_counter()
    before = {row[1]: row for row in connection.execute("PRAGMA index_list(bands)")}
    connection.execute("CREATE INDEX IF NOT EXISTS bands_doc_seq ON bands(doc_seq)")
    index = next(
        row for row in connection.execute("PRAGMA index_list(bands)") if row[1] == "bands_doc_seq"
    )
    columns = [row[2] for row in connection.execute("PRAGMA index_info(bands_doc_seq)")]
    if columns != ["doc_seq"] or index[2] != 0 or index[4] != 0:
        raise ValueError("recovery lookup must be a complete nonunique doc_seq index")
    return {
        "name": "bands_doc_seq",
        "columns": columns,
        "created": "bands_doc_seq" not in before,
        "elapsed_seconds": time.perf_counter() - started,
        "purpose": "Avoid a full bands-table scan for every post-checkpoint document deleted by ON DELETE CASCADE.",
    }


@contextmanager
def recovery_index_policy(observations):
    """Apply the physical index at database open, before the preserved resume logic."""
    original = production_data._database

    def indexed_database(*args, **kwargs):
        connection = original(*args, **kwargs)
        try:
            observations.append(ensure_recovery_index(connection))
            return connection
        except BaseException:
            connection.close()
            raise

    production_data._database = indexed_database
    try:
        yield
    finally:
        production_data._database = original
