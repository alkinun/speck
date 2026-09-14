import pytest

import speck.data.production_data as data
from speck.data.recovery_index import ensure_recovery_index, recovery_index_policy


def populate(path):
    c = data._database(path)
    for i in range(8):
        c.execute(
            "INSERT INTO docs VALUES (?,?,?,?,?,?,?,?)",
            (i, i, 0, "fixture", i, 0, f"{i:064x}", f"{i:064x}"),
        )
        c.executemany("INSERT INTO bands VALUES (?,?,?)", [(band, b"hash", i) for band in range(3)])
    c.commit()
    c.close()


def cleanup(c):
    c.execute("DELETE FROM docs WHERE processed_index>=4")
    c.commit()
    assert c.execute("PRAGMA foreign_key_check").fetchall() == []
    return (
        c.execute("SELECT * FROM docs ORDER BY doc_seq").fetchall(),
        c.execute("SELECT * FROM bands ORDER BY doc_seq,band").fetchall(),
    )


def test_indexed_recovery_matches_legacy_rows_and_avoids_cascade_scan(tmp_path):
    baseline, indexed = tmp_path / "baseline.db", tmp_path / "indexed.db"
    populate(baseline)
    populate(indexed)
    old = data._database(baseline)
    expected = cleanup(old)
    old.close()
    original = data._database
    observed = []
    with recovery_index_policy(observed):
        c = data._database(indexed)
        query = c.execute("EXPLAIN QUERY PLAN DELETE FROM docs WHERE processed_index>=4").fetchall()
        assert any("USING COVERING INDEX bands_doc_seq" in row[-1] for row in query)
        assert cleanup(c) == expected
        assert ensure_recovery_index(c)["created"] is False
        c.close()
    assert observed[0]["created"] is True
    assert data._database is original


def test_wrong_partial_index_is_rejected_and_constructor_hook_restored(tmp_path):
    path = tmp_path / "partial.db"
    populate(path)
    c = data._database(path)
    c.execute("CREATE INDEX bands_doc_seq ON bands(doc_seq) WHERE band=0")
    c.close()
    original = data._database
    with pytest.raises(ValueError, match="complete nonunique"):
        with recovery_index_policy([]):
            data._database(path)
    assert data._database is original
