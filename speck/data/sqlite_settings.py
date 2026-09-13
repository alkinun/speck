"""Validate and apply identity-bound SQLite settings for preparation configs."""

import sqlite3


def validate_sqlite_settings(value):
    if not isinstance(value, dict) or set(value) != {
        "journal_mode",
        "synchronous",
        "wal_autocheckpoint_pages",
        "page_size",
        "cache_size_kib",
    }:
        raise ValueError("SQLite settings require the complete qualified declaration")
    if (
        value["journal_mode"] != "WAL"
        or value["synchronous"] != "FULL"
        or isinstance(value["wal_autocheckpoint_pages"], bool)
        or not isinstance(value["wal_autocheckpoint_pages"], int)
        or value["wal_autocheckpoint_pages"] not in (1000, 65536)
        or type(value["page_size"]) is not int
        or value["page_size"] != 4096
        or type(value["cache_size_kib"]) is not int
        or value["cache_size_kib"] != 2000
    ):
        raise ValueError("SQLite declaration differs from the qualified FULL-sync envelope")
    return dict(value)


def sqlite_runtime(connection):
    return {
        **{
            key: connection.execute(f"PRAGMA {key}").fetchone()[0]
            for key in (
                "journal_mode",
                "synchronous",
                "foreign_keys",
                "wal_autocheckpoint",
                "page_size",
                "cache_size",
            )
        },
        "sqlite_version": sqlite3.sqlite_version,
    }


def verify_sqlite_runtime(declaration, actual):
    declaration = validate_sqlite_settings(declaration)
    expected = {
        "journal_mode": "wal",
        "synchronous": 2,
        "foreign_keys": 1,
        "wal_autocheckpoint": declaration["wal_autocheckpoint_pages"],
        "page_size": declaration["page_size"],
        "cache_size": -declaration["cache_size_kib"],
    }
    if not isinstance(actual, dict) or any(
        actual.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("actual SQLite settings differ from the bound declaration")
    if not isinstance(actual.get("sqlite_version"), str) or not actual["sqlite_version"]:
        raise ValueError("SQLite runtime version is missing")


def configure_sqlite(connection, declaration):
    declaration = validate_sqlite_settings(declaration)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(f"PRAGMA page_size={declaration['page_size']}")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(f"PRAGMA cache_size=-{declaration['cache_size_kib']}")
    connection.execute(f"PRAGMA wal_autocheckpoint={declaration['wal_autocheckpoint_pages']}")
    actual = sqlite_runtime(connection)
    verify_sqlite_runtime(declaration, actual)
    return actual
