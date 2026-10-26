"""SQLite helpers. Later tasks add their tables in init_db()."""
import sqlite3
from pathlib import Path

from backend.config import db_path


def get_connection(path: Path | str | None = None) -> sqlite3.Connection:
    """Open a connection (rows behave like dicts, foreign keys on)."""
    conn = sqlite3.connect(str(path if path is not None else db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path: Path | str | None = None) -> Path:
    """Create the database file, its parent folder and base tables. Safe to call repeatedly."""
    target = Path(path if path is not None else db_path())
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(target)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS app_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT OR IGNORE INTO app_meta (key, value) VALUES ('schema_version', '1')"
        )
        conn.commit()
    finally:
        conn.close()
    return target
