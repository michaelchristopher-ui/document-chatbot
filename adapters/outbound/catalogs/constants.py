"""Constants for `adapters.outbound.catalogs`."""

from __future__ import annotations

TABLE = "documents"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    filename     TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    status       TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    duplicate_of TEXT,
    first_seen   TEXT NOT NULL,
    last_seen    TEXT NOT NULL
)
"""

# `first_seen` is written once and left alone from then on; everything else
# reflects the most recent ingest of that filename.
_UPSERT = f"""
INSERT INTO {TABLE} (
    filename, title, status, chunk_count, duplicate_of, first_seen, last_seen
) VALUES (
    :filename, :title, :status, :chunk_count, :duplicate_of, :seen, :seen
)
ON CONFLICT(filename) DO UPDATE SET
    title        = excluded.title,
    status       = excluded.status,
    chunk_count  = excluded.chunk_count,
    duplicate_of = excluded.duplicate_of,
    last_seen    = excluded.last_seen
"""

_SELECT_ALL = f"""
SELECT filename, title, status, chunk_count, duplicate_of
FROM {TABLE}
ORDER BY title COLLATE NOCASE
"""

_SELECT_TITLE = f"SELECT title FROM {TABLE} WHERE filename = ?"
