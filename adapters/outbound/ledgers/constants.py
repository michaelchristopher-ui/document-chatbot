"""Constants for `adapters.outbound.ledgers`."""

from __future__ import annotations

TABLE = "indexed_documents"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    variant      TEXT NOT NULL,
    filename     TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    title        TEXT NOT NULL,
    status       TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    duplicate_of TEXT,
    signature    BLOB NOT NULL DEFAULT x'',
    ocr_model    TEXT,
    indexed_at   TEXT NOT NULL,
    PRIMARY KEY (variant, filename)
)
"""

# Added to a table that may already exist, since `CREATE TABLE IF NOT EXISTS`
# leaves an older one exactly as it found it. Nullable and without a default on
# purpose: a row written before this column existed gets NULL, which says the
# reading behind it is unknown — and unknown has to stay distinguishable from
# the empty string, which says a document needed no reading at all. `_reusable`
# reads the two differently.
_MIGRATIONS = (f"ALTER TABLE {TABLE} ADD COLUMN ocr_model TEXT",)

_UPSERT = f"""
INSERT INTO {TABLE} (
    variant, filename, content_hash, title, status, chunk_count, duplicate_of,
    signature, ocr_model, indexed_at
) VALUES (
    :variant, :filename, :content_hash, :title, :status, :chunk_count,
    :duplicate_of, :signature, :ocr_model, :indexed_at
)
ON CONFLICT(variant, filename) DO UPDATE SET
    content_hash = excluded.content_hash,
    title        = excluded.title,
    status       = excluded.status,
    chunk_count  = excluded.chunk_count,
    duplicate_of = excluded.duplicate_of,
    signature    = excluded.signature,
    ocr_model    = excluded.ocr_model,
    indexed_at   = excluded.indexed_at
"""

_SELECT_VARIANT = f"""
SELECT filename, content_hash, title, status, chunk_count, duplicate_of,
       signature, ocr_model
FROM {TABLE}
WHERE variant = ?
"""
