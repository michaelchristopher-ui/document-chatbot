"""Implements `IndexLedger` on SQLite.

One row per document per variant, keyed by both. Writes are UPSERTs, so a
document re-indexed after it changed refreshes its row rather than growing a
second one, and a variant is only ever added to.

It shares a database file with `SqliteDocumentCatalog` — two tables, one file —
because the two answer questions about the same documents and are written in
the same breath by the same ingest. They stay separate tables because they are
keyed differently: the catalog knows a filename, this knows a filename within
one index.

A connection is opened per call rather than held, for the reason the catalog
gives: Streamlit runs each script pass on its own thread, and a
`sqlite3.Connection` refuses use from a thread other than the one that opened it.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from adapters.outbound.ledgers.constants import (
    _MIGRATIONS,
    _SCHEMA,
    _SELECT_VARIANT,
    _UPSERT,
)
from domain.models import IndexedDocument, IngestionOutcome, IngestionStatus
from domain.variants import IndexVariant


def _to_entry(row: sqlite3.Row) -> IndexedDocument:
    return IndexedDocument(
        outcome=IngestionOutcome(
            filename=row["filename"],
            title=row["title"],
            status=IngestionStatus.read(row["status"]),
            chunk_count=row["chunk_count"],
            duplicate_of=row["duplicate_of"],
        ),
        content_hash=row["content_hash"],
        signature=bytes(row["signature"] or b""),
        # Straight through, NULL included: only the reader can say what an
        # unrecorded reading should mean, so it is not flattened to "" here.
        ocr_model=row["ocr_model"],
    )


class SqliteIndexLedger:
    """Implements `IndexLedger` over a local SQLite file."""

    def __init__(self, path: str):
        self._path = path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute(_SCHEMA)
            for migration in _MIGRATIONS:
                try:
                    connection.execute(migration)
                except sqlite3.OperationalError:
                    # Already applied — the column is there from `_SCHEMA` on a
                    # database this run created, or from an earlier run on one
                    # it did not. Nothing else here can fail this way, and
                    # re-adding a column is the only thing being asked.
                    pass
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        return connection

    def entries(self, variant: IndexVariant) -> dict[str, IndexedDocument]:
        with closing(self._connect()) as connection:
            rows = connection.execute(_SELECT_VARIANT, (variant.slug,)).fetchall()
        return {row["filename"]: _to_entry(row) for row in rows}

    def record(self, variant: IndexVariant, entry: IndexedDocument) -> None:
        outcome = entry.outcome
        with closing(self._connect()) as connection:
            connection.execute(
                _UPSERT,
                {
                    "variant": variant.slug,
                    "filename": outcome.filename,
                    "content_hash": entry.content_hash,
                    "title": outcome.title,
                    "status": outcome.status.value,
                    "chunk_count": outcome.chunk_count,
                    "duplicate_of": outcome.duplicate_of,
                    "signature": entry.signature,
                    "ocr_model": entry.ocr_model,
                    "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            )
            connection.commit()
