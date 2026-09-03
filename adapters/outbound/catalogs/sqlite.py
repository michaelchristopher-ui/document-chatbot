"""Implements `DocumentCatalog` on SQLite.

One row per document, keyed by filename. Writes are UPSERTs — re-ingesting a
document refreshes its row rather than replacing the table — so the catalog is
the one place that remembers a document whatever became of the chunks made from
it: keyed by filename alone, it spans every index variant and survives all of
them. `SqliteIndexLedger` shares this file and holds the other half, one row per
document per variant.

A connection is opened per call rather than held: Streamlit runs each script pass
on its own thread, and a `sqlite3.Connection` refuses use from a thread other
than the one that opened it. Against a local file the connect costs far less than
the ingest that triggered it.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from adapters.outbound.catalogs.constants import (
    _SCHEMA,
    _SELECT_ALL,
    _SELECT_TITLE,
    _UPSERT,
)
from domain.models import IngestionOutcome, IngestionStatus


def _to_outcome(row: sqlite3.Row) -> IngestionOutcome:
    return IngestionOutcome(
        filename=row["filename"],
        title=row["title"],
        status=IngestionStatus.read(row["status"]),
        chunk_count=row["chunk_count"],
        duplicate_of=row["duplicate_of"],
    )


class SqliteDocumentCatalog:
    """Implements `DocumentCatalog` over a local SQLite file."""

    def __init__(self, path: str):
        self._path = path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute(_SCHEMA)
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        return connection

    def record(self, outcome: IngestionOutcome) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                _UPSERT,
                {
                    "filename": outcome.filename,
                    "title": outcome.title,
                    "status": outcome.status.value,
                    "chunk_count": outcome.chunk_count,
                    "duplicate_of": outcome.duplicate_of,
                    "seen": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            )
            connection.commit()

    def title_of(self, filename: str) -> str:
        with closing(self._connect()) as connection:
            row = connection.execute(_SELECT_TITLE, (filename,)).fetchone()
        return row["title"] if row else ""

    def documents(self) -> list[IngestionOutcome]:
        with closing(self._connect()) as connection:
            rows = connection.execute(_SELECT_ALL).fetchall()
        return [_to_outcome(row) for row in rows]
