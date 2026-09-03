"""Implements `InteractionLog` on SQLite, and `ViewStatistics` as it stands.

Three tables: one row per answered turn, one per search that turn ran, one per
passage a search returned. Writes are plain INSERTs, not the UPSERTs its sibling
`catalogs.sqlite` uses — a turn is an event, so there is no key to conflict on and
nothing that happens later revises one.

Like the catalog, a connection is opened per call rather than held: Streamlit
runs each script pass on its own thread and a `sqlite3.Connection` refuses use
from another. Unlike the catalog, this file has a reader and a writer at the same
moment — the statistics page can be open in one tab while another is answering —
so it runs in WAL mode, where a reader never blocks the writer.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from typing import Sequence

from adapters.outbound.interaction_logs.constants import (
    _ADDED_COLUMNS,
    _INSERT_RETRIEVAL,
    _INSERT_SEARCH,
    _INSERT_TURN,
    _SCHEMA,
    _SELECT_JUDGEMENTS,
    _SELECT_RETRIEVALS,
    _SELECT_SEARCHES,
    _SELECT_TURNS,
    _SELECT_UNJUDGED,
    _UPSERT_JUDGEMENT,
    _WINDOW,
)
from domain.interactions import (
    Judgement,
    RetrievalRecord,
    SearchRecord,
    TokenUsage,
    TurnContext,
    TurnRecord,
)


def _to_search(row: sqlite3.Row) -> SearchRecord:
    return SearchRecord(
        index=row["search_index"],
        query=row["query"],
        result_count=row["result_count"],
    )


def _to_retrieval(row: sqlite3.Row) -> RetrievalRecord:
    return RetrievalRecord(
        search_index=row["search_index"],
        rank=row["rank"],
        source_file=row["source_file"],
        page=row["page"],
        chunk_index=row["chunk_index"],
        citation_index=row["citation_index"],
        cited=bool(row["cited"]),
        score=row["score"],
        text=row["text"],
        keyword_rank=row["keyword_rank"],
        dense_rank=row["dense_rank"],
        fused_rank=row["fused_rank"],
    )


def _to_judgement(row: sqlite3.Row) -> Judgement:
    return Judgement(
        faithfulness=row["faithfulness"],
        unsupported=tuple(json.loads(row["unsupported"])),
        model=row["model"],
        judged_at=row["judged_at"],
    )


def _to_turn(
    row: sqlite3.Row,
    searches: Sequence[SearchRecord],
    retrievals: Sequence[RetrievalRecord],
    judgement: Judgement | None = None,
) -> TurnRecord:
    return TurnRecord(
        id=row["id"],
        judgement=judgement,
        thread_id=row["thread_id"],
        created_at=row["created_at"],
        question=row["question"],
        answer=row["answer"],
        latency_ms=row["latency_ms"],
        first_token_ms=row["first_token_ms"],
        # Absent counts read back as an empty usage, which `domain.statistics`
        # excludes from its totals rather than averaging in as a zero.
        usage=TokenUsage(
            prompt=row["prompt_tokens"] or 0,
            completion=row["completion_tokens"] or 0,
        ),
        context=TurnContext(
            chat_model=row["chat_model"],
            embed_model=row["embed_model"],
            reranker_model=row["reranker_model"],
            chunking_strategy=row["chunking_strategy"],
            vector_backend=row["vector_backend"],
        ),
        searches=tuple(searches),
        retrievals=tuple(retrievals),
        error=row["error"],
    )


def _add_missing_columns(connection: sqlite3.Connection) -> None:
    """Bring a log written by an earlier version up to the current shape.

    Only ever adds nullable columns, so the rows already there keep every value
    they had and read back with None for what was not measured at the time. A log
    of answers already given is not something to drop and recreate.
    """
    for table, column, declaration in _ADDED_COLUMNS:
        existing = {
            row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if column not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def _group(rows: Sequence[sqlite3.Row], build) -> dict:
    grouped: dict = {}
    for row in rows:
        grouped.setdefault(row["turn_id"], []).append(build(row))
    return grouped


class SqliteInteractionLog:
    """Implements `InteractionLog`, and `ViewStatistics` as it stands."""

    def __init__(self, path: str):
        self._path = path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        with closing(self._connect()) as connection:
            # WAL is a property of the file, set once and remembered, so this is
            # the only place it belongs.
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)
            _add_missing_columns(connection)
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        # A write contending with another write waits rather than raising. The
        # alternative is losing a turn to a `database is locked` on a file this
        # app writes once every few seconds at most.
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def record(self, turn: TurnRecord) -> None:
        """Write a turn and its children in one transaction, so none is half-stored."""
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                _INSERT_TURN,
                {
                    "thread_id": turn.thread_id,
                    "created_at": turn.created_at,
                    "question": turn.question,
                    "answer": turn.answer,
                    "latency_ms": turn.latency_ms,
                    "first_token_ms": turn.first_token_ms,
                    "prompt_tokens": turn.usage.prompt or None,
                    "completion_tokens": turn.usage.completion or None,
                    "chat_model": turn.context.chat_model,
                    "embed_model": turn.context.embed_model,
                    "reranker_model": turn.context.reranker_model,
                    "chunking_strategy": turn.context.chunking_strategy,
                    "vector_backend": turn.context.vector_backend,
                    "error": turn.error,
                },
            )
            turn_id = cursor.lastrowid
            connection.executemany(
                _INSERT_SEARCH,
                [
                    (turn_id, search.index, search.query, search.result_count)
                    for search in turn.searches
                ],
            )
            connection.executemany(
                _INSERT_RETRIEVAL,
                [
                    {
                        "turn_id": turn_id,
                        "search_index": retrieval.search_index,
                        "rank": retrieval.rank,
                        "source_file": retrieval.source_file,
                        "page": retrieval.page,
                        "chunk_index": retrieval.chunk_index,
                        "citation_index": retrieval.citation_index,
                        "cited": int(retrieval.cited),
                        "score": retrieval.score,
                        # Only the cited passages keep their text. They are what a
                        # judge has to read to say whether the answer follows from
                        # its sources; storing the rest would make this file a
                        # second copy of the corpus for no question it answers.
                        "text": retrieval.text if retrieval.cited else None,
                        "keyword_rank": retrieval.keyword_rank,
                        "dense_rank": retrieval.dense_rank,
                        "fused_rank": retrieval.fused_rank,
                    }
                    for retrieval in turn.retrievals
                ],
            )
            connection.commit()

    def record_judgement(self, turn_id: int, judgement: Judgement) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                _UPSERT_JUDGEMENT,
                {
                    "turn_id": turn_id,
                    "faithfulness": judgement.faithfulness,
                    "unsupported": json.dumps(list(judgement.unsupported)),
                    "model": judgement.model,
                    "judged_at": judgement.judged_at,
                },
            )
            connection.commit()

    def unjudged(self, limit: int) -> list[TurnRecord]:
        """Answered turns nothing has scored yet, newest first.

        Turns with an empty answer are skipped: there is nothing for a judge to
        read, and scoring them would only ever produce a zero.
        """
        with closing(self._connect()) as connection:
            ids = [row["id"] for row in connection.execute(_SELECT_UNJUDGED, (limit,))]
            # By id rather than by window: an unjudged turn can sit anywhere in
            # the history, so "the most recent N" would quietly miss the old ones
            # and the backlog would never drain.
            return self._load(connection, ids)

    def turns(self, limit: int) -> list[TurnRecord]:
        """The `limit` most recent turns, oldest first, children included."""
        with closing(self._connect()) as connection:
            return self._read(
                connection,
                _SELECT_TURNS, _SELECT_SEARCHES, _SELECT_RETRIEVALS, _SELECT_JUDGEMENTS,
                (limit,),
            )

    def _load(
        self, connection: sqlite3.Connection, ids: Sequence[int]
    ) -> list[TurnRecord]:
        """The named turns, oldest first. Placeholders are built to fit `ids`."""
        if not ids:
            return []
        holes = ",".join("?" * len(ids))
        return self._read(
            connection,
            _SELECT_TURNS.replace(_WINDOW, holes),
            _SELECT_SEARCHES.replace(_WINDOW, holes),
            _SELECT_RETRIEVALS.replace(_WINDOW, holes),
            _SELECT_JUDGEMENTS.replace(_WINDOW, holes),
            tuple(ids),
        )

    @staticmethod
    def _read(
        connection: sqlite3.Connection,
        turns_sql: str,
        searches_sql: str,
        retrievals_sql: str,
        judgements_sql: str,
        params: tuple,
    ) -> list[TurnRecord]:
        """Four statements rather than one join.

        A join would repeat every turn once per passage and the rows would have to
        be folded back apart anyway; this reads each table once and stitches on the
        turn id.
        """
        rows = connection.execute(turns_sql, params).fetchall()
        searches = _group(connection.execute(searches_sql, params).fetchall(), _to_search)
        retrievals = _group(
            connection.execute(retrievals_sql, params).fetchall(), _to_retrieval
        )
        judgements = {
            row["turn_id"]: _to_judgement(row)
            for row in connection.execute(judgements_sql, params).fetchall()
        }
        return [
            _to_turn(
                row,
                searches.get(row["id"], []),
                retrievals.get(row["id"], []),
                judgements.get(row["id"]),
            )
            for row in rows
        ]
